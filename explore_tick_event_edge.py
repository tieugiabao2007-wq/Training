from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gold_ai.data.providers import sha256_file
from gold_ai.direct_net_edge import aggregate_predictions
from gold_ai.tick_event_edge import (
    CYCLE_ID, EDGE_THRESHOLDS_BPS, EXCLUDED_FUTURE_DAY, FAMILY_IDS, FEATURE_COLUMNS,
    HORIZONS, MAX_HORIZON, SCHEMA_VERSION, SLIPPAGE_BUFFER_BPS, STRESS_COST_BPS,
    aggregate_tick_events, build_outcomes, development_tick_files, execute_predictions,
    feature_schema_sha256, fit_predict_bps, stress_predictions, survival_checks,
)
from gold_ai.validation import PurgedWalkForwardSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact Exness tick event-response edge exploration v2")
    parser.add_argument("--tick-dir", type=Path, required=True)
    parser.add_argument("--requirements-prompt", type=Path, required=True)
    parser.add_argument("--polars-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def _splitter(n_splits: int, min_train: int, max_train: int | None) -> PurgedWalkForwardSplit:
    return PurgedWalkForwardSplit(
        n_splits=n_splits, min_train_size=min_train, embargo_bars=MAX_HORIZON,
        max_train_rows=max_train, target_horizon_bars=MAX_HORIZON,
        feature_schema_version=SCHEMA_VERSION,
    )


def _load(args: argparse.Namespace) -> dict[str, Any]:
    files = development_tick_files(args.tick_dir.resolve())
    events = aggregate_tick_events(files)
    outcomes = {h: build_outcomes(events, h) for h in HORIZONS}
    finite = np.isfinite(events.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    finite &= np.isfinite(outcomes[MAX_HORIZON]["asset_return"].to_numpy(dtype=float))
    x = events.loc[finite, FEATURE_COLUMNS].astype(float)
    quotes = events.loc[x.index, ["mid_last", "bid_last", "ask_last"]].astype(float)
    outcomes = {h: outcome.loc[x.index] for h, outcome in outcomes.items()}
    if len(x) < 4_500:
        raise ValueError(f"Insufficient exact tick event rows: {len(x)}")
    if x.index.max().strftime("%Y-%m-%d") >= EXCLUDED_FUTURE_DAY:
        raise ValueError("Future exact tick checkpoint was not kept unseen")
    return {"files": files, "x": x, "quotes": quotes, "outcomes": outcomes}


def _inner_key(metrics: dict[str, Any], stress: dict[str, Any]) -> tuple[float, ...]:
    support = int(metrics["trades"]) >= 80 and int(metrics["buy_trades"]) >= 20 and int(metrics["sell_trades"]) >= 20 and int(metrics["valid_outer_folds"]) >= 2
    return (float(support), float(stress["net_return_sum"] > 0), float(stress["net_return_sum"]), float(metrics["net_return_sum"]), float(stress["profit_factor"]), float(metrics["accuracy_wilson_lower_95"]))


def _select_inner(family_id: str, x: pd.DataFrame, outcomes: dict[int, pd.DataFrame], *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    minimum = max(700, min(1_500, len(x) - 600))
    splits = list(_splitter(2, minimum, 3_500).split(x.index, outcomes[MAX_HORIZON]["exit_time"]))
    parts: dict[tuple[int, float], list[pd.DataFrame]] = {(h, t): [] for h in HORIZONS for t in EDGE_THRESHOLDS_BPS}
    total_rows = 0
    fit_audits: list[dict[str, Any]] = []
    for inner_fold, split in enumerate(splits, start=1):
        total_rows += len(split.test_positions)
        for horizon in HORIZONS:
            y = outcomes[horizon]["asset_return"]
            predictions = fit_predict_bps(
                family_id, x.iloc[split.train_positions], y.iloc[split.train_positions],
                x.iloc[split.test_positions], seed=seed + inner_fold * 100 + horizon,
            )
            fit_audits.append({"inner_fold": inner_fold, "horizon_bars": horizon, "train_rows": len(split.train_positions), "test_rows": len(split.test_positions)})
            for threshold in EDGE_THRESHOLDS_BPS:
                parts[(horizon, threshold)].append(execute_predictions(
                    timestamps=x.index, outcomes=outcomes[horizon], predictions_bps=predictions,
                    test_positions=split.test_positions, horizon=horizon, threshold_bps=threshold,
                    fold=inner_fold, family_id=family_id,
                ))
    diagnostics: list[dict[str, Any]] = []
    for (horizon, threshold), rows in parts.items():
        combined = pd.concat([row for row in rows if len(row)]).sort_index() if any(len(row) for row in rows) else pd.DataFrame(columns=["direction", "actual_direction", "net_return", "gross_strategy_return", "horizon_bars", "fold"])
        metrics = aggregate_predictions(combined, total_test_rows=total_rows)
        stress = aggregate_predictions(stress_predictions(combined), total_test_rows=total_rows)
        diagnostics.append({"horizon_bars": horizon, "threshold_bps": threshold, "metrics": metrics, "three_bps_stress_metrics": stress})
    best = max(diagnostics, key=lambda row: _inner_key(row["metrics"], row["three_bps_stress_metrics"]))
    return {"family_id": family_id, "horizon_bars": int(best["horizon_bars"]), "threshold_bps": float(best["threshold_bps"]), "configuration_frozen_before_outer_test": True}, {"inner_folds": len(splits), "selected_metrics": best["metrics"], "selected_stress_metrics": best["three_bps_stress_metrics"], "configuration_diagnostics": diagnostics, "fit_audits": fit_audits}


def _outer_pair(family_id: str, config: dict[str, Any], x: pd.DataFrame, outcomes: dict[int, pd.DataFrame], train: np.ndarray, test: np.ndarray, *, fold: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon = int(config["horizon_bars"])
    prediction = fit_predict_bps(family_id, x.iloc[train], outcomes[horizon]["asset_return"].iloc[train], x.iloc[test], seed=seed)
    candidate = execute_predictions(timestamps=x.index, outcomes=outcomes[horizon], predictions_bps=prediction, test_positions=test, horizon=horizon, threshold_bps=float(config["threshold_bps"]), fold=fold, family_id=family_id)
    control_prediction = x["mid_return_5m_bps"].iloc[test].to_numpy(dtype=float)
    control = execute_predictions(timestamps=x.index, outcomes=outcomes[horizon], predictions_bps=control_prediction, test_positions=test, horizon=horizon, threshold_bps=float(config["threshold_bps"]), fold=fold, family_id="lagged_mid_momentum_control")
    return candidate, control


def _rank(row: tuple[str, dict[str, Any]]) -> tuple[float, ...]:
    result = row[1]
    return (float(result["three_bps_stress_metrics"]["net_return_sum"]), float(result["metrics"]["net_return_sum"]), float(result["three_bps_stress_metrics"]["profit_factor"]), -abs(float(result["metrics"]["max_drawdown_additive"])))


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to repeat tick event artifact: {output}")
    if not 1 <= args.n_jobs <= 16:
        raise ValueError("n-jobs must be 1..16")
    for path in (args.requirements_prompt.resolve(), args.polars_provenance.resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)
    data = _load(args)
    x, outcomes = data["x"], data["outcomes"]
    outer = list(_splitter(3, 1_800, 4_500).split(x.index, outcomes[MAX_HORIZON]["exit_time"]))
    preflight = {"status": "PASS_PREFLIGHT_NO_TRAINING", "cycle_id": CYCLE_ID, "development_rows": len(x), "development_days": int(x.index.floor("D").nunique()), "development_start": x.index.min().isoformat(), "development_end": x.index.max().isoformat(), "excluded_future_day": EXCLUDED_FUTURE_DAY, "outer_folds": len(outer), "families": list(FAMILY_IDS), "polars_version": pl.__version__, "holdout_access_count": 0}
    if args.preflight:
        print(json.dumps(preflight, indent=2))
        return 0
    candidate_parts = {family: [] for family in FAMILY_IDS}
    control_parts = {family: [] for family in FAMILY_IDS}
    fold_records = {family: [] for family in FAMILY_IDS}
    outer_audits: list[dict[str, Any]] = []
    total_test_rows = 0
    for fold, split in enumerate(outer, start=1):
        total_test_rows += len(split.test_positions)
        audit = split.audit_dict()
        audit["train_start_time"] = x.index[split.train_positions[0]].isoformat()
        audit["train_end_time"] = x.index[split.train_positions[-1]].isoformat()
        outer_audits.append(audit)
        for offset, family in enumerate(FAMILY_IDS):
            print(f"[tick-event-v2] fold={fold}/3 family={family} phase=inner", flush=True)
            train = split.train_positions
            config, inner_audit = _select_inner(family, x.iloc[train], {h: outcomes[h].iloc[train] for h in HORIZONS}, seed=25_000 + fold * 100 + offset * 10)
            print(f"[tick-event-v2] fold={fold}/3 family={family} frozen=h{config['horizon_bars']}/edge{config['threshold_bps']}bps", flush=True)
            candidate, control = _outer_pair(family, config, x, outcomes, train, split.test_positions, fold=fold, seed=26_000 + fold * 100 + offset * 10)
            candidate_parts[family].append(candidate)
            control_parts[family].append(control)
            fold_records[family].append({"fold": fold, "frozen_configuration": config, "inner_selection": inner_audit, "candidate_metrics": aggregate_predictions(candidate, total_test_rows=len(split.test_positions)), "control_metrics": aggregate_predictions(control, total_test_rows=len(split.test_positions)), "three_bps_stress_metrics": aggregate_predictions(stress_predictions(candidate), total_test_rows=len(split.test_positions))})
    results: dict[str, dict[str, Any]] = {}
    saved: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for family in FAMILY_IDS:
        candidate = pd.concat([p for p in candidate_parts[family] if len(p)]).sort_index()
        control = pd.concat([p for p in control_parts[family] if len(p)]).sort_index()
        metrics = aggregate_predictions(candidate, total_test_rows=total_test_rows)
        stress = aggregate_predictions(stress_predictions(candidate), total_test_rows=total_test_rows)
        control_metrics = aggregate_predictions(control, total_test_rows=total_test_rows)
        checks = survival_checks(metrics, stress, control_metrics)
        results[family] = {"metrics": metrics, "three_bps_stress_metrics": stress, "paired_lag_momentum_control_metrics": control_metrics, "survival_checks": checks, "eligible_for_validation": all(checks.values()), "fold_records": fold_records[family]}
        saved[family] = (candidate, control)
    survivors = [row for row in results.items() if row[1]["eligible_for_validation"]]
    selected = max(survivors, key=_rank)[0] if survivors else None
    best = max(results.items(), key=_rank)[0]
    report = selected or best
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "cycle_id": CYCLE_ID,
        "status": "EXPLORATION_SURVIVOR_PENDING_VALIDATION" if selected else "ALL_CANDIDATES_REJECTED",
        "decision": "ESCALATE_BEST_TICK_EVENT_SURVIVOR" if selected else "REJECT_ALL_TICK_EVENT_CANDIDATES",
        "current_mode": "EXPLORATION", "research_only": True, "data_limited_21_days": True,
        "production_promotion_allowed": False, "verified_predictive_improvement_allowed": False,
        "live_trading_enabled": False, "auto_execution": False,
        "selected_for_validation_candidate_id": selected, "best_observed_candidate_id": best,
        "report_candidate_id": report, "candidate_metrics": results[report]["metrics"], "candidate_results": results,
        "protocol": {"pre_registered_before_fit": True, "feature_schema_version": SCHEMA_VERSION, "feature_schema_sha256": feature_schema_sha256(), "feature_columns": list(FEATURE_COLUMNS), "families": list(FAMILY_IDS), "horizons": list(HORIZONS), "inner_only_edge_thresholds_bps": list(EDGE_THRESHOLDS_BPS), "outer_folds": 3, "inner_folds": 2, "purge_and_embargo_bars": MAX_HORIZON, "exact_executable_return": "BUY_future_bid_over_current_ask_or_SELL_current_bid_over_future_ask_minus_0.50bps_buffer", "slippage_buffer_bps": SLIPPAGE_BUFFER_BPS, "stress_cost_bps": STRESS_COST_BPS, "non_overlapping_trades": True, "development_rows": len(x), "development_days": int(x.index.floor('D').nunique()), "development_start": x.index.min().isoformat(), "development_end": x.index.max().isoformat(), "excluded_future_day": EXCLUDED_FUTURE_DAY, "excluded_future_day_accessed": False, "sealed_holdout_access_count": 0},
        "source_provenance": {"tick_files": [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in data["files"]], "requirements_prompt": {"path": str(args.requirements_prompt.resolve()), "sha256": sha256_file(args.requirements_prompt)}, "polars": {"path": str(args.polars_provenance.resolve()), "sha256": sha256_file(args.polars_provenance), "version": pl.__version__}, "trainer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())}, "core": {"path": str((PROJECT_ROOT/'src/gold_ai/tick_event_edge.py').resolve()), "sha256": sha256_file(PROJECT_ROOT/'src/gold_ai/tick_event_edge.py')}},
        "outer_fold_audits": outer_audits, "lockbox_status": "SEALED_NOT_ACCESSED",
    }
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    if temporary.exists(): shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    (temporary/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for family, (candidate, control) in saved.items():
        candidate.to_csv(temporary/f"predictions_{family}.csv", index_label="timestamp")
        control.to_csv(temporary/f"predictions_{family}_control.csv", index_label="timestamp")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    print(json.dumps({"decision": summary["decision"], "selected_for_validation": selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
