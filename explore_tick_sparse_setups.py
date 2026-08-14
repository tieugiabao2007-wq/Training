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
from gold_ai.tick_event_edge import aggregate_tick_events, development_tick_files
from gold_ai.tick_sparse_setups import (
    CYCLE_ID,
    EXCLUDED_FUTURE_DAY,
    FEATURE_COLUMNS,
    HORIZONS,
    MAX_HORIZON,
    META_THRESHOLDS,
    SCHEMA_VERSION,
    SETUP_SPECS,
    SLIPPAGE_BUFFER_BPS,
    STRESS_COST_BPS,
    build_setup_directions,
    build_sparse_outcomes,
    execute_sparse_predictions,
    feature_schema_sha256,
    fit_predict_meta,
    stress_predictions,
    survival_checks,
)
from gold_ai.validation import PurgedWalkForwardSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse causal setup meta-label exploration on exact Exness ticks")
    parser.add_argument("--tick-dir", type=Path, required=True)
    parser.add_argument("--requirements-prompt", type=Path, required=True)
    parser.add_argument("--polars-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def _splitter(n_splits: int, min_train: int, max_train: int | None) -> PurgedWalkForwardSplit:
    return PurgedWalkForwardSplit(
        n_splits=n_splits,
        min_train_size=min_train,
        embargo_bars=MAX_HORIZON,
        max_train_rows=max_train,
        target_horizon_bars=MAX_HORIZON,
        feature_schema_version=SCHEMA_VERSION,
    )


def _load(args: argparse.Namespace) -> dict[str, Any]:
    files = development_tick_files(args.tick_dir.resolve())
    events = aggregate_tick_events(files)
    directions = build_setup_directions(events)
    outcomes = {
        setup_id: {
            horizon: build_sparse_outcomes(events, directions, horizon)[setup_id]
            for horizon in HORIZONS
        }
        for setup_id in SETUP_SPECS
    }
    finite = np.isfinite(events.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    first = next(iter(SETUP_SPECS))
    finite &= np.isfinite(outcomes[first][MAX_HORIZON]["asset_return"].to_numpy(dtype=float))
    x = events.loc[finite, FEATURE_COLUMNS].astype(float)
    directions = directions.loc[x.index]
    outcomes = {
        setup_id: {horizon: frame.loc[x.index] for horizon, frame in by_horizon.items()}
        for setup_id, by_horizon in outcomes.items()
    }
    if len(x) < 4_500:
        raise ValueError(f"Insufficient exact tick event rows: {len(x)}")
    if x.index.max() >= pd.Timestamp(EXCLUDED_FUTURE_DAY, tz="UTC"):
        raise ValueError("Future exact tick checkpoint was not kept unseen")
    return {"files": files, "events": events, "x": x, "directions": directions, "outcomes": outcomes}


def _inner_key(metrics: dict[str, Any], stress: dict[str, Any]) -> tuple[float, ...]:
    support = (
        int(metrics["trades"]) >= 40
        and int(metrics["buy_trades"]) >= 10
        and int(metrics["sell_trades"]) >= 10
        and int(metrics["valid_outer_folds"]) >= 2
    )
    return (
        float(support),
        float(stress["net_return_sum"] > 0),
        float(stress["net_return_sum"]),
        float(metrics["net_return_sum"]),
        float(stress["profit_factor"]),
        float(metrics["accuracy_wilson_lower_95"]),
    )


def _select_inner(
    setup_id: str,
    x: pd.DataFrame,
    outcomes: dict[int, pd.DataFrame],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    minimum = max(700, min(1_500, len(x) - 600))
    splits = list(
        _splitter(2, minimum, 3_500).split(x.index, outcomes[MAX_HORIZON]["label_end_time"])
    )
    parts: dict[tuple[int, float], list[pd.DataFrame]] = {
        (horizon, threshold): []
        for horizon in HORIZONS
        for threshold in META_THRESHOLDS
    }
    audits: list[dict[str, Any]] = []
    total_rows = 0
    for inner_fold, split in enumerate(splits, start=1):
        total_rows += len(split.test_positions)
        for horizon in HORIZONS:
            positions, probabilities, audit = fit_predict_meta(
                x,
                outcomes[horizon],
                split.train_positions,
                split.test_positions,
                seed=seed + inner_fold * 100 + horizon,
            )
            audits.append({"inner_fold": inner_fold, "horizon_bars": horizon, **audit})
            for threshold in META_THRESHOLDS:
                parts[(horizon, threshold)].append(
                    execute_sparse_predictions(
                        timestamps=x.index,
                        outcome=outcomes[horizon],
                        candidate_positions=positions,
                        probabilities=probabilities,
                        threshold=threshold,
                        test_positions=split.test_positions,
                        horizon=horizon,
                        fold=inner_fold,
                        setup_id=setup_id,
                        role="candidate",
                    )
                )
    diagnostics: list[dict[str, Any]] = []
    for (horizon, threshold), rows in parts.items():
        nonempty = [row for row in rows if len(row)]
        combined = (
            pd.concat(nonempty).sort_index()
            if nonempty
            else pd.DataFrame(columns=["direction", "actual_direction", "net_return", "gross_strategy_return", "horizon_bars", "fold"])
        )
        metrics = aggregate_predictions(combined, total_test_rows=total_rows)
        stress = aggregate_predictions(stress_predictions(combined), total_test_rows=total_rows)
        diagnostics.append(
            {
                "horizon_bars": horizon,
                "meta_threshold": threshold,
                "metrics": metrics,
                "three_bps_stress_metrics": stress,
            }
        )
    best = max(diagnostics, key=lambda row: _inner_key(row["metrics"], row["three_bps_stress_metrics"]))
    return (
        {
            "setup_id": setup_id,
            "horizon_bars": int(best["horizon_bars"]),
            "meta_threshold": float(best["meta_threshold"]),
            "configuration_frozen_before_outer_test": True,
        },
        {
            "inner_folds": len(splits),
            "selected_metrics": best["metrics"],
            "selected_stress_metrics": best["three_bps_stress_metrics"],
            "configuration_diagnostics": diagnostics,
            "fit_audits": audits,
        },
    )


def _outer_pair(
    setup_id: str,
    configuration: dict[str, Any],
    x: pd.DataFrame,
    outcomes: dict[int, pd.DataFrame],
    train: np.ndarray,
    test: np.ndarray,
    *,
    fold: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    horizon = int(configuration["horizon_bars"])
    positions, probabilities, audit = fit_predict_meta(
        x, outcomes[horizon], train, test, seed=seed
    )
    candidate = execute_sparse_predictions(
        timestamps=x.index,
        outcome=outcomes[horizon],
        candidate_positions=positions,
        probabilities=probabilities,
        threshold=float(configuration["meta_threshold"]),
        test_positions=test,
        horizon=horizon,
        fold=fold,
        setup_id=setup_id,
        role="candidate",
    )
    control = execute_sparse_predictions(
        timestamps=x.index,
        outcome=outcomes[horizon],
        candidate_positions=positions,
        probabilities=np.ones(len(positions), dtype=float),
        threshold=0.0,
        test_positions=test,
        horizon=horizon,
        fold=fold,
        setup_id=setup_id,
        role="paired_rule_control",
    )
    return candidate, control, audit


def _rank(row: tuple[str, dict[str, Any]]) -> tuple[float, ...]:
    result = row[1]
    return (
        float(result["three_bps_stress_metrics"]["net_return_sum"]),
        float(result["metrics"]["net_return_sum"]),
        float(result["three_bps_stress_metrics"]["profit_factor"]),
        -abs(float(result["metrics"]["max_drawdown_additive"])),
    )


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to repeat sparse tick setup artifact: {output}")
    if not 1 <= int(args.n_jobs) <= 16:
        raise ValueError("n-jobs must be 1..16")
    for path in (args.requirements_prompt.resolve(), args.polars_provenance.resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)

    data = _load(args)
    x = data["x"]
    counts = {
        setup_id: {
            "events": int((data["directions"][setup_id] != 0).sum()),
            "buy": int((data["directions"][setup_id] == 1).sum()),
            "sell": int((data["directions"][setup_id] == -1).sum()),
        }
        for setup_id in SETUP_SPECS
    }
    first = next(iter(SETUP_SPECS))
    outer = list(
        _splitter(3, 1_800, 4_500).split(
            x.index, data["outcomes"][first][MAX_HORIZON]["label_end_time"]
        )
    )
    preflight = {
        "status": "PASS_PREFLIGHT_NO_TRAINING",
        "cycle_id": CYCLE_ID,
        "development_rows": len(x),
        "development_days": int(x.index.floor("D").nunique()),
        "development_start": x.index.min().isoformat(),
        "development_end": x.index.max().isoformat(),
        "excluded_future_day": EXCLUDED_FUTURE_DAY,
        "outer_folds": len(outer),
        "setup_counts": counts,
        "polars_version": pl.__version__,
        "holdout_access_count": 0,
    }
    if args.preflight:
        print(json.dumps(preflight, indent=2))
        return 0

    candidate_parts = {setup_id: [] for setup_id in SETUP_SPECS}
    control_parts = {setup_id: [] for setup_id in SETUP_SPECS}
    fold_records = {setup_id: [] for setup_id in SETUP_SPECS}
    outer_audits: list[dict[str, Any]] = []
    total_test_rows = 0
    for fold, split in enumerate(outer, start=1):
        total_test_rows += len(split.test_positions)
        split_audit = split.audit_dict()
        split_audit["train_start_time"] = x.index[split.train_positions[0]].isoformat()
        split_audit["train_end_time"] = x.index[split.train_positions[-1]].isoformat()
        outer_audits.append(split_audit)
        for offset, setup_id in enumerate(SETUP_SPECS):
            print(f"[tick-sparse] fold={fold}/3 setup={setup_id} phase=inner", flush=True)
            train = split.train_positions
            configuration, inner_audit = _select_inner(
                setup_id,
                x.iloc[train],
                {horizon: data["outcomes"][setup_id][horizon].iloc[train] for horizon in HORIZONS},
                seed=31_000 + fold * 100 + offset * 10,
            )
            print(
                f"[tick-sparse] fold={fold}/3 setup={setup_id} "
                f"frozen=h{configuration['horizon_bars']}/p{configuration['meta_threshold']}",
                flush=True,
            )
            candidate, control, fit_audit = _outer_pair(
                setup_id,
                configuration,
                x,
                data["outcomes"][setup_id],
                train,
                split.test_positions,
                fold=fold,
                seed=32_000 + fold * 100 + offset * 10,
            )
            candidate_parts[setup_id].append(candidate)
            control_parts[setup_id].append(control)
            fold_records[setup_id].append(
                {
                    "fold": fold,
                    "frozen_configuration": configuration,
                    "inner_selection": inner_audit,
                    "outer_fit_audit": fit_audit,
                    "candidate_metrics": aggregate_predictions(candidate, total_test_rows=len(split.test_positions)),
                    "control_metrics": aggregate_predictions(control, total_test_rows=len(split.test_positions)),
                    "candidate_stress_metrics": aggregate_predictions(stress_predictions(candidate), total_test_rows=len(split.test_positions)),
                }
            )

    results: dict[str, dict[str, Any]] = {}
    saved: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for setup_id, specification in SETUP_SPECS.items():
        candidate_nonempty = [part for part in candidate_parts[setup_id] if len(part)]
        control_nonempty = [part for part in control_parts[setup_id] if len(part)]
        empty = pd.DataFrame(columns=["direction", "actual_direction", "net_return", "gross_strategy_return", "horizon_bars", "fold"])
        candidate = pd.concat(candidate_nonempty).sort_index() if candidate_nonempty else empty.copy()
        control = pd.concat(control_nonempty).sort_index() if control_nonempty else empty.copy()
        metrics = aggregate_predictions(candidate, total_test_rows=total_test_rows)
        stress = aggregate_predictions(stress_predictions(candidate), total_test_rows=total_test_rows)
        control_metrics = aggregate_predictions(control, total_test_rows=total_test_rows)
        control_stress = aggregate_predictions(stress_predictions(control), total_test_rows=total_test_rows)
        checks = survival_checks(metrics, stress, control_metrics, control_stress)
        results[setup_id] = {
            "specification": specification,
            "metrics": metrics,
            "three_bps_stress_metrics": stress,
            "paired_rule_control_metrics": control_metrics,
            "paired_rule_control_three_bps_stress_metrics": control_stress,
            "survival_checks": checks,
            "eligible_for_validation": all(checks.values()),
            "fold_records": fold_records[setup_id],
        }
        saved[setup_id] = (candidate, control)

    survivors = [row for row in results.items() if row[1]["eligible_for_validation"]]
    selected = max(survivors, key=_rank)[0] if survivors else None
    best = max(results.items(), key=_rank)[0]
    report = selected or best
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cycle_id": CYCLE_ID,
        "status": "EXPLORATION_SURVIVOR_PENDING_VALIDATION" if selected else "ALL_CANDIDATES_REJECTED",
        "decision": "ESCALATE_BEST_SPARSE_TICK_SETUP_SURVIVOR" if selected else "REJECT_ALL_SPARSE_TICK_SETUP_CANDIDATES",
        "current_mode": "EXPLORATION",
        "research_only": True,
        "data_limited_21_days": True,
        "production_promotion_allowed": False,
        "verified_predictive_improvement_allowed": False,
        "live_trading_enabled": False,
        "auto_execution": False,
        "selected_for_validation_candidate_id": selected,
        "best_observed_candidate_id": best,
        "report_candidate_id": report,
        "candidate_metrics": results[report]["metrics"],
        "candidate_results": results,
        "setup_counts": counts,
        "protocol": {
            "pre_registered_before_fit": True,
            "objective_order": ["outer_oos_three_bps_stress_profit", "exact_executable_profit", "drawdown_and_fold_stability", "accuracy_wilson"],
            "feature_schema_version": SCHEMA_VERSION,
            "feature_schema_sha256": feature_schema_sha256(),
            "feature_columns": list(FEATURE_COLUMNS),
            "setup_ids": list(SETUP_SPECS),
            "model": "fixed_shallow_hist_gradient_boosting_binary_meta_label",
            "horizons": list(HORIZONS),
            "inner_only_meta_thresholds": list(META_THRESHOLDS),
            "outer_folds": 3,
            "inner_folds": 2,
            "purge_and_embargo_bars": MAX_HORIZON,
            "last_quote_cost_approximation": "BUY_future_bid_over_current_ask_or_SELL_current_bid_over_future_ask_minus_0.50bps_stress_assumption; not causal fill evidence",
            "slippage_buffer_bps": SLIPPAGE_BUFFER_BPS,
            "stress_cost_bps": STRESS_COST_BPS,
            "non_overlapping_trades": True,
            "development_rows": len(x),
            "development_days": int(x.index.floor("D").nunique()),
            "development_start": x.index.min().isoformat(),
            "development_end": x.index.max().isoformat(),
            "excluded_future_day": EXCLUDED_FUTURE_DAY,
            "excluded_future_day_accessed": False,
            "sealed_holdout_access_count": 0,
        },
        "source_provenance": {
            "tick_files": [
                {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in data["files"]
            ],
            "requirements_prompt": {"path": str(args.requirements_prompt.resolve()), "sha256": sha256_file(args.requirements_prompt)},
            "polars": {"path": str(args.polars_provenance.resolve()), "sha256": sha256_file(args.polars_provenance), "version": pl.__version__},
            "trainer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "core": {"path": str((PROJECT_ROOT / "src/gold_ai/tick_sparse_setups.py").resolve()), "sha256": sha256_file(PROJECT_ROOT / "src/gold_ai/tick_sparse_setups.py")},
            "tick_event_core": {"path": str((PROJECT_ROOT / "src/gold_ai/tick_event_edge.py").resolve()), "sha256": sha256_file(PROJECT_ROOT / "src/gold_ai/tick_event_edge.py")},
            "meta_model_core": {"path": str((PROJECT_ROOT / "src/gold_ai/setup_meta_label.py").resolve()), "sha256": sha256_file(PROJECT_ROOT / "src/gold_ai/setup_meta_label.py")},
        },
        "outer_fold_audits": outer_audits,
        "lockbox_status": "SEALED_NOT_ACCESSED",
    }

    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    (temporary / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    for setup_id, (candidate, control) in saved.items():
        candidate.to_csv(temporary / f"predictions_{setup_id}.csv", index_label="timestamp")
        control.to_csv(temporary / f"predictions_{setup_id}_control.csv", index_label="timestamp")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    print(json.dumps({"decision": summary["decision"], "selected_for_validation": selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
