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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gold_ai.anchored_auction import (
    CYCLE_ID,
    FEATURE_SCHEMA_VERSION,
    HORIZONS,
    MAX_DEVELOPMENT_ROWS,
    MAX_HORIZON,
    META_THRESHOLDS,
    SETUP_SPECS,
    SLIPPAGE_BUFFER_BPS,
    STRESS_COST_BPS,
    build_auction_features,
    build_horizon_outcomes,
    build_setup_directions,
    feature_schema_sha256,
    stress_predictions,
    survival_checks,
)
from gold_ai.data.providers import load_saved_market_data, sha256_file
from gold_ai.direct_net_edge import aggregate_predictions
from gold_ai.setup_meta_label import execute_setup_predictions, fit_predict_meta
from gold_ai.validation import PurgedWalkForwardSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anchored-auction meta-label exploration on exact Exness M5")
    parser.add_argument("--m5", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--requirements-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _load_dataset(args: argparse.Namespace) -> dict[str, Any]:
    bars, metadata = load_saved_market_data(args.m5)
    if metadata.instrument != "XAUUSDm" or metadata.interval != "M5":
        raise ValueError("Source must be exact legal Exness XAUUSDm M5")
    reference_path = args.reference_artifact.resolve() / "summary.json"
    reference = _json(reference_path)
    rows = reference.get("fold_metrics")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Reference artifact lacks development boundary")
    cutoff = _utc(max(rows, key=lambda row: int(row["fold"]))["test_end"])
    label_bars = bars.loc[: cutoff + pd.Timedelta(minutes=5 * MAX_HORIZON)]
    decisions = label_bars.loc[label_bars.index <= cutoff]
    point = float(metadata.symbol_spec.get("point", 0.0) or 0.0)
    if point <= 0:
        raise ValueError("Exact symbol point must be positive")
    features = build_auction_features(decisions, point)
    setup_directions = build_setup_directions(decisions)
    finite = np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    x = features.loc[finite].astype(float).tail(MAX_DEVELOPMENT_ROWS)
    directions = setup_directions.loc[x.index]
    source_positions = label_bars.index.get_indexer(x.index)
    if len(x) != MAX_DEVELOPMENT_ROWS or (source_positions < 0).any() or not np.all(np.diff(source_positions) == 1):
        raise ValueError("Anchored-auction cycle requires 45,000 contiguous exact development rows")
    outcomes = build_horizon_outcomes(label_bars, x.index, directions, point=point)
    return {
        "bars": label_bars,
        "metadata": metadata,
        "reference_path": reference_path,
        "cutoff": cutoff,
        "x": x,
        "directions": directions,
        "source_positions": source_positions,
        "outcomes": outcomes,
        "point": point,
    }


def _splitter(n_splits: int, min_train: int, max_train: int | None) -> PurgedWalkForwardSplit:
    return PurgedWalkForwardSplit(
        n_splits=n_splits,
        min_train_size=min_train,
        embargo_bars=MAX_HORIZON,
        max_train_rows=max_train,
        target_horizon_bars=MAX_HORIZON,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )


def _inner_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    support = (
        int(metrics["trades"]) >= 40
        and int(metrics["buy_trades"]) >= 10
        and int(metrics["sell_trades"]) >= 10
        and int(metrics["valid_outer_folds"]) >= 2
    )
    return (
        float(support),
        float(float(metrics["net_return_sum"]) > 0),
        float(metrics["net_return_sum"]),
        float(metrics["profit_factor"]),
        float(metrics["accuracy_wilson_lower_95"]),
    )


def _select_inner(
    setup_id: str,
    x: pd.DataFrame,
    outcomes: dict[int, pd.DataFrame],
    source_positions: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    minimum = max(5_000, min(10_000, len(x) - 1_000))
    splits = list(_splitter(2, minimum, 24_000).split(x.index, outcomes[MAX_HORIZON]["label_end_time"]))
    candidate_parts: dict[tuple[int, float], list[pd.DataFrame]] = {
        (h, threshold): [] for h in HORIZONS for threshold in META_THRESHOLDS
    }
    fit_audits: list[dict[str, Any]] = []
    total_test_rows = 0
    for inner_fold, split in enumerate(splits, start=1):
        total_test_rows += len(split.test_positions)
        for horizon in HORIZONS:
            event_positions, probabilities, audit = fit_predict_meta(
                x, outcomes[horizon], split.train_positions, split.test_positions,
                seed=seed + inner_fold * 100 + horizon,
            )
            fit_audits.append({"inner_fold": inner_fold, "horizon_bars": horizon, **audit})
            for threshold in META_THRESHOLDS:
                candidate_parts[(horizon, threshold)].append(
                    execute_setup_predictions(
                        timestamps=x.index,
                        source_positions=source_positions,
                        outcome=outcomes[horizon],
                        candidate_positions=event_positions,
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
    for (horizon, threshold), parts in candidate_parts.items():
        nonempty = [part for part in parts if len(part)]
        combined = pd.concat(nonempty).sort_index() if nonempty else pd.DataFrame(
            columns=["direction", "actual_direction", "net_return", "horizon_bars", "fold"]
        )
        diagnostics.append(
            {
                "horizon_bars": horizon,
                "meta_threshold": threshold,
                "metrics": aggregate_predictions(combined, total_test_rows=total_test_rows),
            }
        )
    best = max(diagnostics, key=lambda row: _inner_key(row["metrics"]))
    return {
        "setup_id": setup_id,
        "horizon_bars": int(best["horizon_bars"]),
        "meta_threshold": float(best["meta_threshold"]),
        "configuration_frozen_before_outer_test": True,
    }, {
        "inner_folds": len(splits),
        "selected_metrics": best["metrics"],
        "configuration_diagnostics": diagnostics,
        "fit_audits": fit_audits,
    }


def _outer_pair(
    setup_id: str,
    configuration: dict[str, Any],
    x: pd.DataFrame,
    outcomes: dict[int, pd.DataFrame],
    source_positions: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    *,
    fold: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    horizon = int(configuration["horizon_bars"])
    event_positions, probabilities, audit = fit_predict_meta(
        x, outcomes[horizon], train_positions, test_positions, seed=seed
    )
    common = {
        "timestamps": x.index,
        "source_positions": source_positions,
        "outcome": outcomes[horizon],
        "candidate_positions": event_positions,
        "test_positions": test_positions,
        "horizon": horizon,
        "fold": fold,
        "setup_id": setup_id,
    }
    candidate = execute_setup_predictions(
        **common,
        probabilities=probabilities,
        threshold=float(configuration["meta_threshold"]),
        role="candidate",
    )
    control = execute_setup_predictions(
        **common,
        probabilities=np.ones(len(event_positions), dtype=float),
        threshold=0.0,
        role="paired_rule_control",
    )
    return candidate, control, audit


def _rank(row: tuple[str, dict[str, Any]]) -> tuple[float, ...]:
    metrics = row[1]["metrics"]
    stress = row[1]["three_bps_stress_metrics"]
    return (
        float(stress["net_return_sum"]),
        float(metrics["net_return_sum"]),
        float(stress["profit_factor"]),
        -abs(float(metrics["max_drawdown_additive"])),
    )


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to repeat anchored-auction artifact: {output}")
    if not 1 <= int(args.n_jobs) <= 16:
        raise ValueError("n-jobs must be 1..16")
    prompt = args.requirements_prompt.resolve()
    if not prompt.is_file():
        raise FileNotFoundError(prompt)
    data = _load_dataset(args)
    x = data["x"]
    counts = {
        setup_id: {
            "events": int((data["directions"][setup_id] != 0).sum()),
            "buy": int((data["directions"][setup_id] == 1).sum()),
            "sell": int((data["directions"][setup_id] == -1).sum()),
        }
        for setup_id in SETUP_SPECS
    }
    outer_splits = list(
        _splitter(3, 12_000, 30_000).split(
            x.index, data["outcomes"][next(iter(SETUP_SPECS))][MAX_HORIZON]["label_end_time"]
        )
    )
    if args.preflight:
        print(json.dumps({
            "status": "PASS_PREFLIGHT_NO_TRAINING",
            "cycle_id": CYCLE_ID,
            "development_rows": len(x),
            "outer_folds": len(outer_splits),
            "causal_setup_counts": counts,
            "dynamic_cost": "side_aware_exact_bar_spread_plus_0.50bps_buffer",
            "stress_cost_bps": STRESS_COST_BPS,
            "holdout_access_count": 0,
        }, indent=2))
        return 0

    candidate_parts: dict[str, list[pd.DataFrame]] = {setup: [] for setup in SETUP_SPECS}
    control_parts: dict[str, list[pd.DataFrame]] = {setup: [] for setup in SETUP_SPECS}
    fold_records: dict[str, list[dict[str, Any]]] = {setup: [] for setup in SETUP_SPECS}
    outer_audits: list[dict[str, Any]] = []
    total_test_rows = 0
    for fold, split in enumerate(outer_splits, start=1):
        total_test_rows += len(split.test_positions)
        audit = split.audit_dict()
        audit["train_start_time"] = x.index[split.train_positions[0]].isoformat()
        audit["train_end_time"] = x.index[split.train_positions[-1]].isoformat()
        outer_audits.append(audit)
        for offset, setup_id in enumerate(SETUP_SPECS):
            print(f"[anchored-auction] fold={fold}/3 setup={setup_id} phase=inner", flush=True)
            outer_train = split.train_positions
            frozen, inner_audit = _select_inner(
                setup_id,
                x.iloc[outer_train],
                {h: data["outcomes"][setup_id][h].iloc[outer_train] for h in HORIZONS},
                data["source_positions"][outer_train],
                seed=21_000 + fold * 100 + offset * 10,
            )
            print(
                f"[anchored-auction] fold={fold}/3 setup={setup_id} "
                f"frozen=h{frozen['horizon_bars']}/p{frozen['meta_threshold']}",
                flush=True,
            )
            candidate, control, fit_audit = _outer_pair(
                setup_id, frozen, x, data["outcomes"][setup_id], data["source_positions"],
                split.train_positions, split.test_positions,
                fold=fold, seed=22_000 + fold * 100 + offset * 10,
            )
            candidate_parts[setup_id].append(candidate)
            control_parts[setup_id].append(control)
            fold_records[setup_id].append({
                "fold": fold,
                "frozen_configuration": frozen,
                "inner_selection": inner_audit,
                "outer_fit_audit": fit_audit,
                "candidate_metrics": aggregate_predictions(candidate, total_test_rows=len(split.test_positions)),
                "control_metrics": aggregate_predictions(control, total_test_rows=len(split.test_positions)),
                "three_bps_stress_metrics": aggregate_predictions(stress_predictions(candidate), total_test_rows=len(split.test_positions)),
            })

    candidate_results: dict[str, dict[str, Any]] = {}
    saved: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for setup_id, spec in SETUP_SPECS.items():
        candidate = pd.concat([p for p in candidate_parts[setup_id] if len(p)]).sort_index() if any(len(p) for p in candidate_parts[setup_id]) else pd.DataFrame(columns=["direction", "actual_direction", "net_return", "horizon_bars", "fold"])
        control = pd.concat([p for p in control_parts[setup_id] if len(p)]).sort_index() if any(len(p) for p in control_parts[setup_id]) else pd.DataFrame(columns=["direction", "actual_direction", "net_return", "horizon_bars", "fold"])
        metrics = aggregate_predictions(candidate, total_test_rows=total_test_rows)
        control_metrics = aggregate_predictions(control, total_test_rows=total_test_rows)
        stress_metrics = aggregate_predictions(stress_predictions(candidate), total_test_rows=total_test_rows)
        checks = survival_checks(metrics, control_metrics, stress_metrics)
        candidate_results[setup_id] = {
            "specification": spec,
            "metrics": metrics,
            "paired_rule_control_metrics": control_metrics,
            "three_bps_stress_metrics": stress_metrics,
            "survival_checks": checks,
            "eligible_for_validation": all(checks.values()),
            "fold_records": fold_records[setup_id],
        }
        saved[setup_id] = (candidate, control)
    survivors = [row for row in candidate_results.items() if row[1]["eligible_for_validation"]]
    selected = max(survivors, key=_rank)[0] if survivors else None
    best_observed = max(candidate_results.items(), key=_rank)[0]
    report_id = selected or best_observed
    decision = "ESCALATE_BEST_ANCHORED_AUCTION_SURVIVOR" if selected else "REJECT_ALL_ANCHORED_AUCTION_CANDIDATES"
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cycle_id": CYCLE_ID,
        "status": "EXPLORATION_SURVIVOR_PENDING_VALIDATION" if selected else "ALL_CANDIDATES_REJECTED",
        "decision": decision,
        "current_mode": "EXPLORATION",
        "active_contract": "V15.1_LEAN",
        "research_only": True,
        "production_promotion_allowed": False,
        "verified_predictive_improvement_allowed": False,
        "live_trading_enabled": False,
        "auto_execution": False,
        "selected_for_validation_candidate_id": selected,
        "best_observed_candidate_id": best_observed,
        "report_candidate_id": report_id,
        "candidate_metrics": candidate_results[report_id]["metrics"],
        "candidate_results": candidate_results,
        "causal_setup_counts": counts,
        "protocol": {
            "pre_registered_before_fit": True,
            "objective_order": ["outer_oos_three_bps_stress_profit", "dynamic_cost_profit_drawdown_stability", "accuracy_wilson"],
            "horizons": list(HORIZONS),
            "inner_only_meta_thresholds": list(META_THRESHOLDS),
            "slippage_latency_buffer_bps": SLIPPAGE_BUFFER_BPS,
            "stress_cost_bps": STRESS_COST_BPS,
            "dynamic_cost": "BUY_entry_spread_SELL_exit_spread_plus_buffer",
            "outer_folds": 3,
            "inner_folds": 2,
            "purge_and_embargo_bars": MAX_HORIZON,
            "maximum_development_rows": MAX_DEVELOPMENT_ROWS,
            "development_rows": len(x),
            "development_cutoff": data["cutoff"].isoformat(),
            "total_outer_test_rows": total_test_rows,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_sha256": feature_schema_sha256(),
            "setup_ids": list(SETUP_SPECS),
            "model": "fixed_shallow_hist_gradient_boosting_binary_meta_label",
            "gpu_policy": "not_used_sklearn_hgb_has_no_smoke_tested_gpu_backend",
            "sealed_holdout_access_count": 0,
            "saved_lockbox_predictions_accessed": False,
        },
        "source_provenance": {
            "m5": {"path": str(args.m5.resolve()), "sha256": sha256_file(args.m5)},
            "m5_metadata": {"path": str(args.m5.with_suffix('.metadata.json').resolve()), "sha256": sha256_file(args.m5.with_suffix('.metadata.json'))},
            "reference_summary": {"path": str(data["reference_path"]), "sha256": sha256_file(data["reference_path"])},
            "requirements_prompt": {"path": str(prompt), "sha256": sha256_file(prompt)},
            "trainer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "core": {"path": str((PROJECT_ROOT / 'src/gold_ai/anchored_auction.py').resolve()), "sha256": sha256_file(PROJECT_ROOT / 'src/gold_ai/anchored_auction.py')},
        },
        "outer_fold_audits": outer_audits,
        "lockbox_status": "SEALED_NOT_ACCESSED",
    }
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    (temporary / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for setup_id, (candidate, control) in saved.items():
        candidate.to_csv(temporary / f"predictions_{setup_id}.csv", index_label="timestamp")
        control.to_csv(temporary / f"predictions_{setup_id}_control.csv", index_label="timestamp")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    print(json.dumps({"decision": decision, "selected_for_validation": selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
