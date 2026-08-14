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

from gold_ai.data.providers import load_saved_market_data, sha256_file
from gold_ai.direct_net_edge import (
    CONTROL_ID,
    CYCLE_ID,
    FAMILY_SPECS,
    FEATURE_SCHEMA_VERSION,
    HORIZONS,
    MAX_DEVELOPMENT_ROWS,
    MAX_LABEL_HORIZON,
    ROUND_TRIP_COST_BPS,
    THRESHOLDS_BPS,
    aggregate_predictions,
    build_control_model,
    build_direct_edge_features,
    build_fixed_horizon_targets,
    build_mean_model,
    execute_value_predictions,
    exploration_survival_checks,
    feature_schema_sha256,
    fit_predict_bps,
    fit_predict_directml_multihorizon,
    fit_predict_quantiles,
    inner_selection_key,
)
from gold_ai.validation import PurgedWalkForwardSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded exact-Exness M5 direct after-cost value exploration cycle."
    )
    parser.add_argument("--m5", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--requirements-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _dataset(args: argparse.Namespace) -> dict[str, Any]:
    bars, metadata = load_saved_market_data(args.m5)
    if metadata.instrument != "XAUUSDm" or metadata.interval != "M5":
        raise ValueError("Source must be the exact legal Exness XAUUSDm M5 snapshot")
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError("Exact M5 timestamps must be unique and chronological")
    reference_path = args.reference_artifact.resolve() / "summary.json"
    reference = _read_json(reference_path)
    fold_rows = reference.get("fold_metrics")
    if not isinstance(fold_rows, list) or not fold_rows:
        raise ValueError("Reference summary has no frozen development boundary")
    cutoff = _utc(max(fold_rows, key=lambda row: int(row["fold"]))["test_end"])
    label_cutoff = cutoff + pd.Timedelta(minutes=5 * MAX_LABEL_HORIZON)
    label_bars = bars.loc[:label_cutoff]
    decisions = label_bars.loc[label_bars.index <= cutoff]
    point = float(metadata.symbol_spec.get("point", 0.0) or 0.0)
    features = build_direct_edge_features(decisions, point)
    targets, ends = build_fixed_horizon_targets(label_bars, decisions.index)
    finite = np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    finite &= np.isfinite(targets.to_numpy(dtype=float)).all(axis=1)
    x = features.loc[finite].astype(float).tail(MAX_DEVELOPMENT_ROWS)
    targets = targets.loc[x.index].astype(float)
    ends = ends.loc[x.index]
    source_positions = label_bars.index.get_indexer(x.index)
    if len(x) != MAX_DEVELOPMENT_ROWS or (source_positions < 0).any():
        raise ValueError("Frozen direct-edge cycle requires exactly 45,000 complete rows")
    if not np.all(np.diff(source_positions) == 1):
        raise ValueError("Development rows must be contiguous exact M5 source observations")
    return {
        "bars": bars,
        "metadata": metadata,
        "reference_path": reference_path,
        "cutoff": cutoff,
        "x": x,
        "targets": targets,
        "ends": ends,
        "source_positions": source_positions,
    }


def _splitter(*, n_splits: int, min_train_size: int, max_train_rows: int | None) -> PurgedWalkForwardSplit:
    return PurgedWalkForwardSplit(
        n_splits=n_splits,
        min_train_size=min_train_size,
        embargo_bars=MAX_LABEL_HORIZON,
        max_train_rows=max_train_rows,
        target_horizon_bars=MAX_LABEL_HORIZON,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )


def _fit_family_predictions(
    family_id: str,
    x_train: pd.DataFrame,
    targets_train: pd.DataFrame,
    x_test: pd.DataFrame,
    *,
    n_jobs: int,
    seed: int,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    outputs: dict[int, dict[str, np.ndarray]] = {}
    audit: dict[str, Any] = {}
    if family_id == "directml_multihorizon_huber_v1":
        predictions, audit = fit_predict_directml_multihorizon(
            x_train,
            targets_train[[f"return_h{h}" for h in HORIZONS]],
            x_test,
            seed=seed,
        )
        for column, horizon in enumerate(HORIZONS):
            outputs[horizon] = {"predicted": predictions[:, column]}
        return outputs, audit
    for horizon in HORIZONS:
        target = targets_train[f"return_h{horizon}"]
        if family_id == "quantile_interval_value_v1":
            lower, upper = fit_predict_quantiles(
                x_train, target, x_test, seed=seed + horizon * 11
            )
            outputs[horizon] = {
                "predicted": (lower + upper) / 2.0,
                "lower": lower,
                "upper": upper,
            }
        else:
            model = (
                build_control_model()
                if family_id == CONTROL_ID
                else build_mean_model(family_id, n_jobs=n_jobs, seed=seed + horizon * 11)
            )
            outputs[horizon] = {
                "predicted": fit_predict_bps(model, x_train, target, x_test)
            }
    return outputs, audit


def _select_inner_configuration(
    family_id: str,
    x: pd.DataFrame,
    targets: pd.DataFrame,
    ends: pd.DataFrame,
    source_positions: np.ndarray,
    *,
    n_jobs: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    minimum = max(5_000, min(10_000, len(x) - 1_000))
    splits = list(
        _splitter(n_splits=2, min_train_size=minimum, max_train_rows=24_000).split(
            x.index, ends[f"end_h{MAX_LABEL_HORIZON}"]
        )
    )
    prediction_parts: dict[tuple[int, float], list[pd.DataFrame]] = {
        (horizon, threshold): []
        for horizon in HORIZONS
        for threshold in THRESHOLDS_BPS
    }
    fit_audits: list[dict[str, Any]] = []
    total_test_rows = 0
    for inner_fold, split in enumerate(splits, start=1):
        total_test_rows += len(split.test_positions)
        outputs, fit_audit = _fit_family_predictions(
            family_id,
            x.iloc[split.train_positions],
            targets.iloc[split.train_positions],
            x.iloc[split.test_positions],
            n_jobs=n_jobs,
            seed=seed + inner_fold * 1_000,
        )
        if fit_audit:
            fit_audits.append({"inner_fold": inner_fold, **fit_audit})
        for horizon in HORIZONS:
            result = outputs[horizon]
            for threshold in THRESHOLDS_BPS:
                prediction_parts[(horizon, threshold)].append(
                    execute_value_predictions(
                        timestamps=x.index,
                        source_positions=source_positions,
                        asset_returns=targets[f"return_h{horizon}"].to_numpy(dtype=float),
                        exit_times=ends[f"end_h{horizon}"],
                        predicted_returns=result["predicted"],
                        lower_returns=result.get("lower"),
                        upper_returns=result.get("upper"),
                        test_positions=split.test_positions,
                        horizon=horizon,
                        threshold_bps=threshold,
                        fold=inner_fold,
                        family_id=family_id,
                    )
                )
    diagnostics: list[dict[str, Any]] = []
    for (horizon, threshold), parts in prediction_parts.items():
        nonempty = [part for part in parts if len(part)]
        combined = pd.concat(nonempty).sort_index() if nonempty else pd.DataFrame(
            columns=["direction", "actual_direction", "net_return", "horizon_bars", "fold"]
        )
        metrics = aggregate_predictions(combined, total_test_rows=total_test_rows)
        diagnostics.append(
            {"horizon_bars": horizon, "threshold_bps": threshold, "metrics": metrics}
        )
    best = max(diagnostics, key=lambda row: inner_selection_key(row["metrics"]))
    frozen = {
        "family_id": family_id,
        "horizon_bars": int(best["horizon_bars"]),
        "threshold_bps": float(best["threshold_bps"]),
        "configuration_frozen_before_outer_test": True,
    }
    return frozen, {
        "inner_folds": len(splits),
        "selected_metrics": best["metrics"],
        "configuration_diagnostics": diagnostics,
        "fit_audits": fit_audits,
    }


def _outer_predictions(
    family_id: str,
    configuration: dict[str, Any],
    x: pd.DataFrame,
    targets: pd.DataFrame,
    ends: pd.DataFrame,
    source_positions: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    *,
    fold: int,
    n_jobs: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outputs, fit_audit = _fit_family_predictions(
        family_id,
        x.iloc[train_positions],
        targets.iloc[train_positions],
        x.iloc[test_positions],
        n_jobs=n_jobs,
        seed=seed,
    )
    horizon = int(configuration["horizon_bars"])
    result = outputs[horizon]
    predictions = execute_value_predictions(
        timestamps=x.index,
        source_positions=source_positions,
        asset_returns=targets[f"return_h{horizon}"].to_numpy(dtype=float),
        exit_times=ends[f"end_h{horizon}"],
        predicted_returns=result["predicted"],
        lower_returns=result.get("lower"),
        upper_returns=result.get("upper"),
        test_positions=test_positions,
        horizon=horizon,
        threshold_bps=float(configuration["threshold_bps"]),
        fold=fold,
        family_id=family_id,
    )
    return predictions, fit_audit


def _ranking_key(row: tuple[str, dict[str, Any]]) -> tuple[float, ...]:
    metrics = row[1]["metrics"]
    return (
        float(metrics["net_return_sum"]),
        float(metrics["profit_factor"]),
        -abs(float(metrics["max_drawdown_additive"])),
        float(metrics["accuracy_wilson_lower_95"]),
    )


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to repeat existing exploration artifact: {output}")
    if not 1 <= int(args.n_jobs) <= 16:
        raise ValueError("n-jobs must be 1..16")
    prompt = args.requirements_prompt.resolve()
    if not prompt.is_file():
        raise FileNotFoundError(prompt)
    data = _dataset(args)
    x = data["x"]
    targets = data["targets"]
    ends = data["ends"]
    source_positions = data["source_positions"]
    outer_splits = list(
        _splitter(n_splits=3, min_train_size=12_000, max_train_rows=30_000).split(
            x.index, ends[f"end_h{MAX_LABEL_HORIZON}"]
        )
    )
    if len(outer_splits) != 3:
        raise AssertionError("Direct-edge exploration requires exactly 3 outer folds")
    if args.preflight:
        import torch_directml

        if torch_directml.device_count() < 1:
            raise RuntimeError("DirectML GPU preflight failed")
        print(
            json.dumps(
                {
                    "status": "PASS_PREFLIGHT_NO_TRAINING",
                    "cycle_id": CYCLE_ID,
                    "development_rows": len(x),
                    "outer_folds": len(outer_splits),
                    "families": list(FAMILY_SPECS),
                    "gpu": torch_directml.device_name(0).replace("\x00", ""),
                    "holdout_access_count": 0,
                },
                indent=2,
            )
        )
        return 0

    family_ids = [CONTROL_ID, *FAMILY_SPECS]
    prediction_parts: dict[str, list[pd.DataFrame]] = {name: [] for name in family_ids}
    fold_records: dict[str, list[dict[str, Any]]] = {name: [] for name in family_ids}
    outer_audits: list[dict[str, Any]] = []
    total_test_rows = 0
    for fold, split in enumerate(outer_splits, start=1):
        total_test_rows += len(split.test_positions)
        audit = split.audit_dict()
        audit["train_start_time"] = x.index[split.train_positions[0]].isoformat()
        audit["train_end_time"] = x.index[split.train_positions[-1]].isoformat()
        outer_audits.append(audit)
        x_train = x.iloc[split.train_positions]
        targets_train = targets.iloc[split.train_positions]
        ends_train = ends.iloc[split.train_positions]
        source_train = source_positions[split.train_positions]
        for family_offset, family_id in enumerate(family_ids):
            print(f"[direct-edge] fold={fold}/3 family={family_id} phase=inner", flush=True)
            frozen, inner_audit = _select_inner_configuration(
                family_id,
                x_train,
                targets_train,
                ends_train,
                source_train,
                n_jobs=args.n_jobs,
                seed=15_200 + fold * 100 + family_offset * 10,
            )
            print(
                f"[direct-edge] fold={fold}/3 family={family_id} "
                f"frozen=h{frozen['horizon_bars']}/t{frozen['threshold_bps']}",
                flush=True,
            )
            predictions, fit_audit = _outer_predictions(
                family_id,
                frozen,
                x,
                targets,
                ends,
                source_positions,
                split.train_positions,
                split.test_positions,
                fold=fold,
                n_jobs=args.n_jobs,
                seed=16_200 + fold * 100 + family_offset * 10,
            )
            prediction_parts[family_id].append(predictions)
            fold_metrics = aggregate_predictions(
                predictions, total_test_rows=len(split.test_positions)
            )
            fold_records[family_id].append(
                {
                    "fold": fold,
                    "frozen_configuration": frozen,
                    "inner_selection": inner_audit,
                    "outer_fit_audit": fit_audit,
                    "outer_metrics": fold_metrics,
                }
            )

    combined: dict[str, pd.DataFrame] = {}
    aggregate: dict[str, dict[str, Any]] = {}
    for family_id, parts in prediction_parts.items():
        nonempty = [part for part in parts if len(part)]
        predictions = pd.concat(nonempty).sort_index() if nonempty else pd.DataFrame(
            columns=["direction", "actual_direction", "net_return", "horizon_bars", "fold"]
        )
        combined[family_id] = predictions
        aggregate[family_id] = aggregate_predictions(
            predictions, total_test_rows=total_test_rows
        )
    control_metrics = aggregate[CONTROL_ID]
    candidate_results: dict[str, dict[str, Any]] = {}
    for family_id, spec in FAMILY_SPECS.items():
        checks = exploration_survival_checks(aggregate[family_id], control_metrics)
        candidate_results[family_id] = {
            "specification": spec,
            "metrics": aggregate[family_id],
            "survival_checks": checks,
            "eligible_for_validation": all(checks.values()),
            "fold_records": fold_records[family_id],
        }
    survivors = [row for row in candidate_results.items() if row[1]["eligible_for_validation"]]
    selected = max(survivors, key=_ranking_key)[0] if survivors else None
    best_observed = max(candidate_results.items(), key=_ranking_key)[0]
    report_id = selected or best_observed
    decision = (
        "ESCALATE_BEST_EXPLORATION_SURVIVOR_TO_VALIDATION"
        if selected
        else "REJECT_ALL_DIRECT_NET_EDGE_CANDIDATES"
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cycle_id": CYCLE_ID,
        "status": "EXPLORATION_SURVIVOR_PENDING_VALIDATION" if selected else "ALL_CANDIDATES_REJECTED",
        "decision": decision,
        "current_mode": "EXPLORATION",
        "active_contract": "V15.1_LEAN",
        "research_only": True,
        "production_promotion_allowed": False,
        "live_trading_enabled": False,
        "auto_execution": False,
        "selected_for_validation_candidate_id": selected,
        "best_observed_candidate_id": best_observed,
        "report_candidate_id": report_id,
        "candidate_metrics": candidate_results[report_id]["metrics"],
        "control_metrics": control_metrics,
        "candidate_results": candidate_results,
        "control_fold_records": fold_records[CONTROL_ID],
        "protocol": {
            "pre_registered_before_fit": True,
            "objective_order": ["outer_oos_after_cost_profit", "drawdown_and_stability", "calibration", "accuracy"],
            "target": "fixed_horizon_close_to_close_gross_return",
            "horizons": list(HORIZONS),
            "inner_only_thresholds_bps": list(THRESHOLDS_BPS),
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "outer_folds": 3,
            "inner_folds": 2,
            "purge_and_embargo_bars": MAX_LABEL_HORIZON,
            "maximum_development_rows": MAX_DEVELOPMENT_ROWS,
            "development_rows": len(x),
            "development_cutoff": data["cutoff"].isoformat(),
            "total_outer_test_rows": total_test_rows,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_sha256": feature_schema_sha256(),
            "family_ids": list(FAMILY_SPECS),
            "paired_control_id": CONTROL_ID,
            "gpu_family": "directml_multihorizon_huber_v1",
            "gpu_required_and_smoke_tested": True,
            "saved_lockbox_predictions_accessed": False,
            "sealed_holdout_access_count": 0,
        },
        "source_provenance": {
            "m5": {"path": str(args.m5.resolve()), "sha256": sha256_file(args.m5)},
            "reference_summary": {"path": str(data["reference_path"]), "sha256": sha256_file(data["reference_path"])},
            "requirements_prompt": {"path": str(prompt), "sha256": sha256_file(prompt)},
            "trainer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "core": {"path": str((PROJECT_ROOT / "src/gold_ai/direct_net_edge.py").resolve()), "sha256": sha256_file(PROJECT_ROOT / "src/gold_ai/direct_net_edge.py")},
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
    for family_id, predictions in combined.items():
        predictions.to_csv(temporary / f"predictions_{family_id}.csv", index_label="timestamp")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    print(json.dumps({"decision": decision, "selected_for_validation": selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
