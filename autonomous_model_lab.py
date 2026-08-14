from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gold_ai.autonomous_lab import (
    CANDIDATE_CATALOG,
    FEATURE_SCHEMA_VERSION,
    FIXED_CONFIDENCE_THRESHOLD,
    HORIZON_BARS,
    MAX_DEVELOPMENT_ROWS,
    ROUND_TRIP_COST_BPS,
    aggregate_executed_predictions,
    align_probabilities,
    build_candidate_model,
    build_causal_m5_features,
    build_control_model,
    feature_schema_columns,
    feature_schema_sha256,
    promotion_checks,
)
from gold_ai.data.providers import load_saved_market_data, sha256_file
from gold_ai.labels import triple_barrier_events
from gold_ai.metrics import triple_barrier_trading_metrics
from gold_ai.validation import PurgedWalkForwardSplit


CATALOG_ID = "XAUUSDM_M5_AUTONOMOUS_LOCAL_CATALOG_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a frozen local-only M5 candidate catalog. No Codex/API, live trading "
            "or sealed holdout access is used."
        )
    )
    parser.add_argument("--m5", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--requirements-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate exact data, prompt, feature rows and folds without fitting or writing artifacts.",
    )
    return parser.parse_args()


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return dict(default)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _incumbent_metrics() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path = PROJECT_ROOT / "artifacts" / "autonomous_lab" / "incumbent.json"
    payload = _read_json(path, {})
    metrics = payload.get("metrics")
    return payload or None, metrics if isinstance(metrics, dict) else None


def _fold_prediction(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    outcomes: pd.DataFrame,
    test_positions: np.ndarray,
    fold: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    probabilities = align_probabilities(model, x.iloc[test_positions])
    metrics, predictions = triple_barrier_trading_metrics(
        probabilities,
        y.iloc[test_positions],
        outcomes.iloc[test_positions],
        threshold=FIXED_CONFIDENCE_THRESHOLD,
        horizon_bars=HORIZON_BARS,
        round_trip_cost_bps=ROUND_TRIP_COST_BPS,
    )
    predictions = predictions.copy()
    predictions["fold"] = fold
    return metrics, predictions


def _ranking_key(row: tuple[str, dict[str, Any]]) -> tuple[float, ...]:
    _, payload = row
    metrics = payload["metrics"]
    return (
        float(metrics["net_return_sum"]),
        float(metrics["profit_factor"]),
        float(metrics["accuracy_wilson_lower_95"]),
        float(metrics["directional_accuracy"]),
    )


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to repeat an existing catalog artifact: {output}")
    if not 1 <= int(args.n_jobs) <= 16:
        raise ValueError("n-jobs must be between 1 and 16")
    prompt_path = args.requirements_prompt.resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Requirements prompt not found: {prompt_path}")

    m5, metadata = load_saved_market_data(args.m5)
    if metadata.instrument != "XAUUSDm" or metadata.interval != "M5":
        raise ValueError("Source must be the exact legal Exness XAUUSDm M5 snapshot")
    if not m5.index.is_monotonic_increasing or m5.index.has_duplicates:
        raise ValueError("Exact M5 timestamps must be unique and chronological")

    reference_summary_path = args.reference_artifact.resolve() / "summary.json"
    reference = _read_json(reference_summary_path, {})
    fold_rows = reference.get("fold_metrics")
    if not isinstance(fold_rows, list) or not fold_rows:
        raise ValueError("Reference summary has no development fold boundary")
    last_fold = max(fold_rows, key=lambda row: int(row["fold"]))
    development_cutoff = _utc_timestamp(last_fold["test_end"])
    label_data_cutoff = development_cutoff + pd.Timedelta(
        minutes=5 * HORIZON_BARS
    )
    label_bars = m5.loc[:label_data_cutoff]
    decision_bars = label_bars.loc[label_bars.index <= development_cutoff]
    point = float(metadata.symbol_spec.get("point", 0.0) or 0.0)
    features = build_causal_m5_features(decision_bars, point)
    events = triple_barrier_events(
        label_bars,
        horizon_bars=HORIZON_BARS,
        atr_multiplier=1.0,
        round_trip_cost_bps=ROUND_TRIP_COST_BPS,
        minimum_edge_bps=2.0,
    ).reindex(decision_bars.index)
    finite = np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    valid = pd.Series(finite, index=features.index) & events["label"].notna()
    x = features.loc[valid].astype(float).tail(MAX_DEVELOPMENT_ROWS)
    y = events.loc[x.index, "label"].astype("int8")
    outcomes = events.loc[
        x.index,
        ["event_return", "barrier_pct", "duration_bars", "ambiguous", "label_end_time"],
    ]
    if len(x) < 15_000 or y.nunique() != 3:
        raise ValueError("Insufficient complete development rows/classes for the frozen catalog")

    splitter = PurgedWalkForwardSplit(
        n_splits=3,
        min_train_size=12_000,
        embargo_bars=HORIZON_BARS,
        max_train_rows=30_000,
        target_horizon_bars=HORIZON_BARS,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    splits = list(splitter.split(x.index, outcomes["label_end_time"]))
    if len(splits) != 3:
        raise AssertionError("Frozen catalog requires exactly three outer folds")
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "PASS_PREFLIGHT_NO_TRAINING_PERFORMED",
                    "catalog_id": CATALOG_ID,
                    "development_rows": len(x),
                    "candidate_ids": list(CANDIDATE_CATALOG),
                    "outer_folds": len(splits),
                    "development_cutoff": development_cutoff.isoformat(),
                    "sealed_holdout_access_count": 0,
                },
                indent=2,
            )
        )
        return 0

    fold_audits: list[dict[str, Any]] = []
    control_parts: list[pd.DataFrame] = []
    control_fold_metrics: list[dict[str, Any]] = []
    candidate_parts: dict[str, list[pd.DataFrame]] = {
        candidate_id: [] for candidate_id in CANDIDATE_CATALOG
    }
    candidate_fold_metrics: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id in CANDIDATE_CATALOG
    }

    for fold, split in enumerate(splits, start=1):
        audit = split.audit_dict()
        audit["train_start_time"] = x.index[split.train_positions[0]].isoformat()
        audit["train_end_time"] = x.index[split.train_positions[-1]].isoformat()
        fold_audits.append(audit)

        control = build_control_model()
        control.fit(x.iloc[split.train_positions], y.iloc[split.train_positions])
        control_metrics, control_predictions = _fold_prediction(
            control, x, y, outcomes, split.test_positions, fold
        )
        control_metrics["fold"] = fold
        control_fold_metrics.append(control_metrics)
        control_parts.append(control_predictions)

        for candidate_id in CANDIDATE_CATALOG:
            print(f"[catalog] fold={fold}/3 candidate={candidate_id}", flush=True)
            model = build_candidate_model(candidate_id, n_jobs=args.n_jobs)
            model.fit(x.iloc[split.train_positions], y.iloc[split.train_positions])
            metrics, predictions = _fold_prediction(
                model, x, y, outcomes, split.test_positions, fold
            )
            metrics["fold"] = fold
            candidate_fold_metrics[candidate_id].append(metrics)
            candidate_parts[candidate_id].append(predictions)

    total_test_rows = sum(int(row["test_rows"]) for row in fold_audits)
    control_predictions = pd.concat(control_parts).sort_index()
    control_metrics = aggregate_executed_predictions(
        control_predictions, total_test_rows=total_test_rows
    )
    incumbent_record, incumbent_metrics = _incumbent_metrics()
    candidate_results: dict[str, dict[str, Any]] = {}
    combined_predictions: dict[str, pd.DataFrame] = {}
    for candidate_id, spec in CANDIDATE_CATALOG.items():
        predictions = pd.concat(candidate_parts[candidate_id]).sort_index()
        metrics = aggregate_executed_predictions(
            predictions, total_test_rows=total_test_rows
        )
        checks = promotion_checks(metrics, control_metrics, incumbent_metrics)
        candidate_results[candidate_id] = {
            "specification": spec,
            "metrics": metrics,
            "checks": checks,
            "eligible_for_lab_keep": all(checks.values()),
            "fold_metrics": candidate_fold_metrics[candidate_id],
        }
        combined_predictions[candidate_id] = predictions

    eligible = [
        (candidate_id, payload)
        for candidate_id, payload in candidate_results.items()
        if payload["eligible_for_lab_keep"]
    ]
    winner_id = max(eligible, key=_ranking_key)[0] if eligible else None
    best_observed_id = max(candidate_results.items(), key=_ranking_key)[0]
    report_candidate_id = winner_id or best_observed_id
    report_metrics = candidate_results[report_candidate_id]["metrics"]
    decision = (
        "KEEP_BETTER_LAB_INCUMBENT_PENDING_INDEPENDENT_VALIDATOR"
        if winner_id
        else "REJECT_ALL_NOT_BETTER"
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_id": CATALOG_ID,
        "status": "LAB_WINNER_PENDING_VALIDATION" if winner_id else "ALL_CANDIDATES_REJECTED",
        "decision": decision,
        "active_contract": "V15.1_LEAN",
        "execution_mode": "LOCAL_PYTHON_ONLY_NO_CODEX_OR_EXTERNAL_AI_API",
        "research_only": True,
        "production_promotion_allowed": False,
        "live_trading_enabled": False,
        "auto_execution": False,
        "winner_candidate_id": winner_id,
        "best_observed_candidate_id": best_observed_id,
        "report_candidate_id": report_candidate_id,
        "candidate_metrics": report_metrics,
        "control_metrics": control_metrics,
        "candidate_results": candidate_results,
        "protocol": {
            "catalog_frozen_before_run": True,
            "candidate_count": len(CANDIDATE_CATALOG),
            "candidate_ids": list(CANDIDATE_CATALOG),
            "hyperparameter_trials": 0,
            "paired_fixed_control": "standardized_balanced_multinomial_logistic",
            "confidence_threshold": FIXED_CONFIDENCE_THRESHOLD,
            "horizon_bars": HORIZON_BARS,
            "atr_barrier_multiplier": 1.0,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "outer_folds": 3,
            "minimum_train_rows": 12_000,
            "maximum_train_rows": 30_000,
            "maximum_development_rows": MAX_DEVELOPMENT_ROWS,
            "purge_and_embargo_bars": HORIZON_BARS,
            "development_cutoff": development_cutoff.isoformat(),
            "development_rows": int(len(x)),
            "total_outer_test_rows": int(total_test_rows),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_columns": list(feature_schema_columns()),
            "feature_schema_sha256": feature_schema_sha256(),
            "sealed_holdout_access_count": 0,
            "saved_lockbox_predictions_accessed": False,
        },
        "incumbent_before_run": incumbent_record,
        "source_provenance": {
            "m5": {"path": str(args.m5.resolve()), "sha256": sha256_file(args.m5)},
            "reference_summary": {
                "path": str(reference_summary_path),
                "sha256": sha256_file(reference_summary_path),
                "access_purpose": "development_cutoff_and_public_baseline_only",
            },
            "requirements_prompt": {
                "path": str(prompt_path),
                "sha256": sha256_file(prompt_path),
                "role": "non_conflicting_requirements_catalog_only",
            },
        },
        "fold_audits": fold_audits,
        "control_fold_metrics": control_fold_metrics,
        "lockbox_status": "SEALED_NOT_ACCESSED",
    }

    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    (temporary / "candidates").mkdir()
    (temporary / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    control_predictions.to_csv(
        temporary / "predictions_control.csv", index_label="timestamp"
    )
    for candidate_id, predictions in combined_predictions.items():
        candidate_dir = temporary / "candidates" / candidate_id
        candidate_dir.mkdir()
        predictions.to_csv(candidate_dir / "predictions.csv", index_label="timestamp")
        (candidate_dir / "fold_metrics.json").write_text(
            json.dumps(
                candidate_fold_metrics[candidate_id],
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

    if winner_id:
        print(f"[catalog] fitting frozen lab winner={winner_id}", flush=True)
        final_model = build_candidate_model(winner_id, n_jobs=args.n_jobs)
        final_model.fit(x, y)
        joblib.dump(final_model, temporary / "winner_model.joblib")
        (temporary / "winner_model_manifest.json").write_text(
            json.dumps(
                {
                    "candidate_id": winner_id,
                    "research_only": True,
                    "production_execution_allowed": False,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "feature_schema_sha256": feature_schema_sha256(),
                    "training_rows": int(len(x)),
                    "training_end_utc": x.index[-1].isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    print(json.dumps({"decision": decision, "winner_candidate_id": winner_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
