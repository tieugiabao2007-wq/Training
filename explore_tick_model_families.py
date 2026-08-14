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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.data.providers import load_saved_market_data, sha256_file
from gold_ai.data.tick_features import TICK_FEATURE_COLUMNS, TICK_FEATURE_SCHEMA_VERSION
from gold_ai.labels import triple_barrier_events
from gold_ai.metrics import triple_barrier_trading_metrics, wilson_lower
from gold_ai.validation import PurgedWalkForwardSplit


CYCLE_ID = "XAUUSD_TICK_21D_MULTI_FAMILY_CYCLE_2"
HORIZON_BARS = 12
INTERVAL = pd.Timedelta(minutes=5)
ROUND_TRIP_COST_BPS = 3.0
CONFIDENCE_THRESHOLD = 1.0 / 3.0
SAVED_LOGISTIC_ACCURACY = 0.4868421052631579
FAMILY_SPECS: dict[str, list[dict[str, Any]]] = {
    "TICK_HGB_LIQUIDITY_THRESHOLDS": [
        {"max_leaf_nodes": leaves, "learning_rate": rate}
        for leaves in (7, 15)
        for rate in (0.05, 0.10)
    ],
    "TICK_EXTRA_TREES_STATE_PARTITIONS": [
        {"min_samples_leaf": leaf, "max_features": features}
        for leaf in (20, 50)
        for features in ("sqrt", 0.75)
    ],
    "TICK_POLYNOMIAL_LOGIT_INTERACTIONS": [
        {"C": value} for value in (0.05, 0.20, 1.0)
    ],
    "TICK_KNN_LOCAL_ANALOGS": [
        {"n_neighbors": value} for value in (25, 75, 150)
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded multi-family tick exploration with inner-only tuning."
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=16)
    return parser.parse_args()


def align_probabilities(model, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    aligned = np.zeros((len(x), 3), dtype=float)
    for column, label in enumerate(model.classes_):
        aligned[:, int(label) + 1] = raw[:, column]
    return aligned


def build_model(family: str, params: dict[str, Any], n_jobs: int):
    if family == "TICK_HGB_LIQUIDITY_THRESHOLDS":
        return HistGradientBoostingClassifier(
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            learning_rate=float(params["learning_rate"]),
            max_iter=120,
            min_samples_leaf=30,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=False,
            random_state=42,
        )
    if family == "TICK_EXTRA_TREES_STATE_PARTITIONS":
        return ExtraTreesClassifier(
            n_estimators=240,
            max_depth=8,
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=42,
        )
    if family == "TICK_POLYNOMIAL_LOGIT_INTERACTIONS":
        return make_pipeline(
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            StandardScaler(),
            LogisticRegression(
                C=float(params["C"]),
                class_weight="balanced",
                max_iter=1_500,
                random_state=42,
            ),
        )
    if family == "TICK_KNN_LOCAL_ANALOGS":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=int(params["n_neighbors"]),
                weights="distance",
                n_jobs=n_jobs,
            ),
        )
    raise KeyError(f"Unknown family: {family}")


def fixed_logistic_control():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1_000,
            random_state=42,
        ),
    )


def profit_factor(predictions: pd.DataFrame) -> float:
    gains = predictions.loc[predictions["net_return"] > 0, "net_return"].sum()
    losses = -predictions.loc[predictions["net_return"] < 0, "net_return"].sum()
    if losses > 0:
        return float(gains / losses)
    return float("inf") if gains > 0 else 0.0


def compact_metrics(predictions: pd.DataFrame, total_test_rows: int) -> dict[str, Any]:
    trades = len(predictions)
    successes = int(predictions["correct"].sum()) if trades else 0
    fold_counts = predictions.groupby("fold").size() if trades else pd.Series(dtype=int)
    return {
        "directional_accuracy": successes / trades if trades else 0.0,
        "accuracy_wilson_lower_95": wilson_lower(successes, trades),
        "trades": trades,
        "buy_trades": int((predictions["direction"] == 1).sum()) if trades else 0,
        "sell_trades": int((predictions["direction"] == -1).sum()) if trades else 0,
        "coverage": float(min(1.0, trades * HORIZON_BARS / max(total_test_rows, 1))),
        "net_return_sum": float(predictions["net_return"].sum()) if trades else 0.0,
        "mean_net_return": float(predictions["net_return"].mean()) if trades else 0.0,
        "profit_factor": profit_factor(predictions),
        "win_rate": float((predictions["net_return"] > 0).mean()) if trades else 0.0,
        "valid_outer_folds": int((fold_counts >= 20).sum()),
        "positive_return_folds": int(
            (predictions.groupby("fold")["net_return"].sum() > 0).sum()
        )
        if trades
        else 0,
    }


def load_research_frame(
    feature_path: Path, gold_path: Path
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict, Any]:
    metadata_path = feature_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(feature_path) != metadata["file_sha256"]:
        raise ValueError("Tick feature checksum mismatch")
    if metadata["schema_version"] != TICK_FEATURE_SCHEMA_VERSION:
        raise ValueError("Tick feature schema mismatch")
    if int(metadata["source_nonempty_days"]) != 21:
        raise ValueError("Cycle 2 is frozen to the validated 21-day feature snapshot")
    features = pd.read_csv(feature_path, parse_dates=["decision_time"]).set_index(
        "decision_time"
    )
    features.index = pd.to_datetime(features.index, utc=True)
    if tuple(features.columns) != TICK_FEATURE_COLUMNS:
        raise ValueError("Unexpected feature columns")
    if features.index.has_duplicates or not features.index.is_monotonic_increasing:
        raise ValueError("Feature timestamps must be unique and monotonic")

    gold, gold_metadata = load_saved_market_data(gold_path)
    if gold_metadata.instrument != "XAUUSDm" or gold_metadata.interval != "M5":
        raise ValueError("Labels require exact XAUUSDm M5")
    events = triple_barrier_events(
        gold,
        horizon_bars=HORIZON_BARS,
        atr_multiplier=1.0,
        round_trip_cost_bps=ROUND_TRIP_COST_BPS,
        minimum_edge_bps=2.0,
    )
    matching_bar_times = pd.DatetimeIndex(features.index - INTERVAL)
    aligned = events.reindex(matching_bar_times).copy()
    aligned.index = features.index
    aligned["label_end_time"] = (
        pd.to_datetime(aligned["label_end_time"], utc=True, errors="coerce")
        + INTERVAL
    )
    valid = pd.Series(
        np.isfinite(features.to_numpy(dtype=float)).all(axis=1),
        index=features.index,
    ) & aligned["label"].notna()
    x = features.loc[valid].astype(float)
    y = aligned.loc[valid, "label"].astype("int8")
    outcomes = aligned.loc[
        valid,
        ["event_return", "barrier_pct", "duration_bars", "ambiguous", "label_end_time"],
    ]
    return x, y, outcomes, metadata, gold_metadata


def evaluate_probabilities(
    probabilities: np.ndarray,
    y: pd.Series,
    outcomes: pd.DataFrame,
    fold: int,
) -> tuple[dict, pd.DataFrame]:
    metrics, predictions = triple_barrier_trading_metrics(
        probabilities,
        y,
        outcomes,
        threshold=CONFIDENCE_THRESHOLD,
        horizon_bars=HORIZON_BARS,
        round_trip_cost_bps=ROUND_TRIP_COST_BPS,
    )
    predictions = predictions.copy()
    predictions["fold"] = fold
    return metrics, predictions


def select_inner_configuration(
    family: str,
    x: pd.DataFrame,
    y: pd.Series,
    outcomes: pd.DataFrame,
    outer_train_positions: np.ndarray,
    outer_fold: int,
    n_jobs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    train_x = x.iloc[outer_train_positions]
    train_y = y.iloc[outer_train_positions]
    train_outcomes = outcomes.iloc[outer_train_positions]
    inner_splitter = PurgedWalkForwardSplit(
        n_splits=2,
        min_train_size=500,
        embargo_bars=HORIZON_BARS,
        target_horizon_bars=HORIZON_BARS,
        feature_schema_version=f"{TICK_FEATURE_SCHEMA_VERSION}_{family}_inner",
    )
    trials: list[dict[str, Any]] = []
    for trial_index, params in enumerate(FAMILY_SPECS[family], start=1):
        parts: list[pd.DataFrame] = []
        inner_audits: list[dict[str, Any]] = []
        for inner_fold, split in enumerate(
            inner_splitter.split(train_x.index, train_outcomes["label_end_time"]),
            start=1,
        ):
            model = build_model(family, params, n_jobs)
            model.fit(train_x.iloc[split.train_positions], train_y.iloc[split.train_positions])
            probabilities = align_probabilities(model, train_x.iloc[split.test_positions])
            _, predictions = evaluate_probabilities(
                probabilities,
                train_y.iloc[split.test_positions],
                train_outcomes.iloc[split.test_positions],
                inner_fold,
            )
            parts.append(predictions)
            audit = split.audit_dict()
            audit["train_start_time"] = train_x.index[split.train_positions[0]].isoformat()
            audit["train_end_time"] = train_x.index[split.train_positions[-1]].isoformat()
            inner_audits.append(audit)
        combined = pd.concat(parts).sort_index()
        metrics = compact_metrics(
            combined, sum(int(row["test_rows"]) for row in inner_audits)
        )
        eligible = (
            metrics["trades"] >= 50
            and metrics["buy_trades"] >= 10
            and metrics["sell_trades"] >= 10
            and metrics["valid_outer_folds"] == 2
        )
        trials.append(
            {
                "outer_fold": outer_fold,
                "trial": trial_index,
                "params": params,
                "selection_eligible": eligible,
                "inner_metrics": metrics,
                "inner_audits": inner_audits,
            }
        )
    selected = max(
        trials,
        key=lambda row: (
            int(row["selection_eligible"]),
            row["inner_metrics"]["net_return_sum"],
            row["inner_metrics"]["directional_accuracy"],
            row["inner_metrics"]["trades"],
            -row["trial"],
        ),
    )
    return dict(selected["params"]), trials, bool(selected["selection_eligible"])


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to repeat exploration cycle: {output}")
    feature_path = args.features.resolve()
    gold_path = args.gold.resolve()
    baseline_summary_path = args.baseline_artifact.resolve() / "summary.json"
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    if baseline_summary["family_id"] != "XAUUSD_M5_EXACT_TICK_MICROSTRUCTURE_V15_1":
        raise ValueError("Unexpected baseline family")
    if baseline_summary["protocol"]["sealed_holdout_access_count"] != 0:
        raise ValueError("Baseline holdout access must be zero")

    x, y, outcomes, feature_metadata, gold_metadata = load_research_frame(
        feature_path, gold_path
    )
    outer_splitter = PurgedWalkForwardSplit(
        n_splits=3,
        min_train_size=1_200,
        embargo_bars=HORIZON_BARS,
        target_horizon_bars=HORIZON_BARS,
        feature_schema_version=f"{TICK_FEATURE_SCHEMA_VERSION}_{CYCLE_ID}",
    )
    outer_splits = list(outer_splitter.split(x.index, outcomes["label_end_time"]))
    family_parts: dict[str, list[pd.DataFrame]] = {name: [] for name in FAMILY_SPECS}
    family_folds: dict[str, list[dict]] = {name: [] for name in FAMILY_SPECS}
    family_trials: dict[str, list[dict]] = {name: [] for name in FAMILY_SPECS}
    control_parts: list[pd.DataFrame] = []
    control_folds: list[dict] = []

    for outer_fold, split in enumerate(outer_splits, start=1):
        audit = split.audit_dict()
        audit["train_start_time"] = x.index[split.train_positions[0]].isoformat()
        audit["train_end_time"] = x.index[split.train_positions[-1]].isoformat()
        control = fixed_logistic_control()
        control.fit(x.iloc[split.train_positions], y.iloc[split.train_positions])
        control_probability = align_probabilities(control, x.iloc[split.test_positions])
        control_metric, control_prediction = evaluate_probabilities(
            control_probability,
            y.iloc[split.test_positions],
            outcomes.iloc[split.test_positions],
            outer_fold,
        )
        control_parts.append(control_prediction)
        control_folds.append({"fold": outer_fold, "audit": audit, "metrics": control_metric})

        for family in FAMILY_SPECS:
            params, trials, eligible = select_inner_configuration(
                family,
                x,
                y,
                outcomes,
                split.train_positions,
                outer_fold,
                args.n_jobs,
            )
            model = build_model(family, params, args.n_jobs)
            model.fit(x.iloc[split.train_positions], y.iloc[split.train_positions])
            probabilities = align_probabilities(model, x.iloc[split.test_positions])
            metrics, predictions = evaluate_probabilities(
                probabilities,
                y.iloc[split.test_positions],
                outcomes.iloc[split.test_positions],
                outer_fold,
            )
            family_parts[family].append(predictions)
            family_trials[family].extend(trials)
            family_folds[family].append(
                {
                    "fold": outer_fold,
                    "audit": audit,
                    "selected_params": params,
                    "inner_selection_eligible": eligible,
                    "metrics": metrics,
                }
            )

    total_test_rows = sum(int(row["audit"]["test_rows"]) for row in control_folds)
    control_predictions = pd.concat(control_parts).sort_index()
    control_metrics = compact_metrics(control_predictions, total_test_rows)
    family_results: dict[str, dict] = {}
    family_prediction_frames: dict[str, pd.DataFrame] = {}
    for family in FAMILY_SPECS:
        predictions = pd.concat(family_parts[family]).sort_index()
        family_prediction_frames[family] = predictions
        metrics = compact_metrics(predictions, total_test_rows)
        metrics["accuracy_delta_vs_paired_control_pp"] = 100 * (
            metrics["directional_accuracy"] - control_metrics["directional_accuracy"]
        )
        metrics["accuracy_delta_vs_saved_logistic_pp"] = 100 * (
            metrics["directional_accuracy"] - SAVED_LOGISTIC_ACCURACY
        )
        checks = {
            "all_outer_inner_selections_eligible": all(
                row["inner_selection_eligible"] for row in family_folds[family]
            ),
            "accuracy_above_paired_control": metrics["directional_accuracy"]
            > control_metrics["directional_accuracy"],
            "accuracy_above_saved_logistic": metrics["directional_accuracy"]
            > SAVED_LOGISTIC_ACCURACY,
            "positive_after_costs": metrics["net_return_sum"] > 0,
            "profit_factor_above_1_05": metrics["profit_factor"] > 1.05,
            "minimum_buy_trades": metrics["buy_trades"] >= 50,
            "minimum_sell_trades": metrics["sell_trades"] >= 50,
            "all_three_outer_folds_valid": metrics["valid_outer_folds"] == 3,
            "at_least_two_positive_return_folds": metrics["positive_return_folds"] >= 2,
        }
        survived = all(checks.values())
        family_results[family] = {
            "exploration_result": "SURVIVED" if survived else "REJECTED",
            "validation_eligible": survived,
            "production_improvement_claim_allowed": False,
            "metrics": metrics,
            "checks": checks,
            "outer_folds": family_folds[family],
            "inner_trials": family_trials[family],
        }

    ranked = sorted(
        family_results,
        key=lambda family: (
            int(family_results[family]["validation_eligible"]),
            family_results[family]["metrics"]["net_return_sum"],
            family_results[family]["metrics"]["directional_accuracy"],
        ),
        reverse=True,
    )
    escalations = [
        family for family in ranked if family_results[family]["validation_eligible"]
    ][:2]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cycle_id": CYCLE_ID,
        "current_mode": "EXPLORATION",
        "status": "EXPLORATION_CYCLE_COMPLETE",
        "research_only": True,
        "data_limit_status": "DATA_LIMITED_21_DAYS",
        "source_mismatch": False,
        "engineering_ready": True,
        "research_cycle_complete": True,
        "accuracy_target_status": "INCONCLUSIVE_EXPLORATION_ONLY",
        "production_certification_status": "NOT_READY",
        "production_improvement_claim_allowed": False,
        "protocol": {
            "outer_folds": 3,
            "inner_folds_per_outer": 2,
            "outer_used_for_tuning": False,
            "purge_and_embargo_bars": HORIZON_BARS,
            "horizon_bars": HORIZON_BARS,
            "atr_barrier_multiplier": 1.0,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "feature_schema_version": TICK_FEATURE_SCHEMA_VERSION,
            "family_config_counts": {
                family: len(configs) for family, configs in FAMILY_SPECS.items()
            },
            "maximum_family_configs": 4,
            "sealed_holdout_access_count": 0,
            "saved_lockbox_predictions_accessed": False,
        },
        "data_quality": {
            "source_nonempty_days": int(feature_metadata["source_nonempty_days"]),
            "aligned_finite_labeled_rows": int(len(x)),
            "label_distribution": {
                str(int(label)): float(rate)
                for label, rate in y.value_counts(normalize=True).sort_index().items()
            },
        },
        "source_provenance": {
            "tick_features": {
                "path": str(feature_path),
                "sha256": sha256_file(feature_path),
                "metadata_path": str(feature_path.with_suffix(".metadata.json")),
                "metadata_sha256": sha256_file(feature_path.with_suffix(".metadata.json")),
            },
            "gold_bars": {
                "path": str(gold_path),
                "sha256": sha256_file(gold_path),
                "provider": gold_metadata.provider,
            },
            "saved_logistic_baseline_summary": {
                "path": str(baseline_summary_path),
                "sha256": sha256_file(baseline_summary_path),
                "directional_accuracy": baseline_summary["metrics"]["directional_accuracy"],
                "reused_without_refit": True,
            },
        },
        "paired_logistic_control_metrics": control_metrics,
        "families": family_results,
        "ranking": ranked,
        "validation_escalations": escalations,
        "lockbox_status": "SEALED_NOT_ACCESSED",
    }

    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    (temporary / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (temporary / "control_folds.json").write_text(
        json.dumps(control_folds, indent=2, default=str), encoding="utf-8"
    )
    control_predictions.to_csv(
        temporary / "predictions_paired_logistic_control.csv",
        index_label="decision_time",
    )
    for family, predictions in family_prediction_frames.items():
        predictions.to_csv(
            temporary / f"predictions_{family.lower()}.csv",
            index_label="decision_time",
        )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "cycle_id": CYCLE_ID,
                "status": summary["status"],
                "paired_logistic_control_metrics": control_metrics,
                "ranking": ranked,
                "validation_escalations": escalations,
                "family_results": {
                    family: {
                        "result": row["exploration_result"],
                        "metrics": row["metrics"],
                    }
                    for family, row in family_results.items()
                },
                "artifact": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
