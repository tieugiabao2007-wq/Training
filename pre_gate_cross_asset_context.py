from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.data.cross_asset_context import join_strict_lag_context
from gold_ai.data.providers import load_saved_market_data, sha256_file
from gold_ai.labels import triple_barrier_events
from gold_ai.metrics import triple_barrier_trading_metrics, wilson_lower
from gold_ai.research_status import classify_accuracy_target
from gold_ai.validation import PurgedWalkForwardSplit


FAMILY_ID = "XAUUSD_M5_EXACT_CROSS_ASSET_LEAD_V15_1"
HORIZON_BARS = 12
INTERVAL_SECONDS = 300
ROUND_TRIP_COST_BPS = 3.0
FIXED_CONFIDENCE_THRESHOLD = 1.0 / 3.0
REFERENCE_BASELINE_ACCURACY = 0.5048749470114455
MINIMUM_ACCURACY_DELTA = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed V15.1 cross-asset pre-gate.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--dxy", type=Path, required=True)
    parser.add_argument("--xag", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def align_probabilities(model, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    aligned = np.zeros((len(x), 3), dtype=float)
    for column, label in enumerate(model.classes_):
        aligned[:, int(label) + 1] = raw[:, column]
    return aligned


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to repeat completed pre-gate: {output}")

    gold, gold_meta = load_saved_market_data(args.gold)
    dxy, dxy_meta = load_saved_market_data(args.dxy)
    xag, xag_meta = load_saved_market_data(args.xag)
    if gold_meta.instrument != "XAUUSDm" or dxy_meta.instrument != "DXYm" or xag_meta.instrument != "XAGUSDm":
        raise ValueError("Pre-gate requires exact XAUUSDm, DXYm and XAGUSDm snapshots")
    if {gold_meta.interval, dxy_meta.interval, xag_meta.interval} != {"M5"}:
        raise ValueError("All pre-gate snapshots must be M5")

    reference = json.loads((args.reference_artifact / "summary.json").read_text(encoding="utf-8"))
    fold_rows = sorted(reference["fold_metrics"], key=lambda row: int(row["fold"]))
    development_cutoff = pd.Timestamp(fold_rows[-1]["test_end"]).tz_convert("UTC")
    label_data_cutoff = development_cutoff + pd.Timedelta(minutes=5 * HORIZON_BARS)
    gold = gold.loc[:label_data_cutoff]
    decision_index = gold.index[gold.index <= development_cutoff]

    context = join_strict_lag_context(
        decision_index,
        {"dxy": dxy, "xag": xag},
        interval_seconds=INTERVAL_SECONDS,
    )
    context["metal_usd_pressure_3"] = context["xag_ret_3"] - context["dxy_ret_3"]
    events = triple_barrier_events(
        gold,
        horizon_bars=HORIZON_BARS,
        atr_multiplier=1.0,
        round_trip_cost_bps=ROUND_TRIP_COST_BPS,
        minimum_edge_bps=2.0,
    ).reindex(decision_index)
    complete_context = context.notna().all(axis=1)
    valid = complete_context & events["label"].notna()
    x = context.loc[valid]
    y = events.loc[valid, "label"].astype("int8")
    outcomes = events.loc[
        valid,
        ["event_return", "barrier_pct", "duration_bars", "ambiguous", "label_end_time"],
    ]
    overlap = float(complete_context.mean())
    splitter = PurgedWalkForwardSplit(
        n_splits=5,
        min_train_size=5_000,
        embargo_bars=HORIZON_BARS,
        target_horizon_bars=HORIZON_BARS,
        feature_schema_version="exact_cross_asset_strict_lag_v1",
    )
    fold_metrics: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    coefficient_signs: dict[str, list[int]] = {column: [] for column in x.columns}
    for fold_number, split in enumerate(splitter.split(x.index, outcomes["label_end_time"]), start=1):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1_000,
                random_state=42,
            ),
        )
        model.fit(x.iloc[split.train_positions], y.iloc[split.train_positions])
        test_x = x.iloc[split.test_positions]
        probabilities = align_probabilities(model, test_x)
        metrics, predictions = triple_barrier_trading_metrics(
            probabilities,
            y.iloc[split.test_positions],
            outcomes.iloc[split.test_positions],
            threshold=FIXED_CONFIDENCE_THRESHOLD,
            horizon_bars=HORIZON_BARS,
            round_trip_cost_bps=ROUND_TRIP_COST_BPS,
        )
        classifier = model.named_steps["logisticregression"]
        class_to_row = {int(label): row for row, label in enumerate(classifier.classes_)}
        directional_coef = classifier.coef_[class_to_row[1]] - classifier.coef_[class_to_row[-1]]
        for feature, value in zip(x.columns, directional_coef):
            coefficient_signs[feature].append(int(np.sign(value)))
        metrics.update({"fold": fold_number, "audit": split.audit_dict()})
        fold_metrics.append(metrics)
        predictions = predictions.copy()
        predictions["fold"] = fold_number
        prediction_parts.append(predictions)

    predictions = pd.concat(prediction_parts).sort_index()
    trades = len(predictions)
    successes = int(predictions["correct"].sum())
    accuracy = successes / trades if trades else 0.0
    buy_trades = int((predictions["direction"] == 1).sum())
    sell_trades = int((predictions["direction"] == -1).sum())
    coverage = float(min(1.0, trades * HORIZON_BARS / max(sum(row["audit"]["test_rows"] for row in fold_metrics), 1)))
    gains = predictions.loc[predictions["net_return"] > 0, "net_return"].sum()
    losses = -predictions.loc[predictions["net_return"] < 0, "net_return"].sum()
    profit_factor = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
    valid_outer_folds = sum(int(row["trades"]) >= 20 for row in fold_metrics)
    expected_signs = {
        "dxy_ret_1": -1,
        "dxy_ret_3": -1,
        "xag_ret_1": 1,
        "xag_ret_3": 1,
        "metal_usd_pressure_3": 1,
    }
    sign_stability = {
        feature: {
            "expected_sign": expected,
            "matching_folds": sum(sign == expected for sign in coefficient_signs[feature]),
            "fold_signs": coefficient_signs[feature],
        }
        for feature, expected in expected_signs.items()
    }
    stable_mechanism = any(row["matching_folds"] >= 4 for row in sign_stability.values())
    net_return = float(predictions["net_return"].sum())
    checks = {
        "context_overlap_at_least_80pct": overlap >= 0.80,
        "mechanism_sign_stable_at_least_4_of_5": stable_mechanism,
        "accuracy_delta_at_least_2pp": accuracy >= REFERENCE_BASELINE_ACCURACY + MINIMUM_ACCURACY_DELTA,
        "positive_after_costs": net_return > 0,
        "profit_factor_above_1_10": profit_factor > 1.10,
        "minimum_buy_trades": buy_trades >= 75,
        "minimum_sell_trades": sell_trades >= 75,
        "at_least_four_valid_folds": valid_outer_folds >= 4,
    }
    target_status = classify_accuracy_target(
        directional_accuracy=accuracy,
        wilson_lower_95=wilson_lower(successes, trades),
        trades=trades,
        buy_trades=buy_trades,
        sell_trades=sell_trades,
        coverage=coverage,
        valid_outer_folds=valid_outer_folds,
        protocol_integrity=True,
    )
    actual = predictions["actual"].to_numpy(dtype=int)
    direction = predictions["direction"].to_numpy(dtype=int)
    summary = {
        "family_id": FAMILY_ID,
        "status": "SURVIVED_PRE_GATE" if all(checks.values()) else "FALSIFIED_PRE_GATE",
        "decision": "RUN_ONE_FULL_NESTED_CHALLENGER" if all(checks.values()) else "CLOSE_FAMILY_NO_FULL_RUN",
        "active_contract": "V15.1_LEAN",
        "protocol": {
            "model": "fixed_standardized_balanced_multinomial_logistic",
            "hyperparameter_trials": 0,
            "confidence_threshold": FIXED_CONFIDENCE_THRESHOLD,
            "horizon_bars": HORIZON_BARS,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "context_availability": "one_full_M5_bar_lag_exact_join_no_forward_fill",
            "outer_folds": 5,
            "purge_and_embargo_bars": HORIZON_BARS,
            "sealed_holdout_access_count": 0,
            "development_cutoff": development_cutoff.isoformat(),
        },
        "source_provenance": {
            "gold": {"path": str(args.gold.resolve()), "sha256": sha256_file(args.gold)},
            "dxy": {"path": str(args.dxy.resolve()), "sha256": sha256_file(args.dxy)},
            "xag": {"path": str(args.xag.resolve()), "sha256": sha256_file(args.xag)},
            "reference_artifact": str(args.reference_artifact.resolve()),
        },
        "metrics": {
            "directional_accuracy": accuracy,
            "accuracy_wilson_lower_95": wilson_lower(successes, trades),
            "macro_f1_executed": float(f1_score(actual, direction, labels=[-1, 0, 1], average="macro", zero_division=0)),
            "trades": trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "coverage": coverage,
            "context_complete_overlap": overlap,
            "net_return_sum": net_return,
            "profit_factor": profit_factor,
            "valid_outer_folds": valid_outer_folds,
            "accuracy_delta_vs_exact_m5_baseline_pp": 100 * (accuracy - REFERENCE_BASELINE_ACCURACY),
        },
        "accuracy_target_status": target_status,
        "checks": checks,
        "coefficient_sign_stability": sign_stability,
        "fold_metrics": fold_metrics,
        "lockbox_status": "sealed_not_accessed",
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    predictions.to_csv(output / "predictions.csv", index_label="timestamp")
    (output / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
