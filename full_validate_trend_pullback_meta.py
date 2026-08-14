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

from explore_setup_meta_label import _load_dataset, _select_inner, _splitter
from gold_ai.data.providers import sha256_file
from gold_ai.direct_net_edge import aggregate_predictions
from gold_ai.setup_meta_label import (
    FEATURE_SCHEMA_VERSION,
    HORIZONS,
    MAX_HORIZON,
    META_THRESHOLDS,
    SLIPPAGE_BUFFER_BPS,
    execute_setup_predictions,
    fit_predict_meta,
)


VALIDATION_ID = "TREND_PULLBACK_META_FULL_VALIDATION_V1"
SETUP_ID = "trend_pullback_resumption_meta_v1"
STRESS_COST_BPS = 3.0
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 15_101


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full nested validation of the sole setup-meta exploration survivor")
    parser.add_argument("--m5", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--exploration-artifact", type=Path, required=True)
    parser.add_argument("--requirements-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def _calibration_metrics(probabilities: pd.DataFrame) -> dict[str, float]:
    if probabilities.empty:
        return {"brier_score": 1.0, "expected_calibration_error": 1.0}
    probability = probabilities["selection_probability"].to_numpy(dtype=float)
    actual = probabilities["positive_net"].to_numpy(dtype=float)
    brier = float(np.mean(np.square(probability - actual)))
    bins = np.linspace(0.0, 1.0, 11)
    assignments = np.clip(np.digitize(probability, bins, right=True) - 1, 0, 9)
    ece = 0.0
    for bucket in range(10):
        mask = assignments == bucket
        if mask.any():
            ece += float(mask.mean()) * abs(float(probability[mask].mean()) - float(actual[mask].mean()))
    return {"brier_score": brier, "expected_calibration_error": float(ece)}


def _stress_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    stressed = predictions.copy()
    if len(stressed):
        stressed["net_return"] = stressed["gross_strategy_return"].astype(float) - STRESS_COST_BPS / 10_000.0
    return stressed


def _day_block_bootstrap(predictions: pd.DataFrame) -> dict[str, Any]:
    daily = predictions["net_return"].astype(float).groupby(predictions.index.floor("D")).sum()
    if len(daily) < 20:
        return {
            "status": "INSUFFICIENT_DATA",
            "days": int(len(daily)),
            "samples": 0,
            "net_return_ci95_lower": None,
            "net_return_ci95_upper": None,
        }
    values = daily.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    totals = values[draws].sum(axis=1)
    return {
        "status": "PASS_COMPUTED",
        "days": int(len(values)),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "net_return_ci95_lower": float(np.quantile(totals, 0.025)),
        "net_return_ci95_upper": float(np.quantile(totals, 0.975)),
        "positive_bootstrap_probability": float(np.mean(totals > 0.0)),
    }


def _validation_checks(
    candidate: dict[str, Any],
    control: dict[str, Any],
    stress: dict[str, Any],
    calibration: dict[str, float],
    bootstrap: dict[str, Any],
) -> dict[str, bool]:
    return {
        "positive_primary_after_dynamic_costs": float(candidate["net_return_sum"]) > 0.0,
        "primary_profit_factor_at_least_1_15": float(candidate["profit_factor"]) >= 1.15,
        "positive_three_bps_stress": float(stress["net_return_sum"]) > 0.0,
        "stress_profit_factor_at_least_1_05": float(stress["profit_factor"]) >= 1.05,
        "minimum_300_trades": int(candidate["trades"]) >= 300,
        "minimum_100_buy_trades": int(candidate["buy_trades"]) >= 100,
        "minimum_100_sell_trades": int(candidate["sell_trades"]) >= 100,
        "all_five_outer_folds_valid": int(candidate["valid_outer_folds"]) >= 5,
        "at_least_four_profitable_outer_folds": int(candidate["positive_net_outer_folds"]) >= 4,
        "positive_daily_sharpe": float(candidate["daily_sharpe"]) > 0.0,
        "positive_daily_sortino": float(candidate["daily_sortino"]) > 0.0,
        "bounded_additive_drawdown": abs(float(candidate["max_drawdown_additive"])) <= 0.15,
        "net_return_above_paired_rule": float(candidate["net_return_sum"]) > float(control["net_return_sum"]),
        "profit_factor_above_paired_rule": float(candidate["profit_factor"]) > float(control["profit_factor"]),
        "brier_at_most_0_25": float(calibration["brier_score"]) <= 0.25,
        "ece_at_most_0_10": float(calibration["expected_calibration_error"]) <= 0.10,
        "day_bootstrap_computed": bootstrap.get("status") == "PASS_COMPUTED",
        "day_bootstrap_ci_lower_positive": (
            bootstrap.get("net_return_ci95_lower") is not None
            and float(bootstrap["net_return_ci95_lower"]) > 0.0
        ),
    }


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to repeat existing full validation: {output}")
    exploration_summary = json.loads((args.exploration_artifact.resolve() / "summary.json").read_text(encoding="utf-8"))
    exploration_validation = json.loads((args.exploration_artifact.resolve() / "validation.json").read_text(encoding="utf-8"))
    if exploration_validation.get("status") != "PASS_SURVIVOR_ELIGIBLE_FOR_VALIDATION":
        raise ValueError("Exploration validator did not authorize a survivor")
    if exploration_validation.get("independently_selected_for_validation") != SETUP_ID:
        raise ValueError("Frozen full-validation candidate does not match independent selection")
    if not 1 <= int(args.n_jobs) <= 16:
        raise ValueError("n-jobs must be 1..16")
    prompt = args.requirements_prompt.resolve()
    if not prompt.is_file():
        raise FileNotFoundError(prompt)
    data = _load_dataset(args)
    x = data["x"]
    outcome_by_horizon = data["outcomes"][SETUP_ID]
    outer_splits = list(
        _splitter(5, 12_000, 30_000).split(
            x.index, outcome_by_horizon[MAX_HORIZON]["label_end_time"]
        )
    )
    if len(outer_splits) != 5:
        raise AssertionError("Full validation requires exactly five outer folds")
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "PASS_PREFLIGHT_NO_TRAINING",
                    "validation_id": VALIDATION_ID,
                    "candidate": SETUP_ID,
                    "development_rows": len(x),
                    "outer_folds": len(outer_splits),
                    "inner_folds": 3,
                    "exploration_selection_verified": True,
                    "holdout_access_count": 0,
                },
                indent=2,
            )
        )
        return 0

    candidate_parts: list[pd.DataFrame] = []
    control_parts: list[pd.DataFrame] = []
    probability_parts: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    outer_audits: list[dict[str, Any]] = []
    total_test_rows = 0
    for fold, split in enumerate(outer_splits, start=1):
        total_test_rows += len(split.test_positions)
        outer_audit = split.audit_dict()
        outer_audit["train_start_time"] = x.index[split.train_positions[0]].isoformat()
        outer_audit["train_end_time"] = x.index[split.train_positions[-1]].isoformat()
        outer_audits.append(outer_audit)
        outer_train = split.train_positions
        outcomes_train = {h: outcome_by_horizon[h].iloc[outer_train] for h in HORIZONS}
        frozen, inner_audit = _select_inner(
            SETUP_ID,
            x.iloc[outer_train],
            outcomes_train,
            data["source_positions"][outer_train],
            seed=19_000 + fold * 100,
            inner_splits=3,
            inner_max_train_rows=24_000,
        )
        horizon = int(frozen["horizon_bars"])
        print(
            f"[full-validation] fold={fold}/5 frozen=h{horizon}/p{frozen['meta_threshold']}",
            flush=True,
        )
        event_positions, probabilities, fit_audit = fit_predict_meta(
            x,
            outcome_by_horizon[horizon],
            split.train_positions,
            split.test_positions,
            seed=20_000 + fold * 100,
        )
        candidate = execute_setup_predictions(
            timestamps=x.index,
            source_positions=data["source_positions"],
            outcome=outcome_by_horizon[horizon],
            candidate_positions=event_positions,
            probabilities=probabilities,
            threshold=float(frozen["meta_threshold"]),
            test_positions=split.test_positions,
            horizon=horizon,
            fold=fold,
            setup_id=SETUP_ID,
            role="candidate",
        )
        control = execute_setup_predictions(
            timestamps=x.index,
            source_positions=data["source_positions"],
            outcome=outcome_by_horizon[horizon],
            candidate_positions=event_positions,
            probabilities=np.ones(len(event_positions), dtype=float),
            threshold=0.0,
            test_positions=split.test_positions,
            horizon=horizon,
            fold=fold,
            setup_id=SETUP_ID,
            role="paired_rule_control",
        )
        outcome = outcome_by_horizon[horizon]
        probability_frame = pd.DataFrame(
            {
                "fold": fold,
                "row_position": event_positions,
                "source_position": data["source_positions"][event_positions],
                "horizon_bars": horizon,
                "meta_threshold": float(frozen["meta_threshold"]),
                "selection_probability": probabilities,
                "positive_net": outcome["positive_net"].to_numpy(dtype=int)[event_positions],
                "direction": outcome["direction"].to_numpy(dtype=int)[event_positions],
                "net_return": outcome["net_return"].to_numpy(dtype=float)[event_positions],
            },
            index=x.index[event_positions],
        )
        candidate_parts.append(candidate)
        control_parts.append(control)
        probability_parts.append(probability_frame)
        fold_records.append(
            {
                "fold": fold,
                "frozen_configuration": frozen,
                "inner_selection": inner_audit,
                "outer_fit_audit": fit_audit,
                "candidate_metrics": aggregate_predictions(candidate, total_test_rows=len(split.test_positions)),
                "control_metrics": aggregate_predictions(control, total_test_rows=len(split.test_positions)),
                "stress_metrics": aggregate_predictions(_stress_predictions(candidate), total_test_rows=len(split.test_positions)),
            }
        )
    candidate = pd.concat([part for part in candidate_parts if len(part)]).sort_index()
    control = pd.concat([part for part in control_parts if len(part)]).sort_index()
    all_probabilities = pd.concat(probability_parts).sort_index()
    candidate_metrics = aggregate_predictions(candidate, total_test_rows=total_test_rows)
    control_metrics = aggregate_predictions(control, total_test_rows=total_test_rows)
    stress_metrics = aggregate_predictions(_stress_predictions(candidate), total_test_rows=total_test_rows)
    calibration = _calibration_metrics(all_probabilities)
    bootstrap = _day_block_bootstrap(candidate)
    checks = _validation_checks(candidate_metrics, control_metrics, stress_metrics, calibration, bootstrap)
    passed = all(checks.values())
    configs = [
        (int(row["frozen_configuration"]["horizon_bars"]), float(row["frozen_configuration"]["meta_threshold"]))
        for row in fold_records
    ]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_id": VALIDATION_ID,
        "status": "VALIDATION_PASS_RESEARCH_CANDIDATE_CAN_FREEZE" if passed else "VALIDATION_FAILED_CANDIDATE_REJECTED",
        "decision": "FREEZE_RESEARCH_CANDIDATE_FOR_FUTURE_UNSEEN_EVIDENCE" if passed else "REJECT_EXACT_VALIDATION_CANDIDATE",
        "current_mode": "VALIDATION",
        "candidate_id": SETUP_ID,
        "candidate_metrics": candidate_metrics,
        "paired_rule_control_metrics": control_metrics,
        "three_bps_stress_metrics": stress_metrics,
        "calibration": calibration,
        "day_block_bootstrap": bootstrap,
        "validation_checks": checks,
        "all_validation_checks_pass": passed,
        "research_only": True,
        "production_promotion_allowed": False,
        "verified_predictive_improvement_allowed": False,
        "verified_predictive_improvement_blocker": "development rows reused after family exploration; future-unseen confirmation required",
        "live_trading_enabled": False,
        "auto_execution": False,
        "protocol": {
            "candidate_frozen_before_validation": True,
            "outer_folds": 5,
            "inner_folds": 3,
            "purge_and_embargo_bars": MAX_HORIZON,
            "horizons": list(HORIZONS),
            "inner_only_meta_thresholds": list(META_THRESHOLDS),
            "dynamic_cost": "BUY_entry_spread_SELL_exit_spread_plus_buffer",
            "slippage_latency_buffer_bps": SLIPPAGE_BUFFER_BPS,
            "stress_cost_bps": STRESS_COST_BPS,
            "non_overlapping_trades": True,
            "development_rows": len(x),
            "development_cutoff": data["cutoff"].isoformat(),
            "total_outer_test_rows": total_test_rows,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "fold_configurations": configs,
            "unique_fold_configurations": len(set(configs)),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "sealed_holdout_access_count": 0,
            "saved_lockbox_predictions_accessed": False,
        },
        "source_provenance": {
            "m5": {"path": str(args.m5.resolve()), "sha256": sha256_file(args.m5)},
            "m5_metadata": {"path": str(args.m5.with_suffix('.metadata.json').resolve()), "sha256": sha256_file(args.m5.with_suffix('.metadata.json'))},
            "reference_summary": {"path": str(data["reference_path"]), "sha256": sha256_file(data["reference_path"])},
            "exploration_summary": {"path": str((args.exploration_artifact.resolve() / 'summary.json')), "sha256": sha256_file(args.exploration_artifact.resolve() / 'summary.json')},
            "exploration_validation": {"path": str((args.exploration_artifact.resolve() / 'validation.json')), "sha256": sha256_file(args.exploration_artifact.resolve() / 'validation.json')},
            "requirements_prompt": {"path": str(prompt), "sha256": sha256_file(prompt)},
            "trainer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        },
        "outer_fold_audits": outer_audits,
        "fold_records": fold_records,
        "lockbox_status": "SEALED_NOT_ACCESSED",
    }
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    (temporary / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    candidate.to_csv(temporary / "predictions_candidate.csv", index_label="timestamp")
    control.to_csv(temporary / "predictions_control.csv", index_label="timestamp")
    all_probabilities.to_csv(temporary / "probabilities_all_setup_events.csv", index_label="timestamp")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    print(json.dumps({"status": summary["status"], "all_checks_pass": passed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
