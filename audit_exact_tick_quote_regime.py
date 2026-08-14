from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gold_ai.m5_markov_transitions import aggregate_markov_features
from gold_ai.m5_volatility_signature import aggregate_signature_features


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def select_rows(manifest: dict[str, Any], count: int, end_day: str | None = None) -> list[dict[str, Any]]:
    rows = [row for row in manifest["partitions"] if int(row.get("rows", 0)) > 0 and (end_day is None or str(row["utc_day"]) < end_day)]
    positions = np.linspace(0, len(rows) - 1, min(count, len(rows)), dtype=int)
    return [rows[int(position)] for position in positions]


def psi(reference: pd.Series, observed: pd.Series) -> float:
    quantiles = np.unique(np.nanquantile(reference.to_numpy(float), np.linspace(0, 1, 11)))
    if len(quantiles) < 3:
        return 0.0 if np.allclose(reference, observed.mean(), rtol=0, atol=1e-12) else 10.0
    bins = np.r_[-np.inf, quantiles[1:-1], np.inf]
    left = np.maximum(np.histogram(reference, bins=bins)[0] / len(reference), 1e-6)
    right = np.maximum(np.histogram(observed, bins=bins)[0] / len(observed), 1e-6)
    return float(np.sum((right - left) * np.log(right / left)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    training_manifest_path = (PROJECT_ROOT / plan["data"]["training_manifest"]).resolve()
    current_manifest_path = (PROJECT_ROOT / plan["data"]["current_manifest"]).resolve()
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    current_manifest = json.loads(current_manifest_path.read_text(encoding="utf-8"))
    count = int(plan["data"]["sample_nonempty_days_per_domain"])
    domains = {
        "historical_2025": select_rows(training_manifest, count),
        "current_2026": select_rows(current_manifest, count, plan["data"]["current_cutoff_exclusive"]),
    }

    daily_rows: list[dict[str, Any]] = []
    domain_frames: dict[str, list[pd.DataFrame]] = {name: [] for name in domains}
    for domain, rows in domains.items():
        for row in rows:
            path = Path(row["csv"])
            markov = aggregate_markov_features([path]).add_prefix("markov__")
            signature = aggregate_signature_features([path]).add_prefix("signature__")
            common = markov.index.intersection(signature.index)
            combined = markov.loc[common].join(signature.loc[common], how="inner")
            domain_frames[domain].append(combined)
            record: dict[str, Any] = {
                "domain": domain,
                "utc_day": row["utc_day"],
                "source_file": str(path),
                "source_sha256": row["file_sha256"],
                "m5_intervals": len(combined),
            }
            record.update({f"median__{column}": float(combined[column].median()) for column in combined.columns})
            daily_rows.append(record)
    daily = pd.DataFrame(daily_rows).sort_values(["domain", "utc_day"])
    daily_path = output_dir / "daily_metrics.csv"
    daily.to_csv(daily_path, index=False)
    historical = pd.concat(domain_frames["historical_2025"])
    current = pd.concat(domain_frames["current_2026"])
    feature_audit: dict[str, dict[str, Any]] = {}
    stable: list[str] = []
    severe: list[str] = []
    for column in historical.columns:
        low, high = np.nanquantile(historical[column], [0.01, 0.99])
        support = float(((current[column] >= low) & (current[column] <= high)).mean())
        shift = psi(historical[column], current[column])
        is_stable = support >= 0.90 and shift <= 0.50
        is_severe = support < 0.70 or shift > 1.0
        if is_stable:
            stable.append(column)
        if is_severe:
            severe.append(column)
        feature_audit[column] = {
            "historical_p01": float(low),
            "historical_p99": float(high),
            "current_within_historical_p01_p99": support,
            "psi": shift,
            "stable": is_stable,
            "severe_shift": is_severe,
        }
    eligible = len(stable) / len(feature_audit) >= 0.80 and not any(
        name.startswith(("markov__transition_", "markov__lag1_", "signature__tick_to_", "signature__one_to_", "signature__jump_"))
        for name in severe
    )
    payload = {
        "schema_version": 1,
        "audit_id": plan["audit_id"],
        "status": "PASS_OLD_HISTORY_PREDICTIVE_FEATURE_ELIGIBLE" if eligible else "FAIL_OLD_HISTORY_QUOTE_REGIME_MISMATCH",
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "source": {
            "historical_manifest": str(training_manifest_path),
            "historical_manifest_sha256": sha256_file(training_manifest_path),
            "current_manifest": str(current_manifest_path),
            "current_manifest_sha256": sha256_file(current_manifest_path),
            "sample_days_per_domain": count,
            "historical_rows": len(historical),
            "current_rows": len(current),
        },
        "feature_summary": {
            "feature_count": len(feature_audit),
            "stable_count": len(stable),
            "stable_fraction": len(stable) / len(feature_audit),
            "severe_shift_count": len(severe),
            "stable_features": stable,
            "severe_shift_features": severe,
        },
        "features": feature_audit,
        "decision": "CURRENT_REGIME_ONLY_FOR_PREDICTIVE_FEATURE_TRAINING_OLD_HISTORY_EXECUTION_DQ_ONLY" if not eligible else "OLD_HISTORY_ELIGIBLE_UNDER_RECORDED_FEATURE_SUBSET",
        "access_audit": {
            "labels_accessed": False,
            "returns_after_decision_accessed": False,
            "future_quotes_accessed": False,
            "final_holdout_access_count": 0,
            "orders_or_account_mutation": False,
        },
        "artifacts": {
            "daily_metrics": str(daily_path),
            "daily_metrics_sha256": sha256_file(daily_path),
        },
    }
    atomic_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
