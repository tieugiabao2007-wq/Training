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


MECHANISM_PREFIXES = (
    "markov__transition_",
    "markov__lag1_",
    "markov__positive_persistence_",
    "markov__negative_persistence_",
    "signature__tick_to_",
    "signature__one_to_",
    "signature__jump_",
)


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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-day", default="2026-01-01")
    parser.add_argument("--end-day-exclusive", default="2026-08-11")
    parser.add_argument("--reference-start", default="2026-07-17")
    parser.add_argument("--reference-end-exclusive", default="2026-08-11")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in manifest["partitions"]
        if int(row.get("rows", 0)) > 0
        and args.start_day <= str(row["utc_day"]) < args.end_day_exclusive
    ]
    frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    for number, row in enumerate(rows, start=1):
        path = Path(row["csv"])
        markov = aggregate_markov_features([path]).add_prefix("markov__")
        signature = aggregate_signature_features([path]).add_prefix("signature__")
        common = markov.index.intersection(signature.index)
        frame = markov.loc[common].join(signature.loc[common], how="inner")
        frame.insert(0, "source_day", row["utc_day"])
        frames.append(frame)
        source_rows.append(
            {
                "utc_day": row["utc_day"],
                "csv": str(path),
                "file_sha256": row["file_sha256"],
                "rows": int(row["rows"]),
                "m5_intervals": len(frame),
            }
        )
        if number % 20 == 0:
            print(f"[monthly-regime] processed {number}/{len(rows)} nonempty days", flush=True)
    features = pd.concat(frames).sort_index(kind="stable")
    features_path = output / "features.csv"
    features.to_csv(features_path, index_label="decision_time")
    sources_path = output / "sources.json"
    sources_path.write_text(json.dumps(source_rows, indent=2), encoding="utf-8")
    decision_time = pd.DatetimeIndex(features.index)
    reference_mask = (
        (decision_time >= pd.Timestamp(args.reference_start, tz="UTC"))
        & (decision_time < pd.Timestamp(args.reference_end_exclusive, tz="UTC"))
    )
    reference = features.loc[reference_mask].drop(columns="source_day")
    months = sorted(pd.Series(decision_time.strftime("%Y-%m")).unique())
    month_audits: dict[str, Any] = {}
    for month in months:
        month_mask = decision_time.strftime("%Y-%m") == month
        observed = features.loc[month_mask].drop(columns="source_day")
        feature_metrics = {}
        stable_count = 0
        severe: list[str] = []
        for column in reference.columns:
            low, high = np.nanquantile(reference[column], [0.01, 0.99])
            support = float(((observed[column] >= low) & (observed[column] <= high)).mean())
            shift = psi(reference[column], observed[column])
            stable = support >= 0.90 and shift <= 0.50
            severe_shift = support < 0.70 or shift > 1.0
            stable_count += int(stable)
            if severe_shift:
                severe.append(column)
            feature_metrics[column] = {
                "current_reference_p01": float(low),
                "current_reference_p99": float(high),
                "month_within_reference_p01_p99": support,
                "psi": shift,
                "stable": stable,
                "severe_shift": severe_shift,
            }
        severe_mechanism = [name for name in severe if name.startswith(MECHANISM_PREFIXES)]
        stable_fraction = stable_count / len(reference.columns)
        eligible = stable_fraction >= 0.80 and not severe_mechanism
        month_audits[month] = {
            "rows": len(observed),
            "stable_count": stable_count,
            "stable_fraction": stable_fraction,
            "severe_shift_count": len(severe),
            "severe_mechanism_features": severe_mechanism,
            "predictive_feature_regime_eligible": eligible,
            "features": feature_metrics,
        }
    eligible_months = [month for month, audit in month_audits.items() if audit["predictive_feature_regime_eligible"]]
    payload = {
        "schema_version": 1,
        "audit_id": "EXACT_TICK_2026_MONTHLY_CURRENT_REGIME_BOUNDARY_V1",
        "status": "PASS_MONTHLY_CHANGE_POINT_AUDIT",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "scope": {
            "start_day": args.start_day,
            "end_day_exclusive": args.end_day_exclusive,
            "reference_start": args.reference_start,
            "reference_end_exclusive": args.reference_end_exclusive,
            "nonempty_days": len(rows),
            "m5_feature_rows": len(features),
        },
        "month_audits": month_audits,
        "eligible_months": eligible_months,
        "decision": "USE_ONLY_RECORDED_ELIGIBLE_MONTHS_FOR_FULL_MARKOV_SIGNATURE_PREDICTIVE_SCHEMA",
        "access_audit": {
            "labels_accessed": False,
            "returns_after_decision_accessed": False,
            "future_quotes_accessed": False,
            "final_holdout_access_count": 0,
            "orders_or_account_mutation": False,
        },
        "artifacts": {
            "features": str(features_path),
            "features_sha256": sha256_file(features_path),
            "sources": str(sources_path),
            "sources_sha256": sha256_file(sources_path),
        },
    }
    atomic_json(output / "summary.json", payload)
    print(json.dumps({"status": payload["status"], "scope": payload["scope"], "eligible_months": eligible_months, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
