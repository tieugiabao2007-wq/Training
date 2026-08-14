from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.data.binance_aggtrades import load_archive, sha256_file
from gold_ai.tick_event_edge import aggregate_tick_events
from gold_ai.transfer_features import (
    COMMON_EVENT_FEATURES,
    domain_shift_metrics,
    exness_common_features,
    paxg_common_features,
)


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Point-in-time PAXG/Exness common-feature domain audit")
    parser.add_argument("--paxg-archive", type=Path, required=True)
    parser.add_argument("--tick-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-07-17T00:00:00Z")
    args = parser.parse_args()
    cutoff = args.cutoff
    raw, entry = load_archive(args.paxg_archive.resolve())
    paxg = paxg_common_features(raw, cutoff)
    tick_files = sorted(
        path for path in args.tick_root.resolve().rglob("XAUUSDm_ticks_*.csv")
        if path.stem[-10:] <= "2026-08-10" and path.stat().st_size > 100
    )
    events = aggregate_tick_events(tick_files)
    exness = exness_common_features(events)
    result = domain_shift_metrics(paxg, exness)
    args.output.mkdir(parents=True, exist_ok=False)
    paxg_path = args.output / "paxg_common_features.csv"
    exness_path = args.output / "exness_common_features.csv"
    paxg.to_csv(paxg_path, index_label="decision_time")
    exness.to_csv(exness_path, index_label="decision_time")
    cutoff_ts = __import__("pandas").Timestamp(cutoff)
    summary = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": result["status"],
        "source_mismatch": True,
        "exact_exness_outer_evidence": False,
        "common_cutoff_utc": cutoff_ts.isoformat(),
        "point_in_time_protocol": "COMMON_CUTOFF_BEFORE_ALL_EXACT_EVALUATION",
        "paxg_raw_max_used_utc": __import__("pandas").to_datetime(
            int(raw.filter(__import__("polars").col("transact_time_us") < int(cutoff_ts.timestamp() * 1_000_000))["transact_time_us"].max()),
            unit="us", utc=True,
        ).isoformat(),
        "paxg_weekend_policy": "EXCLUDE_SATURDAY_SUNDAY",
        "feature_schema": list(COMMON_EVENT_FEATURES),
        "paxg_rows": len(paxg),
        "paxg_days": int(paxg.index.floor("D").nunique()),
        "paxg_first": paxg.index.min().isoformat(),
        "paxg_last": paxg.index.max().isoformat(),
        "exness_rows": len(exness),
        "exness_days": int(exness.index.floor("D").nunique()),
        "exness_first": exness.index.min().isoformat(),
        "exness_last": exness.index.max().isoformat(),
        "paxg_archive": str(args.paxg_archive.resolve()),
        "paxg_archive_sha256": sha256_file(args.paxg_archive.resolve()),
        "paxg_csv_entry": entry,
        "tick_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in tick_files
        ],
        "paxg_features_sha256": sha256_file(paxg_path),
        "exness_features_sha256": sha256_file(exness_path),
        "domain_shift": result,
        "decision": "AUTHORIZE_BOUNDED_EARLIER_MONTH_AUDIT" if result["status"] == "PASS_DOMAIN_PREFLIGHT" else "CLOSE_PAXG_COMMON_FEATURE_SET_NO_TRAIN",
        "holdout_access_count": 0,
        "live_trading_enabled": False,
        "auto_execution": False,
    }
    atomic(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "paxg_rows": len(paxg), "exness_rows": len(exness), "median_psi": result["median_psi"], "checks": result["checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
