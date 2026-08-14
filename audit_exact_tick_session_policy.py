from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.execution_calibration import calibrate_quote_latency


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen session-aware exact Exness fill audit")
    parser.add_argument("--tick-root", type=Path, required=True)
    parser.add_argument("--development-end-day", default="2026-08-10")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(
        path for path in args.tick_root.resolve().rglob("XAUUSDm_ticks_*.csv")
        if path.stem[-10:] <= args.development_end_day and path.stat().st_size > 100
    )
    if not files:
        raise RuntimeError("no development tick files")
    ticks = pd.concat(
        [pd.read_csv(path, usecols=["time_msc", "bid", "ask"]) for path in files],
        ignore_index=True,
    ).sort_values("time_msc", kind="mergesort")
    protocol = {
        "latency_ms": [500],
        "decision_interval_ms": 300_000,
        "exit_horizon_ms": 3_600_000,
        "maximum_wait_ms": 2_000,
        "maximum_entry_wait_ms": 2_000,
        "maximum_exit_wait_ms": 5_000,
        "maximum_reference_age_ms": 2_000,
        "blocked_planned_exit_hours_utc": [21],
        "minimum_full_fill_rate": 0.99,
        "minimum_daily_full_fill_rate": 0.95,
        "require_session_blocked_decisions": True,
    }
    calibration = calibrate_quote_latency(
        ticks,
        latency_ms=protocol["latency_ms"],
        decision_interval_ms=protocol["decision_interval_ms"],
        exit_horizon_ms=protocol["exit_horizon_ms"],
        maximum_wait_ms=protocol["maximum_wait_ms"],
        maximum_reference_age_ms=protocol["maximum_reference_age_ms"],
        blocked_planned_exit_hours_utc=protocol["blocked_planned_exit_hours_utc"],
        maximum_entry_wait_ms=protocol["maximum_entry_wait_ms"],
        maximum_exit_wait_ms=protocol["maximum_exit_wait_ms"],
    )
    row = calibration["latencies"]["500"]
    checks = {
        "full_fill_rate": row["full_fill_rate"] >= protocol["minimum_full_fill_rate"],
        "minimum_daily_full_fill_rate": row["daily_full_fill_rate"]["minimum"] >= protocol["minimum_daily_full_fill_rate"],
        "first_next_exit_within_frozen_timeout": row["exit_wait_after_target_ms"]["max"] <= protocol["maximum_exit_wait_ms"],
        "maintenance_crossing_decisions_explicitly_blocked": calibration["blocked_session_decisions"] > 0,
    }
    summary = {
        "schema_version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_SESSION_AWARE_CAUSAL_FILL_COMPONENT" if all(checks.values()) else "FAIL_SESSION_AWARE_CAUSAL_FILL_COMPONENT",
        "evidence_scope": "DEVELOPMENT_ONLY_QUOTE_AVAILABILITY_AND_GAP_NOT_ACTUAL_ORDER_FILL_OR_BROKER_SLIPPAGE",
        "instrument": "XAUUSDm",
        "timeframe": "M5",
        "source": "exact_Exness_MT5_ticks",
        "development_end_day": args.development_end_day,
        "unseen_days_excluded": ["2026-08-11", "2026-08-12"],
        "protocol": protocol,
        "checks": checks,
        "calibration": calibration,
        "tick_rows": int(len(ticks)),
        "tick_files": [{"path": str(path), "sha256": sha256(path)} for path in files],
        "model_fits": 0,
        "holdout_access_count": 0,
        "live_trading_enabled": False,
        "auto_execution": False,
        "promotion_authorized": False,
        "decision": "AUTHORIZE_COMPONENT_FOR_FUTURE_RESEARCH_POLICY_BINDING" if all(checks.values()) else "CLOSE_SESSION_AWARE_POLICY_REFINEMENT",
    }
    atomic_json(args.output.resolve(), summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": checks, "calibration": calibration}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
