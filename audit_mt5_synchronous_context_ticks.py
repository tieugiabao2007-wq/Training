from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def boundaries(day: date) -> list[int]:
    eastern = ZoneInfo("America/New_York")
    first = datetime.combine(day, time(9, 35), tzinfo=eastern)
    last = datetime.combine(day, time(15, 55), tzinfo=eastern)
    result = []
    current = first
    while current <= last:
        result.append(int(current.astimezone(timezone.utc).timestamp() * 1000))
        current += timedelta(minutes=5)
    return result


def partition_metrics(times: np.ndarray, bids: np.ndarray, asks: np.ndarray, points: list[int], max_pre: int, max_post: int) -> dict:
    before = np.searchsorted(times, points, side="left") - 1
    after = np.searchsorted(times, points, side="left")
    point_array = np.asarray(points, dtype=np.int64)
    valid_before = before >= 0
    valid_after = after < len(times)
    pre_age = np.full(len(points), np.inf)
    post_wait = np.full(len(points), np.inf)
    pre_age[valid_before] = point_array[valid_before] - times[before[valid_before]]
    post_wait[valid_after] = times[after[valid_after]] - point_array[valid_after]
    return {
        "rows": int(len(times)),
        "valid_bidask_fraction": float(np.mean((bids > 0) & (asks >= bids))) if len(times) else 0.0,
        "timestamp_monotonic": bool(np.all(times[1:] >= times[:-1])) if len(times) > 1 else True,
        "pre_boundary_ok": [bool(value <= max_pre) for value in pre_age],
        "post_boundary_ok": [bool(value <= max_post) for value in post_wait],
        "pre_age_ms_p50": float(np.median(pre_age[np.isfinite(pre_age)])) if np.isfinite(pre_age).any() else None,
        "pre_age_ms_p99": float(np.quantile(pre_age[np.isfinite(pre_age)], 0.99)) if np.isfinite(pre_age).any() else None,
        "post_wait_ms_p50": float(np.median(post_wait[np.isfinite(post_wait)])) if np.isfinite(post_wait).any() else None,
        "post_wait_ms_p99": float(np.quantile(post_wait[np.isfinite(post_wait)], 0.99)) if np.isfinite(post_wait).any() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "artifacts/control/m5_synchronous_cross_market_shock_v1_protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/data_audits/mt5_synchronous_context_ticks_v1/summary.json")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    terminal = protocol["data_preflight"]["terminal"]
    if not mt5.initialize(terminal):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        universe = protocol["fixed_universe"]
        before = {symbol: {"selected": bool(mt5.symbol_info(symbol).select), "visible": bool(mt5.symbol_info(symbol).visible)} for symbol in universe}
        if not all(item["selected"] for item in before.values()):
            raise RuntimeError("frozen universe is not already selected; symbol_select mutation is forbidden")
        output_root = ROOT / protocol["data_preflight"]["output_root"]
        partitions = []
        by_symbol_day: dict[tuple[str, str], dict] = {}
        for day_value in protocol["sample_days"]:
            day = date.fromisoformat(day_value)
            start = datetime.combine(day, time.min, tzinfo=timezone.utc)
            end = start + timedelta(days=1)
            points = boundaries(day)
            for symbol in universe:
                ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
                if ticks is None:
                    raise RuntimeError(f"copy_ticks_range failed {symbol} {day}: {mt5.last_error()}")
                frame = pd.DataFrame(ticks)
                if frame.empty:
                    times = np.array([], dtype=np.int64)
                    bids = asks = np.array([], dtype=float)
                    frame = pd.DataFrame(columns=["time", "bid", "ask", "last", "volume", "time_msc", "flags", "volume_real"])
                else:
                    frame = frame.sort_values("time_msc", kind="stable").reset_index(drop=True)
                    times = frame["time_msc"].to_numpy(dtype=np.int64)
                    bids = frame["bid"].to_numpy(dtype=float)
                    asks = frame["ask"].to_numpy(dtype=float)
                destination = output_root / symbol / f"{symbol}_ticks_{day_value}.csv"
                destination.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(destination, index=False)
                metrics = partition_metrics(
                    times, bids, asks, points,
                    int(protocol["data_preflight"]["maximum_pre_boundary_quote_age_ms"]),
                    int(protocol["data_preflight"]["maximum_post_boundary_wait_ms"]),
                )
                record = {
                    "symbol": symbol,
                    "utc_day": day_value,
                    "source": f"local://mt5/{symbol}/ticks/{day_value}",
                    "retrieval": protocol["data_preflight"]["retrieval"],
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                    "csv": str(destination.resolve()),
                    "sha256": sha256(destination),
                    **metrics,
                }
                partitions.append(record)
                by_symbol_day[(symbol, day_value)] = record
        after = {symbol: {"selected": bool(mt5.symbol_info(symbol).select), "visible": bool(mt5.symbol_info(symbol).visible)} for symbol in universe}
    finally:
        mt5.shutdown()

    per_symbol = {}
    for symbol in protocol["fixed_universe"]:
        rows = [item for item in partitions if item["symbol"] == symbol]
        pre = [value for item in rows for value in item["pre_boundary_ok"]]
        post = [value for item in rows for value in item["post_boundary_ok"]]
        per_symbol[symbol] = {
            "total_rows": int(sum(item["rows"] for item in rows)),
            "days_with_at_least_1000_ticks": int(sum(item["rows"] >= 1000 for item in rows)),
            "minimum_valid_bidask_fraction": float(min(item["valid_bidask_fraction"] for item in rows)),
            "all_partitions_timestamp_monotonic": all(item["timestamp_monotonic"] for item in rows),
            "pre_boundary_coverage": float(np.mean(pre)),
            "post_boundary_coverage": float(np.mean(post)),
        }
    shared = []
    for day_value in protocol["sample_days"]:
        for position in range(len(boundaries(date.fromisoformat(day_value)))):
            shared.append(all(
                by_symbol_day[(symbol, day_value)]["pre_boundary_ok"][position]
                and by_symbol_day[(symbol, day_value)]["post_boundary_ok"][position]
                for symbol in protocol["fixed_universe"]
            ))
    gate = protocol["data_preflight"]
    checks = {
        "symbols_selected_state_unchanged": before == after and all(item["selected"] for item in before.values()),
        "no_symbol_select_calls": True,
        "minimum_valid_bidask_fraction": all(item["minimum_valid_bidask_fraction"] >= gate["minimum_valid_bidask_fraction"] for item in per_symbol.values()),
        "all_timestamps_monotonic": all(item["all_partitions_timestamp_monotonic"] for item in per_symbol.values()),
        "minimum_days_with_1000_ticks_per_symbol": all(item["days_with_at_least_1000_ticks"] >= gate["minimum_days_with_1000_ticks_per_symbol"] for item in per_symbol.values()),
        "minimum_pre_boundary_coverage_per_symbol": all(item["pre_boundary_coverage"] >= gate["minimum_pre_boundary_coverage_per_symbol"] for item in per_symbol.values()),
        "minimum_post_boundary_coverage_per_symbol": all(item["post_boundary_coverage"] >= gate["minimum_post_boundary_coverage_per_symbol"] for item in per_symbol.values()),
        "minimum_all_symbol_shared_boundary_coverage": float(np.mean(shared)) >= gate["minimum_all_symbol_shared_boundary_coverage"],
        "outcome_access_zero": all(int(value) == 0 for value in protocol["outcome_access"].values()),
    }
    passed = all(checks.values())
    payload = {
        "schema_version": 1,
        "audit_id": protocol["cycle_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_SYNCHRONOUS_CONTEXT_TICK_PREFLIGHT" if passed else "FAIL_SYNCHRONOUS_CONTEXT_TICK_PREFLIGHT_CLOSE_PATH",
        "decision": "ALLOW_FIXED_FULL_BACKFILL_PROTOCOL" if passed else "CLOSE_WITHOUT_FULL_BACKFILL_OR_MODEL",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "terminal": terminal,
        "provider": "Exness MT5 Real same-account feed",
        "functions_called": ["initialize", "symbol_info", "copy_ticks_range", "shutdown"],
        "functions_not_called": ["symbol_select", "order_send", "order_check", "positions_get", "orders_get"],
        "selected_state_before": before,
        "selected_state_after": after,
        "partitions": partitions,
        "per_symbol": per_symbol,
        "shared_boundaries": {"rows": len(shared), "coverage": float(np.mean(shared))},
        "checks": checks,
        "outcome_access": protocol["outcome_access"],
        "safety": protocol["safety"],
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "decision": payload["decision"], "shared": payload["shared_boundaries"], "per_symbol": per_symbol}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
