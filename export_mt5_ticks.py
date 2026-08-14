from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT
from gold_ai.data.providers import sha256_file


def utc_timestamp(value: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tz is None:
        raise ValueError("Tick export bounds must include a timezone")
    return parsed.tz_convert("UTC")


def daily_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if start != start.floor("D") or end != end.floor("D"):
        raise ValueError("Tick export bounds must be UTC day boundaries")
    if end <= start:
        raise ValueError("end must be after start")
    return [(day, day + pd.Timedelta(days=1)) for day in pd.date_range(start, end, inclusive="left")]


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export completed UTC-day MT5 tick partitions read-only.")
    parser.add_argument("--terminal-path", default=r"D:\Gold\terminal64.exe")
    parser.add_argument("--symbol", default="XAUUSDm")
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--end-utc", required=True, help="Exclusive UTC day boundary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = utc_timestamp(args.start_utc)
    end = utc_timestamp(args.end_utc)
    current_utc_day = pd.Timestamp.now(tz="UTC").floor("D")
    if end > current_utc_day:
        raise ValueError("Only completed UTC days may be exported as immutable partitions")
    windows = daily_windows(start, end)
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 package is required") from exc
    if not mt5.initialize(path=args.terminal_path):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_contract": "V15.1_LEAN",
        "family_id": "XAUUSD_M5_EXACT_TICK_MICROSTRUCTURE_V15_1",
        "symbol": args.symbol,
        "start_utc": start.isoformat(),
        "end_utc_exclusive": end.isoformat(),
        "read_only": True,
        "no_trading": True,
        "partitions": [],
    }
    try:
        symbol_info = mt5.symbol_info(args.symbol)
        if symbol_info is None or not bool(getattr(symbol_info, "visible", True)):
            raise RuntimeError(
                f"Symbol {args.symbol} is not already visible; read-only export refuses to mutate Market Watch"
            )
        for day_start, day_end in windows:
            day = day_start.strftime("%Y-%m-%d")
            partition_dir = (
                PROJECT_ROOT
                / "data"
                / "raw"
                / "mt5_ticks"
                / args.symbol
                / day_start.strftime("%Y")
                / day_start.strftime("%m")
            )
            partition_dir.mkdir(parents=True, exist_ok=True)
            csv_path = partition_dir / f"{args.symbol}_ticks_{day}.csv"
            metadata_path = csv_path.with_suffix(".metadata.json")
            if csv_path.exists() or metadata_path.exists():
                if not csv_path.exists() or not metadata_path.exists():
                    raise FileExistsError(f"Incomplete existing partition: {csv_path}")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if sha256_file(csv_path) != metadata.get("file_sha256"):
                    raise RuntimeError(f"Existing partition checksum mismatch: {csv_path}")
                manifest["partitions"].append(metadata)
                continue
            range_end = day_end.to_pydatetime() - timedelta(microseconds=1)
            ticks = mt5.copy_ticks_range(
                args.symbol,
                day_start.to_pydatetime(),
                range_end,
                mt5.COPY_TICKS_ALL,
            )
            if ticks is None:
                raise RuntimeError(f"MT5 tick request failed for {day}: {mt5.last_error()}")
            frame = pd.DataFrame(ticks)
            if len(frame):
                start_ms = int(day_start.timestamp() * 1000)
                end_ms = int(day_end.timestamp() * 1000)
                frame = frame.loc[(frame["time_msc"] >= start_ms) & (frame["time_msc"] < end_ms)].copy()
                frame.insert(0, "timestamp", pd.to_datetime(frame["time_msc"], unit="ms", utc=True))
                valid_bidask = (frame["bid"] > 0) & (frame["ask"] > 0) & (frame["ask"] >= frame["bid"])
                exact_duplicates = int(frame.duplicated().sum())
            else:
                valid_bidask = pd.Series(dtype=bool)
                exact_duplicates = 0
            temporary_csv = csv_path.with_name(f"{csv_path.name}.{os.getpid()}.tmp")
            frame.to_csv(temporary_csv, index=False)
            os.replace(temporary_csv, csv_path)
            file_sha256 = sha256_file(csv_path)
            metadata = {
                "provider": "MetaTrader 5 local terminal",
                "source_url": f"local://mt5/{args.symbol}/ticks/{day}",
                "retrieval_method": "MetaTrader5.copy_ticks_range(COPY_TICKS_ALL)",
                "license_note": "Private account feed; artifact remains local and is not redistributed.",
                "symbol": args.symbol,
                "utc_day": day,
                "complete_day": True,
                "read_only": True,
                "no_trading": True,
                "rows": int(len(frame)),
                "valid_bidask_pct": float(valid_bidask.mean() * 100) if len(frame) else None,
                "exact_duplicate_rows": exact_duplicates,
                "first_timestamp": frame["timestamp"].iloc[0].isoformat() if len(frame) else None,
                "last_timestamp": frame["timestamp"].iloc[-1].isoformat() if len(frame) else None,
                "csv": str(csv_path),
                "file_sha256": file_sha256,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_text(metadata_path, json.dumps(metadata, indent=2))
            manifest["partitions"].append(metadata)
            print(f"[tick-export] {day} rows={len(frame)}", flush=True)
    finally:
        mt5.shutdown()

    report_dir = PROJECT_ROOT / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"mt5_tick_snapshot_{stamp}.json"
    manifest["total_rows"] = sum(int(row["rows"]) for row in manifest["partitions"])
    manifest["nonempty_days"] = sum(int(row["rows"]) > 0 for row in manifest["partitions"])
    atomic_write_text(report_path, json.dumps(manifest, indent=2))
    print(json.dumps({"manifest": str(report_path), "total_rows": manifest["total_rows"], "nonempty_days": manifest["nonempty_days"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
