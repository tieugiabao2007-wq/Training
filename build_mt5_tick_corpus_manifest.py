from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def day_in_scope(
    day: date,
    start_day: date | None,
    end_day_exclusive: date | None,
) -> bool:
    """Return whether a UTC partition day is inside the optional frozen scope."""
    return (start_day is None or day >= start_day) and (
        end_day_exclusive is None or day < end_day_exclusive
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an immutable manifest over every exact MT5 tick partition.")
    parser.add_argument("--tick-root", type=Path, required=True)
    parser.add_argument("--symbol", default="XAUUSDm")
    parser.add_argument("--history-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--start-day",
        type=date.fromisoformat,
        help="Optional inclusive UTC day (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-day-exclusive",
        type=date.fromisoformat,
        help="Optional exclusive UTC day (YYYY-MM-DD).",
    )
    args = parser.parse_args()
    if (
        args.start_day is not None
        and args.end_day_exclusive is not None
        and args.end_day_exclusive <= args.start_day
    ):
        raise ValueError("end-day-exclusive must be after start-day")
    root = args.tick_root.resolve()
    sidecars = sorted(root.rglob(f"{args.symbol}_ticks_*.metadata.json"))
    if not sidecars:
        raise RuntimeError("no tick partition sidecars")
    partitions = []
    days = set()
    for sidecar in sidecars:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        day = str(metadata.get("utc_day", ""))
        try:
            parsed_day = date.fromisoformat(day)
        except ValueError as exc:
            raise RuntimeError(f"invalid UTC day: {sidecar}") from exc
        if not day_in_scope(parsed_day, args.start_day, args.end_day_exclusive):
            continue
        if not day or day in days:
            raise RuntimeError(f"duplicate or missing UTC day: {sidecar}")
        if metadata.get("symbol") != args.symbol or metadata.get("complete_day") is not True:
            raise RuntimeError(f"invalid symbol/completeness binding: {sidecar}")
        if metadata.get("read_only") is not True or metadata.get("no_trading") is not True:
            raise RuntimeError(f"partition is not read-only/no-trading: {sidecar}")
        csv_path = Path(metadata["csv"]).resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        days.add(day)
        partitions.append(metadata)
    if not partitions:
        raise RuntimeError("no tick partitions inside requested scope")
    partitions.sort(key=lambda row: row["utc_day"])
    history_path = args.history_audit.resolve()
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if history.get("conclusion", {}).get("older_exact_tick_history_found") is not True:
        raise RuntimeError("history audit does not prove older exact broker ticks")
    first_day = date.fromisoformat(partitions[0]["utc_day"])
    end_day_exclusive = date.fromisoformat(partitions[-1]["utc_day"]) + timedelta(days=1)
    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_contract": "V15.1_LEAN",
        "family_id": "XAUUSD_EXACT_EXNESS_TICK_CORPUS",
        "symbol": args.symbol,
        "provider": "Exness MT5 Real account feed via MetaTrader5.copy_ticks_range",
        "tick_root": str(root),
        "history_audit": str(history_path),
        "history_audit_sha256": sha256_file(history_path),
        "start_utc": datetime.combine(first_day, datetime_time.min, timezone.utc).isoformat(),
        "end_utc_exclusive": datetime.combine(
            end_day_exclusive, datetime_time.min, timezone.utc
        ).isoformat(),
        "read_only": True,
        "no_trading": True,
        "account_or_order_state_mutated": False,
        "partitions": partitions,
        "total_rows": sum(int(row["rows"]) for row in partitions),
        "nonempty_days": sum(int(row["rows"]) > 0 for row in partitions),
        "partition_days": len(partitions),
        "requested_start_day": args.start_day.isoformat()
        if args.start_day is not None
        else None,
        "requested_end_day_exclusive": args.end_day_exclusive.isoformat()
        if args.end_day_exclusive is not None
        else None,
        "certification_history_gate_days": 60,
        "certification_history_gate_met": sum(
            int(row["rows"]) > 0 for row in partitions
        )
        >= 60,
    }
    atomic_write(args.output.resolve(), manifest)
    print(json.dumps({key: manifest[key] for key in ("start_utc", "end_utc_exclusive", "partition_days", "nonempty_days", "total_rows")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
