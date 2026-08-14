from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT


EXACT_TICK_START_UTC = pd.Timestamp("2026-07-17T00:00:00Z")
HISTORY_SUFFIXES = {".tkc", ".tks", ".hc", ".hcc", ".fxt", ".hst"}


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit older MT5/Exness tick availability without sending orders."
    )
    parser.add_argument("--terminal-path", default=r"D:\Gold\terminal64.exe")
    parser.add_argument("--symbol", default="XAUUSDm")
    return parser.parse_args()


def tick_bounds(ticks) -> dict:
    if ticks is None or len(ticks) == 0:
        return {"rows": 0, "first_timestamp": None, "last_timestamp": None}
    milliseconds = ticks["time_msc"]
    return {
        "rows": int(len(ticks)),
        "first_timestamp": pd.to_datetime(milliseconds[0], unit="ms", utc=True).isoformat(),
        "last_timestamp": pd.to_datetime(milliseconds[-1], unit="ms", utc=True).isoformat(),
    }


def history_candidates(roots: list[Path], symbol: str) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            try:
                if not path.is_file():
                    continue
            except OSError:
                continue
            lowered = str(path).lower()
            relevant_suffix = path.suffix.lower() in HISTORY_SUFFIXES
            relevant_symbol = symbol.lower() in lowered
            if not relevant_suffix and not relevant_symbol:
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            try:
                stat = path.stat()
            except OSError:
                continue
            month_tokens = re.findall(r"20\d{4}", lowered)
            candidates.append(
                {
                    "path": str(path.resolve()),
                    "suffix": path.suffix.lower(),
                    "bytes": int(stat.st_size),
                    "last_write_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "symbol_in_path": relevant_symbol,
                    "tester_path": "tester" in lowered,
                    "year_month_tokens": sorted(set(month_tokens)),
                }
            )
    return sorted(candidates, key=lambda row: row["path"].lower())


def main() -> int:
    args = parse_args()
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 package is required") from exc
    if not mt5.initialize(path=args.terminal_path):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_contract": "V15.1_LEAN",
        "symbol": args.symbol,
        "read_only_account_audit": True,
        "no_order_functions_called": True,
        "no_trading": True,
        "api_range_probes": [],
        "api_from_probes": [],
    }
    roots: list[Path] = []
    try:
        terminal = mt5.terminal_info()
        if terminal is None or not terminal.connected:
            raise RuntimeError("MT5 terminal is not connected")
        symbol_info = mt5.symbol_info(args.symbol)
        if symbol_info is None:
            raise RuntimeError(f"Symbol not found: {args.symbol}")
        if not bool(symbol_info.visible):
            raise RuntimeError(
                f"Symbol {args.symbol} is not already visible; refusing to mutate Market Watch"
            )

        data_path = Path(terminal.data_path)
        common_path = Path(terminal.commondata_path)
        terminal_parent = Path(args.terminal_path).resolve().parent
        appdata = Path(os.environ.get("APPDATA", ""))
        roots = [
            data_path / "bases",
            data_path / "Tester",
            common_path / "Tester",
            terminal_parent,
            appdata / "MetaQuotes" / "Terminal",
        ]
        report["local_roots_checked"] = [str(path.resolve()) for path in roots if path.exists()]

        probe_days = [
            "2024-01-02",
            "2025-01-02",
            "2026-01-02",
            "2026-06-01",
            "2026-07-01",
            "2026-07-16",
            "2026-07-17",
        ]
        for day in probe_days:
            start = pd.Timestamp(day, tz="UTC")
            end = start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            ticks = mt5.copy_ticks_range(
                args.symbol,
                start.to_pydatetime(),
                end.to_pydatetime(),
                mt5.COPY_TICKS_ALL,
            )
            row = {"start_utc": start.isoformat(), "end_utc": end.isoformat()}
            row.update(tick_bounds(ticks))
            if ticks is None:
                row["last_error"] = list(mt5.last_error())
            report["api_range_probes"].append(row)

        for start_value in [
            "2024-01-01T00:00:00Z",
            "2025-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
        ]:
            start = pd.Timestamp(start_value)
            ticks = mt5.copy_ticks_from(
                args.symbol, start.to_pydatetime(), 10, mt5.COPY_TICKS_ALL
            )
            row = {"requested_from_utc": start.isoformat()}
            row.update(tick_bounds(ticks))
            if ticks is None:
                row["last_error"] = list(mt5.last_error())
            report["api_from_probes"].append(row)
    finally:
        mt5.shutdown()

    candidates = history_candidates(roots, args.symbol)
    report["local_history_candidates"] = candidates
    api_timestamps = [
        pd.Timestamp(row["first_timestamp"])
        for group in (report["api_range_probes"], report["api_from_probes"])
        for row in group
        if row["first_timestamp"]
    ]
    api_earliest = min(api_timestamps) if api_timestamps else None
    older_api_tick_found = bool(api_earliest is not None and api_earliest < EXACT_TICK_START_UTC)
    older_cache_candidates = [
        row
        for row in candidates
        if row["suffix"] in {".tkc", ".tks"}
        and row["symbol_in_path"]
        and any(token < "202607" for token in row["year_month_tokens"])
    ]
    older_tester_candidates = [row for row in older_cache_candidates if row["tester_path"]]
    report["conclusion"] = {
        "earliest_api_tick_utc": api_earliest.isoformat() if api_earliest is not None else None,
        "older_exact_api_tick_found": older_api_tick_found,
        "older_exact_local_cache_candidate_found": bool(older_cache_candidates),
        "older_exact_strategy_tester_candidate_found": bool(older_tester_candidates),
        "older_exact_tick_history_found": bool(
            older_api_tick_found or older_cache_candidates or older_tester_candidates
        ),
        "strategy_tester_bar_files_are_not_assumed_to_be_real_ticks": True,
        "account_or_order_state_mutated": False,
    }
    report_dir = PROJECT_ROOT / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"mt5_tick_history_audit_{stamp}.json"
    atomic_write_text(report_path, json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "report": str(report_path),
                "api_range_probes": report["api_range_probes"],
                "api_from_probes": report["api_from_probes"],
                "history_candidate_count": len(candidates),
                "conclusion": report["conclusion"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
