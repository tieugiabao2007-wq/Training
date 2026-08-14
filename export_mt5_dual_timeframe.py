from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT
from gold_ai.data.providers import MT5Provider
from gold_ai.data.quality import validate_and_clean_bars


INTERVAL_SECONDS = {"M5": 300, "M15": 900}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export immutable, completed-bar Exness MT5 M5/M15 snapshots."
    )
    parser.add_argument("--terminal-path", default=r"D:\Gold\terminal64.exe")
    parser.add_argument("--symbol", default="XAUUSDm")
    parser.add_argument("--max-bars", type=int, default=300_000)
    parser.add_argument("--timeframes", nargs="+", default=["M5", "M15"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = [label.upper() for label in args.timeframes]
    invalid = [label for label in requested if label not in INTERVAL_SECONDS]
    if invalid:
        raise ValueError(f"Only M5/M15 trading snapshots are allowed; got {invalid}")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal_path": str(Path(args.terminal_path).resolve()),
        "requested_symbol": args.symbol,
        "trade_timeframe_binding": "M5 and M15 exported as independent snapshots",
        "read_only": True,
        "snapshots": {},
    }
    for timeframe in requested:
        provider = MT5Provider(
            symbol=args.symbol,
            timeframe=timeframe,
            terminal_path=args.terminal_path,
            default_limit=args.max_bars,
        )
        raw = provider.fetch_bars(limit=args.max_bars)
        clean, quality = validate_and_clean_bars(
            raw,
            expected_interval_seconds=INTERVAL_SECONDS[timeframe],
            min_rows=1_000,
        )
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(clean, index=True).values.tobytes()
        ).hexdigest()
        stem = f"mt5_{provider.symbol}_{timeframe}_{fingerprint[:16]}"
        csv_path, metadata_path = provider.save(clean, stem)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest["snapshots"][timeframe] = {
            "csv": str(csv_path),
            "metadata": str(metadata_path),
            "file_sha256": metadata["file_sha256"],
            "rows": len(clean),
            "first_timestamp": clean.index.min().isoformat(),
            "last_timestamp": clean.index.max().isoformat(),
            "quality": quality.to_dict(),
            "spread_statistics": metadata.get("spread_statistics", {}),
            "complete_bar_policy": metadata.get("complete_bar_policy"),
        }
    report_dir = PROJECT_ROOT / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    immutable = report_dir / f"mt5_dual_timeframe_snapshot_{stamp}.json"
    latest = report_dir / "mt5_dual_timeframe_latest.json"
    payload = json.dumps(manifest, indent=2)
    immutable.write_text(payload, encoding="utf-8")
    latest.write_text(payload, encoding="utf-8")
    print(json.dumps({"manifest": str(immutable), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
