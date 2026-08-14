from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT
from gold_ai.data.providers import MT5Provider
from gold_ai.data.quality import validate_and_clean_bars


INTERVAL_SECONDS = {"M5": 300, "M15": 900}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export immutable completed-bar MT5 context snapshots without trading."
    )
    parser.add_argument("--terminal-path", default=r"D:\Gold\terminal64.exe")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--timeframe", choices=tuple(INTERVAL_SECONDS), default="M5")
    parser.add_argument("--max-bars", type=int, default=99_999)
    parser.add_argument("--hypothesis-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_contract": "V15.1_LEAN",
        "hypothesis_id": args.hypothesis_id,
        "terminal_path": str(Path(args.terminal_path).resolve()),
        "read_only": True,
        "no_trading": True,
        "timeframe": args.timeframe,
        "snapshots": {},
    }
    for requested_symbol in args.symbols:
        provider = MT5Provider(
            symbol=requested_symbol,
            timeframe=args.timeframe,
            terminal_path=args.terminal_path,
            default_limit=args.max_bars,
        )
        raw = provider.fetch_bars(limit=args.max_bars)
        clean, quality = validate_and_clean_bars(
            raw,
            expected_interval_seconds=INTERVAL_SECONDS[args.timeframe],
            min_rows=10_000,
        )
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(clean, index=True).values.tobytes()
        ).hexdigest()
        stem = f"mt5_context_{provider.symbol}_{args.timeframe}_{fingerprint[:16]}"
        expected_csv = PROJECT_ROOT / "data" / "raw" / f"{stem}.csv"
        if expected_csv.exists():
            raise FileExistsError(f"Refusing to overwrite immutable context snapshot: {expected_csv}")
        csv_path, metadata_path = provider.save(clean, stem)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest["snapshots"][provider.symbol] = {
            "csv": str(csv_path),
            "metadata": str(metadata_path),
            "file_sha256": metadata["file_sha256"],
            "rows": len(clean),
            "first_timestamp": clean.index.min().isoformat(),
            "last_timestamp": clean.index.max().isoformat(),
            "quality": quality.to_dict(),
            "complete_bar_policy": metadata.get("complete_bar_policy"),
        }
    report_dir = PROJECT_ROOT / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = report_dir / f"mt5_context_snapshot_{stamp}.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
