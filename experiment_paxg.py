from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT, SETTINGS
from gold_ai.data.providers import BinancePaxgProvider, MarketDataProvider, SourceMetadata
from gold_ai.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-resistant PAXG proxy experiments")
    parser.add_argument("--start", default="2020-01-01", help="inclusive UTC start")
    parser.add_argument("--end", default="", help="exclusive UTC end; empty means now")
    parser.add_argument("--interval", default="5m", choices=("1m", "3m", "5m", "15m", "30m", "1h"))
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 12])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = BinancePaxgProvider(
        SETTINGS.binance_symbol,
        args.interval,
        args.start,
        args.end,
    )
    snapshot = source.fetch_bars()

    class SnapshotProvider(MarketDataProvider):
        def __init__(self, frame, metadata: SourceMetadata):
            self.frame = frame
            self.metadata = metadata

        def fetch_bars(self, limit=None):
            return self.frame.tail(limit).copy() if limit else self.frame.copy()

    results = []
    for horizon in args.horizons:
        name = f"paxg_{args.interval}_h{horizon}"
        settings = replace(
            SETTINGS,
            data_provider="binance_paxg_research",
            bar_interval=args.interval,
            forecast_horizon_bars=horizon,
            binance_start_utc=args.start,
            binance_end_utc=args.end,
        )
        provider = SnapshotProvider(snapshot, source.metadata)
        output = PROJECT_ROOT / "artifacts" / "experiments" / name
        _, summary = run_training(settings, provider=provider, artifact_dir=output)
        row = {"name": name, "horizon": horizon, **summary["validation"]}
        results.append(row)
        print(json.dumps(row))

    path = PROJECT_ROOT / "artifacts" / "reports" / "paxg_experiments.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
