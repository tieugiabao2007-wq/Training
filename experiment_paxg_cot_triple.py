from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT, SETTINGS
from gold_ai.data.context import CftcGoldCotProvider, align_point_in_time_context
from gold_ai.data.providers import (
    BinancePaxgProvider,
    MarketDataProvider,
    SourceMetadata,
    load_saved_market_data,
    sha256_file,
)
from gold_ai.training import run_triple_barrier_training


class SnapshotProvider(MarketDataProvider):
    def __init__(self, frame, metadata: SourceMetadata):
        self.frame = frame
        self.metadata = metadata

    def fetch_bars(self, limit=None):
        return self.frame.tail(limit).copy() if limit else self.frame.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sealed development experiment: Binance PAXG plus point-in-time CFTC Gold COT"
    )
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument(
        "--snapshot-csv",
        default="",
        help="verified saved Binance snapshot; avoids changing the comparison sample",
    )
    parser.add_argument("--interval", default="5m", choices=("1m", "3m", "5m", "15m", "30m", "1h"))
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--atr-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--model-profile",
        default="max_accelerated",
        choices=("memory_safe", "full", "gpu_only", "max_accelerated", "all_models", "auto"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paxg = None
    if args.snapshot_csv:
        bars, paxg_metadata = load_saved_market_data(args.snapshot_csv)
        if paxg_metadata.interval != args.interval:
            raise ValueError(
                f"Snapshot interval {paxg_metadata.interval!r} != requested {args.interval!r}"
            )
    else:
        paxg = BinancePaxgProvider(
            SETTINGS.binance_symbol,
            args.interval,
            args.start,
            args.end,
        )
        bars = paxg.fetch_bars()
        paxg_metadata = paxg.metadata
    start_year = pd_timestamp_year(args.start)
    end_year = pd_timestamp_year(args.end) if args.end else datetime.now(timezone.utc).year
    cot_provider = CftcGoldCotProvider(start_year, end_year)
    cot = cot_provider.fetch_features()
    enriched = align_point_in_time_context(bars, cot)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if paxg is not None:
        snapshot_csv, snapshot_metadata = paxg.save(
            bars, f"binance_{paxg.symbol}_{args.interval}_snapshot_{stamp}"
        )
    else:
        snapshot_csv = Path(args.snapshot_csv).resolve()
        snapshot_metadata = snapshot_csv.with_suffix(".metadata.json")
    context_dir = PROJECT_ROOT / "data" / "raw"
    context_csv = context_dir / f"cftc_gold_cot_context_{start_year}_{end_year}_{stamp}.csv"
    context_meta = context_dir / f"cftc_gold_cot_context_{start_year}_{end_year}_{stamp}.metadata.json"
    cot.to_csv(context_csv, index_label="available_at")
    context_payload = cot_provider.metadata.to_dict()
    context_payload["normalized_context_csv"] = str(context_csv)
    context_payload["normalized_context_sha256"] = sha256_file(context_csv)
    context_meta.write_text(json.dumps(context_payload, indent=2), encoding="utf-8")

    source_metadata = SourceMetadata(
        provider="Binance+CFTC point-in-time",
        instrument=f"{paxg_metadata.instrument}+CFTC_GOLD_088691",
        interval=args.interval,
        timezone="UTC",
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        intended_use="multi_source_gold_proxy_research_only",
        source_url=";".join([paxg_metadata.source_url, *cot_provider.metadata.source_urls]),
        notes=(
            "PAXGUSDT remains the prediction target proxy. CFTC COMEX Gold positioning is "
            "joined backward from a conservative availability timestamp; exact official "
            "catch-up timestamps are used for the 2025 funding-lapse backlog. CFTC features "
            "are slow regime context and must not be interpreted as intraday execution quotes."
        ),
        retrieval_method="Binance Spot REST plus official CFTC yearly ZIP archives",
        license_note="Official/public endpoints; retain checksums and verify redistribution terms.",
    )
    settings = replace(
        SETTINGS,
        data_provider="binance_paxg_research",
        bar_interval=args.interval,
        forecast_horizon_bars=args.horizon,
        atr_barrier_multiplier=args.atr_multiplier,
        walk_forward_splits=args.splits,
        walk_forward_max_train_rows=args.max_train_rows,
        binance_start_utc=args.start,
        binance_end_utc=args.end,
        model_profile=args.model_profile,
    )
    config_tag = (
        f"{args.model_profile}_atr{args.atr_multiplier:g}".replace(".", "p").replace("-", "m")
    )
    split_tag = f"_wf{args.splits}" if args.splits != 5 else ""
    train_tag = f"_tw{args.max_train_rows}" if args.max_train_rows else ""
    name = f"paxg_cot_triple_{args.interval}_h{args.horizon}_{config_tag}{split_tag}{train_tag}_development"
    artifact_dir = PROJECT_ROOT / "artifacts" / "experiments" / name
    _, summary = run_triple_barrier_training(
        settings,
        provider=SnapshotProvider(enriched, source_metadata),
        artifact_dir=artifact_dir,
        open_lockbox=False,
    )
    report = {
        "protocol": "sealed PAXG target plus point-in-time CFTC COT development experiment",
        "artifact": str(artifact_dir),
        "source_snapshot_csv": str(snapshot_csv),
        "source_snapshot_metadata": str(snapshot_metadata),
        "context_csv": str(context_csv),
        "context_metadata": str(context_meta),
        "rows_bars": int(len(bars)),
        "rows_context": int(len(cot)),
        "context_features": list(cot.columns),
        "lockbox_decision": "kept_sealed_for_development_only",
        "summary": summary,
    }
    output = PROJECT_ROOT / "artifacts" / "reports" / f"{name}.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"artifact": str(artifact_dir), "validation": summary["validation"]}, default=str))
    print(f"Saved {output}")


def pd_timestamp_year(value: str) -> int:
    # Local import keeps CLI startup focused and avoids a second timestamp parser.
    import pandas as pd

    return int(pd.Timestamp(value).year)


if __name__ == "__main__":
    main()
