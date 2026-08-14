from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT, SETTINGS
from gold_ai.data.context import FredVintageContextProvider, align_fred_point_in_time_context
from gold_ai.data.providers import (
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
        description="Sealed PAXG experiment with point-in-time FRED initial releases"
    )
    parser.add_argument("--snapshot-csv", required=True)
    parser.add_argument("--fred-api-key-env", default="FRED_API_KEY")
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
    api_key = os.getenv(args.fred_api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"{args.fred_api_key_env} is not set; refusing to substitute revision-leaking latest CSV"
        )
    bars, paxg_metadata = load_saved_market_data(args.snapshot_csv)
    observation_start = bars.index.min().date().isoformat()
    observation_end = bars.index.max().date().isoformat()
    fred_provider = FredVintageContextProvider(
        api_key=api_key,
        observation_start=observation_start,
        observation_end=observation_end,
    )
    fred = fred_provider.fetch_features()
    enriched = align_fred_point_in_time_context(bars, fred)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    context_dir = PROJECT_ROOT / "data" / "raw"
    context_csv = context_dir / f"fred_initial_release_gold_context_{stamp}.csv"
    context_meta = context_dir / f"fred_initial_release_gold_context_{stamp}.metadata.json"
    fred.to_csv(context_csv, index_label="available_at")
    context_payload = fred_provider.metadata.to_dict()
    context_payload["normalized_context_csv"] = str(context_csv)
    context_payload["normalized_context_sha256"] = sha256_file(context_csv)
    context_meta.write_text(json.dumps(context_payload, indent=2), encoding="utf-8")

    source_metadata = SourceMetadata(
        provider="Binance+FRED initial-release point-in-time",
        instrument=f"{paxg_metadata.instrument}+DFII10+DGS10+DTWEXBGS+VIXCLS",
        interval=paxg_metadata.interval,
        timezone="UTC",
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        intended_use="multi_source_gold_proxy_research_only",
        source_url=";".join([paxg_metadata.source_url, *fred_provider.metadata.source_urls]),
        notes=(
            "PAXGUSDT remains the prediction proxy. FRED macro context uses output_type=4 "
            "initial releases and a conservative next-day 23:59 Eastern availability. "
            "No latest-vintage backfill and no API key are stored."
        ),
        retrieval_method="Verified Binance snapshot plus official FRED/ALFRED API v1 JSON",
        license_note="Official/public endpoints; retain series IDs, checksums and terms.",
    )
    settings = replace(
        SETTINGS,
        data_provider="binance_paxg_research",
        bar_interval=paxg_metadata.interval,
        forecast_horizon_bars=args.horizon,
        atr_barrier_multiplier=args.atr_multiplier,
        walk_forward_splits=args.splits,
        walk_forward_max_train_rows=args.max_train_rows,
        model_profile=args.model_profile,
    )
    config_tag = (
        f"{args.model_profile}_atr{args.atr_multiplier:g}".replace(".", "p").replace("-", "m")
    )
    split_tag = f"_wf{args.splits}" if args.splits != 5 else ""
    train_tag = f"_tw{args.max_train_rows}" if args.max_train_rows else ""
    name = (
        f"paxg_fred_initial_triple_{paxg_metadata.interval}_h{args.horizon}_"
        f"{config_tag}{split_tag}{train_tag}_development"
    )
    artifact_dir = PROJECT_ROOT / "artifacts" / "experiments" / name
    if (artifact_dir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {artifact_dir}")
    _, summary = run_triple_barrier_training(
        settings,
        provider=SnapshotProvider(enriched, source_metadata),
        artifact_dir=artifact_dir,
        open_lockbox=False,
    )
    report = {
        "protocol": "sealed PAXG target plus point-in-time FRED initial-release context",
        "artifact": str(artifact_dir),
        "snapshot_csv": str(Path(args.snapshot_csv).resolve()),
        "context_csv": str(context_csv),
        "context_metadata": str(context_meta),
        "rows_bars": int(len(bars)),
        "rows_context_events": int(len(fred)),
        "context_features": list(fred.columns),
        "lockbox_decision": "kept_sealed_for_development_only",
        "summary": summary,
    }
    output = PROJECT_ROOT / "artifacts" / "reports" / f"{name}.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"artifact": str(artifact_dir), "validation": summary["validation"]}))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
