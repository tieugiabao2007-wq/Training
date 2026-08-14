from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT, SETTINGS
from gold_ai.data.providers import MarketDataProvider, SourceMetadata, load_saved_market_data
from gold_ai.training import run_triple_barrier_training


class SnapshotProvider(MarketDataProvider):
    def __init__(self, frame, metadata: SourceMetadata):
        self.frame = frame
        self.metadata = metadata

    def fetch_bars(self, limit=None):
        return self.frame.tail(limit).copy() if limit else self.frame.copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a sealed M5 or M15 experiment on an immutable Exness MT5 snapshot."
    )
    parser.add_argument("--snapshot-csv", required=True)
    parser.add_argument("--experiment-tag", default="exact_feed_baseline")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--atr-multiplier", type=float, default=1.0)
    parser.add_argument("--round-trip-cost-bps", type=float)
    parser.add_argument("--extra-slippage-bps", type=float, default=1.0)
    parser.add_argument(
        "--model-profile",
        default="all_cpu_models",
        choices=("memory_safe", "full", "all_cpu_models", "all_models", "auto"),
    )
    args = parser.parse_args()
    snapshot = Path(args.snapshot_csv).resolve()
    bars, metadata = load_saved_market_data(snapshot)
    if metadata.provider != "MetaTrader 5 local terminal":
        raise ValueError(f"Expected exact MT5 snapshot; got {metadata.provider!r}")
    timeframe = metadata.interval.upper()
    if timeframe not in {"M5", "M15"}:
        raise ValueError(f"Trade timeframe must be M5 or M15; got {timeframe!r}")
    if "XAUUSD" not in metadata.instrument.upper():
        raise ValueError(f"Expected XAUUSD broker symbol; got {metadata.instrument!r}")
    raw_metadata = json.loads(snapshot.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    spread_p99 = float(raw_metadata.get("spread_statistics", {}).get("spread_bps_p99", 0.0))
    cost_bps = (
        float(args.round_trip_cost_bps)
        if args.round_trip_cost_bps is not None
        else max(float(SETTINGS.round_trip_cost_bps), spread_p99 + args.extra_slippage_bps)
    )
    settings = replace(
        SETTINGS,
        data_provider="mt5",
        mt5_symbol=metadata.instrument,
        mt5_timeframe=timeframe,
        forecast_horizon_bars=args.horizon,
        atr_barrier_multiplier=args.atr_multiplier,
        round_trip_cost_bps=cost_bps,
        walk_forward_splits=args.splits,
        walk_forward_max_train_rows=args.max_train_rows,
        model_profile=args.model_profile,
        experiment_family_id=f"XAUUSD_{timeframe}_EXACT_FEED_V15",
        strategy_variant_id=f"{timeframe}_TRIPLE_BARRIER_H{args.horizon}",
    )
    safe_tag = "".join(ch for ch in args.experiment_tag.lower() if ch.isalnum() or ch in "_-")
    cost_tag = f"cost{cost_bps:g}".replace(".", "p")
    config_tag = f"{args.model_profile}_atr{args.atr_multiplier:g}_{cost_tag}".replace(".", "p")
    split_tag = f"_wf{args.splits}" if args.splits != 5 else ""
    train_tag = f"_tw{args.max_train_rows}" if args.max_train_rows else ""
    instrument_tag = "".join(ch.lower() for ch in metadata.instrument if ch.isalnum())
    name = (
        f"mt5_{instrument_tag}_{timeframe.lower()}_{safe_tag}_triple_h{args.horizon}_"
        f"{config_tag}{split_tag}{train_tag}_development"
    )
    artifact_dir = PROJECT_ROOT / "artifacts" / "experiments" / name
    if (artifact_dir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {artifact_dir}")
    _, summary = run_triple_barrier_training(
        settings,
        provider=SnapshotProvider(bars, metadata),
        artifact_dir=artifact_dir,
        open_lockbox=False,
    )
    report = {
        "protocol": "exact Exness MT5 immutable snapshot; sealed development lockbox",
        "artifact": str(artifact_dir),
        "snapshot_csv": str(snapshot),
        "snapshot_sha256": summary["metadata"]["source_file_sha256"],
        "trade_timeframe": timeframe,
        "pipeline_binding": {
            "training": timeframe,
            "target": timeframe,
            "signal": timeframe,
            "execution": timeframe,
            "management": timeframe,
        },
        "round_trip_cost_bps": cost_bps,
        "spread_p99_bps": spread_p99,
        "extra_slippage_bps": args.extra_slippage_bps,
        "feature_set_version": summary["metadata"]["feature_set_version"],
        "lockbox_decision": "kept_sealed_for_development_only",
        "summary": summary,
    }
    output = PROJECT_ROOT / "artifacts" / "reports" / f"{name}.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"artifact": str(artifact_dir), "validation": summary["validation"]}, default=str))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
