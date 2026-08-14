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
        description="Run a sealed feature-set experiment on an immutable verified PAXG snapshot"
    )
    parser.add_argument("--snapshot-csv", required=True)
    parser.add_argument("--experiment-tag", default="traps_v2")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--atr-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--model-profile",
        default="max_accelerated",
        choices=("memory_safe", "full", "gpu_only", "max_accelerated", "all_cpu_models", "all_models", "auto"),
    )
    args = parser.parse_args()
    bars, metadata = load_saved_market_data(args.snapshot_csv)
    interval = metadata.interval
    settings = replace(
        SETTINGS,
        data_provider="binance_paxg_research",
        bar_interval=interval,
        forecast_horizon_bars=args.horizon,
        atr_barrier_multiplier=args.atr_multiplier,
        walk_forward_splits=args.splits,
        walk_forward_max_train_rows=args.max_train_rows,
        model_profile=args.model_profile,
    )
    safe_tag = "".join(ch for ch in args.experiment_tag.lower() if ch.isalnum() or ch in "_-")
    config_tag = (
        f"{args.model_profile}_atr{args.atr_multiplier:g}".replace(".", "p").replace("-", "m")
    )
    split_tag = f"_wf{args.splits}" if args.splits != 5 else ""
    train_tag = f"_tw{args.max_train_rows}" if args.max_train_rows else ""
    name = f"paxg_{safe_tag}_triple_{interval}_h{args.horizon}_{config_tag}{split_tag}{train_tag}_development"
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
        "protocol": "immutable-snapshot sealed feature-set development comparison",
        "artifact": str(artifact_dir),
        "snapshot_csv": str(Path(args.snapshot_csv).resolve()),
        "snapshot_sha256": summary["metadata"]["source_file_sha256"],
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
