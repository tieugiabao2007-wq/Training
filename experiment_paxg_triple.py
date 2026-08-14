from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT, SETTINGS
from gold_ai.data.providers import (
    BinancePaxgProvider,
    MarketDataProvider,
    SourceMetadata,
    sha256_file,
)
from gold_ai.lockbox_registry import (
    claim_lockbox_once,
    finalize_lockbox_claim,
    lockbox_experiment_key,
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
        description="Develop triple-barrier PAXG tracks, then open one selected lockbox once"
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--interval", default="5m", choices=("1m", "3m", "5m", "15m", "30m", "1h"))
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument("--atr-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--model-profile",
        default="memory_safe",
        choices=("memory_safe", "full", "gpu_only", "max_accelerated", "all_models", "auto"),
        help="memory_safe is the default so a shorter dataset cannot accidentally enable ExtraTrees",
    )
    parser.add_argument("--force", action="store_true", help="ignore compatible development checkpoints")
    return parser.parse_args()


def selection_key(row: dict) -> tuple:
    validation = row["validation"]
    return (
        validation["development_qualified"],
        validation["accuracy_wilson_lower_95"],
        validation["net_return_sum"],
        validation["trades"],
    )


def main() -> None:
    args = parse_args()
    source = BinancePaxgProvider(
        SETTINGS.binance_symbol,
        args.interval,
        args.start,
        args.end,
    )
    snapshot = source.fetch_bars()
    snapshot_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_csv, snapshot_metadata = source.save(
        snapshot, f"binance_{source.symbol}_{args.interval}_snapshot_{snapshot_tag}"
    )
    snapshot_sha256 = sha256_file(snapshot_csv)
    developments = []
    config_tag = (
        f"{args.model_profile}_atr{args.atr_multiplier:g}"
        .replace(".", "p")
        .replace("-", "m")
    )
    for horizon in args.horizons:
        settings = replace(
            SETTINGS,
            data_provider="binance_paxg_research",
            bar_interval=args.interval,
            forecast_horizon_bars=horizon,
            atr_barrier_multiplier=args.atr_multiplier,
            binance_start_utc=args.start,
            binance_end_utc=args.end,
            model_profile=args.model_profile,
        )
        name = f"paxg_triple_{args.interval}_h{horizon}_{config_tag}_development"
        artifact_dir = PROJECT_ROOT / "artifacts" / "experiments" / name
        cached_path = artifact_dir / "summary.json"
        summary = None
        if cached_path.exists() and not args.force:
            candidate = json.loads(cached_path.read_text(encoding="utf-8"))
            labeling = candidate.get("metadata", {}).get("labeling", {})
            same_source = candidate.get("metadata", {}).get("source_file_sha256") == snapshot_sha256
            same_protocol = (
                candidate.get("training_mode") == "triple_barrier_multiclass"
                and not candidate.get("lockbox_opened", False)
                and int(labeling.get("horizon_bars", -1)) == horizon
                and float(labeling.get("atr_barrier_multiplier", -1)) == args.atr_multiplier
                and candidate.get("metadata", {}).get("model_profile") == args.model_profile
            )
            if same_source and same_protocol:
                summary = candidate
                print(f"Resuming compatible checkpoint {cached_path}")
        if summary is None:
            _, summary = run_triple_barrier_training(
                settings,
                provider=SnapshotProvider(snapshot, source.metadata),
                artifact_dir=artifact_dir,
                open_lockbox=False,
            )
        developments.append(
            {
                "name": name,
                "horizon": horizon,
                "validation": summary["validation"],
                "artifact": str(PROJECT_ROOT / "artifacts" / "experiments" / name),
            }
        )
        print(json.dumps(developments[-1], default=str))

    selected = max(developments, key=selection_key)
    selected_horizon = int(selected["horizon"])
    final_summary = None
    final_artifact = None
    lockbox_decision = "kept_sealed_because_development_gate_failed"
    if selected["validation"]["development_qualified"]:
        final_settings = replace(
            SETTINGS,
            data_provider="binance_paxg_research",
            bar_interval=args.interval,
            forecast_horizon_bars=selected_horizon,
            atr_barrier_multiplier=args.atr_multiplier,
            binance_start_utc=args.start,
            binance_end_utc=args.end,
            model_profile=args.model_profile,
        )
        final_name = (
            f"paxg_triple_{args.interval}_h{selected_horizon}_{config_tag}_final_lockbox"
        )
        final_artifact_path = PROJECT_ROOT / "artifacts" / "experiments" / final_name
        final_artifact = str(final_artifact_path)
        registry_path = PROJECT_ROOT / "artifacts" / "lockbox_registry.json"
        manifest = {
            "protocol": "paxg_triple_v1",
            "snapshot_sha256": snapshot_sha256,
            "instrument": source.symbol,
            "interval": args.interval,
            "horizon": selected_horizon,
            "atr_multiplier": args.atr_multiplier,
            "round_trip_cost_bps": final_settings.round_trip_cost_bps,
            "minimum_edge_bps": final_settings.minimum_edge_bps,
            "lockbox_fraction": final_settings.lockbox_fraction,
            "model_profile": args.model_profile,
        }
        experiment_key = lockbox_experiment_key(manifest)
        claimed, existing_claim = claim_lockbox_once(registry_path, experiment_key, manifest)
        if claimed:
            try:
                _, final_summary = run_triple_barrier_training(
                    final_settings,
                    provider=SnapshotProvider(snapshot, source.metadata),
                    artifact_dir=final_artifact_path,
                    open_lockbox=True,
                )
                summary_path = final_artifact_path / "summary.json"
                finalize_lockbox_claim(
                    registry_path,
                    experiment_key,
                    status="opened_and_finalized",
                    artifact_summary=str(summary_path),
                )
                lockbox_decision = "opened_once_after_all_development_gates_passed"
            except Exception as exc:
                finalize_lockbox_claim(
                    registry_path,
                    experiment_key,
                    status="opening_failed_lockbox_must_not_be_reopened",
                    error=str(exc),
                )
                raise
        else:
            existing_summary = existing_claim.get("artifact_summary")
            if existing_summary and Path(existing_summary).exists():
                final_summary = json.loads(Path(existing_summary).read_text(encoding="utf-8"))
                lockbox_decision = "reused_prior_final_artifact_without_reopening_lockbox"
            else:
                lockbox_decision = "refused_reopening_previously_claimed_lockbox"
    report = {
        "protocol": "development selection across requested horizons; one selected lockbox opened once",
        "source": source.metadata.__dict__,
        "snapshot_csv": str(snapshot_csv),
        "snapshot_metadata": str(snapshot_metadata),
        "snapshot_sha256": snapshot_sha256,
        "rows_snapshot": len(snapshot),
        "model_profile": args.model_profile,
        "development_trials": developments,
        "selected_horizon": selected_horizon,
        "lockbox_decision": lockbox_decision,
        "final_artifact": final_artifact,
        "final_summary": final_summary,
    }
    path = (
        PROJECT_ROOT
        / "artifacts"
        / "reports"
        / f"paxg_triple_experiments_{args.interval}_{config_tag}.json"
    )
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
