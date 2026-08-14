from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT
from gold_ai.selection_bias import record_experiment


def family_for(name: str, summary: dict) -> str:
    explicit = summary.get("experiment_family_id")
    if explicit:
        return str(explicit)
    normalized = name.lower()
    if normalized.startswith("mt5_xauusdm_m5"):
        return "XAUUSD_M5_EXACT_FEED_V15"
    if normalized.startswith("mt5_xauusdm_m15"):
        return "XAUUSD_M15_EXACT_FEED_V15"
    if "paxg" in normalized and "cot" in normalized:
        return "PAXG_COT_ABLATION"
    if "paxg" in normalized and "fred" in normalized:
        return "PAXG_FRED_ABLATION"
    if "paxg" in normalized:
        return "PAXG_TRIPLE_BARRIER_RESEARCH"
    return "LEGACY_RESEARCH_MIGRATED"


def main() -> int:
    experiments_root = PROJECT_ROOT / "artifacts" / "experiments"
    registry_path = PROJECT_ROOT / "artifacts" / "experiment_registry.json"
    report_path = PROJECT_ROOT / "artifacts" / "reports" / "experiment_registry_migration.json"
    recorded: list[str] = []
    existing: list[str] = []
    skipped: list[dict] = []
    for directory in sorted(path for path in experiments_root.iterdir() if path.is_dir()):
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            skipped.append({"experiment": directory.name, "reason": "no_summary"})
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append({"experiment": directory.name, "reason": str(exc)})
            continue
        validation = summary.get("validation", {})
        metadata = summary.get("metadata", {})
        labeling = metadata.get("labeling", {})
        model_config = {
            "model_name": summary.get("model_name"),
            "model_profile": metadata.get("model_profile"),
            "walk_forward_splits": metadata.get("walk_forward_splits"),
            "walk_forward_max_train_rows": metadata.get("walk_forward_max_train_rows"),
        }
        feature_set = {
            "version": metadata.get("feature_set_version"),
            "columns_sha256": metadata.get("feature_columns_sha256"),
            "count": metadata.get("feature_count"),
        }
        strategy_variant = {
            "training_mode": summary.get("training_mode"),
            "trade_timeframe": metadata.get("trade_timeframe"),
            "horizon_bars": labeling.get("horizon_bars"),
            "atr_multiplier": labeling.get("atr_barrier_multiplier"),
            "round_trip_cost_bps": labeling.get("round_trip_cost_bps"),
        }
        created, _ = record_experiment(
            registry_path,
            experiment_id=directory.name,
            family_id=family_for(directory.name, summary),
            model_config=model_config,
            feature_set=feature_set,
            threshold_config={
                "confidence_threshold": summary.get("confidence_threshold")
            },
            strategy_variant=strategy_variant,
            result={
                "qualified": validation.get("qualified", False),
                "directional_accuracy": validation.get("directional_accuracy"),
                "net_return_sum": validation.get("net_return_sum"),
                "profit_factor": validation.get("profit_factor"),
                "lockbox_status": validation.get("lockbox_status"),
            },
            artifact_path=str(directory),
        )
        (recorded if created else existing).append(directory.name)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    report = {
        "registry": str(registry_path),
        "recorded": recorded,
        "already_present": existing,
        "skipped": skipped,
        "totals": registry["totals"],
        "families": {
            key: value["count"] for key, value in registry["families"].items()
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
