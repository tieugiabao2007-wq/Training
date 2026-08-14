from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT


def main() -> None:
    experiments_root = PROJECT_ROOT / "artifacts" / "experiments"
    trials = []
    for summary_path in sorted(experiments_root.glob("paxg*_triple_*_development/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = summary.get("metadata", {})
        source = metadata.get("source", {})
        labeling = metadata.get("labeling", {})
        trials.append(
            {
                "name": summary_path.parent.name,
                "artifact_dir": str(summary_path.parent),
                "trained_at_utc": metadata.get("trained_at_utc"),
                "instrument": source.get("instrument"),
                "interval": source.get("interval"),
                "intended_use": source.get("intended_use"),
                "source_file_sha256": metadata.get("source_file_sha256"),
                "source_csv": metadata.get("source_csv"),
                "source_provider": source.get("provider"),
                "model_profile": metadata.get("model_profile"),
                "horizon_bars": labeling.get("horizon_bars"),
                "atr_barrier_multiplier": labeling.get("atr_barrier_multiplier"),
                "validation": summary.get("validation"),
            }
        )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "all PAXG development artifacts; lockboxes remain governed per artifact",
        "trial_count": len(trials),
        "trials": trials,
    }
    output = PROJECT_ROOT / "artifacts" / "reports" / "paxg_triple_experiments_all.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved {output} with {len(trials)} trials")


if __name__ == "__main__":
    main()
