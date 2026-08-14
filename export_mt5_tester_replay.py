from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.data.providers import sha256_file


HORIZON_MINUTES = 60
DECISION_LAG_MINUTES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frozen development predictions for MT5 tester-only replay."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--predictions", default="predictions_session.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-utc")
    parser.add_argument("--to-utc")
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def build_replay_signals(
    predictions: pd.DataFrame,
    *,
    from_utc: str | None = None,
    to_utc: str | None = None,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["direction"] = pd.to_numeric(frame["direction"], errors="raise").astype(int)
    frame["barrier_pct"] = pd.to_numeric(
        frame["barrier_pct"], errors="raise"
    ).astype(float)
    decision_time = frame["timestamp"] + pd.Timedelta(minutes=DECISION_LAG_MINUTES)
    if from_utc:
        frame = frame.loc[decision_time >= pd.Timestamp(from_utc).tz_convert("UTC")].copy()
        decision_time = decision_time.loc[frame.index]
    if to_utc:
        frame = frame.loc[decision_time < pd.Timestamp(to_utc).tz_convert("UTC")].copy()
        decision_time = decision_time.loc[frame.index]
    if not frame["direction"].isin([-1, 1]).all():
        raise ValueError("Replay accepts directional BUY/SELL predictions only")
    if not (frame["barrier_pct"] > 0).all():
        raise ValueError("Replay barriers must be positive")
    if decision_time.duplicated().any():
        raise ValueError("Replay decision timestamps must be unique")
    expiry_time = decision_time + pd.Timedelta(minutes=HORIZON_MINUTES)
    output = pd.DataFrame(
        {
            "decision_time_epoch": decision_time.map(lambda value: int(value.timestamp())),
            "direction": frame["direction"].to_numpy(),
            "barrier_pct": frame["barrier_pct"].to_numpy(),
            # Always use the frozen H12 expiry.  Never export label_end_time,
            # duration_bars or any other realized-future field to the tester EA.
            "expiry_time_epoch": expiry_time.map(lambda value: int(value.timestamp())),
            "signal_id": range(1, len(frame) + 1),
        }
    )
    return output.sort_values("decision_time_epoch").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    artifact = args.artifact.resolve()
    summary_path = artifact / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = summary["protocol"]
    if summary.get("research_only") is not True:
        raise ValueError("Only explicitly research-only artifacts may be replayed")
    if summary.get("champion_eligible") is not False:
        raise ValueError("Replay artifact must be Champion-ineligible")
    if int(protocol.get("sealed_holdout_access_count", -1)) != 0:
        raise ValueError("Sealed holdout access must remain zero")
    if protocol.get("saved_lockbox_predictions_accessed") is not False:
        raise ValueError("Saved lockbox predictions must not be accessed")

    prediction_path = artifact / args.predictions
    predictions = pd.read_csv(prediction_path)
    replay = build_replay_signals(
        predictions, from_utc=args.from_utc, to_utc=args.to_utc
    )
    if replay.empty:
        raise ValueError("No replay signals remain after the requested UTC bounds")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.{os.getpid()}.{time.time_ns()}.tmp")
    replay.to_csv(temporary, index=False)
    os.replace(temporary, output)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "MT5_STRATEGY_TESTER_EXECUTION_VALIDATION_ONLY",
        "training_evidence": False,
        "production_improvement_evidence": False,
        "optimization_allowed": False,
        "live_trading": False,
        "source_family": summary["family_id"],
        "source_status": summary["status"],
        "source_artifact": str(artifact),
        "source_summary_sha256": sha256_file(summary_path),
        "source_predictions": str(prediction_path),
        "source_predictions_sha256": sha256_file(prediction_path),
        "decision_semantics": "completed_M5_bar_open_plus_5_minutes",
        "expiry_semantics": "fixed_H12_60_minutes_no_realized_duration_fields",
        "rows": int(len(replay)),
        "first_decision_epoch": int(replay["decision_time_epoch"].iloc[0]),
        "last_decision_epoch": int(replay["decision_time_epoch"].iloc[-1]),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "sealed_holdout_access_count": 0,
    }
    manifest_path = output.with_suffix(".manifest.json")
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), "rows": len(replay)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
