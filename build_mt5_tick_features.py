from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT
from gold_ai.data.providers import sha256_file
from gold_ai.data.tick_features import (
    TICK_FEATURE_COLUMNS,
    TICK_FEATURE_SCHEMA_VERSION,
    build_tick_microstructure_features,
)


MINIMUM_NONEMPTY_DAYS_FOR_RESEARCH_PRE_GATE = 21
MINIMUM_NONEMPTY_DAYS_FOR_CHAMPION = 60


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen causal M5 tick feature snapshot.")
    parser.add_argument("--tick-root", type=Path, required=True)
    parser.add_argument("--symbol", default="XAUUSDm")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_paths = sorted(args.tick_root.resolve().rglob("*.metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No tick metadata under {args.tick_root}")
    partition_rows = []
    frames = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("symbol") != args.symbol or not metadata.get("complete_day"):
            raise ValueError(f"Unexpected tick partition metadata: {metadata_path}")
        csv_path = Path(metadata["csv"])
        if sha256_file(csv_path) != metadata["file_sha256"]:
            raise RuntimeError(f"Tick partition checksum mismatch: {csv_path}")
        partition_rows.append(metadata)
        if int(metadata["rows"]) > 0:
            frames.append(pd.read_csv(csv_path))
    ticks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp", "bid", "ask"])
    features = build_tick_microstructure_features(ticks)
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(features, index=True).values.tobytes()
    ).hexdigest()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    stem = f"mt5_tick_features_{args.symbol}_M5_{TICK_FEATURE_SCHEMA_VERSION}_{fingerprint[:16]}"
    csv_path = processed_dir / f"{stem}.csv"
    metadata_path = processed_dir / f"{stem}.metadata.json"
    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen tick features: {csv_path}")
    temporary_csv = csv_path.with_name(f"{csv_path.name}.{os.getpid()}.tmp")
    features.to_csv(temporary_csv, index_label="decision_time")
    os.replace(temporary_csv, csv_path)
    nonempty_days = sum(int(row["rows"]) > 0 for row in partition_rows)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_contract": "V15.1_LEAN",
        "family_id": "XAUUSD_M5_EXACT_TICK_MICROSTRUCTURE_V15_1",
        "schema_version": TICK_FEATURE_SCHEMA_VERSION,
        "feature_columns": list(TICK_FEATURE_COLUMNS),
        "symbol": args.symbol,
        "timeframe": "M5",
        "decision_semantics": "ticks in [decision-5m, decision); exact decision tick excluded",
        "rows": len(features),
        "first_decision_time": features.index.min().isoformat() if len(features) else None,
        "last_decision_time": features.index.max().isoformat() if len(features) else None,
        "source_calendar_days": len(partition_rows),
        "source_nonempty_days": nonempty_days,
        "minimum_nonempty_days_for_pre_gate": MINIMUM_NONEMPTY_DAYS_FOR_RESEARCH_PRE_GATE,
        "eligible_for_pre_gate": nonempty_days >= MINIMUM_NONEMPTY_DAYS_FOR_RESEARCH_PRE_GATE,
        "research_pre_gate_intended_use": "RESEARCH_ONLY",
        "minimum_nonempty_days_for_champion": MINIMUM_NONEMPTY_DAYS_FOR_CHAMPION,
        "eligible_for_champion": nonempty_days >= MINIMUM_NONEMPTY_DAYS_FOR_CHAMPION,
        "labels_or_holdout_accessed": False,
        "future_unseen_certification_holdout_required": True,
        "source_partitions": [
            {"utc_day": row["utc_day"], "rows": row["rows"], "sha256": row["file_sha256"]}
            for row in partition_rows
        ],
        "csv": str(csv_path),
        "file_sha256": sha256_file(csv_path),
    }
    atomic_write_text(metadata_path, json.dumps(metadata, indent=2))
    report_dir = PROJECT_ROOT / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = report_dir / f"mt5_tick_feature_snapshot_{stamp}.json"
    atomic_write_text(report, json.dumps(metadata, indent=2))
    print(json.dumps({"report": str(report), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
