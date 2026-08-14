from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.data.binance_aggtrades import audit_archive


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit official Binance PAXG aggTrades data.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("checksum", type=Path)
    parser.add_argument("--month", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_archive(args.archive.resolve(), args.checksum.resolve(), args.month)
    audit.update(
        {
            "schema_version": "1.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "provider": "Binance public data",
            "provider_owner": "Binance",
            "instrument": "PAXGUSDT",
            "dataset": "spot/monthly/aggTrades",
            "source_url": (
                f"https://data.binance.vision/data/spot/monthly/aggTrades/PAXGUSDT/"
                f"PAXGUSDT-aggTrades-{args.month}.zip"
            ),
            "checksum_url": (
                f"https://data.binance.vision/data/spot/monthly/aggTrades/PAXGUSDT/"
                f"PAXGUSDT-aggTrades-{args.month}.zip.CHECKSUM"
            ),
            "official_repository": "https://github.com/binance/binance-public-data",
            "license": "MIT license stated by the official binance-public-data repository",
            "license_url": "https://github.com/binance/binance-public-data/blob/master/LICENSE",
            "timestamp_unit": "microseconds",
            "timezone": "UTC",
            "source_mismatch": True,
            "exact_exness": False,
            "intended_use": "RESEARCH_ONLY representation pretraining and domain-shift audit",
            "certification_evidence": False,
            "holdout_access_count": 0,
            "live_trading_enabled": False,
            "auto_execution": False,
            "external_code_executed": False,
            "antivirus_scan": "NOT_RUN_USER_LOW_LAG_POLICY",
        }
    )
    atomic_write(args.output.resolve(), audit)
    print(
        json.dumps(
            {
                "summary": str(args.output.resolve()),
                "quality_gate": audit["quality_gate"],
                "rows": audit["metrics"]["rows"],
                "active_utc_days": audit["metrics"]["active_utc_days"],
                "failed_checks": audit["failed_checks"],
            },
            indent=2,
        )
    )
    return 0 if not audit["failed_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
