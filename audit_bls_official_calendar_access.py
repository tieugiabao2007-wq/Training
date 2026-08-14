from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol_raw = protocol_path.read_bytes()
    protocol = json.loads(protocol_raw)
    rows = []
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Gold-AI-research/1.0; public-data-audit)"
    }
    for source in protocol["official_sources"]:
        response = requests.get(source["url"], timeout=30, headers=headers)
        rows.append(
            {
                **source,
                "status_code": response.status_code,
                "response_bytes": len(response.content),
                "response_sha256": sha256_bytes(response.content),
                "content_type": response.headers.get("content-type"),
                "final_url": response.url,
                "eligible_calendar_payload": response.status_code == 200
                and b"Schedule of Releases" in response.content,
            }
        )
    all_eligible = all(row["eligible_calendar_payload"] for row in rows)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_bytes(protocol_raw),
        "source_checks": rows,
        "calendar_rows_parsed": 0,
        "labels_returns_future_quotes_or_mt5_accessed": False,
        "model_fits": 0,
        "status": (
            "PASS_OFFICIAL_CALENDAR_ACCESS"
            if all_eligible
            else "FAIL_OFFICIAL_BLS_CHANNELS_REFUSED_AUTOMATED_ACCESS_CLOSE_PATH"
        ),
        "decision": (
            "CALENDAR_PARSE_PREFLIGHT_ELIGIBLE"
            if all_eligible
            else "CLOSE_WITHOUT_MIRROR_SCRAPE_MODEL_OR_REFINEMENT"
        ),
        "safety": {
            "live_trading_enabled": False,
            "auto_execution": False,
            "holdout_access_count": 0,
            "order_functions_called": False,
        },
    }
    atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2))
    return 0 if all_eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
