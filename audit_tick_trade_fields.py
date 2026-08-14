from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "artifacts/control/exact_tick_trade_field_availability_v1_protocol.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/data_audits/exact_tick_trade_field_availability_v1/summary.json",
    )
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_path = (ROOT / protocol["source_manifest"]).resolve()
    if sha256(manifest_path) != protocol["source_manifest_sha256"]:
        raise RuntimeError("source manifest hash changed after protocol freeze")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    start = protocol["start_day_inclusive"]
    end = protocol["end_day_exclusive"]
    partitions = [
        item
        for item in manifest["partitions"]
        if start <= item["utc_day"] < end and int(item["rows"]) > 0
    ]
    files = [Path(item["csv"]).resolve() for item in partitions]
    if not files:
        raise RuntimeError("no nonempty partitions in frozen scope")

    source = pl.scan_csv([str(path) for path in files], infer_schema_length=1000)
    row = source.select(
        pl.len().alias("rows"),
        (pl.col("last") != 0).sum().alias("last_nonzero_rows"),
        (pl.col("volume") != 0).sum().alias("volume_nonzero_rows"),
        (pl.col("volume_real") != 0).sum().alias("volume_real_nonzero_rows"),
        pl.col("flags").n_unique().alias("unique_flag_values"),
        *[
            ((pl.col("flags").cast(pl.Int64) & bit) != 0).sum().alias(f"flag_bit_{bit}_rows")
            for bit in protocol["flag_bits_reported_without_semantic_relabeling"]
        ],
    ).collect().to_dicts()[0]
    flag_counts = source.group_by("flags").agg(pl.len().alias("rows")).collect().to_dicts()
    flag_counts.sort(key=lambda item: (-int(item["rows"]), int(item["flags"])))

    gates = protocol["eligibility_gate"]
    any_volume = max(int(row["volume_nonzero_rows"]), int(row["volume_real_nonzero_rows"]))
    checks = {
        "minimum_rows": int(row["rows"]) >= int(gates["minimum_rows"]),
        "minimum_nonzero_last_rows": int(row["last_nonzero_rows"]) >= int(gates["minimum_nonzero_last_rows"]),
        "minimum_nonzero_any_volume_rows": any_volume >= int(gates["minimum_nonzero_any_volume_rows"]),
        "minimum_buy_flag_rows": int(row["flag_bit_32_rows"]) >= int(gates["minimum_buy_flag_rows"]),
        "minimum_sell_flag_rows": int(row["flag_bit_64_rows"]) >= int(gates["minimum_sell_flag_rows"]),
    }
    eligible = all(checks.values())
    payload = {
        "schema_version": 1,
        "audit_id": protocol["protocol_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_TRADE_PRINT_FIELDS_ELIGIBLE" if eligible else "FAIL_NO_TRADE_PRINT_FIELDS_CLOSE_PATH",
        "decision": "ALLOW_ONE_BOUNDED_TRADE_PRINT_PREFLIGHT" if eligible else "CLOSE_TRADE_PRINT_AND_AGGRESSOR_PATH_NO_MODEL",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "scope": {
            "start_day_inclusive": start,
            "end_day_exclusive": end,
            "nonempty_partitions": len(files),
            "source_files": [str(path) for path in files],
            "source_file_sha256": [item["file_sha256"] for item in partitions],
        },
        "metrics": {**{key: int(value) for key, value in row.items()}, "flag_value_counts": flag_counts},
        "eligibility_checks": checks,
        "outcome_access": protocol["outcome_access"],
        "safety": protocol["safety"],
        "interpretation": "Flag bits are counted mechanically. They are not treated as aggressor labels unless nonzero last and volume support also pass.",
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "rows": row["rows"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
