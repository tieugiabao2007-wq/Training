from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROMPT = Path(
    r"C:\Users\Admin\Documents\Codex\2026-08-09\xem-l-i-nh-ng-vi\outputs\GOLD_AI_CODEX_MASTER_PROMPT_V15_M5_M15_MAX_3_ORDERS.txt"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def classify(heading: str) -> str:
    normalized = heading.upper()
    rules = (
        (("PURG", "EMBARGO", "NESTED", "HOLDOUT", "CERTIF", "OVERFITT", "DSR", "PBO"), "V15_VALIDATION"),
        (("NEWS", "MARKET INTELLIGENCE", "FED", "GEOPOL", "TRADINGVIEW", "HOURLY", "DAILY MARKET"), "V13_V14_INTELLIGENCE"),
        (("CHAMPION", "CHALLENGER", "HOT-SWAP", "ROLLBACK", "MODEL VERSION"), "V11_CHAMPION"),
        (("LOT", "MARGIN", "LEVERAGE", "POSITION SIZ", "MARTINGALE"), "V9_V12_RISK_LOT"),
        (("AUTO TRADE", "KILL SWITCH", "AI ENGINE", "EXISTING OPEN", "CONTROL MATRIX"), "V10_RUNTIME_CONTROL"),
        (("M5/M15", "TIMEFRAME", "3 ORDERS", "COMBINED", "TRADE_TIMEFRAME"), "DUAL_TIMEFRAME_EXECUTION"),
        (("DATA", "SOURCE", "MT5", "EXNESS", "CLEAN", "QUALITY"), "DATA_AND_PROVENANCE"),
        (("FEATURE", "INDICATOR", "CONTEXT"), "FEATURES"),
        (("TARGET", "LABEL", "ACCURACY"), "TARGET_AND_ACCURACY"),
        (("MODEL", "ENSEMBLE", "HYPERPARAMETER", "OPTUNA"), "MODELING"),
        (("BACKTEST", "PROFIT", "COST", "SLIPPAGE"), "BACKTEST_ECONOMICS"),
        (("AUTO TRAIN", "RETRAIN", "SCHEDUL"), "AUTO_TRAIN"),
        (("DASHBOARD", "USER INTERFACE", "ONE-CLICK", "WINDOWS"), "UI_AND_OPERATIONS"),
        (("TEST", "REPRODUC", "LOG", "PROJECT_STATE", "AGENTS.MD"), "EVIDENCE_AND_STATE"),
    )
    for needles, family in rules:
        if any(needle in normalized for needle in needles):
            return family
    return "OTHER"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/reports/master_prompt_requirement_inventory.json",
    )
    args = parser.parse_args()
    payload = args.prompt.read_bytes()
    lines = payload.decode("utf-8-sig").splitlines()
    headings: list[dict] = []
    active_override = "BASE"
    heading_pattern = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
    numbered_pattern = re.compile(r"^\s*(\d+)[.)]\s+(.+?)\s*$")
    for line_number, line in enumerate(lines, start=1):
        match = heading_pattern.match(line)
        if match:
            text = match.group(2).strip()
            version = re.search(r"\bV(\d+)\b", text, flags=re.IGNORECASE)
            if version and "OVERRIDE" in text.upper():
                active_override = f"V{version.group(1)}"
            headings.append(
                {
                    "line": line_number,
                    "level": len(match.group(1)),
                    "text": text,
                    "active_override": active_override,
                    "requirement_family": classify(text),
                    "completion_status": "REQUIRES_EVIDENCE_MAPPING",
                }
            )
            continue
        numbered = numbered_pattern.match(line)
        if numbered and len(numbered.group(2)) >= 8:
            text = numbered.group(2).strip()
            headings.append(
                {
                    "line": line_number,
                    "level": "numbered_requirement",
                    "number": int(numbered.group(1)),
                    "text": text,
                    "active_override": active_override,
                    "requirement_family": classify(text),
                    "completion_status": "REQUIRES_EVIDENCE_MAPPING",
                }
            )
    counts: dict[str, int] = {}
    for item in headings:
        counts[item["requirement_family"]] = counts.get(item["requirement_family"], 0) + 1
    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "master_prompt_path": str(args.prompt),
        "master_prompt_sha256": hashlib.sha256(payload).hexdigest(),
        "line_count": len(lines),
        "inventory_item_count": len(headings),
        "family_counts": dict(sorted(counts.items())),
        "completion_policy": (
            "Inventory is not completion evidence. Each item remains REQUIRES_EVIDENCE_MAPPING "
            "until a requirement-specific artifact/test/runtime/OOS result proves it."
        ),
        "items": headings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: output[key] for key in output if key != "items"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
