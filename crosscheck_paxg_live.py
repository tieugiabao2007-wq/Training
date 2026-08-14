from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

from gold_ai.config import PROJECT_ROOT
from gold_ai.data.crossvenue import generate_paxg_crossvenue_report


def main() -> None:
    output = PROJECT_ROOT / "artifacts" / "reports" / "paxg_crossvenue_live.json"
    payload = generate_paxg_crossvenue_report(output)
    print(json.dumps(payload, indent=2, default=str))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
