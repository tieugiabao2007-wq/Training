from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gold_ai.manual_training import run_manual_training_once


if __name__ == "__main__":
    result = run_manual_training_once(PROJECT_ROOT)
    raise SystemExit(0 if result.get("status") != "FAILED_REVIEW_REQUIRED" else 1)
