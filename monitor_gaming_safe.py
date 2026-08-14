from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT
from gold_ai.resource_governor import ResourceGovernor, ResourceGovernorConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reattach gaming-safe CPU controls to an existing training process."
    )
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--cpu-ceiling", type=float, default=88.0)
    parser.add_argument("--max-cores", type=int, default=12)
    parser.add_argument("--min-cores", type=int, default=2)
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument(
        "--full-power",
        action="store_true",
        help="Use all configured cores at Normal priority with no CPU throttling.",
    )
    parser.add_argument("--status-json", required=True, type=Path)
    args = parser.parse_args()
    config = ResourceGovernorConfig(
        cpu_ceiling_pct=100.0 if args.full_power else args.cpu_ceiling,
        hard_pause_pct=100.0 if args.full_power else 88.0,
        max_cores=args.max_cores,
        min_cores=args.min_cores,
        sample_seconds=args.sample_seconds,
        unrestricted_full_power=args.full_power,
    )
    return ResourceGovernor(config, args.status_json).monitor_existing(args.pid)


if __name__ == "__main__":
    raise SystemExit(main())
