from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT, SETTINGS
from gold_ai.data.providers import YahooResearchProvider
from gold_ai.training import run_training


EXPERIMENTS = [
    {"name": "gc_5m_h6", "interval": "5m", "range": "60d", "horizon": 6},
    {"name": "gc_5m_h12", "interval": "5m", "range": "60d", "horizon": 12},
    {"name": "gc_1h_h1", "interval": "1h", "range": "730d", "horizon": 1},
    {"name": "gc_1h_h3", "interval": "1h", "range": "730d", "horizon": 3},
]


results = []
for experiment in EXPERIMENTS:
    settings = replace(
        SETTINGS,
        bar_interval=experiment["interval"],
        bar_range=experiment["range"],
        forecast_horizon_bars=experiment["horizon"],
    )
    provider = YahooResearchProvider(settings.yahoo_symbol, settings.bar_interval, settings.bar_range)
    output = PROJECT_ROOT / "artifacts" / "experiments" / experiment["name"]
    cached_summary = output / "summary.json"
    if cached_summary.exists():
        summary = json.loads(cached_summary.read_text(encoding="utf-8"))
    else:
        _, summary = run_training(settings, provider=provider, artifact_dir=output)
    results.append(
        {
            **experiment,
            "accuracy": summary["validation"]["directional_accuracy"],
            "wilson_lower": summary["validation"]["accuracy_wilson_lower_95"],
            "trades": summary["validation"]["trades"],
            "coverage": summary["validation"]["coverage"],
            "profit_factor": summary["validation"]["profit_factor"],
            "net_return_sum": summary["validation"]["net_return_sum"],
            "qualified": summary["validation"]["qualified"],
        }
    )
    print(json.dumps(results[-1]))

grid_path = PROJECT_ROOT / "artifacts" / "reports" / "experiment_grid.json"
grid_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"Saved {grid_path}")
