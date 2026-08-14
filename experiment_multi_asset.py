from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gold_ai.config import PROJECT_ROOT, SETTINGS
from gold_ai.data.providers import MultiAssetYahooResearchProvider
from gold_ai.training import run_training


results = []
for horizon in (3, 6, 12):
    name = f"gc_multi_5m_h{horizon}"
    settings = replace(
        SETTINGS,
        data_provider="yahoo_multi_research",
        bar_interval="5m",
        bar_range="60d",
        forecast_horizon_bars=horizon,
    )
    provider = MultiAssetYahooResearchProvider(settings.yahoo_symbol, settings.bar_interval, settings.bar_range)
    output = PROJECT_ROOT / "artifacts" / "experiments" / name
    cached = output / "summary.json"
    if cached.exists():
        summary = json.loads(cached.read_text(encoding="utf-8"))
    else:
        _, summary = run_training(settings, provider=provider, artifact_dir=output)
    row = {"name": name, "horizon": horizon, **summary["validation"]}
    results.append(row)
    print(json.dumps(row))

path = PROJECT_ROOT / "artifacts" / "reports" / "multi_asset_experiments.json"
path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"Saved {path}")
