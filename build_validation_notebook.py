from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "artifacts" / "summary.json").read_text(encoding="utf-8"))
validation = summary["validation"]
grid = json.loads((ROOT / "artifacts" / "reports" / "experiment_grid.json").read_text(encoding="utf-8"))
best = max(grid, key=lambda row: row["accuracy"])

notebook = nbf.v4.new_notebook()
notebook["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
notebook["cells"] = [
    nbf.v4.new_markdown_cell(
        f"""# Gold Intraday AI — Validation Notebook

## tl;dr

- Baseline 5-minute model: **{validation['directional_accuracy']:.2%}** OOS directional accuracy; Wilson 95% lower **{validation['accuracy_wilson_lower_95']:.2%}**.
- Best tested horizon configuration: **{best['accuracy']:.2%}** (`{best['name']}`), still far below the **{validation['target_accuracy']:.0%}** target.
- Net result after {validation['round_trip_cost_bps']:.1f} bps round-trip cost: **{validation['net_return_sum']:.4f}** additive return; profit factor **{validation['profit_factor']:.2f}**.
- Verdict: **NOT QUALIFIED**. Current source is research-only and cannot validate an Exness MT5 deployment.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## Context & Methods

The decision is whether the model has enough reproducible, out-of-sample evidence to be used as an intraday Exness/MT5 signal. Features use current/past bars only. Labels use a future horizon and are purged from train/test boundaries. Each expanding walk-forward fold selects its model and confidence threshold on an inner validation slice. Test trades are non-overlapping and include configured cost/slippage.

### Key Assumptions

- Yahoo chart data is bootstrap research evidence, not an exchange-authoritative or broker execution feed.
- Directional accuracy is computed only on executed high-confidence signals; trade count and coverage must therefore stay visible.
- Costs are represented as a fixed round-trip bps assumption; exact Exness spread and slippage remain unverified.
"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
summary = json.loads((ROOT / 'artifacts' / 'summary.json').read_text(encoding='utf-8'))
predictions = pd.read_csv(ROOT / 'artifacts' / 'reports' / 'walk_forward_predictions.csv', parse_dates=['timestamp'])
folds = pd.read_csv(ROOT / 'artifacts' / 'reports' / 'fold_metrics.csv')
experiments = pd.DataFrame(json.loads((ROOT / 'artifacts' / 'reports' / 'experiment_grid.json').read_text(encoding='utf-8')))
summary['status'], predictions.shape, folds.shape"""
    ),
    nbf.v4.new_markdown_cell("## Data"),
    nbf.v4.new_code_cell(
        """source = summary['metadata']['source']
quality = summary['metadata']['quality']
display(pd.DataFrame({
    'value': [source['provider'], source['instrument'], source['interval'], summary['metadata']['rows_clean'], summary['metadata']['rows_supervised'], quality['missing_ohlc_rows'], quality['large_gap_count'], quality['severity']]
}, index=['provider','instrument','interval','clean bars','supervised rows','empty OHLC rows removed','large gaps','quality severity']))"""
    ),
    nbf.v4.new_markdown_cell("## Results"),
    nbf.v4.new_code_cell(
        """recomputed = {
    'trades': len(predictions),
    'directional_accuracy': predictions['correct'].mean(),
    'net_return_sum': predictions['net_return'].sum(),
    'cost_per_trade': (predictions['gross_return'] - predictions['net_return']).median(),
}
display(pd.Series(recomputed, name='independent recomputation'))
assert recomputed['trades'] == summary['validation']['trades']
assert np.isclose(recomputed['directional_accuracy'], summary['validation']['directional_accuracy'])
assert np.isclose(recomputed['net_return_sum'], summary['validation']['net_return_sum'])"""
    ),
    nbf.v4.new_code_cell(
        """display(folds[['fold','model','trades','coverage','directional_accuracy','accuracy_wilson_lower_95','net_return_sum','profit_factor']])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(folds['fold'], folds['directional_accuracy'], color='#d4a72c')
axes[0].axhline(summary['validation']['target_accuracy'], color='#c44e52', linestyle='--', label='80% target')
axes[0].set(title='OOS accuracy by fold', xlabel='fold', ylabel='accuracy', ylim=(0, 1))
axes[0].legend()
axes[1].bar(folds['fold'], folds['net_return_sum'], color=np.where(folds['net_return_sum'] >= 0, '#2ca02c', '#d62728'))
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set(title='Net return after costs by fold', xlabel='fold', ylabel='additive return')
plt.tight_layout()
plt.show()"""
    ),
    nbf.v4.new_code_cell(
        """display(experiments.sort_values('accuracy', ascending=False))
checks = pd.Series(summary['validation']['acceptance_checks'], name='passed')
display(checks.to_frame())
assert not summary['validation']['qualified']"""
    ),
    nbf.v4.new_markdown_cell(
        f"""## Takeaways

1. The observed edge is weak: the baseline achieved **{validation['directional_accuracy']:.2%}**, and the best horizon experiment achieved **{best['accuracy']:.2%}**.
2. Costs remove the apparent edge. The baseline profit factor is **{validation['profit_factor']:.2f}** and net return is negative.
3. The 80% target is not supported by this evidence. Raising selective confidence without minimum coverage/trade-count constraints would be misleading.
4. The next valid experiment requires exact Exness MT5 history, followed by a new untouched out-of-time window and paper trading. Until then, the app must remain clearly labeled advisory/research-only.
"""
    ),
]

output = ROOT / "notebooks" / "gold_intraday_validation.ipynb"
nbf.write(notebook, output)
client = NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbf.write(notebook, output)
print(f"Executed and saved {output}")

