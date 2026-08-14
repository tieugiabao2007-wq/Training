from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "gold_regime_drift_diagnostic.ipynb"


def _code(source: str):
    return nbf.v4.new_code_cell(source.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
notebook["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Gold Intraday AI — Regime Drift Diagnostic

## tl;dr

Execution pending. This section is replaced with observed metrics after the notebook runs successfully.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## Context & Methods

This diagnostic explains why the strongest development model lost its edge in the final 2025–2026 walk-forward folds. It reconstructs the exact immutable PAXG snapshot and the feature columns stored in the best v2 artifact, then compares folds 1–3 with folds 4–5.

### Key Assumptions

- PAXGUSDT is a research proxy, not an XAUUSD/Exness execution feed.
- The 15% trailing lockbox remains unopened; this notebook uses development rows only.
- Feature drift is measured with PSI, KS distance, robust median shift, and early/late feature-label Spearman changes. These are diagnostics, not evidence of tradable causality.
- The Binance checksum and stored feature-column hash must match before findings are accepted.
"""
    ),
    nbf.v4.new_markdown_cell("## Data"),
    _code(
        r"""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, spearmanr

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from gold_ai.backtest import _fold_boundaries
from gold_ai.data.providers import load_saved_market_data
from gold_ai.data.quality import validate_and_clean_bars
from gold_ai.features import build_feature_frame
from gold_ai.labels import triple_barrier_events

SNAPSHOT = ROOT / "data" / "raw" / "binance_PAXGUSDT_5m_51cf1d1b1d1ac096.csv"
BEST_ARTIFACT = ROOT / "artifacts" / "experiments" / "paxg_traps_v2_triple_5m_h12_max_accelerated_atr1_development"
SUMMARY = json.loads((BEST_ARTIFACT / "summary.json").read_text(encoding="utf-8"))
BUNDLE = joblib.load(BEST_ARTIFACT / "model.joblib")

bars, source_metadata = load_saved_market_data(SNAPSHOT)
clean, quality = validate_and_clean_bars(bars, expected_interval_seconds=300)
enriched = build_feature_frame(clean)
events = triple_barrier_events(
    clean,
    horizon_bars=12,
    atr_multiplier=1.0,
    round_trip_cost_bps=3.0,
    minimum_edge_bps=2.0,
)
feature_columns = list(BUNDLE.feature_columns)
valid = enriched[feature_columns].notna().all(axis=1) & events["label"].notna()
x = enriched.loc[valid, feature_columns].copy()
y = events.loc[valid, "label"].astype("int8")
outcomes = events.loc[valid].copy()

feature_hash = hashlib.sha256("\n".join(feature_columns).encode("utf-8")).hexdigest()
assert feature_hash == SUMMARY["metadata"]["feature_columns_sha256"]
assert len(x) == SUMMARY["metadata"]["rows_supervised"]
assert quality.duplicate_timestamps == 0
assert quality.missing_ohlc_rows == 0

source_profile = pd.Series({
    "snapshot": str(SNAPSHOT),
    "provider": source_metadata.provider,
    "instrument": source_metadata.instrument,
    "bars": len(clean),
    "supervised_rows": len(x),
    "features_reproduced": len(feature_columns),
    "first_bar": clean.index.min().isoformat(),
    "last_bar": clean.index.max().isoformat(),
    "duplicate_timestamps": quality.duplicate_timestamps,
    "missing_ohlc_rows": quality.missing_ohlc_rows,
    "large_gaps": quality.large_gap_count,
    "source_sha256": SUMMARY["metadata"]["source_file_sha256"],
    "feature_columns_sha256": feature_hash,
}, name="value")
display(source_profile.to_frame())
"""
    ),
    _code(
        r"""
# Reconstruct the same development/lockbox split and five walk-forward test ranges.
n_rows = len(x)
lockbox_rows = max(int(n_rows * 0.15), min(2000, max(300, n_rows // 5)))
lockbox_start = n_rows - lockbox_rows
development_end = lockbox_start - 12
development_index = x.index[:development_end]
fold_ranges = _fold_boundaries(development_end, n_splits=5)
fold_id = pd.Series(0, index=development_index, dtype="int8", name="fold")
for number, (start, end) in enumerate(fold_ranges, start=1):
    fold_id.iloc[start:end] = number
assert sorted(fold_id[fold_id > 0].unique().tolist()) == [1, 2, 3, 4, 5]
assert x.index[lockbox_start] > fold_id.index.max()

fold_bounds = pd.DataFrame([
    {
        "fold": number,
        "test_start": development_index[start],
        "test_end": development_index[end - 1],
        "rows": end - start,
    }
    for number, (start, end) in enumerate(fold_ranges, start=1)
])
display(fold_bounds)
print(f"Reserved lockbox rows: {lockbox_rows:,}; first lockbox row: {x.index[lockbox_start]}")
"""
    ),
    nbf.v4.new_markdown_cell("## Results"),
    _code(
        r"""
development_mask = fold_id > 0
development_x = x.loc[fold_id.index[development_mask]]
development_y = y.loc[development_x.index]
development_outcomes = outcomes.loc[development_x.index]
fold_values = fold_id.loc[development_x.index]

fold_metrics = pd.read_csv(BEST_ARTIFACT / "reports" / "fold_metrics.csv")
rows = []
for fold in range(1, 6):
    mask = fold_values.eq(fold)
    labels = development_y.loc[mask]
    fold_events = development_outcomes.loc[mask]
    fold_features = development_x.loc[mask]
    reported = fold_metrics.loc[fold_metrics["fold"].eq(fold)].iloc[0]
    rows.append({
        "fold": fold,
        "start": fold_features.index.min(),
        "end": fold_features.index.max(),
        "rows": len(fold_features),
        "down_share": labels.eq(-1).mean(),
        "no_trade_share": labels.eq(0).mean(),
        "up_share": labels.eq(1).mean(),
        "median_barrier_bps": fold_events["barrier_pct"].median() * 10_000,
        "median_duration_bars": fold_events["duration_bars"].median(),
        "median_volume": fold_features["volume_log"].pipe(np.expm1).median(),
        "median_trade_count": fold_features["trade_count_log"].pipe(np.expm1).median(),
        "median_abs_return_12_bps": fold_features["return_12"].abs().median() * 10_000,
        "selected_model": reported["model"],
        "executed_trades": int(reported["trades"]),
        "coverage": reported["coverage"],
        "oos_accuracy": reported["directional_accuracy"],
        "oos_net_return": reported["net_return_sum"],
    })
regime_table = pd.DataFrame(rows)
display(regime_table.round(4))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
regime_table.set_index("fold")[["down_share", "no_trade_share", "up_share"]].plot(
    kind="bar", stacked=True, ax=axes[0], color=["#d9534f", "#999999", "#2ca02c"]
)
axes[0].set(title="Triple-barrier label mix by OOS fold", ylabel="share", ylim=(0, 1))
axes[0].legend(loc="lower right")
axes[1].bar(regime_table["fold"], regime_table["oos_accuracy"], color="#d4a72c")
axes[1].axhline(0.75, color="#c44e52", linestyle="--", label="75% stability gate")
axes[1].axhline(0.80, color="#7f0000", linestyle=":", label="80% target")
axes[1].set(title="Executed-signal OOS accuracy", xlabel="fold", ylabel="accuracy", ylim=(0, 1))
axes[1].legend()
plt.tight_layout()
plt.show()
"""
    ),
    _code(
        r"""
def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = reference.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    cur = current.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(ref) < 100 or len(cur) < 100:
        return np.nan
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=quantiles)[0].astype(float)
    cur_counts = np.histogram(cur, bins=quantiles)[0].astype(float)
    ref_share = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_share = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))

rng = np.random.default_rng(42)
early_index = fold_values.index[fold_values.le(3)]
late_index = fold_values.index[fold_values.ge(4)]
early_sample_index = rng.choice(early_index, size=min(30_000, len(early_index)), replace=False)
late_sample_index = rng.choice(late_index, size=min(30_000, len(late_index)), replace=False)
early_x = development_x.loc[early_sample_index]
late_x = development_x.loc[late_sample_index]
early_y = development_y.loc[early_sample_index]
late_y = development_y.loc[late_sample_index]

drift_rows = []
for feature in feature_columns:
    early = early_x[feature].replace([np.inf, -np.inf], np.nan).dropna()
    late = late_x[feature].replace([np.inf, -np.inf], np.nan).dropna()
    if len(early) < 100 or len(late) < 100:
        continue
    iqr = early.quantile(0.75) - early.quantile(0.25)
    robust_shift = (late.median() - early.median()) / iqr if iqr > 0 else 0.0
    ks = ks_2samp(early, late, method="asymp").statistic
    early_corr = spearmanr(early_x.loc[early.index, feature], early_y.loc[early.index]).statistic
    late_corr = spearmanr(late_x.loc[late.index, feature], late_y.loc[late.index]).statistic
    drift_rows.append({
        "feature": feature,
        "psi": population_stability_index(early, late),
        "ks_distance": float(ks),
        "robust_median_shift_iqr": float(robust_shift),
        "early_label_spearman": float(early_corr) if np.isfinite(early_corr) else 0.0,
        "late_label_spearman": float(late_corr) if np.isfinite(late_corr) else 0.0,
        "correlation_change": float(late_corr - early_corr) if np.isfinite(early_corr) and np.isfinite(late_corr) else 0.0,
        "correlation_sign_flip": bool(
            np.isfinite(early_corr) and np.isfinite(late_corr)
            and np.sign(early_corr) != np.sign(late_corr)
            and abs(early_corr - late_corr) >= 0.03
        ),
    })
drift_table = pd.DataFrame(drift_rows).sort_values(["psi", "ks_distance"], ascending=False)
display(drift_table.head(20).round(4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
top = drift_table.head(15).sort_values("psi")
axes[0].barh(top["feature"], top["psi"], color="#4c78a8")
axes[0].axvline(0.25, color="#c44e52", linestyle="--", label="high PSI")
axes[0].set(title="Top feature population drift (folds 1–3 vs 4–5)", xlabel="PSI")
axes[0].legend()
axes[1].scatter(
    drift_table["early_label_spearman"],
    drift_table["late_label_spearman"],
    alpha=0.65,
    color="#d4a72c",
)
limit = max(0.05, np.nanmax(np.abs(drift_table[["early_label_spearman", "late_label_spearman"]].to_numpy())))
axes[1].plot([-limit, limit], [-limit, limit], color="#777777", linestyle="--")
axes[1].axhline(0, color="#333333", linewidth=0.8)
axes[1].axvline(0, color="#333333", linewidth=0.8)
axes[1].set(
    title="Feature-label relationship drift",
    xlabel="Spearman, folds 1–3",
    ylabel="Spearman, folds 4–5",
    xlim=(-limit, limit),
    ylim=(-limit, limit),
)
plt.tight_layout()
plt.show()
"""
    ),
    _code(
        r"""
trap_features = [
    "sweep_high_48", "sweep_low_48",
    "sweep_high_96", "sweep_low_96",
]
conditional_rows = []
for period, period_index in [("early_folds_1_3", early_index), ("late_folds_4_5", late_index)]:
    for feature in trap_features:
        active = development_x.loc[period_index, feature].eq(1)
        labels = development_y.loc[period_index].loc[active]
        event_returns = development_outcomes.loc[period_index, "event_return"].loc[active]
        conditional_rows.append({
            "period": period,
            "condition": feature,
            "rows": len(labels),
            "down_share": labels.eq(-1).mean() if len(labels) else np.nan,
            "no_trade_share": labels.eq(0).mean() if len(labels) else np.nan,
            "up_share": labels.eq(1).mean() if len(labels) else np.nan,
            "median_event_return_bps": event_returns.median() * 10_000 if len(labels) else np.nan,
        })
conditional_traps = pd.DataFrame(conditional_rows)
display(conditional_traps.round(4))

monthly = clean.copy()
monthly["return"] = np.log(monthly["close"]).diff()
monthly["taker_buy_ratio"] = monthly["taker_buy_base_volume"] / monthly["volume"].replace(0, np.nan)
monthly_profile = monthly.resample("ME").agg(
    bars=("close", "size"),
    return_volatility=("return", "std"),
    median_volume=("volume", "median"),
    median_trade_count=("trade_count", "median"),
    median_taker_buy_ratio=("taker_buy_ratio", "median"),
)
monthly_profile["invalid_taker_ratio_share"] = monthly["taker_buy_ratio"].groupby(
    monthly.index.to_period("M")
).apply(lambda values: ((values < 0) | (values > 1)).mean()).to_numpy()
display(monthly_profile.tail(12).round(5))

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
monthly_profile["return_volatility"].plot(ax=axes[0], color="#c44e52", title="Monthly 5-minute return volatility")
monthly_profile[["median_volume", "median_trade_count"]].plot(ax=axes[1], secondary_y="median_trade_count")
axes[1].set_title("Monthly Binance PAXG microstructure levels")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell("## Takeaways"),
    _code(
        r"""
early_distribution = development_y.loc[early_index].value_counts(normalize=True).reindex([-1, 0, 1], fill_value=0).to_numpy()
late_distribution = development_y.loc[late_index].value_counts(normalize=True).reindex([-1, 0, 1], fill_value=0).to_numpy()
label_js_divergence = float(jensenshannon(early_distribution, late_distribution, base=2) ** 2)
high_psi_count = int(drift_table["psi"].ge(0.25).sum())
sign_flip_count = int(drift_table["correlation_sign_flip"].sum())
late_accuracy = float(
    np.average(
        fold_metrics.loc[fold_metrics["fold"].isin([4, 5]), "directional_accuracy"],
        weights=np.maximum(fold_metrics.loc[fold_metrics["fold"].isin([4, 5]), "trades"], 1),
    )
)
diagnostic = {
    "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "snapshot": str(SNAPSHOT),
    "snapshot_sha256": SUMMARY["metadata"]["source_file_sha256"],
    "feature_set_version": SUMMARY["metadata"]["feature_set_version"],
    "feature_count": len(feature_columns),
    "supervised_rows_reproduced": len(x),
    "development_rows": development_end,
    "lockbox_rows_unopened": lockbox_rows,
    "label_js_divergence_early_vs_late": label_js_divergence,
    "high_psi_feature_count": high_psi_count,
    "feature_label_sign_flip_count": sign_flip_count,
    "top_drift_feature": str(drift_table.iloc[0]["feature"]),
    "top_drift_feature_psi": float(drift_table.iloc[0]["psi"]),
    "late_fold_weighted_accuracy": late_accuracy,
    "fold4_accuracy": float(fold_metrics.loc[fold_metrics["fold"].eq(4), "directional_accuracy"].iloc[0]),
    "fold5_trades": int(fold_metrics.loc[fold_metrics["fold"].eq(5), "trades"].iloc[0]),
    "source_quality": quality.to_dict(),
    "assessment": (
        "material_covariate_and_concept_drift"
        if high_psi_count >= 5 and sign_flip_count >= 5
        else "limited_drift_or_model_selection_instability"
    ),
    "decision": "Do not open lockbox; test a causal drift gate and recency-weighted/rolling training only on development data.",
}

reports = ROOT / "artifacts" / "reports"
reports.mkdir(parents=True, exist_ok=True)
(reports / "regime_drift_diagnostic.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
drift_table.to_csv(reports / "regime_drift_features.csv", index=False)
regime_table.to_csv(reports / "regime_drift_folds.csv", index=False)
conditional_traps.to_csv(reports / "regime_drift_traps.csv", index=False)

display(pd.Series(diagnostic, name="observed_value").to_frame())
print("DRIFT_RESULT_JSON=" + json.dumps(diagnostic))
"""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
client = NotebookClient(
    notebook,
    timeout=900,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
)
client.execute()

result = None
for output in notebook["cells"][-1].get("outputs", []):
    text = output.get("text", "") if output.get("output_type") == "stream" else ""
    for line in text.splitlines():
        if line.startswith("DRIFT_RESULT_JSON="):
            result = json.loads(line.removeprefix("DRIFT_RESULT_JSON="))
if result is None:
    raise RuntimeError("Executed notebook did not emit DRIFT_RESULT_JSON")

notebook["cells"][0]["source"] = f"""# Gold Intraday AI — Regime Drift Diagnostic

## tl;dr

- Reproduced **{result['supervised_rows_reproduced']:,}** supervised rows and all **{result['feature_count']}** v2 feature columns from the checksum-verified immutable snapshot.
- Early-versus-late comparison found **{result['high_psi_feature_count']}** features with PSI ≥0.25 and **{result['feature_label_sign_flip_count']}** material feature-label sign flips.
- Highest population drift: **`{result['top_drift_feature']}`**, PSI **{result['top_drift_feature_psi']:.3f}**. Label-mix Jensen–Shannon divergence: **{result['label_js_divergence_early_vs_late']:.4f}**.
- Fold 4 accuracy was **{result['fold4_accuracy']:.2%}** and fold 5 executed **{result['fold5_trades']}** trades. Assessment: **{result['assessment'].replace('_', ' ')}**.
- Decision: keep the lockbox sealed. Test a causal drift gate and recency-aware training only on development data; do not promote the PAXG proxy to Exness.
"""

nbf.write(notebook, OUTPUT)
print(f"Executed and saved {OUTPUT}")
print(json.dumps(result, indent=2))
