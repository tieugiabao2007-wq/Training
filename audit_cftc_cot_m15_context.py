from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


GOLD_CODE = "088691"
RAW_COLUMNS = {
    "Open_Interest_All": "open_interest",
    "Prod_Merc_Positions_Long_All": "producer_long",
    "Prod_Merc_Positions_Short_All": "producer_short",
    "Swap_Positions_Long_All": "swap_long",
    "Swap__Positions_Short_All": "swap_short",
    "M_Money_Positions_Long_All": "managed_long",
    "M_Money_Positions_Short_All": "managed_short",
    "Other_Rept_Positions_Long_All": "other_long",
    "Other_Rept_Positions_Short_All": "other_short",
}
FEATURES = [
    "managed_money_net_share_oi", "producer_net_share_oi", "swap_net_share_oi",
    "other_reportable_net_share_oi", "open_interest_log_change_1w",
    "managed_money_net_share_change_1w", "managed_money_net_share_z13",
    "producer_net_share_z13", "positioning_disagreement",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def psi(reference: np.ndarray, current: np.ndarray) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0 if np.isclose(reference.mean(), current.mean()) else 99.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.clip(np.histogram(reference, bins=edges)[0] / len(reference), 1e-6, None)
    b = np.clip(np.histogram(current, bins=edges)[0] / len(current), 1e-6, None)
    return float(np.sum((b - a) * np.log(b / a)))


def read_zip(path: Path, year: int) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        names = archive.namelist()
        if bad or len(names) != 1:
            raise ValueError(f"invalid official archive {path}: bad={bad}, names={names}")
        with archive.open(names[0]) as stream:
            frame = pd.read_csv(stream, dtype=str)
    frame.columns = [str(column).strip() for column in frame.columns]
    code = frame["CFTC_Contract_Market_Code"].astype(str).str.strip()
    gold = frame.loc[code == GOLD_CODE].copy()
    gold["source_year"] = year
    return gold, {"archive_member": names[0], "archive_integrity": True, "total_rows": len(frame), "gold_rows": len(gold)}


def make_features(gold: pd.DataFrame) -> pd.DataFrame:
    keep = ["Market_and_Exchange_Names", "Report_Date_as_YYYY-MM-DD", "CFTC_Contract_Market_Code", *RAW_COLUMNS]
    result = gold.loc[:, keep].copy()
    result["report_date"] = pd.to_datetime(result.pop("Report_Date_as_YYYY-MM-DD"), utc=True)
    result["availability_utc"] = result["report_date"] + pd.Timedelta(days=7, hours=21)
    # Annual CFTC files are newest-first.  Every delta/rolling statistic must
    # be calculated in causal ascending report order, never file row order.
    result = result.sort_values("report_date").reset_index(drop=True)
    for raw, clean in RAW_COLUMNS.items():
        result[clean] = pd.to_numeric(result.pop(raw).str.strip(), errors="coerce")
    oi = result["open_interest"]
    result["managed_money_net_share_oi"] = (result["managed_long"] - result["managed_short"]) / oi
    result["producer_net_share_oi"] = (result["producer_long"] - result["producer_short"]) / oi
    result["swap_net_share_oi"] = (result["swap_long"] - result["swap_short"]) / oi
    result["other_reportable_net_share_oi"] = (result["other_long"] - result["other_short"]) / oi
    result["open_interest_log_change_1w"] = np.log(oi).diff()
    result["managed_money_net_share_change_1w"] = result["managed_money_net_share_oi"].diff()
    for field, output in (("managed_money_net_share_oi", "managed_money_net_share_z13"), ("producer_net_share_oi", "producer_net_share_z13")):
        mean = result[field].rolling(13, min_periods=13).mean()
        std = result[field].rolling(13, min_periods=13).std(ddof=0)
        result[output] = (result[field] - mean) / std.replace(0, np.nan)
    result["positioning_disagreement"] = result["managed_money_net_share_oi"] - result["producer_net_share_oi"]
    return result.sort_values("availability_utc").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--m15", type=Path, required=True)
    parser.add_argument("--m15-read-nrows", type=int, default=85010)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source_rows, archives, all_gold = [], [], []
    url_by_name = {Path(url).name: url for url in protocol["official_source"]["urls"]}
    for year in (2024, 2025, 2026):
        path = args.staging / f"fut_disagg_txt_{year}.zip"
        frame, profile = read_zip(path, year)
        all_gold.append(frame)
        archives.append({"year": year, "path": str(path.resolve()), "url": url_by_name[path.name], "size_bytes": path.stat().st_size, "sha256": digest(path), **profile})
    gold = pd.concat(all_gold, ignore_index=True)
    names = sorted(gold["Market_and_Exchange_Names"].astype(str).str.strip().unique())
    codes = sorted(gold["CFTC_Contract_Market_Code"].astype(str).str.strip().unique())
    features = make_features(gold)
    development_reports = features.loc[features.report_date < pd.Timestamp("2026-01-01", tz="UTC")].copy()
    m15 = pd.read_csv(args.m15, usecols=["timestamp"], nrows=args.m15_read_nrows, parse_dates=["timestamp"])
    m15["timestamp"] = pd.to_datetime(m15["timestamp"], utc=True)
    start = pd.Timestamp(protocol["data"]["development_start_utc"])
    end = pd.Timestamp(protocol["data"]["development_end_exclusive_utc"])
    decisions = m15.loc[(m15.timestamp >= start) & (m15.timestamp < end), ["timestamp"]].sort_values("timestamp")
    usable = development_reports.dropna(subset=FEATURES).sort_values("availability_utc")
    joined = pd.merge_asof(decisions, usable[["availability_utc", "report_date", *FEATURES]], left_on="timestamp", right_on="availability_utc", direction="backward", allow_exact_matches=False)
    finite = np.isfinite(joined[FEATURES].to_numpy(float)).all(axis=1)
    coverage = float(finite.mean()) if len(joined) else 0.0
    early = usable.loc[usable.report_date < pd.Timestamp("2025-07-01", tz="UTC")]
    late = usable.loc[(usable.report_date >= pd.Timestamp("2025-07-01", tz="UTC")) & (usable.report_date < pd.Timestamp("2026-01-01", tz="UTC"))]
    domain = {}
    for feature in FEATURES:
        ref, cur = early[feature].to_numpy(float), late[feature].to_numpy(float)
        lo, hi = np.quantile(ref, [.01, .99])
        domain[feature] = {"psi": psi(ref, cur), "late_inside_early_p01_p99": float(np.mean((cur >= lo) & (cur <= hi)))}
    report_dates = development_reports.report_date.sort_values()
    gaps = report_dates.diff().dt.total_seconds().div(86400).dropna()
    duplicate_dates = int(development_reports.report_date.duplicated().sum())
    all_values = development_reports[[*RAW_COLUMNS.values()]].to_numpy(float)
    profile = {
        "official_archives": archives, "gold_contract_names": names, "gold_contract_codes": codes,
        "gold_reports_all_downloads": len(features), "gold_reports_pre_2026": len(development_reports),
        "first_report_date": features.report_date.min().isoformat(), "last_report_date": features.report_date.max().isoformat(),
        "duplicate_report_dates_pre_2026": duplicate_dates, "maximum_report_gap_days_pre_2026": float(gaps.max()),
        "raw_numeric_null_count_pre_2026": int(np.isnan(all_values).sum()), "raw_numeric_nonpositive_open_interest": int((development_reports.open_interest <= 0).sum()),
        "m15_decisions": len(decisions), "m15_joined_finite_rows": int(finite.sum()), "m15_join_coverage": coverage,
        "maximum_availability_not_before_decision_violations": int((joined.loc[finite, "availability_utc"] >= joined.loc[finite, "timestamp"]).sum()),
        "domain_metrics": domain, "median_psi": float(np.median([v["psi"] for v in domain.values()])),
        "features_psi_above_2": int(sum(v["psi"] > 2 for v in domain.values())),
        "minimum_late_support": float(min(v["late_inside_early_p01_p99"] for v in domain.values())),
    }
    gate_cfg = protocol["preflight"]
    gates = {
        "official_https_cftc_archives": all(item["url"].startswith("https://www.cftc.gov/") and item["archive_integrity"] for item in archives),
        "exact_gold_contract_identity": names == [protocol["data"]["gold_name_pattern"]] and codes == [protocol["data"]["gold_contract_market_code"]],
        "minimum_gold_reports": len(development_reports) >= gate_cfg["minimum_gold_reports"],
        "unique_report_dates": duplicate_dates <= gate_cfg["maximum_duplicate_report_dates"],
        "maximum_report_gap": float(gaps.max()) <= gate_cfg["maximum_report_gap_days"],
        "raw_numeric_complete_valid": profile["raw_numeric_null_count_pre_2026"] == 0 and profile["raw_numeric_nonpositive_open_interest"] == 0,
        "point_in_time_join_strict": profile["maximum_availability_not_before_decision_violations"] == 0,
        "minimum_m15_join_coverage": coverage >= gate_cfg["minimum_m15_join_coverage"],
        "feature_finite_rate": float(finite.mean()) >= gate_cfg["minimum_feature_finite_rate"],
        "minimum_late_support": profile["minimum_late_support"] >= gate_cfg["minimum_late_support_inside_early_p01_p99"],
        "median_psi": profile["median_psi"] <= gate_cfg["maximum_median_psi"],
        "features_psi_above_2": profile["features_psi_above_2"] <= gate_cfg["maximum_features_psi_above_2"],
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    curated = output / "cftc_gold_positioning_point_in_time.csv"
    features.to_csv(curated, index=False)
    joined_path = output / "m15_label_free_context_join.csv"
    joined.to_csv(joined_path, index=False)
    payload = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(args.protocol.resolve()), "protocol_sha256": digest(args.protocol),
        "m15_source": str(args.m15.resolve()), "m15_sha256": digest(args.m15), "m15_read_nrows": args.m15_read_nrows,
        "curated_path": str(curated.resolve()), "curated_sha256": digest(curated),
        "joined_path": str(joined_path.resolve()), "joined_sha256": digest(joined_path),
        **profile, "gates": gates, "labels_returns_future_quotes_accessed": False, "model_fits": 0,
        "status": "PASS_LABEL_FREE_PREFLIGHT" if all(gates.values()) else "FAIL_LABEL_FREE_PREFLIGHT_CLOSE_FAMILY",
        "decision": "QUEUE_ONE_FROZEN_CANDIDATE_AFTER_INDEPENDENT_VALIDATION" if all(gates.values()) else "CLOSE_WITHOUT_TRAINING_OR_REFINEMENT",
        "safety": {"source_mismatch_context_only": True, "live_trading_enabled": False, "auto_execution": False, "holdout_access_count": 0, "orders_called": False}
    }
    atomic_json(output / "summary.json", payload)
    print(json.dumps({key: payload[key] for key in ["status", "gold_reports_pre_2026", "m15_join_coverage", "median_psi", "features_psi_above_2", "minimum_late_support", "gates"]}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
