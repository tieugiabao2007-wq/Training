from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES=["real_10y","real_10y_change_1d","real_10y_change_5d","real_10y_z20","real_curve_5s10","real_curve_10s30","real_curve_5s10_change_5d","real_curve_10s30_change_5d"]
REQUIRED=["5 YR","10 YR","30 YR"]


def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def atomic_json(path:Path,payload:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f"{path.name}.{os.getpid()}.tmp");tmp.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8");os.replace(tmp,path)


def psi(ref:np.ndarray,cur:np.ndarray)->float:
    edges=np.unique(np.quantile(ref,np.linspace(0,1,11)))
    if len(edges)<3:return 0. if np.isclose(ref.mean(),cur.mean()) else 99.
    edges[0],edges[-1]=-np.inf,np.inf;a=np.clip(np.histogram(ref,bins=edges)[0]/len(ref),1e-6,None);b=np.clip(np.histogram(cur,bins=edges)[0]/len(cur),1e-6,None);return float(np.sum((b-a)*np.log(b/a)))


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--staging",type=Path,required=True);p.add_argument("--m15",type=Path,required=True);p.add_argument("--m15-read-nrows",type=int,default=85010);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();protocol=json.loads(a.protocol.read_text(encoding="utf-8"));archives=[];frames=[]
    url_by_year={int(url.split(".csv/")[1].split("/")[0]):url for url in protocol["official_source"]["urls"]}
    for year in (2024,2025,2026):
        path=a.staging/f"daily_treasury_real_yield_{year}.csv";frame=pd.read_csv(path);frame["source_year"]=year;frames.append(frame);archives.append({"year":year,"path":str(path.resolve()),"url":url_by_year[year],"size_bytes":path.stat().st_size,"sha256":digest(path),"rows":len(frame),"columns":list(frame.columns[:-1])})
    raw=pd.concat(frames,ignore_index=True);raw["rate_date"]=pd.to_datetime(raw["Date"],format="%m/%d/%Y",utc=True);raw=raw.sort_values("rate_date").reset_index(drop=True);raw["availability_utc"]=raw.rate_date+pd.Timedelta(days=1,hours=6)
    for col in REQUIRED:raw[col]=pd.to_numeric(raw[col],errors="coerce")
    raw["real_10y"]=raw["10 YR"];raw["real_10y_change_1d"]=raw.real_10y.diff();raw["real_10y_change_5d"]=raw.real_10y.diff(5);mean=raw.real_10y.rolling(20,min_periods=20).mean();std=raw.real_10y.rolling(20,min_periods=20).std(ddof=0);raw["real_10y_z20"]=(raw.real_10y-mean)/std.replace(0,np.nan);raw["real_curve_5s10"]=raw["10 YR"]-raw["5 YR"];raw["real_curve_10s30"]=raw["30 YR"]-raw["10 YR"];raw["real_curve_5s10_change_5d"]=raw.real_curve_5s10.diff(5);raw["real_curve_10s30_change_5d"]=raw.real_curve_10s30.diff(5)
    train_source=raw.loc[raw.rate_date<pd.Timestamp("2026-01-01",tz="UTC")].copy();usable=train_source.dropna(subset=FEATURES).copy();m15=pd.read_csv(a.m15,usecols=["timestamp"],nrows=a.m15_read_nrows,parse_dates=["timestamp"]);m15.timestamp=pd.to_datetime(m15.timestamp,utc=True);start=pd.Timestamp(protocol["data"]["development_start_utc"]);end=pd.Timestamp(protocol["data"]["development_end_exclusive_utc"]);decisions=m15.loc[(m15.timestamp>=start)&(m15.timestamp<end),["timestamp"]].sort_values("timestamp");joined=pd.merge_asof(decisions,usable[["availability_utc","rate_date",*FEATURES]].sort_values("availability_utc"),left_on="timestamp",right_on="availability_utc",direction="backward",allow_exact_matches=False);finite=np.isfinite(joined[FEATURES].to_numpy(float)).all(axis=1);coverage=float(finite.mean()) if len(joined) else 0.
    early=usable.loc[(usable.rate_date>=start)&(usable.rate_date<pd.Timestamp("2025-07-01",tz="UTC"))];late=usable.loc[(usable.rate_date>=pd.Timestamp("2025-07-01",tz="UTC"))&(usable.rate_date<end)];domain={}
    for f in FEATURES:
        ref=early[f].to_numpy(float);cur=late[f].to_numpy(float);lo,hi=np.quantile(ref,[.01,.99]);domain[f]={"psi":psi(ref,cur),"late_inside_early_p01_p99":float(np.mean((cur>=lo)&(cur<=hi)))}
    dates=train_source.rate_date.sort_values();gaps=dates.diff().dt.total_seconds().div(86400).dropna();duplicates=int(train_source.rate_date.duplicated().sum());required_complete=float(train_source[REQUIRED].notna().mean().min());profile={"official_files":archives,"rows_2024_2025":len(train_source),"rows_all_downloads":len(raw),"first_rate_date":raw.rate_date.min().isoformat(),"last_rate_date":raw.rate_date.max().isoformat(),"duplicate_dates_2024_2025":duplicates,"maximum_calendar_gap_days_2024_2025":float(gaps.max()),"required_column_complete_rate":required_complete,"m15_decisions":len(decisions),"m15_joined_finite_rows":int(finite.sum()),"m15_join_coverage":coverage,"availability_not_before_decision_violations":int((joined.loc[finite,"availability_utc"]>=joined.loc[finite,"timestamp"]).sum()),"domain_metrics":domain,"median_psi":float(np.median([v["psi"] for v in domain.values()])),"features_psi_above_2":sum(v["psi"]>2 for v in domain.values()),"minimum_late_support":min(v["late_inside_early_p01_p99"] for v in domain.values())}
    c=protocol["preflight"];gates={"official_https_treasury_files":all(x["url"].startswith("https://home.treasury.gov/") for x in archives),"exact_schema":all(set(["Date","5 YR","7 YR","10 YR","20 YR","30 YR"]).issubset(x["columns"]) for x in archives),"minimum_rows":len(train_source)>=c["minimum_2024_2025_rows"],"unique_dates":duplicates<=c["maximum_duplicate_dates"],"maximum_gap":float(gaps.max())<=c["maximum_business_gap_days"],"required_complete":required_complete>=c["minimum_required_column_complete_rate"],"point_in_time_join_strict":profile["availability_not_before_decision_violations"]==0,"minimum_m15_join_coverage":coverage>=c["minimum_m15_join_coverage"],"feature_finite_rate":float(finite.mean())>=c["minimum_feature_finite_rate"],"minimum_late_support":profile["minimum_late_support"]>=c["minimum_late_support_inside_early_p01_p99"],"median_psi":profile["median_psi"]<=c["maximum_median_psi"],"features_psi_above_2":profile["features_psi_above_2"]<=c["maximum_features_psi_above_2"]}
    out=a.output_dir;out.mkdir(parents=True,exist_ok=True);curated=out/"treasury_real_yield_point_in_time.csv";joined_path=out/"m15_label_free_context_join.csv";raw.to_csv(curated,index=False);joined.to_csv(joined_path,index=False);payload={"schema_version":1,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"protocol":str(a.protocol.resolve()),"protocol_sha256":digest(a.protocol),"m15_source":str(a.m15.resolve()),"m15_sha256":digest(a.m15),"m15_read_nrows":a.m15_read_nrows,"curated_path":str(curated.resolve()),"curated_sha256":digest(curated),"joined_path":str(joined_path.resolve()),"joined_sha256":digest(joined_path),**profile,"gates":gates,"labels_returns_future_quotes_accessed":False,"model_fits":0,"status":"PASS_LABEL_FREE_PREFLIGHT" if all(gates.values()) else "FAIL_LABEL_FREE_PREFLIGHT_CLOSE_FAMILY","decision":"QUEUE_ONE_FROZEN_CANDIDATE_AFTER_INDEPENDENT_VALIDATION" if all(gates.values()) else "CLOSE_WITHOUT_TRAINING_OR_REFINEMENT","safety":{"source_mismatch_context_only":True,"live_trading_enabled":False,"auto_execution":False,"holdout_access_count":0,"orders_called":False}};atomic_json(out/"summary.json",payload);print(json.dumps({k:payload[k] for k in ["status","rows_2024_2025","required_column_complete_rate","m15_join_coverage","median_psi","features_psi_above_2","minimum_late_support","gates"]},indent=2));return 0 if all(gates.values()) else 2


if __name__=="__main__":raise SystemExit(main())
