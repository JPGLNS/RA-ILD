#!/usr/bin/env python3
"""Aggregate paired Layer-2 states and quantify incremental feature value."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[3]
MANIFEST=ROOT/"TRB/set/configs/layered_feature_experiments/generated/layer2/LAYER2_CONFIG_MANIFEST.json"
RESULTS=ROOT/"TRB/result/feature_framework/layered_feature_experiments/layer2"
STATES=("F0_static","F1_3mer","F2_ra_hit","F3_3mer_ra_hit")
def mean_ci(values):
 x=np.asarray(values,dtype=float);n=len(x);m=float(np.mean(x));sd=float(np.std(x,ddof=1)) if n>1 else 0.0
 half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else float("nan")
 return m,sd,m-half,m+half
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--manifest",type=Path,default=MANIFEST);p.add_argument("--results-root",type=Path,default=RESULTS);p.add_argument("--apply",action="store_true");a=p.parse_args()
 entries=json.loads(a.manifest.read_text(encoding="utf-8"))["entries"]
 keys=sorted({(x["cohort"],x["algorithm"],x["combination_id"]) for x in entries})
 summary=[];paired=[];missing=[]
 for cohort,algorithm,combo in keys:
  frames={}
  for state in STATES:
   path=a.results_root/cohort/algorithm/combo/state/"06_outer_metrics.csv"
   if not path.is_file(): missing.append(str(path));continue
   frame=pd.read_csv(path)
   if len(frame)!=100: missing.append(f"{path} (n={len(frame)})");continue
   if "outer_repeat" not in frame: raise ValueError(f"outer_repeat missing: {path}")
   if frame["outer_repeat"].duplicated().any(): raise ValueError(f"duplicate repeat IDs: {path}")
   frames[state]=frame.set_index("outer_repeat").sort_index()
  if len(frames)!=4: continue
  common=frames["F0_static"].index
  if any(not frames[s].index.equals(common) for s in STATES[1:]): raise ValueError(f"repeat IDs differ across paired states: {cohort}/{algorithm}/{combo}")
  base=frames["F0_static"]
  for state in STATES:
   frame=frames[state]
   for metric in ("roc_auc","pr_auc","sensitivity_recall","specificity","f1"):
    vals=pd.to_numeric(frame[metric],errors="raise").to_numpy(float)
    m,sd,lo,hi=mean_ci(vals)
    summary.append({"cohort":cohort,"algorithm":algorithm,"combination_id":combo,"feature_state":state,"metric":metric,"mean":m,"sd":sd,"ci95_low":lo,"ci95_high":hi,"n_repeats":100})
  for state in STATES[1:]:
   for metric in ("roc_auc","pr_auc"):
    delta=(pd.to_numeric(frames[state][metric])-pd.to_numeric(base[metric])).to_numpy(float)
    m,sd,lo,hi=mean_ci(delta)
    paired.append({"cohort":cohort,"algorithm":algorithm,"combination_id":combo,"contrast":f"{state}-F0_static","metric":metric,"mean_paired_delta":m,"sd_paired_delta":sd,"ci95_low":lo,"ci95_high":hi,"n_repeats":100,"positive_repeat_fraction":float(np.mean(delta>0))})
 print(f"Promoted combinations: {len(keys)}")
 print(f"Fully paired combinations: {len({(x['cohort'],x['algorithm'],x['combination_id']) for x in paired})}")
 print(f"Missing/incomplete files: {len(missing)}")
 if missing:
  print("Examples:",*missing[:10],sep="\n  ")
  return 2
 if not a.apply:
  print(f"DRY-RUN: would write {len(summary)} state-metric rows and {len(paired)} paired-delta rows.");return 0
 out=a.results_root.parent/"layer2_summary";out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(summary).to_csv(out/"LAYER2_STATE_METRICS.csv",index=False)
 pd.DataFrame(paired).to_csv(out/"LAYER2_PAIRED_INCREMENTAL_VALUE.csv",index=False)
 print(f"Wrote Layer-2 state and paired-delta summaries: {out}");return 0
if __name__=="__main__": raise SystemExit(main())
