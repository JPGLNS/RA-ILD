#!/usr/bin/env python3
"""Aggregate complete 100-repeat Layer-1 runs and promote top-N per cohort/algorithm."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"src"))
from ra_ild_trb.layered_experiments import rank_layer1_results,build_layer2_manifest

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--manifest",type=Path,default=ROOT/"TRB/set/configs/layered_feature_experiments/generated/layer1/LAYER1_CONFIG_MANIFEST.json")
 p.add_argument("--results-root",type=Path,default=ROOT/"TRB/result/feature_framework/layered_feature_experiments/layer1")
 p.add_argument("--top-n",type=int,default=5)
 p.add_argument("--allow-incomplete",action="store_true")
 p.add_argument("--apply",action="store_true")
 a=p.parse_args()
 manifest=json.loads(a.manifest.read_text(encoding="utf-8")); rows=[]; missing=[]
 for item in manifest["entries"]:
  path=a.results_root/item["cohort"]/item["algorithm"]/item["combination_id"]/"06_outer_metrics.csv"
  if not path.is_file(): missing.append((item,path)); continue
  frame=pd.read_csv(path)
  if len(frame)!=100: missing.append((item,path)); continue
  for col in ("roc_auc","pr_auc"):
   frame[col]=pd.to_numeric(frame[col],errors="raise")
   if not np.isfinite(frame[col]).all(): raise ValueError(f"non-finite {col}: {path}")
  rows.append({"cohort":item["cohort"],"algorithm":item["algorithm"],"combination_id":item["combination_id"],
   "feature_groups":list(item["feature_groups"]),"n_repeats":len(frame),
   "roc_auc":float(frame.roc_auc.mean()),"roc_auc_sd":float(frame.roc_auc.std(ddof=1)),
   "roc_auc_median":float(frame.roc_auc.median()),"pr_auc":float(frame.pr_auc.mean()),
   "pr_auc_sd":float(frame.pr_auc.std(ddof=1)),"pr_auc_median":float(frame.pr_auc.median()),
   "sensitivity_mean":float(frame.sensitivity_recall.mean()),"specificity_mean":float(frame.specificity.mean()),
   "f1_mean":float(frame.f1.mean()),"result_dir":str(path.parent)})
 print(f"Complete 100-repeat results: {len(rows)}/{len(manifest['entries'])}")
 print(f"Missing/incomplete results: {len(missing)}")
 if missing and not a.allow_incomplete:
  print("STRICT MODE: no promotion performed until every Layer-1 combination has 100 repeats."); return 2
 if not rows: raise ValueError("no complete Layer-1 results found")
 for cohort in ("total","pbmc","buffycoat"):
  for algorithm in ("elastic_net","ridge","lasso","linear_svm","xgboost"):
   n=sum(x["cohort"]==cohort and x["algorithm"]==algorithm for x in rows)
   if n!=127: raise ValueError(f"promotion requires 127/127 completed combinations for {cohort}/{algorithm}; got {n}")
 selected=rank_layer1_results(rows,top_n=a.top_n,primary_metric="roc_auc",secondary_metric="pr_auc")
 layer2=build_layer2_manifest(selected)
 out=a.results_root.parent/"layer1_summary"
 if not a.apply:
  print(f"DRY-RUN: would write metrics={len(rows)}, promoted={len(selected)}, Layer2 rows={len(layer2)} to {out}"); return 0
 out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(out/"LAYER1_METRICS_ALL.csv",index=False)
 pd.DataFrame(selected).to_csv(out/"LAYER1_PROMOTED_TOPN.csv",index=False)
 pd.DataFrame(layer2).to_csv(out/"LAYER2_INCREMENTAL_MANIFEST.csv",index=False)
 print(f"Wrote summaries and Layer-2 plan: {out}"); return 0
if __name__=="__main__": raise SystemExit(main())
