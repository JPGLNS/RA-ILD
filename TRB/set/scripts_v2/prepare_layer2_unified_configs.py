#!/usr/bin/env python3
"""Generate paired Layer-2 Phase-6 configs for promoted Core83 combinations."""
from __future__ import annotations
import argparse,ast,copy,json
from pathlib import Path
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[3]
TEMPLATE=ROOT/"TRB/set/configs/feature_framework/trb_experiment_unified_ml_template_v1.yaml"
PROMOTED=ROOT/"TRB/result/feature_framework/layered_feature_experiments/layer1_summary/LAYER1_PROMOTED_TOPN.csv"
OUT=ROOT/"TRB/set/configs/layered_feature_experiments/generated/layer2"
STATES=(("F0_static",False,False),("F1_3mer",True,False),("F2_ra_hit",False,True),("F3_3mer_ra_hit",True,True))
def parse_groups(value):
 if isinstance(value,list): return value
 parsed=ast.literal_eval(str(value))
 if not isinstance(parsed,(list,tuple)) or not all(isinstance(x,str) for x in parsed): raise ValueError(f"invalid feature_groups: {value}")
 return list(parsed)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--promoted",type=Path,default=PROMOTED)
 p.add_argument("--apply",action="store_true")
 a=p.parse_args()
 if not a.promoted.is_file(): raise FileNotFoundError(a.promoted)
 promoted=pd.read_csv(a.promoted)
 required={"cohort","algorithm","rank","combination_id","feature_groups","layer1_roc_auc","layer1_pr_auc","layer1_n_repeats"}
 if not required.issubset(promoted.columns): raise ValueError(f"promoted CSV missing {sorted(required-set(promoted.columns))}")
 if not (pd.to_numeric(promoted["layer1_n_repeats"])==100).all(): raise ValueError("Layer 2 requires complete 100-repeat Layer-1 results")
 template=yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
 files=[]; rows=[]
 for item in promoted.to_dict("records"):
  groups=parse_groups(item["feature_groups"])
  for state,use_3mer,use_ra in STATES:
   cfg=copy.deepcopy(template)
   exp_id=f"trb179_{item['cohort']}_{item['algorithm']}_{item['combination_id'].lower()}_{state.lower()}"
   cfg["experiment"]["id"]=exp_id; cfg["cohort"]["scope"]=item["cohort"]
   cfg["repertoire"]={"mode":"topk","top_k":10000}
   cfg["features"]["clinical_groups"]=["clinical_age_sex"]
   cfg["features"]["static_tcr_groups"]=groups
   cfg["features"]["kmer"]={"enabled":use_3mer,"k":3,"representation":"both","top_n":500,"min_sample_count":5,"min_prevalence":0.05,"min_unweighted_variance":1e-12,"min_weighted_variance":1e-12}
   cfg["features"]["enriched_dictionary"]={"enabled":use_ra,"threshold_pct":20.0,"delta_pct":10.0,"features":["RA_dict_hit_rate"] if use_ra else []}
   cfg["model"]["algorithm"]=item["algorithm"]
   # Paired states share the exact frozen split assignment used in Layer 1.
   split={"total":"TRB/set/split_sets/ra_ild_repeat100_v3_179_top10000/repeated_holdout_assignments.csv","pbmc":"TRB/set/split_sets/ra_ild_pbmc_repeat100_v3_180/repeated_holdout_assignments.csv","buffycoat":"TRB/set/split_sets/ra_ild_buffycoat_repeat100_v3_179_top10000/repeated_holdout_assignments.csv"}[item["cohort"]]
   cfg["validation"].update({"repeats":100,"folds":5,"split_assignments":split}); cfg["compute"]["workers"]=1
   rel=Path("TRB/set/configs/layered_feature_experiments/generated/layer2")/item["cohort"]/item["algorithm"]/f"{exp_id}.yaml"
   rows.append({"cohort":item["cohort"],"algorithm":item["algorithm"],"rank":int(item["rank"]),"combination_id":item["combination_id"],"feature_state":state,"feature_groups":groups,"include_3mer":use_3mer,"include_ra_hit":use_ra,"ra_hit_threshold_pct":20.0 if use_ra else None,"ra_hit_delta_pct":10.0 if use_ra else None,"config":rel.as_posix(),"outer_repeats":100})
   files.append((ROOT/rel,cfg))
 print(f"Promoted combinations: {len(promoted)}")
 print(f"Layer-2 configs: {len(files)} = promoted combinations × 4 paired feature states")
 print("RA-hit uses the established T=20% and Delta=10 percentage points; feature=RA_dict_hit_rate")
 print("3-mer uses one fold-local vocabulary with both unweighted and weighted representations (Top500 each)")
 if not a.apply: print("DRY-RUN: no files written; add --apply to materialize configs"); return 0
 for path,cfg in files:
  path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists(): raise FileExistsError(f"refusing to overwrite {path}")
  path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
 manifest=OUT/"LAYER2_CONFIG_MANIFEST.json"; manifest.parent.mkdir(parents=True,exist_ok=True)
 manifest.write_text(json.dumps({"version":"1.0","promoted_combinations":len(promoted),"config_count":len(rows),"entries":rows},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(f"Wrote {len(files)} configs and {manifest}"); return 0
if __name__=="__main__": raise SystemExit(main())
