#!/usr/bin/env python3
"""Generate auditable Phase-6 YAMLs for the TRB V3 Core83 search space.

Every model includes age/sex as fixed adjustment covariates; only the selected
Core83 groups vary. 3-mer and RA-hit are disabled in Layer 1.
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"src"))
from ra_ild_trb.layered_experiments import CORE83_GROUPS, COHORTS, ALGORITHMS, build_layer1_combinations
BASE=ROOT/"TRB/set/configs/feature_framework/trb_experiment_unified_ml_template_v1.yaml"
METADATA={
 "total":("TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total/metadata_train_70.csv","TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total/metadata_test_30.csv","TRB/set/split_sets/ra_ild_repeat100_v3_179_top10000/repeated_holdout_assignments.csv"),
 "pbmc":("TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total/metadata_train_70.csv","TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total/metadata_test_30.csv","TRB/set/split_sets/ra_ild_pbmc_repeat100_v3_180/repeated_holdout_assignments.csv"),
 "buffycoat":("TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total/metadata_train_70.csv","TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total/metadata_test_30.csv","TRB/set/split_sets/ra_ild_buffycoat_repeat100_v3_179_top10000/repeated_holdout_assignments.csv"),
}
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--repository-root",type=Path,default=ROOT); p.add_argument("--apply",action="store_true"); a=p.parse_args(); root=a.repository_root.resolve()
 template=yaml.safe_load((root/BASE.relative_to(ROOT)).read_text(encoding="utf-8"))
 rows=[]; files=[]
 for cohort,n in COHORTS.items():
  _,_,split=METADATA[cohort]
  for algorithm in ALGORITHMS:
   for combo in build_layer1_combinations():
    cfg=copy.deepcopy(template); exp_id=f"trb179_{cohort}_{algorithm}_{combo.combination_id.lower()}"
    cfg["experiment"]["id"]=exp_id; cfg["cohort"]["scope"]=cohort
    cfg["repertoire"]={"mode":"topk","top_k":10000}
    cfg["features"]["clinical_groups"]=["clinical_age_sex"]
    cfg["features"]["static_tcr_groups"]=list(combo.group_ids)
    cfg["features"]["kmer"]={"enabled":False,"k":3,"representation":"both","top_n":None,"min_sample_count":5,"min_prevalence":0.05,"min_unweighted_variance":1e-12,"min_weighted_variance":1e-12}
    cfg["features"]["enriched_dictionary"]={"enabled":False,"threshold_pct":20.0,"delta_pct":10.0,"features":[]}
    cfg["model"]["algorithm"]=algorithm; cfg["validation"].update({"repeats":100,"folds":5,"split_assignments":split}); cfg["compute"]["workers"]=1
    rel=Path("TRB/set/configs/layered_feature_experiments/generated/layer1")/cohort/algorithm/f"{exp_id}.yaml"
    rows.append({"cohort":cohort,"algorithm":algorithm,"combination_id":combo.combination_id,"feature_groups":list(combo.group_ids),"expected_samples":n,"config":rel.as_posix(),"outer_repeats":100,"split_assignments":split})
    files.append((root/rel,cfg))
 print(f"Layer-1 configs: {len(files)} = 3 cohorts × Elastic Net × 127 Core83 combinations")
 print("Fixed adjustment covariates: age + sex; variable predictors: selected Core83 groups")
 print("3-mer OFF; RA-hit OFF; Top10000 V3; frozen cohort-specific split assignments")
 if not a.apply: print("DRY-RUN: no files written; add --apply to materialize configs"); return 0
 for path,cfg in files:
  path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists(): raise FileExistsError(f"refusing to overwrite {path}")
  path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
 manifest=root/"TRB/set/configs/layered_feature_experiments/generated/layer1/LAYER1_CONFIG_MANIFEST.json"
 manifest.parent.mkdir(parents=True,exist_ok=True)
 manifest.write_text(json.dumps({"version":"1.0","count":len(rows),"registry":"TRB/set/configs/feature_framework/trb_feature_registry_v3_179_top10000.yaml","entries":rows},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(f"Wrote {len(files)} configs and {manifest}"); return 0
if __name__=="__main__": raise SystemExit(main())
