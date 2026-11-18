#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create isolated IGH-aligned TRB V3 179 Elastic-Net base schemes."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = Path("TRB/set/configs/schemes/generated/hierarchical_igh_aligned")
BASES = {
 "total":{"old":"trb_v3_179_total_m0_m3_repeat100_top10000_r3_v1","new":"trb_v3_179_igh_aligned_total_m0_m3_repeat100_top10000_v1","n":179,"models":("M0","M1","M2","M3"),"manifest":"TRB/set/training_bundles/trb_v3_179_top10000_total_training_inputs_v1/aa_clone_table_manifest.csv"},
 "pbmc":{"old":"trb_v3_179_pbmc_m0_m2_repeat100_top10000_r3_v1","new":"trb_v3_179_igh_aligned_pbmc_m0_m2_repeat100_top10000_v1","n":116,"models":("M0","M1","M2"),"manifest":"TRB/set/training_bundles/trb_v3_179_top10000_pbmc_training_inputs_v1/aa_clone_table_manifest.csv"},
 "buffycoat":{"old":"trb_v3_179_buffycoat_m0_m2_repeat100_top10000_r3_v1","new":"trb_v3_179_igh_aligned_buffycoat_m0_m2_repeat100_top10000_v1","n":63,"models":("M0","M1","M2"),"manifest":"TRB/set/training_bundles/trb_v3_179_top10000_buffycoat_training_inputs_v1/aa_clone_table_manifest.csv"},
}
PUBLIC=["all_ref_public_clone_ratio","all_ref_public_frequency_sum","shared_ref_clone_ratio","shared_ref_frequency_sum","RA_specific_ref_clone_ratio","RA_specific_ref_frequency_sum","ILD_specific_ref_clone_ratio","ILD_specific_ref_frequency_sum"]

def models_for(names):
 out={}
 for name in names:
  out[name]={
   "description":{"M0":"Clinical baseline: age and sex.","M1":"M0 plus Core83 and inner-fold Top-500 weighted/unweighted 3-mers.","M2":"M1 plus eight raw_bilateral public-reference features.","M3":"M2 plus sample material."}[name],
   "numeric":["age"],
   "categorical":["sex"]+(["material"] if name=="M3" else []),
   "static_feature_groups":[] if name=="M0" else ["static_tcr_candidate_predictors"],
   "dynamic_public":PUBLIC if name in {"M2","M3"} else [],
  }
 return out

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--repository-root",default=str(ROOT));ap.add_argument("--apply",action="store_true");args=ap.parse_args()
 root=Path(args.repository_root).expanduser().resolve(); pending=[]
 for cohort,spec in BASES.items():
  base=root/"TRB/set/experiments"/spec["old"]/"00_config/resolved_config.yaml"
  if not base.is_file():raise FileNotFoundError(base)
  raw=yaml.safe_load(base.read_text(encoding="utf-8"))
  if int(raw["data"]["train"]["expected_samples"])!=spec["n"]:raise ValueError(f"{cohort}: sample count mismatch")
  if int(raw["cross_validation"]["outer_repeats"])!=100:raise ValueError(f"{cohort}: expected 100 repeats")
  expected=("M0_clinical","M1_tcr","M2_tcr_clinical","M3_tcr_clinical_material") if cohort=="total" else ("M0_clinical","M1_tcr","M2_tcr_clinical")
  if tuple(raw["models"])!=expected:raise ValueError(f"{cohort}: historical model contract mismatch: {tuple(raw['models'])}")
  scheme={
   "scheme_version":"1.0",
   "scheme":{"id":spec["new"],"description":f"{cohort.upper()} TRB V3 179 IGH-aligned M0-M3; 100 frozen repeats; inner-fold-only 3-mer selection.","status":"development_igh_aligned","base_config":base.relative_to(root).as_posix(),"output_root":f"TRB/set/experiments/{spec['new']}"},
   "replacements":{"models":models_for(spec["models"])},
   "overrides":{"model_selection":{"expected_resolved_columns":{}}},
   "repeat_3mer_features":{"enabled":True,"selection_scope":"inner_fold_train_only","ranking_rule":"prevalence_then_total_variance","aa_manifest":spec["manifest"],"sequence_column":"cdr3_aa","min_sample_count":5,"min_prevalence":0.05,"min_unweighted_variance":1e-12,"min_weighted_variance":1e-12,"max_kmers":500,"top_k_values":[50,100,200,500],"expected_core_feature_count":83,"core_group_name":"core83","legacy_group_name":"static_tcr_candidate_predictors"}
  }
  pending.append((root/SOURCE_DIR/f"{spec['new']}.yaml",scheme))
 for path,obj in pending:
  print(f"[PLAN] {path.relative_to(root)}")
  if args.apply:
   path.parent.mkdir(parents=True,exist_ok=True); text=yaml.safe_dump(obj,allow_unicode=True,sort_keys=False)
   if path.exists() and path.read_text(encoding="utf-8")!=text:raise FileExistsError(f"Refusing to overwrite {path}")
   if not path.exists():path.write_text(text,encoding="utf-8")
 if not args.apply:print("DRY-RUN PASS; add --apply to write schemes.");return 0
 preparer=root/"TRB/set/scripts_v2/prepare_analysis_scheme.py"
 for path,_ in pending:subprocess.run([sys.executable,str(preparer),"--scheme",str(path),"--repository-root",str(root),"--dry-run"],cwd=root,check=True)
 for path,_ in pending:subprocess.run([sys.executable,str(preparer),"--scheme",str(path),"--repository-root",str(root)],cwd=root,check=True)
 print("IGH-ALIGNED ELASTIC-NET BASE SCHEMES: COMPLETE")
 print("Next: python3 TRB/set/scripts_v2/prepare_hierarchical_five_algorithm_schemes.py --dry-run")
 return 0
if __name__=="__main__":raise SystemExit(main())
