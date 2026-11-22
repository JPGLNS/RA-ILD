#!/usr/bin/env python3
"""Compare TCR-only vs Age+Sex+TCR using identical frozen holdouts across 100 repeats."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "TRB/result/feature_framework/layered_feature_experiments"
MANIFEST = ROOT / "TRB/set/configs/layered_feature_experiments/tcr_only_ablation/TCR_ONLY_ABLATION_MANIFEST.json"

def checked(directory, name):
    needed = ["06_outer_metrics.csv", "06_outer_predictions.csv.gz", "06_run_manifest.json"]
    for item in needed:
        if not (directory/item).is_file():
            raise FileNotFoundError(f"{name}: {directory/item}")
    run = json.loads((directory/"06_run_manifest.json").read_text(encoding="utf-8"))
    execution=run["execution"]
    if execution.get("repeat_count") != 100 or sorted(execution.get("repeat_ids", [])) != list(range(1,101)):
        raise ValueError(f"{name}: incomplete run manifest")
    metrics=pd.read_csv(directory/"06_outer_metrics.csv")
    if sorted(metrics["outer_repeat"].astype(int).tolist()) != list(range(1,101)):
        raise ValueError(f"{name}: incomplete/duplicate metrics")
    metrics=metrics.set_index("outer_repeat").sort_index()
    for k in ("roc_auc","pr_auc"):
        v=pd.to_numeric(metrics[k],errors="raise").to_numpy(float)
        if not np.isfinite(v).all() or ((v<0)|(v>1)).any():
            raise ValueError(f"{name}: invalid {k}")
    pred=pd.read_csv(directory/"06_outer_predictions.csv.gz",usecols=["outer_repeat","sample_id","true_label"])
    if pred.duplicated(["outer_repeat","sample_id"]).any():
        raise ValueError(f"{name}: duplicate sample predictions")
    pred=pred.sort_values(["outer_repeat","sample_id"]).reset_index(drop=True)
    return metrics,pred,run

def assert_no_clinical(directory,run,expected_groups):
    if run["resolved_experiment"]["features"]["clinical_groups"] != []:
        raise ValueError("Clinical groups remain present")
    if run["resolved_experiment"]["features"]["static_tcr_groups"] != expected_groups:
        raise ValueError("TCR group mismatch in run manifest")
    if run["execution"].get("static_tcr_predictor_count",0)<1:
        raise ValueError("No TCR predictors")
    audit=pd.read_csv(directory/"06_preprocessing_audit.csv.gz",usecols=["feature_name"])
    forbidden=audit.feature_name.astype(str).str.fullmatch(r"(?i)(age|sex|sex__.*|material|material__.*)")
    if forbidden.any():
        raise ValueError(f"Clinical predictors in model: {audit.loc[forbidden,'feature_name'].unique().tolist()}")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply",action="store_true")
    args=ap.parse_args()
    entries=json.loads(MANIFEST.read_text(encoding="utf-8"))["entries"]
    if len(entries)!=3 or {e["cohort"] for e in entries}!={"total","pbmc","buffycoat"}:
        raise ValueError("Manifest must have 3 cohorts")
    rows=[];paired=[]
    for e in entries:
        cohort=e["cohort"]
        only_dir=ROOT/e["output_dir"]
        original_dir=RESULT/"layer1"/cohort/"elastic_net"/e["combination_id"]
        a,pa,run=checked(only_dir,f"{cohort} TCR-only")
        b,pb,_=checked(original_dir,f"{cohort} clinical+TCR")
        if not pa.equals(pb):
            raise ValueError(f"{cohort}: same-repeat holdout IDs/labels differ")
        assert_no_clinical(only_dir,run,e["feature_groups"])
        row={"cohort":cohort,"combination_id":e["combination_id"],
             "tcr_groups":" + ".join(e["feature_groups"]),"repeats":100}
        for metric in ("roc_auc","pr_auc"):
            va=a[metric].to_numpy(float);vb=b[metric].to_numpy(float);delta=va-vb
            row[f"tcr_only_{metric}"]=float(va.mean())
            row[f"age_sex_tcr_{metric}"]=float(vb.mean())
            row[f"delta_{metric}"]=float(delta.mean())
            row[f"positive_repeat_fraction_{metric}"]=float((delta>0).mean())
            for repeat, x,y,d in zip(range(1,101),va,vb,delta):
                paired.append({"cohort":cohort,"combination_id":e["combination_id"],
                               "outer_repeat":repeat,"metric":metric,
                               "tcr_only":x,"age_sex_tcr":y,"delta_tcr_only_minus_full":d})
        rows.append(row)
        print(f"[PASS] {cohort}: 100 paired repeats; no age, sex, material predictors")
    summary=pd.DataFrame(rows)
    print("\nTCR-only minus Age+Sex+TCR (paired ROC-AUC and PR-AUC):")
    print(summary.round(4).to_string(index=False))
    if args.apply:
        out=RESULT/"tcr_only_ablation_summary"
        out.mkdir(parents=True,exist_ok=True)
        summary.to_csv(out/"TCR_ONLY_VS_AGE_SEX_TCR_SUMMARY.csv",index=False)
        pd.DataFrame(paired).to_csv(out/"TCR_ONLY_VS_AGE_SEX_TCR_PAIRED_REPEATS.csv",index=False)
        print("Wrote:",out)
    else:
        print("DRY-RUN PASS; add --apply to export summary CSVs")
    return 0
if __name__=="__main__":
    raise SystemExit(main())
