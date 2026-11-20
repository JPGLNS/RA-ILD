#!/usr/bin/env python3
"""Run the generated TRB V3 Layer-1 configurations with bounded concurrency."""
from __future__ import annotations
import argparse, concurrent.futures, json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
CONFIG_ROOT=ROOT/"TRB/set/configs/layered_feature_experiments/generated/layer1"
MANIFEST=CONFIG_ROOT/"LAYER1_CONFIG_MANIFEST.json"
SCRIPT=ROOT/"TRB/set/scripts/run_06_unified_ml.py"
REGISTRY=ROOT/"TRB/set/configs/feature_framework/trb_feature_registry_v3_179_top10000.yaml"
TRAIN_META=ROOT/"TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total/metadata_train_70.csv"
TEST_META=ROOT/"TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total/metadata_test_30.csv"
STEP02=ROOT/"TRB/result/feature_framework/cache/step02/top10000"
OUT=ROOT/"TRB/result/feature_framework/layered_feature_experiments/layer1"
LOG=ROOT/"TRB/result/feature_framework/layered_feature_experiments/logs/layer1"
STATUS=ROOT/"TRB/result/feature_framework/layered_feature_experiments/status/layer1"

def run_one(entry, args):
    cohort,algorithm,combo=entry["cohort"],entry["algorithm"],entry["combination_id"]
    key=f"{cohort}__{algorithm}__{combo}"
    out=OUT/cohort/algorithm/combo
    run_tag = "full100" if args.repeat is None else f"smoke_repeat_{args.repeat:03d}"
    if args.repeat is not None:
        out = out / run_tag
    log=LOG/cohort/algorithm/f"{combo}.{run_tag}.log"
    status=STATUS/cohort/algorithm/f"{combo}.{run_tag}.json"
    if status.is_file():
        try:
            old=json.loads(status.read_text())
            if old.get("status")=="COMPLETE" and (out/"06_outer_metrics.csv").is_file():
                return {"key":key,"status":"SKIPPED_COMPLETE"}
        except Exception: pass
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        return {"key":key,"status":"BLOCKED_OUTPUT_EXISTS","output":str(out)}
    cmd=[sys.executable,str(SCRIPT),"--config",str(ROOT/entry["config"]),
         "--train-step02-dir",str(STEP02/"train"),"--test-step02-dir",str(STEP02/"test"),
         "--train-metadata",str(TRAIN_META),"--test-metadata",str(TEST_META),
         "--registry",str(REGISTRY),"--output-dir",str(out),"--workers","1"]
    if args.repeat is not None: cmd += ["--repeat",str(args.repeat)]
    if args.overwrite: cmd.append("--overwrite")
    log.parent.mkdir(parents=True,exist_ok=True); status.parent.mkdir(parents=True,exist_ok=True)
    started=time.time()
    env=os.environ.copy()
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        env[name]="1"
    with log.open("w",encoding="utf-8") as handle:
        proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    result={"key":key,"cohort":cohort,"algorithm":algorithm,"combination_id":combo,
            "status":"COMPLETE" if proc.returncode==0 else "FAILED","returncode":proc.returncode,
            "runtime_seconds":round(time.time()-started,2),"output":str(out),"log":str(log)}
    status.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=MANIFEST)
    p.add_argument("--cohort",choices=["total","pbmc","buffycoat"])
    p.add_argument("--algorithm",choices=["elastic_net","ridge","lasso","linear_svm","xgboost"])
    p.add_argument("--combination-id")
    p.add_argument("--limit",type=int)
    p.add_argument("--repeat",type=int,help="Run only one frozen repeat for acceptance testing; omit for all 100.")
    p.add_argument("--max-experiments",type=int,default=2)
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    if a.max_experiments<1 or a.max_experiments>8: raise ValueError("--max-experiments must be 1..8")
    if a.repeat is not None and not 1<=a.repeat<=100: raise ValueError("--repeat must be 1..100")
    manifest=json.loads(a.manifest.read_text(encoding="utf-8"))
    entries=manifest["entries"]
    if a.cohort: entries=[x for x in entries if x["cohort"]==a.cohort]
    if a.algorithm: entries=[x for x in entries if x["algorithm"]==a.algorithm]
    if a.combination_id: entries=[x for x in entries if x["combination_id"]==a.combination_id]
    if a.limit is not None: entries=entries[:a.limit]
    if not entries: raise ValueError("no Layer-1 entries matched the filters")
    if not a.execute:
        print(f"DRY-RUN: selected {len(entries)} configs; repeat={'all 100' if a.repeat is None else a.repeat}; max experiments={a.max_experiments}")
        print("No model process launched. Add --execute after Step02 caches and config manifest are audited.")
        return 0
    for path in (SCRIPT,REGISTRY,TRAIN_META,TEST_META,STEP02/"train/02_core_sample_features.csv",STEP02/"test/02_core_sample_features.csv"):
        if not path.exists(): raise FileNotFoundError(f"required prebuilt input missing: {path}")
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.max_experiments) as pool:
        futures=[pool.submit(run_one,e,a) for e in entries]
        for future in concurrent.futures.as_completed(futures):
            result=future.result(); results.append(result); print(f"[{result['status']}] {result['key']}",flush=True)
    report=ROOT/"TRB/result/feature_framework/layered_feature_experiments/layer1_batch_status.csv"
    report.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(results).to_csv(report,index=False)
    failed=[x for x in results if x["status"] in {"FAILED","BLOCKED_OUTPUT_EXISTS"}]
    print(f"Completed report: {report}; failed/blocked={len(failed)}")
    return 1 if failed else 0
if __name__=="__main__": raise SystemExit(main())
