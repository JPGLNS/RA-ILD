#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the final IGH release contract without selecting a final model."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
from typing import Iterable
import pandas as pd

SCRIPT_DIR=Path(__file__).resolve().parent
SRC_DIR=SCRIPT_DIR.parent/"src"
if str(SRC_DIR) not in sys.path: sys.path.insert(0,str(SRC_DIR))
from ra_ild_igh.config import load_experiment_config
from ra_ild_igh.release import EXPECTED_ASSIGNMENT_SHA256, EXPECTED_BRANCH, EXPECTED_SPLIT_SET_ID, validate_shm_aggregation_fixture

def require(ok: bool, message: str):
    if not ok: raise ValueError(message)

def sha(path: Path) -> str:
    h=hashlib.sha256();
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def git(root: Path,*args: str):
    p=subprocess.run(["git",*args],cwd=root,text=True,capture_output=True)
    if p.returncode: raise RuntimeError(p.stderr.strip() or "git failed")
    return p.stdout.strip()

def scan_runtime_files(root: Path):
    candidates=list((root/"IGH/set/src/ra_ild_igh").rglob("*.py"))+list((root/"IGH/set/configs").rglob("*.yaml"))
    bad=[]
    for path in candidates:
        text=path.read_text(encoding="utf-8",errors="replace")
        for token in ("ra_ild_trb", "TRB/set/", "/TRB/"):
            if token in text: bad.append((str(path.relative_to(root)),token))
    return bad,len(candidates)

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--baseline-config",required=True); p.add_argument("--repository-root",default=None); p.add_argument("--allow-dirty-source",action="store_true"); return p.parse_args()

def main():
    args=parse_args(); root=Path(args.repository_root).resolve() if args.repository_root else Path.cwd().resolve()
    config=load_experiment_config(Path(args.config),repository_root=root)
    baseline=load_experiment_config(Path(args.baseline_config),repository_root=root)
    require(config.experiment_id=="igh_scheme_003_repeat3_no_clinical","Unexpected release experiment")
    require(baseline.experiment_id=="igh_baseline_m2_v1","Unexpected historical baseline")
    repeated=config.section("repeated_holdout_training"); final=config.section("final_model")
    require(repeated.get("mode")=="frozen_repeated_holdout","Not a frozen repeated holdout")
    require(repeated.get("split_set_id")==EXPECTED_SPLIT_SET_ID,"Split-set ID mismatch")
    require(final.get("status")=="not_selected","Final model must remain not_selected")
    assignments=config.path("repeated_holdout_training.assignments",must_exist=True,expect="file")
    require(sha(assignments)==EXPECTED_ASSIGNMENT_SHA256,"Assignment SHA256 mismatch")
    split_marker=config.path("repeated_holdout_training.frozen_marker",must_exist=True,expect="file")
    split=json.loads(split_marker.read_text(encoding="utf-8")); require(split.get("status")=="FROZEN","Split marker not frozen")
    bundle_marker=config.path("repeated_holdout_training.training_bundle_marker",must_exist=True,expect="file")
    bundle=json.loads(bundle_marker.read_text(encoding="utf-8")); require(bundle.get("status")=="FROZEN","Bundle marker not frozen")
    require(bundle.get("public_reference_training_partition_only") is True,"Public reference policy invalid")
    task_status=config.path("outer_tasks.status",must_exist=True,expect="file")
    status=pd.read_csv(task_status); counts=status["status"].astype(str).value_counts().to_dict()
    require(int(counts.get("complete",0))==int(config.raw["outer_tasks"]["expected_tasks"]),"Not all tasks complete")
    require(sum(int(counts.get(x,0)) for x in ("missing","incomplete","invalid"))==0,"Task tree contains non-complete tasks")
    aggregate=config.path("aggregation.output_dir")/"08_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json"
    require(aggregate.is_file(),"Aggregation completion marker missing")
    agg=json.loads(aggregate.read_text(encoding="utf-8")); require(agg.get("status")=="COMPLETE","Aggregation incomplete")
    require(agg.get("assignment_sha256")==EXPECTED_ASSIGNMENT_SHA256,"Aggregation assignment mismatch")
    require(agg.get("automatic_final_model_selection") is False,"Aggregation selected a winner")
    require(agg.get("independent_test_read") is False,"Aggregation read independent test")
    bad,scanned=scan_runtime_files(root); require(not bad,f"Forbidden TRB runtime references: {bad[:10]}")
    shm=validate_shm_aggregation_fixture(); require(shm.get("passed") is True,"IGH SHM aggregation fixture failed")
    branch=git(root,"branch","--show-current"); require(branch==EXPECTED_BRANCH,f"Unexpected branch {branch}")
    tracked=git(root,"status","--porcelain","--untracked-files=no")
    all_status=git(root,"status","--porcelain")
    untracked_igh=[x[3:] for x in all_status.splitlines() if x.startswith("?? ") and x[3:].startswith("IGH/")]
    if not args.allow_dirty_source:
        require(tracked=="","Tracked/staged source changes exist")
        require(not untracked_igh,f"Untracked IGH source paths exist: {untracked_igh}")
        head=git(root,"rev-parse","HEAD"); upstream=git(root,"rev-parse","@{u}"); require(head==upstream,"HEAD is not synchronized with upstream")
    print("RA-ILD IGH release contract validation: PASS")
    print(f"Experiment:          {config.experiment_id}")
    print(f"Historical baseline: {baseline.experiment_id}")
    print(f"Runtime files scanned:{scanned}")
    print(f"Completed tasks:     {int(counts.get('complete',0))}")
    print(f"Metric rows:         {int(agg.get('metric_rows',0))}")
    print(f"Prediction rows:     {int(agg.get('prediction_rows',0))}")
    print(f"Stable features:     {int(agg.get('stable_feature_rows',0))}")
    print(f"SHM fixture:         PASS ({shm['nt_clone_number']} NT -> 1 AA)")
    print(f"Source clean strict: {not args.allow_dirty_source}")
    print("IGH_RELEASE_CONTRACT_VALIDATION_PASS")
    return 0
if __name__=="__main__":
    try: raise SystemExit(main())
    except Exception as exc: print(f"RA-ILD IGH release contract validation: FAIL\n{exc}",file=sys.stderr); raise SystemExit(1)
