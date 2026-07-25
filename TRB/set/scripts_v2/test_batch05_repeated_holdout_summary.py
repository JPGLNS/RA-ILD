#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.repeated_holdout_summary import (
    RepeatedHoldoutSummaryError,
    aggregate_repeated_holdout_results,
    compare_aggregated_schemes,
    load_frozen_assignments,
    write_repeated_holdout_aggregate,
)

@dataclass(frozen=True)
class Task:
    task_index: int
    outer_repeat: int
    outer_fold: int
    task_id: str
    output_dir: Path

class Batch05Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.models = ("M1", "M2")
        rows=[]
        for repeat in range(1,4):
            hold={f"S{x:02d}" for x in range((repeat-1)*2, (repeat-1)*2+4)}
            for i in range(10):
                rows.append({"split_set_id":"split_v1","split_id":f"split_{repeat:02d}","repeat_index":repeat,"role":"holdout" if f"S{i:02d}" in hold else "train","sample_id":f"S{i:02d}","patient_id":f"P{i:02d}","cohort":"ILD" if i%2 else "RA"})
        self.assignments=pd.DataFrame(rows)
        self.assign_path=self.root/"assignments.csv"
        self.assignments.to_csv(self.assign_path,index=False)
        sha=hashlib.sha256(self.assign_path.read_bytes()).hexdigest()
        self.marker_path=self.root/"SPLITS_FROZEN.json"
        self.marker_path.write_text(json.dumps({"status":"FROZEN","split_set_id":"split_v1","files":{"assignments":{"sha256":sha}}}),encoding="utf-8")
        self.tasks=[]
        for repeat in range(1,4):
            task=Task(repeat,repeat,1,f"repeat_{repeat:02d}_fold_01",self.root/f"task{repeat}")
            self._write_task(task)
            self.tasks.append(task)
        self.status=pd.DataFrame({"task_id":[x.task_id for x in self.tasks],"status":["complete"]*3})
    def tearDown(self): self.temp.cleanup()
    def _write_task(self,task):
        task.output_dir.mkdir()
        hold=self.assignments.query("repeat_index==@task.outer_repeat and role=='holdout'")["sample_id"].tolist()
        (task.output_dir/"06_TASK_COMPLETE.json").write_text(json.dumps({"status":"COMPLETE"}),encoding="utf-8")
        (task.output_dir/"06_task_configuration.json").write_text(json.dumps({"models":list(self.models)}),encoding="utf-8")
        roles=[]
        for s in [f"S{i:02d}" for i in range(10)]: roles.append({"sample_id":s,"outer_role":"validation" if s in hold else "training"})
        pd.DataFrame(roles).to_csv(task.output_dir/"06_task_sample_roles.csv",index=False)
        metrics=[]; preds=[]; coefs=[]
        for mi,m in enumerate(self.models):
            metrics.append({"model":m,"roc_auc":.60+.02*task.outer_repeat+.01*mi,"pr_auc":.55+.02*task.outer_repeat+.01*mi,"accuracy":.6,"sensitivity_recall":.5,"specificity":.7,"precision":.6,"f1":.55,"threshold":.45+.01*task.outer_repeat,"selected_l1_ratio_alpha":.5,"selected_lambda":10.0,"inner_selected_roc_auc":.65,"inner_selected_pr_auc":.60})
            for s in hold:
                i=int(s[1:]); truth=1 if i%2 else 0
                preds.append({"model":m,"sample_id":s,"true_label":truth,"true_cohort":"ILD" if truth else "RA","probability_ILD":.7 if truth else .3,"threshold":.5,"predicted_label":truth})
            coefs.extend([{"model":m,"feature_name":"f1","coefficient":.2,"nonzero":True},{"model":m,"feature_name":"f2","coefficient":0.0,"nonzero":False},{"model":m,"feature_name":"__INTERCEPT__","coefficient":.1,"nonzero":True}])
        pd.DataFrame(metrics).to_csv(task.output_dir/"06_outer_validation_metrics.csv",index=False)
        pd.DataFrame(preds).to_csv(task.output_dir/"06_outer_validation_predictions.csv",index=False)
        pd.DataFrame(coefs).to_csv(task.output_dir/"06_final_model_coefficients.csv",index=False)
    def _aggregate(self):
        return aggregate_repeated_holdout_results(self.tasks,self.status,self.assignments,expected_models=self.models,expected_split_set_id="split_v1",expected_split_count=3,expected_holdout_size=4,coefficient_tolerance=1e-12,minimum_selection_frequency=.5,minimum_sign_consistency=.8,manifest=pd.DataFrame())
    def test_frozen_assignments_load(self):
        frame,_,_,_=load_frozen_assignments(self.assign_path,self.marker_path,expected_split_set_id="split_v1",expected_splits=3,expected_train_size=6,expected_holdout_size=4)
        self.assertEqual(len(frame),30)
    def test_frozen_hash_mismatch_rejected(self):
        self.assign_path.write_text(self.assign_path.read_text()+"\n",encoding="utf-8")
        with self.assertRaises(RepeatedHoldoutSummaryError): load_frozen_assignments(self.assign_path,self.marker_path,expected_split_set_id="split_v1",expected_splits=3,expected_train_size=6,expected_holdout_size=4)
    def test_incomplete_task_rejected(self):
        bad=self.status.copy(); bad.loc[0,"status"]="missing"
        with self.assertRaises(RepeatedHoldoutSummaryError): aggregate_repeated_holdout_results(self.tasks,bad,self.assignments,expected_models=self.models,expected_split_set_id="split_v1",expected_split_count=3,expected_holdout_size=4,coefficient_tolerance=1e-12,minimum_selection_frequency=.5,minimum_sign_consistency=.8)
    def test_aggregate_counts_and_summary(self):
        a=self._aggregate(); self.assertEqual(len(a.metrics),6); self.assertEqual(len(a.predictions),24); self.assertEqual(len(a.metric_summary),14)
    def test_membership_mismatch_rejected(self):
        p=self.tasks[0].output_dir/"06_outer_validation_predictions.csv"; f=pd.read_csv(p); f.loc[0,"sample_id"]="BAD"; f.to_csv(p,index=False)
        with self.assertRaises(RepeatedHoldoutSummaryError): self._aggregate()
    def test_variable_holdout_coverage_is_supported(self):
        a=self._aggregate(); x=a.sample_prediction_summary.query("model=='M1'"); self.assertEqual(len(x),10); self.assertTrue((x.n_predictions==x.expected_holdout_count).all())
    def test_stable_feature_detected(self):
        a=self._aggregate(); self.assertIn("f1",set(a.stable_features.feature_name)); self.assertNotIn("f2",set(a.stable_features.feature_name))
    def test_write_and_compare_same_assignments(self):
        a=self._aggregate(); dirs=[]
        for name in ("schemeA","schemeB"):
            d=self.root/name
            write_repeated_holdout_aggregate(a,d,experiment_id=name,split_set_id="split_v1",assignment_sha256="same",split_marker_sha256="marker",expected_split_count=3,expected_holdout_size=4)
            dirs.append(d)
        summary,paired,audit=compare_aggregated_schemes(dirs); self.assertEqual(audit["scheme_count"],2); self.assertFalse(paired.empty); self.assertFalse(summary.empty)
    def test_compare_different_assignments_rejected(self):
        a=self._aggregate(); d1=self.root/"a"; d2=self.root/"b"
        write_repeated_holdout_aggregate(a,d1,experiment_id="a",split_set_id="split_v1",assignment_sha256="one",split_marker_sha256="m",expected_split_count=3,expected_holdout_size=4)
        write_repeated_holdout_aggregate(a,d2,experiment_id="b",split_set_id="split_v1",assignment_sha256="two",split_marker_sha256="m",expected_split_count=3,expected_holdout_size=4)
        with self.assertRaises(RepeatedHoldoutSummaryError): compare_aggregated_schemes([d1,d2])

if __name__=="__main__": unittest.main(verbosity=2)
