#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, tempfile, unittest
from pathlib import Path
SCRIPT_DIR=Path(__file__).resolve().parent; SRC_DIR=SCRIPT_DIR.parent/"src"
if str(SRC_DIR) not in sys.path: sys.path.insert(0,str(SRC_DIR))
from ra_ild_igh.release import CheckResult, build_check_plan, build_completion_payload, validate_shm_aggregation_fixture, write_completion_marker

class Batch06ReleaseTests(unittest.TestCase):
    def test_quick_and_full_plans_differ_only_at_nested_mode(self):
        root=Path('/tmp/repo'); config=root/'config.yaml'; baseline=root/'baseline.yaml'
        quick=build_check_plan(root,config,baseline,mode='quick',python_executable='python',allow_dirty_source=True)
        full=build_check_plan(root,config,baseline,mode='full',python_executable='python',allow_dirty_source=True)
        self.assertEqual(len(quick),18); self.assertEqual(len(full),18)
        self.assertIn('nested_cv_quick',[x.name for x in quick]); self.assertIn('nested_cv_full',[x.name for x in full])
        full_nested=next(x for x in full if x.name=='nested_cv_full'); self.assertIn('--full',full_nested.command); self.assertTrue(full_nested.expensive)
    def test_shm_two_nt_to_one_aa_contract(self):
        result=validate_shm_aggregation_fixture(); self.assertTrue(result['passed']); self.assertEqual(result['nt_clone_number'],2); self.assertEqual(result['read_count'],10); self.assertEqual(result['top_nt_cdr3'],'AAA')
    def test_failed_payload_cannot_write_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=CheckResult('x','unit',('false',),1,0.1,'','')
            payload=build_completion_payload(repository_root=Path(tmp),config_path=Path(tmp)/'c',baseline_config_path=Path(tmp)/'b',mode='full',results=[result],unit_test_count=1)
            with self.assertRaises(ValueError): write_completion_marker(Path(tmp)/'marker.json',payload)
    def test_completion_payload_does_not_select_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=CheckResult('x','unit',('true',),0,0.1,'','')
            payload=build_completion_payload(repository_root=Path(tmp),config_path=Path(tmp)/'c',baseline_config_path=Path(tmp)/'b',mode='full',results=[result],unit_test_count=1)
            self.assertEqual(payload['final_model']['status'],'not_selected'); self.assertFalse(payload['final_model']['automatic_final_model_selection']); self.assertTrue(payload['repeated_holdout']['results_are_not_an_independent_test'])
    def test_release_plan_contains_all_batch_regressions(self):
        plan=build_check_plan(Path('/tmp/r'),Path('/tmp/c'),Path('/tmp/b'),python_executable='python',allow_dirty_source=True)
        names={x.name for x in plan}
        for name in ('batch02_feature_management','batch03_repeated_holdout','batch04_training_bundle','batch05_aggregation','batch06_release_unit','historical_baseline','public_reference','preprocessing','modeling','specifications','outer_task_tree','repeated_holdout_aggregation','release_contract'):
            self.assertIn(name,names)

if __name__=='__main__': unittest.main(verbosity=2)
