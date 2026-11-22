#!/usr/bin/env python3
"""No-training static contracts for the three TCR-only experiments."""
from __future__ import annotations
import importlib.util
import unittest
from pathlib import Path
import yaml
AUDIT = Path(__file__).resolve().parents[1] / "scripts_v2/audit_tcr_only_ablation.py"
class TcrOnlyAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location("tcr_only_audit",AUDIT)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.audit=staticmethod(module.audit)
        cls.root=module.ROOT
    def test_source_configuration_delta_is_only_age_sex(self):
        self.assertEqual(len(self.audit(self.root)),3)
    def test_no_clinical_groups(self):
        for cohort in ("total","pbmc","buffycoat"):
            path=self.root/f"TRB/set/configs/layered_feature_experiments/tcr_only_ablation/configs/{cohort}.yaml"
            cfg=yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(cfg["features"]["clinical_groups"],[])
            self.assertTrue(cfg["features"]["static_tcr_groups"])
if __name__=="__main__":
    unittest.main()
