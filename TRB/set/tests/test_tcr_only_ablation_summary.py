#!/usr/bin/env python3
"""Regression tests for TCR-only summary manifest nesting and clinical auditing."""
from __future__ import annotations
import importlib.util
import tempfile
import unittest
from pathlib import Path
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts_v2/summarize_tcr_only_ablation.py"
spec = importlib.util.spec_from_file_location("tcr_only_summary", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)

class TcrOnlySummaryAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        pd.DataFrame({"feature_name": ["physchem_mean", "physchem_std"]}).to_csv(
            self.out / "06_preprocessing_audit.csv.gz", index=False, compression="gzip"
        )
        self.groups = ["tcr_physicochemical_weighted"]
        self.manifest = {
            "resolved_experiment": {
                "phase5": {"features": {
                    "clinical_groups": [], "static_tcr_groups": list(self.groups)
                }},
                "unified_ml": {"algorithm": "elastic_net"},
            },
            "execution": {"static_tcr_predictor_count": 2},
        }

    def test_actual_unified_manifest_nesting_is_supported(self):
        summary.assert_no_clinical(self.out, self.manifest, self.groups)

    def test_detects_clinical_groups_in_manifest(self):
        self.manifest["resolved_experiment"]["phase5"]["features"]["clinical_groups"] = ["clinical_age_sex"]
        with self.assertRaisesRegex(ValueError, "Clinical groups remain"):
            summary.assert_no_clinical(self.out, self.manifest, self.groups)

    def test_detects_wrong_tcr_groups(self):
        self.manifest["resolved_experiment"]["phase5"]["features"]["static_tcr_groups"] = ["tcr_diversity"]
        with self.assertRaisesRegex(ValueError, "TCR group mismatch"):
            summary.assert_no_clinical(self.out, self.manifest, self.groups)

    def test_detects_zero_tcr_predictors(self):
        self.manifest["execution"]["static_tcr_predictor_count"] = 0
        with self.assertRaisesRegex(ValueError, "No TCR predictors"):
            summary.assert_no_clinical(self.out, self.manifest, self.groups)

    def test_detects_age_in_actual_preprocessing(self):
        pd.DataFrame({"feature_name": ["physchem_mean", "age"]}).to_csv(
            self.out / "06_preprocessing_audit.csv.gz", index=False, compression="gzip"
        )
        with self.assertRaisesRegex(ValueError, "Clinical predictors in model"):
            summary.assert_no_clinical(self.out, self.manifest, self.groups)

    def test_detects_sex_dummy_in_actual_preprocessing(self):
        pd.DataFrame({"feature_name": ["physchem_mean", "sex__M_vs_F"]}).to_csv(
            self.out / "06_preprocessing_audit.csv.gz", index=False, compression="gzip"
        )
        with self.assertRaisesRegex(ValueError, "Clinical predictors in model"):
            summary.assert_no_clinical(self.out, self.manifest, self.groups)

if __name__ == "__main__":
    unittest.main()
