#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for the frozen TRB V3 179 model hierarchy contract."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ra_ild_trb.model_contract import (  # noqa: E402
    MODEL_SPECS,
    ModelContractError,
    REPEAT_3MER_EXPECTED,
    validate_hierarchical_model_contract,
)

PUBLIC = list(REPEAT_3MER_EXPECTED.get("public_features", (
    "all_ref_public_clone_ratio", "all_ref_public_frequency_sum",
    "shared_ref_clone_ratio", "shared_ref_frequency_sum",
    "RA_specific_ref_clone_ratio", "RA_specific_ref_frequency_sum",
    "ILD_specific_ref_clone_ratio", "ILD_specific_ref_frequency_sum",
)))


def options():
    return {
        **REPEAT_3MER_EXPECTED,
        "top_k_values": list(REPEAT_3MER_EXPECTED["top_k_values"]),
        "aa_manifest": "TRB/set/training_bundles/test/aa_clone_table_manifest.csv",
    }


def models(cohort="total"):
    names = tuple(MODEL_SPECS) if cohort == "total" else tuple(MODEL_SPECS)[:3]
    result = {}
    for name in names:
        spec = MODEL_SPECS[name]
        result[name] = {
            "numeric": list(spec["numeric"]),
            "categorical": list(spec["categorical"]),
            "static_feature_groups": list(spec["static_feature_groups"]),
            "dynamic_public": list(spec["dynamic_public"]),
        }
    return result


class ModelContractTests(unittest.TestCase):
    def test_total_predictor_counts(self):
        result = validate_hierarchical_model_contract(
            models(), cohort="total", repeat_3mer_options=options()
        )
        self.assertEqual(result["M0_clinical"], {"numeric": 1, "categorical": 1})
        self.assertEqual(result["M1_tcr"], {"numeric": 1084, "categorical": 1})
        self.assertEqual(result["M2_tcr_clinical"], {"numeric": 1092, "categorical": 1})
        self.assertEqual(result["M3_tcr_clinical_material"], {"numeric": 1092, "categorical": 2})

    def test_pbmc_and_buffycoat_exclude_material_model(self):
        for cohort in ("pbmc", "buffycoat"):
            result = validate_hierarchical_model_contract(
                models(cohort), cohort=cohort, repeat_3mer_options=options()
            )
            self.assertEqual(tuple(result), tuple(MODEL_SPECS)[:3])

    def test_rejects_material_model_in_pbmc(self):
        with self.assertRaises(ModelContractError):
            validate_hierarchical_model_contract(
                models(), cohort="pbmc", repeat_3mer_options=options()
            )

    def test_rejects_public_features_in_m1(self):
        value = models()
        value["M1_tcr"]["dynamic_public"] = PUBLIC
        with self.assertRaises(ModelContractError):
            validate_hierarchical_model_contract(
                value, cohort="total", repeat_3mer_options=options()
            )

    def test_rejects_missing_public_feature(self):
        value = models()
        value["M2_tcr_clinical"]["dynamic_public"].pop()
        with self.assertRaises(ModelContractError):
            validate_hierarchical_model_contract(
                value, cohort="total", repeat_3mer_options=options()
            )

    def test_rejects_material_in_m2(self):
        value = models()
        value["M2_tcr_clinical"]["categorical"].append("material")
        with self.assertRaises(ModelContractError):
            validate_hierarchical_model_contract(
                value, cohort="total", repeat_3mer_options=options()
            )

    def test_rejects_inner_fold_vocabulary_scope(self):
        value = options()
        value["selection_scope"] = "inner_fold_train_only"
        with self.assertRaises(ModelContractError):
            validate_hierarchical_model_contract(
                models(), cohort="total", repeat_3mer_options=value
            )

    def test_rejects_wrong_top_k_or_threshold(self):
        for key, bad in (("max_kmers", 400), ("min_sample_count", 4), ("min_prevalence", 0.1)):
            value = options()
            value[key] = bad
            with self.subTest(key=key), self.assertRaises(ModelContractError):
                validate_hierarchical_model_contract(
                    models(), cohort="total", repeat_3mer_options=value
                )

    def test_rejects_predictor_count_drift(self):
        counts = {
            "M0_clinical": {"numeric": 1, "categorical": 1},
            "M1_tcr": {"numeric": 1083, "categorical": 1},
            "M2_tcr_clinical": {"numeric": 1092, "categorical": 1},
            "M3_tcr_clinical_material": {"numeric": 1092, "categorical": 2},
        }
        with self.assertRaises(ModelContractError):
            validate_hierarchical_model_contract(
                models(), cohort="total", repeat_3mer_options=options(),
                expected_resolved_columns=counts,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
