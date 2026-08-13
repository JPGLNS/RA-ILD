#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SET_ROOT = HERE.parent
SRC_DIR = SET_ROOT / "src"
CONFIG_DIR = SET_ROOT / "configs" / "feature_framework"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.feature_framework import (  # noqa: E402
    FeatureFrameworkError,
    FeaturePlan,
    load_feature_plan,
    load_feature_registry,
)

REGISTRY = CONFIG_DIR / "trb_feature_registry_v1.yaml"
PLAN = CONFIG_DIR / "trb_feature_plan_template_v1.yaml"


class TestFeatureFramework(unittest.TestCase):
    def test_registry_loads_and_defines_sources(self):
        registry = load_feature_registry(REGISTRY)
        self.assertEqual(registry.receptor, "TRB")
        self.assertEqual(registry.default_repertoire_source, "top10000")
        self.assertEqual(set(registry.repertoire_sources), {"all", "top10000"})
        self.assertEqual(registry.repertoire_sources["top10000"].top_k, 10000)
        self.assertEqual(registry.repertoire_sources["top10000"].weight_column, "frequency_norm")

    def test_template_plan_resolves_top10000(self):
        registry = load_feature_registry(REGISTRY)
        plan = load_feature_plan(PLAN)
        resolved = registry.resolve_plan(plan)
        self.assertEqual(resolved.repertoire_source.id, "top10000")
        self.assertEqual(len(resolved.feature_groups), 9)
        self.assertTrue(resolved.requires_training_fit)
        self.assertFalse(resolved.warnings)

    def test_all_repertoire_can_be_selected_without_code_change(self):
        registry = load_feature_registry(REGISTRY)
        plan = FeaturePlan(
            id="all_diversity",
            receptor="TRB",
            repertoire_source="all",
            feature_groups=("tcr_diversity", "tcr_clonal_expansion"),
            allow_legacy=False,
            allow_planned=False,
        )
        resolved = registry.resolve_plan(plan)
        self.assertEqual(resolved.repertoire_source.id, "all")
        self.assertFalse(resolved.requires_training_fit)

    def test_unknown_group_is_rejected(self):
        registry = load_feature_registry(REGISTRY)
        plan = FeaturePlan(
            id="bad",
            receptor="TRB",
            repertoire_source="top10000",
            feature_groups=("not_a_group",),
            allow_legacy=False,
            allow_planned=False,
        )
        with self.assertRaises(FeatureFrameworkError):
            registry.resolve_plan(plan)

    def test_planned_enriched_dict_is_rejected_by_default(self):
        registry = load_feature_registry(REGISTRY)
        plan = FeaturePlan(
            id="planned",
            receptor="TRB",
            repertoire_source="top10000",
            feature_groups=("enriched_dict",),
            allow_legacy=False,
            allow_planned=False,
        )
        with self.assertRaisesRegex(FeatureFrameworkError, "planned"):
            registry.resolve_plan(plan)

    def test_planned_enriched_dict_can_be_contract_validated_explicitly(self):
        registry = load_feature_registry(REGISTRY)
        plan = FeaturePlan(
            id="planned",
            receptor="TRB",
            repertoire_source="top10000",
            feature_groups=("enriched_dict",),
            allow_legacy=False,
            allow_planned=True,
        )
        resolved = registry.resolve_plan(plan)
        self.assertTrue(resolved.requires_training_fit)
        self.assertIn("planned feature group selected: enriched_dict", resolved.warnings)

    def test_legacy_public_requires_explicit_opt_in(self):
        registry = load_feature_registry(REGISTRY)
        plan = FeaturePlan(
            id="legacy",
            receptor="TRB",
            repertoire_source="all",
            feature_groups=("public_legacy_reference",),
            allow_legacy=False,
            allow_planned=False,
        )
        with self.assertRaisesRegex(FeatureFrameworkError, "legacy_optional"):
            registry.resolve_plan(plan)
        allowed = FeaturePlan(
            id="legacy",
            receptor="TRB",
            repertoire_source="all",
            feature_groups=("public_legacy_reference",),
            allow_legacy=True,
            allow_planned=False,
        )
        resolved = registry.resolve_plan(allowed)
        self.assertIn("legacy feature group selected: public_legacy_reference", resolved.warnings)

    def test_legacy_public_is_not_claimed_for_top10000(self):
        registry = load_feature_registry(REGISTRY)
        plan = FeaturePlan(
            id="legacy_top",
            receptor="TRB",
            repertoire_source="top10000",
            feature_groups=("public_legacy_reference",),
            allow_legacy=True,
            allow_planned=False,
        )
        with self.assertRaisesRegex(FeatureFrameworkError, "does not support repertoire source"):
            registry.resolve_plan(plan)

    def test_clinical_age_sex_and_material_are_independently_selectable(self):
        registry = load_feature_registry(REGISTRY)
        classified, unmatched = registry.classify_columns(["age", "sex", "material"])
        self.assertFalse(unmatched)
        self.assertEqual(classified["clinical_age_sex"], ["age", "sex"])
        self.assertEqual(classified["clinical_material"], ["material"])
        self.assertEqual(registry.feature_groups["clinical_age_sex"].model_role, "primary_predictor")
        self.assertEqual(registry.feature_groups["clinical_material"].model_role, "optional_predictor")

    def test_existing_step02_columns_classify_deterministically(self):
        registry = load_feature_registry(REGISTRY)
        columns = [
            "Shannon",
            "top10_cumulative_frequency",
            "weighted_length_12_freq",
            "A_frequency",
            "weighted_A_frequency",
            "mean_hydrophobicity",
            "weighted_hydrophobicity",
            "unweighted_3mer_ASS",
            "weighted_3mer_ASS",
        ]
        classified, unmatched = registry.classify_columns(columns)
        self.assertFalse(unmatched)
        self.assertEqual(classified["tcr_diversity"], ["Shannon"])
        self.assertEqual(classified["tcr_clonal_expansion"], ["top10_cumulative_frequency"])
        self.assertEqual(classified["tcr_aa_length"], ["weighted_length_12_freq"])
        self.assertEqual(classified["tcr_aa_composition_unweighted"], ["A_frequency"])
        self.assertEqual(classified["tcr_aa_composition_weighted"], ["weighted_A_frequency"])
        self.assertEqual(classified["tcr_physicochemical_unweighted"], ["mean_hydrophobicity"])
        self.assertEqual(classified["tcr_physicochemical_weighted"], ["weighted_hydrophobicity"])
        self.assertEqual(classified["tcr_3mer_unweighted"], ["unweighted_3mer_ASS"])
        self.assertEqual(classified["tcr_3mer_weighted"], ["weighted_3mer_ASS"])

    def test_reference_drop_contract_matches_legacy_five_columns(self):
        registry = load_feature_registry(REGISTRY)
        observed = set()
        for group in registry.feature_groups.values():
            observed.update(group.reference_drop_columns)
        self.assertEqual(
            observed,
            {
                "Y_frequency",
                "weighted_Y_frequency",
                "weighted_length_other_freq",
                "nonpolar_aa_ratio",
                "weighted_nonpolar_aa_ratio",
            },
        )

    def test_duplicate_feature_groups_in_plan_are_rejected(self):
        raw = {
            "schema_version": "1.0",
            "plan": {
                "id": "dup",
                "receptor": "TRB",
                "repertoire_source": "all",
                "feature_groups": ["tcr_diversity", "tcr_diversity"],
                "allow_legacy": False,
                "allow_planned": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.yaml"
            path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(FeatureFrameworkError, "duplicates"):
                load_feature_plan(path)


    def test_legacy_step02_core_contract_is_96_columns_and_83_candidates(self):
        registry = load_feature_registry(REGISTRY)
        aa = list("ACDEFGHIKLMNPQRSTVWY")
        legacy_core = [
            "total_reads_all_nt", "total_reads_valid_aa", "valid_read_ratio",
            "nt_clone_number_all", "valid_nt_clone_number", "valid_nt_row_ratio",
            "aa_clone_number", "log1p_aa_clone_number",
            "Shannon", "Simpson", "inverse_Simpson", "Gini", "clonality",
            "top1_frequency", "top5_cumulative_frequency", "top10_cumulative_frequency",
            "top20_cumulative_frequency", "top50_cumulative_frequency",
            "mean_aa_length", "median_aa_length", "sd_aa_length",
            "weighted_mean_aa_length", "weighted_sd_aa_length",
            *[f"weighted_length_{i}_freq" for i in range(8, 26)],
            "weighted_length_other_freq",
            *[f"{x}_frequency" for x in aa],
            *[f"weighted_{x}_frequency" for x in aa],
            "mean_hydrophobicity", "weighted_hydrophobicity",
            "mean_charge", "weighted_charge",
            "aromaticity", "weighted_aromaticity",
            "basic_aa_ratio", "weighted_basic_aa_ratio",
            "acidic_aa_ratio", "weighted_acidic_aa_ratio",
            "polar_aa_ratio", "weighted_polar_aa_ratio",
            "nonpolar_aa_ratio", "weighted_nonpolar_aa_ratio",
        ]
        self.assertEqual(len(legacy_core), 96)
        classified, unmatched = registry.classify_columns(legacy_core)
        self.assertFalse(unmatched)
        self.assertEqual(sum(len(x) for x in classified.values()), 96)

        candidate_count = 0
        for group_id, columns in classified.items():
            group = registry.feature_groups[group_id]
            if group.model_role != "candidate_predictor":
                continue
            drops = set(group.reference_drop_columns)
            candidate_count += sum(column not in drops for column in columns)
        self.assertEqual(candidate_count, 83)

    def test_training_derived_groups_are_explicit(self):
        registry = load_feature_registry(REGISTRY)
        for group_id in (
            "tcr_3mer_unweighted",
            "tcr_3mer_weighted",
            "public_legacy_reference",
            "enriched_dict",
        ):
            self.assertEqual(registry.feature_groups[group_id].fit_scope, "training_only")


if __name__ == "__main__":
    unittest.main(verbosity=2)
