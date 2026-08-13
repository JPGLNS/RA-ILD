#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_FILE = Path(__file__).resolve()
SET_DIR = TEST_FILE.parents[1]
SRC_DIR = SET_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.feature_framework import load_feature_registry
from ra_ild_trb.feature_modules import (
    ALL_MODULE_IDS,
    STATIC_GROUP_IDS,
    FeatureModuleError,
    build_modules_from_step02,
    module_summary_table,
    split_core_dataframe,
    validate_kmer_dataframe,
    write_modules,
)

REGISTRY_PATH = SET_DIR / "configs" / "feature_framework" / "trb_feature_registry_v1.yaml"


class TestFeatureModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_feature_registry(REGISTRY_PATH)

    def make_core(self, sample_ids=("S1", "S2")):
        columns = []
        for group_id in STATIC_GROUP_IDS:
            columns.extend(self.registry.feature_groups[group_id].selector.exact)
        data = {"sample_id": list(sample_ids)}
        for index, column in enumerate(columns, start=1):
            data[column] = [index / 1000.0, index / 1000.0 + 0.001]
        return pd.DataFrame(data)

    def make_kmer(self, prefix, sample_ids=("S1", "S2")):
        return pd.DataFrame(
            {
                "sample_id": list(sample_ids),
                f"{prefix}AAA": [0.1, 0.2],
                f"{prefix}AAC": [0.3, 0.4],
                f"{prefix}AAG": [0.5, 0.6],
            }
        )

    def write_step02_fixture(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.make_core().to_csv(root / "02_core_sample_features.csv", index=False)
        self.make_kmer("unweighted_3mer_").to_csv(
            root / "02_3mer_unweighted_features.csv", index=False
        )
        self.make_kmer("weighted_3mer_").to_csv(
            root / "02_3mer_weighted_features.csv", index=False
        )
        payload = {
            "plan": {
                "plan_id": "trb_feature_plan_top10000_exploratory_v1",
                "repertoire_source": {"id": "top10000"},
                "feature_groups": [
                    {"id": "tcr_diversity"},
                    {"id": "tcr_clonal_expansion"},
                    {"id": "tcr_aa_length"},
                    {"id": "tcr_aa_composition_unweighted"},
                    {"id": "tcr_aa_composition_weighted"},
                    {"id": "tcr_physicochemical_unweighted"},
                    {"id": "tcr_physicochemical_weighted"},
                    {"id": "tcr_3mer_unweighted"},
                    {"id": "tcr_3mer_weighted"},
                ],
            }
        }
        (root / "02_resolved_feature_source.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_static_module_counts_sum_to_96(self):
        core = self.make_core()
        modules = split_core_dataframe(core, self.registry)
        counts = {gid: modules[gid].shape[1] - 1 for gid in STATIC_GROUP_IDS}
        self.assertEqual(
            counts,
            {
                "tcr_qc_depth": 8,
                "tcr_diversity": 5,
                "tcr_clonal_expansion": 5,
                "tcr_aa_length": 24,
                "tcr_aa_composition_unweighted": 20,
                "tcr_aa_composition_weighted": 20,
                "tcr_physicochemical_unweighted": 7,
                "tcr_physicochemical_weighted": 7,
            },
        )
        self.assertEqual(sum(counts.values()), 96)

    def test_core_split_is_exactly_reversible(self):
        core = self.make_core()
        modules = split_core_dataframe(core, self.registry)
        owner = {}
        for gid in STATIC_GROUP_IDS:
            for col in modules[gid].columns[1:]:
                owner[col] = modules[gid][col]
        rebuilt = pd.DataFrame({"sample_id": core["sample_id"]})
        for col in core.columns[1:]:
            rebuilt[col] = owner[col]
        self.assertTrue(rebuilt.equals(core))

    def test_extra_core_column_is_rejected(self):
        core = self.make_core()
        core["unexpected_feature"] = 1.0
        with self.assertRaisesRegex(FeatureModuleError, "extra=.*unexpected_feature"):
            split_core_dataframe(core, self.registry)

    def test_duplicate_sample_id_is_rejected(self):
        core = self.make_core(sample_ids=("S1", "S1"))
        with self.assertRaisesRegex(FeatureModuleError, "duplicate sample_id"):
            split_core_dataframe(core, self.registry)

    def test_kmer_prefix_contract_is_strict(self):
        bad = self.make_kmer("unweighted_3mer_")
        bad["wrong_prefix_AAA"] = [0.0, 0.0]
        with self.assertRaisesRegex(FeatureModuleError, "outside prefix"):
            validate_kmer_dataframe(
                bad, self.registry, "tcr_3mer_unweighted", ["S1", "S2"]
            )

    def test_kmer_sample_order_must_match_core(self):
        bad = self.make_kmer("weighted_3mer_", sample_ids=("S2", "S1"))
        with self.assertRaisesRegex(FeatureModuleError, "sample order"):
            validate_kmer_dataframe(
                bad, self.registry, "tcr_3mer_weighted", ["S1", "S2"]
            )

    def test_full_build_has_ten_modules_and_83_static_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            step02 = Path(tmp) / "step02"
            self.write_step02_fixture(step02)
            result = build_modules_from_step02(step02, self.registry)
            self.assertEqual(set(result.modules), set(ALL_MODULE_IDS))
            self.assertEqual(result.core_feature_count, 96)
            self.assertEqual(result.static_candidate_count, 83)
            self.assertEqual(result.sample_count, 2)
            self.assertEqual(result.repertoire_source, "top10000")

    def test_manifest_records_reference_drops_and_plan_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            step02 = Path(tmp) / "step02"
            self.write_step02_fixture(step02)
            result = build_modules_from_step02(step02, self.registry)
            manifest = result.manifest
            self.assertEqual(len(manifest), 96 + 3 + 3)
            refs = set(manifest.loc[manifest["reference_drop"], "feature_name"])
            self.assertEqual(
                refs,
                {
                    "weighted_length_other_freq",
                    "Y_frequency",
                    "weighted_Y_frequency",
                    "nonpolar_aa_ratio",
                    "weighted_nonpolar_aa_ratio",
                },
            )
            qc_selected = manifest.loc[
                manifest["feature_group"] == "tcr_qc_depth", "selected_in_phase2_plan"
            ]
            self.assertFalse(qc_selected.astype(bool).any())
            div_selected = manifest.loc[
                manifest["feature_group"] == "tcr_diversity", "selected_in_phase2_plan"
            ]
            self.assertTrue(div_selected.astype(bool).all())

    def test_module_summary_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            step02 = Path(tmp) / "step02"
            self.write_step02_fixture(step02)
            result = build_modules_from_step02(step02, self.registry)
            summary = module_summary_table(result).set_index("feature_group")
            self.assertEqual(int(summary.loc["tcr_qc_depth", "feature_count"]), 8)
            self.assertEqual(int(summary.loc["tcr_aa_length", "reference_drop_count"]), 1)
            self.assertEqual(int(summary.loc["tcr_3mer_weighted", "feature_count"]), 3)
            self.assertEqual(summary.loc["tcr_3mer_weighted", "fit_scope"], "training_only")

    def test_dry_run_writes_nothing_then_formal_write_has_12_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step02 = root / "step02"
            out = root / "modules"
            self.write_step02_fixture(step02)
            result = build_modules_from_step02(step02, self.registry)
            outputs = write_modules(result, step02, out, dry_run=True)
            self.assertFalse(out.exists())
            self.assertEqual(len(outputs), 12)

            outputs = write_modules(result, step02, out, dry_run=False)
            self.assertTrue(out.is_dir())
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            self.assertEqual(len(list(out.iterdir())), 12)

    def test_written_module_values_are_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step02 = root / "step02"
            out = root / "modules"
            self.write_step02_fixture(step02)
            original_core = pd.read_csv(step02 / "02_core_sample_features.csv")
            original_w = pd.read_csv(step02 / "02_3mer_weighted_features.csv")
            result = build_modules_from_step02(step02, self.registry)
            write_modules(result, step02, out)

            written_w = pd.read_csv(out / "tcr_3mer_weighted.csv")
            self.assertTrue(written_w.equals(original_w))

            module_frames = {
                gid: pd.read_csv(out / f"{gid}.csv") for gid in STATIC_GROUP_IDS
            }
            by_feature = {
                col: module_frames[gid][col]
                for gid in STATIC_GROUP_IDS
                for col in module_frames[gid].columns[1:]
            }
            rebuilt = pd.DataFrame({"sample_id": original_core["sample_id"]})
            for col in original_core.columns[1:]:
                rebuilt[col] = by_feature[col]
            np.testing.assert_allclose(
                rebuilt.iloc[:, 1:].to_numpy(dtype=float),
                original_core.iloc[:, 1:].to_numpy(dtype=float),
                rtol=0,
                atol=0,
            )
            self.assertEqual(rebuilt["sample_id"].tolist(), original_core["sample_id"].tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
