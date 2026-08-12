#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

TEST_FILE = Path(__file__).resolve()
SET_DIR = TEST_FILE.parents[1]
SRC_DIR = SET_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.feature_framework import load_feature_plan, load_feature_registry
from ra_ild_trb.feature_matrix import (
    FeatureMatrixError,
    assemble_feature_matrix,
    expected_output_paths,
    validate_reference_manifest,
    write_feature_matrix_result,
)
from ra_ild_trb.feature_modules import (
    STATIC_GROUP_IDS,
    build_feature_manifest,
    write_modules,
    ModuleBuildResult,
)

REGISTRY_PATH = SET_DIR / "configs/feature_framework/trb_feature_registry_v1.yaml"
PLAN_PATH = SET_DIR / "configs/feature_framework/trb_feature_plan_top10000_primary_weighted_v1.yaml"


class TestFeatureMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_feature_registry(REGISTRY_PATH)
        cls.plan = load_feature_plan(PLAN_PATH)
        cls.resolved = cls.registry.resolve_plan(cls.plan)

    def make_metadata(self, ids=("S1", "S2")):
        return pd.DataFrame(
            {
                "libraryid": list(ids),
                "cohort": ["RA", "ILD"],
                "patient": ["P1", "P2"],
                "age": [52, 67],
                "sex": ["female", "male"],
                "material": ["PBMC", "buffercoat"],
                "batch": ["B1", "B2"],
            }
        )

    def make_modules(self, root: Path, ids=("S1", "S2"), kmer_count=3):
        root.mkdir(parents=True, exist_ok=True)
        modules = {}
        for gid in STATIC_GROUP_IDS:
            cols = self.registry.feature_groups[gid].selector.exact
            data = {"sample_id": list(ids)}
            for i, col in enumerate(cols, start=1):
                data[col] = [i / 1000.0, i / 1000.0 + 0.001]
            modules[gid] = pd.DataFrame(data)

        for gid, prefix in [
            ("tcr_3mer_unweighted", "unweighted_3mer_"),
            ("tcr_3mer_weighted", "weighted_3mer_"),
        ]:
            data = {"sample_id": list(ids)}
            for i in range(kmer_count):
                kmer = ["AAA", "AAC", "AAG", "AAT", "ACA"][i]
                data[f"{prefix}{kmer}"] = [0.01 * (i + 1), 0.02 * (i + 1)]
            modules[gid] = pd.DataFrame(data)

        selected = tuple(group.id for group in self.resolved.feature_groups if group.id.startswith("tcr_"))
        manifest = build_feature_manifest(
            self.registry,
            "top10000",
            "phase2_fixture",
            selected,
            modules,
        )
        result = ModuleBuildResult(
            repertoire_source="top10000",
            plan_id="phase2_fixture",
            selected_feature_groups=selected,
            modules=modules,
            manifest=manifest,
            core_feature_count=96,
            static_candidate_count=83,
            sample_count=len(ids),
        )
        write_modules(result, root.parent, root, dry_run=False)

    def assemble_fixture(self, root: Path, *, keep_refs=False, reference=None, ids=("S1", "S2")):
        root.mkdir(parents=True, exist_ok=True)
        metadata_path = root / "metadata.csv"
        module_dir = root / "modules"
        self.make_metadata(ids).to_csv(metadata_path, index=False)
        self.make_modules(module_dir, ids=ids)
        return assemble_feature_matrix(
            metadata_path=metadata_path,
            module_dir=module_dir,
            registry=self.registry,
            resolved_plan=self.resolved,
            id_col="libraryid",
            context_columns=("cohort", "patient", "age", "sex", "material", "batch"),
            keep_reference_columns=keep_refs,
            reference_manifest_path=reference,
        )

    def test_primary_weighted_plan_resolves(self):
        self.assertEqual(self.resolved.repertoire_source.id, "top10000")
        self.assertIn("clinical_age_sex", [g.id for g in self.resolved.feature_groups])
        self.assertIn("tcr_3mer_weighted", [g.id for g in self.resolved.feature_groups])
        self.assertNotIn("tcr_3mer_unweighted", [g.id for g in self.resolved.feature_groups])

    def test_default_matrix_has_83_static_plus_clinical_plus_weighted_kmer(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.assemble_fixture(Path(tmp))
            # 83 non-QC static + 2 clinical + 3 weighted kmers = 88 predictors.
            self.assertEqual(result.feature_count, 88)
            self.assertEqual(result.matrix.shape, (2, 89))
            self.assertEqual(len(result.dropped_reference_columns), 5)

    def test_unweighted_kmer_is_not_in_primary_weighted_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.assemble_fixture(Path(tmp))
            self.assertFalse(any(c.startswith("unweighted_3mer_") for c in result.matrix.columns))
            self.assertTrue(any(c.startswith("weighted_3mer_") for c in result.matrix.columns))

    def test_outcome_is_separate_from_predictor_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.assemble_fixture(Path(tmp))
            self.assertNotIn("cohort", result.matrix.columns)
            self.assertIn("cohort", result.context.columns)
            self.assertIn("age", result.matrix.columns)
            self.assertIn("sex", result.matrix.columns)

    def test_metadata_types_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.assemble_fixture(Path(tmp))
            manifest = result.manifest.set_index("feature_name")
            self.assertEqual(manifest.loc["age", "data_type"], "numeric")
            self.assertEqual(manifest.loc["sex", "data_type"], "categorical")

    def test_keep_reference_columns_adds_five_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default = self.assemble_fixture(root / "a")
            kept = self.assemble_fixture(root / "b", keep_refs=True)
            self.assertEqual(kept.feature_count, default.feature_count + 5)
            self.assertEqual(len(kept.dropped_reference_columns), 0)

    def test_module_sample_coverage_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata.csv"
            module_dir = root / "modules"
            self.make_metadata().to_csv(metadata, index=False)
            self.make_modules(module_dir, ids=("S1", "S3"))
            with self.assertRaisesRegex(FeatureMatrixError, "sample coverage mismatch"):
                assemble_feature_matrix(
                    metadata_path=metadata,
                    module_dir=module_dir,
                    registry=self.registry,
                    resolved_plan=self.resolved,
                )

    def test_repertoire_source_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata.csv"
            module_dir = root / "modules"
            self.make_metadata().to_csv(metadata, index=False)
            self.make_modules(module_dir)
            manifest_path = module_dir / "02_feature_module_manifest.csv"
            manifest = pd.read_csv(manifest_path)
            manifest["repertoire_source"] = "all"
            manifest.to_csv(manifest_path, index=False)
            with self.assertRaisesRegex(FeatureMatrixError, "repertoire_source mismatch"):
                assemble_feature_matrix(
                    metadata_path=metadata,
                    module_dir=module_dir,
                    registry=self.registry,
                    resolved_plan=self.resolved,
                )

    def test_reference_manifest_schema_lock_passes_and_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self.assemble_fixture(root / "train")
            ref = root / "train_manifest.csv"
            train.manifest.to_csv(ref, index=False)
            test = self.assemble_fixture(root / "test", reference=ref)
            self.assertEqual(test.manifest["feature_name"].tolist(), train.manifest["feature_name"].tolist())

            bad = pd.read_csv(ref)
            bad.loc[0, "feature_name"] = "DRIFTED"
            bad.to_csv(ref, index=False)
            with self.assertRaisesRegex(FeatureMatrixError, "schema mismatch"):
                validate_reference_manifest(train.manifest, ref)

    def test_matrix_follows_metadata_sample_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata.csv"
            module_dir = root / "modules"
            self.make_metadata(ids=("S2", "S1")).to_csv(metadata, index=False)
            self.make_modules(module_dir, ids=("S1", "S2"))
            result = assemble_feature_matrix(
                metadata_path=metadata,
                module_dir=module_dir,
                registry=self.registry,
                resolved_plan=self.resolved,
            )
            self.assertEqual(result.matrix["sample_id"].tolist(), ["S2", "S1"])

    def test_dry_run_writes_nothing_formal_write_has_six_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "fixture"
            result = self.assemble_fixture(fixture)
            out = root / "out"
            outputs = write_feature_matrix_result(
                result,
                self.resolved,
                out,
                id_col="libraryid",
                context_columns=("cohort", "patient", "age", "sex", "material", "batch"),
                keep_reference_columns=False,
                dry_run=True,
            )
            self.assertFalse(out.exists())
            self.assertEqual(len(outputs), 6)

            outputs = write_feature_matrix_result(
                result,
                self.resolved,
                out,
                id_col="libraryid",
                context_columns=("cohort", "patient", "age", "sex", "material", "batch"),
                keep_reference_columns=False,
            )
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            self.assertEqual(len(list(out.iterdir())), 6)

    def test_manifest_matches_matrix_columns_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.assemble_fixture(Path(tmp))
            self.assertEqual(result.manifest["feature_name"].tolist(), list(result.matrix.columns[1:]))
            self.assertEqual(result.manifest["matrix_order"].tolist(), list(range(1, result.feature_count + 1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
