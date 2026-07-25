#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Batch 04 repeated-holdout training preparation."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.repeated_holdout_training import (  # noqa: E402
    RepeatedHoldoutTrainingError,
    build_full_public_sparse_cache,
    discover_aa_clone_tables,
    generate_scheme_yaml,
    generate_v2_assignments,
    load_frozen_assignments,
    load_training_bundle_spec,
    prepare_full_static_matrix,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Batch04RepeatedHoldoutTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / ".gitignore").write_text("# synthetic repo\n", encoding="utf-8")
        for relative in (
            "TRB/set/train/result/04_final_feature_matrix",
            "TRB/set/test/result/04_final_feature_matrix",
            "TRB/set/train/result/01_AA_clone_table",
            "TRB/set/test/result/01_AA_clone_table",
            "TRB/set/split_sets/test_repeat2",
            "TRB/set/configs/repeated_holdout_training",
            "TRB/set/configs/schemes",
        ):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        self.sample_ids = [f"S{i:02d}" for i in range(30)]
        rows = []
        for index, sample_id in enumerate(self.sample_ids):
            rows.append(
                {
                    "sample_id": sample_id,
                    "cohort": "RA" if index < 15 else "ILD",
                    "batch": "A" if index % 2 == 0 else "B",
                    "material": "PBMC" if index % 3 else "buffercoat",
                    "sex": "female" if index % 5 else "male",
                    "age": 35 + index,
                    "aa_clone_number": 2,
                    "tcr_feature_1": float(index),
                    "tcr_feature_2": float(index % 4),
                    "all_ref_public_frequency_sum": 999.0,
                }
            )
        full = pd.DataFrame(rows)
        full.iloc[:21].to_csv(
            self.root / "TRB/set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv",
            index=False,
        )
        full.iloc[21:].to_csv(
            self.root / "TRB/set/test/result/04_final_feature_matrix/04_test_final_feature_matrix.csv",
            index=False,
        )
        pd.DataFrame(
            {
                "feature_name": ["tcr_feature_1", "tcr_feature_2", "ignored"],
                "feature_group": ["tcr_test", "tcr_test", "qc"],
                "feature_role": ["candidate_predictor", "candidate_predictor", "audit"],
                "present_in_train_base": [True, True, True],
                "reference_drop": [False, False, False],
            }
        ).to_csv(
            self.root / "TRB/set/train/result/04_final_feature_matrix/04_feature_manifest.csv",
            index=False,
        )
        for index, sample_id in enumerate(self.sample_ids):
            directory = (
                self.root / "TRB/set/train/result/01_AA_clone_table"
                if index < 21
                else self.root / "TRB/set/test/result/01_AA_clone_table"
            )
            pd.DataFrame(
                {
                    "sample_id": [sample_id, sample_id],
                    "cdr3_aa": [f"CASS{index}A", "CASSCOMMON"],
                    "frequency_norm": [0.6, 0.4],
                }
            ).to_csv(directory / f"{sample_id}_AA_clone_table.csv", index=False)

        assignment_rows = []
        for repeat in (1, 2):
            holdout = set(self.sample_ids[(repeat - 1) * 9 : repeat * 9])
            for index, sample_id in enumerate(self.sample_ids):
                assignment_rows.append(
                    {
                        "split_set_id": "test_repeat2",
                        "split_id": f"split_{repeat:02d}",
                        "repeat_index": repeat,
                        "role": "holdout" if sample_id in holdout else "train",
                        "candidate_seed": 100 + repeat,
                        "sample_id": sample_id,
                        "patient_id": f"P{index:02d}",
                        "cohort": rows[index]["cohort"],
                        "batch": rows[index]["batch"],
                        "material": rows[index]["material"],
                        "sex": rows[index]["sex"],
                        "age": rows[index]["age"],
                    }
                )
        self.assignments_path = self.root / "TRB/set/split_sets/test_repeat2/repeated_holdout_assignments.csv"
        pd.DataFrame(assignment_rows).to_csv(self.assignments_path, index=False)
        marker = {
            "status": "FROZEN",
            "split_set_id": "test_repeat2",
            "files": {"assignments": {"sha256": sha256(self.assignments_path)}},
        }
        (self.root / "TRB/set/split_sets/test_repeat2/SPLITS_FROZEN.json").write_text(
            json.dumps(marker), encoding="utf-8"
        )
        template = {
            "scheme_version": "1.0",
            "scheme": {
                "id": "template_scheme",
                "description": "template",
                "status": "development",
                "base_config": "TRB/set/configs/trb_baseline_m2_v1.yaml",
                "output_root": "TRB/set/experiments/template_scheme",
            },
            "replacements": {
                "models": {
                    "M1": {
                        "description": "static",
                        "numeric": [],
                        "categorical": [],
                        "static_feature_groups": ["static_tcr_candidate_predictors"],
                        "dynamic_public": [],
                    }
                }
            },
            "overrides": {"final_model": {"status": "not_selected"}},
        }
        (self.root / "TRB/set/configs/schemes/template.yaml").write_text(
            yaml.safe_dump(template, sort_keys=False), encoding="utf-8"
        )
        self.spec_path = self.root / "TRB/set/configs/repeated_holdout_training/test.yaml"
        self.spec_path.write_text(
            yaml.safe_dump(
                {
                    "training_bundle_version": "1.0",
                    "bundle": {
                        "id": "test_bundle",
                        "output_root": "TRB/set/training_bundles/test_bundle",
                    },
                    "split_set": {
                        "assignments": "TRB/set/split_sets/test_repeat2/repeated_holdout_assignments.csv",
                        "frozen_marker": "TRB/set/split_sets/test_repeat2/SPLITS_FROZEN.json",
                        "train_role": "train",
                        "holdout_role": "holdout",
                    },
                    "source": {
                        "train_base_matrix": "TRB/set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv",
                        "test_final_matrix": "TRB/set/test/result/04_final_feature_matrix/04_test_final_feature_matrix.csv",
                        "feature_manifest": "TRB/set/train/result/04_final_feature_matrix/04_feature_manifest.csv",
                        "train_aa_clone_table_dir": "TRB/set/train/result/01_AA_clone_table",
                        "test_aa_clone_table_dir": "TRB/set/test/result/01_AA_clone_table",
                        "sample_id_column": "sample_id",
                        "label_column": "cohort",
                        "required_metadata_columns": [
                            "cohort", "batch", "material", "sex", "age", "aa_clone_number"
                        ],
                    },
                    "aa_tables": {
                        "sequence_column": "cdr3_aa",
                        "frequency_column_candidates": ["frequency_norm"],
                        "file_globs": ["**/*.csv"],
                    },
                    "inner_cv": {
                        "folds": 3,
                        "exact_strata": ["cohort"],
                        "balance_categorical": ["batch", "material", "sex"],
                        "balance_numeric": ["age"],
                        "candidates": 20,
                        "base_seed": 2026,
                    },
                    "scheme": {
                        "template": "TRB/set/configs/schemes/template.yaml",
                        "generated_scheme": "TRB/set/configs/schemes/generated/generated.yaml",
                        "id": "generated_scheme",
                        "output_root": "TRB/set/experiments/generated_scheme",
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        self.spec = load_training_bundle_spec(self.spec_path, repository_root=self.root)
        self.assignments, _ = load_frozen_assignments(self.spec)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_full_static_matrix_preserves_frozen_order_and_excludes_old_public(self) -> None:
        combined, metadata, static = prepare_full_static_matrix(self.spec, self.assignments)
        self.assertEqual(combined["sample_id"].tolist(), self.sample_ids)
        self.assertEqual(static, ("tcr_feature_1", "tcr_feature_2"))
        self.assertNotIn("all_ref_public_frequency_sum", combined.columns)
        self.assertEqual(len(metadata), 30)

    def test_one_aa_table_is_discovered_per_sample(self) -> None:
        manifest = discover_aa_clone_tables(self.spec, self.sample_ids)
        self.assertEqual(len(manifest), 30)
        self.assertEqual(manifest["sample_id"].tolist(), self.sample_ids)

    def test_sparse_cache_shape_and_frequency_sums(self) -> None:
        self.spec.output_root.mkdir(parents=True)
        manifest = discover_aa_clone_tables(self.spec, self.sample_ids)
        metadata = build_full_public_sparse_cache(self.spec, self.sample_ids, manifest)
        self.assertEqual(metadata["matrix_shape"], [30, 31])
        presence = sparse.load_npz(self.spec.presence_path)
        frequency = sparse.load_npz(self.spec.frequency_path)
        self.assertEqual(presence.shape, (30, 31))
        np.testing.assert_allclose(np.asarray(frequency.sum(axis=1)).ravel(), 1.0)

    def test_v2_assignments_have_two_outer_folds_and_no_holdout_in_inner(self) -> None:
        outer, inner, selected, audit = generate_v2_assignments(self.spec, self.assignments)
        self.assertEqual(len(outer), 60)
        self.assertEqual(set(outer["outer_fold"]), {1, 2})
        self.assertEqual(len(inner), 42)
        holdouts = set(outer.loc[outer["outer_fold"] == 1, ["outer_repeat", "sample_id"]].itertuples(index=False, name=None))
        inner_pairs = set(inner[["outer_repeat", "sample_id"]].itertuples(index=False, name=None))
        self.assertFalse(holdouts & inner_pairs)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(audit), 6)

    def test_inner_assignment_generation_is_deterministic(self) -> None:
        first = generate_v2_assignments(self.spec, self.assignments)[1]
        second = generate_v2_assignments(self.spec, self.assignments)[1]
        pd.testing.assert_frame_equal(first, second)

    def test_missing_aa_table_is_rejected(self) -> None:
        (self.root / "TRB/set/test/result/01_AA_clone_table/S29_AA_clone_table.csv").unlink()
        with self.assertRaises(RepeatedHoldoutTrainingError):
            discover_aa_clone_tables(self.spec, self.sample_ids)

    def test_frozen_assignment_hash_mismatch_is_rejected(self) -> None:
        frame = pd.read_csv(self.assignments_path)
        frame.loc[0, "role"] = "changed"
        frame.to_csv(self.assignments_path, index=False)
        with self.assertRaises(RepeatedHoldoutTrainingError):
            load_frozen_assignments(self.spec)

    def test_generated_scheme_records_repeated_holdout_contract(self) -> None:
        self.spec.output_root.mkdir(parents=True)
        for path in (
            self.spec.full_metadata_path,
            self.spec.full_base_matrix_path,
            self.spec.catalog_path,
            self.spec.presence_path,
            self.spec.frequency_path,
            self.spec.cache_metadata_path,
            self.spec.outer_assignments_path,
            self.spec.inner_assignments_path,
            self.spec.marker_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".npz":
                sparse.save_npz(path, sparse.csr_matrix((30, 31)))
            else:
                path.write_text("test\n", encoding="utf-8")
        result = generate_scheme_yaml(
            self.spec,
            {"matrix_shape": [30, 31], "catalog_size": 31},
            self.assignments,
        )
        overrides = result["overrides"]
        self.assertEqual(overrides["cross_validation"]["executed_outer_folds"], [1])
        self.assertEqual(overrides["repeated_holdout_training"]["holdout_size"], 9)
        self.assertEqual(overrides["final_model"]["status"], "not_selected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
