#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast tests for Batch 03 repeated-holdout generation; no model fitting."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.repeated_holdout import (  # noqa: E402
    SplitGenerationError,
    generate_repeated_holdout,
    load_combined_metadata,
    load_repeated_holdout_spec,
    validate_frozen_split_set,
    validate_generation_result,
    write_frozen_split_set,
)


class Batch03RepeatedHoldoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / ".gitignore").write_text("# test\n", encoding="utf-8")
        (self.root / "IGH/set/configs/split_sets").mkdir(parents=True)
        (self.root / "IGH/set/train").mkdir(parents=True)
        (self.root / "IGH/set/test").mkdir(parents=True)
        (self.root / "TRB").mkdir()
        self.metadata = self._make_metadata(singleton=True)
        self.metadata.iloc[:50].to_csv(
            self.root / "IGH/set/train/metadata_train_70.csv", index=False
        )
        self.metadata.iloc[50:].to_csv(
            self.root / "IGH/set/test/metadata_test_30.csv", index=False
        )
        self.spec_path = (
            self.root / "IGH/set/configs/split_sets/test_repeat3.yaml"
        )
        self._write_spec()

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _make_metadata(singleton: bool) -> pd.DataFrame:
        rows = []
        # Four splittable exact strata, each with 20 patients.
        for cohort in ("RA", "ILD"):
            for batch in ("A", "B"):
                for i in range(20):
                    index = len(rows)
                    rows.append(
                        {
                            "patient": "P%03d" % index,
                            "libraryid": "L%03d" % index,
                            "cohort": cohort,
                            "batch": batch,
                            "material": "PBMC" if i % 3 else "buffercoat",
                            "sex": "female" if i % 5 else "male",
                            "age": 35 + ((index * 7) % 36),
                        }
                    )
        if singleton:
            rows.append(
                {
                    "patient": "P080",
                    "libraryid": "L080",
                    "cohort": "ILD",
                    "batch": "C",
                    "material": "PBMC",
                    "sex": "female",
                    "age": 56,
                }
            )
        return pd.DataFrame(rows)

    def _spec_mapping(self) -> dict:
        return {
            "split_set_version": "1.1",
            "split_set": {
                "id": "test_repeat3",
                "description": "synthetic IGH singleton test",
                "output_root": "IGH/set/split_sets/test_repeat3",
            },
            "source": {
                "metadata_files": [
                    "IGH/set/train/metadata_train_70.csv",
                    "IGH/set/test/metadata_test_30.csv",
                ],
                "sample_id_column": "libraryid",
                "patient_id_column": "patient",
                "require_one_sample_per_patient": True,
                "expected_samples": 81,
                "expected_patients": 81,
            },
            "split": {
                "strategy": "repeated_holdout",
                "label_column": "cohort",
                "repeats": 3,
                "train_size": 57,
                "holdout_size": 24,
                "exact_strata": ["cohort", "batch"],
                "singleton_strata_policy": "fixed_train",
                "expected_fixed_train_sample_ids": ["L080"],
                "balance_categorical": ["material", "sex"],
                "balance_numeric": ["age"],
                "candidates_per_repeat": 80,
                "base_seed": 20260726,
                "diversity_weight": 0.05,
                "max_pairwise_holdout_jaccard": 0.45,
                "role_names": {"train": "train", "holdout": "holdout"},
            },
        }

    def _write_spec(self, mapping: Optional[dict] = None) -> None:
        self.spec_path.write_text(
            yaml.safe_dump(mapping or self._spec_mapping(), sort_keys=False),
            encoding="utf-8",
        )

    def _generate(self):
        spec = load_repeated_holdout_spec(
            self.spec_path, repository_root=self.root
        )
        metadata = load_combined_metadata(spec)
        return spec, metadata, generate_repeated_holdout(metadata, spec)

    def test_exact_sizes_and_one_role_per_patient(self) -> None:
        spec, _, result = self._generate()
        self.assertEqual(len(result.assignments), 81 * 3)
        for split_id, frame in result.assignments.groupby("split_id"):
            self.assertEqual(frame["patient_id"].nunique(), 81, split_id)
            counts = frame["role"].value_counts().to_dict()
            self.assertEqual(counts, {spec.train_role: 57, spec.holdout_role: 24})

    def test_singleton_is_fixed_train_in_every_repeat(self) -> None:
        spec, _, result = self._generate()
        observed = result.assignments.loc[
            result.assignments["sample_id"] == "L080"
        ]
        self.assertEqual(len(observed), 3)
        self.assertEqual(set(observed["role"]), {spec.train_role})
        self.assertEqual(
            set(observed["assignment_reason"]),
            {"fixed_train_singleton_stratum"},
        )
        frequency = result.sample_holdout_frequency.set_index("sample_id")
        self.assertEqual(int(frequency.loc["L080", "holdout_count"]), 0)
        self.assertTrue(bool(frequency.loc["L080", "fixed_train"]))

    def test_generation_is_deterministic(self) -> None:
        spec, metadata, first = self._generate()
        second = generate_repeated_holdout(metadata, spec)
        columns = [
            "split_id",
            "patient_id",
            "role",
            "assignment_reason",
            "candidate_seed",
        ]
        pd.testing.assert_frame_equal(
            first.assignments[columns], second.assignments[columns]
        )

    def test_repeats_are_distinct_and_below_overlap_limit(self) -> None:
        spec, _, result = self._generate()
        report = validate_generation_result(result, spec)
        self.assertEqual(report["status"], "PASS")
        self.assertLessEqual(
            report["maximum_pairwise_holdout_jaccard"], 0.45 + 1e-12
        )

    def test_wrong_expected_singleton_is_rejected(self) -> None:
        mapping = self._spec_mapping()
        mapping["split"]["expected_fixed_train_sample_ids"] = ["L079"]
        self._write_spec(mapping)
        spec = load_repeated_holdout_spec(
            self.spec_path, repository_root=self.root
        )
        metadata = load_combined_metadata(spec)
        with self.assertRaisesRegex(SplitGenerationError, "differ"):
            generate_repeated_holdout(metadata, spec)

    def test_singleton_error_policy_is_rejected(self) -> None:
        mapping = self._spec_mapping()
        mapping["split"]["singleton_strata_policy"] = "error"
        mapping["split"]["expected_fixed_train_sample_ids"] = []
        self._write_spec(mapping)
        spec = load_repeated_holdout_spec(
            self.spec_path, repository_root=self.root
        )
        metadata = load_combined_metadata(spec)
        with self.assertRaisesRegex(SplitGenerationError, "singleton strata"):
            generate_repeated_holdout(metadata, spec)

    def test_missing_required_column_is_rejected(self) -> None:
        path = self.root / "IGH/set/test/metadata_test_30.csv"
        frame = pd.read_csv(path).drop(columns=["age"])
        frame.to_csv(path, index=False)
        spec = load_repeated_holdout_spec(
            self.spec_path, repository_root=self.root
        )
        with self.assertRaisesRegex(SplitGenerationError, "missing required columns"):
            load_combined_metadata(spec)

    def test_duplicate_sample_is_rejected(self) -> None:
        path = self.root / "IGH/set/test/metadata_test_30.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "libraryid"] = "L000"
        frame.to_csv(path, index=False)
        spec = load_repeated_holdout_spec(
            self.spec_path, repository_root=self.root
        )
        with self.assertRaisesRegex(SplitGenerationError, "duplicate sample IDs"):
            load_combined_metadata(spec)

    def test_frozen_output_is_hashed_and_immutable(self) -> None:
        spec, _, result = self._generate()
        marker = write_frozen_split_set(result, spec)
        self.assertEqual(marker["status"], "FROZEN")
        report = validate_frozen_split_set(spec)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["assignment_rows"], 81 * 3)
        with self.assertRaisesRegex(FileExistsError, "already frozen"):
            write_frozen_split_set(result, spec)

    def test_frozen_tamper_is_detected(self) -> None:
        spec, _, result = self._generate()
        marker = write_frozen_split_set(result, spec)
        assignments = self.root / marker["files"]["assignments"]["path"]
        assignments.write_text(
            assignments.read_text(encoding="utf-8") + "# tampered\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SplitGenerationError, "SHA256 mismatch"):
            validate_frozen_split_set(spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
