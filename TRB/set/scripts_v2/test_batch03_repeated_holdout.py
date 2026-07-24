#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Batch 03 repeated holdout generation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.repeated_holdout import (  # noqa: E402
    SplitGenerationError,
    generate_repeated_holdout,
    load_combined_metadata,
    load_repeated_holdout_spec,
    write_frozen_split_set,
)


class Batch03RepeatedHoldoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / ".gitignore").write_text("# test\n", encoding="utf-8")
        (self.root / "TRB/set/configs/split_sets").mkdir(parents=True)
        (self.root / "TRB/set/train").mkdir(parents=True)
        (self.root / "TRB/set/test").mkdir(parents=True)
        (self.root / "IGH").mkdir()
        self.metadata = self._make_metadata()
        self.metadata.iloc[:50].to_csv(
            self.root / "TRB/set/train/metadata_train_70.csv", index=False
        )
        self.metadata.iloc[50:].to_csv(
            self.root / "TRB/set/test/metadata_test_30.csv", index=False
        )
        self.spec_path = self.root / "TRB/set/configs/split_sets/test_repeat3.yaml"
        self._write_spec()

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _make_metadata() -> pd.DataFrame:
        rows = []
        # Four exact strata, each with 20 patients.
        for cohort in ("RA", "ILD"):
            for batch in ("A", "B"):
                for i in range(20):
                    index = len(rows)
                    rows.append(
                        {
                            "patient": f"P{index:03d}",
                            "libraryid": f"L{index:03d}",
                            "cohort": cohort,
                            "batch": batch,
                            "material": "PBMC" if i % 3 else "buffercoat",
                            "sex": "female" if i % 5 else "male",
                            "age": 35 + ((index * 7) % 36),
                        }
                    )
        return pd.DataFrame(rows)

    def _spec_mapping(self) -> dict:
        return {
            "split_set_version": "1.0",
            "split_set": {
                "id": "test_repeat3",
                "description": "synthetic test",
                "output_root": "TRB/set/split_sets/test_repeat3",
            },
            "source": {
                "metadata_files": [
                    "TRB/set/train/metadata_train_70.csv",
                    "TRB/set/test/metadata_test_30.csv",
                ],
                "sample_id_column": "libraryid",
                "patient_id_column": "patient",
                "require_one_sample_per_patient": True,
                "expected_samples": 80,
                "expected_patients": 80,
            },
            "split": {
                "strategy": "repeated_holdout",
                "label_column": "cohort",
                "repeats": 3,
                "train_size": 56,
                "holdout_size": 24,
                "exact_strata": ["cohort", "batch"],
                "balance_categorical": ["material", "sex"],
                "balance_numeric": ["age"],
                "candidates_per_repeat": 60,
                "base_seed": 20260724,
                "diversity_weight": 0.05,
                "max_pairwise_holdout_jaccard": 0.45,
                "role_names": {"train": "train", "holdout": "holdout"},
            },
        }

    def _write_spec(self, mapping: dict | None = None) -> None:
        self.spec_path.write_text(
            yaml.safe_dump(mapping or self._spec_mapping(), sort_keys=False),
            encoding="utf-8",
        )

    def _generate(self):
        spec = load_repeated_holdout_spec(self.spec_path, repository_root=self.root)
        metadata = load_combined_metadata(spec)
        return spec, metadata, generate_repeated_holdout(metadata, spec)

    def test_exact_sizes_and_one_role_per_patient(self) -> None:
        spec, _, result = self._generate()
        self.assertEqual(len(result.assignments), 80 * 3)
        for split_id, frame in result.assignments.groupby("split_id"):
            self.assertEqual(frame["patient_id"].nunique(), 80, split_id)
            counts = frame["role"].value_counts().to_dict()
            self.assertEqual(counts, {spec.train_role: 56, spec.holdout_role: 24})

    def test_generation_is_deterministic(self) -> None:
        spec, metadata, first = self._generate()
        second = generate_repeated_holdout(metadata, spec)
        columns = ["split_id", "patient_id", "role", "candidate_seed"]
        pd.testing.assert_frame_equal(first.assignments[columns], second.assignments[columns])

    def test_repeats_are_distinct_and_below_overlap_limit(self) -> None:
        spec, _, result = self._generate()
        holdout_sets = []
        for _, frame in result.assignments.groupby("split_id", sort=True):
            holdout_sets.append(set(frame.loc[frame["role"] == spec.holdout_role, "patient_id"]))
        self.assertEqual(len({frozenset(value) for value in holdout_sets}), 3)
        self.assertTrue((result.pairwise_overlap["holdout_jaccard"] <= 0.45 + 1e-12).all())

    def test_missing_required_column_is_rejected(self) -> None:
        path = self.root / "TRB/set/test/metadata_test_30.csv"
        frame = pd.read_csv(path).drop(columns=["age"])
        frame.to_csv(path, index=False)
        spec = load_repeated_holdout_spec(self.spec_path, repository_root=self.root)
        with self.assertRaisesRegex(SplitGenerationError, "missing required columns"):
            load_combined_metadata(spec)

    def test_duplicate_sample_is_rejected(self) -> None:
        path = self.root / "TRB/set/test/metadata_test_30.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "libraryid"] = "L000"
        frame.to_csv(path, index=False)
        spec = load_repeated_holdout_spec(self.spec_path, repository_root=self.root)
        with self.assertRaisesRegex(SplitGenerationError, "duplicate sample IDs"):
            load_combined_metadata(spec)

    def test_multiple_samples_per_patient_is_rejected(self) -> None:
        path = self.root / "TRB/set/test/metadata_test_30.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "patient"] = "P000"
        frame.to_csv(path, index=False)
        spec = load_repeated_holdout_spec(self.spec_path, repository_root=self.root)
        with self.assertRaisesRegex(SplitGenerationError, "exactly one sample per patient"):
            load_combined_metadata(spec)

    def test_label_must_be_in_exact_strata(self) -> None:
        mapping = self._spec_mapping()
        mapping["split"]["exact_strata"] = ["batch"]
        self._write_spec(mapping)
        with self.assertRaisesRegex(SplitGenerationError, "label_column must be included"):
            load_repeated_holdout_spec(self.spec_path, repository_root=self.root)

    def test_frozen_output_cannot_be_overwritten(self) -> None:
        spec, _, result = self._generate()
        marker = write_frozen_split_set(result, spec)
        self.assertEqual(marker["status"], "FROZEN")
        marker_path = spec.output_root / "SPLITS_FROZEN.json"
        self.assertTrue(marker_path.is_file())
        parsed = json.loads(marker_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["split_set_id"], "test_repeat3")
        with self.assertRaisesRegex(FileExistsError, "already frozen"):
            write_frozen_split_set(result, spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
