#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the frozen-bundle repeat-3mer scheme adapter."""
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts_v2" / "prepare_repeat3mer_bundle_schemes.py"
SPEC = importlib.util.spec_from_file_location("repeat3mer_bundle_adapter", SCRIPT)
ADAPTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ADAPTER
SPEC.loader.exec_module(ADAPTER)


class Repeat3merBundleAdapterTests(unittest.TestCase):
    def test_assignment_audit_accepts_100_repeated_holdouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignments.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "repeat_index", "role"])
                writer.writeheader()
                for repeat in range(1, 101):
                    for index in range(10):
                        writer.writerow({
                            "sample_id": f"S{index:02d}",
                            "repeat_index": repeat,
                            "role": "train" if index < 7 else "holdout",
                        })
            marker = {
                "split_count": 100,
                "train_sizes": [7] * 100,
                "holdout_sizes": [3] * 100,
            }
            result = ADAPTER._validate_assignments(path, [f"S{i:02d}" for i in range(10)], marker)
            self.assertEqual(result["repeat_count"], 100)
            self.assertEqual(result["train_size"], 7)
            self.assertEqual(result["holdout_size"], 3)

    def test_assignment_audit_rejects_sample_coverage_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignments.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "repeat_index", "role"])
                writer.writeheader()
                for index in range(10):
                    writer.writerow({
                        "sample_id": f"S{index:02d}",
                        "repeat_index": 1,
                        "role": "train" if index < 7 else "holdout",
                    })
            marker = {"split_count": 1, "train_sizes": [7], "holdout_sizes": [3]}
            with self.assertRaises(ADAPTER.Repeat3merBundleSchemeError):
                ADAPTER._validate_assignments(
                    path, [f"S{i:02d}" for i in range(9)] + ["EXTRA"], marker
                )

    def test_aa_manifest_requires_unique_samples_and_existing_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = root / "S01.csv"
            table.write_text("cdr3_aa,frequency_norm\nCASSF,1\n", encoding="utf-8")
            manifest = root / "aa_clone_table_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["sample_id", "path", "frequency_column"]
                )
                writer.writeheader()
                writer.writerow({
                    "sample_id": "S01",
                    "path": str(table),
                    "frequency_column": "frequency_norm",
                })
            self.assertEqual(ADAPTER._read_aa_manifest(manifest, 1), ("S01",))
            with self.assertRaises(ADAPTER.Repeat3merBundleSchemeError):
                ADAPTER._read_aa_manifest(manifest, 2)

    def test_default_options_match_batch09_contract(self):
        options = ADAPTER.REPEAT_3MER_OPTIONS
        self.assertEqual(options["selection_scope"], "outer_repeat_train_only")
        self.assertEqual(options["ranking_rule"], "prevalence_then_total_variance")
        self.assertEqual(options["min_sample_count"], 5)
        self.assertEqual(options["min_prevalence"], 0.05)
        self.assertEqual(options["max_kmers"], 500)
        self.assertEqual(options["top_k_values"], [50, 100, 200, 500])
        self.assertEqual(options["expected_core_feature_count"], 83)


if __name__ == "__main__":
    unittest.main(verbosity=2)
