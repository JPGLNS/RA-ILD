#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast structural tests for Upgrade Batch 01 (no model fitting)."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.paths import find_repository_root  # noqa: E402
from ra_ild_trb.scheme_management import (  # noqa: E402
    SchemeError,
    build_prepared_scheme,
    deep_merge,
    validate_scheme_id,
)


class SchemeManagementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository_root = find_repository_root(SCRIPT_DIR)
        cls.scheme_path = (
            cls.repository_root
            / "TRB/set/configs/schemes/trb_scheme_001_baseline_compat.yaml"
        )
        cls.prepared = build_prepared_scheme(cls.scheme_path, cls.repository_root)

    def test_scheme_id_validation(self) -> None:
        self.assertEqual(validate_scheme_id("trb_scheme_001"), "trb_scheme_001")
        for bad in ("A", "TRB Scheme", "../escape", "/absolute", "ab"):
            with self.assertRaises(SchemeError):
                validate_scheme_id(bad)

    def test_deep_merge_does_not_mutate_base(self) -> None:
        base = {"a": {"b": 1, "c": 2}, "x": [1]}
        original = copy.deepcopy(base)
        merged = deep_merge(base, {"a": {"b": 9}, "x": [2]})
        self.assertEqual(base, original)
        self.assertEqual(merged, {"a": {"b": 9, "c": 2}, "x": [2]})

    def test_output_paths_are_isolated(self) -> None:
        root = self.prepared.output_root.resolve()
        raw = self.prepared.resolved_config
        configured = [
            raw["outer_tasks"]["output_root"],
            raw["outer_tasks"]["manifest"],
            raw["outer_tasks"]["status"],
            raw["outer_tasks"]["logs_dir"],
            raw["aggregation"]["output_dir"],
            raw["final_model"]["bundle"],
            raw["independent_validation"]["output_dir"],
        ]
        for value in configured:
            candidate = (self.repository_root / value).resolve()
            candidate.relative_to(root)

    def test_baseline_scientific_sections_are_unchanged(self) -> None:
        import yaml

        with self.prepared.base_config_path.open("r", encoding="utf-8") as handle:
            base = yaml.safe_load(handle)
        resolved = self.prepared.resolved_config
        for key in (
            "data",
            "public_reference",
            "cross_validation",
            "model_engine",
            "models",
            "preprocessing",
            "modeling",
            "model_selection",
            "nested_cv",
            "stability",
        ):
            self.assertEqual(base[key], resolved[key], msg=f"Changed section: {key}")

    def test_baseline_counts_and_models_are_preserved(self) -> None:
        raw = self.prepared.resolved_config
        self.assertEqual(raw["data"]["train"]["expected_samples"], 123)
        self.assertEqual(raw["data"]["test"]["expected_samples"], 51)
        self.assertEqual(raw["cross_validation"]["outer_repeats"], 20)
        self.assertEqual(raw["cross_validation"]["outer_folds"], 5)
        self.assertEqual(raw["cross_validation"]["inner_folds"], 5)
        self.assertEqual(
            list(raw["models"]),
            [
                "M0_clinical",
                "M1_static_tcr",
                "M2_static_tcr_public",
                "M3_static_tcr_public_material",
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
