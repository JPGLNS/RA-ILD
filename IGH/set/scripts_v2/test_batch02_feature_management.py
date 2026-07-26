#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast tests for IGH Upgrade Batch 02; no model fitting is performed."""

from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.feature_inputs import (  # noqa: E402
    FeatureInputError,
    load_partition_feature_matrix,
)
from ra_ild_igh.paths import find_repository_root, resolve_project_path  # noqa: E402
from ra_ild_igh.scheme_management import build_prepared_scheme  # noqa: E402


class FakeConfig:
    def __init__(self, root: Path, base: Path, tables):
        self.repository_root = root
        self.raw = {
            "data": {
                "train": {
                    "base_matrix": str(base),
                    "additional_feature_tables": tables,
                }
            }
        }

    def path(self, dotted_key: str, *, must_exist=False, expect=None):
        current = self.raw
        for token in dotted_key.split("."):
            current = current[token]
        return resolve_project_path(
            current,
            self.repository_root,
            must_exist=must_exist,
            expect=expect,
        )


class Batch02FeatureManagementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository_root = find_repository_root(SCRIPT_DIR)
        cls.baseline_scheme = cls.repository_root / (
            "IGH/set/configs/schemes/igh_scheme_001_baseline_compat.yaml"
        )
        cls.no_clinical_scheme = cls.repository_root / (
            "IGH/set/configs/schemes/igh_scheme_002_no_clinical.yaml"
        )

    def test_baseline_remains_four_models_and_480_fits(self) -> None:
        prepared = build_prepared_scheme(self.baseline_scheme, self.repository_root)
        raw = prepared.resolved_config
        self.assertEqual(len(raw["models"]), 4)
        self.assertEqual(raw["nested_cv"]["expected_inner_fits_per_task"], 480)
        self.assertEqual(raw["aggregation"]["expected_metric_rows"], 400)
        self.assertEqual(raw["aggregation"]["expected_prediction_rows"], 9600)

    def test_no_clinical_scheme_replaces_models_and_recalculates_counts(self) -> None:
        prepared = build_prepared_scheme(self.no_clinical_scheme, self.repository_root)
        raw = prepared.resolved_config
        self.assertEqual(
            list(raw["models"]),
            [
                "M1_static_igh",
                "M2_static_igh_public",
                "M3_static_igh_public_material",
            ],
        )
        for spec in raw["models"].values():
            self.assertNotIn("age", spec["numeric"])
            self.assertNotIn("sex", spec["categorical"])
        self.assertEqual(raw["nested_cv"]["expected_inner_fits_per_task"], 360)
        self.assertEqual(raw["aggregation"]["expected_metric_rows"], 300)
        self.assertEqual(raw["aggregation"]["expected_prediction_rows"], 7200)
        self.assertEqual(raw["final_model"]["status"], "not_selected")
        self.assertNotIn("selected_model", raw["final_model"])
        self.assertEqual(
            raw["model_selection"]["expected_resolved_columns"],
            {
                "M1_static_igh": {"numeric": 1083, "categorical": 0},
                "M2_static_igh_public": {"numeric": 1091, "categorical": 0},
                "M3_static_igh_public_material": {
                    "numeric": 1091,
                    "categorical": 1,
                },
            },
        )

    def _write_base_and_extra(self, root: Path):
        base = root / "base.csv"
        extra = root / "extra.csv"
        pd.DataFrame(
            {
                "sample_id": ["S1", "S2", "S3"],
                "cohort": ["RA", "ILD", "RA"],
                "aa_clone_number": [10, 20, 30],
            }
        ).to_csv(base, index=False)
        pd.DataFrame(
            {
                "sample_id": ["S3", "S1", "S2"],
                "new_numeric": [3.0, 1.0, 2.0],
                "new_category": ["C", "A", "B"],
            }
        ).to_csv(extra, index=False)
        return base, extra

    def test_valid_additional_table_preserves_base_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, extra = self._write_base_and_extra(root)
            config = FakeConfig(
                root,
                base,
                [
                    {
                        "path": str(extra),
                        "key": "sample_id",
                        "required_columns": ["new_numeric", "new_category"],
                        "numeric_columns": ["new_numeric"],
                        "categorical_columns": ["new_category"],
                    }
                ],
            )
            merged, audit = load_partition_feature_matrix(config, "train")
            self.assertEqual(merged["sample_id"].tolist(), ["S1", "S2", "S3"])
            self.assertEqual(merged["new_numeric"].tolist(), [1.0, 2.0, 3.0])
            self.assertEqual(merged["new_category"].tolist(), ["A", "B", "C"])
            self.assertEqual(len(audit), 2)
            self.assertEqual(audit.iloc[1]["columns_added"], 2)

    def test_missing_sample_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, extra = self._write_base_and_extra(root)
            pd.read_csv(extra).iloc[:2].to_csv(extra, index=False)
            config = FakeConfig(root, base, [{"path": str(extra)}])
            with self.assertRaisesRegex(FeatureInputError, "coverage mismatch"):
                load_partition_feature_matrix(config, "train")

    def test_duplicate_sample_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, extra = self._write_base_and_extra(root)
            frame = pd.read_csv(extra)
            frame.loc[2, "sample_id"] = "S1"
            frame.to_csv(extra, index=False)
            config = FakeConfig(root, base, [{"path": str(extra)}])
            with self.assertRaisesRegex(FeatureInputError, "duplicate"):
                load_partition_feature_matrix(config, "train")

    def test_column_collision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, extra = self._write_base_and_extra(root)
            pd.read_csv(extra).rename(columns={"new_numeric": "cohort"}).to_csv(
                extra, index=False
            )
            config = FakeConfig(root, base, [{"path": str(extra)}])
            with self.assertRaisesRegex(FeatureInputError, "collide"):
                load_partition_feature_matrix(config, "train")

    def test_non_numeric_declared_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, extra = self._write_base_and_extra(root)
            frame = pd.read_csv(extra)
            frame["new_numeric"] = frame["new_numeric"].astype("object")
            frame.loc[0, "new_numeric"] = "bad"
            frame.to_csv(extra, index=False)
            config = FakeConfig(
                root,
                base,
                [{"path": str(extra), "numeric_columns": ["new_numeric"]}],
            )
            with self.assertRaisesRegex(FeatureInputError, "non-numeric"):
                load_partition_feature_matrix(config, "train")

    def test_template_is_not_silently_preparable(self) -> None:
        template = self.repository_root / (
            "IGH/set/configs/schemes/igh_scheme_additional_features_template.yaml"
        )
        prepared = build_prepared_scheme(template, self.repository_root)
        missing = [row for row in prepared.input_manifest if not row["exists"]]
        self.assertTrue(missing)
        self.assertTrue(
            any("additional_feature_tables" in row["config_key"] for row in missing)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
