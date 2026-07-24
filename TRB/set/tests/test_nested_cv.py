#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the TRB V2 nested-CV orchestration layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

TEST_DIR = Path(__file__).resolve().parent
SET_DIR = TEST_DIR.parent
SRC_DIR = SET_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.nested_cv import (  # noqa: E402
    NestedCVError,
    NestedCVOptions,
    fit_outer_models,
    prepare_inner_folds,
    run_nested_outer_task,
    sample_role_frame,
    tune_models,
    validate_fixed_assignments,
)
from ra_ild_trb.public_reference import ThresholdScheme  # noqa: E402
from ra_ild_trb.specifications import (  # noqa: E402
    HyperparameterCandidate,
    ResolvedModelSpecification,
)


class NestedCVFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sample_ids = [f"S{i:02d}" for i in range(24)]
        cohort = ["RA" if i % 2 == 0 else "ILD" for i in range(24)]
        cls.base = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "cohort": cohort,
                "age": np.linspace(35, 70, 24),
                "sex": ["F" if i % 3 else "M" for i in range(24)],
                "material": ["PBMC" if i % 4 else "buffercoat" for i in range(24)],
                "static_a": np.sin(np.arange(24)) + np.asarray(cohort).astype(object).astype(str).tolist().count("ILD") * 0,
                "static_b": np.cos(np.arange(24) / 3),
            }
        )
        # Six balanced outer folds: each fold contains two RA and two ILD samples.
        outer_fold = [(i // 4) + 1 for i in range(24)]
        cls.outer = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "outer_repeat": 1,
                "outer_fold": outer_fold,
            }
        )
        outer_train = [sid for sid, fold in zip(sample_ids, outer_fold) if fold != 1]
        # Four balanced inner folds over the 20 outer-training samples.
        inner_rows = []
        for index, sample_id in enumerate(outer_train):
            inner_rows.append(
                {
                    "sample_id": sample_id,
                    "outer_repeat": 1,
                    "outer_fold": 1,
                    "inner_fold": ((index // 2) % 4) + 1,
                }
            )
        cls.inner = pd.DataFrame(inner_rows)

        rng = np.random.default_rng(17)
        raw = rng.binomial(1, 0.38, size=(24, 18)).astype(np.uint8)
        # Guarantee each sample has at least three catalog sequences.
        for row in range(24):
            raw[row, row % 18] = 1
            raw[row, (row + 3) % 18] = 1
            raw[row, (row + 7) % 18] = 1
        cls.presence = sparse.csr_matrix(raw)
        frequency = raw.astype(float)
        frequency /= frequency.sum(axis=1, keepdims=True)
        cls.frequency = sparse.csr_matrix(frequency)
        cls.aa_clone_numbers = raw.sum(axis=1).astype(float)
        cls.labels = np.asarray(cohort)
        cls.row_lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        cls.models = {
            "M0_clinical": ResolvedModelSpecification(
                name="M0_clinical",
                description="age + sex",
                numeric=("age",),
                categorical=("sex",),
            ),
            "M2_small": ResolvedModelSpecification(
                name="M2_small",
                description="clinical + static + public",
                numeric=(
                    "age",
                    "static_a",
                    "static_b",
                    "all_ref_public_clone_ratio",
                    "all_ref_public_frequency_sum",
                ),
                categorical=("sex",),
            ),
        }
        cls.candidates = [
            HyperparameterCandidate(alpha=0.1, lambda_value=1.0),
            HyperparameterCandidate(alpha=0.5, lambda_value=3.0),
        ]
        cls.scheme = ThresholdScheme(
            name="fixture",
            global_min_prevalence=0.10,
            global_min_count=2,
            group_min_prevalence=0.10,
            group_min_count=1,
            specific_prevalence_delta=0.10,
        )
        cls.options = NestedCVOptions(
            base_seed=1234,
            class_weight="balanced",
            max_iter=4000,
            tolerance=1e-5,
            zero_sd_tolerance=1e-12,
            epsilon=1e-8,
        )

    def split(self):
        return validate_fixed_assignments(
            self.base["sample_id"],
            self.outer,
            self.inner,
            outer_repeat=1,
            outer_fold=1,
            outer_folds=6,
            inner_folds=4,
        )

    def test_assignment_counts_and_order(self) -> None:
        split = self.split()
        self.assertEqual(split.n_outer_train, 20)
        self.assertEqual(split.n_outer_valid, 4)
        self.assertEqual(split.outer_valid_ids, ("S00", "S01", "S02", "S03"))
        self.assertEqual(split.inner_folds, (1, 2, 3, 4))

    def test_outer_validation_leakage_is_rejected(self) -> None:
        broken = self.inner.copy()
        broken.loc[0, "sample_id"] = "S00"
        with self.assertRaises(NestedCVError):
            validate_fixed_assignments(
                self.base["sample_id"],
                self.outer,
                broken,
                outer_repeat=1,
                outer_fold=1,
                outer_folds=6,
                inner_folds=4,
            )

    def test_missing_inner_fold_is_rejected(self) -> None:
        broken = self.inner.loc[self.inner["inner_fold"] != 4].copy()
        with self.assertRaises(NestedCVError):
            validate_fixed_assignments(
                self.base["sample_id"],
                self.outer,
                broken,
                outer_repeat=1,
                outer_fold=1,
                outer_folds=6,
                inner_folds=4,
            )

    def test_prepare_inner_folds_builds_every_model_and_fold(self) -> None:
        prepared = prepare_inner_folds(
            self.base,
            self.split(),
            self.row_lookup,
            self.presence,
            self.frequency,
            self.labels,
            self.aa_clone_numbers,
            self.models,
            self.scheme,
            self.options,
        )
        self.assertEqual(set(prepared.prepared), set(self.models))
        for folds in prepared.prepared.values():
            self.assertEqual(set(folds), {1, 2, 3, 4})
            for fold in folds.values():
                self.assertGreaterEqual(len(fold.valid_sample_ids), 4)
                self.assertEqual(
                    len(fold.train_sample_ids) + len(fold.valid_sample_ids), 20
                )
                self.assertEqual(set(fold.y_valid.tolist()), {0, 1})
                self.assertTrue(np.isfinite(fold.X_train).all())
                self.assertTrue(np.isfinite(fold.X_valid).all())
        self.assertFalse(prepared.public_reference_audit.empty)
        self.assertFalse(prepared.public_loo_assignments.empty)

    def test_tuning_grid_and_selected_oof_dimensions(self) -> None:
        prepared = prepare_inner_folds(
            self.base,
            self.split(),
            self.row_lookup,
            self.presence,
            self.frequency,
            self.labels,
            self.aa_clone_numbers,
            self.models,
            self.scheme,
            self.options,
        )
        tuned = tune_models(prepared.prepared, self.candidates, self.options)
        self.assertEqual(len(tuned.tuning), len(self.models) * len(self.candidates))
        self.assertEqual(
            len(tuned.selected_oof_predictions),
            len(self.models) * self.split().n_outer_train,
        )
        self.assertEqual(set(tuned.selected), set(self.models))
        self.assertTrue(tuned.tuning["all_fits_converged"].all())

    def test_tuning_is_reproducible(self) -> None:
        prepared = prepare_inner_folds(
            self.base,
            self.split(),
            self.row_lookup,
            self.presence,
            self.frequency,
            self.labels,
            self.aa_clone_numbers,
            self.models,
            self.scheme,
            self.options,
        )
        left = tune_models(prepared.prepared, self.candidates, self.options)
        right = tune_models(prepared.prepared, self.candidates, self.options)
        np.testing.assert_allclose(
            left.selected_oof_predictions["probability"],
            right.selected_oof_predictions["probability"],
            rtol=0,
            atol=0,
        )
        pd.testing.assert_frame_equal(left.tuning, right.tuning)

    def test_outer_fit_outputs_expected_rows(self) -> None:
        prepared = prepare_inner_folds(
            self.base,
            self.split(),
            self.row_lookup,
            self.presence,
            self.frequency,
            self.labels,
            self.aa_clone_numbers,
            self.models,
            self.scheme,
            self.options,
        )
        tuned = tune_models(prepared.prepared, self.candidates, self.options)
        outer = fit_outer_models(
            self.base,
            self.split(),
            self.row_lookup,
            self.presence,
            self.frequency,
            self.labels,
            self.aa_clone_numbers,
            self.models,
            tuned.selected,
            self.scheme,
            self.options,
        )
        self.assertEqual(len(outer.predictions), len(self.models) * 4)
        self.assertEqual(len(outer.metrics), len(self.models))
        self.assertEqual(len(outer.outer_train_public), 20)
        self.assertEqual(len(outer.outer_validation_public), 4)
        self.assertEqual(set(outer.metrics["model"]), set(self.models))

    def test_full_task_is_reproducible(self) -> None:
        kwargs = dict(
            base=self.base,
            outer_assignments=self.outer,
            inner_assignments=self.inner,
            row_lookup=self.row_lookup,
            presence=self.presence,
            frequency=self.frequency,
            labels=self.labels,
            aa_clone_numbers=self.aa_clone_numbers,
            model_specifications=self.models,
            candidates=self.candidates,
            scheme=self.scheme,
            options=self.options,
            outer_repeat=1,
            outer_fold=1,
            outer_folds=6,
            inner_folds=4,
        )
        left = run_nested_outer_task(**kwargs)
        right = run_nested_outer_task(**kwargs)
        pd.testing.assert_frame_equal(left.inner_tuning, right.inner_tuning)
        pd.testing.assert_frame_equal(left.outer_predictions, right.outer_predictions)
        pd.testing.assert_frame_equal(left.coefficients, right.coefficients)

    def test_full_task_has_no_outer_validation_in_inner_oof(self) -> None:
        result = run_nested_outer_task(
            self.base,
            self.outer,
            self.inner,
            self.row_lookup,
            self.presence,
            self.frequency,
            self.labels,
            self.aa_clone_numbers,
            self.models,
            self.candidates,
            self.scheme,
            self.options,
            outer_repeat=1,
            outer_fold=1,
            outer_folds=6,
            inner_folds=4,
        )
        self.assertFalse(
            set(result.inner_selected_oof_predictions["sample_id"])
            & set(result.split.outer_valid_ids)
        )

    def test_sample_roles_cover_base_once(self) -> None:
        roles = sample_role_frame(self.base, self.split(), self.options)
        self.assertEqual(len(roles), len(self.base))
        self.assertFalse(roles["sample_id"].duplicated().any())
        self.assertEqual(
            roles["outer_role"].value_counts().to_dict(),
            {"training": 20, "validation": 4},
        )

    def test_invalid_options_are_rejected(self) -> None:
        with self.assertRaises(NestedCVError):
            NestedCVOptions(base_seed=1, class_weight="invalid")
        with self.assertRaises(NestedCVError):
            NestedCVOptions(base_seed=1, positive_label="RA", negative_label="RA")


if __name__ == "__main__":
    unittest.main(verbosity=2)
