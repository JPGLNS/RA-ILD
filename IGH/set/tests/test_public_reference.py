#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for leakage-controlled V2 public-reference calculations."""

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

from ra_ild_igh.public_reference import (  # noqa: E402
    ALL_PUBLIC_FEATURES,
    ThresholdScheme,
    apply_reference,
    build_reference_from_counts,
    build_reference_from_presence,
    effective_count_threshold,
    external_public_features,
    leave_one_out_public_features,
    public_feature_columns,
)


class PublicReferenceFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Six samples: three RA followed by three ILD; eight catalog sequences.
        dense_presence = np.array(
            [
                [1, 1, 0, 1, 1, 1, 1, 0],
                [1, 1, 0, 1, 0, 0, 1, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 1, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        dense_frequency = dense_presence.astype(float)
        row_sums = dense_frequency.sum(axis=1, keepdims=True)
        dense_frequency = np.divide(
            dense_frequency,
            row_sums,
            out=np.zeros_like(dense_frequency),
            where=row_sums > 0,
        )
        cls.presence = sparse.csr_matrix(dense_presence)
        cls.frequency = sparse.csr_matrix(dense_frequency)
        cls.labels = np.array(["RA", "RA", "RA", "ILD", "ILD", "ILD"])
        cls.clone_numbers = dense_presence.sum(axis=1).astype(float)
        cls.scheme = ThresholdScheme(
            name="synthetic",
            global_min_prevalence=0.0,
            global_min_count=2,
            group_min_prevalence=0.0,
            group_min_count=2,
            specific_prevalence_delta=0.25,
        )
        cls.epsilon = 1e-8

    def test_effective_threshold_uses_ceiling_and_minimum(self) -> None:
        self.assertEqual(effective_count_threshold(123, 0.02, 3), 3)
        self.assertEqual(effective_count_threshold(200, 0.02, 3), 4)
        self.assertEqual(effective_count_threshold(47, 0.05, 3), 3)

    def test_full_reference_masks(self) -> None:
        reference = build_reference_from_presence(
            self.presence,
            np.arange(6),
            self.labels,
            self.scheme,
        )
        np.testing.assert_array_equal(
            reference.global_mask,
            np.array([1, 1, 1, 1, 1, 0, 1, 1], dtype=bool),
        )
        np.testing.assert_array_equal(
            reference.ra_specific_mask,
            np.array([0, 1, 0, 0, 0, 0, 1, 0], dtype=bool),
        )
        np.testing.assert_array_equal(
            reference.ild_specific_mask,
            np.array([0, 0, 1, 0, 0, 0, 0, 1], dtype=bool),
        )
        np.testing.assert_array_equal(
            reference.shared_mask,
            np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=bool),
        )
        self.assertEqual(
            reference.sizes(),
            {"global": 7, "RA_specific": 2, "ILD_specific": 2, "shared": 2},
        )

    def test_count_builder_matches_presence_builder(self) -> None:
        ra_counts = np.asarray(self.presence[:3].sum(axis=0)).ravel()
        ild_counts = np.asarray(self.presence[3:].sum(axis=0)).ravel()
        from_counts = build_reference_from_counts(
            ra_counts,
            ild_counts,
            3,
            3,
            self.scheme,
        )
        from_presence = build_reference_from_presence(
            self.presence,
            np.arange(6),
            self.labels,
            self.scheme,
        )
        for name in (
            "global_mask",
            "ra_specific_mask",
            "ild_specific_mask",
            "shared_mask",
        ):
            np.testing.assert_array_equal(
                getattr(from_counts, name),
                getattr(from_presence, name),
            )
        self.assertEqual(from_counts.thresholds(), from_presence.thresholds())

    def test_apply_reference_returns_all_18_features(self) -> None:
        reference = build_reference_from_presence(
            self.presence,
            np.arange(6),
            self.labels,
            self.scheme,
        )
        output = apply_reference(
            self.presence,
            self.frequency,
            [0, 3],
            self.clone_numbers,
            reference,
            self.epsilon,
        )
        self.assertEqual(tuple(output.columns), ALL_PUBLIC_FEATURES)
        self.assertEqual(output.shape, (2, 18))
        self.assertTrue(np.isfinite(output.to_numpy()).all())
        self.assertAlmostEqual(output.loc[0, "RA_specific_ref_clone_number"], 2.0)
        self.assertAlmostEqual(output.loc[0, "ILD_specific_ref_clone_number"], 0.0)
        self.assertAlmostEqual(output.loc[3, "ILD_specific_ref_clone_number"], 2.0)

    def test_loo_optimized_matches_brute_force(self) -> None:
        optimized = leave_one_out_public_features(
            self.presence,
            self.frequency,
            np.arange(6),
            self.labels,
            self.clone_numbers,
            self.scheme,
            self.epsilon,
            context="unit_test_loo",
        )

        brute_parts = []
        for held_out in range(6):
            reference_rows = np.array([row for row in range(6) if row != held_out])
            reference = build_reference_from_presence(
                self.presence,
                reference_rows,
                self.labels,
                self.scheme,
            )
            values = apply_reference(
                self.presence,
                self.frequency,
                [held_out],
                self.clone_numbers,
                reference,
                self.epsilon,
            )
            brute_parts.append(values)
        brute = pd.concat(brute_parts).sort_index()

        np.testing.assert_allclose(
            optimized.features.loc[:, list(ALL_PUBLIC_FEATURES)].to_numpy(),
            brute.loc[:, list(ALL_PUBLIC_FEATURES)].to_numpy(),
            rtol=0.0,
            atol=1e-14,
        )
        self.assertTrue(
            (optimized.assignment_audit["n_reference_samples"] == 5).all()
        )
        self.assertTrue((optimized.reference_audit["n_reference"] == 5).all())

    def test_external_transform_uses_disjoint_reference(self) -> None:
        result = external_public_features(
            self.presence,
            self.frequency,
            reference_rows=[0, 1, 3, 4],
            target_rows=[2, 5],
            labels=self.labels,
            aa_clone_numbers=self.clone_numbers,
            scheme=self.scheme,
            epsilon=self.epsilon,
            context="unit_test_external",
        )
        self.assertEqual(result.features.shape, (2, 18))
        self.assertEqual(int(result.reference_audit.loc[0, "n_reference"]), 4)
        self.assertEqual(int(result.assignment_audit.loc[0, "n_target_samples"]), 2)

    def test_overlapping_external_rows_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            external_public_features(
                self.presence,
                self.frequency,
                reference_rows=[0, 1, 3, 4],
                target_rows=[1, 2],
                labels=self.labels,
                aa_clone_numbers=self.clone_numbers,
                scheme=self.scheme,
                epsilon=self.epsilon,
            )

    def test_public_feature_set_order(self) -> None:
        raw = public_feature_columns("raw_bilateral")
        self.assertEqual(len(raw), 8)
        self.assertEqual(raw[0], "all_ref_public_clone_ratio")
        self.assertEqual(raw[-1], "ILD_specific_ref_frequency_sum")
        with self.assertRaises(KeyError):
            public_feature_columns("not_a_feature_set")


if __name__ == "__main__":
    unittest.main(verbosity=2)
