#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused tests for IGH frozen-model cross-repeat evaluation."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.cross_repeat_evaluation import (
    ALL_PUBLIC_FEATURES,
    FrozenLinearModel,
    SelectedModelParameters,
    _hybrid_public,
    build_overlap_audit,
    evaluation_role,
    expected_cross_repeat_counts,
    frozen_probability,
)


class IdentityPreprocessor:
    def transform(self, frame):
        class Result:
            matrix = frame[["x1", "x2"]].to_numpy(float)
            feature_names = ("x1", "x2")

        return Result()


def test_expected_counts() -> None:
    one = expected_cross_repeat_counts(3, 1, 49)
    assert one["model_instances"] == 3
    assert one["evaluation_pairs_per_model"] == 9
    assert one["full_metric_rows"] == 9
    assert one["unseen_only_metric_rows"] == 9
    assert one["prediction_rows"] == 441

    all_models = expected_cross_repeat_counts(3, 3, 49)
    assert all_models["model_instances"] == 9
    assert all_models["full_metric_rows"] == 27
    assert all_models["unseen_only_metric_rows"] == 27
    assert all_models["prediction_rows"] == 1323


def test_evaluation_role() -> None:
    training = ("A", "B", "C")
    assert evaluation_role(
        fit_repeat=1,
        evaluation_repeat=1,
        sample_id="D",
        fit_training_ids=training,
    ) == "native_holdout"
    assert evaluation_role(
        fit_repeat=1,
        evaluation_repeat=2,
        sample_id="B",
        fit_training_ids=training,
    ) == "cross_holdout_seen_training"
    assert evaluation_role(
        fit_repeat=1,
        evaluation_repeat=2,
        sample_id="D",
        fit_training_ids=training,
    ) == "cross_holdout_unseen"


def test_overlap_audit() -> None:
    membership = {
        1: {"training": ("A", "B", "C", "D"), "holdout": ("E", "F")},
        2: {"training": ("A", "C", "E", "F"), "holdout": ("B", "D")},
        3: {"training": ("B", "D", "E", "F"), "holdout": ("A", "C")},
    }
    audit = build_overlap_audit(membership)
    assert len(audit) == 9
    assert not audit.duplicated(["fit_repeat", "evaluation_repeat"]).any()
    diagonal = audit.loc[audit["is_native_pair"]]
    assert (diagonal["n_seen_in_fit_training"] == 0).all()
    assert (diagonal["n_unseen_to_fit_model"] == 2).all()
    assert (diagonal["holdout_jaccard"] == 1.0).all()


def test_hybrid_public_splits_seen_and_unseen_rows() -> None:
    fit_train_rows = np.asarray([0, 1, 2, 3], dtype=int)
    evaluation_rows = np.asarray([1, 4, 3], dtype=int)
    saved_train_loo = pd.DataFrame(
        {
            column: [10.0 + index, 20.0 + index, 30.0 + index, 40.0 + index]
            for index, column in enumerate(ALL_PUBLIC_FEATURES)
        },
        index=fit_train_rows,
    )
    unseen_features = pd.DataFrame(
        {
            column: [100.0 + index]
            for index, column in enumerate(ALL_PUBLIC_FEATURES)
        },
        index=[4],
    )

    def fake_external(
        presence,
        frequency,
        reference_rows,
        target_rows,
        labels,
        aa_clone_numbers,
        scheme,
        epsilon,
        *,
        context,
    ):
        assert tuple(map(int, reference_rows)) == (0, 1, 2, 3)
        assert tuple(map(int, target_rows)) == (4,)
        assert not set(map(int, reference_rows)) & set(map(int, target_rows))
        assert context == "cross_repeat_evaluation_application"
        return SimpleNamespace(features=unseen_features)

    with patch(
        "ra_ild_igh.cross_repeat_evaluation.external_public_features",
        side_effect=fake_external,
    ) as mocked:
        observed, methods = _hybrid_public(
            presence=None,
            frequency=None,
            fit_train_rows=fit_train_rows,
            evaluation_rows=evaluation_rows,
            labels=np.asarray(["RA", "RA", "ILD", "ILD", "RA"]),
            aa_clone_numbers=np.ones(5),
            scheme=None,
            epsilon=1.0e-8,
            saved_train_loo=saved_train_loo,
        )

    mocked.assert_called_once()
    assert observed.index.tolist() == [1, 4, 3]
    assert np.allclose(
        observed.loc[1, list(ALL_PUBLIC_FEATURES)].to_numpy(float),
        saved_train_loo.loc[1, list(ALL_PUBLIC_FEATURES)].to_numpy(float),
    )
    assert np.allclose(
        observed.loc[3, list(ALL_PUBLIC_FEATURES)].to_numpy(float),
        saved_train_loo.loc[3, list(ALL_PUBLIC_FEATURES)].to_numpy(float),
    )
    assert np.allclose(
        observed.loc[4, list(ALL_PUBLIC_FEATURES)].to_numpy(float),
        unseen_features.loc[4, list(ALL_PUBLIC_FEATURES)].to_numpy(float),
    )
    assert methods[1] == "saved_fit_training_exact_loo"
    assert methods[3] == "saved_fit_training_exact_loo"
    assert methods[4] == "source_training_reference_only"


def test_frozen_probability_uses_saved_coefficients() -> None:
    selected = SelectedModelParameters(
        alpha=0.5,
        lambda_value=10.0,
        threshold=0.5,
        inner_roc_auc=0.7,
        inner_pr_auc=0.6,
    )
    model = FrozenLinearModel(
        model_name="M",
        model_definition="synthetic",
        preprocessor=IdentityPreprocessor(),
        intercept=-0.25,
        coefficients=np.asarray([0.5, -1.0]),
        feature_names=("x1", "x2"),
        selected=selected,
    )
    frame = pd.DataFrame({"x1": [0.0, 2.0], "x2": [0.0, 1.0]})
    observed = frozen_probability(model, frame)
    expected = 1.0 / (1.0 + np.exp(-np.asarray([-0.25, -0.25])))
    assert np.allclose(observed, expected)


def main() -> int:
    tests = [
        test_expected_counts,
        test_evaluation_role,
        test_overlap_audit,
        test_hybrid_public_splits_seen_and_unseen_rows,
        test_frozen_probability_uses_saved_coefficients,
    ]
    for function in tests:
        function()
        print(f"PASS {function.__name__}")
    print(f"Batch 07 focused tests: {len(tests)}/{len(tests)} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
