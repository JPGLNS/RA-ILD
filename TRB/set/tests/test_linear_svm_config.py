#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

from ra_ild_trb.linear_svm_config import (
    LinearSVMConfigError,
    expand_linear_svm_candidates,
)


class LinearSVMConfigTests(unittest.TestCase):
    def test_main_block_expands_eight_candidates(self) -> None:
        engine = {
            "engine": "linear_svc",
            "candidate_blocks": [
                {
                    "id": "main",
                    "priority": 1,
                    "penalty": "l2",
                    "loss": "squared_hinge",
                    "class_weight_grid": ["balanced"],
                    "C_grid": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                    "dual": "auto",
                }
            ],
        }
        candidates = expand_linear_svm_candidates(engine)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(candidates[0].C, 1e-4)
        self.assertEqual(candidates[-1].C, 1000.0)

    def test_multiple_blocks_are_supported(self) -> None:
        engine = {
            "engine": "linear_svc",
            "candidate_blocks": [
                {
                    "id": "a",
                    "penalty": "l2",
                    "loss": "squared_hinge",
                    "class_weight_grid": ["balanced", "none"],
                    "C_grid": [0.1, 1],
                    "dual": "auto",
                },
                {
                    "id": "b",
                    "penalty": "l1",
                    "loss": "squared_hinge",
                    "class_weight_grid": ["balanced"],
                    "C_grid": [0.1],
                    "dual": False,
                },
            ],
        }
        self.assertEqual(len(expand_linear_svm_candidates(engine)), 5)

    def test_illegal_cartesian_combination_is_rejected(self) -> None:
        engine = {
            "engine": "linear_svc",
            "candidate_blocks": [
                {
                    "id": "bad",
                    "penalty_grid": ["l1", "l2"],
                    "loss_grid": ["hinge"],
                    "class_weight_grid": ["balanced"],
                    "C_grid": [1],
                    "dual": "auto",
                }
            ],
        }
        with self.assertRaises(LinearSVMConfigError):
            expand_linear_svm_candidates(engine)


if __name__ == "__main__":
    unittest.main(verbosity=2)
