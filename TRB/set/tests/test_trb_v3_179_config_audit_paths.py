#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for algorithm-specific TRB V3 output-root audit rules."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "TRB/set/scripts_v2"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_trb_v3_179_igh_aligned_configs import _expected_outer_output_root


def test_elastic_net_family_uses_scheme_manager_layout() -> None:
    experiment = "trb_v3_179_igh_aligned_total_m0_m3_repeat100_top10000_v1"
    expected = f"TRB/set/experiments/{experiment}/02_tasks/outer_nested_cv"
    for algorithm in ("elastic_net", "ridge", "lasso"):
        assert _expected_outer_output_root(algorithm, experiment) == expected


def test_dedicated_engines_use_dedicated_layout() -> None:
    experiment = "trb_v3_179_igh_aligned_total_m0_m3_repeat100_top10000_v1_x"
    expected = f"TRB/set/experiments/{experiment}/01_outer_tasks"
    for algorithm in ("linear_svm", "xgboost"):
        assert _expected_outer_output_root(algorithm, experiment) == expected


def test_unknown_algorithm_is_rejected() -> None:
    try:
        _expected_outer_output_root("unknown", "experiment")
    except RuntimeError:
        return
    raise AssertionError("Unsupported algorithm should raise AuditError")


if __name__ == "__main__":
    test_elastic_net_family_uses_scheme_manager_layout()
    test_dedicated_engines_use_dedicated_layout()
    test_unknown_algorithm_is_rejected()
    print("TRB V3 config audit output-path regression tests: PASS")
