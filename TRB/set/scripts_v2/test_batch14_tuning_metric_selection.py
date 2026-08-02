#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acceptance checks for Batch 14 configurable tuning-metric selection."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SET_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = SET_DIR.parent.parent
TEST_FILE = SET_DIR / "tests" / "test_tuning_metric_selection.py"

REQUIRED_TEXT = {
    SET_DIR / "src" / "ra_ild_trb" / "config.py": (
        "tuning_primary_metric",
        "pooled_log_loss_brier_roc_lambda_alpha",
    ),
    SET_DIR / "src" / "ra_ild_trb" / "specifications.py": (
        "TUNING_RANKING_POLICIES",
        "candidate_selection_policy_name",
    ),
    SET_DIR / "src" / "ra_ild_trb" / "nested_cv.py": (
        "pooled_inner_log_loss",
        "inner_selected_brier_score",
    ),
    SET_DIR / "src" / "ra_ild_trb" / "scheme_management.py": (
        "_normalize_tuning_selection",
        "tuning_primary_metric",
    ),
    SET_DIR / "src" / "ra_ild_trb" / "repeated_holdout_summary.py": (
        "inner_selected_log_loss",
        "tuning_primary_metric",
    ),
    SCRIPT_DIR / "run_nested_cv_task.py": (
        "tuning_primary_metric",
        "candidate_selection_policy",
    ),
}


def compile_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")


def main() -> int:
    for path, tokens in REQUIRED_TEXT.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                raise AssertionError(f"{path} is missing token: {token}")
        compile_source(path)

    if not TEST_FILE.is_file():
        raise FileNotFoundError(TEST_FILE)
    compile_source(TEST_FILE)

    completed = subprocess.run(
        [sys.executable, str(TEST_FILE)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        raise AssertionError(
            f"Batch 14 unit tests failed with return code {completed.returncode}"
        )

    print("TRB_BATCH14_ACCEPTANCE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
