#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acceptance test for Batch 14 hotfix 01."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
SOURCE = ROOT / (
    "TRB/set/configs/schemes/"
    "trb_scheme_pbmc_m0_m2_repeat100_fullgrid_v1.yaml"
)
TEMP_SOURCE = ROOT / (
    "TRB/set/configs/schemes/"
    "_batch14_hotfix01_logloss_tmp.yaml"
)
TEMP_ID = "batch14_hotfix01_logloss_tmp"
TEMP_OUTPUT = ROOT / "TRB/set/experiments" / TEMP_ID
PREPARER = ROOT / "TRB/set/scripts_v2/prepare_feature_ablation_scheme.py"


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    source["scheme"]["id"] = TEMP_ID
    source["scheme"]["description"] = (
        "Temporary Batch 14 hotfix 01 log-loss acceptance scheme."
    )
    source["scheme"]["output_root"] = (
        f"TRB/set/experiments/{TEMP_ID}"
    )
    source["model_selection"] = {
        "tuning_primary_metric": "log_loss"
    }

    try:
        if TEMP_OUTPUT.exists():
            shutil.rmtree(TEMP_OUTPUT)
        TEMP_SOURCE.write_text(
            yaml.safe_dump(
                source,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                PYTHON,
                str(PREPARER),
                "--scheme",
                str(TEMP_SOURCE.relative_to(ROOT)),
                "--repository-root",
                str(ROOT),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        print(completed.stdout, end="")

        resolved_path = (
            TEMP_OUTPUT / "00_config/resolved_config.yaml"
        )
        marker_path = (
            TEMP_OUTPUT
            / "00_config/FEATURE_ABLATION_SCHEME_PREPARED.json"
        )
        resolved = yaml.safe_load(
            resolved_path.read_text(encoding="utf-8")
        )

        selection = resolved["model_selection"]
        nested = resolved["nested_cv"]

        assert selection["tuning_primary_metric"] == "log_loss"
        assert selection["candidate_sort"] == [
            {
                "field": "pooled_inner_log_loss",
                "ascending": True,
            },
            {
                "field": "pooled_inner_brier_score",
                "ascending": True,
            },
            {
                "field": "pooled_inner_roc_auc",
                "ascending": False,
            },
            {"field": "lambda", "ascending": False},
            {
                "field": "l1_ratio_alpha",
                "ascending": False,
            },
        ]
        assert (
            nested["candidate_selection_policy"]
            == "pooled_log_loss_brier_roc_lambda_alpha"
        )
        assert marker_path.is_file()

        print("TRB_BATCH14_HOTFIX01_ACCEPTANCE_PASS")
        return 0
    finally:
        TEMP_SOURCE.unlink(missing_ok=True)
        if TEMP_OUTPUT.exists():
            shutil.rmtree(TEMP_OUTPUT)


if __name__ == "__main__":
    raise SystemExit(main())
