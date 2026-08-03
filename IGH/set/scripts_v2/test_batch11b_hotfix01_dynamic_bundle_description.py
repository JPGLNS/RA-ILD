#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused acceptance tests for IGH Batch 11B Hotfix 01."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.repeated_holdout_training import generate_scheme_yaml


def test_generated_description_uses_actual_design() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        template = root / "template.yaml"
        generated = root / "generated.yaml"
        template.write_text(
            yaml.safe_dump(
                {
                    "scheme_version": "1.0",
                    "scheme": {
                        "id": "template",
                        "description": "template",
                        "status": "development",
                        "base_config": "dummy.yaml",
                        "output_root": "dummy",
                    },
                    "overrides": {},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        def p(name: str) -> Path:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        spec = SimpleNamespace(
            scheme_template=template,
            scheme_id="synthetic_repeat2",
            scheme_output_root="IGH/set/experiments/synthetic_repeat2",
            generated_scheme=generated,
            repository_root=root,
            full_metadata_path=p("bundle/full_metadata.csv"),
            full_base_matrix_path=p("bundle/full_matrix.csv.gz"),
            feature_manifest=p("feature_manifest.csv"),
            output_root=p("bundle"),
            catalog_path=p("bundle/catalog.csv.gz"),
            marker_path=p("bundle/TRAINING_BUNDLE_FROZEN.json"),
            presence_path=p("bundle/presence.npz"),
            frequency_path=p("bundle/frequency.npz"),
            cache_metadata_path=p("bundle/cache.json"),
            inner_folds=5,
            inner_strata=("cohort",),
            balance_categorical=("batch", "sex"),
            outer_assignments_path=p("bundle/outer.csv"),
            inner_assignments_path=p("bundle/inner.csv.gz"),
            split_assignments=p("splits/assignments.csv"),
            split_frozen_marker=p("splits/SPLITS_FROZEN.json"),
            split_train_role="train",
            split_holdout_role="holdout",
        )
        rows = []
        for repeat_index in (1, 2):
            for i in range(4):
                rows.append(
                    {
                        "repeat_index": repeat_index,
                        "role": "train",
                        "split_set_id": "synthetic_split_set",
                        "sample_id": f"T{repeat_index}_{i}",
                    }
                )
            for i in range(2):
                rows.append(
                    {
                        "repeat_index": repeat_index,
                        "role": "holdout",
                        "split_set_id": "synthetic_split_set",
                        "sample_id": f"H{repeat_index}_{i}",
                    }
                )
        assignments = pd.DataFrame(rows)
        result = generate_scheme_yaml(
            spec,
            {"matrix_shape": [6, 10], "catalog_size": 123},
            assignments,
        )
        description = str(result["scheme"]["description"])
        assert description.startswith("2 frozen 4/2 repeated holdouts"), description
        assert "Three frozen 120/49" not in description, description
        loaded = yaml.safe_load(generated.read_text(encoding="utf-8"))
        assert loaded["scheme"]["description"] == description


def test_cli_text_is_generic() -> None:
    script = SCRIPT_DIR / "prepare_repeated_holdout_training.py"
    text = script.read_text(encoding="utf-8")
    assert "three frozen repeated holdouts" not in text.lower()
    assert "Batch 04 repeated-holdout training preparation" not in text


def main() -> int:
    tests = [
        test_generated_description_uses_actual_design,
        test_cli_text_is_generic,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"IGH Batch 11B Hotfix 01 focused tests: {len(tests)}/{len(tests)} PASS")
    print("IGH_BATCH11B_HOTFIX01_ACCEPTANCE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
