#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate a TRB V2 experiment YAML without running a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.config import ConfigError, load_experiment_config  # noqa: E402

PATH_SPECS = (
    ("data.train.metadata", "file"),
    ("data.train.base_matrix", "file"),
    ("data.train.feature_manifest", "file"),
    ("data.train.aa_clone_table_dir", "dir"),
    ("data.train.public_catalog", "file"),
    ("data.train.reference_definition", "file"),
    ("data.test.metadata", "file"),
    ("data.test.final_matrix", "file"),
    ("data.test.reference_features", "file"),
    ("data.test.reference_definition_used", "file"),
    ("cross_validation.outer_assignments", "file"),
    ("cross_validation.inner_assignments", "file"),
    ("final_model.bundle", "file"),
    ("final_model.configuration", "file"),
    ("final_model.reference_masks", "file"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a TRB V2 YAML experiment configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--check-paths", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_experiment_config(
            Path(args.config),
            repository_root=Path(args.repository_root) if args.repository_root else None,
        )
        checked = {}
        if args.check_paths:
            for key, expected_type in PATH_SPECS:
                checked[key] = str(
                    config.path(key, must_exist=True, expect=expected_type)
                )

        summary = {
            "status": "PASS",
            "experiment_id": config.experiment_id,
            "schema_version": str(config.raw["schema_version"]),
            "repository_root": str(config.repository_root),
            "models": list(config.section("models")),
            "path_checks_enabled": bool(args.check_paths),
            "checked_path_count": len(checked),
            "checked_paths": checked,
        }
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print("TRB V2 configuration: PASS")
            print(f"Experiment: {config.experiment_id}")
            print(f"Repository root: {config.repository_root}")
            print("Models:")
            for model in summary["models"]:
                print(f"  - {model}")
            if args.check_paths:
                print(f"Existing paths checked: {len(checked)}")
        return 0
    except (ConfigError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"TRB V2 configuration: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
