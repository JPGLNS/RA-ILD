#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structural and IGH-specific validation for framework-upgrade Batch 01."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

SCRIPT_VERSION = "1.0.0-IGH-Batch01-Check"
AA_SUFFIX = "_IGH-without-DJ_CDR3_AA_clone_table.csv"
FORBIDDEN = ("ra_ild_trb", "TRB/set/", "static_tcr", "tcr_", "_TRB")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", default="/data/users/chenhaisheng/RA-ILD")
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def scan_runtime(root):
    paths = []
    for base in (
        root / "IGH/set/src/ra_ild_igh",
        root / "IGH/set/tests",
        root / "IGH/set/configs",
    ):
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}:
                paths.append(path)
    bad = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token in text:
                bad.append((str(path.relative_to(root)), token))
    require(not bad, f"TRB/TCR runtime leakage: {bad[:20]}")
    return len(paths)


def metadata_aa_contract(root):
    all_ids = set()
    for role, name in (
        ("train", "metadata_train_70.csv"),
        ("test", "metadata_test_30.csv"),
    ):
        metadata = pd.read_csv(root / f"IGH/set/{role}/{name}")
        require("libraryid" in metadata.columns, f"{role} metadata missing libraryid")
        meta_ids = set(metadata["libraryid"].astype(str).str.strip())
        aa_dir = root / f"IGH/set/{role}/result/01_AA_clone_table"
        aa_ids = {
            path.name[:-len(AA_SUFFIX)]
            for path in aa_dir.glob(f"*{AA_SUFFIX}")
        }
        require(
            meta_ids == aa_ids,
            f"{role} metadata/AA mismatch; "
            f"missing={sorted(meta_ids - aa_ids)[:10]}, "
            f"extra={sorted(aa_ids - meta_ids)[:10]}",
        )
        require(not (all_ids & aa_ids), f"{role} AA IDs overlap earlier role")
        all_ids |= aa_ids
    require(len(all_ids) == 169, f"Expected 169 AA tables, got {len(all_ids)}")
    return len(all_ids)


def shm_test(root):
    script = root / "IGH/set/scripts/build_01_AA_clone_table.py"
    require(script.is_file(), f"Missing {script}")
    with tempfile.TemporaryDirectory(prefix="igh_shm_test_") as tmp:
        tmp = Path(tmp)
        input_dir = tmp / "input"
        output_dir = tmp / "output"
        input_dir.mkdir()
        sample = "SYNTHETIC_SHM"
        source = (
            input_dir
            / f"{sample}_IGH-without-DJ_CDR3_NT_frequency_error_correct.csv"
        )
        source.write_text(
            "TGTGCT 10 0.1 0.2\n"
            "TGCGCC 20 0.3 0.4\n",
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(script),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--min-nt-clones-warn",
            "0",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        require(
            completed.returncode == 0,
            "SHM run failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr,
        )
        result = pd.read_csv(output_dir / f"{sample}{AA_SUFFIX}")
        require(len(result) == 1, f"Expected 1 row, got {len(result)}")
        row = result.iloc[0]
        require(str(row["cdr3_aa"]) == "CA", "AA mismatch")
        require(int(row["nt_clone_number"]) == 2, "nt_clone_number mismatch")
        require(int(row["read_count"]) == 30, "read_count mismatch")
        require(
            math.isclose(
                float(row["input_frequency_sum"]),
                0.4,
                abs_tol=1e-12,
            ),
            "frequency sum mismatch",
        )
        require(
            math.isclose(
                float(row["input_cell_frequency_sum"]),
                0.6,
                abs_tol=1e-12,
            ),
            "cell frequency mismatch",
        )
        require(
            math.isclose(float(row["frequency_norm"]), 1.0, abs_tol=1e-12),
            "frequency_norm mismatch",
        )
        require(str(row["top_nt_cdr3"]) == "TGCGCC", "top_nt_cdr3 mismatch")


def main():
    args = parse_args()
    root = Path(args.repository_root).expanduser().resolve()
    for path in (
        root / "IGH/set/configs/igh_baseline_m2_v1.yaml",
        root / "IGH/set/src/ra_ild_igh/config.py",
        root / "IGH/set/scripts_v2/validate_experiment_config.py",
        root / "IGH/set/tests/test_config.py",
        root / "IGH/set/README_V2.md",
    ):
        require(path.is_file(), f"Missing Batch01 file: {path}")

    sys.path.insert(0, str(root / "IGH/set/src"))
    from ra_ild_igh.config import load_experiment_config
    from ra_ild_igh.specifications import select_static_igh_features

    config = load_experiment_config(
        root / "IGH/set/configs/igh_baseline_m2_v1.yaml"
    )
    require(config.raw["experiment"]["receptor"] == "IGH", "receptor mismatch")
    require(config.raw["data"]["train"]["expected_samples"] == 120, "train count")
    require(config.raw["data"]["test"]["expected_samples"] == 49, "test count")

    manifest = pd.read_csv(config.path("data.train.feature_manifest"))
    base = pd.read_csv(config.path("data.train.base_matrix"), nrows=1)
    static = select_static_igh_features(
        manifest,
        available_columns=base.columns,
        require_all_available=True,
    )
    require(
        len(static) == 1083,
        f"Expected 1083 static features, got {len(static)}",
    )

    scanned = scan_runtime(root)
    aa_count = metadata_aa_contract(root)
    shm_test(root)

    print(
        json.dumps(
            {
                "status": "PASS",
                "script_version": SCRIPT_VERSION,
                "runtime_files_scanned": scanned,
                "full_cohort_aa_tables": aa_count,
                "static_igh_features": len(static),
                "shm_nt_to_aa_aggregation": "PASS",
                "historical_results_modified": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
