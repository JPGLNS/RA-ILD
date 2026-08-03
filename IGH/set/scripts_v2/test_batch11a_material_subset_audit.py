#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused acceptance tests for IGH Batch 11A."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.material_subset_audit import (
    MaterialSubsetAuditError,
    audit_material_subsets,
    load_material_subset_audit_config,
    verify_material_subset_audit_marker,
    write_material_subset_audit,
)


def _make_repo(root: Path, *, mismatch: bool = False, duplicate_patient: bool = False) -> Path:
    (root / "data").mkdir(parents=True)
    rows = []
    materials = ["PBMC"] * 8 + ["buffercoat"] * 6
    cohorts = ["RA", "RA", "RA", "RA", "ILD", "ILD", "ILD", "ILD", "RA", "RA", "RA", "ILD", "ILD", "ILD"]
    batches = ["b1", "b1", "b2", "b2", "b1", "b1", "b2", "b2", "b1", "b1", "b2", "b1", "b1", "b2"]
    for i, (material, cohort, batch) in enumerate(zip(materials, cohorts, batches), start=1):
        rows.append({
            "libraryid": f"S{i:02d}",
            "patient": "P01" if duplicate_patient and i == 2 else f"P{i:02d}",
            "cohort": cohort,
            "material": material,
            "batch": batch,
            "sex": "female" if i % 2 else "male",
            "age": 40 + i,
        })
    metadata = pd.DataFrame(rows)
    metadata.iloc[:9].to_csv(root / "data/train_meta.csv", index=False)
    metadata.iloc[9:].to_csv(root / "data/test_meta.csv", index=False)
    matrix = metadata.rename(columns={"libraryid": "sample_id"}).copy()
    matrix["feature_x"] = range(len(matrix))
    if mismatch:
        matrix.loc[0, "material"] = "WRONG"
    matrix.iloc[:9].to_csv(root / "data/train_matrix.csv", index=False)
    matrix.iloc[9:].to_csv(root / "data/test_matrix.csv", index=False)
    config = {
        "material_subset_audit_version": "1.0",
        "audit": {
            "id": "synthetic_material_audit",
            "description": "synthetic",
            "output_dir": "IGH/set/material_subset_audits/synthetic_material_audit",
        },
        "source": {
            "metadata_files": ["data/train_meta.csv", "data/test_meta.csv"],
            "matrix_files": ["data/train_matrix.csv", "data/test_matrix.csv"],
            "require_one_sample_per_patient": True,
        },
        "columns": {
            "metadata_sample_id": "libraryid",
            "matrix_sample_id": "sample_id",
            "patient_id": "patient",
            "label": "cohort",
            "material": "material",
            "batch": "batch",
            "sex": "sex",
            "age": "age",
        },
        "expected": {"samples": 14, "patients": 14, "labels": ["RA", "ILD"]},
        "split_recommendation": {
            "train_fraction": 0.70,
            "minimum_exact_stratum_count": 2,
            "candidate_exact_strata": [["cohort", "batch"], ["cohort"]],
            "balance_categorical": ["batch", "sex"],
            "balance_numeric": ["age"],
        },
    }
    path = root / "audit.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_counts_and_recommendations() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        path = _make_repo(root)
        config = load_material_subset_audit_config(path, repository_root=root)
        result = audit_material_subsets(config)
        counts = dict(zip(result.material_summary["material"], result.material_summary["samples"]))
        assert counts == {"PBMC": 8, "buffercoat": 6}
        recommendations = result.recommendations.set_index("material")
        assert recommendations.loc["PBMC", "recommended_exact_strata"] == "cohort+batch"
        assert recommendations.loc["buffercoat", "recommended_exact_strata"] == "cohort"


def test_identity_mismatch_rejected() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        path = _make_repo(root, mismatch=True)
        config = load_material_subset_audit_config(path, repository_root=root)
        try:
            audit_material_subsets(config)
        except MaterialSubsetAuditError as exc:
            assert "value mismatch" in str(exc)
        else:
            raise AssertionError("Expected metadata/matrix mismatch rejection")


def test_duplicate_patient_rejected() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        path = _make_repo(root, duplicate_patient=True)
        config = load_material_subset_audit_config(path, repository_root=root)
        try:
            audit_material_subsets(config)
        except MaterialSubsetAuditError as exc:
            assert "One sample per patient" in str(exc)
        else:
            raise AssertionError("Expected duplicate patient rejection")


def test_atomic_output_and_marker_verification() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        path = _make_repo(root)
        config = load_material_subset_audit_config(path, repository_root=root)
        result = audit_material_subsets(config)
        output = write_material_subset_audit(config, result)
        marker = output / "AUDIT_COMPLETE.json"
        verified = verify_material_subset_audit_marker(marker, repository_root=root)
        assert verified["status"] == "complete"
        assert (output / "material_split_recommendations.csv").is_file()


def test_existing_output_requires_overwrite() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        path = _make_repo(root)
        config = load_material_subset_audit_config(path, repository_root=root)
        result = audit_material_subsets(config)
        write_material_subset_audit(config, result)
        try:
            write_material_subset_audit(config, result)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Expected existing output rejection")


def test_marker_detects_tampering() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        path = _make_repo(root)
        config = load_material_subset_audit_config(path, repository_root=root)
        result = audit_material_subsets(config)
        output = write_material_subset_audit(config, result)
        target = output / "material_summary.csv"
        target.write_text(target.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
        try:
            verify_material_subset_audit_marker(output / "AUDIT_COMPLETE.json", repository_root=root)
        except MaterialSubsetAuditError as exc:
            assert "SHA256 mismatch" in str(exc)
        else:
            raise AssertionError("Expected marker tamper detection")


def main() -> int:
    tests = [
        test_counts_and_recommendations,
        test_identity_mismatch_rejected,
        test_duplicate_patient_rejected,
        test_atomic_output_and_marker_verification,
        test_existing_output_requires_overwrite,
        test_marker_detects_tampering,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"IGH Batch 11A focused tests: {len(tests)}/{len(tests)} PASS")
    print("IGH_BATCH11A_ACCEPTANCE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
