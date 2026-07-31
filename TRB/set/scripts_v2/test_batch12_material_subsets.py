#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused synthetic tests for Batch 12 material-defined subcohorts."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.cohort_subset import (  # noqa: E402
    CohortSubsetError,
    load_cohort_subset_spec,
    prepare_cohort_subset,
    verify_frozen_subset,
    write_frozen_cohort_subset,
)


def _population_rows() -> pd.DataFrame:
    groups = [
        ("PBMC", "batch1", "ILD", 5),
        ("PBMC", "batch1", "RA", 8),
        ("PBMC", "batch2", "ILD", 12),
        ("PBMC", "batch2", "RA", 11),
        ("PBMC", "batch3", "ILD", 14),
        ("PBMC", "batch3", "RA", 28),
        ("PBMC", "batch4", "ILD", 13),
        ("PBMC", "batch4", "RA", 20),
        ("buffercoat", "batch2", "ILD", 4),
        ("buffercoat", "batch2", "RA", 13),
        ("buffercoat", "batch4", "ILD", 1),
        ("buffercoat", "batch4", "RA", 2),
        ("buffercoat", "batch5", "ILD", 17),
        ("buffercoat", "batch5", "RA", 26),
    ]
    rows = []
    index = 0
    for material, batch, cohort, count in groups:
        for _ in range(count):
            index += 1
            rows.append(
                {
                    "libraryid": f"S{index:03d}",
                    "patient": f"P{index:03d}",
                    "cohort": cohort,
                    "material": material,
                    "batch": batch,
                    "sex": "female" if index % 8 else "male",
                    "age": 30 + (index % 45),
                }
            )
    frame = pd.DataFrame(rows)
    assert frame.shape[0] == 174
    return frame


def _matrix(metadata: pd.DataFrame) -> pd.DataFrame:
    # 7 metadata/identifier columns + 1083 static columns = 1090 columns.
    base = metadata.rename(columns={"libraryid": "sample_id"})[
        ["sample_id", "cohort", "patient", "age", "sex", "material", "batch"]
    ].copy()
    feature_count_remaining = 1082
    values = np.arange(len(base), dtype=float)[:, None] * 0.001
    offsets = np.arange(feature_count_remaining, dtype=float)[None, :] * 0.0001
    features = pd.DataFrame(
        values + offsets,
        columns=[f"feature_{index:04d}" for index in range(feature_count_remaining)],
    )
    features.insert(0, "aa_clone_number", np.arange(1, len(base) + 1, dtype=float) + 100.0)
    base = pd.concat([base.reset_index(drop=True), features], axis=1)
    assert base.shape[1] == 1090
    return base


def _build_repo(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = _population_rows().sample(frac=1.0, random_state=20260731).reset_index(drop=True)
    matrix = _matrix(metadata)
    train_meta = metadata.iloc[:123].copy()
    test_meta = metadata.iloc[123:].copy()
    train_matrix = matrix.iloc[:123].copy()
    test_matrix = matrix.iloc[123:].copy()

    (root / "TRB/set/train/result/04_final_feature_matrix").mkdir(parents=True)
    (root / "TRB/set/test/result/04_final_feature_matrix").mkdir(parents=True)
    train_meta.to_csv(root / "TRB/set/train/metadata_train_70.csv", index=False)
    test_meta.to_csv(root / "TRB/set/test/metadata_test_30.csv", index=False)
    train_matrix.to_csv(
        root / "TRB/set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv",
        index=False,
    )
    test_matrix.to_csv(
        root / "TRB/set/test/result/04_final_feature_matrix/04_test_final_feature_matrix.csv",
        index=False,
    )
    return metadata, train_matrix, test_matrix


def _spec(
    root: Path,
    *,
    subset_id: str,
    include_values: Iterable[str],
    expected_samples: int,
    expected_labels: Dict[str, int],
    expected_filters: Optional[Dict[str, int]],
) -> Path:
    path = root / f"TRB/set/configs/material_subsets/{subset_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "material_subset_version": "1.0",
        "subset": {
            "id": subset_id,
            "description": "synthetic Batch 12 test",
            "output_root": f"TRB/set/cohort_subsets/{subset_id}",
        },
        "source": {
            "metadata_files": [
                "TRB/set/train/metadata_train_70.csv",
                "TRB/set/test/metadata_test_30.csv",
            ],
            "train_base_matrix": (
                "TRB/set/train/result/04_final_feature_matrix/"
                "04_train_base_feature_matrix.csv"
            ),
            "test_final_matrix": (
                "TRB/set/test/result/04_final_feature_matrix/"
                "04_test_final_feature_matrix.csv"
            ),
            "metadata_sample_id_column": "libraryid",
            "matrix_sample_id_column": "sample_id",
            "patient_id_column": "patient",
            "label_column": "cohort",
            "require_one_sample_per_patient": True,
        },
        "filter": {
            "column": "material",
            "include_values": list(include_values),
            "case_insensitive": True,
            "strip_whitespace": True,
        },
        "expected": {
            "samples": expected_samples,
            "patients": expected_samples,
            "label_counts": expected_labels,
        },
    }
    if expected_filters is not None:
        payload["expected"]["filter_counts"] = expected_filters
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_pbmc_subset_realistic_174_by_1090() -> None:
    with tempfile.TemporaryDirectory(prefix="trb_batch12_pbmc_") as temp:
        root = Path(temp)
        _build_repo(root)
        path = _spec(
            root,
            subset_id="test_pbmc_only",
            include_values=["PBMC"],
            expected_samples=111,
            expected_labels={"RA": 67, "ILD": 44},
            expected_filters={"PBMC": 111},
        )
        spec = load_cohort_subset_spec(path, repository_root=root)
        result = prepare_cohort_subset(spec)
        assert result.metadata.shape[0] == 111
        assert result.train_matrix.shape[0] + result.test_matrix.shape[0] == 111
        assert result.train_matrix.shape[1] == 1090
        assert result.test_matrix.shape[1] == 1090
        assert set(result.metadata["material"]) == {"PBMC"}
        assert result.metadata["cohort"].value_counts().to_dict() == {"RA": 67, "ILD": 44}
        assert result.sample_audit["sample_id"].tolist() == result.metadata["libraryid"].tolist()


def test_buffercoat_subset_realistic_counts() -> None:
    with tempfile.TemporaryDirectory(prefix="trb_batch12_buffer_") as temp:
        root = Path(temp)
        _build_repo(root)
        path = _spec(
            root,
            subset_id="test_buffercoat_only",
            include_values=["buffercoat"],
            expected_samples=63,
            expected_labels={"RA": 41, "ILD": 22},
            expected_filters={"buffercoat": 63},
        )
        spec = load_cohort_subset_spec(path, repository_root=root)
        result = prepare_cohort_subset(spec)
        assert result.metadata.shape[0] == 63
        assert set(result.metadata["material"]) == {"buffercoat"}
        assert result.metadata["cohort"].value_counts().to_dict() == {"RA": 41, "ILD": 22}
        table = pd.crosstab(result.metadata["batch"], result.metadata["cohort"])
        assert int(table.loc["batch4", "ILD"]) == 1


def test_case_insensitive_and_whitespace_filter() -> None:
    with tempfile.TemporaryDirectory(prefix="trb_batch12_case_") as temp:
        root = Path(temp)
        metadata, _, _ = _build_repo(root)
        train_path = root / "TRB/set/train/metadata_train_70.csv"
        test_path = root / "TRB/set/test/metadata_test_30.csv"
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        for frame in (train, test):
            mask = frame["material"].eq("PBMC")
            indices = frame.index[mask]
            frame.loc[indices[::2], "material"] = " pbmc "
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        # Keep matrices synchronized with the altered metadata strings.
        lookup = pd.concat([train, test]).set_index("libraryid")["material"]
        for matrix_path in (
            root / "TRB/set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv",
            root / "TRB/set/test/result/04_final_feature_matrix/04_test_final_feature_matrix.csv",
        ):
            matrix = pd.read_csv(matrix_path)
            matrix["material"] = matrix["sample_id"].map(lookup)
            matrix.to_csv(matrix_path, index=False)
        path = _spec(
            root,
            subset_id="test_casefold_pbmc",
            include_values=["PbMc"],
            expected_samples=111,
            expected_labels={"RA": 67, "ILD": 44},
            expected_filters=None,
        )
        result = prepare_cohort_subset(
            load_cohort_subset_spec(path, repository_root=root)
        )
        assert len(result.metadata) == 111


def test_unknown_material_is_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="trb_batch12_unknown_") as temp:
        root = Path(temp)
        _build_repo(root)
        path = _spec(
            root,
            subset_id="test_unknown_material",
            include_values=["serum"],
            expected_samples=1,
            expected_labels={"RA": 1},
            expected_filters=None,
        )
        spec = load_cohort_subset_spec(path, repository_root=root)
        try:
            prepare_cohort_subset(spec)
        except CohortSubsetError as exc:
            assert "not observed" in str(exc)
        else:
            raise AssertionError("Unknown material should be rejected")


def test_matrix_metadata_mismatch_is_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="trb_batch12_mismatch_") as temp:
        root = Path(temp)
        _build_repo(root)
        train_path = root / "TRB/set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv"
        train = pd.read_csv(train_path)
        train.loc[0, "material"] = "buffercoat" if train.loc[0, "material"] == "PBMC" else "PBMC"
        train.to_csv(train_path, index=False)
        path = _spec(
            root,
            subset_id="test_matrix_mismatch",
            include_values=["PBMC"],
            expected_samples=111,
            expected_labels={"RA": 67, "ILD": 44},
            expected_filters={"PBMC": 111},
        )
        spec = load_cohort_subset_spec(path, repository_root=root)
        try:
            prepare_cohort_subset(spec)
        except CohortSubsetError as exc:
            assert "disagree" in str(exc)
        else:
            raise AssertionError("Matrix/metadata mismatch should be rejected")


def test_frozen_marker_hash_verification() -> None:
    with tempfile.TemporaryDirectory(prefix="trb_batch12_hash_") as temp:
        root = Path(temp)
        _build_repo(root)
        path = _spec(
            root,
            subset_id="test_frozen_hash",
            include_values=["PBMC"],
            expected_samples=111,
            expected_labels={"RA": 67, "ILD": 44},
            expected_filters={"PBMC": 111},
        )
        spec = load_cohort_subset_spec(path, repository_root=root)
        result = prepare_cohort_subset(spec)
        marker = write_frozen_cohort_subset(result, spec)
        assert marker["status"] == "FROZEN"
        verified = verify_frozen_subset(spec.marker_output, repository_root=root)
        assert verified["subset_id"] == "test_frozen_hash"
        metadata = pd.read_csv(spec.metadata_output)
        metadata.loc[0, "age"] = 999
        metadata.to_csv(spec.metadata_output, index=False)
        try:
            verify_frozen_subset(spec.marker_output, repository_root=root)
        except CohortSubsetError as exc:
            assert "hash mismatch" in str(exc)
        else:
            raise AssertionError("Changed frozen output should fail hash verification")


def test_frozen_subset_is_immutable() -> None:
    with tempfile.TemporaryDirectory(prefix="trb_batch12_immutable_") as temp:
        root = Path(temp)
        _build_repo(root)
        path = _spec(
            root,
            subset_id="test_immutable",
            include_values=["buffercoat"],
            expected_samples=63,
            expected_labels={"RA": 41, "ILD": 22},
            expected_filters={"buffercoat": 63},
        )
        spec = load_cohort_subset_spec(path, repository_root=root)
        result = prepare_cohort_subset(spec)
        write_frozen_cohort_subset(result, spec)
        try:
            write_frozen_cohort_subset(result, spec, replace_incomplete=True)
        except FileExistsError as exc:
            assert "immutable" in str(exc)
        else:
            raise AssertionError("Frozen subset should not be replaceable")


def _package_root() -> Path:
    return SCRIPT_DIR.parents[2]


def test_existing_v1_contracts_are_reused_unchanged() -> None:
    root = _package_root()
    pbmc_split = yaml.safe_load(
        (root / "TRB/set/configs/repeated_holdout_splits/ra_ild_pbmc_repeat100_v1.yaml")
        .read_text(encoding="utf-8")
    )
    buffer_split = yaml.safe_load(
        (root / "TRB/set/configs/repeated_holdout_splits/ra_ild_buffercoat_repeat100_v1.yaml")
        .read_text(encoding="utf-8")
    )
    pbmc_training = yaml.safe_load(
        (root / "TRB/set/configs/repeated_holdout_training/trb_pbmc_repeat100_training_v1.yaml")
        .read_text(encoding="utf-8")
    )
    assert pbmc_split["split_set_version"] == "1.0"
    assert buffer_split["split_set_version"] == "1.0"
    assert pbmc_training["training_bundle_version"] == "1.0"
    assert pbmc_split["split"]["train_size"] + pbmc_split["split"]["holdout_size"] == 111
    assert buffer_split["split"]["train_size"] + buffer_split["split"]["holdout_size"] == 63
    assert pbmc_training["source"]["train_base_matrix"].endswith(
        "subset_train_base_matrix.csv.gz"
    )


def test_material_specific_stratification_rules() -> None:
    root = _package_root()
    pbmc = yaml.safe_load(
        (root / "TRB/set/configs/repeated_holdout_splits/ra_ild_pbmc_repeat100_v1.yaml")
        .read_text(encoding="utf-8")
    )
    buffercoat = yaml.safe_load(
        (root / "TRB/set/configs/repeated_holdout_splits/ra_ild_buffercoat_repeat100_v1.yaml")
        .read_text(encoding="utf-8")
    )
    assert pbmc["split"]["exact_strata"] == ["cohort", "batch"]
    assert "material" not in pbmc["split"]["balance_categorical"]
    assert buffercoat["split"]["exact_strata"] == ["cohort"]
    assert "batch" in buffercoat["split"]["balance_categorical"]


def test_material_is_not_a_subset_model_predictor() -> None:
    root = _package_root()
    base_template = yaml.safe_load(
        (
            root
            / "TRB/set/configs/schemes/"
            "trb_scheme_material_homogeneous_no_clinical_template.yaml"
        ).read_text(encoding="utf-8")
    )
    base_models = base_template["replacements"]["models"]
    assert set(base_models) == {"M1_static_tcr", "M2_static_tcr_public"}
    for model in base_models.values():
        assert "material" not in model.get("categorical", [])
        assert "material" not in model.get("numeric", [])

    for filename in (
        "trb_scheme_pbmc_a1_a6_repeat100_v1.yaml",
        "trb_scheme_buffercoat_a1_a6_repeat100_v1.yaml",
    ):
        payload = yaml.safe_load(
            (root / "TRB/set/configs/schemes" / filename).read_text(encoding="utf-8")
        )
        for model in payload["models"].values():
            assert "material" not in model.get("categorical", [])
            assert "material" not in model.get("numeric", [])

    for filename in (
        "trb_pbmc_repeat100_training_v1.yaml",
        "trb_buffercoat_repeat100_training_v1.yaml",
    ):
        training = yaml.safe_load(
            (
                root / "TRB/set/configs/repeated_holdout_training" / filename
            ).read_text(encoding="utf-8")
        )
        assert training["scheme"]["template"].endswith(
            "trb_scheme_material_homogeneous_no_clinical_template.yaml"
        )


def main() -> int:
    tests = [
        test_pbmc_subset_realistic_174_by_1090,
        test_buffercoat_subset_realistic_counts,
        test_case_insensitive_and_whitespace_filter,
        test_unknown_material_is_rejected,
        test_matrix_metadata_mismatch_is_rejected,
        test_frozen_marker_hash_verification,
        test_frozen_subset_is_immutable,
        test_existing_v1_contracts_are_reused_unchanged,
        test_material_specific_stratification_rules,
        test_material_is_not_a_subset_model_predictor,
    ]
    for function in tests:
        function()
        print(f"PASS {function.__name__}")
    print(f"Batch 12 focused tests: {len(tests)}/{len(tests)} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
