#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate an IGH repeated-holdout training bundle for configurable repeat counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.paths import find_repository_root
from ra_ild_igh.repeated_holdout_training import (
    discover_aa_clone_tables,
    generate_v2_assignments,
    load_frozen_assignments,
    load_training_bundle_spec,
    prepare_full_static_matrix,
)
from ra_ild_igh.scheme_management import build_prepared_scheme

EXPECTED_SAMPLES = 169
EXPECTED_STATIC = 1083
FIXED_TRAIN_ID = "MGI260202N01-31"
EXPECTED_MODELS = [
    "M1_static_igh",
    "M2_static_igh_public",
    "M3_static_igh_public_material",
]
DYNAMIC_PUBLIC = {
    "all_ref_public_clone_ratio",
    "all_ref_public_frequency_sum",
    "shared_ref_clone_ratio",
    "shared_ref_frequency_sum",
    "RA_specific_ref_clone_ratio",
    "RA_specific_ref_frequency_sum",
    "ILD_specific_ref_clone_ratio",
    "ILD_specific_ref_frequency_sum",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--mode", choices=("dry-run", "frozen"), default="dry-run")
    return parser.parse_args()


def validate_structure(spec):
    assignments, split_marker = load_frozen_assignments(spec)

    split_set_id = str(split_marker.get("split_set_id", "")).strip()
    assert split_set_id, "Frozen split marker is missing split_set_id"
    assert set(assignments["split_set_id"].astype(str)) == {split_set_id}

    expected_splits = int(assignments["repeat_index"].nunique())
    assert expected_splits >= 1

    role_counts = (
        assignments.groupby(["repeat_index", "role"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    assert list(role_counts.index) == list(range(1, expected_splits + 1))
    assert spec.split_train_role in role_counts.columns
    assert spec.split_holdout_role in role_counts.columns

    train_sizes = role_counts[spec.split_train_role].astype(int)
    holdout_sizes = role_counts[spec.split_holdout_role].astype(int)
    assert train_sizes.nunique() == 1, train_sizes.tolist()
    assert holdout_sizes.nunique() == 1, holdout_sizes.tolist()

    expected_train = int(train_sizes.iloc[0])
    expected_holdout = int(holdout_sizes.iloc[0])

    combined, metadata, static_features = prepare_full_static_matrix(spec, assignments)
    aa_manifest = discover_aa_clone_tables(
        spec, combined["sample_id"].astype(str).tolist()
    )
    outer, inner, selected, audit = generate_v2_assignments(spec, assignments)

    assert len(combined) == EXPECTED_SAMPLES
    assert len(metadata) == EXPECTED_SAMPLES
    assert len(static_features) == EXPECTED_STATIC
    assert len(aa_manifest) == EXPECTED_SAMPLES
    assert not (DYNAMIC_PUBLIC & set(combined.columns))
    assert combined["sample_id"].is_unique
    assert set(combined["cohort"].astype(str).str.upper()) == {"RA", "ILD"}

    singleton = assignments.loc[
        assignments["sample_id"].astype(str) == FIXED_TRAIN_ID
    ]
    assert len(singleton) == expected_splits
    assert singleton["role"].astype(str).eq(spec.split_train_role).all()

    assert len(outer) == EXPECTED_SAMPLES * expected_splits
    assert set(outer["outer_fold"].astype(int)) == {1, 2}
    for repeat, frame in outer.groupby("outer_repeat"):
        counts = frame["outer_fold"].value_counts().to_dict()
        assert counts == {2: expected_train, 1: expected_holdout}, (repeat, counts)

    holdout_pairs = set(
        outer.loc[
            outer["outer_fold"].astype(int) == 1,
            ["outer_repeat", "sample_id"],
        ].itertuples(index=False, name=None)
    )
    inner_pairs = set(
        inner[["outer_repeat", "sample_id"]].itertuples(index=False, name=None)
    )
    assert not (holdout_pairs & inner_pairs)

    assert len(inner) == expected_splits * expected_train
    assert len(selected) == expected_splits
    assert len(audit) == expected_splits * spec.inner_folds
    for _, frame in inner.groupby("outer_repeat"):
        assert set(frame["inner_fold"].astype(int)) == set(
            range(1, spec.inner_folds + 1)
        )
        assert len(frame) == expected_train

    return {
        "combined": combined,
        "outer": outer,
        "inner": inner,
        "selected": selected,
        "split_set_id": split_set_id,
        "expected_splits": expected_splits,
        "expected_train": expected_train,
        "expected_holdout": expected_holdout,
    }


def validate_frozen(spec, structure):
    marker = json.loads(spec.marker_path.read_text(encoding="utf-8"))

    split_set_id = structure["split_set_id"]
    expected_splits = structure["expected_splits"]
    expected_train = structure["expected_train"]
    expected_holdout = structure["expected_holdout"]

    assert marker["status"] == "FROZEN"
    assert marker["bundle_id"] == spec.bundle_id
    assert marker["split_set_id"] == split_set_id
    assert marker["samples"] == EXPECTED_SAMPLES
    assert marker["static_feature_count"] == EXPECTED_STATIC
    assert marker["split_count"] == expected_splits
    assert marker["train_sizes"] == [expected_train] * expected_splits
    assert marker["holdout_sizes"] == [expected_holdout] * expected_splits
    assert marker["split_frozen_marker_sha256"] == sha256(spec.split_frozen_marker)

    for item in marker["files"].values():
        file_path = spec.repository_root / item["path"]
        assert file_path.is_file(), file_path
        assert sha256(file_path) == item["sha256"], file_path

    full = pd.read_csv(spec.full_base_matrix_path)
    cache = json.loads(spec.cache_metadata_path.read_text(encoding="utf-8"))
    presence = sparse.load_npz(spec.presence_path).tocsr()
    frequency = sparse.load_npz(spec.frequency_path).tocsr()
    shape = tuple(cache["matrix_shape"])

    assert shape[0] == EXPECTED_SAMPLES
    assert presence.shape == shape == frequency.shape
    assert cache["sample_ids"] == full["sample_id"].astype(str).tolist()
    np.testing.assert_allclose(
        np.asarray(frequency.sum(axis=1)).ravel(),
        1.0,
        rtol=0,
        atol=1e-10,
    )
    assert int(marker["catalog_size"]) == shape[1]
    assert int(marker["matrix_nnz"]) == presence.nnz

    generated = spec.repository_root / marker["generated_scheme"]
    assert generated.is_file()
    assert sha256(generated) == marker["generated_scheme_sha256"]

    raw = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert raw["scheme"]["id"] == spec.scheme_id
    assert list(raw["replacements"]["models"]) == EXPECTED_MODELS
    assert raw["overrides"]["final_model"]["status"] == "not_selected"

    prepared = build_prepared_scheme(generated, spec.repository_root)
    resolved = prepared.resolved_config
    assert list(resolved["models"]) == EXPECTED_MODELS

    cv = resolved["cross_validation"]
    engine = resolved["model_engine"]
    assert cv["outer_repeats"] == expected_splits
    assert cv["outer_folds"] == 2
    assert cv["executed_outer_folds"] == [1]

    expected_tasks = expected_splits * len(cv["executed_outer_folds"])
    candidate_count = len(engine["alpha_grid"]) * len(engine["lambda_grid"])
    expected_inner_fits = (
        len(EXPECTED_MODELS) * candidate_count * spec.inner_folds
    )

    assert resolved["nested_cv"]["expected_outer_tasks"] == expected_tasks
    assert (
        resolved["nested_cv"]["expected_inner_fits_per_task"]
        == expected_inner_fits
    )
    assert resolved["outer_tasks"]["expected_tasks"] == expected_tasks
    assert (
        resolved["aggregation"]["expected_metric_rows"]
        == expected_tasks * len(EXPECTED_MODELS)
    )
    assert (
        resolved["aggregation"]["expected_prediction_rows"]
        == expected_holdout * expected_tasks * len(EXPECTED_MODELS)
    )

    repeated = resolved["repeated_holdout_training"]
    assert int(repeated["split_count"]) == expected_splits
    assert str(repeated["split_set_id"]) == split_set_id
    assert int(repeated["train_size"]) == expected_train
    assert int(repeated["holdout_size"]) == expected_holdout

    return marker, prepared


def main() -> int:
    args = parse_args()
    spec_path = Path(args.spec).expanduser().resolve()
    root = (
        Path(args.repository_root).expanduser().resolve()
        if args.repository_root
        else find_repository_root(spec_path)
    )
    spec = load_training_bundle_spec(spec_path, repository_root=root)
    structure = validate_structure(spec)

    print("RA-ILD IGH Batch 04 training-bundle validation: PASS")
    print(f"Mode:              {args.mode}")
    print(f"Split set:         {structure['split_set_id']}")
    print(f"Splits:            {structure['expected_splits']}")
    print(
        "Per split:         "
        f"train={structure['expected_train']}, "
        f"holdout={structure['expected_holdout']}"
    )
    print(f"Samples:           {len(structure['combined'])}")
    print(f"Static features:   {EXPECTED_STATIC}")
    print(f"AA tables:         {EXPECTED_SAMPLES}")
    print(f"Outer rows:        {len(structure['outer'])}")
    print(f"Inner rows:        {len(structure['inner'])}")
    print(f"Inner candidates:  {len(structure['selected'])} repeats")
    print(
        "Max inner balance: "
        f"{structure['selected']['max_balance_score'].astype(float).max():.8f}"
    )

    if args.mode == "frozen":
        marker, prepared = validate_frozen(spec, structure)
        print(f"Catalog size:      {marker['catalog_size']}")
        print(f"Matrix nnz:        {marker['matrix_nnz']}")
        print(f"Generated scheme:  {prepared.source_path}")

    print("IGH_REPEATED_HOLDOUT_TRAINING_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"RA-ILD IGH Batch 04 training-bundle validation: FAIL\n{exc}",
            file=sys.stderr,
        )
        raise
