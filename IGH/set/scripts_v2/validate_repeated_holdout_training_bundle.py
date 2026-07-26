#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the IGH Batch 04 repeated-holdout training bundle."""
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
EXPECTED_SPLITS = 3
EXPECTED_TRAIN = 120
EXPECTED_HOLDOUT = 49
EXPECTED_ASSIGNMENT_SHA = "5537365e41ac77a144c81fa7b7167896fc1ceb9db84664cd661b546575f8a030"
FIXED_TRAIN_ID = "MGI260202N01-31"
DYNAMIC_PUBLIC = {
    "all_ref_public_clone_ratio", "all_ref_public_frequency_sum",
    "shared_ref_clone_ratio", "shared_ref_frequency_sum",
    "RA_specific_ref_clone_ratio", "RA_specific_ref_frequency_sum",
    "ILD_specific_ref_clone_ratio", "ILD_specific_ref_frequency_sum",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--repository-root", default=None)
    p.add_argument("--mode", choices=("dry-run", "frozen"), default="dry-run")
    return p.parse_args()


def validate_structure(spec):
    assignments, split_marker = load_frozen_assignments(spec)
    assert sha256(spec.split_assignments) == EXPECTED_ASSIGNMENT_SHA
    assert split_marker.get("split_set_id") == "igh_ra_ild_repeat3_v1"
    combined, metadata, static_features = prepare_full_static_matrix(spec, assignments)
    aa_manifest = discover_aa_clone_tables(spec, combined["sample_id"].astype(str).tolist())
    outer, inner, selected, audit = generate_v2_assignments(spec, assignments)

    assert len(combined) == EXPECTED_SAMPLES
    assert len(metadata) == EXPECTED_SAMPLES
    assert len(static_features) == EXPECTED_STATIC
    assert len(aa_manifest) == EXPECTED_SAMPLES
    assert not (DYNAMIC_PUBLIC & set(combined.columns))
    assert combined["sample_id"].is_unique
    assert set(combined["cohort"].astype(str).str.upper()) == {"RA", "ILD"}

    assert assignments["repeat_index"].nunique() == EXPECTED_SPLITS
    role_counts = assignments.groupby(["repeat_index", "role"]).size().unstack(fill_value=0)
    assert (role_counts[spec.split_train_role] == EXPECTED_TRAIN).all()
    assert (role_counts[spec.split_holdout_role] == EXPECTED_HOLDOUT).all()
    singleton = assignments.loc[assignments["sample_id"].astype(str) == FIXED_TRAIN_ID]
    assert len(singleton) == EXPECTED_SPLITS
    assert singleton["role"].astype(str).eq(spec.split_train_role).all()

    assert len(outer) == EXPECTED_SAMPLES * EXPECTED_SPLITS
    assert set(outer["outer_fold"].astype(int)) == {1, 2}
    for repeat, frame in outer.groupby("outer_repeat"):
        counts = frame["outer_fold"].value_counts().to_dict()
        assert counts == {2: EXPECTED_TRAIN, 1: EXPECTED_HOLDOUT}, (repeat, counts)
    holdout_pairs = set(
        outer.loc[outer["outer_fold"].astype(int) == 1, ["outer_repeat", "sample_id"]]
        .itertuples(index=False, name=None)
    )
    inner_pairs = set(inner[["outer_repeat", "sample_id"]].itertuples(index=False, name=None))
    assert not (holdout_pairs & inner_pairs)
    assert len(inner) == EXPECTED_SPLITS * EXPECTED_TRAIN
    assert len(selected) == EXPECTED_SPLITS
    assert len(audit) == EXPECTED_SPLITS * spec.inner_folds
    for _, frame in inner.groupby("outer_repeat"):
        assert set(frame["inner_fold"].astype(int)) == set(range(1, spec.inner_folds + 1))
        assert len(frame) == EXPECTED_TRAIN
    return assignments, combined, outer, inner, selected, audit


def validate_frozen(spec):
    marker = json.loads(spec.marker_path.read_text(encoding="utf-8"))
    assert marker["status"] == "FROZEN"
    assert marker["bundle_id"] == spec.bundle_id
    assert marker["samples"] == EXPECTED_SAMPLES
    assert marker["static_feature_count"] == EXPECTED_STATIC
    assert marker["split_count"] == EXPECTED_SPLITS
    assert marker["train_sizes"] == [EXPECTED_TRAIN] * EXPECTED_SPLITS
    assert marker["holdout_sizes"] == [EXPECTED_HOLDOUT] * EXPECTED_SPLITS
    for item in marker["files"].values():
        path = spec.repository_root / item["path"]
        assert path.is_file(), path
        assert sha256(path) == item["sha256"], path

    full = pd.read_csv(spec.full_base_matrix_path)
    cache = json.loads(spec.cache_metadata_path.read_text(encoding="utf-8"))
    presence = sparse.load_npz(spec.presence_path).tocsr()
    frequency = sparse.load_npz(spec.frequency_path).tocsr()
    shape = tuple(cache["matrix_shape"])
    assert shape[0] == EXPECTED_SAMPLES
    assert presence.shape == shape == frequency.shape
    assert cache["sample_ids"] == full["sample_id"].astype(str).tolist()
    np.testing.assert_allclose(np.asarray(frequency.sum(axis=1)).ravel(), 1.0, rtol=0, atol=1e-10)
    assert int(marker["catalog_size"]) == shape[1]
    assert int(marker["matrix_nnz"]) == presence.nnz

    generated = spec.repository_root / marker["generated_scheme"]
    assert generated.is_file()
    assert sha256(generated) == marker["generated_scheme_sha256"]
    raw = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert raw["scheme"]["id"] == "igh_scheme_003_repeat3_no_clinical"
    assert list(raw["replacements"]["models"]) == [
        "M1_static_igh", "M2_static_igh_public", "M3_static_igh_public_material"
    ]
    assert raw["overrides"]["final_model"]["status"] == "not_selected"

    prepared = build_prepared_scheme(generated, spec.repository_root)
    resolved = prepared.resolved_config
    assert list(resolved["models"]) == [
        "M1_static_igh", "M2_static_igh_public", "M3_static_igh_public_material"
    ]
    assert resolved["cross_validation"]["outer_repeats"] == 3
    assert resolved["cross_validation"]["outer_folds"] == 2
    assert resolved["cross_validation"]["executed_outer_folds"] == [1]
    assert resolved["nested_cv"]["expected_outer_tasks"] == 3
    assert resolved["nested_cv"]["expected_inner_fits_per_task"] == 360
    assert resolved["outer_tasks"]["expected_tasks"] == 3
    assert resolved["aggregation"]["expected_metric_rows"] == 9
    assert resolved["aggregation"]["expected_prediction_rows"] == 441
    return marker, prepared


def main():
    args = parse_args()
    spec_path = Path(args.spec).expanduser().resolve()
    root = Path(args.repository_root).expanduser().resolve() if args.repository_root else find_repository_root(spec_path)
    spec = load_training_bundle_spec(spec_path, repository_root=root)
    _, combined, outer, inner, selected, audit = validate_structure(spec)
    print("RA-ILD IGH Batch 04 training-bundle validation: PASS")
    print(f"Mode:              {args.mode}")
    print(f"Samples:           {len(combined)}")
    print(f"Static features:   {EXPECTED_STATIC}")
    print(f"AA tables:         {EXPECTED_SAMPLES}")
    print(f"Outer rows:        {len(outer)}")
    print(f"Inner rows:        {len(inner)}")
    print(f"Inner candidates:  {selected['candidate_seed'].astype(int).tolist()}")
    print(f"Max inner balance: {selected['max_balance_score'].astype(float).max():.8f}")
    if args.mode == "frozen":
        marker, prepared = validate_frozen(spec)
        print(f"Catalog size:      {marker['catalog_size']}")
        print(f"Matrix nnz:        {marker['matrix_nnz']}")
        print(f"Generated scheme: {prepared.source_path}")
    print("IGH_REPEATED_HOLDOUT_TRAINING_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"RA-ILD IGH Batch 04 training-bundle validation: FAIL\n{exc}", file=sys.stderr)
        raise
