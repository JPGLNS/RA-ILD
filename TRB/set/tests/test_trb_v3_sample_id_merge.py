#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression test for sample_id index/column ambiguity in fold feature merges."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "TRB/set/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ra_ild_trb.nested_cv import assemble_model_dataframe
from ra_ild_trb.public_reference import ALL_PUBLIC_FEATURES


def test_assembled_frames_merge_inner_fold_3mer_features_by_sample_id() -> None:
    base = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "cohort": ["RA", "ILD", "RA"],
            "age": [51, 62, 47],
        }
    ).set_index("sample_id", drop=False)

    # Public-reference features are aligned by the original base-matrix row IDs.
    public = pd.DataFrame(
        0.0,
        index=[10, 20, 30],
        columns=list(ALL_PUBLIC_FEATURES),
    )
    row_lookup = {"s1": 10, "s2": 20, "s3": 30}
    requested_ids = ("s3", "s1")

    assembled = assemble_model_dataframe(
        base,
        requested_ids,
        public,
        row_lookup,
    )

    # The selected copy keeps the sample_id column but no longer has a
    # conflicting index name. The original indexed base remains unchanged.
    assert assembled.index.name is None
    assert "sample_id" in assembled.columns
    assert assembled["sample_id"].tolist() == list(requested_ids)
    assert base.index.name == "sample_id"

    fold_3mer = pd.DataFrame(
        {
            "sample_id": ["s3", "s1"],
            "AA_3mer_AAA_unweighted": [0.25, 0.75],
            "AA_3mer_AAA_weighted": [0.10, 0.90],
        }
    )
    merged = assembled.merge(
        fold_3mer,
        on="sample_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    assert merged["sample_id"].tolist() == list(requested_ids)
    assert merged["AA_3mer_AAA_unweighted"].tolist() == [0.25, 0.75]
    assert merged["AA_3mer_AAA_weighted"].tolist() == [0.10, 0.90]


if __name__ == "__main__":
    test_assembled_frames_merge_inner_fold_3mer_features_by_sample_id()
    print("TRB V3 sample_id merge regression test: PASS")
