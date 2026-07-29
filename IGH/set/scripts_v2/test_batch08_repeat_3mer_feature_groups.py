#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression tests for Batch 08 repeat-specific 3-mer features."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.repeat_3mer_features import (  # noqa: E402
    Repeat3merOptions,
    build_repeat_3mer_features,
    split_legacy_static_features,
)


def _write_sample(path: Path, sample_id: str, sequences: list[tuple[str, float]]) -> None:
    pd.DataFrame(
        {
            "sample_id": [sample_id] * len(sequences),
            "cdr3_aa": [value[0] for value in sequences],
            "frequency_norm": [value[1] for value in sequences],
        }
    ).to_csv(path, index=False)


def test_split_1083_style_groups() -> None:
    core = [f"core_{index:03d}" for index in range(1, 84)]
    kmers = ["AAA", "AAC"]
    static = core + [f"unweighted_3mer_{value}" for value in kmers] + [
        f"weighted_3mer_{value}" for value in kmers
    ]
    observed_core, observed_u, observed_w = split_legacy_static_features(
        static, expected_core_count=83
    )
    assert observed_core == tuple(core)
    assert len(observed_u) == 2
    assert len(observed_w) == 2


def test_outer_repeat_vocabulary_and_nested_top_k_groups() -> None:
    with tempfile.TemporaryDirectory(prefix="igh_batch08_") as temp:
        root = Path(temp)
        sample_sequences = {
            "S1": [("AAACCC", 0.7), ("GGGAAA", 0.3)],
            "S2": [("AAAGGG", 0.6), ("CCCAAA", 0.4)],
            "S3": [("AAATTT", 0.5), ("GGGCCC", 0.5)],
            # Holdout-only WWW must not enter the fitted vocabulary.
            "S4": [("WWWWWW", 1.0)],
        }
        manifest_rows = []
        for sample_id, sequences in sample_sequences.items():
            path = root / f"{sample_id}_IGH_CDR3_AA_clone_table.csv"
            _write_sample(path, sample_id, sequences)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "path": str(path),
                    "frequency_column": "frequency_norm",
                }
            )
        manifest_path = root / "aa_clone_table_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

        core = [f"core_{index:03d}" for index in range(1, 84)]
        legacy_kmers = ["AAA", "AAC"]
        legacy_static = core + [
            f"unweighted_3mer_{value}" for value in legacy_kmers
        ] + [f"weighted_3mer_{value}" for value in legacy_kmers]
        base_rows = []
        for row_index, sample_id in enumerate(sample_sequences, start=1):
            row = {
                "sample_id": sample_id,
                "cohort": "ILD" if row_index % 2 else "RA",
                "aa_clone_number": 2,
            }
            row.update({name: float(row_index) for name in core})
            for name in legacy_static[83:]:
                row[name] = 0.0
            base_rows.append(row)
        base = pd.DataFrame(base_rows)

        options = Repeat3merOptions(
            aa_manifest=manifest_path,
            min_sample_count=1,
            min_prevalence=0.0,
            min_unweighted_variance=0.0,
            min_weighted_variance=0.0,
            max_kmers=2,
            top_k_values=(1, 2),
            expected_core_feature_count=83,
        )
        build = build_repeat_3mer_features(
            base,
            legacy_static_features=legacy_static,
            outer_train_ids=("S1", "S2", "S3"),
            outer_repeat=1,
            options=options,
        )

        selected = build.vocabulary.loc[
            build.vocabulary["keep"].astype(bool), "kmer"
        ].astype(str).tolist()
        assert len(selected) == 2
        assert "WWW" not in selected
        assert build.feature_matrix.shape == (4, 5)
        assert len(build.static_feature_groups["core83"]) == 83
        assert len(build.static_feature_groups["repeat_3mer_unweighted_top1"]) == 1
        assert len(build.static_feature_groups["repeat_3mer_weighted_top2"]) == 2
        assert len(build.static_feature_groups["repeat_3mer_both_top2"]) == 4
        assert len(build.static_feature_groups["static_igh_candidate_predictors"]) == 87
        assert not any(
            name in build.augmented_base.columns
            for name in (
                "unweighted_3mer_AAC",
                "weighted_3mer_AAC",
            )
            if name not in build.feature_matrix.columns
        )


def main() -> int:
    tests = [
        test_split_1083_style_groups,
        test_outer_repeat_vocabulary_and_nested_top_k_groups,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"Batch 08 focused tests: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
