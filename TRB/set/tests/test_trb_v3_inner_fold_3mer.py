#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for inner-fold-only TRB V3 3-mer vocabulary selection."""
from __future__ import annotations

import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "TRB/set/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ra_ild_trb.repeat_3mer_features import (
    Repeat3merOptions,
    Repeat3merInputs,
    build_repeat_3mer_fold_features,
)


def _options(manifest: Path) -> Repeat3merOptions:
    return Repeat3merOptions(
        aa_manifest=manifest,
        selection_scope="inner_fold_train_only",
        min_sample_count=1,
        min_prevalence=0.25,
        min_unweighted_variance=0.0,
        min_weighted_variance=0.0,
        max_kmers=2,
        top_k_values=(1, 2),
    )


def test_inner_validation_only_kmers_do_not_enter_vocabulary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.csv"
        manifest.write_text("sample_id,path,frequency_column\n", encoding="utf-8")
        options = _options(manifest)

        # The validation-only CDR3 motif "YYY" is absent from every training
        # repertoire. Its presence must not affect the candidate universe.
        ids = ("tr1", "tr2", "tr3", "va1")
        unweighted = {
            "tr1": {"AAA": 1.0, "CCC": 0.5},
            "tr2": {"AAA": 1.0, "CCC": 0.5},
            "tr3": {"AAA": 1.0, "CCC": 0.5},
            "va1": {"AAA": 1.0, "CCC": 0.5, "YYY": 1.0},
        }
        weighted = {
            "tr1": {"AAA": 0.8, "CCC": 0.2},
            "tr2": {"AAA": 0.7, "CCC": 0.3},
            "tr3": {"AAA": 0.9, "CCC": 0.1},
            "va1": {"AAA": 0.2, "CCC": 0.1, "YYY": 0.7},
        }
        inputs = Repeat3merInputs(
            sample_ids=ids,
            unweighted_by_sample=unweighted,
            weighted_by_sample=weighted,
            clone_numbers={sid: 10000 for sid in ids},
            manifest_path=manifest,
        )
        built = build_repeat_3mer_fold_features(
            ("tr1", "tr2", "tr3"),
            ("va1",),
            inputs,
            options,
            outer_repeat=1,
            outer_fold=1,
            inner_fold=2,
        )
        selected = set(built.vocabulary.loc[built.vocabulary.keep, "kmer"])
        assert "YYY" not in selected
        assert len(selected) == 2
        assert set(c.split("_3mer_", 1)[1] for c in built.train_features.columns if c != "sample_id") == selected
        assert built.audit["selection_scope"] == "inner_fold_train_only"
        assert built.audit["training_sample_count"] == 3
        assert built.audit["validation_sample_count"] == 1


def test_training_and_validation_share_same_fold_vocabulary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.csv"
        manifest.write_text("sample_id,path,frequency_column\n", encoding="utf-8")
        options = _options(manifest)
        ids = ("a", "b", "c", "v")
        unweighted = {
            "a": {"AAA": 1.0, "CCC": 0.5},
            "b": {"AAA": 1.0, "CCC": 0.5},
            "c": {"AAA": 1.0, "CCC": 0.5},
            "v": {"AAA": 0.5, "CCC": 0.5, "DDD": 0.5},
        }
        weighted = {
            "a": {"AAA": 0.8, "CCC": 0.2},
            "b": {"AAA": 0.7, "CCC": 0.3},
            "c": {"AAA": 0.9, "CCC": 0.1},
            "v": {"AAA": 0.3, "CCC": 0.2, "DDD": 0.5},
        }
        inputs = Repeat3merInputs(
            sample_ids=ids,
            unweighted_by_sample=unweighted,
            weighted_by_sample=weighted,
            clone_numbers={sid: 10000 for sid in ids},
            manifest_path=manifest,
        )
        built = build_repeat_3mer_fold_features(
            ("a", "b", "c"), ("v",), inputs, options,
            outer_repeat=1, outer_fold=1, inner_fold=1,
        )
        train_cols = [x for x in built.train_features if x != "sample_id"]
        valid_cols = [x for x in built.valid_features if x != "sample_id"]
        assert train_cols == valid_cols
        assert len(train_cols) == 4
        assert built.train_features.shape == (3, 5)
        assert built.valid_features.shape == (1, 5)


if __name__ == "__main__":
    test_inner_validation_only_kmers_do_not_enter_vocabulary()
    test_training_and_validation_share_same_fold_vocabulary()
    print("TRB V3 inner-fold 3-mer regression tests: PASS")
