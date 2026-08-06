#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression tests for Batch 16 enrich-dictionary repeated 5-fold CV."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SET_DIR = SCRIPT_DIR.parent
SRC_DIR = SET_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.enrich_dictionary_cv import (  # noqa: E402
    DictionaryResult,
    RawRepertoire,
    ScopeDataset,
    auc_from_scores,
    build_aa_file_map,
    build_dictionary,
    direction_fields,
    jaccard,
    optimize_split,
    plot_scope_result,
    run_scope_cv,
    score_sample,
    validate_fold_assignment,
)

PASS = 0
FAIL = 0


def ok(condition: bool, name: str) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print("PASS {}".format(name))
    else:
        FAIL += 1
        print("FAIL {}".format(name))


def expect_error(fn, contains: str, name: str) -> None:
    try:
        fn()
    except Exception as exc:  # intentionally broad for focused CLI tests
        ok(contains in str(exc), name)
    else:
        ok(False, name)


def make_dataset(
    n_ra: int = 10,
    n_ild: int = 10,
    include_signal: bool = True,
    scope: str = "total",
) -> ScopeDataset:
    sample_ids = ["RA{:02d}".format(i + 1) for i in range(n_ra)] + [
        "ILD{:02d}".format(i + 1) for i in range(n_ild)
    ]
    labels = np.asarray(["RA"] * n_ra + ["ILD"] * n_ild, dtype=str)
    meta = pd.DataFrame(
        {
            "libraryid": sample_ids,
            "cohort": labels,
            "material": ["PBMC" if i % 2 == 0 else "buffycoat" for i in range(len(sample_ids))],
            "batch": ["batch{}".format(i % 3 + 1) for i in range(len(sample_ids))],
            "sex": ["female" if i % 4 else "male" for i in range(len(sample_ids))],
        }
    )

    # 0=shared, 1=RA prevalence signal, 2=ILD prevalence signal,
    # 3/4 weak background, remaining IDs are sample-private.
    clone_names = ["SHARED", "RA_SIGNAL", "ILD_SIGNAL", "BG_RA", "BG_ILD"]
    clone_ids = []
    read_fractions = []
    total_unique = []
    next_private = 5
    for i, label in enumerate(labels):
        ids = [0]
        rfs = [0.10]
        if include_signal:
            if label == "RA":
                ids.extend([1, 3])
                rfs.extend([0.30, 0.05])
            else:
                ids.extend([2, 4])
                rfs.extend([0.28, 0.05])
        else:
            ids.append(3 if i % 2 == 0 else 4)
            rfs.append(0.05)
        private_name = "PRIVATE_{}".format(sample_ids[i])
        clone_names.append(private_name)
        ids.append(next_private)
        next_private += 1
        rfs.append(0.01)
        clone_ids.append(np.asarray(ids, dtype=np.int32))
        read_fractions.append(np.asarray(rfs, dtype=np.float64))
        total_unique.append(len(ids))

    return ScopeDataset(
        scope=scope,
        metadata=meta,
        sample_ids=sample_ids,
        labels=labels,
        clone_names=np.asarray(clone_names, dtype=object),
        clone_ids=clone_ids,
        read_fractions=read_fractions,
        total_unique_clones=np.asarray(total_unique, dtype=np.int32),
        prescreen_min_presence=1,
        prescreen_original_universe=len(clone_names),
    )


def main() -> int:
    print("== T1 deterministic optimized stratified split ==")
    d = make_dataset()
    f1, score1, seed1, sig1 = optimize_split(d.metadata, 5, 40, 20260711, ["batch", "material", "sex"])
    f2, score2, seed2, sig2 = optimize_split(d.metadata, 5, 40, 20260711, ["batch", "material", "sex"])
    ok(np.array_equal(f1, f2) and score1 == score2 and seed1 == seed2 and sig1 == sig2,
       "T1 deterministic split for identical seed")
    validate_fold_assignment(d.metadata, f1, 5)
    ok(True, "T1b fold validation passes")

    print("\n== T2 cohort balance and unique partitions ==")
    used = set()
    signatures = []
    for repeat in range(1, 4):
        folds, _, _, sig = optimize_split(d.metadata, 5, 30, 1000 + repeat, ["batch"], forbidden=used)
        used.add(sig)
        signatures.append(sig)
        validate_fold_assignment(d.metadata, folds, 5)
    ok(len(set(signatures)) == 3, "T2 three repeats use unique partitions")

    print("\n== T3 dictionary is training-only ==")
    # Clone ID 5 is private to RA01. Holding RA01 out must exclude it.
    train = np.arange(1, d.n_samples, dtype=int)
    dictionary = build_dictionary(d, train, threshold_pct=20.0, delta_pct=10.0)
    ok(5 not in set(dictionary.ra_ids.tolist()) and 5 not in set(dictionary.ild_ids.tolist()),
       "T3 held-out private clone never enters dictionary")

    print("\n== T4 scoring formulas ==")
    manual = DictionaryResult(
        ra_ids=np.asarray([0, 1], dtype=np.int32),
        ild_ids=np.asarray([2], dtype=np.int32),
        ra_threshold_count=1,
        ild_threshold_count=1,
        n_ra_train=5,
        n_ild_train=5,
    )
    scored = score_sample(d, 0, manual)
    ok(scored["RA_dict_clone_count"] == 2 and scored["ILD_dict_clone_count"] == 0,
       "T4a dictionary hit counts")
    ok(np.isclose(scored["RA_dict_hit_rate"], 1.0) and np.isclose(scored["RA_hit_fraction_of_sample"], 0.5),
       "T4b dictionary coverage and sample-hit fraction")
    ok(np.isclose(scored["RA_dict_read_fraction_sum"], 0.40),
       "T4c cumulative abundance")

    print("\n== T5 full repeated 5-fold OOF invariants ==")
    result = run_scope_cv(d, folds=5, repeats=3, candidates=30, seed=20260711,
                          threshold_pct=20.0, delta_pct=10.0, save_dictionaries=True)
    ok(len(result.predictions) == d.n_samples * 3,
       "T5a one OOF prediction per sample per repeat")
    ok(result.predictions.groupby(["repeat", "sample_id"]).size().eq(1).all(),
       "T5b no duplicated or missing OOF score")
    constant_within_fold = result.predictions.groupby(["repeat", "fold"])[["RA_dict_size", "ILD_dict_size"]].nunique().eq(1).all().all()
    ok(bool(constant_within_fold), "T5c all validation samples in a fold share one dictionary")
    ok(set(result.assignments["repeat"].unique()) == {1, 2, 3},
       "T5d assignments cover all repeats")
    expected_repeat_rows = 3 * len(result.metric_summary)
    expected_fold_rows = 3 * 5 * len(result.metric_summary)
    ok(len(result.auc_by_repeat) == expected_repeat_rows and
       result.auc_by_repeat["is_primary_repeat_result"].all(),
       "T5e one primary pooled OOF AUC per metric per complete repeat")
    ok(len(result.auc_by_fold) == expected_fold_rows and
       result.auc_by_fold["role"].eq("diagnostic_fold_auc_not_independent").all(),
       "T5f five diagnostic fold AUC rows per metric per repeat")
    first_repeat = result.predictions.loc[result.predictions["repeat"] == 1]
    direct_auc, _ = auc_from_scores(first_repeat["cohort"], first_repeat["RA_dict_clone_count"])
    stored_auc = result.auc_by_repeat.loc[
        (result.auc_by_repeat["repeat"] == 1) &
        (result.auc_by_repeat["metric"] == "RA_dict_clone_count"),
        "auc_RA_positive_raw",
    ].iloc[0]
    ok(np.isclose(direct_auc, stored_auc),
       "T5g repeat AUC is calculated from pooled OOF scores across all five folds")

    print("\n== T6 synthetic direction is recovered out of fold ==")
    summary = result.metric_summary.set_index("metric")
    ra_auc = float(summary.loc["RA_dict_clone_count", "sample_averaged_auc_RA_positive_raw"])
    ild_auc = float(summary.loc["ILD_dict_clone_count", "sample_averaged_auc_RA_positive_raw"])
    ok(ra_auc > 0.95, "T6a RA dictionary score is higher in RA")
    ok(ild_auc < 0.05, "T6b RA-ILD dictionary score is higher in RA-ILD")
    dir_auc, direction, ild_positive_auc = direction_fields(ild_auc)
    ok(dir_auc > 0.95 and direction == "RA-ILD_higher" and ild_positive_auc > 0.95,
       "T6c direction-aware fields correctly report RA-ILD-higher score")

    print("\n== T7 AUC edge cases ==")
    auc_constant, n_constant = auc_from_scores(["RA", "RA", "ILD", "ILD"], [1, 1, 1, 1])
    ok(auc_constant == 0.5 and n_constant == 4, "T7a constant score gives AUC 0.5")
    auc_missing, n_missing = auc_from_scores(["RA", "ILD", "RA"], [1.0, np.nan, 0.5])
    ok(np.isnan(auc_missing) and n_missing == 2, "T7b one-class finite subset gives NA AUC")

    print("\n== T8 empty dictionary is safe ==")
    empty = DictionaryResult(
        ra_ids=np.asarray([], dtype=np.int32),
        ild_ids=np.asarray([], dtype=np.int32),
        ra_threshold_count=1,
        ild_threshold_count=1,
        n_ra_train=5,
        n_ild_train=5,
    )
    empty_score = score_sample(d, 0, empty)
    ok(np.isnan(empty_score["RA_dict_hit_rate"]) and np.isnan(empty_score["ILD_dict_hit_rate"]),
       "T8 empty dictionary coverage is NA, not crash")

    print("\n== T9 dictionary stability metric ==")
    ok(np.isclose(jaccard(np.asarray([1, 2]), np.asarray([2, 3])), 1.0 / 3.0),
       "T9a Jaccard overlap")
    ok(jaccard(np.asarray([], dtype=int), np.asarray([], dtype=int)) == 1.0,
       "T9b two empty dictionaries are identical")
    ok(result.stability["jaccard"].between(0, 1).all(),
       "T9c observed stability values stay within [0,1]")

    print("\n== T10 explicit sample-file mapping ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for sid in ("s3", "s1", "s2"):
            pd.DataFrame({"cdr3_aa": ["ONLY_{}".format(sid)], "read_fraction": [1.0]}).to_csv(
                root / "{}_AA_clone_table.csv".format(sid), index=False
            )
        mapping = build_aa_file_map(root, ["s1", "s2", "s3"])
        ok(set(mapping) == {"s1", "s2", "s3"} and mapping["s2"].name.startswith("s2_"),
           "T10a mapping is keyed by libraryid, not creation order")
        expect_error(lambda: build_aa_file_map(root, ["s1", "s2", "s3", "missing"]),
                     "missing AA clone tables", "T10b missing sample file is rejected")

    print("\n== T11 richness diagnostics are emitted ==")
    ok(set(result.richness_correlation["metric"]) == set(result.metric_summary["metric"]),
       "T11 richness correlation covers every metric")
    ok("sample_unique_clone_count" in result.sample_averaged.columns,
       "T11b sample unique-clone diagnostic retained")

    print("\n== T12 dictionary member export is training-fold-specific ==")
    ok(result.dictionary_members is not None and
       set(result.dictionary_members["repeat"].unique()) == {1, 2, 3} and
       set(result.dictionary_members["fold"].unique()) == {1, 2, 3, 4, 5},
       "T12 fold-specific dictionary members are exportable")

    print("\n== T13 Matplotlib boxplot compatibility and plot smoke ==")
    with tempfile.TemporaryDirectory() as td:
        plot_paths = plot_scope_result(result, Path(td))
        ok(len(plot_paths) == 5 and all(path.is_file() and path.stat().st_size > 0 for path in plot_paths),
           "T13 all five plots render with the installed Matplotlib API")

    print("\nFocused tests: {}/{} PASS".format(PASS, PASS + FAIL))
    if FAIL:
        raise SystemExit("{} focused tests failed".format(FAIL))
    return 0


if __name__ == "__main__":
    sys.exit(main())
