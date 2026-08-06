#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused tests for Batch 17 threshold-grid repeated five-fold validation."""

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

from ra_ild_trb.enrich_dictionary_cv import ScopeDataset  # noqa: E402
from ra_ild_trb.enrich_dictionary_grid import (  # noqa: E402
    ALL_METRICS,
    MAX_WORKERS,
    auc_matrix,
    generate_threshold_pairs,
    inclusive_decimal_range,
    plot_scope_grid,
    run_scope_grid,
    threshold_grid_frame,
    validate_workers,
    write_scope_outputs,
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
    except Exception as exc:
        ok(contains in str(exc), name)
    else:
        ok(False, name)


def make_dataset(n_ra: int = 15, n_ild: int = 10, scope: str = "total") -> ScopeDataset:
    sample_ids = ["RA{:02d}".format(i + 1) for i in range(n_ra)] + [
        "ILD{:02d}".format(i + 1) for i in range(n_ild)
    ]
    labels = np.asarray(["RA"] * n_ra + ["ILD"] * n_ild, dtype=str)
    metadata = pd.DataFrame(
        {
            "libraryid": sample_ids,
            "cohort": labels,
            "material": ["PBMC" if i % 2 == 0 else "buffycoat" for i in range(len(labels))],
            "batch": ["batch{}".format(i % 3 + 1) for i in range(len(labels))],
            "sex": ["female" if i % 4 else "male" for i in range(len(labels))],
        }
    )
    clone_names = ["SHARED", "RA_SIGNAL", "ILD_SIGNAL", "RA_WEAK", "ILD_WEAK"]
    clone_ids = []
    read_fractions = []
    total_unique = []
    next_id = 5
    for i, label in enumerate(labels):
        ids = [0]
        rf = [0.10]
        if label == "RA":
            ids.extend([1, 3])
            rf.extend([0.30, 0.05])
        else:
            ids.extend([2, 4])
            rf.extend([0.28, 0.05])
        # deterministic background variation
        if i % 3 == 0:
            ids.append(3 if label == "ILD" else 4)
            rf.append(0.02)
        clone_names.append("PRIVATE_{}".format(sample_ids[i]))
        ids.append(next_id)
        next_id += 1
        rf.append(0.01)
        clone_ids.append(np.asarray(ids, dtype=np.int32))
        read_fractions.append(np.asarray(rf, dtype=np.float64))
        total_unique.append(len(ids))
    return ScopeDataset(
        scope=scope,
        metadata=metadata,
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
    print("== T1 threshold-grid construction ==")
    t_values = inclusive_decimal_range(10, 30, 2)
    d_values = inclusive_decimal_range(5, 15, 1)
    pairs = generate_threshold_pairs(t_values, d_values)
    ok(t_values == [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
       "T1a inclusive T range")
    ok(d_values == [float(x) for x in range(5, 16)], "T1b inclusive Delta range")
    ok(len(pairs) == 112, "T1c exactly 112 valid threshold pairs")
    ok(all(pair.delta_pct <= pair.threshold_pct for pair in pairs), "T1d every pair satisfies Delta<=T")
    frame = threshold_grid_frame(pairs)
    ok(frame["threshold_id"].is_unique and len(frame) == 112, "T1e threshold IDs are unique")

    print("\n== T2 worker contract ==")
    ok(validate_workers(1) == 1 and validate_workers(16) == 16 and MAX_WORKERS == 16,
       "T2a worker range includes 1 and 16")
    expect_error(lambda: validate_workers(0), "between 1 and 16", "T2b workers=0 rejected")
    expect_error(lambda: validate_workers(17), "between 1 and 16", "T2c workers>16 rejected")
    expect_error(lambda: validate_workers(18), "between 1 and 16", "T2d requested workers=18 rejected")

    print("\n== T3 vectorized AUC ==")
    labels = np.asarray(["RA", "RA", "ILD", "ILD"])
    scores = np.asarray([[4, 1], [3, 1], [2, 1], [1, 1]], dtype=float)
    aucs, n_valid = auc_matrix(labels, scores)
    ok(np.allclose(aucs, [1.0, 0.5]) and np.array_equal(n_valid, [4, 4]),
       "T3a perfect and constant-score AUC")
    scores[3, 0] = np.nan
    aucs, n_valid = auc_matrix(labels, scores)
    ok(np.isclose(aucs[0], 1.0) and n_valid[0] == 3, "T3b finite-subset AUC")

    print("\n== T4 serial grid run invariants ==")
    dataset = make_dataset()
    small_pairs = generate_threshold_pairs([10, 20], [5, 10])
    serial = run_scope_grid(
        dataset,
        small_pairs,
        folds=5,
        repeats=3,
        candidates=20,
        seed=20260711,
        workers=1,
        save_fold_auc=True,
        progress_every=10,
    )
    n_pairs = len(small_pairs)
    n_metrics = len(ALL_METRICS)
    ok(len(serial.auc_by_repeat) == 3 * n_pairs * n_metrics,
       "T4a one pooled OOF AUC per repeat/pair/metric")
    ok(len(serial.auc_by_fold) == 3 * 5 * n_pairs * n_metrics,
       "T4b five diagnostic fold AUCs per repeat/pair/metric")
    ok(len(serial.assignments) == dataset.n_samples * 3,
       "T4c one fold assignment per sample per repeat")
    ok(len(serial.sample_averaged_scores) == dataset.n_samples * n_pairs,
       "T4d sample-averaged scores cover every sample/pair")
    ok(len(serial.threshold_summary) == n_pairs * n_metrics,
       "T4e threshold summary covers every pair/metric")
    ok(len(serial.best_thresholds) == n_metrics,
       "T4f one exploratory best pair per metric")
    ok(serial.auc_by_repeat["is_primary_repeat_result"].all(),
       "T4g repeat rows are marked primary")
    ok(serial.auc_by_fold["role"].eq("diagnostic_fold_auc_not_independent").all(),
       "T4h fold rows are marked diagnostic")
    ok(serial.fold_dictionary_summary.groupby(["repeat", "fold", "threshold_id"]).size().eq(1).all(),
       "T4i one dictionary summary per fold and threshold")
    ok(serial.stability["jaccard"].between(0, 1).all(),
       "T4j Jaccard remains in [0,1]")
    no_fold = run_scope_grid(
        dataset,
        small_pairs[:1],
        folds=5,
        repeats=1,
        candidates=10,
        seed=20260711,
        workers=1,
        save_fold_auc=False,
        progress_every=10,
    )
    ok(no_fold.auc_by_fold.empty and "metric" in no_fold.auc_by_fold.columns,
       "T4k no-fold-AUC mode emits an empty table with headers")

    print("\n== T5 threshold direction recovered ==")
    summary = serial.threshold_summary
    ra_signal = summary.loc[summary["metric"] == "RA_dict_clone_count"]
    ild_signal = summary.loc[summary["metric"] == "ILD_dict_clone_count"]
    ok(float(ra_signal["repeat_directional_auc_median"].max()) > 0.9,
       "T5a RA dictionary signal recovered")
    ok(float(ild_signal["repeat_directional_auc_median"].max()) > 0.9,
       "T5b RA-ILD dictionary signal recovered directionally")

    print("\n== T6 serial/parallel reproducibility ==")
    parallel = run_scope_grid(
        dataset,
        small_pairs,
        folds=5,
        repeats=3,
        candidates=20,
        seed=20260711,
        workers=2,
        save_fold_auc=True,
        progress_every=10,
    )
    left = serial.auc_by_repeat.sort_values(["repeat", "threshold_id", "metric"])["auc_RA_positive_raw"].to_numpy()
    right = parallel.auc_by_repeat.sort_values(["repeat", "threshold_id", "metric"])["auc_RA_positive_raw"].to_numpy()
    ok(np.allclose(left, right, equal_nan=True), "T6a serial and two-worker repeat AUCs match")
    left_scores = serial.sample_averaged_scores.sort_values(["threshold_id", "sample_id"])[list(ALL_METRICS)].to_numpy()
    right_scores = parallel.sample_averaged_scores.sort_values(["threshold_id", "sample_id"])[list(ALL_METRICS)].to_numpy()
    ok(np.allclose(left_scores, right_scores, equal_nan=True), "T6b serial and parallel sample means match")

    print("\n== T7 outputs and plots ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        outputs = write_scope_outputs(serial, root / "result", root / "plots", make_plots=True)
        ok(all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in outputs.values()),
           "T7a all declared output files exist")
        plot_paths = [Path(path) for name, path in outputs.items() if name.startswith("plot_")]
        ok(len(plot_paths) == 11, "T7b eight heatmaps plus three summary plots")
        ok((root / "result" / "total" / "grid_auc_by_repeat_total.csv.gz").is_file(),
           "T7c compressed repeat AUC table written")
        ok((root / "result" / "total" / "grid_best_thresholds_by_metric_total.csv").is_file(),
           "T7d best-threshold table written")

    print("\nFocused tests: {}/{} PASS".format(PASS, PASS + FAIL))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
