#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI for threshold-grid repeated five-fold enrich-dictionary validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a valid T/Delta threshold grid with repeated stratified five-fold "
            "out-of-fold ROC analysis. One pooled OOF AUC is produced per complete repeat, "
            "threshold pair, scope and metric."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", default="/data/users/chenhaisheng/RA-ILD")
    parser.add_argument("--scopes", default="total,pbmc,buffycoat")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260711)

    parser.add_argument("--threshold-start", type=float, default=10.0)
    parser.add_argument("--threshold-stop", type=float, default=30.0)
    parser.add_argument("--threshold-step", type=float, default=2.0)
    parser.add_argument("--delta-start", type=float, default=5.0)
    parser.add_argument("--delta-stop", type=float, default=15.0)
    parser.add_argument("--delta-step", type=float, default=1.0)
    parser.add_argument(
        "--threshold-values",
        default=None,
        help="Optional comma-separated explicit T values; overrides threshold start/stop/step.",
    )
    parser.add_argument(
        "--delta-values",
        default=None,
        help="Optional comma-separated explicit Delta values; overrides delta start/stop/step.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Linux process workers. Supported range: 1-16. Each worker processes one complete repeat.",
    )
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--no-fold-auc", action="store_true", help="Do not write diagnostic per-fold AUC values.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast end-to-end test: one repeat, at most 20 split candidates, and the first "
            "three valid threshold pairs."
        ),
    )
    args = parser.parse_args()
    args.scopes = [x.strip().lower() for x in args.scopes.split(",") if x.strip()]
    invalid = sorted(set(args.scopes).difference({"total", "pbmc", "buffycoat"}))
    if invalid:
        parser.error("Unsupported scopes: {}".format(invalid))
    if args.folds < 2:
        parser.error("--folds must be >=2")
    if args.repeats < 1:
        parser.error("--repeats must be >=1")
    if args.candidates < 1:
        parser.error("--candidates must be >=1")
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16 inclusive")
    return args


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    src = root / "TRB/set/src"
    if not src.is_dir():
        raise FileNotFoundError("Missing source directory: {}".format(src))
    sys.path.insert(0, str(src))
    from ra_ild_trb.enrich_dictionary_grid import (
        generate_threshold_pairs,
        inclusive_decimal_range,
        parse_numeric_list,
        run_project_grid_analysis,
    )

    threshold_values = parse_numeric_list(args.threshold_values, "threshold values")
    if threshold_values is None:
        threshold_values = inclusive_decimal_range(
            args.threshold_start, args.threshold_stop, args.threshold_step
        )
    delta_values = parse_numeric_list(args.delta_values, "delta values")
    if delta_values is None:
        delta_values = inclusive_decimal_range(args.delta_start, args.delta_stop, args.delta_step)
    pairs = generate_threshold_pairs(threshold_values, delta_values)

    repeats = args.repeats
    candidates = args.candidates
    if args.smoke:
        repeats = 1
        candidates = min(candidates, 20)
        pairs = pairs[:3]
        threshold_values = sorted(set(pair.threshold_pct for pair in pairs))
        delta_values = sorted(set(pair.delta_pct for pair in pairs))

    print(
        "Threshold grid: T={} | Delta={} | valid pairs={} | rule Delta<=T".format(
            threshold_values, delta_values, len(pairs)
        ),
        flush=True,
    )
    print(
        "Design: scopes={} | folds={} | repeats={} | workers={}".format(
            args.scopes, args.folds, repeats, args.workers
        ),
        flush=True,
    )

    manifest = run_project_grid_analysis(
        project_root=root,
        scopes=args.scopes,
        folds=args.folds,
        repeats=repeats,
        candidates=candidates,
        seed=args.seed,
        threshold_values=threshold_values,
        delta_values=delta_values,
        workers=args.workers,
        allow_count_mismatch=args.allow_count_mismatch,
        save_fold_auc=not args.no_fold_auc,
        make_plots=not args.no_plots,
        overwrite=args.overwrite,
        result_dir=Path(args.result_dir).expanduser().resolve() if args.result_dir else None,
        plot_dir=Path(args.plot_dir).expanduser().resolve() if args.plot_dir else None,
    )
    print(json.dumps(manifest["scope_counts"], ensure_ascii=False, indent=2))
    print("Completed enrich-dictionary threshold-grid repeated five-fold validation.")
    print(
        "Summary: {}".format(
            Path(manifest["configuration"]["result_dir"]) / "threshold_grid_summary.md"
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise
