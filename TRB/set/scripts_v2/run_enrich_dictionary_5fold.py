#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI for repeated stratified 5-fold enrich-dictionary validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated stratified 5-fold validation of RA/RA-ILD enrich dictionaries; one pooled OOF AUC per complete repeat.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", default="/data/users/chenhaisheng/RA-ILD")
    parser.add_argument("--scopes", default="total,pbmc,buffycoat")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--threshold-pct", type=float, default=20.0)
    parser.add_argument("--delta-pct", type=float, default=10.0)
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--save-dictionaries", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one repeat and 20 split candidates for a fast code-path test.",
    )
    args = parser.parse_args()
    args.scopes = [x.strip().lower() for x in args.scopes.split(",") if x.strip()]
    invalid = sorted(set(args.scopes).difference({"total", "pbmc", "buffycoat"}))
    if invalid:
        parser.error("Unsupported scopes: {}".format(invalid))
    if args.smoke:
        args.repeats = 1
        args.candidates = min(args.candidates, 20)
    return args


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    src = root / "TRB/set/src"
    if not src.is_dir():
        raise FileNotFoundError("Missing source directory: {}".format(src))
    sys.path.insert(0, str(src))
    from ra_ild_trb.enrich_dictionary_cv import run_project_analysis

    manifest = run_project_analysis(
        project_root=root,
        scopes=args.scopes,
        folds=args.folds,
        repeats=args.repeats,
        candidates=args.candidates,
        seed=args.seed,
        threshold_pct=args.threshold_pct,
        delta_pct=args.delta_pct,
        allow_count_mismatch=args.allow_count_mismatch,
        save_dictionaries=args.save_dictionaries,
        make_plots=not args.no_plots,
        overwrite=args.overwrite,
        result_dir=Path(args.result_dir).expanduser().resolve() if args.result_dir else None,
        plot_dir=Path(args.plot_dir).expanduser().resolve() if args.plot_dir else None,
    )
    print(json.dumps(manifest["scope_counts"], ensure_ascii=False, indent=2))
    print("Completed enrich-dictionary repeated 5-fold validation.")
    print("Summary: {}".format(
        Path(manifest["configuration"]["result_dir"]) / "enrich_5fold_summary.md"
    ))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise
