#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare/validate the all or arbitrary Top-K repertoire cache selected by config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
SET_DIR = SCRIPT_DIR.parent
SRC_DIR = SET_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.dynamic_repertoire import (  # noqa: E402
    read_and_filter_metadata,
    resolve_dynamic_repertoire,
    validate_dynamic_source,
)
from ra_ild_trb.experiment_spec import load_advanced_experiment  # noqa: E402
from ra_ild_trb.paths import find_repository_root  # noqa: E402

TOPK_SCRIPT = "TRB/set/scripts/build_01b_AA_clone_table_topk.py"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare arbitrary Top-K repertoire cache selected by an advanced experiment config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--repository-root", default=None)
    p.add_argument("--id-col", default="libraryid")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo = (
        Path(args.repository_root).expanduser().resolve()
        if args.repository_root
        else find_repository_root(SCRIPT_DIR)
    )
    spec = load_advanced_experiment(args.config)
    metadata = read_and_filter_metadata(args.metadata, spec.cohort.scope, id_col=args.id_col)
    source = resolve_dynamic_repertoire(spec, repo)

    print("=" * 78)
    print("TRB Feature Framework Phase 5: repertoire cache")
    print("=" * 78)
    print(f"Experiment:        {spec.experiment_id}")
    print(f"Source:            {source.source_id}")
    print(f"Selected samples:  {len(metadata)}")

    if spec.repertoire.mode == "all":
        audit = validate_dynamic_source(source, metadata[args.id_col].tolist())
        print(f"- ALL source already materialized: {source.aa_dir}")
        print(f"- VALIDATION PASS: {audit['sample_count']} selected samples covered.")
        return 0

    # Reuse a valid cache instead of recomputing it.
    try:
        audit = validate_dynamic_source(source, metadata[args.id_col].tolist())
        print(f"- CACHE HIT: validated existing {source.source_id} source at {source.aa_dir}")
        print(f"- VALIDATION PASS: {audit['sample_count']} selected samples covered.")
        return 0
    except (FileNotFoundError, ValueError) as initial_error:
        print(f"- Cache not ready: {initial_error}")

    script = repo / TOPK_SCRIPT
    if not script.is_file():
        raise FileNotFoundError(script)
    command = [
        sys.executable,
        str(script),
        "--input-dir", str(repo / "TRB/result/01_AA_clone_table"),
        "--output-dir", str(source.aa_dir),
        "--top-k", str(source.top_k),
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.overwrite:
        command.append("--overwrite")
    print("- Running existing deterministic Top-K builder:")
    print("  " + " ".join(command))
    completed = subprocess.run(command, cwd=repo)
    if completed.returncode != 0:
        raise RuntimeError(f"Top-K builder failed with code {completed.returncode}")
    if args.dry_run:
        print("- DRY-RUN PASS: Top-K cohort preflight completed; no cache written.")
        return 0
    audit = validate_dynamic_source(source, metadata[args.id_col].tolist())
    print(f"- WRITE/VALIDATION PASS: {source.source_id}; selected samples={audit['sample_count']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
