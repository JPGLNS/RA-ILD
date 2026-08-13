#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate and resolve a Phase-5 advanced experiment configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SET_DIR = SCRIPT_DIR.parent
SRC_DIR = SET_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.dynamic_repertoire import (  # noqa: E402
    read_and_filter_metadata,
    render_dynamic_registry,
    render_phase2_plan,
    resolve_dynamic_repertoire,
    validate_dynamic_source,
)
from ra_ild_trb.experiment_spec import load_advanced_experiment  # noqa: E402
from ra_ild_trb.paths import find_repository_root  # noqa: E402

SCRIPT_VERSION = "1.0.0-phase5"
DEFAULT_REGISTRY = "TRB/set/configs/feature_framework/trb_feature_registry_v1.yaml"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate Phase-5 advanced experiment config and render dynamic Phase-2 registry/plan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--registry", default=DEFAULT_REGISTRY)
    p.add_argument("--repository-root", default=None)
    p.add_argument("--id-col", default="libraryid")
    p.add_argument("--deep-source-check", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def _write(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    source_audit = validate_dynamic_source(
        source, metadata[args.id_col].tolist(), deep=args.deep_source_check
    )
    registry_path = (repo / args.registry).resolve() if not Path(args.registry).is_absolute() else Path(args.registry).resolve()
    dynamic_registry = render_dynamic_registry(registry_path, spec)
    phase2_plan = render_phase2_plan(spec)

    print("=" * 78)
    print("TRB Feature Framework Phase 5: advanced experiment resolution")
    print("=" * 78)
    print(f"Script version:      {SCRIPT_VERSION}")
    print(f"Experiment:          {spec.experiment_id}")
    print(f"Cohort scope:        {spec.cohort.scope}")
    print(f"Selected samples:    {len(metadata)}")
    print(f"Repertoire source:   {source.source_id}")
    print(f"Repertoire path:     {source.aa_dir}")
    print(f"Static TCR groups:   {len(spec.features.static_tcr_groups)}")
    if spec.features.kmer.enabled:
        print(
            f"3-mer:               {spec.features.kmer.representation}; "
            f"top_n={spec.features.kmer.top_n or 'all retained'}"
        )
    else:
        print("3-mer:               disabled")
    if spec.features.enriched_dictionary.enabled:
        e = spec.features.enriched_dictionary
        print(
            f"Enriched dictionary: T={e.threshold_pct:g}%; Delta={e.delta_pct:g}pp; "
            f"features={','.join(e.features)}"
        )
        print(f"Enrich source:       {source.source_id} (inherited; no independent override)")
    else:
        print("Enriched dictionary: disabled")
    print(f"Model contract:      {spec.model.algorithm} (execution deferred to unified ML phase)")
    print(f"Validation:          folds={spec.validation.folds}; repeats={spec.validation.repeats}")
    print(f"Workers:             {spec.compute.workers} (allowed 1..16; omitted config defaults to 1)")
    print("- PASS: experiment configuration contract validated.")
    print("- PASS: repertoire cache/source covers every selected sample.")
    print("- PASS: enriched dictionary is locked to the main repertoire source.")

    if args.dry_run:
        print("- DRY-RUN PASS: no resolved config files written.")
        return 0

    out = Path(args.output_dir).expanduser().resolve()
    resolved = spec.as_dict()
    resolved["repertoire_cache"] = source.as_dict()
    resolved["source_validation"] = source_audit
    _write(
        out / "05_resolved_experiment.json",
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
        args.overwrite,
    )
    _write(
        out / "05_resolved_registry.yaml",
        yaml.safe_dump(dynamic_registry, sort_keys=False, allow_unicode=True),
        args.overwrite,
    )
    _write(
        out / "05_resolved_phase2_plan.yaml",
        yaml.safe_dump(phase2_plan, sort_keys=False, allow_unicode=True),
        args.overwrite,
    )
    print(f"- WRITE PASS: resolved experiment contract -> {out}")
    print("")
    print("[Phase-2 k-mer options derived from config]")
    k = spec.features.kmer
    if k.enabled:
        print(f"--min-kmer-sample-count {k.min_sample_count}")
        print(f"--min-kmer-prevalence {k.min_prevalence:g}")
        print(f"--min-kmer-unweighted-variance {k.min_unweighted_variance:g}")
        print(f"--min-kmer-weighted-variance {k.min_weighted_variance:g}")
        print(f"--max-kmers {k.top_n or 0}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
