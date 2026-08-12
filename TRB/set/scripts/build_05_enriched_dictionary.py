#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-5 enriched-dictionary CLI for all/Top-K repertoires."""

from __future__ import annotations

import argparse
import json
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
from ra_ild_trb.enriched_dictionary import (  # noqa: E402
    fit_enriched_dictionary,
    load_reference,
    load_repertoires,
    run_existing_grid_cv_via_source_shim,
    transform_repertoires,
    write_reference,
)
from ra_ild_trb.experiment_spec import load_advanced_experiment  # noqa: E402
from ra_ild_trb.paths import find_repository_root  # noqa: E402

SCRIPT_VERSION = "1.0.0-phase5"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit/transform/CV the Phase-5 enriched dictionary using the experiment repertoire source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["fit", "transform", "cv"], required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--reference-dir", default=None, help="Required for transform mode.")
    p.add_argument("--repository-root", default=None)
    p.add_argument("--id-col", default="libraryid")
    p.add_argument("--deep-source-check", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def _write_feature_table(frame, output_dir: Path, selected_features, *, overwrite: bool) -> Path:
    path = output_dir / "05_enriched_dictionary_features.csv"
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = ["sample_id", *selected_features]
    missing = [c for c in keep if c not in frame.columns]
    if missing:
        raise ValueError(f"selected enriched features missing from score table: {missing}")
    frame.loc[:, keep].to_csv(path, index=False)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo = (
        Path(args.repository_root).expanduser().resolve()
        if args.repository_root
        else find_repository_root(SCRIPT_DIR)
    )
    spec = load_advanced_experiment(args.config)
    if not spec.features.enriched_dictionary.enabled:
        raise ValueError("features.enriched_dictionary.enabled is false")
    metadata = read_and_filter_metadata(args.metadata, spec.cohort.scope, id_col=args.id_col)
    source = resolve_dynamic_repertoire(spec, repo)
    validate_dynamic_source(source, metadata[args.id_col].tolist(), deep=args.deep_source_check)
    enrich = spec.features.enriched_dictionary
    out = Path(args.output_dir).expanduser().resolve()

    print("=" * 78)
    print("TRB Feature Framework Phase 5: enriched dictionary")
    print("=" * 78)
    print(f"Script version:      {SCRIPT_VERSION}")
    print(f"Mode:                {args.mode}")
    print(f"Experiment:          {spec.experiment_id}")
    print(f"Cohort scope:        {spec.cohort.scope}")
    print(f"Repertoire source:   {source.source_id}")
    print(f"AA directory:        {source.aa_dir}")
    print(f"Samples:             {len(metadata)}")
    print(f"T prevalence:        {enrich.threshold_pct:g}%")
    print(f"Delta:               {enrich.delta_pct:g} percentage points")
    print(f"Selected features:   {', '.join(enrich.features)}")
    print(f"Repeats/workers:     {spec.validation.repeats} / {spec.compute.workers}")
    print("- PASS: enriched dictionary source is inherited from the main repertoire.")

    if args.mode == "cv":
        manifest = run_existing_grid_cv_via_source_shim(
            repository_root=repo,
            metadata=metadata,
            source=source,
            output_dir=out,
            threshold_pct=enrich.threshold_pct,
            delta_pct=enrich.delta_pct,
            folds=spec.validation.folds,
            repeats=spec.validation.repeats,
            split_candidates=spec.validation.split_candidates,
            seed=spec.validation.seed,
            workers=spec.compute.workers,
            overwrite=args.overwrite,
        )
        wrapper = {
            "phase": "phase5_enriched_dictionary_cv",
            "experiment": spec.as_dict(),
            "source": source.as_dict(),
            "metadata": str(Path(args.metadata).resolve()),
            "selected_sample_count": len(metadata),
            "legacy_grid_manifest": manifest,
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / "05_enriched_dictionary_cv_manifest.json").write_text(
            json.dumps(wrapper, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("- CV PASS: existing optimized repeated-5-fold grid engine completed one T/Delta pair.")
        print(f"- Output: {out}")
        return 0

    repertoires = load_repertoires(metadata, source, id_col=args.id_col)
    if args.mode == "fit":
        reference = fit_enriched_dictionary(
            metadata,
            repertoires,
            source_id=source.source_id,
            threshold_pct=enrich.threshold_pct,
            delta_pct=enrich.delta_pct,
            id_col=args.id_col,
        )
        outputs = write_reference(reference, out, overwrite=args.overwrite)
        in_sample = transform_repertoires(metadata, repertoires, reference, id_col=args.id_col)
        diag_path = out / "05_enriched_dictionary_fit_scores_DIAGNOSTIC_ONLY.csv"
        if diag_path.exists() and not args.overwrite:
            raise FileExistsError(diag_path)
        in_sample.to_csv(diag_path, index=False)
        print(f"- RA dictionary size:  {reference.ra_size}")
        print(f"- ILD dictionary size: {reference.ild_size}")
        print("- WRITE PASS: reference fitted on the selected training metadata.")
        print("- NOTE: fit-sample scores are DIAGNOSTIC_ONLY and must not be used as CV predictors.")
        for name, path in outputs.items():
            print(f"  - {name}: {path}")
        print(f"  - diagnostic_scores: {diag_path}")
        return 0

    if not args.reference_dir:
        raise ValueError("--reference-dir is required in transform mode")
    reference = load_reference(args.reference_dir)
    if reference.source_id != source.source_id:
        raise ValueError(
            f"reference source {reference.source_id!r} does not match experiment source {source.source_id!r}"
        )
    frame = transform_repertoires(metadata, repertoires, reference, id_col=args.id_col)
    path = _write_feature_table(frame, out, enrich.features, overwrite=args.overwrite)
    audit_path = out / "05_enriched_dictionary_transform_audit.json"
    audit = {
        "experiment_id": spec.experiment_id,
        "source_id": source.source_id,
        "reference_dir": str(Path(args.reference_dir).resolve()),
        "sample_count": len(frame),
        "features": list(enrich.features),
        "status": "PASS",
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("- TRANSFORM PASS: fixed TRAIN dictionary applied to held-out samples.")
    print(f"  - features: {path}")
    print(f"  - audit: {audit_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
