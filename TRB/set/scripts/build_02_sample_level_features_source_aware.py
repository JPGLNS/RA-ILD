#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-2 source-aware entry point for TRB Step02 feature generation.

The legacy ``build_02_sample_level_features.py`` remains the numerical kernel.
This wrapper resolves a feature plan, chooses the registered repertoire source,
adapts source-specific QC/summary semantics, and delegates all biological
feature calculations to the existing Step02 functions.

Phase 2 deliberately still produces the complete legacy Step02 core plus both
3-mer representations. Config-driven *execution* of individual feature modules
is deferred to Phase 3; the feature plan is used here to route the repertoire
source and record provenance without changing feature mathematics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SET_DIR = SCRIPT_DIR.parent
SRC_DIR = SET_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.feature_framework import (  # noqa: E402
    load_feature_plan,
    load_feature_registry,
)
from ra_ild_trb.paths import find_repository_root, resolve_project_path  # noqa: E402
from ra_ild_trb.repertoire_input import (  # noqa: E402
    build_adapted_summary_table,
    resolve_repertoire_input,
)

SCRIPT_VERSION = "2.0.1-source-aware-phase2"
DEFAULT_REGISTRY = "TRB/set/configs/feature_framework/trb_feature_registry_v1.yaml"
DEFAULT_PLAN = "TRB/set/configs/feature_framework/trb_feature_plan_template_v1.yaml"
DEFAULT_LEGACY_STEP02 = "TRB/set/scripts/build_02_sample_level_features.py"


def load_legacy_step02(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"legacy Step02 script not found: {path}")
    spec = importlib.util.spec_from_file_location("ra_ild_trb_legacy_step02", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy Step02 script: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses consult sys.modules while decorating classes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build TRB Step02 features using a Feature Framework repertoire source "
            "while preserving the legacy numerical feature definitions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id-col", default="libraryid")

    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--legacy-step02-script", default=DEFAULT_LEGACY_STEP02)

    parser.add_argument(
        "--aa-dir",
        default=None,
        help="Optional override of the selected registry source directory.",
    )
    parser.add_argument(
        "--source-summary-file",
        default=None,
        help="Optional override of the selected source summary file.",
    )
    parser.add_argument(
        "--parent-summary-file",
        default=None,
        help="Optional parent Step01 summary override for derived sources such as Top-K.",
    )

    parser.add_argument("--vocabulary-file", default=None)
    parser.add_argument("--min-kmer-sample-count", type=int, default=5)
    parser.add_argument("--min-kmer-prevalence", type=float, default=0.05)
    parser.add_argument("--min-kmer-unweighted-variance", type=float, default=1e-12)
    parser.add_argument("--min-kmer-weighted-variance", type=float, default=1e-12)
    parser.add_argument("--max-kmers", type=int, default=0)
    parser.add_argument("--norm-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--invalid-aa-policy", choices=["error", "skip"], default="error"
    )
    parser.add_argument("--write-merged", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if not 0.0 <= args.min_kmer_prevalence <= 1.0:
        parser.error("--min-kmer-prevalence must be between 0 and 1")
    if args.min_kmer_sample_count < 1:
        parser.error("--min-kmer-sample-count must be >= 1")
    if args.min_kmer_unweighted_variance < 0 or args.min_kmer_weighted_variance < 0:
        parser.error("3-mer variance thresholds must be >= 0")
    if args.max_kmers < 0:
        parser.error("--max-kmers must be >= 0")
    if args.norm_tolerance <= 0:
        parser.error("--norm-tolerance must be > 0")
    if args.mode == "test" and not args.vocabulary_file:
        parser.error("--vocabulary-file is required in TEST mode")
    return args


def _resolve_config_path(value: str, repo_root: Path) -> Path:
    return resolve_project_path(value, repo_root, must_exist=True, expect="file")


def _canonical_split_result_dir(metadata: Path, repo_root: Path, mode: str) -> Optional[Path]:
    """Return TRB/set/<mode>/result for canonical split metadata, else None.

    A canonical train/test metadata file may have any filename, but it must live
    under ``TRB/set/train`` or ``TRB/set/test`` matching ``--mode``.  This keeps
    source routing tied to the current split instead of silently falling back to
    historical global Step01 outputs.
    """
    root = repo_root.resolve()
    metadata = metadata.resolve()
    detected = None
    for split in ("train", "test"):
        split_dir = (root / "TRB" / "set" / split).resolve()
        try:
            metadata.relative_to(split_dir)
        except ValueError:
            continue
        detected = split
        break
    if detected is None:
        return None
    if detected != mode:
        raise ValueError(
            f"metadata belongs to split {detected!r} but --mode is {mode!r}: {metadata}"
        )
    return root / "TRB" / "set" / detected / "result"


def _split_aware_overrides(
    args: argparse.Namespace,
    registry,
    resolved_plan,
    metadata: Path,
    repo_root: Path,
):
    """Infer canonical split-specific Step01 paths when the user did not override them."""
    aa_override = args.aa_dir
    source_summary_override = args.source_summary_file
    parent_summary_override = args.parent_summary_file
    notes: List[str] = []

    split_result = _canonical_split_result_dir(metadata, repo_root, args.mode)
    if split_result is None:
        return aa_override, source_summary_override, parent_summary_override, notes

    source = registry.repertoire_sources[resolved_plan.repertoire_source.id]

    # Full repertoire analyses should use the Step01 output belonging to the
    # current train/test split, not a historical global TRB/result directory.
    if source.representation == "full":
        layer_dir = split_result / source.source_layer
        summary_path = layer_dir / Path(source.summary_path).name
        if aa_override is None:
            if not layer_dir.is_dir():
                raise FileNotFoundError(
                    f"canonical split Step01 directory not found for {args.mode}: {layer_dir}"
                )
            aa_override = str(layer_dir)
            notes.append(f"AA directory inferred from {args.mode} split: {layer_dir}")
        if source_summary_override is None:
            if not summary_path.is_file():
                raise FileNotFoundError(
                    f"canonical split Step01 summary not found for {args.mode}: {summary_path}"
                )
            source_summary_override = str(summary_path)
            notes.append(f"source summary inferred from {args.mode} split: {summary_path}")

    # Derived sources such as Top-K may remain global, but their parent full
    # Step01 provenance/QC must follow the current split.
    if source.parent_source is not None and parent_summary_override is None:
        parent = registry.repertoire_sources[source.parent_source]
        parent_summary = (
            split_result / parent.source_layer / Path(parent.summary_path).name
        )
        if not parent_summary.is_file():
            raise FileNotFoundError(
                f"canonical parent Step01 summary not found for {args.mode}: {parent_summary}"
            )
        parent_summary_override = str(parent_summary)
        notes.append(
            f"parent Step01 summary inferred from {args.mode} split: {parent_summary}"
        )

    return aa_override, source_summary_override, parent_summary_override, notes


def _legacy_args(args: argparse.Namespace, resolved_input, metadata: Path, output_dir: Path):
    # Namespace fields intentionally mirror the current legacy Step02 contract.
    return argparse.Namespace(
        mode=args.mode,
        aa_dir=str(resolved_input.aa_dir),
        summary_file=str(resolved_input.summary_path),
        metadata=str(metadata),
        id_col=args.id_col,
        output_dir=str(output_dir),
        aa_input_suffix="",  # file names are resolved from the registry, not a suffix.
        vocabulary_file=args.vocabulary_file,
        min_kmer_sample_count=args.min_kmer_sample_count,
        min_kmer_prevalence=args.min_kmer_prevalence,
        min_kmer_unweighted_variance=args.min_kmer_unweighted_variance,
        min_kmer_weighted_variance=args.min_kmer_weighted_variance,
        max_kmers=args.max_kmers,
        norm_tolerance=args.norm_tolerance,
        summary_mismatch_policy="error",
        invalid_aa_policy=args.invalid_aa_policy,
        write_merged=args.write_merged,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


def _expected_outputs(legacy, output_dir: Path, mode: str, write_merged: bool) -> Dict[str, Path]:
    outputs = dict(legacy.expected_output_paths(output_dir, mode, write_merged))
    outputs["repertoire_source_audit"] = output_dir / "02_repertoire_source_audit.csv"
    outputs["resolved_feature_source"] = output_dir / "02_resolved_feature_source.json"
    return outputs


def _check_overwrite(outputs: Mapping[str, Path], overwrite: bool, dry_run: bool) -> None:
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite and not dry_run:
        shown = "\n".join(f"  - {path}" for path in existing[:20])
        raise FileExistsError(
            "source-aware Step02 outputs already exist; add --overwrite after review:\n" + shown
        )


def _write_json_atomic(payload: Mapping[str, object], path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _source_markdown_section(resolved_plan, resolved_input, audit_df: pd.DataFrame) -> str:
    selected = [group.id for group in resolved_plan.feature_groups]
    lines = [
        "",
        "## Feature Framework Phase-2 source routing",
        "",
        f"- Plan: `{resolved_plan.plan_id}`",
        f"- Repertoire source: `{resolved_input.source.id}`",
        f"- Representation: `{resolved_input.source.representation}`",
        f"- Source layer: `{resolved_input.source.source_layer}`",
        f"- AA directory: `{resolved_input.aa_dir}`",
        f"- Source summary: `{resolved_input.summary_path}`",
        f"- Parent summary: `{resolved_input.parent_summary_path or 'none'}`",
        f"- Canonical clone weight: `{resolved_input.source.weight_column}`",
        f"- Top-K: `{resolved_input.source.top_k}`",
        f"- Feature groups selected in plan: `{', '.join(selected)}`",
        "- Phase-2 execution scope: the complete legacy Step02 core + both 3-mer matrices are still calculated.",
        "- Per-group execution is intentionally deferred to Phase 3; this run only makes repertoire routing source-aware.",
        "",
        "### Source audit overview",
        "",
        f"- Samples audited: **{len(audit_df)}**",
        f"- Analysis AA clone range: **{int(audit_df['analysis_aa_clone_number'].min())} – {int(audit_df['analysis_aa_clone_number'].max())}**",
        f"- Retained read-mass range: **{audit_df['retained_read_mass'].min():.6g} – {audit_df['retained_read_mass'].max():.6g}**",
        f"- Source QC pass: **{int(audit_df['source_qc_pass'].astype(bool).sum())}/{len(audit_df)}**",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    started = time.time()

    if args.repository_root:
        repo_root = Path(args.repository_root).expanduser().resolve()
    else:
        repo_root = find_repository_root(SCRIPT_DIR)

    registry_path = _resolve_config_path(args.registry, repo_root)
    plan_path = _resolve_config_path(args.plan, repo_root)
    legacy_path = _resolve_config_path(args.legacy_step02_script, repo_root)
    metadata = resolve_project_path(args.metadata, repo_root, must_exist=True, expect="file")
    output_dir = resolve_project_path(args.output_dir, repo_root, must_exist=False)

    registry = load_feature_registry(registry_path)
    plan = load_feature_plan(plan_path)
    resolved_plan = registry.resolve_plan(plan)
    (
        aa_override,
        source_summary_override,
        parent_summary_override,
        routing_notes,
    ) = _split_aware_overrides(args, registry, resolved_plan, metadata, repo_root)
    resolved_input = resolve_repertoire_input(
        registry,
        resolved_plan.repertoire_source.id,
        repo_root,
        aa_dir_override=aa_override,
        summary_override=source_summary_override,
        parent_summary_override=parent_summary_override,
        must_exist=True,
    )
    legacy = load_legacy_step02(legacy_path)
    legacy_args = _legacy_args(args, resolved_input, metadata, output_dir)

    sample_ids = legacy.read_metadata_ids(metadata, args.id_col)
    adapted_summary, source_audit = build_adapted_summary_table(
        resolved_input,
        sample_ids,
        normalization_tolerance=args.norm_tolerance,
    )

    aa_files = {sample_id: resolved_input.sample_path(sample_id) for sample_id in sample_ids}
    missing = [path for path in aa_files.values() if not path.is_file()]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} repertoire sample files:\n{preview}")

    outputs = _expected_outputs(legacy, output_dir, args.mode, args.write_merged)
    _check_overwrite(outputs, args.overwrite, args.dry_run)

    training_vocab_df = None
    retained_kmers_from_train: Optional[List[str]] = None
    retained_set: Optional[Set[str]] = None
    if args.mode == "test":
        vocab_path = resolve_project_path(
            args.vocabulary_file, repo_root, must_exist=True, expect="file"
        )
        legacy_args.vocabulary_file = str(vocab_path)
        training_vocab_df, retained_kmers_from_train = legacy.read_training_vocabulary(vocab_path)
        retained_set = set(retained_kmers_from_train)

    print("")
    print("=" * 78)
    print("02_sample_level_features Phase 2：source-aware routing")
    print("=" * 78)
    print(f"脚本版本: {SCRIPT_VERSION}")
    print(f"Feature plan: {resolved_plan.plan_id}")
    print(f"Repertoire source: {resolved_input.source.id}")
    print(f"Representation: {resolved_input.source.representation}")
    print(f"AA clone 表目录: {resolved_input.aa_dir}")
    print(f"Source summary: {resolved_input.summary_path}")
    if resolved_input.parent_summary_path:
        print(f"Parent Step01 summary: {resolved_input.parent_summary_path}")
    for note in routing_notes:
        print(f"Routing: {note}")
    print(f"样本数: {len(sample_ids)}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Dry-run: {args.dry_run}")
    print("")

    core_rows = []
    unweighted_by_sample = {}
    weighted_by_sample = {}
    logs = []

    for index, sample_id in enumerate(sample_ids, start=1):
        core, unweighted, weighted, log = legacy.process_one_sample(
            sample_id=sample_id,
            aa_file=aa_files[sample_id],
            summary_row=adapted_summary.loc[sample_id],
            retained_kmers=retained_set,
            enumerate_all_kmers=(args.mode == "train"),
            args=legacy_args,
        )
        if log.summary_mismatch_count:
            raise RuntimeError(
                f"{sample_id}: source adapter failed legacy summary contract: "
                f"{log.summary_mismatch_detail}"
            )
        core_rows.append(core)
        unweighted_by_sample[sample_id] = unweighted
        weighted_by_sample[sample_id] = weighted
        logs.append(log)
        print(
            f"[{index:>3}/{len(sample_ids)}] {sample_id} | "
            f"source={resolved_input.source.id} | AA={log.aa_clone_number:,} | "
            f"Shannon={log.Shannon:.4f} | top1={log.top1_frequency:.6f} | PASS"
        )

    core_df = pd.DataFrame(core_rows)
    if args.mode == "train":
        vocab_df = legacy.determine_vocabulary(
            sample_ids=sample_ids,
            unweighted_by_sample=unweighted_by_sample,
            weighted_by_sample=weighted_by_sample,
            args=legacy_args,
        )
        retained_kmers = vocab_df.loc[vocab_df["keep"], "kmer"].astype(str).tolist()
    else:
        assert training_vocab_df is not None
        assert retained_kmers_from_train is not None
        vocab_df = training_vocab_df.copy()
        retained_kmers = retained_kmers_from_train

    unweighted_df = legacy.build_kmer_matrix(
        sample_ids, unweighted_by_sample, retained_kmers, prefix="unweighted"
    )
    weighted_df = legacy.build_kmer_matrix(
        sample_ids, weighted_by_sample, retained_kmers, prefix="weighted"
    )

    merged_df = None
    if args.write_merged:
        merged_df = (
            core_df.merge(unweighted_df, on="sample_id", how="left", validate="one_to_one")
            .merge(weighted_df, on="sample_id", how="left", validate="one_to_one")
        )

    for name, frame in [
        ("core", core_df),
        ("unweighted", unweighted_df),
        ("weighted", weighted_df),
    ]:
        if frame["sample_id"].astype(str).tolist() != list(sample_ids):
            raise RuntimeError(f"{name} matrix sample order differs from metadata")
        if frame["sample_id"].duplicated().any():
            raise RuntimeError(f"{name} matrix has duplicate sample_id")
        numeric = frame.drop(columns=["sample_id"]).to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"{name} matrix contains NaN/Inf")

    resolved_payload = {
        "phase": "feature_framework_v1_phase2_source_aware_step02",
        "script_version": SCRIPT_VERSION,
        "registry": str(registry_path),
        "plan": resolved_plan.as_dict(),
        "repertoire_input": resolved_input.as_dict(),
        "metadata": str(metadata),
        "mode": args.mode,
        "sample_count": len(sample_ids),
        "phase2_generation_scope": "legacy_step02_complete_core_plus_both_3mer",
        "feature_group_execution_deferred_to_phase3": True,
    }

    total_seconds = time.time() - started
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        legacy.write_dataframe_atomic(core_df, outputs["core"])
        legacy.write_dataframe_atomic(unweighted_df, outputs["unweighted"])
        legacy.write_dataframe_atomic(weighted_df, outputs["weighted"])
        if args.mode == "train":
            legacy.write_dataframe_atomic(vocab_df, outputs["vocabulary"])
        else:
            vocab_used = vocab_df.copy()
            vocab_used.insert(0, "vocabulary_source", str(legacy_args.vocabulary_file))
            legacy.write_dataframe_atomic(vocab_used, outputs["vocabulary_used"])
        legacy.write_dataframe_atomic(pd.DataFrame([vars(x) for x in logs]), outputs["build_log"])
        if args.write_merged and merged_df is not None:
            legacy.write_dataframe_atomic(merged_df, outputs["merged"])
        legacy.write_dataframe_atomic(source_audit, outputs["repertoire_source_audit"])
        _write_json_atomic(resolved_payload, outputs["resolved_feature_source"])

        markdown = legacy.make_markdown_summary(
            args=legacy_args,
            sample_ids=sample_ids,
            core_df=core_df,
            vocab_df=vocab_df,
            retained_kmers=retained_kmers,
            logs=logs,
            outputs=outputs,
            total_seconds=total_seconds,
        )
        markdown += _source_markdown_section(resolved_plan, resolved_input, source_audit)
        temp = outputs["summary"].with_suffix(".md.tmp")
        temp.write_text(markdown + "\n", encoding="utf-8")
        temp.replace(outputs["summary"])

    print("")
    print("[Phase-2 source contract]")
    print(f"- Repertoire source: {resolved_input.source.id}")
    print(
        f"- Analysis AA clone range: {int(source_audit['analysis_aa_clone_number'].min()):,} – "
        f"{int(source_audit['analysis_aa_clone_number'].max()):,}"
    )
    print(
        f"- Retained read-mass range: {source_audit['retained_read_mass'].min():.6g} – "
        f"{source_audit['retained_read_mass'].max():.6g}"
    )
    print(f"- Core columns including sample_id: {core_df.shape[1]}")
    print(f"- Retained 3-mers: {len(retained_kmers):,}")
    if args.dry_run:
        print("- DRY-RUN PASS: complete calculation/QC finished; no Step02 files written.")
    else:
        print(f"- WRITE PASS: {output_dir}")
    print(f"- Runtime: {total_seconds:.2f} seconds")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nERROR: user interrupted run", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
