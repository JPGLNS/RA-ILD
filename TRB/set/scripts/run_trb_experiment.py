#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-7 one-command TRB experiment orchestrator."""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_PATH = Path(__file__).resolve()
SRC_DIR = SCRIPT_PATH.parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.orchestrator import (  # noqa:E402
    ORCHESTRATOR_VERSION,
    OrchestratorError,
    build_orchestrator_plan,
    ml_command,
    remove_step02_cache,
    run_checked,
    source_builder_command,
    step02_commands,
    validate_step02_cache,
    validate_dynamic_source,
    write_orchestrator_manifest,
    write_resolved_contract,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a complete TRB experiment from one unified YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True)
    p.add_argument("--repository-root", default=None)
    p.add_argument("--train-metadata", default="TRB/set/train/metadata_train_70.csv")
    p.add_argument("--test-metadata", default="TRB/set/test/metadata_test_30.csv")
    p.add_argument("--registry", default="TRB/set/configs/feature_framework/trb_feature_registry_v1.yaml")
    p.add_argument("--workspace-root", default=None)
    p.add_argument("--train-step02-dir", default=None, help="Optional validated Step02 cache override.")
    p.add_argument("--test-step02-dir", default=None, help="Optional validated Step02 cache override.")
    p.add_argument("--repeat", type=int, default=None, help="Run one frozen repeat as a real-data smoke test.")
    p.add_argument("--workers", type=int, default=None, help="Override config workers (1..16).")
    p.add_argument("--prepare-only", action="store_true", help="Prepare repertoire + Step02 static cache, then stop before ML.")
    p.add_argument("--dry-run", action="store_true", help="Resolve/validate and print planned work without writing or fitting.")
    p.add_argument("--deep-source-check", action="store_true")
    p.add_argument("--rebuild-step02", action="store_true", help="Rebuild only canonical source-level Step02 caches.")
    p.add_argument("--overwrite-ml", action="store_true")
    args = p.parse_args()
    if args.repeat is not None and args.repeat < 1:
        p.error("--repeat must be >= 1")
    if args.workers is not None and not 1 <= args.workers <= 16:
        p.error("--workers must be in [1,16]")
    if args.dry_run and args.rebuild_step02:
        p.error("--dry-run cannot be combined with --rebuild-step02")
    return args


def repo_root() -> Path:
    # TRB/set/scripts/run_trb_experiment.py -> repository root
    return SCRIPT_PATH.parents[3]


def show_command(label: str, command: List[str]) -> None:
    print(f"- {label}:")
    print("  " + " ".join(shlex.quote(str(x)) for x in command))


def main() -> int:
    args = parse_args()
    root = Path(args.repository_root).expanduser().resolve() if args.repository_root else repo_root()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()
    else:
        config_path = config_path.resolve()

    plan = build_orchestrator_plan(
        config_path,
        repository_root=root,
        train_metadata=args.train_metadata,
        test_metadata=args.test_metadata,
        registry=args.registry,
        workspace_root=args.workspace_root,
        train_step02_override=args.train_step02_dir,
        test_step02_override=args.test_step02_dir,
        repeat_override=args.repeat,
        workers_override=args.workers,
        deep_source_check=args.deep_source_check,
    )

    print("=" * 78)
    print("TRB Feature Framework Phase 7: Final Experiment Orchestrator")
    print("=" * 78)
    print(f"Script version:       {ORCHESTRATOR_VERSION}")
    print(f"Experiment:           {plan.spec.base.experiment_id}")
    print(f"Algorithm:            {plan.spec.algorithm}")
    print(f"Cohort scope:         {plan.spec.base.cohort.scope}")
    print(f"Repertoire source:    {plan.source.source_id}")
    print(f"3-mer:                enabled={plan.spec.base.features.kmer.enabled}; "
          f"representation={plan.spec.base.features.kmer.representation}; "
          f"top_n={plan.spec.base.features.kmer.top_n}")
    print(f"Enriched dictionary:  enabled={plan.spec.base.features.enriched_dictionary.enabled}; "
          f"features={list(plan.spec.base.features.enriched_dictionary.features)}")
    print(f"Model repeats:        {plan.spec.cv.repeats}")
    print(f"Repeat override:      {plan.repeat_override}")
    print(f"Workers:              {plan.workers}")
    print(f"Experiment root:      {plan.paths.experiment_root}")
    print(f"Step02 TRAIN:         {plan.paths.train_step02_dir}")
    print(f"Step02 TEST:          {plan.paths.test_step02_dir}")
    print("")
    print("[Preflight]")
    print(f"- Repertoire cache:   {'PASS/CACHE HIT' if plan.source_ready else 'NEEDS BUILD'}")
    if plan.source_error:
        print(f"  detail: {plan.source_error}")
    print(f"- TRAIN Step02 core:  {'PASS/CACHE HIT' if plan.train_step02_ready else 'NEEDS BUILD'}")
    if plan.train_step02_error:
        print(f"  detail: {plan.train_step02_error}")
    print(f"- TEST Step02 core:   {'PASS/CACHE HIT' if plan.test_step02_ready else 'NEEDS BUILD'}")
    if plan.test_step02_error:
        print(f"  detail: {plan.test_step02_error}")
    print("- PASS: Step02 cache is static-core/provenance only; its precomputed 3-mers are ignored by ML.")
    print("- PASS: Phase 6 remains responsible for fold-local 3-mer, Enrich, preprocessing and tuning.")

    source_cmd = source_builder_command(plan)
    if args.dry_run:
        print("")
        print("[Planned actions]")
        if source_cmd is not None:
            show_command("build repertoire source", source_cmd)
        else:
            print("- repertoire source: reuse validated cache")
        if not plan.step02_ready:
            print("- Step02 static-core cache: will be generated after resolved registry/plan are written")
        else:
            print("- Step02 static-core cache: reuse validated cache")
        if args.prepare_only:
            print("- unified ML: skipped (--prepare-only)")
        elif plan.step02_ready and plan.source_ready:
            show_command(
                "Phase-6 ML preflight",
                ml_command(plan, config_path=config_path, dry_run=True),
            )
        else:
            print("- Phase-6 ML: will run after missing prerequisites are materialized")
        print("DRY-RUN PASS: no files written; no model fits executed.")
        return 0

    commands: Dict[str, List[str]] = {}
    source_action = "cache_hit"
    if source_cmd is not None:
        commands["repertoire_builder"] = source_cmd
        run_checked(source_cmd, cwd=root, label="repertoire builder")
        train_ids = list(pd_read_ids(plan.train_metadata))
        test_ids = list(pd_read_ids(plan.test_metadata))
        validate_dynamic_source(plan.source, [*train_ids, *test_ids], deep=args.deep_source_check)
        source_action = "built_and_validated"

    resolved = write_resolved_contract(plan, overwrite=True)
    print(f"- Resolved contract written: {plan.paths.resolved_dir}")

    train_ids = list(pd_read_ids(plan.train_metadata))
    test_ids = list(pd_read_ids(plan.test_metadata))
    train_ready = plan.train_step02_ready
    test_ready = plan.test_step02_ready
    if args.rebuild_step02:
        remove_step02_cache(plan.paths.train_step02_dir, canonical_cache_root=plan.paths.step02_cache_root)
        remove_step02_cache(plan.paths.test_step02_dir, canonical_cache_root=plan.paths.step02_cache_root)
        train_ready = False
        test_ready = False

    step02_action = "cache_hit"
    if not (train_ready and test_ready):
        if (plan.paths.train_step02_dir.exists() or plan.paths.test_step02_dir.exists()) and not args.rebuild_step02:
            raise OrchestratorError(
                "Step02 cache is incomplete/stale. Review it, then rerun with --rebuild-step02 only for canonical cache paths."
            )
        train_cmd, test_cmd = step02_commands(plan, resolved)
        commands["step02_train"] = train_cmd
        commands["step02_test"] = test_cmd
        run_checked(train_cmd, cwd=root, label="Step02 TRAIN static-core cache")
        run_checked(test_cmd, cwd=root, label="Step02 TEST static-core cache")
        validate_step02_cache(
            plan.paths.train_step02_dir,
            expected_ids=train_ids,
            expected_source_id=plan.source.source_id,
            mode="train",
        )
        validate_step02_cache(
            plan.paths.test_step02_dir,
            expected_ids=test_ids,
            expected_source_id=plan.source.source_id,
            mode="test",
        )
        step02_action = "built_and_validated"

    if args.prepare_only:
        write_orchestrator_manifest(
            plan,
            config_path=config_path,
            source_action=source_action,
            step02_action=step02_action,
            ml_action="skipped_prepare_only",
            commands=commands,
        )
        print("PREPARE PASS: repertoire and Step02 static-core prerequisites are ready; ML not executed.")
        return 0

    command = ml_command(
        plan,
        config_path=config_path,
        overwrite_ml=args.overwrite_ml,
        dry_run=False,
    )
    commands["unified_ml"] = command
    show_command("running Phase-6 unified ML", command)
    run_checked(command, cwd=root, label="Phase-6 unified ML")
    write_orchestrator_manifest(
        plan,
        config_path=config_path,
        source_action=source_action,
        step02_action=step02_action,
        ml_action="completed",
        commands=commands,
    )
    print("")
    print("PHASE 7 RUN PASS: end-to-end experiment orchestration completed.")
    print(f"- ML output: {plan.paths.ml_output_dir}")
    print(f"- Orchestrator manifest: {plan.paths.manifest_path}")
    return 0


def pd_read_ids(path: Path):
    import pandas as pd
    frame = pd.read_csv(path, dtype=str)
    return frame["libraryid"].astype(str).str.strip().tolist()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
