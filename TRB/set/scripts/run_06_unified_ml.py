#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Phase-6 unified leakage-controlled TRB ML evaluation."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0-phase6"
SCRIPT_PATH = Path(__file__).resolve()
SRC_DIR = SCRIPT_PATH.parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.fold_features import load_fold_feature_inputs  # noqa:E402
from ra_ild_trb.unified_cv import load_repeated_holdout_bank, run_repeats  # noqa:E402
from ra_ild_trb.unified_experiment import (  # noqa:E402
    auto_split_assignment_path,
    load_unified_experiment,
)
from ra_ild_trb.unified_modeling import build_candidates  # noqa:E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified TRB repeated-holdout ML with fold-local 3-mer/enriched features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-step02-dir", required=True)
    parser.add_argument("--test-step02-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-metadata", default="TRB/set/train/metadata_train_70.csv")
    parser.add_argument("--test-metadata", default="TRB/set/test/metadata_test_30.csv")
    parser.add_argument("--registry", default="TRB/set/configs/feature_framework/trb_feature_registry_v1.yaml")
    parser.add_argument("--split-assignments", default=None)
    parser.add_argument("--repeat", type=int, default=None, help="Run only one frozen repeat for smoke validation.")
    parser.add_argument("--workers", type=int, default=None, help="Override config workers for this run.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.repeat is not None and args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.workers is not None and not 1 <= args.workers <= 16:
        parser.error("--workers must be in [1,16]")
    return args


def repo_root_from_script() -> Path:
    # TRB/set/scripts/file.py -> repository root is parents[3]
    return SCRIPT_PATH.parents[3]


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def expected_outputs(out: Path) -> Dict[str, Path]:
    return {
        "metrics": out / "06_outer_metrics.csv",
        "predictions": out / "06_outer_predictions.csv.gz",
        "inner_tuning": out / "06_inner_tuning.csv.gz",
        "selected": out / "06_selected_hyperparameters.csv",
        "explanations": out / "06_feature_explanations.csv.gz",
        "feature_audit": out / "06_feature_fit_audit.csv.gz",
        "preprocessing": out / "06_preprocessing_audit.csv.gz",
        "inner_assignments": out / "06_inner_assignment_audit.csv.gz",
        "manifest": out / "06_run_manifest.json",
        "summary": out / "06_run_summary.md",
    }


def prepare_output(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty; use --overwrite: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def summary_markdown(spec, metrics: pd.DataFrame, candidates: int, repeats: List[int]) -> str:
    numeric = ["roc_auc", "pr_auc", "accuracy", "sensitivity_recall", "specificity", "precision", "f1"]
    if bool(metrics["probability_metrics_available"].iloc[0]):
        numeric += ["log_loss", "brier_score"]
    lines = [
        "# Phase-6 Unified ML Summary",
        "",
        f"- Experiment: `{spec.base.experiment_id}`",
        f"- Algorithm: `{spec.algorithm}`",
        f"- Cohort scope: `{spec.base.cohort.scope}`",
        f"- Repertoire source: `{spec.repertoire_source_id}`",
        f"- Repeats completed: **{len(repeats)}**",
        f"- Inner folds: **{spec.cv.inner_folds}**",
        f"- Tuning candidates per repeat: **{candidates}**",
        f"- Positive class: **{spec.model.positive_label}**",
        "",
        "## Across-repeat outer-holdout metrics",
        "",
        "| Metric | Mean | Median | SD | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for col in numeric:
        values = pd.to_numeric(metrics[col], errors="coerce").dropna().to_numpy(float)
        if not len(values):
            continue
        lines.append(
            f"| {col} | {np.mean(values):.4f} | {np.median(values):.4f} | "
            f"{(np.std(values, ddof=1) if len(values)>1 else 0.0):.4f} | {np.min(values):.4f} | {np.max(values):.4f} |"
        )
    lines += [
        "",
        "## Leakage controls",
        "",
        "- 3-mer vocabulary: fit separately on each current training partition.",
        "- Enriched dictionary: validation/holdout uses training-only reference.",
        "- Enriched dictionary training values: exact leave-one-out reference for every training sample.",
        "- Categorical coding, zero-variance filtering, means and SDs: fit on current training partition only.",
        "- Hyperparameters and decision threshold: selected from pooled inner-OOF predictions only.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = repo_root_from_script()
    spec = load_unified_experiment(resolve_path(args.config, root))
    workers = spec.workers if args.workers is None else args.workers
    train_metadata = resolve_path(args.train_metadata, root)
    test_metadata = resolve_path(args.test_metadata, root)
    registry = resolve_path(args.registry, root)
    train_step02 = resolve_path(args.train_step02_dir, root)
    test_step02 = resolve_path(args.test_step02_dir, root)
    out = resolve_path(args.output_dir, root)
    split_path = (
        resolve_path(args.split_assignments, root)
        if args.split_assignments
        else auto_split_assignment_path(spec, root)
    )

    print("=" * 78)
    print("TRB Feature Framework Phase 6: Unified ML + leakage-controlled CV")
    print("=" * 78)
    print(f"Script version:       {SCRIPT_VERSION}")
    print(f"Experiment:           {spec.base.experiment_id}")
    print(f"Algorithm:            {spec.algorithm}")
    print(f"Positive class:       {spec.model.positive_label}")
    print(f"Cohort scope:         {spec.base.cohort.scope}")
    print(f"Repertoire source:    {spec.repertoire_source_id}")
    print(f"3-mer:                enabled={spec.base.features.kmer.enabled}; "
          f"representation={spec.base.features.kmer.representation}; top_n={spec.base.features.kmer.top_n}")
    print(f"Enriched dictionary:  enabled={spec.base.features.enriched_dictionary.enabled}; "
          f"T={spec.base.features.enriched_dictionary.threshold_pct:g}; "
          f"Delta={spec.base.features.enriched_dictionary.delta_pct:g}; "
          f"features={list(spec.base.features.enriched_dictionary.features)}")
    print(f"Inner folds:          {spec.cv.inner_folds}")
    print(f"Configured repeats:   {spec.cv.repeats}")
    print(f"Workers:              {workers}")
    print(f"Split assignments:    {split_path}")

    inputs = load_fold_feature_inputs(
        spec,
        repository_root=root,
        train_metadata=train_metadata,
        test_metadata=test_metadata,
        train_step02_dir=train_step02,
        test_step02_dir=test_step02,
        registry_path=registry,
        validate_source_files=True,
    )
    candidates = build_candidates(spec)
    bank = load_repeated_holdout_bank(
        split_path,
        inputs.sample_ids,
        requested_repeats=spec.cv.repeats,
    )
    repeat_ids = [args.repeat] if args.repeat is not None else list(bank.repeat_ids)
    unavailable = [r for r in repeat_ids if r not in bank.repeat_ids]
    if unavailable:
        raise ValueError(f"requested --repeat is absent from selected frozen bank: {unavailable}")

    print("")
    print("[Resolved contract]")
    print(f"- Samples in scope:          {len(inputs.sample_ids)}")
    print(f"- Static TCR predictors:     {len(inputs.static_feature_names)}")
    print(f"- Clinical numeric:          {list(inputs.clinical_numeric)}")
    print(f"- Clinical categorical:      {list(inputs.clinical_categorical)}")
    print(f"- Hyperparameter candidates: {len(candidates)}")
    print(f"- Repeats to run:            {repeat_ids if len(repeat_ids)<=10 else str(len(repeat_ids)) + ' repeats'}")
    print("- PASS: learned features will be rebuilt inside every inner/outer training partition.")
    print("- PASS: enriched training scores use exact leave-one-out references.")
    print("- PASS: outer holdout is not used for vocabulary, dictionary, preprocessing, tuning or threshold fitting.")

    if args.dry_run:
        print("DRY-RUN PASS: preflight completed; no model fits or output files written.")
        return 0

    prepare_output(out, args.overwrite)
    effective_workers = min(int(workers), len(repeat_ids))
    results = run_repeats(
        inputs,
        spec,
        bank,
        repeat_ids,
        workers=effective_workers,
    )

    metrics = pd.concat([r.metrics for r in results], ignore_index=True)
    predictions = pd.concat([r.predictions for r in results], ignore_index=True)
    tuning = pd.concat([r.inner_tuning for r in results], ignore_index=True, sort=False)
    selected = pd.concat([r.selected_hyperparameters for r in results], ignore_index=True, sort=False)
    explanations = pd.concat([r.explanations for r in results], ignore_index=True, sort=False)
    feature_audit = pd.concat([r.feature_audit for r in results], ignore_index=True, sort=False)
    preprocessing = pd.concat([r.preprocessing_audit for r in results], ignore_index=True, sort=False)
    inner_assignment = pd.concat([r.inner_assignment_audit for r in results], ignore_index=True, sort=False)

    paths = expected_outputs(out)
    metrics.to_csv(paths["metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False, compression="gzip")
    tuning.to_csv(paths["inner_tuning"], index=False, compression="gzip")
    selected.to_csv(paths["selected"], index=False)
    explanations.to_csv(paths["explanations"], index=False, compression="gzip")
    feature_audit.to_csv(paths["feature_audit"], index=False, compression="gzip")
    preprocessing.to_csv(paths["preprocessing"], index=False, compression="gzip")
    inner_assignment.to_csv(paths["inner_assignments"], index=False, compression="gzip")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(root),
        "resolved_experiment": spec.as_dict(),
        "inputs": {
            "train_metadata": str(train_metadata),
            "test_metadata": str(test_metadata),
            "train_step02_dir": str(train_step02),
            "test_step02_dir": str(test_step02),
            "registry": str(registry),
            "split_assignments": str(split_path),
            "repertoire_source": inputs.source.as_dict(),
        },
        "execution": {
            "repeat_ids": repeat_ids,
            "repeat_count": len(repeat_ids),
            "workers": effective_workers,
            "candidate_count": len(candidates),
            "static_tcr_predictor_count": len(inputs.static_feature_names),
        },
        "leakage_contract": {
            "kmer_vocabulary": "current_training_only",
            "enriched_training": "exact_leave_one_out",
            "enriched_validation": "current_training_reference_only",
            "preprocessing": "current_training_only",
            "hyperparameter_selection": "pooled_inner_oof_only",
            "threshold_selection": "selected_candidate_pooled_inner_oof_youden",
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["summary"].write_text(summary_markdown(spec, metrics, len(candidates), repeat_ids), encoding="utf-8")

    print("")
    print(f"WRITE PASS: completed {len(repeat_ids)} repeated holdout(s).")
    print(f"- metrics:      {paths['metrics']}")
    print(f"- predictions:  {paths['predictions']}")
    print(f"- tuning:       {paths['inner_tuning']}")
    print(f"- selected:     {paths['selected']}")
    print(f"- summary:      {paths['summary']}")
    print("")
    print("Metric medians:")
    for col in ("roc_auc", "pr_auc", "accuracy", "sensitivity_recall", "specificity", "f1"):
        print(f"  {col}: {pd.to_numeric(metrics[col]).median():.4f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
