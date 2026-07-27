#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run frozen-model n×n cross-repeat holdout evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, Sequence, Tuple

import pandas as pd
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.config import load_experiment_config
from ra_ild_trb.cross_repeat_evaluation import (
    FrozenTaskArtifacts,
    SelectedModelParameters,
    evaluate_cross_repeat_matrix,
    write_cross_repeat_evaluation,
)
from ra_ild_trb.feature_inputs import load_partition_feature_matrix
from ra_ild_trb.nested_cv import NestedCVOptions
from ra_ild_trb.outer_cv import build_outer_tasks, scan_outer_tasks, task_output_paths
from ra_ild_trb.public_reference import ThresholdScheme
from ra_ild_trb.repeated_holdout_summary import load_frozen_assignments
from ra_ild_trb.specifications import resolve_all_model_specifications, select_static_tcr_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every repeat-specific final model i on every frozen "
            "holdout j, producing an n×n result matrix."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated configured model names or 'all'.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reproduction-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    duplicate = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate:
        raise ValueError(f"{label} duplicated columns: {duplicate[:20]}")
    return frame


def parse_models(text: str, configured: Sequence[str]) -> Tuple[str, ...]:
    configured_models = tuple(map(str, configured))
    if str(text).strip().lower() == "all":
        return configured_models
    models = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not models or len(models) != len(set(models)):
        raise ValueError("--models must be 'all' or a unique non-empty list")
    unknown = sorted(set(models) - set(configured_models))
    if unknown:
        raise ValueError(f"Unknown configured models: {unknown}")
    return models


def load_inputs(config):
    base, _ = load_partition_feature_matrix(config, "train")
    manifest = read_csv(
        config.path("data.train.feature_manifest", must_exist=True, expect="file"),
        "feature manifest",
    )
    outer = read_csv(
        config.path("cross_validation.outer_assignments", must_exist=True, expect="file"),
        "outer assignments",
    )
    inner = read_csv(
        config.path("cross_validation.inner_assignments", must_exist=True, expect="file"),
        "inner assignments",
    )
    metadata_path = config.path("public_reference.cache.metadata", must_exist=True, expect="file")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample_ids = [str(value) for value in metadata.get("sample_ids", [])]
    if "sample_id" not in base.columns:
        raise ValueError("Base matrix is missing sample_id")
    base["sample_id"] = base["sample_id"].astype(str)
    if base["sample_id"].duplicated().any() or base["sample_id"].tolist() != sample_ids:
        raise ValueError("Sparse-cache row order must match unique base-matrix sample order")
    presence = sparse.load_npz(
        config.path("public_reference.cache.presence", must_exist=True, expect="file")
    ).tocsr()
    frequency = sparse.load_npz(
        config.path("public_reference.cache.frequency", must_exist=True, expect="file")
    ).tocsr()
    if presence.shape != frequency.shape or presence.shape[0] != len(base):
        raise ValueError("Sparse cache shape does not match the base matrix")
    return base, manifest, outer, inner, presence, frequency, metadata


def selected_parameters(configuration: Mapping[str, object], models: Sequence[str]):
    values = configuration.get("selected", {})
    if not isinstance(values, Mapping):
        raise ValueError("Task configuration.selected must be a mapping")
    result: Dict[str, SelectedModelParameters] = {}
    for model in models:
        selected = values.get(model)
        if not isinstance(selected, Mapping):
            raise ValueError(f"Missing selected parameters for {model}")
        result[model] = SelectedModelParameters(
            alpha=float(selected["alpha"]),
            lambda_value=float(selected["lambda"]),
            threshold=float(selected["threshold"]),
            inner_roc_auc=float(selected["inner_roc_auc"]),
            inner_pr_auc=float(selected["inner_pr_auc"]),
        )
    return MappingProxyType(result)


def load_task_artifacts(task_dirs: Mapping[int, Path], configured_models: Sequence[str]):
    result: Dict[int, FrozenTaskArtifacts] = {}
    configured_models = tuple(map(str, configured_models))
    for repeat, directory in sorted(task_dirs.items()):
        paths = task_output_paths(directory)
        configuration = json.loads(paths["configuration"].read_text(encoding="utf-8"))
        if int(configuration.get("outer_repeat", -1)) != int(repeat):
            raise ValueError(f"Task configuration repeat mismatch for repeat {repeat}")
        if tuple(map(str, configuration.get("models", []))) != configured_models:
            raise ValueError(f"Task model order mismatch for repeat {repeat}")
        result[int(repeat)] = FrozenTaskArtifacts(
            selected=selected_parameters(configuration, configured_models),
            coefficients=read_csv(paths["coefficients"], f"repeat {repeat} coefficients"),
            preprocessing=read_csv(paths["preprocessing"], f"repeat {repeat} preprocessing"),
            outer_train_public=read_csv(paths["outer_train_public"], f"repeat {repeat} outer-train public"),
            outer_validation_public=read_csv(paths["outer_validation_public"], f"repeat {repeat} outer-validation public"),
            native_predictions=read_csv(paths["outer_predictions"], f"repeat {repeat} native predictions"),
        )
    return MappingProxyType(result)


def main() -> int:
    args = parse_args()
    try:
        if args.reproduction_tolerance < 0:
            raise ValueError("--reproduction-tolerance must be >= 0")
        config = load_experiment_config(
            Path(args.config),
            repository_root=Path(args.repository_root) if args.repository_root else None,
        )
        repeated = config.section("repeated_holdout_training")
        if str(repeated.get("mode")) != "frozen_repeated_holdout":
            raise ValueError("Requires repeated_holdout_training.mode=frozen_repeated_holdout")
        cv = config.section("cross_validation")
        executed_folds = tuple(int(value) for value in cv.get("executed_outer_folds", (1,)))
        if len(executed_folds) != 1:
            raise ValueError("Exactly one outer fold must be executed per repeated holdout")
        executed_fold = executed_folds[0]
        repeats = int(repeated["split_count"])
        configured_models = tuple(config.section("models"))
        models = parse_models(args.models, configured_models)
        engine = config.section("model_engine")
        selection = config.section("model_selection")

        all_tasks = build_outer_tasks(
            int(cv["outer_repeats"]),
            int(cv["outer_folds"]),
            config.path("outer_tasks.output_root"),
        )
        tasks = tuple(task for task in all_tasks if task.outer_fold == executed_fold)
        if len(tasks) != repeats:
            raise ValueError(f"Executed tasks={len(tasks)}, expected repeats={repeats}")
        status = scan_outer_tasks(
            tasks,
            expected_models=configured_models,
            expected_samples=int(config.raw["data"]["train"]["expected_samples"]),
            expected_candidates_per_model=len(engine["alpha_grid"]) * len(engine["lambda_grid"]),
            expected_static_features=int(selection["expected_static_feature_count"]),
        )
        if not status["status"].eq("complete").all():
            bad = status.loc[status["status"] != "complete", ["task_id", "status", "reason"]]
            raise ValueError("All repeated-holdout tasks must be complete:\n" + bad.to_string(index=False))
        artifacts = load_task_artifacts(
            {int(task.outer_repeat): task.output_dir for task in tasks},
            configured_models,
        )

        base, manifest, outer, inner, presence, frequency, _ = load_inputs(config)
        static_features = select_static_tcr_features(
            manifest,
            available_columns=base.columns,
            require_all_available=True,
        )
        public = config.section("public_reference")
        model_specs = resolve_all_model_specifications(
            config.section("models"),
            static_features=static_features,
            allowed_dynamic_public=public["dynamic_features"],
            static_group_name=str(config.raw["model_selection"]["static_feature_group"]),
        )
        modeling = config.section("modeling")
        options = NestedCVOptions(
            base_seed=int(config.raw["experiment"]["random_seed"]),
            class_weight=str(engine["class_weight"]),
            max_iter=int(engine["max_iter"]),
            tolerance=float(engine["tolerance"]),
            zero_sd_tolerance=float(engine["zero_sd_tolerance"]),
            epsilon=float(public["epsilon"]),
            coefficient_nonzero_tolerance=float(modeling["coefficient_nonzero_tolerance"]),
            positive_label=str(modeling["positive_label"]),
            negative_label=str(modeling["negative_label"]),
        )
        sample_ids = base["sample_id"].astype(str).tolist()
        row_lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        labels = base[options.label_column].astype(str).str.upper().to_numpy()
        aa_clone_numbers = pd.to_numeric(base["aa_clone_number"], errors="raise").to_numpy(float)
        scheme = ThresholdScheme.from_mapping(public)

        assignments, _, assignment_sha, _ = load_frozen_assignments(
            config.path("repeated_holdout_training.assignments", must_exist=True, expect="file"),
            config.path("repeated_holdout_training.frozen_marker", must_exist=True, expect="file"),
            expected_split_set_id=str(repeated["split_set_id"]),
            expected_splits=repeats,
            expected_train_size=int(repeated["train_size"]),
            expected_holdout_size=int(repeated["holdout_size"]),
        )
        expected_assignment_rows = int(config.raw["data"]["train"]["expected_samples"]) * repeats
        if len(assignments) != expected_assignment_rows:
            raise ValueError(f"Frozen assignment rows={len(assignments)}, expected={expected_assignment_rows}")

        result = evaluate_cross_repeat_matrix(
            base=base,
            outer_assignments=outer,
            inner_assignments=inner,
            row_lookup=row_lookup,
            presence=presence,
            frequency=frequency,
            labels=labels,
            aa_clone_numbers=aa_clone_numbers,
            model_specifications=model_specs,
            artifacts_by_repeat=artifacts,
            scheme=scheme,
            options=options,
            repeats=repeats,
            outer_folds=int(cv["outer_folds"]),
            inner_folds=int(cv["inner_folds"]),
            executed_outer_fold=executed_fold,
            selected_models=models,
            reproduction_tolerance=float(args.reproduction_tolerance),
        )
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else config.path("aggregation.output_dir").parent / "04_cross_repeat_evaluation"
        )
        paths = write_cross_repeat_evaluation(
            result,
            output_dir,
            experiment_id=config.experiment_id,
            split_set_id=str(repeated["split_set_id"]),
            assignment_sha256=assignment_sha,
            expected_repeats=repeats,
            expected_models=models,
            expected_holdout_size=int(repeated["holdout_size"]),
            reproduction_tolerance=float(args.reproduction_tolerance),
            overwrite=args.overwrite,
        )
        print("TRB frozen-model cross-repeat evaluation: COMPLETE")
        print(f"Repeats:              {repeats}")
        print(f"Selected models:      {len(models)}")
        print(f"Pairs per model:      {repeats} x {repeats} = {repeats * repeats}")
        print(f"Full metric rows:     {len(result.full_metrics)}")
        print(f"Unseen metric rows:   {len(result.unseen_only_metrics)}")
        print(f"Prediction rows:      {len(result.predictions)}")
        print(f"Output:               {output_dir}")
        print(f"Completion marker:    {paths['complete']}")
        return 0
    except Exception as exc:
        print(f"TRB frozen-model cross-repeat evaluation: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
