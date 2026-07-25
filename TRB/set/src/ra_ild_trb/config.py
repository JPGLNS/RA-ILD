#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load and validate TRB V2 experiment YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .paths import find_repository_root, resolve_project_path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _mapping(parent: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context}.{key} must be a mapping")
    return value


def _string(parent: Mapping[str, Any], key: str, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _integer(parent: Mapping[str, Any], key: str, context: str, minimum: int = 0) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{context}.{key} must be an integer >= {minimum}")
    return value


def _number(
    parent: Mapping[str, Any],
    key: str,
    context: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    strict_minimum: bool = False,
) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context}.{key} must be numeric")
    value = float(value)
    if minimum is not None:
        bad = value <= minimum if strict_minimum else value < minimum
        if bad:
            op = ">" if strict_minimum else ">="
            raise ConfigError(f"{context}.{key} must be {op} {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{context}.{key} must be <= {maximum}")
    return value


def _string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{context} must be a list")
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{context} must contain non-empty strings")
        out.append(item.strip())
    if len(out) != len(set(out)):
        raise ConfigError(f"{context} contains duplicates")
    return out


def _positive_grid(value: Any, context: str, maximum: Optional[float] = None) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ConfigError(f"{context} must be a non-empty list")
    seen = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigError(f"{context} must contain numeric values")
        number = float(item)
        if number <= 0 or (maximum is not None and number > maximum):
            raise ConfigError(f"Invalid value {number} in {context}")
        if number in seen:
            raise ConfigError(f"Duplicate value {number} in {context}")
        seen.add(number)



def _validate_additional_feature_tables(
    partition: Mapping[str, Any],
    context: str,
) -> None:
    value = partition.get("additional_feature_tables", [])
    if value is None:
        value = []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{context}.additional_feature_tables must be a list")
    observed_paths = []
    for index, item in enumerate(value):
        item_context = f"{context}.additional_feature_tables[{index}]"
        if isinstance(item, str):
            path = item.strip()
            if not path:
                raise ConfigError(f"{item_context} must be a non-empty path")
            observed_paths.append(path)
            continue
        if not isinstance(item, Mapping):
            raise ConfigError(f"{item_context} must be a mapping or path string")
        path = _string(item, "path", item_context)
        key = item.get("key", "sample_id")
        if not isinstance(key, str) or not key.strip():
            raise ConfigError(f"{item_context}.key must be a non-empty string")
        required = _string_list(item.get("required_columns", []), f"{item_context}.required_columns")
        numeric = _string_list(item.get("numeric_columns", []), f"{item_context}.numeric_columns")
        categorical = _string_list(
            item.get("categorical_columns", []),
            f"{item_context}.categorical_columns",
        )
        overlap = sorted(set(numeric) & set(categorical))
        if overlap:
            raise ConfigError(
                f"{item_context} declares columns as both numeric and categorical: {overlap}"
            )
        declared = required + numeric + categorical
        if len(declared) != len(set(declared)):
            # Repetition across required/numeric/categorical is allowed only because a
            # required column is commonly also typed. Duplicates within each list were
            # already rejected by _string_list.
            pass
        observed_paths.append(path)
    if len(observed_paths) != len(set(observed_paths)):
        raise ConfigError(f"{context}.additional_feature_tables contains duplicate paths")


def validate_experiment_mapping(raw: Mapping[str, Any]) -> None:
    """Validate the current TRB V2 configuration schema."""
    if not isinstance(raw, Mapping):
        raise ConfigError("YAML root must be a mapping")
    if str(raw.get("schema_version")) != "1.0":
        raise ConfigError("schema_version must be '1.0'")

    experiment = _mapping(raw, "experiment", "root")
    _string(experiment, "id", "experiment")
    if _string(experiment, "receptor", "experiment") != "TRB":
        raise ConfigError("experiment.receptor must be TRB")
    _integer(experiment, "random_seed", "experiment", 0)

    data = _mapping(raw, "data", "root")
    train = _mapping(data, "train", "data")
    test = _mapping(data, "test", "data")
    _integer(train, "expected_samples", "data.train", 1)
    _integer(test, "expected_samples", "data.test", 1)
    for key in (
        "metadata", "base_matrix", "feature_manifest", "aa_clone_table_dir",
        "public_catalog", "reference_definition",
    ):
        _string(train, key, "data.train")
    for key in (
        "metadata", "final_matrix", "reference_features", "reference_definition_used",
    ):
        _string(test, key, "data.test")

    _validate_additional_feature_tables(train, "data.train")
    _validate_additional_feature_tables(test, "data.test")

    public = _mapping(raw, "public_reference", "root")
    _number(public, "global_min_prevalence", "public_reference", 0, 1)
    _integer(public, "global_min_count", "public_reference", 1)
    _number(public, "group_min_prevalence", "public_reference", 0, 1)
    _integer(public, "group_min_count", "public_reference", 1)
    _number(public, "specific_prevalence_delta", "public_reference", 0, 1)
    _number(public, "epsilon", "public_reference", 0, strict_minimum=True)
    _string(public, "feature_set", "public_reference")
    _string_list(public.get("dynamic_features", []), "public_reference.dynamic_features")
    expected_sizes = _mapping(public, "expected_sizes", "public_reference")
    for key in ("catalog", "global", "RA_specific", "ILD_specific", "shared"):
        _integer(expected_sizes, key, "public_reference.expected_sizes", 0)

    cache = _mapping(public, "cache", "public_reference")
    for key in ("presence", "frequency", "metadata"):
        _string(cache, key, "public_reference.cache")

    regression_task = _mapping(public, "regression_task", "public_reference")
    _integer(regression_task, "outer_repeat", "public_reference.regression_task", 1)
    _integer(regression_task, "outer_fold", "public_reference.regression_task", 1)
    _string(regression_task, "task_dir", "public_reference.regression_task")

    cv = _mapping(raw, "cross_validation", "root")
    outer_repeats = _integer(cv, "outer_repeats", "cross_validation", 1)
    outer_folds = _integer(cv, "outer_folds", "cross_validation", 2)
    inner_folds = _integer(cv, "inner_folds", "cross_validation", 2)
    _string(cv, "outer_assignments", "cross_validation")
    _string(cv, "inner_assignments", "cross_validation")
    configured_executed_folds = cv.get("executed_outer_folds")
    if configured_executed_folds is None:
        executed_outer_folds = tuple(range(1, outer_folds + 1))
    else:
        if not isinstance(configured_executed_folds, Sequence) or isinstance(
            configured_executed_folds, (str, bytes)
        ) or not configured_executed_folds:
            raise ConfigError("cross_validation.executed_outer_folds must be a non-empty list")
        parsed_folds = []
        for index, value in enumerate(configured_executed_folds):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(
                    f"cross_validation.executed_outer_folds[{index}] must be an integer"
                )
            if not 1 <= int(value) <= outer_folds:
                raise ConfigError(
                    "cross_validation.executed_outer_folds contains a fold outside 1..outer_folds"
                )
            parsed_folds.append(int(value))
        if len(parsed_folds) != len(set(parsed_folds)):
            raise ConfigError("cross_validation.executed_outer_folds contains duplicates")
        executed_outer_folds = tuple(parsed_folds)

    repeated_holdout = raw.get("repeated_holdout_training")
    if repeated_holdout is not None:
        if not isinstance(repeated_holdout, Mapping):
            raise ConfigError("repeated_holdout_training must be a mapping")
        if _string(repeated_holdout, "mode", "repeated_holdout_training") != "frozen_repeated_holdout":
            raise ConfigError(
                "repeated_holdout_training.mode must be frozen_repeated_holdout"
            )
        for key in ("split_set_id", "assignments", "frozen_marker", "training_bundle_marker"):
            _string(repeated_holdout, key, "repeated_holdout_training")
        split_count = _integer(
            repeated_holdout, "split_count", "repeated_holdout_training", 1
        )
        train_size = _integer(
            repeated_holdout, "train_size", "repeated_holdout_training", 1
        )
        holdout_size = _integer(
            repeated_holdout, "holdout_size", "repeated_holdout_training", 1
        )
        validation_outer_fold = _integer(
            repeated_holdout,
            "validation_outer_fold",
            "repeated_holdout_training",
            1,
        )
        if split_count != outer_repeats:
            raise ConfigError(
                "repeated_holdout_training.split_count must equal cross_validation.outer_repeats"
            )
        if train_size + holdout_size != int(train["expected_samples"]):
            raise ConfigError(
                "repeated_holdout_training train_size + holdout_size must equal data.train.expected_samples"
            )
        if validation_outer_fold not in executed_outer_folds:
            raise ConfigError(
                "repeated_holdout_training.validation_outer_fold must be executed"
            )
        if tuple(executed_outer_folds) != (validation_outer_fold,):
            raise ConfigError(
                "frozen repeated holdout currently requires exactly one executed validation fold"
            )

    engine = _mapping(raw, "model_engine", "root")
    if _string(engine, "engine", "model_engine") != "elastic_net_logistic":
        raise ConfigError("Current framework supports only elastic_net_logistic")
    if engine.get("class_weight") not in {"balanced", "none"}:
        raise ConfigError("model_engine.class_weight must be balanced or none")
    _positive_grid(engine.get("alpha_grid"), "model_engine.alpha_grid", 1)
    _positive_grid(engine.get("lambda_grid"), "model_engine.lambda_grid")
    _integer(engine, "max_iter", "model_engine", 1)
    _number(engine, "tolerance", "model_engine", 0, strict_minimum=True)
    _number(engine, "zero_sd_tolerance", "model_engine", 0)

    models = _mapping(raw, "models", "root")
    if not models:
        raise ConfigError("models must not be empty")
    for model_name, spec in models.items():
        if not isinstance(spec, Mapping):
            raise ConfigError(f"models.{model_name} must be a mapping")
        predictor_count = 0
        for key in ("numeric", "categorical", "static_feature_groups", "dynamic_public"):
            predictor_count += len(_string_list(spec.get(key, []), f"models.{model_name}.{key}"))
        if predictor_count == 0:
            raise ConfigError(f"models.{model_name} has no predictors")

    preprocessing = _mapping(raw, "preprocessing", "root")
    if _string(preprocessing, "numeric_standardization", "preprocessing") != "zscore_population_ddof0":
        raise ConfigError(
            "preprocessing.numeric_standardization must be zscore_population_ddof0"
        )
    if _string(preprocessing, "categorical_encoding", "preprocessing") != "sorted_reference_dummy":
        raise ConfigError(
            "preprocessing.categorical_encoding must be sorted_reference_dummy"
        )
    if preprocessing.get("zero_variance_filter") is not True:
        raise ConfigError("preprocessing.zero_variance_filter must be true")
    if _string(preprocessing, "unseen_category_policy", "preprocessing") != "reference_all_zero_with_audit":
        raise ConfigError(
            "preprocessing.unseen_category_policy must be reference_all_zero_with_audit"
        )
    preprocessing_task = _mapping(preprocessing, "regression_task", "preprocessing")
    _integer(preprocessing_task, "outer_repeat", "preprocessing.regression_task", 1)
    _integer(preprocessing_task, "outer_fold", "preprocessing.regression_task", 1)
    for key in (
        "task_dir",
        "preprocessing_summary",
        "static_feature_list",
        "outer_train_public",
        "outer_validation_public",
        "coefficients",
    ):
        _string(preprocessing_task, key, "preprocessing.regression_task")

    modeling = _mapping(raw, "modeling", "root")
    positive_label = _string(modeling, "positive_label", "modeling")
    negative_label = _string(modeling, "negative_label", "modeling")
    if positive_label == negative_label:
        raise ConfigError("modeling positive_label and negative_label must differ")
    if _string(modeling, "threshold_selection", "modeling") != "youden_closest_to_0.5":
        raise ConfigError(
            "modeling.threshold_selection must be youden_closest_to_0.5"
        )
    _number(modeling, "coefficient_nonzero_tolerance", "modeling", 0)
    modeling_task = _mapping(modeling, "regression_task", "modeling")
    _integer(modeling_task, "outer_repeat", "modeling.regression_task", 1)
    _integer(modeling_task, "outer_fold", "modeling.regression_task", 1)
    for key in (
        "inner_selected_oof",
        "outer_predictions",
        "outer_metrics",
        "coefficients",
        "independent_predictions",
        "independent_metrics",
    ):
        _string(modeling_task, key, "modeling.regression_task")

    selection = _mapping(raw, "model_selection", "root")
    _string(selection, "static_feature_group", "model_selection")
    _integer(selection, "expected_static_feature_count", "model_selection", 1)
    manifest_rule = _mapping(selection, "manifest_rule", "model_selection")
    if manifest_rule.get("present_in_train_base") is not True:
        raise ConfigError(
            "model_selection.manifest_rule.present_in_train_base must be true"
        )
    _string(manifest_rule, "feature_group_prefix", "model_selection.manifest_rule")
    _string(manifest_rule, "feature_role", "model_selection.manifest_rule")
    if manifest_rule.get("reference_drop") is not False:
        raise ConfigError(
            "model_selection.manifest_rule.reference_drop must be false"
        )
    if _string(selection, "hyperparameter_grid_order", "model_selection") != "alpha_then_lambda":
        raise ConfigError(
            "model_selection.hyperparameter_grid_order must be alpha_then_lambda"
        )
    candidate_sort = selection.get("candidate_sort")
    if not isinstance(candidate_sort, Sequence) or isinstance(candidate_sort, (str, bytes)):
        raise ConfigError("model_selection.candidate_sort must be a list")
    expected_sort = [
        ("pooled_inner_roc_auc", False),
        ("pooled_inner_pr_auc", False),
        ("lambda", False),
        ("l1_ratio_alpha", False),
    ]
    observed_sort = []
    for index, item in enumerate(candidate_sort):
        if not isinstance(item, Mapping):
            raise ConfigError(
                f"model_selection.candidate_sort[{index}] must be a mapping"
            )
        field = _string(item, "field", f"model_selection.candidate_sort[{index}]")
        ascending = item.get("ascending")
        if not isinstance(ascending, bool):
            raise ConfigError(
                f"model_selection.candidate_sort[{index}].ascending must be boolean"
            )
        observed_sort.append((field, ascending))
    if observed_sort != expected_sort:
        raise ConfigError(
            "model_selection.candidate_sort does not match the frozen V1 ranking rule"
        )
    expected_columns = _mapping(selection, "expected_resolved_columns", "model_selection")
    if set(expected_columns) != set(models):
        raise ConfigError(
            "model_selection.expected_resolved_columns must define every model exactly once"
        )
    for model_name, counts in expected_columns.items():
        if not isinstance(counts, Mapping):
            raise ConfigError(
                f"model_selection.expected_resolved_columns.{model_name} must be a mapping"
            )
        _integer(
            counts,
            "numeric",
            f"model_selection.expected_resolved_columns.{model_name}",
            0,
        )
        _integer(
            counts,
            "categorical",
            f"model_selection.expected_resolved_columns.{model_name}",
            0,
        )
    selection_task = _mapping(selection, "regression_task", "model_selection")
    _string(selection_task, "inner_tuning_results", "model_selection.regression_task")

    nested = _mapping(raw, "nested_cv", "root")
    expected_policies = {
        "assignment_policy": "fixed_precomputed",
        "training_public_policy": "exact_leave_one_out",
        "validation_public_policy": "training_reference_only",
        "preprocessing_policy": "fit_on_current_training_partition",
        "candidate_selection_policy": "pooled_roc_pr_lambda_alpha",
    }
    for key, expected in expected_policies.items():
        observed = _string(nested, key, "nested_cv")
        if observed != expected:
            raise ConfigError(f"nested_cv.{key} must be {expected}")
    expected_outer_tasks = _integer(
        nested, "expected_outer_tasks", "nested_cv", 1
    )
    expected_configured_tasks = int(outer_repeats) * len(executed_outer_folds)
    if expected_outer_tasks != expected_configured_tasks:
        raise ConfigError(
            "nested_cv.expected_outer_tasks must equal outer_repeats * executed outer folds"
        )
    expected_inner_fits = _integer(
        nested, "expected_inner_fits_per_task", "nested_cv", 1
    )
    expected_grid_size = len(engine["alpha_grid"]) * len(engine["lambda_grid"])
    expected_fit_count = len(models) * int(cv["inner_folds"]) * expected_grid_size
    if expected_inner_fits != expected_fit_count:
        raise ConfigError(
            "nested_cv.expected_inner_fits_per_task must equal "
            "models * inner_folds * alpha/lambda candidates"
        )
    nested_task = _mapping(nested, "regression_task", "nested_cv")
    _integer(nested_task, "outer_repeat", "nested_cv.regression_task", 1)
    _integer(nested_task, "outer_fold", "nested_cv.regression_task", 1)
    for key in (
        "task_dir",
        "configuration",
        "sample_roles",
        "public_reference_summary",
        "public_loo_assignments",
        "inner_tuning_results",
        "inner_selected_oof",
        "outer_predictions",
        "outer_metrics",
        "coefficients",
        "preprocessing_summary",
        "outer_train_public",
        "outer_validation_public",
    ):
        _string(nested_task, key, "nested_cv.regression_task")

    outer_tasks = _mapping(raw, "outer_tasks", "root")
    for key in ("output_root", "runner_script", "manifest", "status", "logs_dir"):
        _string(outer_tasks, key, "outer_tasks")
    expected_outer_task_count = _integer(
        outer_tasks, "expected_tasks", "outer_tasks", 1
    )
    if expected_outer_task_count != expected_outer_tasks:
        raise ConfigError(
            "outer_tasks.expected_tasks must equal nested_cv.expected_outer_tasks"
        )
    default_workers = _integer(outer_tasks, "default_workers", "outer_tasks", 1)
    max_workers = _integer(outer_tasks, "max_workers", "outer_tasks", 1)
    if default_workers > max_workers:
        raise ConfigError("outer_tasks.default_workers cannot exceed max_workers")
    minimum_complete = _integer(
        outer_tasks, "minimum_complete_tasks_for_validation", "outer_tasks", 0
    )
    if minimum_complete > expected_outer_task_count:
        raise ConfigError(
            "outer_tasks.minimum_complete_tasks_for_validation exceeds expected_tasks"
        )

    aggregation = _mapping(raw, "aggregation", "root")
    _string(aggregation, "output_dir", "aggregation")
    if not isinstance(aggregation.get("require_all_tasks"), bool):
        raise ConfigError("aggregation.require_all_tasks must be boolean")
    expected_metric_rows = _integer(
        aggregation, "expected_metric_rows", "aggregation", 1
    )
    if expected_metric_rows != expected_outer_tasks * len(models):
        raise ConfigError(
            "aggregation.expected_metric_rows must equal outer tasks * models"
        )
    expected_predictions_per_sample = _integer(
        aggregation, "expected_predictions_per_sample", "aggregation", 1
    )
    if expected_predictions_per_sample != int(cv["outer_repeats"]):
        raise ConfigError(
            "aggregation.expected_predictions_per_sample must equal outer_repeats"
        )
    expected_prediction_rows = _integer(
        aggregation, "expected_prediction_rows", "aggregation", 1
    )
    if repeated_holdout is None:
        expected_prediction_total = (
            int(train["expected_samples"])
            * int(cv["outer_repeats"])
            * len(models)
        )
        prediction_rule = "train samples * outer repeats * models"
    else:
        expected_prediction_total = (
            int(repeated_holdout["holdout_size"])
            * int(repeated_holdout["split_count"])
            * len(models)
        )
        prediction_rule = "holdout size * split count * models"
    if expected_prediction_rows != expected_prediction_total:
        raise ConfigError(
            "aggregation.expected_prediction_rows must equal " + prediction_rule
        )

    stability = _mapping(raw, "stability", "root")
    _number(stability, "minimum_selection_frequency", "stability", 0, 1)
    _number(stability, "minimum_sign_consistency", "stability", 0, 1)
    _number(stability, "coefficient_tolerance", "stability", 0)

    final = _mapping(raw, "final_model", "root")
    final_status = str(final.get("status", "locked")).strip()
    if final_status not in {"locked", "not_selected"}:
        raise ConfigError("final_model.status must be locked or not_selected")
    for key in ("bundle", "configuration", "reference_masks"):
        _string(final, key, "final_model")

    if final_status == "locked":
        selected_model = _string(final, "selected_model", "final_model")
        if selected_model not in models:
            raise ConfigError("final_model.selected_model is not defined in models")
        selected_alpha = _number(
            final, "selected_alpha", "final_model", 0, 1, strict_minimum=True
        )
        selected_lambda = _number(
            final, "selected_lambda", "final_model", 0, strict_minimum=True
        )
        candidate_pairs = final.get("candidate_pairs")
        if not isinstance(candidate_pairs, Sequence) or isinstance(candidate_pairs, (str, bytes)) or not candidate_pairs:
            raise ConfigError("final_model.candidate_pairs must be a non-empty list")
        parsed_pairs = []
        for index, pair in enumerate(candidate_pairs):
            if not isinstance(pair, Mapping):
                raise ConfigError(f"final_model.candidate_pairs[{index}] must be a mapping")
            alpha = _number(
                pair,
                "alpha",
                f"final_model.candidate_pairs[{index}]",
                0,
                1,
                strict_minimum=True,
            )
            lambda_value = _number(
                pair,
                "lambda",
                f"final_model.candidate_pairs[{index}]",
                0,
                strict_minimum=True,
            )
            parsed_pairs.append((alpha, lambda_value))
        if len(parsed_pairs) != len(set(parsed_pairs)):
            raise ConfigError("final_model.candidate_pairs contains duplicates")
        if (selected_alpha, selected_lambda) not in set(parsed_pairs):
            raise ConfigError(
                "final_model selected alpha/lambda must appear in candidate_pairs"
            )
        threshold = _number(
            final, "locked_threshold", "final_model", 0, 1, strict_minimum=True
        )
        if threshold >= 1:
            raise ConfigError("final_model.locked_threshold must be < 1")

    validation = _mapping(raw, "independent_validation", "root")
    if not isinstance(validation.get("single_use"), bool):
        raise ConfigError("independent_validation.single_use must be boolean")
    _integer(validation, "bootstrap_reps", "independent_validation", 100)
    _integer(validation, "bootstrap_seed", "independent_validation", 0)
    _string(validation, "output_dir", "independent_validation")


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path
    repository_root: Path
    raw: Mapping[str, Any]

    @property
    def experiment_id(self) -> str:
        return str(self.raw["experiment"]["id"])

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, Mapping):
            raise ConfigError(f"Section {name} is not a mapping")
        return value

    def path(self, dotted_key: str, *, must_exist: bool = False, expect: Optional[str] = None) -> Path:
        current: Any = self.raw
        for token in dotted_key.split("."):
            if not isinstance(current, Mapping) or token not in current:
                raise ConfigError(f"Unknown configuration key: {dotted_key}")
            current = current[token]
        if not isinstance(current, (str, Path)):
            raise ConfigError(f"Configuration key is not a path: {dotted_key}")
        return resolve_project_path(current, self.repository_root, must_exist=must_exist, expect=expect)


def load_experiment_config(path: Path, *, repository_root: Optional[Path] = None) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required. Install with: mamba install -c conda-forge pyyaml"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    validate_experiment_mapping(raw)
    root = Path(repository_root).resolve() if repository_root else find_repository_root(config_path)
    return ExperimentConfig(config_path, root, raw)
