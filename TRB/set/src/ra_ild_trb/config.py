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
    _integer(cv, "outer_repeats", "cross_validation", 1)
    _integer(cv, "outer_folds", "cross_validation", 2)
    _integer(cv, "inner_folds", "cross_validation", 2)
    _string(cv, "outer_assignments", "cross_validation")
    _string(cv, "inner_assignments", "cross_validation")

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

    stability = _mapping(raw, "stability", "root")
    _number(stability, "minimum_selection_frequency", "stability", 0, 1)
    _number(stability, "minimum_sign_consistency", "stability", 0, 1)
    _number(stability, "coefficient_tolerance", "stability", 0)

    final = _mapping(raw, "final_model", "root")
    selected_model = _string(final, "selected_model", "final_model")
    if selected_model not in models:
        raise ConfigError("final_model.selected_model is not defined in models")
    _number(final, "selected_alpha", "final_model", 0, 1, strict_minimum=True)
    _number(final, "selected_lambda", "final_model", 0, strict_minimum=True)
    threshold = _number(final, "locked_threshold", "final_model", 0, 1, strict_minimum=True)
    if threshold >= 1:
        raise ConfigError("final_model.locked_threshold must be < 1")
    for key in ("bundle", "configuration", "reference_masks"):
        _string(final, key, "final_model")

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
