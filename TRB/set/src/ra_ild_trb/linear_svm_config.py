#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration loading and candidate expansion for TRB Linear SVM schemes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from .linear_svm import (
    LinearSVMCandidate,
    LinearSVMError,
    normalize_dual,
    validate_linear_svc_combination,
)


class LinearSVMConfigError(ValueError):
    """Raised when a Linear SVM scheme/configuration is invalid."""


def _mapping(parent: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise LinearSVMConfigError(f"{context}.{key} must be a mapping")
    return value


def _nonempty_string(parent: Mapping[str, Any], key: str, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LinearSVMConfigError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _positive_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LinearSVMConfigError(f"{context} must be numeric")
    number = float(value)
    if number <= 0:
        raise LinearSVMConfigError(f"{context} must be > 0")
    return number


def _list(value: object, context: str) -> list:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise LinearSVMConfigError(f"{context} must be a non-empty list")
    return list(value)


def find_repository_root(start: Path) -> Path:
    current = start.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (
            (candidate / "TRB").is_dir() and (candidate / "IGH").exists()
        ):
            return candidate
    raise LinearSVMConfigError(f"Could not locate repository root from {start}")


@dataclass(frozen=True)
class LinearSVMExperimentConfig:
    raw: Mapping[str, Any]
    source_path: Path
    repository_root: Path

    @property
    def experiment_id(self) -> str:
        return str(self.raw["experiment"]["id"])

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, Mapping):
            raise LinearSVMConfigError(f"Missing mapping section: {name}")
        return value

    def path(
        self,
        dotted: str,
        *,
        must_exist: bool = False,
        expect: Optional[str] = None,
    ) -> Path:
        value: Any = self.raw
        for key in dotted.split("."):
            if not isinstance(value, Mapping) or key not in value:
                raise LinearSVMConfigError(f"Missing path setting: {dotted}")
            value = value[key]
        if not isinstance(value, str) or not value.strip():
            raise LinearSVMConfigError(f"{dotted} must be a path string")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.repository_root / path
        path = path.resolve()
        if must_exist and not path.exists():
            raise LinearSVMConfigError(f"Required path does not exist: {path}")
        if expect == "file" and path.exists() and not path.is_file():
            raise LinearSVMConfigError(f"Expected file: {path}")
        if expect == "dir" and path.exists() and not path.is_dir():
            raise LinearSVMConfigError(f"Expected directory: {path}")
        return path


def _candidate_id(block_id: str, penalty: str, loss: str, weight: str, C: float) -> str:
    c_text = format(float(C), ".12g").replace("-", "m").replace(".", "p").replace("+", "")
    return f"{block_id}__{penalty}__{loss}__cw_{weight}__C_{c_text}"


def expand_linear_svm_candidates(engine: Mapping[str, Any]) -> List[LinearSVMCandidate]:
    if str(engine.get("engine", "")).strip() != "linear_svc":
        raise LinearSVMConfigError("model_engine.engine must be linear_svc")
    blocks = _list(engine.get("candidate_blocks"), "model_engine.candidate_blocks")
    candidates: List[LinearSVMCandidate] = []
    seen_ids = set()
    seen_combinations = set()
    for index, raw in enumerate(blocks):
        context = f"model_engine.candidate_blocks[{index}]"
        if not isinstance(raw, Mapping):
            raise LinearSVMConfigError(f"{context} must be a mapping")
        block_id = _nonempty_string(raw, "id", context)
        priority = raw.get("priority", index + 1)
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 1:
            raise LinearSVMConfigError(f"{context}.priority must be an integer >= 1")
        penalty_values = raw.get("penalty_grid", [raw.get("penalty")])
        loss_values = raw.get("loss_grid", [raw.get("loss")])
        weight_values = raw.get("class_weight_grid", [raw.get("class_weight", "balanced")])
        C_values = raw.get("C_grid")
        penalties = [str(x).strip().lower() for x in _list(penalty_values, f"{context}.penalty_grid")]
        losses = [str(x).strip().lower() for x in _list(loss_values, f"{context}.loss_grid")]
        weights = [str(x).strip().lower() for x in _list(weight_values, f"{context}.class_weight_grid")]
        Cs = [_positive_float(x, f"{context}.C_grid") for x in _list(C_values, f"{context}.C_grid")]
        dual = normalize_dual(raw.get("dual", "auto"))
        for penalty, loss, weight, C in product(penalties, losses, weights, Cs):
            if weight not in {"none", "balanced"}:
                raise LinearSVMConfigError(
                    f"{context}.class_weight_grid values must be none or balanced"
                )
            try:
                validate_linear_svc_combination(
                    penalty=penalty, loss=loss, dual=dual
                )
            except LinearSVMError as exc:
                raise LinearSVMConfigError(f"{context}: {exc}") from exc
            candidate_id = _candidate_id(block_id, penalty, loss, weight, C)
            combination = (penalty, loss, weight, float(C), str(dual))
            if candidate_id in seen_ids or combination in seen_combinations:
                raise LinearSVMConfigError(
                    f"Duplicate Linear SVM candidate in {context}: {combination}"
                )
            seen_ids.add(candidate_id)
            seen_combinations.add(combination)
            candidates.append(
                LinearSVMCandidate(
                    candidate_id=candidate_id,
                    block_id=block_id,
                    priority=int(priority),
                    C=float(C),
                    penalty=penalty,
                    loss=loss,
                    class_weight=weight,
                    dual=dual,
                )
            )
    if not candidates:
        raise LinearSVMConfigError("No Linear SVM candidates were expanded")
    return candidates


def validate_linear_svm_mapping(raw: Mapping[str, Any]) -> None:
    if not isinstance(raw, Mapping):
        raise LinearSVMConfigError("YAML root must be a mapping")
    if str(raw.get("schema_version")) != "1.0":
        raise LinearSVMConfigError("schema_version must be '1.0'")
    experiment = _mapping(raw, "experiment", "root")
    _nonempty_string(experiment, "id", "experiment")
    if _nonempty_string(experiment, "receptor", "experiment") != "TRB":
        raise LinearSVMConfigError("experiment.receptor must be TRB")
    for section in (
        "data",
        "public_reference",
        "cross_validation",
        "models",
        "preprocessing",
        "modeling",
        "model_selection",
        "nested_cv",
        "outer_tasks",
    ):
        _mapping(raw, section, "root")
    models = raw["models"]
    if not models:
        raise LinearSVMConfigError("models must not be empty")
    for name, spec in models.items():
        if not isinstance(spec, Mapping):
            raise LinearSVMConfigError(f"models.{name} must be a mapping")
        count = 0
        for key in ("numeric", "categorical", "static_feature_groups", "dynamic_public"):
            value = spec.get(key, [])
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise LinearSVMConfigError(f"models.{name}.{key} must be a list")
            count += len(value)
        if count == 0:
            raise LinearSVMConfigError(f"models.{name} has no predictors")

    engine = _mapping(raw, "model_engine", "root")
    candidates = expand_linear_svm_candidates(engine)
    max_iter = engine.get("max_iter", 10000)
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise LinearSVMConfigError("model_engine.max_iter must be an integer >= 1")
    _positive_float(engine.get("tolerance", 1.0e-4), "model_engine.tolerance")
    _positive_float(
        engine.get("intercept_scaling", 1.0), "model_engine.intercept_scaling"
    )
    zero_sd = engine.get("zero_sd_tolerance", 1.0e-12)
    if isinstance(zero_sd, bool) or not isinstance(zero_sd, (int, float)) or float(zero_sd) < 0:
        raise LinearSVMConfigError("model_engine.zero_sd_tolerance must be >= 0")
    if engine.get("fit_intercept", True) is not True:
        raise LinearSVMConfigError("Batch 15 requires model_engine.fit_intercept=true")

    selection = raw["model_selection"]
    if str(selection.get("tuning_primary_metric", "roc_auc")) != "roc_auc":
        raise LinearSVMConfigError(
            "Linear SVM tuning_primary_metric must be roc_auc; probability-only log_loss is unsupported"
        )
    threshold = str(selection.get("threshold_source", "inner_oof_youden"))
    if threshold != "inner_oof_youden":
        raise LinearSVMConfigError(
            "model_selection.threshold_source must be inner_oof_youden"
        )
    expected_count = selection.get("expected_candidate_count")
    if expected_count is not None and int(expected_count) != len(candidates):
        raise LinearSVMConfigError(
            f"expected_candidate_count={expected_count}, expanded={len(candidates)}"
        )


def load_linear_svm_config(
    path: Path, *, repository_root: Optional[Path] = None
) -> LinearSVMExperimentConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise LinearSVMConfigError(f"Configuration not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    validate_linear_svm_mapping(raw)
    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else find_repository_root(source)
    )
    return LinearSVMExperimentConfig(raw=raw, source_path=source, repository_root=root)


def candidate_summary(candidates: Sequence[LinearSVMCandidate]) -> List[Dict[str, object]]:
    return [candidate.as_dict() for candidate in candidates]
