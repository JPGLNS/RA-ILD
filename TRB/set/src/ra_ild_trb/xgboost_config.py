#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration and deterministic Random Search candidate-bank support for Batch 18."""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from .xgboost_model import XGBoostCandidate, canonical_parameter_json


class XGBoostConfigError(ValueError):
    pass


RESERVED_HYPERPARAMETERS = {"random_state", "seed", "n_jobs", "nthread", "early_stopping_rounds", "callbacks"}
SUPPORTED_TUNING_METRICS = {"roc_auc", "log_loss"}


def find_repository_root(start: Path) -> Path:
    current = start.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or ((candidate / "TRB").is_dir() and (candidate / "IGH").exists()):
            return candidate
    raise XGBoostConfigError(f"Could not locate repository root from {start}")


def _mapping(parent: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise XGBoostConfigError(f"{context}.{key} must be a mapping")
    return value


def _scalar_or_nonempty_list(value: Any, context: str) -> List[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise XGBoostConfigError(f"{context} list must not be empty")
        values = list(value)
    else:
        values = [value]
    for item in values:
        if isinstance(item, (Mapping, list, tuple, set)):
            raise XGBoostConfigError(f"{context} values must be scalar")
        if isinstance(item, float) and not np.isfinite(item):
            raise XGBoostConfigError(f"{context} contains non-finite value")
    return values


def _validate_common_parameter(name: str, value: Any, context: str) -> None:
    numeric_nonnegative = {
        "gamma", "min_child_weight", "reg_alpha", "reg_lambda", "max_delta_step"
    }
    unit_interval_open_left = {"subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode"}
    positive_float = {"learning_rate", "eta", "scale_pos_weight"}
    positive_int = {"n_estimators", "max_leaves", "max_bin"}
    if name == "max_depth":
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise XGBoostConfigError(f"{context} must be an integer >= 0")
    elif name in positive_int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise XGBoostConfigError(f"{context} must be an integer >= 1")
    elif name in numeric_nonnegative:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or float(value) < 0:
            raise XGBoostConfigError(f"{context} must be numeric >= 0")
    elif name in positive_float:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or float(value) <= 0:
            raise XGBoostConfigError(f"{context} must be numeric > 0")
    elif name in unit_interval_open_left:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not 0 < float(value) <= 1:
            raise XGBoostConfigError(f"{context} must be in (0, 1]")


def _candidate_id(sample_order: int, parameters: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_parameter_json(parameters).encode("utf-8")).hexdigest()[:10]
    return f"xgb_{int(sample_order):03d}_{digest}"


def build_xgboost_candidate_bank(engine: Mapping[str, Any]) -> Tuple[List[XGBoostCandidate], int]:
    if str(engine.get("engine", "")).strip().lower() != "xgboost":
        raise XGBoostConfigError("model_engine.engine must be xgboost")
    hyper = _mapping(engine, "hyperparameters", "model_engine")
    if not hyper:
        raise XGBoostConfigError("model_engine.hyperparameters must not be empty")
    illegal = sorted(RESERVED_HYPERPARAMETERS & set(map(str, hyper)))
    if illegal:
        raise XGBoostConfigError(f"Runtime-owned/unsupported hyperparameters: {illegal}")
    names = list(map(str, hyper.keys()))
    value_lists: List[List[Any]] = []
    for name in names:
        values = _scalar_or_nonempty_list(hyper[name], f"model_engine.hyperparameters.{name}")
        for value in values:
            _validate_common_parameter(name, value, f"model_engine.hyperparameters.{name}")
        value_lists.append(values)
    pool = [dict(zip(names, values)) for values in itertools.product(*value_lists)]
    if not pool:
        raise XGBoostConfigError("XGBoost candidate pool is empty")
    # Product of unique value lists should be unique, but explicitly guard accidental duplicates.
    canonical = [canonical_parameter_json(item) for item in pool]
    if len(canonical) != len(set(canonical)):
        raise XGBoostConfigError("XGBoost candidate pool contains duplicate parameter combinations")

    search = _mapping(engine, "search", "model_engine")
    strategy = str(search.get("strategy", "random")).strip().lower()
    if strategy not in {"random", "grid"}:
        raise XGBoostConfigError("model_engine.search.strategy must be random or grid")
    if strategy == "grid":
        selected_indices = np.arange(len(pool), dtype=int)
    else:
        n_iter = search.get("n_iter")
        if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
            raise XGBoostConfigError("model_engine.search.n_iter must be an integer >= 1")
        replacement = bool(search.get("replacement", False))
        if replacement:
            raise XGBoostConfigError("Batch 18 v1 requires random search replacement=false")
        if n_iter > len(pool):
            raise XGBoostConfigError(
                f"Random search n_iter={n_iter} exceeds candidate pool size={len(pool)}"
            )
        seed = search.get("random_state")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise XGBoostConfigError("model_engine.search.random_state must be an integer >= 0")
        rng = np.random.default_rng(int(seed))
        selected_indices = rng.choice(len(pool), size=int(n_iter), replace=False)

    candidates: List[XGBoostCandidate] = []
    for order, pool_index in enumerate(selected_indices.tolist(), start=1):
        params = pool[int(pool_index)]
        candidates.append(
            XGBoostCandidate(
                candidate_id=_candidate_id(order, params),
                pool_index=int(pool_index),
                sample_order=int(order),
                parameters=params,
            )
        )
    return candidates, len(pool)


def candidate_bank_frame(candidates: Sequence[XGBoostCandidate], *, pool_size: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        row = {
            "candidate_id": candidate.candidate_id,
            "candidate_sample_order": candidate.sample_order,
            "candidate_pool_index": candidate.pool_index,
            "candidate_pool_size": int(pool_size),
            "parameters_json": candidate.parameters_json,
        }
        row.update({f"xgb_{key}": value for key, value in sorted(candidate.parameters.items())})
        rows.append(row)
    return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate_bank(path: Path, *, expected_sha256: Optional[str] = None) -> List[XGBoostCandidate]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise XGBoostConfigError(f"Candidate bank not found: {source}")
    if expected_sha256 and sha256_file(source) != str(expected_sha256):
        raise XGBoostConfigError("Candidate bank SHA256 does not match resolved configuration")
    frame = pd.read_csv(source)
    required = {"candidate_id", "candidate_sample_order", "candidate_pool_index", "parameters_json"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise XGBoostConfigError(f"Candidate bank missing columns: {missing}")
    if frame.empty or frame["candidate_id"].astype(str).duplicated().any():
        raise XGBoostConfigError("Candidate bank is empty or contains duplicate candidate_id values")
    candidates: List[XGBoostCandidate] = []
    for row in frame.itertuples(index=False):
        try:
            parameters = json.loads(str(row.parameters_json))
        except Exception as exc:
            raise XGBoostConfigError(f"Invalid parameters_json for {row.candidate_id}") from exc
        candidates.append(
            XGBoostCandidate(
                candidate_id=str(row.candidate_id),
                sample_order=int(row.candidate_sample_order),
                pool_index=int(row.candidate_pool_index),
                parameters=parameters,
            )
        )
    return candidates


@dataclass(frozen=True)
class XGBoostExperimentConfig:
    raw: Mapping[str, Any]
    source_path: Path
    repository_root: Path

    @property
    def experiment_id(self) -> str:
        return str(self.raw["experiment"]["id"])

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, Mapping):
            raise XGBoostConfigError(f"Missing mapping section: {name}")
        return value

    def path(self, dotted: str, *, must_exist: bool = False, expect: Optional[str] = None) -> Path:
        value: Any = self.raw
        for key in dotted.split("."):
            if not isinstance(value, Mapping) or key not in value:
                raise XGBoostConfigError(f"Missing path setting: {dotted}")
            value = value[key]
        if not isinstance(value, str) or not value.strip():
            raise XGBoostConfigError(f"{dotted} must be a path string")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.repository_root / path
        path = path.resolve()
        if must_exist and not path.exists():
            raise XGBoostConfigError(f"Required path does not exist: {path}")
        if expect == "file" and path.exists() and not path.is_file():
            raise XGBoostConfigError(f"Expected file: {path}")
        if expect == "dir" and path.exists() and not path.is_dir():
            raise XGBoostConfigError(f"Expected directory: {path}")
        return path


def validate_xgboost_mapping(raw: Mapping[str, Any], *, require_candidate_bank: bool = False) -> None:
    if not isinstance(raw, Mapping):
        raise XGBoostConfigError("YAML root must be a mapping")
    if str(raw.get("schema_version")) != "1.0":
        raise XGBoostConfigError("schema_version must be '1.0'")
    experiment = _mapping(raw, "experiment", "root")
    if str(experiment.get("receptor")) != "TRB":
        raise XGBoostConfigError("experiment.receptor must be TRB")
    if not str(experiment.get("id", "")).strip():
        raise XGBoostConfigError("experiment.id is required")
    for section in (
        "data", "public_reference", "cross_validation", "models", "preprocessing",
        "modeling", "model_selection", "nested_cv", "outer_tasks", "model_engine",
    ):
        _mapping(raw, section, "root")
    models = raw["models"]
    if not models:
        raise XGBoostConfigError("models must not be empty")
    for name, spec in models.items():
        if not isinstance(spec, Mapping):
            raise XGBoostConfigError(f"models.{name} must be a mapping")
        count = 0
        for key in ("numeric", "categorical", "static_feature_groups", "dynamic_public"):
            value = spec.get(key, [])
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise XGBoostConfigError(f"models.{name}.{key} must be a list")
            count += len(value)
        if count == 0:
            raise XGBoostConfigError(f"models.{name} has no predictors")
    engine = raw["model_engine"]
    candidates, pool_size = build_xgboost_candidate_bank(engine)
    fixed = engine.get("fixed_parameters", {})
    if not isinstance(fixed, Mapping):
        raise XGBoostConfigError("model_engine.fixed_parameters must be a mapping")
    if set(fixed) & set(engine["hyperparameters"]):
        raise XGBoostConfigError("fixed_parameters and hyperparameters must not overlap")
    illegal = sorted(RESERVED_HYPERPARAMETERS & set(map(str, fixed)))
    if illegal:
        raise XGBoostConfigError(f"Runtime-owned/unsupported fixed_parameters: {illegal}")
    objective = str(fixed.get("objective", "binary:logistic"))
    if objective != "binary:logistic":
        raise XGBoostConfigError("Batch 18 requires objective='binary:logistic'")
    if bool(engine.get("early_stopping", False)):
        raise XGBoostConfigError("Batch 18 v1 requires model_engine.early_stopping=false")
    runtime = _mapping(engine, "runtime", "model_engine")
    n_jobs = runtime.get("n_jobs_per_fit", 1)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise XGBoostConfigError("model_engine.runtime.n_jobs_per_fit must be an integer >= 1")
    selection = raw["model_selection"]
    metric = str(selection.get("tuning_primary_metric", "roc_auc")).strip().lower()
    if metric not in SUPPORTED_TUNING_METRICS:
        raise XGBoostConfigError("XGBoost tuning_primary_metric must be roc_auc or log_loss")
    if str(selection.get("threshold_source", "inner_oof_youden")) != "inner_oof_youden":
        raise XGBoostConfigError("model_selection.threshold_source must be inner_oof_youden")
    expected = selection.get("expected_candidate_count")
    if expected is not None and int(expected) != len(candidates):
        raise XGBoostConfigError(f"expected_candidate_count={expected}, sampled={len(candidates)}")
    if require_candidate_bank:
        bank = _mapping(engine, "candidate_bank", "model_engine")
        if int(bank.get("candidate_count", -1)) != len(candidates):
            raise XGBoostConfigError("candidate_bank candidate_count mismatch")
        if int(bank.get("pool_size", -1)) != pool_size:
            raise XGBoostConfigError("candidate_bank pool_size mismatch")


def load_xgboost_config(path: Path, *, repository_root: Optional[Path] = None) -> XGBoostExperimentConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise XGBoostConfigError(f"Configuration not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    validate_xgboost_mapping(raw, require_candidate_bank=True)
    root = Path(repository_root).expanduser().resolve() if repository_root is not None else find_repository_root(source)
    config = XGBoostExperimentConfig(raw=raw, source_path=source, repository_root=root)
    engine = config.section("model_engine")
    bank = engine["candidate_bank"]
    loaded = load_candidate_bank(
        config.path("model_engine.candidate_bank.path", must_exist=True, expect="file"),
        expected_sha256=str(bank["sha256"]),
    )
    expected, expected_pool = build_xgboost_candidate_bank(engine)
    observed_signature = [(x.candidate_id, x.pool_index, x.parameters_json) for x in loaded]
    expected_signature = [(x.candidate_id, x.pool_index, x.parameters_json) for x in expected]
    if observed_signature != expected_signature or int(bank["pool_size"]) != int(expected_pool):
        raise XGBoostConfigError(
            "Frozen candidate bank does not match the search seed/hyperparameter definition"
        )
    return config


def candidates_from_config(config: XGBoostExperimentConfig) -> List[XGBoostCandidate]:
    bank = config.section("model_engine")["candidate_bank"]
    return load_candidate_bank(
        config.path("model_engine.candidate_bank.path", must_exist=True, expect="file"),
        expected_sha256=str(bank["sha256"]),
    )
