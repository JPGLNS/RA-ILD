#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a repeat-specific 3-mer feature-ablation experiment.

This preparer intentionally avoids changing the general Batch-02 scheme manager.
It starts from an already validated resolved repeated-holdout configuration,
replaces only the feature-ablation models, enables repeat-level 3-mer fitting,
recalculates all model/task counts, and writes an isolated experiment directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required: mamba install -c conda-forge pyyaml") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.config import validate_experiment_mapping  # noqa: E402
from ra_ild_trb.specifications import (  # noqa: E402
    candidate_selection_policy_name,
    normalize_tuning_primary_metric,
    tuning_sort_policy,
)


SCHEME_VERSION = "1.0"
SCHEME_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
ALLOWED_ENGINE_OVERRIDE_KEYS = {"alpha_grid", "lambda_grid"}


class FeatureAblationSchemeError(ValueError):
    """Raised when a feature-ablation scheme is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a TRB repeat-specific 3-mer feature-ablation scheme.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"YAML not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FeatureAblationSchemeError(f"YAML root must be a mapping: {path}")
    return value


def _string(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FeatureAblationSchemeError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _mapping(mapping: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise FeatureAblationSchemeError(f"{context}.{key} must be a mapping")
    return value


def _resolve(path_value: str, repository_root: Path) -> Path:
    candidate = Path(path_value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (repository_root / candidate).resolve()


def _repo_relative(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise FeatureAblationSchemeError(f"Path is outside repository root: {path}") from exc


def _set_nested(mapping: MutableMapping[str, Any], keys: Sequence[str], value: Any) -> None:
    current = mapping
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, MutableMapping):
            raise FeatureAblationSchemeError(f"Configuration key is not a mapping: {'.'.join(keys)}")
        current = child
    current[keys[-1]] = value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_count(group: str, repeat_options: Mapping[str, Any], legacy_count: int) -> int:
    core_name = str(repeat_options.get("core_group_name", "core83"))
    legacy_name = str(
        repeat_options.get("legacy_group_name", "static_tcr_candidate_predictors")
    )
    core_count = int(repeat_options.get("expected_core_feature_count", 83))
    if group == core_name:
        return core_count
    if group == legacy_name:
        return legacy_count
    match = re.fullmatch(r"repeat_3mer_(unweighted|weighted|both)_top(\d+)", group)
    if not match:
        raise FeatureAblationSchemeError(f"Unsupported feature group: {group}")
    representation, k_text = match.groups()
    k = int(k_text)
    top_values = {int(value) for value in repeat_options.get("top_k_values", [])}
    if k not in top_values:
        raise FeatureAblationSchemeError(
            f"Feature group {group} uses K={k}, absent from repeat_3mer_features.top_k_values"
        )
    return k if representation in {"unweighted", "weighted"} else 2 * k


def _resolved_model_counts(
    models: Mapping[str, Mapping[str, Any]],
    repeat_options: Mapping[str, Any],
    legacy_count: int,
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for model_name, spec in models.items():
        if not isinstance(spec, Mapping):
            raise FeatureAblationSchemeError(f"models.{model_name} must be a mapping")
        numeric = list(spec.get("numeric", []) or [])
        categorical = list(spec.get("categorical", []) or [])
        dynamic = list(spec.get("dynamic_public", []) or [])
        groups = list(spec.get("static_feature_groups", []) or [])
        numeric_count = len(numeric) + len(dynamic)
        seen_features = 0
        for group in groups:
            seen_features += _group_count(str(group), repeat_options, legacy_count)
        numeric_count += seen_features
        counts[str(model_name)] = {
            "numeric": int(numeric_count),
            "categorical": int(len(categorical)),
        }
    return counts


def _rewrite_output_paths(
    resolved: MutableMapping[str, Any], output_root: Path, repository_root: Path
) -> None:
    replacements = {
        ("outer_tasks", "output_root"): output_root / "02_tasks" / "outer_nested_cv",
        ("outer_tasks", "manifest"): output_root / "02_tasks" / "outer_nested_cv" / "07_outer_task_manifest.csv",
        ("outer_tasks", "status"): output_root / "02_tasks" / "outer_nested_cv" / "07_outer_task_status.csv",
        ("outer_tasks", "logs_dir"): output_root / "logs" / "outer_tasks",
        ("aggregation", "output_dir"): output_root / "03_summary",
        ("final_model", "bundle"): output_root / "04_final_model" / "final_model.joblib",
        ("final_model", "configuration"): output_root / "04_final_model" / "final_model_configuration.json",
        ("final_model", "reference_masks"): output_root / "04_final_model" / "final_public_reference_masks.npz",
        ("independent_validation", "output_dir"): output_root / "05_independent_validation",
    }
    for keys, path in replacements.items():
        _set_nested(resolved, keys, _repo_relative(path, repository_root))


def prepare(source_path: Path, repository_root: Path, *, overwrite: bool) -> Path:
    source = _load_yaml(source_path)
    if str(source.get("feature_ablation_scheme_version")) != SCHEME_VERSION:
        raise FeatureAblationSchemeError(
            f"feature_ablation_scheme_version must be {SCHEME_VERSION!r}"
        )
    scheme = _mapping(source, "scheme", "root")
    scheme_id = _string(scheme, "id", "scheme")
    if not SCHEME_ID_PATTERN.fullmatch(scheme_id):
        raise FeatureAblationSchemeError("scheme.id has invalid characters")
    description = _string(scheme, "description", "scheme")
    base_path = _resolve(_string(scheme, "base_resolved_config", "scheme"), repository_root)
    output_root = _resolve(_string(scheme, "output_root", "scheme"), repository_root)
    allowed_root = (repository_root / "TRB/set/experiments").resolve()
    try:
        output_root.relative_to(allowed_root)
    except ValueError as exc:
        raise FeatureAblationSchemeError(
            f"scheme.output_root must be under {allowed_root}"
        ) from exc
    if output_root.name != scheme_id:
        raise FeatureAblationSchemeError("scheme.output_root must end with scheme.id")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Experiment output already exists and is not empty: {output_root}"
        )

    resolved = copy.deepcopy(dict(_load_yaml(base_path)))
    models = _mapping(source, "models", "root")
    repeat_options = copy.deepcopy(dict(_mapping(source, "repeat_3mer_features", "root")))

    engine_override_raw = source.get("model_engine")
    engine_override: Mapping[str, Any] | None
    if engine_override_raw is None:
        engine_override = None
    elif not isinstance(engine_override_raw, Mapping):
        raise FeatureAblationSchemeError("root.model_engine must be a mapping")
    else:
        engine_override = engine_override_raw
        unknown_engine_keys = sorted(
            set(engine_override) - ALLOWED_ENGINE_OVERRIDE_KEYS
        )
        if unknown_engine_keys:
            raise FeatureAblationSchemeError(
                "root.model_engine only supports alpha_grid/lambda_grid overrides; "
                f"unknown keys: {unknown_engine_keys}"
            )
        if not engine_override:
            raise FeatureAblationSchemeError(
                "root.model_engine must contain alpha_grid and/or lambda_grid"
            )
    repeat_options["enabled"] = True
    repeat_options.setdefault("selection_scope", "outer_repeat_train_only")
    repeat_options.setdefault("ranking_rule", "prevalence_then_total_variance")
    repeat_options.setdefault("core_group_name", "core83")
    repeat_options.setdefault("legacy_group_name", "static_tcr_candidate_predictors")
    repeat_options.setdefault("expected_core_feature_count", 83)
    repeat_options.setdefault("top_k_values", [50, 100, 200, 500])
    repeat_options.setdefault("max_kmers", 500)

    resolved["experiment"]["id"] = scheme_id
    resolved["experiment"]["description"] = description
    resolved["experiment"]["status"] = "development"
    resolved["models"] = copy.deepcopy(dict(models))
    resolved["repeat_3mer_features"] = repeat_options
    if engine_override is not None:
        resolved_engine = copy.deepcopy(
            dict(_mapping(resolved, "model_engine", "base_resolved_config"))
        )
        for key in ALLOWED_ENGINE_OVERRIDE_KEYS:
            if key in engine_override:
                resolved_engine[key] = copy.deepcopy(engine_override[key])
        resolved["model_engine"] = resolved_engine

    source_selection = source.get("model_selection", {})
    if source_selection is None:
        source_selection = {}
    if not isinstance(source_selection, Mapping):
        raise FeatureAblationSchemeError("root.model_selection must be a mapping")

    tuning_primary_metric = normalize_tuning_primary_metric(
        source_selection.get("tuning_primary_metric", "roc_auc")
    )
    sort_columns, sort_ascending = tuning_sort_policy(tuning_primary_metric)

    resolved_selection = resolved.get("model_selection")
    resolved_nested = resolved.get("nested_cv")
    if not isinstance(resolved_selection, MutableMapping):
        raise FeatureAblationSchemeError(
            "base_resolved_config.model_selection must be a mapping"
        )
    if not isinstance(resolved_nested, MutableMapping):
        raise FeatureAblationSchemeError(
            "base_resolved_config.nested_cv must be a mapping"
        )

    resolved_selection["tuning_primary_metric"] = tuning_primary_metric
    resolved_selection["candidate_sort"] = [
        {"field": field, "ascending": bool(ascending)}
        for field, ascending in zip(sort_columns, sort_ascending)
    ]
    resolved_nested["candidate_selection_policy"] = (
        candidate_selection_policy_name(tuning_primary_metric)
    )

    _rewrite_output_paths(resolved, output_root, repository_root)

    legacy_count = int(resolved["model_selection"].get("expected_static_feature_count", 1083))
    resolved["model_selection"]["expected_resolved_columns"] = _resolved_model_counts(
        models,
        repeat_options,
        legacy_count,
    )

    cv = resolved["cross_validation"]
    engine = resolved["model_engine"]
    executed_folds = list(cv.get("executed_outer_folds", []))
    outer_tasks = int(cv["outer_repeats"]) * len(executed_folds)
    candidate_count = len(engine["alpha_grid"]) * len(engine["lambda_grid"])
    model_count = len(models)
    resolved["nested_cv"]["expected_outer_tasks"] = outer_tasks
    resolved["nested_cv"]["expected_inner_fits_per_task"] = (
        model_count * int(cv["inner_folds"]) * candidate_count
    )
    resolved["outer_tasks"]["expected_tasks"] = outer_tasks
    resolved["aggregation"]["expected_metric_rows"] = outer_tasks * model_count
    repeated = resolved.get("repeated_holdout_training")
    if isinstance(repeated, Mapping):
        resolved["aggregation"]["expected_prediction_rows"] = (
            int(repeated["holdout_size"]) * int(repeated["split_count"]) * model_count
        )
    resolved["aggregation"]["expected_predictions_per_sample"] = int(cv["outer_repeats"])

    final = resolved["final_model"]
    final["status"] = "not_selected"
    for key in (
        "selected_model",
        "tuning_mode",
        "candidate_pairs",
        "selection_rule",
        "selected_alpha",
        "selected_lambda",
        "locked_threshold",
    ):
        final.pop(key, None)

    # Fail before writing if the resolved configuration no longer satisfies the
    # existing V2 validation contract.
    validate_experiment_mapping(resolved)

    config_dir = output_root / "00_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = config_dir / "resolved_config.yaml"
    source_copy = config_dir / "source_feature_ablation_scheme.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    shutil.copy2(source_path, source_copy)

    inputs = []
    for label, path in (
        ("base_resolved_config", base_path),
        ("source_feature_ablation_scheme", source_path),
        ("repeat_3mer_aa_manifest", _resolve(str(repeat_options["aa_manifest"]), repository_root)),
    ):
        inputs.append(
            {
                "label": label,
                "path": _repo_relative(path, repository_root),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    marker = {
        "status": "PREPARED",
        "scheme_id": scheme_id,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_config": _repo_relative(resolved_path, repository_root),
        "model_count": model_count,
        "models": list(models),
        "top_k_values": repeat_options["top_k_values"],
        "selection_scope": repeat_options["selection_scope"],
        "alpha_grid": list(engine["alpha_grid"]),
        "lambda_grid": list(engine["lambda_grid"]),
        "tuning_primary_metric": tuning_primary_metric,
        "candidate_selection_policy": resolved["nested_cv"][
            "candidate_selection_policy"
        ],
        "candidate_count": int(candidate_count),
        "expected_inner_fits_per_task": int(
            resolved["nested_cv"]["expected_inner_fits_per_task"]
        ),
        "input_files": inputs,
    }
    (config_dir / "FEATURE_ABLATION_SCHEME_PREPARED.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved_path


def main() -> int:
    args = parse_args()
    try:
        root = (
            Path(args.repository_root).expanduser().resolve()
            if args.repository_root
            else Path.cwd().resolve()
        )
        source = Path(args.scheme).expanduser()
        if not source.is_absolute():
            source = (root / source).resolve()
        resolved = prepare(source, root, overwrite=args.overwrite)
        print("TRB feature-ablation scheme preparation: COMPLETE")
        print(f"Resolved config: {resolved}")
        return 0
    except Exception as exc:
        print(f"TRB feature-ablation scheme preparation: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
