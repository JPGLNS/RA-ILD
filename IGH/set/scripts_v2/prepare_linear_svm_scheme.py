#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare an isolated IGH Linear SVM scheme from any completed IGH resolved config."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.linear_svm_config import (  # noqa: E402
    expand_linear_svm_candidates,
    find_repository_root,
    validate_linear_svm_mapping,
)


SCHEME_VERSION = "1.0"


class LinearSVMSchemeError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a IGH Linear SVM analysis scheme.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(value: object, root: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise LinearSVMSchemeError(f"{context} must be a path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _set_path(mapping: MutableMapping[str, Any], keys, value: str) -> None:
    current: MutableMapping[str, Any] = mapping
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, MutableMapping):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def prepare(
    source_path: Path,
    repository_root: Path,
    *,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, Mapping):
        raise LinearSVMSchemeError("Scheme YAML root must be a mapping")
    if str(source.get("linear_svm_scheme_version")) != SCHEME_VERSION:
        raise LinearSVMSchemeError(
            f"linear_svm_scheme_version must be {SCHEME_VERSION!r}"
        )
    scheme = source.get("scheme")
    if not isinstance(scheme, Mapping):
        raise LinearSVMSchemeError("scheme must be a mapping")
    scheme_id = str(scheme.get("id", "")).strip()
    description = str(scheme.get("description", "")).strip()
    if not scheme_id or not description:
        raise LinearSVMSchemeError("scheme.id and scheme.description are required")
    base_path = resolve_path(
        scheme.get("base_resolved_config"), repository_root, "scheme.base_resolved_config"
    )
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    output_root = resolve_path(
        scheme.get("output_root"), repository_root, "scheme.output_root"
    )
    resolved = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(resolved, MutableMapping):
        raise LinearSVMSchemeError("Base resolved config must be a mapping")
    resolved = copy.deepcopy(resolved)

    experiment = resolved.get("experiment")
    if not isinstance(experiment, MutableMapping):
        raise LinearSVMSchemeError("Base config experiment must be a mapping")
    if str(experiment.get("receptor")) != "IGH":
        raise LinearSVMSchemeError("Base config receptor must be IGH")
    experiment["id"] = scheme_id
    experiment["description"] = description
    experiment["status"] = "linear_svm_scheme"
    experiment["model_engine"] = "linear_svc"

    if "models" in source:
        if not isinstance(source["models"], Mapping) or not source["models"]:
            raise LinearSVMSchemeError("models override must be a non-empty mapping")
        resolved["models"] = copy.deepcopy(source["models"])

    if "repeat_3mer_features" in source:
        value = source["repeat_3mer_features"]
        if value is None:
            resolved.pop("repeat_3mer_features", None)
        elif not isinstance(value, Mapping):
            raise LinearSVMSchemeError("repeat_3mer_features must be a mapping or null")
        else:
            resolved["repeat_3mer_features"] = copy.deepcopy(value)

    engine = source.get("model_engine")
    if not isinstance(engine, Mapping):
        raise LinearSVMSchemeError("model_engine must be a mapping")
    resolved_engine = copy.deepcopy(engine)
    resolved_engine["engine"] = "linear_svc"
    resolved_engine.setdefault("max_iter", 10000)
    resolved_engine.setdefault("final_max_iter", 100000)
    resolved_engine.setdefault("tolerance", 1.0e-4)
    resolved_engine.setdefault("zero_sd_tolerance", 1.0e-12)
    resolved_engine.setdefault("fit_intercept", True)
    resolved_engine.setdefault("intercept_scaling", 1.0)
    resolved["model_engine"] = resolved_engine
    candidates = expand_linear_svm_candidates(resolved_engine)

    selection = resolved.get("model_selection")
    if not isinstance(selection, MutableMapping):
        raise LinearSVMSchemeError("Base config model_selection must be a mapping")
    # A models override invalidates model-specific column-count audits
    # inherited from the base resolved configuration.
    if "models" in source:
        selection.pop("expected_resolved_columns", None)

    selection["tuning_primary_metric"] = "roc_auc"
    selection["candidate_selection_policy"] = "pooled_roc_ap_C_priority"
    selection["candidate_sort"] = [
        {"field": "pooled_inner_roc_auc", "ascending": False},
        {"field": "pooled_inner_average_precision", "ascending": False},
        {"field": "C", "ascending": True},
        {"field": "candidate_priority", "ascending": True},
        {"field": "candidate_id", "ascending": True},
    ]
    selection["threshold_source"] = "inner_oof_youden"
    selection["threshold_tie_breaker"] = "closest_to_zero_then_larger"
    selection["natural_threshold"] = 0.0
    selection["expected_candidate_count"] = len(candidates)

    modeling = resolved.get("modeling")
    if not isinstance(modeling, MutableMapping):
        raise LinearSVMSchemeError("Base config modeling must be a mapping")
    modeling["score_type"] = "decision_function"
    modeling["probability_metrics_available"] = False

    cv = resolved.get("cross_validation")
    nested = resolved.get("nested_cv")
    outer_tasks = resolved.get("outer_tasks")
    if not isinstance(cv, Mapping) or not isinstance(nested, MutableMapping):
        raise LinearSVMSchemeError("Base config CV sections are invalid")
    if not isinstance(outer_tasks, MutableMapping):
        raise LinearSVMSchemeError("Base config outer_tasks must be a mapping")
    executed_folds = cv.get("executed_outer_folds")
    if executed_folds is None:
        executed_folds = list(range(1, int(cv["outer_folds"]) + 1))
    expected_tasks = int(cv["outer_repeats"]) * len(executed_folds)
    expected_inner_fits = (
        len(resolved["models"])
        * len(candidates)
        * int(cv["inner_folds"])
    )
    nested["candidate_selection_policy"] = "pooled_roc_ap_C_priority"
    nested["threshold_selection"] = "youden_closest_to_zero"
    nested["expected_inner_fits_per_task"] = expected_inner_fits

    output_relative = _relative_or_absolute(output_root, repository_root)
    outer_tasks["output_root"] = f"{output_relative}/01_outer_tasks"
    outer_tasks["manifest"] = f"{output_relative}/00_config/outer_task_manifest.csv"
    outer_tasks["status"] = f"{output_relative}/00_config/outer_task_status.csv"
    outer_tasks["logs_dir"] = f"{output_relative}/02_logs"
    outer_tasks["runner_script"] = (
        "IGH/set/scripts_v2/run_linear_svm_nested_cv_task.py"
    )
    outer_tasks["expected_tasks"] = expected_tasks

    regression_task = nested.get("regression_task")
    if not isinstance(regression_task, MutableMapping):
        regression_task = {}
        nested["regression_task"] = regression_task
    regression_task["outer_repeat"] = 1
    regression_task["outer_fold"] = int(executed_folds[0])
    regression_task["task_dir"] = (
        f"{output_relative}/01_outer_tasks/repeat_01_fold_{int(executed_folds[0]):02d}"
    )

    resolved["linear_svm_scheme"] = {
        "version": SCHEME_VERSION,
        "source_scheme": _relative_or_absolute(source_path, repository_root),
        "base_resolved_config": _relative_or_absolute(base_path, repository_root),
        "candidate_count": len(candidates),
        "probability_metrics_available": False,
        "unsupported_metrics": ["log_loss", "brier_score"],
    }
    validate_linear_svm_mapping(resolved)

    resolved_path = output_root / "00_config" / "resolved_config.yaml"
    marker_path = output_root / "00_config" / "LINEAR_SVM_SCHEME_PREPARED.json"
    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Output root already exists: {output_root}; use a new scheme ID or --overwrite"
        )
    if dry_run:
        print("RA-ILD IGH Linear SVM scheme preparation")
        print(f"Scheme ID:       {scheme_id}")
        print(f"Base config:     {base_path}")
        print(f"Output root:     {output_root}")
        print(f"Models:          {len(resolved['models'])}")
        print(f"Candidates:      {len(candidates)}")
        print(f"Tasks:           {expected_tasks}")
        print(f"Inner fits/task: {expected_inner_fits}")
        print("Dry run only; no files were written.")
        return resolved_path

    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    marker = {
        "status": "PREPARED",
        "scheme_id": scheme_id,
        "engine": "linear_svc",
        "source_scheme": str(source_path),
        "base_resolved_config": str(base_path),
        "resolved_config": str(resolved_path),
        "source_scheme_sha256": sha256(source_path),
        "base_resolved_config_sha256": sha256(base_path),
        "resolved_config_sha256": sha256(resolved_path),
        "model_count": len(resolved["models"]),
        "candidate_count": len(candidates),
        "expected_tasks": expected_tasks,
        "expected_inner_fits_per_task": expected_inner_fits,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("RA-ILD IGH Linear SVM scheme preparation")
    print(f"Scheme ID:       {scheme_id}")
    print(f"Base config:     {base_path}")
    print(f"Output root:     {output_root}")
    print(f"Resolved config: {resolved_path}")
    print(f"Models:          {len(resolved['models'])}")
    print(f"Candidates:      {len(candidates)}")
    print(f"Tasks:           {expected_tasks}")
    print(f"Inner fits/task: {expected_inner_fits}")
    print("Linear SVM scheme preparation: COMPLETE")
    return resolved_path


def main() -> int:
    args = parse_args()
    try:
        source_path = Path(args.scheme).expanduser().resolve()
        root = (
            Path(args.repository_root).expanduser().resolve()
            if args.repository_root
            else find_repository_root(source_path)
        )
        prepare(
            source_path,
            root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        return 0
    except Exception as exc:
        print(f"Linear SVM scheme preparation: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
