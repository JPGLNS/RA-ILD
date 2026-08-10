#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare an isolated Batch 15 IGH XGBoost scheme and freeze its candidate bank."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
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

from ra_ild_igh.xgboost_config import (  # noqa: E402
    build_xgboost_candidate_bank,
    candidate_bank_frame,
    find_repository_root,
    validate_xgboost_mapping,
)

SCHEME_VERSION = "1.0"


class XGBoostSchemeError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a IGH Batch 15 XGBoost scheme.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(value: object, root: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise XGBoostSchemeError(f"{context} must be a path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _serialize_csv(frame) -> bytes:
    stream = io.StringIO()
    frame.to_csv(stream, index=False, lineterminator="\n")
    return stream.getvalue().encode("utf-8")


def prepare(source_path: Path, repository_root: Path, *, overwrite: bool, dry_run: bool) -> Path:
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, Mapping):
        raise XGBoostSchemeError("Scheme YAML root must be a mapping")
    if str(source.get("xgboost_scheme_version")) != SCHEME_VERSION:
        raise XGBoostSchemeError(f"xgboost_scheme_version must be {SCHEME_VERSION!r}")
    scheme = source.get("scheme")
    if not isinstance(scheme, Mapping):
        raise XGBoostSchemeError("scheme must be a mapping")
    scheme_id = str(scheme.get("id", "")).strip()
    description = str(scheme.get("description", "")).strip()
    if not scheme_id or not description:
        raise XGBoostSchemeError("scheme.id and scheme.description are required")
    base_path = resolve_path(scheme.get("base_resolved_config"), repository_root, "scheme.base_resolved_config")
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    output_root = resolve_path(scheme.get("output_root"), repository_root, "scheme.output_root")
    resolved = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(resolved, MutableMapping):
        raise XGBoostSchemeError("Base resolved config must be a mapping")
    resolved = copy.deepcopy(resolved)

    experiment = resolved.get("experiment")
    if not isinstance(experiment, MutableMapping) or str(experiment.get("receptor")) != "IGH":
        raise XGBoostSchemeError("Base config must contain experiment.receptor=IGH")
    experiment["id"] = scheme_id
    experiment["description"] = description
    experiment["status"] = "xgboost_scheme"
    experiment["model_engine"] = "xgboost"

    if "models" in source:
        if not isinstance(source["models"], Mapping) or not source["models"]:
            raise XGBoostSchemeError("models override must be a non-empty mapping")
        resolved["models"] = copy.deepcopy(source["models"])
    if "repeat_3mer_features" in source:
        value = source["repeat_3mer_features"]
        if value is None:
            resolved.pop("repeat_3mer_features", None)
        elif not isinstance(value, Mapping):
            raise XGBoostSchemeError("repeat_3mer_features must be a mapping or null")
        else:
            resolved["repeat_3mer_features"] = copy.deepcopy(value)

    cv = resolved.get("cross_validation")
    if not isinstance(cv, MutableMapping):
        raise XGBoostSchemeError("Base config cross_validation must be a mapping")
    if "outer_repeats" in source:
        outer_repeats = source["outer_repeats"]
        if isinstance(outer_repeats, bool) or not isinstance(outer_repeats, int) or outer_repeats < 1:
            raise XGBoostSchemeError("outer_repeats override must be an integer >= 1")
        inherited = int(cv.get("outer_repeats", 0))
        if inherited and int(outer_repeats) > inherited:
            raise XGBoostSchemeError(
                f"outer_repeats override={outer_repeats} exceeds inherited frozen assignments={inherited}"
            )
        cv["outer_repeats"] = int(outer_repeats)

    engine = source.get("model_engine")
    if not isinstance(engine, Mapping):
        raise XGBoostSchemeError("model_engine must be a mapping")
    resolved_engine = copy.deepcopy(engine)
    resolved_engine["engine"] = "xgboost"
    resolved_engine.setdefault("fixed_parameters", {})
    resolved_engine["fixed_parameters"].setdefault("objective", "binary:logistic")
    resolved_engine["fixed_parameters"].setdefault("eval_metric", "logloss")
    resolved_engine["fixed_parameters"].setdefault("tree_method", "hist")
    resolved_engine["fixed_parameters"].setdefault("verbosity", 0)
    resolved_engine.setdefault("early_stopping", False)
    resolved_engine.setdefault("runtime", {})
    resolved_engine["runtime"].setdefault("n_jobs_per_fit", 1)
    candidates, pool_size = build_xgboost_candidate_bank(resolved_engine)
    bank_frame = candidate_bank_frame(candidates, pool_size=pool_size)
    bank_bytes = _serialize_csv(bank_frame)
    bank_path = output_root / "00_config" / "xgboost_candidate_bank.csv"
    resolved_engine["candidate_bank"] = {
        "path": relative_or_absolute(bank_path, repository_root),
        "sha256": sha256_bytes(bank_bytes),
        "candidate_count": len(candidates),
        "pool_size": pool_size,
        "frozen": True,
    }
    resolved["model_engine"] = resolved_engine

    selection = resolved.get("model_selection")
    if not isinstance(selection, MutableMapping):
        raise XGBoostSchemeError("Base config model_selection must be a mapping")
    if "models" in source:
        selection.pop("expected_resolved_columns", None)
    requested_metric = str(source.get("tuning_primary_metric", selection.get("tuning_primary_metric", "roc_auc"))).strip().lower()
    if requested_metric not in {"roc_auc", "log_loss"}:
        raise XGBoostSchemeError("tuning_primary_metric must be roc_auc or log_loss")
    selection["tuning_primary_metric"] = requested_metric
    if requested_metric == "roc_auc":
        policy = "pooled_roc_pr_logloss_candidate"
        selection["candidate_sort"] = [
            {"field": "pooled_inner_roc_auc", "ascending": False},
            {"field": "pooled_inner_pr_auc", "ascending": False},
            {"field": "pooled_inner_log_loss", "ascending": True},
            {"field": "candidate_sample_order", "ascending": True},
            {"field": "candidate_id", "ascending": True},
        ]
    else:
        policy = "pooled_logloss_roc_pr_candidate"
        selection["candidate_sort"] = [
            {"field": "pooled_inner_log_loss", "ascending": True},
            {"field": "pooled_inner_roc_auc", "ascending": False},
            {"field": "pooled_inner_pr_auc", "ascending": False},
            {"field": "candidate_sample_order", "ascending": True},
            {"field": "candidate_id", "ascending": True},
        ]
    selection["candidate_selection_policy"] = policy
    selection["threshold_source"] = "inner_oof_youden"
    selection["threshold_tie_breaker"] = "closest_to_0.5"
    selection["expected_candidate_count"] = len(candidates)

    modeling = resolved.get("modeling")
    if not isinstance(modeling, MutableMapping):
        raise XGBoostSchemeError("Base config modeling must be a mapping")
    modeling["score_type"] = "probability"
    modeling["probability_metrics_available"] = True

    nested = resolved.get("nested_cv")
    outer_tasks = resolved.get("outer_tasks")
    if not isinstance(nested, MutableMapping) or not isinstance(outer_tasks, MutableMapping):
        raise XGBoostSchemeError("Base config nested_cv/outer_tasks sections are invalid")
    executed_folds = cv.get("executed_outer_folds")
    if executed_folds is None:
        executed_folds = list(range(1, int(cv["outer_folds"]) + 1))
    expected_tasks = int(cv["outer_repeats"]) * len(executed_folds)
    expected_inner_fits = len(resolved["models"]) * len(candidates) * int(cv["inner_folds"])
    nested["candidate_selection_policy"] = policy
    nested["threshold_selection"] = "youden_closest_to_0.5"
    nested["expected_inner_fits_per_task"] = expected_inner_fits

    output_relative = relative_or_absolute(output_root, repository_root)
    outer_tasks["output_root"] = f"{output_relative}/01_outer_tasks"
    outer_tasks["manifest"] = f"{output_relative}/00_config/outer_task_manifest.csv"
    outer_tasks["status"] = f"{output_relative}/00_config/outer_task_status.csv"
    outer_tasks["logs_dir"] = f"{output_relative}/02_logs"
    outer_tasks["runner_script"] = "IGH/set/scripts_v2/run_xgboost_nested_cv_task.py"
    outer_tasks["expected_tasks"] = expected_tasks
    regression = nested.get("regression_task")
    if not isinstance(regression, MutableMapping):
        regression = {}
        nested["regression_task"] = regression
    regression["outer_repeat"] = 1
    regression["outer_fold"] = int(executed_folds[0])
    regression["task_dir"] = f"{output_relative}/01_outer_tasks/repeat_01_fold_{int(executed_folds[0]):02d}"

    resolved["xgboost_scheme"] = {
        "version": SCHEME_VERSION,
        "source_scheme": relative_or_absolute(source_path, repository_root),
        "base_resolved_config": relative_or_absolute(base_path, repository_root),
        "candidate_count": len(candidates),
        "candidate_pool_size": pool_size,
        "candidate_bank": relative_or_absolute(bank_path, repository_root),
        "candidate_bank_sha256": sha256_bytes(bank_bytes),
        "probability_metrics_available": True,
        "early_stopping_enabled": False,
    }
    validate_xgboost_mapping(resolved, require_candidate_bank=True)

    resolved_path = output_root / "00_config" / "resolved_config.yaml"
    marker_path = output_root / "00_config" / "XGBOOST_SCHEME_PREPARED.json"
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"Output root already exists: {output_root}; use a new scheme ID or --overwrite")
    if dry_run:
        print("RA-ILD IGH XGBoost scheme preparation")
        print(f"Scheme ID:       {scheme_id}")
        print(f"Base config:     {base_path}")
        print(f"Output root:     {output_root}")
        print(f"Models:          {len(resolved['models'])}")
        print(f"Candidate pool:  {pool_size}")
        print(f"Candidates:      {len(candidates)}")
        print(f"Tasks:           {expected_tasks}")
        print(f"Inner fits/task: {expected_inner_fits}")
        print(f"Tuning metric:   {requested_metric}")
        print(f"n_jobs/fit:      {resolved_engine['runtime']['n_jobs_per_fit']}")
        print("Dry run only; no files were written.")
        return resolved_path

    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_bytes(bank_bytes)
    resolved_path.write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")
    marker = {
        "status": "PREPARED", "scheme_id": scheme_id, "engine": "xgboost",
        "source_scheme": str(source_path), "base_resolved_config": str(base_path),
        "resolved_config": str(resolved_path), "candidate_bank": str(bank_path),
        "source_scheme_sha256": sha256(source_path), "base_resolved_config_sha256": sha256(base_path),
        "resolved_config_sha256": sha256(resolved_path), "candidate_bank_sha256": sha256(bank_path),
        "model_count": len(resolved["models"]), "candidate_pool_size": pool_size,
        "candidate_count": len(candidates), "expected_tasks": expected_tasks,
        "expected_inner_fits_per_task": expected_inner_fits,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("RA-ILD IGH XGBoost scheme preparation")
    print(f"Scheme ID:       {scheme_id}")
    print(f"Resolved config: {resolved_path}")
    print(f"Candidate bank:  {bank_path}")
    print(f"Candidate pool:  {pool_size}")
    print(f"Candidates:      {len(candidates)}")
    print(f"Tasks:           {expected_tasks}")
    print(f"Inner fits/task: {expected_inner_fits}")
    print("XGBoost scheme preparation: COMPLETE")
    return resolved_path


def main() -> int:
    args = parse_args()
    try:
        source_path = Path(args.scheme).expanduser().resolve()
        root = Path(args.repository_root).expanduser().resolve() if args.repository_root else find_repository_root(source_path)
        prepare(source_path, root, overwrite=args.overwrite, dry_run=args.dry_run)
        return 0
    except Exception as exc:
        print(f"XGBoost scheme preparation: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
