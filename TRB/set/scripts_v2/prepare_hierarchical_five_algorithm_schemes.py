#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the 15 five-algorithm schemes for the frozen TRB V3 179-sample cohorts.

The three existing Elastic Net schemes and their results are read-only inputs.
Ridge/Lasso are prepared as alpha=0/1 endpoints of the validated Elastic Net
backend. Linear SVM and XGBoost inherit the exact cohort model definitions,
frozen assignments, training bundles, public-reference cache, and repeat-3mer
settings from the three existing resolved configurations.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.model_contract import validate_hierarchical_model_contract


ROOT = Path(__file__).resolve().parents[3]
BASES = {
    "total": {
        "experiment": "trb_v3_179_total_m0_m3_repeat100_top10000_r3_v1",
        "samples": 179,
        "models": (
            "M0_clinical",
            "M1_tcr",
            "M2_tcr_clinical",
            "M3_tcr_clinical_material",
        ),
    },
    "pbmc": {
        "experiment": "trb_v3_179_pbmc_m0_m2_repeat100_top10000_r3_v1",
        "samples": 116,
        "models": ("M0_clinical", "M1_tcr", "M2_tcr_clinical"),
    },
    "buffycoat": {
        "experiment": "trb_v3_179_buffycoat_m0_m2_repeat100_top10000_r3_v1",
        "samples": 63,
        "models": ("M0_clinical", "M1_tcr", "M2_tcr_clinical"),
    },
}
SOURCE_SCHEME_DIR = Path("TRB/set/configs/schemes/generated/hierarchical_repeat3mer")
OUTPUT_SCHEME_DIR = Path("TRB/set/configs/schemes/generated/hierarchical_five_algorithm")
SVM_CONFIG_DIR = Path("TRB/set/configs/linear_svm/hierarchical_repeat3mer")
XGB_CONFIG_DIR = Path("TRB/set/configs/xgboost/hierarchical_repeat3mer")
MANIFEST_PATH = OUTPUT_SCHEME_DIR / "FIVE_ALGORITHM_RUN_MANIFEST.json"

SVM_C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
XGB_HYPERPARAMETERS = {
    "max_depth": [1, 2, 3],
    "min_child_weight": [3, 5, 10],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.70, 0.85],
    "colsample_bytree": [0.30, 0.50, 0.70, 1.00],
    "reg_alpha": [0.1, 0.5, 1.0],
    "reg_lambda": [3.0, 5.0, 10.0],
    "gamma": 0.0,
    "scale_pos_weight": 1.0,
    "n_estimators": [200, 300, 500],
}
XGB_CANDIDATES = 100


class PreparationError(RuntimeError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PreparationError(f"YAML root must be a mapping: {path}")
    return copy.deepcopy(dict(value))


def _dump_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _config_path(root: Path, experiment_id: str) -> Path:
    return root / "TRB/set/experiments" / experiment_id / "00_config/resolved_config.yaml"


def _validate_base(root: Path, cohort: str, spec: Mapping[str, Any]) -> Path:
    path = _config_path(root, str(spec["experiment"]))
    raw = _load_yaml(path)
    observed_samples = int(raw["data"]["train"]["expected_samples"])
    if observed_samples != int(spec["samples"]):
        raise PreparationError(
            f"{cohort}: expected {spec['samples']} samples in resolved config, "
            f"observed {observed_samples}"
        )
    observed_models = tuple(raw.get("models", {}).keys())
    if observed_models != tuple(spec["models"]):
        raise PreparationError(
            f"{cohort}: model contract mismatch; observed={observed_models}, "
            f"expected={tuple(spec['models'])}"
        )
    cv = raw["cross_validation"]
    if int(cv["outer_repeats"]) != 100:
        raise PreparationError(f"{cohort}: expected 100 outer repeats")
    executed = tuple(int(x) for x in cv.get("executed_outer_folds", [1]))
    if executed != (1,):
        raise PreparationError(
            f"{cohort}: frozen repeated-holdout contract requires executed_outer_folds=[1], "
            f"observed={executed}"
        )
    repeat3mer = raw.get("repeat_3mer_features", {})
    if not isinstance(repeat3mer, Mapping) or repeat3mer.get("enabled") is not True:
        raise PreparationError(f"{cohort}: repeat-specific 3-mer features are not enabled")
    try:
        predictor_counts = validate_hierarchical_model_contract(
            raw["models"],
            cohort=cohort,
            repeat_3mer_options=repeat3mer,
            expected_resolved_columns=raw.get("model_selection", {}).get(
                "expected_resolved_columns"
            ),
        )
    except Exception as exc:
        raise PreparationError(
            f"{cohort}: frozen M0-M3/3-mer model contract failed: {exc}"
        ) from exc
    print(f"[MODEL CONTRACT PASS] {cohort}: {predictor_counts}")
    if int(raw["outer_tasks"]["expected_tasks"]) != 100:
        raise PreparationError(f"{cohort}: expected exactly 100 outer tasks")
    return path


def _new_id(base_id: str, algorithm: str) -> str:
    suffix = {
        "ridge": "ridge_v1",
        "lasso": "lasso_v1",
        "linear_svm": "linear_svm_v1",
        "xgboost": "xgboost_v1",
    }[algorithm]
    return f"{base_id}_{suffix}"


def _build_ridge_lasso(
    root: Path, cohort: str, base_id: str, algorithm: str, alpha: float
) -> tuple[Path, Path]:
    source_path = root / SOURCE_SCHEME_DIR / f"{base_id}.yaml"
    source = _load_yaml(source_path)
    if str(source.get("scheme_version")) != "1.0":
        raise PreparationError(
            f"{cohort}: generated hierarchical source must have scheme_version='1.0': "
            f"{source_path}"
        )
    scheme = source.get("scheme")
    if not isinstance(scheme, Mapping):
        raise PreparationError(f"{source_path}: scheme section is missing")
    result = copy.deepcopy(source)
    scheme_out = result["scheme"]
    scheme_id = _new_id(base_id, algorithm)
    scheme_out["id"] = scheme_id
    scheme_out["description"] = (
        f"{cohort.upper()} frozen 100-repeat hierarchical TRB comparison using "
        f"{algorithm.upper()} (Elastic Net logistic endpoint alpha={alpha}). "
        "All cohort inputs, models, splits, public-reference cache, and "
        "repeat-specific 3-mer settings are inherited unchanged."
    )
    scheme_out["output_root"] = f"TRB/set/experiments/{scheme_id}"
    overrides = result.setdefault("overrides", {})
    if not isinstance(overrides, dict):
        raise PreparationError(f"{source_path}: overrides must be a mapping")
    engine = overrides.setdefault("model_engine", {})
    if not isinstance(engine, dict):
        raise PreparationError(f"{source_path}: overrides.model_engine must be a mapping")
    engine["alpha_grid"] = [float(alpha)]
    output_path = root / OUTPUT_SCHEME_DIR / f"{scheme_id}.yaml"
    return output_path, result


def _build_svm(cohort: str, base_id: str) -> tuple[str, dict[str, Any]]:
    scheme_id = _new_id(base_id, "linear_svm")
    return (
        f"TRB/set/configs/linear_svm/hierarchical_repeat3mer/{scheme_id}.yaml",
        {
            "linear_svm_scheme_version": "1.0",
            "scheme": {
                "id": scheme_id,
                "description": (
                    f"{cohort.upper()} frozen 100-repeat hierarchical TRB comparison "
                    "using LinearSVC. The exact cohort model definitions, outer/inner "
                    "assignments, repeat-specific 3-mer contract, and public-reference "
                    "cache are inherited from the corresponding V3 resolved config."
                ),
                "base_resolved_config": (
                    f"TRB/set/experiments/{base_id}/00_config/resolved_config.yaml"
                ),
                "output_root": f"TRB/set/experiments/{scheme_id}",
            },
            "model_engine": {
                "engine": "linear_svc",
                "candidate_blocks": [
                    {
                        "id": "l2_squared_hinge_balanced",
                        "priority": 1,
                        "penalty": "l2",
                        "loss": "squared_hinge",
                        "class_weight_grid": ["balanced"],
                        "C_grid": list(SVM_C_GRID),
                        "dual": "auto",
                    }
                ],
                "max_iter": 10000,
                "tolerance": 1.0e-4,
                "zero_sd_tolerance": 1.0e-12,
                "fit_intercept": True,
                "intercept_scaling": 1.0,
            },
        },
    )


def _build_xgb(cohort: str, base_id: str) -> tuple[str, dict[str, Any]]:
    scheme_id = _new_id(base_id, "xgboost")
    return (
        f"TRB/set/configs/xgboost/hierarchical_repeat3mer/{scheme_id}.yaml",
        {
            "xgboost_scheme_version": "1.0",
            "scheme": {
                "id": scheme_id,
                "description": (
                    f"{cohort.upper()} frozen 100-repeat hierarchical TRB comparison "
                    "using XGBoost with a deterministic frozen 100-candidate random "
                    "search. The exact cohort model definitions, frozen assignments, "
                    "repeat-specific 3-mer contract, and public-reference cache are "
                    "inherited from the corresponding V3 resolved config."
                ),
                "base_resolved_config": (
                    f"TRB/set/experiments/{base_id}/00_config/resolved_config.yaml"
                ),
                "output_root": f"TRB/set/experiments/{scheme_id}",
            },
            "tuning_primary_metric": "roc_auc",
            "model_engine": {
                "engine": "xgboost",
                "fixed_parameters": {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "tree_method": "hist",
                    "verbosity": 0,
                },
                "early_stopping": False,
                "hyperparameters": copy.deepcopy(XGB_HYPERPARAMETERS),
                "search": {
                    "strategy": "random",
                    "n_iter": XGB_CANDIDATES,
                    "random_state": 20260807,
                    "replacement": False,
                },
                "runtime": {"n_jobs_per_fit": 1},
                "zero_sd_tolerance": 1.0e-12,
            },
        },
    )


def _run_preparer(root: Path, script: str, scheme_path: Path, dry_run: bool) -> None:
    command = [
        sys.executable,
        str(root / "TRB/set/scripts_v2" / script),
        "--scheme",
        str(scheme_path),
        "--repository-root",
        str(root),
    ]
    if dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=root, check=True)


def prepare(root: Path, apply: bool) -> Path:
    root = root.resolve()
    entries: list[dict[str, Any]] = []
    pending_files: list[tuple[Path, dict[str, Any]]] = []
    pending_preparers: list[tuple[str, Path]] = []

    for cohort, spec in BASES.items():
        base_id = str(spec["experiment"])
        base_config = _validate_base(root, cohort, spec)
        entries.append(
            {
                "cohort": cohort,
                "algorithm": "elastic_net",
                "config": base_config.relative_to(root).as_posix(),
                "base_experiment": base_id,
                "expected_samples": int(spec["samples"]),
                "expected_models": list(spec["models"]),
                "outer_repeats": 100,
                "outer_tasks": 100,
                "candidate_count_per_model": 24,
                "existing_repeat_01_expected": True,
            }
        )

        for algorithm, alpha in (("ridge", 0.0), ("lasso", 1.0)):
            source_path, source = _build_ridge_lasso(
                root, cohort, base_id, algorithm, alpha
            )
            scheme_id = str(source["scheme"]["id"])
            output_root = root / "TRB/set/experiments" / scheme_id
            if source_path.exists() or output_root.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing {algorithm} scheme: "
                    f"{source_path} or {output_root}"
                )
            pending_files.append((source_path, source))
            pending_preparers.append(("prepare_analysis_scheme.py", source_path))
            entries.append(
                {
                    "cohort": cohort,
                    "algorithm": algorithm,
                    "config": f"TRB/set/experiments/{scheme_id}/00_config/resolved_config.yaml",
                    "scheme": source_path.relative_to(root).as_posix(),
                    "base_experiment": base_id,
                    "expected_samples": int(spec["samples"]),
                    "expected_models": list(spec["models"]),
                    "outer_repeats": 100,
                    "outer_tasks": 100,
                    "candidate_count_per_model": 8,
                }
            )

        svm_relative, svm_source = _build_svm(cohort, base_id)
        svm_path = root / svm_relative
        svm_id = str(svm_source["scheme"]["id"])
        svm_output = root / "TRB/set/experiments" / svm_id
        if svm_path.exists() or svm_output.exists():
            raise FileExistsError(f"Refusing to overwrite existing Linear SVM scheme: {svm_path}")
        pending_files.append((svm_path, svm_source))
        pending_preparers.append(("prepare_linear_svm_scheme.py", svm_path))
        entries.append(
            {
                "cohort": cohort,
                "algorithm": "linear_svm",
                "config": f"TRB/set/experiments/{svm_id}/00_config/resolved_config.yaml",
                "scheme": svm_relative,
                "base_experiment": base_id,
                "expected_samples": int(spec["samples"]),
                "expected_models": list(spec["models"]),
                "outer_repeats": 100,
                "outer_tasks": 100,
                "candidate_count_per_model": len(SVM_C_GRID),
            }
        )

        xgb_relative, xgb_source = _build_xgb(cohort, base_id)
        xgb_path = root / xgb_relative
        xgb_id = str(xgb_source["scheme"]["id"])
        xgb_output = root / "TRB/set/experiments" / xgb_id
        if xgb_path.exists() or xgb_output.exists():
            raise FileExistsError(f"Refusing to overwrite existing XGBoost scheme: {xgb_path}")
        pending_files.append((xgb_path, xgb_source))
        pending_preparers.append(("prepare_xgboost_scheme.py", xgb_path))
        pool_size = math.prod(
            len(value) if isinstance(value, list) else 1
            for value in XGB_HYPERPARAMETERS.values()
        )
        entries.append(
            {
                "cohort": cohort,
                "algorithm": "xgboost",
                "config": f"TRB/set/experiments/{xgb_id}/00_config/resolved_config.yaml",
                "scheme": xgb_relative,
                "base_experiment": base_id,
                "expected_samples": int(spec["samples"]),
                "expected_models": list(spec["models"]),
                "outer_repeats": 100,
                "outer_tasks": 100,
                "candidate_count_per_model": XGB_CANDIDATES,
                "candidate_pool_size": pool_size,
                "n_jobs_per_fit": 1,
            }
        )

    manifest = {
        "manifest_version": "1.0",
        "purpose": "TRB V3 179-sample five-algorithm hierarchical repeated-holdout training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_branch": "feature/trb-v3-179-repeat3mer-bundle-adapter",
        "cohort_contract": {
            name: {
                "experiment": value["experiment"],
                "samples": value["samples"],
                "models": list(value["models"]),
                "outer_repeats": 100,
                "train_holdout": {
                    "total": [125, 54],
                    "pbmc": [81, 35],
                    "buffycoat": [44, 19],
                }[name],
            }
            for name, value in BASES.items()
        },
        "algorithm_contract": {
            "elastic_net": {"alpha_grid": [0.1, 0.5, 0.9], "lambda_grid_size": 8},
            "ridge": {"alpha_grid": [0.0], "lambda_grid_size": 8},
            "lasso": {"alpha_grid": [1.0], "lambda_grid_size": 8},
            "linear_svm": {"C_grid": list(SVM_C_GRID)},
            "xgboost": {
                "candidate_count": XGB_CANDIDATES,
                "candidate_pool_size": math.prod(
                    len(value) if isinstance(value, list) else 1
                    for value in XGB_HYPERPARAMETERS.values()
                ),
                "n_jobs_per_fit": 1,
                "early_stopping": False,
            },
        },
        "entries": entries,
    }

    manifest_path = root / MANIFEST_PATH
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite run manifest: {manifest_path}")

    print("TRB V3 hierarchical five-algorithm scheme preparation")
    print(f"Repository: {root}")
    print(f"New scheme YAMLs: {len(pending_files)}")
    print(f"Training configurations: {len(entries)}")
    print(f"Outer tasks: {sum(int(x['outer_tasks']) for x in entries)}")
    print("Existing Elastic Net resolved configs/results: READ-ONLY")
    for entry in entries:
        print(
            f"[PLAN] {entry['cohort']:9s} {entry['algorithm']:12s} "
            f"samples={entry['expected_samples']:3d} "
            f"models={len(entry['expected_models'])} "
            f"repeats={entry['outer_repeats']} "
            f"candidates/model={entry['candidate_count_per_model']}"
        )
    if not apply:
        print("DRY-RUN PASS: no files or experiment directories were written.")
        print("Add --apply to create the new Ridge/Lasso/SVM/XGBoost schemes.")
        return manifest_path

    for path, source in pending_files:
        _dump_yaml(path, source)
        print(f"[WRITE] {path.relative_to(root)}")

    # Dry-run every new preparer before writing any resolved configuration.
    for script, path in pending_preparers:
        _run_preparer(root, script, path, dry_run=True)

    # Prepare isolated experiment roots. Existing Elastic Net roots are untouched.
    for script, path in pending_preparers:
        _run_preparer(root, script, path, dry_run=False)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[WRITE] {manifest_path.relative_to(root)}")
    print("FIVE-ALGORITHM PREPARATION: COMPLETE")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Ridge, Lasso, Linear SVM and XGBoost schemes for the frozen TRB V3 cohorts."
    )
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        prepare(Path(args.repository_root), apply=bool(args.apply))
        return 0
    except Exception as exc:
        print(f"FIVE-ALGORITHM PREPARATION: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
