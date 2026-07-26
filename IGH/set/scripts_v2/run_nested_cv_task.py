#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one IGH V2 nested-CV outer task into a separate V2 output directory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.config import load_experiment_config  # noqa: E402
from ra_ild_igh.feature_inputs import load_partition_feature_matrix  # noqa: E402
from ra_ild_igh.nested_cv import NestedCVOptions, run_nested_outer_task  # noqa: E402
from ra_ild_igh.public_reference import ThresholdScheme  # noqa: E402
from ra_ild_igh.specifications import (  # noqa: E402
    build_hyperparameter_grid,
    resolve_all_model_specifications,
    select_static_igh_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one leakage-controlled IGH V2 nested-CV outer task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--outer-repeat", type=int, default=None)
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"{label} has duplicated columns: {duplicates[:20]}")
    return frame


def load_inputs(config) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    sparse.csr_matrix,
    sparse.csr_matrix,
    Mapping[str, object],
]:
    base, feature_input_audit = load_partition_feature_matrix(config, "train")
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
    metadata_path = config.path(
        "public_reference.cache.metadata", must_exist=True, expect="file"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample_ids = [str(value) for value in metadata.get("sample_ids", [])]
    if "sample_id" not in base.columns:
        raise ValueError("Training base matrix is missing sample_id.")
    base["sample_id"] = base["sample_id"].astype(str)
    if base["sample_id"].duplicated().any():
        raise ValueError("Training base matrix contains duplicate sample_id values.")
    if base["sample_id"].tolist() != sample_ids:
        raise ValueError(
            "Sparse-cache row order must exactly match the training base-matrix order."
        )
    presence = sparse.load_npz(
        config.path("public_reference.cache.presence", must_exist=True, expect="file")
    ).tocsr()
    frequency = sparse.load_npz(
        config.path("public_reference.cache.frequency", must_exist=True, expect="file")
    ).tocsr()
    expected_shape = (
        int(config.raw["data"]["train"]["expected_samples"]),
        int(config.raw["public_reference"]["expected_sizes"]["catalog"]),
    )
    if presence.shape != expected_shape or frequency.shape != expected_shape:
        raise ValueError(
            f"Sparse cache shape mismatch: presence={presence.shape}, "
            f"frequency={frequency.shape}, expected={expected_shape}."
        )
    return base, feature_input_audit, manifest, outer, inner, presence, frequency, metadata


def output_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "configuration": output_dir / "06_task_configuration.json",
        "feature_input_audit": output_dir / "06_feature_input_audit.csv",
        "sample_roles": output_dir / "06_task_sample_roles.csv",
        "outer_train_public": output_dir / "06_dynamic_public_features_outer_train_loo.csv",
        "outer_valid_public": output_dir / "06_dynamic_public_features_outer_validation.csv",
        "public_reference_summary": output_dir / "06_public_reference_build_summary.csv",
        "public_loo_assignments": output_dir / "06_public_loo_assignments.csv",
        "inner_tuning": output_dir / "06_inner_tuning_results.csv",
        "inner_oof": output_dir / "06_inner_selected_oof_predictions.csv",
        "outer_predictions": output_dir / "06_outer_validation_predictions.csv",
        "outer_metrics": output_dir / "06_outer_validation_metrics.csv",
        "coefficients": output_dir / "06_final_model_coefficients.csv",
        "preprocessing": output_dir / "06_preprocessing_summary.csv",
        "static_feature_list": output_dir / "06_static_igh_feature_list.csv",
        "complete": output_dir / "06_TASK_COMPLETE.json",
    }


def enforce_output_policy(paths: Mapping[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "V2 nested-CV outputs already exist; add --overwrite only for a "
            "documented rerun:\n" + "\n".join(f"  - {path}" for path in existing)
        )


def main() -> int:
    args = parse_args()
    started = time.time()
    try:
        config = load_experiment_config(
            Path(args.config),
            repository_root=Path(args.repository_root) if args.repository_root else None,
        )
        nested = config.section("nested_cv")
        task = nested["regression_task"]
        outer_repeat = int(args.outer_repeat or task["outer_repeat"])
        outer_fold = int(args.outer_fold or task["outer_fold"])
        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
        else:
            output_dir = (
                config.path("outer_tasks.output_root")
                / f"repeat_{outer_repeat:02d}_fold_{outer_fold:02d}"
            )
        paths = output_paths(output_dir)
        enforce_output_policy(paths, args.overwrite)
        output_dir.mkdir(parents=True, exist_ok=True)

        base, feature_input_audit, manifest, outer, inner, presence, frequency, metadata = load_inputs(config)
        static_features = select_static_igh_features(
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
        engine = config.section("model_engine")
        candidates = build_hyperparameter_grid(
            engine["alpha_grid"], engine["lambda_grid"]
        )
        scheme = ThresholdScheme.from_mapping(public)
        modeling = config.section("modeling")
        options = NestedCVOptions(
            base_seed=int(config.raw["experiment"]["random_seed"]),
            class_weight=str(engine["class_weight"]),
            max_iter=int(engine["max_iter"]),
            tolerance=float(engine["tolerance"]),
            zero_sd_tolerance=float(engine["zero_sd_tolerance"]),
            epsilon=float(public["epsilon"]),
            coefficient_nonzero_tolerance=float(
                modeling["coefficient_nonzero_tolerance"]
            ),
            positive_label=str(modeling["positive_label"]),
            negative_label=str(modeling["negative_label"]),
        )
        sample_ids = base["sample_id"].astype(str).tolist()
        row_lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        labels = base[options.label_column].astype(str).str.upper().to_numpy()
        aa_clone_numbers = pd.to_numeric(
            base["aa_clone_number"], errors="raise"
        ).to_numpy(float)

        result = run_nested_outer_task(
            base,
            outer,
            inner,
            row_lookup,
            presence,
            frequency,
            labels,
            aa_clone_numbers,
            model_specs,
            candidates,
            scheme,
            options,
            outer_repeat=outer_repeat,
            outer_fold=outer_fold,
            outer_folds=int(config.raw["cross_validation"]["outer_folds"]),
            inner_folds=int(config.raw["cross_validation"]["inner_folds"]),
        )

        feature_input_audit.to_csv(paths["feature_input_audit"], index=False)
        result.sample_roles.to_csv(paths["sample_roles"], index=False)
        result.outer_train_public.to_csv(paths["outer_train_public"], index=False)
        result.outer_validation_public.to_csv(paths["outer_valid_public"], index=False)
        result.public_reference_audit.to_csv(paths["public_reference_summary"], index=False)
        result.public_loo_assignments.to_csv(paths["public_loo_assignments"], index=False)
        result.inner_tuning.to_csv(paths["inner_tuning"], index=False)
        result.inner_selected_oof_predictions.to_csv(paths["inner_oof"], index=False)
        result.outer_predictions.to_csv(paths["outer_predictions"], index=False)
        result.outer_metrics.to_csv(paths["outer_metrics"], index=False)
        result.coefficients.to_csv(paths["coefficients"], index=False)
        result.preprocessing_audit.to_csv(paths["preprocessing"], index=False)
        pd.DataFrame(
            {
                "feature_order": np.arange(1, len(static_features) + 1),
                "feature_name": static_features,
            }
        ).to_csv(paths["static_feature_list"], index=False)

        selected = {
            model: {
                "alpha": value.candidate.alpha,
                "lambda": value.candidate.lambda_value,
                "threshold": value.threshold,
                "inner_roc_auc": value.inner_roc_auc,
                "inner_pr_auc": value.inner_pr_auc,
            }
            for model, value in result.selected.items()
        }
        configuration = {
            "experiment_id": config.experiment_id,
            "merged_feature_column_count": int(len(base.columns)),
            "additional_feature_table_count": int((feature_input_audit["source_type"] == "additional_feature_table").sum()),
            "outer_repeat": outer_repeat,
            "outer_fold": outer_fold,
            "outer_train_samples": result.split.n_outer_train,
            "outer_validation_samples": result.split.n_outer_valid,
            "inner_folds": list(result.split.inner_folds),
            "models": list(model_specs),
            "candidate_count_per_model": len(candidates),
            "expected_inner_fit_count": len(model_specs)
            * len(candidates)
            * len(result.split.inner_folds),
            "selected": selected,
            "cache_metadata": str(
                config.path("public_reference.cache.metadata", must_exist=True)
            ),
            "cache_matrix_shape": metadata.get("matrix_shape"),
            "independent_test_read": False,
            "runtime_seconds": time.time() - started,
        }
        paths["configuration"].write_text(
            json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        complete = {
            "status": "COMPLETE",
            "outer_repeat": outer_repeat,
            "outer_fold": outer_fold,
            "output_dir": str(output_dir),
            "runtime_seconds": time.time() - started,
        }
        paths["complete"].write_text(
            json.dumps(complete, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print("IGH V2 nested-CV task: COMPLETE")
        print(f"Outer task: repeat={outer_repeat}, fold={outer_fold}")
        print(
            f"Samples: train={result.split.n_outer_train}, "
            f"validation={result.split.n_outer_valid}"
        )
        print(
            f"Fits: {len(model_specs)} models × {len(candidates)} candidates × "
            f"{len(result.split.inner_folds)} folds = "
            f"{len(model_specs) * len(candidates) * len(result.split.inner_folds)} inner fits"
        )
        print(f"Output: {output_dir}")
        return 0
    except Exception as exc:
        print(f"IGH V2 nested-CV task: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
