#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one leakage-controlled TRB Batch 18 XGBoost outer task."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.feature_inputs import load_partition_feature_matrix  # noqa: E402
from ra_ild_trb.nested_cv import validate_fixed_assignments  # noqa: E402
from ra_ild_trb.public_reference import ThresholdScheme  # noqa: E402
from ra_ild_trb.repeat_3mer_features import (  # noqa: E402
    build_repeat_3mer_features, options_from_config, split_legacy_static_features,
    write_repeat_3mer_outputs,
)
from ra_ild_trb.specifications import (  # noqa: E402
    parse_model_specifications, resolve_model_specification, select_static_tcr_features,
)
from ra_ild_trb.xgboost_config import candidates_from_config, load_xgboost_config  # noqa: E402
from ra_ild_trb.xgboost_model import xgboost_version  # noqa: E402
from ra_ild_trb.xgboost_nested_cv import XGBoostNestedCVOptions, run_xgboost_outer_task  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one TRB XGBoost nested-CV outer task.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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


def load_inputs(config):
    base, feature_input_audit = load_partition_feature_matrix(config, "train")
    manifest = read_csv(config.path("data.train.feature_manifest", must_exist=True, expect="file"), "feature manifest")
    outer = read_csv(config.path("cross_validation.outer_assignments", must_exist=True, expect="file"), "outer assignments")
    inner = read_csv(config.path("cross_validation.inner_assignments", must_exist=True, expect="file"), "inner assignments")
    metadata_path = config.path("public_reference.cache.metadata", must_exist=True, expect="file")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample_ids = [str(value) for value in metadata.get("sample_ids", [])]
    if "sample_id" not in base.columns:
        raise ValueError("Training base matrix is missing sample_id")
    base["sample_id"] = base["sample_id"].astype(str)
    if base["sample_id"].duplicated().any():
        raise ValueError("Training base matrix contains duplicate sample_id values")
    if base["sample_id"].tolist() != sample_ids:
        raise ValueError("Sparse-cache row order must exactly match training base-matrix order")
    presence = sparse.load_npz(config.path("public_reference.cache.presence", must_exist=True, expect="file")).tocsr()
    frequency = sparse.load_npz(config.path("public_reference.cache.frequency", must_exist=True, expect="file")).tocsr()
    expected_shape = (
        int(config.raw["data"]["train"]["expected_samples"]),
        int(config.raw["public_reference"]["expected_sizes"]["catalog"]),
    )
    if presence.shape != expected_shape or frequency.shape != expected_shape:
        raise ValueError(f"Sparse cache shape mismatch: presence={presence.shape}, frequency={frequency.shape}, expected={expected_shape}")
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
        "feature_importance": output_dir / "06_final_model_feature_importance.csv",
        "preprocessing": output_dir / "06_preprocessing_summary.csv",
        "static_feature_list": output_dir / "06_static_tcr_feature_list.csv",
        "feature_group_manifest": output_dir / "06_feature_group_manifest.csv",
        "repeat_3mer_vocabulary": output_dir / "06_repeat_3mer_vocabulary.csv",
        "repeat_3mer_matrix": output_dir / "06_repeat_3mer_feature_matrix.csv.gz",
        "repeat_3mer_audit": output_dir / "06_repeat_3mer_audit.json",
        "complete": output_dir / "06_TASK_COMPLETE.json",
    }


def enforce_output_policy(paths: Mapping[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("XGBoost outputs already exist; use --overwrite only for a documented rerun:\n" + "\n".join(f"  - {path}" for path in existing))


def _legacy_feature_groups(static_features: Sequence[str], *, group_name: str) -> Mapping[str, Tuple[str, ...]]:
    core, unweighted, weighted = split_legacy_static_features(
        static_features,
        expected_core_count=len([name for name in static_features if not str(name).startswith(("unweighted_3mer_", "weighted_3mer_"))]),
    )
    groups: Dict[str, Tuple[str, ...]] = {group_name: tuple(map(str, static_features)), "core83": tuple(core)}
    max_k = min(len(unweighted), len(weighted))
    for k in (50, 100, 200, 500):
        if k <= max_k:
            groups[f"repeat_3mer_unweighted_top{k}"] = tuple(unweighted[:k])
            groups[f"repeat_3mer_weighted_top{k}"] = tuple(weighted[:k])
            groups[f"repeat_3mer_both_top{k}"] = tuple(unweighted[:k]) + tuple(weighted[:k])
    return groups


def _resolve_models(model_mapping, *, static_feature_groups, allowed_dynamic_public):
    parsed = parse_model_specifications(model_mapping)
    return {
        name: resolve_model_specification(
            specification,
            static_feature_groups=static_feature_groups,
            allowed_dynamic_public=allowed_dynamic_public,
        )
        for name, specification in parsed.items()
    }


def _validate_expected_columns(config, model_specs) -> None:
    expected = config.raw.get("model_selection", {}).get("expected_resolved_columns", {})
    if not isinstance(expected, Mapping) or not expected:
        return
    for model_name, specification in model_specs.items():
        counts = expected.get(model_name)
        if not isinstance(counts, Mapping):
            raise ValueError(f"model_selection.expected_resolved_columns lacks {model_name}")
        observed = (len(specification.numeric), len(specification.categorical))
        wanted = (int(counts.get("numeric", -1)), int(counts.get("categorical", -1)))
        if observed != wanted:
            raise ValueError(f"Resolved predictor-count mismatch for {model_name}: expected={wanted}, observed={observed}")


def _write_static_union(path: Path, groups: Mapping[str, Sequence[str]]) -> None:
    ordered = []
    for features in groups.values():
        for feature in features:
            value = str(feature)
            if value not in ordered:
                ordered.append(value)
    pd.DataFrame({"feature_order": np.arange(1, len(ordered) + 1), "feature_name": ordered}).to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    started = time.time()
    try:
        config = load_xgboost_config(Path(args.config), repository_root=Path(args.repository_root) if args.repository_root else None)
        nested = config.section("nested_cv")
        task = nested.get("regression_task", {})
        outer_repeat = int(args.outer_repeat or task.get("outer_repeat", 1))
        outer_fold = int(args.outer_fold or task.get("outer_fold", 1))
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.path("outer_tasks.output_root") / f"repeat_{outer_repeat:02d}_fold_{outer_fold:02d}"
        paths = output_paths(output_dir)
        enforce_output_policy(paths, args.overwrite)
        output_dir.mkdir(parents=True, exist_ok=True)

        base, feature_input_audit, manifest, outer, inner, presence, frequency, metadata = load_inputs(config)
        legacy_static_features = select_static_tcr_features(manifest, available_columns=base.columns, require_all_available=True)
        sample_ids = base["sample_id"].astype(str).tolist()
        split = validate_fixed_assignments(
            sample_ids, outer, inner,
            outer_repeat=outer_repeat, outer_fold=outer_fold,
            outer_folds=int(config.raw["cross_validation"]["outer_folds"]),
            inner_folds=int(config.raw["cross_validation"]["inner_folds"]),
        )

        repeat_options = options_from_config(config)
        repeat_audit = None
        if repeat_options is not None:
            repeat_build = build_repeat_3mer_features(
                base, legacy_static_features=legacy_static_features,
                outer_train_ids=split.outer_train_ids, outer_repeat=outer_repeat,
                options=repeat_options,
            )
            base = repeat_build.augmented_base
            static_feature_groups = repeat_build.static_feature_groups
            repeat_audit = dict(repeat_build.audit)
            write_repeat_3mer_outputs(
                repeat_build,
                vocabulary_path=paths["repeat_3mer_vocabulary"],
                matrix_path=paths["repeat_3mer_matrix"],
                group_manifest_path=paths["feature_group_manifest"],
                audit_path=paths["repeat_3mer_audit"],
            )
        else:
            legacy_group_name = str(config.raw["model_selection"]["static_feature_group"])
            static_feature_groups = _legacy_feature_groups(legacy_static_features, group_name=legacy_group_name)
            rows = []
            for group_order, (group_name, features) in enumerate(static_feature_groups.items(), start=1):
                for feature_order, feature_name in enumerate(features, start=1):
                    rows.append({"group_order": group_order, "feature_group": group_name, "feature_order": feature_order, "feature_name": feature_name})
            pd.DataFrame(rows).to_csv(paths["feature_group_manifest"], index=False)

        public = config.section("public_reference")
        model_specs = _resolve_models(
            config.section("models"), static_feature_groups=static_feature_groups,
            allowed_dynamic_public=public["dynamic_features"],
        )
        _validate_expected_columns(config, model_specs)
        candidates = candidates_from_config(config)
        engine = config.section("model_engine")
        modeling = config.section("modeling")
        options = XGBoostNestedCVOptions(
            base_seed=int(config.raw["experiment"]["random_seed"]),
            tuning_primary_metric=str(config.raw["model_selection"].get("tuning_primary_metric", "roc_auc")),
            zero_sd_tolerance=float(engine.get("zero_sd_tolerance", 1.0e-12)),
            epsilon=float(public["epsilon"]),
            positive_label=str(modeling["positive_label"]),
            negative_label=str(modeling["negative_label"]),
            n_jobs_per_fit=int(engine.get("runtime", {}).get("n_jobs_per_fit", 1)),
            fixed_parameters=dict(engine.get("fixed_parameters", {})),
        )
        scheme = ThresholdScheme.from_mapping(public)
        row_lookup = {sample_id: index for index, sample_id in enumerate(base["sample_id"].astype(str).tolist())}
        labels = base[options.label_column].astype(str).str.upper().to_numpy()
        aa_clone_numbers = pd.to_numeric(base["aa_clone_number"], errors="raise").to_numpy(float)
        result = run_xgboost_outer_task(
            base, outer, inner, row_lookup, presence, frequency, labels, aa_clone_numbers,
            model_specs, candidates, scheme, options,
            outer_repeat=outer_repeat, outer_fold=outer_fold,
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
        result.feature_importance.to_csv(paths["feature_importance"], index=False)
        result.preprocessing_audit.to_csv(paths["preprocessing"], index=False)
        _write_static_union(paths["static_feature_list"], static_feature_groups)

        selected = {
            model: {
                **value.candidate.as_dict(),
                "threshold": value.threshold,
                "inner_roc_auc": value.inner_roc_auc,
                "inner_pr_auc": value.inner_pr_auc,
                "inner_log_loss": value.inner_log_loss,
                "inner_brier_score": value.inner_brier_score,
                "threshold_source": value.threshold_source,
                "candidate_selection_policy": value.candidate_selection_policy,
                "tuning_primary_metric": value.tuning_primary_metric,
            }
            for model, value in result.selected.items()
        }
        bank = engine["candidate_bank"]
        configuration = {
            "experiment_id": config.experiment_id,
            "model_engine": "xgboost",
            "xgboost_version": xgboost_version(),
            "score_type": "probability",
            "probability_metrics_available": True,
            "outer_repeat": outer_repeat,
            "outer_fold": outer_fold,
            "outer_train_samples": result.split.n_outer_train,
            "outer_validation_samples": result.split.n_outer_valid,
            "inner_folds": list(result.split.inner_folds),
            "models": list(model_specs),
            "candidate_count_per_model": len(candidates),
            "candidate_bank": str(config.path("model_engine.candidate_bank.path", must_exist=True)),
            "candidate_bank_sha256": str(bank["sha256"]),
            "candidate_pool_size": int(bank["pool_size"]),
            "expected_inner_fit_count": len(model_specs) * len(candidates) * len(result.split.inner_folds),
            "selected": selected,
            "n_jobs_per_fit": options.n_jobs_per_fit,
            "repeat_3mer_features": repeat_audit,
            "cache_metadata": str(config.path("public_reference.cache.metadata", must_exist=True)),
            "cache_matrix_shape": metadata.get("matrix_shape"),
            "independent_test_read": False,
            "runtime_seconds": time.time() - started,
        }
        paths["configuration"].write_text(json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths["complete"].write_text(json.dumps({
            "status": "COMPLETE", "engine": "xgboost", "outer_repeat": outer_repeat,
            "outer_fold": outer_fold, "output_dir": str(output_dir), "runtime_seconds": time.time() - started,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("TRB XGBoost nested-CV task: COMPLETE")
        print(f"Outer task: repeat={outer_repeat}, fold={outer_fold}")
        print(f"Samples: train={result.split.n_outer_train}, validation={result.split.n_outer_valid}")
        print(f"Fits: {len(model_specs)} models × {len(candidates)} candidates × {len(result.split.inner_folds)} folds = {len(model_specs) * len(candidates) * len(result.split.inner_folds)} inner fits")
        print(f"XGBoost: {xgboost_version()} | n_jobs/fit={options.n_jobs_per_fit}")
        print(f"Output: {output_dir}")
        return 0
    except Exception as exc:
        print(f"TRB XGBoost nested-CV task: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
