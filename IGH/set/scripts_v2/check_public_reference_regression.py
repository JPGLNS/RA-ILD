#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression-check the V2 public-reference module against frozen V1 outputs.

The script performs two independent checks without changing any analysis files:

1. Rebuild the full 123-sample training reference from the existing sparse cache
   and compare every saved mask with the locked step-07 reference-mask NPZ.
2. Recompute one frozen outer task (default repeat 1, fold 1) and compare all 18
   LOO/external public features with the existing V1 step-05 CSV outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.config import load_experiment_config  # noqa: E402
from ra_ild_igh.public_reference import (  # noqa: E402
    ALL_PUBLIC_FEATURES,
    ThresholdScheme,
    build_reference_from_presence,
    external_public_features,
    leave_one_out_public_features,
)


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str

    def as_dict(self) -> Dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare V2 public-reference calculations with frozen V1 outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument(
        "--outer-repeat",
        type=int,
        default=None,
        help="Override public_reference.regression_task.outer_repeat.",
    )
    parser.add_argument(
        "--outer-fold",
        type=int,
        default=None,
        help="Override public_reference.regression_task.outer_fold.",
    )
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument(
        "--skip-task-features",
        action="store_true",
        help="Check only the full-training reference masks.",
    )
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def add_result(
    results: List[CheckResult],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    results.append(CheckResult(name, "PASS" if passed else "FAIL", detail))


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def read_base_and_cache_metadata(
    base_path: Path,
    cache_metadata_path: Path,
    expected_samples: int,
) -> Tuple[pd.DataFrame, Mapping[str, Any], List[str]]:
    require_file(base_path, "Training base matrix")
    require_file(cache_metadata_path, "Sparse-cache metadata")
    base = pd.read_csv(base_path)
    required = {"sample_id", "cohort", "aa_clone_number"}
    missing = sorted(required - set(base.columns))
    if missing:
        raise ValueError(f"Training base matrix missing columns: {missing}")
    if len(base) != expected_samples:
        raise ValueError(
            f"Training base matrix rows={len(base)}, expected={expected_samples}."
        )
    base["sample_id"] = base["sample_id"].astype(str)
    if base["sample_id"].duplicated().any():
        raise ValueError("Training base matrix contains duplicate sample_id values.")
    base["cohort"] = base["cohort"].astype(str).str.strip().str.upper()
    if set(base["cohort"]) != {"RA", "ILD"}:
        raise ValueError("Training cohort must contain exactly RA and ILD.")
    base["aa_clone_number"] = pd.to_numeric(
        base["aa_clone_number"], errors="raise"
    )

    metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    sample_ids = [str(value) for value in metadata.get("sample_ids", [])]
    if len(sample_ids) != expected_samples or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Sparse-cache metadata has invalid sample_ids.")
    if set(sample_ids) != set(base["sample_id"]):
        only_cache = sorted(set(sample_ids) - set(base["sample_id"]))[:10]
        only_base = sorted(set(base["sample_id"]) - set(sample_ids))[:10]
        raise ValueError(
            "Sparse-cache and base sample sets differ; "
            f"only_cache={only_cache}, only_base={only_base}."
        )
    ordered = base.set_index("sample_id", drop=False).loc[sample_ids].copy()
    return ordered, metadata, sample_ids


def load_sparse_cache(
    presence_path: Path,
    frequency_path: Path,
    cache_metadata: Mapping[str, Any],
    expected_samples: int,
    expected_catalog: int,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    require_file(presence_path, "Public presence cache")
    require_file(frequency_path, "Public frequency cache")
    presence = sparse.load_npz(presence_path).tocsr()
    frequency = sparse.load_npz(frequency_path).tocsr()
    expected_shape = (expected_samples, expected_catalog)
    if presence.shape != expected_shape:
        raise ValueError(
            f"Presence cache shape={presence.shape}, expected={expected_shape}."
        )
    if frequency.shape != expected_shape:
        raise ValueError(
            f"Frequency cache shape={frequency.shape}, expected={expected_shape}."
        )
    metadata_shape = tuple(cache_metadata.get("matrix_shape", []))
    if metadata_shape and metadata_shape != expected_shape:
        raise ValueError(
            f"Cache metadata shape={metadata_shape}, expected={expected_shape}."
        )
    return presence, frequency


def check_full_reference(
    results: List[CheckResult],
    presence: sparse.csr_matrix,
    labels: np.ndarray,
    scheme: ThresholdScheme,
    expected_sizes: Mapping[str, Any],
    masks_path: Path,
) -> None:
    require_file(masks_path, "Step-07 reference masks")
    reference = build_reference_from_presence(
        presence,
        np.arange(presence.shape[0], dtype=int),
        labels,
        scheme,
    )
    observed_sizes = reference.sizes()
    for key in ("global", "RA_specific", "ILD_specific", "shared"):
        expected = int(expected_sizes[key])
        observed = int(observed_sizes[key])
        add_result(
            results,
            f"Full reference size {key}",
            observed == expected,
            f"observed={observed}, expected={expected}",
        )

    locked = np.load(masks_path)
    comparisons = {
        "global_mask": reference.global_mask,
        "RA_specific_mask": reference.ra_specific_mask,
        "ILD_specific_mask": reference.ild_specific_mask,
        "shared_mask": reference.shared_mask,
    }
    for key, rebuilt in comparisons.items():
        if key not in locked.files:
            add_result(results, f"Exact mask equality {key}", False, "missing in NPZ")
            continue
        saved = np.asarray(locked[key]).astype(bool)
        differing = int(np.count_nonzero(saved != rebuilt)) if saved.shape == rebuilt.shape else -1
        add_result(
            results,
            f"Exact mask equality {key}",
            saved.shape == rebuilt.shape and differing == 0,
            f"shape_saved={saved.shape}, shape_v2={rebuilt.shape}, differing={differing}",
        )

    add_result(
        results,
        "Full reference effective thresholds",
        True,
        (
            f"global={reference.global_threshold}, RA={reference.ra_threshold}, "
            f"ILD={reference.ild_threshold}; n_RA={reference.n_ra}, n_ILD={reference.n_ild}"
        ),
    )


def align_v1_features(
    path: Path,
    expected_ids: Sequence[str],
    label: str,
) -> pd.DataFrame:
    require_file(path, label)
    frame = pd.read_csv(path)
    required = {"sample_id", *ALL_PUBLIC_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate sample_id values.")
    if set(frame["sample_id"]) != set(expected_ids):
        raise ValueError(f"{label} sample IDs differ from expected task IDs.")
    return frame.set_index("sample_id").loc[list(expected_ids), list(ALL_PUBLIC_FEATURES)]


def compare_feature_frames(
    results: List[CheckResult],
    name: str,
    v2: pd.DataFrame,
    v1: pd.DataFrame,
    rtol: float,
    atol: float,
) -> None:
    left = v2.loc[v1.index, list(ALL_PUBLIC_FEATURES)].to_numpy(dtype=float)
    right = v1.loc[:, list(ALL_PUBLIC_FEATURES)].to_numpy(dtype=float)
    difference = np.abs(left - right)
    max_abs = float(np.max(difference)) if difference.size else 0.0
    passed = bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False))
    if difference.size:
        flat_index = int(np.argmax(difference))
        row_index, column_index = np.unravel_index(flat_index, difference.shape)
        location = (
            f"sample={v1.index[row_index]}, feature={ALL_PUBLIC_FEATURES[column_index]}"
        )
    else:
        location = "empty"
    add_result(
        results,
        name,
        passed,
        f"max_abs_diff={max_abs:.12g}, location={location}, rtol={rtol}, atol={atol}",
    )


def check_one_outer_task(
    results: List[CheckResult],
    config: Any,
    presence: sparse.csr_matrix,
    frequency: sparse.csr_matrix,
    ordered_base: pd.DataFrame,
    sample_ids: Sequence[str],
    labels: np.ndarray,
    aa_clone_numbers: np.ndarray,
    scheme: ThresholdScheme,
    epsilon: float,
    outer_repeat: int,
    outer_fold: int,
    rtol: float,
    atol: float,
) -> None:
    outer_path = config.path(
        "cross_validation.outer_assignments", must_exist=True, expect="file"
    )
    outer = pd.read_csv(outer_path)
    required = {"sample_id", "outer_repeat", "outer_fold"}
    missing = sorted(required - set(outer.columns))
    if missing:
        raise ValueError(f"Outer assignments missing columns: {missing}")
    outer["sample_id"] = outer["sample_id"].astype(str)
    current = outer.loc[
        pd.to_numeric(outer["outer_repeat"], errors="raise").astype(int)
        == outer_repeat
    ].copy()
    if len(current) != len(sample_ids) or current["sample_id"].duplicated().any():
        raise ValueError(f"Invalid outer assignment coverage for repeat {outer_repeat}.")
    if set(current["sample_id"]) != set(sample_ids):
        raise ValueError("Outer assignment sample IDs differ from sparse cache.")
    fold_by_id = current.set_index("sample_id")["outer_fold"].astype(int).to_dict()
    train_ids = [sample_id for sample_id in sample_ids if fold_by_id[sample_id] != outer_fold]
    valid_ids = [sample_id for sample_id in sample_ids if fold_by_id[sample_id] == outer_fold]
    row_lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    train_rows = np.array([row_lookup[value] for value in train_ids], dtype=int)
    valid_rows = np.array([row_lookup[value] for value in valid_ids], dtype=int)

    loo = leave_one_out_public_features(
        presence,
        frequency,
        train_rows,
        labels,
        aa_clone_numbers,
        scheme,
        epsilon,
        context=f"regression_outer_{outer_repeat}_{outer_fold}_train_loo",
    )
    external = external_public_features(
        presence,
        frequency,
        train_rows,
        valid_rows,
        labels,
        aa_clone_numbers,
        scheme,
        epsilon,
        context=f"regression_outer_{outer_repeat}_{outer_fold}_validation",
    )

    v2_train = loo.features.copy()
    v2_train.index = [sample_ids[int(row)] for row in v2_train.index]
    v2_valid = external.features.copy()
    v2_valid.index = [sample_ids[int(row)] for row in v2_valid.index]

    regression = config.raw["public_reference"]["regression_task"]
    configured_repeat = int(regression["outer_repeat"])
    configured_fold = int(regression["outer_fold"])
    task_dir = config.path(
        "public_reference.regression_task.task_dir", must_exist=True, expect="dir"
    )
    if (outer_repeat, outer_fold) != (configured_repeat, configured_fold):
        task_dir = task_dir.parent / f"repeat_{outer_repeat:02d}_fold_{outer_fold:02d}"
    train_v1_path = task_dir / "05_dynamic_public_features_outer_train_loo.csv"
    valid_v1_path = task_dir / "05_dynamic_public_features_outer_validation.csv"
    v1_train = align_v1_features(train_v1_path, train_ids, "V1 outer-train LOO features")
    v1_valid = align_v1_features(valid_v1_path, valid_ids, "V1 outer-validation features")

    compare_feature_frames(
        results,
        f"Outer task {outer_repeat}/{outer_fold} train LOO all-18 feature equality",
        v2_train,
        v1_train,
        rtol,
        atol,
    )
    compare_feature_frames(
        results,
        f"Outer task {outer_repeat}/{outer_fold} validation all-18 feature equality",
        v2_valid,
        v1_valid,
        rtol,
        atol,
    )

    expected_loo_reference = len(train_rows) - 1
    add_result(
        results,
        "LOO reference sample count",
        bool((loo.assignment_audit["n_reference_samples"] == expected_loo_reference).all()),
        f"expected_per_sample={expected_loo_reference}, n_training_samples={len(train_rows)}",
    )
    add_result(
        results,
        "External validation reference sample count",
        int(external.reference_audit.loc[0, "n_reference"]) == len(train_rows),
        f"observed={external.reference_audit.loc[0, 'n_reference']}, expected={len(train_rows)}",
    )
    add_result(
        results,
        "Outer task sample counts",
        len(train_ids) + len(valid_ids) == len(sample_ids),
        f"train={len(train_ids)}, validation={len(valid_ids)}, total={len(sample_ids)}",
    )


def main() -> int:
    args = parse_args()
    if args.rtol < 0 or args.atol < 0:
        print("--rtol and --atol must be non-negative.", file=sys.stderr)
        return 2

    results: List[CheckResult] = []
    try:
        config = load_experiment_config(
            Path(args.config),
            repository_root=(
                Path(args.repository_root) if args.repository_root else None
            ),
        )
        public = config.raw["public_reference"]
        scheme = ThresholdScheme.from_mapping(public)
        epsilon = float(public["epsilon"])
        expected_sizes = public["expected_sizes"]
        expected_samples = int(config.raw["data"]["train"]["expected_samples"])
        expected_catalog = int(expected_sizes["catalog"])

        ordered_base, cache_metadata, sample_ids = read_base_and_cache_metadata(
            config.path("data.train.base_matrix", must_exist=True, expect="file"),
            config.path(
                "public_reference.cache.metadata", must_exist=True, expect="file"
            ),
            expected_samples,
        )
        presence, frequency = load_sparse_cache(
            config.path(
                "public_reference.cache.presence", must_exist=True, expect="file"
            ),
            config.path(
                "public_reference.cache.frequency", must_exist=True, expect="file"
            ),
            cache_metadata,
            expected_samples,
            expected_catalog,
        )
        labels = ordered_base["cohort"].to_numpy(dtype=str)
        aa_clone_numbers = ordered_base["aa_clone_number"].to_numpy(dtype=float)

        add_result(
            results,
            "Sparse cache sample order",
            ordered_base["sample_id"].tolist() == sample_ids,
            f"n_samples={len(sample_ids)}",
        )
        check_full_reference(
            results,
            presence,
            labels,
            scheme,
            expected_sizes,
            config.path("final_model.reference_masks", must_exist=True, expect="file"),
        )

        if not args.skip_task_features:
            regression = public["regression_task"]
            outer_repeat = (
                int(args.outer_repeat)
                if args.outer_repeat is not None
                else int(regression["outer_repeat"])
            )
            outer_fold = (
                int(args.outer_fold)
                if args.outer_fold is not None
                else int(regression["outer_fold"])
            )
            check_one_outer_task(
                results,
                config,
                presence,
                frequency,
                ordered_base,
                sample_ids,
                labels,
                aa_clone_numbers,
                scheme,
                epsilon,
                outer_repeat,
                outer_fold,
                args.rtol,
                args.atol,
            )

    except Exception as exc:
        add_result(results, "Regression script execution", False, repr(exc))

    failed = [item for item in results if item.status == "FAIL"]
    print(f"Checks={len(results)} PASS={len(results)-len(failed)} FAIL={len(failed)}")
    for item in results:
        print(f"[{item.status}] {item.name}: {item.detail}")

    report = {
        "status": "PASS" if not failed else "FAIL",
        "n_checks": len(results),
        "n_pass": len(results) - len(failed),
        "n_fail": len(failed),
        "checks": [item.as_dict() for item in results],
    }
    if args.json_output:
        output = Path(args.json_output).expanduser()
        if not output.is_absolute():
            try:
                output = config.repository_root / output
            except UnboundLocalError:
                output = Path.cwd() / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {output.resolve()}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
