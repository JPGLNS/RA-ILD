#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare IGH V2 preprocessing against one frozen V1 outer task.

The checker reconstructs the outer-final design matrices for every configured
model using:

* the frozen training base matrix;
* the frozen outer-fold assignment;
* the V1 static-feature list;
* the V1 outer-train LOO and outer-validation public-feature tables.

It then compares V2 preprocessing metadata with
``05_preprocessing_summary.csv`` and compares the retained feature order with
``05_final_model_coefficients.csv``. No model is fitted and no existing result
is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.config import ConfigError, load_experiment_config  # noqa: E402
from ra_ild_igh.preprocessing import (  # noqa: E402
    PreprocessingError,
    PreparedDesign,
    prepare_design_matrices,
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
        description="Compare V2 preprocessing with a frozen V1 outer task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def add(results: List[CheckResult], name: str, ok: bool, detail: str) -> None:
    results.append(CheckResult(name, "PASS" if ok else "FAIL", detail))


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "1", "yes", "false", "0", "no"}
    bad = ~normalized.isin(allowed)
    if bad.any():
        examples = normalized[bad].drop_duplicates().head(10).tolist()
        raise ValueError(f"Invalid boolean values: {examples}")
    return normalized.isin({"true", "1", "yes"})


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    frame = frame.drop(
        columns=[c for c in frame.columns if str(c).startswith("Unnamed:")],
        errors="ignore",
    )
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{label} has duplicated columns: {duplicates[:20]}")
    return frame


def require_unique_ids(frame: pd.DataFrame, id_col: str, label: str) -> None:
    if id_col not in frame.columns:
        raise ValueError(f"{label} is missing {id_col}")
    ids = frame[id_col].astype(str)
    if ids.duplicated().any():
        examples = ids[ids.duplicated(keep=False)].unique().tolist()[:10]
        raise ValueError(f"{label} has duplicated {id_col}: {examples}")
    frame[id_col] = ids


def load_static_features(path: Path) -> List[str]:
    frame = read_csv(path, "V1 static feature list")
    if "feature_name" not in frame.columns:
        raise ValueError("V1 static feature list is missing feature_name")
    features = frame["feature_name"].astype(str).tolist()
    if not features or len(features) != len(set(features)):
        raise ValueError("V1 static feature list is empty or duplicated")
    return features


def model_columns(
    model_spec: Mapping[str, Any],
    static_features: Sequence[str],
) -> Tuple[List[str], List[str]]:
    numeric = [str(x) for x in model_spec.get("numeric", [])]
    if model_spec.get("static_feature_groups"):
        numeric.extend(static_features)
    numeric.extend(str(x) for x in model_spec.get("dynamic_public", []))
    categorical = [str(x) for x in model_spec.get("categorical", [])]
    if len(numeric) != len(set(numeric)):
        raise ValueError(f"Resolved numeric columns contain duplicates: {numeric}")
    if len(categorical) != len(set(categorical)):
        raise ValueError(
            f"Resolved categorical columns contain duplicates: {categorical}"
        )
    return numeric, categorical


def merge_public(
    base: pd.DataFrame,
    public: pd.DataFrame,
    ids: Sequence[str],
    id_col: str,
    label: str,
) -> pd.DataFrame:
    require_unique_ids(public, id_col, label)
    requested = list(map(str, ids))
    observed = set(public[id_col])
    missing = [sample_id for sample_id in requested if sample_id not in observed]
    extra = sorted(observed - set(requested))
    if missing or extra:
        raise ValueError(
            f"{label} sample mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )
    aligned = public.set_index(id_col).loc[requested]
    output = base.loc[requested].copy()
    collisions = sorted(set(output.columns) & set(aligned.columns))
    if collisions:
        raise ValueError(f"{label} collides with base columns: {collisions}")
    for column in aligned.columns:
        output[column] = pd.to_numeric(aligned[column], errors="raise").to_numpy(float)
    return output


def compare_audit(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    model_name: str,
    rtol: float,
    atol: float,
) -> Tuple[bool, str]:
    columns = [
        "feature_name",
        "source_type",
        "training_mean",
        "training_sd",
        "kept_after_zero_variance_filter",
    ]
    missing = [column for column in columns if column not in expected.columns]
    if missing:
        return False, f"V1 preprocessing summary missing columns: {missing}"

    exp = expected.loc[expected["model"].astype(str) == model_name, columns].copy()
    obs = observed[columns].copy()
    if exp.empty:
        return False, f"No V1 preprocessing rows found for {model_name}"

    exp["kept_after_zero_variance_filter"] = as_bool(
        exp["kept_after_zero_variance_filter"]
    )
    obs["kept_after_zero_variance_filter"] = as_bool(
        obs["kept_after_zero_variance_filter"]
    )

    if len(exp) != len(obs):
        return False, f"row_count V1={len(exp)}, V2={len(obs)}"
    if exp["feature_name"].tolist() != obs["feature_name"].tolist():
        for index, (left, right) in enumerate(
            zip(exp["feature_name"], obs["feature_name"])
        ):
            if left != right:
                return False, f"feature order differs at row {index}: V1={left}, V2={right}"
        return False, "feature order differs"
    if exp["source_type"].tolist() != obs["source_type"].tolist():
        return False, "source_type sequence differs"
    if exp["kept_after_zero_variance_filter"].tolist() != obs[
        "kept_after_zero_variance_filter"
    ].tolist():
        return False, "zero-variance keep flags differ"

    numeric_columns = ["training_mean", "training_sd"]
    max_difference = 0.0
    max_location = ""
    for column in numeric_columns:
        left = pd.to_numeric(exp[column], errors="raise").to_numpy(float)
        right = pd.to_numeric(obs[column], errors="raise").to_numpy(float)
        differences = np.abs(left - right)
        index = int(np.argmax(differences)) if len(differences) else 0
        if len(differences) and differences[index] > max_difference:
            max_difference = float(differences[index])
            max_location = f"{obs.iloc[index]['feature_name']}:{column}"
        if not np.allclose(left, right, rtol=rtol, atol=atol):
            return (
                False,
                f"{column} differs; max_abs_diff={differences[index]:.12g}, "
                f"feature={obs.iloc[index]['feature_name']}",
            )
    return True, f"rows={len(obs)}, max_abs_diff={max_difference:.12g}, location={max_location}"


def coefficient_feature_order(
    coefficient_frame: pd.DataFrame,
    model_name: str,
) -> List[str]:
    required = {"model", "feature_name"}
    missing = sorted(required - set(coefficient_frame.columns))
    if missing:
        raise ValueError(f"V1 coefficient table missing columns: {missing}")
    sub = coefficient_frame.loc[
        coefficient_frame["model"].astype(str) == model_name,
        "feature_name",
    ].astype(str)
    return [name for name in sub.tolist() if name != "__INTERCEPT__"]


def main() -> int:
    args = parse_args()
    results: List[CheckResult] = []
    try:
        config = load_experiment_config(
            Path(args.config),
            repository_root=(
                Path(args.repository_root) if args.repository_root else None
            ),
        )
        section = config.section("preprocessing")
        task = section["regression_task"]
        outer_repeat = int(task["outer_repeat"])
        outer_fold = int(task["outer_fold"])
        task_dir = config.path("preprocessing.regression_task.task_dir", must_exist=True, expect="dir")

        base = read_csv(config.path("data.train.base_matrix", must_exist=True), "train base matrix")
        require_unique_ids(base, "sample_id", "train base matrix")
        base = base.set_index("sample_id", drop=False)

        outer = read_csv(
            config.path("cross_validation.outer_assignments", must_exist=True),
            "outer assignments",
        )
        required_outer = {"outer_repeat", "outer_fold", "sample_id"}
        missing_outer = sorted(required_outer - set(outer.columns))
        if missing_outer:
            raise ValueError(f"Outer assignments missing columns: {missing_outer}")
        selected = outer.loc[
            pd.to_numeric(outer["outer_repeat"], errors="raise").astype(int)
            == outer_repeat
        ].copy()
        selected["sample_id"] = selected["sample_id"].astype(str)
        train_set = set(
            selected.loc[
                pd.to_numeric(selected["outer_fold"], errors="raise").astype(int)
                != outer_fold,
                "sample_id",
            ]
        )
        valid_set = set(
            selected.loc[
                pd.to_numeric(selected["outer_fold"], errors="raise").astype(int)
                == outer_fold,
                "sample_id",
            ]
        )
        train_ids = [sample_id for sample_id in base.index if sample_id in train_set]
        valid_ids = [sample_id for sample_id in base.index if sample_id in valid_set]
        if train_set & valid_set or train_set | valid_set != set(base.index):
            raise ValueError("Outer train/validation partition is invalid")

        static_features = load_static_features(
            config.path(
                "preprocessing.regression_task.static_feature_list",
                must_exist=True,
                expect="file",
            )
        )
        public_train = read_csv(
            config.path(
                "preprocessing.regression_task.outer_train_public",
                must_exist=True,
                expect="file",
            ),
            "V1 outer-train public features",
        )
        public_valid = read_csv(
            config.path(
                "preprocessing.regression_task.outer_validation_public",
                must_exist=True,
                expect="file",
            ),
            "V1 outer-validation public features",
        )
        train_frame = merge_public(
            base,
            public_train,
            train_ids,
            "sample_id",
            "V1 outer-train public features",
        )
        valid_frame = merge_public(
            base,
            public_valid,
            valid_ids,
            "sample_id",
            "V1 outer-validation public features",
        )

        v1_audit = read_csv(
            config.path(
                "preprocessing.regression_task.preprocessing_summary",
                must_exist=True,
                expect="file",
            ),
            "V1 preprocessing summary",
        )
        v1_coefficients = read_csv(
            config.path(
                "preprocessing.regression_task.coefficients",
                must_exist=True,
                expect="file",
            ),
            "V1 coefficient table",
        )

        add(
            results,
            "Outer task sample counts",
            len(train_ids) + len(valid_ids) == len(base),
            f"train={len(train_ids)}, validation={len(valid_ids)}, total={len(base)}",
        )
        add(
            results,
            "Static feature count",
            len(static_features) > 0,
            f"n_static={len(static_features)}",
        )

        tolerance = float(config.raw["model_engine"]["zero_sd_tolerance"])
        for model_name, model_spec in config.section("models").items():
            numeric, categorical = model_columns(model_spec, static_features)
            prepared: PreparedDesign = prepare_design_matrices(
                train_frame,
                valid_frame,
                numeric,
                categorical,
                zero_sd_tolerance=tolerance,
                strict_unseen_categories=False,
            )
            audit = prepared.audit.copy()
            audit["stage"] = "outer_final"
            audit["model"] = model_name
            audit["inner_fold"] = ""

            ok, detail = compare_audit(
                audit,
                v1_audit,
                model_name,
                args.rtol,
                args.atol,
            )
            add(results, f"{model_name} preprocessing audit equality", ok, detail)

            expected_order = coefficient_feature_order(v1_coefficients, model_name)
            observed_order = list(prepared.feature_names)
            add(
                results,
                f"{model_name} retained feature order",
                observed_order == expected_order,
                f"V2={len(observed_order)}, V1={len(expected_order)}",
            )

            train_means = prepared.X_train.mean(axis=0)
            train_sds = prepared.X_train.std(axis=0, ddof=0)
            standard_ok = bool(
                np.allclose(train_means, 0.0, rtol=args.rtol, atol=1e-12)
                and np.allclose(train_sds, 1.0, rtol=args.rtol, atol=1e-12)
            )
            add(
                results,
                f"{model_name} standardized training matrix",
                standard_ok,
                f"max_abs_mean={np.max(np.abs(train_means)):.12g}, "
                f"max_abs_sd_minus_1={np.max(np.abs(train_sds-1)):.12g}",
            )
            add(
                results,
                f"{model_name} finite validation matrix",
                bool(np.isfinite(prepared.X_valid).all()),
                f"shape={prepared.X_valid.shape}, unseen={dict(prepared.valid_unseen_category_counts)}",
            )

        failed = [result for result in results if result.status == "FAIL"]
        print(
            f"Outer task={outer_repeat}/{outer_fold} "
            f"Checks={len(results)} PASS={len(results)-len(failed)} FAIL={len(failed)}"
        )
        for result in results:
            print(f"[{result.status}] {result.name}: {result.detail}")

        report = {
            "experiment_id": config.experiment_id,
            "outer_repeat": outer_repeat,
            "outer_fold": outer_fold,
            "status": "PASS" if not failed else "FAIL",
            "n_checks": len(results),
            "n_pass": len(results) - len(failed),
            "n_fail": len(failed),
            "checks": [result.as_dict() for result in results],
        }
        if args.json_output:
            output = Path(args.json_output).expanduser()
            if not output.is_absolute():
                output = config.repository_root / output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"JSON report: {output.resolve()}")
        return 0 if not failed else 1

    except (ConfigError, PreprocessingError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"Preprocessing regression check failed before completion: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
