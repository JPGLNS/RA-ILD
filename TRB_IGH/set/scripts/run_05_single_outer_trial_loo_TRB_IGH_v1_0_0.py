#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run one leakage-controlled outer-fold Elastic Net trial for the paired
RA / RA-ILD TRB+IGH project.

The modeling unit is the patient. TRB and IGH static features are joined at
patient level. Dynamic public CDR3-AA features are constructed separately for
TRB and IGH and merged only after receptor-specific reference construction.

LOO in this script refers to exact leave-one-out generation of dynamic public
features for samples used to fit a model. The frozen outer CV remains repeated
5-fold CV and the inner hyperparameter search remains 5-fold CV.

Leakage controls
----------------
1. The independent test set is never read.
2. The frozen patient-level outer/inner assignments are reused.
3. For every inner task, TRB and IGH public references are rebuilt using only
   the corresponding inner-training patients.
4. Public features for model-fitting patients use exact receptor-specific LOO:
   a patient contributes neither TRB nor IGH sequences to its own reference.
5. Outer-validation public features use references built only from the outer
   training patients.
6. Encoding, zero-variance filtering and standardization are learned only from
   the current model-training partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

SCRIPT_VERSION = "1.0.0-PAIRED-TRB-IGH-loo"
ROOT = Path("/data/users/chenhaisheng/RA-ILD/TRB_IGH")

DEFAULT_BASE = (
    ROOT / "set/train/result/04_final_feature_matrix/"
           "04_train_base_feature_matrix.csv"
)
DEFAULT_MANIFEST = (
    ROOT / "set/train/result/04_final_feature_matrix/"
           "04_feature_manifest.csv"
)
DEFAULT_OUTER = (
    ROOT / "set/train/result/05_modeling/cv_splits/"
           "05_outer_fold_assignments.csv"
)
DEFAULT_INNER = (
    ROOT / "set/train/result/05_modeling/cv_splits/"
           "05_inner_fold_assignments.csv.gz"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "set/train/result/05_modeling/single_outer_trial_loo_TRB_IGH"
)

TRB_ROOT = ROOT / "set/train/TRB/result"
IGH_ROOT = ROOT / "set/train/IGH/result"

ALL_PUBLIC_FEATURES = [
    "all_ref_public_clone_number",
    "all_ref_public_clone_ratio",
    "all_ref_public_frequency_sum",
    "all_ref_public_set_coverage",
    "RA_specific_ref_clone_number",
    "RA_specific_ref_clone_ratio",
    "RA_specific_ref_frequency_sum",
    "RA_specific_ref_set_coverage",
    "ILD_specific_ref_clone_number",
    "ILD_specific_ref_clone_ratio",
    "ILD_specific_ref_frequency_sum",
    "ILD_specific_ref_set_coverage",
    "shared_ref_clone_number",
    "shared_ref_clone_ratio",
    "shared_ref_frequency_sum",
    "shared_ref_set_coverage",
    "ILD_RA_specific_ref_frequency_delta",
    "ILD_RA_specific_ref_frequency_log_ratio",
]

PUBLIC_FEATURE_SETS = {
    "raw_bilateral": [
        "all_ref_public_clone_ratio",
        "all_ref_public_frequency_sum",
        "shared_ref_clone_ratio",
        "shared_ref_frequency_sum",
        "RA_specific_ref_clone_ratio",
        "RA_specific_ref_frequency_sum",
        "ILD_specific_ref_clone_ratio",
        "ILD_specific_ref_frequency_sum",
    ],
    "log_ratio": [
        "all_ref_public_clone_ratio",
        "all_ref_public_frequency_sum",
        "shared_ref_clone_ratio",
        "shared_ref_frequency_sum",
        "ILD_RA_specific_ref_frequency_log_ratio",
    ],
    "all18": ALL_PUBLIC_FEATURES,
}

MODEL_LABELS = {
    "M0_clinical": "age + sex",
    "M1_static_trb_igh": "age + sex + static TRB + static IGH",
    "M2_static_trb_igh_public": (
        "age + sex + static TRB + static IGH + dynamic TRB public "
        "+ dynamic IGH public"
    ),
    "M3_static_trb_igh_public_material": (
        "age + sex + material + static TRB + static IGH "
        "+ dynamic TRB public + dynamic IGH public"
    ),
}


@dataclass(frozen=True)
class ReceptorSpec:
    receptor: str
    prefix: str
    library_id_col: str
    aa_clone_number_col: str
    clone_file_pattern: str
    public_catalog: Path
    reference_definition: Path
    clone_table_dir: Path
    cache_dir: Path
    expected_catalog_size: int
    expected_global_size: int
    expected_ra_specific_size: int
    expected_ild_specific_size: int
    expected_shared_size: int


@dataclass
class PublicReference:
    global_mask: np.ndarray
    ra_specific_mask: np.ndarray
    ild_specific_mask: np.ndarray
    shared_mask: np.ndarray
    n_reference: int
    n_ra: int
    n_ild: int
    global_threshold: int
    ra_threshold: int
    ild_threshold: int


@dataclass
class PreparedFold:
    X_train: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    valid_sample_ids: List[str]
    feature_names: List[str]
    preprocess_rows: List[Dict[str, object]]


def parse_float_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one leakage-controlled paired TRB+IGH outer-fold "
            "Elastic Net trial with receptor-specific LOO public features."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--base-matrix", default=str(DEFAULT_BASE))
    parser.add_argument("--feature-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--outer-assignments", default=str(DEFAULT_OUTER))
    parser.add_argument("--inner-assignments", default=str(DEFAULT_INNER))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    parser.add_argument(
        "--trb-public-catalog",
        default=str(TRB_ROOT / "03_public_features/03_public_aa_catalog.csv.gz"),
    )
    parser.add_argument(
        "--igh-public-catalog",
        default=str(IGH_ROOT / "03_public_features/03_public_aa_catalog.csv.gz"),
    )
    parser.add_argument(
        "--trb-reference-definition",
        default=str(TRB_ROOT / "03_public_features/03_reference_definition.json"),
    )
    parser.add_argument(
        "--igh-reference-definition",
        default=str(IGH_ROOT / "03_public_features/03_reference_definition.json"),
    )
    parser.add_argument(
        "--trb-clone-table-dir",
        default=str(TRB_ROOT / "01_AA_clone_table"),
    )
    parser.add_argument(
        "--igh-clone-table-dir",
        default=str(IGH_ROOT / "01_AA_clone_table"),
    )
    parser.add_argument(
        "--trb-cache-dir",
        default=str(ROOT / "set/train/result/05_modeling/cache_loo/TRB"),
    )
    parser.add_argument(
        "--igh-cache-dir",
        default=str(ROOT / "set/train/result/05_modeling/cache_loo/IGH"),
    )

    parser.add_argument("--outer-repeat", type=int, default=1)
    parser.add_argument("--outer-fold", type=int, default=1)
    parser.add_argument("--inner-folds", type=int, default=5)

    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--label-col", default="cohort")
    parser.add_argument("--positive-label", default="ILD")
    parser.add_argument("--age-col", default="age")
    parser.add_argument("--sex-col", default="sex")
    parser.add_argument("--material-col", default="material")
    parser.add_argument("--trb-library-id-col", default="trb_libraryid")
    parser.add_argument("--igh-library-id-col", default="igh_libraryid")
    parser.add_argument("--trb-aa-clone-number-col", default="trb_aa_clone_number")
    parser.add_argument("--igh-aa-clone-number-col", default="igh_aa_clone_number")

    parser.add_argument(
        "--trb-clone-file-pattern",
        default="{library_id}_TRB_CDR3_AA_clone_table.csv",
    )
    parser.add_argument(
        "--igh-clone-file-pattern",
        default="{library_id}_IGH-without-DJ_CDR3_AA_clone_table.csv",
    )
    parser.add_argument("--sequence-col", default="cdr3_aa")
    parser.add_argument("--frequency-col", default="frequency_norm")

    parser.add_argument("--global-min-prevalence", type=float, default=0.02)
    parser.add_argument("--global-min-count", type=int, default=3)
    parser.add_argument("--group-min-prevalence", type=float, default=0.05)
    parser.add_argument("--group-min-count", type=int, default=3)
    parser.add_argument("--specific-prevalence-delta", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=1e-8)

    parser.add_argument(
        "--public-feature-set",
        choices=sorted(PUBLIC_FEATURE_SETS),
        default="raw_bilateral",
    )
    parser.add_argument(
        "--l1-ratios",
        type=parse_float_list,
        default=parse_float_list("0.1,0.5,0.9"),
    )
    parser.add_argument(
        "--lambdas",
        type=parse_float_list,
        default=parse_float_list("0.01,0.03,0.1,0.3,1,3,10,30"),
    )
    parser.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default="balanced",
    )
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--zero-sd-tolerance", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=20260711)

    parser.add_argument("--expected-samples", type=int, default=118)

    parser.add_argument("--expected-trb-catalog-size", type=int, default=965027)
    parser.add_argument("--expected-trb-global-size", type=int, default=385472)
    parser.add_argument("--expected-trb-ra-specific-size", type=int, default=44662)
    parser.add_argument("--expected-trb-ild-specific-size", type=int, default=27563)
    parser.add_argument("--expected-trb-shared-size", type=int, default=36119)

    parser.add_argument("--expected-igh-catalog-size", type=int, default=48594)
    parser.add_argument("--expected-igh-global-size", type=int, default=6786)
    parser.add_argument("--expected-igh-ra-specific-size", type=int, default=320)
    parser.add_argument("--expected-igh-ild-specific-size", type=int, default=571)
    parser.add_argument("--expected-igh-shared-size", type=int, default=274)

    parser.add_argument("--skip-full-reference-validation", action="store_true")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.outer_repeat < 1 or args.outer_fold < 1:
        parser.error("--outer-repeat and --outer-fold must be >= 1.")
    if args.inner_folds < 2:
        parser.error("--inner-folds must be >= 2.")
    for value in args.l1_ratios:
        if not (0 < value <= 1):
            parser.error("Every l1 ratio must be in (0, 1].")
    for value in args.lambdas:
        if value <= 0:
            parser.error("Every lambda must be > 0.")
    if args.epsilon <= 0:
        parser.error("--epsilon must be > 0.")
    return args


def receptor_specs(args: argparse.Namespace) -> Dict[str, ReceptorSpec]:
    return {
        "TRB": ReceptorSpec(
            receptor="TRB",
            prefix="trb_",
            library_id_col=args.trb_library_id_col,
            aa_clone_number_col=args.trb_aa_clone_number_col,
            clone_file_pattern=args.trb_clone_file_pattern,
            public_catalog=Path(args.trb_public_catalog).expanduser().resolve(),
            reference_definition=Path(
                args.trb_reference_definition
            ).expanduser().resolve(),
            clone_table_dir=Path(
                args.trb_clone_table_dir
            ).expanduser().resolve(),
            cache_dir=Path(args.trb_cache_dir).expanduser().resolve(),
            expected_catalog_size=args.expected_trb_catalog_size,
            expected_global_size=args.expected_trb_global_size,
            expected_ra_specific_size=args.expected_trb_ra_specific_size,
            expected_ild_specific_size=args.expected_trb_ild_specific_size,
            expected_shared_size=args.expected_trb_shared_size,
        ),
        "IGH": ReceptorSpec(
            receptor="IGH",
            prefix="igh_",
            library_id_col=args.igh_library_id_col,
            aa_clone_number_col=args.igh_aa_clone_number_col,
            clone_file_pattern=args.igh_clone_file_pattern,
            public_catalog=Path(args.igh_public_catalog).expanduser().resolve(),
            reference_definition=Path(
                args.igh_reference_definition
            ).expanduser().resolve(),
            clone_table_dir=Path(
                args.igh_clone_table_dir
            ).expanduser().resolve(),
            cache_dir=Path(args.igh_cache_dir).expanduser().resolve(),
            expected_catalog_size=args.expected_igh_catalog_size,
            expected_global_size=args.expected_igh_global_size,
            expected_ra_specific_size=args.expected_igh_ra_specific_size,
            expected_ild_specific_size=args.expected_igh_ild_specific_size,
            expected_shared_size=args.expected_igh_shared_size,
        ),
    }


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(base_seed: int, *parts: int) -> int:
    text = ":".join(str(x) for x in (base_seed, *parts))
    return int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:8],
        "little",
    ) % (2**32 - 1)


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def ensure_no_missing(df: pd.DataFrame, label: str) -> None:
    n = int(df.isna().sum().sum())
    if n:
        cols = df.isna().sum()
        bad = cols[cols > 0].sort_values(ascending=False).head(20)
        raise ValueError(f"{label} has {n} missing values: {bad.to_dict()}")



def load_and_validate_reference_definition(
    args: argparse.Namespace,
    spec: ReceptorSpec,
) -> Tuple[Path, Dict[str, object]]:
    path = spec.reference_definition
    ensure_file(path, f"{spec.receptor} reference definition")
    definition = json.loads(path.read_text(encoding="utf-8"))

    receptor = str(definition.get("receptor", "")).upper()
    if receptor != spec.receptor:
        raise ValueError(
            f"{spec.receptor} reference definition receptor="
            f"{receptor or '<missing>'}."
        )

    expected_suffix = spec.clone_file_pattern.format(
        library_id="", sample_id="", patient_id=""
    )
    observed_suffix = str(definition.get("aa_input_suffix", ""))
    if observed_suffix != expected_suffix:
        raise ValueError(
            f"{spec.receptor} reference AA suffix mismatch: "
            f"definition={observed_suffix}, expected={expected_suffix}."
        )

    scheme = definition.get("scheme", {})
    if str(scheme.get("name", "")) != "main":
        raise ValueError(
            f"{spec.receptor} reference scheme={scheme.get('name')}; "
            "expected main."
        )

    expected_scheme = {
        "global_min_prevalence": float(args.global_min_prevalence),
        "global_min_count": int(args.global_min_count),
        "group_min_prevalence": float(args.group_min_prevalence),
        "group_min_count": int(args.group_min_count),
        "specific_prevalence_delta": float(
            args.specific_prevalence_delta
        ),
    }
    for key, expected in expected_scheme.items():
        if key not in scheme:
            raise ValueError(
                f"{spec.receptor} reference scheme missing {key}."
            )
        observed = float(scheme[key])
        if not math.isclose(
            observed, float(expected), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"{spec.receptor} reference {key}={observed}; "
                f"command expects {expected}."
            )

    n_training = int(definition.get("n_training_samples", -1))
    if args.expected_samples > 0 and n_training != args.expected_samples:
        raise ValueError(
            f"{spec.receptor} reference training samples={n_training}; "
            f"expected {args.expected_samples}."
        )

    candidate_count = int(
        definition.get("candidate_sequence_count", -1)
    )
    if (
        spec.expected_catalog_size > 0
        and candidate_count != spec.expected_catalog_size
    ):
        raise ValueError(
            f"{spec.receptor} candidate count={candidate_count}; "
            f"expected {spec.expected_catalog_size}."
        )

    sizes = definition.get("reference_set_sizes", {})
    checks = {
        "global_public": spec.expected_global_size,
        "RA_specific_ref": spec.expected_ra_specific_size,
        "ILD_specific_ref": spec.expected_ild_specific_size,
        "between_group_shared": spec.expected_shared_size,
    }
    for key, expected in checks.items():
        if expected > 0 and int(sizes.get(key, -1)) != expected:
            raise ValueError(
                f"{spec.receptor} reference {key}={sizes.get(key)}; "
                f"expected {expected}."
            )
    return path, definition


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def read_core_inputs(args: argparse.Namespace):
    base_path = Path(args.base_matrix).expanduser().resolve()
    manifest_path = Path(args.feature_manifest).expanduser().resolve()
    outer_path = Path(args.outer_assignments).expanduser().resolve()
    inner_path = Path(args.inner_assignments).expanduser().resolve()

    for path, label in [
        (base_path, "paired base matrix"),
        (manifest_path, "paired feature manifest"),
        (outer_path, "outer assignments"),
        (inner_path, "inner assignments"),
    ]:
        ensure_file(path, label)

    base = pd.read_csv(base_path)
    manifest = pd.read_csv(manifest_path)
    outer = pd.read_csv(outer_path)
    inner = pd.read_csv(inner_path)

    required_base = {
        args.sample_id_col,
        args.label_col,
        args.age_col,
        args.sex_col,
        args.material_col,
        args.trb_library_id_col,
        args.igh_library_id_col,
        args.trb_aa_clone_number_col,
        args.igh_aa_clone_number_col,
    }
    missing = required_base - set(base.columns)
    if missing:
        raise ValueError(
            f"Paired base matrix missing columns: {sorted(missing)}"
        )
    if base[args.sample_id_col].duplicated().any():
        raise ValueError("Paired base matrix has duplicated patient IDs.")
    ensure_no_missing(base, "paired base matrix")

    for column in [
        args.sample_id_col,
        args.trb_library_id_col,
        args.igh_library_id_col,
    ]:
        base[column] = base[column].astype(str).str.strip()
        if base[column].eq("").any():
            raise ValueError(f"Empty values in base matrix column {column}.")
        if base[column].duplicated().any():
            raise ValueError(
                f"Duplicated values in base matrix column {column}."
            )

    base[args.label_col] = (
        base[args.label_col].astype(str).str.upper()
    )
    if set(base[args.label_col]) != {"RA", "ILD"}:
        raise ValueError("Paired base labels must be exactly RA and ILD.")

    if args.expected_samples > 0 and len(base) != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} patients, observed {len(base)}."
        )

    required_manifest = {
        "analysis_view",
        "receptor",
        "feature_name",
        "present_in_train_base",
        "feature_group",
        "feature_role",
        "reference_drop",
    }
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise ValueError(
            f"Feature manifest missing columns: {sorted(missing_manifest)}"
        )
    views = set(manifest["analysis_view"].astype(str))
    if views != {"TRB_IGH"}:
        raise ValueError(
            f"Expected a TRB_IGH manifest; observed analysis_view={sorted(views)}."
        )

    required_outer = {
        "outer_repeat", "outer_fold", args.sample_id_col, args.label_col
    }
    required_inner = {
        "outer_repeat", "outer_fold", "inner_fold",
        args.sample_id_col, args.label_col
    }
    if not required_outer.issubset(outer.columns):
        raise ValueError(
            f"Outer assignments missing: "
            f"{sorted(required_outer - set(outer.columns))}"
        )
    if not required_inner.issubset(inner.columns):
        raise ValueError(
            f"Inner assignments missing: "
            f"{sorted(required_inner - set(inner.columns))}"
        )

    outer_task = outer[
        outer["outer_repeat"] == args.outer_repeat
    ].copy()
    if len(outer_task) != len(base):
        raise ValueError(
            f"Outer repeat {args.outer_repeat} has {len(outer_task)} rows; "
            f"expected {len(base)}."
        )
    outer_task[args.sample_id_col] = (
        outer_task[args.sample_id_col].astype(str)
    )
    if set(outer_task[args.sample_id_col]) != set(
        base[args.sample_id_col]
    ):
        raise ValueError(
            "Outer assignment patient IDs do not match paired base matrix."
        )
    if args.outer_fold not in set(
        pd.to_numeric(outer_task["outer_fold"], errors="raise").astype(int)
    ):
        raise ValueError(
            f"Outer fold {args.outer_fold} is absent from repeat "
            f"{args.outer_repeat}."
        )

    # Cross-check frozen patient-library mapping when the columns are available.
    for column in [args.trb_library_id_col, args.igh_library_id_col]:
        if column in outer_task.columns:
            check = (
                base[[args.sample_id_col, column]]
                .merge(
                    outer_task[[args.sample_id_col, column]],
                    on=args.sample_id_col,
                    validate="one_to_one",
                    suffixes=("_base", "_split"),
                )
            )
            if not (
                check[f"{column}_base"].astype(str).to_numpy()
                == check[f"{column}_split"].astype(str).to_numpy()
            ).all():
                raise ValueError(
                    f"Frozen CV mapping differs from base matrix for {column}."
                )

    outer_train_ids = set(
        outer_task.loc[
            outer_task["outer_fold"] != args.outer_fold,
            args.sample_id_col,
        ].astype(str)
    )
    outer_valid_ids = set(
        outer_task.loc[
            outer_task["outer_fold"] == args.outer_fold,
            args.sample_id_col,
        ].astype(str)
    )
    if outer_train_ids & outer_valid_ids:
        raise ValueError("Outer training and validation patients overlap.")
    if outer_train_ids | outer_valid_ids != set(base[args.sample_id_col]):
        raise ValueError("Outer task does not cover all patients.")

    inner_task = inner[
        (inner["outer_repeat"] == args.outer_repeat)
        & (inner["outer_fold"] == args.outer_fold)
    ].copy()
    inner_task[args.sample_id_col] = (
        inner_task[args.sample_id_col].astype(str)
    )
    if set(inner_task[args.sample_id_col]) != outer_train_ids:
        raise ValueError(
            "Inner assignment patients are not exactly the selected "
            "outer-training patients."
        )
    if inner_task[args.sample_id_col].duplicated().any():
        raise ValueError("Inner assignment contains duplicated patients.")
    if set(pd.to_numeric(
        inner_task["inner_fold"], errors="raise"
    ).astype(int)) != set(range(1, args.inner_folds + 1)):
        raise ValueError("Inner assignment does not contain all requested folds.")
    if set(inner_task[args.sample_id_col]) & outer_valid_ids:
        raise ValueError(
            "Outer-validation patients leaked into inner assignments."
        )

    base = base.set_index(args.sample_id_col, drop=False)
    outer_train_order = [
        patient for patient in base.index.tolist()
        if patient in outer_train_ids
    ]
    outer_valid_order = [
        patient for patient in base.index.tolist()
        if patient in outer_valid_ids
    ]

    return (
        base,
        manifest,
        outer_task,
        inner_task,
        outer_train_order,
        outer_valid_order,
        {
            "base": base_path,
            "manifest": manifest_path,
            "outer": outer_path,
            "inner": inner_path,
        },
    )


def static_joint_features_from_manifest(
    manifest: pd.DataFrame,
    base: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    present = as_bool(manifest["present_in_train_base"])
    reference_drop = as_bool(manifest["reference_drop"])
    role = manifest["feature_role"].astype(str)
    receptor = manifest["receptor"].astype(str).str.upper()

    selected: Dict[str, List[str]] = {}
    for rec, prefix in [("TRB", "trb_"), ("IGH", "igh_")]:
        features = manifest.loc[
            present
            & receptor.eq(rec)
            & role.eq("candidate_predictor")
            & (~reference_drop),
            "feature_name",
        ].astype(str).tolist()
        features = [feature for feature in features if feature in base.columns]
        if not features:
            raise ValueError(
                f"No static {rec} candidate predictors selected from manifest."
            )
        if len(features) != len(set(features)):
            raise ValueError(f"Duplicated static {rec} feature names.")
        bad_prefix = [
            feature for feature in features
            if not feature.startswith(prefix)
        ]
        if bad_prefix:
            raise ValueError(
                f"{rec} manifest features lack {prefix} prefix: "
                f"{bad_prefix[:10]}"
            )
        selected[rec] = features

    overlap = set(selected["TRB"]) & set(selected["IGH"])
    if overlap:
        raise ValueError(
            f"TRB/IGH static feature collision: {sorted(overlap)[:10]}"
        )
    return selected["TRB"], selected["IGH"]



def cache_paths(cache_dir: Path) -> Dict[str, Path]:
    return {
        "presence": cache_dir / "05_train_public_presence.npz",
        "frequency": cache_dir / "05_train_public_frequency.npz",
        "metadata": cache_dir / "05_train_public_sparse_cache.json",
    }


def format_clone_path(
    spec: ReceptorSpec,
    patient_id: str,
    library_id: str,
) -> Path:
    filename = spec.clone_file_pattern.format(
        library_id=library_id,
        sample_id=library_id,
        patient_id=patient_id,
    )
    return spec.clone_table_dir / filename


def source_file_signature(
    patient_ids: Sequence[str],
    library_ids: Sequence[str],
    spec: ReceptorSpec,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for patient_id, library_id in zip(patient_ids, library_ids):
        path = format_clone_path(spec, patient_id, library_id)
        ensure_file(path, f"{spec.receptor} clone table for {patient_id}")
        stat = path.stat()
        rows.append(
            {
                "patient_id": patient_id,
                "library_id": library_id,
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return rows


def cache_matches(
    metadata_path: Path,
    patient_ids: Sequence[str],
    library_ids: Sequence[str],
    spec: ReceptorSpec,
    clone_signatures: Sequence[Mapping[str, object]],
    reference_definition_sha256: str,
) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        meta.get("receptor") == spec.receptor
        and meta.get("patient_ids") == list(patient_ids)
        and meta.get("library_ids") == list(library_ids)
        and meta.get("catalog_sha256") == sha256sum(spec.public_catalog)
        and meta.get("clone_file_pattern") == spec.clone_file_pattern
        and meta.get("reference_definition_sha256")
        == reference_definition_sha256
        and meta.get("clone_files") == list(clone_signatures)
    )


def build_or_load_sparse_cache(
    args: argparse.Namespace,
    base: pd.DataFrame,
    spec: ReceptorSpec,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, Dict[str, object]]:
    ensure_file(spec.public_catalog, f"{spec.receptor} public AA catalog")
    ensure_file(
        spec.reference_definition,
        f"{spec.receptor} reference definition",
    )
    if not spec.clone_table_dir.is_dir():
        raise FileNotFoundError(
            f"{spec.receptor} clone table directory not found: "
            f"{spec.clone_table_dir}"
        )
    spec.cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(spec.cache_dir)

    patient_ids = base.index.astype(str).tolist()
    library_ids = (
        base[spec.library_id_col].astype(str).tolist()
    )
    clone_signatures = source_file_signature(
        patient_ids, library_ids, spec
    )
    definition_sha = sha256sum(spec.reference_definition)

    valid_cache = (
        not args.force_rebuild_cache
        and paths["presence"].is_file()
        and paths["frequency"].is_file()
        and cache_matches(
            paths["metadata"],
            patient_ids,
            library_ids,
            spec,
            clone_signatures,
            definition_sha,
        )
    )

    if valid_cache:
        presence = sparse.load_npz(paths["presence"]).tocsr()
        frequency = sparse.load_npz(paths["frequency"]).tocsr()
        meta = json.loads(
            paths["metadata"].read_text(encoding="utf-8")
        )
        if presence.shape != frequency.shape:
            raise ValueError(
                f"Cached {spec.receptor} presence/frequency shapes differ."
            )
        if presence.shape[0] != len(patient_ids):
            raise ValueError(
                f"Cached {spec.receptor} row count differs from base matrix."
            )
        print(
            f"[{spec.receptor} sparse cache] reuse "
            f"shape={presence.shape}, nnz={presence.nnz:,}"
        )
        return presence, frequency, meta

    catalog = pd.read_csv(
        spec.public_catalog, usecols=[args.sequence_col]
    )
    catalog[args.sequence_col] = (
        catalog[args.sequence_col].astype(str)
    )
    if catalog[args.sequence_col].duplicated().any():
        raise ValueError(
            f"{spec.receptor} public AA catalog contains duplicate sequences."
        )
    if (
        spec.expected_catalog_size > 0
        and len(catalog) != spec.expected_catalog_size
    ):
        raise ValueError(
            f"Expected {spec.receptor} catalog size "
            f"{spec.expected_catalog_size}, observed {len(catalog)}."
        )

    sequence_to_col = {
        sequence: index
        for index, sequence in enumerate(
            catalog[args.sequence_col].tolist()
        )
    }

    row_parts: List[np.ndarray] = []
    col_parts: List[np.ndarray] = []
    freq_parts: List[np.ndarray] = []
    observed_clone_counts: Dict[str, int] = {}

    for row_index, (patient_id, library_id) in enumerate(
        zip(patient_ids, library_ids)
    ):
        path = format_clone_path(spec, patient_id, library_id)
        clone = pd.read_csv(
            path,
            usecols=[args.sequence_col, args.frequency_col],
        )
        if clone[args.sequence_col].duplicated().any():
            raise ValueError(
                f"{spec.receptor} {library_id} clone table has "
                "duplicated AA sequences."
            )
        clone[args.sequence_col] = (
            clone[args.sequence_col].astype(str)
        )
        clone[args.frequency_col] = pd.to_numeric(
            clone[args.frequency_col], errors="raise"
        )
        if (clone[args.frequency_col] < 0).any():
            raise ValueError(
                f"{spec.receptor} {library_id} has negative frequency_norm."
            )
        observed_clone_counts[patient_id] = len(clone)

        expected_aa = int(round(float(
            base.loc[patient_id, spec.aa_clone_number_col]
        )))
        if expected_aa != len(clone):
            raise ValueError(
                f"{spec.receptor} {patient_id}/{library_id}: "
                f"{spec.aa_clone_number_col}={expected_aa}, "
                f"01 rows={len(clone)}."
            )

        mapped = clone[args.sequence_col].map(sequence_to_col)
        keep = mapped.notna().to_numpy()
        cols = mapped[keep].astype(np.int64).to_numpy()
        values = clone.loc[
            keep, args.frequency_col
        ].to_numpy(dtype=np.float64)

        row_parts.append(
            np.full(len(cols), row_index, dtype=np.int32)
        )
        col_parts.append(cols)
        freq_parts.append(values)

        if (
            row_index == 0
            or (row_index + 1) % 20 == 0
            or row_index + 1 == len(patient_ids)
        ):
            print(
                f"[{spec.receptor} sparse cache] processed "
                f"{row_index + 1}/{len(patient_ids)} patients"
            )

    rows = (
        np.concatenate(row_parts)
        if row_parts else np.array([], dtype=np.int32)
    )
    cols = (
        np.concatenate(col_parts)
        if col_parts else np.array([], dtype=np.int64)
    )
    frequency_values = (
        np.concatenate(freq_parts)
        if freq_parts else np.array([], dtype=np.float64)
    )

    shape = (len(patient_ids), len(catalog))
    presence = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=shape,
    )
    frequency = sparse.csr_matrix(
        (frequency_values, (rows, cols)),
        shape=shape,
    )
    presence.sum_duplicates()
    frequency.sum_duplicates()

    sparse.save_npz(paths["presence"], presence, compressed=True)
    sparse.save_npz(paths["frequency"], frequency, compressed=True)

    meta = {
        "script_version": SCRIPT_VERSION,
        "receptor": spec.receptor,
        "library_id_col": spec.library_id_col,
        "aa_clone_number_col": spec.aa_clone_number_col,
        "clone_file_pattern": spec.clone_file_pattern,
        "reference_definition_sha256": definition_sha,
        "patient_ids": patient_ids,
        "library_ids": library_ids,
        "catalog_path": str(spec.public_catalog),
        "catalog_sha256": sha256sum(spec.public_catalog),
        "catalog_size": len(catalog),
        "matrix_shape": list(shape),
        "presence_nnz": int(presence.nnz),
        "frequency_nnz": int(frequency.nnz),
        "clone_files": clone_signatures,
        "observed_clone_counts": observed_clone_counts,
    }
    paths["metadata"].write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return presence, frequency, meta


def effective_threshold(n: int, prevalence: float, minimum: int) -> int:
    return max(minimum, int(math.ceil(prevalence * n)))


def build_public_reference(
    presence: sparse.csr_matrix,
    reference_rows: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> PublicReference:
    reference_rows = np.asarray(reference_rows, dtype=int)
    ref_labels = labels[reference_rows]
    ra_rows = reference_rows[ref_labels == "RA"]
    ild_rows = reference_rows[ref_labels == "ILD"]

    if len(ra_rows) < args.group_min_count:
        raise ValueError("Too few RA reference samples.")
    if len(ild_rows) < args.group_min_count:
        raise ValueError("Too few ILD reference samples.")

    total_counts = np.asarray(
        presence[reference_rows].sum(axis=0)
    ).ravel()
    ra_counts = np.asarray(presence[ra_rows].sum(axis=0)).ravel()
    ild_counts = np.asarray(presence[ild_rows].sum(axis=0)).ravel()

    global_threshold = effective_threshold(
        len(reference_rows),
        args.global_min_prevalence,
        args.global_min_count,
    )
    ra_threshold = effective_threshold(
        len(ra_rows),
        args.group_min_prevalence,
        args.group_min_count,
    )
    ild_threshold = effective_threshold(
        len(ild_rows),
        args.group_min_prevalence,
        args.group_min_count,
    )

    global_mask = total_counts >= global_threshold
    ra_group = ra_counts >= ra_threshold
    ild_group = ild_counts >= ild_threshold

    ra_prevalence = ra_counts / len(ra_rows)
    ild_prevalence = ild_counts / len(ild_rows)

    ra_specific_mask = (
        ra_group
        & ((ra_prevalence - ild_prevalence) >= args.specific_prevalence_delta)
    )
    ild_specific_mask = (
        ild_group
        & ((ild_prevalence - ra_prevalence) >= args.specific_prevalence_delta)
    )
    shared_mask = (
        ra_group
        & ild_group
        & ~ra_specific_mask
        & ~ild_specific_mask
    )

    return PublicReference(
        global_mask=global_mask,
        ra_specific_mask=ra_specific_mask,
        ild_specific_mask=ild_specific_mask,
        shared_mask=shared_mask,
        n_reference=len(reference_rows),
        n_ra=len(ra_rows),
        n_ild=len(ild_rows),
        global_threshold=global_threshold,
        ra_threshold=ra_threshold,
        ild_threshold=ild_threshold,
    )


def apply_public_reference(
    presence: sparse.csr_matrix,
    frequency: sparse.csr_matrix,
    target_rows: np.ndarray,
    aa_clone_numbers: np.ndarray,
    ref: PublicReference,
    epsilon: float,
) -> pd.DataFrame:
    target_rows = np.asarray(target_rows, dtype=int)
    target_presence = presence[target_rows]
    target_frequency = frequency[target_rows]

    result: Dict[str, np.ndarray] = {}

    masks = {
        "all_ref_public": ref.global_mask,
        "RA_specific_ref": ref.ra_specific_mask,
        "ILD_specific_ref": ref.ild_specific_mask,
        "shared_ref": ref.shared_mask,
    }

    for prefix, mask in masks.items():
        set_size = int(mask.sum())
        clone_number = np.asarray(
            target_presence[:, mask].sum(axis=1)
        ).ravel().astype(float)
        frequency_sum = np.asarray(
            target_frequency[:, mask].sum(axis=1)
        ).ravel().astype(float)

        result[f"{prefix}_clone_number"] = clone_number
        result[f"{prefix}_clone_ratio"] = np.divide(
            clone_number,
            aa_clone_numbers[target_rows],
            out=np.zeros_like(clone_number, dtype=float),
            where=aa_clone_numbers[target_rows] > 0,
        )
        result[f"{prefix}_frequency_sum"] = frequency_sum
        result[f"{prefix}_set_coverage"] = (
            clone_number / set_size if set_size > 0 else np.zeros_like(clone_number)
        )

    ild_sum = result["ILD_specific_ref_frequency_sum"]
    ra_sum = result["RA_specific_ref_frequency_sum"]
    result["ILD_RA_specific_ref_frequency_delta"] = ild_sum - ra_sum
    result["ILD_RA_specific_ref_frequency_log_ratio"] = np.log(
        (ild_sum + epsilon) / (ra_sum + epsilon)
    )

    out = pd.DataFrame(result)
    return out[ALL_PUBLIC_FEATURES]


def build_public_reference_from_counts(
    ra_counts: np.ndarray,
    ild_counts: np.ndarray,
    n_ra_reference: int,
    n_ild_reference: int,
    args: argparse.Namespace,
) -> PublicReference:
    """Build step-03-compatible masks from precomputed sample counts."""
    if n_ra_reference < args.group_min_count:
        raise ValueError("Too few RA reference samples.")
    if n_ild_reference < args.group_min_count:
        raise ValueError("Too few ILD reference samples.")

    ra_counts = np.asarray(ra_counts, dtype=np.int32)
    ild_counts = np.asarray(ild_counts, dtype=np.int32)
    if ra_counts.shape != ild_counts.shape:
        raise ValueError("RA and ILD count arrays have different shapes.")

    n_reference = n_ra_reference + n_ild_reference
    global_threshold = effective_threshold(
        n_reference,
        args.global_min_prevalence,
        args.global_min_count,
    )
    ra_threshold = effective_threshold(
        n_ra_reference,
        args.group_min_prevalence,
        args.group_min_count,
    )
    ild_threshold = effective_threshold(
        n_ild_reference,
        args.group_min_prevalence,
        args.group_min_count,
    )

    total_counts = ra_counts + ild_counts
    ra_prevalence = ra_counts.astype(np.float64) / n_ra_reference
    ild_prevalence = ild_counts.astype(np.float64) / n_ild_reference

    global_mask = total_counts >= global_threshold
    ra_group = ra_counts >= ra_threshold
    ild_group = ild_counts >= ild_threshold

    ra_specific_mask = (
        ra_group
        & ((ra_prevalence - ild_prevalence) >= args.specific_prevalence_delta)
    )
    ild_specific_mask = (
        ild_group
        & ((ild_prevalence - ra_prevalence) >= args.specific_prevalence_delta)
    )
    if np.any(ra_specific_mask & ild_specific_mask):
        raise RuntimeError("RA-specific and ILD-specific masks overlap.")

    shared_mask = (
        ra_group
        & ild_group
        & ~ra_specific_mask
        & ~ild_specific_mask
    )

    return PublicReference(
        global_mask=global_mask,
        ra_specific_mask=ra_specific_mask,
        ild_specific_mask=ild_specific_mask,
        shared_mask=shared_mask,
        n_reference=n_reference,
        n_ra=n_ra_reference,
        n_ild=n_ild_reference,
        global_threshold=global_threshold,
        ra_threshold=ra_threshold,
        ild_threshold=ild_threshold,
    )


def _single_sample_public_from_loo_templates(
    presence_row: sparse.csr_matrix,
    frequency_row: sparse.csr_matrix,
    aa_clone_number: float,
    absent_ref: PublicReference,
    present_ref: PublicReference,
    epsilon: float,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Calculate exact LOO features without rebuilding a million-element mask for
    every sample.

    For sequences absent from the held-out sample, the LOO count equals the
    partition total count. For sequences present in the held-out sample, one is
    subtracted from the held-out sample's cohort count. Thus only two mask
    templates are needed per held-out cohort.
    """
    presence_row = presence_row.tocsr(copy=True)
    frequency_row = frequency_row.tocsr(copy=True)
    presence_row.sort_indices()
    frequency_row.sort_indices()

    cols = presence_row.indices
    freq_cols = frequency_row.indices
    if not np.array_equal(cols, freq_cols):
        # Defensive fallback if sparse structures ever differ.
        freq_values = np.asarray(frequency_row[:, cols].toarray()).ravel()
    else:
        freq_values = frequency_row.data.astype(np.float64, copy=False)

    result: Dict[str, float] = {}
    set_sizes: Dict[str, int] = {}
    masks = {
        "all_ref_public": (
            absent_ref.global_mask,
            present_ref.global_mask,
        ),
        "RA_specific_ref": (
            absent_ref.ra_specific_mask,
            present_ref.ra_specific_mask,
        ),
        "ILD_specific_ref": (
            absent_ref.ild_specific_mask,
            present_ref.ild_specific_mask,
        ),
        "shared_ref": (
            absent_ref.shared_mask,
            present_ref.shared_mask,
        ),
    }

    for prefix, (absent_mask, present_mask) in masks.items():
        present_status = present_mask[cols]
        absent_status_at_sample = absent_mask[cols]

        # Start from the valid absent-sample mask for all sequences, then replace
        # statuses only at sequences actually present in this held-out sample.
        set_size = int(np.count_nonzero(absent_mask)) + int(
            np.sum(
                present_status.astype(np.int8)
                - absent_status_at_sample.astype(np.int8)
            )
        )
        if set_size < 0:
            raise RuntimeError(f"Negative LOO set size for {prefix}.")

        clone_number = float(np.count_nonzero(present_status))
        frequency_sum = float(np.sum(freq_values[present_status]))

        result[f"{prefix}_clone_number"] = clone_number
        result[f"{prefix}_clone_ratio"] = (
            clone_number / aa_clone_number if aa_clone_number > 0 else 0.0
        )
        result[f"{prefix}_frequency_sum"] = frequency_sum
        result[f"{prefix}_set_coverage"] = (
            clone_number / set_size if set_size > 0 else 0.0
        )
        set_sizes[prefix] = set_size

    ild_sum = result["ILD_specific_ref_frequency_sum"]
    ra_sum = result["RA_specific_ref_frequency_sum"]
    result["ILD_RA_specific_ref_frequency_delta"] = ild_sum - ra_sum
    result["ILD_RA_specific_ref_frequency_log_ratio"] = float(
        np.log((ild_sum + epsilon) / (ra_sum + epsilon))
    )
    return result, set_sizes


def leave_one_out_public_features(
    presence: sparse.csr_matrix,
    frequency: sparse.csr_matrix,
    sample_rows: np.ndarray,
    labels: np.ndarray,
    aa_clone_numbers: np.ndarray,
    args: argparse.Namespace,
    context: str,
    inner_fold: Optional[int],
    reference_rows_log: List[Dict[str, object]],
    loo_assignment_log: List[Dict[str, object]],
) -> pd.DataFrame:
    """
    Generate exact, efficient leave-one-out public features for model-fitting
    samples. Every sample uses all other samples in the current training
    partition as its reference.
    """
    sample_rows = np.asarray(sample_rows, dtype=int)
    local_labels = labels[sample_rows]
    ra_rows = sample_rows[local_labels == "RA"]
    ild_rows = sample_rows[local_labels == "ILD"]

    if len(ra_rows) <= args.group_min_count:
        raise ValueError("Too few RA samples for leave-one-out reference.")
    if len(ild_rows) <= args.group_min_count:
        raise ValueError("Too few ILD samples for leave-one-out reference.")

    ra_counts = np.asarray(
        presence[ra_rows].sum(axis=0)
    ).ravel().astype(np.int32)
    ild_counts = np.asarray(
        presence[ild_rows].sum(axis=0)
    ).ravel().astype(np.int32)

    templates: Dict[str, Tuple[PublicReference, PublicReference]] = {}
    for held_label in ("RA", "ILD"):
        n_ra_loo = len(ra_rows) - (1 if held_label == "RA" else 0)
        n_ild_loo = len(ild_rows) - (1 if held_label == "ILD" else 0)

        # Absent template: the held-out sample does not contain the sequence.
        absent_ref = build_public_reference_from_counts(
            ra_counts=ra_counts,
            ild_counts=ild_counts,
            n_ra_reference=n_ra_loo,
            n_ild_reference=n_ild_loo,
            args=args,
        )

        # Present template: subtract the held-out sample's contribution from its
        # cohort count. Clipping only affects sequences absent from that sample;
        # those entries are never used from the present template.
        if held_label == "RA":
            present_ra = np.maximum(ra_counts - 1, 0)
            present_ild = ild_counts
        else:
            present_ra = ra_counts
            present_ild = np.maximum(ild_counts - 1, 0)

        present_ref = build_public_reference_from_counts(
            ra_counts=present_ra,
            ild_counts=present_ild,
            n_ra_reference=n_ra_loo,
            n_ild_reference=n_ild_loo,
            args=args,
        )
        templates[held_label] = (absent_ref, present_ref)

    output_rows: List[Dict[str, float]] = []
    output_index: List[int] = []

    for row in sample_rows:
        held_label = str(labels[row])
        absent_ref, present_ref = templates[held_label]
        values, set_sizes = _single_sample_public_from_loo_templates(
            presence_row=presence.getrow(row),
            frequency_row=frequency.getrow(row),
            aa_clone_number=float(aa_clone_numbers[row]),
            absent_ref=absent_ref,
            present_ref=present_ref,
            epsilon=args.epsilon,
        )
        output_rows.append(values)
        output_index.append(int(row))

        reference_rows_log.append({
            "context": context,
            "generation_method": "leave_one_out",
            "outer_repeat": args.outer_repeat,
            "outer_fold": args.outer_fold,
            "inner_fold": inner_fold if inner_fold is not None else "",
            "held_out_row_index": int(row),
            "held_out_label": held_label,
            "n_reference": absent_ref.n_reference,
            "n_RA_reference": absent_ref.n_ra,
            "n_ILD_reference": absent_ref.n_ild,
            "global_threshold": absent_ref.global_threshold,
            "RA_threshold": absent_ref.ra_threshold,
            "ILD_threshold": absent_ref.ild_threshold,
            "all_ref_public_size": set_sizes["all_ref_public"],
            "RA_specific_ref_size": set_sizes["RA_specific_ref"],
            "ILD_specific_ref_size": set_sizes["ILD_specific_ref"],
            "shared_ref_size": set_sizes["shared_ref"],
        })
        loo_assignment_log.append({
            "context": context,
            "generation_method": "leave_one_out",
            "outer_repeat": args.outer_repeat,
            "outer_fold": args.outer_fold,
            "inner_fold": inner_fold if inner_fold is not None else "",
            "held_out_row_index": int(row),
            "held_out_label": held_label,
            "n_partition_samples": int(len(sample_rows)),
            "n_reference_samples": int(len(sample_rows) - 1),
        })

    output = pd.DataFrame(output_rows, index=output_index)
    output = output[ALL_PUBLIC_FEATURES].sort_index()
    if output.isna().any().any():
        raise RuntimeError(f"Missing LOO public values in {context}.")
    return output


def validate_full_reference_for_receptor(
    presence: sparse.csr_matrix,
    labels: np.ndarray,
    args: argparse.Namespace,
    spec: ReceptorSpec,
) -> Dict[str, int]:
    ref = build_public_reference(
        presence,
        np.arange(presence.shape[0], dtype=int),
        labels,
        args,
    )
    observed = {
        "global": int(ref.global_mask.sum()),
        "RA_specific": int(ref.ra_specific_mask.sum()),
        "ILD_specific": int(ref.ild_specific_mask.sum()),
        "shared": int(ref.shared_mask.sum()),
    }
    expected = {
        "global": spec.expected_global_size,
        "RA_specific": spec.expected_ra_specific_size,
        "ILD_specific": spec.expected_ild_specific_size,
        "shared": spec.expected_shared_size,
    }
    for key, expected_value in expected.items():
        if expected_value > 0 and observed[key] != expected_value:
            raise ValueError(
                f"{spec.receptor} dynamic public definition does not "
                f"reproduce step-03. {key}: expected {expected_value}, "
                f"observed {observed[key]}."
            )
    return observed


def add_receptor_to_new_rows(
    rows: List[Dict[str, object]],
    start: int,
    receptor: str,
) -> None:
    for row in rows[start:]:
        row["receptor"] = receptor


def prefix_public_frame(
    frame: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    renamed = {
        column: f"{prefix}{column}"
        for column in ALL_PUBLIC_FEATURES
    }
    out = frame.rename(columns=renamed)
    expected = [renamed[column] for column in ALL_PUBLIC_FEATURES]
    return out[expected]


def receptor_loo_public(
    presence: sparse.csr_matrix,
    frequency: sparse.csr_matrix,
    sample_rows: np.ndarray,
    labels: np.ndarray,
    aa_clone_numbers: np.ndarray,
    args: argparse.Namespace,
    spec: ReceptorSpec,
    context: str,
    inner_fold: Optional[int],
    reference_rows_log: List[Dict[str, object]],
    loo_assignment_log: List[Dict[str, object]],
) -> pd.DataFrame:
    reference_start = len(reference_rows_log)
    loo_start = len(loo_assignment_log)
    frame = leave_one_out_public_features(
        presence=presence,
        frequency=frequency,
        sample_rows=sample_rows,
        labels=labels,
        aa_clone_numbers=aa_clone_numbers,
        args=args,
        context=context,
        inner_fold=inner_fold,
        reference_rows_log=reference_rows_log,
        loo_assignment_log=loo_assignment_log,
    )
    add_receptor_to_new_rows(
        reference_rows_log, reference_start, spec.receptor
    )
    add_receptor_to_new_rows(
        loo_assignment_log, loo_start, spec.receptor
    )
    return prefix_public_frame(frame, spec.prefix)


def receptor_external_public(
    presence: sparse.csr_matrix,
    frequency: sparse.csr_matrix,
    reference_rows: np.ndarray,
    target_rows: np.ndarray,
    labels: np.ndarray,
    aa_clone_numbers: np.ndarray,
    args: argparse.Namespace,
    spec: ReceptorSpec,
    context: str,
    inner_fold: Optional[int],
    reference_rows_log: List[Dict[str, object]],
) -> pd.DataFrame:
    ref = build_public_reference(
        presence, reference_rows, labels, args
    )
    values = apply_public_reference(
        presence,
        frequency,
        target_rows,
        aa_clone_numbers,
        ref,
        args.epsilon,
    )
    values.index = np.asarray(target_rows, dtype=int)
    reference_rows_log.append(
        {
            "receptor": spec.receptor,
            "context": context,
            "generation_method": "external_training_partition",
            "outer_repeat": args.outer_repeat,
            "outer_fold": args.outer_fold,
            "inner_fold": inner_fold if inner_fold is not None else "",
            "held_out_row_index": "",
            "held_out_label": "",
            "n_reference": ref.n_reference,
            "n_RA_reference": ref.n_ra,
            "n_ILD_reference": ref.n_ild,
            "global_threshold": ref.global_threshold,
            "RA_threshold": ref.ra_threshold,
            "ILD_threshold": ref.ild_threshold,
            "all_ref_public_size": int(ref.global_mask.sum()),
            "RA_specific_ref_size": int(
                ref.ra_specific_mask.sum()
            ),
            "ILD_specific_ref_size": int(
                ref.ild_specific_mask.sum()
            ),
            "shared_ref_size": int(ref.shared_mask.sum()),
        }
    )
    return prefix_public_frame(values.sort_index(), spec.prefix)


def build_joint_public_for_partition(
    receptor_data: Mapping[str, Mapping[str, object]],
    sample_rows: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    context: str,
    inner_fold: Optional[int],
    reference_rows_log: List[Dict[str, object]],
    loo_assignment_log: List[Dict[str, object]],
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for receptor in ("TRB", "IGH"):
        data = receptor_data[receptor]
        parts.append(
            receptor_loo_public(
                presence=data["presence"],
                frequency=data["frequency"],
                sample_rows=sample_rows,
                labels=labels,
                aa_clone_numbers=data["aa_clone_numbers"],
                args=args,
                spec=data["spec"],
                context=context,
                inner_fold=inner_fold,
                reference_rows_log=reference_rows_log,
                loo_assignment_log=loo_assignment_log,
            )
        )
    out = pd.concat(parts, axis=1)
    if out.columns.duplicated().any():
        raise RuntimeError("Duplicated joint LOO public feature columns.")
    if out.isna().any().any():
        raise RuntimeError(f"Missing joint LOO public values in {context}.")
    return out.sort_index()


def build_joint_external_public(
    receptor_data: Mapping[str, Mapping[str, object]],
    reference_rows: np.ndarray,
    target_rows: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    context: str,
    inner_fold: Optional[int],
    reference_rows_log: List[Dict[str, object]],
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for receptor in ("TRB", "IGH"):
        data = receptor_data[receptor]
        parts.append(
            receptor_external_public(
                presence=data["presence"],
                frequency=data["frequency"],
                reference_rows=reference_rows,
                target_rows=target_rows,
                labels=labels,
                aa_clone_numbers=data["aa_clone_numbers"],
                args=args,
                spec=data["spec"],
                context=context,
                inner_fold=inner_fold,
                reference_rows_log=reference_rows_log,
            )
        )
    out = pd.concat(parts, axis=1)
    if out.columns.duplicated().any():
        raise RuntimeError(
            "Duplicated joint external public feature columns."
        )
    if out.isna().any().any():
        raise RuntimeError(
            f"Missing joint external public values in {context}."
        )
    return out.sort_index()


def build_design(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    zero_sd_tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, object]]]:
    train_parts: List[np.ndarray] = []
    valid_parts: List[np.ndarray] = []
    feature_names: List[str] = []
    source_types: List[str] = []

    for col in numeric_cols:
        train_values = pd.to_numeric(train_df[col], errors="raise").to_numpy(float)
        valid_values = pd.to_numeric(valid_df[col], errors="raise").to_numpy(float)
        train_parts.append(train_values[:, None])
        valid_parts.append(valid_values[:, None])
        feature_names.append(col)
        source_types.append("numeric")

    for col in categorical_cols:
        train_values = train_df[col].astype(str)
        valid_values = valid_df[col].astype(str)
        categories = sorted(train_values.unique().tolist())
        if not categories:
            raise ValueError(f"No training category found for {col}.")
        reference = categories[0]
        for category in categories[1:]:
            train_parts.append((train_values == category).to_numpy(float)[:, None])
            valid_parts.append((valid_values == category).to_numpy(float)[:, None])
            feature_names.append(f"{col}__{category}_vs_{reference}")
            source_types.append("categorical_dummy")

    if not train_parts:
        raise ValueError("No predictor columns were assembled.")

    X_train_raw = np.hstack(train_parts)
    X_valid_raw = np.hstack(valid_parts)

    means = X_train_raw.mean(axis=0)
    sds = X_train_raw.std(axis=0, ddof=0)
    keep = np.isfinite(sds) & (sds > zero_sd_tolerance)

    if not keep.any():
        raise ValueError("All predictors had zero/invalid standard deviation.")

    X_train = (X_train_raw[:, keep] - means[keep]) / sds[keep]
    X_valid = (X_valid_raw[:, keep] - means[keep]) / sds[keep]
    kept_names = [name for name, flag in zip(feature_names, keep) if flag]

    rows: List[Dict[str, object]] = []
    for name, source_type, mean, sd, flag in zip(
        feature_names, source_types, means, sds, keep
    ):
        rows.append({
            "feature_name": name,
            "source_type": source_type,
            "training_mean": float(mean),
            "training_sd": float(sd),
            "kept_after_zero_variance_filter": bool(flag),
        })

    if not np.isfinite(X_train).all() or not np.isfinite(X_valid).all():
        raise ValueError("Non-finite values were produced during preprocessing.")
    return X_train, X_valid, kept_names, rows


def model_specifications(
    static_features: Sequence[str],
    public_features: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, List[str]]]:
    static_numeric = [args.age_col, *static_features]
    public_numeric = [*static_numeric, *public_features]

    return {
        "M0_clinical": {
            "numeric": [args.age_col],
            "categorical": [args.sex_col],
        },
        "M1_static_igh": {
            "numeric": static_numeric,
            "categorical": [args.sex_col],
        },
        "M2_static_igh_public": {
            "numeric": public_numeric,
            "categorical": [args.sex_col],
        },
        "M3_static_igh_public_material": {
            "numeric": public_numeric,
            "categorical": [args.sex_col, args.material_col],
        },
    }


def fit_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    l1_ratio: float,
    lambda_value: float,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[LogisticRegression, bool, int]:
    class_weight = None if args.class_weight == "none" else args.class_weight
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        C=1.0 / lambda_value,
        class_weight=class_weight,
        max_iter=args.max_iter,
        tol=args.tolerance,
        random_state=seed,
        fit_intercept=True,
    )
    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
        converged = not any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )
    n_iter = int(np.max(model.n_iter_))
    return model, converged, n_iter


def safe_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(y, probability))


def choose_threshold_youden(
    y: np.ndarray,
    probability: np.ndarray,
) -> float:
    fpr, tpr, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    j = tpr[finite] - fpr[finite]
    candidates = thresholds[finite][j == np.max(j)]
    return float(candidates[np.argmin(np.abs(candidates - 0.5))])


def classification_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "roc_auc": safe_auc(y, probability),
        "pr_auc": float(average_precision_score(y, probability)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity_recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def model_specifications(
    static_features: Sequence[str],
    public_features: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, List[str]]]:
    static_numeric = [args.age_col, *static_features]
    public_numeric = [*static_numeric, *public_features]
    return {
        "M0_clinical": {
            "numeric": [args.age_col],
            "categorical": [args.sex_col],
        },
        "M1_static_trb_igh": {
            "numeric": static_numeric,
            "categorical": [args.sex_col],
        },
        "M2_static_trb_igh_public": {
            "numeric": public_numeric,
            "categorical": [args.sex_col],
        },
        "M3_static_trb_igh_public_material": {
            "numeric": public_numeric,
            "categorical": [args.sex_col, args.material_col],
        },
    }


def assemble_model_dataframe(
    base: pd.DataFrame,
    sample_ids: Sequence[str],
    public_values: Optional[pd.DataFrame],
    row_lookup: Mapping[str, int],
) -> pd.DataFrame:
    out = base.loc[list(sample_ids)].copy()
    if public_values is not None:
        row_ids = [row_lookup[sample_id] for sample_id in sample_ids]
        aligned = public_values.loc[row_ids].copy()
        aligned.index = list(sample_ids)
        for column in aligned.columns:
            if column in out.columns:
                raise ValueError(
                    f"Dynamic public feature collides with base: {column}"
                )
            out[column] = aligned[column].to_numpy(float)
    return out


def prepare_inner_folds(
    base: pd.DataFrame,
    inner_task: pd.DataFrame,
    outer_train_ids: Sequence[str],
    row_lookup: Mapping[str, int],
    receptor_data: Mapping[str, Mapping[str, object]],
    labels: np.ndarray,
    model_specs: Mapping[str, Mapping[str, List[str]]],
    args: argparse.Namespace,
    reference_rows_log: List[Dict[str, object]],
    loo_assignment_log: List[Dict[str, object]],
) -> Dict[str, Dict[int, PreparedFold]]:
    prepared: Dict[str, Dict[int, PreparedFold]] = {
        model: {} for model in model_specs
    }

    for inner_fold in range(1, args.inner_folds + 1):
        valid_ids = inner_task.loc[
            inner_task["inner_fold"] == inner_fold,
            args.sample_id_col,
        ].astype(str).tolist()
        valid_set = set(valid_ids)
        train_ids = [
            sample_id for sample_id in outer_train_ids
            if sample_id not in valid_set
        ]

        train_rows = np.array(
            [row_lookup[sample_id] for sample_id in train_ids],
            dtype=int,
        )
        valid_rows = np.array(
            [row_lookup[sample_id] for sample_id in valid_ids],
            dtype=int,
        )

        train_public = build_joint_public_for_partition(
            receptor_data=receptor_data,
            sample_rows=train_rows,
            labels=labels,
            args=args,
            context="inner_training_leave_one_out",
            inner_fold=inner_fold,
            reference_rows_log=reference_rows_log,
            loo_assignment_log=loo_assignment_log,
        )
        valid_public = build_joint_external_public(
            receptor_data=receptor_data,
            reference_rows=train_rows,
            target_rows=valid_rows,
            labels=labels,
            args=args,
            context="inner_validation_application",
            inner_fold=inner_fold,
            reference_rows_log=reference_rows_log,
        )

        train_with_public = assemble_model_dataframe(
            base, train_ids, train_public, row_lookup
        )
        valid_with_public = assemble_model_dataframe(
            base, valid_ids, valid_public, row_lookup
        )

        y_train = (
            train_with_public[args.label_col].astype(str).str.upper()
            == args.positive_label.upper()
        ).astype(int).to_numpy()
        y_valid = (
            valid_with_public[args.label_col].astype(str).str.upper()
            == args.positive_label.upper()
        ).astype(int).to_numpy()

        if len(np.unique(y_train)) != 2 or len(np.unique(y_valid)) != 2:
            raise ValueError(
                f"Inner fold {inner_fold} does not contain both classes."
            )

        for model_name, specification in model_specs.items():
            X_train, X_valid, names, prep_rows = build_design(
                train_with_public,
                valid_with_public,
                numeric_cols=specification["numeric"],
                categorical_cols=specification["categorical"],
                zero_sd_tolerance=args.zero_sd_tolerance,
            )
            for row in prep_rows:
                row.update(
                    {
                        "stage": "inner",
                        "model": model_name,
                        "inner_fold": inner_fold,
                    }
                )
            prepared[model_name][inner_fold] = PreparedFold(
                X_train=X_train,
                X_valid=X_valid,
                y_train=y_train,
                y_valid=y_valid,
                valid_sample_ids=valid_ids,
                feature_names=names,
                preprocess_rows=prep_rows,
            )
        print(
            f"[inner fold {inner_fold}/{args.inner_folds}] "
            f"train={len(train_ids)}, validation={len(valid_ids)}"
        )
    return prepared


def fit_outer_models(
    base: pd.DataFrame,
    outer_train_ids: Sequence[str],
    outer_valid_ids: Sequence[str],
    row_lookup: Mapping[str, int],
    receptor_data: Mapping[str, Mapping[str, object]],
    labels: np.ndarray,
    model_specs: Mapping[str, Mapping[str, List[str]]],
    selected: Mapping[str, Mapping[str, float]],
    args: argparse.Namespace,
    reference_rows_log: List[Dict[str, object]],
    loo_assignment_log: List[Dict[str, object]],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    train_rows = np.array(
        [row_lookup[sample_id] for sample_id in outer_train_ids],
        dtype=int,
    )
    valid_rows = np.array(
        [row_lookup[sample_id] for sample_id in outer_valid_ids],
        dtype=int,
    )

    train_public = build_joint_public_for_partition(
        receptor_data=receptor_data,
        sample_rows=train_rows,
        labels=labels,
        args=args,
        context="outer_training_leave_one_out",
        inner_fold=None,
        reference_rows_log=reference_rows_log,
        loo_assignment_log=loo_assignment_log,
    )
    valid_public = build_joint_external_public(
        receptor_data=receptor_data,
        reference_rows=train_rows,
        target_rows=valid_rows,
        labels=labels,
        args=args,
        context="outer_validation_application",
        inner_fold=None,
        reference_rows_log=reference_rows_log,
    )

    train_with_public = assemble_model_dataframe(
        base, outer_train_ids, train_public, row_lookup
    )
    valid_with_public = assemble_model_dataframe(
        base, outer_valid_ids, valid_public, row_lookup
    )

    prediction_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    coefficient_rows: List[Dict[str, object]] = []
    preprocess_rows: List[Dict[str, object]] = []

    for model_index, (model_name, specification) in enumerate(
        model_specs.items()
    ):
        X_train, X_valid, names, prep = build_design(
            train_with_public,
            valid_with_public,
            numeric_cols=specification["numeric"],
            categorical_cols=specification["categorical"],
            zero_sd_tolerance=args.zero_sd_tolerance,
        )
        for row in prep:
            row.update(
                {
                    "stage": "outer_final",
                    "model": model_name,
                    "inner_fold": "",
                }
            )
        preprocess_rows.extend(prep)

        y_train = (
            train_with_public[args.label_col].astype(str).str.upper()
            == args.positive_label.upper()
        ).astype(int).to_numpy()
        y_valid = (
            valid_with_public[args.label_col].astype(str).str.upper()
            == args.positive_label.upper()
        ).astype(int).to_numpy()

        params = selected[model_name]
        model, converged, n_iter = fit_elastic_net(
            X_train,
            y_train,
            l1_ratio=params["l1_ratio"],
            lambda_value=params["lambda"],
            args=args,
            seed=derive_seed(args.seed, 400, model_index),
        )
        probability = model.predict_proba(X_valid)[:, 1]
        threshold = params["threshold"]
        metrics = classification_metrics(
            y_valid, probability, threshold
        )
        metrics.update(
            {
                "model": model_name,
                "model_definition": MODEL_LABELS[model_name],
                "outer_repeat": args.outer_repeat,
                "outer_fold": args.outer_fold,
                "n_outer_train": len(outer_train_ids),
                "n_outer_validation": len(outer_valid_ids),
                "selected_l1_ratio_alpha": params["l1_ratio"],
                "selected_lambda": params["lambda"],
                "inner_selected_roc_auc": params["inner_roc_auc"],
                "inner_selected_pr_auc": params["inner_pr_auc"],
                "fit_converged": converged,
                "iterations_used": n_iter,
                "n_final_predictors": len(names),
                "n_nonzero_coefficients": int(
                    np.sum(np.abs(model.coef_.ravel()) > 1e-12)
                ),
            }
        )
        metric_rows.append(metrics)

        predicted = (probability >= threshold).astype(int)
        for sample_id, y_value, prob, pred in zip(
            outer_valid_ids, y_valid, probability, predicted
        ):
            prediction_rows.append(
                {
                    "model": model_name,
                    "outer_repeat": args.outer_repeat,
                    "outer_fold": args.outer_fold,
                    "sample_id": sample_id,
                    "trb_libraryid": base.loc[
                        sample_id, args.trb_library_id_col
                    ],
                    "igh_libraryid": base.loc[
                        sample_id, args.igh_library_id_col
                    ],
                    "true_label": int(y_value),
                    "true_cohort": (
                        args.positive_label if y_value == 1 else "RA"
                    ),
                    "probability_ILD": float(prob),
                    "threshold": float(threshold),
                    "predicted_label": int(pred),
                    "predicted_cohort": (
                        args.positive_label if pred == 1 else "RA"
                    ),
                }
            )

        coefficient_rows.append(
            {
                "model": model_name,
                "feature_name": "__INTERCEPT__",
                "receptor": "none",
                "coefficient": float(model.intercept_[0]),
                "absolute_coefficient": abs(
                    float(model.intercept_[0])
                ),
                "nonzero": True,
            }
        )
        for name, coefficient in zip(names, model.coef_.ravel()):
            receptor = (
                "TRB" if name.startswith("trb_")
                else "IGH" if name.startswith("igh_")
                else "clinical"
            )
            coefficient_rows.append(
                {
                    "model": model_name,
                    "feature_name": name,
                    "receptor": receptor,
                    "coefficient": float(coefficient),
                    "absolute_coefficient": abs(float(coefficient)),
                    "nonzero": bool(abs(coefficient) > 1e-12),
                }
            )

    sample_ids = base.index.astype(str).tolist()
    train_public_out = train_public.copy()
    train_public_out.insert(
        0,
        args.sample_id_col,
        [sample_ids[row] for row in train_public_out.index],
    )
    valid_public_out = valid_public.copy()
    valid_public_out.insert(
        0,
        args.sample_id_col,
        [sample_ids[row] for row in valid_public_out.index],
    )

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(preprocess_rows),
        train_public_out.reset_index(drop=True),
        valid_public_out.reset_index(drop=True),
    )


def tune_models(
    prepared: Mapping[str, Mapping[int, PreparedFold]],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    tuning_rows: List[Dict[str, object]] = []
    prediction_cache: Dict[
        Tuple[str, float, float], List[Dict[str, object]]
    ] = {}

    for model_idx, model_name in enumerate(prepared):
        for l1_ratio in args.l1_ratios:
            for lambda_value in args.lambdas:
                pred_rows: List[Dict[str, object]] = []
                fold_aucs: List[float] = []
                convergence: List[bool] = []
                iterations: List[int] = []

                for inner_fold, fold_data in prepared[model_name].items():
                    model, converged, n_iter = fit_elastic_net(
                        fold_data.X_train,
                        fold_data.y_train,
                        l1_ratio=l1_ratio,
                        lambda_value=lambda_value,
                        args=args,
                        seed=derive_seed(
                            args.seed,
                            200,
                            model_idx,
                            inner_fold,
                            int(round(l1_ratio * 1000)),
                            int(round(lambda_value * 1000)),
                        ),
                    )
                    probability = model.predict_proba(fold_data.X_valid)[:, 1]
                    fold_aucs.append(
                        safe_auc(fold_data.y_valid, probability)
                    )
                    convergence.append(converged)
                    iterations.append(n_iter)

                    for sample_id, y, prob in zip(
                        fold_data.valid_sample_ids,
                        fold_data.y_valid,
                        probability,
                    ):
                        pred_rows.append({
                            "model": model_name,
                            "inner_fold": inner_fold,
                            "sample_id": sample_id,
                            "true_label": int(y),
                            "probability": float(prob),
                            "l1_ratio_alpha": l1_ratio,
                            "lambda": lambda_value,
                        })

                pred_df = pd.DataFrame(pred_rows)
                pooled_auc = safe_auc(
                    pred_df["true_label"].to_numpy(),
                    pred_df["probability"].to_numpy(),
                )
                pooled_pr = float(
                    average_precision_score(
                        pred_df["true_label"],
                        pred_df["probability"],
                    )
                )
                tuning_rows.append({
                    "model": model_name,
                    "l1_ratio_alpha": l1_ratio,
                    "lambda": lambda_value,
                    "C_inverse_lambda": 1.0 / lambda_value,
                    "pooled_inner_roc_auc": pooled_auc,
                    "pooled_inner_pr_auc": pooled_pr,
                    "mean_fold_roc_auc": float(np.mean(fold_aucs)),
                    "sd_fold_roc_auc": float(np.std(fold_aucs, ddof=1)),
                    "all_fits_converged": bool(all(convergence)),
                    "max_iterations_used": int(max(iterations)),
                })
                prediction_cache[
                    (model_name, l1_ratio, lambda_value)
                ] = pred_rows

    tuning = pd.DataFrame(tuning_rows)

    selected: Dict[str, Dict[str, float]] = {}
    selected_oof_parts: List[pd.DataFrame] = []

    for model_name in prepared:
        candidates = tuning[tuning["model"] == model_name].copy()
        candidates = candidates.sort_values(
            [
                "pooled_inner_roc_auc",
                "pooled_inner_pr_auc",
                "lambda",
                "l1_ratio_alpha",
            ],
            ascending=[False, False, False, False],
        )
        best = candidates.iloc[0]
        l1_ratio = float(best["l1_ratio_alpha"])
        lambda_value = float(best["lambda"])
        pred = pd.DataFrame(
            prediction_cache[(model_name, l1_ratio, lambda_value)]
        )
        threshold = choose_threshold_youden(
            pred["true_label"].to_numpy(),
            pred["probability"].to_numpy(),
        )
        pred["selected_threshold"] = threshold
        selected_oof_parts.append(pred)
        selected[model_name] = {
            "l1_ratio": l1_ratio,
            "lambda": lambda_value,
            "threshold": threshold,
            "inner_roc_auc": float(best["pooled_inner_roc_auc"]),
            "inner_pr_auc": float(best["pooled_inner_pr_auc"]),
        }

    selected_oof = pd.concat(selected_oof_parts, ignore_index=True)
    return tuning, selected_oof, selected


def output_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "configuration": output_dir / "05_trial_configuration.json",
        "sample_roles": output_dir / "05_trial_sample_roles.csv",
        "outer_train_public": (
            output_dir
            / "05_dynamic_public_features_outer_train_loo_TRB_IGH.csv"
        ),
        "outer_valid_public": (
            output_dir
            / "05_dynamic_public_features_outer_validation_TRB_IGH.csv"
        ),
        "public_reference_summary": (
            output_dir / "05_public_reference_build_summary_TRB_IGH.csv"
        ),
        "public_loo_assignments": (
            output_dir / "05_public_loo_assignments_TRB_IGH.csv"
        ),
        "inner_tuning": output_dir / "05_inner_tuning_results.csv",
        "inner_oof": output_dir / "05_inner_selected_oof_predictions.csv",
        "outer_predictions": (
            output_dir / "05_outer_validation_predictions.csv"
        ),
        "outer_metrics": output_dir / "05_outer_validation_metrics.csv",
        "coefficients": output_dir / "05_final_model_coefficients.csv",
        "preprocessing": output_dir / "05_preprocessing_summary.csv",
        "static_feature_list": (
            output_dir / "05_static_TRB_IGH_feature_list.csv"
        ),
        "summary": output_dir / "05_single_outer_trial_summary.md",
    }


def enforce_overwrite(
    paths: Iterable[Path],
    overwrite: bool,
) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        text = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Outputs already exist. Add --overwrite:\n" + text
        )


def write_summary(
    path: Path,
    args: argparse.Namespace,
    n_train: int,
    n_valid: int,
    n_static_trb: int,
    n_static_igh: int,
    public_features: Sequence[str],
    metrics: pd.DataFrame,
    full_reference_counts: Optional[
        Mapping[str, Mapping[str, int]]
    ],
    runtime: float,
) -> None:
    lines = [
        "# 05 Paired TRB+IGH Single Outer-Fold Trial Summary",
        "",
        "## Task",
        "",
        f"- Outer repeat: **{args.outer_repeat}**",
        f"- Outer fold: **{args.outer_fold}**",
        f"- Outer training patients: **{n_train}**",
        f"- Outer validation patients: **{n_valid}**",
        f"- Inner CV: **{args.inner_folds} folds**",
        (
            "- Training-patient public generation: "
            "**exact receptor-specific leave-one-out (LOO)**"
        ),
        "",
        "## Leakage controls",
        "",
        "- Independent test data were not read.",
        "- The frozen CV unit was the paired patient.",
        "- Outer-validation patients were absent from all inner folds.",
        (
            "- TRB and IGH reference sets were rebuilt independently "
            "inside each training partition."
        ),
        (
            "- A fitting patient contributed neither TRB nor IGH sequences "
            "to its own LOO references."
        ),
        (
            "- Encoding, zero-variance filtering and standardization were "
            "learned from each training partition only."
        ),
        "",
        "## Feature definitions",
        "",
        f"- Static TRB features selected from manifest: **{n_static_trb}**",
        f"- Static IGH features selected from manifest: **{n_static_igh}**",
        f"- Public feature set per receptor: `{args.public_feature_set}`",
        f"- Joint dynamic public predictors: **{len(public_features)}**",
    ]
    lines.extend(
        f"- Public predictor: `{feature}`"
        for feature in public_features
    )

    if full_reference_counts is not None:
        lines.extend(
            ["", "## Step-03 definition reproduction", ""]
        )
        for receptor, counts in full_reference_counts.items():
            lines.append(f"### {receptor}")
            lines.append("")
            for key, value in counts.items():
                lines.append(f"- {key}: **{value:,}**")
            lines.append("")

    lines.extend(
        [
            "## Outer validation results",
            "",
            (
                "| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity "
                "| F1 | Nonzero coefficients |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['model']} | {row['roc_auc']:.4f} | "
            f"{row['pr_auc']:.4f} | "
            f"{row['sensitivity_recall']:.4f} | "
            f"{row['specificity']:.4f} | {row['f1']:.4f} | "
            f"{int(row['n_nonzero_coefficients'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation warning",
            "",
            (
                "- This is one outer fold for implementation validation, "
                "not the final performance estimate."
            ),
            "- Do not choose the final model from this single fold.",
            (
                "- After validation, run the identical workflow over all "
                "100 frozen outer tasks."
            ),
            "",
            f"- Runtime: **{runtime:.2f} seconds**",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")



def main() -> int:
    args = parse_args()
    started = time.time()
    specs = receptor_specs(args)

    (
        base,
        manifest,
        outer_task,
        inner_task,
        outer_train_ids,
        outer_valid_ids,
        input_paths,
    ) = read_core_inputs(args)

    reference_definitions: Dict[str, Dict[str, object]] = {}
    reference_paths: Dict[str, Path] = {}
    for receptor in ("TRB", "IGH"):
        path, definition = load_and_validate_reference_definition(
            args, specs[receptor]
        )
        reference_paths[receptor] = path
        reference_definitions[receptor] = definition

    output_dir = (
        Path(args.output_root).expanduser().resolve()
        / f"public_{args.public_feature_set}"
        / f"repeat_{args.outer_repeat:02d}_fold_{args.outer_fold:02d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    enforce_overwrite(paths.values(), args.overwrite)

    static_trb, static_igh = static_joint_features_from_manifest(
        manifest, base
    )
    static_features = [*static_trb, *static_igh]
    selected_public_base = PUBLIC_FEATURE_SETS[
        args.public_feature_set
    ]
    public_features = [
        *[f"trb_{feature}" for feature in selected_public_base],
        *[f"igh_{feature}" for feature in selected_public_base],
    ]
    model_specs = model_specifications(
        static_features, public_features, args
    )

    print("=" * 88)
    print("05 Paired TRB+IGH Single Outer-Fold Modeling Trial")
    print("   Exact receptor-specific LOO public features")
    print("=" * 88)
    print(
        f"Task: repeat={args.outer_repeat}, "
        f"outer_fold={args.outer_fold}"
    )
    print(
        f"Outer train={len(outer_train_ids)}, "
        f"outer validation={len(outer_valid_ids)}"
    )
    print(f"Static TRB predictors: {len(static_trb)}")
    print(f"Static IGH predictors: {len(static_igh)}")
    print(
        f"Public predictor set: {args.public_feature_set} "
        f"({len(selected_public_base)} per receptor; "
        f"{len(public_features)} joint)"
    )
    print("CV unit: paired patient")
    print("Independent test set: NOT READ")
    print("")

    labels = (
        base[args.label_col].astype(str).str.upper().to_numpy()
    )
    sample_ids = base.index.astype(str).tolist()
    row_lookup = {
        sample_id: index
        for index, sample_id in enumerate(sample_ids)
    }

    receptor_data: Dict[str, Dict[str, object]] = {}
    cache_metadata: Dict[str, Dict[str, object]] = {}
    for receptor in ("TRB", "IGH"):
        spec = specs[receptor]
        presence, frequency, cache_meta = (
            build_or_load_sparse_cache(args, base, spec)
        )
        aa_clone_numbers = pd.to_numeric(
            base[spec.aa_clone_number_col], errors="raise"
        ).to_numpy(float)
        receptor_data[receptor] = {
            "spec": spec,
            "presence": presence,
            "frequency": frequency,
            "aa_clone_numbers": aa_clone_numbers,
        }
        cache_metadata[receptor] = cache_meta

    full_reference_counts = None
    if not args.skip_full_reference_validation:
        full_reference_counts = {}
        for receptor in ("TRB", "IGH"):
            data = receptor_data[receptor]
            counts = validate_full_reference_for_receptor(
                presence=data["presence"],
                labels=labels,
                args=args,
                spec=data["spec"],
            )
            full_reference_counts[receptor] = counts
            print(
                f"[PASS] {receptor} full-train reference reproduces "
                "step-03: "
                + ", ".join(
                    f"{key}={value:,}"
                    for key, value in counts.items()
                )
            )

    reference_rows_log: List[Dict[str, object]] = []
    loo_assignment_log: List[Dict[str, object]] = []

    prepared = prepare_inner_folds(
        base=base,
        inner_task=inner_task,
        outer_train_ids=outer_train_ids,
        row_lookup=row_lookup,
        receptor_data=receptor_data,
        labels=labels,
        model_specs=model_specs,
        args=args,
        reference_rows_log=reference_rows_log,
        loo_assignment_log=loo_assignment_log,
    )
    tuning, selected_oof, selected = tune_models(
        prepared, args
    )

    (
        outer_predictions,
        outer_metrics,
        coefficients,
        preprocessing,
        outer_train_public,
        outer_valid_public,
    ) = fit_outer_models(
        base=base,
        outer_train_ids=outer_train_ids,
        outer_valid_ids=outer_valid_ids,
        row_lookup=row_lookup,
        receptor_data=receptor_data,
        labels=labels,
        model_specs=model_specs,
        selected=selected,
        args=args,
        reference_rows_log=reference_rows_log,
        loo_assignment_log=loo_assignment_log,
    )

    roles = outer_task.copy()
    roles["selected_outer_task"] = np.where(
        roles["outer_fold"] == args.outer_fold,
        "validation",
        "training",
    )
    role_columns = [
        "outer_repeat",
        "outer_fold",
        args.sample_id_col,
        args.label_col,
    ]
    for column in [
        args.trb_library_id_col,
        args.igh_library_id_col,
        "batch",
        "material",
        "sex",
    ]:
        if column in roles.columns and column not in role_columns:
            role_columns.append(column)
    role_columns.append("selected_outer_task")
    roles = roles[role_columns]

    static_feature_table = pd.DataFrame(
        {
            "feature_order": np.arange(
                1, len(static_features) + 1
            ),
            "feature_name": static_features,
            "receptor": [
                "TRB" if feature.startswith("trb_") else "IGH"
                for feature in static_features
            ],
        }
    )

    reference_summary = pd.DataFrame(reference_rows_log)
    if "held_out_row_index" in reference_summary.columns:
        reference_summary[args.sample_id_col] = (
            pd.to_numeric(
                reference_summary["held_out_row_index"],
                errors="coerce",
            )
            .map({index: sid for index, sid in enumerate(sample_ids)})
        )
    loo_assignments = pd.DataFrame(loo_assignment_log)
    if not loo_assignments.empty:
        loo_assignments[args.sample_id_col] = (
            loo_assignments["held_out_row_index"]
            .map({index: sid for index, sid in enumerate(sample_ids)})
        )

    roles.to_csv(paths["sample_roles"], index=False)
    outer_train_public.to_csv(
        paths["outer_train_public"], index=False
    )
    outer_valid_public.to_csv(
        paths["outer_valid_public"], index=False
    )
    reference_summary.to_csv(
        paths["public_reference_summary"], index=False
    )
    loo_assignments.to_csv(
        paths["public_loo_assignments"], index=False
    )
    tuning.to_csv(paths["inner_tuning"], index=False)
    selected_oof.to_csv(paths["inner_oof"], index=False)
    outer_predictions.to_csv(
        paths["outer_predictions"], index=False
    )
    outer_metrics.to_csv(paths["outer_metrics"], index=False)
    coefficients.to_csv(paths["coefficients"], index=False)
    preprocessing.to_csv(paths["preprocessing"], index=False)
    static_feature_table.to_csv(
        paths["static_feature_list"], index=False
    )

    config = {
        "script_version": SCRIPT_VERSION,
        "analysis_view": "TRB_IGH",
        "cv_unit": "patient",
        "outer_repeat": args.outer_repeat,
        "outer_fold": args.outer_fold,
        "input_files": {
            key: str(value) for key, value in input_paths.items()
        },
        "input_sha256": {
            key: sha256sum(value)
            for key, value in input_paths.items()
        },
        "receptor_inputs": {
            receptor: {
                "public_catalog": str(specs[receptor].public_catalog),
                "public_catalog_sha256": sha256sum(
                    specs[receptor].public_catalog
                ),
                "reference_definition": str(
                    reference_paths[receptor]
                ),
                "reference_definition_sha256": sha256sum(
                    reference_paths[receptor]
                ),
                "reference_definition_validated": True,
                "reference_definition_snapshot": (
                    reference_definitions[receptor]
                ),
                "clone_table_dir": str(
                    specs[receptor].clone_table_dir
                ),
                "cache_metadata": cache_metadata[receptor],
            }
            for receptor in ("TRB", "IGH")
        },
        "n_outer_train": len(outer_train_ids),
        "n_outer_validation": len(outer_valid_ids),
        "n_static_trb_features": len(static_trb),
        "n_static_igh_features": len(static_igh),
        "public_feature_set_per_receptor": args.public_feature_set,
        "public_predictors": public_features,
        "models": MODEL_LABELS,
        "l1_ratios_alpha": args.l1_ratios,
        "lambdas": args.lambdas,
        "class_weight": args.class_weight,
        "threshold_rule": (
            "Youden J from selected inner OOF predictions"
        ),
        "training_public_generation": (
            "exact receptor-specific leave-one-out"
        ),
        "validation_public_generation": (
            "complete corresponding training partition, separately for TRB "
            "and IGH"
        ),
        "public_thresholds": {
            "global_min_prevalence": args.global_min_prevalence,
            "global_min_count": args.global_min_count,
            "group_min_prevalence": args.group_min_prevalence,
            "group_min_count": args.group_min_count,
            "specific_prevalence_delta": (
                args.specific_prevalence_delta
            ),
            "epsilon": args.epsilon,
        },
        "selected_parameters": selected,
        "full_reference_counts": full_reference_counts,
        "seed": args.seed,
    }
    paths["configuration"].write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    runtime = time.time() - started
    write_summary(
        path=paths["summary"],
        args=args,
        n_train=len(outer_train_ids),
        n_valid=len(outer_valid_ids),
        n_static_trb=len(static_trb),
        n_static_igh=len(static_igh),
        public_features=public_features,
        metrics=outer_metrics,
        full_reference_counts=full_reference_counts,
        runtime=runtime,
    )

    print("")
    print("[Selected parameters]")
    for model_name, params in selected.items():
        print(
            f"- {model_name}: alpha={params['l1_ratio']}, "
            f"lambda={params['lambda']}, "
            f"threshold={params['threshold']:.4f}, "
            f"inner AUC={params['inner_roc_auc']:.4f}"
        )
    print("")
    print("[Outer validation metrics]")
    print(
        outer_metrics[
            [
                "model",
                "roc_auc",
                "pr_auc",
                "sensitivity_recall",
                "specificity",
                "f1",
                "n_nonzero_coefficients",
                "fit_converged",
            ]
        ].to_string(index=False)
    )
    print("")
    print("[Output files]")
    for key, path in paths.items():
        print(f"- {key}: {path}")
    print(f"Runtime: {runtime:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
