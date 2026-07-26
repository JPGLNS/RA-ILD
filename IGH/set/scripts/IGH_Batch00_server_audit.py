#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch 00 read-only audit for the RA/RA-ILD IGH framework upgrade.

This utility does not modify historical IGH inputs or results. It inventories:
- Git refs and repository trees;
- full/train/test metadata and patient/sample consistency;
- AA clone-table coverage and IGH filename contracts;
- step-02/04 feature resources and manifest-derived static predictors;
- public catalog/reference/cache resources;
- historical final-model and independent-test provenance;
- file sizes and unresolved design questions.

Outputs are written to a separate audit directory.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0-IGH-Batch00"
EXPECTED_RECEPTOR = "IGH"
AA_SUFFIX = "_IGH-without-DJ_CDR3_AA_clone_table.csv"
NT_SUFFIX = "_IGH-without-DJ_CDR3_NT_frequency_error_correct.csv"
EXPECTED_TRB_HEAD = "c772f367f5ad8865f1217ab760dcb8a2f18d158b"

LIKELY_PATHS = {
    "full_metadata": "IGH/metadata.csv",
    "train_metadata": "IGH/set/train/metadata_train_70.csv",
    "test_metadata": "IGH/set/test/metadata_test_30.csv",
    "train_aa_dir": "IGH/set/train/result/01_AA_clone_table",
    "test_aa_dir": "IGH/set/test/result/01_AA_clone_table",
    "train_02": "IGH/set/train/result/02_sample_level_features/02_sample_level_features_merged.csv",
    "test_02": "IGH/set/test/result/02_sample_level_features/02_sample_level_features_merged.csv",
    "train_3mer_vocab": "IGH/set/train/result/02_sample_level_features/02_3mer_vocabulary.csv",
    "train_base": "IGH/set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv",
    "test_final": "IGH/set/test/result/04_final_feature_matrix/04_test_final_feature_matrix.csv",
    "feature_manifest": "IGH/set/train/result/04_final_feature_matrix/04_feature_manifest.csv",
    "public_catalog": "IGH/set/train/result/03_public_features/03_public_aa_catalog.csv.gz",
    "reference_definition": "IGH/set/train/result/03_public_features/03_reference_definition.json",
    "cache_presence": "IGH/set/train/result/05_modeling/cache/05_train_public_presence.npz",
    "cache_frequency": "IGH/set/train/result/05_modeling/cache/05_train_public_frequency.npz",
    "cache_metadata": "IGH/set/train/result/05_modeling/cache/05_train_public_sparse_cache.json",
    "outer_assignments": "IGH/set/train/result/05_modeling/cv_splits/05_outer_fold_assignments.csv",
    "inner_assignments": "IGH/set/train/result/05_modeling/cv_splits/05_inner_fold_assignments.csv.gz",
    "final_model_config": "IGH/set/train/result/07_final_M2_model/07_final_M2_configuration.json",
    "independent_test_config": "IGH/set/test/result/08_final_M2_validation/08_independent_test_configuration.json",
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read-only Batch 00 audit for the IGH framework upgrade.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project-root", default="/data/users/chenhaisheng/RA-ILD")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--main-ref", default="main")
    p.add_argument("--trb-ref", default="feature/trb-scheme-upgrade-batch01")
    p.add_argument("--expected-trb-head", default=EXPECTED_TRB_HEAD)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def now_iso() -> str:
    import datetime as dt
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")

def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def run_cmd(cmd: Sequence[str], cwd: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        list(cmd), cwd=str(cwd), text=True, capture_output=True, check=False
    )
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.rstrip(),
        "stderr": proc.stderr.rstrip(),
    }

def git_value(root: Path, *args: str) -> Optional[str]:
    result = run_cmd(["git", *args], root)
    if result["returncode"] != 0:
        return None
    value = str(result["stdout"]).strip()
    return value or None

def git_lines(root: Path, *args: str) -> List[str]:
    value = git_value(root, *args)
    return [] if value is None else [x for x in value.splitlines() if x.strip()]

def ensure_output(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Audit outputs already exist. Add --overwrite for a documented rerun:\n"
            + "\n".join(f"  - {p}" for p in existing)
        )

def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path)
    df = df.drop(
        columns=[c for c in df.columns if str(c).startswith("Unnamed:")],
        errors="ignore",
    )
    return df

def lower_map(columns: Iterable[str]) -> Dict[str, str]:
    return {str(c).strip().lower(): str(c) for c in columns}

def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    mapping = lower_map(df.columns)
    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]
    return None

def normalize_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()

def value_counts_dict(series: pd.Series) -> Dict[str, int]:
    values = series.astype(str).str.strip().replace({"": "<EMPTY>"})
    return {str(k): int(v) for k, v in values.value_counts(dropna=False).items()}

def metadata_audit(path: Path, label: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"label": label, "path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    df = read_csv(path, label)
    columns = {
        "sample_id": find_column(df, ["libraryid", "sample_id", "library_id"]),
        "patient_id": find_column(df, ["patient", "patient_id"]),
        "cohort": find_column(df, ["cohort", "group"]),
        "batch": find_column(df, ["batch"]),
        "material": find_column(df, ["material"]),
        "sex": find_column(df, ["sex", "gender"]),
        "age": find_column(df, ["age"]),
    }
    result.update({
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "resolved_columns": columns,
        "missing_by_column": {str(k): int(v) for k, v in df.isna().sum().items() if int(v) > 0},
    })
    for key in ["sample_id", "patient_id"]:
        col = columns[key]
        if col:
            ids = normalize_ids(df[col])
            result[f"{key}_unique"] = int(ids.nunique(dropna=False))
            result[f"{key}_duplicates"] = sorted(
                ids[ids.duplicated(keep=False)].unique().tolist()
            )[:100]
            result[f"{key}_empty"] = int(ids.eq("").sum())
    for key in ["cohort", "batch", "material", "sex"]:
        col = columns[key]
        if col:
            result[f"{key}_counts"] = value_counts_dict(df[col])
    age_col = columns["age"]
    if age_col:
        age = pd.to_numeric(df[age_col], errors="coerce")
        result["age_summary"] = {
            "missing_or_invalid": int(age.isna().sum()),
            "min": None if age.dropna().empty else float(age.min()),
            "median": None if age.dropna().empty else float(age.median()),
            "mean": None if age.dropna().empty else float(age.mean()),
            "max": None if age.dropna().empty else float(age.max()),
        }
    return result

def compare_metadata_sets(
    full_info: Dict[str, Any],
    train_info: Dict[str, Any],
    test_info: Dict[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not all(x.get("exists") for x in [full_info, train_info, test_info]):
        return {"status": "INCOMPLETE_INPUT"}
    frames = {}
    for info in [full_info, train_info, test_info]:
        path = Path(info["path"])
        df = read_csv(path, info["label"])
        sample_col = info["resolved_columns"].get("sample_id")
        patient_col = info["resolved_columns"].get("patient_id")
        frames[info["label"]] = {
            "sample": set(normalize_ids(df[sample_col])) if sample_col else set(),
            "patient": set(normalize_ids(df[patient_col])) if patient_col else set(),
        }
    for key in ["sample", "patient"]:
        full = frames["full"][key]
        train = frames["train"][key]
        test = frames["test"][key]
        result[key] = {
            "train_test_overlap_count": len(train & test),
            "train_test_overlap_examples": sorted(train & test)[:50],
            "union_equals_full": (train | test) == full if full else None,
            "only_full_count": len(full - (train | test)),
            "only_train_or_test_count": len((train | test) - full),
            "full_count": len(full),
            "train_count": len(train),
            "test_count": len(test),
        }
    return result

def batch6_role_audit(train_info: Dict[str, Any], test_info: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for role, info in [("historical_train", train_info), ("historical_test", test_info)]:
        if not info.get("exists"):
            continue
        df = read_csv(Path(info["path"]), info["label"])
        batch_col = info["resolved_columns"].get("batch")
        cohort_col = info["resolved_columns"].get("cohort")
        sample_col = info["resolved_columns"].get("sample_id")
        patient_col = info["resolved_columns"].get("patient_id")
        if not batch_col:
            continue
        norm = df[batch_col].astype(str).str.strip().str.lower()
        mask = norm.isin({"batch6", "6", "batch_6", "batch 6"})
        for _, row in df.loc[mask].iterrows():
            rows.append({
                "historical_role": role,
                "batch_raw": str(row[batch_col]),
                "cohort": None if not cohort_col else str(row[cohort_col]),
                "sample_id": None if not sample_col else str(row[sample_col]),
                "patient_id": None if not patient_col else str(row[patient_col]),
            })
    return {
        "n_batch6_records": len(rows),
        "roles": dict(Counter(x["historical_role"] for x in rows)),
        "records": rows,
        "interpretation_required": (
            "Determine whether the historical fixed-train rule was a source constraint, "
            "an independent-test design constraint, or a temporary balancing rule."
        ),
    }

def file_record(path: Path, root: Path) -> Dict[str, Any]:
    rec = {
        "path": str(path),
        "relative_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }
    if path.is_file():
        stat = path.stat()
        rec.update({
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": sha256sum(path),
        })
    return rec

def clone_dir_audit(path: Path, suffix: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "exists": path.is_dir(), "suffix": suffix}
    if not path.is_dir():
        return result
    files = sorted(path.glob(f"*{suffix}"))
    all_csv = sorted(path.glob("*.csv"))
    ids = [f.name[:-len(suffix)] for f in files]
    result.update({
        "matching_file_count": len(files),
        "all_csv_count": len(all_csv),
        "sample_ids": ids,
        "duplicate_sample_ids": sorted([x for x, n in Counter(ids).items() if n > 1]),
        "nonmatching_csv_examples": [f.name for f in all_csv if not f.name.endswith(suffix)][:100],
        "total_size_bytes": int(sum(f.stat().st_size for f in files)),
    })
    return result

def matrix_audit(path: Path, label: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "label": label, "exists": path.is_file()}
    if not path.is_file():
        return result
    df = read_csv(path, label)
    id_col = find_column(df, ["sample_id", "libraryid", "library_id"])
    result.update({
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(map(str, df.columns)),
        "duplicated_column_names": list(map(str, df.columns[df.columns.duplicated()])),
        "missing_cells": int(df.isna().sum().sum()),
        "sample_id_column": id_col,
    })
    if id_col:
        ids = normalize_ids(df[id_col])
        result["unique_sample_ids"] = int(ids.nunique(dropna=False))
        result["duplicate_sample_ids"] = sorted(
            ids[ids.duplicated(keep=False)].unique().tolist()
        )[:100]
    return result

def compare_matrix_columns(train: Dict[str, Any], test: Dict[str, Any]) -> Dict[str, Any]:
    if not train.get("exists") or not test.get("exists"):
        return {"status": "INCOMPLETE_INPUT"}
    train_cols = train["column_names"]
    test_cols = test["column_names"]
    return {
        "exact_same_order": train_cols == test_cols,
        "same_set": set(train_cols) == set(test_cols),
        "only_train": [x for x in train_cols if x not in set(test_cols)],
        "only_test": [x for x in test_cols if x not in set(train_cols)],
        "first_order_mismatches": [
            {"index": i, "train": a, "test": b}
            for i, (a, b) in enumerate(zip(train_cols, test_cols))
            if a != b
        ][:50],
    }

def manifest_audit(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    df = read_csv(path, "feature manifest")
    fmap = lower_map(df.columns)
    feature_col = fmap.get("feature_name")
    group_col = fmap.get("feature_group")
    role_col = fmap.get("feature_role")
    present_col = fmap.get("present_in_train_base")
    ref_col = fmap.get("reference_drop")
    result.update({
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "resolved_columns": {
            "feature_name": feature_col,
            "feature_group": group_col,
            "feature_role": role_col,
            "present_in_train_base": present_col,
            "reference_drop": ref_col,
        },
    })
    if group_col:
        result["feature_group_counts"] = value_counts_dict(df[group_col])
    if role_col:
        result["feature_role_counts"] = value_counts_dict(df[role_col])
    if feature_col and group_col and role_col:
        group = df[group_col].astype(str)
        role = df[role_col].astype(str)
        mask = group.str.startswith("igh_") & role.eq("candidate_predictor")
        if present_col:
            present = df[present_col].astype(str).str.lower().isin({"true", "1", "yes"})
            mask &= present
        if ref_col:
            ref = df[ref_col].astype(str).str.lower().isin({"true", "1", "yes"})
            mask &= ~ref
        static = df.loc[mask, feature_col].astype(str).tolist()
        result["derived_static_candidate_count"] = len(static)
        result["derived_static_candidates"] = static
    return result

def json_audit(path: Path, label: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "label": label, "exists": path.is_file()}
    if not path.is_file():
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        result["json"] = data
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result

def catalog_audit(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    df = read_csv(path, "public catalog")
    result.update({
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
    })
    return result

def sparse_audit(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    try:
        from scipy import sparse
        matrix = sparse.load_npz(path)
        result.update({
            "shape": [int(x) for x in matrix.shape],
            "nnz": int(matrix.nnz),
            "dtype": str(matrix.dtype),
            "sum": float(matrix.sum()),
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result

def assignments_audit(path: Path, compressed: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    df = read_csv(path, "assignments")
    result.update({
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "missing_cells": int(df.isna().sum().sum()),
    })
    for col in ["outer_repeat", "outer_fold", "inner_fold", "sample_id", "role"]:
        if col in df.columns:
            result[f"{col}_unique"] = int(df[col].nunique(dropna=False))
    return result

def script_inventory(root: Path, main_ref: str, trb_ref: str) -> Dict[str, Any]:
    return {
        "main_igh_scripts": git_lines(
            root, "ls-tree", "-r", "--name-only", main_ref, "--", "IGH/set/scripts"
        ),
        "main_trb_scripts": git_lines(
            root, "ls-tree", "-r", "--name-only", main_ref, "--", "TRB/set/scripts"
        ),
        "trb_configs": git_lines(
            root, "ls-tree", "-r", "--name-only", trb_ref, "--", "TRB/set/configs"
        ),
        "trb_scripts_v2": git_lines(
            root, "ls-tree", "-r", "--name-only", trb_ref, "--", "TRB/set/scripts_v2"
        ),
        "trb_src": git_lines(
            root, "ls-tree", "-r", "--name-only", trb_ref, "--", "TRB/set/src/ra_ild_trb"
        ),
        "trb_tests": git_lines(
            root, "ls-tree", "-r", "--name-only", trb_ref, "--", "TRB/set/tests"
        ),
    }

def flatten_json(prefix: str, value: Any, rows: List[Dict[str, Any]]) -> None:
    if isinstance(value, Mapping):
        for key, sub in value.items():
            flatten_json(f"{prefix}.{key}" if prefix else str(key), sub, rows)
    elif isinstance(value, list):
        if len(value) <= 20 and all(not isinstance(x, (dict, list)) for x in value):
            rows.append({"item": prefix, "value": json.dumps(value, ensure_ascii=False)})
        else:
            rows.append({"item": prefix, "value": f"<list length={len(value)}>"})
    else:
        rows.append({"item": prefix, "value": value})

def module_mapping_rows() -> List[Dict[str, str]]:
    return [
        {"trb_final_module":"config.py","igh_existing_source":"Hard-coded constants across build_04, run_05, collectors and final scripts","igh_migration_action":"PORT+ADAPT: create ra_ild_igh config contract; receptor/path checks; no hard-coded sample/public counts in repeated holdout"},
        {"trb_final_module":"paths.py","igh_existing_source":"Absolute IGH ROOT declarations in original scripts","igh_migration_action":"PORT: generic path resolver with mandatory IGH root and no TRB path leakage"},
        {"trb_final_module":"feature_inputs.py","igh_existing_source":"No equivalent reusable module; one-to-one merges embedded in build_04","igh_migration_action":"PORT GENERIC + TEST: CSV/CSV.GZ additional tables, exact IDs, collision/NA/Inf/type audit"},
        {"trb_final_module":"metrics.py","igh_existing_source":"Metric helpers in run_05/run_07/run_08","igh_migration_action":"PORT MOSTLY UNCHANGED; retain positive class ILD and arbitrary holdout size"},
        {"trb_final_module":"thresholds.py","igh_existing_source":"Youden/max-F1/sensitivity helpers in run_05/run_07","igh_migration_action":"PORT; threshold must come only from inner OOF"},
        {"trb_final_module":"preprocessing.py","igh_existing_source":"build_design and preprocessing rows inside run_05_single_outer_trial_loo.py","igh_migration_action":"PORT+REGRESSION: training-only encoding, zero variance and scaling"},
        {"trb_final_module":"modeling.py","igh_existing_source":"fit_elastic_net and model metrics inside run_05","igh_migration_action":"PORT+REGRESSION: same SAGA Elastic Net parameterization and deterministic seeds"},
        {"trb_final_module":"public_reference.py","igh_existing_source":"build_03 outputs plus PublicReference/build/LOO/external functions in run_05","igh_migration_action":"PORT+IGH CONTRACT: IGH suffix, frequency_norm, partition-only masks, exact LOO; historical sizes regression-only"},
        {"trb_final_module":"specifications.py","igh_existing_source":"MODEL_LABELS/model_specifications/static_igh_features_from_manifest in run_05","igh_migration_action":"REWRITE CONFIG LAYER: igh_* manifest groups, arbitrary named Elastic Net feature combinations"},
        {"trb_final_module":"nested_cv.py","igh_existing_source":"run_05_single_outer_trial_loo.py","igh_migration_action":"PORT SCIENTIFIC ENGINE; replace fixed four-model/100-task assumptions with resolved scheme"},
        {"trb_final_module":"outer_cv.py","igh_existing_source":"batch runner, status collector and step-06 scripts","igh_migration_action":"PORT+GENERALIZE: variable repeats/model count/holdout size, resumable structural checks"},
        {"trb_final_module":"scheme_management.py","igh_existing_source":"No original IGH equivalent","igh_migration_action":"PORT GENERIC; unique scheme ID, resolved YAML, isolated outputs, no silent overwrite"},
        {"trb_final_module":"repeated_holdout.py","igh_existing_source":"Historical build_05_cv_splits.py plus metadata split rules","igh_migration_action":"PORT+REDESIGN AFTER AUDIT: full 169 cohort, patient-level frozen splits, batch6 policy explicit"},
        {"trb_final_module":"repeated_holdout_training.py","igh_existing_source":"build_04 + run_05 cache/public/nested CV logic","igh_migration_action":"PORT+IGH REBUILD: combine old roles into full static matrix and full sparse bundle; exclude old test public predictors"},
        {"trb_final_module":"repeated_holdout_summary.py","igh_existing_source":"collect_06/analyze_06/plot_06/review parameter scripts","igh_migration_action":"PORT+GENERALIZE: repeated holdout membership, variable sample coverage, descriptive ranking only"},
        {"trb_final_module":"release.py","igh_existing_source":"No unified original IGH release checker","igh_migration_action":"PORT GENERIC + IGH checks: SHM test, receptor/path audit, hashes and completion marker"},
        {"trb_final_module":"AA aggregation regression","igh_existing_source":"build_01_AA_clone_table.py","igh_migration_action":"REUSE SCIENTIFIC SEMANTICS; add synthetic two-NT-to-one-AA SHM regression test"},
    ]

def make_markdown(audit: Dict[str, Any], mapping_rows: List[Dict[str, str]]) -> str:
    repo = audit["repository"]
    meta = audit["metadata"]
    resources = audit["resources"]
    lines = [
        "# IGH Upgrade Batch 00 Audit",
        "",
        f"- Generated: `{audit['generated_at']}`",
        f"- Audit script: `{SCRIPT_VERSION}`",
        f"- Project root: `{audit['project_root']}`",
        "- Mode: **read-only**",
        "",
        "## 1. Repository",
        "",
        f"- Current branch: `{repo.get('current_branch')}`",
        f"- Current HEAD: `{repo.get('current_head')}`",
        f"- main ref: `{repo.get('main_head')}`",
        f"- TRB framework ref: `{repo.get('trb_head')}`",
        f"- Expected TRB head match: `{repo.get('trb_head_matches_expected')}`",
        f"- Git worktree clean: `{repo.get('worktree_clean')}`",
        "",
        "## 2. Metadata",
        "",
    ]
    for key in ["full", "train", "test"]:
        info = meta[key]
        lines.append(
            f"- {key}: exists=`{info.get('exists')}`, rows=`{info.get('rows')}`, "
            f"unique samples=`{info.get('sample_id_unique')}`, "
            f"unique patients=`{info.get('patient_id_unique')}`"
        )
    lines.extend([
        "",
        f"- Train/test/full set audit: `{meta['set_comparison']}`",
        f"- Batch6 audit: `{meta['batch6']}`",
        "",
        "## 3. Core resources",
        "",
    ])
    for key in [
        "train_aa", "test_aa", "train_02", "test_02", "train_base",
        "test_final", "feature_manifest", "public_catalog", "cache_presence",
        "cache_frequency", "reference_definition", "final_model_config",
        "independent_test_config",
    ]:
        value = resources.get(key, {})
        lines.append(
            f"- {key}: exists=`{value.get('exists')}`, "
            f"rows/shape=`{value.get('rows', value.get('shape'))}`, "
            f"columns/nnz=`{value.get('columns', value.get('nnz'))}`"
        )
    lines.extend([
        "",
        "## 4. TRB → IGH migration mapping",
        "",
        "| TRB final module | IGH existing source | Migration action |",
        "|---|---|---|",
    ])
    for row in mapping_rows:
        lines.append(
            f"| `{row['trb_final_module']}` | {row['igh_existing_source']} | "
            f"{row['igh_migration_action']} |"
        )
    lines.extend([
        "",
        "## 5. Decisions still requiring user confirmation",
        "",
        "1. Initial repeated-holdout repeat count.",
        "2. Train/holdout size after the full-cohort audit.",
        "3. Exact strata and balance columns.",
        "4. Batch6 holdout policy and scientific justification.",
        "5. Initial model set/scheme.",
        "6. Git branch base and future PR order.",
        "",
        "## 6. Gate",
        "",
        "Batch 01 must not start until the metadata/resource audit contains no "
        "unresolved sample-union, duplicate-patient, AA-file coverage, feature-column "
        "or receptor/path failures, and the six design decisions above are recorded.",
        "",
    ])
    return "\n".join(lines)

def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    if not (root / ".git").exists():
        raise FileNotFoundError(f"Not a Git working tree: {root}")
    igh_root = root / "IGH"
    if not igh_root.is_dir():
        raise FileNotFoundError(f"IGH project directory not found: {igh_root}")

    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else igh_root / "set/audit_batch00"
    )
    files = {
        "json": output / "IGH_upgrade_audit_raw.json",
        "flat": output / "IGH_upgrade_audit_items.csv",
        "scripts": output / "IGH_repository_file_inventory.csv",
        "mapping": output / "IGH_TRB_module_mapping.csv",
        "audit_md": output / "IGH_upgrade_audit.md",
        "plan_md": output / "IGH_upgrade_plan.md",
        "complete": output / "BATCH00_AUDIT_COMPLETE.json",
    }
    ensure_output(files.values(), args.overwrite)
    output.mkdir(parents=True, exist_ok=True)

    status = run_cmd(["git", "status", "--short", "--branch"], root)
    current_branch = git_value(root, "branch", "--show-current")
    current_head = git_value(root, "rev-parse", "HEAD")
    main_head = git_value(root, "rev-parse", args.main_ref)
    trb_head = git_value(root, "rev-parse", args.trb_ref)
    merge_base = git_value(root, "merge-base", args.main_ref, args.trb_ref)
    inventory = script_inventory(root, args.main_ref, args.trb_ref)

    repo_audit = {
        "git_status": status,
        "current_branch": current_branch,
        "current_head": current_head,
        "main_ref": args.main_ref,
        "main_head": main_head,
        "trb_ref": args.trb_ref,
        "trb_head": trb_head,
        "expected_trb_head": args.expected_trb_head,
        "trb_head_matches_expected": trb_head == args.expected_trb_head,
        "merge_base": merge_base,
        "trb_based_on_main": merge_base == main_head if main_head and merge_base else None,
        "worktree_clean": not any(
            line and not line.startswith("##")
            for line in status["stdout"].splitlines()
        ),
        "inventory": inventory,
    }

    metadata = {
        "full": metadata_audit(root / LIKELY_PATHS["full_metadata"], "full"),
        "train": metadata_audit(root / LIKELY_PATHS["train_metadata"], "train"),
        "test": metadata_audit(root / LIKELY_PATHS["test_metadata"], "test"),
    }
    metadata["set_comparison"] = compare_metadata_sets(
        metadata["full"], metadata["train"], metadata["test"]
    )
    metadata["batch6"] = batch6_role_audit(metadata["train"], metadata["test"])

    resources: Dict[str, Any] = {}
    resources["train_aa"] = clone_dir_audit(root / LIKELY_PATHS["train_aa_dir"], AA_SUFFIX)
    resources["test_aa"] = clone_dir_audit(root / LIKELY_PATHS["test_aa_dir"], AA_SUFFIX)
    resources["train_02"] = matrix_audit(root / LIKELY_PATHS["train_02"], "train step-02")
    resources["test_02"] = matrix_audit(root / LIKELY_PATHS["test_02"], "test step-02")
    resources["step02_column_comparison"] = compare_matrix_columns(
        resources["train_02"], resources["test_02"]
    )
    resources["train_3mer_vocab"] = matrix_audit(
        root / LIKELY_PATHS["train_3mer_vocab"], "train 3-mer vocabulary"
    )
    resources["train_base"] = matrix_audit(root / LIKELY_PATHS["train_base"], "train base")
    resources["test_final"] = matrix_audit(root / LIKELY_PATHS["test_final"], "test final")
    resources["feature_manifest"] = manifest_audit(root / LIKELY_PATHS["feature_manifest"])
    resources["public_catalog"] = catalog_audit(root / LIKELY_PATHS["public_catalog"])
    resources["reference_definition"] = json_audit(
        root / LIKELY_PATHS["reference_definition"], "reference definition"
    )
    resources["cache_presence"] = sparse_audit(root / LIKELY_PATHS["cache_presence"])
    resources["cache_frequency"] = sparse_audit(root / LIKELY_PATHS["cache_frequency"])
    resources["cache_metadata"] = json_audit(
        root / LIKELY_PATHS["cache_metadata"], "cache metadata"
    )
    resources["outer_assignments"] = assignments_audit(
        root / LIKELY_PATHS["outer_assignments"]
    )
    resources["inner_assignments"] = assignments_audit(
        root / LIKELY_PATHS["inner_assignments"], compressed=True
    )
    resources["final_model_config"] = json_audit(
        root / LIKELY_PATHS["final_model_config"], "final model configuration"
    )
    resources["independent_test_config"] = json_audit(
        root / LIKELY_PATHS["independent_test_config"],
        "independent test configuration",
    )
    resources["file_records"] = {
        key: file_record(root / rel, root) for key, rel in LIKELY_PATHS.items()
    }

    mapping = module_mapping_rows()
    audit = {
        "script_version": SCRIPT_VERSION,
        "generated_at": now_iso(),
        "project_root": str(root),
        "receptor": EXPECTED_RECEPTOR,
        "repository": repo_audit,
        "metadata": metadata,
        "resources": resources,
        "historical_values_are_regression_only": True,
        "core_data_modified": False,
    }

    files["json"].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    flat_rows: List[Dict[str, Any]] = []
    flatten_json("", audit, flat_rows)
    pd.DataFrame(flat_rows).to_csv(files["flat"], index=False)

    inventory_rows = []
    for category, paths in inventory.items():
        for path in paths:
            inventory_rows.append({"category": category, "path": path})
    pd.DataFrame(inventory_rows).to_csv(files["scripts"], index=False)
    pd.DataFrame(mapping).to_csv(files["mapping"], index=False)

    files["audit_md"].write_text(
        make_markdown(audit, mapping), encoding="utf-8"
    )
    plan = """# IGH Upgrade Plan after Batch 00

## Gate 1 — confirm the real cohort and resources
Use `IGH_upgrade_audit.md` and `IGH_upgrade_audit_raw.json` to resolve every
sample, patient, feature, public-reference and batch6 question.

## Gate 2 — record design choices
Record:
- branch base and PR order;
- repeat count;
- train/holdout size;
- exact strata;
- balance columns;
- batch6 policy;
- initial model set.

## Batch 01
Create the independent `ra_ild_igh` package, baseline YAML, validation tools,
IGH receptor contracts, unit tests and output isolation. Do not alter historical
`IGH/set/scripts/` or historical results.

## Batch 02
Add flexible model mappings, complete model replacement, additional feature
tables and feature audits.

## Batch 03
Generate and freeze the full-cohort patient-level repeated-holdout split set.

## Batch 04
Build the frozen full-cohort static/sparse training bundle, fixed inner folds,
and run the leakage-controlled repeated-holdout engine.

## Batch 05
Aggregate repeated holdouts and compare schemes only under the same split-set ID
and assignment SHA256.

## Batch 06
Run all regression/release checks, write completion provenance and prepare the
source-only Git commits/PR.
"""
    files["plan_md"].write_text(plan, encoding="utf-8")

    marker = {
        "status": "COMPLETE",
        "script_version": SCRIPT_VERSION,
        "generated_at": audit["generated_at"],
        "project_root": str(root),
        "output_dir": str(output),
        "core_data_modified": False,
        "output_sha256": {
            key: sha256sum(path)
            for key, path in files.items()
            if key != "complete" and path.is_file()
        },
    }
    files["complete"].write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("IGH Framework Upgrade — Batch 00 Read-only Audit")
    print("=" * 88)
    print(f"Project root:        {root}")
    print(f"Current branch:      {current_branch}")
    print(f"main:                {main_head}")
    print(f"TRB framework head:  {trb_head}")
    print(f"Expected TRB head:   {args.expected_trb_head}")
    print(f"TRB head verified:   {trb_head == args.expected_trb_head}")
    print(f"Worktree clean:      {repo_audit['worktree_clean']}")
    print("")
    print("[Metadata]")
    for key in ["full", "train", "test"]:
        info = metadata[key]
        print(
            f"- {key:5s}: exists={info.get('exists')}, rows={info.get('rows')}, "
            f"samples={info.get('sample_id_unique')}, patients={info.get('patient_id_unique')}"
        )
    print("")
    print("[Core resource snapshot]")
    print(f"- train AA tables:   {resources['train_aa'].get('matching_file_count')}")
    print(f"- test AA tables:    {resources['test_aa'].get('matching_file_count')}")
    print(f"- train step-02:     {resources['train_02'].get('rows')} × {resources['train_02'].get('columns')}")
    print(f"- test step-02:      {resources['test_02'].get('rows')} × {resources['test_02'].get('columns')}")
    print(f"- static candidates: {resources['feature_manifest'].get('derived_static_candidate_count')}")
    print(f"- public catalog:    {resources['public_catalog'].get('rows')}")
    print(f"- presence cache:    {resources['cache_presence'].get('shape')}, nnz={resources['cache_presence'].get('nnz')}")
    print(f"- frequency cache:   {resources['cache_frequency'].get('shape')}, nnz={resources['cache_frequency'].get('nnz')}")
    print("")
    print("[Outputs]")
    for key, path in files.items():
        print(f"- {key}: {path}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
