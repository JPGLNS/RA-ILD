#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only audit and metadata materialization for the complete TRB 180 cohort.

Default mode is audit-only. It never modifies the supplied metadata or raw NT
repertoire files. Metadata copies and the repeat-1 compatibility split are
written only when the corresponding mode is explicitly selected.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

REQUIRED = ("patient", "libraryid", "batch", "sex", "age", "cohort", "material")
RAW_SUFFIX = "_TRB_CDR3_NT_frequency_error_correct.csv"
DATA_VERSION = Path("TRB/set/data_versions/trb_v3_180")
EXPECTED_COHORT = {"RA": 111, "ILD": 69}
EXPECTED_MATERIAL = {"pbmc": 116, "buffycoat": 64}
EXPECTED_COHORT_MATERIAL = {
    ("RA", "pbmc"): 69,
    ("ILD", "pbmc"): 47,
    ("RA", "buffycoat"): 42,
    ("ILD", "buffycoat"): 22,
}
EXPECTED_COHORT_BATCH = {
    "total": {
        ("ILD", "batch1"): 5, ("ILD", "batch2"): 16, ("ILD", "batch3"): 14,
        ("ILD", "batch4"): 14, ("ILD", "batch5"): 17, ("ILD", "batch6"): 3,
        ("RA", "batch1"): 8, ("RA", "batch2"): 25, ("RA", "batch3"): 28,
        ("RA", "batch4"): 22, ("RA", "batch5"): 26, ("RA", "batch6"): 2,
    },
    "pbmc": {
        ("ILD", "batch1"): 5, ("ILD", "batch2"): 12, ("ILD", "batch3"): 14,
        ("ILD", "batch4"): 13, ("ILD", "batch6"): 3,
        ("RA", "batch1"): 8, ("RA", "batch2"): 11, ("RA", "batch3"): 28,
        ("RA", "batch4"): 20, ("RA", "batch6"): 2,
    },
    "buffycoat": {
        ("ILD", "batch2"): 4, ("ILD", "batch4"): 1, ("ILD", "batch5"): 17,
        ("RA", "batch2"): 14, ("RA", "batch4"): 2, ("RA", "batch5"): 26,
    },
}


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def normalize_material(series: pd.Series) -> pd.Series:
    values = (
        series.astype(str).str.strip().str.lower()
        .str.replace(r"[\s_-]", "", regex=True)
    )
    return values.replace({
        "buffcoat": "buffycoat",
        "buffercoat": "buffycoat",
        "buffycoat": "buffycoat",
        "pbmc": "pbmc",
    })


def read_metadata(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    frame.columns = frame.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise AuditError(f"{label} missing required columns: {missing}")
    frame = frame.loc[:, list(REQUIRED)].copy()
    for column in REQUIRED:
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or frame[column].eq("").any():
            raise AuditError(f"{label}: null/blank values in {column}")
    for column in ("patient", "libraryid"):
        duplicated = frame.loc[frame[column].duplicated(keep=False), column].unique().tolist()
        if duplicated:
            raise AuditError(f"{label}: duplicated {column}: {duplicated[:20]}")
    frame["cohort"] = frame["cohort"].str.upper().replace({
        "RA-ILD": "ILD", "RA_ILD": "ILD", "RAILD": "ILD"
    })
    frame["material"] = normalize_material(frame["material"])
    frame["batch"] = frame["batch"].str.lower()
    if not frame["cohort"].isin(["RA", "ILD"]).all():
        raise AuditError(f"{label}: cohort must contain only RA/ILD")
    if not frame["material"].isin(["pbmc", "buffycoat"]).all():
        observed = sorted(frame["material"].dropna().unique().tolist())
        raise AuditError(f"{label}: unexpected material values: {observed}")
    age = pd.to_numeric(frame["age"], errors="coerce")
    if age.isna().any():
        raise AuditError(f"{label}: age contains non-numeric values")
    frame["age"] = age.astype(float)
    return frame


def counts_dict(frame: pd.DataFrame, columns: list[str]) -> dict:
    return {
        tuple(map(str, key if isinstance(key, tuple) else (key,))): int(value)
        for key, value in frame.groupby(columns, dropna=False).size().items()
    }


def compare_counts(label: str, observed: dict, expected: dict) -> None:
    if observed != expected:
        missing = {key: value for key, value in expected.items() if observed.get(key) != value}
        extra = {key: value for key, value in observed.items() if key not in expected}
        raise AuditError(
            f"{label} counts differ from the supplied cohort table; "
            f"mismatched_expected={missing}; unexpected={extra}"
        )


def audit_metadata(frame: pd.DataFrame) -> None:
    if len(frame) != 180:
        raise AuditError(f"Expected 180 metadata rows, observed {len(frame)}")
    if frame["patient"].nunique() != 180 or frame["libraryid"].nunique() != 180:
        raise AuditError("Expected 180 unique patients and 180 unique library IDs")
    compare_counts(
        "cohort",
        {tuple([str(k)]): int(v) for k, v in frame["cohort"].value_counts().items()},
        {tuple([k]): v for k, v in EXPECTED_COHORT.items()},
    )
    compare_counts(
        "material",
        {tuple([str(k)]): int(v) for k, v in frame["material"].value_counts().items()},
        {tuple([k]): v for k, v in EXPECTED_MATERIAL.items()},
    )
    compare_counts(
        "cohort x material",
        counts_dict(frame, ["cohort", "material"]),
        EXPECTED_COHORT_MATERIAL,
    )
    for scope in ("total", "pbmc", "buffycoat"):
        part = frame if scope == "total" else frame.loc[frame["material"] == scope]
        compare_counts(
            f"{scope} cohort x batch",
            counts_dict(part, ["cohort", "batch"]),
            EXPECTED_COHORT_BATCH[scope],
        )


def audit_raw_files(frame: pd.DataFrame, raw_dir: Path) -> None:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw input directory not found: {raw_dir}")
    files = sorted(path for path in raw_dir.glob(f"*{RAW_SUFFIX}") if path.is_file())
    names = [path.name[:-len(RAW_SUFFIX)] for path in files]
    if len(names) != len(set(names)):
        raise AuditError("Raw input inventory contains duplicate sample IDs")
    expected = set(frame["libraryid"].astype(str))
    found = set(names)
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    empty = [path.name for path in files if path.stat().st_size == 0]
    print("\n[RAW INPUT INVENTORY]")
    print(f"  directory={raw_dir}")
    print(f"  suffix={RAW_SUFFIX}")
    print(f"  files={len(files)} expected={len(expected)} missing={len(missing)} extra={len(extra)} empty={len(empty)}")
    if missing:
        print("  missing:", missing[:30])
    if extra:
        print("  extra:", extra[:30])
    if empty:
        print("  empty:", empty[:30])
    if missing or extra or empty or len(files) != 180:
        raise AuditError("Raw NT input inventory does not exactly match the 180 metadata samples")

    malformed = []
    for path in files:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            first = next((line.strip() for line in handle if line.strip()), "")
        fields = [value for value in re.split(r"[,\t\s]+", first) if value]
        if len(fields) != 4:
            malformed.append((path.name, len(fields), first[:120]))
    if malformed:
        print("  malformed first records:", malformed[:10])
        raise AuditError(
            "Some raw files do not look like the expected four-column, headerless Step01 input"
        )
    print("  first non-empty record: four fields in every file")


def compare_old_metadata(root: Path, args: argparse.Namespace, new: pd.DataFrame) -> None:
    old_paths = [
        resolve(root, args.old_train_metadata),
        resolve(root, args.old_test_metadata),
    ]
    if not all(path.is_file() for path in old_paths):
        print("\n[OLD COHORT COMPARISON] skipped: one or both old metadata files are absent")
        return
    old = pd.concat(
        [read_metadata(path, f"old metadata {index}") for index, path in enumerate(old_paths, 1)],
        ignore_index=True,
    )
    if old["patient"].duplicated().any() or old["libraryid"].duplicated().any():
        raise AuditError("Old 174-sample train/test metadata contain duplicated patients or library IDs")
    old_patients = set(old["patient"].astype(str))
    new_patients = set(new["patient"].astype(str))
    added = sorted(new_patients - old_patients)
    removed = sorted(old_patients - new_patients)
    print("\n[OLD 174 -> COMPLETE 180 RECONCILIATION]")
    print(f"  old={len(old)} new={len(new)} added_patients={len(added)} removed_patients={len(removed)}")
    if added:
        print("  added:", added)
    if removed:
        print("  removed:", removed)
    old_by_patient = old.set_index("patient")
    new_by_patient = new.set_index("patient")
    shared = sorted(old_patients & new_patients)
    id_changes = [
        (patient, old_by_patient.loc[patient, "libraryid"], new_by_patient.loc[patient, "libraryid"])
        for patient in shared
        if old_by_patient.loc[patient, "libraryid"] != new_by_patient.loc[patient, "libraryid"]
    ]
    annotation_changes = []
    for patient in shared:
        for column in ("age", "sex", "cohort", "material", "batch"):
            if str(old_by_patient.loc[patient, column]) != str(new_by_patient.loc[patient, column]):
                annotation_changes.append((
                    patient, column, str(old_by_patient.loc[patient, column]),
                    str(new_by_patient.loc[patient, column])
                ))
    print(f"  library ID changes={len(id_changes)}; annotation changes={len(annotation_changes)}")
    for item in id_changes[:20]:
        print("  ID_CHANGE", *item)
    for item in annotation_changes[:30]:
        print("  META_CHANGE", *item)
    if removed or id_changes or annotation_changes:
        raise AuditError(
            "Old-to-new cohort reconciliation requires review before materializing the new data version"
        )


def write_csv(path: Path, frame: pd.DataFrame, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing metadata: {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8")
    temp.replace(path)
    print(f"[WRITE] {path} rows={len(frame)} sha256={sha256(path)}")


def materialize_metadata(root: Path, output_root: Path, frame: pd.DataFrame, overwrite: bool) -> None:
    metadata_dir = output_root / "metadata"
    write_csv(metadata_dir / "metadata_total.csv", frame, overwrite)
    write_csv(
        metadata_dir / "metadata_pbmc.csv",
        frame.loc[frame["material"] == "pbmc"].copy(),
        overwrite,
    )
    write_csv(
        metadata_dir / "metadata_buffycoat.csv",
        frame.loc[frame["material"] == "buffycoat"].copy(),
        overwrite,
    )


def materialize_repeat1(
    root: Path,
    output_root: Path,
    frame: pd.DataFrame,
    assignments_path: Path,
    overwrite: bool,
) -> None:
    if not assignments_path.is_file():
        raise FileNotFoundError(f"Repeated-holdout assignments not found: {assignments_path}")
    assignments = pd.read_csv(assignments_path, dtype={"sample_id": str, "role": str})
    required = {"repeat_index", "sample_id", "role"}
    if not required.issubset(assignments.columns):
        raise AuditError(f"Assignments missing columns: {sorted(required - set(assignments.columns))}")
    repeat = assignments.loc[pd.to_numeric(assignments["repeat_index"], errors="coerce") == 1].copy()
    if repeat.empty:
        raise AuditError("Assignments contain no repeat_index=1 rows")
    if repeat["sample_id"].duplicated().any():
        raise AuditError("Repeat 1 contains duplicate sample IDs")
    ids = set(frame["libraryid"].astype(str))
    assigned = set(repeat["sample_id"].astype(str))
    if ids != assigned:
        raise AuditError(
            f"Repeat 1 sample coverage mismatch: missing={sorted(ids-assigned)[:20]}, "
            f"extra={sorted(assigned-ids)[:20]}"
        )
    roles = repeat.set_index("sample_id")["role"].astype(str).str.lower()
    train_ids = set(roles.index[roles == "train"])
    holdout_ids = set(roles.index[roles == "holdout"])
    if train_ids & holdout_ids or train_ids | holdout_ids != ids:
        raise AuditError("Repeat 1 train/holdout roles do not form a disjoint full-cohort partition")
    if (len(train_ids), len(holdout_ids)) != (126, 54):
        raise AuditError(
            f"Expected repeat 1 train/holdout sizes 126/54, observed {len(train_ids)}/{len(holdout_ids)}"
        )
    train = frame.loc[frame["libraryid"].astype(str).isin(train_ids)].copy()
    holdout = frame.loc[frame["libraryid"].astype(str).isin(holdout_ids)].copy()
    out = output_root / "ml_compat" / "total"
    write_csv(out / "metadata_train_70.csv", train, overwrite)
    write_csv(out / "metadata_test_30.csv", holdout, overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--metadata", default="TRB/metadata_TCR_with_batch.csv")
    parser.add_argument("--raw-dir", default="TRB/完整数据/origin_result")
    parser.add_argument("--old-train-metadata", default="TRB/set/train/metadata_train_70.csv")
    parser.add_argument("--old-test-metadata", default="TRB/set/test/metadata_test_30.csv")
    parser.add_argument("--output-root", default=str(DATA_VERSION))
    parser.add_argument(
        "--mode",
        choices=("audit", "write-metadata", "materialize-repeat1"),
        default="audit",
    )
    parser.add_argument(
        "--split-assignments",
        default="TRB/set/split_sets/ra_ild_repeat100_v3_180/repeated_holdout_assignments.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo.expanduser().resolve()
    metadata_path = resolve(root, args.metadata)
    raw_dir = resolve(root, args.raw_dir)
    output_root = resolve(root, args.output_root)

    print("TRB V3 COMPLETE-COHORT INPUT AUDIT")
    print("Repository:", root)
    print("Metadata:", metadata_path)
    print("Output data version:", output_root)

    frame = read_metadata(metadata_path, "complete TRB metadata")
    audit_metadata(frame)
    print("\n[METADATA PASS]")
    print("  rows/patients/libraries:", len(frame), frame["patient"].nunique(), frame["libraryid"].nunique())
    print("  cohort:", frame["cohort"].value_counts().to_dict())
    print("  material:", frame["material"].value_counts().to_dict())
    print("  cohort x material:", frame.groupby(["cohort", "material"]).size().to_dict())
    for scope in ("total", "pbmc", "buffycoat"):
        part = frame if scope == "total" else frame.loc[frame["material"] == scope]
        print(f"  {scope} cohort x batch:", part.groupby(["cohort", "batch"]).size().to_dict())

    compare_old_metadata(root, args, frame)
    audit_raw_files(frame, raw_dir)

    if args.mode == "write-metadata":
        materialize_metadata(root, output_root, frame, args.overwrite)
    elif args.mode == "materialize-repeat1":
        materialize_repeat1(
            root,
            output_root,
            frame,
            resolve(root, args.split_assignments),
            args.overwrite,
        )
    else:
        print("\n[AUDIT ONLY] No files were written or modified.")

    print("\nTRB V3 180 INPUT AUDIT PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"TRB V3 180 INPUT AUDIT FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
