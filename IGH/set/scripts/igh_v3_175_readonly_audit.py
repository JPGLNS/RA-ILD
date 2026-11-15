#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only IGH complete cohort (175) vs frozen v2 (172) source audit.

This program does not change metadata, splits, clone tables, or model results.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys

import pandas as pd

REQUIRED = ("patient", "libraryid", "batch", "sex", "age", "cohort", "material")
SUFFIX = "_IGH_CDR3_NT_frequency_error_correct.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_material(s: pd.Series) -> pd.Series:
    v = s.astype(str).str.lower().str.replace(r"[\s_-]", "", regex=True)
    return v.replace({"buffcoat": "buffycoat", "buffercoat": "buffycoat"})


def read_metadata(path: Path, title: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{title}: {path}")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"{title}: missing columns: {missing}")
    df = df.loc[:, list(REQUIRED)].copy()
    for col in REQUIRED:
        df[col] = df[col].astype("string").str.strip()
        if df[col].isna().any() or df[col].eq("").any():
            raise ValueError(f"{title}: null/blank column {col}")
    for key in ("patient", "libraryid"):
        if df[key].duplicated().any():
            raise ValueError(f"{title}: duplicate {key}: {df.loc[df[key].duplicated(), key].tolist()[:10]}")
    df["cohort"] = df["cohort"].str.upper().replace({"RA-ILD": "ILD", "RA_ILD": "ILD", "RAILD": "ILD"})
    df["material"] = norm_material(df["material"])
    if not df["cohort"].isin(["RA", "ILD"]).all():
        raise ValueError(f"{title}: unknown cohort labels")
    if not df["material"].isin(["pbmc", "buffycoat"]).all():
        raise ValueError(f"{title}: unknown material labels")
    if pd.to_numeric(df["age"], errors="coerce").isna().any():
        raise ValueError(f"{title}: non-numeric age")
    print(f"[{title}] rows={len(df)}; patients={df.patient.nunique()}; libraries={df.libraryid.nunique()}")
    print("  cohort:", df.cohort.value_counts().to_dict())
    print("  material:", df.material.value_counts().to_dict())
    print("  cohort x material:", df.groupby(["cohort", "material"]).size().to_dict())
    print("  metadata SHA256:", sha256(path))
    return df


def report_diff(old: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    issues = []
    o = old.set_index("patient"); n = new.set_index("patient")
    added = sorted(set(n.index) - set(o.index))
    removed = sorted(set(o.index) - set(n.index))
    print("\n[Patient-level cohort reconciliation]")
    print("  Added patients:", len(added))
    for patient in added:
        r = n.loc[patient]
        print(f"    + {patient}: {r.libraryid}, {r.cohort}, {r.material}, {r.batch}")
    print("  Removed patients:", len(removed))
    for patient in removed:
        print(f"    - {patient}: {o.loc[patient, 'libraryid']}")
    if removed:
        issues.append("old patients not present in complete metadata")
    shared = sorted(set(o.index) & set(n.index))
    library_changes = []
    annotation_changes = []
    for patient in shared:
        old_row = o.loc[patient]; new_row = n.loc[patient]
        if old_row.libraryid != new_row.libraryid:
            library_changes.append((patient, old_row.libraryid, new_row.libraryid))
        for field in ("age", "sex", "cohort", "material", "batch"):
            if str(old_row[field]) != str(new_row[field]):
                annotation_changes.append((patient, field, old_row[field], new_row[field]))
    print("  Library ID replacements (same patient):", len(library_changes))
    for vals in library_changes:
        print("    ID_CHANGE", *vals)
    print("  Annotation field changes:", len(annotation_changes))
    for vals in annotation_changes[:40]:
        print("    META_CHANGE", *vals)
    if len(annotation_changes) > 40:
        print(f"    ...and {len(annotation_changes)-40} further changes")
    if library_changes:
        issues.append("patient library IDs replaced; check mapping/provenance")
    if annotation_changes:
        issues.append("metadata fields changed; review source metadata")
    old_ids = set(old.libraryid); new_ids = set(new.libraryid)
    print("  Added library IDs:", len(new_ids-old_ids), sorted(new_ids-old_ids))
    print("  Removed library IDs:", len(old_ids-new_ids), sorted(old_ids-new_ids))
    print("  Reused library IDs:", len(old_ids & new_ids))
    return issues


def print_singletons(new: pd.DataFrame) -> None:
    print("\n[Exact-stratum singleton audit: cohort x batch]")
    for scope in ("total", "pbmc", "buffycoat"):
        part = new if scope == "total" else new.loc[new["material"] == scope]
        counts = part.groupby(["cohort", "batch"]).size()
        singles = counts[counts == 1]
        print(f"  {scope}: {len(singles)} singleton strata")
        for cohort, batch in singles.index:
            sid = part.loc[(part["cohort"] == cohort) & (part["batch"] == batch), "libraryid"].iloc[0]
            print(f"    fixed_train candidate {cohort}/{batch}: {sid}")


def check_raw(new: pd.DataFrame, raw_dir: Path, suffix: str, old: pd.DataFrame) -> list[str]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw directory not found: {raw_dir}")
    files = list(raw_dir.glob("*" + suffix))
    names = [p.name[:-len(suffix)] for p in files]
    if len(set(names)) != len(names):
        raise ValueError("duplicate source IDs in file inventory")
    expected = set(new.libraryid.astype(str)); found = set(names)
    missing = sorted(expected-found); extra = sorted(found-expected)
    empty = [p.name for p in files if p.stat().st_size <= 0]
    print("\n[Complete source file inventory]")
    print("  raw directory:", raw_dir)
    print("  source suffix:", suffix)
    print(f"  matching files={len(files)}; expected={len(expected)}; missing={len(missing)}; extra={len(extra)}; empty={len(empty)}")
    if missing: print("  Missing:", missing)
    if extra: print("  Extra:", extra)
    if empty: print("  Empty:", empty)
    issues=[]
    if missing or extra or empty:
        issues.append("raw input inventory missing/extra/empty")
    for sample in sorted(set(sorted(expected)[:2]) | (expected - set(old.libraryid.astype(str)))):
        p=raw_dir/(sample+suffix)
        if p.is_file():
            with p.open("r", encoding="utf-8-sig", errors="replace") as fh:
                line=next((s.strip() for s in fh if s.strip()), "")
            fields=[x for x in re.split(r"[,\t\s]+", line) if x]
            print(f"  Quick schema check: {sample}, first nonempty record fields={len(fields)}; numeric second field={len(fields)>1 and fields[1].replace('.', '', 1).isdigit()}")
    return issues


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--old-metadata", default="IGH/set/data_versions/igh_v2_172/metadata.csv")
    ap.add_argument("--full-metadata", default="IGH/metadata_BCR_with_batch.csv")
    ap.add_argument("--raw-dir", default="IGH/完整数据/origin_result")
    ap.add_argument("--raw-suffix", default=SUFFIX)
    args=ap.parse_args()
    root=args.repo.resolve()
    def full(p):
        p=Path(p)
        return p if p.is_absolute() else root/p
    print("IGH COMPLETE-COHORT PREFLIGHT (READ ONLY)")
    print("Repository:",root)
    old=read_metadata(full(args.old_metadata),"v2 old cohort")
    new=read_metadata(full(args.full_metadata),"complete cohort")
    issues=report_diff(old,new)
    print_singletons(new)
    issues+=check_raw(new,full(args.raw_dir),args.raw_suffix,old)
    print("\n[Data-version recommendation]")
    print("  Preserve old: IGH/set/data_versions/igh_v2_172")
    print("  New proposed version: IGH/set/data_versions/igh_v3_175")
    print("  IMPORTANT: the default Step01 input suffix is _IGH-without-DJ_CDR3_NT_frequency_error_correct.csv;")
    print("  complete input uses _IGH_CDR3_NT_frequency_error_correct.csv. Validate source semantics before Step01.")
    if len(new)!=175 or len(old)!=172:
        issues.append("unexpected cohort counts: expect 172 -> 175")
    if issues:
        print("  REVIEW REQUIRED:", '; '.join(issues))
        return 2
    print("  METADATA/FILE INVENTORY PASS; review source schema and data provenance next.")
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("PREFLIGHT FAIL:",exc,file=sys.stderr)
        sys.exit(1)
