#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare and audit the additive 179-sample TRB Top-10000 data version.

The 180-sample Step01 tables remain immutable. This script creates a metadata
view and a symlink-only whitelist of 179 Step01 AA tables, audits the exact-K
Top10000 outputs, and materializes repeat-1 train/holdout metadata for the
existing Step02-Step04 builders.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

DATA_VERSION = Path("TRB/set/data_versions/trb_v3_179_top10000")
SOURCE_METADATA = Path("TRB/set/data_versions/trb_v3_180/metadata/metadata_total.csv")
SOURCE_STEP01 = Path("TRB/set/data_versions/trb_v3_180/result/01_AA_clone_table")
EXCLUDED_LIBRARY_ID = "MGI260107R01-86"
EXCLUDED_PATIENT = "B1197705"
STEP01_SUFFIX = "_TRB_CDR3_AA_clone_table.csv"
TOPK_SUFFIX = "_TRB_CDR3_AA_clone_table_top10000.csv"
TOP_K = 10000
REQUIRED_METADATA = ("patient", "libraryid", "batch", "sex", "age", "cohort", "material")
EXPECTED_TOTAL = {"RA": 110, "ILD": 69}
EXPECTED_MATERIAL = {"pbmc": 116, "buffycoat": 63}
EXPECTED_COHORT_MATERIAL = {
    ("RA", "pbmc"): 69,
    ("ILD", "pbmc"): 47,
    ("RA", "buffycoat"): 41,
    ("ILD", "buffycoat"): 22,
}


class PreparationError(RuntimeError):
    """Raised when the 179-sample data-version contract is violated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def normalize_metadata(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = frame.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    missing = sorted(set(REQUIRED_METADATA) - set(frame.columns))
    if missing:
        raise PreparationError(f"{label} is missing required metadata columns: {missing}")
    frame = frame.loc[:, list(REQUIRED_METADATA)].copy()
    for column in REQUIRED_METADATA:
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or frame[column].eq("").any():
            raise PreparationError(f"{label}: blank/null values in {column}")
    frame["cohort"] = frame["cohort"].str.upper().replace(
        {"RA-ILD": "ILD", "RA_ILD": "ILD", "RAILD": "ILD"}
    )
    frame["material"] = (
        frame["material"].str.lower().str.replace(r"[\s_-]", "", regex=True)
        .replace({"buffcoat": "buffycoat", "buffercoat": "buffycoat"})
    )
    frame["batch"] = frame["batch"].str.lower()
    if not frame["cohort"].isin(["RA", "ILD"]).all():
        raise PreparationError(f"{label}: cohort must contain only RA/ILD")
    if not frame["material"].isin(["pbmc", "buffycoat"]).all():
        raise PreparationError(f"{label}: material must contain only pbmc/buffycoat")
    age = pd.to_numeric(frame["age"], errors="coerce")
    if age.isna().any():
        raise PreparationError(f"{label}: age contains non-numeric values")
    frame["age"] = age.astype(float)
    for column in ("patient", "libraryid"):
        duplicates = frame.loc[frame[column].duplicated(keep=False), column].unique().tolist()
        if duplicates:
            raise PreparationError(f"{label}: duplicated {column}: {duplicates[:20]}")
    return frame


def _counts(frame: pd.DataFrame, columns: list[str]) -> dict[tuple[str, ...], int]:
    result = {}
    for key, value in frame.groupby(columns, dropna=False).size().items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        result[tuple(str(item) for item in key_tuple)] = int(value)
    return result


def build_filtered_metadata(
    source: pd.DataFrame,
    *,
    exclude_library_id: str = EXCLUDED_LIBRARY_ID,
    exclude_patient: str = EXCLUDED_PATIENT,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Remove exactly the approved sample after verifying its clinical identity."""
    frame = normalize_metadata(source, "180-sample source metadata")
    if len(frame) != 180 or frame["patient"].nunique() != 180 or frame["libraryid"].nunique() != 180:
        raise PreparationError(
            "The source metadata must contain exactly 180 unique patients/libraries"
        )
    selected = frame.loc[frame["libraryid"] == exclude_library_id]
    if len(selected) != 1:
        raise PreparationError(
            f"Expected exactly one excluded library {exclude_library_id}; observed {len(selected)}"
        )
    row = selected.iloc[0]
    observed = {
        "patient": str(row["patient"]),
        "cohort": str(row["cohort"]),
        "material": str(row["material"]),
        "batch": str(row["batch"]),
    }
    expected = {
        "patient": exclude_patient,
        "cohort": "RA",
        "material": "buffycoat",
        "batch": "batch2",
    }
    if observed != expected:
        raise PreparationError(
            f"Excluded sample identity mismatch: expected={expected}, observed={observed}"
        )
    filtered = frame.loc[frame["libraryid"] != exclude_library_id].copy().reset_index(drop=True)
    if len(filtered) != 179 or filtered["patient"].nunique() != 179:
        raise PreparationError("Filtered metadata must contain exactly 179 unique patients")
    cohort = {str(k): int(v) for k, v in filtered["cohort"].value_counts().items()}
    material = {str(k): int(v) for k, v in filtered["material"].value_counts().items()}
    cohort_material = _counts(filtered, ["cohort", "material"])
    if cohort != EXPECTED_TOTAL:
        raise PreparationError(f"179-sample cohort counts mismatch: {cohort}")
    if material != EXPECTED_MATERIAL:
        raise PreparationError(f"179-sample material counts mismatch: {material}")
    if cohort_material != EXPECTED_COHORT_MATERIAL:
        raise PreparationError(f"179-sample cohort x material counts mismatch: {cohort_material}")
    return filtered, observed


def _write_csv_immutable(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        expected = frame.astype(str)
        if existing.astype(str).equals(expected):
            return
        raise FileExistsError(f"Existing metadata differs; refusing to overwrite: {path}")
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8")
    temp.replace(path)


def _step01_inventory(
    metadata: pd.DataFrame, step01_dir: Path, excluded_library_id: str
) -> tuple[dict[str, Path], Path]:
    if not step01_dir.is_dir():
        raise FileNotFoundError(f"Step01 AA clone-table directory not found: {step01_dir}")
    source_ids = metadata["libraryid"].astype(str).tolist()
    expected_ids = set(source_ids)
    excluded_path = step01_dir / f"{excluded_library_id}{STEP01_SUFFIX}"
    if not excluded_path.is_file():
        raise FileNotFoundError(f"Excluded sample Step01 table not found: {excluded_path}")
    files = sorted(step01_dir.glob(f"*{STEP01_SUFFIX}"))
    by_id = {path.name[: -len(STEP01_SUFFIX)]: path for path in files}
    if len(by_id) != len(files):
        raise PreparationError("Step01 directory contains duplicate sample IDs")
    missing = sorted(expected_ids - set(by_id))
    extra = sorted(set(by_id) - expected_ids - {excluded_library_id})
    if missing or extra:
        raise PreparationError(
            f"Step01 inventory mismatch: missing={missing[:20]}, unexpected={extra[:20]}"
        )
    if len(files) != 180:
        raise PreparationError(
            f"Expected 180 Step01 AA tables before exclusion; observed {len(files)}"
        )
    return {sample_id: by_id[sample_id] for sample_id in source_ids}, excluded_path


def _verify_or_create_symlinks(link_dir: Path, inventory: Mapping[str, Path]) -> None:
    link_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{sample_id}{STEP01_SUFFIX}" for sample_id in inventory}
    existing = {path.name: path for path in link_dir.iterdir()}
    unexpected = sorted(set(existing) - expected_names)
    if unexpected:
        raise PreparationError(f"Whitelist directory contains unexpected entries: {unexpected[:20]}")
    for sample_id, source in inventory.items():
        destination = link_dir / f"{sample_id}{STEP01_SUFFIX}"
        source_resolved = source.resolve()
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() or destination.resolve() != source_resolved:
                raise PreparationError(f"Existing whitelist entry is not the expected symlink: {destination}")
            continue
        destination.symlink_to(source_resolved)


def _verify_prepared_marker(
    marker_path: Path,
    repository_root: Path,
    metadata_dir: Path,
    link_dir: Path,
    expected_ids: set[str],
) -> None:
    if not marker_path.is_file():
        return
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "PREPARED":
        raise PreparationError(f"Unexpected preparation marker status: {marker.get('status')}")
    for key, record in marker.get("files", {}).items():
        path = resolve(repository_root, record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise PreparationError(f"Prepared metadata hash mismatch for {key}: {path}")
    actual = {
        path.name[: -len(STEP01_SUFFIX)]
        for path in link_dir.glob(f"*{STEP01_SUFFIX}")
        if path.is_file()
    }
    if actual != expected_ids:
        raise PreparationError(
            f"Prepared Step01 whitelist mismatch: missing={sorted(expected_ids-actual)[:10]}, "
            f"extra={sorted(actual-expected_ids)[:10]}"
        )


def prepare_metadata_and_whitelist(
    root: Path,
    metadata_path: Path,
    step01_dir: Path,
    output_root: Path,
    *,
    exclude_library_id: str,
    exclude_patient: str,
    dry_run: bool,
) -> None:
    source = pd.read_csv(metadata_path, dtype=str, encoding="utf-8-sig")
    filtered, excluded = build_filtered_metadata(
        source,
        exclude_library_id=exclude_library_id,
        exclude_patient=exclude_patient,
    )
    inventory, excluded_path = _step01_inventory(filtered, step01_dir, exclude_library_id)
    pbmc = filtered.loc[filtered["material"] == "pbmc"].copy()
    buffycoat = filtered.loc[filtered["material"] == "buffycoat"].copy()
    metadata_dir = output_root / "metadata"
    link_dir = output_root / "source" / "01_AA_clone_table"
    marker_path = output_root / "PREPARED_179_TOP10000.json"

    print("[179 SAMPLE CONTRACT]")
    print(f"  samples={len(filtered)}; RA={int((filtered['cohort'] == 'RA').sum())}; ILD={int((filtered['cohort'] == 'ILD').sum())}")
    print(f"  PBMC={len(pbmc)}; Buffycoat={len(buffycoat)}")
    print(f"  excluded={exclude_library_id}; patient={excluded['patient']}; reason=insufficient_AA_clonotypes_for_top10000")
    print(f"  Step01 tables={len(inventory)}; excluded Step01 table retained upstream={excluded_path}")
    if dry_run:
        print("DRY-RUN: no metadata or symlinks were written.")
        return

    if marker_path.exists():
        _verify_prepared_marker(marker_path, root, metadata_dir, link_dir, set(filtered["libraryid"].astype(str)))
        print(f"Already prepared and verified: {output_root}")
        return

    _write_csv_immutable(filtered, metadata_dir / "metadata_total.csv")
    _write_csv_immutable(pbmc, metadata_dir / "metadata_pbmc.csv")
    _write_csv_immutable(buffycoat, metadata_dir / "metadata_buffycoat.csv")
    _verify_or_create_symlinks(link_dir, inventory)

    files = {
        "metadata_total": metadata_dir / "metadata_total.csv",
        "metadata_pbmc": metadata_dir / "metadata_pbmc.csv",
        "metadata_buffycoat": metadata_dir / "metadata_buffycoat.csv",
    }
    payload = {
        "status": "PREPARED",
        "data_version": "trb_v3_179_top10000",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata": str(metadata_path),
        "source_metadata_sha256": sha256(metadata_path),
        "source_step01_dir": str(step01_dir),
        "source_step01_sample_count_before_exclusion": 180,
        "top_k": TOP_K,
        "excluded_sample": {
            "libraryid": exclude_library_id,
            **excluded,
            "reason": "insufficient_AA_clonotypes_for_top10000",
            "source_step01_table": str(excluded_path),
        },
        "sample_counts": {
            "total": len(filtered),
            "RA": int((filtered["cohort"] == "RA").sum()),
            "ILD": int((filtered["cohort"] == "ILD").sum()),
            "pbmc": len(pbmc),
            "buffycoat": len(buffycoat),
        },
        "files": {
            key: {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for key, path in files.items()
        },
        "step01_whitelist": {
            "path": link_dir.relative_to(root).as_posix(),
            "sample_count": len(inventory),
            "excluded_sample_absent": exclude_library_id not in inventory,
            "symlink_targets": "original v3_180 Step01 AA tables; no source table copied or modified",
        },
        "original_v3_180_untouched": True,
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PREPARE PASS: metadata and {len(inventory)} Step01 symlinks -> {output_root}")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def audit_topk(
    output_root: Path,
    topk_dir: Path,
    *,
    top_k: int = TOP_K,
) -> None:
    metadata_path = output_root / "metadata" / "metadata_total.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"179-sample metadata not found: {metadata_path}")
    metadata = normalize_metadata(pd.read_csv(metadata_path, dtype=str), "179-sample metadata")
    expected_ids = set(metadata["libraryid"].astype(str))
    summary_path = topk_dir / f"01b_top{top_k}_summary.csv"
    qc_path = topk_dir / f"01b_top{top_k}_qc.csv"
    config_path = topk_dir / f"01b_top{top_k}_configuration.json"
    for path in (summary_path, qc_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Top-K output missing: {path}")
    summary = pd.read_csv(summary_path)
    qc = pd.read_csv(qc_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("top_k", -1)) != top_k or int(config.get("sample_count", -1)) != len(expected_ids):
        raise PreparationError(f"Top-K configuration mismatch: {config}")
    if not bool(config.get("all_samples_qc_pass")):
        raise PreparationError("Top-K configuration reports one or more QC failures")
    for label, frame in (("summary", summary), ("QC", qc)):
        if "sample_id" not in frame.columns or frame["sample_id"].duplicated().any():
            raise PreparationError(f"Top-K {label} has missing/duplicated sample_id")
        observed_ids = set(frame["sample_id"].astype(str))
        if observed_ids != expected_ids:
            raise PreparationError(
                f"Top-K {label} sample coverage mismatch: missing={sorted(expected_ids-observed_ids)[:10]}, "
                f"extra={sorted(observed_ids-expected_ids)[:10]}"
            )
    if not summary["top_k"].astype(int).eq(top_k).all():
        raise PreparationError("Top-K summary contains a top_k value different from the requested K")
    if not summary["retained_aa_clone_number"].astype(int).eq(top_k).all():
        raise PreparationError("At least one sample did not retain exactly Top-K AA clonotypes")
    if not summary["qc_pass"].map(_truthy).all() or not qc["qc_pass"].map(_truthy).all():
        raise PreparationError("At least one Top-K summary/QC row failed")
    files = sorted(topk_dir.glob(f"*{TOPK_SUFFIX}"))
    names = {path.name[: -len(TOPK_SUFFIX)] for path in files}
    if names != expected_ids or len(files) != len(expected_ids):
        raise PreparationError(
            f"Top-K table inventory mismatch: missing={sorted(expected_ids-names)[:10]}, "
            f"extra={sorted(names-expected_ids)[:10]}"
        )
    for path in files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "topk_rank" not in reader.fieldnames:
                raise PreparationError(f"Top-K table missing topk_rank: {path}")
            count = 0
            for expected_rank, row in enumerate(reader, start=1):
                count += 1
                if int(row["topk_rank"]) != expected_rank:
                    raise PreparationError(f"Non-consecutive topk_rank in {path} at row {expected_rank}")
            if count != top_k:
                raise PreparationError(f"{path}: expected {top_k} rows, observed {count}")
    audit_path = output_root / "TOP10000_179_AUDIT.json"
    if not audit_path.exists():
        payload = {
            "status": "PASS",
            "data_version": "trb_v3_179_top10000",
            "top_k": top_k,
            "sample_count": len(expected_ids),
            "excluded_sample_absent": EXCLUDED_LIBRARY_ID not in expected_ids,
            "summary_sha256": sha256(summary_path),
            "qc_sha256": sha256(qc_path),
            "configuration_sha256": sha256(config_path),
            "audited_table_count": len(files),
            "all_tables_exact_k_and_ranked": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"TOP-K AUDIT PASS: {len(files)} samples; each has exactly {top_k} ranked AA clonotypes.")
    print(f"Audit marker: {audit_path}")


def materialize_repeat1(
    output_root: Path,
    assignments_path: Path,
) -> None:
    metadata_path = output_root / "metadata" / "metadata_total.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"179-sample metadata not found: {metadata_path}")
    metadata = normalize_metadata(pd.read_csv(metadata_path, dtype=str), "179-sample metadata")
    expected_ids = set(metadata["libraryid"].astype(str))
    if not assignments_path.is_file():
        raise FileNotFoundError(f"Repeated-holdout assignments not found: {assignments_path}")
    assignments = pd.read_csv(assignments_path, dtype={"sample_id": str, "role": str})
    required = {"split_id", "repeat_index", "role", "sample_id"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise PreparationError(f"Repeated-holdout assignments missing columns: {missing}")
    assignments["repeat_index"] = pd.to_numeric(assignments["repeat_index"], errors="raise").astype(int)
    repeats = sorted(assignments["repeat_index"].unique().tolist())
    if repeats != list(range(1, 101)):
        raise PreparationError(f"Expected repeat_index 1..100, observed {repeats}")
    for repeat_index, frame in assignments.groupby("repeat_index", sort=True):
        ids = set(frame["sample_id"].astype(str))
        if ids != expected_ids or frame["sample_id"].duplicated().any():
            raise PreparationError(f"Repeat {repeat_index} does not cover the exact 179-sample cohort")
        counts = frame["role"].astype(str).value_counts().to_dict()
        if counts.get("train", 0) != 125 or counts.get("holdout", 0) != 54:
            raise PreparationError(
                f"Repeat {repeat_index} expected train/holdout=125/54, observed "
                f"{counts.get('train', 0)}/{counts.get('holdout', 0)}"
            )
        if frame["split_id"].astype(str).nunique() != 1:
            raise PreparationError(f"Repeat {repeat_index} maps to multiple split IDs")
    repeat1 = assignments.loc[assignments["repeat_index"] == 1].copy()
    role_map = repeat1.set_index("sample_id")["role"].astype(str).str.lower()
    train_ids = set(role_map.index[role_map == "train"])
    holdout_ids = set(role_map.index[role_map == "holdout"])
    if train_ids & holdout_ids or train_ids | holdout_ids != expected_ids:
        raise PreparationError("Repeat 1 train/holdout roles do not form a disjoint partition")
    train = metadata.loc[metadata["libraryid"].astype(str).isin(train_ids)].copy()
    holdout = metadata.loc[metadata["libraryid"].astype(str).isin(holdout_ids)].copy()
    if (len(train), len(holdout)) != (125, 54):
        raise PreparationError("Repeat 1 metadata partition is not 125/54")
    out_dir = output_root / "ml_compat" / "total"
    train_path = out_dir / "metadata_train_repeat01.csv"
    holdout_path = out_dir / "metadata_holdout_repeat01.csv"
    _write_csv_immutable(train, train_path)
    _write_csv_immutable(holdout, holdout_path)
    _write_csv_immutable(train, output_root / "train" / "metadata_train.csv")
    _write_csv_immutable(holdout, output_root / "test" / "metadata_test.csv")
    marker_path = out_dir / "REPEAT01_COMPAT_FROZEN.json"
    if not marker_path.exists():
        marker = {
            "status": "FROZEN",
            "split_assignments": str(assignments_path),
            "split_assignments_sha256": sha256(assignments_path),
            "repeat_index": 1,
            "train_samples": len(train),
            "holdout_samples": len(holdout),
            "train_library_ids_sha256": hashlib.sha256("\n".join(train["libraryid"].astype(str)).encode()).hexdigest(),
            "holdout_library_ids_sha256": hashlib.sha256("\n".join(holdout["libraryid"].astype(str)).encode()).hexdigest(),
            "files": {
                "train": {"path": train_path.name, "sha256": sha256(train_path)},
                "holdout": {"path": holdout_path.name, "sha256": sha256(holdout_path)},
            },
        }
        marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"REPEAT-1 MATERIALIZATION PASS: train={len(train)}, holdout={len(holdout)}")
    print(f"Train metadata:   {train_path}")
    print(f"Holdout metadata: {holdout_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--mode",
        choices=("prepare", "audit-topk", "materialize-repeat1"),
        default="prepare",
    )
    parser.add_argument("--metadata", default=str(SOURCE_METADATA))
    parser.add_argument("--step01-dir", default=str(SOURCE_STEP01))
    parser.add_argument("--output-root", default=str(DATA_VERSION))
    parser.add_argument("--topk-dir", default=None)
    parser.add_argument(
        "--assignments",
        default="TRB/set/split_sets/ra_ild_repeat100_v3_179_top10000/repeated_holdout_assignments.csv",
    )
    parser.add_argument("--exclude-library-id", default=EXCLUDED_LIBRARY_ID)
    parser.add_argument("--exclude-patient", default=EXCLUDED_PATIENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        root = args.repo.expanduser().resolve()
        output_root = resolve(root, args.output_root)
        if args.mode == "prepare":
            prepare_metadata_and_whitelist(
                root,
                resolve(root, args.metadata),
                resolve(root, args.step01_dir),
                output_root,
                exclude_library_id=args.exclude_library_id,
                exclude_patient=args.exclude_patient,
                dry_run=args.dry_run,
            )
        elif args.mode == "audit-topk":
            topk_dir = (
                resolve(root, args.topk_dir)
                if args.topk_dir
                else output_root / "result" / "01b_AA_clone_table_top10000"
            )
            audit_topk(output_root, topk_dir)
        else:
            materialize_repeat1(output_root, resolve(root, args.assignments))
        return 0
    except Exception as exc:
        print(f"TRB V3 179 Top10000 preparation: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
