#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copy fixed train/test TRB repertoire files according to split metadata.

This script only copies repertoire files. It does not perform train/test splitting,
feature extraction, or model training.

Expected default source filename pattern:
    <libraryid>_TRB_CDR3_NT_frequency_error_correct.csv

Example:
    MGI260113R01-16_TRB_CDR3_NT_frequency_error_correct.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


DEFAULT_SUFFIX = "_TRB_CDR3_NT_frequency_error_correct.csv"


@dataclass
class CopyRecord:
    dataset: str
    sample_id: str
    source_file: str
    destination_file: str
    status: str
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy TRB repertoire files into fixed train/test directories according "
            "to metadata_train_70.csv and metadata_test_30.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-metadata",
        required=True,
        help="Path to fixed training metadata CSV.",
    )
    parser.add_argument(
        "--test-metadata",
        required=True,
        help="Path to fixed testing metadata CSV.",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing all original TRB repertoire CSV files.",
    )
    parser.add_argument(
        "--train-dest",
        required=True,
        help="Destination directory for training repertoire files.",
    )
    parser.add_argument(
        "--test-dest",
        required=True,
        help="Destination directory for testing repertoire files.",
    )
    parser.add_argument(
        "--id-col",
        default="libraryid",
        help="Metadata column used to construct source filenames.",
    )
    parser.add_argument(
        "--filename-suffix",
        default=DEFAULT_SUFFIX,
        help="Text appended to each sample ID to obtain the source filename.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "CSV copy log path. If omitted, a log named "
            "copy_train_test_files_log.csv is written beside train/test directories."
        ),
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help=(
            "Markdown summary path. If omitted, a summary named "
            "copy_train_test_files_summary.md is written beside train/test directories."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check all paths and report planned operations without copying files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files that already exist.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=("error", "warn"),
        default="error",
        help=(
            "How to handle source files missing from source-dir. 'error' performs "
            "no copy after preflight failure; 'warn' copies available files."
        ),
    )
    parser.add_argument(
        "--clean-destination",
        action="store_true",
        help=(
            "Before copying, remove files in each destination that match the configured "
            "filename suffix. Requires --overwrite and is ignored in --dry-run mode."
        ),
    )
    return parser.parse_args()


def read_metadata(path: Path, id_col: str, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} metadata does not exist: {path}")

    df = pd.read_csv(path, dtype=str)

    # Drop common index columns accidentally written by pandas/R.
    index_like = [
        col for col in df.columns
        if str(col).startswith("Unnamed:") or str(col).strip() == ""
    ]
    if index_like:
        df = df.drop(columns=index_like)

    if id_col not in df.columns:
        raise ValueError(
            f"Required ID column '{id_col}' is absent from {label} metadata. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )

    ids = df[id_col]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        bad_rows = df.index[ids.isna() | ids.astype(str).str.strip().eq("")].tolist()
        preview = ", ".join(str(i + 2) for i in bad_rows[:10])
        raise ValueError(
            f"{label} metadata contains missing/blank '{id_col}' values at CSV row(s): {preview}"
        )

    df[id_col] = ids.astype(str).str.strip()
    duplicate_ids = df.loc[df[id_col].duplicated(keep=False), id_col].unique().tolist()
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:10])
        raise ValueError(
            f"{label} metadata contains duplicate '{id_col}' values: {preview}"
        )

    return df



def build_plan(
    dataset: str,
    metadata: pd.DataFrame,
    id_col: str,
    source_dir: Path,
    destination_dir: Path,
    suffix: str,
    overwrite: bool,
) -> List[CopyRecord]:
    records: List[CopyRecord] = []

    for sample_id in metadata[id_col].tolist():
        filename = f"{sample_id}{suffix}"
        source = source_dir / filename
        destination = destination_dir / filename

        if not source.is_file():
            status = "missing_source"
            message = "Source file does not exist"
        elif destination.exists() and not overwrite:
            status = "skipped_exists"
            message = "Destination already exists; use --overwrite to replace"
        elif destination.exists() and overwrite:
            status = "would_overwrite"
            message = "Destination exists and will be overwritten"
        else:
            status = "would_copy"
            message = ""

        records.append(
            CopyRecord(
                dataset=dataset,
                sample_id=sample_id,
                source_file=str(source),
                destination_file=str(destination),
                status=status,
                message=message,
            )
        )

    return records


def clean_destination(destination: Path, suffix: str) -> int:
    removed = 0
    if not destination.exists():
        return removed
    for path in destination.iterdir():
        if path.is_file() and path.name.endswith(suffix):
            path.unlink()
            removed += 1
    return removed


def execute_plan(records: Sequence[CopyRecord], dry_run: bool) -> None:
    if dry_run:
        return

    for record in records:
        if record.status not in {"would_copy", "would_overwrite"}:
            continue

        source = Path(record.source_file)
        destination = Path(record.destination_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        record.status = "copied"
        record.message = ""


def write_log(records: Sequence[CopyRecord], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "sample_id",
                "source_file",
                "destination_file",
                "status",
                "message",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def count_statuses(records: Sequence[CopyRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    return counts


def format_status_counts(counts: Dict[str, int]) -> str:
    order = [
        "copied",
        "would_copy",
        "would_overwrite",
        "skipped_exists",
        "missing_source",
    ]
    lines = []
    for status in order:
        lines.append(f"- {status}: **{counts.get(status, 0)}**")
    return "\n".join(lines)


def write_summary(
    summary_file: Path,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    records: Sequence[CopyRecord],
    train_test_overlap: Sequence[str],
    removed_train: int,
    removed_test: int,
) -> None:
    train_records = [r for r in records if r.dataset == "train"]
    test_records = [r for r in records if r.dataset == "test"]
    train_counts = count_statuses(train_records)
    test_counts = count_statuses(test_records)
    all_counts = count_statuses(records)

    missing = [r for r in records if r.status == "missing_source"]
    skipped = [r for r in records if r.status == "skipped_exists"]

    lines = [
        "# Train/Test TRB File Copy Summary",
        "",
        f"- Run time: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Dry run: `{args.dry_run}`",
        f"- Overwrite: `{args.overwrite}`",
        f"- Missing policy: `{args.missing_policy}`",
        f"- ID column: `{args.id_col}`",
        f"- Filename suffix: `{args.filename_suffix}`",
        f"- Source directory: `{Path(args.source_dir).resolve()}`",
        f"- Train metadata: `{Path(args.train_metadata).resolve()}`",
        f"- Test metadata: `{Path(args.test_metadata).resolve()}`",
        f"- Train destination: `{Path(args.train_dest).resolve()}`",
        f"- Test destination: `{Path(args.test_dest).resolve()}`",
        "",
        "## Metadata integrity",
        "",
        f"- Train metadata samples: **{len(train_df)}**",
        f"- Test metadata samples: **{len(test_df)}**",
        f"- Train/test ID overlap: **{len(train_test_overlap)}**",
        f"- Metadata total: **{len(train_df) + len(test_df)}**",
        "",
        "## Overall copy status",
        "",
        format_status_counts(all_counts),
        "",
        "## Train copy status",
        "",
        format_status_counts(train_counts),
        "",
        "## Test copy status",
        "",
        format_status_counts(test_counts),
        "",
        "## Destination cleanup",
        "",
        f"- Removed matching files from train destination: **{removed_train}**",
        f"- Removed matching files from test destination: **{removed_test}**",
        "",
    ]

    if missing:
        lines.extend(["## Missing source files", ""])
        for record in missing:
            lines.append(f"- `{record.dataset}` | `{record.sample_id}` | `{record.source_file}`")
        lines.append("")
    else:
        lines.extend(["## Missing source files", "", "None.", ""])

    if skipped:
        lines.extend(["## Existing destination files skipped", ""])
        for record in skipped:
            lines.append(f"- `{record.dataset}` | `{record.sample_id}` | `{record.destination_file}`")
        lines.append("")

    if train_test_overlap:
        lines.extend(["## ERROR: train/test ID overlap", ""])
        for sample_id in train_test_overlap:
            lines.append(f"- `{sample_id}`")
        lines.append("")

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text("\n".join(lines), encoding="utf-8")


def print_console_summary(records: Sequence[CopyRecord], dry_run: bool) -> None:
    print("\nSummary:")
    counts = count_statuses(records)
    for status in [
        "copied",
        "would_copy",
        "would_overwrite",
        "skipped_exists",
        "missing_source",
    ]:
        print(f"  {status}: {counts.get(status, 0)}")
    if dry_run:
        print("\nDry run completed: no files were copied.")


def main() -> int:
    args = parse_args()

    train_metadata = Path(args.train_metadata).expanduser().resolve()
    test_metadata = Path(args.test_metadata).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    train_dest = Path(args.train_dest).expanduser().resolve()
    test_dest = Path(args.test_dest).expanduser().resolve()

    if not source_dir.is_dir():
        print(f"ERROR: source directory does not exist: {source_dir}", file=sys.stderr)
        return 2

    if train_dest == test_dest:
        print("ERROR: --train-dest and --test-dest must be different directories.", file=sys.stderr)
        return 2

    try:
        train_df = read_metadata(train_metadata, args.id_col, "train")
        test_df = read_metadata(test_metadata, args.id_col, "test")
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    train_ids = set(train_df[args.id_col])
    test_ids = set(test_df[args.id_col])
    overlap = sorted(train_ids & test_ids)
    if overlap:
        preview = ", ".join(overlap[:10])
        print(
            f"ERROR: train/test metadata share {len(overlap)} sample ID(s): {preview}",
            file=sys.stderr,
        )
        return 2

    # For paths such as set/train/origin_result and set/test/origin_result,
    # the common directory is set/.
    common_output = Path(os.path.commonpath([str(train_dest), str(test_dest)]))
    if common_output in {train_dest, test_dest}:
        common_output = common_output.parent
    log_file = (
        Path(args.log_file).expanduser().resolve()
        if args.log_file
        else common_output / "copy_train_test_files_log.csv"
    )
    summary_file = (
        Path(args.summary_file).expanduser().resolve()
        if args.summary_file
        else common_output / "copy_train_test_files_summary.md"
    )

    # Preflight before any file is copied.
    records = build_plan(
        "train",
        train_df,
        args.id_col,
        source_dir,
        train_dest,
        args.filename_suffix,
        args.overwrite,
    )
    records.extend(
        build_plan(
            "test",
            test_df,
            args.id_col,
            source_dir,
            test_dest,
            args.filename_suffix,
            args.overwrite,
        )
    )

    missing_records = [r for r in records if r.status == "missing_source"]

    # Always write a preflight log/summary, including when missing-policy=error aborts.
    removed_train = 0
    removed_test = 0

    if missing_records and args.missing_policy == "error":
        write_log(records, log_file)
        write_summary(
            summary_file,
            args,
            train_df,
            test_df,
            records,
            overlap,
            removed_train,
            removed_test,
        )
        print(
            f"ERROR: {len(missing_records)} source file(s) are missing. No files were copied.\n"
            f"See log: {log_file}\nSee summary: {summary_file}",
            file=sys.stderr,
        )
        print_console_summary(records, args.dry_run)
        return 3

    if args.clean_destination and not args.overwrite:
        print(
            "ERROR: --clean-destination requires --overwrite to reduce accidental deletion risk.",
            file=sys.stderr,
        )
        return 2

    if args.clean_destination and not args.dry_run:
        train_dest.mkdir(parents=True, exist_ok=True)
        test_dest.mkdir(parents=True, exist_ok=True)
        removed_train = clean_destination(train_dest, args.filename_suffix)
        removed_test = clean_destination(test_dest, args.filename_suffix)
        # Rebuild plan because previous destination status may have changed.
        records = build_plan(
            "train",
            train_df,
            args.id_col,
            source_dir,
            train_dest,
            args.filename_suffix,
            args.overwrite,
        )
        records.extend(
            build_plan(
                "test",
                test_df,
                args.id_col,
                source_dir,
                test_dest,
                args.filename_suffix,
                args.overwrite,
            )
        )

    execute_plan(records, args.dry_run)

    # Post-copy verification for files expected to have been copied.
    if not args.dry_run:
        failed_after_copy: List[CopyRecord] = []
        for record in records:
            if record.status == "copied" and not Path(record.destination_file).is_file():
                record.status = "copy_failed"
                record.message = "Destination file absent after copy operation"
                failed_after_copy.append(record)
        if failed_after_copy:
            write_log(records, log_file)
            write_summary(
                summary_file,
                args,
                train_df,
                test_df,
                records,
                overlap,
                removed_train,
                removed_test,
            )
            print(
                f"ERROR: {len(failed_after_copy)} copied file(s) failed post-copy verification.",
                file=sys.stderr,
            )
            return 4

    write_log(records, log_file)
    write_summary(
        summary_file,
        args,
        train_df,
        test_df,
        records,
        overlap,
        removed_train,
        removed_test,
    )

    print("Completed successfully.")
    print(f"Train metadata: {train_metadata}")
    print(f"Test metadata: {test_metadata}")
    print(f"Source directory: {source_dir}")
    print(f"Train destination: {train_dest}")
    print(f"Test destination: {test_dest}")
    print(f"Log file: {log_file}")
    print(f"Summary file: {summary_file}")
    print_console_summary(records, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
