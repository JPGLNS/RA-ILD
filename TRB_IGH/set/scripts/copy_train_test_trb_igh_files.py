#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copy paired TRB and IGH repertoire files for a fixed train/test split.

The script performs a complete preflight check before copying any file. Each
metadata row must contain one patient ID, one TRB library ID, and one IGH library
ID. By default, source filenames are constructed as:

    <trb_libraryid>_TRB_CDR3_NT_frequency_error_correct.csv
    <igh_libraryid>_IGH-without-DJ_CDR3_NT_frequency_error_correct.csv

It does not perform train/test splitting, feature extraction, or model training.
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
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd


DEFAULT_TRB_SUFFIX = "_TRB_CDR3_NT_frequency_error_correct.csv"
DEFAULT_IGH_SUFFIX = "_IGH-without-DJ_CDR3_NT_frequency_error_correct.csv"


@dataclass
class CopyRecord:
    dataset: str
    repertoire: str
    patient: str
    library_id: str
    source_file: str
    destination_file: str
    status: str
    message: str = ""


class CopyError(RuntimeError):
    """Raised for invalid metadata, paths, or copy state."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy paired TRB and IGH repertoire files into fixed train/test "
            "directories after a complete preflight check."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--test-metadata", required=True)
    parser.add_argument("--trb-source-dir", required=True)
    parser.add_argument("--igh-source-dir", required=True)
    parser.add_argument("--train-trb-dest", required=True)
    parser.add_argument("--train-igh-dest", required=True)
    parser.add_argument("--test-trb-dest", required=True)
    parser.add_argument("--test-igh-dest", required=True)
    parser.add_argument(
        "--patient-col",
        default="patient",
        help="Patient identifier column used as the paired split unit.",
    )
    parser.add_argument(
        "--trb-id-col",
        default="trb_libraryid",
        help="Metadata column used to construct TRB source filenames.",
    )
    parser.add_argument(
        "--igh-id-col",
        default="igh_libraryid",
        help="Metadata column used to construct IGH source filenames.",
    )
    parser.add_argument(
        "--trb-filename-suffix",
        default=DEFAULT_TRB_SUFFIX,
        help="Text appended to each TRB library ID.",
    )
    parser.add_argument(
        "--igh-filename-suffix",
        default=DEFAULT_IGH_SUFFIX,
        help="Text appended to each IGH library ID.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "CSV copy log path. If omitted, "
            "copy_train_test_trb_igh_files_log.csv is written under the common "
            "output root."
        ),
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help=(
            "Markdown summary path. If omitted, "
            "copy_train_test_trb_igh_files_summary.md is written under the common "
            "output root."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all inputs and report planned operations without copying.",
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
            "How to handle missing source files. With 'error', no file is copied "
            "when any source is missing. With 'warn', available files are copied."
        ),
    )
    parser.add_argument(
        "--clean-destination",
        action="store_true",
        help=(
            "Remove files matching each repertoire suffix from all four destination "
            "directories before copying. Requires --overwrite and is ignored during "
            "--dry-run."
        ),
    )
    return parser.parse_args()


def drop_index_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in df.columns
        if str(column).startswith("Unnamed:")
        or str(column).strip() == ""
        or str(column).strip() == "X"
    ]
    return df.drop(columns=columns) if columns else df


def normalize_nonblank(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def read_metadata(
    path: Path,
    label: str,
    patient_col: str,
    trb_id_col: str,
    igh_id_col: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise CopyError(f"{label} metadata does not exist: {path}")

    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception as exc:
        raise CopyError(f"Failed to read {label} metadata: {exc}") from exc

    df = drop_index_like_columns(df)
    if df.empty:
        raise CopyError(f"{label} metadata contains zero rows: {path}")

    required = [patient_col, trb_id_col, igh_id_col]
    absent = [column for column in required if column not in df.columns]
    if absent:
        raise CopyError(
            f"{label} metadata is missing required column(s): {', '.join(absent)}. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )

    for column in required:
        values = normalize_nonblank(df[column])
        missing = values.isna() | values.eq("")
        if missing.any():
            rows = (df.index[missing] + 2).tolist()[:10]
            raise CopyError(
                f"{label} metadata contains {int(missing.sum())} missing/blank "
                f"'{column}' value(s), for example CSV row(s): {rows}"
            )
        df[column] = values.astype(str)

        duplicate_mask = df[column].duplicated(keep=False)
        if duplicate_mask.any():
            duplicates = sorted(df.loc[duplicate_mask, column].unique().tolist())
            raise CopyError(
                f"{label} metadata contains duplicate '{column}' values: "
                + ", ".join(duplicates[:20])
            )

    return df.reset_index(drop=True)


def validate_train_test_integrity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    patient_col: str,
    trb_id_col: str,
    igh_id_col: str,
) -> Dict[str, object]:
    results: Dict[str, object] = {}
    for column, label in (
        (patient_col, "patient"),
        (trb_id_col, "TRB library"),
        (igh_id_col, "IGH library"),
    ):
        train_values = set(train_df[column])
        test_values = set(test_df[column])
        overlap = sorted(train_values & test_values)
        results[f"{label}_overlap_count"] = len(overlap)
        results[f"{label}_overlap_examples"] = overlap[:20]
        if overlap:
            raise CopyError(
                f"Train/test metadata share {len(overlap)} {label} ID(s): "
                + ", ".join(overlap[:20])
            )

    combined = pd.concat(
        [train_df.assign(dataset="train"), test_df.assign(dataset="test")],
        ignore_index=True,
    )
    for column in (patient_col, trb_id_col, igh_id_col):
        duplicate_mask = combined[column].duplicated(keep=False)
        if duplicate_mask.any():
            duplicates = sorted(combined.loc[duplicate_mask, column].unique().tolist())
            raise CopyError(
                f"Combined train/test metadata contain duplicate '{column}' values: "
                + ", ".join(duplicates[:20])
            )

    results["train_rows"] = len(train_df)
    results["test_rows"] = len(test_df)
    results["total_rows"] = len(combined)
    return results


def destination_common_root(destinations: Sequence[Path]) -> Path:
    common = Path(os.path.commonpath([str(path) for path in destinations]))
    if common in destinations:
        common = common.parent
    return common


def ensure_distinct_destinations(destinations: Mapping[str, Path]) -> None:
    resolved = {name: path.resolve() for name, path in destinations.items()}
    reverse: Dict[Path, List[str]] = {}
    for name, path in resolved.items():
        reverse.setdefault(path, []).append(name)
    duplicates = {path: names for path, names in reverse.items() if len(names) > 1}
    if duplicates:
        detail = "; ".join(
            f"{path}: {', '.join(names)}" for path, names in duplicates.items()
        )
        raise CopyError(f"Destination directories must be distinct: {detail}")


def build_records_for_dataset(
    dataset: str,
    metadata: pd.DataFrame,
    patient_col: str,
    trb_id_col: str,
    igh_id_col: str,
    trb_source_dir: Path,
    igh_source_dir: Path,
    trb_destination: Path,
    igh_destination: Path,
    trb_suffix: str,
    igh_suffix: str,
    overwrite: bool,
) -> List[CopyRecord]:
    records: List[CopyRecord] = []
    specs = (
        (
            "TRB",
            trb_id_col,
            trb_source_dir,
            trb_destination,
            trb_suffix,
        ),
        (
            "IGH",
            igh_id_col,
            igh_source_dir,
            igh_destination,
            igh_suffix,
        ),
    )

    for row in metadata.itertuples(index=False):
        row_map = row._asdict()
        patient = str(row_map[patient_col])
        for repertoire, id_col, source_dir, destination_dir, suffix in specs:
            library_id = str(row_map[id_col])
            filename = f"{library_id}{suffix}"
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
                    repertoire=repertoire,
                    patient=patient,
                    library_id=library_id,
                    source_file=str(source),
                    destination_file=str(destination),
                    status=status,
                    message=message,
                )
            )
    return records


def build_all_records(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    paths: Mapping[str, Path],
) -> List[CopyRecord]:
    records = build_records_for_dataset(
        dataset="train",
        metadata=train_df,
        patient_col=args.patient_col,
        trb_id_col=args.trb_id_col,
        igh_id_col=args.igh_id_col,
        trb_source_dir=paths["trb_source"],
        igh_source_dir=paths["igh_source"],
        trb_destination=paths["train_trb_dest"],
        igh_destination=paths["train_igh_dest"],
        trb_suffix=args.trb_filename_suffix,
        igh_suffix=args.igh_filename_suffix,
        overwrite=args.overwrite,
    )
    records.extend(
        build_records_for_dataset(
            dataset="test",
            metadata=test_df,
            patient_col=args.patient_col,
            trb_id_col=args.trb_id_col,
            igh_id_col=args.igh_id_col,
            trb_source_dir=paths["trb_source"],
            igh_source_dir=paths["igh_source"],
            trb_destination=paths["test_trb_dest"],
            igh_destination=paths["test_igh_dest"],
            trb_suffix=args.trb_filename_suffix,
            igh_suffix=args.igh_filename_suffix,
            overwrite=args.overwrite,
        )
    )
    return records


def clean_destination(destination: Path, suffix: str) -> int:
    if not destination.exists():
        return 0
    removed = 0
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


def count_statuses(records: Iterable[CopyRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    return counts


def count_grouped(
    records: Sequence[CopyRecord],
) -> Dict[Tuple[str, str], Dict[str, int]]:
    grouped: Dict[Tuple[str, str], Dict[str, int]] = {}
    for record in records:
        key = (record.dataset, record.repertoire)
        grouped.setdefault(key, {})
        grouped[key][record.status] = grouped[key].get(record.status, 0) + 1
    return grouped


def write_log(records: Sequence[CopyRecord], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "repertoire",
                "patient",
                "library_id",
                "source_file",
                "destination_file",
                "status",
                "message",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def format_status_counts(counts: Mapping[str, int]) -> List[str]:
    statuses = [
        "copied",
        "would_copy",
        "would_overwrite",
        "skipped_exists",
        "missing_source",
        "copy_failed",
    ]
    return [f"- {status}: **{counts.get(status, 0)}**" for status in statuses]


def write_summary(
    summary_file: Path,
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    integrity: Mapping[str, object],
    records: Sequence[CopyRecord],
    removed: Mapping[str, int],
) -> None:
    overall_counts = count_statuses(records)
    grouped = count_grouped(records)
    missing = [record for record in records if record.status == "missing_source"]
    skipped = [record for record in records if record.status == "skipped_exists"]

    lines = [
        "# Paired TRB+IGH Train/Test File Copy Summary",
        "",
        f"- Run time: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Dry run: `{args.dry_run}`",
        f"- Overwrite: `{args.overwrite}`",
        f"- Missing policy: `{args.missing_policy}`",
        f"- Patient column: `{args.patient_col}`",
        f"- TRB ID column: `{args.trb_id_col}`",
        f"- IGH ID column: `{args.igh_id_col}`",
        f"- TRB filename suffix: `{args.trb_filename_suffix}`",
        f"- IGH filename suffix: `{args.igh_filename_suffix}`",
        "",
        "## Input and output paths",
        "",
        f"- Train metadata: `{paths['train_metadata']}`",
        f"- Test metadata: `{paths['test_metadata']}`",
        f"- TRB source directory: `{paths['trb_source']}`",
        f"- IGH source directory: `{paths['igh_source']}`",
        f"- Train TRB destination: `{paths['train_trb_dest']}`",
        f"- Train IGH destination: `{paths['train_igh_dest']}`",
        f"- Test TRB destination: `{paths['test_trb_dest']}`",
        f"- Test IGH destination: `{paths['test_igh_dest']}`",
        "",
        "## Metadata integrity",
        "",
        f"- Train patients: **{len(train_df)}**",
        f"- Test patients: **{len(test_df)}**",
        f"- Total paired patients: **{integrity['total_rows']}**",
        f"- Train/test patient overlap: **{integrity['patient_overlap_count']}**",
        f"- Train/test TRB library overlap: **{integrity['TRB library_overlap_count']}**",
        f"- Train/test IGH library overlap: **{integrity['IGH library_overlap_count']}**",
        f"- Expected copy records: **{2 * integrity['total_rows']}**",
        f"- Actual copy records: **{len(records)}**",
        "",
        "## Overall copy status",
        "",
        *format_status_counts(overall_counts),
        "",
        "## Status by dataset and repertoire",
        "",
    ]

    for dataset in ("train", "test"):
        for repertoire in ("TRB", "IGH"):
            lines.append(f"### {dataset} {repertoire}")
            lines.append("")
            lines.extend(format_status_counts(grouped.get((dataset, repertoire), {})))
            lines.append("")

    lines.extend(["## Destination cleanup", ""])
    for name in (
        "train_trb_dest",
        "train_igh_dest",
        "test_trb_dest",
        "test_igh_dest",
    ):
        lines.append(f"- Removed from {name}: **{removed.get(name, 0)}**")

    lines.extend(["", "## Missing source files", ""])
    if missing:
        for record in missing:
            lines.append(
                f"- `{record.dataset}` | `{record.repertoire}` | "
                f"`{record.patient}` | `{record.library_id}` | `{record.source_file}`"
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Existing destination files skipped", ""])
    if skipped:
        for record in skipped:
            lines.append(
                f"- `{record.dataset}` | `{record.repertoire}` | "
                f"`{record.patient}` | `{record.destination_file}`"
            )
    else:
        lines.append("None.")

    lines.append("")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text("\n".join(lines), encoding="utf-8")


def print_console_summary(records: Sequence[CopyRecord], dry_run: bool) -> None:
    print("\nSummary:")
    counts = count_statuses(records)
    for status in (
        "copied",
        "would_copy",
        "would_overwrite",
        "skipped_exists",
        "missing_source",
        "copy_failed",
    ):
        print(f"  {status}: {counts.get(status, 0)}")
    if dry_run:
        print("\nDry run completed: no files were copied.")


def main() -> int:
    args = parse_args()

    paths: Dict[str, Path] = {
        "train_metadata": Path(args.train_metadata).expanduser().resolve(),
        "test_metadata": Path(args.test_metadata).expanduser().resolve(),
        "trb_source": Path(args.trb_source_dir).expanduser().resolve(),
        "igh_source": Path(args.igh_source_dir).expanduser().resolve(),
        "train_trb_dest": Path(args.train_trb_dest).expanduser().resolve(),
        "train_igh_dest": Path(args.train_igh_dest).expanduser().resolve(),
        "test_trb_dest": Path(args.test_trb_dest).expanduser().resolve(),
        "test_igh_dest": Path(args.test_igh_dest).expanduser().resolve(),
    }

    try:
        if not args.trb_filename_suffix:
            raise CopyError("--trb-filename-suffix cannot be empty.")
        if not args.igh_filename_suffix:
            raise CopyError("--igh-filename-suffix cannot be empty.")

        for name in ("trb_source", "igh_source"):
            if not paths[name].is_dir():
                raise CopyError(f"Source directory does not exist: {paths[name]}")

        destination_paths = {
            key: paths[key]
            for key in (
                "train_trb_dest",
                "train_igh_dest",
                "test_trb_dest",
                "test_igh_dest",
            )
        }
        ensure_distinct_destinations(destination_paths)

        train_df = read_metadata(
            paths["train_metadata"],
            "train",
            args.patient_col,
            args.trb_id_col,
            args.igh_id_col,
        )
        test_df = read_metadata(
            paths["test_metadata"],
            "test",
            args.patient_col,
            args.trb_id_col,
            args.igh_id_col,
        )
        integrity = validate_train_test_integrity(
            train_df,
            test_df,
            args.patient_col,
            args.trb_id_col,
            args.igh_id_col,
        )

        common_root = destination_common_root(list(destination_paths.values()))
        log_file = (
            Path(args.log_file).expanduser().resolve()
            if args.log_file
            else common_root / "copy_train_test_trb_igh_files_log.csv"
        )
        summary_file = (
            Path(args.summary_file).expanduser().resolve()
            if args.summary_file
            else common_root / "copy_train_test_trb_igh_files_summary.md"
        )

        records = build_all_records(train_df, test_df, args, paths)
        expected_records = 2 * (len(train_df) + len(test_df))
        if len(records) != expected_records:
            raise CopyError(
                f"Internal plan-size mismatch: expected {expected_records}, got {len(records)}"
            )

        missing_records = [record for record in records if record.status == "missing_source"]
        removed = {name: 0 for name in destination_paths}

        # Always write preflight evidence, including an abort caused by missing files.
        if missing_records and args.missing_policy == "error":
            write_log(records, log_file)
            write_summary(
                summary_file,
                args,
                paths,
                train_df,
                test_df,
                integrity,
                records,
                removed,
            )
            print_console_summary(records, args.dry_run)
            print(
                f"ERROR: {len(missing_records)} source file(s) are missing. "
                "No files were copied.\n"
                f"See log: {log_file}\nSee summary: {summary_file}",
                file=sys.stderr,
            )
            return 3

        if args.clean_destination and not args.overwrite:
            raise CopyError("--clean-destination requires --overwrite.")

        if args.clean_destination and not args.dry_run:
            for name, suffix in (
                ("train_trb_dest", args.trb_filename_suffix),
                ("train_igh_dest", args.igh_filename_suffix),
                ("test_trb_dest", args.trb_filename_suffix),
                ("test_igh_dest", args.igh_filename_suffix),
            ):
                paths[name].mkdir(parents=True, exist_ok=True)
                removed[name] = clean_destination(paths[name], suffix)
            records = build_all_records(train_df, test_df, args, paths)

        execute_plan(records, args.dry_run)

        if not args.dry_run:
            failed: List[CopyRecord] = []
            for record in records:
                if record.status == "copied" and not Path(record.destination_file).is_file():
                    record.status = "copy_failed"
                    record.message = "Destination file absent after copy operation"
                    failed.append(record)
            if failed:
                write_log(records, log_file)
                write_summary(
                    summary_file,
                    args,
                    paths,
                    train_df,
                    test_df,
                    integrity,
                    records,
                    removed,
                )
                print(
                    f"ERROR: {len(failed)} copied file(s) failed post-copy verification.",
                    file=sys.stderr,
                )
                return 4

        write_log(records, log_file)
        write_summary(
            summary_file,
            args,
            paths,
            train_df,
            test_df,
            integrity,
            records,
            removed,
        )

        print("Completed successfully.")
        print(f"Train metadata: {paths['train_metadata']}")
        print(f"Test metadata: {paths['test_metadata']}")
        print(f"TRB source directory: {paths['trb_source']}")
        print(f"IGH source directory: {paths['igh_source']}")
        print(f"Log file: {log_file}")
        print(f"Summary file: {summary_file}")
        print_console_summary(records, args.dry_run)
        return 0

    except CopyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("ERROR: Interrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
