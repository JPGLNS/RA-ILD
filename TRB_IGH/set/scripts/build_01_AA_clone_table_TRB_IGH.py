#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build paired TRB and IGH CDR3 amino-acid clone tables.

This script is the paired-cohort version of the original IGH 01_AA_clone_table
workflow.  It reads one paired metadata file, uses ``trb_libraryid`` and
``igh_libraryid`` independently, and writes separate TRB and IGH output trees.
The two repertoires are intentionally NOT merged at this stage.

Default input files
-------------------
TRB:
    <trb_libraryid>_TRB_CDR3_NT_frequency_error_correct.csv
IGH:
    <igh_libraryid>_IGH-without-DJ_CDR3_NT_frequency_error_correct.csv

Expected input columns (headerless by default)
----------------------------------------------
1. cdr3_nt
2. read_count
3. input_frequency
4. input_cell_frequency

Main per-sample output columns
------------------------------
sample_id, cdr3_aa, aa_length, nt_clone_number, read_count,
read_fraction, input_frequency_sum, input_cell_frequency_sum,
frequency, frequency_norm, top_nt_cdr3

The default ``frequency`` definition remains the fourth input column aggregated
at AA level, matching the existing TRB/IGH workflow.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, TextIO, Tuple


SCRIPT_VERSION = "1.0.0-PAIRED-TRB-IGH"

TRB_INPUT_SUFFIX = "_TRB_CDR3_NT_frequency_error_correct.csv"
TRB_OUTPUT_SUFFIX = "_TRB_CDR3_AA_clone_table.csv"
IGH_INPUT_SUFFIX = "_IGH-without-DJ_CDR3_NT_frequency_error_correct.csv"
IGH_OUTPUT_SUFFIX = "_IGH-without-DJ_CDR3_AA_clone_table.csv"

CODON_TABLE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

OUTPUT_COLUMNS = [
    "sample_id",
    "cdr3_aa",
    "aa_length",
    "nt_clone_number",
    "read_count",
    "read_fraction",
    "input_frequency_sum",
    "input_cell_frequency_sum",
    "frequency",
    "frequency_norm",
    "top_nt_cdr3",
]

INVALID_REASONS = [
    "empty_sequence",
    "length_not_multiple_of_3",
    "non_acgt_base",
    "stop_codon",
    "malformed_columns",
    "invalid_read_count",
    "invalid_frequency",
]

REASON_CHINESE = {
    "empty_sequence": "空序列",
    "length_not_multiple_of_3": "长度不是3的倍数",
    "non_acgt_base": "包含非ACGT碱基",
    "stop_codon": "翻译产生终止密码子",
    "malformed_columns": "输入列数不足",
    "invalid_read_count": "read_count无效",
    "invalid_frequency": "频率列无效",
}


@dataclass(frozen=True)
class ReceptorConfig:
    name: str
    id_col: str
    input_dir: Path
    output_dir: Path
    input_suffix: str
    output_suffix: str


@dataclass
class NTRecord:
    read_count: int = 0
    input_frequency_sum: float = 0.0
    input_cell_frequency_sum: float = 0.0


@dataclass
class AARecord:
    read_count: int = 0
    input_frequency_sum: float = 0.0
    input_cell_frequency_sum: float = 0.0
    nt_records: Dict[str, NTRecord] = field(default_factory=dict)


@dataclass
class SampleStats:
    sample_id: str
    source_file: str
    output_file: str
    status: str = ""

    input_nt_rows: int = 0
    valid_nt_rows: int = 0
    invalid_nt_rows: int = 0
    valid_nt_row_ratio: float = 0.0

    total_reads_all_nt: int = 0
    total_reads_valid_nt: int = 0
    total_reads_invalid_nt: int = 0
    valid_read_ratio: float = 0.0

    unique_valid_nt_clones: int = 0
    aa_clone_number: int = 0
    aa_clone_reduction_ratio: float = 0.0

    frequency_source: str = "none"
    frequency_sum: float = 0.0
    frequency_norm_sum: float = 0.0
    read_fraction_sum: float = 0.0

    empty_sequence_rows: int = 0
    length_not_multiple_of_3_rows: int = 0
    non_acgt_base_rows: int = 0
    stop_codon_rows: int = 0
    malformed_columns_rows: int = 0
    invalid_read_count_rows: int = 0
    invalid_frequency_rows: int = 0
    invalid_rows_without_parseable_reads: int = 0

    qc_pass_01: bool = False
    has_warning_01: bool = False
    qc_warning_01: str = ""
    processing_seconds: float = 0.0


@dataclass
class InvalidReasonStats:
    invalid_nt_rows: int = 0
    invalid_read_count: int = 0
    rows_without_parseable_reads: int = 0


@dataclass
class ReceptorRunResult:
    receptor: str
    sample_count: int
    qc_pass_count: int
    qc_fail_count: int
    warning_count: int
    total_aa_clones: int
    output_dir: Path
    summary_file: Path
    exclude_file: Path


class InvalidDetailWriter:
    FIELDNAMES = [
        "sample_id",
        "line_number",
        "cdr3_nt",
        "read_count_raw",
        "input_frequency_raw",
        "input_cell_frequency_raw",
        "reason",
    ]

    def __init__(self, path: Path, max_rows: int) -> None:
        self.path = path
        self.max_rows = max_rows
        self.rows_written = 0
        self.rows_skipped_due_to_limit = 0
        self.handle: Optional[TextIO] = None
        self.writer: Optional[csv.DictWriter] = None

    def __enter__(self) -> "InvalidDetailWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(self.path, "wt", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDNAMES)
        self.writer.writeheader()
        return self

    def write(self, row: Dict[str, object]) -> None:
        if self.max_rows > 0 and self.rows_written >= self.max_rows:
            self.rows_skipped_due_to_limit += 1
            return
        if self.writer is None:
            raise RuntimeError("InvalidDetailWriter is not open")
        self.writer.writerow(row)
        self.rows_written += 1

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is not None:
            self.handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "构建配对队列的 TRB 与 IGH CDR3 AA clone table。"
            "两种组库分别处理、分别输出，本阶段不横向合并。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", required=True, help="配对 train/test metadata CSV。")
    parser.add_argument(
        "--mode",
        choices=("both", "trb", "igh"),
        default="both",
        help="处理两种组库或只处理其中一种。",
    )
    parser.add_argument("--patient-col", default="patient", help="患者ID列，用于完整性检查。")
    parser.add_argument("--trb-id-col", default="trb_libraryid", help="TRB library ID列。")
    parser.add_argument("--igh-id-col", default="igh_libraryid", help="IGH library ID列。")

    parser.add_argument("--trb-input-dir", default=None, help="TRB NT clone文件目录。")
    parser.add_argument("--igh-input-dir", default=None, help="IGH NT clone文件目录。")
    parser.add_argument("--trb-output-dir", default=None, help="TRB 01_AA_clone_table输出目录。")
    parser.add_argument("--igh-output-dir", default=None, help="IGH 01_AA_clone_table输出目录。")

    parser.add_argument("--trb-input-suffix", default=TRB_INPUT_SUFFIX)
    parser.add_argument("--trb-output-suffix", default=TRB_OUTPUT_SUFFIX)
    parser.add_argument("--igh-input-suffix", default=IGH_INPUT_SUFFIX)
    parser.add_argument("--igh-output-suffix", default=IGH_OUTPUT_SUFFIX)

    parser.add_argument(
        "--frequency-source",
        choices=("input_cell_frequency_sum", "input_frequency_sum", "read_fraction", "auto"),
        default="input_cell_frequency_sum",
        help=(
            "frequency列来源。auto按第4列、第3列、read_fraction顺序回退；"
            "正式分析建议保持默认以确保TRB和IGH定义一致。"
        ),
    )
    parser.add_argument(
        "--invalid-policy",
        choices=("skip", "error"),
        default="skip",
        help="无效行处理方式；skip会计入QC，error会立即停止。",
    )
    parser.add_argument(
        "--input-has-header",
        action="store_true",
        help="两种输入文件的第一条非空记录均为表头。当前无表头文件不要使用。",
    )
    parser.add_argument("--dry-run", action="store_true", help="完整计算和QC，但不写任何结果文件。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有AA表和报告。")
    parser.add_argument("--min-valid-read-ratio", type=float, default=0.70)
    parser.add_argument("--min-valid-nt-row-ratio", type=float, default=0.70)
    parser.add_argument("--min-nt-clones-warn", type=int, default=10)
    parser.add_argument("--norm-tolerance", type=float, default=1e-8)
    parser.add_argument("--top-n-low-quality", type=int, default=5)
    parser.add_argument("--invalid-example-per-reason", type=int, default=5)
    parser.add_argument("--write-invalid-log", action="store_true")
    parser.add_argument("--invalid-log-max-rows", type=int, default=100000)
    parser.add_argument(
        "--joint-summary-file",
        default=None,
        help=(
            "联合运行Markdown摘要。默认写在TRB和IGH输出目录的最近共同父目录下，"
            "文件名为01_TRB_IGH_build_summary.md。"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("min_valid_read_ratio", "min_valid_nt_row_ratio"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} 必须在0到1之间。")
    if args.min_nt_clones_warn < 0:
        raise ValueError("--min-nt-clones-warn不能为负数。")
    if args.norm_tolerance <= 0:
        raise ValueError("--norm-tolerance必须大于0。")
    if args.top_n_low_quality < 0:
        raise ValueError("--top-n-low-quality不能为负数。")
    if args.invalid_example_per_reason < 0:
        raise ValueError("--invalid-example-per-reason不能为负数。")
    if args.invalid_log_max_rows < 0:
        raise ValueError("--invalid-log-max-rows不能为负数。")

    if args.mode in ("both", "trb"):
        missing = [name for name in ("trb_input_dir", "trb_output_dir") if not getattr(args, name)]
        if missing:
            raise ValueError("mode包含TRB时必须提供--trb-input-dir和--trb-output-dir。")
    if args.mode in ("both", "igh"):
        missing = [name for name in ("igh_input_dir", "igh_output_dir") if not getattr(args, name)]
        if missing:
            raise ValueError("mode包含IGH时必须提供--igh-input-dir和--igh-output-dir。")


def normalize_nt(value: str) -> str:
    return value.strip().upper().replace("U", "T")


def translate_nt(nt: str) -> Tuple[Optional[str], Optional[str]]:
    if not nt:
        return None, "empty_sequence"
    if len(nt) % 3 != 0:
        return None, "length_not_multiple_of_3"
    if re.fullmatch(r"[ACGT]+", nt) is None:
        return None, "non_acgt_base"
    aa_chars: List[str] = []
    for start in range(0, len(nt), 3):
        aa = CODON_TABLE[nt[start:start + 3]]
        if aa == "*":
            return None, "stop_codon"
        aa_chars.append(aa)
    return "".join(aa_chars), None


def split_input_line(line: str) -> List[str]:
    return [token for token in re.split(r"[\s,]+", line.strip()) if token]


def parse_nonnegative_int(value: str) -> int:
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(value)
    return int(number)


def parse_nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(value)
    return number


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def format_float(value: object) -> str:
    return f"{float(value):.12g}"


def format_int(value: int) -> str:
    return f"{int(value):,}"


def format_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def read_metadata(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"metadata不存在: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"metadata没有表头: {path}")
        rows = []
        for row in reader:
            cleaned = {str(key): "" if value is None else str(value).strip() for key, value in row.items()}
            rows.append(cleaned)
    if not rows:
        raise ValueError(f"metadata中没有样本: {path}")
    return list(reader.fieldnames), rows


def validate_metadata(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    patient_col: str,
    receptor_configs: Sequence[ReceptorConfig],
) -> None:
    required = {patient_col}
    required.update(config.id_col for config in receptor_configs)
    absent = sorted(required - set(fieldnames))
    if absent:
        raise ValueError(
            "metadata缺少必要列: " + ", ".join(absent)
            + "; 可用列: " + ", ".join(fieldnames)
        )

    columns_to_check = [patient_col] + [config.id_col for config in receptor_configs]
    for column in columns_to_check:
        values: List[str] = []
        for row_number, row in enumerate(rows, start=2):
            value = str(row.get(column, "")).strip()
            if not value:
                raise ValueError(f"metadata第{row_number}行的{column}为空。")
            values.append(value)
        seen: Set[str] = set()
        duplicates: Set[str] = set()
        for value in values:
            if value in seen:
                duplicates.add(value)
            seen.add(value)
        if duplicates:
            raise ValueError(
                f"metadata列{column}存在重复值: " + ", ".join(sorted(duplicates)[:20])
            )


def sample_ids_from_metadata(rows: Sequence[Mapping[str, str]], id_col: str) -> List[str]:
    return [str(row[id_col]).strip() for row in rows]


def discover_samples(input_dir: Path, input_suffix: str) -> List[str]:
    files = sorted(input_dir.glob(f"*{input_suffix}"))
    if not files:
        raise FileNotFoundError(f"在{input_dir}中未发现*{input_suffix}")
    return [path.name[:-len(input_suffix)] for path in files]


def report_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "sample_summary": output_dir / "01_AA_clone_table_summary.csv",
        "sample_qc": output_dir / "01_sample_qc_status.csv",
        "invalid_summary": output_dir / "01_invalid_status_summary.csv",
        "invalid_by_sample": output_dir / "01_invalid_status_by_sample.csv",
        "norm_check": output_dir / "01_frequency_norm_check.csv",
        "exclude_samples": output_dir / "01_exclude_samples.txt",
        "invalid_examples": output_dir / "01_invalid_examples.csv",
        "markdown_summary": output_dir / "01_AA_clone_table_build_summary.md",
        "invalid_detail": output_dir / "01_invalid_nt_translation_log.csv.gz",
    }


def check_existing_outputs(
    output_files: Sequence[Path],
    reports: Mapping[str, Path],
    overwrite: bool,
    dry_run: bool,
    write_invalid_log: bool,
) -> List[Path]:
    candidates = list(output_files)
    for key, path in reports.items():
        if key == "invalid_detail" and not write_invalid_log:
            continue
        candidates.append(path)
    existing = [path for path in candidates if path.exists()]
    if existing and not overwrite and not dry_run:
        preview = "\n".join(f"  - {path}" for path in existing[:20])
        extra = "" if len(existing) <= 20 else f"\n  ...另有{len(existing) - 20}个"
        raise FileExistsError("发现已有输出文件。请确认后使用--overwrite:\n" + preview + extra)
    return existing


def preflight_receptor(
    config: ReceptorConfig,
    sample_ids: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Path], Dict[str, Path], List[Path]]:
    if not config.input_dir.is_dir():
        raise NotADirectoryError(f"{config.name}输入目录不存在: {config.input_dir}")

    input_paths = {
        sample_id: config.input_dir / f"{sample_id}{config.input_suffix}"
        for sample_id in sample_ids
    }
    missing_inputs = [path for path in input_paths.values() if not path.is_file()]
    if missing_inputs:
        preview = "\n".join(f"  - {path}" for path in missing_inputs[:20])
        extra = "" if len(missing_inputs) <= 20 else f"\n  ...另有{len(missing_inputs) - 20}个"
        raise FileNotFoundError(
            f"{config.name}缺少{len(missing_inputs)}个metadata对应输入文件:\n{preview}{extra}"
        )

    discovered = set(discover_samples(config.input_dir, config.input_suffix))
    extras = sorted(discovered - set(sample_ids))
    if extras:
        print(
            f"WARNING: {config.name}输入目录另有{len(extras)}个不在metadata中的文件，"
            "本次不会处理。"
        )

    reports = report_paths(config.output_dir)
    output_files = [
        config.output_dir / f"{sample_id}{config.output_suffix}"
        for sample_id in sample_ids
    ]
    existing = check_existing_outputs(
        output_files=output_files,
        reports=reports,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        write_invalid_log=args.write_invalid_log,
    )
    return input_paths, reports, existing


def add_invalid_reason(
    sample_stats: SampleStats,
    global_reason_stats: Dict[str, InvalidReasonStats],
    sample_reason_stats: Dict[str, InvalidReasonStats],
    reason: str,
    parseable_read_count: Optional[int],
) -> None:
    if reason not in INVALID_REASONS:
        raise AssertionError(f"Unknown invalid reason: {reason}")
    sample_stats.invalid_nt_rows += 1
    attr = f"{reason}_rows"
    setattr(sample_stats, attr, getattr(sample_stats, attr) + 1)

    global_record = global_reason_stats[reason]
    sample_record = sample_reason_stats[reason]
    global_record.invalid_nt_rows += 1
    sample_record.invalid_nt_rows += 1
    if parseable_read_count is None:
        sample_stats.invalid_rows_without_parseable_reads += 1
        global_record.rows_without_parseable_reads += 1
        sample_record.rows_without_parseable_reads += 1
    else:
        sample_stats.total_reads_invalid_nt += parseable_read_count
        global_record.invalid_read_count += parseable_read_count
        sample_record.invalid_read_count += parseable_read_count


def invalid_detail_row(
    sample_id: str,
    line_number: int,
    fields_in: Sequence[str],
    reason: str,
) -> Dict[str, object]:
    padded = list(fields_in[:4]) + [""] * max(0, 4 - len(fields_in))
    return {
        "sample_id": sample_id,
        "line_number": line_number,
        "cdr3_nt": padded[0],
        "read_count_raw": padded[1],
        "input_frequency_raw": padded[2],
        "input_cell_frequency_raw": padded[3],
        "reason": reason,
    }


def choose_frequency_source(
    requested_source: str,
    rows: Sequence[Dict[str, object]],
) -> Tuple[str, float]:
    sums = {
        "input_cell_frequency_sum": sum(float(row["input_cell_frequency_sum"]) for row in rows),
        "input_frequency_sum": sum(float(row["input_frequency_sum"]) for row in rows),
        "read_fraction": sum(float(row["read_fraction"]) for row in rows),
    }
    if requested_source == "auto":
        for source in ("input_cell_frequency_sum", "input_frequency_sum", "read_fraction"):
            if math.isfinite(sums[source]) and sums[source] > 0:
                return source, sums[source]
        return "none", 0.0
    total = sums[requested_source]
    if not math.isfinite(total) or total <= 0:
        return "none", total
    return requested_source, total


def build_one_sample(
    sample_id: str,
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    global_reason_stats: Dict[str, InvalidReasonStats],
    invalid_examples: Dict[str, List[Dict[str, object]]],
    invalid_writer: Optional[InvalidDetailWriter],
) -> Tuple[List[Dict[str, object]], SampleStats, Dict[str, InvalidReasonStats]]:
    started = time.perf_counter()
    stats = SampleStats(
        sample_id=sample_id,
        source_file=input_path.name,
        output_file=output_path.name,
    )
    sample_reason_stats = {reason: InvalidReasonStats() for reason in INVALID_REASONS}
    aa_records: Dict[str, AARecord] = {}
    first_nonempty_seen = False

    def record_invalid(
        reason: str,
        line_number: int,
        row_fields: Sequence[str],
        parseable_read_count: Optional[int],
    ) -> None:
        add_invalid_reason(
            sample_stats=stats,
            global_reason_stats=global_reason_stats,
            sample_reason_stats=sample_reason_stats,
            reason=reason,
            parseable_read_count=parseable_read_count,
        )
        detail = invalid_detail_row(sample_id, line_number, row_fields, reason)
        if (
            args.invalid_example_per_reason > 0
            and len(invalid_examples[reason]) < args.invalid_example_per_reason
        ):
            invalid_examples[reason].append(detail)
        if invalid_writer is not None:
            invalid_writer.write(detail)
        if args.invalid_policy == "error":
            raise ValueError(f"{input_path}第{line_number}行无效: {reason}")

    with input_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if args.input_has_header and not first_nonempty_seen:
                first_nonempty_seen = True
                continue
            first_nonempty_seen = True
            stats.input_nt_rows += 1

            row_fields = split_input_line(stripped)
            if len(row_fields) < 4:
                read_count: Optional[int] = None
                if len(row_fields) >= 2:
                    try:
                        read_count = parse_nonnegative_int(row_fields[1])
                        stats.total_reads_all_nt += read_count
                    except ValueError:
                        read_count = None
                record_invalid("malformed_columns", line_number, row_fields, read_count)
                continue

            try:
                read_count = parse_nonnegative_int(row_fields[1])
                stats.total_reads_all_nt += read_count
            except ValueError:
                record_invalid("invalid_read_count", line_number, row_fields, None)
                continue

            try:
                input_frequency = parse_nonnegative_float(row_fields[2])
                input_cell_frequency = parse_nonnegative_float(row_fields[3])
            except ValueError:
                record_invalid("invalid_frequency", line_number, row_fields, read_count)
                continue

            nt = normalize_nt(row_fields[0])
            aa, reason = translate_nt(nt)
            if aa is None:
                record_invalid(reason or "empty_sequence", line_number, row_fields, read_count)
                continue

            stats.valid_nt_rows += 1
            stats.total_reads_valid_nt += read_count
            aa_record = aa_records.setdefault(aa, AARecord())
            nt_record = aa_record.nt_records.setdefault(nt, NTRecord())
            nt_record.read_count += read_count
            nt_record.input_frequency_sum += input_frequency
            nt_record.input_cell_frequency_sum += input_cell_frequency
            aa_record.read_count += read_count
            aa_record.input_frequency_sum += input_frequency
            aa_record.input_cell_frequency_sum += input_cell_frequency

    if stats.input_nt_rows == 0:
        raise ValueError(f"输入文件没有数据行: {input_path}")

    stats.valid_nt_row_ratio = safe_ratio(stats.valid_nt_rows, stats.input_nt_rows)
    stats.valid_read_ratio = safe_ratio(stats.total_reads_valid_nt, stats.total_reads_all_nt)
    stats.unique_valid_nt_clones = sum(len(record.nt_records) for record in aa_records.values())
    stats.aa_clone_number = len(aa_records)
    stats.aa_clone_reduction_ratio = safe_ratio(stats.aa_clone_number, stats.unique_valid_nt_clones)

    rows: List[Dict[str, object]] = []
    if stats.aa_clone_number > 0 and stats.total_reads_valid_nt > 0:
        for aa, aa_record in aa_records.items():
            top_nt_cdr3 = sorted(
                aa_record.nt_records.items(),
                key=lambda item: (-item[1].read_count, item[0]),
            )[0][0]
            rows.append(
                {
                    "sample_id": sample_id,
                    "cdr3_aa": aa,
                    "aa_length": len(aa),
                    "nt_clone_number": len(aa_record.nt_records),
                    "read_count": aa_record.read_count,
                    "read_fraction": aa_record.read_count / stats.total_reads_valid_nt,
                    "input_frequency_sum": aa_record.input_frequency_sum,
                    "input_cell_frequency_sum": aa_record.input_cell_frequency_sum,
                    "frequency": 0.0,
                    "frequency_norm": 0.0,
                    "top_nt_cdr3": top_nt_cdr3,
                }
            )

        source, frequency_total = choose_frequency_source(args.frequency_source, rows)
        stats.frequency_source = source
        stats.frequency_sum = frequency_total
        if source != "none" and frequency_total > 0:
            for row in rows:
                row["frequency"] = float(row[source])
                row["frequency_norm"] = float(row["frequency"]) / frequency_total
            stats.frequency_norm_sum = sum(float(row["frequency_norm"]) for row in rows)
            stats.read_fraction_sum = sum(float(row["read_fraction"]) for row in rows)
            rows.sort(
                key=lambda row: (
                    -float(row["frequency_norm"]),
                    -int(row["read_count"]),
                    str(row["cdr3_aa"]),
                )
            )

    hard_fail_reasons: List[str] = []
    warning_reasons: List[str] = []
    if stats.input_nt_rows < args.min_nt_clones_warn:
        warning_reasons.append("原始NT clone行数过少")
    if stats.aa_clone_number == 0:
        hard_fail_reasons.append("无有效AA clone")
    if stats.total_reads_valid_nt == 0:
        hard_fail_reasons.append("有效翻译reads为0")
    if stats.frequency_source == "none":
        hard_fail_reasons.append("所选frequency来源总和为0或不可用")
    if 0 < stats.valid_read_ratio < args.min_valid_read_ratio:
        warning_reasons.append(f"有效reads比例偏低({stats.valid_read_ratio:.4f})")
    if 0 < stats.valid_nt_row_ratio < args.min_valid_nt_row_ratio:
        warning_reasons.append(f"有效NT行比例偏低({stats.valid_nt_row_ratio:.4f})")

    if stats.aa_clone_number > 0 and stats.frequency_source != "none":
        if abs(stats.frequency_norm_sum - 1.0) > args.norm_tolerance:
            hard_fail_reasons.append(f"frequency_norm求和异常({stats.frequency_norm_sum:.12g})")
        if abs(stats.read_fraction_sum - 1.0) > args.norm_tolerance:
            hard_fail_reasons.append(f"read_fraction求和异常({stats.read_fraction_sum:.12g})")

    all_reasons = hard_fail_reasons + warning_reasons
    stats.qc_pass_01 = len(hard_fail_reasons) == 0
    stats.has_warning_01 = len(all_reasons) > 0
    stats.qc_warning_01 = "; ".join(all_reasons)
    stats.processing_seconds = time.perf_counter() - started
    return rows, stats, sample_reason_stats


def write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def write_aa_table(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    float_columns = {
        "read_fraction",
        "input_frequency_sum",
        "input_cell_frequency_sum",
        "frequency",
        "frequency_norm",
    }
    formatted: List[Dict[str, object]] = []
    for row in rows:
        out = dict(row)
        for column in float_columns:
            out[column] = format_float(out[column])
        formatted.append(out)
    write_csv_atomic(path, OUTPUT_COLUMNS, formatted)


def sample_stats_rows(stats_list: Sequence[SampleStats]) -> List[Dict[str, object]]:
    float_names = {
        "valid_nt_row_ratio",
        "valid_read_ratio",
        "aa_clone_reduction_ratio",
        "frequency_sum",
        "frequency_norm_sum",
        "read_fraction_sum",
        "processing_seconds",
    }
    rows: List[Dict[str, object]] = []
    for stats in stats_list:
        row = asdict(stats)
        for name in float_names:
            row[name] = format_float(row[name])
        rows.append(row)
    return rows


def invalid_summary_rows(
    global_reason_stats: Mapping[str, InvalidReasonStats],
    total_invalid_rows: int,
    total_invalid_reads: int,
    total_reads_all: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for reason in INVALID_REASONS:
        record = global_reason_stats[reason]
        rows.append(
            {
                "translate_status": reason,
                "description_zh": REASON_CHINESE[reason],
                "invalid_nt_rows": record.invalid_nt_rows,
                "invalid_nt_row_ratio_among_invalid": format_float(
                    safe_ratio(record.invalid_nt_rows, total_invalid_rows)
                ),
                "invalid_read_count": record.invalid_read_count,
                "invalid_read_ratio_among_invalid": format_float(
                    safe_ratio(record.invalid_read_count, total_invalid_reads)
                ),
                "invalid_read_ratio_among_all_reads": format_float(
                    safe_ratio(record.invalid_read_count, total_reads_all)
                ),
                "rows_without_parseable_reads": record.rows_without_parseable_reads,
            }
        )
    rows.sort(key=lambda row: int(row["invalid_nt_rows"]), reverse=True)
    return rows


def invalid_by_sample_rows(
    all_sample_reason_stats: Mapping[str, Mapping[str, InvalidReasonStats]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for sample_id, reason_map in all_sample_reason_stats.items():
        for reason in INVALID_REASONS:
            record = reason_map[reason]
            if record.invalid_nt_rows == 0:
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "translate_status": reason,
                    "description_zh": REASON_CHINESE[reason],
                    "invalid_nt_rows": record.invalid_nt_rows,
                    "invalid_read_count": record.invalid_read_count,
                    "rows_without_parseable_reads": record.rows_without_parseable_reads,
                }
            )
    return rows


def mean_or_zero(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def median_or_zero(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def print_rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_configuration(
    config: ReceptorConfig,
    args: argparse.Namespace,
    metadata_path: Path,
    sample_ids: Sequence[str],
    existing_outputs: Sequence[Path],
) -> None:
    print_rule(f"01_AA_clone_table（{config.name}）：运行配置")
    print(f"脚本版本: {SCRIPT_VERSION}")
    print(f"输入目录: {config.input_dir}")
    print(f"输出目录: {config.output_dir}")
    print(f"metadata: {metadata_path}")
    print(f"样本ID列: {config.id_col}")
    print(f"待处理样本数: {len(sample_ids)}")
    print(f"输入文件后缀: {config.input_suffix}")
    print(f"输出文件后缀: {config.output_suffix}")
    print(f"frequency来源: {args.frequency_source}")
    print(f"无效行策略: {args.invalid_policy}")
    print(f"dry-run: {'是（完整计算QC但不写文件）' if args.dry_run else '否'}")
    print(f"覆盖旧文件: {'是' if args.overwrite else '否'}")
    if existing_outputs and args.dry_run:
        print(f"提示: 发现{len(existing_outputs)}个已有输出；dry-run不会修改。")


def print_progress(
    receptor: str,
    index: int,
    total: int,
    stats: SampleStats,
) -> None:
    marker = "PASS"
    if not stats.qc_pass_01:
        marker = "FAIL"
    elif stats.has_warning_01:
        marker = "WARN"
    print(
        f"[{receptor} {index:>3}/{total}] {stats.sample_id} | "
        f"NT={format_int(stats.input_nt_rows)} | "
        f"有效={format_int(stats.valid_nt_rows)} ({format_pct(stats.valid_nt_row_ratio)}) | "
        f"有效reads={format_pct(stats.valid_read_ratio)} | "
        f"AA={format_int(stats.aa_clone_number)} | "
        f"无效={format_int(stats.invalid_nt_rows)} | {marker} | {stats.status}"
    )


def write_markdown_summary(
    path: Path,
    config: ReceptorConfig,
    args: argparse.Namespace,
    metadata_path: Path,
    stats_list: Sequence[SampleStats],
    invalid_rows: Sequence[Mapping[str, object]],
    reports: Mapping[str, Path],
    invalid_writer: Optional[InvalidDetailWriter],
) -> None:
    total_input_rows = sum(item.input_nt_rows for item in stats_list)
    total_valid_rows = sum(item.valid_nt_rows for item in stats_list)
    total_invalid_rows = sum(item.invalid_nt_rows for item in stats_list)
    total_reads = sum(item.total_reads_all_nt for item in stats_list)
    total_valid_reads = sum(item.total_reads_valid_nt for item in stats_list)
    total_invalid_reads = sum(item.total_reads_invalid_nt for item in stats_list)

    lines = [
        f"# 01_AA_clone_table {config.name} Build Summary",
        "",
        "## Run configuration",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Input directory: `{config.input_dir}`",
        f"- Output directory: `{config.output_dir}`",
        f"- Metadata: `{metadata_path}`",
        f"- ID column: `{config.id_col}`",
        f"- Input suffix: `{config.input_suffix}`",
        f"- Output suffix: `{config.output_suffix}`",
        f"- Frequency source requested: `{args.frequency_source}`",
        "",
        "## Overall QC",
        "",
        f"- Samples: **{len(stats_list)}**",
        f"- QC-pass samples: **{sum(item.qc_pass_01 for item in stats_list)}**",
        f"- QC-fail samples: **{sum(not item.qc_pass_01 for item in stats_list)}**",
        f"- Samples with warning: **{sum(item.has_warning_01 for item in stats_list)}**",
        f"- Input NT rows: **{total_input_rows}**",
        f"- Valid NT rows: **{total_valid_rows} ({format_pct(safe_ratio(total_valid_rows, total_input_rows))})**",
        f"- Invalid NT rows: **{total_invalid_rows} ({format_pct(safe_ratio(total_invalid_rows, total_input_rows))})**",
        f"- Total parseable reads: **{total_reads}**",
        f"- Valid translated reads: **{total_valid_reads} ({format_pct(safe_ratio(total_valid_reads, total_reads), 4)})**",
        f"- Invalid-sequence reads: **{total_invalid_reads} ({format_pct(safe_ratio(total_invalid_reads, total_reads), 4)})**",
        "",
        "## Interpretation note",
        "",
        f"- Multiple {config.name} CDR3 NT sequences encoding the same AA sequence are intentionally merged.",
        "- `nt_clone_number` records the number of distinct NT variants contributing to each AA clone.",
        "- Frameshift-like lengths, stop codons and non-ACGT sequences are excluded and reported.",
        "",
        "## Invalid-reason summary",
        "",
        "| reason | description | invalid_nt_rows | pct_of_invalid_rows | invalid_reads | pct_of_all_reads | unparseable_read_rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in invalid_rows:
        lines.append(
            f"| {row['translate_status']} | {row['description_zh']} | "
            f"{row['invalid_nt_rows']} | "
            f"{format_pct(float(row['invalid_nt_row_ratio_among_invalid']))} | "
            f"{row['invalid_read_count']} | "
            f"{format_pct(float(row['invalid_read_ratio_among_all_reads']), 4)} | "
            f"{row['rows_without_parseable_reads']} |"
        )

    lines.extend(
        [
            "",
            "## Per-sample QC",
            "",
            "| sample_id | status | input_nt_rows | valid_nt_row_ratio | valid_read_ratio | unique_valid_nt | aa_clones | frequency_source | frequency_norm_sum | read_fraction_sum | qc_pass | warning |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for item in stats_list:
        warning = item.qc_warning_01.replace("|", "/")
        lines.append(
            f"| {item.sample_id} | {item.status} | {item.input_nt_rows} | "
            f"{item.valid_nt_row_ratio:.6f} | {item.valid_read_ratio:.6f} | "
            f"{item.unique_valid_nt_clones} | {item.aa_clone_number} | "
            f"{item.frequency_source} | {item.frequency_norm_sum:.12g} | "
            f"{item.read_fraction_sum:.12g} | {item.qc_pass_01} | {warning} |"
        )

    lines.extend(["", "## Output files", ""])
    for key, report_path in reports.items():
        if key == "invalid_detail" and not args.write_invalid_log:
            continue
        if key == "invalid_examples" and args.invalid_example_per_reason == 0:
            continue
        lines.append(f"- `{report_path}`")

    if args.write_invalid_log and invalid_writer is not None:
        lines.extend(
            [
                "",
                "## Detailed invalid-log status",
                "",
                f"- Rows written: **{invalid_writer.rows_written}**",
                f"- Rows omitted because of max-row limit: **{invalid_writer.rows_skipped_due_to_limit}**",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp_path.replace(path)


def write_formal_reports(
    config: ReceptorConfig,
    args: argparse.Namespace,
    metadata_path: Path,
    stats_list: Sequence[SampleStats],
    global_reason_stats: Mapping[str, InvalidReasonStats],
    all_sample_reason_stats: Mapping[str, Mapping[str, InvalidReasonStats]],
    invalid_examples: Mapping[str, List[Dict[str, object]]],
    reports: Mapping[str, Path],
    invalid_writer: Optional[InvalidDetailWriter],
) -> List[Dict[str, object]]:
    total_invalid_rows = sum(item.invalid_nt_rows for item in stats_list)
    total_invalid_reads = sum(item.total_reads_invalid_nt for item in stats_list)
    total_reads_all = sum(item.total_reads_all_nt for item in stats_list)
    invalid_rows = invalid_summary_rows(
        global_reason_stats,
        total_invalid_rows=total_invalid_rows,
        total_invalid_reads=total_invalid_reads,
        total_reads_all=total_reads_all,
    )

    sample_rows = sample_stats_rows(stats_list)
    sample_fieldnames = [field_info.name for field_info in fields(SampleStats)]
    write_csv_atomic(reports["sample_summary"], sample_fieldnames, sample_rows)
    write_csv_atomic(reports["sample_qc"], sample_fieldnames, sample_rows)

    write_csv_atomic(
        reports["invalid_summary"],
        [
            "translate_status",
            "description_zh",
            "invalid_nt_rows",
            "invalid_nt_row_ratio_among_invalid",
            "invalid_read_count",
            "invalid_read_ratio_among_invalid",
            "invalid_read_ratio_among_all_reads",
            "rows_without_parseable_reads",
        ],
        invalid_rows,
    )
    write_csv_atomic(
        reports["invalid_by_sample"],
        [
            "sample_id",
            "translate_status",
            "description_zh",
            "invalid_nt_rows",
            "invalid_read_count",
            "rows_without_parseable_reads",
        ],
        invalid_by_sample_rows(all_sample_reason_stats),
    )

    norm_rows = [
        {
            "sample_id": item.sample_id,
            "aa_clone_number": item.aa_clone_number,
            "frequency_source": item.frequency_source,
            "frequency_sum": format_float(item.frequency_sum),
            "frequency_norm_sum": format_float(item.frequency_norm_sum),
            "read_fraction_sum": format_float(item.read_fraction_sum),
            "norm_pass": (
                item.aa_clone_number > 0
                and abs(item.frequency_norm_sum - 1.0) <= args.norm_tolerance
                and abs(item.read_fraction_sum - 1.0) <= args.norm_tolerance
            ),
        }
        for item in stats_list
    ]
    write_csv_atomic(
        reports["norm_check"],
        [
            "sample_id",
            "aa_clone_number",
            "frequency_source",
            "frequency_sum",
            "frequency_norm_sum",
            "read_fraction_sum",
            "norm_pass",
        ],
        norm_rows,
    )

    temp_exclude = reports["exclude_samples"].with_suffix(".txt.tmp")
    temp_exclude.parent.mkdir(parents=True, exist_ok=True)
    with temp_exclude.open("w", encoding="utf-8") as handle:
        for item in stats_list:
            if not item.qc_pass_01:
                handle.write(item.sample_id + "\n")
    temp_exclude.replace(reports["exclude_samples"])

    if args.invalid_example_per_reason > 0:
        example_rows: List[Dict[str, object]] = []
        for reason in INVALID_REASONS:
            example_rows.extend(invalid_examples[reason])
        write_csv_atomic(
            reports["invalid_examples"],
            InvalidDetailWriter.FIELDNAMES,
            example_rows,
        )

    write_markdown_summary(
        reports["markdown_summary"],
        config=config,
        args=args,
        metadata_path=metadata_path,
        stats_list=stats_list,
        invalid_rows=invalid_rows,
        reports=reports,
        invalid_writer=invalid_writer,
    )
    return invalid_rows


def print_final_report(
    config: ReceptorConfig,
    args: argparse.Namespace,
    stats_list: Sequence[SampleStats],
    invalid_rows: Sequence[Mapping[str, object]],
    reports: Mapping[str, Path],
    invalid_writer: Optional[InvalidDetailWriter],
) -> None:
    input_samples = len(stats_list)
    pass_samples = sum(1 for item in stats_list if item.qc_pass_01)
    fail_samples = input_samples - pass_samples
    warning_samples = sum(1 for item in stats_list if item.has_warning_01)
    total_input_rows = sum(item.input_nt_rows for item in stats_list)
    total_valid_rows = sum(item.valid_nt_rows for item in stats_list)
    total_invalid_rows = sum(item.invalid_nt_rows for item in stats_list)
    total_reads = sum(item.total_reads_all_nt for item in stats_list)
    total_valid_reads = sum(item.total_reads_valid_nt for item in stats_list)
    total_invalid_reads = sum(item.total_reads_invalid_nt for item in stats_list)
    total_aa = sum(item.aa_clone_number for item in stats_list)

    print_rule(f"01_AA_clone_table（{config.name}）：运行完成")
    print(f"运行模式: {'DRY-RUN（未写文件）' if args.dry_run else '正式运行'}")
    print(f"输入样本数: {input_samples}")
    print(f"QC通过样本: {pass_samples}")
    print(f"QC失败样本: {fail_samples}")
    print(f"存在warning的样本: {warning_samples}")
    print(f"输出AA clone总数: {format_int(total_aa)}")
    print()
    print("[总体NT clone行与reads QC]")
    print(f"- 原始NT输入行: {format_int(total_input_rows)}")
    print(f"- 有效翻译NT行: {format_int(total_valid_rows)} ({format_pct(safe_ratio(total_valid_rows, total_input_rows))})")
    print(f"- 无效NT行: {format_int(total_invalid_rows)} ({format_pct(safe_ratio(total_invalid_rows, total_input_rows))})")
    print(f"- 全部可解析reads: {format_int(total_reads)}")
    print(f"- 有效翻译reads: {format_int(total_valid_reads)} ({format_pct(safe_ratio(total_valid_reads, total_reads), 4)})")
    print(f"- 无效序列对应reads: {format_int(total_invalid_reads)} ({format_pct(safe_ratio(total_invalid_reads, total_reads), 4)})")

    print()
    print("[无效NT原因汇总]")
    nonzero = [row for row in invalid_rows if int(row["invalid_nt_rows"]) > 0]
    if not nonzero:
        print("- 未发现无效NT序列。")
    else:
        for row in nonzero:
            print(
                f"- {row['translate_status']}（{row['description_zh']}）: "
                f"NT行{format_int(int(row['invalid_nt_rows']))}, "
                f"reads{format_int(int(row['invalid_read_count']))}"
            )

    valid_read_ratios = [item.valid_read_ratio for item in stats_list]
    valid_nt_ratios = [item.valid_nt_row_ratio for item in stats_list]
    print()
    print("[样本级QC概览]")
    print(
        "- valid_read_ratio: "
        f"mean={mean_or_zero(valid_read_ratios):.4f}, "
        f"median={median_or_zero(valid_read_ratios):.4f}, "
        f"min={min(valid_read_ratios):.4f}, max={max(valid_read_ratios):.4f}"
    )
    print(
        "- valid_nt_row_ratio: "
        f"mean={mean_or_zero(valid_nt_ratios):.4f}, "
        f"median={median_or_zero(valid_nt_ratios):.4f}, "
        f"min={min(valid_nt_ratios):.4f}, max={max(valid_nt_ratios):.4f}"
    )

    if args.top_n_low_quality > 0:
        print()
        print(f"[最低valid_read_ratio样本Top {args.top_n_low_quality}]")
        for item in sorted(stats_list, key=lambda value: value.valid_read_ratio)[:args.top_n_low_quality]:
            print(
                f"- {item.sample_id}: valid_read_ratio={item.valid_read_ratio:.4f}, "
                f"valid_nt_row_ratio={item.valid_nt_row_ratio:.4f}, "
                f"AA clones={format_int(item.aa_clone_number)}"
            )

    fail_list = [item for item in stats_list if not item.qc_pass_01]
    print()
    print("[QC失败样本，建议后续02/03/04默认排除]")
    if not fail_list:
        print("- 无。")
    else:
        for item in fail_list:
            print(f"- {item.sample_id}: {item.qc_warning_01}")

    abnormal_norm = [
        item
        for item in stats_list
        if item.aa_clone_number > 0
        and (
            abs(item.frequency_norm_sum - 1.0) > args.norm_tolerance
            or abs(item.read_fraction_sum - 1.0) > args.norm_tolerance
        )
    ]
    print()
    print("[归一化检查]")
    if not abnormal_norm:
        print(
            f"- PASS：全部样本frequency_norm与read_fraction求和均在"
            f"{args.norm_tolerance:g}容差内接近1。"
        )
    else:
        print(f"- FAIL：{len(abnormal_norm)}个样本求和异常，见QC文件。")

    print()
    print(f"[{config.name}结果解释提示]")
    print(f"- 多个{config.name} CDR3 NT序列翻译为同一AA序列时会合并。")
    print("- nt_clone_number保留其NT变体数量。")
    print("- 当前frequency默认使用输入第4列汇总值，与既有TRB/IGH流程保持一致。")

    print()
    if args.dry_run:
        print("[输出文件]")
        print("- 本次为dry-run：未写入AA clone表或QC报告。")
    else:
        print("[输出文件]")
        print(f"- AA clone表目录: {config.output_dir}")
        print(f"- 样本总结表: {reports['sample_summary']}")
        print(f"- 样本QC状态: {reports['sample_qc']}")
        print(f"- 无效原因汇总: {reports['invalid_summary']}")
        print(f"- 归一化检查: {reports['norm_check']}")
        print(f"- 建议排除样本: {reports['exclude_samples']}")
        print(f"- Markdown总结: {reports['markdown_summary']}")
        if args.write_invalid_log:
            print(f"- 逐行无效明细日志: {reports['invalid_detail']}")
            if invalid_writer is not None:
                print(
                    f"  已写{format_int(invalid_writer.rows_written)}行；"
                    f"受上限影响未写{format_int(invalid_writer.rows_skipped_due_to_limit)}行。"
                )


def run_receptor(
    config: ReceptorConfig,
    args: argparse.Namespace,
    metadata_path: Path,
    sample_ids: Sequence[str],
    input_paths: Mapping[str, Path],
    reports: Mapping[str, Path],
    existing_outputs: Sequence[Path],
) -> ReceptorRunResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print_configuration(
        config=config,
        args=args,
        metadata_path=metadata_path,
        sample_ids=sample_ids,
        existing_outputs=existing_outputs,
    )

    global_reason_stats = {reason: InvalidReasonStats() for reason in INVALID_REASONS}
    all_sample_reason_stats: Dict[str, Dict[str, InvalidReasonStats]] = {}
    invalid_examples: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    stats_list: List[SampleStats] = []

    invalid_writer_context: Optional[InvalidDetailWriter] = None
    if args.write_invalid_log and not args.dry_run:
        invalid_writer_context = InvalidDetailWriter(
            reports["invalid_detail"],
            max_rows=args.invalid_log_max_rows,
        )

    run_started = time.perf_counter()

    def process_all(invalid_writer: Optional[InvalidDetailWriter]) -> None:
        for index, sample_id in enumerate(sample_ids, start=1):
            input_path = input_paths[sample_id]
            output_path = config.output_dir / f"{sample_id}{config.output_suffix}"
            rows, sample_stats, sample_reasons = build_one_sample(
                sample_id=sample_id,
                input_path=input_path,
                output_path=output_path,
                args=args,
                global_reason_stats=global_reason_stats,
                invalid_examples=invalid_examples,
                invalid_writer=invalid_writer,
            )
            if args.dry_run:
                sample_stats.status = "WOULD_WRITE"
            else:
                write_aa_table(output_path, rows)
                sample_stats.status = "WRITTEN"
            stats_list.append(sample_stats)
            all_sample_reason_stats[sample_id] = sample_reasons
            print_progress(config.name, index, len(sample_ids), sample_stats)

    if invalid_writer_context is not None:
        with invalid_writer_context as opened_writer:
            process_all(opened_writer)
    else:
        process_all(None)

    total_invalid_rows = sum(item.invalid_nt_rows for item in stats_list)
    total_invalid_reads = sum(item.total_reads_invalid_nt for item in stats_list)
    total_reads_all = sum(item.total_reads_all_nt for item in stats_list)
    invalid_rows = invalid_summary_rows(
        global_reason_stats,
        total_invalid_rows=total_invalid_rows,
        total_invalid_reads=total_invalid_reads,
        total_reads_all=total_reads_all,
    )

    if not args.dry_run:
        invalid_rows = write_formal_reports(
            config=config,
            args=args,
            metadata_path=metadata_path,
            stats_list=stats_list,
            global_reason_stats=global_reason_stats,
            all_sample_reason_stats=all_sample_reason_stats,
            invalid_examples=invalid_examples,
            reports=reports,
            invalid_writer=invalid_writer_context,
        )

    print_final_report(
        config=config,
        args=args,
        stats_list=stats_list,
        invalid_rows=invalid_rows,
        reports=reports,
        invalid_writer=invalid_writer_context,
    )
    print(f"{config.name}总运行时间: {time.perf_counter() - run_started:.2f}秒")

    return ReceptorRunResult(
        receptor=config.name,
        sample_count=len(stats_list),
        qc_pass_count=sum(item.qc_pass_01 for item in stats_list),
        qc_fail_count=sum(not item.qc_pass_01 for item in stats_list),
        warning_count=sum(item.has_warning_01 for item in stats_list),
        total_aa_clones=sum(item.aa_clone_number for item in stats_list),
        output_dir=config.output_dir,
        summary_file=reports["markdown_summary"],
        exclude_file=reports["exclude_samples"],
    )


def default_joint_summary_path(configs: Sequence[ReceptorConfig]) -> Path:
    if len(configs) == 1:
        return configs[0].output_dir.parent / "01_TRB_IGH_build_summary.md"
    common = Path(os.path.commonpath([str(config.output_dir) for config in configs]))
    return common / "01_TRB_IGH_build_summary.md"


def write_joint_summary(
    path: Path,
    metadata_path: Path,
    args: argparse.Namespace,
    results: Sequence[ReceptorRunResult],
) -> None:
    lines = [
        "# Paired TRB+IGH 01_AA_clone_table Summary",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Metadata: `{metadata_path}`",
        f"- Mode: `{args.mode}`",
        f"- Dry run: `{args.dry_run}`",
        f"- Frequency source: `{args.frequency_source}`",
        "",
        "| receptor | samples | QC pass | QC fail | warnings | total AA clones | output directory |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result.receptor} | {result.sample_count} | {result.qc_pass_count} | "
            f"{result.qc_fail_count} | {result.warning_count} | {result.total_aa_clones} | "
            f"`{result.output_dir}` |"
        )
    lines.extend(
        [
            "",
            "## Important",
            "",
            "- TRB and IGH are processed independently in 01_AA_clone_table.",
            "- Do not merge clone rows across receptors.",
            "- Patient-level integration should occur only after receptor-specific feature construction.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("\n".join(lines), encoding="utf-8")
    temp.replace(path)


def build_configs(args: argparse.Namespace) -> List[ReceptorConfig]:
    configs: List[ReceptorConfig] = []
    if args.mode in ("both", "trb"):
        configs.append(
            ReceptorConfig(
                name="TRB",
                id_col=args.trb_id_col,
                input_dir=Path(args.trb_input_dir).expanduser().resolve(),
                output_dir=Path(args.trb_output_dir).expanduser().resolve(),
                input_suffix=args.trb_input_suffix,
                output_suffix=args.trb_output_suffix,
            )
        )
    if args.mode in ("both", "igh"):
        configs.append(
            ReceptorConfig(
                name="IGH",
                id_col=args.igh_id_col,
                input_dir=Path(args.igh_input_dir).expanduser().resolve(),
                output_dir=Path(args.igh_output_dir).expanduser().resolve(),
                input_suffix=args.igh_input_suffix,
                output_suffix=args.igh_output_suffix,
            )
        )
    return configs


def main() -> int:
    args = parse_args()
    validate_args(args)
    metadata_path = Path(args.metadata).expanduser().resolve()
    configs = build_configs(args)

    fieldnames, metadata_rows = read_metadata(metadata_path)
    validate_metadata(
        fieldnames=fieldnames,
        rows=metadata_rows,
        patient_col=args.patient_col,
        receptor_configs=configs,
    )

    # Preflight every requested receptor before writing any output.
    receptor_inputs: Dict[str, Dict[str, Path]] = {}
    receptor_reports: Dict[str, Dict[str, Path]] = {}
    receptor_existing: Dict[str, List[Path]] = {}
    receptor_sample_ids: Dict[str, List[str]] = {}

    for config in configs:
        sample_ids = sample_ids_from_metadata(metadata_rows, config.id_col)
        input_paths, reports, existing = preflight_receptor(config, sample_ids, args)
        receptor_sample_ids[config.name] = sample_ids
        receptor_inputs[config.name] = input_paths
        receptor_reports[config.name] = reports
        receptor_existing[config.name] = existing

    results: List[ReceptorRunResult] = []
    for config in configs:
        results.append(
            run_receptor(
                config=config,
                args=args,
                metadata_path=metadata_path,
                sample_ids=receptor_sample_ids[config.name],
                input_paths=receptor_inputs[config.name],
                reports=receptor_reports[config.name],
                existing_outputs=receptor_existing[config.name],
            )
        )

    print_rule("配对TRB+IGH 01_AA_clone_table：总览")
    print(f"metadata患者数: {len(metadata_rows)}")
    for result in results:
        print(
            f"- {result.receptor}: samples={result.sample_count}, "
            f"QC pass={result.qc_pass_count}, QC fail={result.qc_fail_count}, "
            f"warnings={result.warning_count}, AA clones={format_int(result.total_aa_clones)}"
        )

    if args.dry_run:
        print("- DRY-RUN：未写入联合或受体特异性结果文件。")
    else:
        joint_summary = (
            Path(args.joint_summary_file).expanduser().resolve()
            if args.joint_summary_file
            else default_joint_summary_path(configs)
        )
        write_joint_summary(
            path=joint_summary,
            metadata_path=metadata_path,
            args=args,
            results=results,
        )
        print(f"- 联合摘要: {joint_summary}")

    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nERROR: 用户中断运行。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
