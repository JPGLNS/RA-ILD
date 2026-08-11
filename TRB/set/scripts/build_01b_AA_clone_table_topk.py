#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01b_AA_clone_table_topk
======================

Build an exact Top-K CDR3-AA repertoire from the full Step-01 AA clone table.

Selection rule (deterministic):
    1. frequency descending
    2. read_count descending
    3. cdr3_aa ascending

The original clone-level abundance columns are preserved. The original
full-repertoire normalization columns are renamed to:
    - read_fraction_full
    - frequency_norm_full

Within the retained Top-K repertoire, the canonical downstream columns are
recomputed:
    - read_fraction = read_count / sum(read_count among retained Top-K)
    - frequency_norm = frequency / sum(frequency among retained Top-K)

This script intentionally does NOT modify the full Step-01 tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_VERSION = "1.0.0"
DEFAULT_INPUT_GLOB = "*_AA_clone_table.csv"

REQUIRED_COLUMNS = [
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

OUTPUT_COLUMNS = [
    "sample_id",
    "topk_rank",
    "cdr3_aa",
    "aa_length",
    "nt_clone_number",
    "read_count",
    "read_fraction_full",
    "read_fraction",
    "input_frequency_sum",
    "input_cell_frequency_sum",
    "frequency",
    "frequency_norm_full",
    "frequency_norm",
    "top_nt_cdr3",
]

SUMMARY_COLUMNS = [
    "sample_id",
    "source_file",
    "output_file",
    "top_k",
    "full_aa_clone_number",
    "retained_aa_clone_number",
    "removed_aa_clone_number",
    "retained_clone_fraction",
    "full_read_count",
    "retained_read_count",
    "retained_read_mass",
    "full_frequency_sum",
    "retained_frequency_sum",
    "retained_frequency_mass",
    "cutoff_frequency",
    "cutoff_frequency_norm_full",
    "cutoff_read_count",
    "boundary_same_frequency_clone_number",
    "frequency_norm_sum_topk",
    "read_fraction_sum_topk",
    "qc_pass",
    "qc_warning",
]

QC_COLUMNS = [
    "sample_id",
    "input_rows_ge_k",
    "unique_cdr3_aa",
    "input_frequency_norm_sum",
    "input_frequency_norm_ok",
    "input_read_fraction_sum",
    "input_read_fraction_ok",
    "exact_k_ok",
    "topk_rank_ok",
    "topk_frequency_norm_sum",
    "topk_frequency_norm_ok",
    "topk_read_fraction_sum",
    "topk_read_fraction_ok",
    "retained_frequency_mass_consistent",
    "retained_read_mass_consistent",
    "boundary_order_ok",
    "preserved_fields_ok",
    "qc_pass",
    "qc_warning",
]

PRESERVED_FIELDS = [
    "sample_id",
    "cdr3_aa",
    "aa_length",
    "nt_clone_number",
    "read_count",
    "input_frequency_sum",
    "input_cell_frequency_sum",
    "frequency",
    "top_nt_cdr3",
]


class TopKError(RuntimeError):
    """Raised when a sample violates the Top-K input/output contract."""


@dataclass
class ParsedRow:
    raw: Dict[str, str]
    sample_id: str
    cdr3_aa: str
    read_count: int
    frequency: float
    read_fraction_full: float
    frequency_norm_full: float


@dataclass
class SamplePlan:
    source_path: Path
    sample_id: str
    output_path: Path
    selected: List[ParsedRow]
    summary: Dict[str, object]
    qc: Dict[str, object]


@dataclass
class SampleAudit:
    source_path: Path
    sample_id: str
    output_path: Path
    summary: Dict[str, object]
    qc: Dict[str, object]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从完整 Step-01 CDR3-AA 表构建严格、可审计的 Top-K AA repertoire。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, help="完整 Step-01 AA clone table 目录。")
    parser.add_argument("--output-dir", required=True, help="Step-01b Top-K 输出目录。")
    parser.add_argument("--top-k", type=int, default=10000, help="每个样本严格保留的 AA clonotype 数。")
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help="输入文件 glob；默认同时兼容仓库标准文件名和 *_AA_clone_table.csv。",
    )
    parser.add_argument(
        "--normalization-tol",
        type=float,
        default=1e-6,
        help="检查 full repertoire 归一化总和与 1 的允许误差。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只完成完整预检/QC，不写任何结果。")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的 Top-K 输出文件。")
    return parser.parse_args(argv)


def _finite_nonnegative(value: float, label: str, sample_id: str) -> float:
    if not math.isfinite(value):
        raise TopKError(f"{sample_id}: {label} 不是有限数值: {value!r}")
    if value < 0:
        raise TopKError(f"{sample_id}: {label} 为负数: {value!r}")
    return value


def _parse_int(text: str, label: str, sample_id: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise TopKError(f"{sample_id}: {label} 不是整数: {text!r}") from exc
    if value < 0:
        raise TopKError(f"{sample_id}: {label} 为负数: {value}")
    return value


def _parse_float(text: str, label: str, sample_id: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise TopKError(f"{sample_id}: {label} 不是数值: {text!r}") from exc
    return _finite_nonnegative(value, label, sample_id)


def _format_float(value: float) -> str:
    return format(value, ".17g")


def _approx_equal(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def discover_input_files(input_dir: Path, input_glob: str) -> List[Path]:
    if not input_dir.is_dir():
        raise TopKError(f"输入目录不存在或不是目录: {input_dir}")
    files = sorted(p for p in input_dir.glob(input_glob) if p.is_file())
    if not files:
        raise TopKError(f"输入目录未找到文件: {input_dir} / {input_glob}")
    return files


def load_sample(path: Path, normalization_tol: float) -> Tuple[List[ParsedRow], float, float]:
    rows: List[ParsedRow] = []
    seen_cdr3 = set()
    sample_ids = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise TopKError(f"{path}: 缺少 CSV header")
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise TopKError(f"{path}: 缺少必需列: {', '.join(missing)}")
        unexpected = [col for col in reader.fieldnames if col not in REQUIRED_COLUMNS]
        if unexpected:
            raise TopKError(
                f"{path}: 出现未定义列，Step-01b 为避免静默丢列而拒绝处理: {', '.join(unexpected)}"
            )

        for line_number, raw_row in enumerate(reader, start=2):
            raw = {key: (value if value is not None else "") for key, value in raw_row.items()}
            sample_id = raw["sample_id"].strip()
            cdr3_aa = raw["cdr3_aa"].strip()
            if not sample_id:
                raise TopKError(f"{path}:{line_number}: sample_id 为空")
            if sample_id in {".", ".."} or "/" in sample_id or "\\" in sample_id:
                raise TopKError(f"{path}:{line_number}: sample_id 含不安全路径字符: {sample_id!r}")
            if not cdr3_aa:
                raise TopKError(f"{path}:{line_number}: cdr3_aa 为空")
            if cdr3_aa in seen_cdr3:
                raise TopKError(f"{sample_id}: cdr3_aa 重复，AA 聚合层不应重复: {cdr3_aa}")
            seen_cdr3.add(cdr3_aa)
            sample_ids.add(sample_id)

            read_count = _parse_int(raw["read_count"].strip(), "read_count", sample_id)
            frequency = _parse_float(raw["frequency"].strip(), "frequency", sample_id)
            read_fraction_full = _parse_float(raw["read_fraction"].strip(), "read_fraction", sample_id)
            frequency_norm_full = _parse_float(raw["frequency_norm"].strip(), "frequency_norm", sample_id)

            rows.append(
                ParsedRow(
                    raw=raw,
                    sample_id=sample_id,
                    cdr3_aa=cdr3_aa,
                    read_count=read_count,
                    frequency=frequency,
                    read_fraction_full=read_fraction_full,
                    frequency_norm_full=frequency_norm_full,
                )
            )

    if not rows:
        raise TopKError(f"{path}: 没有数据行")
    if len(sample_ids) != 1:
        raise TopKError(f"{path}: 一个文件中出现多个 sample_id: {sorted(sample_ids)}")

    sample_id = next(iter(sample_ids))
    input_frequency_norm_sum = math.fsum(r.frequency_norm_full for r in rows)
    input_read_fraction_sum = math.fsum(r.read_fraction_full for r in rows)
    if not _approx_equal(input_frequency_norm_sum, 1.0, normalization_tol):
        raise TopKError(
            f"{sample_id}: full frequency_norm 总和不是 1: {input_frequency_norm_sum:.12g} "
            f"(tol={normalization_tol:g})"
        )
    if not _approx_equal(input_read_fraction_sum, 1.0, normalization_tol):
        raise TopKError(
            f"{sample_id}: full read_fraction 总和不是 1: {input_read_fraction_sum:.12g} "
            f"(tol={normalization_tol:g})"
        )

    return rows, input_frequency_norm_sum, input_read_fraction_sum


def deterministic_sort(rows: Iterable[ParsedRow]) -> List[ParsedRow]:
    return sorted(rows, key=lambda r: (-r.frequency, -r.read_count, r.cdr3_aa))


def _output_filename(sample_id: str, top_k: int) -> str:
    return f"{sample_id}_TRB_CDR3_AA_clone_table_top{top_k}.csv"


def build_sample_plan(
    path: Path,
    output_dir: Path,
    top_k: int,
    normalization_tol: float,
) -> SamplePlan:
    if top_k <= 0:
        raise TopKError(f"top_k 必须 > 0，当前为 {top_k}")

    rows, input_frequency_norm_sum, input_read_fraction_sum = load_sample(path, normalization_tol)
    sample_id = rows[0].sample_id
    if len(rows) < top_k:
        raise TopKError(
            f"{sample_id}: full AA clone number={len(rows)} < top_k={top_k}; "
            "固定-K 模式禁止静默保留不足 K 的样本。"
        )

    all_sorted = deterministic_sort(rows)
    selected = all_sorted[:top_k]

    full_read_count = sum(r.read_count for r in rows)
    retained_read_count = sum(r.read_count for r in selected)
    full_frequency_sum = math.fsum(r.frequency for r in rows)
    retained_frequency_sum = math.fsum(r.frequency for r in selected)
    if full_read_count <= 0 or retained_read_count <= 0:
        raise TopKError(f"{sample_id}: read_count 总和必须 > 0")
    if full_frequency_sum <= 0 or retained_frequency_sum <= 0:
        raise TopKError(f"{sample_id}: frequency 总和必须 > 0")

    retained_read_mass_from_full = math.fsum(r.read_fraction_full for r in selected)
    retained_frequency_mass_from_full = math.fsum(r.frequency_norm_full for r in selected)
    retained_read_mass_from_counts = retained_read_count / full_read_count
    retained_frequency_mass_from_values = retained_frequency_sum / full_frequency_sum

    mass_tol = max(normalization_tol, 1e-10)
    retained_read_mass_consistent = _approx_equal(
        retained_read_mass_from_full, retained_read_mass_from_counts, mass_tol
    )
    retained_frequency_mass_consistent = _approx_equal(
        retained_frequency_mass_from_full, retained_frequency_mass_from_values, mass_tol
    )
    if not retained_read_mass_consistent:
        raise TopKError(
            f"{sample_id}: read_fraction 与 read_count 推导的 retained mass 不一致: "
            f"{retained_read_mass_from_full:.12g} vs {retained_read_mass_from_counts:.12g}"
        )
    if not retained_frequency_mass_consistent:
        raise TopKError(
            f"{sample_id}: frequency_norm 与 frequency 推导的 retained mass 不一致: "
            f"{retained_frequency_mass_from_full:.12g} vs {retained_frequency_mass_from_values:.12g}"
        )

    cutoff = selected[-1]
    boundary_same_frequency_clone_number = sum(1 for r in all_sorted if r.frequency == cutoff.frequency)

    boundary_order_ok = True
    if len(all_sorted) > top_k:
        a = all_sorted[top_k - 1]
        b = all_sorted[top_k]
        boundary_order_ok = (-a.frequency, -a.read_count, a.cdr3_aa) <= (
            -b.frequency,
            -b.read_count,
            b.cdr3_aa,
        )
    if not boundary_order_ok:
        raise TopKError(f"{sample_id}: Top-K 边界排序检查失败")

    warnings: List[str] = []
    if boundary_same_frequency_clone_number > 1:
        warnings.append(f"boundary_frequency_tie:{boundary_same_frequency_clone_number}")

    topk_frequency_norm_sum = math.fsum(r.frequency / retained_frequency_sum for r in selected)
    topk_read_fraction_sum = math.fsum(r.read_count / retained_read_count for r in selected)
    exact_k_ok = len(selected) == top_k
    topk_rank_ok = exact_k_ok
    unique_cdr3_aa = len({r.cdr3_aa for r in rows}) == len(rows)
    preserved_fields_ok = True  # output is created directly from raw selected rows; rechecked on write.

    qc_pass = all(
        [
            len(rows) >= top_k,
            unique_cdr3_aa,
            _approx_equal(input_frequency_norm_sum, 1.0, normalization_tol),
            _approx_equal(input_read_fraction_sum, 1.0, normalization_tol),
            exact_k_ok,
            topk_rank_ok,
            _approx_equal(topk_frequency_norm_sum, 1.0, 1e-12),
            _approx_equal(topk_read_fraction_sum, 1.0, 1e-12),
            retained_frequency_mass_consistent,
            retained_read_mass_consistent,
            boundary_order_ok,
            preserved_fields_ok,
        ]
    )
    if not qc_pass:
        raise TopKError(f"{sample_id}: 内部 Top-K QC 未通过")

    output_path = output_dir / _output_filename(sample_id, top_k)
    summary: Dict[str, object] = {
        "sample_id": sample_id,
        "source_file": str(path),
        "output_file": str(output_path),
        "top_k": top_k,
        "full_aa_clone_number": len(rows),
        "retained_aa_clone_number": len(selected),
        "removed_aa_clone_number": len(rows) - len(selected),
        "retained_clone_fraction": len(selected) / len(rows),
        "full_read_count": full_read_count,
        "retained_read_count": retained_read_count,
        "retained_read_mass": retained_read_mass_from_full,
        "full_frequency_sum": full_frequency_sum,
        "retained_frequency_sum": retained_frequency_sum,
        "retained_frequency_mass": retained_frequency_mass_from_full,
        "cutoff_frequency": cutoff.frequency,
        "cutoff_frequency_norm_full": cutoff.frequency_norm_full,
        "cutoff_read_count": cutoff.read_count,
        "boundary_same_frequency_clone_number": boundary_same_frequency_clone_number,
        "frequency_norm_sum_topk": topk_frequency_norm_sum,
        "read_fraction_sum_topk": topk_read_fraction_sum,
        "qc_pass": True,
        "qc_warning": ";".join(warnings),
    }
    qc: Dict[str, object] = {
        "sample_id": sample_id,
        "input_rows_ge_k": len(rows) >= top_k,
        "unique_cdr3_aa": unique_cdr3_aa,
        "input_frequency_norm_sum": input_frequency_norm_sum,
        "input_frequency_norm_ok": _approx_equal(input_frequency_norm_sum, 1.0, normalization_tol),
        "input_read_fraction_sum": input_read_fraction_sum,
        "input_read_fraction_ok": _approx_equal(input_read_fraction_sum, 1.0, normalization_tol),
        "exact_k_ok": exact_k_ok,
        "topk_rank_ok": topk_rank_ok,
        "topk_frequency_norm_sum": topk_frequency_norm_sum,
        "topk_frequency_norm_ok": _approx_equal(topk_frequency_norm_sum, 1.0, 1e-12),
        "topk_read_fraction_sum": topk_read_fraction_sum,
        "topk_read_fraction_ok": _approx_equal(topk_read_fraction_sum, 1.0, 1e-12),
        "retained_frequency_mass_consistent": retained_frequency_mass_consistent,
        "retained_read_mass_consistent": retained_read_mass_consistent,
        "boundary_order_ok": boundary_order_ok,
        "preserved_fields_ok": preserved_fields_ok,
        "qc_pass": qc_pass,
        "qc_warning": ";".join(warnings),
    }
    return SamplePlan(
        source_path=path,
        sample_id=sample_id,
        output_path=output_path,
        selected=selected,
        summary=summary,
        qc=qc,
    )


def _build_output_row(row: ParsedRow, rank: int, retained_read_count: int, retained_frequency_sum: float) -> Dict[str, str]:
    out = {
        "sample_id": row.raw["sample_id"],
        "topk_rank": str(rank),
        "cdr3_aa": row.raw["cdr3_aa"],
        "aa_length": row.raw["aa_length"],
        "nt_clone_number": row.raw["nt_clone_number"],
        "read_count": row.raw["read_count"],
        "read_fraction_full": row.raw["read_fraction"],
        "read_fraction": _format_float(row.read_count / retained_read_count),
        "input_frequency_sum": row.raw["input_frequency_sum"],
        "input_cell_frequency_sum": row.raw["input_cell_frequency_sum"],
        "frequency": row.raw["frequency"],
        "frequency_norm_full": row.raw["frequency_norm"],
        "frequency_norm": _format_float(row.frequency / retained_frequency_sum),
        "top_nt_cdr3": row.raw["top_nt_cdr3"],
    }

    for field in PRESERVED_FIELDS:
        source_value = row.raw[field]
        output_value = out[field]
        if source_value != output_value:
            raise TopKError(
                f"{row.sample_id}: preserved field 被意外修改: {field}: {source_value!r} -> {output_value!r}"
            )
    return out


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_sample(plan: SamplePlan, overwrite: bool) -> None:
    if plan.output_path.exists() and not overwrite:
        raise TopKError(f"输出已存在，拒绝覆盖: {plan.output_path}；如确认重跑请使用 --overwrite")
    retained_read_count = int(plan.summary["retained_read_count"])
    retained_frequency_sum = float(plan.summary["retained_frequency_sum"])
    rows = (
        _build_output_row(row, rank, retained_read_count, retained_frequency_sum)
        for rank, row in enumerate(plan.selected, start=1)
    )
    _atomic_write_csv(plan.output_path, OUTPUT_COLUMNS, rows)

    # Read-back validation catches serialization/column-order mistakes.
    with plan.output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)
    if len(written) != int(plan.summary["top_k"]):
        raise TopKError(f"{plan.sample_id}: 写出后行数不是 exact K")
    ranks = [int(row["topk_rank"]) for row in written]
    if ranks != list(range(1, len(written) + 1)):
        raise TopKError(f"{plan.sample_id}: 写出后 topk_rank 不连续")
    freq_sum = math.fsum(float(row["frequency_norm"]) for row in written)
    read_sum = math.fsum(float(row["read_fraction"]) for row in written)
    if not _approx_equal(freq_sum, 1.0, 1e-12):
        raise TopKError(f"{plan.sample_id}: 写出后 frequency_norm 总和异常: {freq_sum}")
    if not _approx_equal(read_sum, 1.0, 1e-12):
        raise TopKError(f"{plan.sample_id}: 写出后 read_fraction 总和异常: {read_sum}")


def write_metadata_outputs(output_dir: Path, top_k: int, audits: Sequence[SampleAudit], args: argparse.Namespace) -> None:
    summary_path = output_dir / f"01b_top{top_k}_summary.csv"
    qc_path = output_dir / f"01b_top{top_k}_qc.csv"
    config_path = output_dir / f"01b_top{top_k}_configuration.json"

    _atomic_write_csv(summary_path, SUMMARY_COLUMNS, (p.summary for p in audits))
    _atomic_write_csv(qc_path, QC_COLUMNS, (p.qc for p in audits))

    config = {
        "schema_version": 1,
        "script": "build_01b_AA_clone_table_topk.py",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "input_glob": args.input_glob,
        "top_k": top_k,
        "selection_unit": "CDR3-AA clonotype after Step-01 synonymous NT collapse",
        "sort_rule": ["frequency DESC", "read_count DESC", "cdr3_aa ASC"],
        "selection_rule": "exact first K rows after deterministic sorting",
        "preserved_columns": PRESERVED_FIELDS,
        "full_normalization_columns": ["frequency_norm_full", "read_fraction_full"],
        "recomputed_columns": {
            "frequency_norm": "frequency / sum(frequency within retained Top-K)",
            "read_fraction": "read_count / sum(read_count within retained Top-K)",
        },
        "normalization_tolerance_input": args.normalization_tol,
        "sample_count": len(audits),
        "all_samples_qc_pass": all(bool(p.qc["qc_pass"]) for p in audits),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_path.with_name(config_path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, config_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def preflight(args: argparse.Namespace) -> List[SampleAudit]:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    files = discover_input_files(input_dir, args.input_glob)

    seen_sample_ids = set()
    audits: List[SampleAudit] = []
    for path in files:
        # build_sample_plan holds one sample in memory only. Store only compact QC/audit
        # metadata so cohort-scale preflight does not retain all clone tables in RAM.
        plan = build_sample_plan(path, output_dir, args.top_k, args.normalization_tol)
        if plan.sample_id in seen_sample_ids:
            raise TopKError(
                f"sample_id 在多个输入文件中重复: {plan.sample_id}; 请收紧 --input-glob 或清理输入目录。"
            )
        seen_sample_ids.add(plan.sample_id)
        audits.append(
            SampleAudit(
                source_path=plan.source_path,
                sample_id=plan.sample_id,
                output_path=plan.output_path,
                summary=plan.summary,
                qc=plan.qc,
            )
        )
        print(
            f"[QC PASS] {plan.sample_id}: full={plan.summary['full_aa_clone_number']} "
            f"-> top{args.top_k}; retained_frequency_mass={plan.summary['retained_frequency_mass']:.6f}; "
            f"retained_read_mass={plan.summary['retained_read_mass']:.6f}"
        )

    return audits


def run_pipeline(args: argparse.Namespace) -> List[SampleAudit]:
    if args.top_k <= 0:
        raise TopKError("--top-k 必须 > 0")
    if args.normalization_tol <= 0:
        raise TopKError("--normalization-tol 必须 > 0")

    # Phase 1: validate ALL samples before any output is written. Only compact
    # per-sample audit metadata is retained, avoiding cohort-scale clone-table RAM use.
    audits = preflight(args)
    print(f"\nPreflight PASS: {len(audits)} samples; exact Top-K={args.top_k}.")

    if args.dry_run:
        print("DRY-RUN: 未写入任何 Top-K 结果。")
        return audits

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = [p.output_path for p in audits if p.output_path.exists()]
    if existing and not args.overwrite:
        preview = "\n".join(f"  - {p}" for p in existing[:10])
        extra = "" if len(existing) <= 10 else f"\n  ... 另有 {len(existing) - 10} 个"
        raise TopKError(
            "发现已存在的 Top-K 样本输出，正式运行前未写任何新样本结果。\n"
            + preview
            + extra
            + "\n如确认需要覆盖，请使用 --overwrite。"
        )

    # Phase 2: re-read one sample at a time and write only after the cohort-wide
    # preflight has passed. This trades extra I/O for fail-before-write safety.
    for audit in audits:
        plan = build_sample_plan(
            audit.source_path, output_dir, args.top_k, args.normalization_tol
        )
        if plan.sample_id != audit.sample_id or plan.summary != audit.summary:
            raise TopKError(
                f"{audit.sample_id}: preflight 与写出阶段结果不一致；输入文件可能在运行期间发生变化。"
            )
        write_sample(plan, overwrite=args.overwrite)

    write_metadata_outputs(output_dir, args.top_k, audits, args)

    print(f"WRITE PASS: {len(audits)} samples -> {output_dir}")
    print(f"Summary: {output_dir / f'01b_top{args.top_k}_summary.csv'}")
    print(f"QC:      {output_dir / f'01b_top{args.top_k}_qc.csv'}")
    return audits


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run_pipeline(args)
    except TopKError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
