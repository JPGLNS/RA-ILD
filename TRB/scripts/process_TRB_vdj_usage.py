#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse TRB VDJ_Gene_Usage_Proportion files and prepare tcrdist3 clone tables.

Input directory (default):
    /data/users/chenhaisheng/RA-ILD/TRB/vdj/origin

Expected input filename:
    {libraryid}_TRB_VDJ_Gene_Usage_Proportion.txt

Input structure:
    >CDR3_NT<TAB>CDR3_TOTAL_COUNT
    V///D///J<TAB>ASSIGNMENT_COUNT<TAB>ASSIGNMENT_PERCENT
    ...

Processing:
    1. Expand each V/D/J assignment beneath its CDR3 header.
    2. Remove only assignment rows with ambiguous/multiple V genes (V contains ';').
    3. Ignore D for tcrdist3 preparation, but retain it in the detailed table.
    4. Aggregate NT-level duplicates by libraryid + V + J + CDR3_NT.
    5. Translate CDR3_NT in reading frame 0 using the standard genetic code.
    6. Exclude invalid translations from the final tcrdist table:
       - non-ACGT nucleotide characters
       - nucleotide length not divisible by 3
       - stop codon
       - empty translated sequence
    7. Aggregate the final table by libraryid + V + J + CDR3_AA.
       Counts are summed; distinct NT sequences are counted and the highest-count
       NT sequence is retained as top_cdr3_b_nt.

Outputs:
    result/nt_assignment/{libraryid}_TRB_VJ_CDR3_NT_assignment.csv
    result/tcrdist_clone_table/{libraryid}_TRB_tcrdist_clone_table.csv
    result/TRB_VJ_CDR3_NT_assignment_all_samples.csv
    result/TRB_tcrdist_clone_table_all_samples.csv
    result/TRB_vdj_processing_qc.csv
    result/TRB_vdj_excluded_records.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CODON_TABLE = {
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

INPUT_SUFFIX = "_TRB_VDJ_Gene_Usage_Proportion.txt"

NT_FIELDS = [
    "libraryid",
    "cdr3_b_nt",
    "cdr3_total_count",
    "count",
    "assignment_pct_sum",
    "v_b_gene",
    "d_b_gene_values",
    "j_b_gene",
    "translation_status",
    "cdr3_b_aa",
]

AA_FIELDS = [
    "libraryid",
    "clone_id",
    "count",
    "v_b_gene",
    "j_b_gene",
    "cdr3_b_aa",
    "aa_length",
    "nt_clone_number",
    "top_cdr3_b_nt",
    "top_cdr3_b_nt_count",
]

EXCLUDED_FIELDS = [
    "libraryid",
    "source_file",
    "line_no",
    "cdr3_b_nt",
    "cdr3_total_count",
    "raw_assignment",
    "v_b_gene",
    "d_b_gene",
    "j_b_gene",
    "count",
    "assignment_pct",
    "stage",
    "reason",
]

QC_FIELDS = [
    "libraryid",
    "source_file",
    "cdr3_blocks",
    "assignment_rows_total",
    "assignment_rows_kept",
    "assignment_rows_excluded_multi_v",
    "assignment_rows_excluded_multi_j",
    "assignment_rows_malformed",
    "orphan_assignment_rows",
    "nt_rows_after_aggregation",
    "nt_rows_valid_translation",
    "nt_rows_invalid_translation",
    "aa_rows_after_aggregation",
    "sum_assignment_count_total",
    "sum_assignment_count_kept",
    "sum_count_valid_translation",
    "status",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse TRB VDJ_Gene_Usage_Proportion files, remove ambiguous V "
            "assignments, translate CDR3 NT, and build tcrdist3 clone tables."
        )
    )
    parser.add_argument(
        "--indir",
        default="/data/users/chenhaisheng/RA-ILD/TRB/vdj/origin",
        help="Directory containing *_TRB_VDJ_Gene_Usage_Proportion.txt files",
    )
    parser.add_argument(
        "--outdir",
        default="/data/users/chenhaisheng/RA-ILD/TRB/vdj/result",
        help="Output directory",
    )
    parser.add_argument(
        "--pattern",
        default="*_TRB_VDJ_Gene_Usage_Proportion.txt",
        help="Input filename glob pattern",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-sample output files",
    )
    return parser.parse_args()


def libraryid_from_filename(path: Path) -> str:
    name = path.name
    if name.endswith(INPUT_SUFFIX):
        return name[: -len(INPUT_SUFFIX)]
    return path.stem


def safe_int(value: str) -> int:
    # Some text exports may contain integer-looking floats.
    number = float(str(value).strip())
    if number < 0 or not number.is_integer():
        raise ValueError(f"invalid non-negative integer count: {value!r}")
    return int(number)


def safe_float(value: str) -> float:
    return float(str(value).strip())


def translate_cdr3_nt(nt: str) -> Tuple[Optional[str], str]:
    seq = str(nt).strip().upper().replace("U", "T")
    if not seq:
        return None, "empty_nt"
    invalid = sorted(set(seq) - set("ACGT"))
    if invalid:
        return None, "invalid_nt_character:" + "".join(invalid)
    if len(seq) % 3 != 0:
        return None, f"out_of_frame_length:{len(seq)}"

    aa_chars: List[str] = []
    for i in range(0, len(seq), 3):
        codon = seq[i : i + 3]
        aa = CODON_TABLE[codon]
        aa_chars.append(aa)

    aa_seq = "".join(aa_chars)
    if "*" in aa_seq:
        return None, "stop_codon"
    if not aa_seq:
        return None, "empty_aa"
    return aa_seq, "valid"


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def format_float(value: float) -> str:
    return f"{value:.10f}".rstrip("0").rstrip(".")


def parse_sample_file(path: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    libraryid = libraryid_from_filename(path)

    current_nt: Optional[str] = None
    current_total_count: Optional[int] = None

    raw_kept: List[Dict[str, object]] = []
    excluded: List[Dict[str, object]] = []

    qc: Dict[str, object] = {
        "libraryid": libraryid,
        "source_file": str(path),
        "cdr3_blocks": 0,
        "assignment_rows_total": 0,
        "assignment_rows_kept": 0,
        "assignment_rows_excluded_multi_v": 0,
        "assignment_rows_excluded_multi_j": 0,
        "assignment_rows_malformed": 0,
        "orphan_assignment_rows": 0,
        "nt_rows_after_aggregation": 0,
        "nt_rows_valid_translation": 0,
        "nt_rows_invalid_translation": 0,
        "aa_rows_after_aggregation": 0,
        "sum_assignment_count_total": 0,
        "sum_assignment_count_kept": 0,
        "sum_count_valid_translation": 0,
        "status": "ok",
        "note": "",
    }

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                parts = line[1:].split("\t")
                if len(parts) < 2:
                    current_nt = None
                    current_total_count = None
                    qc["assignment_rows_malformed"] = int(qc["assignment_rows_malformed"]) + 1
                    excluded.append({
                        "libraryid": libraryid,
                        "source_file": str(path),
                        "line_no": line_no,
                        "cdr3_b_nt": parts[0].strip() if parts else "",
                        "cdr3_total_count": "",
                        "raw_assignment": line,
                        "v_b_gene": "",
                        "d_b_gene": "",
                        "j_b_gene": "",
                        "count": "",
                        "assignment_pct": "",
                        "stage": "parse_header",
                        "reason": "malformed_cdr3_header",
                    })
                    continue
                try:
                    current_nt = parts[0].strip().upper().replace("U", "T")
                    current_total_count = safe_int(parts[1])
                except ValueError as exc:
                    current_nt = None
                    current_total_count = None
                    qc["assignment_rows_malformed"] = int(qc["assignment_rows_malformed"]) + 1
                    excluded.append({
                        "libraryid": libraryid,
                        "source_file": str(path),
                        "line_no": line_no,
                        "cdr3_b_nt": parts[0].strip() if parts else "",
                        "cdr3_total_count": parts[1].strip() if len(parts) > 1 else "",
                        "raw_assignment": line,
                        "v_b_gene": "",
                        "d_b_gene": "",
                        "j_b_gene": "",
                        "count": "",
                        "assignment_pct": "",
                        "stage": "parse_header",
                        "reason": f"invalid_cdr3_header:{exc}",
                    })
                    continue
                qc["cdr3_blocks"] = int(qc["cdr3_blocks"]) + 1
                continue

            qc["assignment_rows_total"] = int(qc["assignment_rows_total"]) + 1

            if current_nt is None or current_total_count is None:
                qc["orphan_assignment_rows"] = int(qc["orphan_assignment_rows"]) + 1
                excluded.append({
                    "libraryid": libraryid,
                    "source_file": str(path),
                    "line_no": line_no,
                    "cdr3_b_nt": "",
                    "cdr3_total_count": "",
                    "raw_assignment": line,
                    "v_b_gene": "",
                    "d_b_gene": "",
                    "j_b_gene": "",
                    "count": "",
                    "assignment_pct": "",
                    "stage": "parse_assignment",
                    "reason": "assignment_without_valid_cdr3_header",
                })
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                qc["assignment_rows_malformed"] = int(qc["assignment_rows_malformed"]) + 1
                excluded.append({
                    "libraryid": libraryid,
                    "source_file": str(path),
                    "line_no": line_no,
                    "cdr3_b_nt": current_nt,
                    "cdr3_total_count": current_total_count,
                    "raw_assignment": line,
                    "v_b_gene": "",
                    "d_b_gene": "",
                    "j_b_gene": "",
                    "count": "",
                    "assignment_pct": "",
                    "stage": "parse_assignment",
                    "reason": "malformed_assignment_columns",
                })
                continue

            vdj = parts[0].strip().split("///")
            if len(vdj) != 3:
                qc["assignment_rows_malformed"] = int(qc["assignment_rows_malformed"]) + 1
                excluded.append({
                    "libraryid": libraryid,
                    "source_file": str(path),
                    "line_no": line_no,
                    "cdr3_b_nt": current_nt,
                    "cdr3_total_count": current_total_count,
                    "raw_assignment": line,
                    "v_b_gene": vdj[0].strip() if vdj else "",
                    "d_b_gene": "",
                    "j_b_gene": "",
                    "count": parts[1].strip(),
                    "assignment_pct": parts[2].strip(),
                    "stage": "parse_assignment",
                    "reason": "vdj_field_not_three_parts",
                })
                continue

            v_gene, d_gene, j_gene = (x.strip() for x in vdj)
            try:
                count = safe_int(parts[1])
                assignment_pct = safe_float(parts[2])
            except ValueError as exc:
                qc["assignment_rows_malformed"] = int(qc["assignment_rows_malformed"]) + 1
                excluded.append({
                    "libraryid": libraryid,
                    "source_file": str(path),
                    "line_no": line_no,
                    "cdr3_b_nt": current_nt,
                    "cdr3_total_count": current_total_count,
                    "raw_assignment": line,
                    "v_b_gene": v_gene,
                    "d_b_gene": d_gene,
                    "j_b_gene": j_gene,
                    "count": parts[1].strip(),
                    "assignment_pct": parts[2].strip(),
                    "stage": "parse_assignment",
                    "reason": f"invalid_count_or_percentage:{exc}",
                })
                continue

            qc["sum_assignment_count_total"] = int(qc["sum_assignment_count_total"]) + count

            if ";" in v_gene:
                qc["assignment_rows_excluded_multi_v"] = int(qc["assignment_rows_excluded_multi_v"]) + 1
                excluded.append({
                    "libraryid": libraryid,
                    "source_file": str(path),
                    "line_no": line_no,
                    "cdr3_b_nt": current_nt,
                    "cdr3_total_count": current_total_count,
                    "raw_assignment": line,
                    "v_b_gene": v_gene,
                    "d_b_gene": d_gene,
                    "j_b_gene": j_gene,
                    "count": count,
                    "assignment_pct": assignment_pct,
                    "stage": "v_filter",
                    "reason": "multiple_v_annotation",
                })
                continue

            if ";" in j_gene:
                qc["assignment_rows_excluded_multi_j"] = int(qc["assignment_rows_excluded_multi_j"]) + 1
                excluded.append({
                    "libraryid": libraryid,
                    "source_file": str(path),
                    "line_no": line_no,
                    "cdr3_b_nt": current_nt,
                    "cdr3_total_count": current_total_count,
                    "raw_assignment": line,
                    "v_b_gene": v_gene,
                    "d_b_gene": d_gene,
                    "j_b_gene": j_gene,
                    "count": count,
                    "assignment_pct": assignment_pct,
                    "stage": "j_filter",
                    "reason": "multiple_j_annotation",
                })
                continue

            if not v_gene or v_gene == "NA" or not j_gene or j_gene == "NA":
                qc["assignment_rows_malformed"] = int(qc["assignment_rows_malformed"]) + 1
                excluded.append({
                    "libraryid": libraryid,
                    "source_file": str(path),
                    "line_no": line_no,
                    "cdr3_b_nt": current_nt,
                    "cdr3_total_count": current_total_count,
                    "raw_assignment": line,
                    "v_b_gene": v_gene,
                    "d_b_gene": d_gene,
                    "j_b_gene": j_gene,
                    "count": count,
                    "assignment_pct": assignment_pct,
                    "stage": "vj_filter",
                    "reason": "missing_v_or_j_gene",
                })
                continue

            qc["assignment_rows_kept"] = int(qc["assignment_rows_kept"]) + 1
            qc["sum_assignment_count_kept"] = int(qc["sum_assignment_count_kept"]) + count
            raw_kept.append({
                "libraryid": libraryid,
                "cdr3_b_nt": current_nt,
                "cdr3_total_count": current_total_count,
                "count": count,
                "assignment_pct": assignment_pct,
                "v_b_gene": v_gene,
                "d_b_gene": d_gene,
                "j_b_gene": j_gene,
                "source_file": str(path),
                "line_no": line_no,
            })

    # NT-level aggregation: ignore D for clonotype identity but retain all observed D annotations.
    nt_groups: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for row in raw_kept:
        key = (
            str(row["libraryid"]),
            str(row["v_b_gene"]),
            str(row["j_b_gene"]),
            str(row["cdr3_b_nt"]),
        )
        if key not in nt_groups:
            nt_groups[key] = {
                "libraryid": row["libraryid"],
                "cdr3_b_nt": row["cdr3_b_nt"],
                "cdr3_total_count": row["cdr3_total_count"],
                "count": 0,
                "assignment_pct_sum": 0.0,
                "v_b_gene": row["v_b_gene"],
                "d_values": set(),
                "j_b_gene": row["j_b_gene"],
            }
        group = nt_groups[key]
        group["count"] = int(group["count"]) + int(row["count"])
        group["assignment_pct_sum"] = float(group["assignment_pct_sum"]) + float(row["assignment_pct"])
        d_value = str(row["d_b_gene"])
        if d_value:
            group["d_values"].add(d_value)
        # Header total should be identical within a CDR3; retain the maximum defensively.
        group["cdr3_total_count"] = max(int(group["cdr3_total_count"]), int(row["cdr3_total_count"]))

    nt_rows: List[Dict[str, object]] = []
    valid_nt_rows: List[Dict[str, object]] = []
    for key in sorted(nt_groups):
        group = nt_groups[key]
        aa_seq, translation_status = translate_cdr3_nt(str(group["cdr3_b_nt"]))
        nt_row = {
            "libraryid": group["libraryid"],
            "cdr3_b_nt": group["cdr3_b_nt"],
            "cdr3_total_count": group["cdr3_total_count"],
            "count": group["count"],
            "assignment_pct_sum": format_float(float(group["assignment_pct_sum"])),
            "v_b_gene": group["v_b_gene"],
            "d_b_gene_values": "|".join(sorted(group["d_values"])),
            "j_b_gene": group["j_b_gene"],
            "translation_status": translation_status,
            "cdr3_b_aa": aa_seq or "",
        }
        nt_rows.append(nt_row)

        if aa_seq is None:
            excluded.append({
                "libraryid": libraryid,
                "source_file": str(path),
                "line_no": "",
                "cdr3_b_nt": group["cdr3_b_nt"],
                "cdr3_total_count": group["cdr3_total_count"],
                "raw_assignment": "",
                "v_b_gene": group["v_b_gene"],
                "d_b_gene": "|".join(sorted(group["d_values"])),
                "j_b_gene": group["j_b_gene"],
                "count": group["count"],
                "assignment_pct": format_float(float(group["assignment_pct_sum"])),
                "stage": "translation",
                "reason": translation_status,
            })
        else:
            valid_nt_rows.append({**nt_row, "cdr3_b_aa": aa_seq})

    qc["nt_rows_after_aggregation"] = len(nt_rows)
    qc["nt_rows_valid_translation"] = len(valid_nt_rows)
    qc["nt_rows_invalid_translation"] = len(nt_rows) - len(valid_nt_rows)
    qc["sum_count_valid_translation"] = sum(int(row["count"]) for row in valid_nt_rows)

    # AA-level aggregation for tcrdist3: V + J + CDR3_AA.
    aa_groups: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for row in valid_nt_rows:
        key = (
            str(row["libraryid"]),
            str(row["v_b_gene"]),
            str(row["j_b_gene"]),
            str(row["cdr3_b_aa"]),
        )
        if key not in aa_groups:
            aa_groups[key] = {
                "libraryid": row["libraryid"],
                "count": 0,
                "v_b_gene": row["v_b_gene"],
                "j_b_gene": row["j_b_gene"],
                "cdr3_b_aa": row["cdr3_b_aa"],
                "nt_counts": defaultdict(int),
            }
        group = aa_groups[key]
        count = int(row["count"])
        group["count"] = int(group["count"]) + count
        group["nt_counts"][str(row["cdr3_b_nt"])] += count

    aa_rows: List[Dict[str, object]] = []
    for clone_number, key in enumerate(sorted(aa_groups), start=1):
        group = aa_groups[key]
        nt_counts: Dict[str, int] = dict(group["nt_counts"])
        top_nt, top_nt_count = sorted(nt_counts.items(), key=lambda x: (-x[1], x[0]))[0]
        aa_rows.append({
            "libraryid": group["libraryid"],
            "clone_id": f"{libraryid}_TRB_{clone_number:07d}",
            "count": group["count"],
            "v_b_gene": group["v_b_gene"],
            "j_b_gene": group["j_b_gene"],
            "cdr3_b_aa": group["cdr3_b_aa"],
            "aa_length": len(str(group["cdr3_b_aa"])),
            "nt_clone_number": len(nt_counts),
            "top_cdr3_b_nt": top_nt,
            "top_cdr3_b_nt_count": top_nt_count,
        })

    qc["aa_rows_after_aggregation"] = len(aa_rows)

    if int(qc["cdr3_blocks"]) == 0:
        qc["status"] = "warning"
        qc["note"] = "no valid CDR3 blocks parsed"
    elif int(qc["assignment_rows_kept"]) == 0:
        qc["status"] = "warning"
        qc["note"] = "no V/J assignment rows remained after filtering"

    return nt_rows, aa_rows, excluded, qc


def main() -> int:
    args = parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    nt_dir = outdir / "nt_assignment"
    aa_dir = outdir / "tcrdist_clone_table"

    if not indir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {indir}")

    input_files = sorted(indir.glob(args.pattern))
    if not input_files:
        raise FileNotFoundError(
            f"No input files matched pattern {args.pattern!r} in {indir}"
        )

    outdir.mkdir(parents=True, exist_ok=True)
    nt_dir.mkdir(parents=True, exist_ok=True)
    aa_dir.mkdir(parents=True, exist_ok=True)

    combined_nt: List[Dict[str, object]] = []
    combined_aa: List[Dict[str, object]] = []
    all_excluded: List[Dict[str, object]] = []
    qc_rows: List[Dict[str, object]] = []

    processed = 0
    skipped_existing = 0

    for path in input_files:
        libraryid = libraryid_from_filename(path)
        nt_path = nt_dir / f"{libraryid}_TRB_VJ_CDR3_NT_assignment.csv"
        aa_path = aa_dir / f"{libraryid}_TRB_tcrdist_clone_table.csv"

        nt_rows, aa_rows, excluded, qc = parse_sample_file(path)

        if (nt_path.exists() or aa_path.exists()) and not args.overwrite:
            skipped_existing += 1
            print(
                f"KEEP existing per-sample output for {libraryid}; "
                "combined files are rebuilt from the raw input. "
                "Use --overwrite to replace per-sample files.",
                file=sys.stderr,
            )
        else:
            write_csv(nt_path, nt_rows, NT_FIELDS)
            write_csv(aa_path, aa_rows, AA_FIELDS)

        combined_nt.extend(nt_rows)
        combined_aa.extend(aa_rows)
        all_excluded.extend(excluded)
        qc_rows.append(qc)
        processed += 1

    # Combined outputs reflect samples processed in this run. Use --overwrite for a full rebuild.
    write_csv(outdir / "TRB_VJ_CDR3_NT_assignment_all_samples.csv", combined_nt, NT_FIELDS)
    write_csv(outdir / "TRB_tcrdist_clone_table_all_samples.csv", combined_aa, AA_FIELDS)
    write_csv(outdir / "TRB_vdj_excluded_records.csv", all_excluded, EXCLUDED_FIELDS)
    write_csv(outdir / "TRB_vdj_processing_qc.csv", qc_rows, QC_FIELDS)

    print("Done.")
    print(f"Input directory: {indir}")
    print(f"Output directory: {outdir}")
    print(f"Matched input files: {len(input_files)}")
    print(f"Processed files: {processed}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Combined NT rows: {len(combined_nt)}")
    print(f"Combined AA rows: {len(combined_aa)}")
    print(f"Excluded/logged records: {len(all_excluded)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
