#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run step-03 public CDR3-AA threshold-stability evaluation for paired TRB+IGH.

The paired metadata is validated once. TRB and IGH are then evaluated separately
with the same patient order, stratification columns, reference fraction, and
random seed. Consequently, each receptor uses identical reference/held-out
patient splits while maintaining receptor-specific public-sequence catalogs and
sparse caches.

Only the fixed training set is accepted. This script does not read the
independent test set and does not use model performance to select thresholds.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


SCRIPT_VERSION = "1.0.0-PAIRED-TRB-IGH"
TRB_SUFFIX = "_TRB_CDR3_AA_clone_table.csv"
IGH_SUFFIX = "_IGH-without-DJ_CDR3_AA_clone_table.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TRB and IGH public CDR3-AA threshold stability using the "
            "same paired training patients and identical resampling splits."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--trb-aa-dir", required=True)
    parser.add_argument("--igh-aa-dir", required=True)
    parser.add_argument("--trb-summary-file", required=True)
    parser.add_argument("--igh-summary-file", required=True)
    parser.add_argument("--trb-output-dir", required=True)
    parser.add_argument("--igh-output-dir", required=True)

    parser.add_argument("--patient-col", default="patient")
    parser.add_argument("--trb-id-col", default="trb_libraryid")
    parser.add_argument("--igh-id-col", default="igh_libraryid")
    parser.add_argument("--cohort-col", default="cohort")
    parser.add_argument("--batch-col", default="batch")
    parser.add_argument("--strata-cols", default="cohort,batch")

    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--reference-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--candidate-min-total-count", type=int, default=2)
    parser.add_argument("--sort-memory", default="4G")
    parser.add_argument("--sort-parallel", type=int, default=4)
    parser.add_argument("--max-sequence-output-rows", type=int, default=0)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--engine-script",
        default=None,
        help=(
            "Path to evaluate_03_public_thresholds_receptor.py. By default the "
            "script beside this paired launcher is used."
        ),
    )
    parser.add_argument(
        "--joint-summary-file",
        default=None,
        help=(
            "Optional paired summary path. Default: common parent of the two "
            "output directories / 03_TRB_IGH_threshold_stability_summary.md"
        ),
    )
    args = parser.parse_args()

    if args.iterations < 2:
        parser.error("--iterations must be >= 2")
    if not 0.0 < args.reference_fraction < 1.0:
        parser.error("--reference-fraction must be between 0 and 1")
    if args.epsilon <= 0:
        parser.error("--epsilon must be > 0")
    if args.candidate_min_total_count < 2:
        parser.error("--candidate-min-total-count must be >= 2")
    if args.sort_parallel < 1:
        parser.error("--sort-parallel must be >= 1")
    if args.max_sequence_output_rows < 0:
        parser.error("--max-sequence-output-rows must be >= 0")
    return args


def clean_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df.reset_index(drop=True)


def normalized_nonblank(series: pd.Series, column: str) -> pd.Series:
    values = series.astype("string").str.strip()
    if values.isna().any() or values.eq("").any():
        bad = values.index[values.isna() | values.eq("")].tolist()[:10]
        raise ValueError(f"metadata column '{column}' contains blank values at rows {bad}")
    return values.astype(str)


def validate_metadata(args: argparse.Namespace, path: Path) -> tuple[pd.DataFrame, List[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"metadata does not exist: {path}")
    df = clean_metadata(path)
    strata_cols = [x.strip() for x in args.strata_cols.split(",") if x.strip()]
    if not strata_cols:
        raise ValueError("--strata-cols must contain at least one column")

    required = {
        args.patient_col,
        args.trb_id_col,
        args.igh_id_col,
        args.cohort_col,
        *strata_cols,
    }
    if args.batch_col:
        required.add(args.batch_col)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"metadata missing columns: {missing}; available={list(df.columns)}")
    if df.empty:
        raise ValueError("metadata contains zero rows")

    for column in [args.patient_col, args.trb_id_col, args.igh_id_col, args.cohort_col, *strata_cols]:
        df[column] = normalized_nonblank(df[column], column)

    for column in [args.patient_col, args.trb_id_col, args.igh_id_col]:
        duplicated = df.loc[df[column].duplicated(keep=False), column].unique().tolist()
        if duplicated:
            raise ValueError(f"metadata column '{column}' is not unique, e.g. {duplicated[:10]}")

    df[args.cohort_col] = df[args.cohort_col].str.upper()
    cohorts = set(df[args.cohort_col])
    if cohorts != {"RA", "ILD"}:
        raise ValueError(f"{args.cohort_col} must contain exactly RA and ILD; observed={sorted(cohorts)}")

    return df, strata_cols


def validate_receptor_inputs(
    label: str,
    metadata: pd.DataFrame,
    id_col: str,
    aa_dir: Path,
    summary_file: Path,
    suffix: str,
) -> None:
    if not aa_dir.is_dir():
        raise NotADirectoryError(f"{label} AA directory does not exist: {aa_dir}")
    if not summary_file.is_file():
        raise FileNotFoundError(f"{label} 01 summary does not exist: {summary_file}")

    summary = pd.read_csv(summary_file, dtype={"sample_id": str})
    if "sample_id" not in summary.columns or "aa_clone_number" not in summary.columns:
        raise ValueError(f"{label} 01 summary lacks sample_id or aa_clone_number: {summary_file}")
    summary_ids = summary["sample_id"].astype(str).str.strip()
    if summary_ids.duplicated().any():
        raise ValueError(f"{label} 01 summary contains duplicate sample_id")

    ids = metadata[id_col].tolist()
    missing_summary = [sample_id for sample_id in ids if sample_id not in set(summary_ids)]
    if missing_summary:
        raise ValueError(f"{label}: {len(missing_summary)} metadata IDs absent from 01 summary, e.g. {missing_summary[:10]}")

    missing_files = [aa_dir / f"{sample_id}{suffix}" for sample_id in ids]
    missing_files = [p for p in missing_files if not p.is_file()]
    if missing_files:
        preview = "\n".join(f"  - {p}" for p in missing_files[:20])
        raise FileNotFoundError(f"{label}: missing {len(missing_files)} AA clone tables:\n{preview}")


def common_parent(a: Path, b: Path) -> Path:
    import os

    path = Path(os.path.commonpath([str(a), str(b)]))
    if path in {a, b}:
        return path.parent
    return path


def engine_command(
    args: argparse.Namespace,
    engine: Path,
    receptor: str,
    aa_dir: Path,
    summary_file: Path,
    id_col: str,
    output_dir: Path,
    suffix: str,
) -> List[str]:
    command = [
        sys.executable,
        str(engine),
        "--receptor", receptor,
        "--aa-dir", str(aa_dir),
        "--summary-file", str(summary_file),
        "--metadata", str(Path(args.metadata).expanduser().resolve()),
        "--id-col", id_col,
        "--cohort-col", args.cohort_col,
        "--batch-col", args.batch_col,
        "--strata-cols", args.strata_cols,
        "--aa-input-suffix", suffix,
        "--output-dir", str(output_dir),
        "--iterations", str(args.iterations),
        "--reference-fraction", str(args.reference_fraction),
        "--seed", str(args.seed),
        "--epsilon", str(args.epsilon),
        "--candidate-min-total-count", str(args.candidate_min_total_count),
        "--sort-memory", args.sort_memory,
        "--sort-parallel", str(args.sort_parallel),
        "--max-sequence-output-rows", str(args.max_sequence_output_rows),
    ]
    if args.reuse_cache:
        command.append("--reuse-cache")
    if args.keep_temp:
        command.append("--keep-temp")
    if args.overwrite:
        command.append("--overwrite")
    return command


def read_receptor_result(output_dir: Path, receptor: str) -> Dict[str, object]:
    config_file = output_dir / "03_threshold_stability_configuration.json"
    overview_file = output_dir / "03_threshold_stability_overview.csv"
    full_file = output_dir / "03_full_train_reference_set_sizes.csv"
    for path in [config_file, overview_file, full_file]:
        if not path.is_file():
            raise RuntimeError(f"{receptor} output is incomplete; missing: {path}")

    config = json.loads(config_file.read_text(encoding="utf-8"))
    overview = pd.read_csv(overview_file)
    full = pd.read_csv(full_file)
    main_full = full.loc[full["scheme"].eq("main")]
    if len(main_full) != 1:
        raise RuntimeError(f"{receptor}: expected one main row in {full_file}")
    main_row = main_full.iloc[0]

    main_overview = overview.loc[overview["scheme"].eq("main")].copy()
    median_jaccard = {
        str(row["category"]): float(row["median_jaccard_to_full"])
        for _, row in main_overview.iterrows()
    }
    return {
        "receptor": receptor,
        "training_samples": int(config["cache_manifest"]["presence_shape"][0]),
        "candidate_sequences": int(config["cache_manifest"]["candidate_sequence_count"]),
        "candidate_incidences": int(config["cache_manifest"]["candidate_incidence_count"]),
        "main_global": int(main_row["global_public_set_size"]),
        "main_RA_specific": int(main_row["RA_specific_set_size"]),
        "main_ILD_specific": int(main_row["ILD_specific_set_size"]),
        "main_shared": int(main_row["shared_set_size"]),
        "main_median_jaccard": median_jaccard,
        "output_dir": str(output_dir),
    }


def write_joint_summary(
    path: Path,
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    strata_cols: Sequence[str],
    results: Sequence[Dict[str, object]],
    runtime: float,
) -> None:
    lines = [
        "# 03 Paired TRB+IGH Public Threshold Stability Summary",
        "",
        "## Scope",
        "",
        "- Only the fixed paired training set was used.",
        "- TRB and IGH public catalogs and sparse caches were constructed independently.",
        "- The same metadata row order, stratification, reference fraction, and seed were used for both receptors, so resampling splits are identical.",
        "- The independent test set was not read and model AUC was not used.",
        "",
        "## Configuration",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Paired training patients: **{len(metadata)}**",
        f"- RA: **{int((metadata[args.cohort_col] == 'RA').sum())}**",
        f"- ILD: **{int((metadata[args.cohort_col] == 'ILD').sum())}**",
        f"- Iterations: **{args.iterations}**",
        f"- Reference fraction: **{args.reference_fraction:.2f}**",
        f"- Seed: **{args.seed}**",
        f"- Stratification: `{', '.join(strata_cols)}`",
        f"- Candidate minimum total sample count: **{args.candidate_min_total_count}**",
        f"- Total runtime: **{runtime:.2f} seconds**",
        "",
        "## Main-scheme overview",
        "",
        "| receptor | candidates | incidences | global | RA-specific | ILD-specific | shared | median Jaccard global | RA-specific | ILD-specific | shared |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        jac = result["main_median_jaccard"]
        lines.append(
            f"| {result['receptor']} | {result['candidate_sequences']:,} | "
            f"{result['candidate_incidences']:,} | {result['main_global']:,} | "
            f"{result['main_RA_specific']:,} | {result['main_ILD_specific']:,} | "
            f"{result['main_shared']:,} | {jac.get('global_public', float('nan')):.4f} | "
            f"{jac.get('RA_specific', float('nan')):.4f} | "
            f"{jac.get('ILD_specific', float('nan')):.4f} | "
            f"{jac.get('shared', float('nan')):.4f} |"
        )
    lines.extend(["", "## Receptor output directories", ""])
    for result in results:
        lines.append(f"- `{result['receptor']}`: `{result['output_dir']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    args = parse_args()
    started = time.time()

    metadata_file = Path(args.metadata).expanduser().resolve()
    trb_aa_dir = Path(args.trb_aa_dir).expanduser().resolve()
    igh_aa_dir = Path(args.igh_aa_dir).expanduser().resolve()
    trb_summary = Path(args.trb_summary_file).expanduser().resolve()
    igh_summary = Path(args.igh_summary_file).expanduser().resolve()
    trb_output = Path(args.trb_output_dir).expanduser().resolve()
    igh_output = Path(args.igh_output_dir).expanduser().resolve()

    if trb_output == igh_output:
        raise ValueError("TRB and IGH output directories must be different")

    script_dir = Path(__file__).resolve().parent
    engine = (
        Path(args.engine_script).expanduser().resolve()
        if args.engine_script
        else script_dir / "evaluate_03_public_thresholds_receptor.py"
    )
    if not engine.is_file():
        raise FileNotFoundError(f"single-receptor engine script does not exist: {engine}")
    if shutil.which("sort") is None:
        raise RuntimeError("GNU sort is not available in PATH")

    metadata, strata_cols = validate_metadata(args, metadata_file)
    validate_receptor_inputs(
        "TRB", metadata, args.trb_id_col, trb_aa_dir, trb_summary, TRB_SUFFIX
    )
    validate_receptor_inputs(
        "IGH", metadata, args.igh_id_col, igh_aa_dir, igh_summary, IGH_SUFFIX
    )

    print("\n" + "=" * 82)
    print("03 Paired TRB+IGH Public CDR3-AA threshold-stability evaluation")
    print("=" * 82)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Paired training patients: {len(metadata)}")
    print(f"RA: {(metadata[args.cohort_col] == 'RA').sum()}")
    print(f"ILD: {(metadata[args.cohort_col] == 'ILD').sum()}")
    print(f"Stratification: {strata_cols}")
    print(f"Iterations: {args.iterations}")
    print(f"Reference fraction: {args.reference_fraction}")
    print(f"Seed shared by TRB and IGH: {args.seed}")
    print("Preflight: all TRB and IGH AA tables and 01 summaries are present.")
    print("The independent test set is not used.\n")

    receptor_specs = [
        ("TRB", trb_aa_dir, trb_summary, args.trb_id_col, trb_output, TRB_SUFFIX),
        ("IGH", igh_aa_dir, igh_summary, args.igh_id_col, igh_output, IGH_SUFFIX),
    ]

    for receptor, aa_dir, summary, id_col, output, suffix in receptor_specs:
        print("\n" + "#" * 82)
        print(f"Start {receptor} threshold-stability evaluation")
        print("#" * 82 + "\n")
        sys.stdout.flush()
        command = engine_command(
            args, engine, receptor, aa_dir, summary, id_col, output, suffix
        )
        subprocess.run(command, check=True)

    results = [
        read_receptor_result(trb_output, "TRB"),
        read_receptor_result(igh_output, "IGH"),
    ]
    summary_file = (
        Path(args.joint_summary_file).expanduser().resolve()
        if args.joint_summary_file
        else common_parent(trb_output, igh_output)
        / "03_TRB_IGH_threshold_stability_summary.md"
    )
    runtime = time.time() - started
    write_joint_summary(
        summary_file, args, metadata, strata_cols, results, runtime
    )

    print("\n" + "=" * 82)
    print("Paired TRB+IGH threshold-stability evaluation completed successfully")
    print("=" * 82)
    for result in results:
        print(
            f"{result['receptor']}: candidates={result['candidate_sequences']:,}, "
            f"main global={result['main_global']:,}, "
            f"RA-specific={result['main_RA_specific']:,}, "
            f"ILD-specific={result['main_ILD_specific']:,}, "
            f"shared={result['main_shared']:,}"
        )
    print(f"Joint summary: {summary_file}")
    print(f"Total runtime: {runtime:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nERROR: interrupted by user", file=sys.stderr)
        raise SystemExit(130)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: receptor engine failed with return code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode or 1)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
