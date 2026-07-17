#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paired TRB+IGH entry point for final step-03 public CDR3-AA construction.

TRAIN:
  Build separate TRB and IGH public catalogs, fixed full-training reference
  sets, and descriptive sample-level features from the confirmed stability
  caches. The full-training descriptive features are self-inclusive and must
  not be used as fixed predictors inside ordinary cross-validation.

TEST:
  Apply each receptor's fixed TRAIN reference to the paired independent test
  samples. Test labels are never used to create or modify reference sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd

SCRIPT_VERSION = "1.0.0-PAIRED-TRB-IGH"
ENGINE_FILENAME = "build_03_public_features_receptor_v1_2_0.py"

RECEPTORS = {
    "TRB": {
        "id_arg": "trb_id_col",
        "aa_dir_arg": "trb_aa_dir",
        "summary_arg": "trb_summary_file",
        "output_arg": "trb_output_dir",
        "stability_config_arg": "trb_stability_configuration",
        "stability_cache_arg": "trb_stability_cache_dir",
        "reference_file_arg": "trb_reference_file",
        "reference_definition_arg": "trb_reference_definition",
        "suffix": "_TRB_CDR3_AA_clone_table.csv",
    },
    "IGH": {
        "id_arg": "igh_id_col",
        "aa_dir_arg": "igh_aa_dir",
        "summary_arg": "igh_summary_file",
        "output_arg": "igh_output_dir",
        "stability_config_arg": "igh_stability_configuration",
        "stability_cache_arg": "igh_stability_cache_dir",
        "reference_file_arg": "igh_reference_file",
        "reference_definition_arg": "igh_reference_definition",
        "suffix": "_IGH-without-DJ_CDR3_AA_clone_table.csv",
    },
}

TRAIN_OUTPUT_NAMES = [
    "03_descriptive_public_features.csv",
    "03_public_feature_build_log.csv",
    "03_public_feature_build_summary.md",
    "03_public_aa_catalog.csv.gz",
    "03_final_reference_public_sets.csv.gz",
    "03_reference_set_summary.csv",
    "03_reference_definition.json",
]
TEST_OUTPUT_NAMES = [
    "03_descriptive_public_features.csv",
    "03_public_feature_build_log.csv",
    "03_public_feature_build_summary.md",
    "03_test_reference_public_features.csv",
    "03_reference_definition_used.json",
]
CACHE_FILENAMES = [
    "candidate_presence_matrix.npz",
    "candidate_frequency_matrix.npz",
    "candidate_sequences.txt.gz",
    "sample_info.csv",
    "aa_clone_numbers.npy",
    "cache_manifest.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paired TRB+IGH final step-03 public features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", required=True, choices=["train", "test"])
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--patient-col", default="patient")
    parser.add_argument("--trb-id-col", default="trb_libraryid")
    parser.add_argument("--igh-id-col", default="igh_libraryid")
    parser.add_argument("--cohort-col", default="cohort")
    parser.add_argument("--scheme-name", default="main")
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--catalog-chunk-size", type=int, default=100_000)
    parser.add_argument("--engine-script", default=None)
    parser.add_argument("--joint-summary", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--trb-aa-dir", required=True)
    parser.add_argument("--igh-aa-dir", required=True)
    parser.add_argument("--trb-summary-file", required=True)
    parser.add_argument("--igh-summary-file", required=True)
    parser.add_argument("--trb-output-dir", required=True)
    parser.add_argument("--igh-output-dir", required=True)

    parser.add_argument("--trb-stability-configuration")
    parser.add_argument("--igh-stability-configuration")
    parser.add_argument("--trb-stability-cache-dir")
    parser.add_argument("--igh-stability-cache-dir")

    parser.add_argument("--trb-reference-file")
    parser.add_argument("--igh-reference-file")
    parser.add_argument("--trb-reference-definition")
    parser.add_argument("--igh-reference-definition")

    args = parser.parse_args()
    if args.epsilon <= 0:
        parser.error("--epsilon must be > 0.")
    if args.catalog_chunk_size < 1:
        parser.error("--catalog-chunk-size must be >= 1.")

    if args.mode == "train":
        required = [
            "trb_stability_configuration", "igh_stability_configuration",
            "trb_stability_cache_dir", "igh_stability_cache_dir",
        ]
    else:
        required = [
            "trb_reference_file", "igh_reference_file",
            "trb_reference_definition", "igh_reference_definition",
        ]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(
            f"The following arguments are required in {args.mode} mode: "
            + ", ".join("--" + x.replace("_", "-") for x in missing)
        )
    return args


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_metadata(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.metadata).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"metadata不存在: {path}")
    df = pd.read_csv(path, dtype=str)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    required = {args.patient_col, args.trb_id_col, args.igh_id_col}
    if args.mode == "train":
        required.add(args.cohort_col)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"metadata缺少列: {missing}; 实际列: {list(df.columns)}")

    for column in [args.patient_col, args.trb_id_col, args.igh_id_col]:
        df[column] = df[column].astype(str).str.strip()
        if df[column].isna().any() or df[column].eq("").any():
            raise ValueError(f"metadata的{column}存在空值。")
        if df[column].duplicated().any():
            examples = df.loc[df[column].duplicated(keep=False), column].unique().tolist()[:10]
            raise ValueError(f"metadata的{column}存在重复，例如: {examples}")

    if args.cohort_col in df.columns:
        df[args.cohort_col] = df[args.cohort_col].astype(str).str.strip().str.upper()
    if args.mode == "train":
        cohorts = set(df[args.cohort_col].unique())
        if cohorts != {"RA", "ILD"}:
            raise ValueError(
                f"训练metadata的{args.cohort_col}必须且只能包含RA/ILD，实际={sorted(cohorts)}"
            )
    return df.reset_index(drop=True)


def read_summary(path: Path, sample_ids: Sequence[str], receptor: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{receptor} 01 summary不存在: {path}")
    df = pd.read_csv(path)
    required = {"sample_id", "aa_clone_number"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{receptor} 01 summary缺少列: {missing}")
    df["sample_id"] = df["sample_id"].astype(str)
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{receptor} 01 summary中sample_id重复。")
    index = df.set_index("sample_id", drop=False)
    missing_ids = [x for x in sample_ids if x not in index.index]
    if missing_ids:
        raise ValueError(
            f"{receptor}有{len(missing_ids)}个metadata样本不在01 summary，例如: {missing_ids[:10]}"
        )
    return index.loc[list(sample_ids)].copy()


def expected_outputs(output_dir: Path, mode: str) -> List[Path]:
    names = TRAIN_OUTPUT_NAMES if mode == "train" else TEST_OUTPUT_NAMES
    return [output_dir / name for name in names]


def preflight_receptor(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    receptor: str,
) -> Dict[str, object]:
    cfg = RECEPTORS[receptor]
    id_col = getattr(args, cfg["id_arg"])
    sample_ids = metadata[id_col].astype(str).tolist()
    aa_dir = Path(getattr(args, cfg["aa_dir_arg"])).expanduser().resolve()
    summary_file = Path(getattr(args, cfg["summary_arg"])).expanduser().resolve()
    output_dir = Path(getattr(args, cfg["output_arg"])).expanduser().resolve()
    suffix = str(cfg["suffix"])

    if not aa_dir.is_dir():
        raise NotADirectoryError(f"{receptor} AA目录不存在: {aa_dir}")
    summary_df = read_summary(summary_file, sample_ids, receptor)
    missing_aa = [aa_dir / f"{sample_id}{suffix}" for sample_id in sample_ids]
    missing_aa = [p for p in missing_aa if not p.is_file()]
    if missing_aa:
        preview = "\n".join(f"  - {p}" for p in missing_aa[:20])
        raise FileNotFoundError(f"{receptor}缺失{len(missing_aa)}个AA表:\n{preview}")

    existing = [p for p in expected_outputs(output_dir, args.mode) if p.exists()]
    if existing and not args.overwrite:
        shown = "\n".join(f"  - {p}" for p in existing)
        raise FileExistsError(
            f"{receptor}以下输出已存在；确认覆盖时添加--overwrite:\n{shown}"
        )

    result: Dict[str, object] = {
        "receptor": receptor,
        "id_col": id_col,
        "sample_ids": sample_ids,
        "aa_dir": aa_dir,
        "summary_file": summary_file,
        "summary_df": summary_df,
        "output_dir": output_dir,
        "suffix": suffix,
    }

    if args.mode == "train":
        config_file = Path(getattr(args, cfg["stability_config_arg"])).expanduser().resolve()
        cache_dir = Path(getattr(args, cfg["stability_cache_arg"])).expanduser().resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"{receptor}稳定性配置不存在: {config_file}")
        missing_cache = [cache_dir / name for name in CACHE_FILENAMES if not (cache_dir / name).is_file()]
        if missing_cache:
            raise FileNotFoundError(
                f"{receptor}稳定性cache不完整:\n"
                + "\n".join(f"  - {p}" for p in missing_cache)
            )

        configuration = json.loads(config_file.read_text(encoding="utf-8"))
        schemes = [x for x in configuration.get("schemes", []) if x.get("name") == args.scheme_name]
        if len(schemes) != 1:
            raise ValueError(
                f"{receptor}稳定性配置中scheme={args.scheme_name}应恰好出现一次，实际={len(schemes)}"
            )
        manifest_file = cache_dir / "cache_manifest.json"
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if str(manifest.get("receptor", "")).upper() != receptor:
            raise RuntimeError(
                f"{receptor} cache manifest receptor={manifest.get('receptor')}不匹配。"
            )
        if manifest.get("aa_input_suffix") != suffix:
            raise RuntimeError(
                f"{receptor} cache suffix={manifest.get('aa_input_suffix')}，预期={suffix}。"
            )
        manifest_ids = [str(x) for x in manifest.get("sample_ids", [])]
        if manifest_ids and manifest_ids != sample_ids:
            raise RuntimeError(f"{receptor} cache manifest样本顺序与metadata不一致。")

        sample_info = pd.read_csv(cache_dir / "sample_info.csv", dtype=str)
        if id_col not in sample_info.columns:
            raise ValueError(f"{receptor} cache sample_info缺少{id_col}。")
        if sample_info[id_col].astype(str).tolist() != sample_ids:
            raise RuntimeError(f"{receptor} cache sample_info样本顺序与metadata不一致。")

        config_manifest = configuration.get("cache_manifest")
        if isinstance(config_manifest, Mapping):
            for key in [
                "receptor", "aa_input_suffix", "sample_ids",
                "candidate_min_total_count", "candidate_sequence_count",
                "candidate_incidence_count", "presence_shape", "presence_nnz",
                "frequency_nnz",
            ]:
                if key in config_manifest and config_manifest.get(key) != manifest.get(key):
                    raise RuntimeError(f"{receptor}稳定性配置与cache manifest字段不一致: {key}")

        result.update({
            "stability_configuration": config_file,
            "stability_cache_dir": cache_dir,
            "cache_manifest": manifest,
        })
    else:
        reference_file = Path(getattr(args, cfg["reference_file_arg"])).expanduser().resolve()
        definition_file = Path(getattr(args, cfg["reference_definition_arg"])).expanduser().resolve()
        if not reference_file.is_file():
            raise FileNotFoundError(f"{receptor}训练reference不存在: {reference_file}")
        if not definition_file.is_file():
            raise FileNotFoundError(f"{receptor}训练reference definition不存在: {definition_file}")
        definition = json.loads(definition_file.read_text(encoding="utf-8"))
        if str(definition.get("receptor", "")).upper() != receptor:
            raise RuntimeError(
                f"{receptor} reference definition receptor={definition.get('receptor')}不匹配。"
            )
        if definition.get("aa_input_suffix") != suffix:
            raise RuntimeError(
                f"{receptor} reference definition suffix={definition.get('aa_input_suffix')}，预期={suffix}。"
            )
        if definition.get("scheme", {}).get("name") != args.scheme_name:
            raise RuntimeError(
                f"{receptor} reference scheme={definition.get('scheme', {}).get('name')}，"
                f"命令要求={args.scheme_name}。"
            )
        actual_hash = sha256_file(reference_file)
        expected_hash = definition.get("reference_file_sha256")
        if actual_hash != expected_hash:
            raise RuntimeError(f"{receptor}训练reference SHA256与definition不一致。")
        result.update({
            "reference_file": reference_file,
            "reference_definition": definition_file,
            "definition": definition,
        })
    return result


def build_engine_command(
    args: argparse.Namespace,
    engine: Path,
    item: Mapping[str, object],
) -> List[str]:
    command = [
        sys.executable,
        str(engine),
        "--mode", args.mode,
        "--receptor", str(item["receptor"]),
        "--metadata", str(Path(args.metadata).expanduser().resolve()),
        "--summary-file", str(item["summary_file"]),
        "--aa-dir", str(item["aa_dir"]),
        "--output-dir", str(item["output_dir"]),
        "--id-col", str(item["id_col"]),
        "--cohort-col", args.cohort_col,
        "--scheme-name", args.scheme_name,
        "--epsilon", str(args.epsilon),
        "--aa-input-suffix", str(item["suffix"]),
        "--catalog-chunk-size", str(args.catalog_chunk_size),
    ]
    if args.mode == "train":
        command += [
            "--stability-configuration", str(item["stability_configuration"]),
            "--stability-cache-dir", str(item["stability_cache_dir"]),
        ]
    else:
        command += [
            "--reference-file", str(item["reference_file"]),
            "--reference-definition", str(item["reference_definition"]),
        ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def resolve_joint_summary(args: argparse.Namespace, items: Sequence[Mapping[str, object]]) -> Path:
    if args.joint_summary:
        return Path(args.joint_summary).expanduser().resolve()
    common = Path(os.path.commonpath([str(x["output_dir"]) for x in items]))
    return common / "03_TRB_IGH_public_features_summary.md"


def write_joint_summary(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    items: Sequence[Mapping[str, object]],
    path: Path,
    runtime: float,
) -> None:
    lines = [
        f"# 03 Paired TRB+IGH Public Features Summary — {args.mode.upper()}",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Paired patients: **{len(metadata)}**",
        f"- Scheme: `{args.scheme_name}`",
        f"- Metadata: `{Path(args.metadata).expanduser().resolve()}`",
        "",
    ]
    if args.mode == "train":
        lines.extend([
            "- Only the fixed training set and receptor-specific stability caches were used.",
            "- Full-training descriptive features are self-inclusive and are not fixed CV predictors.",
            "- Cross-validation must rebuild TRB and IGH reference sets within each training fold.",
            "",
        ])
        for item in items:
            definition_path = Path(item["output_dir"]) / "03_reference_definition.json"
            definition = json.loads(definition_path.read_text(encoding="utf-8"))
            sizes = definition["reference_set_sizes"]
            lines.extend([
                f"## {item['receptor']}",
                "",
                f"- Candidate sequences: **{definition['candidate_sequence_count']:,}**",
                f"- Global public: **{sizes['global_public']:,}**",
                f"- RA-specific: **{sizes['RA_specific_ref']:,}**",
                f"- ILD-specific: **{sizes['ILD_specific_ref']:,}**",
                f"- Shared: **{sizes['between_group_shared']:,}**",
                f"- Reference definition: `{definition_path}`",
                "",
            ])
    else:
        lines.extend([
            "- Each receptor used its own fixed full-training reference set.",
            "- Test cohort labels were not used to build or modify references.",
            "",
        ])
        for item in items:
            ref_features = pd.read_csv(
                Path(item["output_dir"]) / "03_test_reference_public_features.csv"
            )
            lines.extend([
                f"## {item['receptor']}",
                "",
                f"- Reference-feature matrix: **{ref_features.shape[0]} × {ref_features.shape[1]}**",
                f"- Output directory: `{item['output_dir']}`",
                "",
            ])
    lines.append(f"Total runtime: **{runtime:.2f} seconds**")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    args = parse_args()
    started = time.time()
    script_dir = Path(__file__).resolve().parent
    engine = (
        Path(args.engine_script).expanduser().resolve()
        if args.engine_script
        else script_dir / ENGINE_FILENAME
    )
    if not engine.is_file():
        raise FileNotFoundError(
            f"单受体引擎不存在: {engine}\n"
            f"请将{ENGINE_FILENAME}与本脚本放在同一目录，或使用--engine-script指定。"
        )

    metadata = read_metadata(args)
    items = [preflight_receptor(args, metadata, receptor) for receptor in ["TRB", "IGH"]]
    joint_summary = resolve_joint_summary(args, items)
    if joint_summary.exists() and not args.overwrite:
        raise FileExistsError(
            f"联合摘要已存在；确认覆盖时添加--overwrite: {joint_summary}"
        )

    print("=" * 86)
    print("03 Paired TRB+IGH final public-feature construction")
    print("=" * 86)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Paired patients: {len(metadata)}")
    if args.mode == "train":
        print(f"RA: {(metadata[args.cohort_col] == 'RA').sum()}")
        print(f"ILD: {(metadata[args.cohort_col] == 'ILD').sum()}")
    print(f"Scheme: {args.scheme_name}")
    print("Preflight: all paired metadata, AA tables, summaries and receptor-specific inputs passed.")
    if args.mode == "test":
        print("Test labels are not used to construct or modify reference sets.")

    if args.preflight_only:
        print("Preflight-only completed successfully; no output files were written.")
        return 0

    for item in items:
        print("\n" + "#" * 86)
        print(f"Start {item['receptor']} final public-feature construction")
        print("#" * 86 + "\n")
        command = build_engine_command(args, engine, item)
        completed = subprocess.run(command)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{item['receptor']} receptor engine failed with return code {completed.returncode}"
            )

    runtime = time.time() - started
    write_joint_summary(args, metadata, items, joint_summary, runtime)
    print("\n" + "=" * 86)
    print("Paired TRB+IGH final public-feature construction completed successfully")
    print("=" * 86)
    for item in items:
        descriptive = pd.read_csv(Path(item["output_dir"]) / "03_descriptive_public_features.csv")
        print(f"{item['receptor']}: descriptive shape={descriptive.shape}")
    print(f"Joint summary: {joint_summary}")
    print(f"Total runtime: {runtime:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
