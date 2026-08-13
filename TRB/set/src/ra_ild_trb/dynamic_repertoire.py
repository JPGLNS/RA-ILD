#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic all/Top-K repertoire resolution for TRB Feature Framework Phase 5."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import yaml

from .experiment_spec import AdvancedExperimentSpec

PathLike = Union[str, Path]


class DynamicRepertoireError(ValueError):
    pass


@dataclass(frozen=True)
class DynamicRepertoireSource:
    source_id: str
    representation: str
    top_k: Optional[int]
    aa_dir: Path
    summary_path: Optional[Path]
    sample_filename_template: str
    weight_column: str = "frequency_norm"

    def sample_filename(self, sample_id: str) -> str:
        sid = str(sample_id).strip()
        if not sid:
            raise DynamicRepertoireError("sample_id must be non-empty")
        return self.sample_filename_template.format(sample_id=sid)

    def sample_path(self, sample_id: str) -> Path:
        return self.aa_dir / self.sample_filename(sample_id)

    def as_dict(self) -> Dict[str, object]:
        return {
            "source_id": self.source_id,
            "representation": self.representation,
            "top_k": self.top_k,
            "aa_dir": str(self.aa_dir),
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "sample_filename_template": self.sample_filename_template,
            "weight_column": self.weight_column,
        }


def resolve_dynamic_repertoire(
    spec: AdvancedExperimentSpec,
    repository_root: PathLike,
) -> DynamicRepertoireSource:
    root = Path(repository_root).expanduser().resolve()
    if spec.repertoire.mode == "all":
        return DynamicRepertoireSource(
            source_id="all",
            representation="full",
            top_k=None,
            aa_dir=root / "TRB/result/01_AA_clone_table",
            summary_path=root / "TRB/result/01_AA_clone_table/01_AA_clone_table_summary.csv",
            sample_filename_template="{sample_id}_TRB_CDR3_AA_clone_table.csv",
        )
    k = int(spec.repertoire.top_k)
    return DynamicRepertoireSource(
        source_id=f"top{k}",
        representation="topk",
        top_k=k,
        aa_dir=root / f"TRB/result/01b_AA_clone_table_top{k}",
        summary_path=root / f"TRB/result/01b_AA_clone_table_top{k}/01b_top{k}_summary.csv",
        sample_filename_template=f"{{sample_id}}_TRB_CDR3_AA_clone_table_top{k}.csv",
    )


def normalize_material(value: object) -> str:
    text = str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if text == "pbmc":
        return "pbmc"
    if text in {"buffycoat", "buffercoat", "buffcoat"}:
        return "buffycoat"
    return text


def read_and_filter_metadata(
    metadata_path: PathLike,
    scope: str,
    *,
    id_col: str = "libraryid",
) -> pd.DataFrame:
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str)
    frame = frame.drop(columns=[c for c in frame.columns if str(c).startswith("Unnamed:")], errors="ignore")
    required = {id_col, "cohort"}
    if scope != "total":
        required.add("material")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DynamicRepertoireError(f"metadata missing columns: {missing}")
    ids = frame[id_col].astype(str).str.strip()
    if ids.eq("").any() or ids.isna().any() or ids.duplicated().any():
        raise DynamicRepertoireError(f"metadata {id_col} must be non-empty and unique")
    frame = frame.copy()
    frame[id_col] = ids
    cohort = frame["cohort"].astype(str).str.strip().str.upper().replace({"RA-ILD": "ILD", "RA_ILD": "ILD", "RAILD": "ILD"})
    if not cohort.isin({"RA", "ILD"}).all():
        bad = sorted(cohort[~cohort.isin({"RA", "ILD"})].unique().tolist())
        raise DynamicRepertoireError(f"unsupported cohort labels: {bad}")
    frame["cohort"] = cohort
    if scope != "total":
        normalized = frame["material"].map(normalize_material)
        frame = frame.loc[normalized == scope].copy()
    if frame.empty:
        raise DynamicRepertoireError(f"scope {scope!r} contains no samples")
    if set(frame["cohort"]) != {"RA", "ILD"}:
        raise DynamicRepertoireError(f"scope {scope!r} must contain both RA and ILD")
    return frame.reset_index(drop=True)


def validate_dynamic_source(
    source: DynamicRepertoireSource,
    sample_ids: Sequence[str],
    *,
    deep: bool = False,
) -> Dict[str, object]:
    ids = [str(x).strip() for x in sample_ids]
    if not source.aa_dir.is_dir():
        raise FileNotFoundError(f"repertoire directory not found: {source.aa_dir}")
    missing = [str(source.sample_path(sid)) for sid in ids if not source.sample_path(sid).is_file()]
    if missing:
        preview = "\n".join(f"  - {p}" for p in missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} repertoire files:\n{preview}")

    summary_rows = None
    if source.representation == "topk":
        if source.summary_path is None or not source.summary_path.is_file():
            raise FileNotFoundError(f"Top-K summary not found: {source.summary_path}")
        summary = pd.read_csv(source.summary_path)
        required = {"sample_id", "top_k", "retained_aa_clone_number", "qc_pass"}
        missing_cols = sorted(required - set(summary.columns))
        if missing_cols:
            raise DynamicRepertoireError(f"Top-K summary missing columns: {missing_cols}")
        summary["sample_id"] = summary["sample_id"].astype(str).str.strip()
        if summary["sample_id"].duplicated().any():
            raise DynamicRepertoireError("Top-K summary contains duplicate sample_id")
        sub = summary.set_index("sample_id")
        absent = [sid for sid in ids if sid not in sub.index]
        if absent:
            raise DynamicRepertoireError(f"Top-K summary missing samples: {absent[:20]}")
        chosen = sub.loc[ids]
        if not (pd.to_numeric(chosen["top_k"]) == int(source.top_k)).all():
            raise DynamicRepertoireError("Top-K summary top_k does not match experiment")
        if not (pd.to_numeric(chosen["retained_aa_clone_number"]) == int(source.top_k)).all():
            raise DynamicRepertoireError("Top-K summary retained_aa_clone_number is not exact K")
        q = chosen["qc_pass"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        if not q.all():
            raise DynamicRepertoireError("Top-K summary has qc_pass=false for selected samples")
        summary_rows = int(len(chosen))

    if deep:
        for sid in ids:
            path = source.sample_path(sid)
            frame = pd.read_csv(path, usecols=["sample_id", "cdr3_aa"])
            observed = frame["sample_id"].astype(str).str.strip().unique().tolist()
            if observed != [sid]:
                raise DynamicRepertoireError(f"sample/file mismatch for {sid}: {observed[:10]}")
            if source.top_k is not None and len(frame) != int(source.top_k):
                raise DynamicRepertoireError(f"{sid}: rows={len(frame)} expected={source.top_k}")

    return {
        "source_id": source.source_id,
        "representation": source.representation,
        "top_k": source.top_k,
        "sample_count": len(ids),
        "summary_rows_checked": summary_rows,
        "deep_validation": bool(deep),
        "status": "PASS",
    }


def _topk_registry_entry(k: int) -> dict:
    return {
        "status": "available",
        "representation": "topk",
        "source_layer": f"01b_AA_clone_table_top{k}",
        "default_path": f"TRB/result/01b_AA_clone_table_top{k}",
        "file_glob": f"*_TRB_CDR3_AA_clone_table_top{k}.csv",
        "sample_filename_template": f"{{sample_id}}_TRB_CDR3_AA_clone_table_top{k}.csv",
        "summary_kind": "step01b_topk",
        "summary_path": f"TRB/result/01b_AA_clone_table_top{k}/01b_top{k}_summary.csv",
        "parent_source": "all",
        "required_columns": [
            "sample_id", "topk_rank", "cdr3_aa", "aa_length", "nt_clone_number",
            "read_count", "read_fraction_full", "read_fraction", "input_frequency_sum",
            "input_cell_frequency_sum", "frequency", "frequency_norm_full",
            "frequency_norm", "top_nt_cdr3",
        ],
        "weight_column": "frequency_norm",
        "top_k": k,
    }


def render_dynamic_registry(
    base_registry_path: PathLike,
    spec: AdvancedExperimentSpec,
) -> Mapping[str, object]:
    path = Path(base_registry_path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if spec.repertoire.mode == "all":
        return raw
    k = int(spec.repertoire.top_k)
    source_id = f"top{k}"
    raw = json.loads(json.dumps(raw))
    sources = raw["repertoire_sources"]
    if source_id not in sources:
        sources[source_id] = _topk_registry_entry(k)
    # Groups already declared compatible with Top10000 are representation-compatible
    # with other deterministic Top-K sizes. All-only legacy groups remain all-only.
    for group in raw.get("feature_groups", {}).values():
        if group.get("repertoire_dependency") != "repertoire":
            continue
        supported = list(group.get("supported_repertoire_sources", []))
        if "top10000" in supported and source_id not in supported:
            supported.append(source_id)
            group["supported_repertoire_sources"] = supported
    return raw


def render_phase2_plan(spec: AdvancedExperimentSpec) -> Mapping[str, object]:
    groups = list(spec.features.clinical_groups) + list(spec.features.static_tcr_groups)
    # Phase2 calculates both 3-mer representations, but selecting one group in the plan
    # records the intended downstream representation.
    if spec.features.kmer.enabled:
        groups.append(f"tcr_3mer_{spec.features.kmer.representation}")
    return {
        "schema_version": "1.0",
        "plan": {
            "id": f"{spec.experiment_id}_phase2",
            "receptor": "TRB",
            "repertoire_source": spec.repertoire.source_id,
            "feature_groups": groups,
            "allow_legacy": False,
            "allow_planned": False,
        },
    }
