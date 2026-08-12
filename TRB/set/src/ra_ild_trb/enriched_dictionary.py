#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable enriched-dictionary fit/transform primitives.

The mathematical definitions intentionally match the existing exploratory
``enrich_dictionary_cv.py`` implementation:

* dictionary membership is based on CDR3-AA sample prevalence;
* RA dictionary: P_RA >= T and P_RA - P_ILD >= Delta;
* ILD dictionary: P_ILD >= T and P_ILD - P_RA >= Delta;
* RA_dict_hit_rate = sample RA-dictionary hits / RA dictionary size;
* RA_hit_fraction_of_sample = sample RA-dictionary hits / sample unique clones.

No abundance cutoff is applied. read_fraction is only used for cumulative-
abundance metrics, not dictionary membership or hit-rate metrics.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .dynamic_repertoire import DynamicRepertoireSource

PathLike = Union[str, Path]


class EnrichedDictionaryError(ValueError):
    pass


@dataclass(frozen=True)
class SampleRepertoire:
    sample_id: str
    clones: Tuple[str, ...]
    read_fraction: Mapping[str, float]

    @property
    def unique_clone_count(self) -> int:
        return len(self.clones)


@dataclass(frozen=True)
class EnrichedReference:
    threshold_pct: float
    delta_pct: float
    source_id: str
    n_ra_train: int
    n_ild_train: int
    ra_threshold_count: int
    ild_threshold_count: int
    ra_clones: Tuple[str, ...]
    ild_clones: Tuple[str, ...]
    member_table: pd.DataFrame

    @property
    def ra_size(self) -> int:
        return len(self.ra_clones)

    @property
    def ild_size(self) -> int:
        return len(self.ild_clones)

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "threshold_pct": self.threshold_pct,
            "delta_pct": self.delta_pct,
            "source_id": self.source_id,
            "n_ra_train": self.n_ra_train,
            "n_ild_train": self.n_ild_train,
            "ra_threshold_count": self.ra_threshold_count,
            "ild_threshold_count": self.ild_threshold_count,
            "RA_dict_size": self.ra_size,
            "ILD_dict_size": self.ild_size,
            "primary_feature": "RA_dict_hit_rate",
            "RA_dict_hit_rate_formula": "RA_dict_clone_count / RA_dict_size",
            "low_abundance_filter": "none",
            "dictionary_basis": "CDR3-AA sample prevalence",
        }


ALL_SCORE_COLUMNS = (
    "RA_dict_clone_count",
    "ILD_dict_clone_count",
    "RA_dict_hit_rate",
    "ILD_dict_hit_rate",
    "RA_dict_read_fraction_sum",
    "ILD_dict_read_fraction_sum",
    "RA_minus_ILD_count",
    "RA_minus_ILD_read_fraction",
    "RA_minus_ILD_hit_rate",
    "RA_hit_fraction_of_sample",
    "ILD_hit_fraction_of_sample",
    "RA_minus_ILD_sample_fraction",
    "sample_unique_clone_count",
    "RA_dict_size",
    "ILD_dict_size",
)


def _safe_divide(num: float, den: float) -> float:
    if not np.isfinite(den) or den <= 0:
        return float("nan")
    return float(num) / float(den)


def read_sample_repertoire(path: PathLike, sample_id: str) -> SampleRepertoire:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        frame = pd.read_csv(path, usecols=["cdr3_aa", "read_fraction"])
    except ValueError as exc:
        raise EnrichedDictionaryError(
            f"{path} must contain cdr3_aa and read_fraction"
        ) from exc
    frame["cdr3_aa"] = frame["cdr3_aa"].astype("string").str.strip().str.upper()
    frame["read_fraction"] = pd.to_numeric(frame["read_fraction"], errors="coerce")
    valid = (
        frame["cdr3_aa"].notna()
        & frame["cdr3_aa"].ne("")
        & frame["read_fraction"].notna()
        & np.isfinite(frame["read_fraction"].to_numpy(dtype=float))
        & frame["read_fraction"].ge(0)
    )
    frame = frame.loc[valid].copy()
    if frame.empty:
        return SampleRepertoire(str(sample_id), tuple(), {})
    agg = frame.groupby("cdr3_aa", sort=False, as_index=False)["read_fraction"].sum()
    clones = tuple(agg["cdr3_aa"].astype(str).tolist())
    rf = dict(zip(clones, agg["read_fraction"].astype(float).tolist()))
    return SampleRepertoire(str(sample_id), clones, rf)


def load_repertoires(
    metadata: pd.DataFrame,
    source: DynamicRepertoireSource,
    *,
    id_col: str = "libraryid",
) -> Dict[str, SampleRepertoire]:
    if id_col not in metadata.columns:
        raise EnrichedDictionaryError(f"metadata missing {id_col}")
    result: Dict[str, SampleRepertoire] = {}
    for sid in metadata[id_col].astype(str).str.strip().tolist():
        result[sid] = read_sample_repertoire(source.sample_path(sid), sid)
    return result


def _presence_counts(sample_ids: Iterable[str], repertoires: Mapping[str, SampleRepertoire]) -> Counter:
    counts: Counter[str] = Counter()
    for sid in sample_ids:
        counts.update(repertoires[str(sid)].clones)
    return counts


def fit_enriched_dictionary(
    metadata: pd.DataFrame,
    repertoires: Mapping[str, SampleRepertoire],
    *,
    source_id: str,
    threshold_pct: float = 20.0,
    delta_pct: float = 10.0,
    id_col: str = "libraryid",
    cohort_col: str = "cohort",
) -> EnrichedReference:
    if threshold_pct <= 0 or threshold_pct > 100:
        raise EnrichedDictionaryError("threshold_pct must be in (0,100]")
    if delta_pct < 0 or delta_pct > threshold_pct:
        raise EnrichedDictionaryError("delta_pct must be in [0, threshold_pct]")
    for col in (id_col, cohort_col):
        if col not in metadata.columns:
            raise EnrichedDictionaryError(f"metadata missing {col}")
    frame = metadata[[id_col, cohort_col]].copy()
    frame[id_col] = frame[id_col].astype(str).str.strip()
    frame[cohort_col] = (
        frame[cohort_col].astype(str).str.strip().str.upper()
        .replace({"RA-ILD": "ILD", "RA_ILD": "ILD", "RAILD": "ILD"})
    )
    if frame[id_col].duplicated().any():
        raise EnrichedDictionaryError("metadata sample IDs must be unique")
    if not frame[cohort_col].isin({"RA", "ILD"}).all():
        raise EnrichedDictionaryError("cohort labels must be RA/ILD")
    ra_ids = frame.loc[frame[cohort_col] == "RA", id_col].tolist()
    ild_ids = frame.loc[frame[cohort_col] == "ILD", id_col].tolist()
    if not ra_ids or not ild_ids:
        raise EnrichedDictionaryError("training metadata must contain both RA and ILD")
    missing = [sid for sid in frame[id_col] if sid not in repertoires]
    if missing:
        raise EnrichedDictionaryError(f"repertoires missing samples: {missing[:20]}")

    ra_counts = _presence_counts(ra_ids, repertoires)
    ild_counts = _presence_counts(ild_ids, repertoires)
    universe = sorted(set(ra_counts) | set(ild_counts))
    n_ra = len(ra_ids)
    n_ild = len(ild_ids)
    ra_threshold_count = int(math.ceil(n_ra * threshold_pct / 100.0))
    ild_threshold_count = int(math.ceil(n_ild * threshold_pct / 100.0))

    rows: List[dict] = []
    ra_dict: List[str] = []
    ild_dict: List[str] = []
    for clone in universe:
        ra_count = int(ra_counts.get(clone, 0))
        ild_count = int(ild_counts.get(clone, 0))
        ra_pct = ra_count / float(n_ra) * 100.0
        ild_pct = ild_count / float(n_ild) * 100.0
        delta = ra_pct - ild_pct
        is_ra = bool(ra_pct >= threshold_pct and delta >= delta_pct)
        is_ild = bool(ild_pct >= threshold_pct and delta <= -delta_pct)
        if is_ra and is_ild:
            raise EnrichedDictionaryError(f"dictionary overlap for clone {clone}")
        if is_ra:
            ra_dict.append(clone)
        if is_ild:
            ild_dict.append(clone)
        if is_ra or is_ild:
            rows.append({
                "cdr3_aa": clone,
                "dictionary": "RA" if is_ra else "RA-ILD",
                "RA_presence_count": ra_count,
                "ILD_presence_count": ild_count,
                "RA_prevalence_pct": ra_pct,
                "ILD_prevalence_pct": ild_pct,
                "RA_minus_ILD_prevalence_pct": delta,
            })

    members = pd.DataFrame(rows, columns=[
        "cdr3_aa", "dictionary", "RA_presence_count", "ILD_presence_count",
        "RA_prevalence_pct", "ILD_prevalence_pct", "RA_minus_ILD_prevalence_pct",
    ])
    return EnrichedReference(
        threshold_pct=float(threshold_pct),
        delta_pct=float(delta_pct),
        source_id=str(source_id),
        n_ra_train=n_ra,
        n_ild_train=n_ild,
        ra_threshold_count=ra_threshold_count,
        ild_threshold_count=ild_threshold_count,
        ra_clones=tuple(sorted(ra_dict)),
        ild_clones=tuple(sorted(ild_dict)),
        member_table=members,
    )


def score_sample(rep: SampleRepertoire, reference: EnrichedReference) -> Dict[str, float]:
    clone_set = set(rep.clones)
    ra_set = set(reference.ra_clones)
    ild_set = set(reference.ild_clones)
    ra_hits = clone_set & ra_set
    ild_hits = clone_set & ild_set
    ra_count = len(ra_hits)
    ild_count = len(ild_hits)
    ra_rf = float(sum(rep.read_fraction.get(clone, 0.0) for clone in ra_hits))
    ild_rf = float(sum(rep.read_fraction.get(clone, 0.0) for clone in ild_hits))
    ra_rate = _safe_divide(ra_count, reference.ra_size)
    ild_rate = _safe_divide(ild_count, reference.ild_size)
    sample_total = rep.unique_clone_count
    ra_sample_fraction = _safe_divide(ra_count, sample_total)
    ild_sample_fraction = _safe_divide(ild_count, sample_total)
    return {
        "RA_dict_clone_count": float(ra_count),
        "ILD_dict_clone_count": float(ild_count),
        "RA_dict_hit_rate": ra_rate,
        "ILD_dict_hit_rate": ild_rate,
        "RA_dict_read_fraction_sum": ra_rf,
        "ILD_dict_read_fraction_sum": ild_rf,
        "RA_minus_ILD_count": float(ra_count - ild_count),
        "RA_minus_ILD_read_fraction": ra_rf - ild_rf,
        "RA_minus_ILD_hit_rate": ra_rate - ild_rate if np.isfinite(ra_rate) and np.isfinite(ild_rate) else float("nan"),
        "RA_hit_fraction_of_sample": ra_sample_fraction,
        "ILD_hit_fraction_of_sample": ild_sample_fraction,
        "RA_minus_ILD_sample_fraction": ra_sample_fraction - ild_sample_fraction if np.isfinite(ra_sample_fraction) and np.isfinite(ild_sample_fraction) else float("nan"),
        "sample_unique_clone_count": float(sample_total),
        "RA_dict_size": float(reference.ra_size),
        "ILD_dict_size": float(reference.ild_size),
    }


def transform_repertoires(
    metadata: pd.DataFrame,
    repertoires: Mapping[str, SampleRepertoire],
    reference: EnrichedReference,
    *,
    id_col: str = "libraryid",
) -> pd.DataFrame:
    ids = metadata[id_col].astype(str).str.strip().tolist()
    rows = []
    for sid in ids:
        if sid not in repertoires:
            raise EnrichedDictionaryError(f"repertoire missing sample {sid}")
        row = {"sample_id": sid}
        row.update(score_sample(repertoires[sid], reference))
        rows.append(row)
    return pd.DataFrame(rows)


def write_reference(reference: EnrichedReference, output_dir: PathLike, *, overwrite: bool = False) -> Dict[str, Path]:
    out = Path(output_dir).expanduser().resolve()
    members_path = out / "05_enriched_dictionary_members.csv"
    json_path = out / "05_enriched_dictionary_reference.json"
    if not overwrite and (members_path.exists() or json_path.exists()):
        raise FileExistsError(f"reference output exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    reference.member_table.to_csv(members_path, index=False)
    payload = reference.as_dict()
    payload["members_file"] = members_path.name
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"members": members_path, "reference": json_path}


def load_reference(reference_dir: PathLike) -> EnrichedReference:
    root = Path(reference_dir).expanduser().resolve()
    json_path = root / "05_enriched_dictionary_reference.json"
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    members_path = root / raw["members_file"]
    members = pd.read_csv(members_path)
    ra = tuple(sorted(members.loc[members["dictionary"] == "RA", "cdr3_aa"].astype(str).tolist()))
    ild = tuple(sorted(members.loc[members["dictionary"] == "RA-ILD", "cdr3_aa"].astype(str).tolist()))
    return EnrichedReference(
        threshold_pct=float(raw["threshold_pct"]),
        delta_pct=float(raw["delta_pct"]),
        source_id=str(raw["source_id"]),
        n_ra_train=int(raw["n_ra_train"]),
        n_ild_train=int(raw["n_ild_train"]),
        ra_threshold_count=int(raw["ra_threshold_count"]),
        ild_threshold_count=int(raw["ild_threshold_count"]),
        ra_clones=ra,
        ild_clones=ild,
        member_table=members,
    )


def run_existing_grid_cv_via_source_shim(
    *,
    repository_root: PathLike,
    metadata: pd.DataFrame,
    source: DynamicRepertoireSource,
    output_dir: PathLike,
    threshold_pct: float,
    delta_pct: float,
    folds: int,
    repeats: int,
    split_candidates: int,
    seed: int,
    workers: int,
    overwrite: bool = False,
) -> Mapping[str, object]:
    """Run the repository's existing threshold-grid engine with one T/Delta pair.

    A temporary project shim exposes the selected all/Top-K tables using the old
    ``*_AA_clone_table.csv`` filename contract. The underlying old engine still
    owns optimized stratified folds, pooled-OOF AUC, and Linux workers=1..16.
    """
    if workers < 1 or workers > 16:
        raise EnrichedDictionaryError("workers must be in [1,16]")
    try:
        from .enrich_dictionary_grid import run_project_grid_analysis
    except ImportError as exc:
        raise RuntimeError("existing enrich_dictionary_grid.py is required for CV mode") from exc

    repo = Path(repository_root).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    # Keep the shim on the same filesystem as the repertoire source so that
    # hard links are normally available.  A symlink is deliberately NOT used:
    # the legacy CV loader resolves file-map paths and then validates the
    # resolved basename against ``*_AA_clone_table.csv``.  Resolving a symlink
    # to a dynamic Top-K target (for example ``*_top10000.csv``) therefore
    # breaks that legacy filename contract.
    shim_parent = source.aa_dir.parent
    with tempfile.TemporaryDirectory(
        prefix=".trb_enrich_source_shim_",
        dir=str(shim_parent),
    ) as tmp:
        shim = Path(tmp)
        aa_dir = shim / "TRB/result/01_AA_clone_table"
        aa_dir.mkdir(parents=True, exist_ok=True)
        meta = metadata.copy()
        if "material" not in meta.columns:
            meta["material"] = "unknown"
        meta.to_csv(shim / "TRB/metadata.csv", index=False)
        for sid in meta["libraryid"].astype(str).str.strip():
            src = source.sample_path(sid)
            if not src.is_file():
                raise FileNotFoundError(src)
            link = aa_dir / f"{sid}_AA_clone_table.csv"
            try:
                os.link(src, link)
            except OSError:
                # Defensive fallback for filesystems that disallow hard links.
                # A physical copy keeps the legacy basename stable after
                # Path.resolve(), unlike a symlink.
                shutil.copy2(src, link)

        manifest = run_project_grid_analysis(
            project_root=shim,
            scopes=("total",),
            folds=int(folds),
            repeats=int(repeats),
            candidates=int(split_candidates),
            seed=int(seed),
            threshold_values=[float(threshold_pct)],
            delta_values=[float(delta_pct)],
            workers=int(workers),
            allow_count_mismatch=True,
            save_fold_auc=True,
            make_plots=False,
            overwrite=bool(overwrite),
            result_dir=out,
            plot_dir=out / "plots",
        )
    return manifest
