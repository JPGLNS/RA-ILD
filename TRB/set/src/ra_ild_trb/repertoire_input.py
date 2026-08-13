#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve and audit repertoire inputs for source-aware TRB feature generation.

Phase 2 keeps biological feature calculations in the legacy Step02 implementation.
This module is responsible only for the representation-dependent input contract:

* resolve ``all`` versus ``top10000`` from the feature registry/plan;
* map sample IDs to deterministic repertoire file names;
* load the correct source summary (Step01 full or Step01b Top-K);
* for Top-K, retain Step01 as the parent technical/provenance denominator;
* adapt source-specific summary values to the legacy Step02 calculation kernel;
* emit an auditable source/QC record without mixing full and Top-K equality checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .feature_framework import FeatureFrameworkError, FeatureRegistry, RepertoireSource
from .paths import resolve_project_path

PathLike = Union[str, Path]


class RepertoireInputError(ValueError):
    """Raised when a repertoire source violates its source-specific contract."""


FULL_SUMMARY_REQUIRED = {
    "sample_id",
    "input_nt_rows",
    "valid_nt_rows",
    "valid_nt_row_ratio",
    "total_reads_all_nt",
    "total_reads_valid_nt",
    "unique_valid_nt_clones",
    "aa_clone_number",
}

TOPK_SUMMARY_REQUIRED = {
    "sample_id",
    "top_k",
    "full_aa_clone_number",
    "retained_aa_clone_number",
    "retained_read_count",
    "retained_read_mass",
    "full_read_count",
    "full_frequency_sum",
    "retained_frequency_sum",
    "retained_frequency_mass",
    "frequency_norm_sum_topk",
    "read_fraction_sum_topk",
    "qc_pass",
    "qc_warning",
}

TOPK_SCAN_COLUMNS = {
    "sample_id",
    "topk_rank",
    "nt_clone_number",
    "read_count",
    "frequency",
    "read_fraction_full",
    "read_fraction",
    "frequency_norm_full",
    "frequency_norm",
}


@dataclass(frozen=True)
class ResolvedRepertoireInput:
    source: RepertoireSource
    repository_root: Path
    aa_dir: Path
    summary_path: Path
    parent_source: Optional[RepertoireSource]
    parent_summary_path: Optional[Path]

    def sample_path(self, sample_id: str) -> Path:
        return self.aa_dir / self.source.sample_filename(sample_id)

    def as_dict(self) -> Dict[str, object]:
        return {
            "source_id": self.source.id,
            "representation": self.source.representation,
            "source_layer": self.source.source_layer,
            "aa_dir": str(self.aa_dir),
            "summary_kind": self.source.summary_kind,
            "summary_path": str(self.summary_path),
            "parent_source": self.parent_source.id if self.parent_source else None,
            "parent_summary_path": (
                str(self.parent_summary_path) if self.parent_summary_path else None
            ),
            "sample_filename_template": self.source.sample_filename_template,
            "weight_column": self.source.weight_column,
            "top_k": self.source.top_k,
        }


def _coerce_numeric(frame: pd.DataFrame, columns: Iterable[str], label: str) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise RepertoireInputError(f"{label}.{column} contains invalid numeric values")
        out[column] = values
    return out


def _bool_value(value: object, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise RepertoireInputError(f"{label} must be boolean-like, observed {value!r}")


def _assert_unique_sample_ids(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if "sample_id" not in frame.columns:
        raise RepertoireInputError(f"{label} is missing sample_id")
    out = frame.copy()
    if out["sample_id"].isna().any():
        raise RepertoireInputError(f"{label}.sample_id contains missing values")
    out["sample_id"] = out["sample_id"].astype(str).str.strip()
    if out["sample_id"].eq("").any():
        raise RepertoireInputError(f"{label}.sample_id contains empty values")
    if out["sample_id"].duplicated().any():
        values = out.loc[out["sample_id"].duplicated(keep=False), "sample_id"].unique()
        raise RepertoireInputError(f"{label} contains duplicated sample_id values: {values[:10].tolist()}")
    return out


def load_summary(path: PathLike, summary_kind: str) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"repertoire summary not found: {path}")
    frame = pd.read_csv(path)
    frame = frame.drop(
        columns=[c for c in frame.columns if str(c).startswith("Unnamed:")],
        errors="ignore",
    )
    frame = _assert_unique_sample_ids(frame, f"{summary_kind} summary")
    if summary_kind == "step01_full":
        required = FULL_SUMMARY_REQUIRED
        numeric = required - {"sample_id"}
    elif summary_kind == "step01b_topk":
        required = TOPK_SUMMARY_REQUIRED
        numeric = required - {"sample_id", "qc_pass", "qc_warning"}
    else:
        raise RepertoireInputError(f"unsupported summary_kind: {summary_kind!r}")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RepertoireInputError(f"{summary_kind} summary missing columns: {missing}")
    frame = _coerce_numeric(frame, numeric, f"{summary_kind} summary")
    return frame.set_index("sample_id", drop=False)


def resolve_repertoire_input(
    registry: FeatureRegistry,
    source_id: str,
    repository_root: PathLike,
    *,
    aa_dir_override: Optional[PathLike] = None,
    summary_override: Optional[PathLike] = None,
    parent_summary_override: Optional[PathLike] = None,
    must_exist: bool = True,
) -> ResolvedRepertoireInput:
    if source_id not in registry.repertoire_sources:
        raise FeatureFrameworkError(
            f"unknown repertoire source {source_id!r}; available={sorted(registry.repertoire_sources)}"
        )
    source = registry.repertoire_sources[source_id]
    if source.status != "available":
        raise FeatureFrameworkError(
            f"repertoire source {source_id!r} is not available (status={source.status})"
        )
    root = Path(repository_root).expanduser().resolve()
    aa_dir = resolve_project_path(
        aa_dir_override if aa_dir_override is not None else source.default_path,
        root,
        must_exist=must_exist,
        expect="dir",
    )
    summary_path = resolve_project_path(
        summary_override if summary_override is not None else source.summary_path,
        root,
        must_exist=must_exist,
        expect="file",
    )

    parent_source: Optional[RepertoireSource] = None
    parent_summary_path: Optional[Path] = None
    if source.parent_source is not None:
        parent_source = registry.repertoire_sources[source.parent_source]
        if parent_source.summary_kind != "step01_full":
            raise RepertoireInputError(
                f"parent source {parent_source.id!r} must provide step01_full summary"
            )
        parent_summary_path = resolve_project_path(
            parent_summary_override
            if parent_summary_override is not None
            else parent_source.summary_path,
            root,
            must_exist=must_exist,
            expect="file",
        )
    elif parent_summary_override is not None:
        raise RepertoireInputError(
            f"source {source.id!r} has no parent_source; parent summary override is invalid"
        )

    return ResolvedRepertoireInput(
        source=source,
        repository_root=root,
        aa_dir=aa_dir,
        summary_path=summary_path,
        parent_source=parent_source,
        parent_summary_path=parent_summary_path,
    )


def assert_summary_coverage(
    sample_ids: Sequence[str],
    summary: pd.DataFrame,
    label: str,
) -> None:
    requested = [str(x).strip() for x in sample_ids]
    missing = [sample_id for sample_id in requested if sample_id not in summary.index]
    if missing:
        raise RepertoireInputError(f"{label} missing metadata samples: {missing[:20]}")


def _close(a: float, b: float, tolerance: float) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=tolerance, atol=tolerance))


def _validate_full_row(
    sample_id: str,
    row: pd.Series,
) -> None:
    if int(row["input_nt_rows"]) < int(row["valid_nt_rows"]):
        raise RepertoireInputError(f"{sample_id}: full summary valid_nt_rows > input_nt_rows")
    if int(row["total_reads_all_nt"]) < int(row["total_reads_valid_nt"]):
        raise RepertoireInputError(f"{sample_id}: full summary valid reads > all reads")
    if int(row["aa_clone_number"]) < 1:
        raise RepertoireInputError(f"{sample_id}: full summary aa_clone_number < 1")
    if int(row["unique_valid_nt_clones"]) < int(row["aa_clone_number"]):
        raise RepertoireInputError(
            f"{sample_id}: full summary unique_valid_nt_clones < aa_clone_number"
        )


def adapt_summary_for_legacy_step02(
    resolved: ResolvedRepertoireInput,
    sample_id: str,
    source_summary_row: pd.Series,
    *,
    parent_summary_row: Optional[pd.Series] = None,
    normalization_tolerance: float = 1e-8,
) -> Tuple[pd.Series, Dict[str, object]]:
    """Return a legacy-Step02 summary row plus source audit metadata.

    For ``all``, the original Step01 summary values are preserved exactly.
    For ``topk``, current-repertoire counts are derived from the Top-K table while
    full Step01 counts remain denominators/provenance. This prevents the old
    full-repertoire equality checks from being incorrectly applied to Top-K.
    """

    source = resolved.source
    sample_id = str(sample_id).strip()
    if str(source_summary_row["sample_id"]).strip() != sample_id:
        raise RepertoireInputError(f"{sample_id}: source summary sample_id mismatch")

    if source.representation == "full":
        _validate_full_row(sample_id, source_summary_row)
        legacy = source_summary_row.copy()
        audit = {
            "sample_id": sample_id,
            "repertoire_source": source.id,
            "representation": source.representation,
            "source_layer": source.source_layer,
            "source_summary_kind": source.summary_kind,
            "source_summary_file": str(resolved.summary_path),
            "parent_summary_file": "",
            "analysis_aa_clone_number": int(source_summary_row["aa_clone_number"]),
            "full_aa_clone_number": int(source_summary_row["aa_clone_number"]),
            "analysis_read_count": int(source_summary_row["total_reads_valid_nt"]),
            "full_valid_read_count": int(source_summary_row["total_reads_valid_nt"]),
            "retained_read_mass": 1.0,
            "analysis_nt_clone_number": int(source_summary_row["unique_valid_nt_clones"]),
            "full_valid_nt_clone_number": int(source_summary_row["unique_valid_nt_clones"]),
            "retained_nt_clone_fraction": 1.0,
            "frequency_norm_sum": 1.0,
            "read_fraction_sum": 1.0,
            "source_qc_pass": True,
            "source_qc_warning": "",
        }
        return legacy, audit

    if source.representation != "topk":
        raise RepertoireInputError(
            f"source {source.id!r} representation {source.representation!r} is unsupported in Phase 2"
        )
    if parent_summary_row is None:
        raise RepertoireInputError(f"{sample_id}: Top-K source requires parent Step01 summary")
    _validate_full_row(sample_id, parent_summary_row)
    if str(parent_summary_row["sample_id"]).strip() != sample_id:
        raise RepertoireInputError(f"{sample_id}: parent summary sample_id mismatch")
    if not _bool_value(source_summary_row["qc_pass"], f"{sample_id}.qc_pass"):
        raise RepertoireInputError(
            f"{sample_id}: Step01b source summary qc_pass is false: "
            f"{source_summary_row.get('qc_warning', '')}"
        )
    if source.top_k is None:
        raise RepertoireInputError(f"{sample_id}: Top-K source lacks top_k contract")
    if int(source_summary_row["top_k"]) != int(source.top_k):
        raise RepertoireInputError(
            f"{sample_id}: source summary top_k={int(source_summary_row['top_k'])} "
            f"does not match registry top_k={source.top_k}"
        )
    if int(source_summary_row["retained_aa_clone_number"]) != int(source.top_k):
        raise RepertoireInputError(
            f"{sample_id}: retained_aa_clone_number is not exact K={source.top_k}"
        )
    if int(source_summary_row["full_aa_clone_number"]) != int(parent_summary_row["aa_clone_number"]):
        raise RepertoireInputError(
            f"{sample_id}: Step01b full_aa_clone_number does not match parent Step01 aa_clone_number"
        )
    if int(source_summary_row["full_read_count"]) != int(parent_summary_row["total_reads_valid_nt"]):
        raise RepertoireInputError(
            f"{sample_id}: Step01b full_read_count does not match parent Step01 total_reads_valid_nt"
        )
    if not _close(source_summary_row["frequency_norm_sum_topk"], 1.0, normalization_tolerance):
        raise RepertoireInputError(f"{sample_id}: Step01b frequency_norm_sum_topk != 1")
    if not _close(source_summary_row["read_fraction_sum_topk"], 1.0, normalization_tolerance):
        raise RepertoireInputError(f"{sample_id}: Step01b read_fraction_sum_topk != 1")

    path = resolved.sample_path(sample_id)
    if not path.is_file():
        raise FileNotFoundError(f"Top-K sample table not found: {path}")
    header = pd.read_csv(path, nrows=0)
    missing_scan = sorted(TOPK_SCAN_COLUMNS - set(header.columns))
    if missing_scan:
        raise RepertoireInputError(f"{sample_id}: Top-K table missing columns: {missing_scan}")
    frame = pd.read_csv(path, usecols=sorted(TOPK_SCAN_COLUMNS))
    if len(frame) != int(source.top_k):
        raise RepertoireInputError(
            f"{sample_id}: Top-K table rows={len(frame)} expected={source.top_k}"
        )
    ids = frame["sample_id"].astype(str).str.strip().unique().tolist()
    if ids != [sample_id]:
        raise RepertoireInputError(f"{sample_id}: Top-K table sample_id mismatch: {ids[:10]}")
    frame = _coerce_numeric(
        frame,
        TOPK_SCAN_COLUMNS - {"sample_id"},
        f"{sample_id} Top-K table",
    )
    expected_ranks = np.arange(1, int(source.top_k) + 1, dtype=int)
    ranks = frame["topk_rank"].to_numpy(dtype=int)
    if not np.array_equal(ranks, expected_ranks):
        raise RepertoireInputError(f"{sample_id}: topk_rank is not exact 1..K")

    current_read_count = int(round(float(frame["read_count"].sum())))
    current_nt_count = int(round(float(frame["nt_clone_number"].sum())))
    current_frequency_sum = float(frame["frequency"].sum())
    frequency_norm_sum = float(frame["frequency_norm"].sum())
    read_fraction_sum = float(frame["read_fraction"].sum())
    frequency_norm_full_sum = float(frame["frequency_norm_full"].sum())
    read_fraction_full_sum = float(frame["read_fraction_full"].sum())

    if current_read_count != int(source_summary_row["retained_read_count"]):
        raise RepertoireInputError(
            f"{sample_id}: Top-K table read_count sum does not match Step01b summary"
        )
    if not _close(current_frequency_sum, source_summary_row["retained_frequency_sum"], normalization_tolerance):
        raise RepertoireInputError(
            f"{sample_id}: Top-K table frequency sum does not match Step01b summary"
        )
    if not _close(frequency_norm_sum, 1.0, normalization_tolerance):
        raise RepertoireInputError(f"{sample_id}: Top-K table frequency_norm sum != 1")
    if not _close(read_fraction_sum, 1.0, normalization_tolerance):
        raise RepertoireInputError(f"{sample_id}: Top-K table read_fraction sum != 1")
    if not _close(
        frequency_norm_full_sum,
        source_summary_row["retained_frequency_mass"],
        normalization_tolerance,
    ):
        raise RepertoireInputError(
            f"{sample_id}: frequency_norm_full retained mass does not match Step01b summary"
        )
    if not _close(
        read_fraction_full_sum,
        source_summary_row["retained_read_mass"],
        normalization_tolerance,
    ):
        raise RepertoireInputError(
            f"{sample_id}: read_fraction_full retained mass does not match Step01b summary"
        )

    parent_input_nt_rows = int(parent_summary_row["input_nt_rows"])
    parent_valid_nt = int(parent_summary_row["unique_valid_nt_clones"])
    if current_nt_count > parent_valid_nt:
        raise RepertoireInputError(
            f"{sample_id}: retained NT clone count exceeds parent valid NT clone count"
        )
    valid_nt_row_ratio = (
        current_nt_count / parent_input_nt_rows if parent_input_nt_rows > 0 else 0.0
    )

    legacy = pd.Series(
        {
            "sample_id": sample_id,
            "input_nt_rows": parent_input_nt_rows,
            "valid_nt_rows": current_nt_count,
            "valid_nt_row_ratio": valid_nt_row_ratio,
            "total_reads_all_nt": int(parent_summary_row["total_reads_all_nt"]),
            "total_reads_valid_nt": current_read_count,
            "unique_valid_nt_clones": current_nt_count,
            "aa_clone_number": int(source.top_k),
        }
    )
    audit = {
        "sample_id": sample_id,
        "repertoire_source": source.id,
        "representation": source.representation,
        "source_layer": source.source_layer,
        "source_summary_kind": source.summary_kind,
        "source_summary_file": str(resolved.summary_path),
        "parent_summary_file": str(resolved.parent_summary_path or ""),
        "analysis_aa_clone_number": int(source.top_k),
        "full_aa_clone_number": int(parent_summary_row["aa_clone_number"]),
        "analysis_read_count": current_read_count,
        "full_valid_read_count": int(parent_summary_row["total_reads_valid_nt"]),
        "retained_read_mass": float(source_summary_row["retained_read_mass"]),
        "analysis_nt_clone_number": current_nt_count,
        "full_valid_nt_clone_number": parent_valid_nt,
        "retained_nt_clone_fraction": (
            current_nt_count / parent_valid_nt if parent_valid_nt > 0 else 0.0
        ),
        "frequency_norm_sum": frequency_norm_sum,
        "read_fraction_sum": read_fraction_sum,
        "source_qc_pass": True,
        "source_qc_warning": str(source_summary_row.get("qc_warning", "")),
    }
    return legacy, audit


def build_adapted_summary_table(
    resolved: ResolvedRepertoireInput,
    sample_ids: Sequence[str],
    *,
    normalization_tolerance: float = 1e-8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    source_summary = load_summary(resolved.summary_path, resolved.source.summary_kind)
    assert_summary_coverage(sample_ids, source_summary, f"{resolved.source.id} summary")

    parent_summary: Optional[pd.DataFrame] = None
    if resolved.parent_summary_path is not None:
        assert resolved.parent_source is not None
        parent_summary = load_summary(
            resolved.parent_summary_path, resolved.parent_source.summary_kind
        )
        assert_summary_coverage(sample_ids, parent_summary, "parent summary")

    legacy_rows = []
    audits = []
    for sample_id in sample_ids:
        parent_row = parent_summary.loc[sample_id] if parent_summary is not None else None
        legacy, audit = adapt_summary_for_legacy_step02(
            resolved,
            sample_id,
            source_summary.loc[sample_id],
            parent_summary_row=parent_row,
            normalization_tolerance=normalization_tolerance,
        )
        legacy_rows.append(dict(legacy))
        audit["input_file"] = str(resolved.sample_path(sample_id))
        audits.append(audit)

    legacy_frame = pd.DataFrame(legacy_rows)
    legacy_frame["sample_id"] = legacy_frame["sample_id"].astype(str)
    legacy_frame = legacy_frame.set_index("sample_id", drop=False)
    audit_frame = pd.DataFrame(audits)
    return legacy_frame, audit_frame
