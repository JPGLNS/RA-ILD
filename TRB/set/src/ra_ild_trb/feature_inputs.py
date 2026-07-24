#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe loading and one-to-one merging of scheme-specific feature tables.

The base feature matrix remains the authoritative sample order. Optional tables are
joined by ``sample_id`` (or another configured key) only after strict uniqueness,
coverage, column-name, finite-value, and type checks. This module does not perform
model fitting or any label-dependent feature construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class FeatureInputError(ValueError):
    """Raised when a configured feature table cannot be merged safely."""


@dataclass(frozen=True)
class AdditionalFeatureTable:
    path: Path
    key: str
    required_columns: Tuple[str, ...]
    numeric_columns: Tuple[str, ...]
    categorical_columns: Tuple[str, ...]

    @property
    def declared_columns(self) -> Tuple[str, ...]:
        ordered: List[str] = []
        for name in (*self.required_columns, *self.numeric_columns, *self.categorical_columns):
            if name not in ordered:
                ordered.append(name)
        return tuple(ordered)


def _string_list(value: Any, context: str) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FeatureInputError(f"{context} must be a list of strings")
    values: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise FeatureInputError(f"{context} must contain non-empty strings")
        values.append(item.strip())
    if len(values) != len(set(values)):
        raise FeatureInputError(f"{context} contains duplicate column names")
    return tuple(values)


def parse_additional_feature_tables(config, partition: str) -> Tuple[AdditionalFeatureTable, ...]:
    """Parse optional ``data.<partition>.additional_feature_tables`` entries."""

    section = config.raw.get("data", {}).get(partition, {})
    raw_tables = section.get("additional_feature_tables", [])
    if raw_tables is None:
        raw_tables = []
    if not isinstance(raw_tables, Sequence) or isinstance(raw_tables, (str, bytes)):
        raise FeatureInputError(
            f"data.{partition}.additional_feature_tables must be a list"
        )

    tables: List[AdditionalFeatureTable] = []
    for index, raw in enumerate(raw_tables):
        context = f"data.{partition}.additional_feature_tables[{index}]"
        if isinstance(raw, str):
            raw = {"path": raw}
        if not isinstance(raw, Mapping):
            raise FeatureInputError(f"{context} must be a mapping or path string")
        value = raw.get("path")
        if not isinstance(value, str) or not value.strip():
            raise FeatureInputError(f"{context}.path must be a non-empty string")
        key = raw.get("key", "sample_id")
        if not isinstance(key, str) or not key.strip():
            raise FeatureInputError(f"{context}.key must be a non-empty string")
        required = _string_list(raw.get("required_columns", []), f"{context}.required_columns")
        numeric = _string_list(raw.get("numeric_columns", []), f"{context}.numeric_columns")
        categorical = _string_list(
            raw.get("categorical_columns", []), f"{context}.categorical_columns"
        )
        overlap = set(numeric) & set(categorical)
        if overlap:
            raise FeatureInputError(
                f"{context} declares columns as both numeric and categorical: {sorted(overlap)}"
            )
        # Resolve the individual table path through the repository root without requiring
        # ExperimentConfig.path(), which addresses only scalar dotted keys.
        from .paths import resolve_project_path

        resolved = resolve_project_path(value.strip(), config.repository_root)
        tables.append(
            AdditionalFeatureTable(
                path=resolved,
                key=key.strip(),
                required_columns=required,
                numeric_columns=numeric,
                categorical_columns=categorical,
            )
        )
    return tuple(tables)


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    duplicated = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicated:
        raise FeatureInputError(f"{label} contains duplicated columns: {duplicated[:20]}")
    return frame


def _assert_unique_key(frame: pd.DataFrame, key: str, label: str) -> None:
    if key not in frame.columns:
        raise FeatureInputError(f"{label} is missing merge key {key!r}")
    if frame[key].isna().any():
        raise FeatureInputError(f"{label} contains missing values in merge key {key!r}")
    frame[key] = frame[key].astype(str)
    duplicates = frame.loc[frame[key].duplicated(keep=False), key].astype(str).unique().tolist()
    if duplicates:
        raise FeatureInputError(
            f"{label} contains duplicate {key} values: {duplicates[:20]}"
        )


def _validate_declared_types(
    frame: pd.DataFrame,
    table: AdditionalFeatureTable,
    label: str,
) -> None:
    for column in table.numeric_columns:
        if column not in frame.columns:
            raise FeatureInputError(f"{label} is missing declared numeric column {column!r}")
        converted = pd.to_numeric(frame[column], errors="coerce")
        bad = converted.isna() & frame[column].notna()
        if bad.any():
            examples = frame.loc[bad, column].astype(str).head(10).tolist()
            raise FeatureInputError(
                f"{label} column {column!r} contains non-numeric values: {examples}"
            )
        values = converted.to_numpy(float)
        if not np.isfinite(values).all():
            raise FeatureInputError(
                f"{label} column {column!r} contains NA, NaN or infinite values"
            )
        frame[column] = converted.astype(float)

    for column in table.categorical_columns:
        if column not in frame.columns:
            raise FeatureInputError(
                f"{label} is missing declared categorical column {column!r}"
            )
        if frame[column].isna().any():
            raise FeatureInputError(
                f"{label} categorical column {column!r} contains missing values"
            )
        frame[column] = frame[column].astype(str)


def load_partition_feature_matrix(
    config,
    partition: str = "train",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a base matrix and merge all configured additional feature tables.

    Returns ``(merged_matrix, audit_frame)``. The merged matrix preserves the base
    matrix row order exactly. Every additional table must contain exactly the same
    sample keys as the base matrix; silent partial joins are forbidden.
    """

    base_path = config.path(
        f"data.{partition}.base_matrix" if partition == "train" else f"data.{partition}.final_matrix",
        must_exist=True,
        expect="file",
    )
    base = _read_csv(base_path, f"{partition} base feature matrix")
    key = "sample_id"
    _assert_unique_key(base, key, f"{partition} base feature matrix")
    base_ids = base[key].astype(str).tolist()
    base_set = set(base_ids)

    audit_rows: List[Dict[str, Any]] = [
        {
            "partition": partition,
            "source_index": 0,
            "source_type": "base_matrix",
            "path": str(base_path),
            "merge_key": key,
            "rows": int(len(base)),
            "columns_before": 0,
            "columns_added": int(len(base.columns)),
            "columns_after": int(len(base.columns)),
            "added_column_names": "|".join(map(str, base.columns)),
            "missing_base_samples": 0,
            "extra_samples": 0,
        }
    ]

    merged = base.copy()
    for index, table in enumerate(parse_additional_feature_tables(config, partition), start=1):
        label = f"{partition} additional feature table {index}"
        frame = _read_csv(table.path, label)
        _assert_unique_key(frame, table.key, label)
        if table.key != key:
            if key in frame.columns:
                raise FeatureInputError(
                    f"{label} contains both configured key {table.key!r} and reserved key {key!r}"
                )
            frame = frame.rename(columns={table.key: key})

        required = set(table.declared_columns)
        missing_required = sorted(required - set(frame.columns))
        if missing_required:
            raise FeatureInputError(
                f"{label} is missing required/declared columns: {missing_required}"
            )
        _validate_declared_types(frame, table, label)

        frame_ids = frame[key].astype(str).tolist()
        frame_set = set(frame_ids)
        missing_samples = sorted(base_set - frame_set)
        extra_samples = sorted(frame_set - base_set)
        if missing_samples or extra_samples:
            details = []
            if missing_samples:
                details.append(f"missing base samples={missing_samples[:20]}")
            if extra_samples:
                details.append(f"extra samples={extra_samples[:20]}")
            raise FeatureInputError(f"{label} sample coverage mismatch: " + "; ".join(details))

        added_columns = [column for column in frame.columns if column != key]
        collisions = sorted(set(added_columns) & set(merged.columns))
        if collisions:
            raise FeatureInputError(
                f"{label} columns collide with existing feature columns: {collisions[:20]}"
            )
        if not added_columns:
            raise FeatureInputError(f"{label} does not provide any feature columns")
        if frame[added_columns].isna().any().any():
            missing_columns = frame[added_columns].columns[
                frame[added_columns].isna().any()
            ].astype(str).tolist()
            raise FeatureInputError(
                f"{label} contains missing values in feature columns: {missing_columns[:20]}"
            )

        before = len(merged.columns)
        ordered = frame.set_index(key).loc[base_ids].reset_index()
        merged = merged.merge(ordered, on=key, how="left", validate="one_to_one", sort=False)
        if merged[key].astype(str).tolist() != base_ids:
            raise FeatureInputError(f"{label} changed the base sample order")
        audit_rows.append(
            {
                "partition": partition,
                "source_index": index,
                "source_type": "additional_feature_table",
                "path": str(table.path),
                "merge_key": table.key,
                "rows": int(len(frame)),
                "columns_before": int(before),
                "columns_added": int(len(added_columns)),
                "columns_after": int(len(merged.columns)),
                "added_column_names": "|".join(map(str, added_columns)),
                "missing_base_samples": 0,
                "extra_samples": 0,
            }
        )

    if merged.columns.duplicated().any():
        duplicates = merged.columns[merged.columns.duplicated()].astype(str).tolist()
        raise FeatureInputError(f"Merged feature matrix contains duplicate columns: {duplicates}")
    return merged, pd.DataFrame(audit_rows)
