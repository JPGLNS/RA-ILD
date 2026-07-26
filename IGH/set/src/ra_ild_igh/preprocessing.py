#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leakage-controlled design-matrix preprocessing for IGH V2.

This module reproduces the preprocessing behavior of the V1
``run_05_single_outer_trial_loo.py`` worker while separating fit and transform
operations explicitly:

* numeric columns are parsed as floating-point values;
* categorical levels are learned from the current training partition only;
* the lexicographically first training category is the reference level;
* one dummy is created for every remaining training category;
* means, population SDs (``ddof=0``), and zero-variance decisions are learned
  from the current training partition only;
* validation/test rows are transformed with those frozen training parameters.

The V1 behavior for a category not observed during fitting is preserved: all
learned dummy indicators for that variable are zero, so the row receives the
same raw dummy code as the training reference category. Such occurrences are
reported by :meth:`FittedPreprocessor.unseen_category_counts` and can optionally
be rejected with ``strict_unseen_categories=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class PreprocessingError(ValueError):
    """Raised when a design matrix cannot be fitted or transformed safely."""


def _unique_columns(columns: Sequence[str], label: str) -> Tuple[str, ...]:
    out: List[str] = []
    for value in columns:
        if not isinstance(value, str) or not value.strip():
            raise PreprocessingError(f"{label} must contain non-empty strings.")
        out.append(value.strip())
    if len(out) != len(set(out)):
        raise PreprocessingError(f"{label} contains duplicate columns.")
    return tuple(out)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise PreprocessingError(f"{label} is missing columns: {missing}")


def _require_no_missing(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    columns = list(columns)
    if not columns:
        return
    counts = frame[columns].isna().sum()
    bad = counts[counts > 0]
    if not bad.empty:
        raise PreprocessingError(
            f"{label} contains missing predictor values: {bad.to_dict()}"
        )


def _numeric_values(frame: pd.DataFrame, column: str, label: str) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    except Exception as exc:  # pandas raises several possible numeric errors
        raise PreprocessingError(
            f"{label}.{column} cannot be converted to numeric values: {exc}"
        ) from exc
    if not np.isfinite(values).all():
        raise PreprocessingError(f"{label}.{column} contains non-finite values.")
    return values


@dataclass(frozen=True)
class CategoryEncoding:
    """Training-derived dummy-coding metadata for one categorical variable."""

    column: str
    categories: Tuple[str, ...]
    reference: str
    dummy_categories: Tuple[str, ...]
    dummy_feature_names: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "column": self.column,
            "categories": list(self.categories),
            "reference": self.reference,
            "dummy_categories": list(self.dummy_categories),
            "dummy_feature_names": list(self.dummy_feature_names),
        }


@dataclass(frozen=True)
class DesignTransform:
    """A transformed design matrix plus category-audit information."""

    matrix: np.ndarray
    feature_names: Tuple[str, ...]
    unseen_category_counts: Mapping[str, int]


@dataclass(frozen=True)
class PreparedDesign:
    """Training and validation matrices built from one fitted preprocessor."""

    X_train: np.ndarray
    X_valid: np.ndarray
    feature_names: Tuple[str, ...]
    audit: pd.DataFrame
    preprocessor: "FittedPreprocessor"
    valid_unseen_category_counts: Mapping[str, int]


@dataclass(frozen=True)
class FittedPreprocessor:
    """Frozen, training-derived preprocessing parameters."""

    numeric_columns: Tuple[str, ...]
    categorical_columns: Tuple[str, ...]
    category_encodings: Mapping[str, CategoryEncoding]
    raw_feature_names: Tuple[str, ...]
    source_types: Tuple[str, ...]
    training_means: np.ndarray
    training_sds: np.ndarray
    keep_mask: np.ndarray
    zero_sd_tolerance: float
    ddof: int = 0

    def __post_init__(self) -> None:
        n_features = len(self.raw_feature_names)
        arrays = (self.training_means, self.training_sds, self.keep_mask)
        if any(len(array) != n_features for array in arrays):
            raise PreprocessingError("Fitted preprocessing arrays have inconsistent lengths.")
        if len(self.source_types) != n_features:
            raise PreprocessingError("source_types length does not match feature names.")
        if not np.asarray(self.keep_mask, dtype=bool).any():
            raise PreprocessingError("All predictors were removed during preprocessing.")

    @property
    def kept_feature_names(self) -> Tuple[str, ...]:
        return tuple(
            name
            for name, keep in zip(self.raw_feature_names, self.keep_mask)
            if bool(keep)
        )

    @property
    def dropped_feature_names(self) -> Tuple[str, ...]:
        return tuple(
            name
            for name, keep in zip(self.raw_feature_names, self.keep_mask)
            if not bool(keep)
        )

    @property
    def n_raw_features(self) -> int:
        return len(self.raw_feature_names)

    @property
    def n_kept_features(self) -> int:
        return int(np.asarray(self.keep_mask, dtype=bool).sum())

    def unseen_category_counts(self, frame: pd.DataFrame) -> Mapping[str, int]:
        """Count non-missing transform categories absent from the fit partition."""
        _require_columns(frame, self.categorical_columns, "transform data")
        _require_no_missing(frame, self.categorical_columns, "transform data")
        counts: Dict[str, int] = {}
        for column in self.categorical_columns:
            values = frame[column].astype(str)
            known = set(self.category_encodings[column].categories)
            counts[column] = int((~values.isin(known)).sum())
        return MappingProxyType(counts)

    def transform_raw(
        self,
        frame: pd.DataFrame,
        *,
        strict_unseen_categories: bool = False,
    ) -> DesignTransform:
        """Build the unstandardized raw matrix using frozen fit-time encodings."""
        required = (*self.numeric_columns, *self.categorical_columns)
        _require_columns(frame, required, "transform data")
        _require_no_missing(frame, required, "transform data")

        parts: List[np.ndarray] = []
        for column in self.numeric_columns:
            parts.append(_numeric_values(frame, column, "transform data")[:, None])

        unseen = dict(self.unseen_category_counts(frame))
        if strict_unseen_categories and any(unseen.values()):
            raise PreprocessingError(
                "Unseen categorical levels were detected: "
                + ", ".join(f"{key}={value}" for key, value in unseen.items() if value)
            )

        for column in self.categorical_columns:
            values = frame[column].astype(str)
            encoding = self.category_encodings[column]
            for category in encoding.dummy_categories:
                parts.append((values == category).to_numpy(dtype=float)[:, None])

        if not parts:
            raise PreprocessingError("No predictor columns were assembled.")
        matrix = np.hstack(parts)
        if matrix.shape[1] != self.n_raw_features:
            raise PreprocessingError(
                "Transform matrix width does not match fitted preprocessing metadata."
            )
        if not np.isfinite(matrix).all():
            raise PreprocessingError("Non-finite raw values were produced.")
        return DesignTransform(
            matrix=matrix,
            feature_names=self.raw_feature_names,
            unseen_category_counts=MappingProxyType(unseen),
        )

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        strict_unseen_categories: bool = False,
    ) -> DesignTransform:
        """Apply zero-variance filtering and training-derived z-scoring."""
        raw = self.transform_raw(
            frame,
            strict_unseen_categories=strict_unseen_categories,
        )
        keep = np.asarray(self.keep_mask, dtype=bool)
        matrix = (
            raw.matrix[:, keep] - self.training_means[keep]
        ) / self.training_sds[keep]
        if not np.isfinite(matrix).all():
            raise PreprocessingError("Non-finite standardized values were produced.")
        return DesignTransform(
            matrix=matrix,
            feature_names=self.kept_feature_names,
            unseen_category_counts=raw.unseen_category_counts,
        )

    def audit_frame(self) -> pd.DataFrame:
        """Return the five V1-compatible preprocessing audit columns."""
        return pd.DataFrame(
            {
                "feature_name": list(self.raw_feature_names),
                "source_type": list(self.source_types),
                "training_mean": np.asarray(self.training_means, dtype=float),
                "training_sd": np.asarray(self.training_sds, dtype=float),
                "kept_after_zero_variance_filter": np.asarray(
                    self.keep_mask, dtype=bool
                ),
            }
        )

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-serializable preprocessing metadata."""
        return {
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "category_encodings": {
                key: value.as_dict() for key, value in self.category_encodings.items()
            },
            "raw_feature_names": list(self.raw_feature_names),
            "source_types": list(self.source_types),
            "training_means": self.training_means.astype(float).tolist(),
            "training_sds": self.training_sds.astype(float).tolist(),
            "keep_mask": self.keep_mask.astype(bool).tolist(),
            "kept_feature_names": list(self.kept_feature_names),
            "dropped_feature_names": list(self.dropped_feature_names),
            "zero_sd_tolerance": float(self.zero_sd_tolerance),
            "ddof": int(self.ddof),
        }


def fit_preprocessor(
    train_df: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    zero_sd_tolerance: float = 1e-12,
) -> FittedPreprocessor:
    """Fit V1-compatible encoding, filtering, and scaling on training rows only."""
    if zero_sd_tolerance < 0:
        raise PreprocessingError("zero_sd_tolerance must be >= 0.")

    numeric = _unique_columns(numeric_columns, "numeric_columns")
    categorical = _unique_columns(categorical_columns, "categorical_columns")
    overlap = sorted(set(numeric) & set(categorical))
    if overlap:
        raise PreprocessingError(
            f"Columns cannot be both numeric and categorical: {overlap}"
        )

    required = (*numeric, *categorical)
    _require_columns(train_df, required, "training data")
    _require_no_missing(train_df, required, "training data")

    parts: List[np.ndarray] = []
    feature_names: List[str] = []
    source_types: List[str] = []
    encodings: Dict[str, CategoryEncoding] = {}

    for column in numeric:
        parts.append(_numeric_values(train_df, column, "training data")[:, None])
        feature_names.append(column)
        source_types.append("numeric")

    for column in categorical:
        values = train_df[column].astype(str)
        categories = tuple(sorted(values.unique().tolist()))
        if not categories:
            raise PreprocessingError(f"No training category found for {column}.")
        reference = categories[0]
        dummy_categories = categories[1:]
        dummy_names = tuple(
            f"{column}__{category}_vs_{reference}" for category in dummy_categories
        )
        encodings[column] = CategoryEncoding(
            column=column,
            categories=categories,
            reference=reference,
            dummy_categories=dummy_categories,
            dummy_feature_names=dummy_names,
        )
        for category, feature_name in zip(dummy_categories, dummy_names):
            parts.append((values == category).to_numpy(dtype=float)[:, None])
            feature_names.append(feature_name)
            source_types.append("categorical_dummy")

    if not parts:
        raise PreprocessingError("No predictor columns were assembled.")

    raw = np.hstack(parts)
    if not np.isfinite(raw).all():
        raise PreprocessingError("Non-finite training values were assembled.")

    means = raw.mean(axis=0)
    sds = raw.std(axis=0, ddof=0)
    keep = np.isfinite(sds) & (sds > zero_sd_tolerance)
    if not keep.any():
        raise PreprocessingError(
            "All predictors had zero or invalid standard deviation."
        )

    return FittedPreprocessor(
        numeric_columns=numeric,
        categorical_columns=categorical,
        category_encodings=MappingProxyType(encodings),
        raw_feature_names=tuple(feature_names),
        source_types=tuple(source_types),
        training_means=np.asarray(means, dtype=float),
        training_sds=np.asarray(sds, dtype=float),
        keep_mask=np.asarray(keep, dtype=bool),
        zero_sd_tolerance=float(zero_sd_tolerance),
        ddof=0,
    )


def prepare_design_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    zero_sd_tolerance: float = 1e-12,
    strict_unseen_categories: bool = False,
) -> PreparedDesign:
    """Fit on ``train_df`` and transform both training and validation rows."""
    preprocessor = fit_preprocessor(
        train_df,
        numeric_columns,
        categorical_columns,
        zero_sd_tolerance=zero_sd_tolerance,
    )
    transformed_train = preprocessor.transform(train_df)
    transformed_valid = preprocessor.transform(
        valid_df,
        strict_unseen_categories=strict_unseen_categories,
    )
    return PreparedDesign(
        X_train=transformed_train.matrix,
        X_valid=transformed_valid.matrix,
        feature_names=preprocessor.kept_feature_names,
        audit=preprocessor.audit_frame(),
        preprocessor=preprocessor,
        valid_unseen_category_counts=transformed_valid.unseen_category_counts,
    )
