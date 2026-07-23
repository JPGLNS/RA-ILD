"""Reusable components for the RA/RA-ILD TRB V2 framework."""

from .config import ConfigError, ExperimentConfig, load_experiment_config
from .paths import RepositoryRootError, find_repository_root, resolve_project_path
from .public_reference import (
    ALL_PUBLIC_FEATURES,
    MAIN_THRESHOLD_SCHEME,
    PUBLIC_FEATURE_SETS,
    PublicFeatureResult,
    PublicReference,
    ThresholdScheme,
    apply_reference,
    build_reference_from_counts,
    build_reference_from_presence,
    external_public_features,
    leave_one_out_public_features,
    public_feature_columns,
)

__all__ = [
    "ALL_PUBLIC_FEATURES",
    "ConfigError",
    "ExperimentConfig",
    "MAIN_THRESHOLD_SCHEME",
    "PUBLIC_FEATURE_SETS",
    "PublicFeatureResult",
    "PublicReference",
    "RepositoryRootError",
    "ThresholdScheme",
    "apply_reference",
    "build_reference_from_counts",
    "build_reference_from_presence",
    "external_public_features",
    "find_repository_root",
    "leave_one_out_public_features",
    "load_experiment_config",
    "public_feature_columns",
    "resolve_project_path",
]

__version__ = "0.2.0"
