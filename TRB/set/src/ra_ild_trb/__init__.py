"""Reusable components for the RA/RA-ILD TRB V2 framework.

The package initializer intentionally imports only lightweight modules.
Optional scientific dependencies such as scikit-learn are imported by their
specific submodules (for example ``ra_ild_trb.modeling`` and
``ra_ild_trb.metrics``). This keeps configuration and path validation usable
before modeling dependencies are installed.
"""

from .config import ConfigError, ExperimentConfig, load_experiment_config
from .paths import RepositoryRootError, find_repository_root, resolve_project_path

__all__ = [
    "ConfigError",
    "ExperimentConfig",
    "RepositoryRootError",
    "find_repository_root",
    "load_experiment_config",
    "resolve_project_path",
]

__version__ = "0.7.0"
