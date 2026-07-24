#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration-driven analysis-scheme preparation for the RA-ILD TRB framework.

Batch 01 deliberately leaves the scientific modeling engine unchanged.  It turns a
frozen V2 configuration into an isolated, auditable analysis scheme by resolving a
small scheme YAML, rewriting only output destinations, and snapshotting inputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Dict, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .config import validate_experiment_mapping
from .paths import resolve_project_path


SCHEME_VERSION = "1.0"
SCHEME_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
DEFAULT_SCHEME_ROOT = Path("TRB/set/experiments")


class SchemeError(ValueError):
    """Raised when a scheme definition or output layout is unsafe or invalid."""


@dataclass(frozen=True)
class PreparedScheme:
    """Resolved scheme metadata returned before or after writing snapshots."""

    scheme_id: str
    description: str
    source_path: Path
    base_config_path: Path
    output_root: Path
    resolved_config: Mapping[str, Any]
    input_manifest: Sequence[Mapping[str, Any]]

    @property
    def resolved_config_path(self) -> Path:
        return self.output_root / "00_config" / "resolved_config.yaml"


OUTPUT_PATH_KEYS: Tuple[Tuple[str, ...], ...] = (
    ("outer_tasks", "output_root"),
    ("outer_tasks", "manifest"),
    ("outer_tasks", "status"),
    ("outer_tasks", "logs_dir"),
    ("aggregation", "output_dir"),
    ("final_model", "bundle"),
    ("final_model", "configuration"),
    ("final_model", "reference_masks"),
    ("independent_validation", "output_dir"),
)

INPUT_PATH_KEYS: Tuple[Tuple[str, ...], ...] = (
    ("data", "train", "metadata"),
    ("data", "train", "base_matrix"),
    ("data", "train", "feature_manifest"),
    ("data", "train", "aa_clone_table_dir"),
    ("data", "train", "public_catalog"),
    ("data", "train", "reference_definition"),
    ("data", "test", "metadata"),
    ("data", "test", "final_matrix"),
    ("data", "test", "reference_features"),
    ("data", "test", "reference_definition_used"),
    ("public_reference", "cache", "presence"),
    ("public_reference", "cache", "frequency"),
    ("public_reference", "cache", "metadata"),
    ("cross_validation", "outer_assignments"),
    ("cross_validation", "inner_assignments"),
    ("outer_tasks", "runner_script"),
)


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required. Install with: mamba install -c conda-forge pyyaml"
        )


def load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    """Read a YAML file and require a mapping at the root."""

    _require_yaml()
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise SchemeError(f"YAML root must be a mapping: {path}")
    return value


def dump_yaml(mapping: Mapping[str, Any]) -> str:
    """Return deterministic, readable UTF-8 YAML text."""

    _require_yaml()
    return yaml.safe_dump(
        mapping,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge mappings; non-mapping values are replaced by the override."""

    result: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _get_nested(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise SchemeError(f"Missing configuration key: {'.'.join(keys)}")
        current = current[key]
    return current


def _set_nested(mapping: MutableMapping[str, Any], keys: Sequence[str], value: Any) -> None:
    current: MutableMapping[str, Any] = mapping
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, MutableMapping):
            raise SchemeError(f"Configuration key is not a mapping: {'.'.join(keys[:-1])}")
        current = child
    current[keys[-1]] = value


def _repo_relative(path: Path, repository_root: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    root = Path(repository_root).expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise SchemeError(f"Path must remain inside repository root: {resolved}") from exc


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_scheme_id(value: Any) -> str:
    """Validate a filesystem-safe, stable scheme identifier."""

    if not isinstance(value, str) or not SCHEME_ID_PATTERN.fullmatch(value.strip()):
        raise SchemeError(
            "scheme.id must be 3-80 lowercase characters using only a-z, 0-9, '.', '_' or '-'"
        )
    return value.strip()


def _resolve_scheme_output_root(
    scheme_id: str,
    raw_value: Any,
    repository_root: Path,
) -> Path:
    if raw_value is None or raw_value == "":
        relative = DEFAULT_SCHEME_ROOT / scheme_id
    else:
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise SchemeError("scheme.output_root must be a non-empty repository-relative path")
        candidate = Path(raw_value.strip()).expanduser()
        if candidate.is_absolute():
            raise SchemeError("scheme.output_root must be repository-relative, not absolute")
        relative = candidate

    output_root = resolve_project_path(relative, repository_root)
    allowed_root = resolve_project_path(DEFAULT_SCHEME_ROOT, repository_root)
    if not _is_within(output_root, allowed_root):
        raise SchemeError(
            f"scheme.output_root must be inside {DEFAULT_SCHEME_ROOT.as_posix()}: {relative}"
        )
    if output_root.name != scheme_id:
        raise SchemeError(
            "scheme.output_root must end with the exact scheme.id so schemes cannot share outputs"
        )
    return output_root


def _rewrite_output_paths(
    resolved: MutableMapping[str, Any],
    output_root: Path,
    repository_root: Path,
) -> None:
    """Rewrite only scheme-owned outputs while preserving every scientific input/rule."""

    outer_root = output_root / "02_tasks" / "outer_nested_cv"
    replacements = {
        ("outer_tasks", "output_root"): outer_root,
        ("outer_tasks", "manifest"): outer_root / "07_outer_task_manifest.csv",
        ("outer_tasks", "status"): outer_root / "07_outer_task_status.csv",
        ("outer_tasks", "logs_dir"): output_root / "logs" / "outer_tasks",
        ("aggregation", "output_dir"): output_root / "03_summary",
        ("final_model", "bundle"): output_root / "04_final_model" / "final_model.joblib",
        ("final_model", "configuration"): output_root
        / "04_final_model"
        / "final_model_configuration.json",
        ("final_model", "reference_masks"): output_root
        / "04_final_model"
        / "final_public_reference_masks.npz",
        ("independent_validation", "output_dir"): output_root
        / "05_independent_validation",
    }
    for keys, absolute_path in replacements.items():
        _set_nested(resolved, keys, _repo_relative(absolute_path, repository_root))


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_input_manifest(
    resolved: Mapping[str, Any],
    repository_root: Path,
) -> Tuple[Mapping[str, Any], ...]:
    """Record immutable identifiers for configured input files without copying large data."""

    rows = []
    for keys in INPUT_PATH_KEYS:
        configured = _get_nested(resolved, keys)
        if not isinstance(configured, str) or not configured.strip():
            raise SchemeError(f"Configured input path is invalid: {'.'.join(keys)}")
        path = resolve_project_path(configured, repository_root)
        row: Dict[str, Any] = {
            "config_key": ".".join(keys),
            "configured_path": configured,
            "resolved_path": str(path),
            "exists": path.exists(),
            "kind": "missing",
            "size_bytes": None,
            "mtime_utc": None,
            "sha256": None,
        }
        if path.is_file():
            stat = path.stat()
            row.update(
                {
                    "kind": "file",
                    "size_bytes": int(stat.st_size),
                    "mtime_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "sha256": _sha256_file(path),
                }
            )
        elif path.is_dir():
            stat = path.stat()
            row.update(
                {
                    "kind": "directory",
                    "mtime_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        rows.append(row)
    return tuple(rows)


def _git_commit(repository_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def environment_snapshot(repository_root: Path) -> Mapping[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "PyYAML", "joblib"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "working_directory": os.getcwd(),
        "repository_root": str(repository_root),
        "git_commit": _git_commit(repository_root),
        "packages": packages,
    }


def build_prepared_scheme(
    scheme_path: Path,
    repository_root: Path,
) -> PreparedScheme:
    """Resolve a scheme YAML into a validated V2-compatible configuration."""

    scheme_path = Path(scheme_path).expanduser().resolve()
    repository_root = Path(repository_root).expanduser().resolve()
    raw_scheme = load_yaml_mapping(scheme_path)
    if str(raw_scheme.get("scheme_version")) != SCHEME_VERSION:
        raise SchemeError(f"scheme_version must be '{SCHEME_VERSION}'")

    scheme_section = raw_scheme.get("scheme")
    if not isinstance(scheme_section, Mapping):
        raise SchemeError("scheme must be a mapping")
    scheme_id = validate_scheme_id(scheme_section.get("id"))
    description = str(scheme_section.get("description", "")).strip()
    base_value = scheme_section.get("base_config")
    if not isinstance(base_value, str) or not base_value.strip():
        raise SchemeError("scheme.base_config must be a non-empty path")
    base_config_path = resolve_project_path(
        base_value.strip(), repository_root, must_exist=True, expect="file"
    )
    output_root = _resolve_scheme_output_root(
        scheme_id, scheme_section.get("output_root"), repository_root
    )

    base = load_yaml_mapping(base_config_path)
    overrides = raw_scheme.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise SchemeError("overrides must be a mapping")
    resolved = deep_merge(base, overrides)
    experiment = resolved.get("experiment")
    if not isinstance(experiment, MutableMapping):
        raise SchemeError("Resolved experiment section must be a mapping")
    experiment["id"] = scheme_id
    experiment["description"] = description or str(
        experiment.get("description", "RA-ILD TRB analysis scheme")
    )
    experiment["status"] = str(scheme_section.get("status", "development"))
    experiment["scheme_base_config"] = _repo_relative(base_config_path, repository_root)
    experiment["scheme_output_root"] = _repo_relative(output_root, repository_root)

    _rewrite_output_paths(resolved, output_root, repository_root)
    validate_experiment_mapping(resolved)

    # Defense in depth: verify every rewritten output resolves under this scheme root.
    for keys in OUTPUT_PATH_KEYS:
        configured = _get_nested(resolved, keys)
        path = resolve_project_path(configured, repository_root)
        if not _is_within(path, output_root):
            raise SchemeError(
                f"Output path escapes scheme root: {'.'.join(keys)} -> {path}"
            )

    manifest = collect_input_manifest(resolved, repository_root)
    return PreparedScheme(
        scheme_id=scheme_id,
        description=description,
        source_path=scheme_path,
        base_config_path=base_config_path,
        output_root=output_root,
        resolved_config=resolved,
        input_manifest=manifest,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _directory_has_results(output_root: Path) -> bool:
    for relative in (
        Path("02_tasks"),
        Path("03_summary"),
        Path("04_final_model"),
        Path("05_independent_validation"),
    ):
        path = output_root / relative
        if path.exists() and any(item.is_file() for item in path.rglob("*")):
            return True
    return False


def write_prepared_scheme(
    prepared: PreparedScheme,
    repository_root: Path,
    *,
    allow_missing_inputs: bool = False,
    refresh_config_only: bool = False,
) -> Mapping[str, Any]:
    """Create scheme directories and write reproducibility snapshots.

    Existing scientific result files are never removed.  ``refresh_config_only`` is
    accepted only when no result files are present, preventing a configuration from
    silently changing underneath completed tasks.
    """

    repository_root = Path(repository_root).expanduser().resolve()
    missing = [row for row in prepared.input_manifest if not bool(row["exists"])]
    if missing and not allow_missing_inputs:
        labels = "\n".join(
            f"  - {row['config_key']}: {row['resolved_path']}" for row in missing
        )
        raise FileNotFoundError("Configured scheme inputs are missing:\n" + labels)

    marker_path = prepared.output_root / "SCHEME_PREPARED.json"
    if prepared.output_root.exists():
        if _directory_has_results(prepared.output_root):
            raise SchemeError(
                "Scheme output already contains scientific results; preparation refuses "
                "to change its configuration. Create a new scheme.id instead."
            )
        if marker_path.exists() and not refresh_config_only:
            raise FileExistsError(
                f"Scheme is already prepared: {prepared.output_root}. "
                "Use --refresh-config-only only before any model result exists."
            )

    for relative in (
        "00_config",
        "01_splits",
        "02_tasks",
        "03_summary",
        "04_final_model",
        "05_independent_validation",
        "logs",
    ):
        (prepared.output_root / relative).mkdir(parents=True, exist_ok=True)

    config_dir = prepared.output_root / "00_config"
    source_text = prepared.source_path.read_text(encoding="utf-8")
    base_text = prepared.base_config_path.read_text(encoding="utf-8")
    resolved_text = dump_yaml(prepared.resolved_config)

    (config_dir / "source_scheme.yaml").write_text(source_text, encoding="utf-8")
    (config_dir / "base_config_snapshot.yaml").write_text(base_text, encoding="utf-8")
    (config_dir / "resolved_config.yaml").write_text(resolved_text, encoding="utf-8")
    _write_json(config_dir / "input_manifest.json", list(prepared.input_manifest))
    env = environment_snapshot(repository_root)
    _write_json(config_dir / "environment.json", env)

    metadata = {
        "scheme_version": SCHEME_VERSION,
        "scheme_id": prepared.scheme_id,
        "description": prepared.description,
        "source_scheme": str(prepared.source_path),
        "base_config": str(prepared.base_config_path),
        "output_root": str(prepared.output_root),
        "resolved_config": str(prepared.resolved_config_path),
        "resolved_config_sha256": hashlib.sha256(
            resolved_text.encode("utf-8")
        ).hexdigest(),
        "base_config_sha256": _sha256_file(prepared.base_config_path),
        "source_scheme_sha256": _sha256_file(prepared.source_path),
        "missing_input_count": len(missing),
        "scientific_engine_changed_by_batch01": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": env.get("git_commit"),
    }
    _write_json(config_dir / "scheme_metadata.json", metadata)
    _write_json(
        marker_path,
        {
            "status": "PREPARED",
            "scheme_id": prepared.scheme_id,
            "resolved_config": str(prepared.resolved_config_path),
            "created_at_utc": metadata["created_at_utc"],
            "scientific_engine_changed_by_batch01": False,
        },
    )
    return metadata
