#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Release-validation helpers for the TRB V2 framework.

This module keeps the unified validation plan and completion-marker schema in
one standard-library-only module. Importing package metadata therefore remains
lightweight and does not eagerly import scikit-learn.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


FRAMEWORK_NAME = "RA/RA-ILD TRB V2"
FRAMEWORK_VERSION = "1.0.0"
DEFAULT_CONFIG = "TRB/set/configs/trb_baseline_m2_v1.yaml"
DEFAULT_MARKER = "TRB/set/FRAMEWORK_V2_COMPLETE.json"

# The complete 480-inner-fit regression was already executed successfully on
# the project server before the release batch. Keeping this evidence separate
# prevents a quick release check from pretending that it repeated the expensive
# full nested-CV run.
PRIOR_FULL_NESTED_VALIDATION: Mapping[str, Any] = {
    "date": "2026-07-24",
    "outer_task": {"repeat": 1, "fold": 1},
    "summary": "Mode=full Outer task=1/1 Checks=43 PASS=43 FAIL=0",
    "inner_fits": 480,
    "status": "verified_before_release_batch",
}


@dataclass(frozen=True)
class CheckSpec:
    """One command in the unified release-validation plan."""

    name: str
    command: Tuple[str, ...]
    category: str
    expensive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        return payload


@dataclass(frozen=True)
class CheckResult:
    """Normalized result from one validation command."""

    name: str
    category: str
    command: Tuple[str, ...]
    returncode: int
    duration_seconds: float
    stdout_tail: str
    stderr_tail: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "command": list(self.command),
            "returncode": int(self.returncode),
            "passed": bool(self.passed),
            "duration_seconds": float(self.duration_seconds),
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


def _script_command(
    python_executable: str,
    repository_root: Path,
    script_name: str,
    config_path: Path,
    *extra: str,
) -> Tuple[str, ...]:
    script = repository_root / "TRB/set/scripts_v2" / script_name
    return (
        python_executable,
        str(script),
        "--config",
        str(config_path),
        *extra,
    )


def build_check_plan(
    repository_root: Path,
    config_path: Path,
    *,
    mode: str = "quick",
    python_executable: Optional[str] = None,
    include_outer_orchestration: bool = True,
) -> Tuple[CheckSpec, ...]:
    """Build the ordered, auditable framework-validation plan.

    ``quick`` uses frozen V1 inner-selection outputs and refits only the four
    outer-final models. ``full`` repeats all 480 inner fits for the frozen
    repeat-01/fold-01 task. Outer-orchestration validation can be skipped while
    an optional background V2 mirror task is actively writing its directory.
    """
    root = Path(repository_root).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be 'quick' or 'full'.")
    python = python_executable or sys.executable

    nested_extra: Tuple[str, ...] = ("--full",) if mode == "full" else ()
    checks = [
        CheckSpec(
            "configuration",
            _script_command(python, root, "validate_experiment_config.py", config),
            "configuration",
        ),
        CheckSpec(
            "configured_paths",
            _script_command(
                python,
                root,
                "validate_experiment_config.py",
                config,
                "--check-paths",
            ),
            "configuration",
        ),
        CheckSpec(
            "unit_tests",
            (
                python,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(root / "TRB/set/tests"),
                "-p",
                "test_*.py",
                "-v",
            ),
            "unit",
        ),
        CheckSpec(
            "v1_baseline",
            _script_command(
                python,
                root,
                "check_v1_baseline_outputs.py",
                config,
                "--scope",
                "full",
            ),
            "real_data_regression",
        ),
        CheckSpec(
            "public_reference",
            _script_command(
                python, root, "check_public_reference_regression.py", config
            ),
            "real_data_regression",
        ),
        CheckSpec(
            "preprocessing",
            _script_command(
                python, root, "check_preprocessing_regression.py", config
            ),
            "real_data_regression",
        ),
        CheckSpec(
            "modeling",
            _script_command(python, root, "check_modeling_regression.py", config),
            "real_data_regression",
        ),
        CheckSpec(
            "specifications",
            _script_command(
                python, root, "check_specification_regression.py", config
            ),
            "real_data_regression",
        ),
        CheckSpec(
            f"nested_cv_{mode}",
            _script_command(
                python,
                root,
                "check_nested_cv_regression.py",
                config,
                *nested_extra,
            ),
            "real_data_regression",
            expensive=(mode == "full"),
        ),
    ]
    if include_outer_orchestration:
        checks.append(
            CheckSpec(
                "outer_orchestration",
                _script_command(
                    python, root, "check_outer_orchestration.py", config
                ),
                "orchestration",
            )
        )
    return tuple(checks)


def package_versions(names: Sequence[str]) -> Dict[str, Optional[str]]:
    """Return installed package versions without importing scientific modules."""
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_metadata(repository_root: Path) -> Dict[str, Any]:
    """Collect best-effort Git metadata for the completion marker."""
    root = Path(repository_root).resolve()

    def run(*args: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        value = completed.stdout.strip()
        return value or None

    dirty = run("status", "--porcelain")
    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "working_tree_clean_before_marker": (
            None if dirty is None else not bool(dirty)
        ),
    }


def build_completion_payload(
    *,
    repository_root: Path,
    config_path: Path,
    mode: str,
    results: Sequence[CheckResult],
    framework_version: str = FRAMEWORK_VERSION,
    unit_test_count: Optional[int] = None,
    locked_model: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create the JSON-serializable framework completion record."""
    result_list = [result.to_dict() for result in results]
    all_passed = bool(result_list) and all(item["passed"] for item in result_list)
    return {
        "framework": FRAMEWORK_NAME,
        "framework_version": framework_version,
        "status": "complete" if all_passed else "validation_failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(Path(repository_root).resolve()),
        "configuration": str(Path(config_path).resolve()),
        "validation_mode": mode,
        "all_checks_passed": all_passed,
        "unit_test_count": unit_test_count,
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": package_versions(
                ("numpy", "pandas", "scipy", "scikit-learn", "PyYAML")
            ),
        },
        "git": git_metadata(Path(repository_root)),
        "checks": result_list,
        "prior_full_nested_cv_validation": dict(PRIOR_FULL_NESTED_VALIDATION),
        "locked_model": dict(locked_model or {}),
        "independent_test_policy": {
            "nested_cv_engine_reads_independent_test": False,
            "independent_test_refit_or_reprediction_in_v2": False,
            "frozen_v1_test_outputs_read_by_regression_checks": True,
        },
        "outer_task_status": {
            "full_v2_recomputation_required_for_release": False,
            "remaining_tasks_are_optional_mirror_computation": True,
        },
    }


def write_completion_marker(path: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write a completion marker after successful validation."""
    if payload.get("status") != "complete" or not payload.get("all_checks_passed"):
        raise ValueError("A completion marker can only be written after all checks pass.")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
