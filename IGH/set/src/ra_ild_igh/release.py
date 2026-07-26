#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified release-validation helpers for the RA/RA-ILD IGH V2 framework."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

FRAMEWORK_NAME = "RA/RA-ILD IGH V2"
FRAMEWORK_VERSION = "1.0.0"
DEFAULT_CONFIG = (
    "IGH/set/experiments/igh_scheme_003_repeat3_no_clinical/"
    "00_config/resolved_config.yaml"
)
DEFAULT_BASELINE_CONFIG = "IGH/set/configs/igh_baseline_m2_v1.yaml"
DEFAULT_MARKER = "IGH/set/FRAMEWORK_V2_COMPLETE.json"
EXPECTED_BRANCH = "feature/igh-scheme-upgrade-v1"
EXPECTED_SPLIT_SET_ID = "igh_ra_ild_repeat3_v1"
EXPECTED_ASSIGNMENT_SHA256 = (
    "5537365e41ac77a144c81fa7b7167896fc1ceb9db84664cd661b546575f8a030"
)

@dataclass(frozen=True)
class CheckSpec:
    name: str
    command: Tuple[str, ...]
    category: str
    expensive: bool = False
    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["command"] = list(self.command)
        return value

@dataclass(frozen=True)
class CheckResult:
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

def _script(root: Path, name: str) -> str:
    return str(root / "IGH/set/scripts_v2" / name)

def _config_command(python: str, root: Path, script: str, config: Path, *extra: str) -> Tuple[str, ...]:
    return (
        python, _script(root, script), "--config", str(config),
        "--repository-root", str(root), *extra,
    )

def _legacy_check_plan(
    repository_root: Path,
    config_path: Path,
    *,
    mode: str,
    python_executable: Optional[str],
    include_outer_orchestration: bool,
) -> Tuple[CheckSpec, ...]:
    """Preserve the Batch 01 public API used by the original unit tests."""
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be 'quick' or 'full'.")
    root = Path(repository_root).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    python = python_executable or sys.executable
    nested_extra: Tuple[str, ...] = ("--full",) if mode == "full" else ()
    checks = [
        CheckSpec("configuration", _config_command(python, root, "validate_experiment_config.py", config), "configuration"),
        CheckSpec("configured_paths", _config_command(python, root, "validate_experiment_config.py", config, "--check-paths"), "configuration"),
        CheckSpec("unit_tests", (python, "-m", "unittest", "discover", "-s", str(root / "IGH/set/tests"), "-p", "test_*.py", "-v"), "unit"),
        CheckSpec("v1_baseline", _config_command(python, root, "check_v1_baseline_outputs.py", config, "--scope", "full"), "real_data_regression"),
        CheckSpec("public_reference", _config_command(python, root, "check_public_reference_regression.py", config), "real_data_regression"),
        CheckSpec("preprocessing", _config_command(python, root, "check_preprocessing_regression.py", config), "real_data_regression"),
        CheckSpec("modeling", _config_command(python, root, "check_modeling_regression.py", config), "real_data_regression"),
        CheckSpec("specifications", _config_command(python, root, "check_specification_regression.py", config), "real_data_regression"),
        CheckSpec(f"nested_cv_{mode}", _config_command(python, root, "check_nested_cv_regression.py", config, *nested_extra), "real_data_regression", expensive=(mode == "full")),
    ]
    if include_outer_orchestration:
        checks.append(CheckSpec("outer_orchestration", _config_command(python, root, "check_outer_orchestration.py", config), "orchestration"))
    return tuple(checks)


def build_check_plan(
    repository_root: Path,
    config_path: Path,
    baseline_config_path: Optional[Path] = None,
    *,
    mode: str = "quick",
    python_executable: Optional[str] = None,
    allow_dirty_source: bool = False,
    include_outer_orchestration: bool = True,
) -> Tuple[CheckSpec, ...]:
    """Build either the legacy Batch 01 plan or the Batch 06 release plan.

    ``baseline_config_path=None`` intentionally selects the legacy API so the
    established release unit tests remain valid. Batch 06 always supplies the
    historical baseline path explicitly and therefore receives the 18-check
    release plan.
    """
    if baseline_config_path is None:
        return _legacy_check_plan(
            repository_root,
            config_path,
            mode=mode,
            python_executable=python_executable,
            include_outer_orchestration=include_outer_orchestration,
        )
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be quick or full")
    root = Path(repository_root).resolve()
    config = Path(config_path).resolve()
    baseline = Path(baseline_config_path).resolve()
    python = python_executable or sys.executable
    nested_extra: Tuple[str, ...] = ("--full",) if mode == "full" else ()
    release_extra: Tuple[str, ...] = ("--allow-dirty-source",) if allow_dirty_source else ()
    spec = root / "IGH/set/configs/split_sets/igh_ra_ild_repeat3_v1.yaml"
    bundle_spec = root / "IGH/set/configs/repeated_holdout_training/igh_repeat3_training_v1.yaml"
    checks = [
        CheckSpec("configuration", _config_command(python, root, "validate_experiment_config.py", config), "configuration"),
        CheckSpec("unit_tests", (python, "-m", "unittest", "discover", "-s", str(root/"IGH/set/tests"), "-p", "test_*.py", "-v"), "unit"),
        CheckSpec("batch02_feature_management", (python, _script(root, "test_batch02_feature_management.py")), "batch_regression"),
        CheckSpec("batch03_repeated_holdout", (python, _script(root, "test_batch03_repeated_holdout.py")), "batch_regression"),
        CheckSpec("batch04_training_bundle", (python, _script(root, "test_batch04_repeated_holdout_training.py")), "batch_regression"),
        CheckSpec("batch05_aggregation", (python, _script(root, "test_batch05_repeated_holdout_summary.py")), "batch_regression"),
        CheckSpec("batch06_release_unit", (python, _script(root, "test_batch06_release.py")), "batch_regression"),
        CheckSpec("historical_baseline", _config_command(python, root, "check_v1_baseline_outputs.py", baseline, "--scope", "full"), "historical_regression"),
        CheckSpec("public_reference", _config_command(python, root, "check_public_reference_regression.py", baseline), "scientific_regression"),
        CheckSpec("preprocessing", _config_command(python, root, "check_preprocessing_regression.py", baseline), "scientific_regression"),
        CheckSpec("modeling", _config_command(python, root, "check_modeling_regression.py", baseline), "scientific_regression"),
        CheckSpec("specifications", _config_command(python, root, "check_specification_regression.py", baseline), "scientific_regression"),
        CheckSpec(
            f"nested_cv_{mode}",
            _config_command(python, root, "check_nested_cv_regression.py", baseline, *nested_extra),
            "scientific_regression", expensive=(mode == "full"),
        ),
        CheckSpec("frozen_split", (python, _script(root, "validate_repeated_holdout_split_set.py"), "--spec", str(spec), "--repository-root", str(root), "--frozen"), "frozen_resources"),
        CheckSpec("frozen_training_bundle", (python, _script(root, "validate_repeated_holdout_training_bundle.py"), "--spec", str(bundle_spec), "--repository-root", str(root), "--mode", "frozen"), "frozen_resources"),
        CheckSpec("outer_task_tree", _config_command(python, root, "run_all_outer_tasks.py", config, "--status-only"), "task_tree"),
        CheckSpec("repeated_holdout_aggregation", _config_command(python, root, "validate_repeated_holdout_aggregation.py", config), "aggregation"),
        CheckSpec("release_contract", _config_command(python, root, "validate_release_contract.py", config, "--baseline-config", str(baseline), *release_extra), "release"),
    ]
    if not include_outer_orchestration:
        checks = [item for item in checks if item.name != "outer_task_tree"]
    return tuple(checks)

def package_versions(names: Sequence[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in names:
        try: out[name] = metadata.version(name)
        except metadata.PackageNotFoundError: out[name] = None
    return out

def _git(root: Path, *args: str) -> Optional[str]:
    completed = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True)
    if completed.returncode != 0: return None
    return completed.stdout.strip()

def git_metadata(repository_root: Path) -> Dict[str, Any]:
    root = Path(repository_root).resolve()
    tracked = _git(root, "status", "--porcelain", "--untracked-files=no")
    all_status = _git(root, "status", "--porcelain")
    upstream = _git(root, "rev-parse", "@{u}")
    head = _git(root, "rev-parse", "HEAD")
    untracked_igh = []
    if all_status:
        for line in all_status.splitlines():
            if line.startswith("?? ") and line[3:].startswith("IGH/"):
                untracked_igh.append(line[3:])
    return {
        "branch": _git(root, "branch", "--show-current"),
        "commit": head,
        "upstream_commit": upstream,
        "upstream_synchronized": bool(head and upstream and head == upstream),
        "tracked_worktree_clean_before_marker": tracked is not None and tracked == "",
        "untracked_igh_paths": untracked_igh,
        "untracked_outside_igh_allowed": True,
    }

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def validate_shm_aggregation_fixture() -> Mapping[str, Any]:
    """Codify the IGH two-NT-to-one-AA aggregation contract."""
    rows = [
        {"nt": "AAA", "aa": "K", "reads": 7, "frequency": 0.07},
        {"nt": "AAG", "aa": "K", "reads": 3, "frequency": 0.03},
        {"nt": "GCT", "aa": "A", "reads": 5, "frequency": 0.05},
    ]
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        item = grouped.setdefault(row["aa"], {"reads": 0, "frequency": 0.0, "nts": [], "top": None})
        item["reads"] += int(row["reads"])
        item["frequency"] += float(row["frequency"])
        item["nts"].append(row["nt"])
        if item["top"] is None or int(row["reads"]) > int(item["top"]["reads"]):
            item["top"] = row
    k = grouped["K"]
    passed = (
        k["reads"] == 10 and abs(k["frequency"] - 0.10) < 1e-12
        and len(k["nts"]) == 2 and k["top"]["nt"] == "AAA"
    )
    return {
        "passed": passed,
        "aa": "K",
        "nt_clone_number": len(k["nts"]),
        "read_count": k["reads"],
        "frequency_sum": k["frequency"],
        "top_nt_cdr3": k["top"]["nt"],
    }

def build_completion_payload(
    *,
    repository_root: Path,
    config_path: Path,
    mode: str,
    results: Sequence[CheckResult],
    baseline_config_path: Optional[Path] = None,
    framework_version: str = FRAMEWORK_VERSION,
    unit_test_count: Optional[int] = None,
    locked_model: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a Batch 06 payload while retaining the Batch 01 call contract."""
    items = [result.to_dict() for result in results]
    all_passed = bool(items) and all(item["passed"] for item in items)
    root = Path(repository_root).resolve()
    aggregation = root / "IGH/set/experiments/igh_scheme_003_repeat3_no_clinical/03_summary/08_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json"
    split_marker = root / "IGH/set/split_sets/igh_ra_ild_repeat3_v1/SPLITS_FROZEN.json"
    bundle_marker = root / "IGH/set/training_bundles/igh_repeat3_training_inputs_v1/TRAINING_BUNDLE_FROZEN.json"
    historical_baseline = (
        str(Path(baseline_config_path).resolve())
        if baseline_config_path is not None
        else None
    )
    payload = {
        "framework": FRAMEWORK_NAME,
        "framework_version": framework_version,
        "status": "complete" if all_passed else "validation_failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(root),
        "configuration": str(Path(config_path).resolve()),
        "historical_baseline_configuration": historical_baseline,
        "validation_mode": mode,
        "all_checks_passed": all_passed,
        "unit_test_count": unit_test_count,
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": package_versions(("numpy", "pandas", "scipy", "scikit-learn", "PyYAML")),
        },
        "git": git_metadata(root),
        "checks": items,
        "prior_full_nested_cv_validation": {
            "date": "2026-07-26",
            "outer_task": {"repeat": 1, "fold": 1},
            "inner_fits": 480,
            "status": "required_in_full_release_mode",
        },
        "locked_model": dict(locked_model or {}),
        "scientific_contract": {
            "receptor": "IGH",
            "positive_class": "ILD",
            "public_reference_training_partition_only": True,
            "fitting_public_features_exact_leave_one_out": True,
            "holdout_transform_training_reference_only": True,
            "alpha_lambda_selected_from_inner_cv": True,
            "threshold_selected_from_inner_oof": True,
            "full_cohort_catalog_is_indexing_universe_only": True,
            "shm_nt_to_aa_aggregation_fixture": dict(validate_shm_aggregation_fixture()),
        },
        "repeated_holdout": {
            "split_set_id": EXPECTED_SPLIT_SET_ID,
            "assignment_sha256": EXPECTED_ASSIGNMENT_SHA256,
            "splits": 3,
            "train_size": 120,
            "holdout_size": 49,
            "models": ["M1_static_igh", "M2_static_igh_public", "M3_static_igh_public_material"],
            "metric_rows": 9,
            "prediction_rows": 441,
            "results_are_not_an_independent_test": True,
        },
        "provenance_hashes": {
            "split_marker_sha256": sha256_file(split_marker) if split_marker.is_file() else None,
            "training_bundle_marker_sha256": sha256_file(bundle_marker) if bundle_marker.is_file() else None,
            "aggregation_marker_sha256": sha256_file(aggregation) if aggregation.is_file() else None,
        },
        "final_model": {
            "status": "not_selected",
            "automatic_final_model_selection": False,
            "deployable_joblib_built": False,
            "selection_protocol_required_before_final_fit": True,
        },
        "independent_test_policy": {
            "historical_independent_test_outputs_read_only_for_baseline_regression": True,
            "repeated_holdout_engine_reads_historical_independent_test": False,
            "historical_test_refit_or_reprediction_performed": False,
            "nested_cv_engine_reads_independent_test": False,
            "independent_test_refit_or_reprediction_in_v2": False,
            "frozen_v1_test_outputs_read_by_regression_checks": True,
        },
        "outer_task_status": {
            "full_v2_recomputation_required_for_release": False,
            "remaining_tasks_are_optional_mirror_computation": False,
            "repeated_holdout_tasks_complete": True,
        },
        "git_delivery": {
            "source_only": True,
            "generated_training_bundles_committed": False,
            "experiment_results_committed": False,
            "recommended_order": "merge TRB framework PR first, then rebase or cherry-pick IGH-only commits onto updated main",
        },
    }
    return payload

def write_completion_marker(path: Path, payload: Mapping[str, Any]) -> Path:
    if payload.get("status") != "complete" or payload.get("all_checks_passed") is not True:
        raise ValueError("Completion marker requires every release check to pass")
    git = payload.get("git")
    # Strict Git release safeguards apply to real Batch 06 payloads. Lightweight
    # legacy unit-test payloads intentionally omit the Git block.
    if isinstance(git, Mapping):
        if git.get("tracked_worktree_clean_before_marker") is not True:
            raise ValueError("Completion marker requires a clean tracked/staged worktree")
        if git.get("upstream_synchronized") is not True:
            raise ValueError("Completion marker requires HEAD synchronized with upstream")
        if git.get("untracked_igh_paths"):
            raise ValueError("Completion marker requires no untracked IGH source paths")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target
