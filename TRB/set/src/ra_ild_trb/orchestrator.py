#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-7 final experiment orchestration for the TRB feature framework.

This module does not implement new biological features or ML algorithms.  It
coordinates the already-tested Phase-2/5/6 components so one experiment YAML can
be executed from repertoire-source preparation through leakage-controlled ML.

Key policies
------------
* all/Top-K repertoire source is resolved from the unified experiment config;
* one shared Step02 *static-core cache* is maintained per repertoire source;
* the Step02 cache is never used as a source of preselected 3-mers for modeling;
* 3-mer vocabulary and enriched-dictionary features remain fold-local in Phase 6;
* total/PBMC/buffycoat experiments may share the same full train/test Step02 core;
* existing valid caches are reused, while stale caches require explicit rebuild;
* output directories are deterministic and experiment-specific.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml

from .dynamic_repertoire import (
    DynamicRepertoireSource,
    render_dynamic_registry,
    render_phase2_plan,
    resolve_dynamic_repertoire,
    validate_dynamic_source,
)
from .unified_experiment import UnifiedExperimentSpec, load_unified_experiment

PathLike = Union[str, Path]
ORCHESTRATOR_VERSION = "1.0.0-phase7"
EXPERIMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,99}$")


class OrchestratorError(ValueError):
    pass


@dataclass(frozen=True)
class OrchestratorPaths:
    repository_root: Path
    experiment_root: Path
    resolved_dir: Path
    step02_cache_root: Path
    train_step02_dir: Path
    test_step02_dir: Path
    ml_output_dir: Path
    manifest_path: Path
    summary_path: Path

    def as_dict(self) -> Dict[str, str]:
        return {
            "experiment_root": str(self.experiment_root),
            "resolved_dir": str(self.resolved_dir),
            "step02_cache_root": str(self.step02_cache_root),
            "train_step02_dir": str(self.train_step02_dir),
            "test_step02_dir": str(self.test_step02_dir),
            "ml_output_dir": str(self.ml_output_dir),
            "manifest": str(self.manifest_path),
            "summary": str(self.summary_path),
        }


@dataclass(frozen=True)
class Step02CacheAudit:
    path: Path
    mode: str
    source_id: str
    sample_count: int
    status: str = "PASS"

    def as_dict(self) -> Dict[str, object]:
        return {
            "path": str(self.path),
            "mode": self.mode,
            "source_id": self.source_id,
            "sample_count": self.sample_count,
            "status": self.status,
        }


@dataclass(frozen=True)
class OrchestratorPlan:
    spec: UnifiedExperimentSpec
    source: DynamicRepertoireSource
    paths: OrchestratorPaths
    train_metadata: Path
    test_metadata: Path
    registry_path: Path
    repeat_override: Optional[int]
    workers: int
    source_ready: bool
    source_error: Optional[str]
    train_step02_ready: bool
    train_step02_error: Optional[str]
    test_step02_ready: bool
    test_step02_error: Optional[str]

    @property
    def step02_ready(self) -> bool:
        return self.train_step02_ready and self.test_step02_ready


def _resolve(value: PathLike, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _safe_experiment_id(value: str) -> str:
    text = str(value).strip()
    if not EXPERIMENT_ID_RE.fullmatch(text):
        raise OrchestratorError(
            "experiment.id must be 2-100 characters using only letters, digits, '.', '_' or '-'"
        )
    return text


def _read_full_metadata_ids(path: PathLike, *, id_col: str = "libraryid") -> Tuple[str, ...]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source, dtype=str)
    if id_col not in frame.columns:
        raise OrchestratorError(f"metadata missing {id_col}: {source}")
    ids = frame[id_col].astype(str).str.strip()
    if ids.eq("").any() or ids.isna().any() or ids.duplicated().any():
        raise OrchestratorError(f"metadata {id_col} must be non-empty and unique: {source}")
    return tuple(ids.tolist())


def resolve_orchestrator_paths(
    spec: UnifiedExperimentSpec,
    repository_root: PathLike,
    *,
    workspace_root: Optional[PathLike] = None,
    train_step02_override: Optional[PathLike] = None,
    test_step02_override: Optional[PathLike] = None,
    repeat_override: Optional[int] = None,
) -> OrchestratorPaths:
    root = Path(repository_root).expanduser().resolve()
    experiment_id = _safe_experiment_id(spec.base.experiment_id)
    experiment_root = (
        _resolve(workspace_root, root)
        if workspace_root is not None
        else root / "TRB/result/feature_framework/experiments" / experiment_id
    )
    source_id = spec.repertoire_source_id
    cache_root = root / "TRB/result/feature_framework/cache/step02" / source_id
    train_step02 = (
        _resolve(train_step02_override, root)
        if train_step02_override is not None
        else cache_root / "train"
    )
    test_step02 = (
        _resolve(test_step02_override, root)
        if test_step02_override is not None
        else cache_root / "test"
    )
    ml_leaf = f"smoke_repeat{int(repeat_override)}" if repeat_override is not None else "full"
    ml_output = experiment_root / "06_unified_ml" / ml_leaf
    return OrchestratorPaths(
        repository_root=root,
        experiment_root=experiment_root,
        resolved_dir=experiment_root / "00_resolved",
        step02_cache_root=cache_root,
        train_step02_dir=train_step02,
        test_step02_dir=test_step02,
        ml_output_dir=ml_output,
        manifest_path=experiment_root / "07_orchestrator_manifest.json",
        summary_path=experiment_root / "07_orchestrator_summary.md",
    )


def validate_step02_cache(
    step02_dir: PathLike,
    *,
    expected_ids: Sequence[str],
    expected_source_id: str,
    mode: str,
) -> Step02CacheAudit:
    root = Path(step02_dir).expanduser().resolve()
    if mode not in {"train", "test"}:
        raise OrchestratorError("Step02 cache mode must be train/test")
    core_path = root / "02_core_sample_features.csv"
    source_path = root / "02_resolved_feature_source.json"
    if not core_path.is_file():
        raise FileNotFoundError(f"Step02 core cache missing: {core_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Step02 provenance missing: {source_path}")
    core = pd.read_csv(core_path, usecols=["sample_id"])
    ids = core["sample_id"].astype(str).tolist()
    expected = [str(x) for x in expected_ids]
    if ids != expected:
        raise OrchestratorError(
            f"{mode} Step02 cache sample order/coverage mismatch: observed={len(ids)}, expected={len(expected)}"
        )
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    observed_mode = str(payload.get("mode", ""))
    observed_source = str((payload.get("repertoire_input") or {}).get("source_id", ""))
    observed_count = int(payload.get("sample_count", -1))
    if observed_mode != mode:
        raise OrchestratorError(
            f"{mode} Step02 cache mode mismatch: observed={observed_mode!r}"
        )
    if observed_source != expected_source_id:
        raise OrchestratorError(
            f"{mode} Step02 source mismatch: observed={observed_source!r}, expected={expected_source_id!r}"
        )
    if observed_count != len(expected):
        raise OrchestratorError(
            f"{mode} Step02 sample_count mismatch: observed={observed_count}, expected={len(expected)}"
        )
    return Step02CacheAudit(
        path=root,
        mode=mode,
        source_id=expected_source_id,
        sample_count=len(expected),
    )


def _try_source(source: DynamicRepertoireSource, ids: Sequence[str], deep: bool) -> Tuple[bool, Optional[str]]:
    try:
        validate_dynamic_source(source, ids, deep=deep)
        return True, None
    except (FileNotFoundError, ValueError) as exc:
        return False, str(exc)


def _try_step02(path: Path, ids: Sequence[str], source_id: str, mode: str) -> Tuple[bool, Optional[str]]:
    try:
        validate_step02_cache(
            path,
            expected_ids=ids,
            expected_source_id=source_id,
            mode=mode,
        )
        return True, None
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return False, str(exc)


def build_orchestrator_plan(
    config: PathLike,
    *,
    repository_root: PathLike,
    train_metadata: PathLike = "TRB/set/train/metadata_train_70.csv",
    test_metadata: PathLike = "TRB/set/test/metadata_test_30.csv",
    registry: PathLike = "TRB/set/configs/feature_framework/trb_feature_registry_v1.yaml",
    workspace_root: Optional[PathLike] = None,
    train_step02_override: Optional[PathLike] = None,
    test_step02_override: Optional[PathLike] = None,
    repeat_override: Optional[int] = None,
    workers_override: Optional[int] = None,
    deep_source_check: bool = False,
) -> OrchestratorPlan:
    root = Path(repository_root).expanduser().resolve()
    config_path = _resolve(config, root)
    spec = load_unified_experiment(config_path)
    if repeat_override is not None and repeat_override < 1:
        raise OrchestratorError("repeat_override must be >= 1")
    workers = spec.workers if workers_override is None else int(workers_override)
    if workers < 1 or workers > 16:
        raise OrchestratorError("workers must be in [1,16]")
    train_meta = _resolve(train_metadata, root)
    test_meta = _resolve(test_metadata, root)
    registry_path = _resolve(registry, root)
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    train_ids = _read_full_metadata_ids(train_meta)
    test_ids = _read_full_metadata_ids(test_meta)
    if set(train_ids) & set(test_ids):
        raise OrchestratorError("train/test metadata sample IDs overlap")
    source = resolve_dynamic_repertoire(spec.base, root)
    paths = resolve_orchestrator_paths(
        spec,
        root,
        workspace_root=workspace_root,
        train_step02_override=train_step02_override,
        test_step02_override=test_step02_override,
        repeat_override=repeat_override,
    )
    all_ids = (*train_ids, *test_ids)
    source_ready, source_error = _try_source(source, all_ids, deep_source_check)
    train_ready, train_error = _try_step02(
        paths.train_step02_dir, train_ids, source.source_id, "train"
    )
    test_ready, test_error = _try_step02(
        paths.test_step02_dir, test_ids, source.source_id, "test"
    )
    return OrchestratorPlan(
        spec=spec,
        source=source,
        paths=paths,
        train_metadata=train_meta,
        test_metadata=test_meta,
        registry_path=registry_path,
        repeat_override=repeat_override,
        workers=workers,
        source_ready=source_ready,
        source_error=source_error,
        train_step02_ready=train_ready,
        train_step02_error=train_error,
        test_step02_ready=test_ready,
        test_step02_error=test_error,
    )


def _python_command(script: Path, *args: object) -> List[str]:
    return [sys.executable, str(script), *[str(x) for x in args]]


def source_builder_command(plan: OrchestratorPlan) -> Optional[List[str]]:
    if plan.source_ready:
        return None
    if plan.spec.base.repertoire.mode == "all":
        raise OrchestratorError(
            "full repertoire source is not ready; Phase 7 will not regenerate Step01 automatically: "
            + str(plan.source_error)
        )
    root = plan.paths.repository_root
    script = root / "TRB/set/scripts/build_01b_AA_clone_table_topk.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    return _python_command(
        script,
        "--input-dir", root / "TRB/result/01_AA_clone_table",
        "--output-dir", plan.source.aa_dir,
        "--top-k", int(plan.source.top_k),
    )


def render_resolved_contract(plan: OrchestratorPlan) -> Mapping[str, object]:
    registry = render_dynamic_registry(plan.registry_path, plan.spec.base)
    phase2_plan = render_phase2_plan(plan.spec.base)
    return {
        "resolved_experiment": plan.spec.as_dict(),
        "repertoire_source": plan.source.as_dict(),
        "dynamic_registry": registry,
        "phase2_plan": phase2_plan,
    }


def write_resolved_contract(plan: OrchestratorPlan, *, overwrite: bool = True) -> Dict[str, Path]:
    out = plan.paths.resolved_dir
    if out.exists() and not overwrite:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = render_resolved_contract(plan)
    paths = {
        "experiment": out / "07_resolved_experiment.json",
        "registry": out / "07_resolved_registry.yaml",
        "phase2_plan": out / "07_resolved_phase2_plan.yaml",
    }
    paths["experiment"].write_text(
        json.dumps(
            {
                "resolved_experiment": payload["resolved_experiment"],
                "repertoire_source": payload["repertoire_source"],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    paths["registry"].write_text(
        yaml.safe_dump(payload["dynamic_registry"], sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    paths["phase2_plan"].write_text(
        yaml.safe_dump(payload["phase2_plan"], sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return paths


def step02_commands(plan: OrchestratorPlan, resolved_paths: Mapping[str, Path]) -> Tuple[List[str], List[str]]:
    root = plan.paths.repository_root
    script = root / "TRB/set/scripts/build_02_sample_level_features_source_aware.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    # Phase 7 needs Step02's label-independent static core and provenance only.
    # A one-kmer scaffold minimizes materialized precomputed k-mer width. Phase 6
    # ignores this vocabulary and rebuilds the configured Top-N fold-locally.
    common = [
        "--registry", str(resolved_paths["registry"]),
        "--plan", str(resolved_paths["phase2_plan"]),
        "--max-kmers", "1",
    ]
    train = _python_command(
        script,
        "--mode", "train",
        "--metadata", plan.train_metadata,
        "--output-dir", plan.paths.train_step02_dir,
        *common,
    )
    vocab = plan.paths.train_step02_dir / "02_3mer_vocabulary.csv"
    test = _python_command(
        script,
        "--mode", "test",
        "--metadata", plan.test_metadata,
        "--output-dir", plan.paths.test_step02_dir,
        "--vocabulary-file", vocab,
        *common,
    )
    return train, test


def ml_command(
    plan: OrchestratorPlan,
    *,
    config_path: Path,
    overwrite_ml: bool = False,
    dry_run: bool = False,
) -> List[str]:
    root = plan.paths.repository_root
    script = root / "TRB/set/scripts/run_06_unified_ml.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    command = _python_command(
        script,
        "--config", config_path,
        "--train-step02-dir", plan.paths.train_step02_dir,
        "--test-step02-dir", plan.paths.test_step02_dir,
        "--train-metadata", plan.train_metadata,
        "--test-metadata", plan.test_metadata,
        "--registry", plan.registry_path,
        "--output-dir", plan.paths.ml_output_dir,
        "--workers", plan.workers,
    )
    if plan.repeat_override is not None:
        command += ["--repeat", str(plan.repeat_override)]
    if dry_run:
        command.append("--dry-run")
    if overwrite_ml:
        command.append("--overwrite")
    return command


def run_checked(command: Sequence[str], *, cwd: Path, label: str) -> subprocess.CompletedProcess:
    result = subprocess.run(list(command), cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    return result


def write_orchestrator_manifest(
    plan: OrchestratorPlan,
    *,
    config_path: Path,
    source_action: str,
    step02_action: str,
    ml_action: str,
    commands: Mapping[str, Sequence[str]],
) -> None:
    plan.paths.experiment_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": plan.spec.base.experiment_id,
        "config": str(config_path),
        "algorithm": plan.spec.algorithm,
        "cohort_scope": plan.spec.base.cohort.scope,
        "repertoire_source": plan.source.as_dict(),
        "repeat_override": plan.repeat_override,
        "configured_repeats": plan.spec.cv.repeats,
        "workers": plan.workers,
        "paths": plan.paths.as_dict(),
        "actions": {
            "repertoire": source_action,
            "step02_static_core": step02_action,
            "unified_ml": ml_action,
        },
        "commands": {key: list(value) for key, value in commands.items()},
        "modeling_leakage_policy": {
            "precomputed_step02_kmer_used_for_ml": False,
            "kmer_vocabulary": "current_training_only_in_phase6",
            "enriched_dictionary": "current_training_reference; exact_LOO_for_training",
            "preprocessing": "current_training_only",
            "hyperparameter_and_threshold_selection": "inner_OOF_only",
        },
    }
    plan.paths.manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Phase-7 Experiment Orchestrator Summary",
        "",
        f"- Experiment: `{plan.spec.base.experiment_id}`",
        f"- Algorithm: `{plan.spec.algorithm}`",
        f"- Cohort scope: `{plan.spec.base.cohort.scope}`",
        f"- Repertoire source: `{plan.source.source_id}`",
        f"- Configured repeats: **{plan.spec.cv.repeats}**",
        f"- Executed repeat override: `{plan.repeat_override if plan.repeat_override is not None else 'none (full configured bank)'}`",
        f"- Workers: **{plan.workers}**",
        f"- Repertoire action: `{source_action}`",
        f"- Step02 static-core action: `{step02_action}`",
        f"- Unified ML action: `{ml_action}`",
        "",
        "## Leakage contract",
        "",
        "- Step02 cache supplies label-independent static core/provenance only.",
        "- Precomputed Step02 3-mer matrices are not consumed by Phase 6.",
        "- 3-mer vocabulary, enriched dictionary, preprocessing, tuning and threshold selection remain fold-local.",
        "",
    ]
    plan.paths.summary_path.write_text("\n".join(lines), encoding="utf-8")


def remove_step02_cache(path: Path, *, canonical_cache_root: Path) -> None:
    resolved = path.expanduser().resolve()
    allowed = canonical_cache_root.expanduser().resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise OrchestratorError(
            "refusing to rebuild a Step02 override outside the canonical cache root; delete/rebuild it explicitly: "
            + str(resolved)
        ) from exc
    if resolved.exists():
        shutil.rmtree(resolved)
