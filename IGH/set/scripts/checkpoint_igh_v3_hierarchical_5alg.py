#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpointed IGH V3 Phase-6 nested CV, one configuration at a time.

Uses the existing ra_ild_igh.unified_cv.run_repeat; does not implement new
features, hyperparameter selection, thresholds, or models. Data are loaded
once per invocation and each completed repeat is atomically committed.

For a reproducible full run, freeze source/configuration/input snapshots first.
This runner fingerprints key input and code files, but does not hash the
contents of every Top10000 per-sample repertoire file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_VERSION = "0.1.0-checkpoint-smoke"
OUTPUT_NAMES = (
    "metrics", "predictions", "inner_tuning", "selected", "explanations",
    "feature_audit", "preprocessing", "inner_assignments", "manifest", "summary",
)
EXPECTED_N = {"total": (175, 53), "pbmc": (113, 34), "buffycoat": (62, 19)}
SOURCE_PATHS = [
    "IGH/set/src/ra_ild_igh/experiment_spec.py",
    "IGH/set/src/ra_ild_igh/unified_experiment.py",
    "IGH/set/src/ra_ild_igh/unified_modeling.py",
    "IGH/set/src/ra_ild_igh/unified_cv.py",
    "IGH/set/src/ra_ild_igh/fold_features.py",
    "IGH/set/src/ra_ild_igh/v3_public_adapter.py",
    "IGH/set/src/ra_ild_igh/dynamic_repertoire.py",
    "IGH/set/scripts/run_06_unified_ml.py",
    "IGH/set/configs/feature_framework/igh_feature_registry_v3_175.yaml",
    "IGH/set/data_versions/igh_v3_175/result/01b_AA_clone_table_top10000/01b_top10000_summary.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".writing")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def resolve(value: str, root: Path) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def build_contract(root: Path, config: Path, spec, registry: Path, split: Path, data: Path) -> dict:
    paths = [root / rel for rel in SOURCE_PATHS]
    paths.extend([
        config, registry, split,
        data / "metadata_train.csv", data / "metadata_test.csv",
        data / "train_step02/02_core_sample_features.csv",
        data / "test_step02/02_core_sample_features.csv",
    ])
    # A file may be listed twice (registry); hash once by canonical path.
    checksums = {}
    for p in sorted(set(p.resolve() for p in paths)):
        if not p.is_file():
            raise FileNotFoundError(f"Required frozen input absent: {p}")
        checksums[str(p)] = sha256(p)
    payload = {
        "runner_version": SCRIPT_VERSION,
        "experiment_id": spec.base.experiment_id,
        "algorithm": spec.algorithm,
        "scope": spec.base.cohort.scope,
        "config": str(config),
        "hashes": checksums,
        "note": "Per-sample repertoire file contents are NOT individually hashed.",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    payload["contract_sha256"] = digest
    return payload


def assert_contract(out: Path, current: dict, *, run: bool) -> None:
    target = out / "_run_contract.json"
    if target.is_file():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old != current:
            raise RuntimeError(
                "Input/code/config contract changed since previous repeats. "
                "STOP: use a fresh --output-dir or restore the frozen inputs."
            )
    elif run:
        # Exclusive creation, never overwrite an existing contract.
        with target.open("x", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")


def validate_repeat(folder: Path, *, repeat: int, holdout: int,
                    expected_candidates: int, contract_sha: str, expected_outputs) -> None:
    files = expected_outputs(folder)
    for name in OUTPUT_NAMES:
        if name not in files or not files[name].is_file() or files[name].stat().st_size == 0:
            raise RuntimeError(f"Missing/nonempty output {folder}: {name}")
    marker = folder / "RUN_COMPLETE.json"
    if not marker.is_file():
        raise RuntimeError(f"Missing completion marker: {marker}")
    m = json.loads(marker.read_text(encoding="utf-8"))
    if (int(m["repeat"]) != repeat or m["contract_sha256"] != contract_sha):
        raise RuntimeError(f"Repeat/contract mismatch: {marker}")
    metrics = pd.read_csv(files["metrics"])
    preds = pd.read_csv(files["predictions"])
    tuning = pd.read_csv(files["inner_tuning"])
    audit = pd.read_csv(files["feature_audit"])
    inner = pd.read_csv(files["inner_assignments"])
    if len(metrics) != 1 or int(metrics["outer_repeat"].iloc[0]) != repeat:
        raise RuntimeError(f"Bad metrics rows: {folder}")
    if (len(preds) != holdout or preds["sample_id"].nunique() != holdout
            or set(preds["outer_repeat"]) != {repeat}):
        raise RuntimeError(f"Bad holdout predictions: {folder}")
    if len(tuning) != expected_candidates:
        raise RuntimeError(f"Bad candidate count: {folder}")
    if tuning["selected"].astype(str).str.lower().eq("true").sum() != 1:
        raise RuntimeError(f"No unique selected candidate: {folder}")
    if not metrics["fit_converged"].astype(str).str.lower().eq("true").all():
        raise RuntimeError(f"Final fit nonconvergence: {folder}")
    if len(inner) != 5 or len(set(inner["inner_fold"].astype(int))) != 5:
        raise RuntimeError(f"Incomplete inner folds: {folder}")
    if set(audit["stage"]) != {"inner", "outer_final"}:
        raise RuntimeError(f"Incomplete feature audit: {folder}")
    numeric = metrics[["roc_auc", "pr_auc", "accuracy", "f1"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"Nonfinite outer metrics: {folder}")


def write_result(tempdir: Path, finaldir: Path, result, spec, inputs, candidate_count: int,
                 split_path: Path, registry: Path, data: Path, contract: dict,
                 elapsed: float, expected_outputs, summary_markdown) -> None:
    file_paths = expected_outputs(tempdir)
    payloads = {
        "metrics": result.metrics,
        "predictions": result.predictions,
        "inner_tuning": result.inner_tuning,
        "selected": result.selected_hyperparameters,
        "explanations": result.explanations,
        "feature_audit": result.feature_audit,
        "preprocessing": result.preprocessing_audit,
        "inner_assignments": result.inner_assignment_audit,
    }
    zipped = {"predictions", "inner_tuning", "explanations", "feature_audit", "preprocessing", "inner_assignments"}
    for key, frame in payloads.items():
        frame.to_csv(file_paths[key], index=False, compression="gzip" if key in zipped else None)
    manifest = {
        "runner_version": SCRIPT_VERSION,
        "source_runner": "run_06_unified_ml.py-compatible Phase-6 unified_cv.run_repeat",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": contract["contract_sha256"],
        "resolved_experiment": spec.as_dict(),
        "inputs": {
            "train_metadata": str(data / "metadata_train.csv"),
            "test_metadata": str(data / "metadata_test.csv"),
            "train_step02_dir": str(data / "train_step02"),
            "test_step02_dir": str(data / "test_step02"),
            "registry": str(registry),
            "split_assignments": str(split_path),
            "repertoire_source": inputs.source.as_dict(),
        },
        "execution": {
            "repeat_ids": [int(result.repeat)],
            "repeat_count": 1,
            "candidate_count": candidate_count,
            "elapsed_seconds": elapsed,
        },
        "outputs": {name: str(path) for name, path in expected_outputs(finaldir).items()},
    }
    file_paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    file_paths["summary"].write_text(
        summary_markdown(spec, result.metrics, candidate_count, [int(result.repeat)]),
        encoding="utf-8",
    )
    json_dump_atomic(tempdir / "RUN_COMPLETE.json", {
        "repeat": int(result.repeat),
        "contract_sha256": contract["contract_sha256"],
        "elapsed_seconds": elapsed,
        "created_at_utc": manifest["created_at_utc"],
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True,
                    help="Dedicated output root for one config, outside tracked source/config dirs")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=1)
    ap.add_argument("--mode", choices=("plan", "run", "status"), default="plan")
    args = ap.parse_args()

    if not (1 <= args.start <= args.end <= 100):
        ap.error("require 1 <= --start <= --end <= 100")
    repo = Path(args.repo).expanduser().resolve()
    config = resolve(args.config, repo)
    out = resolve(args.output_dir, repo)
    if not (repo / ".git").is_dir():
        raise RuntimeError("--repo must point to a local git working tree")
    src = repo / "IGH/set/src"
    scripts = repo / "IGH/set/scripts"
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(scripts))

    from ra_ild_igh.fold_features import load_fold_feature_inputs
    from ra_ild_igh.unified_cv import load_repeated_holdout_bank, run_repeat
    from ra_ild_igh.unified_experiment import auto_split_assignment_path, load_unified_experiment
    from ra_ild_igh.unified_modeling import build_candidates
    from run_06_unified_ml import expected_outputs, summary_markdown

    spec = load_unified_experiment(config)
    scope = spec.base.cohort.scope
    if scope not in EXPECTED_N:
        raise RuntimeError(f"Unsupported scope: {scope}")
    expected_n, holdout = EXPECTED_N[scope]
    if spec.cv.repeats != 100:
        raise RuntimeError("STOP: frozen protocol requires 100 repeats")
    if spec.repertoire_source_id != "top10000":
        raise RuntimeError("STOP: frozen protocol requires top10000 repertoire")
    data = repo / f"IGH/set/data_versions/igh_v3_175/ml_compat/{scope}"
    registry = repo / "IGH/set/configs/feature_framework/igh_feature_registry_v3_175.yaml"
    split_path = auto_split_assignment_path(spec, repo)
    contract = build_contract(repo, config, spec, registry, split_path, data)
    if out == repo or str(out).startswith(str(repo / "IGH/set/configs") + os.sep):
        raise RuntimeError("Output dir cannot be repo root or inside configs")

    print(f"[CONFIG] {config}")
    print(f"[SCOPE/ALG] {scope} / {spec.algorithm}")
    print(f"[CONTRACT] {contract['contract_sha256']}")
    print(f"[OUTPUT] {out}")
    print(f"[REPEATS] {args.start}..{args.end} mode={args.mode}")
    if out.is_dir():
        assert_contract(out, contract, run=False)
    if args.mode == "plan":
        print("[PLAN ONLY] No data loaded and no outputs written")
        return 0
    if args.mode == "run":
        out.mkdir(parents=True, exist_ok=True)
        assert_contract(out, contract, run=True)
    candidate_count = len(build_candidates(spec))
    if args.mode == "status":
        good, missing, bad = [], [], []
        for r in range(args.start, args.end + 1):
            folder = out / f"repeat_{r:03d}"
            if not folder.exists():
                missing.append(r)
                continue
            try:
                validate_repeat(folder, repeat=r, holdout=holdout,
                                expected_candidates=candidate_count,
                                contract_sha=contract["contract_sha256"],
                                expected_outputs=expected_outputs)
                good.append(r)
            except Exception as exc:
                bad.append((r, str(exc)))
        print(f"[STATUS] verified={len(good)} missing={len(missing)} invalid={len(bad)}")
        for row in bad[:10]:
            print(f"[INVALID] repeat {row[0]}: {row[1]}")
        return 2 if bad else 0

    pending = []
    for r in range(args.start, args.end + 1):
        folder = out / f"repeat_{r:03d}"
        if folder.exists():
            validate_repeat(folder, repeat=r, holdout=holdout,
                            expected_candidates=candidate_count,
                            contract_sha=contract["contract_sha256"],
                            expected_outputs=expected_outputs)
            print(f"[SKIP PASS] repeat={r:03d}: verified complete", flush=True)
        else:
            pending.append(r)
    if not pending:
        print("[DONE] All requested repeats already completed and validated")
        return 0

    # Load once per configuration; checkpoint each repeat immediately afterwards.
    inputs = load_fold_feature_inputs(
        spec,
        repository_root=repo,
        train_metadata=data / "metadata_train.csv",
        test_metadata=data / "metadata_test.csv",
        train_step02_dir=data / "train_step02",
        test_step02_dir=data / "test_step02",
        registry_path=registry,
        validate_source_files=True,
    )
    if len(inputs.sample_ids) != expected_n:
        raise RuntimeError(f"Expected {expected_n} samples but found {len(inputs.sample_ids)}")
    bank = load_repeated_holdout_bank(split_path, inputs.sample_ids, requested_repeats=100)

    for r in pending:
        finaldir = out / f"repeat_{r:03d}"
        # Atomic checkpoint: never expose a partially-written repeat as complete.
        tempdir = Path(tempfile.mkdtemp(prefix=f".repeat_{r:03d}_INCOMPLETE_", dir=out))
        print(f"[START] repeat={r:03d} temporary={tempdir}", flush=True)
        start = time.perf_counter()
        try:
            result = run_repeat(inputs, spec, bank, repeat=r)
            elapsed = time.perf_counter() - start
            write_result(tempdir, finaldir, result, spec, inputs, candidate_count,
                         split_path, registry, data, contract, elapsed,
                         expected_outputs, summary_markdown)
            validate_repeat(tempdir, repeat=r, holdout=holdout,
                            expected_candidates=candidate_count,
                            contract_sha=contract["contract_sha256"],
                            expected_outputs=expected_outputs)
            # Rename only after validation; fail if the target appeared unexpectedly.
            if finaldir.exists():
                raise RuntimeError(f"Checkpoint target already exists: {finaldir}")
            tempdir.rename(finaldir)
            print(f"[REPEAT PASS] {r:03d} elapsed_s={elapsed:.1f} stored={finaldir}", flush=True)
        except BaseException:
            print(f"[STOP] repeat {r:03d} failed; diagnostic directory preserved: {tempdir}",
                  file=sys.stderr, flush=True)
            raise
    print(f"[DONE] {len(pending)} new repeats completed; no official 100-repeat summary generated")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
