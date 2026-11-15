#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IGH V3 hierarchical five-algorithm checkpoint batch orchestration.

Commands:
    plan    Read-only selected workload and checkpoint counts.
    freeze  Fingerprint all 50 YAMLs, engine code, split/metadata/Step02 inputs,
            and every one of the 175 Top10000 repertoire files. No training.
    run     Validate frozen fingerprint, launch selected independent configs;
            each child delegates to the tested per-repeat checkpoint runner.
    status  Read-only fast checkpoint counts (marker-based, NOT deep validation).
    verify  Delegate full per-repeat validation to the checkpoint runner.
    collect Require all 50x100 checkpoints and deep validation; write
            aggregate metrics, per-config summaries and paired deltas.

Linux implementation: every config has a nonblocking flock lock, whose file
handle is inherited by the child process. A parent SIGKILL cannot release a
lock while that child is still running. Each job's stdout/stderr goes to a log.
No existing run outputs are overwritten. Default command is read-only plan.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

ALGOS = ("elastic_net", "ridge", "lasso", "linear_svm", "xgboost")
SCOPE_MODELS = {
    "total": ("M0", "M1", "M2", "M3"),
    "pbmc": ("M0", "M1", "M2"),
    "buffycoat": ("M0", "M1", "M2"),
}
SCOPE_COUNTS = {"total": 175, "pbmc": 113, "buffycoat": 62}
EXPECTED_HOLDOUT = {"total": 53, "pbmc": 34, "buffycoat": 19}
SPLIT_PREFIXES = {
    "total": "455eba17c74a",
    "pbmc": "4726d9b48301",
    "buffycoat": "da1e598559dc",
}
CONFIG_REL = Path("IGH/set/configs/experiments/igh_v3_hierarchical_5alg_top10000_v3_175")
REGISTRY_REL = Path("IGH/set/configs/feature_framework/igh_feature_registry_v3_175.yaml")
CHECKPOINT_REL = Path("IGH/set/scripts/checkpoint_igh_v3_hierarchical_5alg.py")
RUNNER_REL = Path("IGH/set/scripts/run_06_unified_ml.py")
FROZEN_NAME = "_FROZEN_INPUTS.json"
METRICS = ("roc_auc", "pr_auc", "accuracy", "sensitivity_recall", "specificity", "precision", "f1")
STATIC_GROUPS = (
    "tcr_diversity", "tcr_clonal_expansion", "tcr_aa_length",
    "tcr_aa_composition_unweighted", "tcr_aa_composition_weighted",
    "tcr_physicochemical_unweighted", "tcr_physicochemical_weighted",
)


def ensure(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(target: Path, payload: dict) -> None:
    tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def read_yaml(p: Path) -> dict:
    ensure(p.is_file(), f"Missing YAML: {p}")
    x = yaml.safe_load(p.read_text(encoding="utf8"))
    ensure(isinstance(x, dict), f"Invalid YAML document: {p}")
    return x


def all_configs(repo: Path) -> dict[str, dict]:
    folder = repo / CONFIG_REL
    ensure(folder.is_dir(), f"Missing 50-config directory: {folder}")
    expected = {f"{scope}_{model}_{alg}.yaml" for scope, models in SCOPE_MODELS.items()
                for model in models for alg in ALGOS}
    actual = {p.name for p in folder.glob("*.yaml")}
    ensure(actual == expected, f"Configuration filenames differ; missing={sorted(expected-actual)[:5]}, "
            f"extra={sorted(actual-expected)[:5]}")
    result = {}
    ids = set()
    for scope, models in SCOPE_MODELS.items():
        for model in models:
            for alg in ALGOS:
                key = f"{scope}_{model}_{alg}"
                path = folder / f"{key}.yaml"
                cfg = read_yaml(path)
                feat = cfg["features"]
                eid = cfg["experiment"]["id"]
                ensure(eid not in ids, f"Duplicate experiment ID {eid}")
                ids.add(eid)
                ensure(cfg["cohort"]["scope"] == scope, f"Scope mismatch: {key}")
                ensure(cfg["repertoire"] == {"mode": "topk", "top_k": 10000}, f"Repertoire mismatch: {key}")
                ensure(cfg["model"]["algorithm"] == alg, f"Algorithm mismatch: {key}")
                ensure(cfg["model"]["positive_label"] == "ILD", f"Positive label mismatch: {key}")
                ensure(cfg["validation"]["repeats"] == 100 and cfg["validation"]["folds"] == 5,
                       f"Repeat/fold mismatch: {key}")
                ensure(feat["clinical_groups"] == (["clinical_age_sex", "clinical_material"] if model == "M3"
                                                  else ["clinical_age_sex"]), f"Clinical mismatch {key}")
                ensure(tuple(feat["static_tcr_groups"]) == (STATIC_GROUPS if model != "M0" else ()),
                       f"Core83 group mismatch: {key}")
                ensure(feat["kmer"]["enabled"] == (model != "M0"), f"3mer mismatch: {key}")
                if model != "M0":
                    ensure(feat["kmer"]["representation"] == "both" and feat["kmer"]["top_n"] == 500,
                           f"Double 3mer mismatch: {key}")
                ensure(feat["public_reference"]["enabled"] == (model in {"M2", "M3"}),
                       f"Public mismatch: {key}")
                ensure(not feat["enriched_dictionary"]["enabled"], f"Legacy dictionary enabled: {key}")
                split = (repo / cfg["validation"]["split_assignments"]).resolve()
                ensure(split.is_file(), f"Missing split bank: {key} -> {split}")
                ensure(sha(split).startswith(SPLIT_PREFIXES[scope]), f"Frozen split differs: {scope}")
                result[key] = {"key": key, "scope": scope, "model": model, "algorithm": alg,
                               "path": path, "split": split}
    ensure(len(result) == 50, "Expected 50 configs")
    return result


def selected_configs(configs: dict, args) -> list[dict]:
    sel = []
    for conf in configs.values():
        if args.scope != "all" and conf["scope"] != args.scope:
            continue
        if args.model != "all" and conf["model"] != args.model:
            continue
        if args.algorithm != "all" and conf["algorithm"] != args.algorithm:
            continue
        sel.append(conf)
    ensure(sel, "No matching configurations")
    return sel


def paths_to_freeze(repo: Path, configs: dict) -> list[Path]:
    paths = set()
    for c in configs.values():
        paths.add(c["path"])
        paths.add(c["split"])
    paths.update({repo / REGISTRY_REL, repo / CHECKPOINT_REL, repo / RUNNER_REL,
                  repo / "IGH/set/scripts/orchestrate_igh_v3_hierarchical_5alg.py",
                  repo / "IGH/set/data_versions/igh_v3_175/result/01b_AA_clone_table_top10000/01b_top10000_summary.csv"})
    pkg = repo / "IGH/set/src/ra_ild_igh"
    ensure(pkg.is_dir(), f"Missing engine package: {pkg}")
    paths.update(pkg.rglob("*.py"))
    total_ids = set()
    all_scope_ids = {}
    for scope, n in SCOPE_COUNTS.items():
        data = repo / f"IGH/set/data_versions/igh_v3_175/ml_compat/{scope}"
        scope_ids = set()
        for name in ("metadata_train.csv", "metadata_test.csv"):
            p = data / name
            ensure(p.is_file(), f"Missing metadata: {p}")
            paths.add(p)
            with p.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    sid = row["libraryid"].strip()
                    ensure(sid and sid not in scope_ids, f"Duplicate/blank metadata ID {scope}/{sid}")
                    scope_ids.add(sid)
        ensure(len(scope_ids) == n, f"Wrong sample count {scope}: {len(scope_ids)} != {n}")
        all_scope_ids[scope] = scope_ids
        for sub in ("train_step02", "test_step02"):
            d = data / sub
            for name in ("02_core_sample_features.csv", "02_resolved_feature_source.json"):
                p = d / name
                if name == "02_resolved_feature_source.json" and not p.exists():
                    # The loader treats this file as optional; hash when present.
                    continue
                ensure(p.is_file(), f"Missing Step02 input: {p}")
                paths.add(p)
    ensure(all_scope_ids["total"] == all_scope_ids["pbmc"] | all_scope_ids["buffycoat"]
           and not all_scope_ids["pbmc"] & all_scope_ids["buffycoat"],
           "Cohort ID membership mismatch")
    registry = read_yaml(repo / REGISTRY_REL)
    source = registry["repertoire_sources"]["top10000"]
    folder = (repo / source["default_path"]).resolve()
    pattern = source["sample_filename_template"]
    for sid in sorted(all_scope_ids["total"]):
        paths.add(folder / pattern.format(sample_id=sid))
    return sorted({p.resolve() for p in paths})


def fingerprint(repo: Path, configs: dict) -> dict:
    paths = paths_to_freeze(repo, configs)
    files = {}
    for i, path in enumerate(paths, 1):
        ensure(path.is_file(), f"Missing frozen file: {path}")
        # Relative names ensure the same contract is stable within one repo.
        try:
            rel = str(path.relative_to(repo))
        except ValueError:
            # A symlinked input may resolve outside the repository.
            rel = "@external:" + str(path)
        files[rel] = {"sha256": sha(path), "bytes": path.stat().st_size}
        if i % 100 == 0:
            print(f"[HASH] {i}/{len(paths)} files", flush=True)
    packages = {}
    for modname in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "yaml"):
        mod = __import__(modname)
        packages[modname] = str(getattr(mod, "__version__", "unknown"))
    record = {"protocol": "IGH-V3-5ALG-100REPEAT-v1", "python": platform.python_version(),
              "packages": packages, "files": files}
    record["manifest_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return record


def validate_freeze(repo: Path, out: Path, configs: dict) -> dict:
    p = out / FROZEN_NAME
    ensure(p.is_file(), f"Frozen fingerprint absent: {p}; run --mode freeze first")
    prior = json.loads(p.read_text(encoding="utf8"))
    current = fingerprint(repo, configs)
    ensure(prior["manifest_sha256"] == current["manifest_sha256"],
           "Frozen sources/configs/environment have changed! STOP. Keep outputs separate and investigate.")
    print(f"[FREEZE PASS] {len(current['files'])} files SHA256, manifest={current['manifest_sha256']}")
    return prior


def fast_counts(folder: Path, start: int, end: int) -> tuple[int, int, int]:
    present = 0
    broken = 0
    for r in range(start, end+1):
        p = folder / f"repeat_{r:03d}"
        if not p.exists():
            continue
        if p.is_dir() and (p / "RUN_COMPLETE.json").is_file():
            present += 1
        else:
            broken += 1
    return present, end-start+1-present-broken, broken


def start_job(conf: dict, repo: Path, out: Path, start: int, end: int) -> dict:
    folder = out / conf["key"]
    folder.mkdir(parents=True, exist_ok=True)
    lockfile = folder / ".config.lock"
    fd = os.open(lockfile, os.O_CREAT | os.O_RDWR, 0o640)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Config already running: {conf['key']}") from exc
        logfolder = out / "_logs"
        logfolder.mkdir(exist_ok=True)
        log = logfolder / f"{conf['key']}_{timestamp()}_{os.getpid()}.log"
        cmd = [sys.executable, str(repo / CHECKPOINT_REL), "--repo", str(repo),
               "--config", str(conf["path"]), "--output-dir", str(folder),
               "--start", str(start), "--end", str(end), "--mode", "run"]
        env = os.environ.copy()
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            env[var] = "1"
        with log.open("x", encoding="utf8") as handle:
            handle.write("COMMAND: " + " ".join(cmd) + "\n")
            handle.flush()
            # Parent and child are in separate sessions. SIGINT/SSH disconnection
            # never launches the queued jobs; scheduler sends signals explicitly.
            # The child INHERITS the flock fd, so an abrupt parent SIGKILL cannot
            # release the per-config lock while the child keeps writing results.
            proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT, pass_fds=(fd,),
                                    start_new_session=True)
        return {"config": conf["key"], "proc": proc,
                "started": time.monotonic(), "log": str(log)}
    finally:
        os.close(fd)


def run_queue(configs: list[dict], repo: Path, out: Path,
              start: int, end: int, jobs: int) -> int:
    queued = list(configs)
    active: dict[int, dict] = {}
    successes = 0
    failures = []
    stop = {"requested": False}
    def on_stop(signum, _frame):
        stop["requested"] = True
        print(f"\n[INTERRUPT] signal={signum}; stopping active jobs and not launching queued jobs",
              flush=True)
    old_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for s in old_handlers:
        signal.signal(s, on_stop)
    try:
        while queued or active:
            while queued and len(active) < jobs and not stop["requested"]:
                c = queued.pop(0)
                try:
                    r = start_job(c, repo, out, start, end)
                    active[r["proc"].pid] = r
                    print(f"[JOB START] {r['config']} pid={r['proc'].pid} log={r['log']}", flush=True)
                except Exception as exc:
                    failures.append(c["key"])
                    print(f"[JOB START FAILED] {c['key']}: {exc}", file=sys.stderr, flush=True)
            for pid, r in list(active.items()):
                rc = r["proc"].poll()
                if rc is None:
                    continue
                elapsed = round(time.monotonic()-r["started"], 1)
                print(f"[JOB END] {r['config']} exit={rc} seconds={elapsed} log={r['log']}", flush=True)
                if rc == 0:
                    successes += 1
                else:
                    failures.append(r["config"])
                del active[pid]
            if stop["requested"]:
                for r in active.values():
                    try:
                        os.killpg(r["proc"].pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                deadline = time.monotonic()+10
                while active and time.monotonic() < deadline:
                    for pid, r in list(active.items()):
                        if r["proc"].poll() is not None:
                            del active[pid]
                    time.sleep(0.2)
                for r in active.values():
                    try:
                        os.killpg(r["proc"].pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                print(f"[STOPPED] {len(queued)} queued jobs were not launched", flush=True)
                return 130
            if active:
                time.sleep(1)
        print(f"[BATCH END] successful_configs={successes}, failed_configs={len(failures)}")
        if failures:
            print("[FAILED] " + ", ".join(failures[:50]), file=sys.stderr)
        return 2 if failures else 0
    finally:
        for sig, old in old_handlers.items():
            signal.signal(sig, old)

def deep_verify(config: dict, repo: Path, out: Path, start: int, end: int) -> None:
    cmd = [sys.executable, str(repo / CHECKPOINT_REL), "--repo", str(repo),
           "--config", str(config["path"]), "--output-dir", str(out / config["key"]),
           "--start", str(start), "--end", str(end), "--mode", "status"]
    process = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"DEEP VERIFY FAILED: {config['key']}\n{process.stdout}\n{process.stderr}")
    # checkpoint status mode returns 0 when missing; additionally require all done.
    expected = f"[STATUS] verified={end-start+1} missing=0 invalid=0"
    ensure(expected in process.stdout, f"Incomplete deep verification: {config['key']}\n{process.stdout}")
    print(f"[VERIFY PASS] {config['key']}: {end-start+1} repeats", flush=True)


def collect(repo: Path, out: Path, configs: dict, *, max_rows: int = 100) -> None:
    """Require all configs complete, then deep verify; write aggregate CSV files."""
    import numpy as np
    import pandas as pd

    for c in configs.values():
        deep_verify(c, repo, out, 1, max_rows)
    rows = []
    selected = []
    result_hash_rows = []
    output_names = (
        "06_outer_metrics.csv", "06_outer_predictions.csv.gz", "06_inner_tuning.csv.gz",
        "06_selected_hyperparameters.csv", "06_feature_explanations.csv.gz",
        "06_feature_fit_audit.csv.gz", "06_preprocessing_audit.csv.gz",
        "06_inner_assignment_audit.csv.gz", "06_run_manifest.json",
        "06_run_summary.md", "RUN_COMPLETE.json",
    )
    for c in configs.values():
        for r in range(1, max_rows+1):
            folder = out / c["key"] / f"repeat_{r:03d}"
            for name in output_names:
                result_hash_rows.append({"config_id": c["key"], "outer_repeat": r,
                                         "filename": name, "sha256": sha(folder / name)})
            m = pd.read_csv(folder / "06_outer_metrics.csv")
            ensure(len(m) == 1 and int(m.iloc[0]["outer_repeat"]) == r, f"Metric identity mismatch: {c['key']} {r}")
            row = m.iloc[0].to_dict()
            row.update({"scope": c["scope"], "model": c["model"], "config_id": c["key"]})
            rows.append(row)
            sel = pd.read_csv(folder / "06_selected_hyperparameters.csv")
            ensure(len(sel) == 1, f"Selected candidates mismatch: {c['key']} {r}")
            z = sel.iloc[0].to_dict()
            z.update({"scope": c["scope"], "model": c["model"], "algorithm": c["algorithm"],
                      "config_id": c["key"]})
            selected.append(z)
    frame = pd.DataFrame(rows)
    ensure(len(frame) == len(configs)*max_rows, "Unexpected collected row count")
    ensure(not frame.duplicated(["scope", "model", "algorithm", "outer_repeat"]).any(), "Duplicate outer repeat")
    overview = frame.groupby(["scope", "model", "algorithm"], sort=True).agg(
        n=("roc_auc", "count"), auc_mean=("roc_auc", "mean"), auc_median=("roc_auc", "median"),
        auc_std=("roc_auc", "std"), auc_q25=("roc_auc", lambda x: x.quantile(.25)),
        auc_q75=("roc_auc", lambda x: x.quantile(.75)),
        ap_mean=("pr_auc", "mean"), ap_median=("pr_auc", "median"),
        sensitivity_mean=("sensitivity_recall", "mean"),
        specificity_mean=("specificity", "mean"), f1_mean=("f1", "mean")
    ).reset_index()
    # Within each scope+algorithm, models reuse the same frozen repeated-holdout
    # assignments. Paired deltas describe results, not independent-replicate CI.
    pairs = []
    for scope, models in SCOPE_MODELS.items():
        for alg in ALGOS:
            sub = frame[(frame["scope"] == scope) & (frame["algorithm"] == alg)]
            pivot = sub.pivot(index="outer_repeat", columns="model", values="roc_auc")
            for a, b in zip(models[:-1], models[1:]):
                ensure(a in pivot and b in pivot, f"Missing model comparison {scope}/{alg}/{a}-{b}")
                vals = (pivot[b] - pivot[a]).dropna()
                ensure(len(vals) == max_rows, f"Incomplete paired comparison {scope}/{alg}/{a}-{b}")
                pairs.append({"scope": scope, "algorithm": alg,
                              "comparison": f"{b}-{a}", "n_repeats": len(vals),
                              "delta_auc_mean": float(vals.mean()),
                              "delta_auc_median": float(vals.median()),
                              "delta_auc_q25": float(vals.quantile(.25)),
                              "delta_auc_q75": float(vals.quantile(.75))})
    result_hashes = pd.DataFrame(result_hash_rows)
    output_hash_digest = hashlib.sha256(result_hashes.to_csv(index=False).encode("utf-8")).hexdigest()
    dest = out / "_summary"
    existing = dest / "COLLECTION_MANIFEST.json"
    if existing.is_file():
        old_manifest = json.loads(existing.read_text(encoding="utf8"))
        ensure(old_manifest.get("outputs_sha256_digest") == output_hash_digest,
               "Previously collected output checksums differ. STOP and investigate; never silently rewrite.")
    dest.mkdir(exist_ok=True)
    for name, data in (("outputs_sha256.csv", result_hashes),
                       ("outer_metrics_all.csv", frame),
                       ("selected_candidates_all.csv", pd.DataFrame(selected)),
                       ("config_summary.csv", overview),
                       ("paired_auc_deltas.csv", pd.DataFrame(pairs))):
        target = dest / name
        tmp = dest / f".{name}.writing.{os.getpid()}"
        data.to_csv(tmp, index=False)
        os.replace(tmp, target)
        print(f"[COLLECT] {target}")
    atomic_json(dest / "COLLECTION_MANIFEST.json", {
        "created_utc": timestamp(), "configs": len(configs), "repeats_per_config": max_rows,
        "outer_rows": len(frame), "paired_comparisons": len(pairs),
        "output_files_hashed": len(result_hash_rows),
        "outputs_sha256_digest": output_hash_digest,
        "note": "Repeated holdout estimates are correlated; do not treat repeats as independent samples.",
    })
    print(f"[COLLECT PASS] {len(frame)} outer evaluations, {len(pairs)} paired model comparisons")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--mode", choices=("plan", "freeze", "run", "status", "verify", "collect"), default="plan")
    p.add_argument("--scope", choices=("all", *SCOPE_MODELS), default="all")
    p.add_argument("--model", choices=("all", "M0", "M1", "M2", "M3"), default="all")
    p.add_argument("--algorithm", choices=("all", *ALGOS), default="all")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=100)
    p.add_argument("--jobs", type=int, default=1,
                   help="Max concurrent config processes. Start at 1; benchmark 2 later.")
    a = p.parse_args()
    if not (1 <= a.start <= a.end <= 100):
        p.error("require 1 <= start <= end <= 100")
    if not (1 <= a.jobs <= 8):
        p.error("require 1 <= jobs <= 8")
    repo = a.repo.expanduser().resolve()
    out = a.output_root.expanduser().resolve()
    ensure((repo / ".git").exists(), "Not a Git repo")
    ensure(not out.is_relative_to(repo), "Output root MUST be outside repository")
    cfgs = all_configs(repo)
    selected = selected_configs(cfgs, a)
    print(f"[BATCH] mode={a.mode}, configs={len(selected)}/50, repeats={a.start}..{a.end}, jobs={a.jobs}")
    print(f"[OUTPUT] {out}")
    if a.mode in {"plan", "status"}:
        grand = Counter()
        for c in selected:
            present, missing, broken = fast_counts(out / c["key"], a.start, a.end)
            grand.update({"present": present, "missing": missing, "broken": broken})
            print(f"[CONFIG] {c['key']}: marker_present={present} missing={missing} broken={broken}")
        print(f"[TOTAL] {dict(grand)}; marker_present is NOT deep verification")
        return 0 if grand["broken"] == 0 else 2
    if a.mode == "freeze":
        ensure(a.scope == a.model == a.algorithm == "all", "Freeze requires entire 50-config plan")
        if out.exists() and not (out / FROZEN_NAME).is_file():
            existing_runs = [c["key"] for c in cfgs.values() if (out / c["key"]).exists()]
            ensure(not existing_runs, "Cannot retroactively freeze existing results: " + ", ".join(existing_runs[:5]))
        out.mkdir(parents=True, exist_ok=True)
        snapshot = fingerprint(repo, cfgs)
        path = out / FROZEN_NAME
        if path.exists():
            ensure(json.loads(path.read_text(encoding="utf8")) == snapshot,
                   "Existing frozen snapshot differs; do not overwrite")
            print(f"[FREEZE ALREADY PRESENT] {snapshot['manifest_sha256']}")
        else:
            atomic_json(path, snapshot)
            print(f"[FREEZE PASS] {len(snapshot['files'])} SHA256-checked files; "
                  f"manifest={snapshot['manifest_sha256']}")
        return 0
    validate_freeze(repo, out, cfgs)
    if a.mode == "verify":
        for c in selected:
            deep_verify(c, repo, out, a.start, a.end)
        return 0
    if a.mode == "collect":
        ensure(a.scope == a.model == a.algorithm == "all", "Collect requires all 50 configs")
        ensure(a.start == 1 and a.end == 100, "Collect requires repeats 1..100")
        collect(repo, out, cfgs)
        return 0
    ensure(a.mode == "run", f"Unexpected mode {a.mode}")
    print("[RUN] Checkpoint outputs are per-config, per-repeat. "
          "The child processes are CPU-bound; no Python threads run models.")
    return run_queue(selected, repo, out, a.start, a.end, a.jobs)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)
