#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe and resumable (per experiment) runner for IGH v3_175 Layer 1.

No model runs without --execute. Existing outputs are NEVER overwritten/deleted.
A PASS is inferred from validated model artifacts, not a stale status file.
The 100 repeats for one config are one atomic unit: no mid-config checkpointing.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
VER = REPO / "IGH/set/data_versions/igh_v3_175"
CFG_ROOT = REPO / "IGH/set/configs/experiments"
SPLIT_ROOT = REPO / "IGH/set/split_sets"
REGISTRY = REPO / "IGH/set/configs/feature_framework/igh_feature_registry_v3_175.yaml"
MODEL_SCRIPT = REPO / "IGH/set/scripts/run_06_unified_ml.py"
MANAGEMENT = VER / "runs/layer1_batch_v3_175"

SCOPES = {
    "total": (
        175, 122, 53,
        "455eba17c74a86667fd2939dc8645ac1c50f0cdb25053ab951dccad0ab8f1788"
    ),
    "pbmc": (
        113, 79, 34,
        "4726d9b48301c4cd535369f9b39f6bdf3c6247c0722b9c9663385a6c041a68b8"
    ),
    "buffycoat": (
        62, 43, 19,
        "da1e598559dc734c0e0296bd1f99620351c2d81e903100c85fda23b670047c05"
    ),
}

GROUPS = (
    "tcr_diversity",
    "tcr_clonal_expansion",
    "tcr_aa_length",
    "tcr_aa_composition_unweighted",
    "tcr_aa_composition_weighted",
    "tcr_physicochemical_unweighted",
    "tcr_physicochemical_weighted",
)
REQUIRED_OUTPUTS = (
    "06_outer_metrics.csv",
    "06_outer_predictions.csv.gz",
    "06_inner_tuning.csv.gz",
    "06_selected_hyperparameters.csv",
    "06_feature_explanations.csv.gz",
    "06_feature_fit_audit.csv.gz",
    "06_preprocessing_audit.csv.gz",
    "06_inner_assignment_audit.csv.gz",
    "06_run_manifest.json",
    "06_run_summary.md",
)
STOP = False
STOP_AT = None
ACTIVE = {}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_ok(condition, message):
    if not condition:
        raise ValueError(message)


def split_path(scope):
    name = "igh_ra_ild_repeat100_v3_175" if scope == "total" else f"igh_ra_ild_{scope}_repeat100_v3_175"
    return SPLIT_ROOT / name / "repeated_holdout_assignments.csv"


def compat_dir(scope):
    return VER / "ml_compat" / scope


def config_dir(scope):
    return CFG_ROOT / f"igh_layer1_{scope}_top10000_v3_175"


def output_dir(scope, stem):
    return VER / "runs" / f"layer1_{scope}_top10000" / stem / "repeat100"


def status_path(scope, stem):
    return MANAGEMENT / "status" / scope / f"{stem}.json"


def log_path(scope, stem):
    return MANAGEMENT / "logs" / scope / f"{stem}.log"


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def checked_integer_series(frame, column):
    return pd.to_numeric(frame[column], errors="raise").astype(int)


def preflight_scope(scope):
    n, train_n, holdout_n, expected_sha = SCOPES[scope]
    bank = split_path(scope)
    assert_ok(bank.is_file(), f"frozen bank missing: {bank}")
    assert_ok(sha256(bank) == expected_sha, f"STOP: frozen bank SHA256 mismatch: {bank}")
    marker = bank.parent / "SPLITS_FROZEN.json"
    assert_ok(marker.is_file(), f"frozen marker missing: {marker}")
    freeze = json.loads(marker.read_text(encoding="utf-8"))
    assert_ok(freeze.get("status") == "FROZEN", f"not frozen: {marker}")
    assert_ok(freeze.get("files", {}).get("assignments", {}).get("sha256") == expected_sha,
              f"frozen marker SHA does not agree: {marker}")

    if scope == "total":
        meta_path = VER / "metadata.csv"
    else:
        meta_path = VER / f"metadata_{scope}.csv"
    metadata = pd.read_csv(meta_path, dtype=str)
    metadata["libraryid"] = metadata["libraryid"].str.strip()
    assert_ok(len(metadata) == n and metadata["libraryid"].is_unique, f"metadata count or unique ID failed for {scope}")
    ids = set(metadata["libraryid"])
    assert_ok(metadata["cohort"].isin(["RA", "ILD"]).all(), f"cohort invalid: {scope}")

    df = pd.read_csv(bank, dtype={"sample_id": str})
    assert_ok({"repeat_index", "sample_id", "role"}.issubset(df.columns), f"split columns invalid: {scope}")
    df["repeat_index"] = checked_integer_series(df, "repeat_index")
    assert_ok(set(df["repeat_index"]) == set(range(1, 101)), f"not exact 100 repeats: {scope}")
    by_repeat = {}
    first_roles = None
    for rep, part in df.groupby("repeat_index", sort=True):
        assert_ok(len(part) == n and part["sample_id"].is_unique, f"repeat size/unique mismatch {scope}:{rep}")
        assert_ok(set(part["sample_id"]) == ids, f"sample coverage mismatch {scope}:{rep}")
        counts = part["role"].value_counts().to_dict()
        assert_ok(counts == {"train": train_n, "holdout": holdout_n}, f"roles mismatch {scope}:{rep}: {counts}")
        by_repeat[int(rep)] = set(part.loc[part["role"] == "holdout", "sample_id"].astype(str))
        if rep == 1:
            first_roles = {role: set(part.loc[part["role"] == role, "sample_id"].astype(str))
                           for role in ("train", "holdout")}
        if scope == "buffycoat":
            assert_ok("MGI260202N01-31" not in by_repeat[int(rep)], "singleton leaked to holdout")

    compat = compat_dir(scope)
    core_sets = {}
    for label, role in (("train", "train"), ("test", "holdout")):
        mpath = compat / f"metadata_{label}.csv"
        spath = compat / f"{label}_step02/02_core_sample_features.csv"
        jpath = compat / f"{label}_step02/02_resolved_feature_source.json"
        for path in (mpath, spath, jpath):
            assert_ok(path.is_file(), f"compat missing: {path}")
        mini_meta = pd.read_csv(mpath, dtype=str)
        mini_core = pd.read_csv(spath, dtype={"sample_id": str})
        m_ids = set(mini_meta["libraryid"].str.strip())
        c_ids = set(mini_core["sample_id"].str.strip())
        assert_ok(m_ids == c_ids == first_roles[role], f"compat partition mismatch {scope}:{label}")
        assert_ok(mini_core.shape[1] == 97, f"core feature count not 96: {scope}:{label}")
        payload = json.loads(jpath.read_text(encoding="utf-8"))
        assert_ok(payload.get("repertoire_input", {}).get("source_id") == "top10000",
                  f"source not top10000: {scope}:{label}")
        source_input = payload.get("repertoire_input", {})
        expected_aa = VER / "result/01b_AA_clone_table_top10000"
        expected_summary = expected_aa / "01b_top10000_summary.csv"
        assert_ok(
            Path(source_input.get("aa_dir", "")).resolve()
            == expected_aa.resolve(),
            f"wrong V3 AA source: {scope}:{label}"
        )
        assert_ok(
            Path(source_input.get("summary_path", "")).resolve()
            == expected_summary.resolve(),
            f"wrong V3 source summary: {scope}:{label}"
        )
        core_sets[label] = c_ids
    assert_ok(core_sets["train"].isdisjoint(core_sets["test"]), f"compat overlap: {scope}")

    paths = sorted(config_dir(scope).glob("S*.yaml"))
    assert_ok(len(paths) == 127, f"expected 127 configs for {scope}, got {len(paths)}")
    combos, stems, jobs = set(), set(), []
    for path in paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        stem = path.stem
        assert_ok(stem not in stems, f"duplicate config stem: {stem}")
        stems.add(stem)
        assert_ok(cfg["experiment"]["receptor"] == "IGH", f"receptor wrong: {path}")
        assert_ok(cfg["cohort"]["scope"] == scope, f"scope wrong: {path}")
        assert_ok(cfg["experiment"]["id"] == f"igh_layer1_{scope}_top10000_{stem}_v3_175",
                  f"id not canonical: {path}")
        assert_ok(cfg["repertoire"]["mode"] == "topk" and cfg["repertoire"]["top_k"] == 10000,
                  f"source not top10000: {path}")
        f = cfg["features"]
        combo = tuple(f["static_tcr_groups"])
        assert_ok(1 <= len(combo) <= 7 and len(combo) == len(set(combo)) and set(combo).issubset(GROUPS),
                  f"invalid static combo: {path}")
        assert_ok(combo not in combos, f"duplicate combination: {path}")
        combos.add(combo)
        assert_ok(f["clinical_groups"] == [] and not f["kmer"]["enabled"]
                  and not f["enriched_dictionary"]["enabled"], f"Layer1 flags invalid: {path}")
        assert_ok(cfg["model"]["algorithm"] == "elastic_net" and cfg["model"]["positive_label"] == "ILD",
                  f"model not expected: {path}")
        assert_ok(cfg["validation"]["folds"] == 5 and cfg["validation"]["repeats"] == 100,
                  f"CV setting mismatch: {path}")
        configured_path = Path(cfg["validation"]["split_assignments"])
        if not configured_path.is_absolute():
            configured_path = REPO / configured_path
        assert_ok(configured_path.resolve() == bank.resolve(), f"config split mismatch: {path}")
        grid = cfg["model"]["parameters"]["elastic_net"]
        assert_ok(len(grid["alpha_grid"]) * len(grid["lambda_grid"]) == 24,
                  f"expected 24 candidate grid: {path}")
        jobs.append((scope, stem, path))
    assert_ok(len(combos) == 127, f"not 127 distinct combos: {scope}")
    return {"metadata": metadata.set_index("libraryid"), "holdouts": by_repeat,
            "jobs": jobs, "train_n": train_n, "holdout_n": holdout_n,
            "split_path": bank, "split_sha": expected_sha}


def result_check(scope, stem, contract):
    path = output_dir(scope, stem)
    if not path.is_dir():
        return False, "result directory absent"
    try:
        absent = [name for name in REQUIRED_OUTPUTS if not (path / name).is_file()]
        assert_ok(not absent, f"missing outputs: {absent}")
        metrics = pd.read_csv(path / "06_outer_metrics.csv")
        pred = pd.read_csv(path / "06_outer_predictions.csv.gz", dtype={"sample_id": str})
        tuning = pd.read_csv(path / "06_inner_tuning.csv.gz")
        selected = pd.read_csv(path / "06_selected_hyperparameters.csv")
        manifest = json.loads((path / "06_run_manifest.json").read_text(encoding="utf-8"))
        assert_ok(len(metrics) == len(selected) == 100, "metrics/selected row counts not 100")
        assert_ok(len(pred) == 100 * contract["holdout_n"], "prediction count differs from 100*holdout")
        assert_ok(len(tuning) == 2400, "inner tuning rows not 100*24")
        for frame, name in ((metrics, "metrics"), (selected, "selected"), (pred, "predictions")):
            reps = checked_integer_series(frame, "outer_repeat")
            assert_ok(set(reps) == set(range(1, 101)), f"{name} repeat coverage is not 1..100")
        assert_ok((checked_integer_series(metrics, "n_outer_train") == contract["train_n"]).all(), "train sizes wrong")
        assert_ok((checked_integer_series(metrics, "n_outer_holdout") == contract["holdout_n"]).all(), "holdout sizes wrong")
        assert_ok(metrics["fit_converged"].astype(str).str.lower().eq("true").all(), "nonconverged final fit")
        for rep, part in pred.groupby("outer_repeat"):
            sample_ids = part["sample_id"].astype(str)
            assert_ok(sample_ids.is_unique and set(sample_ids) == contract["holdouts"][int(rep)],
                      f"prediction sample IDs mismatch frozen holdout at repeat={rep}")
            assert_ok((part["true_cohort"].astype(str) == contract["metadata"].loc[sample_ids, "cohort"].to_numpy()).all(),
                      f"true cohort mismatch at repeat={rep}")
        execution = manifest.get("execution", {})
        assert_ok(execution.get("repeat_count") == 100 and execution.get("candidate_count") == 24,
                  "manifest repeats/candidates mismatch")
        recorded = manifest.get("inputs", {}).get("split_assignments", "")
        assert_ok(Path(recorded).resolve() == contract["split_path"].resolve(), "manifest frozen split path mismatch")
        return True, "100 repeats, predictions, tuning, frozen IDs, manifest PASS"
    except Exception as exc:
        return False, f"validation error: {exc}"


def request_stop(signum, frame):
    global STOP, STOP_AT
    if STOP:
        return
    STOP = True
    STOP_AT = time.monotonic()
    print(f"\nSIGNAL {signum}: stop launching; terminate active process groups", flush=True)
    for job in list(ACTIVE.values()):
        proc = job["proc"]
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def launch_job(job, workers):
    scope, stem, path = job
    compat = compat_dir(scope)
    out = output_dir(scope, stem)
    log = log_path(scope, stem)
    stat = status_path(scope, stem)
    log.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(MODEL_SCRIPT),
           "--config", str(path),
           "--train-metadata", str(compat / "metadata_train.csv"),
           "--test-metadata", str(compat / "metadata_test.csv"),
           "--train-step02-dir", str(compat / "train_step02"),
           "--test-step02-dir", str(compat / "test_step02"),
           "--registry", str(REGISTRY),
           "--split-assignments", str(split_path(scope)),
           "--output-dir", str(out),
           "--workers", str(workers)]
    env = os.environ.copy()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[k] = "1"
    with log.open("w", encoding="utf-8") as stream:
        stream.write(f"Started UTC: {utcnow()}\nCommand: {json.dumps(cmd, ensure_ascii=False)}\n\n")
        stream.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
    atomic_json(stat, {"state": "RUNNING", "scope": scope, "name": stem, "pid": proc.pid,
                       "started_utc": utcnow(), "output": str(out), "log": str(log)})
    print(f"START {scope}/{stem} pid={proc.pid} workers={workers} log={log}", flush=True)
    return {"proc": proc, "scope": scope, "stem": stem, "status": stat, "log": log, "started": time.monotonic()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--scope", choices=["all", *SCOPES], default="all", help="Scope(s) to run")
    ap.add_argument("--name", action="append", default=[], help="Only selected config stems; repeatable, e.g. --name S001_D")
    ap.add_argument("--limit", type=int, default=1, help="Max NEW experiments to start; 0=all eligible")
    ap.add_argument("--max-concurrent", type=int, default=1, help="Independent experiment processes at once")
    ap.add_argument("--workers-per-experiment", type=int, default=1, help="Repeat workers inside each experiment")
    ap.add_argument("--execute", action="store_true", help="Actually run training (otherwise preflight and plan only)")
    args = ap.parse_args()
    assert_ok(args.limit >= 0, "--limit must be >=0")
    assert_ok(1 <= args.max_concurrent <= 6, "--max-concurrent must be 1..6")
    assert_ok(1 <= args.workers_per_experiment <= 10, "--workers-per-experiment must be 1..10")
    assert_ok(args.max_concurrent * args.workers_per_experiment <= 50,
              "total model worker slots must be <=50")
    assert_ok(MODEL_SCRIPT.is_file() and REGISTRY.is_file(), "model script or feature registry missing")
    scopes = list(SCOPES) if args.scope == "all" else [args.scope]
    contexts = {scope: preflight_scope(scope) for scope in scopes}
    jobs = [job for scope in scopes for job in contexts[scope]["jobs"]]
    if args.name:
        unknown = set(args.name) - {stem for _, stem, _ in jobs}
        assert_ok(not unknown, f"unknown --name(s) for scope {args.scope}: {sorted(unknown)}")
        jobs = [job for job in jobs if job[1] in set(args.name)]
    statuses = []
    for scope, stem, path in jobs:
        out = output_dir(scope, stem)
        if out.exists():
            valid, reason = result_check(scope, stem, contexts[scope])
            state = "PASS" if valid else "BLOCKED"
        else:
            state, reason = "READY", "no production output exists"
        statuses.append((state, reason, (scope, stem, path)))
    counts = Counter(state for state, _, _ in statuses)
    selected = [j for state, _, j in statuses if state == "READY"]
    if args.limit:
        selected = selected[:args.limit]
    print(f"REPO: {REPO}", flush=True)
    print(f"Scopes: {', '.join(scopes)}; config count selected by filters: {len(jobs)}", flush=True)
    print("Frozen banks: " + ", ".join(f"{s}={contexts[s]['split_sha'][:12]}" for s in scopes), flush=True)
    print(f"Status: {dict(counts)}; next jobs: {len(selected)}; mode: {'EXECUTE' if args.execute else 'PREFLIGHT ONLY'}", flush=True)
    for state, reason, j in statuses:
        if state == "BLOCKED":
            print(f"BLOCKED {j[0]}/{j[1]}: {reason}; inspect {output_dir(j[0], j[1])}", flush=True)
    for j in selected[:12]:
        print(f"NEXT {j[0]}/{j[1]}", flush=True)
    if not args.execute:
        print("PREFLIGHT PASS (read-only). To train, add --execute.", flush=True)
        return 0
    if counts.get("BLOCKED", 0):
        print("REFUSING EXECUTE: incomplete/corrupt result directory exists; never overwrite automatically.", flush=True)
        return 2
    if not selected:
        print("NOTHING TO DO: all selected experiments already passed or there are no new jobs.", flush=True)
        return 0

    MANAGEMENT.mkdir(parents=True, exist_ok=True)
    lock_file = MANAGEMENT / ".batch_runner.lock"
    with lock_file.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("REFUSING EXECUTE: another IGH Layer1 batch runner holds the lock.", flush=True)
            return 3
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        queue = list(selected)
        failed = 0
        passed = 0
        while (queue and not STOP and not failed) or ACTIVE:
            while queue and len(ACTIVE) < args.max_concurrent and not STOP and not failed:
                item = queue.pop(0)
                key = f"{item[0]}/{item[1]}"
                ACTIVE[key] = launch_job(item, args.workers_per_experiment)
            time.sleep(1)
            for key, job in list(ACTIVE.items()):
                proc = job["proc"]
                rc = proc.poll()
                if rc is None:
                    continue
                scope, stem = job["scope"], job["stem"]
                valid, detail = result_check(scope, stem, contexts[scope]) if rc == 0 else (False, f"process exit code={rc}")
                state = "PASS" if valid and not STOP else ("INTERRUPTED" if STOP else "FAIL")
                atomic_json(job["status"], {
                    "state": state, "scope": scope, "name": stem, "pid": proc.pid,
                    "return_code": rc, "checked_utc": utcnow(), "detail": detail,
                    "output": str(output_dir(scope, stem)), "log": str(job["log"]),
                })
                print(f"{state} {key}: {detail} (rc={rc})", flush=True)
                if state == "PASS":
                    passed += 1
                else:
                    failed += 1
                    print(f"Inspect log: tail -n 50 '{job['log']}'", flush=True)
                del ACTIVE[key]
            if STOP and ACTIVE and STOP_AT is not None and time.monotonic() - STOP_AT > 20:
                # Escalate only after allowing subprocess groups to exit gracefully.
                for job in list(ACTIVE.values()):
                    if job["proc"].poll() is None:
                        try:
                            os.killpg(job["proc"].pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
        print(f"BATCH FINISHED: PASS={passed}, FAIL/INTERRUPTED={failed}, unstarted={len(queue)}", flush=True)
        return 130 if STOP else (1 if failed else 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
