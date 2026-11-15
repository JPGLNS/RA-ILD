#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IGH V3 Layer2 F1-only, leakage-safe batch orchestration (analysis-policy locked).

Scope: 15 already-generated Top5 F1 YAMLs, NEVER F2/F3. The original 45 YAMLs
are retained as historical configs. No model training without BOTH --lock-plan
(first use, explicit) and --execute (subsequent use, explicit).

Validated results are reusable; partial production output is BLOCKED rather than
replaced. Each experiment's 100 repeats are an atomic unit (no partial resume).
V2, Layer1 and frozen banks are read-only.  Python 3.9+; pandas, PyYAML.
"""
from __future__ import annotations

import argparse
import copy
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
MANAGEMENT = VER / "runs/layer2_f1_batch_v3_175"
PLAN_PATH = VER / "analysis/layer2_f1_only_plan_v3_175.json"

# Pinned from the previously completed 381-row Layer1 experiment-level summary.
# These are actual 100-repeat mean ROC-AUC Top5 *within each scope*.
TOP5 = {
    "total": ("S008_D_E", "S049_E_AU_PU", "S001_D", "S069_D_E_AU_PU", "S092_E_AU_PU_PW"),
    "pbmc": ("S003_L", "S021_L_PU", "S008_D_E", "S022_L_PW", "S059_L_PU_PW"),
    "buffycoat": ("S002_E", "S008_D_E", "S001_D", "S003_L", "S014_E_L"),
}
SCOPES = {
    "total": (175, 122, 53, "igh_ra_ild_repeat100_v3_175", "455eba17c74a86667fd2939dc8645ac1c50f0cdb25053ab951dccad0ab8f1788"),
    "pbmc": (113, 79, 34, "igh_ra_ild_pbmc_repeat100_v3_175", "4726d9b48301c4cd535369f9b39f6bdf3c6247c0722b9c9663385a6c041a68b8"),
    "buffycoat": (62, 43, 19, "igh_ra_ild_buffycoat_repeat100_v3_175", "da1e598559dc734c0e0296bd1f99620351c2d81e903100c85fda23b670047c05"),
}
REQUIRED_OUTPUTS = (
    "06_outer_metrics.csv", "06_outer_predictions.csv.gz", "06_inner_tuning.csv.gz",
    "06_selected_hyperparameters.csv", "06_feature_explanations.csv.gz",
    "06_feature_fit_audit.csv.gz", "06_preprocessing_audit.csv.gz",
    "06_inner_assignment_audit.csv.gz", "06_run_manifest.json", "06_run_summary.md",
)
GROUPS = {
    "tcr_diversity", "tcr_clonal_expansion", "tcr_aa_length",
    "tcr_aa_composition_unweighted", "tcr_aa_composition_weighted",
    "tcr_physicochemical_unweighted", "tcr_physicochemical_weighted",
}
STOP = False
STOP_AT = None
ACTIVE = {}


def assert_ok(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def split_path(scope):
    return SPLIT_ROOT / SCOPES[scope][3] / "repeated_holdout_assignments.csv"


def config_dir(scope):
    return CFG_ROOT / f"igh_layer2_{scope}_top10000_v3_175"


def layer1_dir(scope):
    return CFG_ROOT / f"igh_layer1_{scope}_top10000_v3_175"


def compat_dir(scope):
    return VER / "ml_compat" / scope


def output_dir(scope, stem):
    return VER / "runs" / f"layer2_{scope}_top10000" / stem / "repeat100"


def status_path(scope, stem):
    return MANAGEMENT / "status" / scope / f"{stem}.json"


def log_path(scope, stem):
    return MANAGEMENT / "logs" / scope / f"{stem}.log"


def as_ints(series):
    nums = pd.to_numeric(series, errors="raise")
    assert_ok(nums.notna().all() and (nums == nums.astype(int)).all(), "non-integral repeat/count column")
    return nums.astype(int)


def preflight_scope(scope):
    n_total, n_train, n_holdout, bank_name, frozen_sha = SCOPES[scope]
    bank = split_path(scope)
    frozen = bank.parent / "SPLITS_FROZEN.json"
    assert_ok(bank.is_file() and frozen.is_file(), f"missing frozen bank: {bank.parent}")
    actual_sha = sha256(bank)
    frozen_data = json.loads(frozen.read_text(encoding="utf-8"))
    assert_ok(actual_sha == frozen_sha, f"split SHA256 mismatch for {scope}: {actual_sha}")
    assert_ok(frozen_data.get("status") == "FROZEN", f"bank is not marked FROZEN: {scope}")
    assert_ok(frozen_data.get("files", {}).get("assignments", {}).get("sha256") == frozen_sha,
              f"frozen marker SHA mismatch: {scope}")
    bank_df = pd.read_csv(bank, dtype={"sample_id": str})
    assert_ok({"repeat_index", "sample_id", "role"}.issubset(bank_df.columns),
              f"missing split columns: {scope}")
    bank_df["repeat_index"] = as_ints(bank_df["repeat_index"])
    assert_ok(set(bank_df["repeat_index"]) == set(range(1, 101)),
              f"not 100 splits: {scope}")
    holdouts = {}
    repeat1 = None
    bank_ids = None
    for r, part in bank_df.groupby("repeat_index"):
        assert_ok(len(part) == n_total and part["sample_id"].is_unique,
                  f"split duplicated/missing sample: {scope}:{r}")
        ids = set(part["sample_id"].astype(str))
        if bank_ids is None:
            bank_ids = ids
        assert_ok(ids == bank_ids, f"split sample universe changed: {scope}:{r}")
        counts = part["role"].value_counts().to_dict()
        assert_ok(counts == {"train": n_train, "holdout": n_holdout},
                  f"train/holdout counts: {scope}:{r}: {counts}")
        holdouts[int(r)] = set(part.loc[part.role == "holdout", "sample_id"])
        if int(r) == 1:
            repeat1 = {
                role: set(part.loc[part.role == role, "sample_id"])
                for role in ("train", "holdout")
            }
        if scope == "buffycoat":
            assert_ok("MGI260202N01-31" not in holdouts[int(r)],
                      "buffycoat singleton leaked to holdout")
        elif scope == "total":
            assert_ok("MGI260201R01-24" not in holdouts[int(r)],
                      "total singleton leaked to holdout")
        elif scope == "pbmc":
            assert_ok(not ({"MGI260201R01-23", "MGI260201R01-24"} & holdouts[int(r)]),
                      "pbmc singleton leaked to holdout")
    assert_ok(repeat1 is not None, f"missing repeat 1 in {scope}")

    compat = compat_dir(scope)
    cohort_by_id = {}
    for part_name, role in (("train", "train"), ("test", "holdout")):
        meta_path = compat / f"metadata_{part_name}.csv"
        core_path = compat / f"{part_name}_step02/02_core_sample_features.csv"
        source_path = compat / f"{part_name}_step02/02_resolved_feature_source.json"
        assert_ok(all(p.is_file() for p in (meta_path, core_path, source_path)),
                  f"missing ml_compat: {scope}:{part_name}")
        metadata = pd.read_csv(meta_path, dtype={"libraryid": str})
        core = pd.read_csv(core_path, usecols=["sample_id"], dtype={"sample_id": str})
        assert_ok(metadata["libraryid"].is_unique and core["sample_id"].is_unique,
                  f"duplicate ml_compat IDs: {scope}:{part_name}")
        meta_ids = set(metadata["libraryid"].str.strip())
        core_ids = set(core["sample_id"].str.strip())
        assert_ok(meta_ids == core_ids == repeat1[role],
                  f"ml_compat partition differs from split 1: {scope}:{part_name}")
        assert_ok(metadata["cohort"].isin(["RA", "ILD"]).all(),
                  f"invalid labels: {scope}:{part_name}")
        cohort_by_id.update(dict(zip(metadata["libraryid"].str.strip(), metadata["cohort"])))
        source = json.loads(source_path.read_text(encoding="utf-8"))
        assert_ok(source.get("repertoire_input", {}).get("source_id") == "top10000",
                  f"ml_compat origin is not top10000: {scope}:{part_name}")
    assert_ok(len(cohort_by_id) == n_total and set(cohort_by_id) == bank_ids,
              f"metadata scope coverage: {scope}")

    # The historical 30 F2/F3 YAMLs stay on disk, but cannot be selected.
    old_dir, current_dir = layer1_dir(scope), config_dir(scope)
    assert_ok(old_dir.is_dir() and current_dir.is_dir(), f"missing config directory: {scope}")
    assert_ok(len(list(old_dir.glob("S*.yaml"))) == 127, f"Layer1 config count mismatch: {scope}")
    yaml_paths = sorted(current_dir.glob("S*.yaml"))
    assert_ok(len(yaml_paths) == 15, f"expected 15 historical Layer2 YAMLs: {scope}")
    stems_all = {p.stem for p in yaml_paths}
    expected_all = {f"{stem}_{form}" for stem in TOP5[scope] for form in ("F1", "F2", "F3")}
    assert_ok(stems_all == expected_all, f"15 Layer2 YAML filenames differ from Top5: {scope}")
    history = json.loads((current_dir / "LAYER2_MANIFEST.json").read_text(encoding="utf-8"))
    assert_ok(len(history) == 15 and
              {h["layer2_stem"] for h in history} == expected_all,
              f"Layer2 historical manifest mismatch: {scope}")
    jobs = []
    for rank, old_stem in enumerate(TOP5[scope], 1):
        stem = old_stem + "_F1"
        cfg_path = current_dir / f"{stem}.yaml"
        original_path = old_dir / f"{old_stem}.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        old = yaml.safe_load(original_path.read_text(encoding="utf-8"))
        assert_ok(isinstance(cfg, dict) and isinstance(old, dict),
                  f"malformed YAML: {cfg_path}")
        assert_ok(cfg["experiment"]["id"] == f"igh_layer2_{scope}_top10000_{stem}_v3_175",
                  f"Layer2 experiment ID differs: {cfg_path}")
        assert_ok(old["experiment"]["id"] == f"igh_layer1_{scope}_top10000_{old_stem}_v3_175",
                  f"Layer1 experiment ID differs: {original_path}")
        assert_ok(cfg["cohort"]["scope"] == scope and cfg["experiment"]["receptor"] == "IGH",
                  f"incorrect receptor/scope: {cfg_path}")
        assert_ok(cfg["model"]["algorithm"] == "elastic_net" and
                  cfg["model"]["positive_label"] == "ILD",
                  f"incorrect model: {cfg_path}")
        assert_ok(cfg["validation"]["folds"] == 5 and
                  cfg["validation"]["repeats"] == 100,
                  f"incorrect CV: {cfg_path}")
        ref = Path(cfg["validation"]["split_assignments"])
        if not ref.is_absolute():
            ref = REPO / ref
        assert_ok(ref.resolve() == bank.resolve(), f"config split mismatch: {cfg_path}")
        assert_ok(cfg["repertoire"]["mode"] == "topk" and
                  int(cfg["repertoire"]["top_k"]) == 10000,
                  f"source not Top10000: {cfg_path}")
        feat = cfg["features"]
        kmer = feat["kmer"]
        enrich = feat["enriched_dictionary"]
        assert_ok(feat["clinical_groups"] == [], f"Layer2 clinical features enabled: {cfg_path}")
        assert_ok(kmer["enabled"] is True and kmer["k"] == 3 and
                  kmer["representation"] == "unweighted" and kmer["top_n"] == 500 and
                  kmer["min_sample_count"] == 5 and float(kmer["min_prevalence"]) == 0.05,
                  f"F1 3mer contract mismatch: {cfg_path}")
        assert_ok(enrich["enabled"] is False, f"F1 enriched dictionary MUST BE OFF: {cfg_path}")
        assert_ok(1 <= len(feat["static_tcr_groups"]) <= 7 and
                  len(feat["static_tcr_groups"]) == len(set(feat["static_tcr_groups"])) and
                  set(feat["static_tcr_groups"]).issubset(GROUPS),
                  f"static groups invalid: {cfg_path}")
        assert_ok(cfg["model"] == old["model"] and
                  feat["static_tcr_groups"] == old["features"]["static_tcr_groups"],
                  f"Layer2 static/model settings diverge from Layer1: {cfg_path}")
        # Only selected differences allowed relative to Layer1 config.
        expected = copy.deepcopy(old)
        expected["experiment"]["id"] = cfg["experiment"]["id"]
        expected["features"]["kmer"] = copy.deepcopy(kmer)
        expected["features"]["enriched_dictionary"] = copy.deepcopy(enrich)
        assert_ok(expected == cfg, f"unexpected configuration changes from Layer1: {cfg_path}")
        grid = cfg["model"]["parameters"]["elastic_net"]
        assert_ok(len(grid["alpha_grid"]) * len(grid["lambda_grid"]) == 24,
                  f"hyperparameter grid mismatch: {cfg_path}")
        matching = [entry for entry in history if entry["layer2_stem"] == stem]
        assert_ok(len(matching) == 1 and matching[0]["form"] == "F1" and
                  matching[0]["layer1_rank"] == rank and matching[0]["layer1_stem"] == old_stem and
                  matching[0]["split_sha256"] == frozen_sha,
                  f"historical manifest F1 mismatch: {cfg_path}")
        jobs.append((scope, stem, cfg_path))
    assert_ok(len(jobs) == 5, f"expected five F1-only jobs: {scope}")
    return {"jobs": jobs, "holdouts": holdouts, "cohort": cohort_by_id,
            "split_sha": frozen_sha, "split_path": bank,
            "train_n": n_train, "holdout_n": n_holdout}


def make_plan(contexts):
    selected = []
    for scope in SCOPES:
        for _, stem, cfg in contexts[scope]["jobs"]:
            selected.append({"scope": scope, "name": stem,
                             "config": str(cfg.relative_to(REPO)),
                             "config_sha256": sha256(cfg),
                             "frozen_split_sha256": contexts[scope]["split_sha"]})
    assert_ok(len(selected) == 15, "plan must contain 15 F1-only jobs")
    return {
        "schema_version": 1,
        "decision": "IGH_V3_LAYER2_F1_ONLY",
        "included": "F1: selected Layer1 Top5 static features + training-fold Top500 unweighted CDR3 AA 3mer",
        "excluded": ["F2: RA-enriched dictionary", "F3: 3mer + RA-enriched dictionary"],
        "reason": "IGH exact-CDR3 enriched dictionary showed empty/unstable reference at TRB thresholds; not part of IGH main analysis.",
        "legacy_yaml_policy": "Keep 45 existing F1/F2/F3 YAML files unchanged; execute exactly the 15 F1 YAMLs listed below.",
        "model": "Elastic Net logistic regression; 5-fold inner CV; 24 candidates; 100 frozen repeated holdouts; no clinical variables",
        "experiments": selected,
    }


def validate_plan(plan):
    if not PLAN_PATH.is_file():
        return False
    existing = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert_ok(existing == plan,
              f"F1-only plan exists but does not match verified inputs: {PLAN_PATH}")
    return True


def publish_plan(plan):
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if validate_plan(plan):
        print(f"LOCK-PLAN ALREADY VERIFIED: {PLAN_PATH}", flush=True)
        return
    try:
        with PLAN_PATH.open("x", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except FileExistsError:
        validate_plan(plan)
    print(f"LOCK-PLAN PASS: 15 F1 jobs; 30 F2/F3 configs excluded; {PLAN_PATH}", flush=True)


def result_check(scope, stem, context):
    out = output_dir(scope, stem)
    if not out.is_dir():
        return False, "production output directory absent"
    try:
        absent = [f for f in REQUIRED_OUTPUTS if not (out / f).is_file()]
        assert_ok(not absent, f"missing outputs: {absent}")
        metrics = pd.read_csv(out / "06_outer_metrics.csv")
        pred = pd.read_csv(out / "06_outer_predictions.csv.gz", dtype={"sample_id": str})
        tune = pd.read_csv(out / "06_inner_tuning.csv.gz")
        selected = pd.read_csv(out / "06_selected_hyperparameters.csv")
        audit = pd.read_csv(out / "06_feature_fit_audit.csv.gz")
        manifest = json.loads((out / "06_run_manifest.json").read_text(encoding="utf-8"))
        assert_ok(len(metrics) == len(selected) == 100, "metrics/selected != 100 rows")
        assert_ok(len(tune) == 2400, "inner tuning != 2400 rows")
        assert_ok(len(pred) == 100 * context["holdout_n"], "holdout predictions count wrong")
        for df, label in ((metrics, "metrics"), (selected, "selected"), (pred, "predictions")):
            assert_ok(set(as_ints(df["outer_repeat"])) == set(range(1, 101)),
                      f"outer repeat 1..100 incomplete: {label}")
        assert_ok((as_ints(metrics["n_outer_train"]) == context["train_n"]).all(),
                  "training counts mismatch")
        assert_ok((as_ints(metrics["n_outer_holdout"]) == context["holdout_n"]).all(),
                  "holdout counts mismatch")
        assert_ok(metrics["fit_converged"].astype(str).str.lower().eq("true").all(),
                  "final model not converged in all repeats")
        for rep, df in pred.groupby("outer_repeat"):
            ids = df["sample_id"].astype(str)
            assert_ok(ids.is_unique and set(ids) == context["holdouts"][int(rep)],
                      f"frozen holdout IDs mismatch: repeat {rep}")
            expected_labels = [context["cohort"][sid] for sid in ids]
            assert_ok(df["true_cohort"].astype(str).tolist() == expected_labels,
                      f"cohort labels mismatch: repeat {rep}")
        execution = manifest.get("execution", {})
        assert_ok(execution.get("repeat_count") == 100 and
                  execution.get("candidate_count") == 24,
                  "manifest repeat/candidate contract mismatch")
        path = Path(manifest.get("inputs", {}).get("split_assignments", ""))
        if not path.is_absolute():
            path = REPO / path
        assert_ok(path.resolve() == context["split_path"].resolve(),
                  "manifest frozen split path wrong")
        assert_ok({"outer_repeat", "feature_family", "fit_policy"}.issubset(audit.columns),
                  "feature fit audit lacks 3mer provenance fields")
        kmer = audit.loc[audit["feature_family"].astype(str) == "3mer"]
        assert_ok(len(kmer) >= 100, "missing 3mer fit audits for repeats")
        assert_ok(set(as_ints(kmer["outer_repeat"])) == set(range(1, 101)),
                  "3mer audit repeats incomplete")
        assert_ok(kmer["fit_policy"].astype(str).eq("fit_vocabulary_on_current_training_only").all(),
                  "3mer fitted with unexpected policy")
        assert_ok(not audit["feature_family"].astype(str).eq("enriched_dictionary").any(),
                  "F1 output unexpectedly used enriched dictionary")
        return True, "100 repeats, frozen holdout IDs, 3mer fold-fit audits, manifest PASS"
    except Exception as ex:
        return False, f"validation error: {ex}"


def request_stop(signum, _frame):
    global STOP, STOP_AT
    if STOP:
        return
    STOP = True
    STOP_AT = time.monotonic()
    print(f"\nSIGNAL {signum}: stop launching; terminate active process groups", flush=True)
    for item in list(ACTIVE.values()):
        proc = item["proc"]
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def launch_job(job, workers):
    scope, stem, cfg = job
    out = output_dir(scope, stem)
    log = log_path(scope, stem)
    state = status_path(scope, stem)
    assert_ok(not out.exists(), f"refuse overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(MODEL_SCRIPT),
           "--config", str(cfg),
           "--train-metadata", str(compat_dir(scope) / "metadata_train.csv"),
           "--test-metadata", str(compat_dir(scope) / "metadata_test.csv"),
           "--train-step02-dir", str(compat_dir(scope) / "train_step02"),
           "--test-step02-dir", str(compat_dir(scope) / "test_step02"),
           "--registry", str(REGISTRY),
           "--split-assignments", str(split_path(scope)),
           "--output-dir", str(out),
           "--workers", str(workers)]
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    with log.open("w", encoding="utf-8") as stream:
        stream.write(f"Started UTC: {utcnow()}\nCommand: {json.dumps(cmd, ensure_ascii=False)}\n\n")
        stream.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True,
                                env=env)
    atomic_json(state, {"state": "RUNNING", "scope": scope, "name": stem,
                        "pid": proc.pid, "started_utc": utcnow(),
                        "output": str(out), "log": str(log)})
    print(f"START {scope}/{stem} pid={proc.pid} workers={workers} log={log}", flush=True)
    return {"proc": proc, "scope": scope, "stem": stem, "log": log,
            "status": state, "started": time.monotonic()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--scope", choices=["all", *SCOPES], default="all")
    ap.add_argument("--name", action="append", default=[], help="Exact F1 config stem, e.g. S008_D_E_F1 (repeatable)")
    ap.add_argument("--limit", type=int, default=1, help="Max new experiments to run, 0=all")
    ap.add_argument("--max-concurrent", type=int, default=1)
    ap.add_argument("--workers-per-experiment", type=int, default=1)
    ap.add_argument("--lock-plan", action="store_true", help="Persist immutable 15-F1 design manifest; no training")
    ap.add_argument("--execute", action="store_true", help="Train selected READY jobs (requires --lock-plan previously)")
    args = ap.parse_args()
    assert_ok(not (args.execute and args.lock_plan), "--lock-plan and --execute are mutually exclusive")
    assert_ok(args.limit >= 0, "--limit must be nonnegative")
    assert_ok(1 <= args.max_concurrent <= 6, "--max-concurrent must be 1..6")
    assert_ok(1 <= args.workers_per_experiment <= 8, "--workers-per-experiment must be 1..8")
    assert_ok(args.max_concurrent * args.workers_per_experiment <= 48,
              "combined model workers must be <=48")
    assert_ok(MODEL_SCRIPT.is_file() and REGISTRY.is_file(), "missing model runner or IGH V3 registry")
    # Lock/plan must always check all 15 F1 configs and all 3 frozen banks.
    all_contexts = {scope: preflight_scope(scope) for scope in SCOPES}
    plan = make_plan(all_contexts)
    if args.lock_plan:
        publish_plan(plan)
        return 0
    plan_exists = validate_plan(plan)
    scopes = list(SCOPES) if args.scope == "all" else [args.scope]
    jobs = [job for scope in scopes for job in all_contexts[scope]["jobs"]]
    if args.name:
        unmatched = set(args.name) - {j[1] for j in jobs}
        assert_ok(not unmatched, f"unknown F1-only --name for requested scope: {sorted(unmatched)}")
        jobs = [j for j in jobs if j[1] in args.name]
    statuses = []
    for scope, stem, cfg in jobs:
        if output_dir(scope, stem).exists():
            valid, detail = result_check(scope, stem, all_contexts[scope])
            statuses.append(("PASS" if valid else "BLOCKED", detail, (scope, stem, cfg)))
        else:
            statuses.append(("READY", "no production output", (scope, stem, cfg)))
    counts = Counter(x[0] for x in statuses)
    selected = [j for state, _, j in statuses if state == "READY"]
    if args.limit:
        selected = selected[:args.limit]
    print(f"REPO: {REPO}", flush=True)
    print("Layer2 policy: F1 ONLY. F2/F3 EXCLUDED; their YAMLs retained unchanged.", flush=True)
    bank_text = ", ".join(scope + "=" + all_contexts[scope]["split_sha"][:12] for scope in scopes)
    print(f"Frozen banks: {bank_text}", flush=True)
    print(f"Selected configs: {len(jobs)} (15 when --scope all); locked plan: {plan_exists}", flush=True)
    print(f"Status: {dict(counts)}; next jobs: {len(selected)}; mode: {'EXECUTE' if args.execute else 'PREFLIGHT ONLY'}", flush=True)
    for state, detail, j in statuses:
        if state == "BLOCKED":
            print(f"BLOCKED {j[0]}/{j[1]}: {detail}; inspect {output_dir(j[0], j[1])}", flush=True)
    for job in selected[:15]:
        print(f"NEXT {job[0]}/{job[1]}", flush=True)
    if not args.execute:
        if not plan_exists:
            print("PREFLIGHT READ-ONLY PASS; publish design using --lock-plan before --execute.", flush=True)
        else:
            print("PREFLIGHT PASS (read-only). To train, add --execute.", flush=True)
        return 0
    assert_ok(plan_exists, f"REFUSE EXECUTE: run --lock-plan first; missing {PLAN_PATH}")
    if counts.get("BLOCKED", 0):
        print("REFUSE EXECUTE: incomplete/corrupt production result; never overwrite.", flush=True)
        return 2
    if not selected:
        print("NOTHING TO DO: no READY F1 jobs selected.", flush=True)
        return 0
    MANAGEMENT.mkdir(parents=True, exist_ok=True)
    with (MANAGEMENT / ".batch_runner.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("REFUSE EXECUTE: another F1 batch runner holds the lock.", flush=True)
            return 3
        # Python explicitly ignores SIGHUP instead of replacing nohup's SIG_IGN.
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        # Recheck after acquiring flock: a previous independent runner may have
        # finished jobs between our initial preflight and this lock acquisition.
        queue = []
        for scope, stem, cfg in selected:
            out = output_dir(scope, stem)
            if out.exists():
                valid, detail = result_check(scope, stem, all_contexts[scope])
                if valid:
                    print(f"SKIP newly PASS {scope}/{stem} after lock", flush=True)
                    continue
                print(f"REFUSE newly BLOCKED {scope}/{stem}: {detail}", flush=True)
                return 2
            queue.append((scope, stem, cfg))
        passed = 0
        failed = 0
        while (queue and not STOP and not failed) or ACTIVE:
            while queue and not STOP and not failed and len(ACTIVE) < args.max_concurrent:
                job = queue.pop(0)
                key = f"{job[0]}/{job[1]}"
                ACTIVE[key] = launch_job(job, args.workers_per_experiment)
            time.sleep(1)
            for key, item in list(ACTIVE.items()):
                rc = item["proc"].poll()
                if rc is None:
                    continue
                scope, stem = item["scope"], item["stem"]
                valid, detail = (result_check(scope, stem, all_contexts[scope]) if rc == 0
                                 else (False, f"process exit code={rc}"))
                state = "PASS" if valid and not STOP else ("INTERRUPTED" if STOP else "FAIL")
                atomic_json(item["status"], {
                    "state": state, "scope": scope, "name": stem, "pid": item["proc"].pid,
                    "return_code": rc, "checked_utc": utcnow(), "detail": detail,
                    "output": str(output_dir(scope, stem)), "log": str(item["log"]),
                })
                print(f"{state} {key}: {detail} (rc={rc})", flush=True)
                if state == "PASS":
                    passed += 1
                else:
                    failed += 1
                    print(f"Inspect log: tail -n 50 '{item['log']}'", flush=True)
                del ACTIVE[key]
            if STOP and ACTIVE and STOP_AT is not None and time.monotonic() - STOP_AT > 20:
                for item in list(ACTIVE.values()):
                    proc = item["proc"]
                    if proc.poll() is None:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
        print(f"BATCH FINISHED: PASS={passed}, FAIL/INTERRUPTED={failed}, unstarted={len(queue)}", flush=True)
        return 130 if STOP else (1 if failed else 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
