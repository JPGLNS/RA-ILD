#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IGH V3 selected-BCR-only ablation (Total PW, PBMC E, Buffycoat E).

Two safe paths:
  audit-existing: validate matching IGH-only Layer1 100-repeat results (no new fits).
  collect-existing: deliver those validated results plus paired comparisons against
                    the clinical baseline and clinical+BCR Layer1 experiments.
  run:             optional independent rerun using the frozen Phase-6 per-repeat
                    checkpoint engine; --execute is required.
  status:          deep-check independent per-repeat checkpoints.
  collect-new:     aggregate independently rerun checkpoints and paired results.

The matching Layer1 YAML files already have clinical_groups: [], kmer disabled,
enriched dictionary disabled. This script never changes those files, existing
training outputs, any split bank, or the frozen five-algorithm experiment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CASES = {
    "total": {
        "stem": "S007_PW", "code": "PW",
        "group": "tcr_physicochemical_weighted",
        "n": 175, "n_train": 122, "n_holdout": 53,
        "split_sha256": "455eba17c74a86667fd2939dc8645ac1c50f0cdb25053ab951dccad0ab8f1788",
    },
    "pbmc": {
        "stem": "S002_E", "code": "E",
        "group": "tcr_clonal_expansion",
        "n": 113, "n_train": 79, "n_holdout": 34,
        "split_sha256": "4726d9b48301c4cd535369f9b39f6bdf3c6247c0722b9c9663385a6c041a68b8",
    },
    "buffycoat": {
        "stem": "S002_E", "code": "E",
        "group": "tcr_clonal_expansion",
        "n": 62, "n_train": 43, "n_holdout": 19,
        "split_sha256": "da1e598559dc734c0e0296bd1f99620351c2d81e903100c85fda23b670047c05",
    },
}
METRICS = ("roc_auc", "pr_auc", "accuracy", "sensitivity_recall", "specificity", "f1")
OUTPUT_NAMES = (
    "06_outer_metrics.csv", "06_outer_predictions.csv.gz", "06_inner_tuning.csv.gz",
    "06_selected_hyperparameters.csv", "06_feature_explanations.csv.gz",
    "06_feature_fit_audit.csv.gz", "06_preprocessing_audit.csv.gz",
    "06_inner_assignment_audit.csv.gz", "06_run_manifest.json", "06_run_summary.md",
)


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def source_config(repo, scope):
    x = CASES[scope]
    return repo / "IGH/set/configs/experiments" / (
        f"igh_layer1_{scope}_top10000_v3_175/{x['stem']}.yaml"
    )


def clinical_config(repo, scope, stem):
    return repo / "IGH/set/configs/experiments" / (
        f"igh_clinical_layer1_{scope}_top10000_v3_175/{stem}.yaml"
    )


def legacy_dir(repo, scope, stem, *, clinical=False):
    kind = "clinical_layer1" if clinical else "layer1"
    return (repo / "IGH/set/data_versions/igh_v3_175/runs" /
            f"{kind}_{scope}_top10000" / stem / "repeat100")


def split_path(repo, scope):
    name = ("igh_ra_ild_repeat100_v3_175" if scope == "total"
            else f"igh_ra_ild_{scope}_repeat100_v3_175")
    return repo / "IGH/set/split_sets" / name / "repeated_holdout_assignments.csv"


def validate_config(repo, scope):
    x = CASES[scope]
    source = source_config(repo, scope)
    clinical = clinical_config(repo, scope, x["stem"])
    baseline = clinical_config(repo, scope, "S000_none")
    need(all(p.is_file() for p in (source, clinical, baseline)),
         f"Missing source/clinical YAML for {scope}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    joint = yaml.safe_load(clinical.read_text(encoding="utf-8"))
    base = yaml.safe_load(baseline.read_text(encoding="utf-8"))
    f = raw["features"]
    need(f["clinical_groups"] == [], f"{scope}: clinical predictors must be empty")
    need(f["static_tcr_groups"] == [x["group"]], f"{scope}: wrong BCR group")
    need(f["kmer"]["enabled"] is False and f["enriched_dictionary"]["enabled"] is False,
         f"{scope}: unexpected dynamic feature")
    need(not f.get("public_reference", {}).get("enabled", False),
         f"{scope}: unexpected public reference")
    need(raw["model"]["algorithm"] == "elastic_net"
         and raw["model"]["positive_label"] == "ILD", f"{scope}: model mismatch")
    need(raw["repertoire"] == {"mode": "topk", "top_k": 10000},
         f"{scope}: repertoire mismatch")
    need(raw["validation"]["repeats"] == 100
         and raw["validation"]["folds"] == 5, f"{scope}: CV settings mismatch")
    grid = raw["model"]["parameters"]["elastic_net"]
    need(len(grid["alpha_grid"]) * len(grid["lambda_grid"]) == 24,
         f"{scope}: expected 24 Elastic Net candidates")
    actual_split = repo / raw["validation"]["split_assignments"]
    need(actual_split.resolve() == split_path(repo, scope).resolve(),
         f"{scope}: frozen split path mismatch")
    need(digest(actual_split) == x["split_sha256"],
         f"{scope}: frozen split SHA256 mismatch")
    expected_joint = copy.deepcopy(raw)
    expected_joint["features"]["clinical_groups"] = ["clinical_age_sex"]
    expected_joint["experiment"]["id"] = joint["experiment"]["id"]
    need(expected_joint == joint,
         f"{scope}: selected joint YAML is NOT just BCR-only + Age/Sex")
    expected_base = copy.deepcopy(joint)
    expected_base["features"]["static_tcr_groups"] = []
    expected_base["experiment"]["id"] = base["experiment"]["id"]
    need(expected_base == base, f"{scope}: Age+Sex baseline YAML mismatch")
    bank = pd.read_csv(actual_split, dtype={"sample_id": str})
    need({"repeat_index", "sample_id", "role"} <= set(bank),
         f"{scope}: split schema mismatch")
    bank["repeat_index"] = bank["repeat_index"].astype(int)
    need(len(bank) == x["n"] * 100
         and set(bank["repeat_index"]) == set(range(1, 101)),
         f"{scope}: unexpected split size/repeats")
    holdouts = {}
    for repeat, part in bank.groupby("repeat_index"):
        need(len(part) == x["n"] and part["sample_id"].is_unique,
             f"{scope}/{repeat}: repeated IDs")
        counts = part.role.value_counts().to_dict()
        need(counts == {"train": x["n_train"], "holdout": x["n_holdout"]},
             f"{scope}/{repeat}: holdout/train sizes differ")
        holdouts[int(repeat)] = set(part.loc[part.role == "holdout", "sample_id"])
    meta_root = repo / f"IGH/set/data_versions/igh_v3_175/ml_compat/{scope}"
    metas = []
    for part in ("train", "test"):
        mp = meta_root / f"metadata_{part}.csv"
        need(mp.is_file(), f"{scope}: missing {mp}")
        metas.append(pd.read_csv(mp, dtype={"libraryid": str})[["libraryid", "cohort"]])
    meta = pd.concat(metas, ignore_index=True).set_index("libraryid")["cohort"]
    need(len(meta) == x["n"] and meta.index.is_unique,
         f"{scope}: invalid metadata sample set")
    for repeat, holdout in holdouts.items():
        need(holdout <= set(meta.index),
             f"{scope}/{repeat}: split contains unknown sample")
    return {"source": source, "clinical": clinical, "baseline": baseline,
            "holdouts": holdouts, "cohorts": meta}


def check_metrics(frame, scope, label, reps):
    x = CASES[scope]
    need(len(frame) == len(reps) and frame["outer_repeat"].is_unique,
         f"{scope}/{label}: metrics count or repeat duplicate")
    need(set(frame["outer_repeat"].astype(int)) == set(reps),
         f"{scope}/{label}: unexpected repeat IDs")
    need(all(c in frame for c in ("fit_converged", *METRICS,
                                 "n_outer_train", "n_outer_holdout")),
         f"{scope}/{label}: missing metric fields")
    need(frame["fit_converged"].astype(str).str.lower().eq("true").all(),
         f"{scope}/{label}: unconverged fit")
    need((frame["n_outer_train"] == x["n_train"]).all()
         and (frame["n_outer_holdout"] == x["n_holdout"]).all(),
         f"{scope}/{label}: wrong sample counts")
    need(np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all(),
         f"{scope}/{label}: nonfinite metrics")
    return frame.sort_values("outer_repeat").reset_index(drop=True)


def validate_predictions(pred, scope, label, ctx, reps):
    x = CASES[scope]
    need(len(pred) == len(reps) * x["n_holdout"],
         f"{scope}/{label}: wrong prediction count")
    need({"outer_repeat", "sample_id", "true_cohort", "true_label",
          "prediction_score"} <= set(pred),
         f"{scope}/{label}: missing prediction fields")
    need(np.isfinite(pred["prediction_score"].to_numpy(float)).all(),
         f"{scope}/{label}: nonfinite prediction scores")
    for rep, part in pred.groupby("outer_repeat", sort=True):
        need(int(rep) in reps and part.sample_id.is_unique
             and set(part.sample_id.astype(str)) == ctx["holdouts"][int(rep)],
             f"{scope}/{label}/{rep}: predictions do not match frozen holdout")
        labels = ctx["cohorts"].loc[part.sample_id.astype(str)].to_numpy(str)
        need(np.array_equal(part.true_cohort.astype(str).to_numpy(), labels),
             f"{scope}/{label}/{rep}: cohort mismatch")
        need(np.array_equal(part.true_label.to_numpy(int),
                            (labels == "ILD").astype(int)),
             f"{scope}/{label}/{rep}: positive-label mismatch")


def read_existing(repo, scope, stem, ctx, clinical=False):
    folder = legacy_dir(repo, scope, stem, clinical=clinical)
    need(folder.is_dir(), f"Missing existing 100-repeat results: {folder}")
    need(all((folder / f).is_file() for f in OUTPUT_NAMES),
         f"Incomplete existing results: {folder}")
    metrics = check_metrics(pd.read_csv(folder / "06_outer_metrics.csv"),
                            scope, stem, range(1, 101))
    pred = pd.read_csv(folder / "06_outer_predictions.csv.gz",
                       dtype={"sample_id": str})
    validate_predictions(pred, scope, stem, ctx, set(range(1, 101)))
    selected = pd.read_csv(folder / "06_selected_hyperparameters.csv")
    need(len(selected) == 100 and selected.outer_repeat.nunique() == 100,
         f"{scope}/{stem}: wrong selected candidates")
    tuning = pd.read_csv(folder / "06_inner_tuning.csv.gz")
    need(len(tuning) == 2400, f"{scope}/{stem}: expected 24x100 tuning rows")
    manifest = json.loads((folder / "06_run_manifest.json").read_text(encoding="utf-8"))
    need(manifest.get("execution", {}).get("repeat_count") == 100,
         f"{scope}/{stem}: unexpected manifest repeat count")
    recorded = manifest.get("inputs", {}).get("split_assignments")
    need(recorded and Path(recorded).resolve() == split_path(repo, scope).resolve(),
         f"{scope}/{stem}: old result used different split bank")
    return metrics


def read_new(repo, out, scope, ctx):
    x = CASES[scope]
    folder = out / f"{scope}_{x['code']}_elastic_net"
    frames = []
    for r in range(1, 101):
        rep = folder / f"repeat_{r:03d}"
        need((rep / "RUN_COMPLETE.json").is_file(),
             f"Missing checkpoint: {rep}")
        m = pd.read_csv(rep / "06_outer_metrics.csv")
        frames.append(m)
        p = pd.read_csv(rep / "06_outer_predictions.csv.gz",
                        dtype={"sample_id": str})
        validate_predictions(p, scope, f"checkpoint_{r}", ctx, {r})
    return check_metrics(pd.concat(frames, ignore_index=True),
                         scope, "checkpoint", range(1, 101))


def collect(repo, output_root, *, source):
    rows = []
    comparisons = []
    overview = []
    origins = {}
    for scope, x in CASES.items():
        ctx = validate_config(repo, scope)
        if source == "existing":
            bcr = read_existing(repo, scope, x["stem"], ctx)
        else:
            bcr = read_new(repo, output_root, scope, ctx)
        clinical = read_existing(repo, scope, "S000_none", ctx, clinical=True)
        joint = read_existing(repo, scope, x["stem"], ctx, clinical=True)
        origins[scope] = {"bcr_config": str(ctx["source"]),
                          "bcr_config_sha256": digest(ctx["source"]),
                          "clinical_config": str(ctx["clinical"]),
                          "baseline_config": str(ctx["baseline"]),
                          "split_sha256": x["split_sha256"],
                          "bcr_results": str(
                              legacy_dir(repo, scope, x["stem"]) if source == "existing"
                              else output_root / f"{scope}_{x['code']}_elastic_net"),
                          "clinical_baseline_results": str(
                              legacy_dir(repo, scope, "S000_none", clinical=True)),
                          "clinical_joint_results": str(
                              legacy_dir(repo, scope, x["stem"], clinical=True))}
        series = {"BCR_only": bcr, "Age_Sex": clinical, "Age_Sex_plus_BCR": joint}
        for name, frame in series.items():
            for row in frame.to_dict("records"):
                rows.append({"scope": scope, "group": x["code"], "model": name, **row})
            summary = {"scope": scope, "group": x["code"], "model": name,
                       "n_repeats": 100}
            for metric in METRICS:
                arr = frame[metric].to_numpy(float)
                summary.update({
                    f"{metric}_mean": float(arr.mean()),
                    f"{metric}_sd": float(arr.std(ddof=1)),
                    f"{metric}_median": float(np.median(arr)),
                    f"{metric}_q25": float(np.quantile(arr, 0.25)),
                    f"{metric}_q75": float(np.quantile(arr, 0.75)),
                })
            overview.append(summary)
        for name, left, right in (
            ("BCR_only_minus_Age_Sex", bcr, clinical),
            ("BCR_only_minus_Age_Sex_plus_BCR", bcr, joint),
            ("Age_Sex_plus_BCR_minus_Age_Sex", joint, clinical),
        ):
            for metric in ("roc_auc", "pr_auc"):
                delta = (left[metric] - right[metric]).to_numpy(float)
                comparisons.append({
                    "scope": scope, "group": x["code"], "comparison": name,
                    "metric": metric, "n_paired_repeats": 100,
                    "delta_mean": float(delta.mean()),
                    "delta_median": float(np.median(delta)),
                    "delta_sd": float(delta.std(ddof=1)),
                    "delta_q25": float(np.quantile(delta, .25)),
                    "delta_q75": float(np.quantile(delta, .75)),
                    "fraction_positive": float((delta > 0).mean()),
                })
        print(f"[{scope}] BCR-only AUC={bcr.roc_auc.mean():.4f} "
              f"vs Age+Sex={clinical.roc_auc.mean():.4f} "
              f"vs joint={joint.roc_auc.mean():.4f}", flush=True)
    dest = output_root / ("summary_existing" if source == "existing" else "summary_new")
    dest.mkdir(parents=True, exist_ok=True)
    payloads = {
        "01_metrics_900.csv": pd.DataFrame(rows),
        "02_model_summary_9.csv": pd.DataFrame(overview),
        "03_paired_deltas_18.csv": pd.DataFrame(comparisons),
    }
    for name, frame in payloads.items():
        target = dest / name
        tmp = dest / f".{name}.tmp.{os.getpid()}"
        frame.to_csv(tmp, index=False)
        os.replace(tmp, target)
        print(f"[WRITE] {target}", flush=True)
    manifest = {
        "protocol": "IGH_V3_SELECTED_BCR_ONLY_AGE_SEX_ABLATION_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": source,
        "n_scopes": 3, "n_models_per_scope": 3,
        "n_repeats": 100, "n_outer_metrics_rows": 900,
        "note": "The selected feature groups were chosen using the same cohort's "
                "repeated holdouts; post-selection comparison is exploratory, "
                "not unbiased external validation. The 100 repeated holdouts are correlated.",
        "origins": origins,
        "files_sha256": {name: digest(dest / name) for name in payloads},
    }
    (dest / "COLLECTION_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[COLLECT PASS] 3 scopes x 3 models x 100 repeats; output={dest}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path,
                    default=Path(__file__).resolve().parents[3])
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--mode",
                    choices=("plan", "audit-existing", "collect-existing",
                             "run", "status", "collect-new"),
                    default="plan")
    ap.add_argument("--scope", choices=("all", *CASES), default="all")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=100)
    ap.add_argument("--execute", action="store_true",
                    help="Required for mode=run; otherwise no fits are made")
    args = ap.parse_args()
    need(1 <= args.start <= args.end <= 100, "Invalid repeat range")
    repo = args.repo.expanduser().resolve()
    need((repo / ".git").is_dir(), "--repo must point to a Git working tree")
    output_root = (args.output_root or
                   repo.parent / "igh_v3_selected_bcr_only_ablation_v3_175").resolve()
    need(not output_root.is_relative_to(repo),
         "Output root must be outside the Git repository")
    scopes = list(CASES) if args.scope == "all" else [args.scope]
    need(args.scope == "all" or args.mode not in {"collect-existing", "collect-new"},
         "Collection requires all three scopes")
    need(args.mode != "run" or args.execute,
         "Refusing to fit: add --execute explicitly")

    for scope in scopes:
        x = CASES[scope]
        ctx = validate_config(repo, scope)
        newdir = output_root / f"{scope}_{x['code']}_elastic_net"
        existing = legacy_dir(repo, scope, x["stem"])
        complete = sum((newdir / f"repeat_{r:03d}" / "RUN_COMPLETE.json").is_file()
                       for r in range(args.start, args.end + 1))
        print(f"[CONTRACT PASS] {scope}: BCR-only {x['code']}, "
              f"Elastic Net, frozen 100-repeat bank, no Age/Sex, "
              f"no 3mer/public; existing={existing.is_dir()} "
              f"new_markers={complete}/{args.end - args.start + 1}", flush=True)
        if args.mode == "audit-existing":
            result = read_existing(repo, scope, x["stem"], ctx)
            print(f"[EXISTING PASS] {scope}/{x['stem']}: "
                  f"100 repeats, AUC mean={result.roc_auc.mean():.4f}", flush=True)
        if args.mode in {"run", "status"}:
            command = [
                sys.executable,
                str(repo / "IGH/set/scripts/checkpoint_igh_v3_hierarchical_5alg.py"),
                "--repo", str(repo), "--config", str(ctx["source"]),
                "--output-dir", str(newdir), "--start", str(args.start),
                "--end", str(args.end),
                "--mode", "run" if args.mode == "run" else "status",
            ]
            print("[CHECKPOINT] " + " ".join(command), flush=True)
            if args.mode == "run":
                output_root.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(command, cwd=repo, check=False)
            need(result.returncode == 0,
                 f"{scope}: checkpoint exited with {result.returncode}")
            if args.mode == "status":
                need(complete == args.end-args.start+1,
                     f"{scope}: {args.end-args.start+1-complete} missing markers")
    if args.mode == "collect-existing":
        collect(repo, output_root, source="existing")
    elif args.mode == "collect-new":
        collect(repo, output_root, source="new")
    elif args.mode == "plan":
        print("[PLAN PASS] Read-only contract verification; no training.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
