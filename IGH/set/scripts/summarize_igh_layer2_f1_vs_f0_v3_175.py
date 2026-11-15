#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IGH V3 Layer2 F1-only: read-only, paired 100-repeat comparison with Layer1 F0.

Requires pandas, numpy and PyYAML. Reads the locked design, frozen banks and
15 pairs of completed 100-repeat model results. Writes only to a NEW analysis
folder after all inputs have passed verification. No training, no overwriting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

VER_REL = Path("IGH/set/data_versions/igh_v3_175")
CFG_REL = Path("IGH/set/configs/experiments")
BANK_REL = Path("IGH/set/split_sets")
SCOPES = {
    "total": (122, 53, "igh_ra_ild_repeat100_v3_175", "455eba17c74a86667fd2939dc8645ac1c50f0cdb25053ab951dccad0ab8f1788"),
    "pbmc": (79, 34, "igh_ra_ild_pbmc_repeat100_v3_175", "4726d9b48301c4cd535369f9b39f6bdf3c6247c0722b9c9663385a6c041a68b8"),
    "buffycoat": (43, 19, "igh_ra_ild_buffycoat_repeat100_v3_175", "da1e598559dc734c0e0296bd1f99620351c2d81e903100c85fda23b670047c05"),
}
TOP5 = {
    "total": ("S008_D_E", "S049_E_AU_PU", "S001_D", "S069_D_E_AU_PU", "S092_E_AU_PU_PW"),
    "pbmc": ("S003_L", "S021_L_PU", "S008_D_E", "S022_L_PW", "S059_L_PU_PW"),
    "buffycoat": ("S002_E", "S008_D_E", "S001_D", "S003_L", "S014_E_L"),
}
REPEATS = set(range(1, 101))
REQUIRED = (
    "06_outer_metrics.csv", "06_outer_predictions.csv.gz", "06_inner_tuning.csv.gz",
    "06_selected_hyperparameters.csv", "06_feature_explanations.csv.gz",
    "06_feature_fit_audit.csv.gz", "06_preprocessing_audit.csv.gz",
    "06_inner_assignment_audit.csv.gz", "06_run_manifest.json", "06_run_summary.md",
)
PRED_METRICS = ("roc_auc", "pr_auc", "accuracy", "sensitivity_recall", "specificity", "f1")


def need(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def boolcol(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin(("true", "yes", "1"))


def ints(values: pd.Series, label: str) -> pd.Series:
    v = pd.to_numeric(values, errors="raise")
    need(v.notna().all() and (v % 1 == 0).all(), f"{label}: non-integer repeat")
    return v.astype(int)


def check_repeats(df: pd.DataFrame, label: str, unique: bool = False) -> pd.DataFrame:
    need("outer_repeat" in df.columns, f"{label}: missing outer_repeat")
    df = df.copy()
    df["outer_repeat"] = ints(df["outer_repeat"], label)
    need(set(df["outer_repeat"]) == REPEATS, f"{label}: repeat 1..100 incomplete")
    if unique:
        need(df["outer_repeat"].is_unique and len(df) == 100, f"{label}: expected 100 unique repeats")
    return df


def validate_banks(repo: Path) -> dict:
    output = {}
    for scope, (_, n_holdout, dirname, expected_sha) in SCOPES.items():
        path = repo / BANK_REL / dirname / "repeated_holdout_assignments.csv"
        need(path.is_file() and sha256(path) == expected_sha, f"{scope}: frozen split SHA mismatch")
        df = pd.read_csv(path, dtype={"sample_id": str})
        need({"repeat_index", "sample_id", "role"}.issubset(df.columns), f"{scope}: bank columns missing")
        df["repeat_index"] = ints(df["repeat_index"], f"{scope} bank")
        need(set(df["repeat_index"]) == REPEATS, f"{scope}: bank repeats incomplete")
        holdouts = {}
        for rep, group in df.groupby("repeat_index"):
            need(group["sample_id"].is_unique, f"{scope} repeat {rep}: duplicate sample IDs")
            ids = set(group.loc[group["role"] == "holdout", "sample_id"])
            need(len(ids) == n_holdout, f"{scope} repeat {rep}: wrong holdout count")
            holdouts[int(rep)] = ids
        output[scope] = holdouts
    return output


def validate_locked_plan(repo: Path) -> list:
    ver = repo / VER_REL
    path = ver / "analysis/layer2_f1_only_plan_v3_175.json"
    need(path.is_file(), f"Locked F1-only plan missing: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    need(plan.get("decision") == "IGH_V3_LAYER2_F1_ONLY", "Wrong Layer2 plan decision")
    entries = plan.get("experiments", [])
    need(len(entries) == 15, f"Expected exactly 15 F1 entries; got {len(entries)}")
    jobmap = {}
    for entry in entries:
        scope = entry["scope"]
        stem = entry["name"]
        need(scope in SCOPES and stem.endswith("_F1"), f"Invalid plan job: {scope}/{stem}")
        base = stem[:-3]
        need(base in TOP5[scope], f"Not a pinned Top5 F1 job: {scope}/{stem}")
        expected_config = CFG_REL / f"igh_layer2_{scope}_top10000_v3_175" / f"{stem}.yaml"
        need(Path(entry["config"]) == expected_config, f"Config path mismatch: {scope}/{stem}")
        cfg = repo / expected_config
        need(cfg.is_file() and sha256(cfg) == entry["config_sha256"], f"Config hash mismatch: {cfg}")
        need(entry["frozen_split_sha256"] == SCOPES[scope][3], f"Split hash mismatch in plan: {scope}")
        obj = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        need(obj["features"]["kmer"]["enabled"] is True and
             obj["features"]["kmer"]["representation"] == "unweighted" and
             int(obj["features"]["kmer"]["top_n"]) == 500 and
             obj["features"]["enriched_dictionary"]["enabled"] is False,
             f"F1-only feature contract mismatch: {scope}/{stem}")
        key = (scope, stem)
        need(key not in jobmap, f"Duplicate locked plan entry: {key}")
        jobmap[key] = entry
    expected = {(scope, stem + "_F1") for scope, stems in TOP5.items() for stem in stems}
    need(set(jobmap) == expected, "Locked plan differs from exactly 15 pinned Top5 F1 jobs")
    return [jobmap[(scope, base + "_F1")] for scope, names in TOP5.items() for base in names]


def model_data(run: Path, label: str, scope: str, holdouts: dict, is_f1: bool):
    need(run.is_dir(), f"{label}: model output directory missing: {run}")
    missing = [x for x in REQUIRED if not (run / x).is_file()]
    need(not missing, f"{label}: missing output files: {missing}")
    m = check_repeats(pd.read_csv(run / "06_outer_metrics.csv"), f"{label} metrics", True)
    p = check_repeats(pd.read_csv(run / "06_outer_predictions.csv.gz", dtype={"sample_id": str}), f"{label} predictions")
    s = check_repeats(pd.read_csv(run / "06_selected_hyperparameters.csv"), f"{label} selected", True)
    c = check_repeats(pd.read_csv(run / "06_feature_explanations.csv.gz"), f"{label} coefficients")
    tune = pd.read_csv(run / "06_inner_tuning.csv.gz", usecols=["outer_repeat"])
    need(len(tune) == 2400 and set(ints(tune["outer_repeat"], label + " tuning")) == REPEATS,
         f"{label}: expected 2400 tuning rows across 100 repeats")
    need({"n_outer_train", "n_outer_holdout", "fit_converged", "iterations_used", "n_final_predictors", *PRED_METRICS}.issubset(m.columns),
         f"{label}: unexpected metric schema")
    need((pd.to_numeric(m["n_outer_train"]) == SCOPES[scope][0]).all() and
         (pd.to_numeric(m["n_outer_holdout"]) == SCOPES[scope][1]).all(),
         f"{label}: train/holdout sizes wrong")
    need(boolcol(m["fit_converged"]).all(), f"{label}: nonconverged outer models")
    need(np.isfinite(m[list(PRED_METRICS)].to_numpy(dtype=float)).all(), f"{label}: nonfinite model metrics")
    need(len(p) == SCOPES[scope][1] * 100 and
         {"sample_id", "true_cohort", "prediction_score"}.issubset(p.columns),
         f"{label}: prediction count/schema wrong")
    need(np.isfinite(pd.to_numeric(p["prediction_score"]).to_numpy(dtype=float)).all(),
         f"{label}: nonfinite scores")
    # Must be same participants in both models and the frozen bank for each repeat.
    for rep, group in p.groupby("outer_repeat"):
        ids = group["sample_id"].tolist()
        need(len(ids) == SCOPES[scope][1] and len(set(ids)) == len(ids),
             f"{label}/repeat{rep}: duplicate/missing sample")
        need(set(ids) == holdouts[int(rep)], f"{label}/repeat{rep}: frozen holdout IDs mismatch")
    if is_f1:
        audit = check_repeats(pd.read_csv(run / "06_feature_fit_audit.csv.gz"), f"{label} feature audit")
        need({"feature_family", "fit_policy"}.issubset(audit.columns), f"{label}: invalid feature audit schema")
        k = audit.loc[audit["feature_family"].astype(str).eq("3mer")]
        need(set(k["outer_repeat"]) == REPEATS and len(k) >= 100,
             f"{label}: missing inner/outer 3mer fit audits")
        need(k["fit_policy"].astype(str).eq("fit_vocabulary_on_current_training_only").all(),
             f"{label}: 3mer fit policy mismatch")
        need(not audit["feature_family"].astype(str).eq("enriched_dictionary").any(),
             f"{label}: F2/F3 audit unexpectedly present")
    need({"feature_name", "nonzero", "absolute_coefficient"}.issubset(c.columns),
         f"{label}: coefficient schema mismatch")
    coeff = c.loc[c["feature_name"].astype(str).ne("__INTERCEPT__")].copy()
    coeff["chosen"] = boolcol(coeff["nonzero"]).astype(int)
    nonzero = coeff.groupby("outer_repeat")["chosen"].sum()
    need(set(nonzero.index) == REPEATS, f"{label}: missing coefficient repeats")
    score_unique = p.groupby("outer_repeat")["prediction_score"].nunique()
    m["n_nonzero"] = m["outer_repeat"].map(nonzero).astype(int)
    m["zero_model"] = m["n_nonzero"].eq(0)
    m["constant_score"] = m["outer_repeat"].map(score_unique).eq(1)
    need({"parameter_alpha", "parameter_lambda"}.issubset(s.columns), f"{label}: missing hyperparameter fields")
    m = m.merge(s[["outer_repeat", "parameter_alpha", "parameter_lambda"]], on="outer_repeat", validate="one_to_one")
    return m.set_index("outer_repeat").sort_index(), p, coeff


def summarise_pair(scope: str, base: str, rank: int, f0, f1) -> tuple[dict, pd.DataFrame]:
    m0, m1 = f0, f1
    need(m0.index.equals(m1.index), f"{scope}/{base}: repeat indices differ")
    df = pd.DataFrame({"scope": scope, "base_stem": base, "outer_repeat": m0.index.to_numpy()})
    for measure in PRED_METRICS:
        df[f"{measure}_F0"] = m0[measure].to_numpy(dtype=float)
        df[f"{measure}_F1"] = m1[measure].to_numpy(dtype=float)
        df[f"delta_{measure}"] = df[f"{measure}_F1"] - df[f"{measure}_F0"]
    for label, m in (("F0", m0), ("F1", m1)):
        df[f"n_final_predictors_{label}"] = m["n_final_predictors"].to_numpy(dtype=int)
        df[f"n_nonzero_{label}"] = m["n_nonzero"].to_numpy(dtype=int)
        df[f"zero_model_{label}"] = m["zero_model"].to_numpy(dtype=bool)
        df[f"constant_score_{label}"] = m["constant_score"].to_numpy(dtype=bool)
        df[f"parameter_alpha_{label}"] = m["parameter_alpha"].to_numpy(dtype=float)
        df[f"parameter_lambda_{label}"] = m["parameter_lambda"].to_numpy(dtype=float)
    info = {"scope": scope, "layer1_rank": rank, "base_stem": base,
            "feature_groups": "+".join(base.split("_")[1:]), "f1_stem": base + "_F1",
            "n_paired_repeats": len(df)}
    for measure in ("roc_auc", "pr_auc"):
        for arm in ("F0", "F1"):
            vals = df[f"{measure}_{arm}"]
            info[f"{measure}_{arm}_mean"] = float(vals.mean())
            info[f"{measure}_{arm}_sd"] = float(vals.std(ddof=1))
        delta = df[f"delta_{measure}"]
        info[f"delta_{measure}_mean"] = float(delta.mean())
        info[f"delta_{measure}_median"] = float(delta.median())
        info[f"delta_{measure}_sd"] = float(delta.std(ddof=1))
        info[f"delta_{measure}_q025"] = float(delta.quantile(.025))
        info[f"delta_{measure}_q975"] = float(delta.quantile(.975))
        info[f"fraction_{measure}_F1_better"] = float((delta > 0).mean())
        info[f"fraction_{measure}_F1_equal"] = float((delta == 0).mean())
        info[f"fraction_{measure}_F1_worse"] = float((delta < 0).mean())
    for arm in ("F0", "F1"):
        info[f"zero_coefficient_rate_{arm}"] = float(df[f"zero_model_{arm}"].mean())
        info[f"constant_score_rate_{arm}"] = float(df[f"constant_score_{arm}"].mean())
        info[f"n_nonzero_mean_{arm}"] = float(df[f"n_nonzero_{arm}"].mean())
        info[f"n_predictors_median_{arm}"] = float(df[f"n_final_predictors_{arm}"].median())
        info[f"median_lambda_{arm}"] = float(df[f"parameter_lambda_{arm}"].median())
    return info, df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", help="RA-ILD repo root")
    ap.add_argument("--output-dir", help="New output directory (must not already exist)")
    args = ap.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    need((repo / "IGH/set").is_dir(), f"Not a RA-ILD repository: {repo}")
    ver = repo / VER_REL
    print("[1/4] Validating 15-job locked F1-only plan and 3 frozen banks...", flush=True)
    jobs = validate_locked_plan(repo)
    banks = validate_banks(repo)
    summary, paired, selection = [], [], []
    print("[2/4] Reading and validating 15 x (F0, F1) 100-repeat experiment pairs...", flush=True)
    for index, job in enumerate(jobs, 1):
        scope, stem = job["scope"], job["name"]
        base = stem[:-3]
        name = f"{scope}/{stem}"
        f0_dir = ver / "runs" / f"layer1_{scope}_top10000" / base / "repeat100"
        f1_dir = ver / "runs" / f"layer2_{scope}_top10000" / stem / "repeat100"
        m0, p0, _ = model_data(f0_dir, name + "/F0", scope, banks[scope], False)
        m1, p1, c1 = model_data(f1_dir, name + "/F1", scope, banks[scope], True)
        # Stronger-than-repeat-level check: same patient and labels in paired holdouts.
        cols = ["outer_repeat", "sample_id", "true_cohort"]
        a = p0.loc[:, cols].sort_values(cols[:2]).reset_index(drop=True)
        b = p1.loc[:, cols].sort_values(cols[:2]).reset_index(drop=True)
        need(a.equals(b), f"{name}: F0/F1 holdout patient/label pairs differ")
        info, detail = summarise_pair(scope, base, TOP5[scope].index(base) + 1, m0, m1)
        summary.append(info)
        paired.append(detail)
        # Full per-feature selected frequency, retains raw feature names to avoid
        # incorrectly inferring the feature prefix used by Unified ML.
        grouped = c1.groupby("feature_name", as_index=False).agg(
            n_selected=("chosen", "sum"), n_observed=("chosen", "size"),
            mean_abs_coefficient=("absolute_coefficient", "mean"),
        )
        grouped["selection_rate"] = grouped["n_selected"] / 100.0
        grouped.insert(0, "scope", scope)
        grouped.insert(1, "f1_stem", stem)
        selection.append(grouped)
        print(f"  [{index:02d}/15] PASS {name}: mean ΔAUC={info['delta_roc_auc_mean']:+.4f}, "
              f"mean ΔPR-AUC={info['delta_pr_auc_mean']:+.4f}", flush=True)
    need(len(summary) == 15, "Expected exactly 15 paired comparisons")
    full = pd.DataFrame(summary)
    reps = pd.concat(paired, ignore_index=True)
    sels = pd.concat(selection, ignore_index=True)
    need(len(reps) == 1500, f"Expected 1500 paired repeats, got {len(reps)}")
    print("[3/4] Ranking F1-only candidates and comparing within-repeat gains...", flush=True)
    full["rank_F1_auc_within_top5"] = full.groupby("scope")["roc_auc_F1_mean"].rank(
        method="first", ascending=False).astype(int)
    full["rank_gain_auc_within_top5"] = full.groupby("scope")["delta_roc_auc_mean"].rank(
        method="first", ascending=False).astype(int)
    full = full.sort_values(["scope", "rank_F1_auc_within_top5"])
    # If the original 381-row summary is available, verify all F0 means match.
    original = ver / "analysis/layer1_summary_20261114T090147Z/01_experiment_summary_381.csv"
    if original.is_file():
        l1 = pd.read_csv(original)
        for _, row in full.iterrows():
            matching = l1[(l1["scope"] == row["scope"]) & (l1["stem"] == row["base_stem"])]
            need(len(matching) == 1, f"Layer1 original summary missing {row['scope']}/{row['base_stem']}")
            for metric in ("roc_auc", "pr_auc"):
                need(abs(float(matching.iloc[0][metric + "_mean"]) - row[metric + "_F0_mean"]) < 1e-8,
                     f"Original Layer1 summary differs for {row['scope']}/{row['base_stem']}/{metric}")
    out = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
           ver / "analysis" / ("layer2_f1_summary_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    need(not out.exists(), f"Refusing to overwrite analysis: {out}")
    print("[4/4] Writing NEW analysis-only outputs...", flush=True)
    out.mkdir(parents=True)
    full.to_csv(out / "01_f1_vs_f0_summary_15.csv", index=False)
    reps.to_csv(out / "02_paired_per_repeat_1500.csv.gz", index=False, compression="gzip")
    sels.to_csv(out / "03_f1_feature_selection_frequency.csv.gz", index=False, compression="gzip")
    report = ["# IGH V3 Layer 2: F1-only vs F0 (paired frozen holdouts)", "",
              "15 static Top5 + training-fold Top500 unweighted 3-mer experiments, 100 matched repeats per experiment.",
              "F2/F3 are excluded from the locked V3 IGH Layer2 plan.", "",
              "**Statistical limits:** Each repeat reuses the same patients. These repeat distributions are descriptive, NOT 100 independent samples or 95% confidence intervals. "
              "The Layer1 Top5 were selected on overlapping holdout results, so neither winning F1 AUC nor observed gains constitute fully independent post-selection validation.", "",
              "## Comparison by cohort", ""]
    shown = ["layer1_rank", "base_stem", "feature_groups", "roc_auc_F0_mean", "roc_auc_F1_mean",
             "delta_roc_auc_mean", "fraction_roc_auc_F1_better", "pr_auc_F0_mean", "pr_auc_F1_mean",
             "delta_pr_auc_mean", "zero_coefficient_rate_F0", "zero_coefficient_rate_F1"]
    for scope in SCOPES:
        d = full.loc[full.scope == scope].sort_values("rank_F1_auc_within_top5")
        need(len(d) == 5, f"{scope}: expected exactly five F1 results")
        d.to_csv(out / f"ranking_{scope}_5.csv", index=False)
        txt = d[shown].to_string(index=False, float_format=lambda x: f"{x:+.4f}")
        print(f"\n=== {scope.upper()} F1 TOP5 (by mean ROC-AUC) ===\n{txt}", flush=True)
        report.extend([f"### {scope}", "", "```text", txt, "```", ""])
    report.extend(["## Files", "", "- 01_f1_vs_f0_summary_15.csv: all 15 matched experiment comparisons",
                   "- 02_paired_per_repeat_1500.csv.gz: all paired holdout metrics by repeat",
                   "- 03_f1_feature_selection_frequency.csv.gz: F1 coefficients and selection frequency by feature",
                   "- ranking_{scope}_5.csv: cohort-specific F1 top-five rankings", "",
                   "Primary comparison: per-repeat ROC-AUC and PR-AUC F1 minus the identical-split F0.",
                   "Empty/constant classifiers are reported as zero_coefficient_rate and constant_score_rate.", ""])
    (out / "README_summary.md").write_text("\n".join(report), encoding="utf-8")
    audit = {
        "status": "PASS", "policy": "F1_ONLY", "matched_experiments": 15,
        "matched_repeat_pairs": 1500, "checked_model_outputs": 30,
        "scope_experiments": {scope: 5 for scope in SCOPES},
        "frozen_split_sha256": {scope: value[3] for scope, value in SCOPES.items()},
        "locked_plan_sha256": sha256(ver / "analysis/layer2_f1_only_plan_v3_175.json"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "00_validation_report.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nIGH V3 LAYER2 F1-ONLY SUMMARY PASS: 15 paired experiments / 1500 matched repeats", flush=True)
    print("OUTPUT_DIR:", out, flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
