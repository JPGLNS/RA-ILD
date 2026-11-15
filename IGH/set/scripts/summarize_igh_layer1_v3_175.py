#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only validation and analysis of IGH V3 175 Layer 1 (381 x 100).

Requires pandas, numpy, PyYAML. Never modifies modeling inputs or results.
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

SCOPE = {
    "total": (175, 122, 53, "igh_ra_ild_repeat100_v3_175", "455eba17c74a86667fd2939dc8645ac1c50f0cdb25053ab951dccad0ab8f1788"),
    "pbmc": (113, 79, 34, "igh_ra_ild_pbmc_repeat100_v3_175", "4726d9b48301c4cd535369f9b39f6bdf3c6247c0722b9c9663385a6c041a68b8"),
    "buffycoat": (62, 43, 19, "igh_ra_ild_buffycoat_repeat100_v3_175", "da1e598559dc734c0e0296bd1f99620351c2d81e903100c85fda23b670047c05"),
}
GROUPS = {
    "tcr_diversity": "D",
    "tcr_clonal_expansion": "E",
    "tcr_aa_length": "L",
    "tcr_aa_composition_unweighted": "AU",
    "tcr_aa_composition_weighted": "AW",
    "tcr_physicochemical_unweighted": "PU",
    "tcr_physicochemical_weighted": "PW",
}
EXPECTED = set(range(1, 101))
REQUIRED = [
    "06_outer_metrics.csv", "06_outer_predictions.csv.gz",
    "06_inner_tuning.csv.gz", "06_selected_hyperparameters.csv",
    "06_feature_explanations.csv.gz", "06_feature_fit_audit.csv.gz",
    "06_preprocessing_audit.csv.gz", "06_inner_assignment_audit.csv.gz",
    "06_run_manifest.json", "06_run_summary.md",
]
METRICS = ["roc_auc", "pr_auc", "accuracy", "sensitivity_recall", "specificity", "f1", "log_loss", "brier_score"]


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def positive(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "yes", "1"])


def unique_repeats(df, name):
    check("outer_repeat" in df.columns, f"{name}: outer_repeat missing")
    s = pd.to_numeric(df["outer_repeat"], errors="raise").astype(int)
    check(set(s) == EXPECTED, f"{name}: repeat indices not 1..100")
    return s


def source_banks(repo):
    banks = {}
    for scope, (n, train_n, holdout_n, name, expected_sha) in SCOPE.items():
        split_dir = repo / "IGH/set/split_sets" / name
        bank = split_dir / "repeated_holdout_assignments.csv"
        marker = json.loads((split_dir / "SPLITS_FROZEN.json").read_text(encoding="utf-8"))
        digest = sha256(bank)
        check(digest == expected_sha, f"{scope}: frozen bank SHA mismatch")
        check(marker["files"]["assignments"]["sha256"] == digest, f"{scope}: marker SHA mismatch")
        df = pd.read_csv(bank, dtype={"sample_id": str})
        df["repeat_index"] = pd.to_numeric(df["repeat_index"], errors="raise").astype(int)
        check(set(df["repeat_index"]) == EXPECTED, f"{scope}: bank repeats incomplete")
        holdouts = {}
        for rep, g in df.groupby("repeat_index", sort=True):
            check(len(g) == n and g["sample_id"].is_unique, f"{scope}/{rep}: bank sample count/duplicates")
            check((g["role"] == "train").sum() == train_n and (g["role"] == "holdout").sum() == holdout_n,
                  f"{scope}/{rep}: wrong train/holdout counts")
            holdouts[int(rep)] = set(g.loc[g["role"] == "holdout", "sample_id"])
        banks[scope] = holdouts
    return banks


def load_experiment(repo, scope, yaml_file, banks):
    _, train_n, holdout_n, _, _ = SCOPE[scope]
    stem = yaml_file.stem
    cfg = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    exp_id = f"igh_layer1_{scope}_top10000_{stem}_v3_175"
    check(cfg["experiment"]["id"] == exp_id, f"{yaml_file}: experiment ID mismatch")
    group_names = cfg["features"]["static_tcr_groups"]
    check(len(group_names) == len(set(group_names)) and set(group_names).issubset(GROUPS), f"{stem}: group mismatch")
    codes = [GROUPS[g] for g in group_names]
    run = repo / "IGH/set/data_versions/igh_v3_175/runs" / f"layer1_{scope}_top10000" / stem / "repeat100"
    missing = [x for x in REQUIRED if not (run / x).is_file()]
    check(not missing, f"{scope}/{stem}: missing {missing}")
    m = pd.read_csv(run / "06_outer_metrics.csv")
    p = pd.read_csv(run / "06_outer_predictions.csv.gz", dtype={"sample_id": str})
    s = pd.read_csv(run / "06_selected_hyperparameters.csv")
    t = pd.read_csv(run / "06_inner_tuning.csv.gz", usecols=["outer_repeat"])
    c = pd.read_csv(run / "06_feature_explanations.csv.gz")
    check(len(m) == len(s) == 100, f"{scope}/{stem}: metric/selected != 100")
    check(len(p) == holdout_n * 100, f"{scope}/{stem}: prediction count wrong")
    check(len(t) == 2400, f"{scope}/{stem}: tuning != 2400")
    for name, frame in [("metrics", m), ("selected", s), ("pred", p), ("tuning", t), ("coeff", c)]:
        unique_repeats(frame, f"{scope}/{stem}/{name}")
    check(m["outer_repeat"].is_unique and s["outer_repeat"].is_unique, f"{scope}/{stem}: duplicate repeat metrics/selected")
    check((pd.to_numeric(m["n_outer_train"]) == train_n).all(), f"{scope}/{stem}: wrong train n")
    check((pd.to_numeric(m["n_outer_holdout"]) == holdout_n).all(), f"{scope}/{stem}: wrong holdout n")
    check(positive(m["fit_converged"]).all(), f"{scope}/{stem}: nonconverged final fit")
    check(p["prediction_score"].notna().all(), f"{scope}/{stem}: NaN predictions")
    check(m[METRICS].notna().all().all(), f"{scope}/{stem}: NaN metric(s)")
    for rep, g in p.groupby("outer_repeat", sort=True):
        check(len(g) == holdout_n and g["sample_id"].is_unique,
              f"{scope}/{stem}/{rep}: prediction count or duplicates")
        check(set(g["sample_id"]) == banks[scope][int(rep)], f"{scope}/{stem}/{rep}: holdout IDs mismatch")

    non_intercept = c.loc[c["feature_name"].ne("__INTERCEPT__")].copy()
    non_intercept["nonzero_flag"] = positive(non_intercept["nonzero"]).astype(int)
    num_active = non_intercept.groupby("outer_repeat")["nonzero_flag"].sum()
    check(set(num_active.index) == EXPECTED, f"{scope}/{stem}: missing coefficient repeats")
    score_unique = p.groupby("outer_repeat")["prediction_score"].nunique()
    pos_frac = p.groupby("outer_repeat")["true_cohort"].apply(lambda x: (x.astype(str).str.upper() == "ILD").mean())
    m = m.merge(s[["outer_repeat", "parameter_alpha", "parameter_lambda"]], on="outer_repeat", how="left", validate="one_to_one")
    for key, val in [("n_nonzero", num_active), ("score_unique", score_unique), ("holdout_prevalence", pos_frac)]:
        m[key] = m["outer_repeat"].map(val)
    check(m[["n_nonzero", "score_unique", "holdout_prevalence"]].notna().all().all(), f"{scope}/{stem}: missing audit data")
    m["zero_coefficient_model"] = m["n_nonzero"].eq(0)
    m["constant_score"] = m["score_unique"].eq(1)
    m["pr_auc_minus_prevalence"] = m["pr_auc"] - m["holdout_prevalence"]
    m["pr_auc_lift"] = m["pr_auc"] / m["holdout_prevalence"].replace(0, np.nan)
    m.insert(0, "stem", stem)
    m.insert(0, "scope", scope)
    m.insert(2, "n_groups", len(group_names))
    m.insert(3, "group_codes", "+".join(codes))
    m.insert(4, "group_names", "+".join(group_names))
    m.insert(5, "experiment_id", exp_id)

    summary = {"scope": scope, "stem": stem, "experiment_id": exp_id,
               "n_groups": len(group_names), "group_codes": "+".join(codes),
               "group_names": "+".join(group_names), "n_predictors_configured": int(m["n_final_predictors"].median()),
               "n_repeats": len(m), "n_train": train_n, "n_holdout": holdout_n}
    for metric in METRICS:
        v = pd.to_numeric(m[metric], errors="raise")
        summary[metric + "_mean"] = float(v.mean())
        summary[metric + "_sd"] = float(v.std(ddof=1))
        summary[metric + "_median"] = float(v.median())
        summary[metric + "_q025"] = float(v.quantile(.025))
        summary[metric + "_q975"] = float(v.quantile(.975))
    summary.update({
        "zero_coefficient_model_rate": float(m["zero_coefficient_model"].mean()),
        "constant_score_rate": float(m["constant_score"].mean()),
        "n_nonzero_mean": float(m["n_nonzero"].mean()),
        "convergence_rate": float(positive(m["fit_converged"]).mean()),
        "max_iterations": int(m["iterations_used"].max()),
        "median_lambda": float(m["parameter_lambda"].median()),
        "high_lambda_ge10_rate": float((m["parameter_lambda"] >= 10).mean()),
        "holdout_prevalence_mean": float(m["holdout_prevalence"].mean()),
        "pr_auc_minus_prevalence_mean": float(m["pr_auc_minus_prevalence"].mean()),
        "pr_auc_lift_mean": float(m["pr_auc_lift"].mean()),
    })
    selections = non_intercept.groupby("feature_name", as_index=False).agg(
        n_selected=("nonzero_flag", "sum"),
        n_observed=("nonzero_flag", "size"),
        mean_abs_coefficient=("absolute_coefficient", "mean"),
    )
    selections.insert(0, "stem", stem)
    selections.insert(0, "scope", scope)
    selections["selection_rate"] = selections["n_selected"] / 100
    return summary, m, selections


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", help="Root of RA-ILD Git repository (default: current directory)")
    ap.add_argument("--output-dir", default=None, help="New, NONEXISTENT output directory; default timestamped inside V3 analysis")
    args = ap.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    check((repo / "IGH/set").is_dir(), f"Not an RA-ILD repo: {repo}")
    print("[1/4] Validating the three frozen banks...", flush=True)
    banks = source_banks(repo)
    summaries, repeats, selections, errors = [], [], [], []
    print("[2/4] Reading 381 experiments and validating their output...", flush=True)
    for scope in SCOPE:
        cfg_dir = repo / "IGH/set/configs/experiments" / f"igh_layer1_{scope}_top10000_v3_175"
        configs = sorted(cfg_dir.glob("S*.yaml"))
        check(len(configs) == 127, f"{scope}: need 127 configs, found {len(configs)}")
        for i, cfg in enumerate(configs, 1):
            try:
                summary, rep, sel = load_experiment(repo, scope, cfg, banks)
                summaries.append(summary)
                repeats.append(rep)
                selections.append(sel)
            except Exception as exc:
                errors.append({"scope": scope, "stem": cfg.stem, "error": str(exc)})
            if i % 25 == 0 or i == 127:
                print(f"  {scope}: {i}/127 checked", flush=True)
    if errors:
        print(f"ERROR: {len(errors)} failed validation(s). No ranking files written.", file=sys.stderr)
        for item in errors[:30]:
            print(item, file=sys.stderr)
        return 2
    check(len(summaries) == 381, f"Expected 381, got {len(summaries)}")
    all_sum = pd.DataFrame(summaries)
    all_rep = pd.concat(repeats, ignore_index=True)
    all_sel = pd.concat(selections, ignore_index=True)
    check(len(all_rep) == 38100, f"Expected 38100 repeat rows, got {len(all_rep)}")
    for scope in SCOPE:
        check(len(set(all_sum.loc[all_sum.scope == scope, "group_codes"])) == 127,
              f"{scope}: duplicate group combinations")

    print("[3/4] Computing within-split paired comparisons and group-addition effects...", flush=True)
    pair_rows = []
    effect_rows = []
    for scope in SCOPE:
        rs = all_rep[all_rep.scope == scope]
        ss = all_sum[all_sum.scope == scope]
        base = rs[rs.stem == "S001_D"].set_index("outer_repeat")
        check(len(base) == 100, f"{scope}: S001_D missing")
        for stem, g in rs.groupby("stem", sort=False):
            cur = g.set_index("outer_repeat")
            delta_auc = cur["roc_auc"].sort_index() - base["roc_auc"].sort_index()
            delta_ap = cur["pr_auc"].sort_index() - base["pr_auc"].sort_index()
            info = ss.loc[ss.stem == stem].iloc[0]
            pair_rows.append({"scope": scope, "stem": stem,
                              "group_codes": info["group_codes"],
                              "delta_auc_mean_vs_D": float(delta_auc.mean()),
                              "delta_auc_median_vs_D": float(delta_auc.median()),
                              "delta_auc_sd_vs_D": float(delta_auc.std(ddof=1)),
                              "fraction_repeats_better_auc_than_D": float((delta_auc > 0).mean()),
                              "delta_pr_auc_mean_vs_D": float(delta_ap.mean()),
                              "fraction_repeats_better_pr_auc_than_D": float((delta_ap > 0).mean())})
        lookup = {frozenset(x.split("+")): stem for x, stem in zip(ss.group_codes, ss.stem)}
        by_stem = {stem: g.set_index("outer_repeat") for stem, g in rs.groupby("stem")}
        for feature in GROUPS.values():
            comparisons = []
            for base_set, a in lookup.items():
                if feature in base_set:
                    continue
                b = lookup[frozenset(set(base_set) | {feature})]
                a_df, b_df = by_stem[a], by_stem[b]
                comparisons.append((
                    float((b_df["roc_auc"] - a_df["roc_auc"]).mean()),
                    float((b_df["pr_auc"] - a_df["pr_auc"]).mean()),
                ))
            check(len(comparisons) == 63, f"{scope}/{feature}: expected 63 group-addition contrasts")
            vals = np.array(comparisons)
            effect_rows.append({"scope": scope, "added_group": feature,
                                "n_paired_combinations": len(comparisons),
                                "mean_delta_auc_when_added": float(vals[:, 0].mean()),
                                "median_delta_auc_when_added": float(np.median(vals[:, 0])),
                                "fraction_positive_auc_contrasts": float((vals[:, 0] > 0).mean()),
                                "mean_delta_pr_auc_when_added": float(vals[:, 1].mean()),
                                "fraction_positive_pr_auc_contrasts": float((vals[:, 1] > 0).mean())})

    print("[4/4] Writing analysis-only CSVs and short report...", flush=True)
    all_sum = all_sum.sort_values(["scope", "roc_auc_mean", "pr_auc_mean", "zero_coefficient_model_rate", "n_groups"],
                                  ascending=[True, False, False, True, True])
    all_sum["rank_auc_in_scope"] = all_sum.groupby("scope").cumcount() + 1
    out = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
           repo / "IGH/set/data_versions/igh_v3_175/analysis" / (
               "layer1_summary_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    check(not out.exists(), f"Refusing to overwrite existing analysis: {out}")
    out.mkdir(parents=True)
    all_sum.to_csv(out / "01_experiment_summary_381.csv", index=False)
    all_rep.to_csv(out / "02_per_repeat_38100.csv.gz", index=False, compression="gzip")
    pd.DataFrame(pair_rows).to_csv(out / "03_paired_vs_diversity.csv", index=False)
    pd.DataFrame(effect_rows).to_csv(out / "04_group_addition_effects.csv", index=False)
    all_sel.to_csv(out / "05_feature_selection_frequency.csv.gz", index=False, compression="gzip")
    report = [
        "# IGH V3 (175 samples) Layer 1 — complete analysis",
        "", "Validation: 381/381 experiment outputs validated, 38,100 outer repeats, frozen holdout IDs matched.",
        "", "**Exploratory rankings only.** Selecting the best from 127 models using the same outer holdouts can give optimistically biased performance. Do not treat these as independently validated final AUCs.",
        "", "## Best candidates per scope (ranked by mean ROC-AUC)", "",
    ]
    shown = ["rank_auc_in_scope", "stem", "group_codes", "roc_auc_mean", "roc_auc_sd", "pr_auc_mean",
             "holdout_prevalence_mean", "zero_coefficient_model_rate", "convergence_rate"]
    for scope in SCOPE:
        table = all_sum.loc[all_sum.scope == scope].copy()
        table.to_csv(out / f"ranking_{scope}_127.csv", index=False)
        top = table.head(10)[shown]
        headers = list(top.columns)
        rows_md = ["| " + " | ".join(headers) + " |",
                   "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in top.iterrows():
            cells = [f"{row[col]:.4f}" if isinstance(row[col], (float, np.floating)) else str(row[col]) for col in headers]
            rows_md.append("| " + " | ".join(cells) + " |")
        report.extend([f"### {scope} (top 10)", "", *rows_md, ""])
        print(f"\n=== {scope.upper()} TOP 10 by mean ROC-AUC ===")
        print(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    report.extend(["## Interpretation notes", "",
                   "- The 100 outer repeats reuse participants; their SD and per-repeat differences are descriptive, not 100 independent-sample tests.",
                   "- PR-AUC should be interpreted against the ILD holdout prevalence for each cohort.",
                   "- `zero_coefficient_model_rate` is the share of intercept-only Elastic Net fits.",
                   "- `03_paired_vs_diversity.csv` compares experiments with S001_D using identical outer holdout assignments.",
                   "- `04_group_addition_effects.csv` averages each group's addition over 63 nonempty background combinations; descriptive, not causal.",
                   "- Layer 2 design and independent/nested selection assessment remain separate tasks.", ""])
    (out / "README_summary.md").write_text("\n".join(report), encoding="utf-8")
    audit = {"status": "PASS", "experiments": 381, "repeats": 38100,
             "scopes": {s: int((all_sum.scope == s).sum()) for s in SCOPE},
             "created_utc": datetime.now(timezone.utc).isoformat(), "output": str(out),
             "frozen_sha256": {s: x[4] for s, x in SCOPE.items()}}
    (out / "00_validation_report.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nIGH V3 LAYER1 SUMMARY PASS: 381 experiments / 38100 repeats")
    print("OUTPUT_DIR:", out)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
