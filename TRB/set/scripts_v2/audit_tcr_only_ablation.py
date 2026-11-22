#!/usr/bin/env python3
"""Audit that three TCR-only configs change only id and age/sex predictors."""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "TRB/set/configs/layered_feature_experiments/tcr_only_ablation/TCR_ONLY_ABLATION_MANIFEST.json"

def audit(root: Path):
    manifest = json.loads((root / MANIFEST.relative_to(ROOT)).read_text(encoding="utf-8"))
    expected = {
        "total": ("L1_007_PW", ["tcr_physicochemical_weighted"], 179),
        "pbmc": ("L1_001_D", ["tcr_diversity"], 116),
        "buffycoat": ("L1_017_E_PU", ["tcr_clonal_expansion", "tcr_physicochemical_unweighted"], 63),
    }
    rows = manifest["entries"]
    if len(rows) != 3 or {x["cohort"] for x in rows} != set(expected):
        raise ValueError("Expected exactly 3 cohorts")
    for e in rows:
        cohort = e["cohort"]
        combo, groups, n = expected[cohort]
        if e["combination_id"] != combo or e["feature_groups"] != groups:
            raise ValueError(f"Unexpected best TCR groups: {cohort}")
        if e["expected_samples"] != n or e["algorithm"] != "elastic_net":
            raise ValueError(f"Unexpected sample count or algorithm: {cohort}")
        source = yaml.safe_load((root / e["base_config"]).read_text(encoding="utf-8"))
        actual = yaml.safe_load((root / e["config"]).read_text(encoding="utf-8"))
        reference = copy.deepcopy(source)
        reference["experiment"]["id"] = e["experiment_id"]
        reference["features"]["clinical_groups"] = []
        if actual != reference:
            raise ValueError(f"{cohort}: unexpected change beyond id and age/sex removal")
        if actual["features"]["static_tcr_groups"] != groups:
            raise ValueError(f"{cohort}: TCR groups mismatch")
        if actual["features"]["kmer"]["enabled"] or actual["features"]["enriched_dictionary"]["enabled"]:
            raise ValueError(f"{cohort}: extra feature enabled")
        if actual["repertoire"] != {"mode": "topk", "top_k": 10000}:
            raise ValueError(f"{cohort}: repertoire mismatch")
        if actual["validation"]["repeats"] != 100 or actual["validation"]["folds"] != 5:
            raise ValueError(f"{cohort}: CV mismatch")
        if actual["validation"]["split_assignments"] != e["split_assignments"]:
            raise ValueError(f"{cohort}: split path mismatch")
        if not (root / e["split_assignments"]).is_file():
            raise FileNotFoundError(e["split_assignments"])
        print(f"[PASS] {cohort}: age/sex removed, TCR={groups}, Repeat100, independent tuning")
    print("ALL THREE ABLATION CONFIGS PASSED")
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repository-root",type=Path,default=ROOT)
    args=ap.parse_args()
    audit(args.repository_root.resolve())
    return 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ABLATION AUDIT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
