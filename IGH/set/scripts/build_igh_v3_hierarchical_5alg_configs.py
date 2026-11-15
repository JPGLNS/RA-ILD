#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and validate the frozen IGH V3 hierarchical 5-algorithm YAML plan.

By default this is a read-only dry run (--check).  --apply writes only new
configs under a dedicated directory and refuses to overwrite changed files.
No models are trained and no original YAML, data, or split bank is changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required") from exc

BRANCH = "feature/igh-v3-hierarchical-5alg"
ALGOS = ("elastic_net", "ridge", "lasso", "linear_svm", "xgboost")
SCOPE_MODELS = {
    "total": ("M0", "M1", "M2", "M3"),
    "pbmc": ("M0", "M1", "M2"),
    "buffycoat": ("M0", "M1", "M2"),
}
CORE83_GROUPS = (
    "tcr_diversity",
    "tcr_clonal_expansion",
    "tcr_aa_length",
    "tcr_aa_composition_unweighted",
    "tcr_aa_composition_weighted",
    "tcr_physicochemical_unweighted",
    "tcr_physicochemical_weighted",
)
SPLITS = {
    "total": ("igh_ra_ild_repeat100_v3_175", "455eba17c74a"),
    "pbmc": ("igh_ra_ild_pbmc_repeat100_v3_175", "4726d9b48301"),
    "buffycoat": ("igh_ra_ild_buffycoat_repeat100_v3_175", "da1e598559dc"),
}
OUTPUT_REL = Path("IGH/set/configs/experiments/igh_v3_hierarchical_5alg_top10000_v3_175")

# Values from the Phase-2B PublicReferenceSpec configuration defaults.
PUBLIC_PARAMETERS = {
    "feature_set": "raw_bilateral",  # exactly eight dynamic Public features
    "global_min_prevalence": 0.02,
    "global_min_count": 3,
    "group_min_prevalence": 0.05,
    "group_min_count": 3,
    "specific_prevalence_delta": 0.05,
    "epsilon": 1.0e-8,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_inputs(root: Path) -> None:
    import subprocess

    require((root / ".git").exists(), f"Not a Git worktree: {root}")
    branch = subprocess.check_output(
        ["git", "-C", str(root), "branch", "--show-current"], text=True,
    ).strip()
    require(branch == BRANCH, f"Wrong branch {branch!r}; expected {BRANCH!r}")
    print(f"[GIT PASS] {branch}")

    for scope, (bank_name, prefix) in SPLITS.items():
        bank = root / "IGH/set/split_sets" / bank_name / "repeated_holdout_assignments.csv"
        require(bank.is_file(), f"Missing frozen split bank: {bank}")
        digest = sha256_file(bank)
        require(digest.startswith(prefix),
                f"Frozen split SHA mismatch: {scope}; expected={prefix} actual={digest}")
        print(f"[SPLIT HASH PASS] {scope}: {digest}")

    registry = root / "IGH/set/configs/feature_framework/igh_feature_registry_v3_175.yaml"
    require(registry.is_file(), f"Missing Core83 registry: {registry}")
    groups = yaml.safe_load(registry.read_text(encoding="utf-8"))["feature_groups"]
    names = []
    for group_name in CORE83_GROUPS:
        require(group_name in groups, f"Missing Core83 group {group_name}")
        group = groups[group_name]
        selector = group["selector"]
        require(not selector.get("prefixes"), f"Unexpected prefix selector {group_name}")
        excluded = set(group.get("reference_drop_columns", []))
        names.extend(x for x in selector.get("exact", []) if x not in excluded)
    require(len(names) == len(set(names)) == 83,
            f"Core83 registry mismatch: total={len(names)} unique={len(set(names))}")
    print("[CORE83 PASS] 7 frozen groups, 83 unique selectors")


def make_config(template: dict, scope: str, model: str, algorithm: str) -> dict:
    cfg = copy.deepcopy(template)
    cfg["experiment"]["id"] = (
        f"igh_v3_hierarchical_{scope}_{model.lower()}_{algorithm}_top10000_v3_175"
    )
    cfg["cohort"]["scope"] = scope
    cfg["repertoire"]["mode"] = "topk"
    cfg["repertoire"]["top_k"] = 10000

    features = cfg["features"]
    features["clinical_groups"] = ["clinical_age_sex"]
    if model == "M3":
        features["clinical_groups"].append("clinical_material")
    features["static_tcr_groups"] = list(CORE83_GROUPS) if model != "M0" else []
    features["kmer"]["enabled"] = model != "M0"
    features["kmer"]["k"] = 3
    features["kmer"]["representation"] = "both" if model != "M0" else "unweighted"
    features["kmer"]["top_n"] = 500
    features["enriched_dictionary"]["enabled"] = False
    features["public_reference"] = {
        "enabled": model in {"M2", "M3"},
        **PUBLIC_PARAMETERS,
    }

    cfg["model"]["algorithm"] = algorithm
    cfg["validation"]["folds"] = 5
    cfg["validation"]["repeats"] = 100
    cfg["validation"]["split_assignments"] = (
        f"IGH/set/split_sets/{SPLITS[scope][0]}/repeated_holdout_assignments.csv"
    )
    return cfg


def verify_config(cfg: dict, scope: str, model: str, algorithm: str, parsed, root: Path) -> None:
    require(parsed.algorithm == algorithm, f"Algorithm parse mismatch: {scope} {model} {algorithm}")
    base = parsed.base
    feat = base.features
    require(base.cohort.scope == scope, "Wrong scope")
    require(base.repertoire.source_id == "top10000", "Wrong repertoire")
    require(tuple(feat.clinical_groups) ==
            (("clinical_age_sex", "clinical_material") if model == "M3" else ("clinical_age_sex",)),
            "Clinical hierarchy mismatch")
    require(tuple(feat.static_tcr_groups) == (CORE83_GROUPS if model != "M0" else ()),
            "Core83 hierarchy mismatch")
    require(feat.kmer.enabled == (model != "M0"), "3-mer enabled mismatch")
    require(feat.kmer.representation == ("unweighted" if model == "M0" else "both"),
            "3-mer representation mismatch")
    require(feat.kmer.top_n == 500 and feat.kmer.k == 3, "3-mer Top500/k=3 mismatch")
    require(not feat.enriched_dictionary.enabled, "Legacy enriched dictionary must be disabled")
    require(feat.public_reference.enabled == (model in {"M2", "M3"}),
            "Public-reference hierarchy mismatch")
    require(feat.public_reference.feature_set == "raw_bilateral", "Public-reference feature_set mismatch")
    for field, expected in PUBLIC_PARAMETERS.items():
        actual = getattr(feat.public_reference, field)
        require(actual == expected, f"Public-reference parameter mismatch: {field} {actual} != {expected}")
    require(parsed.cv.repeats == 100 and parsed.cv.inner_folds == 5, "CV repeats/folds mismatch")
    require(cfg["model"]["positive_label"] == "ILD" and cfg["model"]["negative_label"] == "RA",
            "Unexpected label orientation")
    require(cfg["repertoire"] == {"mode": "topk", "top_k": 10000}, "Repertoire config mismatch")
    split_path = root / cfg["validation"]["split_assignments"]
    require(split_path.is_file(), f"Frozen split does not resolve: {split_path}")
    require(parsed.cv.split_assignments == cfg["validation"]["split_assignments"],
            "Resolved split field differs")


def build_plan(root: Path) -> dict[str, str]:
    sys.path.insert(0, str(root / "IGH/set/src"))
    from ra_ild_igh.unified_experiment import load_unified_experiment

    plan: dict[str, str] = {}
    identities = set()
    with tempfile.TemporaryDirectory(prefix="igh_v3_5alg_check_") as temp_dir:
        scratch = Path(temp_dir)
        for scope, models in SCOPE_MODELS.items():
            template_path = (root / "IGH/set/configs/experiments"
                             / f"igh_clinical_layer1_{scope}_top10000_v3_175" / "S000_none.yaml")
            require(template_path.is_file(), f"Missing M0 scope template: {template_path}")
            template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
            require(template["cohort"]["scope"] == scope, f"Template scope mismatch: {scope}")
            require(template["repertoire"] == {"mode": "topk", "top_k": 10000},
                    f"Template must be frozen Top10000: {scope}")
            require(template["validation"]["split_assignments"].endswith(
                f"/{SPLITS[scope][0]}/repeated_holdout_assignments.csv"),
                f"Template split bank mismatch for {scope}")
            for model in models:
                for algorithm in ALGOS:
                    cfg = make_config(template, scope, model, algorithm)
                    name = f"{scope}_{model}_{algorithm}.yaml"
                    tmp = scratch / name
                    content = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
                    tmp.write_text(content, encoding="utf-8")
                    parsed = load_unified_experiment(tmp)
                    verify_config(cfg, scope, model, algorithm, parsed, root)
                    identity = cfg["experiment"]["id"]
                    require(identity not in identities, f"Duplicate experiment ID: {identity}")
                    identities.add(identity)
                    plan[name] = content
            print(f"[CONFIG PASS] {scope}: {len(models)} models x {len(ALGOS)} algorithms = {len(models)*len(ALGOS)}")
    require(len(plan) == len(identities) == 50, "Expected exactly 50 unique configs")
    require(Counter(x.split("_")[0] for x in plan) == {"total": 20, "pbmc": 15, "buffycoat": 15},
            "Scope distribution mismatch")
    print("[DRY RUN PASS] All 50 YAMLs parsed by the active unified_experiment loader")
    return plan


def apply_plan(root: Path, plan: dict[str, str]) -> None:
    out = root / OUTPUT_REL
    # Check all existing targets before creating or writing anything.
    conflicts = [p for name, body in plan.items()
                 if (p := out / name).exists() and p.read_text(encoding="utf-8") != body]
    require(not conflicts, f"Refusing to overwrite edited files: {conflicts[:5]}")
    unexpected = [p for p in out.glob("*.yaml") if p.name not in plan] if out.is_dir() else []
    require(not unexpected, f"Unexpected YAMLs in output directory: {unexpected[:5]}")
    out.mkdir(parents=True, exist_ok=True)
    created = 0
    for name, body in sorted(plan.items()):
        target = out / name
        if target.exists():
            continue
        # Exclusive create guarantees no silent overwrite.
        with target.open("x", encoding="utf-8") as handle:
            handle.write(body)
        created += 1
    print(f"[APPLY PASS] {len(plan)} configs present; created={created}; directory={out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Default: dry-run without writing project files")
    group.add_argument("--apply", action="store_true", help="Create checked configurations")
    args = parser.parse_args()
    root = args.repo.expanduser().resolve()
    assert_inputs(root)
    plan = build_plan(root)
    if args.apply:
        apply_plan(root, plan)
    else:
        print(f"[CHECK ONLY] No repository files written. Output would be: {root / OUTPUT_REL}")
    print("[SUMMARY] 5 algorithms x 10 scope/model choices = 50 YAMLs x 100 repeats = 5000 outer holdouts")
    print("[NOTE] YAML dry run is not a model fit or a full-data feature-loading test")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[STOP] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
