#!/usr/bin/env python3
"""Strict audit for the 381 Elastic-Net Layer-1 configs and shared Core83 cache."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ra_ild_trb.layered_experiments import ALGORITHMS, COHORTS, CORE83_GROUPS, build_layer1_combinations
from ra_ild_trb.feature_framework import load_feature_registry
from ra_ild_trb.fold_features import select_static_features
MANIFEST = ROOT / "TRB/set/configs/layered_feature_experiments/generated/layer1/LAYER1_CONFIG_MANIFEST.json"
REGISTRY = ROOT / "TRB/set/configs/feature_framework/trb_feature_registry_v3_179_top10000.yaml"
COMPAT = ROOT / "TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total"
CACHE = ROOT / "TRB/result/feature_framework/cache/step02/top10000"
EXPECTED_SPLITS = {"total": "TRB/set/split_sets/ra_ild_repeat100_v3_179_top10000/repeated_holdout_assignments.csv", "pbmc": "TRB/set/split_sets/ra_ild_pbmc_repeat100_v3_180/repeated_holdout_assignments.csv", "buffycoat": "TRB/set/split_sets/ra_ild_buffycoat_repeat100_v3_179_top10000/repeated_holdout_assignments.csv"}

def fail(message: str) -> None:
    raise SystemExit("AUDIT FAIL: " + message)

def audit_configs(root: Path):
    manifest_path = root / MANIFEST.relative_to(ROOT)
    if not manifest_path.is_file(): fail(f"manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    expected_total = len(COHORTS) * len(build_layer1_combinations())
    if payload.get("count") != expected_total or len(entries) != expected_total: fail(f"manifest count must be {expected_total}; declared={payload.get('count')}, actual={len(entries)}")
    if ALGORITHMS != ("elastic_net",): fail(f"code algorithm contract changed unexpectedly: {ALGORITHMS}")
    expected_combo = {c.combination_id: list(c.group_ids) for c in build_layer1_combinations()}
    seen = set(); counts = {c: 0 for c in COHORTS}
    for entry in entries:
        cohort, algorithm, combo = entry.get("cohort"), entry.get("algorithm"), entry.get("combination_id")
        key = (cohort, algorithm, combo)
        if key in seen: fail(f"duplicate manifest entry: {key}")
        seen.add(key)
        if cohort not in COHORTS: fail(f"unexpected cohort: {cohort}")
        counts[cohort] += 1
        if algorithm != "elastic_net": fail(f"non-Elastic-Net config in manifest: {key}")
        if combo not in expected_combo or entry.get("feature_groups") != expected_combo[combo]: fail(f"Core83 combination mismatch: {key}")
        if entry.get("expected_samples") != COHORTS[cohort]: fail(f"expected sample count mismatch: {key}")
        split = EXPECTED_SPLITS[cohort]
        if entry.get("split_assignments") != split or not (root / split).is_file(): fail(f"split bank path mismatch/missing: {key}")
        cfg_path = root / entry["config"]
        if not cfg_path.is_file(): fail(f"config file missing: {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if cfg["experiment"]["id"] != f"trb179_{cohort}_elastic_net_{combo.lower()}": fail(f"experiment ID mismatch: {cfg_path}")
        if cfg["cohort"]["scope"] != cohort: fail(f"cohort mismatch: {cfg_path}")
        if cfg["model"]["algorithm"] != "elastic_net": fail(f"algorithm mismatch: {cfg_path}")
        if cfg["repertoire"] != {"mode": "topk", "top_k": 10000}: fail(f"repertoire contract mismatch: {cfg_path}")
        if cfg["features"]["clinical_groups"] != ["clinical_age_sex"]: fail(f"age/sex adjustment mismatch: {cfg_path}")
        if cfg["features"]["static_tcr_groups"] != expected_combo[combo]: fail(f"static feature group mismatch: {cfg_path}")
        if cfg["features"]["kmer"]["enabled"] is not False: fail(f"Layer-1 3-mer must be disabled: {cfg_path}")
        if cfg["features"]["enriched_dictionary"]["enabled"] is not False: fail(f"Layer-1 RA-hit must be disabled: {cfg_path}")
        if cfg["validation"]["repeats"] != 100 or cfg["validation"]["folds"] != 5: fail(f"CV contract mismatch: {cfg_path}")
        if cfg["validation"]["split_assignments"] != split: fail(f"config split bank mismatch: {cfg_path}")
    if counts != {c: 127 for c in COHORTS}: fail(f"cohort config counts mismatch: {counts}")
    if len([e for e in entries if e["cohort"] == "total"]) != 127: fail("Total must contain 127 unique combinations")
    registry = load_feature_registry(root / REGISTRY.relative_to(ROOT))
    print(f"[PASS] Layer-1 manifest/configs: {len(entries)}")
    print(f"[PASS] Cohort counts: {counts}")
    print(f"[PASS] Algorithms: {sorted({e['algorithm'] for e in entries})}")
    print(f"[PASS] Core83 combinations: {len(expected_combo)}")
    return registry

def audit_cache(root: Path, registry) -> None:
    train_dir = root / CACHE.relative_to(ROOT) / "train"
    test_dir = root / CACHE.relative_to(ROOT) / "test"
    train_path, test_path = train_dir / "02_core_sample_features.csv", test_dir / "02_core_sample_features.csv"
    for path in (train_path, test_path):
        if not path.is_file(): fail(f"Step02 cache file missing: {path}")
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    for label, frame in (("TRAIN", train), ("TEST", test)):
        if "sample_id" not in frame.columns: fail(f"{label} core table lacks sample_id")
        if frame["sample_id"].astype(str).duplicated().any(): fail(f"{label} core table has duplicate sample_id")
        if frame.isna().any().any():
            missing = frame.isna().sum(); fail(f"{label} core table contains missing values: {missing[missing>0].to_dict()}")
        selected = select_static_features(registry, frame.columns, CORE83_GROUPS)
        if len(selected) != 83 or len(set(selected)) != 83: fail(f"{label} Core83 resolution expected 83 unique predictors; got {len(selected)}")
        print(f"[PASS] {label} Step02 Core83 columns: {len(selected)}")
    train_meta = pd.read_csv(root / COMPAT.relative_to(ROOT) / "metadata_train_70.csv", dtype=str)
    test_meta = pd.read_csv(root / COMPAT.relative_to(ROOT) / "metadata_test_30.csv", dtype=str)
    pairs = (("TRAIN", set(train["sample_id"].astype(str)), set(train_meta["libraryid"].astype(str))), ("TEST", set(test["sample_id"].astype(str)), set(test_meta["libraryid"].astype(str))))
    for label, observed, expected in pairs:
        if observed != expected: fail(f"{label} Step02 sample IDs differ from compatibility metadata: missing={sorted(expected-observed)[:10]}, extra={sorted(observed-expected)[:10]}")
        print(f"[PASS] {label} Step02 sample IDs: {len(observed)}")
    for label, directory in (("TRAIN", train_dir), ("TEST", test_dir)):
        provenance = directory / "02_resolved_feature_source.json"
        if not provenance.is_file(): fail(f"{label} source provenance missing: {provenance}")
        info = json.loads(provenance.read_text(encoding="utf-8"))
        source_id = (info.get("repertoire_input") or {}).get("source_id")
        if source_id != "top10000": fail(f"{label} Step02 source_id must be top10000, got {source_id!r}")
    print("[PASS] Step02 provenance: V3 top10000")
    print("[PASS] Combined cache covers the 179-sample compatibility metadata")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, default=ROOT)
    p.add_argument("--check-cache", action="store_true")
    a = p.parse_args(); root = a.repository_root.resolve()
    registry = audit_configs(root)
    if a.check_cache: audit_cache(root, registry)
    print("ALL REQUESTED LAYER-1 AUDITS PASSED")

if __name__ == "__main__": main()
