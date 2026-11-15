#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only IGH V3 hierarchical 5-algorithm migration preflight.

Checks the immutable V3 inputs/splits, V3 engine contracts, dependencies,
feature registry and the two known migration gaps (both 3-mer and public).
No training, output creation, Git modification or data transformation.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

# Keep this preflight read-only with respect to the project source tree.
sys.dont_write_bytecode = True

SPECS = {
    "total": {
        "n": 175, "ra": 107, "ild": 68, "train": 122, "holdout": 53,
        "bank": "igh_ra_ild_repeat100_v3_175",
        "sha_prefix": "455eba17c74a",
    },
    "pbmc": {
        "n": 113, "ra": 67, "ild": 46, "train": 79, "holdout": 34,
        "bank": "igh_ra_ild_pbmc_repeat100_v3_175",
        "sha_prefix": "4726d9b48301",
    },
    "buffycoat": {
        "n": 62, "ra": 40, "ild": 22, "train": 43, "holdout": 19,
        "bank": "igh_ra_ild_buffycoat_repeat100_v3_175",
        "sha_prefix": "da1e598559dc",
    },
}
GROUPS = (
    "tcr_diversity", "tcr_clonal_expansion", "tcr_aa_length",
    "tcr_aa_composition_unweighted", "tcr_aa_composition_weighted",
    "tcr_physicochemical_unweighted", "tcr_physicochemical_weighted",
)
ALGORITHMS = ("elastic_net", "ridge", "lasso", "linear_svm", "xgboost")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.PIPE).strip()


def read_metadata(root: Path, scope: str, info: dict[str, Any]) -> set[str]:
    d = root / f"IGH/set/data_versions/igh_v3_175/ml_compat/{scope}"
    paths = [d / "metadata_train.csv", d / "metadata_test.csv"]
    rows: list[dict[str, str]] = []
    for p in paths:
        require(p.is_file(), f"missing metadata: {p}")
        with p.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames is not None, f"empty metadata: {p}")
            require({"libraryid", "cohort", "age", "sex"}.issubset(reader.fieldnames),
                    f"metadata missing required columns: {p}")
            rows.extend(reader)
    ids = [r["libraryid"].strip() for r in rows]
    require(all(ids) and len(ids) == len(set(ids)), f"{scope}: duplicate or blank IDs")
    require(len(ids) == info["n"], f"{scope}: n={len(ids)} != {info['n']}")
    counts = Counter(r["cohort"].strip().upper().replace("RA-ILD", "ILD") for r in rows)
    require(counts == {"RA": info["ra"], "ILD": info["ild"]},
            f"{scope}: unexpected RA/ILD labels/counts {dict(counts)}")
    require(all(r["age"].strip() and r["sex"].strip() for r in rows),
            f"{scope}: blank age/sex metadata")
    for row in rows:
        try:
            float(row["age"])
        except (ValueError, TypeError):
            raise RuntimeError(f"{scope}: invalid age for {row['libraryid']}") from None
    for sub in ("train_step02", "test_step02"):
        require((d / sub).is_dir(), f"{scope}: missing Step02 directory {d/sub}")
    print(f"[METADATA PASS] {scope}: n={len(ids)}, RA={counts['RA']}, ILD={counts['ILD']}, Step02 dirs present")
    return set(ids)


def read_bank(root: Path, scope: str, info: dict[str, Any], ids: set[str]) -> None:
    p = root / "IGH/set/split_sets" / info["bank"] / "repeated_holdout_assignments.csv"
    require(p.is_file(), f"{scope}: missing frozen assignment bank {p}")
    actual = sha256(p)
    # Only the frozen 12-character prefixes are independently recorded here.
    # Never invent the remaining digest. Print the actual full hash for auditing.
    require(actual.startswith(info["sha_prefix"]),
            f"{scope}: SHA256 prefix mismatch expected_prefix={info['sha_prefix']} actual={actual}")
    repeat_rows: dict[int, dict[str, set[str]]] = {}
    count_rows = 0
    with p.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and
                {"repeat_index", "sample_id", "role"}.issubset(reader.fieldnames),
                f"{scope}: missing frozen bank columns")
        for row in reader:
            idx = int(row["repeat_index"])
            role = row["role"].strip().lower()
            sid = row["sample_id"].strip()
            require(role in {"train", "holdout"}, f"{scope}/{idx}: bad role {role}")
            r = repeat_rows.setdefault(idx, {"train": set(), "holdout": set()})
            require(sid not in r[role], f"{scope}/{idx}: duplicate {sid}/{role}")
            r[role].add(sid)
            count_rows += 1
    require(set(repeat_rows) == set(range(1, 101)),
            f"{scope}: repeat indices not exactly 1..100")
    for idx, r in repeat_rows.items():
        require(len(r["train"]) == info["train"] and len(r["holdout"]) == info["holdout"],
                f"{scope}/{idx}: wrong split sizes")
        require(not r["train"] & r["holdout"] and r["train"] | r["holdout"] == ids,
                f"{scope}/{idx}: split sample coverage mismatch")
    print(f"[SPLITS PASS] {scope}: 100 repeats, {info['train']}/{info['holdout']}, full_SHA256={actual}, rows={count_rows}")


def source_checks(root: Path, all_ids: set[str]) -> None:
    import yaml  # noqa: PLC0415
    p = root / "IGH/set/configs/feature_framework/igh_feature_registry_v3_175.yaml"
    require(p.is_file(), f"feature registry absent: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    raw_groups = data["feature_groups"]
    selected: list[str] = []
    for gid in GROUPS:
        require(gid in raw_groups, f"missing feature group {gid}")
        group = raw_groups[gid]
        selector = group["selector"]
        names = list(selector.get("exact", []))
        drops = set(group.get("reference_drop_columns", []))
        if selector.get("prefixes"):
            raise RuntimeError(f"{gid}: prefix selector needs manual audit before Core83 claim")
        names = [n for n in names if n not in drops]
        selected.extend(names)
        print(f"[CORE83] {gid}: {len(names)} columns")
    require(len(selected) == 83 and len(set(selected)) == 83,
            f"Core83 selectors: {len(selected)} columns, unique={len(set(selected))}, expected 83/83")
    source = data["repertoire_sources"]["top10000"]
    dirname = root / source["default_path"]
    require(dirname.is_dir(), f"Top10000 directory missing: {dirname}")
    missing = [sid for sid in sorted(all_ids) if not (dirname / source["sample_filename_template"].format(sample_id=sid)).is_file()]
    require(not missing, f"Top10000 missing {len(missing)} samples, first: {missing[:5]}")
    summary = root / source["summary_path"]
    require(summary.is_file(), f"Top10000 QC summary missing: {summary}")
    print(f"[REPERTOIRE PASS] source=top10000, available files={len(all_ids)}, Core83 selected=83")


def algorithm_checks(root: Path) -> None:
    import yaml  # noqa: PLC0415
    pkg = root / "IGH/set/src"
    sys.path.insert(0, str(pkg))
    from ra_ild_igh.unified_experiment import load_unified_experiment  # noqa: PLC0415
    from ra_ild_igh.experiment_spec import KMER_REPRESENTATIONS, FeatureSpec  # noqa: PLC0415
    sample = root / "IGH/set/configs/experiments/igh_clinical_layer1_total_top10000_v3_175/S000_none.yaml"
    require(sample.is_file(), f"S000 M0 template missing: {sample}")
    base = yaml.safe_load(sample.read_text(encoding="utf-8"))
    for alg in ALGORITHMS:
        payload = yaml.safe_load(yaml.safe_dump(base))
        payload["experiment"]["id"] = f"igh_v3_preflight_{alg}_m0"
        payload["model"]["algorithm"] = alg
        with tempfile.TemporaryDirectory(prefix="igh_v3_preflight_") as td:
            candidate = Path(td) / "m0.yaml"
            candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            parsed = load_unified_experiment(candidate)
            require(parsed.algorithm == alg, f"{alg}: parsed algorithm mismatch")
        print(f"[ENGINE CONFIG PASS] {alg}")
    print("[ADAPTER STATUS] weighted+unweighted 3-mer: " +
          ("READY" if "both" in KMER_REPRESENTATIONS else "NOT YET SUPPORTED (expected before migration)"))
    feature_fields = {field.name for field in dataclasses.fields(FeatureSpec)}
    print("[ADAPTER STATUS] V2 Public-reference in V3 fold features: " +
          ("SPEC FIELD PRESENT - implementation still requires audit" if "public_reference" in feature_fields
           else "NOT YET SUPPORTED (expected before migration)"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="Path to local RA-ILD repository")
    args = p.parse_args()
    root = Path(args.repo).expanduser().resolve()
    require((root / ".git").exists(), f"not a Git working tree: {root}")
    branch = git(root, "branch", "--show-current")
    print(f"REPO: {root}")
    print(f"BRANCH: {branch}")
    require(branch == "feature/igh-v3-hierarchical-5alg",
            "STOP: please create/switch to feature/igh-v3-hierarchical-5alg first")
    base = git(root, "rev-parse", "feature/igh-v3-ml-freeze")
    head = git(root, "rev-parse", "HEAD")
    rc = subprocess.run(["git", "merge-base", "--is-ancestor", base, head],
                        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    require(rc == 0, "current branch is not a descendant of frozen IGH V3 commit")
    print(f"[GIT PASS] frozen ancestor={base[:12]}, HEAD={head[:12]}")
    for lib in ("numpy", "pandas", "scipy", "scikit-learn", "PyYAML", "xgboost"):
        # The conda environment can provide an importable package without a
        # discoverable Python distribution metadata record (observed for XGBoost).
        # A functional import is the authoritative availability check here.
        module_name = {"scikit-learn": "sklearn", "PyYAML": "yaml"}.get(lib, lib)
        try:
            module = __import__(module_name)
            if lib == "xgboost":
                from xgboost import XGBClassifier  # noqa: F401, PLC0415
        except Exception as exc:
            raise RuntimeError(
                f"Cannot import required package {lib} ({module_name}): {exc!r}"
            ) from exc
        try:
            version = importlib.metadata.version(lib)
            origin = "distribution metadata"
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "unknown")
            origin = "module import; distribution metadata unavailable"
        print(f"[DEPENDENCY PASS] {lib}=={version} ({origin})")
    identities = {}
    for scope, info in SPECS.items():
        identities[scope] = read_metadata(root, scope, info)
        read_bank(root, scope, info, identities[scope])
    require(not identities["pbmc"] & identities["buffycoat"], "PBMC and buffycoat IDs overlap")
    require(identities["pbmc"] | identities["buffycoat"] == identities["total"],
            "scope membership mismatch: total != PBMC + buffycoat")
    print("[COHORT MEMBERSHIP PASS] PBMC and buffycoat are a non-overlapping partition of Total")
    source_checks(root, identities["total"])
    algorithm_checks(root)
    print("\nIGH V3 HIERARCHICAL 5-ALGORITHM INPUT PREFLIGHT PASS")
    print("Expected fixed models: Total M0-M3 (4) + PBMC M0-M2 (3) + Buffycoat M0-M2 (3)")
    print("Algorithms: 5; total configurations: 50; proposed 100 repeats/config = 5000 outer holdouts")
    print("No models trained. No persistent project files created or changed.")
    print("NEXT: implement/audit 'both' 3-mer and strict fold-local V2 Public reference adapters.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"PREFLIGHT ERROR: {e}", file=sys.stderr)
        raise SystemExit(2) from e
