#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IGH V3 Layer 2: safely create 45 F1/F2/F3 YAML configs from Layer 1 Top5.

Default is read-only. Use --write to publish new directories. Requires pandas + PyYAML.
No training is performed. V2 and all Layer 1 configs/results are not modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import yaml

SCOPES = {
    "total": ("igh_ra_ild_repeat100_v3_175", "455eba17c74a86667fd2939dc8645ac1c50f0cdb25053ab951dccad0ab8f1788", 122, 53),
    "pbmc": ("igh_ra_ild_pbmc_repeat100_v3_175", "4726d9b48301c4cd535369f9b39f6bdf3c6247c0722b9c9663385a6c041a68b8", 79, 34),
    "buffycoat": ("igh_ra_ild_buffycoat_repeat100_v3_175", "da1e598559dc734c0e0296bd1f99620351c2d81e903100c85fda23b670047c05", 43, 19),
}
GROUPS = {"D", "E", "L", "AU", "AW", "PU", "PW"}
FORMS = {
    "F1": (True, False),
    "F2": (False, True),
    "F3": (True, True),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd(), help="RA-ILD repository root")
    p.add_argument("--summary", type=Path, required=True, help="01_experiment_summary_381.csv, repo-relative or absolute")
    p.add_argument("--write", action="store_true", help="Publish 45 YAMLs; otherwise read-only verification")
    args = p.parse_args()
    root = args.repo.resolve()
    summary_path = args.summary if args.summary.is_absolute() else root / args.summary
    base = root / "IGH/set/configs/experiments"
    splits = root / "IGH/set/split_sets"
    assert summary_path.is_file(), f"missing summary: {summary_path}"
    assert base.is_dir(), f"missing config root: {base}"

    table = pd.read_csv(summary_path, dtype={"scope": str, "stem": str, "group_codes": str})
    required = {"scope", "stem", "rank_auc_in_scope", "group_codes", "n_repeats", "roc_auc_mean"}
    assert required.issubset(table.columns), f"summary missing: {sorted(required-set(table.columns))}"
    assert len(table) == 381, f"expected 381 summary rows; got {len(table)}"
    assert not table.duplicated(["scope", "stem"]).any(), "duplicate scope/stem"
    assert set(table.scope) == set(SCOPES), "unexpected scopes"
    assert table.n_repeats.eq(100).all(), "some Layer1 repeats are not 100"

    staged = {}
    for scope, (bank_name, frozen_sha, train_n, holdout_n) in SCOPES.items():
        part = table[table.scope == scope].copy()
        assert len(part) == 127, f"wrong Layer1 count for {scope}"
        ranks = part.rank_auc_in_scope.astype(int)
        assert set(ranks) == set(range(1, 128)), f"incomplete rank 1..127 for {scope}"
        part = part.sort_values("rank_auc_in_scope").head(5)
        old_dir = base / f"igh_layer1_{scope}_top10000_v3_175"
        new_dir = base / f"igh_layer2_{scope}_top10000_v3_175"
        assert old_dir.is_dir(), f"missing Layer1 configs: {old_dir}"
        assert not new_dir.exists(), f"REFUSE OVERWRITE: {new_dir}"
        assert len(list(old_dir.glob("S*.yaml"))) == 127, f"Layer1 YAML count mismatch: {scope}"

        bank_dir = splits / bank_name
        assignment = bank_dir / "repeated_holdout_assignments.csv"
        marker = bank_dir / "SPLITS_FROZEN.json"
        assert assignment.is_file() and marker.is_file(), f"frozen bank missing: {bank_name}"
        real_sha = sha256(assignment)
        frozen = json.loads(marker.read_text(encoding="utf-8"))
        assert real_sha == frozen_sha == frozen["files"]["assignments"]["sha256"], f"frozen SHA mismatch: {scope}"

        configs = {}
        manifest = []
        for row in part.itertuples(index=False):
            stem = row.stem
            source_path = old_dir / f"{stem}.yaml"
            assert source_path.is_file(), f"missing Layer1 YAML: {source_path}"
            src = yaml.safe_load(source_path.read_text(encoding="utf-8"))
            assert isinstance(src, dict), f"not a YAML mapping: {source_path}"
            original_id = f"igh_layer1_{scope}_top10000_{stem}_v3_175"
            assert src["experiment"]["id"] == original_id
            assert src["experiment"]["receptor"] == "IGH"
            assert src["cohort"]["scope"] == scope
            assert src["model"]["algorithm"] == "elastic_net"
            assert src["model"]["positive_label"] == "ILD"
            assert src["validation"]["repeats"] == 100
            assert src["validation"]["folds"] == 5
            configured_assignment = Path(src["validation"]["split_assignments"])
            if not configured_assignment.is_absolute():
                configured_assignment = root / configured_assignment
            assert configured_assignment.resolve() == assignment.resolve(), (
                f"split assignments mismatch for {source_path}"
            )
            assert src["repertoire"]["mode"] == "topk" and int(src["repertoire"]["top_k"]) == 10000
            feat = src["features"]
            assert not feat["kmer"]["enabled"] and not feat["enriched_dictionary"]["enabled"]
            assert feat["clinical_groups"] == [], "clinical groups must remain excluded"
            codes = str(row.group_codes).split("+")
            assert set(codes).issubset(GROUPS) and len(set(codes)) == len(codes)
            assert len(feat["static_tcr_groups"]) == len(codes)
            grid = src["model"]["parameters"]["elastic_net"]
            assert len(grid["alpha_grid"]) * len(grid["lambda_grid"]) == 24
            for form, (km, ra) in FORMS.items():
                new = copy.deepcopy(src)
                new_stem = f"{stem}_{form}"
                new_id = f"igh_layer2_{scope}_top10000_{new_stem}_v3_175"
                new["experiment"]["id"] = new_id
                kmer = new["features"]["kmer"]
                for key in ("enabled", "k", "representation", "top_n", "min_sample_count", "min_prevalence", "min_unweighted_variance", "min_weighted_variance"):
                    assert key in kmer, f"missing kmer.{key}: {source_path}"
                kmer.update(enabled=km, k=3, representation="unweighted", top_n=500,
                            min_sample_count=5, min_prevalence=0.05,
                            min_unweighted_variance=1e-12, min_weighted_variance=1e-12)
                dictionary = new["features"]["enriched_dictionary"]
                for key in ("enabled", "threshold_pct", "delta_pct", "features"):
                    assert key in dictionary, f"missing enriched_dictionary.{key}: {source_path}"
                dictionary.update(enabled=ra, threshold_pct=20, delta_pct=10, features=["RA_dict_hit_rate"])
                assert new["validation"]["split_assignments"] == src["validation"]["split_assignments"]
                assert new["features"]["static_tcr_groups"] == src["features"]["static_tcr_groups"]
                assert new["model"] == src["model"], "model options unexpectedly changed"
                # Semantic diff contract: ONLY experiment.id and dynamic features may differ.
                expected = copy.deepcopy(src)
                expected["experiment"]["id"] = new_id
                expected["features"]["kmer"] = copy.deepcopy(kmer)
                expected["features"]["enriched_dictionary"] = copy.deepcopy(dictionary)
                assert expected == new
                name = f"{new_stem}.yaml"
                assert name not in configs
                serialized = yaml.safe_dump(new, allow_unicode=True, sort_keys=False)
                assert yaml.safe_load(serialized) == new, f"YAML roundtrip failed: {name}"
                configs[name] = serialized
                manifest.append({"scope": scope, "layer1_rank": int(row.rank_auc_in_scope), "layer1_stem": stem,
                                 "groups": row.group_codes, "layer1_auc_mean": float(row.roc_auc_mean),
                                 "layer2_stem": new_stem, "form": form, "experiment_id": new_id,
                                 "split_sha256": frozen_sha, "train_n": train_n, "holdout_n": holdout_n})
        assert len(configs) == 15 and len(manifest) == 15
        staged[scope] = (new_dir, configs, manifest)
        print(f"[{scope.upper()}] 5 Layer1 candidates -> 15 Layer2 configs; split SHA {real_sha[:12]}")
        for r in manifest[::3]:
            print(f"  rank {r['layer1_rank']}: {r['layer1_stem']} ({r['groups']}, AUC={r['layer1_auc_mean']:.4f})")

    assert sum(len(v[1]) for v in staged.values()) == 45
    if not args.write:
        print("\nLAYER2 CONFIG PREFLIGHT PASS: 45 configs ready; no files written. Use --write to publish.")
        return

    with tempfile.TemporaryDirectory(prefix=".layer2_v3_stage_", dir=base) as temp:
        stage = Path(temp)
        for scope, (dest, configs, manifest) in staged.items():
            folder = stage / dest.name
            folder.mkdir()
            for filename, value in configs.items():
                with (folder / filename).open("x", encoding="utf-8") as f:
                    f.write(value)
            (folder / "LAYER2_MANIFEST.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            assert len(list(folder.glob("S*.yaml"))) == 15
            for yaml_path in folder.glob("S*.yaml"):
                assert yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["experiment"]["id"].endswith("_v3_175")
        # No overwrite, distinct folders, original V1/V2/Layer1 artifacts untouched.
        for scope, (dest, _, _) in staged.items():
            assert not dest.exists(), f"REFUSE OVERWRITE: {dest}"
        for scope, (dest, _, _) in staged.items():
            os.rename(stage / dest.name, dest)
            print(f"[WRITE PASS] {scope}: {dest}")
    print("\nIGH V3 LAYER2 CONFIG WRITE PASS: 45 YAMLs (15 per scope). No training was run.")


if __name__ == "__main__":
    main()
