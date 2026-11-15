#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only real-data smoke check for IGH V3 hierarchical 5-algorithm YAMLs.

Defaults: real-data feature construction and train-only preprocessing only.
--fit: fit one existing candidate (NOT hyperparameter selection) for each algorithm.
Uses only frozen Repeat 001, does NOT run the 100-repeats training entry point.
No project data/config/splits are modified; optional JSON report only on --report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

BRANCH = "feature/igh-v3-hierarchical-5alg"
BASE = Path("IGH/set")
CONF = BASE / "configs/experiments/igh_v3_hierarchical_5alg_top10000_v3_175"
REGISTRY = BASE / "configs/feature_framework/igh_feature_registry_v3_175.yaml"
ALGORITHMS = ("elastic_net", "ridge", "lasso", "linear_svm", "xgboost")
LAYERS = {"total": ("M0", "M1", "M2", "M3"),
          "pbmc": ("M0", "M1", "M2"),
          "buffycoat": ("M0", "M1", "M2")}
SPLITS = {
    "total": ("igh_ra_ild_repeat100_v3_175", "455eba17c74a" ,122, 53),
    "pbmc": ("igh_ra_ild_pbmc_repeat100_v3_175", "4726d9b48301",79, 34),
    "buffycoat": ("igh_ra_ild_buffycoat_repeat100_v3_175", "da1e598559dc",43, 19),
}
CORE83 = (
    "tcr_diversity", "tcr_clonal_expansion", "tcr_aa_length",
    "tcr_aa_composition_unweighted", "tcr_aa_composition_weighted",
    "tcr_physicochemical_unweighted", "tcr_physicochemical_weighted",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen_split(root: Path, scope: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    name, prefix, expected_train, expected_test = SPLITS[scope]
    path = root / BASE / "split_sets" / name / "repeated_holdout_assignments.csv"
    require(path.is_file(), f"Missing frozen split file: {path}")
    digest = sha256(path)
    require(digest.startswith(prefix), f"Frozen split SHA mismatch {scope}: {digest}")
    frame = pd.read_csv(path, dtype={"sample_id": str, "role": str})
    require({"repeat_index", "sample_id", "role"}.issubset(frame.columns),
            "Invalid split CSV schema")
    require(set(frame["repeat_index"].astype(int)) == set(range(1, 101)),
            f"{scope}: frozen bank is not 100 repeats")
    sub = frame.loc[frame["repeat_index"].astype(int) == 1].copy()
    sub["role"] = sub["role"].astype(str).str.lower().str.strip()
    require(set(sub["role"]) == {"train", "holdout"}, f"Invalid role values {scope}")
    require(not sub["sample_id"].duplicated().any(), f"Duplicate split IDs {scope}")
    train = tuple(sub.loc[sub["role"] == "train", "sample_id"].astype(str).tolist())
    test = tuple(sub.loc[sub["role"] == "holdout", "sample_id"].astype(str).tolist())
    require((len(train), len(test)) == (expected_train, expected_test),
            f"{scope}: unexpected frozen Repeat001 counts {len(train)}/{len(test)}")
    require(not set(train) & set(test), f"{scope}: train/holdout overlap")
    print(f"[FROZEN PASS] {scope} SHA256={digest} repeat=001 train={len(train)} holdout={len(test)}", flush=True)
    return train, test


def load_spec(root: Path, scope: str, layer: str, alg: str, loader):
    path = root / CONF / f"{scope}_{layer}_{alg}.yaml"
    require(path.is_file(), f"Missing YAML {path}")
    spec = loader(path)
    f = spec.base.features
    require(spec.algorithm == alg and spec.base.cohort.scope == scope, f"Spec identity mismatch {path.name}")
    require(spec.base.repertoire.source_id == "top10000", f"Wrong repertoire {path.name}")
    require(spec.cv.repeats == 100 and spec.cv.inner_folds == 5, f"CV mismatch {path.name}")
    require(tuple(f.clinical_groups) ==
            (("clinical_age_sex", "clinical_material") if layer == "M3" else ("clinical_age_sex",)),
            f"Clinical layer mismatch {path.name}")
    require(tuple(f.static_tcr_groups) == (CORE83 if layer != "M0" else ()),
            f"Core83 layer mismatch {path.name}")
    require(f.kmer.enabled == (layer != "M0") and f.kmer.top_n == 500 and f.kmer.k == 3,
            f"Kmer enabled/topN mismatch {path.name}")
    if layer != "M0":
        require(f.kmer.representation == "both", f"Kmer representation mismatch {path.name}")
    require(not f.enriched_dictionary.enabled, f"Unexpected old enriched_dictionary {path.name}")
    require(f.public_reference.enabled == (layer in {"M2", "M3"}),
            f"Public-reference layer mismatch {path.name}")
    require(spec.model.positive_label.upper() == "ILD", f"Positive label mismatch {path.name}")
    expected_bank = SPLITS[scope][0]
    require(str(spec.cv.split_assignments).endswith(
            f"/{expected_bank}/repeated_holdout_assignments.csv"),
            f"Frozen bank path mismatch {path.name}")
    return spec


def make_inputs(root: Path, scope: str, spec, loader):
    d = root / BASE / "data_versions/igh_v3_175/ml_compat" / scope
    return loader(
        spec, repository_root=root,
        train_metadata=d / "metadata_train.csv",
        test_metadata=d / "metadata_test.csv",
        train_step02_dir=d / "train_step02",
        test_step02_dir=d / "test_step02",
        registry_path=root / REGISTRY,
        validate_source_files=True,
    )


def check_frames(scope, layer, built, train_ids, valid_ids, previous):
    numeric = tuple(built.numeric_columns)
    categoric = tuple(built.categorical_columns)
    require(not set(numeric) & set(categoric), f"{scope}/{layer}: overlapping predictors")
    require(len(numeric) == len(set(numeric)), f"{scope}/{layer}: duplicated numeric features")
    require(set(built.train.sample_id.astype(str)) == set(train_ids), f"{scope}/{layer}: train IDs changed")
    require(set(built.valid.sample_id.astype(str)) == set(valid_ids), f"{scope}/{layer}: holdout IDs changed")
    require(len(built.train) == len(train_ids) and len(built.valid) == len(valid_ids),
            f"{scope}/{layer}: incorrect fold row counts")
    require("age" in numeric and "sex" in categoric, f"{scope}/{layer}: missing age/sex")
    require(("material" in categoric) == (layer == "M3"), f"{scope}/{layer}: material layer error")
    u = [c for c in numeric if c.startswith("unweighted_3mer_")]
    w = [c for c in numeric if c.startswith("weighted_3mer_")]
    expected_k = 0 if layer == "M0" else 500
    require(len(u) == len(w) == expected_k,
            f"{scope}/{layer}: 3mer counts wrong, unweighted={len(u)} weighted={len(w)}")
    if layer != "M0":
        require({c[len('unweighted_3mer_'):] for c in u} ==
                {c[len('weighted_3mer_'):] for c in w},
                f"{scope}/{layer}: weighted/unweighted vocab differs")
        require(len(built.kmer_vocabulary) == 500, f"{scope}/{layer}: not a Top500 vocabulary")
    expected_nums = {"M0": 1, "M1": 1+83+1000, "M2": 1+83+1000+8, "M3": 1+83+1000+8}[layer]
    require(len(numeric) == expected_nums,
            f"{scope}/{layer}: expected numeric={expected_nums} got={len(numeric)}")
    for part, frame in (("train", built.train), ("holdout", built.valid)):
        require(not frame[[*numeric, *categoric]].isna().any().any(),
                f"{scope}/{layer}/{part}: NaN present")
        values = frame.loc[:, list(numeric)].to_numpy(dtype=float)
        require(np.isfinite(values).all(), f"{scope}/{layer}/{part}: non-finite numeric input")
    if previous is not None:
        old_numeric = list(previous.numeric_columns)
        old_categorical = list(previous.categorical_columns)
        require(set(old_numeric).issubset(numeric), f"{scope}/{layer}: numerical hierarchy broken")
        require(set(old_categorical).issubset(categoric), f"{scope}/{layer}: categorical hierarchy broken")
        for part in ("train", "valid"):
            a = getattr(previous, part).set_index("sample_id").loc[list(train_ids if part == "train" else valid_ids)]
            b = getattr(built, part).set_index("sample_id").loc[list(train_ids if part == "train" else valid_ids)]
            for col in old_numeric:
                require(np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), rtol=0, atol=1e-12),
                        f"{scope}/{layer}/{part}: previous-layer feature changed {col}")
            for col in old_categorical:
                require(a[col].astype(str).equals(b[col].astype(str)),
                        f"{scope}/{layer}/{part}: previous-layer category changed {col}")
        delta = set(numeric) - set(old_numeric)
        if layer == "M1":
            require(len(delta) == 83+1000, f"{scope}: M1 should add 83+1000 numeric features")
        elif layer == "M2":
            require(len(delta) == 8, f"{scope}: M2 should add exactly 8 public features")
        elif layer == "M3":
            require(not delta and set(categoric)-set(old_categorical)=={"material"},
                    f"{scope}: M3 should add material only")
    return {"numeric": len(numeric), "categorical": list(categoric),
            "unweighted_3mer": len(u), "weighted_3mer": len(w),
            "raw_predictor_columns": len(numeric)+len(categoric)}


def preflight_fold(spec, built, prepared, scope, layer):
    require(prepared.X_train.shape[0] == len(built.train), f"{scope}/{layer} train matrix rows")
    require(prepared.X_valid.shape[0] == len(built.valid), f"{scope}/{layer} holdout matrix rows")
    require(prepared.X_train.shape[1] == prepared.X_valid.shape[1] > 0,
            f"{scope}/{layer} train/holdout feature widths differ")
    require(np.isfinite(prepared.X_train).all() and np.isfinite(prepared.X_valid).all(),
            f"{scope}/{layer} non-finite preprocessed features")
    require(set(np.unique(prepared.y_train)) == {0,1} and set(np.unique(prepared.y_valid)) == {0,1},
            f"{scope}/{layer} not binary in each split")
    return {"post_preprocessing_features": int(prepared.X_train.shape[1]),
            "train_classes": Counter(prepared.y_train.astype(int).tolist()),
            "holdout_classes": Counter(prepared.y_valid.astype(int).tolist())}


def fit_one(alg, spec, prepared, build_candidates, fit_candidate):
    from sklearn.metrics import roc_auc_score
    bank = build_candidates(spec)
    require(bool(bank), f"No candidates for {alg}")
    # One real configured candidate; no CV candidate selection or hyperparameter optimization.
    candidate = bank[0]
    t0 = time.monotonic()
    fitted = fit_candidate(candidate, prepared.X_train, prepared.y_train, spec,
                           random_state=20261115)
    scores = np.asarray(fitted.score(prepared.X_valid), dtype=float)
    require(scores.shape == (len(prepared.y_valid),) and np.isfinite(scores).all(),
            f"{alg} predictions are not finite")
    auc = float(roc_auc_score(prepared.y_valid, scores))
    require(math.isfinite(auc), f"{alg}: holdout AUC non-finite")
    return {"candidate_id":candidate.candidate_id,"converged":bool(fitted.converged),
            "roc_auc_diagnostic_only":round(auc,6),
            "score_type":fitted.score_type,
            "elapsed_seconds":round(time.monotonic()-t0,3)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--scope", choices=("all", *LAYERS), default="total",
                   help="Default total; 'all' covers Total+PBMC+Buffycoat")
    p.add_argument("--fit", action="store_true", help="Fit one configured candidate for each of five algorithms")
    p.add_argument("--report", type=Path, help="Optional JSON report path (only path written)")
    a=p.parse_args()
    root=a.repo.expanduser().resolve()
    require((root/".git").exists(), f"Not a Git worktree: {root}")
    branch=subprocess.check_output(["git","-C",str(root),"branch","--show-current"],text=True).strip()
    require(branch==BRANCH,f"Wrong branch {branch!r}")
    sys.path.insert(0,str(root/BASE/"src"))
    from ra_ild_igh.unified_experiment import load_unified_experiment
    from ra_ild_igh.fold_features import load_fold_feature_inputs, build_fold_features
    from ra_ild_igh.unified_modeling import prepare_fold, build_candidates, fit_candidate

    scopes=list(LAYERS) if a.scope=="all" else [a.scope]
    result={"mode":"one_candidate_fit" if a.fit else "features_only",
            "repeat_index":1,"scopes":{},"notes":
            "Diagnostic smoke only; not nested CV, model selection, or reportable outer test performance"}
    passed=0
    for scope in scopes:
        train_ids,holdout_ids=frozen_split(root,scope)
        known_specs={}
        scope_result={}
        for layer in LAYERS[scope]:
            for algorithm in ALGORITHMS:
                known_specs[algorithm]=load_spec(root,scope,layer,algorithm,load_unified_experiment)
            # All algorithm variants must have identical features and split contract.
            ref=known_specs[ALGORITHMS[0]]
            for alg,spec in known_specs.items():
                require(spec.base.features==ref.base.features and spec.cv==ref.cv,
                        f"{scope}/{layer}: algorithm YAML differs in features or CV ({alg})")
            tick=time.monotonic()
            inputs=make_inputs(root,scope,ref,load_fold_feature_inputs)
            require(set(inputs.sample_ids)==set(train_ids)|set(holdout_ids),
                    f"{scope}/{layer}: loaded sample IDs don't match frozen Repeat001")
            built=build_fold_features(inputs,ref,train_ids,holdout_ids,
                                      context=f"smoke_{scope}_{layer}_repeat001")
            prev=scope_result.get("_previous_build")
            counts=check_frames(scope,layer,built,train_ids,holdout_ids,prev)
            prepared=prepare_fold(built.train,built.valid,built.numeric_columns,built.categorical_columns,
                                   positive_label=ref.model.positive_label,
                                   zero_sd_tolerance=ref.model.zero_sd_tolerance)
            checks=preflight_fold(ref,built,prepared,scope,layer)
            print(f"[FEATURE PASS] {scope}/{layer} raw={counts['raw_predictor_columns']} "
                  f"numeric={counts['numeric']} cat={counts['categorical']} "
                  f"kept={checks['post_preprocessing_features']} "
                  f"time_s={time.monotonic()-tick:.1f}",flush=True)
            layer_result={"features":counts, "preprocessor":checks}
            if a.fit:
                models={}
                for alg in ALGORITHMS:
                    model_start=time.monotonic()
                    fitted=fit_one(alg,known_specs[alg],prepared,build_candidates,fit_candidate)
                    models[alg]=fitted
                    print(f"[FIT PASS] {scope}/{layer}/{alg} "
                          f"candidate={fitted['candidate_id']} "
                          f"converged={fitted['converged']} "
                          f"diagnostic_auc={fitted['roc_auc_diagnostic_only']:.3f} "
                          f"time_s={time.monotonic()-model_start:.1f}",flush=True)
                    passed+=1
                layer_result["models"]=models
            else:
                passed+=1
            scope_result[layer]=layer_result
            scope_result["_previous_build"]=built
        scope_result.pop("_previous_build",None)
        result["scopes"][scope]=scope_result
    print(f"[SMOKE PASS] scope={a.scope} mode={result['mode']} "
          f"{'fits' if a.fit else 'feature profiles'}={passed} "
          "Repeat001 only; no hyperparameter optimization or final training",flush=True)
    if a.report:
        output=a.report.expanduser().resolve()
        require(not output.exists(),f"Refusing to overwrite report {output}")
        output.parent.mkdir(parents=True,exist_ok=True)
        # Convert Counter keys to JSON-compatible strings.
        output.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=int)+"\n",encoding="utf-8")
        print(f"[REPORT] {output}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[SMOKE FAIL] {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        raise SystemExit(2) from exc
