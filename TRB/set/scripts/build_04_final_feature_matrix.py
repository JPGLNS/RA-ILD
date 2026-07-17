#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd

VERSION = "1.0.0"
ROOT = Path("/data/users/chenhaisheng/RA-ILD/TRB")

DEFAULTS = {
    "train_metadata": ROOT/"set/train/metadata_train_70.csv",
    "test_metadata": ROOT/"set/test/metadata_test_30.csv",
    "train_02": ROOT/"set/train/result/02_sample_level_features/02_sample_level_features_merged.csv",
    "test_02": ROOT/"set/test/result/02_sample_level_features/02_sample_level_features_merged.csv",
    "train_03_desc": ROOT/"set/train/result/03_public_features/03_descriptive_public_features.csv",
    "test_03_desc": ROOT/"set/test/result/03_public_features/03_descriptive_public_features.csv",
    "test_03_ref": ROOT/"set/test/result/03_public_features/03_test_reference_public_features.csv",
    "train_out": ROOT/"set/train/result/04_final_feature_matrix",
    "test_out": ROOT/"set/test/result/04_final_feature_matrix",
}

REFERENCE_DROP = {
    "Y_frequency","weighted_Y_frequency","weighted_length_other_freq",
    "nonpolar_aa_ratio","weighted_nonpolar_aa_ratio"
}
QC = {
    "total_reads_all_nt","total_reads_valid_aa","valid_read_ratio",
    "nt_clone_number_all","valid_nt_clone_number","valid_nt_row_ratio",
    "aa_clone_number","log1p_aa_clone_number"
}
DIVERSITY = {"Shannon","Simpson","inverse_Simpson","Gini","clonality"}
EXPANSION = {
    "top1_frequency","top5_cumulative_frequency","top10_cumulative_frequency",
    "top20_cumulative_frequency","top50_cumulative_frequency"
}
PUBLIC_DYNAMIC_PREFIXES = (
    "all_ref_public_","RA_specific_ref_","ILD_specific_ref_",
    "shared_ref_","ILD_RA_specific_ref_"
)
PUBLIC_DESC_PREFIXES = (
    "global_public_aa_","RA_group_public_aa_","ILD_group_public_aa_",
    "between_group_shared_aa_"
)

def parse_args():
    p = argparse.ArgumentParser(
        description="Build step-04 train/test feature matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["train","test","both"], default="both")
    p.add_argument("--train-metadata", default=str(DEFAULTS["train_metadata"]))
    p.add_argument("--test-metadata", default=str(DEFAULTS["test_metadata"]))
    p.add_argument("--train-02", default=str(DEFAULTS["train_02"]))
    p.add_argument("--test-02", default=str(DEFAULTS["test_02"]))
    p.add_argument("--train-03-descriptive", default=str(DEFAULTS["train_03_desc"]))
    p.add_argument("--test-03-descriptive", default=str(DEFAULTS["test_03_desc"]))
    p.add_argument("--test-03-reference", default=str(DEFAULTS["test_03_ref"]))
    p.add_argument("--train-output-dir", default=str(DEFAULTS["train_out"]))
    p.add_argument("--test-output-dir", default=str(DEFAULTS["test_out"]))
    p.add_argument("--metadata-id-col", default="libraryid")
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--metadata-output-cols",
                   default="cohort,patient,age,sex,material,batch")
    p.add_argument("--expected-train-samples", type=int, default=123)
    p.add_argument("--expected-test-samples", type=int, default=51)
    p.add_argument("--expected-02-columns", type=int, default=1097)
    p.add_argument("--expected-03-descriptive-columns", type=int, default=26)
    p.add_argument("--expected-test-reference-columns", type=int, default=19)
    p.add_argument("--float-tolerance", type=float, default=1e-10)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    a.metadata_output_cols = [x.strip() for x in a.metadata_output_cols.split(",") if x.strip()]
    return a

def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")], errors="ignore")
    if df.columns.duplicated().any():
        dup = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"{label} duplicated columns: {dup[:20]}")
    return df

def check_expected(actual: int, expected: int, label: str):
    if expected > 0 and actual != expected:
        raise ValueError(f"{label}: expected {expected}, observed {actual}")

def check_ids(df: pd.DataFrame, id_col: str, label: str):
    if id_col not in df.columns:
        raise ValueError(f"{label} missing {id_col}")
    ids = df[id_col].astype(str).str.strip()
    if ids.eq("").any() or ids.isna().any():
        raise ValueError(f"{label} has empty {id_col}")
    if ids.duplicated().any():
        vals = ids[ids.duplicated(keep=False)].unique().tolist()[:20]
        raise ValueError(f"{label} duplicated IDs: {vals}")
    df[id_col] = ids

def check_no_missing(df: pd.DataFrame, label: str):
    n = int(df.isna().sum().sum())
    if n:
        bad = df.isna().sum()
        raise ValueError(f"{label} has {n} missing values: {bad[bad>0].head(20).to_dict()}")

def load_metadata(path, id_col, out_id, out_cols, expected, label):
    df = read_csv(path, label)
    need = [id_col] + list(out_cols)
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{label} missing columns: {miss}")
    df = df[need].rename(columns={id_col: out_id}).copy()
    check_ids(df, out_id, label)
    check_no_missing(df, label)
    check_expected(len(df), expected, f"{label} rows")
    df["cohort"] = df["cohort"].astype(str).str.upper()
    if set(df["cohort"]) != {"RA","ILD"}:
        raise ValueError(f"{label} cohort must be RA/ILD only")
    df["age"] = pd.to_numeric(df["age"], errors="raise")
    return df

def load_numeric_matrix(path, id_col, expected_rows, expected_cols, label):
    df = read_csv(path, label)
    check_ids(df, id_col, label)
    check_no_missing(df, label)
    check_expected(len(df), expected_rows, f"{label} rows")
    check_expected(len(df.columns), expected_cols, f"{label} columns")
    for c in df.columns:
        if c == id_col: continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.isna().any():
            raise ValueError(f"{label}.{c} is not fully numeric")
        df[c] = x
    return df

def exact_id_set(a, b, id_col, la, lb):
    sa, sb = set(a[id_col]), set(b[id_col])
    if sa != sb:
        raise ValueError(
            f"ID mismatch {la} vs {lb}; only_{la}={sorted(sa-sb)[:20]}, "
            f"only_{lb}={sorted(sb-sa)[:20]}")

def merge_base(meta, f02, id_col, meta_cols, label):
    exact_id_set(meta, f02, id_col, f"{label}_meta", f"{label}_02")
    out = meta.merge(f02, on=id_col, how="left", validate="one_to_one", sort=False)
    order = [id_col] + list(meta_cols) + [c for c in f02.columns if c != id_col]
    out = out[order]
    check_no_missing(out, f"{label}_base")
    return out

def compare_numeric(a, b, tol, label):
    aa = pd.to_numeric(a, errors="raise").to_numpy(float)
    bb = pd.to_numeric(b, errors="raise").to_numpy(float)
    if not np.allclose(aa, bb, rtol=tol, atol=tol):
        d = np.abs(aa-bb)
        i = int(np.argmax(d))
        raise ValueError(f"{label} mismatch at row {i}; diff={d[i]}")

def merge_desc(base, desc, id_col, tol, label):
    exact_id_set(base, desc, id_col, f"{label}_base", f"{label}_03_desc")
    d = desc.copy()
    if "aa_clone_number" in d.columns:
        chk = base[[id_col,"aa_clone_number"]].merge(
            d[[id_col,"aa_clone_number"]], on=id_col, validate="one_to_one",
            suffixes=("_base","_03"), sort=False)
        compare_numeric(chk["aa_clone_number_base"], chk["aa_clone_number_03"], tol,
                        f"{label} aa_clone_number")
        d = d.drop(columns="aa_clone_number")
    collisions = sorted((set(base.columns)&set(d.columns))-{id_col})
    if collisions:
        raise ValueError(f"{label} descriptive collisions: {collisions}")
    out = base.merge(d, on=id_col, how="left", validate="one_to_one", sort=False)
    check_no_missing(out, f"{label}_descriptive")
    return out

def merge_test_final(base, ref, id_col):
    exact_id_set(base, ref, id_col, "test_base", "test_03_ref")
    collisions = sorted((set(base.columns)&set(ref.columns))-{id_col})
    if collisions:
        raise ValueError(f"test final collisions: {collisions}")
    out = base.merge(ref, on=id_col, how="left", validate="one_to_one", sort=False)
    check_no_missing(out, "test_final")
    return out

def infer_type(s: pd.Series):
    if pd.api.types.is_integer_dtype(s): return "integer"
    if pd.api.types.is_numeric_dtype(s): return "numeric"
    return "categorical"

def group_of(n: str):
    if n=="sample_id": return "identifier"
    if n=="cohort": return "outcome"
    if n=="patient": return "identifier_trace"
    if n in {"age","sex"}: return "clinical"
    if n=="material": return "clinical_optional"
    if n=="batch": return "technical_qc"
    if n in QC: return "tcr_qc_depth"
    if n in DIVERSITY: return "tcr_diversity"
    if n in EXPANSION: return "tcr_clonal_expansion"
    if n.startswith("unweighted_3mer_"): return "tcr_3mer_unweighted"
    if n.startswith("weighted_3mer_"): return "tcr_3mer_weighted"
    if n.startswith(PUBLIC_DESC_PREFIXES): return "public_descriptive"
    if n.startswith(PUBLIC_DYNAMIC_PREFIXES): return "public_reference_dynamic"
    if "aa_length" in n or n.startswith("weighted_length_"): return "tcr_aa_length"
    if n.endswith("_frequency"):
        core = n.removeprefix("weighted_").split("_")[0]
        if len(core)==1:
            return "tcr_aa_composition_weighted" if n.startswith("weighted_") else "tcr_aa_composition_unweighted"
    if any(t in n for t in ("hydrophobicity","charge","aromaticity",
                            "basic_aa_ratio","acidic_aa_ratio","polar_aa_ratio","nonpolar_aa_ratio")):
        return "tcr_physicochemical"
    return "tcr_other_static"

def role_of(n, g):
    if n=="sample_id": return "join_key"
    if n=="cohort": return "outcome_label"
    if n=="patient": return "trace_only"
    if n=="batch": return "qc_only"
    if g=="clinical": return "primary_predictor"
    if g=="clinical_optional": return "optional_predictor"
    if g=="tcr_qc_depth": return "qc_or_sensitivity_predictor"
    if g=="public_descriptive": return "descriptive_only"
    if g=="public_reference_dynamic": return "dynamic_predictor_train_fixed_predictor_test"
    if n in REFERENCE_DROP: return "composition_reference_drop"
    if g.startswith("tcr_"): return "candidate_predictor"
    return "review"

def model_default(role):
    if role in {"join_key","outcome_label","trace_only","qc_only","descriptive_only","composition_reference_drop"}:
        return "no"
    if role=="optional_predictor": return "optional"
    if role=="qc_or_sensitivity_predictor": return "sensitivity"
    if role=="dynamic_predictor_train_fixed_predictor_test": return "dynamic"
    return "yes"

def transform_text(n, dtype, role):
    if n=="cohort": return "binary encode in model pipeline"
    if n in {"sex","material"}: return "one-hot encode in model pipeline"
    if n=="batch": return "exclude from primary model"
    if n=="patient": return "exclude from predictors"
    if n in REFERENCE_DROP: return "drop before model fitting"
    if dtype in {"numeric","integer"} and role not in {"join_key","outcome_label","trace_only","qc_only","descriptive_only"}:
        return "standardize within each training fold"
    return "none"

def build_manifest(train_base, test_base, train_desc, test_desc, test_final):
    union = []
    for df in (train_base, train_desc, test_final, test_desc):
        for c in df.columns:
            if c not in union: union.append(c)
    meta_set = {"sample_id","cohort","patient","age","sex","material","batch"}
    rows = []
    for i,c in enumerate(union,1):
        if c in train_base.columns:
            source = "metadata" if c in meta_set else "02_sample_level_features_merged"
        elif c in train_desc.columns:
            source = "03_descriptive_public_features"
        else:
            source = "03_test_reference_public_features"
        exemplar = next(df[c] for df in (train_base,train_desc,test_final,test_desc) if c in df.columns)
        dtype = infer_type(exemplar)
        group = group_of(c)
        role = role_of(c,group)
        rows.append({
            "feature_name": c,
            "feature_order": i,
            "source_file": source,
            "feature_group": group,
            "feature_role": role,
            "data_type": dtype,
            "present_in_train_base": c in train_base.columns,
            "present_in_test_base": c in test_base.columns,
            "present_in_train_descriptive": c in train_desc.columns,
            "present_in_test_descriptive": c in test_desc.columns,
            "present_in_test_final": c in test_final.columns,
            "eligible_for_fixed_train_cv_input": (
                c in train_base.columns and role not in {
                    "join_key","outcome_label","trace_only","qc_only",
                    "descriptive_only","composition_reference_drop"}),
            "dynamic_in_training": group=="public_reference_dynamic",
            "model_default": model_default(role),
            "transformation": transform_text(c,dtype,role),
            "reference_drop": c in REFERENCE_DROP,
            "notes": ("Static train descriptive values have self-inclusion; generate "
                      "fold-specific values during cross-validation."
                      if group=="public_reference_dynamic" else "")
        })
    return pd.DataFrame(rows)

def ensure_overwrite(paths, overwrite):
    exists = [p for p in paths if p.exists()]
    if exists and not overwrite:
        raise FileExistsError("Outputs exist; add --overwrite:\n" + "\n".join(map(str,exists)))

def write_summary(path, title, inputs, outputs, shapes, runtime):
    lines = [f"# {title}","",f"- Script version: `{VERSION}`",f"- Runtime: `{runtime:.2f} seconds`","",
             "## Inputs",""]
    lines += [f"- `{k}`: `{v}`" for k,v in inputs.items()]
    lines += ["","## Output shapes",""]
    lines += [f"- `{k}`: **{v[0]} × {v[1]}**" for k,v in shapes.items()]
    lines += ["","## Validation","",
              "- Train/test sample IDs are disjoint.",
              "- All merges were one-to-one by sample_id.",
              "- Train/test step-02 schemas are identical.",
              "- Train/test base schemas are identical.",
              "- Train/test descriptive schemas are identical.",
              "- Step-02 and step-03 aa_clone_number values are identical.",
              "- No missing values were found.",
              "- No fixed 04_train_final_feature_matrix.csv was generated.",
              "","## Notes","",
              "- patient is retained only for traceability and is not a predictor.",
              "- batch is retained for QC and sensitivity analysis, not the primary model.",
              "- Static train descriptive public features have self-inclusion.",
              "- Training reference-public features must be generated inside cross-validation folds.",
              "- Composition reference columns are retained in files but flagged in the manifest.",
              "","## Outputs",""]
    lines += [f"- `{k}`: `{v}`" for k,v in outputs.items()]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")

def main():
    a = parse_args()
    t0 = time.time()
    P = {k:Path(getattr(a,k)).expanduser().resolve() for k in [
        "train_metadata","test_metadata","train_02","test_02",
        "train_03_descriptive","test_03_descriptive","test_03_reference"]}
    train_out = Path(a.train_output_dir).expanduser().resolve()
    test_out = Path(a.test_output_dir).expanduser().resolve()
    train_out.mkdir(parents=True,exist_ok=True)
    test_out.mkdir(parents=True,exist_ok=True)
    train_files = {
        "train_base":train_out/"04_train_base_feature_matrix.csv",
        "train_descriptive":train_out/"04_train_descriptive_feature_matrix.csv",
        "manifest":train_out/"04_feature_manifest.csv",
        "summary":train_out/"04_feature_matrix_build_summary.md"}
    test_files = {
        "test_base":test_out/"04_test_base_feature_matrix.csv",
        "test_descriptive":test_out/"04_test_descriptive_feature_matrix.csv",
        "test_final":test_out/"04_test_final_feature_matrix.csv",
        "summary":test_out/"04_feature_matrix_build_summary.md"}
    selected = []
    if a.mode in {"train","both"}: selected += list(train_files.values())
    if a.mode in {"test","both"}: selected += list(test_files.values())
    ensure_overwrite(selected,a.overwrite)

    tm = load_metadata(P["train_metadata"],a.metadata_id_col,a.sample_id_col,
                       a.metadata_output_cols,a.expected_train_samples,"train_metadata")
    sm = load_metadata(P["test_metadata"],a.metadata_id_col,a.sample_id_col,
                       a.metadata_output_cols,a.expected_test_samples,"test_metadata")
    overlap = set(tm[a.sample_id_col]) & set(sm[a.sample_id_col])
    if overlap: raise ValueError(f"Train/test sample overlap: {sorted(overlap)[:20]}")

    t02 = load_numeric_matrix(P["train_02"],a.sample_id_col,a.expected_train_samples,
                              a.expected_02_columns,"train_02")
    s02 = load_numeric_matrix(P["test_02"],a.sample_id_col,a.expected_test_samples,
                              a.expected_02_columns,"test_02")
    if t02.columns.tolist()!=s02.columns.tolist():
        raise ValueError("Train/test step-02 columns or order differ")

    train_base = merge_base(tm,t02,a.sample_id_col,a.metadata_output_cols,"train")
    test_base = merge_base(sm,s02,a.sample_id_col,a.metadata_output_cols,"test")
    if train_base.columns.tolist()!=test_base.columns.tolist():
        raise RuntimeError("Train/test base schemas differ")

    td = load_numeric_matrix(P["train_03_descriptive"],a.sample_id_col,
                             a.expected_train_samples,a.expected_03_descriptive_columns,
                             "train_03_descriptive")
    sd = load_numeric_matrix(P["test_03_descriptive"],a.sample_id_col,
                             a.expected_test_samples,a.expected_03_descriptive_columns,
                             "test_03_descriptive")
    if td.columns.tolist()!=sd.columns.tolist():
        raise ValueError("Train/test step-03 descriptive schemas differ")
    train_desc = merge_desc(train_base,td,a.sample_id_col,a.float_tolerance,"train")
    test_desc = merge_desc(test_base,sd,a.sample_id_col,a.float_tolerance,"test")
    if train_desc.columns.tolist()!=test_desc.columns.tolist():
        raise RuntimeError("Train/test descriptive schemas differ")

    sr = load_numeric_matrix(P["test_03_reference"],a.sample_id_col,
                             a.expected_test_samples,a.expected_test_reference_columns,
                             "test_03_reference")
    test_final = merge_test_final(test_base,sr,a.sample_id_col)
    manifest = build_manifest(train_base,test_base,train_desc,test_desc,test_final)

    if a.mode in {"train","both"}:
        train_base.to_csv(train_files["train_base"],index=False)
        train_desc.to_csv(train_files["train_descriptive"],index=False)
        manifest.to_csv(train_files["manifest"],index=False)
    if a.mode in {"test","both"}:
        test_base.to_csv(test_files["test_base"],index=False)
        test_desc.to_csv(test_files["test_descriptive"],index=False)
        test_final.to_csv(test_files["test_final"],index=False)

    runtime = time.time()-t0
    if a.mode in {"train","both"}:
        write_summary(train_files["summary"],"04 Train Feature Matrix Build Summary",
                      {"train_metadata":P["train_metadata"],"train_02":P["train_02"],
                       "train_03_descriptive":P["train_03_descriptive"]},
                      train_files,
                      {"04_train_base_feature_matrix.csv":train_base.shape,
                       "04_train_descriptive_feature_matrix.csv":train_desc.shape,
                       "04_feature_manifest.csv":manifest.shape},runtime)
    if a.mode in {"test","both"}:
        write_summary(test_files["summary"],"04 Test Feature Matrix Build Summary",
                      {"test_metadata":P["test_metadata"],"test_02":P["test_02"],
                       "test_03_descriptive":P["test_03_descriptive"],
                       "test_03_reference":P["test_03_reference"]},
                      test_files,
                      {"04_test_base_feature_matrix.csv":test_base.shape,
                       "04_test_descriptive_feature_matrix.csv":test_desc.shape,
                       "04_test_final_feature_matrix.csv":test_final.shape},runtime)

    print("="*80)
    print("04 Final Feature Matrix Build")
    print("="*80)
    print("Train base:",train_base.shape)
    print("Test base:",test_base.shape)
    print("Train descriptive:",train_desc.shape)
    print("Test descriptive:",test_desc.shape)
    print("Test final:",test_final.shape)
    print("Manifest:",manifest.shape)
    print("No 04_train_final_feature_matrix.csv was generated.")
    print("Training reference-public features remain dynamic for cross-validation.")
    for d in ([train_files] if a.mode=="train" else [test_files] if a.mode=="test" else [train_files,test_files]):
        for k,v in d.items(): print(f"- {k}: {v}")
    print(f"Runtime: {runtime:.2f}s")
    return 0

if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}",file=sys.stderr)
        raise
