#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import pandas as pd
import yaml

TRUE = {"true", "1", "yes"}

def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin(TRUE)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    manifest_path = root / config["data"]["train"]["feature_manifest"]
    base_path = root / config["data"]["train"]["base_matrix"]
    manifest = pd.read_csv(manifest_path)
    base = pd.read_csv(base_path, nrows=1)
    required = {
        "feature_name", "feature_group", "feature_role",
        "present_in_train_base", "reference_drop",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    selected = manifest.loc[
        as_bool(manifest["present_in_train_base"])
        & manifest["feature_group"].astype(str).str.startswith("igh_")
        & manifest["feature_role"].astype(str).eq("candidate_predictor")
        & ~as_bool(manifest["reference_drop"]),
        "feature_name",
    ].astype(str).tolist()
    absent = [name for name in selected if name not in base.columns]
    weighted = [x for x in selected if re.fullmatch(r"weighted_3mer_[A-Z]{3}", x)]
    unweighted = [x for x in selected if re.fullmatch(r"unweighted_3mer_[A-Z]{3}", x)]
    core = [x for x in selected if x not in set(weighted) | set(unweighted)]
    weighted_k = {x.rsplit("_", 1)[-1] for x in weighted}
    unweighted_k = {x.rsplit("_", 1)[-1] for x in unweighted}
    result = {
        "selected_static": len(selected),
        "core": len(core),
        "weighted_3mer": len(weighted),
        "unweighted_3mer": len(unweighted),
        "same_3mer_vocabulary": weighted_k == unweighted_k,
        "missing_from_base": len(absent),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    expected = {
        "selected_static": 1083,
        "core": 83,
        "weighted_3mer": 500,
        "unweighted_3mer": 500,
        "same_3mer_vocabulary": True,
        "missing_from_base": 0,
    }
    if result != expected:
        raise SystemExit(
            "IGH_STATIC_FEATURE_COMPOSITION_FAIL\n"
            f"observed={result}\nexpected={expected}"
        )
    print("IGH_STATIC_FEATURE_COMPOSITION_PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
