#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create Scheme 013: add public features to the A9 and A3 model structures."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


A3 = "A3_core83_unweighted500"
A5 = "A5_core83_public"
A9 = "A9_core83_unweighted200"

A13 = "A13_core83_unweighted200_public"
A14 = "A14_core83_unweighted500_public"

DEFAULT_SCHEME011 = (
    "TRB/set/configs/schemes/"
    "trb_scheme_011_a1_a6_repeat100_fullgrid.yaml"
)
DEFAULT_SCHEME012 = (
    "TRB/set/configs/schemes/"
    "trb_scheme_012_additional6_repeat100_fullgrid.yaml"
)
DEFAULT_OUTPUT = (
    "TRB/set/configs/schemes/"
    "trb_scheme_013_a9_a3_public_repeat100_fullgrid.yaml"
)
SCHEME_ID = "trb_scheme_013_a9_a3_public_repeat100_fullgrid"
OUTPUT_ROOT = f"TRB/set/experiments/{SCHEME_ID}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the two-model TRB Scheme 013 public-addon YAML."
    )
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--scheme011", default=DEFAULT_SCHEME011)
    parser.add_argument("--scheme012", default=DEFAULT_SCHEME012)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"YAML not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def require_model(models: Mapping[str, Any], name: str, source: str) -> dict[str, Any]:
    value = models.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} does not contain model {name}")
    return copy.deepcopy(dict(value))


def main() -> int:
    args = parse_args()
    repo = Path(args.repository_root).expanduser().resolve()
    scheme011_path = resolve(repo, args.scheme011)
    scheme012_path = resolve(repo, args.scheme012)
    output_path = resolve(repo, args.output)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists; use --overwrite only for a documented replacement: "
            f"{output_path}"
        )

    s011 = load_yaml(scheme011_path)
    s012 = load_yaml(scheme012_path)

    if s011.get("feature_ablation_scheme_version") != s012.get(
        "feature_ablation_scheme_version"
    ):
        raise ValueError("Scheme 011 and Scheme 012 versions differ")

    for key in ("model_engine", "repeat_3mer_features"):
        if s011.get(key) != s012.get(key):
            raise ValueError(f"Scheme 011 and Scheme 012 differ in {key}")

    base011 = s011.get("scheme", {}).get("base_resolved_config")
    base012 = s012.get("scheme", {}).get("base_resolved_config")
    if base011 != base012:
        raise ValueError(
            "Scheme 011 and Scheme 012 use different base_resolved_config values"
        )

    models011 = s011.get("models")
    models012 = s012.get("models")
    if not isinstance(models011, Mapping) or not isinstance(models012, Mapping):
        raise ValueError("Both source schemes must contain a models mapping")

    a3 = require_model(models011, A3, "Scheme 011")
    a5 = require_model(models011, A5, "Scheme 011")
    a9 = require_model(models012, A9, "Scheme 012")

    public_features = list(a5.get("dynamic_public", []) or [])
    if len(public_features) != 8:
        raise ValueError(
            f"Expected exactly 8 public features from {A5}; "
            f"observed {len(public_features)}: {public_features}"
        )
    if len(public_features) != len(set(map(str, public_features))):
        raise ValueError("A5 dynamic_public contains duplicate feature names")

    if list(a3.get("dynamic_public", []) or []):
        raise ValueError(f"{A3} already contains dynamic_public features")
    if list(a9.get("dynamic_public", []) or []):
        raise ValueError(f"{A9} already contains dynamic_public features")

    groups_a3 = list(a3.get("static_feature_groups", []) or [])
    groups_a9 = list(a9.get("static_feature_groups", []) or [])
    if "repeat_3mer_unweighted_top500" not in groups_a3:
        raise ValueError(f"{A3} does not contain unweighted Top-500")
    if "repeat_3mer_unweighted_top200" not in groups_a9:
        raise ValueError(f"{A9} does not contain unweighted Top-200")

    a13 = copy.deepcopy(a9)
    a13["description"] = (
        "Core 83 plus repeat-specific Top-200 unweighted 3-mers "
        "and eight dynamic public-reference features."
    )
    a13["dynamic_public"] = copy.deepcopy(public_features)

    a14 = copy.deepcopy(a3)
    a14["description"] = (
        "Core 83 plus repeat-specific Top-500 unweighted 3-mers "
        "and eight dynamic public-reference features."
    )
    a14["dynamic_public"] = copy.deepcopy(public_features)

    scheme013 = copy.deepcopy(s011)
    scheme013["scheme"]["id"] = SCHEME_ID
    scheme013["scheme"]["description"] = (
        "Public-reference add-on test for the leading A9 Top-200 and "
        "A3 Top-500 unweighted 3-mer model structures across the same "
        "100 frozen repeated holdouts and full Elastic Net grid."
    )
    scheme013["scheme"]["output_root"] = OUTPUT_ROOT
    scheme013["models"] = {
        A13: a13,
        A14: a14,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(scheme013, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    engine = scheme013["model_engine"]
    candidates = len(engine["alpha_grid"]) * len(engine["lambda_grid"])
    print("Scheme 013 creation: COMPLETE")
    print(f"Output: {output_path}")
    print(f"Models: {list(scheme013['models'])}")
    print(f"Public features ({len(public_features)}): {public_features}")
    print(f"Candidates/model: {candidates}")
    print("Expected resolved predictor counts after preparation:")
    print(f"  {A13}: 83 + 200 + 8 = 291")
    print(f"  {A14}: 83 + 500 + 8 = 591")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
