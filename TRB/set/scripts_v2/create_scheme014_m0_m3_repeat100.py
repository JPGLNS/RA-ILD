#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create Scheme 014: original M0-M3 models under 100 repeated holdouts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


REPOSITORY_ROOT = Path(
    "/data/users/chenhaisheng/RA-ILD"
).resolve()

SOURCE_SCHEME = (
    REPOSITORY_ROOT
    / "TRB/set/configs/schemes/"
      "trb_scheme_011_a1_a6_repeat100_fullgrid.yaml"
)

OUTPUT_SCHEME = (
    REPOSITORY_ROOT
    / "TRB/set/configs/schemes/"
      "trb_scheme_014_m0_m3_repeat100.yaml"
)

SCHEME_ID = "trb_scheme_014_m0_m3_repeat100"

OUTPUT_ROOT = (
    "TRB/set/experiments/"
    "trb_scheme_014_m0_m3_repeat100"
)

SOURCE_TRB_MODEL = "A4_core83_both500"
SOURCE_PUBLIC_MODEL = "A5_core83_public"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"YAML not found: {path}")

    value = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"YAML root must be a mapping: {path}"
        )

    return value


def require_model(
    models: Mapping[str, Any],
    model_name: str,
) -> dict[str, Any]:

    value = models.get(model_name)

    if not isinstance(value, Mapping):
        raise ValueError(
            f"Source scheme does not contain model: "
            f"{model_name}"
        )

    return copy.deepcopy(dict(value))


def main() -> int:
    if OUTPUT_SCHEME.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_SCHEME}\n"
            "Delete it explicitly before regenerating."
        )

    source = load_yaml(SOURCE_SCHEME)

    models = source.get("models")

    if not isinstance(models, Mapping):
        raise ValueError(
            "Source Scheme 011 does not contain "
            "a models mapping"
        )

    # A4 represents Core83 + repeat-specific
    # unweighted500 + weighted500.
    source_trb = require_model(
        models,
        SOURCE_TRB_MODEL,
    )

    static_groups = list(
        source_trb.get(
            "static_feature_groups",
            [],
        )
        or []
    )

    expected_groups = {
        "core83",
        "repeat_3mer_both_top500",
    }

    if set(static_groups) != expected_groups:
        raise ValueError(
            f"Unexpected A4 feature groups: "
            f"{static_groups}"
        )

    # Extract the exact 8 public features from A5.
    source_public = require_model(
        models,
        SOURCE_PUBLIC_MODEL,
    )

    public_features = list(
        source_public.get(
            "dynamic_public",
            [],
        )
        or []
    )

    if len(public_features) != 8:
        raise ValueError(
            "Expected exactly 8 dynamic public "
            f"features; observed {len(public_features)}: "
            f"{public_features}"
        )

    if len(public_features) != len(
        set(map(str, public_features))
    ):
        raise ValueError(
            "Duplicate dynamic public feature names"
        )

    scheme014 = copy.deepcopy(source)

    scheme014["scheme"]["id"] = SCHEME_ID

    scheme014["scheme"]["description"] = (
        "Original M0-M3 hierarchical TRB models "
        "evaluated across the same 100 frozen "
        "repeated holdouts. Repeat-specific Top-500 "
        "unweighted and weighted 3-mer selection is "
        "used inside each outer-training partition."
    )

    scheme014["scheme"]["output_root"] = OUTPUT_ROOT

    # Reproduce the original M0-M3 tuning grid.
    # Lambda grid remains the same eight values.
    scheme014["model_engine"] = {
        "alpha_grid": [
            0.1,
            0.5,
            0.9,
        ],
        "lambda_grid": [
            0.01,
            0.03,
            0.1,
            0.3,
            1.0,
            3.0,
            10.0,
            30.0,
        ],
    }

    scheme014["models"] = {
        "M0_clinical": {
            "description": (
                "Clinical baseline with age and sex."
            ),
            "numeric": [
                "age",
            ],
            "categorical": [
                "sex",
            ],
            "static_feature_groups": [],
            "dynamic_public": [],
        },

        "M1_static_tcr": {
            "description": (
                "Age and sex plus Core83 and "
                "repeat-specific Top-500 unweighted "
                "and weighted 3-mer features."
            ),
            "numeric": [
                "age",
            ],
            "categorical": [
                "sex",
            ],
            "static_feature_groups": [
                "core83",
                "repeat_3mer_both_top500",
            ],
            "dynamic_public": [],
        },

        "M2_static_tcr_public": {
            "description": (
                "M1 plus eight dynamic public-reference "
                "features."
            ),
            "numeric": [
                "age",
            ],
            "categorical": [
                "sex",
            ],
            "static_feature_groups": [
                "core83",
                "repeat_3mer_both_top500",
            ],
            "dynamic_public": copy.deepcopy(
                public_features
            ),
        },

        "M3_static_tcr_public_material": {
            "description": (
                "M2 plus sample material."
            ),
            "numeric": [
                "age",
            ],
            "categorical": [
                "sex",
                "material",
            ],
            "static_feature_groups": [
                "core83",
                "repeat_3mer_both_top500",
            ],
            "dynamic_public": copy.deepcopy(
                public_features
            ),
        },
    }

    OUTPUT_SCHEME.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_SCHEME.write_text(
        yaml.safe_dump(
            scheme014,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print("Scheme 014 creation: COMPLETE")
    print(f"Source: {SOURCE_SCHEME}")
    print(f"Output: {OUTPUT_SCHEME}")
    print(f"Scheme ID: {SCHEME_ID}")
    print(f"Models: {list(scheme014['models'])}")
    print(
        f"Public features ({len(public_features)}): "
        f"{public_features}"
    )
    print(
        "Alpha grid:",
        scheme014["model_engine"]["alpha_grid"],
    )
    print(
        "Lambda grid:",
        scheme014["model_engine"]["lambda_grid"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
