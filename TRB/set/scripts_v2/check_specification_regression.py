#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression-check V2 model specifications against frozen V1 artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.config import load_experiment_config  # noqa: E402
from ra_ild_trb.specifications import (  # noqa: E402
    HyperparameterCandidate,
    build_hyperparameter_grid,
    parse_candidate_pair_mapping,
    resolve_all_model_specifications,
    select_best_tuning_candidate,
    select_static_tcr_features,
    selected_pairs_from_oof,
    validate_locked_candidate_selection,
    validate_tuning_grid_frame,
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare V2 model specifications and tuning candidates with V1 outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--float-tolerance", type=float, default=1e-12)
    return parser.parse_args()


def add(results: List[CheckResult], name: str, passed: bool, detail: str) -> None:
    results.append(CheckResult(name=name, passed=bool(passed), detail=detail))


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path)


def unique_static_list(frame: pd.DataFrame) -> List[str]:
    if "feature_name" not in frame.columns:
        raise ValueError("V1 static feature list lacks feature_name")
    values = frame["feature_name"].astype(str).tolist()
    if not values or len(values) != len(set(values)):
        raise ValueError("V1 static feature list is empty or duplicated")
    return values


def step07_selected_pair(configuration: Mapping[str, Any]) -> HyperparameterCandidate:
    selected = configuration.get("selected_parameters")
    if isinstance(selected, Mapping):
        alpha = selected.get("alpha", selected.get("l1_ratio_alpha"))
        lambda_value = selected.get("lambda")
    else:
        alpha = configuration.get("selected_alpha")
        lambda_value = configuration.get("selected_lambda")
    if alpha is None or lambda_value is None:
        raise ValueError("Step 07 configuration lacks selected alpha/lambda")
    return HyperparameterCandidate(float(alpha), float(lambda_value))


def same_pair(
    left: HyperparameterCandidate,
    right: HyperparameterCandidate,
    tolerance: float,
) -> bool:
    return math.isclose(left.alpha, right.alpha, rel_tol=tolerance, abs_tol=tolerance) and math.isclose(
        left.lambda_value,
        right.lambda_value,
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def main() -> int:
    args = parse_args()
    results: List[CheckResult] = []
    try:
        config = load_experiment_config(
            Path(args.config),
            repository_root=Path(args.repository_root) if args.repository_root else None,
        )
        selection = config.section("model_selection")
        engine = config.section("model_engine")
        models = config.section("models")
        public = config.section("public_reference")
        final = config.section("final_model")

        base = read_csv(
            config.path("data.train.base_matrix", must_exist=True, expect="file"),
            "train base matrix",
        )
        manifest = read_csv(
            config.path("data.train.feature_manifest", must_exist=True, expect="file"),
            "feature manifest",
        )
        v1_static_frame = read_csv(
            config.path(
                "preprocessing.regression_task.static_feature_list",
                must_exist=True,
                expect="file",
            ),
            "V1 static feature list",
        )
        v1_static = unique_static_list(v1_static_frame)
        selected_static = select_static_tcr_features(
            manifest,
            available_columns=base.columns,
            require_all_available=True,
        )

        add(
            results,
            "Training base sample count",
            len(base) == int(config.raw["data"]["train"]["expected_samples"]),
            f"observed={len(base)}, expected={config.raw['data']['train']['expected_samples']}",
        )
        expected_static = int(selection["expected_static_feature_count"])
        add(
            results,
            "Static feature count",
            len(selected_static) == expected_static,
            f"observed={len(selected_static)}, expected={expected_static}",
        )
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(selected_static, v1_static))
                if left != right
            ),
            None,
        )
        exact_static = selected_static == v1_static
        detail = f"V2={len(selected_static)}, V1={len(v1_static)}"
        if mismatch is not None:
            detail += f", first_mismatch={mismatch}:{selected_static[mismatch]}!={v1_static[mismatch]}"
        add(results, "Static feature exact order", exact_static, detail)
        add(
            results,
            "Static features present in base matrix",
            all(name in base.columns for name in selected_static),
            f"n_features={len(selected_static)}",
        )

        resolved = resolve_all_model_specifications(
            models,
            static_features=selected_static,
            allowed_dynamic_public=public["dynamic_features"],
            static_group_name=str(selection["static_feature_group"]),
        )
        add(
            results,
            "Resolved model set",
            list(resolved) == list(models),
            f"models={list(resolved)}",
        )
        expected_columns = selection["expected_resolved_columns"]
        for name, specification in resolved.items():
            expected = expected_columns[name]
            passed = (
                len(specification.numeric) == int(expected["numeric"])
                and len(specification.categorical) == int(expected["categorical"])
            )
            add(
                results,
                f"{name} resolved column counts",
                passed,
                (
                    f"numeric={len(specification.numeric)}/{expected['numeric']}, "
                    f"categorical={len(specification.categorical)}/{expected['categorical']}"
                ),
            )

        public_features = list(map(str, public["dynamic_features"]))
        dynamic_models_ok = (
            list(resolved["M2_static_tcr_public"].numeric[-len(public_features):])
            == public_features
            and list(
                resolved["M3_static_tcr_public_material"].numeric[-len(public_features):]
            )
            == public_features
        )
        add(
            results,
            "Dynamic public feature order",
            dynamic_models_ok,
            f"n_dynamic={len(public_features)}",
        )

        grid = build_hyperparameter_grid(engine["alpha_grid"], engine["lambda_grid"])
        expected_grid_size = len(engine["alpha_grid"]) * len(engine["lambda_grid"])
        add(
            results,
            "Configured candidate grid size",
            len(grid) == expected_grid_size,
            f"observed={len(grid)}, expected={expected_grid_size}",
        )

        tuning = read_csv(
            config.path(
                "model_selection.regression_task.inner_tuning_results",
                must_exist=True,
                expect="file",
            ),
            "V1 inner tuning results",
        )
        validate_tuning_grid_frame(
            tuning,
            model_names=list(resolved),
            candidates=grid,
            inverse_lambda_tolerance=args.float_tolerance,
        )
        add(
            results,
            "V1 tuning grid exact coverage",
            True,
            f"rows={len(tuning)}, expected={len(resolved) * len(grid)}",
        )
        add(
            results,
            "V1 tuning result row count",
            len(tuning) == len(resolved) * len(grid),
            f"observed={len(tuning)}, expected={len(resolved) * len(grid)}",
        )

        oof = read_csv(
            config.path(
                "modeling.regression_task.inner_selected_oof",
                must_exist=True,
                expect="file",
            ),
            "V1 selected inner OOF predictions",
        )
        oof_selected = selected_pairs_from_oof(oof)
        for model_name in resolved:
            ranked_selected = select_best_tuning_candidate(
                tuning.loc[tuning["model"].astype(str) == model_name]
            )
            observed = oof_selected[model_name]
            add(
                results,
                f"{model_name} selected tuning pair",
                same_pair(ranked_selected, observed, args.float_tolerance),
                (
                    f"ranked=({ranked_selected.alpha:g},{ranked_selected.lambda_value:g}), "
                    f"V1=({observed.alpha:g},{observed.lambda_value:g})"
                ),
            )

        final_candidates = parse_candidate_pair_mapping(
            final["candidate_pairs"], label="final_model.candidate_pairs"
        )
        locked = HyperparameterCandidate(
            alpha=float(final["selected_alpha"]),
            lambda_value=float(final["selected_lambda"]),
        )
        validate_locked_candidate_selection(
            full_grid=grid,
            candidate_subset=final_candidates,
            selected=locked,
        )
        add(
            results,
            "Final candidate subset and locked pair",
            True,
            (
                f"candidate_pairs={[(x.alpha, x.lambda_value) for x in final_candidates]}, "
                f"selected=({locked.alpha:g},{locked.lambda_value:g})"
            ),
        )

        step07_path = config.path(
            "final_model.configuration", must_exist=True, expect="file"
        )
        step07 = json.loads(step07_path.read_text(encoding="utf-8"))
        step07_pair = step07_selected_pair(step07)
        add(
            results,
            "Step 07 selected pair",
            same_pair(step07_pair, locked, args.float_tolerance),
            (
                f"Step07=({step07_pair.alpha:g},{step07_pair.lambda_value:g}), "
                f"config=({locked.alpha:g},{locked.lambda_value:g})"
            ),
        )

    except Exception as exc:
        add(results, "Unhandled regression-check error", False, f"{type(exc).__name__}: {exc}")

    passed = sum(item.passed for item in results)
    failed = len(results) - passed
    print(f"Checks={len(results)} PASS={passed} FAIL={failed}")
    for item in results:
        print(f"[{'PASS' if item.passed else 'FAIL'}] {item.name}: {item.detail}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
