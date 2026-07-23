#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check key files and locked values from the frozen TRB V1 baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.config import load_experiment_config  # noqa: E402


@dataclass
class Result:
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the frozen TRB V1 baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--scope", choices=("inputs", "training", "full"), default="full")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def add(results: list[Result], name: str, ok: bool, detail: str) -> None:
    results.append(Result(name, "PASS" if ok else "FAIL", detail))


def csv_rows(
    results: list[Result],
    path: Path,
    expected: int,
    label: str,
    unique_col: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        add(results, label, False, str(exc))
        return None
    add(results, f"{label} rows", len(frame) == expected, f"observed={len(frame)}, expected={expected}")
    if unique_col:
        if unique_col not in frame.columns:
            add(results, f"{label} unique {unique_col}", False, "column missing")
        else:
            duplicates = int(frame[unique_col].astype(str).duplicated().sum())
            add(results, f"{label} unique {unique_col}", duplicates == 0, f"duplicates={duplicates}")
    return frame


def dotted_get(mapping: Mapping[str, Any], key: str) -> Any:
    value: Any = mapping
    for token in key.split("."):
        if not isinstance(value, Mapping) or token not in value:
            raise KeyError(key)
        value = value[token]
    return value


def first_value(mapping: Mapping[str, Any], keys: Sequence[str]) -> tuple[Optional[str], Any]:
    for key in keys:
        try:
            return key, dotted_get(mapping, key)
        except KeyError:
            pass
    return None, None


def close(observed: Any, expected: float, tolerance: float) -> bool:
    try:
        return math.isclose(float(observed), float(expected), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def check_step07(results: list[Result], config, tolerance: float) -> None:
    final = config.raw["final_model"]
    paths = {
        "bundle": config.path("final_model.bundle"),
        "configuration": config.path("final_model.configuration"),
        "reference_masks": config.path("final_model.reference_masks"),
    }
    for label, path in paths.items():
        add(results, f"Step 07 {label} exists", path.is_file(), str(path))

    if paths["configuration"].is_file():
        payload = json.loads(paths["configuration"].read_text(encoding="utf-8"))
        checks = (
            ("alpha", ("selected_parameters.alpha", "selected_alpha"), final["selected_alpha"]),
            ("lambda", ("selected_parameters.lambda", "selected_lambda"), final["selected_lambda"]),
            ("threshold", ("locked_threshold", "threshold"), final["locked_threshold"]),
        )
        for label, keys, expected in checks:
            key, observed = first_value(payload, keys)
            add(results, f"Step 07 locked {label}", key is not None and close(observed, expected, tolerance), f"key={key}, observed={observed}, expected={expected}")
        key, observed = first_value(payload, ("independent_test_read",))
        add(results, "Step 07 independent test unread", key is not None and observed is False, f"observed={observed}")

    if paths["reference_masks"].is_file():
        masks = np.load(paths["reference_masks"])
        expected = config.raw["public_reference"]["expected_sizes"]
        key_map = {
            "global": "global_mask",
            "RA_specific": "RA_specific_mask",
            "ILD_specific": "ILD_specific_mask",
            "shared": "shared_mask",
        }
        loaded = {}
        for name, npz_key in key_map.items():
            if npz_key not in masks.files:
                add(results, f"Reference mask {npz_key}", False, "missing")
                continue
            mask = np.asarray(masks[npz_key]).astype(bool)
            loaded[npz_key] = mask
            observed = int(mask.sum())
            add(results, f"Reference size {name}", observed == int(expected[name]), f"observed={observed}, expected={expected[name]}")
        required = {"RA_specific_mask", "ILD_specific_mask", "shared_mask"}
        if required.issubset(loaded):
            ra, ild, shared = (loaded[x] for x in ("RA_specific_mask", "ILD_specific_mask", "shared_mask"))
            disjoint = not (np.any(ra & ild) or np.any(ra & shared) or np.any(ild & shared))
            add(results, "Specific/shared masks disjoint", disjoint, "RA-specific, ILD-specific, shared")


def metric_map(frame: pd.DataFrame) -> dict[str, float]:
    aliases = {"sensitivity_recall": "sensitivity", "recall": "sensitivity", "brier_score": "brier"}
    if {"metric", "value"}.issubset(frame.columns):
        return {aliases.get(str(k).lower(), str(k).lower()): float(v) for k, v in zip(frame["metric"], frame["value"])}
    if len(frame) == 1:
        out = {}
        for column in frame.columns:
            try:
                out[aliases.get(column.lower(), column.lower())] = float(frame.iloc[0][column])
            except (TypeError, ValueError):
                pass
        return out
    return {}


def check_step08(results: list[Result], config, tolerance: float) -> None:
    output = config.path("independent_validation.output_dir")
    complete = output / "08_VALIDATION_COMPLETE.json"
    predictions = output / "08_independent_test_predictions.csv"
    metrics = output / "08_independent_test_metrics.csv"
    add(results, "Step 08 completion marker exists", complete.is_file(), str(complete))
    csv_rows(results, predictions, int(config.raw["data"]["test"]["expected_samples"]), "Step 08 predictions", "sample_id")
    if not metrics.is_file():
        add(results, "Step 08 metrics exists", False, str(metrics))
        return
    observed = metric_map(pd.read_csv(metrics))
    if not observed:
        add(results, "Step 08 metric table structure", False, str(metrics))
        return
    for name, expected in config.raw["independent_validation"]["expected_metrics"].items():
        value = observed.get(name.lower())
        add(results, f"Step 08 metric {name}", value is not None and close(value, expected, tolerance), f"observed={value}, expected={expected}")


def main() -> int:
    args = parse_args()
    config = load_experiment_config(Path(args.config), repository_root=Path(args.repository_root) if args.repository_root else None)
    results: list[Result] = []
    train_n = int(config.raw["data"]["train"]["expected_samples"])
    test_n = int(config.raw["data"]["test"]["expected_samples"])

    csv_rows(results, config.path("data.train.metadata"), train_n, "Train metadata", "libraryid")
    csv_rows(results, config.path("data.test.metadata"), test_n, "Test metadata", "libraryid")
    csv_rows(results, config.path("data.train.base_matrix"), train_n, "Train base matrix", "sample_id")
    csv_rows(results, config.path("data.test.final_matrix"), test_n, "Test final matrix", "sample_id")

    repeats = int(config.raw["cross_validation"]["outer_repeats"])
    folds = int(config.raw["cross_validation"]["outer_folds"])
    outer = csv_rows(results, config.path("cross_validation.outer_assignments"), train_n * repeats, "Outer CV assignments")
    if outer is not None:
        required = {"outer_repeat", "outer_fold", "sample_id"}
        missing = sorted(required - set(outer.columns))
        add(results, "Outer CV required columns", not missing, f"missing={missing}")
        if not missing:
            duplicates = int(outer.duplicated(["outer_repeat", "sample_id"]).sum())
            add(results, "Outer CV sample/repeat uniqueness", duplicates == 0, f"duplicates={duplicates}")
            observed_repeats = set(pd.to_numeric(outer["outer_repeat"], errors="coerce"))
            observed_folds = set(pd.to_numeric(outer["outer_fold"], errors="coerce"))
            add(results, "Outer CV repeat coverage", observed_repeats == set(range(1, repeats + 1)), f"observed={sorted(observed_repeats)}")
            add(results, "Outer CV fold coverage", observed_folds == set(range(1, folds + 1)), f"observed={sorted(observed_folds)}")
    inner = config.path("cross_validation.inner_assignments")
    add(results, "Inner CV assignments exist", inner.is_file(), str(inner))

    if args.scope in {"training", "full"}:
        check_step07(results, config, args.tolerance)
    if args.scope == "full":
        check_step08(results, config, args.tolerance)

    failed = [item for item in results if item.status == "FAIL"]
    print(f"Experiment: {config.experiment_id}")
    print(f"Checks={len(results)} PASS={len(results)-len(failed)} FAIL={len(failed)}")
    for item in results:
        print(f"[{item.status}] {item.name}: {item.detail}")

    report = {
        "experiment_id": config.experiment_id,
        "scope": args.scope,
        "status": "PASS" if not failed else "FAIL",
        "checks": [asdict(item) for item in results],
    }
    if args.json_output:
        path = Path(args.json_output).expanduser()
        if not path.is_absolute():
            path = config.repository_root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"JSON report: {path.resolve()}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
