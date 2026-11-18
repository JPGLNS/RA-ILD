#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only audit for all 15 TRB V3 179 five-algorithm resolved schemes."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ra_ild_trb.model_contract import (  # noqa: E402
    ModelContractError,
    validate_hierarchical_model_contract,
)

EXPECTED = {
    "total": {"samples": 179, "models": 4},
    "pbmc": {"samples": 116, "models": 3},
    "buffycoat": {"samples": 63, "models": 3},
}
ALGORITHMS = ("elastic_net", "ridge", "lasso", "linear_svm", "xgboost")


def load_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be a mapping: {path}")
    return dict(value)


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _stable_model_signature(models: Mapping[str, Any]) -> str:
    # Model definitions are copied exactly between algorithms; key order and
    # predictor order are scientifically meaningful and intentionally retained.
    return json.dumps(models, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def audit(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 15:
        raise ValueError(f"Expected 15 manifest entries; observed {len(entries) if isinstance(entries, list) else 'invalid'}")

    seen = set()
    cohort_configs: dict[str, list[tuple[str, dict[str, Any], Path]]] = defaultdict(list)
    rows = []
    for entry in entries:
        cohort = str(entry["cohort"]).lower()
        algorithm = str(entry["algorithm"]).lower()
        key = (cohort, algorithm)
        if key in seen:
            raise ValueError(f"Duplicate cohort/algorithm entry: {key}")
        seen.add(key)
        if cohort not in EXPECTED or algorithm not in ALGORITHMS:
            raise ValueError(f"Unexpected cohort/algorithm: {key}")
        if int(entry.get("expected_samples", -1)) != EXPECTED[cohort]["samples"]:
            raise ValueError(f"{key}: manifest sample count mismatch")
        config_path = resolve(root, str(entry["config"]))
        config = load_mapping(config_path, f"{key} resolved config")
        cv = config["cross_validation"]
        if int(config["data"]["train"]["expected_samples"]) != EXPECTED[cohort]["samples"]:
            raise ValueError(f"{key}: resolved sample count mismatch")
        if int(cv["outer_repeats"]) != 100:
            raise ValueError(f"{key}: outer_repeats must be 100")
        if tuple(int(x) for x in cv.get("executed_outer_folds", [1])) != (1,):
            raise ValueError(f"{key}: executed_outer_folds must be [1]")
        if tuple(config["models"]) != tuple(entry["expected_models"]):
            raise ValueError(f"{key}: manifest/resolved model names differ")
        counts = validate_hierarchical_model_contract(
            config["models"],
            cohort=cohort,
            repeat_3mer_options=config.get("repeat_3mer_features"),
            expected_resolved_columns=config.get("model_selection", {}).get("expected_resolved_columns"),
        )
        if len(config["models"]) != EXPECTED[cohort]["models"]:
            raise ValueError(f"{key}: wrong model count")
        cohort_configs[cohort].append((algorithm, config, config_path))
        rows.append({
            "cohort": cohort,
            "algorithm": algorithm,
            "samples": EXPECTED[cohort]["samples"],
            "models": len(config["models"]),
            "outer_repeats": int(cv["outer_repeats"]),
            "executed_outer_folds": list(cv.get("executed_outer_folds", [1])),
            "predictor_counts": counts,
            "config": str(config_path),
        })

    expected_pairs = {(c, a) for c in EXPECTED for a in ALGORITHMS}
    if seen != expected_pairs:
        raise ValueError(f"Manifest cohort/algorithm coverage mismatch: missing={sorted(expected_pairs-seen)} extra={sorted(seen-expected_pairs)}")

    cross_algorithm_checks = {}
    for cohort, group in cohort_configs.items():
        signatures = {_stable_model_signature(config["models"]) for _, config, _ in group}
        if len(signatures) != 1:
            raise ValueError(f"{cohort}: model definitions differ between algorithms")
        base = group[0][1]
        base_cv = base["cross_validation"]
        base_repeat = base["repeat_3mer_features"]
        base_inputs = base.get("repeated_holdout_training", {})
        for algorithm, config, _ in group[1:]:
            if config["repeat_3mer_features"] != base_repeat:
                raise ValueError(f"{cohort}/{algorithm}: repeat-3mer config differs")
            if config["cross_validation"] != base_cv:
                raise ValueError(f"{cohort}/{algorithm}: frozen CV config differs")
            if config.get("repeated_holdout_training", {}) != base_inputs:
                raise ValueError(f"{cohort}/{algorithm}: frozen bundle references differ")
            for section in ("data", "public_reference"):
                if config.get(section) != base.get(section):
                    raise ValueError(f"{cohort}/{algorithm}: {section} differs from cohort baseline")
        cross_algorithm_checks[cohort] = {
            "algorithms": [algorithm for algorithm, _, _ in group],
            "model_definitions_identical": True,
            "repeat_3mer_identical": True,
            "cv_identical": True,
            "training_bundle_references_identical": True,
            "data_and_public_reference_identical": True,
        }

    return {
        "status": "PASS",
        "manifest": str(manifest_path),
        "entry_count": len(entries),
        "expected_entry_count": 15,
        "cohorts": EXPECTED,
        "cross_algorithm_checks": cross_algorithm_checks,
        "entries": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument(
        "--manifest",
        default="TRB/set/configs/schemes/generated/hierarchical_five_algorithm/FIVE_ALGORITHM_RUN_MANIFEST.json",
    )
    parser.add_argument("--output", default=None, help="Optional JSON audit report path")
    args = parser.parse_args()
    root = Path(args.repository_root).expanduser().resolve()
    manifest_path = resolve(root, args.manifest)
    try:
        result = audit(root, manifest_path)
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        print("TRB V3 179 FIVE-ALGORITHM MODEL CONTRACT AUDIT: PASS")
        print(f"Manifest entries: {result['entry_count']}")
        for cohort, checks in result["cross_algorithm_checks"].items():
            print(f"{cohort}: five algorithms share identical model/data/split/3-mer contracts")
        if args.output:
            output = resolve(root, args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite audit report: {output}")
            output.write_text(rendered + "\n", encoding="utf-8")
            print(f"Audit report: {output}")
        return 0
    except Exception as exc:
        print(f"TRB V3 179 MODEL CONTRACT AUDIT: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
