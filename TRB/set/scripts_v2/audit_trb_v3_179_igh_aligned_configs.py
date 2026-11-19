#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the 15 prepared IGH-aligned TRB V3 179 configurations before Repeat-001."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.public_reference import PUBLIC_FEATURE_SETS  # noqa: E402
from ra_ild_trb.repeat_3mer_features import split_legacy_static_features  # noqa: E402
from ra_ild_trb.specifications import select_static_tcr_features  # noqa: E402

MANIFEST = Path(
    "TRB/set/configs/schemes/generated/hierarchical_igh_aligned_five_algorithm/"
    "FIVE_ALGORITHM_RUN_MANIFEST.json"
)
GLOBAL_ROOT = Path(
    "TRB/set/experiments/trb_v3_179_igh_aligned_five_algorithm_global"
)
COHORTS = {
    "total": {
        "base_experiment": "trb_v3_179_igh_aligned_total_m0_m3_repeat100_top10000_v1",
        "samples": 179,
        "models": ("M0", "M1", "M2", "M3"),
        "train_holdout": (125, 54),
    },
    "pbmc": {
        "base_experiment": "trb_v3_179_igh_aligned_pbmc_m0_m2_repeat100_top10000_v1",
        "samples": 116,
        "models": ("M0", "M1", "M2"),
        "train_holdout": (81, 35),
    },
    "buffycoat": {
        "base_experiment": "trb_v3_179_igh_aligned_buffycoat_m0_m2_repeat100_top10000_v1",
        "samples": 63,
        "models": ("M0", "M1", "M2"),
        "train_holdout": (44, 19),
    },
}
ALGORITHMS = ("elastic_net", "ridge", "lasso", "linear_svm", "xgboost")
PUBLIC = tuple(PUBLIC_FEATURE_SETS["raw_bilateral"])
EXPECTED_3MER = {
    "enabled": True,
    "selection_scope": "inner_fold_train_only",
    "ranking_rule": "prevalence_then_total_variance",
    "sequence_column": "cdr3_aa",
    "min_sample_count": 5,
    "min_prevalence": 0.05,
    "min_unweighted_variance": 1.0e-12,
    "min_weighted_variance": 1.0e-12,
    "max_kmers": 500,
    "top_k_values": [50, 100, 200, 500],
    "expected_core_feature_count": 83,
    "core_group_name": "core83",
    "legacy_group_name": "static_tcr_candidate_predictors",
}
RUNNERS = {
    "elastic_net": "TRB/set/scripts_v2/run_nested_cv_task.py",
    "ridge": "TRB/set/scripts_v2/run_nested_cv_task.py",
    "lasso": "TRB/set/scripts_v2/run_nested_cv_task.py",
    "linear_svm": "TRB/set/scripts_v2/run_linear_svm_nested_cv_task.py",
    "xgboost": "TRB/set/scripts_v2/run_xgboost_nested_cv_task.py",
}


class AuditError(RuntimeError):
    pass


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise AuditError(f"YAML root must be a mapping: {path}")
    return dict(value)


def _resolve(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AuditError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _tuple(mapping: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key, [])
    if not isinstance(value, list):
        raise AuditError(f"{key} must be a list")
    return tuple(str(item) for item in value)


def _expected_model_definition(cohort: str, model: str) -> dict[str, tuple[str, ...]]:
    if model not in COHORTS[cohort]["models"]:
        raise AuditError(f"Unexpected model {cohort}/{model}")
    return {
        "numeric": ("age",),
        "categorical": ("sex", "material") if model == "M3" else ("sex",),
        "static_feature_groups": () if model == "M0" else (
            "static_tcr_candidate_predictors",
        ),
        "dynamic_public": PUBLIC if model in {"M2", "M3"} else (),
    }


def _experiment_id(cohort: str, algorithm: str) -> str:
    base = COHORTS[cohort]["base_experiment"]
    suffix = {
        "elastic_net": "",
        "ridge": "_ridge_v1",
        "lasso": "_lasso_v1",
        "linear_svm": "_linear_svm_v1",
        "xgboost": "_xgboost_v1",
    }[algorithm]
    return base + suffix


def _expected_outer_output_root(algorithm: str, experiment_id: str) -> str:
    """Return the output layout used by each algorithm's actual scheme preparer.

    Elastic Net/Ridge/Lasso use scheme_management._rewrite_output_paths(),
    which writes task outputs under 02_tasks/outer_nested_cv. Linear SVM and
    XGBoost use their dedicated preparers, which write under 01_outer_tasks.
    """
    base = f"TRB/set/experiments/{experiment_id}"
    if algorithm in {"elastic_net", "ridge", "lasso"}:
        return f"{base}/02_tasks/outer_nested_cv"
    if algorithm in {"linear_svm", "xgboost"}:
        return f"{base}/01_outer_tasks"
    raise AuditError(f"Unsupported algorithm for output layout: {algorithm}")


def _input_signature(raw: Mapping[str, Any]) -> tuple[str, ...]:
    data = raw["data"]["train"]
    cv = raw["cross_validation"]
    public_cache = raw["public_reference"]["cache"]
    repeat = raw["repeat_3mer_features"]
    keys = (
        str(data["metadata"]),
        str(data["feature_manifest"]),
        str(data["base_matrix"]),
        str(data["aa_clone_table_dir"]),
        str(cv["outer_assignments"]),
        str(cv["inner_assignments"]),
        str(public_cache["presence"]),
        str(public_cache["frequency"]),
        str(public_cache["metadata"]),
        str(repeat["aa_manifest"]),
    )
    return keys


def _check_aa_manifest(path: Path, expected_samples: int) -> tuple[str, str]:
    frame = pd.read_csv(path)
    required = {"sample_id", "path", "frequency_column"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AuditError(f"{path}: missing AA manifest columns {missing}")
    frame = frame.loc[:, ["sample_id", "path", "frequency_column"]].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise AuditError(f"{path}: duplicate sample IDs in AA manifest")
    if len(frame) != expected_samples:
        raise AuditError(
            f"{path}: expected {expected_samples} AA rows, observed {len(frame)}"
        )
    missing_paths = []
    for value in frame["path"]:
        item = Path(str(value)).expanduser()
        if not item.is_absolute():
            item = (path.parent / item).resolve()
        if not item.is_file():
            missing_paths.append(str(item))
    if missing_paths:
        raise AuditError(
            f"{path}: {len(missing_paths)} clone tables are missing; "
            f"examples={missing_paths[:5]}"
        )
    if frame["frequency_column"].isna().any() or (
        frame["frequency_column"].astype(str).str.strip() == ""
    ).any():
        raise AuditError(f"{path}: empty frequency_column values")
    digest = hashlib.sha256(
        "\n".join(frame["sample_id"].tolist()).encode("utf-8")
    ).hexdigest()
    return digest, str(path.resolve())


def _candidate_count(root: Path, config_path: Path, algorithm: str, raw: Mapping[str, Any]) -> int:
    engine = raw["model_engine"]
    if algorithm in {"elastic_net", "ridge", "lasso"}:
        return len(engine["alpha_grid"]) * len(engine["lambda_grid"])
    marker_name = (
        "LINEAR_SVM_SCHEME_PREPARED.json"
        if algorithm == "linear_svm"
        else "XGBOOST_SCHEME_PREPARED.json"
    )
    marker = config_path.parent / marker_name
    if not marker.is_file():
        raise FileNotFoundError(f"Prepared algorithm marker missing: {marker}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    return int(payload["candidate_count"])


def audit(root: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    if manifest.get("base_branch") != "feature/trb-v3-179-igh-aligned-hierarchy":
        errors.append(f"Manifest base_branch mismatch: {manifest.get('base_branch')!r}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 15:
        raise AuditError(f"Expected 15 manifest entries, observed {len(entries) if isinstance(entries, list) else 'invalid'}")

    expected_pairs = {(c, a) for c in COHORTS for a in ALGORITHMS}
    observed_pairs = [(str(x.get("cohort")), str(x.get("algorithm"))) for x in entries]
    if len(set(observed_pairs)) != 15 or set(observed_pairs) != expected_pairs:
        errors.append(f"Manifest cohort/algorithm matrix mismatch: {observed_pairs}")

    cohort_signatures: dict[str, tuple[str, ...]] = {}
    cohort_static: dict[str, tuple[str, ...]] = {}
    cohort_aa_ids: dict[str, str] = {}
    for entry in entries:
        cohort = str(entry.get("cohort"))
        algorithm = str(entry.get("algorithm"))
        row: dict[str, Any] = {
            "cohort": cohort,
            "algorithm": algorithm,
            "config": str(entry.get("config", "")),
            "status": "PASS",
            "checks": [],
            "errors": [],
        }
        try:
            if cohort not in COHORTS or algorithm not in ALGORITHMS:
                raise AuditError(f"Unknown cohort/algorithm: {cohort}/{algorithm}")
            spec = COHORTS[cohort]
            config_path = _resolve(root, entry["config"], "manifest config")
            raw = _read_yaml(config_path)
            experiment_id = _experiment_id(cohort, algorithm)
            if str(raw["experiment"]["id"]) != experiment_id:
                raise AuditError(
                    f"experiment.id mismatch: {raw['experiment']['id']!r} != {experiment_id!r}"
                )
            if int(raw["data"]["train"]["expected_samples"]) != spec["samples"]:
                raise AuditError("training sample count mismatch")
            if int(entry["expected_samples"]) != spec["samples"]:
                raise AuditError("manifest expected_samples mismatch")
            expected_models = tuple(spec["models"])
            if tuple(raw["models"].keys()) != expected_models:
                raise AuditError(
                    f"model names/order mismatch: {tuple(raw['models'].keys())}"
                )
            if tuple(entry["expected_models"]) != expected_models:
                raise AuditError("manifest model contract mismatch")
            cv = raw["cross_validation"]
            if int(cv["outer_repeats"]) != 100:
                raise AuditError("outer_repeats must equal 100")
            if tuple(int(x) for x in cv.get("executed_outer_folds", [1])) != (1,):
                raise AuditError("executed_outer_folds must equal [1]")
            if int(cv["inner_folds"]) != 5:
                raise AuditError("inner_folds must equal 5")
            if int(raw["outer_tasks"]["expected_tasks"]) != 100:
                raise AuditError("outer_tasks.expected_tasks must equal 100")
            if str(raw["outer_tasks"]["runner_script"]) != RUNNERS[algorithm]:
                raise AuditError(
                    f"runner mismatch: {raw['outer_tasks']['runner_script']!r}"
                )
            observed_output_root = str(raw["outer_tasks"]["output_root"]).rstrip("/")
            expected_output_root = _expected_outer_output_root(algorithm, experiment_id)
            if observed_output_root != expected_output_root:
                raise AuditError(
                    "outer task output_root mismatch: "
                    f"observed={observed_output_root!r}, expected={expected_output_root!r}"
                )

            repeat = raw.get("repeat_3mer_features")
            if not isinstance(repeat, Mapping):
                raise AuditError("repeat_3mer_features section missing")
            for key, expected in EXPECTED_3MER.items():
                observed = repeat.get(key)
                if isinstance(expected, float):
                    if abs(float(observed) - expected) > 1e-15:
                        raise AuditError(f"3-mer option {key} mismatch: {observed!r}")
                elif observed != expected:
                    raise AuditError(f"3-mer option {key} mismatch: {observed!r}")
            aa_manifest = _resolve(root, repeat["aa_manifest"], "aa_manifest")
            aa_hash, aa_path = _check_aa_manifest(aa_manifest, spec["samples"])
            previous_hash = cohort_aa_ids.setdefault(cohort, aa_hash)
            if previous_hash != aa_hash:
                raise AuditError("AA manifest sample-ID order differs across algorithms")

            feature_manifest = _resolve(
                root, raw["data"]["train"]["feature_manifest"], "feature_manifest"
            )
            frame = pd.read_csv(feature_manifest)
            static = tuple(select_static_tcr_features(frame))
            core, unweighted, weighted = split_legacy_static_features(
                static, expected_core_count=83
            )
            if len(static) != 1083 or len(core) != 83 or len(unweighted) != 500 or len(weighted) != 500:
                raise AuditError(
                    "static candidate contract mismatch: "
                    f"total={len(static)}, core={len(core)}, "
                    f"unweighted={len(unweighted)}, weighted={len(weighted)}"
                )
            static_hash = hashlib.sha256("\n".join(static).encode("utf-8")).hexdigest()
            previous_static = cohort_static.setdefault(cohort, static)
            if previous_static != static:
                raise AuditError("static predictor order differs across algorithms")

            public = raw["public_reference"]
            if tuple(public["dynamic_features"]) != PUBLIC:
                raise AuditError(
                    f"public_reference.dynamic_features must be raw_bilateral 8: "
                    f"{tuple(public['dynamic_features'])}"
                )
            for model_name in expected_models:
                definition = raw["models"][model_name]
                expected_definition = _expected_model_definition(cohort, model_name)
                for key, expected in expected_definition.items():
                    observed = _tuple(definition, key)
                    if observed != expected:
                        raise AuditError(
                            f"{model_name}.{key} mismatch: observed={observed}, "
                            f"expected={expected}"
                        )
            resolved_counts = {}
            for model_name in expected_models:
                definition = _expected_model_definition(cohort, model_name)
                numeric_count = (
                    len(definition["numeric"])
                    + (len(static) if definition["static_feature_groups"] else 0)
                    + len(definition["dynamic_public"])
                )
                categorical_count = len(definition["categorical"])
                resolved_counts[model_name] = (numeric_count, categorical_count)
            expected_counts = {
                "M0": (1, 1),
                "M1": (1084, 1),
                "M2": (1092, 1),
                "M3": (1092, 2),
            }
            for model_name, counts in resolved_counts.items():
                if counts != expected_counts[model_name]:
                    raise AuditError(
                        f"{model_name} resolved predictor count mismatch: {counts}"
                    )

            signature = _input_signature(raw)
            old_signature = cohort_signatures.setdefault(cohort, signature)
            if old_signature != signature:
                raise AuditError("input/split/public-cache paths differ across algorithms")

            candidate_count = _candidate_count(root, config_path, algorithm, raw)
            if candidate_count != int(entry["candidate_count_per_model"]):
                raise AuditError(
                    f"candidate count mismatch: config={candidate_count}, "
                    f"manifest={entry['candidate_count_per_model']}"
                )
            if int(entry["outer_tasks"]) != 100 or int(entry["outer_repeats"]) != 100:
                raise AuditError("manifest outer task/repeat count mismatch")

            row.update({
                "experiment_id": experiment_id,
                "samples": spec["samples"],
                "models": ",".join(expected_models),
                "static_features": len(static),
                "core83": len(core),
                "unweighted_3mer": len(unweighted),
                "weighted_3mer": len(weighted),
                "public_features": len(PUBLIC),
                "resolved_predictor_counts": json.dumps(
                    {k: list(v) for k, v in resolved_counts.items()},
                    sort_keys=True,
                ),
                "candidate_count_per_model": candidate_count,
                "aa_manifest": aa_path,
                "aa_sample_id_sha256": aa_hash,
                "static_feature_sha256": static_hash,
                "status": "PASS",
                "checks": [
                    "sample/model contract",
                    "100-repeat/5-inner-fold contract",
                    "runner/output path",
                    "inner-fold-only 3-mer settings",
                    "AA manifest coverage/files",
                    "Core83 + 500 weighted + 500 unweighted",
                    "eight raw_bilateral Public features",
                    "M0-M3 predictor definitions/counts",
                    "same cohort inputs across algorithms",
                    "candidate count",
                ],
            })
        except Exception as exc:
            row["status"] = "FAIL"
            row["errors"] = [str(exc)]
            errors.append(f"{cohort}/{algorithm}: {exc}")
        rows.append(row)

    return rows, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--output-dir", default=str(GLOBAL_ROOT / "00_config_audit"))
    args = parser.parse_args()
    root = Path(args.repository_root).expanduser().resolve()
    manifest_path = _resolve(root, args.manifest, "manifest")
    output_dir = _resolve(root, args.output_dir, "output_dir")
    try:
        rows, errors = audit(root, manifest_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "08_FIVE_ALGORITHM_CONFIG_AUDIT.csv"
        json_path = output_dir / "08_FIVE_ALGORITHM_CONFIG_AUDIT.json"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        payload = {
            "status": "PASS" if not errors else "FAIL",
            "manifest": str(manifest_path),
            "branch": "feature/trb-v3-179-igh-aligned-hierarchy",
            "configuration_count": len(rows),
            "passed": sum(row["status"] == "PASS" for row in rows),
            "failed": sum(row["status"] == "FAIL" for row in rows),
            "errors": errors,
            "rows": rows,
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("TRB V3 179 IGH-aligned configuration audit")
        print(f"Configurations: {payload['passed']}/{payload['configuration_count']} passed")
        print(f"Errors: {len(errors)}")
        print(f"CSV: {csv_path}")
        print(f"JSON: {json_path}")
        if errors:
            for error in errors:
                print(f"[FAIL] {error}")
            print("CONFIG AUDIT: FAIL")
            return 1
        print("CONFIG AUDIT: PASS")
        return 0
    except Exception as exc:
        print(f"CONFIG AUDIT: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
