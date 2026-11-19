#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print mean ROC-AUC across 100 repeated outer holdouts for TRB V3.

This is a read-only terminal summary. It does not create figures, CSV files,
or other aggregation artifacts, and it never reruns model training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ALGORITHMS = ("elastic_net", "ridge", "lasso", "linear_svm", "xgboost")
ALGORITHM_LABELS = {
    "elastic_net": "Elastic Net",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "linear_svm": "Linear SVM",
    "xgboost": "XGBoost",
}
COHORT_ORDER = ("total", "pbmc", "buffycoat")
COHORT_LABELS = {
    "total": "TOTAL",
    "pbmc": "PBMC",
    "buffycoat": "BUFFYCOAT",
}
MARKER_FILE = "06_TASK_COMPLETE.json"
METRIC_FILE = "06_outer_validation_metrics.csv"
PREDICTION_FILE = "06_outer_validation_predictions.csv"


class SummaryError(RuntimeError):
    """Raised when completed task outputs fail aggregation checks."""


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise SummaryError(f"Configuration root is not a mapping: {path}")
    return dict(value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise SummaryError(f"JSON root is not a mapping: {path}")
    return dict(value)


def validate_repeat_holdout_config(
    cv: Mapping[str, Any], *, cohort: str, algorithm: str, expected_repeats: int
) -> None:
    """Validate the scheduler's executed-fold contract, not outer_folds."""
    if int(cv.get("outer_repeats", -1)) != expected_repeats:
        raise SummaryError(
            f"{cohort}/{algorithm}: configured outer_repeats="
            f"{cv.get('outer_repeats')}, expected={expected_repeats}"
        )
    # The original nested-CV assignment scheme may define multiple possible
    # outer folds. This experiment explicitly executes only fold 1 per repeat.
    executed_folds = tuple(int(x) for x in cv.get("executed_outer_folds", [1]))
    if executed_folds != (1,):
        raise SummaryError(
            f"{cohort}/{algorithm}: expected executed_outer_folds=[1], "
            f"observed={list(executed_folds)}"
        )


def _score_column(frame: pd.DataFrame, path: Path) -> str:
    # Linear SVM uses an uncalibrated decision_function score. The other
    # algorithms use probabilities. Both are valid continuous ROC scores.
    for column in ("prediction_score", "probability_ILD", "decision_score_ILD"):
        if column in frame.columns:
            return column
    raise SummaryError(
        f"No continuous prediction score found in {path}: {list(frame.columns)}"
    )


def summarize_model_metrics(
    metrics: pd.DataFrame,
    *,
    cohort: str,
    algorithm: str,
    expected_models: Sequence[str],
    expected_repeats: int,
) -> pd.DataFrame:
    required = {"model", "outer_repeat", "outer_fold", "roc_auc"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise SummaryError(f"{cohort}/{algorithm}: metrics missing columns {missing}")

    frame = metrics.copy()
    frame["model"] = frame["model"].astype(str)
    frame["outer_repeat"] = pd.to_numeric(
        frame["outer_repeat"], errors="raise"
    ).astype(int)
    frame["outer_fold"] = pd.to_numeric(
        frame["outer_fold"], errors="raise"
    ).astype(int)
    frame["roc_auc"] = pd.to_numeric(frame["roc_auc"], errors="raise")

    if frame.duplicated(["model", "outer_repeat", "outer_fold"]).any():
        raise SummaryError(f"{cohort}/{algorithm}: duplicate model/repeat/fold metric rows")
    observed_models = set(frame["model"])
    if observed_models != set(expected_models):
        raise SummaryError(
            f"{cohort}/{algorithm}: model set mismatch; "
            f"observed={sorted(observed_models)}, expected={list(expected_models)}"
        )
    if set(frame["outer_fold"]) != {1}:
        raise SummaryError(
            f"{cohort}/{algorithm}: observed outer folds={sorted(frame['outer_fold'].unique())}, "
            "expected only fold 1"
        )
    if not np.isfinite(frame["roc_auc"].to_numpy(float)).all():
        raise SummaryError(f"{cohort}/{algorithm}: ROC-AUC contains non-finite values")
    if ((frame["roc_auc"] < 0) | (frame["roc_auc"] > 1)).any():
        raise SummaryError(f"{cohort}/{algorithm}: ROC-AUC values outside [0, 1]")

    rows: list[dict[str, Any]] = []
    for model in expected_models:
        subset = frame.loc[frame["model"] == model].sort_values("outer_repeat")
        repeats = subset["outer_repeat"].tolist()
        expected_sequence = list(range(1, expected_repeats + 1))
        if repeats != expected_sequence:
            raise SummaryError(
                f"{cohort}/{algorithm}/{model}: repeat coverage mismatch; "
                f"observed {len(repeats)} rows, expected repeats 1..{expected_repeats}"
            )
        values = subset["roc_auc"].to_numpy(float)
        rows.append({
            "cohort": cohort,
            "algorithm": algorithm,
            "model": model,
            "n_repeats": len(values),
            "roc_auc_mean": float(np.mean(values)),
            "roc_auc_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "roc_auc_median": float(np.median(values)),
            "roc_auc_min": float(np.min(values)),
            "roc_auc_max": float(np.max(values)),
        })
    return pd.DataFrame(rows)


def _load_entry(
    root: Path,
    entry: Mapping[str, Any],
    *,
    expected_repeats: int,
    split_signatures: dict[tuple[str, int], tuple[tuple[str, int], ...]],
    validation_counts: dict[str, int],
) -> pd.DataFrame:
    cohort = str(entry["cohort"])
    algorithm = str(entry["algorithm"])
    if algorithm not in ALGORITHMS:
        raise SummaryError(f"Unsupported algorithm in manifest: {algorithm}")

    config_path = _resolve(root, str(entry["config"]))
    if not config_path.is_file():
        raise SummaryError(f"Resolved config not found: {config_path}")
    config = _read_yaml(config_path)
    models = tuple(map(str, config["models"].keys()))
    manifest_models = tuple(map(str, entry.get("expected_models", models)))
    if models != manifest_models:
        raise SummaryError(
            f"{cohort}/{algorithm}: manifest/config model mismatch "
            f"{manifest_models} != {models}"
        )
    expected_samples = int(config["data"]["train"]["expected_samples"])
    if "expected_samples" in entry and int(entry["expected_samples"]) != expected_samples:
        raise SummaryError(f"{cohort}/{algorithm}: expected sample count mismatch")

    validate_repeat_holdout_config(
        config["cross_validation"],
        cohort=cohort,
        algorithm=algorithm,
        expected_repeats=expected_repeats,
    )
    output_root = _resolve(root, str(config["outer_tasks"]["output_root"]))
    metric_parts: list[pd.DataFrame] = []

    for repeat in range(1, expected_repeats + 1):
        task_dir = output_root / f"repeat_{repeat:02d}_fold_01"
        marker_path = task_dir / MARKER_FILE
        metric_path = task_dir / METRIC_FILE
        prediction_path = task_dir / PREDICTION_FILE
        missing = [
            path.name for path in (marker_path, metric_path, prediction_path)
            if not path.is_file()
        ]
        if missing:
            raise SummaryError(
                f"{cohort}/{algorithm}/repeat_{repeat:03d}: missing {missing}"
            )

        marker = _read_json(marker_path)
        if str(marker.get("status", "")).upper() != "COMPLETE":
            raise SummaryError(f"Task marker is not COMPLETE: {marker_path}")
        if "outer_repeat" in marker and int(marker["outer_repeat"]) != repeat:
            raise SummaryError(f"Repeat identity mismatch: {marker_path}")
        if "outer_fold" in marker and int(marker["outer_fold"]) != 1:
            raise SummaryError(f"Fold identity mismatch: {marker_path}")

        metrics = pd.read_csv(
            metric_path,
            float_precision="round_trip",
        )
        if "outer_repeat" not in metrics.columns or "outer_fold" not in metrics.columns:
            raise SummaryError(f"Metrics lack repeat/fold identity: {metric_path}")
        if set(pd.to_numeric(metrics["outer_repeat"]).astype(int)) != {repeat}:
            raise SummaryError(f"Metric repeat identity mismatch: {metric_path}")
        if set(pd.to_numeric(metrics["outer_fold"]).astype(int)) != {1}:
            raise SummaryError(f"Metric fold identity mismatch: {metric_path}")

        predictions = pd.read_csv(
            prediction_path,
            float_precision="round_trip",
        )

        required_predictions = {"model", "sample_id", "true_label"}
        if not required_predictions.issubset(predictions.columns):
            raise SummaryError(
                f"{prediction_path} missing columns: "
                f"{sorted(required_predictions - set(predictions.columns))}"
            )
        score_col = _score_column(predictions, prediction_path)
        if predictions.duplicated(["model", "sample_id"]).any():
            raise SummaryError(f"Duplicate model/sample predictions: {prediction_path}")

        for model in models:
            pred = predictions.loc[
                predictions["model"].astype(str) == model
            ].copy()
            if pred.empty:
                raise SummaryError(
                    f"No predictions for {cohort}/{algorithm}/{model}/repeat_{repeat}"
                )
            pred["sample_id"] = pred["sample_id"].astype(str)
            pred["true_label"] = pd.to_numeric(
                pred["true_label"], errors="raise"
            ).astype(int)
            if set(pred["true_label"].unique()) != {0, 1}:
                raise SummaryError(
                    f"{cohort}/{algorithm}/{model}/{repeat}: validation lacks both classes"
                )
            signature = tuple(sorted(
                zip(pred["sample_id"].tolist(), pred["true_label"].tolist())
            ))
            split_key = (cohort, repeat)
            previous = split_signatures.get(split_key)
            if previous is None:
                split_signatures[split_key] = signature
            elif previous != signature:
                raise SummaryError(
                    f"Holdout sample IDs/labels differ across algorithms/models: "
                    f"{cohort}/repeat_{repeat:03d}"
                )

            n_valid = len(pred)
            if n_valid != pred["sample_id"].nunique():
                raise SummaryError(f"Duplicate validation samples: {prediction_path}")
            if n_valid != expected_samples - int(config["data"]["train"]["expected_samples"]) + n_valid:
                # Kept intentionally inert: expected_samples is the cohort size,
                # while the actual holdout size is validated across algorithms below.
                raise SummaryError("Unexpected validation sample count arithmetic")

            previous_n = validation_counts.get(cohort)
            if previous_n is None:
                validation_counts[cohort] = n_valid
            elif previous_n != n_valid:
                raise SummaryError(
                    f"{cohort}: validation sample count varies across repeats "
                    f"({previous_n} vs {n_valid})"
                )

            score = pd.to_numeric(pred[score_col], errors="raise").to_numpy(float)
            if not np.isfinite(score).all():
                raise SummaryError(f"Non-finite prediction scores: {prediction_path}")
            observed_auc = float(roc_auc_score(
                pred["true_label"].to_numpy(int), score
            ))
            metric_rows = metrics.loc[
                metrics["model"].astype(str) == model, "roc_auc"
            ]
            if len(metric_rows) != 1:
                raise SummaryError(
                    f"{cohort}/{algorithm}/{model}/{repeat}: expected one ROC metric row"
                )
            stored_auc = float(metric_rows.iloc[0])
            if not np.isclose(observed_auc, stored_auc, atol=1e-6, rtol=0):
                raise SummaryError(
                    f"ROC-AUC mismatch from predictions: {cohort}/{algorithm}/{model}/"
                    f"{repeat}: stored={stored_auc}, recomputed={observed_auc}"
                )

        metric_parts.append(metrics)

    all_metrics = pd.concat(metric_parts, ignore_index=True)
    return summarize_model_metrics(
        all_metrics,
        cohort=cohort,
        algorithm=algorithm,
        expected_models=models,
        expected_repeats=expected_repeats,
    )


def aggregate(
    root: Path,
    manifest_path: Path,
    *,
    expected_repeats: int = 100,
) -> pd.DataFrame:
    manifest = _read_json(manifest_path)
    entries = list(manifest.get("entries", []))
    if len(entries) != 15:
        raise SummaryError(
            f"Expected 15 cohort/algorithm entries, observed {len(entries)}"
        )

    observed_pairs = {
        (str(entry["cohort"]), str(entry["algorithm"])) for entry in entries
    }
    expected_pairs = {
        (cohort, algorithm)
        for cohort in COHORT_ORDER
        for algorithm in ALGORITHMS
    }
    if observed_pairs != expected_pairs:
        raise SummaryError(
            f"Manifest cohort/algorithm matrix mismatch: "
            f"missing={sorted(expected_pairs - observed_pairs)}, "
            f"extra={sorted(observed_pairs - expected_pairs)}"
        )

    split_signatures: dict[tuple[str, int], tuple[tuple[str, int], ...]] = {}
    validation_counts: dict[str, int] = {}
    summary_parts: list[pd.DataFrame] = []

    for cohort in COHORT_ORDER:
        for algorithm in ALGORITHMS:
            entry = next(
                item for item in entries
                if str(item["cohort"]) == cohort
                and str(item["algorithm"]) == algorithm
            )
            result = _load_entry(
                root,
                entry,
                expected_repeats=expected_repeats,
                split_signatures=split_signatures,
                validation_counts=validation_counts,
            )
            summary_parts.append(result)
            print(
                f"[PASS] {COHORT_LABELS[cohort]} / {ALGORITHM_LABELS[algorithm]}: "
                f"{len(result)} models × {expected_repeats} repeats"
            )

    summary = pd.concat(summary_parts, ignore_index=True)
    expected_combinations = sum(
        len(_read_yaml(_resolve(root, str(entry["config"])))["models"])
        for entry in entries
    )
    if len(summary) != expected_combinations:
        raise SummaryError(
            f"Summary row count mismatch: {len(summary)} != {expected_combinations}"
        )
    return summary


def print_summary(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("TRB V3 — 179 SAMPLES — FIVE-ALGORITHM REPEAT100 ROC-AUC SUMMARY")
    print("=" * 88)
    print("Values are mean ROC-AUC ± SD across 100 repeated outer holdouts.")
    print("Linear SVM uses continuous decision scores; other algorithms use probabilities.")

    for cohort in COHORT_ORDER:
        subset = summary.loc[summary["cohort"] == cohort]
        print(f"\n{'-' * 88}")
        print(f"COHORT: {COHORT_LABELS[cohort]}")
        print("-" * 88)
        for algorithm in ALGORITHMS:
            alg = subset.loc[subset["algorithm"] == algorithm]
            if alg.empty:
                continue
            print(f"\n{ALGORITHM_LABELS[algorithm]}")
            for row in alg.itertuples(index=False):
                print(
                    f"  {row.model:<4s} ROC-AUC = "
                    f"{row.roc_auc_mean:.4f} ± {row.roc_auc_sd:.4f} "
                    f"(n={row.n_repeats})"
                )

    print("\n" + "=" * 88)
    print(f"Model combinations summarized: {len(summary)}")
    print("Training outputs modified: NO")
    print("Figures/CSV aggregation files generated: NO")
    print("TRB V3 REPEAT100 ROC-AUC SUMMARY: COMPLETE")
    print("=" * 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print TRB V3 179-sample five-algorithm Repeat100 mean ROC-AUC."
    )
    parser.add_argument("--repository-root", default=None)
    parser.add_argument(
        "--manifest",
        default=(
            "TRB/set/configs/schemes/generated/"
            "hierarchical_igh_aligned_five_algorithm/"
            "FIVE_ALGORITHM_RUN_MANIFEST.json"
        ),
    )
    parser.add_argument("--expected-repeats", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(
        args.repository_root or Path(__file__).resolve().parents[3]
    ).expanduser().resolve()
    manifest_path = _resolve(root, args.manifest)
    try:
        summary = aggregate(
            root,
            manifest_path,
            expected_repeats=args.expected_repeats,
        )
        print_summary(summary)
        return 0
    except Exception as exc:
        print(
            f"TRB V3 ROC-AUC summary: FAIL\n{exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
