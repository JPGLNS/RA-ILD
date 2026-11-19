#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate 100-repeat TRB V3 hierarchical five-algorithm outer validation.

The script reads completed task outputs only. It does not retrain models or
modify task directories. It validates repeat/model coverage and cross-algorithm
holdout alignment, then writes repeat-level metrics, mean/SD summaries, and
cohort-specific mean ROC curves.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

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
ROC_GRID = np.linspace(0.0, 1.0, 201)
METRIC_FILE = "06_outer_validation_metrics.csv"
PREDICTION_FILE = "06_outer_validation_predictions.csv"
MARKER_FILE = "06_TASK_COMPLETE.json"


class SummaryError(RuntimeError):
    """Raised when the completed outer-task results fail aggregation checks."""


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


def _ap_column(frame: pd.DataFrame, path: Path) -> str:
    # SVM names this metric average_precision; the probability models generally
    # name the same average_precision_score quantity pr_auc.
    for column in ("average_precision", "pr_auc", "ap"):
        if column in frame.columns:
            return column
    raise SummaryError(f"No AP/PR-AUC column found in {path}: {list(frame.columns)}")


def _score_column(frame: pd.DataFrame, path: Path) -> str:
    # prediction_score is a probability for LR/XGBoost and a decision score
    # for the uncalibrated Linear SVM. Both are valid continuous ROC scores.
    for column in ("prediction_score", "probability_ILD", "decision_score_ILD"):
        if column in frame.columns:
            return column
    raise SummaryError(f"No continuous prediction score found in {path}: {list(frame.columns)}")


def interpolate_roc_curve(
    y_true: Sequence[int],
    score: Sequence[float],
    grid: np.ndarray = ROC_GRID,
) -> np.ndarray:
    """Interpolate one repeat's ROC curve onto a shared FPR grid."""
    y = np.asarray(y_true, dtype=int)
    values = np.asarray(score, dtype=float)
    if y.ndim != 1 or values.ndim != 1 or len(y) != len(values) or len(y) < 2:
        raise SummaryError("ROC inputs must be one-dimensional and have equal length >= 2")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise SummaryError("Each outer validation split must contain both labels 0 and 1")
    if not np.isfinite(values).all():
        raise SummaryError("Prediction scores contain non-finite values")

    fpr, tpr, _ = roc_curve(y, values)
    unique_fpr = np.unique(fpr)
    # At repeated FPR coordinates retain the upper edge of the vertical ROC step.
    upper_tpr = np.asarray([np.max(tpr[fpr == value]) for value in unique_fpr])
    result = np.interp(grid, unique_fpr, upper_tpr)
    result[0] = 0.0
    result[-1] = 1.0
    return result


def summarize_repeat_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeat-level ROC-AUC and AP without treating repeats as subjects."""
    required = {"cohort", "algorithm", "model", "outer_repeat", "roc_auc", "average_precision"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise SummaryError(f"Repeat-level metrics missing columns: {missing}")

    rows: list[dict[str, Any]] = []
    group_columns = ["cohort", "algorithm", "model"]
    for (cohort, algorithm, model), subset in metrics.groupby(group_columns, sort=False):
        row: dict[str, Any] = {
            "cohort": cohort,
            "cohort_label": COHORT_LABELS.get(str(cohort), str(cohort)),
            "algorithm": algorithm,
            "algorithm_label": ALGORITHM_LABELS.get(str(algorithm), str(algorithm)),
            "model": model,
            "n_repeats": int(subset["outer_repeat"].nunique()),
        }
        for metric in ("roc_auc", "average_precision"):
            values = pd.to_numeric(subset[metric], errors="coerce").dropna().to_numpy(float)
            if len(values) != row["n_repeats"]:
                raise SummaryError(
                    f"{cohort}/{algorithm}/{model}: {metric} has "
                    f"{len(values)} valid values for {row['n_repeats']} repeats"
                )
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            row[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            row[f"{metric}_q75"] = float(np.quantile(values, 0.75))
            row[f"{metric}_q975"] = float(np.quantile(values, 0.975))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)

    result = pd.DataFrame(rows)
    result["roc_auc_rank_within_cohort_model"] = (
        result.groupby(["cohort", "model"])["roc_auc_mean"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return result


def _load_entry(
    root: Path,
    entry: Mapping[str, Any],
    expected_repeats: int,
    split_signatures: dict[tuple[str, int], tuple[tuple[str, int], ...]],
    validation_counts: dict[str, int],
    curve_store: dict[tuple[str, str, str], list[np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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

    cv = config["cross_validation"]
    if int(cv["outer_repeats"]) != expected_repeats:
        raise SummaryError(
            f"{cohort}/{algorithm}: configured outer_repeats="
            f"{cv['outer_repeats']}, expected={expected_repeats}"
        )
    if int(cv["outer_folds"]) != 1:
        raise SummaryError(f"{cohort}/{algorithm}: expected one outer fold per repeat")

    output_root = _resolve(root, str(config["outer_tasks"]["output_root"]))
    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    for repeat in range(1, expected_repeats + 1):
        task_dir = output_root / f"repeat_{repeat:02d}_fold_01"
        marker_path = task_dir / MARKER_FILE
        metric_path = task_dir / METRIC_FILE
        prediction_path = task_dir / PREDICTION_FILE
        missing = [
            str(path.name)
            for path in (marker_path, metric_path, prediction_path)
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

        metrics = pd.read_csv(metric_path)
        predictions = pd.read_csv(prediction_path)
        metric_required = {"model", "roc_auc", "outer_repeat", "outer_fold"}
        prediction_required = {"model", "sample_id", "true_label"}
        if not metric_required.issubset(metrics.columns):
            raise SummaryError(
                f"{metric_path} missing columns: {sorted(metric_required - set(metrics.columns))}"
            )
        if not prediction_required.issubset(predictions.columns):
            raise SummaryError(
                f"{prediction_path} missing columns: "
                f"{sorted(prediction_required - set(predictions.columns))}"
            )
        ap_col = _ap_column(metrics, metric_path)
        score_col = _score_column(predictions, prediction_path)

        if metrics["model"].astype(str).duplicated().any():
            raise SummaryError(f"Duplicate model metric rows: {metric_path}")
        observed_models = tuple(metrics["model"].astype(str).tolist())
        if set(observed_models) != set(models) or len(observed_models) != len(models):
            raise SummaryError(
                f"{cohort}/{algorithm}/repeat_{repeat:03d}: "
                f"metrics model set {observed_models} != {models}"
            )
        if predictions.duplicated(["model", "sample_id"]).any():
            raise SummaryError(f"Duplicate model/sample predictions: {prediction_path}")

        metric_repeat = set(pd.to_numeric(metrics["outer_repeat"]).astype(int))
        metric_fold = set(pd.to_numeric(metrics["outer_fold"]).astype(int))
        if metric_repeat != {repeat} or metric_fold != {1}:
            raise SummaryError(f"Metric repeat/fold identity mismatch: {metric_path}")

        for model in models:
            metric = metrics.loc[metrics["model"].astype(str) == model].iloc[0]
            pred = predictions.loc[predictions["model"].astype(str) == model].copy()
            if pred.empty:
                raise SummaryError(f"No predictions for {cohort}/{algorithm}/{model}/{repeat}")
            pred["sample_id"] = pred["sample_id"].astype(str)
            pred["true_label"] = pd.to_numeric(pred["true_label"], errors="raise").astype(int)
            if set(pred["true_label"].unique()) != {0, 1}:
                raise SummaryError(
                    f"{cohort}/{algorithm}/{model}/{repeat}: validation data lacks one class"
                )
            if len(pred) != pred["sample_id"].nunique():
                raise SummaryError(f"Duplicate sample predictions for {model}: {prediction_path}")

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
            previous_n = validation_counts.get(cohort)
            if previous_n is None:
                validation_counts[cohort] = n_valid
            elif previous_n != n_valid:
                raise SummaryError(
                    f"{cohort}: validation sample count varies across repeats "
                    f"({previous_n} vs {n_valid})"
                )
            if "n_outer_validation" in metric and int(metric["n_outer_validation"]) != n_valid:
                raise SummaryError(
                    f"{cohort}/{algorithm}/{model}/{repeat}: metric/prediction sample count mismatch"
                )

            y = pred["true_label"].to_numpy(int)
            score = pd.to_numeric(pred[score_col], errors="raise").to_numpy(float)
            if not np.isfinite(score).all():
                raise SummaryError(f"Non-finite prediction scores: {prediction_path}")
            observed_roc = float(roc_auc_score(y, score))
            observed_ap = float(average_precision_score(y, score))
            stored_roc = float(metric["roc_auc"])
            stored_ap = float(metric[ap_col])
            if not np.isclose(observed_roc, stored_roc, atol=1e-6, rtol=0):
                raise SummaryError(
                    f"ROC-AUC mismatch from predictions: {cohort}/{algorithm}/{model}/"
                    f"{repeat}: stored={stored_roc}, recomputed={observed_roc}"
                )
            if not np.isclose(observed_ap, stored_ap, atol=1e-6, rtol=0):
                raise SummaryError(
                    f"AP mismatch from predictions: {cohort}/{algorithm}/{model}/"
                    f"{repeat}: stored={stored_ap}, recomputed={observed_ap}"
                )

            row: dict[str, Any] = {
                "cohort": cohort,
                "algorithm": algorithm,
                "model": model,
                "outer_repeat": repeat,
                "outer_fold": 1,
                "roc_auc": stored_roc,
                "average_precision": stored_ap,
                "n_validation_samples": n_valid,
                "score_type": str(pred["score_type"].iloc[0]) if "score_type" in pred.columns else (
                    "decision_function" if algorithm == "linear_svm" else "probability"
                ),
                "source_task_dir": str(task_dir),
            }
            for optional in ("n_outer_train", "n_final_predictors", "selected_threshold"):
                if optional in metric.index:
                    row[optional] = metric[optional]
            metric_rows.append(row)

            interpolated = interpolate_roc_curve(y, score)
            curve_store.setdefault((cohort, algorithm, model), []).append(interpolated)

    return metric_rows, curve_rows


def _build_curve_table(
    curve_store: Mapping[tuple[str, str, str], Sequence[np.ndarray]],
    expected_repeats: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cohort, algorithm, model), curves in curve_store.items():
        if len(curves) != expected_repeats:
            raise SummaryError(
                f"{cohort}/{algorithm}/{model}: found {len(curves)} ROC curves, "
                f"expected {expected_repeats}"
            )
        matrix = np.vstack(curves)
        mean_tpr = matrix.mean(axis=0)
        sd_tpr = matrix.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros(matrix.shape[1])
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        for fpr, mean, sd in zip(ROC_GRID, mean_tpr, sd_tpr):
            rows.append({
                "cohort": cohort,
                "algorithm": algorithm,
                "model": model,
                "false_positive_rate": float(fpr),
                "mean_true_positive_rate": float(mean),
                "sd_true_positive_rate": float(sd),
                "lower_mean_minus_sd": float(max(0.0, mean - sd)),
                "upper_mean_plus_sd": float(min(1.0, mean + sd)),
                "n_repeats": int(len(curves)),
            })
    return pd.DataFrame(rows)


def _plot_cohort_roc(
    cohort: str,
    models: Sequence[str],
    curve_table: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(models)
    ncols = 2 if n_models == 4 else 3
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7.0 * ncols, 5.8 * nrows),
        sharex=True, sharey=True,
        squeeze=False,
    )
    for index, model in enumerate(models):
        ax = axes.flat[index]
        for algorithm in ALGORITHMS:
            subset = curve_table.loc[
                (curve_table["cohort"] == cohort)
                & (curve_table["algorithm"] == algorithm)
                & (curve_table["model"] == model)
            ].sort_values("false_positive_rate")
            if subset.empty:
                continue
            stats = summary.loc[
                (summary["cohort"] == cohort)
                & (summary["algorithm"] == algorithm)
                & (summary["model"] == model)
            ].iloc[0]
            x = subset["false_positive_rate"].to_numpy(float)
            y = subset["mean_true_positive_rate"].to_numpy(float)
            low = subset["lower_mean_minus_sd"].to_numpy(float)
            high = subset["upper_mean_plus_sd"].to_numpy(float)
            auc_mean = float(stats["roc_auc_mean"])
            auc_sd = float(stats["roc_auc_sd"])
            label = f"{ALGORITHM_LABELS[algorithm]} (AUC {auc_mean:.3f} ± {auc_sd:.3f})"
            ax.plot(x, y, label=label, linewidth=1.8)
            ax.fill_between(x, low, high, alpha=0.10)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Chance")
        ax.set_title(model)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    for index in range(n_models, nrows * ncols):
        axes.flat[index].set_visible(False)

    fig.suptitle(
        f"{COHORT_LABELS.get(cohort, cohort)} cohort — mean ROC across "
        f"{int(summary.loc[summary['cohort'] == cohort, 'n_repeats'].min())} repeated holdouts",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths = []
    stem = output_dir / f"10_mean_roc_curves_{COHORT_LABELS.get(cohort, cohort)}"
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def aggregate(
    root: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    expected_repeats: int = 100,
    overwrite: bool = False,
) -> dict[str, Path]:
    manifest = _read_json(manifest_path)
    entries = list(manifest.get("entries", []))
    if len(entries) != 15:
        raise SummaryError(f"Expected 15 cohort/algorithm entries, observed {len(entries)}")

    observed_pairs = {(str(e["cohort"]), str(e["algorithm"])) for e in entries}
    expected_pairs = {
        (cohort, algorithm)
        for cohort in COHORT_ORDER
        for algorithm in ALGORITHMS
    }
    if observed_pairs != expected_pairs:
        raise SummaryError(
            f"Manifest cohort/algorithm matrix mismatch: missing={sorted(expected_pairs-observed_pairs)}, "
            f"extra={sorted(observed_pairs-expected_pairs)}"
        )

    split_signatures: dict[tuple[str, int], tuple[tuple[str, int], ...]] = {}
    validation_counts: dict[str, int] = {}
    curve_store: dict[tuple[str, str, str], list[np.ndarray]] = {}
    metric_rows: list[dict[str, Any]] = []

    for cohort in COHORT_ORDER:
        for algorithm in ALGORITHMS:
            entry = next(
                e for e in entries
                if str(e["cohort"]) == cohort and str(e["algorithm"]) == algorithm
            )
            rows, _ = _load_entry(
                root, entry, expected_repeats,
                split_signatures, validation_counts, curve_store,
            )
            metric_rows.extend(rows)
            print(
                f"[PASS] {cohort}:{algorithm}: "
                f"{len(rows)} model-repeat metric rows"
            )

    repeat_metrics = pd.DataFrame(metric_rows)
    expected_metric_rows = sum(
        expected_repeats * len(
            _read_yaml(_resolve(root, str(entry["config"])) )["models"]
        )
        for entry in entries
    )
    if len(repeat_metrics) != expected_metric_rows:
        raise SummaryError(
            f"Metric row count mismatch: {len(repeat_metrics)} != {expected_metric_rows}"
        )
    if repeat_metrics.duplicated(
        ["cohort", "algorithm", "model", "outer_repeat", "outer_fold"]
    ).any():
        raise SummaryError("Duplicate cohort/algorithm/model/repeat metric rows")

    summary = summarize_repeat_metrics(repeat_metrics)
    if not (summary["n_repeats"] == expected_repeats).all():
        raise SummaryError("At least one cohort/algorithm/model lacks all repeats")
    curve_table = _build_curve_table(curve_store, expected_repeats)

    output_dir = output_dir.expanduser().resolve()
    output_files = [
        output_dir / "10_repeat_level_metrics.csv",
        output_dir / "10_model_performance_summary.csv",
        output_dir / "10_mean_roc_curve_points.csv",
        output_dir / "10_PERFORMANCE_AGGREGATION_COMPLETE.json",
    ]
    for cohort in COHORT_ORDER:
        stem = output_dir / f"10_mean_roc_curves_{COHORT_LABELS[cohort]}"
        output_files.extend(stem.with_suffix(f".{suffix}") for suffix in ("png", "pdf", "svg"))
    existing = [path for path in output_files if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Performance summary outputs already exist; use --overwrite to replace:\n"
            + "\n".join(map(str, existing))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    repeat_metrics.to_csv(output_files[0], index=False)
    summary.to_csv(output_files[1], index=False)
    curve_table.to_csv(output_files[2], index=False)

    generated_figures: list[Path] = []
    for cohort in COHORT_ORDER:
        models = list(
            _read_yaml(_resolve(
                root,
                str(next(e for e in entries if e["cohort"] == cohort and e["algorithm"] == "elastic_net")["config"])
            ))["models"].keys()
        )
        generated_figures.extend(
            _plot_cohort_roc(cohort, models, curve_table, summary, output_dir)
        )

    payload = {
        "status": "COMPLETE",
        "aggregation": "100-repeat outer holdout mean ROC and performance summary",
        "expected_repeats_per_model": expected_repeats,
        "cohort_algorithm_entries": len(entries),
        "repeat_level_metric_rows": len(repeat_metrics),
        "model_combinations": len(summary),
        "roc_curve_grid_points_per_combination": len(ROC_GRID),
        "cohorts": list(COHORT_ORDER),
        "algorithms": list(ALGORITHMS),
        "validation_sample_counts": validation_counts,
        "interpretation": {
            "roc_auc": "Mean and SD across repeated outer validation splits.",
            "average_precision": "Average precision (AP); sourced from pr_auc or average_precision in task metrics.",
            "roc_curve": "Per-repeat ROC curves interpolated to a common FPR grid, then averaged. Shading is mean TPR ± 1 SD, not a confidence interval.",
            "linear_svm": "ROC/AP use the continuous decision_function score; scores are not calibrated probabilities.",
            "repeated_holdout": "The 100 repeats are overlapping repeated holdout splits, not 100 independent patient cohorts."
        },
        "generated_files": [str(path) for path in output_files],
    }
    marker_path = output_files[3]
    marker_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n========== PERFORMANCE SUMMARY ==========")
    display_columns = [
        "cohort_label", "algorithm_label", "model", "n_repeats",
        "roc_auc_mean", "roc_auc_sd", "average_precision_mean", "average_precision_sd",
    ]
    print(summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nOutput directory: {output_dir}")
    print(f"Summary table: {output_files[1]}")
    print(f"Repeat-level metrics: {output_files[0]}")
    print(f"Mean ROC points: {output_files[2]}")
    print(f"Completion marker: {marker_path}")
    for path in generated_figures:
        print(f"Figure: {path}")
    print("TRB V3 REPEAT100 PERFORMANCE AGGREGATION: COMPLETE")

    return {
        "repeat_metrics": output_files[0],
        "summary": output_files[1],
        "roc_points": output_files[2],
        "complete": marker_path,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize TRB V3 179-sample 100-repeat five-algorithm outer validation."
    )
    parser.add_argument("--repository-root", default=None)
    parser.add_argument(
        "--manifest",
        default="TRB/set/configs/schemes/generated/hierarchical_igh_aligned_five_algorithm/FIVE_ALGORITHM_RUN_MANIFEST.json",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-repeats", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.repository_root or Path(__file__).resolve().parents[3]).expanduser().resolve()
    manifest_path = _resolve(root, args.manifest)
    global_dir = root / (
        "TRB/set/experiments/trb_v3_179_igh_aligned_five_algorithm_global"
    )
    output_dir = (
        _resolve(root, args.output_dir)
        if args.output_dir
        else global_dir / "10_performance_summary"
    )
    try:
        aggregate(
            root,
            manifest_path,
            output_dir,
            expected_repeats=args.expected_repeats,
            overwrite=args.overwrite,
        )
        return 0
    except Exception as exc:
        print(
            f"TRB V3 performance aggregation: FAIL\n{exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
