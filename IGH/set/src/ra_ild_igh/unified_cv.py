#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leakage-controlled repeated-holdout orchestration for Phase 6."""
from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .fold_features import FoldFeatureInputs, build_fold_features
from .unified_experiment import UnifiedExperimentSpec
from .unified_modeling import (
    ModelCandidate,
    PreparedFold,
    build_candidates,
    candidate_tuning_row,
    choose_youden_threshold,
    classification_metrics,
    derive_seed,
    fit_candidate,
    prepare_fold,
    select_candidate,
)

PathLike = Union[str, Path]


class UnifiedCVError(ValueError):
    pass


@dataclass(frozen=True)
class RepeatedHoldoutBank:
    assignments: pd.DataFrame
    repeat_ids: Tuple[int, ...]
    train_role: str = "train"
    holdout_role: str = "holdout"


@dataclass(frozen=True)
class UnifiedRepeatResult:
    repeat: int
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    inner_tuning: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    explanations: pd.DataFrame
    feature_audit: pd.DataFrame
    preprocessing_audit: pd.DataFrame
    inner_assignment_audit: pd.DataFrame


def load_repeated_holdout_bank(
    path: PathLike,
    expected_sample_ids: Sequence[str],
    *,
    requested_repeats: int,
) -> RepeatedHoldoutBank:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"repeated-holdout assignments not found: {source}")
    frame = pd.read_csv(source)
    if "sample_id" not in frame.columns and "libraryid" in frame.columns:
        frame = frame.rename(columns={"libraryid": "sample_id"})
    required = {"sample_id", "repeat_index", "role"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise UnifiedCVError(f"assignment file missing columns: {missing}")
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["repeat_index"] = pd.to_numeric(frame["repeat_index"], errors="raise").astype(int)
    frame["role"] = frame["role"].astype(str).str.strip().str.lower()
    expected = set(map(str, expected_sample_ids))
    repeat_ids = tuple(sorted(frame["repeat_index"].unique().tolist()))
    if not repeat_ids:
        raise UnifiedCVError("assignment file contains no repeats")
    if requested_repeats > len(repeat_ids):
        raise UnifiedCVError(
            f"requested repeats={requested_repeats} but frozen assignment bank contains {len(repeat_ids)}"
        )
    selected = tuple(repeat_ids[:requested_repeats])
    frame = frame.loc[frame["repeat_index"].isin(selected)].copy()
    for repeat in selected:
        part = frame.loc[frame["repeat_index"] == repeat]
        if part["sample_id"].duplicated().any():
            raise UnifiedCVError(f"repeat {repeat} contains duplicate sample_id")
        observed = set(part["sample_id"])
        if observed != expected:
            missing_ids = sorted(expected - observed)[:20]
            extra_ids = sorted(observed - expected)[:20]
            raise UnifiedCVError(
                f"repeat {repeat} sample coverage mismatch; missing={missing_ids}, extra={extra_ids}"
            )
        roles = set(part["role"])
        if roles != {"train", "holdout"}:
            raise UnifiedCVError(f"repeat {repeat} roles must be train/holdout; observed={sorted(roles)}")
        for role in roles:
            labels = set(part.loc[part["role"] == role].merge(
                pd.DataFrame({"sample_id": list(expected)}), on="sample_id", how="inner"
            )["sample_id"])
            if not labels:
                raise UnifiedCVError(f"repeat {repeat} has empty role {role}")
    return RepeatedHoldoutBank(assignments=frame, repeat_ids=selected)


def _categorical_score(full: pd.DataFrame, valid: pd.DataFrame, columns: Sequence[str]) -> float:
    scores: List[float] = []
    for column in columns:
        if column not in full.columns or full[column].nunique(dropna=False) <= 1:
            continue
        levels = sorted(full[column].astype(str).unique().tolist())
        full_prop = full[column].astype(str).value_counts(normalize=True)
        valid_prop = valid[column].astype(str).value_counts(normalize=True)
        scores.extend(abs(float(full_prop.get(x, 0.0)) - float(valid_prop.get(x, 0.0))) for x in levels)
    return max(scores, default=0.0)


def _numeric_score(full: pd.DataFrame, valid: pd.DataFrame, columns: Sequence[str]) -> float:
    scores: List[float] = []
    for column in columns:
        if column not in full.columns:
            continue
        all_values = pd.to_numeric(full[column], errors="raise").to_numpy(float)
        valid_values = pd.to_numeric(valid[column], errors="raise").to_numpy(float)
        sd = float(np.std(all_values, ddof=0))
        scores.append(abs(float(np.mean(valid_values)) - float(np.mean(all_values))) / sd if sd > 0 else 0.0)
    return max(scores, default=0.0)


def generate_inner_folds(
    metadata: pd.DataFrame,
    outer_train_ids: Sequence[str],
    spec: UnifiedExperimentSpec,
    *,
    repeat: int,
) -> Tuple[Mapping[int, Tuple[str, ...]], pd.DataFrame]:
    train = metadata.set_index("sample_id").loc[list(map(str, outer_train_ids))].reset_index()
    folds = spec.cv.inner_folds
    strata = train["cohort"].astype(str)
    counts = strata.value_counts()
    if int(counts.min()) < folds:
        raise UnifiedCVError(
            f"outer repeat {repeat}: minimum cohort count={int(counts.min())} < inner_folds={folds}"
        )
    balance_cat = [c for c in ("batch", "material", "sex") if c in train.columns and train[c].nunique() > 1]
    balance_num = [c for c in ("age",) if c in train.columns]
    best: Optional[Tuple[float, int, np.ndarray]] = None
    for candidate_index in range(spec.cv.inner_split_candidates):
        seed = int(spec.cv.inner_split_seed + int(repeat) * 100000 + candidate_index)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        labels = np.zeros(len(train), dtype=int)
        fold_scores = []
        for fold, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(train)), strata.to_numpy()), start=1):
            labels[valid_idx] = fold
            valid = train.iloc[valid_idx]
            fold_scores.append(max(
                _categorical_score(train, valid, ["cohort", *balance_cat]),
                _numeric_score(train, valid, balance_num),
            ))
        score = float(max(fold_scores))
        if best is None or (score, seed) < (best[0], best[1]):
            best = (score, seed, labels.copy())
    assert best is not None
    score, seed, labels = best
    result: Dict[int, Tuple[str, ...]] = {}
    audit = []
    for fold in range(1, folds + 1):
        ids = tuple(train.loc[labels == fold, "sample_id"].astype(str))
        if not ids:
            raise UnifiedCVError(f"repeat {repeat} inner fold {fold} is empty")
        sub = train.loc[labels == fold]
        if set(sub["cohort"].astype(str).str.upper()) != {"RA", "ILD"}:
            raise UnifiedCVError(f"repeat {repeat} inner fold {fold} lacks both cohorts")
        result[fold] = ids
        audit.append({
            "outer_repeat": repeat,
            "inner_fold": fold,
            "validation_samples": len(ids),
            "candidate_seed": seed,
            "candidates_evaluated": spec.cv.inner_split_candidates,
            "max_balance_score": score,
            "categorical_max_abs_difference": _categorical_score(train, sub, ["cohort", *balance_cat]),
            "numeric_max_standardized_difference": _numeric_score(train, sub, balance_num),
        })
    return result, pd.DataFrame(audit)


def _outer_ids(bank: RepeatedHoldoutBank, repeat: int) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    part = bank.assignments.loc[bank.assignments["repeat_index"] == int(repeat)]
    train = tuple(part.loc[part["role"] == "train", "sample_id"].astype(str))
    holdout = tuple(part.loc[part["role"] == "holdout", "sample_id"].astype(str))
    if not train or not holdout:
        raise UnifiedCVError(f"repeat {repeat} has empty train/holdout")
    return train, holdout


def _prepare_inner_folds(
    inputs: FoldFeatureInputs,
    spec: UnifiedExperimentSpec,
    outer_train_ids: Sequence[str],
    inner_valid: Mapping[int, Tuple[str, ...]],
    *,
    repeat: int,
) -> Tuple[Mapping[int, PreparedFold], pd.DataFrame, pd.DataFrame]:
    prepared: Dict[int, PreparedFold] = {}
    feature_audits: List[pd.DataFrame] = []
    prep_audits: List[pd.DataFrame] = []
    outer_set = set(map(str, outer_train_ids))
    for fold, valid_ids in inner_valid.items():
        valid_set = set(valid_ids)
        train_ids = tuple(sid for sid in outer_train_ids if sid not in valid_set)
        if set(train_ids) | valid_set != outer_set or set(train_ids) & valid_set:
            raise UnifiedCVError("invalid inner partition")
        feature = build_fold_features(
            inputs,
            spec,
            train_ids,
            valid_ids,
            context=f"repeat{repeat}_inner{fold}",
        )
        design = prepare_fold(
            feature.train,
            feature.valid,
            feature.numeric_columns,
            feature.categorical_columns,
            positive_label=spec.model.positive_label,
            zero_sd_tolerance=spec.model.zero_sd_tolerance,
        )
        prepared[fold] = design
        fa = feature.audit.copy()
        fa["outer_repeat"] = repeat
        fa["stage"] = "inner"
        fa["inner_fold"] = fold
        feature_audits.append(fa)
        pa = design.preprocessing_audit.copy()
        pa["outer_repeat"] = repeat
        pa["stage"] = "inner"
        pa["inner_fold"] = fold
        prep_audits.append(pa)
    return (
        prepared,
        pd.concat(feature_audits, ignore_index=True),
        pd.concat(prep_audits, ignore_index=True),
    )


def _tune(
    prepared: Mapping[int, PreparedFold],
    candidates: Sequence[ModelCandidate],
    spec: UnifiedExperimentSpec,
    *,
    repeat: int,
) -> Tuple[pd.DataFrame, ModelCandidate, float, pd.DataFrame]:
    tuning_rows: List[Dict[str, Any]] = []
    cache: Dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        predictions: List[Dict[str, Any]] = []
        converged: List[bool] = []
        feature_counts: List[int] = []
        for fold, data in prepared.items():
            seed = derive_seed(
                spec.base.validation.seed,
                "inner", repeat, fold, candidate.candidate_id,
            )
            fit = fit_candidate(candidate, data.X_train, data.y_train, spec, random_state=seed)
            score = fit.score(data.X_valid)
            converged.append(bool(fit.converged))
            feature_counts.append(int(data.X_train.shape[1]))
            predictions.extend({
                "outer_repeat": repeat,
                "inner_fold": fold,
                "candidate_id": candidate.candidate_id,
                "sample_id": sid,
                "true_label": int(y),
                "prediction_score": float(s),
                "score_type": fit.score_type,
            } for sid, y, s in zip(data.valid_ids, data.y_valid, score))
        pred = pd.DataFrame(predictions)
        cache[candidate.candidate_id] = pred
        tuning_rows.append(candidate_tuning_row(
            candidate,
            pred["true_label"].to_numpy(int),
            pred["prediction_score"].to_numpy(float),
            score_type=str(pred["score_type"].iloc[0]),
            all_converged=all(converged),
            min_features=min(feature_counts),
            max_features=max(feature_counts),
        ))
    tuning = pd.DataFrame(tuning_rows)
    best = select_candidate(tuning, spec)
    candidate_lookup = {c.candidate_id: c for c in candidates}
    selected = candidate_lookup[str(best["candidate_id"])]
    selected_pred = cache[selected.candidate_id].copy()
    score_type = str(selected_pred["score_type"].iloc[0])
    threshold = choose_youden_threshold(
        selected_pred["true_label"].to_numpy(int),
        selected_pred["prediction_score"].to_numpy(float),
        score_type=score_type,
    )
    selected_pred["selected_threshold"] = threshold
    tuning["selected"] = tuning["candidate_id"].astype(str).eq(selected.candidate_id)
    tuning["outer_repeat"] = repeat
    return tuning, selected, threshold, selected_pred


def run_repeat(
    inputs: FoldFeatureInputs,
    spec: UnifiedExperimentSpec,
    bank: RepeatedHoldoutBank,
    *,
    repeat: int,
) -> UnifiedRepeatResult:
    outer_train, outer_holdout = _outer_ids(bank, repeat)
    inner_valid, inner_audit = generate_inner_folds(
        inputs.metadata,
        outer_train,
        spec,
        repeat=repeat,
    )
    prepared, inner_feature_audit, inner_prep_audit = _prepare_inner_folds(
        inputs,
        spec,
        outer_train,
        inner_valid,
        repeat=repeat,
    )
    candidates = build_candidates(spec)
    tuning, selected, threshold, selected_oof = _tune(
        prepared,
        candidates,
        spec,
        repeat=repeat,
    )

    outer_feature = build_fold_features(
        inputs,
        spec,
        outer_train,
        outer_holdout,
        context=f"repeat{repeat}_outer",
    )
    outer_design = prepare_fold(
        outer_feature.train,
        outer_feature.valid,
        outer_feature.numeric_columns,
        outer_feature.categorical_columns,
        positive_label=spec.model.positive_label,
        zero_sd_tolerance=spec.model.zero_sd_tolerance,
    )
    seed = derive_seed(spec.base.validation.seed, "outer", repeat, selected.candidate_id)
    fit = fit_candidate(
        selected,
        outer_design.X_train,
        outer_design.y_train,
        spec,
        random_state=seed,
    )
    if not fit.converged:
        raise UnifiedCVError(f"repeat {repeat}: final selected model did not converge")
    score = fit.score(outer_design.X_valid)
    metrics = classification_metrics(
        outer_design.y_valid,
        score,
        threshold,
        score_type=fit.score_type,
    )
    metrics.update({
        "outer_repeat": repeat,
        "algorithm": spec.algorithm,
        "positive_label": spec.model.positive_label,
        "negative_label": spec.model.negative_label,
        "n_outer_train": len(outer_train),
        "n_outer_holdout": len(outer_holdout),
        "selected_candidate_id": selected.candidate_id,
        "selected_threshold": threshold,
        "score_type": fit.score_type,
        "n_final_predictors": len(outer_design.feature_names),
        "fit_converged": fit.converged,
        "iterations_used": fit.iterations if fit.iterations is not None else "",
    })
    metric_frame = pd.DataFrame([metrics])

    predicted = (score >= threshold).astype(int)
    predictions = pd.DataFrame({
        "outer_repeat": repeat,
        "sample_id": outer_design.valid_ids,
        "true_label": outer_design.y_valid,
        "true_cohort": [spec.model.positive_label if y == 1 else spec.model.negative_label for y in outer_design.y_valid],
        "prediction_score": score,
        "score_type": fit.score_type,
        "threshold": threshold,
        "predicted_label": predicted,
        "predicted_cohort": [spec.model.positive_label if y == 1 else spec.model.negative_label for y in predicted],
    })

    selected_row = selected.as_dict()
    selected_row.update({
        "outer_repeat": repeat,
        "selected_threshold": threshold,
        "inner_roc_auc": float(tuning.loc[tuning["selected"], "pooled_inner_roc_auc"].iloc[0]),
        "inner_pr_auc": float(tuning.loc[tuning["selected"], "pooled_inner_pr_auc"].iloc[0]),
        "inner_log_loss": tuning.loc[tuning["selected"], "pooled_inner_log_loss"].iloc[0],
    })
    selected_frame = pd.DataFrame([selected_row])

    explanation = fit.explanation_frame(outer_design.feature_names)
    explanation["outer_repeat"] = repeat

    outer_fa = outer_feature.audit.copy()
    outer_fa["outer_repeat"] = repeat
    outer_fa["stage"] = "outer_final"
    outer_fa["inner_fold"] = ""
    feature_audit = pd.concat([inner_feature_audit, outer_fa], ignore_index=True, sort=False)

    outer_pa = outer_design.preprocessing_audit.copy()
    outer_pa["outer_repeat"] = repeat
    outer_pa["stage"] = "outer_final"
    outer_pa["inner_fold"] = ""
    prep_audit = pd.concat([inner_prep_audit, outer_pa], ignore_index=True, sort=False)

    return UnifiedRepeatResult(
        repeat=repeat,
        metrics=metric_frame,
        predictions=predictions,
        inner_tuning=tuning,
        selected_hyperparameters=selected_frame,
        explanations=explanation,
        feature_audit=feature_audit,
        preprocessing_audit=prep_audit,
        inner_assignment_audit=inner_audit,
    )


_WORKER_CONTEXT: Optional[Tuple[FoldFeatureInputs, UnifiedExperimentSpec, RepeatedHoldoutBank]] = None


def _worker(repeat: int) -> UnifiedRepeatResult:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("worker context is not initialized")
    inputs, spec, bank = _WORKER_CONTEXT
    return run_repeat(inputs, spec, bank, repeat=int(repeat))


def run_repeats(
    inputs: FoldFeatureInputs,
    spec: UnifiedExperimentSpec,
    bank: RepeatedHoldoutBank,
    repeats: Sequence[int],
    *,
    workers: int,
) -> List[UnifiedRepeatResult]:
    repeat_ids = tuple(int(x) for x in repeats)
    if not repeat_ids:
        raise UnifiedCVError("no repeats requested")
    if workers < 1 or workers > 16:
        raise UnifiedCVError("workers must be in [1,16]")
    if workers == 1:
        results = []
        for index, r in enumerate(repeat_ids, start=1):
            result = run_repeat(inputs, spec, bank, repeat=r)
            results.append(result)
            print(f"[unified-cv] completed repeat {r} ({index}/{len(repeat_ids)})", flush=True)
        return results
    if "fork" not in mp.get_all_start_methods():
        raise UnifiedCVError("workers>1 requires Linux multiprocessing start method 'fork'")
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = (inputs, spec, bank)
    context = mp.get_context("fork")
    results: Dict[int, UnifiedRepeatResult] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        future_map = {executor.submit(_worker, repeat): repeat for repeat in repeat_ids}
        for future in concurrent.futures.as_completed(future_map):
            repeat = future_map[future]
            results[repeat] = future.result()
            print(f"[unified-cv] completed repeat {repeat} ({len(results)}/{len(repeat_ids)})", flush=True)
    return [results[r] for r in repeat_ids]
