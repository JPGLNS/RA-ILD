#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified experiment-spec contract for TRB Feature Framework advanced sources.

Phase 5 deliberately validates configuration only. It does not fit ML models.
The validated fields are consumed by dynamic repertoire, k-mer, enriched-
dictionary, cohort, and later model-dispatch layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Tuple, Union

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for advanced experiment specs") from exc

SCHEMA_VERSION = "1.0"
COHORT_SCOPES = {"total", "pbmc", "buffycoat"}
REPERTOIRE_MODES = {"all", "topk"}
KMER_REPRESENTATIONS = {"unweighted", "weighted"}
MODEL_ALGORITHMS = {"elastic_net", "ridge", "lasso", "linear_svm", "xgboost"}
STATIC_TCR_GROUPS = {
    "tcr_qc_depth",
    "tcr_diversity",
    "tcr_clonal_expansion",
    "tcr_aa_length",
    "tcr_aa_composition_unweighted",
    "tcr_aa_composition_weighted",
    "tcr_physicochemical_unweighted",
    "tcr_physicochemical_weighted",
}
CLINICAL_GROUPS = {"clinical_age_sex", "clinical_material"}
ENRICH_FEATURES = {
    "RA_dict_clone_count",
    "ILD_dict_clone_count",
    "RA_dict_hit_rate",
    "ILD_dict_hit_rate",
    "RA_dict_read_fraction_sum",
    "ILD_dict_read_fraction_sum",
    "RA_minus_ILD_count",
    "RA_minus_ILD_read_fraction",
    "RA_minus_ILD_hit_rate",
    "RA_hit_fraction_of_sample",
    "ILD_hit_fraction_of_sample",
    "RA_minus_ILD_sample_fraction",
}


class ExperimentSpecError(ValueError):
    pass


def _mapping(value: Any, context: str, *, default_empty: bool = False) -> Mapping[str, Any]:
    if value is None and default_empty:
        return {}
    if not isinstance(value, Mapping):
        raise ExperimentSpecError(f"{context} must be a mapping")
    return value


def _str(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentSpecError(f"{context} must be a non-empty string")
    return value.strip()


def _bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ExperimentSpecError(f"{context} must be true/false")
    return bool(value)


def _int(value: Any, context: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExperimentSpecError(f"{context} must be an integer >= {minimum}")
    return int(value)


def _float(value: Any, context: str, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentSpecError(f"{context} must be numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ExperimentSpecError(f"{context} must be >= {minimum}")
    return result


def _strings(value: Any, context: str, *, allow_empty: bool = True) -> Tuple[str, ...]:
    if value is None:
        if allow_empty:
            return tuple()
        raise ExperimentSpecError(f"{context} must be a non-empty list")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExperimentSpecError(f"{context} must be a list of strings")
    out = tuple(_str(item, f"{context}[]") for item in value)
    if not allow_empty and not out:
        raise ExperimentSpecError(f"{context} must not be empty")
    if len(out) != len(set(out)):
        raise ExperimentSpecError(f"{context} contains duplicates")
    return out


@dataclass(frozen=True)
class CohortSpec:
    scope: str = "total"


@dataclass(frozen=True)
class RepertoireSpec:
    mode: str = "topk"
    top_k: Optional[int] = 10000

    @property
    def source_id(self) -> str:
        return "all" if self.mode == "all" else f"top{self.top_k}"


@dataclass(frozen=True)
class KmerSpec:
    enabled: bool = True
    k: int = 3
    representation: str = "unweighted"
    top_n: Optional[int] = 500
    min_sample_count: int = 5
    min_prevalence: float = 0.05
    min_unweighted_variance: float = 1.0e-12
    min_weighted_variance: float = 1.0e-12


@dataclass(frozen=True)
class EnrichedDictionarySpec:
    enabled: bool = True
    threshold_pct: float = 20.0
    delta_pct: float = 10.0
    features: Tuple[str, ...] = ("RA_dict_hit_rate",)


@dataclass(frozen=True)
class FeatureSpec:
    clinical_groups: Tuple[str, ...]
    static_tcr_groups: Tuple[str, ...]
    kmer: KmerSpec
    enriched_dictionary: EnrichedDictionarySpec

    @property
    def phase2_feature_groups(self) -> Tuple[str, ...]:
        groups = list(self.clinical_groups) + list(self.static_tcr_groups)
        if self.kmer.enabled:
            groups.append(f"tcr_3mer_{self.kmer.representation}")
        return tuple(groups)


@dataclass(frozen=True)
class ModelSpec:
    algorithm: str = "elastic_net"


@dataclass(frozen=True)
class ValidationSpec:
    folds: int = 5
    repeats: int = 100
    split_candidates: int = 300
    seed: int = 20260711


@dataclass(frozen=True)
class ComputeSpec:
    workers: int = 1


@dataclass(frozen=True)
class AdvancedExperimentSpec:
    experiment_id: str
    receptor: str
    cohort: CohortSpec
    repertoire: RepertoireSpec
    features: FeatureSpec
    model: ModelSpec
    validation: ValidationSpec
    compute: ComputeSpec
    source_path: Path

    @property
    def repertoire_source_id(self) -> str:
        return self.repertoire.source_id

    def as_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "receptor": self.receptor,
            "cohort": {"scope": self.cohort.scope},
            "repertoire": {
                "mode": self.repertoire.mode,
                "top_k": self.repertoire.top_k,
                "resolved_source_id": self.repertoire.source_id,
            },
            "features": {
                "clinical_groups": list(self.features.clinical_groups),
                "static_tcr_groups": list(self.features.static_tcr_groups),
                "kmer": {
                    "enabled": self.features.kmer.enabled,
                    "k": self.features.kmer.k,
                    "representation": self.features.kmer.representation,
                    "top_n": self.features.kmer.top_n,
                    "min_sample_count": self.features.kmer.min_sample_count,
                    "min_prevalence": self.features.kmer.min_prevalence,
                    "min_unweighted_variance": self.features.kmer.min_unweighted_variance,
                    "min_weighted_variance": self.features.kmer.min_weighted_variance,
                },
                "enriched_dictionary": {
                    "enabled": self.features.enriched_dictionary.enabled,
                    "threshold_pct": self.features.enriched_dictionary.threshold_pct,
                    "delta_pct": self.features.enriched_dictionary.delta_pct,
                    "features": list(self.features.enriched_dictionary.features),
                    "repertoire_source": self.repertoire.source_id,
                    "source_policy": "inherit_main_repertoire",
                },
            },
            "model": {"algorithm": self.model.algorithm},
            "validation": {
                "folds": self.validation.folds,
                "repeats": self.validation.repeats,
                "split_candidates": self.validation.split_candidates,
                "seed": self.validation.seed,
            },
            "compute": {"workers": self.compute.workers},
            "source_path": str(self.source_path),
        }


def load_advanced_experiment(path: Union[str, Path]) -> AdvancedExperimentSpec:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _mapping(raw, "experiment spec")
    if str(raw.get("schema_version")) != SCHEMA_VERSION:
        raise ExperimentSpecError(f"schema_version must be {SCHEMA_VERSION}")

    exp = _mapping(raw.get("experiment"), "experiment")
    experiment_id = _str(exp.get("id"), "experiment.id")
    receptor = _str(exp.get("receptor", "TRB"), "experiment.receptor").upper()
    if receptor != "TRB":
        raise ExperimentSpecError("experiment.receptor must be TRB")

    cohort_raw = _mapping(raw.get("cohort", {}), "cohort")
    scope = _str(cohort_raw.get("scope", "total"), "cohort.scope").lower()
    if scope not in COHORT_SCOPES:
        raise ExperimentSpecError(f"cohort.scope must be one of {sorted(COHORT_SCOPES)}")

    rep_raw = _mapping(raw.get("repertoire", {}), "repertoire")
    mode = _str(rep_raw.get("mode", "topk"), "repertoire.mode").lower()
    if mode not in REPERTOIRE_MODES:
        raise ExperimentSpecError(f"repertoire.mode must be one of {sorted(REPERTOIRE_MODES)}")
    top_k_raw = rep_raw.get("top_k", 10000 if mode == "topk" else None)
    if mode == "topk":
        top_k = _int(top_k_raw, "repertoire.top_k", 1)
    else:
        if top_k_raw not in (None, ""):
            raise ExperimentSpecError("repertoire.top_k must be null/omitted when mode=all")
        top_k = None

    features_raw = _mapping(raw.get("features", {}), "features")
    clinical = _strings(
        features_raw.get("clinical_groups", ["clinical_age_sex"]),
        "features.clinical_groups",
    )
    unknown = sorted(set(clinical) - CLINICAL_GROUPS)
    if unknown:
        raise ExperimentSpecError(f"unknown clinical_groups: {unknown}")
    static = _strings(
        features_raw.get(
            "static_tcr_groups",
            [
                "tcr_diversity",
                "tcr_clonal_expansion",
                "tcr_aa_length",
                "tcr_aa_composition_unweighted",
                "tcr_aa_composition_weighted",
                "tcr_physicochemical_unweighted",
                "tcr_physicochemical_weighted",
            ],
        ),
        "features.static_tcr_groups",
    )
    unknown = sorted(set(static) - STATIC_TCR_GROUPS)
    if unknown:
        raise ExperimentSpecError(f"unknown static_tcr_groups: {unknown}")

    kraw = _mapping(features_raw.get("kmer", {}), "features.kmer")
    k_enabled = _bool(kraw.get("enabled", True), "features.kmer.enabled")
    k = _int(kraw.get("k", 3), "features.kmer.k", 1)
    if k != 3:
        raise ExperimentSpecError("Phase-5 currently supports amino-acid k=3 only")
    representation = _str(kraw.get("representation", "unweighted"), "features.kmer.representation").lower()
    if representation not in KMER_REPRESENTATIONS:
        raise ExperimentSpecError(
            f"features.kmer.representation must be one of {sorted(KMER_REPRESENTATIONS)}"
        )
    top_n_raw = kraw.get("top_n", 500)
    top_n = None if top_n_raw in (None, 0) else _int(top_n_raw, "features.kmer.top_n", 1)
    min_sample_count = _int(kraw.get("min_sample_count", 5), "features.kmer.min_sample_count", 1)
    min_prevalence = _float(kraw.get("min_prevalence", 0.05), "features.kmer.min_prevalence", 0.0)
    if min_prevalence > 1:
        raise ExperimentSpecError("features.kmer.min_prevalence must be <= 1")
    min_uw_var = _float(kraw.get("min_unweighted_variance", 1e-12), "features.kmer.min_unweighted_variance", 0.0)
    min_w_var = _float(kraw.get("min_weighted_variance", 1e-12), "features.kmer.min_weighted_variance", 0.0)

    eraw = _mapping(features_raw.get("enriched_dictionary", {}), "features.enriched_dictionary")
    # Explicit source override is forbidden: enrich must inherit the main repertoire.
    for forbidden in ("repertoire_source", "source", "source_id"):
        if forbidden in eraw:
            raise ExperimentSpecError(
                f"features.enriched_dictionary.{forbidden} is not allowed; enriched dictionary must inherit repertoire.mode/top_k"
            )
    e_enabled = _bool(eraw.get("enabled", True), "features.enriched_dictionary.enabled")
    threshold = _float(eraw.get("threshold_pct", 20.0), "features.enriched_dictionary.threshold_pct", 0.0)
    delta = _float(eraw.get("delta_pct", 10.0), "features.enriched_dictionary.delta_pct", 0.0)
    if threshold <= 0 or threshold > 100:
        raise ExperimentSpecError("enriched threshold_pct must be in (0, 100]")
    if delta > threshold:
        raise ExperimentSpecError("enriched delta_pct must be <= threshold_pct")
    e_features = _strings(
        eraw.get("features", ["RA_dict_hit_rate"]),
        "features.enriched_dictionary.features",
        allow_empty=not e_enabled,
    )
    unknown = sorted(set(e_features) - ENRICH_FEATURES)
    if unknown:
        raise ExperimentSpecError(f"unknown enriched_dictionary features: {unknown}")

    model_raw = _mapping(raw.get("model", {}), "model")
    algorithm = _str(model_raw.get("algorithm", "elastic_net"), "model.algorithm").lower()
    if algorithm not in MODEL_ALGORITHMS:
        raise ExperimentSpecError(f"model.algorithm must be one of {sorted(MODEL_ALGORITHMS)}")

    val_raw = _mapping(raw.get("validation", {}), "validation")
    folds = _int(val_raw.get("folds", 5), "validation.folds", 2)
    repeats = _int(val_raw.get("repeats", 100), "validation.repeats", 1)
    candidates = _int(val_raw.get("split_candidates", 300), "validation.split_candidates", 1)
    seed = _int(val_raw.get("seed", 20260711), "validation.seed", 0)

    compute_raw = _mapping(raw.get("compute", {}), "compute")
    workers = _int(compute_raw.get("workers", 1), "compute.workers", 1)
    if workers > 16:
        raise ExperimentSpecError("compute.workers must be <= 16")

    return AdvancedExperimentSpec(
        experiment_id=experiment_id,
        receptor=receptor,
        cohort=CohortSpec(scope=scope),
        repertoire=RepertoireSpec(mode=mode, top_k=top_k),
        features=FeatureSpec(
            clinical_groups=clinical,
            static_tcr_groups=static,
            kmer=KmerSpec(
                enabled=k_enabled,
                k=k,
                representation=representation,
                top_n=top_n,
                min_sample_count=min_sample_count,
                min_prevalence=min_prevalence,
                min_unweighted_variance=min_uw_var,
                min_weighted_variance=min_w_var,
            ),
            enriched_dictionary=EnrichedDictionarySpec(
                enabled=e_enabled,
                threshold_pct=threshold,
                delta_pct=delta,
                features=e_features,
            ),
        ),
        model=ModelSpec(algorithm=algorithm),
        validation=ValidationSpec(
            folds=folds,
            repeats=repeats,
            split_candidates=candidates,
            seed=seed,
        ),
        compute=ComputeSpec(workers=workers),
        source_path=path,
    )
