#!/usr/bin/env python3
"""Deterministic contracts for TRB V3 Layer-1/Layer-2 feature experiments.

Layer 1 enumerates every non-empty combination of the seven Core83 groups.
Layer 2 promotes complete Layer-1 results separately per cohort and algorithm,
then creates paired static/3-mer/RA-hit/combined contrasts. Model execution is
intentionally delegated to the repository's nested-CV engines.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

CORE83_GROUPS = (
    "tcr_diversity", "tcr_clonal_expansion", "tcr_aa_length",
    "tcr_aa_composition_unweighted", "tcr_aa_composition_weighted",
    "tcr_physicochemical_unweighted", "tcr_physicochemical_weighted",
)
GROUP_CODES = dict(zip(CORE83_GROUPS, ("D","E","L","AU","AW","PU","PW")))
COHORTS = {"total":179, "pbmc":116, "buffycoat":63}
ALGORITHMS = ("elastic_net","ridge","lasso","linear_svm","xgboost")
RA_HIT_THRESHOLD_PCT = 20.0
RA_HIT_DELTA_PCT = 10.0

@dataclass(frozen=True)
class Layer1Combination:
    combination_id: str
    group_ids: tuple[str, ...]
    feature_code: str

def build_layer1_combinations(groups: Sequence[str] = CORE83_GROUPS) -> tuple[Layer1Combination,...]:
    ordered=tuple(groups)
    if len(ordered)!=7 or set(ordered)!=set(CORE83_GROUPS) or len(set(ordered))!=7:
        raise ValueError("Layer 1 requires exactly the seven unique Core83 groups")
    out=[]
    for size in range(1,8):
        for selected in combinations(ordered,size):
            code="_".join(GROUP_CODES[x] for x in selected)
            out.append(Layer1Combination(f"L1_{len(out)+1:03d}_{code}",selected,code))
    if len(out)!=127: raise AssertionError(len(out))
    return tuple(out)

def build_layer1_manifest() -> list[dict]:
    rows=[]
    for cohort,n in COHORTS.items():
        for algorithm in ALGORITHMS:
            for c in build_layer1_combinations():
                rows.append({"layer":"layer1","cohort":cohort,"expected_samples":n,
                    "algorithm":algorithm,"combination_id":c.combination_id,
                    "feature_code":c.feature_code,"feature_groups":list(c.group_ids),
                    "n_feature_groups":len(c.group_ids),"outer_repeats":100,
                    "outer_fold":1,"status":"planned"})
    return rows

def rank_layer1_results(metrics: Iterable[Mapping], *, top_n:int=5,
                        primary_metric:str="roc_auc", secondary_metric:str="pr_auc") -> list[dict]:
    """Promote top-N per cohort×algorithm; incomplete repeats are never eligible."""
    if top_n<1: raise ValueError("top_n must be >=1")
    if primary_metric not in {"roc_auc","pr_auc","average_precision"}:
        raise ValueError("unsupported primary metric")
    rows=[dict(x) for x in metrics]
    required={"cohort","algorithm","combination_id",primary_metric,secondary_metric,"n_repeats"}
    grouped={}
    for row in rows:
        missing=required-set(row)
        if missing: raise ValueError(f"Layer-1 result missing {sorted(missing)}")
        if int(row["n_repeats"])!=100:
            raise ValueError("only complete 100-repeat combinations may be promoted")
        key=(str(row["cohort"]),str(row["algorithm"]))
        if key[0] not in COHORTS or key[1] not in ALGORITHMS: raise ValueError(f"invalid group {key}")
        grouped.setdefault(key,[]).append(row)
    selected=[]
    for (cohort,algorithm),group in sorted(grouped.items()):
        group.sort(key=lambda r:(-float(r[primary_metric]),-float(r[secondary_metric]),
            float(r.get(primary_metric+"_sd",float("inf"))),str(r["combination_id"])))
        for rank,row in enumerate(group[:top_n],1):
            selected.append({"cohort":cohort,"algorithm":algorithm,"rank":rank,
                "combination_id":row["combination_id"],"feature_groups":row.get("feature_groups",[]),
                "layer1_roc_auc":row.get("roc_auc"),
                "layer1_pr_auc":row.get("pr_auc",row.get("average_precision")),
                "layer1_n_repeats":100})
    return selected

def build_layer2_manifest(selected: Iterable[Mapping]) -> list[dict]:
    """Build F0/F1/F2/F3 paired contrasts for every promoted Layer-1 combination."""
    rows=[]; seen=set()
    states=(("F0_static",False,False),("F1_3mer",True,False),
            ("F2_ra_hit",False,True),("F3_3mer_ra_hit",True,True))
    for item in selected:
        cohort=str(item["cohort"]); algorithm=str(item["algorithm"]); combo=str(item["combination_id"])
        if cohort not in COHORTS or algorithm not in ALGORITHMS: raise ValueError("invalid cohort/algorithm")
        key=(cohort,algorithm,combo)
        if key in seen: raise ValueError(f"duplicate Layer-1 promotion {key}")
        seen.add(key)
        for state,use_3mer,use_ra in states:
            rows.append({"layer":"layer2","cohort":cohort,"expected_samples":COHORTS[cohort],
                "algorithm":algorithm,"rank":int(item["rank"]),"combination_id":combo,
                "feature_groups":list(item.get("feature_groups",[])),"feature_state":state,
                "include_3mer":use_3mer,"include_ra_hit":use_ra,
                "ra_hit_threshold_pct":RA_HIT_THRESHOLD_PCT if use_ra else None,
                "ra_hit_delta_pct":RA_HIT_DELTA_PCT if use_ra else None,
                "outer_repeats":100,"outer_fold":1,"status":"planned"})
    return rows
