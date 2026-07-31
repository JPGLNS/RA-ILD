#!/usr/bin/env bash
set -Eeuo pipefail

trap 'status=$?; echo "FAILED: line=${LINENO}, exit_status=${status}, time=$(date "+%F %T")"; exit "${status}"' ERR

cd /data/users/chenhaisheng/RA-ILD

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "Scheme 011: A1-A6 100-repeat full-grid benchmark"
echo "Started: $(date '+%F %T')"
echo "Repository: /data/users/chenhaisheng/RA-ILD"
echo "============================================================"

# ============================================================
# 1. 创建100-repeat split、training bundle和Scheme 011配置
# ============================================================

/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python - <<'PY'
from copy import deepcopy
from pathlib import Path

import yaml

root = Path("/data/users/chenhaisheng/RA-ILD")
config_root = root / "TRB/set/configs"


def read_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return data


def write_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            data,
            handle,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )


# ------------------------------------------------------------
# 1.1 找到原始repeat3 split specification
# ------------------------------------------------------------

split_matches = []

for path in config_root.rglob("*.yaml"):
    try:
        data = read_yaml(path)
    except Exception:
        continue

    split_set = data.get("split_set")
    if (
        isinstance(split_set, dict)
        and split_set.get("id") == "ra_ild_repeat3_v1"
    ):
        split_matches.append((path, data))

if len(split_matches) != 1:
    found = [str(path) for path, _ in split_matches]
    raise SystemExit(
        "Expected exactly one split YAML with "
        "split_set.id=ra_ild_repeat3_v1; "
        f"found {len(split_matches)}: {found}"
    )

split3_path, split3 = split_matches[0]
split100 = deepcopy(split3)

split100["split_set"]["id"] = "ra_ild_repeat100_v1"
split100["split_set"]["description"] = (
    "One hundred deterministic metadata-balanced "
    "123/51 repeated holdout splits."
)
split100["split_set"]["output_root"] = (
    "TRB/set/split_sets/ra_ild_repeat100_v1"
)
split100["split"]["repeats"] = 100

split100_path = (
    config_root
    / "repeated_holdout_splits"
    / "ra_ild_repeat100_v1.yaml"
)
write_yaml(split100_path, split100)

# ------------------------------------------------------------
# 1.2 创建100-repeat training bundle
# ------------------------------------------------------------

training3_path = (
    config_root
    / "repeated_holdout_training"
    / "trb_repeat3_training_v1.yaml"
)

if not training3_path.is_file():
    raise SystemExit(
        f"Original training-bundle specification not found: {training3_path}"
    )

training100 = deepcopy(read_yaml(training3_path))

training100["bundle"]["id"] = "trb_repeat100_training_inputs_v1"
training100["bundle"]["description"] = (
    "Full-cohort static and sequence inputs for one hundred "
    "frozen 123/51 repeated holdouts."
)
training100["bundle"]["output_root"] = (
    "TRB/set/training_bundles/"
    "trb_repeat100_training_inputs_v1"
)

training100["split_set"]["assignments"] = (
    "TRB/set/split_sets/ra_ild_repeat100_v1/"
    "repeated_holdout_assignments.csv"
)
training100["split_set"]["frozen_marker"] = (
    "TRB/set/split_sets/ra_ild_repeat100_v1/"
    "SPLITS_FROZEN.json"
)

training100["scheme"]["generated_scheme"] = (
    "TRB/set/configs/schemes/generated/"
    "trb_scheme_003_repeat100_no_clinical.generated.yaml"
)
training100["scheme"]["id"] = (
    "trb_scheme_003_repeat100_no_clinical"
)
training100["scheme"]["output_root"] = (
    "TRB/set/experiments/"
    "trb_scheme_003_repeat100_no_clinical"
)

training100_path = (
    config_root
    / "repeated_holdout_training"
    / "trb_repeat100_training_v1.yaml"
)
write_yaml(training100_path, training100)

# ------------------------------------------------------------
# 1.3 复制原Scheme 004的A1-A6模型，创建Scheme 011
# ------------------------------------------------------------

scheme4_path = (
    config_root
    / "schemes"
    / "trb_scheme_004_feature_ablation_repeat3.yaml"
)

if not scheme4_path.is_file():
    raise SystemExit(f"Scheme 004 not found: {scheme4_path}")

scheme11 = deepcopy(read_yaml(scheme4_path))

scheme11["scheme"]["id"] = (
    "trb_scheme_011_a1_a6_repeat100_fullgrid"
)
scheme11["scheme"]["description"] = (
    "One hundred frozen repeated holdouts comparing the original "
    "A1-A6 feature-ablation models under one common Ridge, "
    "Elastic Net and Lasso hyperparameter grid."
)
scheme11["scheme"]["base_resolved_config"] = (
    "TRB/set/experiments/"
    "trb_scheme_003_repeat100_no_clinical/"
    "00_config/resolved_config.yaml"
)
scheme11["scheme"]["output_root"] = (
    "TRB/set/experiments/"
    "trb_scheme_011_a1_a6_repeat100_fullgrid"
)

scheme11["model_engine"] = {
    "alpha_grid": [
        0.0,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        1.0,
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

scheme11["repeat_3mer_features"]["aa_manifest"] = (
    "TRB/set/training_bundles/"
    "trb_repeat100_training_inputs_v1/"
    "aa_clone_table_manifest.csv"
)
scheme11["repeat_3mer_features"]["max_kmers"] = 500
scheme11["repeat_3mer_features"]["top_k_values"] = [
    50,
    100,
    200,
    500,
]

expected_models = [
    "A1_core83",
    "A2_core83_weighted500",
    "A3_core83_unweighted500",
    "A4_core83_both500",
    "A5_core83_public",
    "A6_core83_both50",
    "A6_core83_both100",
    "A6_core83_both200",
]

observed_models = list(scheme11["models"])
if observed_models != expected_models:
    raise SystemExit(
        "Unexpected Scheme 004 model definitions.\n"
        f"Observed: {observed_models}\n"
        f"Expected: {expected_models}"
    )

scheme11_path = (
    config_root
    / "schemes"
    / "trb_scheme_011_a1_a6_repeat100_fullgrid.yaml"
)
write_yaml(scheme11_path, scheme11)

print("Configuration generation: PASS")
print(f"Original split specification: {split3_path}")
print(f"100-repeat split specification: {split100_path}")
print(f"100-repeat training specification: {training100_path}")
print(f"Scheme 011 specification: {scheme11_path}")
print("Models:")
for model in expected_models:
    print(f"  {model}")
PY

# ============================================================
# 2. 生成并冻结100次外层holdout
# ============================================================

if [[ ! -f \
  /data/users/chenhaisheng/RA-ILD/TRB/set/split_sets/ra_ild_repeat100_v1/SPLITS_FROZEN.json
]]
then
    echo "Preflight: generating 100 splits in memory..."

    /data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
      /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/generate_repeated_holdout_splits.py \
      --spec \
      /data/users/chenhaisheng/RA-ILD/TRB/set/configs/repeated_holdout_splits/ra_ild_repeat100_v1.yaml \
      --dry-run

    echo "Preflight split generation: PASS"
    echo "Writing and freezing 100 splits..."

    /data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
      /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/generate_repeated_holdout_splits.py \
      --spec \
      /data/users/chenhaisheng/RA-ILD/TRB/set/configs/repeated_holdout_splits/ra_ild_repeat100_v1.yaml \
      --replace-incomplete
else
    echo "Frozen 100-repeat split set already exists; skipping generation."
fi

# ============================================================
# 3. 准备100-repeat training bundle
# ============================================================

if [[ ! -f \
  /data/users/chenhaisheng/RA-ILD/TRB/set/training_bundles/trb_repeat100_training_inputs_v1/TRAINING_BUNDLE_FROZEN.json
]]
then
    /data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
      /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/prepare_repeated_holdout_training.py \
      --spec \
      /data/users/chenhaisheng/RA-ILD/TRB/set/configs/repeated_holdout_training/trb_repeat100_training_v1.yaml \
      --replace-incomplete
else
    echo "Frozen 100-repeat training bundle already exists; skipping preparation."
fi

# ============================================================
# 4. 准备100-repeat基础resolved config
# ============================================================

if [[ ! -f \
  /data/users/chenhaisheng/RA-ILD/TRB/set/experiments/trb_scheme_003_repeat100_no_clinical/00_config/resolved_config.yaml
]]
then
    /data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
      /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/prepare_analysis_scheme.py \
      --scheme \
      /data/users/chenhaisheng/RA-ILD/TRB/set/configs/schemes/generated/trb_scheme_003_repeat100_no_clinical.generated.yaml
else
    echo "100-repeat base resolved config already exists; skipping preparation."
fi

# ============================================================
# 5. 准备Scheme 011 resolved config
# ============================================================

if [[ ! -f \
  /data/users/chenhaisheng/RA-ILD/TRB/set/experiments/trb_scheme_011_a1_a6_repeat100_fullgrid/00_config/resolved_config.yaml
]]
then
    /data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
      /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/prepare_feature_ablation_scheme.py \
      --scheme \
      /data/users/chenhaisheng/RA-ILD/TRB/set/configs/schemes/trb_scheme_011_a1_a6_repeat100_fullgrid.yaml
else
    echo "Scheme 011 resolved config already exists; skipping preparation."
fi

# ============================================================
# 6. 严格校验配置、模型数、特征数和任务规模
# ============================================================

/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python - <<'PY'
from pathlib import Path

import yaml

path = Path(
    "/data/users/chenhaisheng/RA-ILD/"
    "TRB/set/experiments/"
    "trb_scheme_011_a1_a6_repeat100_fullgrid/"
    "00_config/resolved_config.yaml"
)

with path.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

expected_alpha = [
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    1.0,
]

expected_lambda = [
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
]

expected_counts = {
    "A1_core83": 83,
    "A2_core83_weighted500": 583,
    "A3_core83_unweighted500": 583,
    "A4_core83_both500": 1083,
    "A5_core83_public": 91,
    "A6_core83_both50": 183,
    "A6_core83_both100": 283,
    "A6_core83_both200": 483,
}

assert cfg["model_engine"]["alpha_grid"] == expected_alpha
assert cfg["model_engine"]["lambda_grid"] == expected_lambda
assert list(cfg["models"]) == list(expected_counts)

counts = cfg["model_selection"]["expected_resolved_columns"]

for model, expected_numeric in expected_counts.items():
    observed_numeric = counts[model]["numeric"]
    observed_categorical = counts[model]["categorical"]

    assert observed_numeric == expected_numeric, (
        model,
        observed_numeric,
        expected_numeric,
    )
    assert observed_categorical == 0, (
        model,
        observed_categorical,
    )

assert cfg["nested_cv"]["expected_outer_tasks"] == 100
assert cfg["nested_cv"]["expected_inner_fits_per_task"] == 2240
assert cfg["aggregation"]["expected_metric_rows"] == 800
assert cfg["aggregation"]["expected_prediction_rows"] == 40800

print("Scheme 011 validation: PASS")
print()
print("Feature counts:")
for model, count in expected_counts.items():
    print(f"  {model}: {count}")
print()
print("Outer repeats:              100")
print("Models:                     8")
print("Candidates per model:       56")
print("Inner folds:                5")
print("Inner fits per repeat:      2240")
print("Total planned inner fits:   224000")
print("Expected metric rows:       800")
print("Expected prediction rows:   40800")
PY

# ============================================================
# 7. 查看初始状态
# ============================================================

/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
  /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/run_all_outer_tasks.py \
  --config \
  /data/users/chenhaisheng/RA-ILD/TRB/set/experiments/trb_scheme_011_a1_a6_repeat100_fullgrid/00_config/resolved_config.yaml \
  --status-only

# ============================================================
# 8. 使用3个并行worker运行100个任务
# ============================================================

/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
  /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/run_all_outer_tasks.py \
  --config \
  /data/users/chenhaisheng/RA-ILD/TRB/set/experiments/trb_scheme_011_a1_a6_repeat100_fullgrid/00_config/resolved_config.yaml \
  --workers 3 \
  --execute

# ============================================================
# 9. 运行结束后再次检查任务状态
# ============================================================

/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
  /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/run_all_outer_tasks.py \
  --config \
  /data/users/chenhaisheng/RA-ILD/TRB/set/experiments/trb_scheme_011_a1_a6_repeat100_fullgrid/00_config/resolved_config.yaml \
  --status-only

# ============================================================
# 10. 自动汇总
# ============================================================

if [[ ! -f \
  /data/users/chenhaisheng/RA-ILD/TRB/set/experiments/trb_scheme_011_a1_a6_repeat100_fullgrid/03_summary/08_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json
]]
then
    /data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
      /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/aggregate_repeated_holdout_results.py \
      --config \
      /data/users/chenhaisheng/RA-ILD/TRB/set/experiments/trb_scheme_011_a1_a6_repeat100_fullgrid/00_config/resolved_config.yaml
else
    echo "Repeated-holdout aggregation already exists; skipping aggregation."
fi

# ============================================================
# 11. 生成30、50、75、100次累计排名
# ============================================================

/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python - <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd

summary_dir = Path(
    "/data/users/chenhaisheng/RA-ILD/"
    "TRB/set/experiments/"
    "trb_scheme_011_a1_a6_repeat100_fullgrid/"
    "03_summary"
)

input_path = summary_dir / "08_holdout_task_metrics.csv"
output_path = summary_dir / "08_cumulative_checkpoint_ranking.csv"

df = pd.read_csv(input_path)

required = {
    "outer_repeat",
    "model",
    "roc_auc",
    "pr_auc",
    "sensitivity_recall",
    "specificity",
    "f1",
}

missing = sorted(required - set(df.columns))
if missing:
    raise SystemExit(f"Missing metric columns: {missing}")

rows = []

for checkpoint in [30, 50, 75, 100]:
    subset = df.loc[df["outer_repeat"] <= checkpoint].copy()

    observed_repeats = subset["outer_repeat"].nunique()
    if observed_repeats != checkpoint:
        raise SystemExit(
            f"Checkpoint {checkpoint}: observed "
            f"{observed_repeats} repeats"
        )

    grouped = (
        subset.groupby("model", sort=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            roc_auc_median=("roc_auc", "median"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            pr_auc_median=("pr_auc", "median"),
            sensitivity_mean=("sensitivity_recall", "mean"),
            specificity_mean=("specificity", "mean"),
            f1_mean=("f1", "mean"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values(
        ["roc_auc_mean", "pr_auc_mean", "f1_mean", "model"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped.insert(
        0,
        "descriptive_rank",
        np.arange(1, len(grouped) + 1),
    )
    grouped.insert(0, "checkpoint_repeats", checkpoint)

    rows.append(grouped)

result = pd.concat(rows, ignore_index=True)
result.to_csv(output_path, index=False)

print("Cumulative checkpoint ranking: COMPLETE")
print(output_path)

for checkpoint in [30, 50, 75, 100]:
    print()
    print("=" * 72)
    print(f"Checkpoint: first {checkpoint} repeats")
    print(
        result.loc[
            result["checkpoint_repeats"] == checkpoint,
            [
                "descriptive_rank",
                "model",
                "roc_auc_mean",
                "roc_auc_std",
                "pr_auc_mean",
                "f1_mean",
            ],
        ].round(4).to_string(index=False)
    )
PY

echo "============================================================"
echo "Scheme 011 pipeline finished successfully."
echo "Finished: $(date '+%F %T')"
echo "Summary directory:"
echo "/data/users/chenhaisheng/RA-ILD/TRB/set/experiments/trb_scheme_011_a1_a6_repeat100_fullgrid/03_summary"
echo "============================================================"
