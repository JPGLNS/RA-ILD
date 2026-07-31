#!/usr/bin/env bash
set -Eeuo pipefail

cd /data/users/chenhaisheng/RA-ILD

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "TRB Scheme 009: 30-repeat parameter tuning"
echo "Started: $(date '+%F %T')"
echo "Repository: /data/users/chenhaisheng/RA-ILD"
echo "============================================================"

# ------------------------------------------------------------------
# 1. 根据现有repeat3配置，生成独立的repeat30配置文件。
#    不覆盖Scheme 008和原3-repeat结果。
# ------------------------------------------------------------------

/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python - <<'PY'
from copy import deepcopy
from pathlib import Path

import yaml

root = Path("/data/users/chenhaisheng/RA-ILD")
config_root = root / "TRB/set/configs"


def read_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


# 查找原始3-repeat split specification。
split_matches = []

for path in config_root.rglob("*.yaml"):
    try:
        data = read_yaml(path)
    except Exception:
        continue

    if (
        isinstance(data, dict)
        and isinstance(data.get("split_set"), dict)
        and data["split_set"].get("id") == "ra_ild_repeat3_v1"
    ):
        split_matches.append((path, data))

if len(split_matches) != 1:
    found = [str(path) for path, _ in split_matches]
    raise SystemExit(
        "Expected exactly one split YAML with "
        "split_set.id=ra_ild_repeat3_v1; "
        f"found {len(split_matches)}: {found}"
    )

split_source_path, split_source = split_matches[0]
split30 = deepcopy(split_source)

split30["split_set"]["id"] = "ra_ild_repeat30_v1"
split30["split_set"]["description"] = (
    "Thirty deterministic metadata-balanced 123/51 repeated holdout splits."
)
split30["split_set"]["output_root"] = (
    "TRB/set/split_sets/ra_ild_repeat30_v1"
)
split30["split"]["repeats"] = 30

split30_path = (
    config_root
    / "repeated_holdout_splits"
    / "ra_ild_repeat30_v1.yaml"
)
write_yaml(split30_path, split30)

# 创建30-repeat training-bundle specification。
training3_path = (
    config_root
    / "repeated_holdout_training"
    / "trb_repeat3_training_v1.yaml"
)

if not training3_path.is_file():
    raise SystemExit(
        f"Original training-bundle specification not found: {training3_path}"
    )

training30 = deepcopy(read_yaml(training3_path))

training30["bundle"]["id"] = "trb_repeat30_training_inputs_v1"
training30["bundle"]["description"] = (
    "Full-cohort static and sequence inputs for thirty frozen "
    "123/51 repeated holdouts."
)
training30["bundle"]["output_root"] = (
    "TRB/set/training_bundles/trb_repeat30_training_inputs_v1"
)

training30["split_set"]["assignments"] = (
    "TRB/set/split_sets/ra_ild_repeat30_v1/"
    "repeated_holdout_assignments.csv"
)
training30["split_set"]["frozen_marker"] = (
    "TRB/set/split_sets/ra_ild_repeat30_v1/"
    "SPLITS_FROZEN.json"
)

training30["scheme"]["generated_scheme"] = (
    "TRB/set/configs/schemes/generated/"
    "trb_scheme_003_repeat30_no_clinical.generated.yaml"
)
training30["scheme"]["id"] = (
    "trb_scheme_003_repeat30_no_clinical"
)
training30["scheme"]["output_root"] = (
    "TRB/set/experiments/"
    "trb_scheme_003_repeat30_no_clinical"
)

training30_path = (
    config_root
    / "repeated_holdout_training"
    / "trb_repeat30_training_v1.yaml"
)
write_yaml(training30_path, training30)

# 从已验证的Scheme 008复制模型和参数网格，建立正式30-repeat Scheme 009。
scheme8_path = (
    config_root
    / "schemes"
    / "trb_scheme_008_parameter_tuning_repeat3.yaml"
)

if not scheme8_path.is_file():
    raise SystemExit(f"Scheme 008 not found: {scheme8_path}")

scheme9 = deepcopy(read_yaml(scheme8_path))

scheme9["scheme"]["id"] = (
    "trb_scheme_009_parameter_tuning_repeat30"
)
scheme9["scheme"]["description"] = (
    "Thirty frozen repeated holdouts comparing weighted Top-500 "
    "and both Top-50 feature combinations across Ridge, Elastic Net "
    "and Lasso parameter values."
)
scheme9["scheme"]["base_resolved_config"] = (
    "TRB/set/experiments/"
    "trb_scheme_003_repeat30_no_clinical/"
    "00_config/resolved_config.yaml"
)
scheme9["scheme"]["output_root"] = (
    "TRB/set/experiments/"
    "trb_scheme_009_parameter_tuning_repeat30"
)

scheme9["model_engine"]["alpha_grid"] = [
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    1.0,
]
scheme9["model_engine"]["lambda_grid"] = [
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
]

scheme9["repeat_3mer_features"]["aa_manifest"] = (
    "TRB/set/training_bundles/"
    "trb_repeat30_training_inputs_v1/"
    "aa_clone_table_manifest.csv"
)

scheme9_path = (
    config_root
    / "schemes"
    / "trb_scheme_009_parameter_tuning_repeat30.yaml"
)
write_yaml(scheme9_path, scheme9)

print("Configuration generation: PASS")
print(f"Original split specification: {split_source_path}")
print(f"30-repeat split specification: {split30_path}")
print(f"30-repeat training specification: {training30_path}")
print(f"30-repeat parameter scheme: {scheme9_path}")
PY

# ------------------------------------------------------------------
# 2. 生成并冻结30次外层holdout。
#    --replace-incomplete只允许替换未冻结的残缺目录。
# ------------------------------------------------------------------

if [[ ! -f \
  /data/users/chenhaisheng/RA-ILD/TRB/set/split_sets/ra_ild_repeat30_v1/SPLITS_FROZEN.json
]]
then
    /data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
      /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/generate_repeated_holdout_splits.py \
      --spec \
      /data/users/chenhaisheng/RA-ILD/TRB/set/configs/repeated_holdout_splits/ra_ild_repeat30_v1.yaml \
      --replace-incomplete
else
    echo "Frozen 30-repeat split set already exists; skipping generation."
fi

# ------------------------------------------------------------------
# 3. 构建30-repeat training bundle和5折inner-CV assignments。
# ------------------------------------------------------------------

if [[ ! -f \
  /data/users/chenhaisheng/RA-ILD/TRB/set/training_bundles/trb_repeat30_training_inputs_v1/TRAINING_BUNDLE_FROZEN.json
]]
then
    /data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
      /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/prepare_repeated_holdout_training.py \
      --spec \
      /data/users/chenhaisheng/RA-ILD/TRB/set/configs/repeated_holdout_training/trb_repeat30_training_v1.yaml \
      --replace-incomplete
else
    echo "Frozen 30-repeat training bundle already exists; skipping preparation."
fi

# ------------------------------------------------------------------
# 4. 生成30-repeat基础实验resolved config。
# ------------------------------------------------------------------

if [[ ! -f \
  /data/users/chenhaisheng/RA-ILD/TRB/set/experiments/trb_scheme_003_repeat30_no_clinical/00_config/resolved_config.yaml
]]
then
    /data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
      /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/prepare_analysis_scheme.py \
      --scheme \
      /data/users/chenhaisheng/RA-ILD/TRB/set/configs/schemes/generated/trb_scheme_003_repeat30_no_clinical.generated.yaml
else
    echo "30-repeat base resolved config already exists; skipping preparation."
fi

# ------------------------------------------------------------------
# 5. 生成正式Scheme 009 resolved config。
# ------------------------------------------------------------------

if [[ ! -f \
  /data/users/chenhaisheng/RA-ILD/TRB/set/experiments/trb_scheme_009_parameter_tuning_repeat30/00_config/resolved_config.yaml
]]
then
    /data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
      /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/prepare_feature_ablation_scheme.py \
      --scheme \
      /data/users/chenhaisheng/RA-ILD/TRB/set/configs/schemes/trb_scheme_009_parameter_tuning_repeat30.yaml
else
    echo "Scheme 009 resolved config already exists; skipping preparation."
fi

# ------------------------------------------------------------------
# 6. 运行前严格检查任务数、网格和预期输出规模。
# ------------------------------------------------------------------

/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python - <<'PY'
from pathlib import Path

import yaml

path = Path(
    "/data/users/chenhaisheng/RA-ILD/"
    "TRB/set/experiments/"
    "trb_scheme_009_parameter_tuning_repeat30/"
    "00_config/resolved_config.yaml"
)

with path.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

assert cfg["model_engine"]["alpha_grid"] == [
    0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0
]
assert cfg["model_engine"]["lambda_grid"] == [
    0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0
]
assert cfg["nested_cv"]["expected_outer_tasks"] == 30
assert cfg["nested_cv"]["expected_inner_fits_per_task"] == 560
assert cfg["aggregation"]["expected_metric_rows"] == 60
assert cfg["aggregation"]["expected_prediction_rows"] == 3060

counts = cfg["model_selection"]["expected_resolved_columns"]

assert counts["P1_core83_weighted500"]["numeric"] == 583
assert counts["P2_core83_both50"]["numeric"] == 183

print("Scheme 009 validation: PASS")
print("Outer tasks: 30")
print("Models: 2")
print("Inner fits per repeat: 560")
print("Total planned inner fits: 16800")
print("Expected metric rows: 60")
print("Expected prediction rows: 3060")
PY

# ------------------------------------------------------------------
# 7. 一次并行运行3个任务，共完成30个repeat。
#    已完成任务会由outer-task manager识别，不重复运行。
# ------------------------------------------------------------------

/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
  /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/run_all_outer_tasks.py \
  --config \
  /data/users/chenhaisheng/RA-ILD/TRB/set/experiments/trb_scheme_009_parameter_tuning_repeat30/00_config/resolved_config.yaml \
  --workers 3 \
  --execute

# ------------------------------------------------------------------
# 8. 检查状态并自动汇总。
# ------------------------------------------------------------------

/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
  /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/run_all_outer_tasks.py \
  --config \
  /data/users/chenhaisheng/RA-ILD/TRB/set/experiments/trb_scheme_009_parameter_tuning_repeat30/00_config/resolved_config.yaml \
  --status-only

/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python \
  /data/users/chenhaisheng/RA-ILD/TRB/set/scripts_v2/aggregate_repeated_holdout_results.py \
  --config \
  /data/users/chenhaisheng/RA-ILD/TRB/set/experiments/trb_scheme_009_parameter_tuning_repeat30/00_config/resolved_config.yaml

echo "============================================================"
echo "Scheme 009 pipeline finished successfully."
echo "Finished: $(date '+%F %T')"
echo "Summary:"
echo "/data/users/chenhaisheng/RA-ILD/TRB/set/experiments/trb_scheme_009_parameter_tuning_repeat30/03_summary"
echo "============================================================"
