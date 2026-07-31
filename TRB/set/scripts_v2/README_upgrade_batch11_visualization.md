# TRB Upgrade Batch 11 — Configurable Repeated-Holdout Visualization

> **Hotfix 01 — version 1.0.1:** NumPy 2.4 and later removed the deprecated
> `numpy.trapz` alias. The mean-ROC area calculation now uses
> `numpy.trapezoid` with a fallback for older NumPy environments. A focused
> regression test is included.

## 1. Purpose

Batch 11 adds a **read-only, configuration-driven visualization layer** for
completed native repeated-holdout experiments. It does not rerun inner CV,
refit Elastic Net, rebuild public references, select a new threshold from
holdout labels, or modify any existing training result.

The module reads the aggregation files produced by Batch 05:

```text
<scheme root>/03_summary/
├── 08_holdout_task_metrics.csv
├── 08_holdout_predictions.csv.gz
├── 08_aggregation_summary.json
└── 08_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json
```

The latest reviewed TRB branch is:

```text
feature/trb-scheme-upgrade-batch01
```

## 2. Implemented figures

### Figure 1 — Model performance boxplots

One PDF contains one subplot per selected metric. For example:

- Figure 1A: ROC-AUC distribution across repeats;
- Figure 1B: PR-AUC (Average Precision) distribution;
- Figure 1C: F1 distribution;
- Figure 1D: Sensitivity distribution;
- Figure 1E: Specificity distribution.

Within each subplot, the x-axis contains the user-selected models and each
boxplot contains that model's repeat-level values. Repeat points and mean
markers are optional.

### Figure 2 — Mean ROC curves

Each selected model receives one panel. The module:

1. computes one ROC curve per native holdout repeat;
2. interpolates TPR onto a common FPR grid;
3. macro-averages TPR across repeats;
4. optionally draws an across-repeat percentile variability band;
5. reports mean repeat ROC-AUC ± SD.

The shaded band is an across-repeat variability summary, **not** an
independent-sample confidence interval, because repeated holdouts can share
patients.

### Figure 3 — Best observed repeat ROC curves

Each selected model receives one panel from its deterministically selected best
observed native holdout. The default ordering is:

```text
ROC-AUC descending
→ PR-AUC descending
→ F1 descending
→ outer_repeat ascending
```

The saved threshold operating point is optionally marked. This figure is
explicitly descriptive and does not replace Figure 1 or Figure 2 as the model's
overall performance evidence.

### Figure 4 — Best observed repeat prediction boxplots

Each selected model receives one panel containing:

- RA patient predicted ILD probabilities;
- ILD patient predicted ILD probabilities;
- individual patient points;
- the saved inner-training-derived threshold as a horizontal dashed line.

Figure 4 can reuse the exact Figure 3 best-repeat selection, ensuring both
figures refer to the same model instance and holdout.

## 3. Multi-panel layout

All four figure types support a maximum of three columns:

```text
1 panel  → one large panel
2 panels → one row with two equal panels
3 panels → one row with three panels
4 panels → 3 + 1, final panel centered
5 panels → 3 + 2, final two panels equally divided
6 panels → 3 + 3
```

PDF output remains vector based.

## 4. Installed files

```text
TRB/set/src/ra_ild_trb/repeated_holdout_visualization.py
TRB/set/scripts_v2/plot_repeated_holdout_results.py
TRB/set/scripts_v2/test_batch11_repeated_holdout_visualization.py
TRB/set/scripts_v2/README_upgrade_batch11_visualization.md
TRB/set/scripts_v2/BATCH11_REPEATED_HOLDOUT_VISUALIZATION_MANIFEST.json
TRB/set/configs/visualization/trb_repeated_holdout_visualization_example.yaml
```

All files are additive. No existing source file is replaced.

## 5. Environment dependencies

The module uses packages already expected in the Python modeling environment:

```text
numpy
pandas
matplotlib
scikit-learn
pyyaml
```

Check them with:

```bash
conda activate ra-ild
python3 - <<'PY'
import numpy, pandas, matplotlib, sklearn, yaml
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("matplotlib", matplotlib.__version__)
print("scikit-learn", sklearn.__version__)
print("pyyaml", yaml.__version__)
PY
```

If only plotting dependencies are missing:

```bash
mamba install -n ra-ild -c conda-forge matplotlib pyyaml
```

## 6. Focused validation

From the repository root:

```bash
conda activate ra-ild
cd /data/users/chenhaisheng/RA-ILD

python3 \
  TRB/set/scripts_v2/test_batch11_repeated_holdout_visualization.py
```

Expected:

```text
PASS test_subplot_layout_policy
PASS test_all_four_figures_and_audits
PASS test_best_repeat_tie_breaker_is_deterministic
PASS test_models_all_selector
PASS test_unknown_model_is_rejected
PASS test_incomplete_aggregation_is_rejected
PASS test_existing_output_requires_explicit_overwrite
Batch 11 focused tests: 8/8 PASS
```

The synthetic test writes all four PDFs, verifies their PDF signatures, checks
plot-data row counts, confirms Figure 3/Figure 4 selection reuse, and exercises
input-safety failures.

## 7. Prepare a visualization YAML

Copy the example rather than editing it in place:

```bash
cp \
  TRB/set/configs/visualization/trb_repeated_holdout_visualization_example.yaml \
  TRB/set/configs/visualization/my_scheme011_visualization.yaml
```

Edit these fields:

```yaml
visualization:
  id: my_scheme011_visualization
  aggregation_dir: TRB/set/experiments/<scheme_id>/03_summary
  output_dir: TRB/set/experiments/<scheme_id>/06_visualization/my_scheme011_visualization
```

Then select models and metrics separately under each figure. Use `models: all`
to include every model in the aggregation result, or provide an explicit model
list. Explicit model names must exactly match the `model` column in:

```text
<aggregation_dir>/08_holdout_task_metrics.csv
```

A quick way to list them is:

```bash
python3 - <<'PY'
import pandas as pd
path = "TRB/set/experiments/<scheme_id>/03_summary/08_holdout_task_metrics.csv"
print("\n".join(pd.read_csv(path)["model"].drop_duplicates().astype(str)))
PY
```

## 8. Run visualization

```bash
conda activate ra-ild
cd /data/users/chenhaisheng/RA-ILD

python3 \
  TRB/set/scripts_v2/plot_repeated_holdout_results.py \
  --config \
  TRB/set/configs/visualization/my_scheme011_visualization.yaml
```

The command refuses to overwrite existing output. A documented intentional
rerun can use:

```bash
python3 \
  TRB/set/scripts_v2/plot_repeated_holdout_results.py \
  --config \
  TRB/set/configs/visualization/my_scheme011_visualization.yaml \
  --overwrite
```

A safer routine is to use a new `visualization.id` and output directory for a
new model/figure selection instead of overwriting an earlier result.

## 9. Output structure

For all four enabled figures:

```text
<output_dir>/
├── Figure1_model_performance_boxplots.pdf
├── Figure1_plot_data.csv
├── Figure2_mean_ROC_curves.pdf
├── Figure2_mean_ROC_coordinates.csv.gz
├── Figure2_repeat_ROC_AUC.csv
├── Figure3_best_repeat_ROC_curves.pdf
├── Figure3_best_repeat_selection.csv
├── Figure3_best_repeat_ROC_coordinates.csv.gz
├── Figure4_best_repeat_prediction_boxplots.pdf
├── Figure4_plot_data.csv
├── visualization_input_audit.csv
├── visualization_resolved_config.yaml
├── visualization_manifest.json
└── VISUALIZATION_COMPLETE.json
```

`VISUALIZATION_COMPLETE.json` is written last and contains hashes for the prior
outputs. `visualization_input_audit.csv` independently recalculates ROC-AUC,
PR-AUC, F1, sensitivity and specificity from the saved patient predictions and
requires agreement with the saved metric table before plotting.

## 10. Scientific and safety rules

- Only native repeated-holdout files are accepted.
- Batch 06 `n × n` cross-repeat files are not read by this module.
- A COMPLETE aggregation marker is required by default.
- Every model/split prediction group must contain both RA and ILD.
- Saved probabilities and thresholds must remain within `[0, 1]`.
- Saved metrics must reproduce from saved predictions within `1e-6`.
- Unknown model names stop the run.
- Figure 2 averages complete per-repeat ROC curves rather than pooling duplicate
  patient rows across repeats.
- Figure 3 and Figure 4 use saved thresholds; no threshold is optimized on the
  displayed holdout.
- Best-repeat figures are descriptive extreme observations.

## 11. Git workflow suggestion

Suggested branch:

```text
feature/trb-configurable-visualization
```

Suggested commit:

```text
Add configurable repeated-holdout visualization module
```

Suggested PR title:

```text
Add configurable repeated-holdout visualization module
```

Suggested PR summary:

```text
- add YAML-driven PDF generation for repeat-level metric boxplots
- add macro-averaged ROC curves with across-repeat variability bands
- add deterministic best-repeat ROC and probability boxplots
- add input reproduction audits, output manifests and focused synthetic tests
- keep training, threshold selection and cross-repeat evaluation unchanged
```
