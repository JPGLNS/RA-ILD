# IGH Upgrade Batch 10 — Configurable Repeated-Holdout Visualization

## Purpose

Batch 10 adds a read-only, YAML-driven visualization layer for the dedicated
IGH repeated-holdout aggregation outputs written under `03_summary/`:

```text
07_all_outer_metrics.csv
07_all_outer_predictions.csv.gz
07_aggregation_summary.json
07_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json
```

It does not refit Elastic Net, retune alpha/lambda, rebuild public references,
change thresholds, read the cross-repeat n×n evaluation, or read the independent
test set.

## Figures

1. Multi-metric model-performance boxplots across repeats.
2. Per-model macro-averaged ROC curves, calculated by interpolating each native
   holdout ROC curve before averaging. The band is across-repeat variability,
   not an independent-sample confidence interval.
3. Deterministically selected best observed repeat ROC curves.
4. RA versus RA-ILD predicted-probability boxplots from the same selected repeat.

The best-repeat figures are descriptive extreme observations and are not used to
select a final model.

## Installed files

```text
IGH/set/src/ra_ild_igh/repeated_holdout_visualization.py
IGH/set/scripts_v2/plot_repeated_holdout_results.py
IGH/set/scripts_v2/test_batch10_repeated_holdout_visualization.py
IGH/set/scripts_v2/README_upgrade_batch10_visualization.md
IGH/set/scripts_v2/BATCH10_REPEATED_HOLDOUT_VISUALIZATION_MANIFEST.json
IGH/set/configs/visualization/igh_repeated_holdout_visualization_example.yaml
```

## Acceptance test

```bash
cd /data/users/chenhaisheng/RA-ILD
python3 IGH/set/scripts_v2/test_batch10_repeated_holdout_visualization.py
```

Expected final marker:

```text
IGH_BATCH10_ACCEPTANCE_PASS
```

## Configure and run

Copy the example rather than editing it directly:

```bash
cp \
  IGH/set/configs/visualization/igh_repeated_holdout_visualization_example.yaml \
  IGH/set/configs/visualization/igh_scheme006_visualization_v1.yaml
```

Edit `visualization.id`, `aggregation_dir`, `output_dir`, and the selected models.
Then run:

```bash
python3 \
  IGH/set/scripts_v2/plot_repeated_holdout_results.py \
  --config IGH/set/configs/visualization/igh_scheme006_visualization_v1.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD
```

A documented intentional rerun may add `--overwrite`. Prefer a new
`visualization.id` and output directory for a new figure selection.

## Outputs

```text
Figure1_model_performance_boxplots.pdf
Figure1_plot_data.csv
Figure2_mean_ROC_curves.pdf
Figure2_mean_ROC_coordinates.csv.gz
Figure2_repeat_ROC_AUC.csv
Figure3_best_repeat_ROC_curves.pdf
Figure3_best_repeat_selection.csv
Figure3_best_repeat_ROC_coordinates.csv.gz
Figure4_best_repeat_prediction_boxplots.pdf
Figure4_plot_data.csv
visualization_input_audit.csv
visualization_resolved_config.yaml
visualization_manifest.json
VISUALIZATION_COMPLETE.json
```

Before plotting, every saved metric that can be reconstructed from predictions
is recalculated and compared within the configured tolerance. Saved predicted
labels must also reproduce from the saved threshold.

## Git staging

Generated PDFs and visualization outputs remain under `experiments/` and should
not be committed. Stage only the six Batch 10 source/configuration files:

```bash
git add -- \
  IGH/set/src/ra_ild_igh/repeated_holdout_visualization.py \
  IGH/set/scripts_v2/plot_repeated_holdout_results.py \
  IGH/set/scripts_v2/test_batch10_repeated_holdout_visualization.py \
  IGH/set/scripts_v2/README_upgrade_batch10_visualization.md \
  IGH/set/scripts_v2/BATCH10_REPEATED_HOLDOUT_VISUALIZATION_MANIFEST.json \
  IGH/set/configs/visualization/igh_repeated_holdout_visualization_example.yaml
```

Suggested commit:

```text
Add configurable IGH repeated-holdout visualization
```
