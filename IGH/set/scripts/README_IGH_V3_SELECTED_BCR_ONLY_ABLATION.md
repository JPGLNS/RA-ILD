# IGH V3 — selected BCR-only ablation (100 repeated holdouts)

**Branch:** \`feature/igh-v3-hierarchical-5alg\`

Goal: remove **Age and Sex entirely** while keeping the selected static BCR feature
group unchanged. Reuse the **same three frozen 100-repeat banks**, Elastic Net
(24 alpha/lambda candidates, five inner folds), Top10000 source and test protocol.

| Cohort | Original selected joint model | BCR-only group | Canonical no-clinical config |
| --- | --- | --- | --- |
| Total | Age+Sex+PW | PW = \`tcr_physicochemical_weighted\` | \`igh_layer1_total_top10000_v3_175/S007_PW.yaml\` |
| PBMC | Age+Sex+E | E = \`tcr_clonal_expansion\` | \`igh_layer1_pbmc_top10000_v3_175/S002_E.yaml\` |
| Buffycoat | Age+Sex+E | E = \`tcr_clonal_expansion\` | \`igh_layer1_buffycoat_top10000_v3_175/S002_E.yaml\` |

All three canonical YAMLs have \`features.clinical_groups: []\`,
\`kmer.enabled: false\`, \`enriched_dictionary.enabled: false\` and
\`public_reference.enabled: false\` (or omitted, default false).
**No modification to the frozen five-algorithm M0–M3 experiment.**

## Read-only checks

From repo root on **neo3** in the \`ra-ild\` environment:

\`\`\`bash
cd /data/users/chenhaisheng/RA-ILD
git switch feature/igh-v3-hierarchical-5alg
git pull --ff-only origin feature/igh-v3-hierarchical-5alg
python -m py_compile IGH/set/scripts/igh_v3_selected_bcr_only_ablation.py
PY=IGH/set/scripts/igh_v3_selected_bcr_only_ablation.py

python "$PY" --mode plan
python "$PY" --mode audit-existing
\`\`\`

The earlier **IGH-only Layer1** run may already be the *exact* requested
BCR-only ablation. \`audit-existing\` validates 100 outer metrics, 100
selected candidates, 2400 tuning rows, true class labels and every frozen
holdout sample ID. If this passes, **do not retrain unnecessarily**.

## Collect verified existing 100-repeat results

\`\`\`bash
python "$PY" --mode collect-existing
\`\`\`

Default delivery root (outside the Git repo):
\`/data/users/chenhaisheng/igh_v3_selected_bcr_only_ablation_v3_175/summary_existing/\`

- \`01_metrics_900.csv\`: 3 cohorts × (BCR-only / Age+Sex / Age+Sex+BCR) × 100 repeats
- \`02_model_summary_9.csv\`: means, SD, medians, quartiles for 6 metrics
- \`03_paired_deltas_18.csv\`: 3 comparisons × ROC-AUC/AP × 3 cohorts
- \`COLLECTION_MANIFEST.json\`: config/result paths, SHA256 provenance

The **source Layer1 raw predictions** remain at
\`IGH/set/data_versions/igh_v3_175/runs/layer1_<scope>_top10000/<stem>/repeat100/\`.
The two clinical comparison models are from
\`runs/clinical_layer1_<scope>_top10000/S000_none/repeat100\` and
\`runs/clinical_layer1_<scope>_top10000/<stem>/repeat100\`.

## Optional fresh independent rerun

If existing outputs are missing, or a separate new run is explicitly needed,
the exact same canonical BCR-only YAMLs can be run with the Phase-6
**per-repeat checkpoint** engine. No output is overwritten.

Start with Repeat001:

\`\`\`bash
python "$PY" --mode run --start 1 --end 1 --execute
python "$PY" --mode status --start 1 --end 1
\`\`\`

Then run all 100. \`run\` is resumable and skips verified checkpoints:

\`\`\`bash
nohup python -u "$PY" --mode run --start 1 --end 100 --execute \
  > /data/users/chenhaisheng/igh_v3_selected_bcr_only_ablation_v3_175/nohup.log 2>&1 < /dev/null &
\`\`\`

**Create the output directory first** if using the nohup redirection above:

\`\`\`bash
mkdir -p /data/users/chenhaisheng/igh_v3_selected_bcr_only_ablation_v3_175
\`\`\`

Then verify and collect:

\`\`\`bash
python "$PY" --mode status --start 1 --end 100
python "$PY" --mode collect-new
\`\`\`

The fresh rerun uses sequential execution across the three configurations
and the checkpoint engine for per-repeat atomic output. To run different
cohorts in parallel, invoke separate \`--scope\` commands in distinct
processes; their output directories are isolated. Do not start duplicate
processes for the same scope.

## Interpretation

The top PW/E feature groups were selected using repeated holdout evaluations
on the same patients. These ablations are **post-selection/exploratory**
rather than an independent estimate of an untouched model-selection protocol.
The 100 repeated holdouts overlap and are not independent patient cohorts.

For any published comparison, label ROC-AUC and Average Precision (AP)
clearly, and report the *paired* differences using identical frozen repeat
indices. Do not interpret repeated-holdout quartiles as independent-sample
95% confidence intervals.
