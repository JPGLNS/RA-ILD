# TRB V3 — TCR-only clinical ablation result record

Recorded: 2026-11-22. Branch: `feature/trb-v3-179-tcr-only-ablation` (to be fast-forward integrated into `feature/trb-v3-179-layer1-layer2`).

## Objective and experimental contract

To assess the predictive ability of the *already-selected* best TCR feature groups without the age/sex covariates, train three independent Elastic Net models using exactly the same 100 frozen repeated outer holdout splits and inner five-fold CV as the corresponding Layer-1 models. Hyperparameters are **retuned independently within each repeat**, not copied from the full models. Top10000 TCR input, static feature definitions and samples remain unchanged.

| Cohort | N | Full model | Ablation model | Layer-1 combination |
|---|---:|---|---|---|
| total | 179 | Age + Sex + PW | PW | L1_007_PW |
| pbmc | 116 | Age + Sex + D | D | L1_001_D |
| buffycoat | 63 | Age + Sex + E + PU | E + PU | L1_017_E_PU |

D=diversity; E=clonal expansion; PU=unweighted physicochemical; PW=weighted physicochemical.

## Final reported results

Means across 100 repeated holdout evaluations. The M0 age+sex-only results come from the earlier *separate* conventional Elastic Net experiment, so M0 comparisons are descriptive, rather than asserted to have the same held-out sample identity as this pairwise ablation.

| Cohort | M0 ROC-AUC | Age+Sex+TCR ROC-AUC | TCR-only ROC-AUC | M0 PR-AUC | Age+Sex+TCR PR-AUC | TCR-only PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| total | 0.7250 | 0.7201 | 0.4728 | 0.6303 | 0.6211 | 0.3959 |
| pbmc | 0.6779 | 0.6565 | 0.4676 | 0.6219 | 0.5853 | 0.4182 |
| buffycoat | 0.7921 | 0.8656 | 0.8571 | 0.7211 | 0.8060 | 0.8003 |

Paired `TCR-only - Age+Sex+TCR` changes from the summary, with matched holdout sample IDs and labels in every repeat:

| Cohort | Delta ROC-AUC | Fraction repeats with ROC gain | Delta PR-AUC | Fraction repeats with PR gain |
|---|---:|---:|---:|---:|
| total | -0.2473 | 0.00 | -0.2252 | 0.00 |
| pbmc | -0.1888 | 0.00 | -0.1671 | 0.00 |
| buffycoat | -0.0085 | 0.27 | -0.0057 | 0.33 |

**Run/quality status:** The user-supplied neo3 output confirms 3/3 Repeat100 training jobs exited successfully and completed, all 300 evaluation repeats match corresponding full-model holdout sample IDs and class labels, actual preprocessing excludes age/sex/material predictors, and both summary CSVs were written. Repository-side config audit and two static tests passed; six regression tests for the corrected summary-manifest `resolved_experiment.phase5.features` path passed on neo3. This record is based on those execution logs and summary values, not a Git commit of raw prediction outputs.

## Interpretation and limitations

- Total (PW) and PBMC (D) TCR-only models have mean ROC-AUC below 0.50; these feature sets did not show useful independent discriminative performance in the current evaluation. A separate review of prediction-score direction and per-repeat distributions is advisable before publication; this has **not** been documented as complete.
- Buffycoat retains a high TCR-only ROC-AUC (~0.8571); removing age+sex results in only a small decrease in mean ROC-AUC, but this does **not** establish independence from age-related biology, blood-material cell composition, or other confounders.
- Best TCR groups were selected from the same study's repeated evaluations. Performance and comparisons are exploratory; selected-best ROC-AUC may be optimistic.
- Repeated holdouts share patients and are not 100 independent biological validation cohorts.

## Runtime source files on neo3 (excluded from Git)

Relative to `/data/users/chenhaisheng/RA-ILD`:

- TCR-only outputs: `TRB/result/feature_framework/layered_feature_experiments/tcr_only_ablation/{total,pbmc,buffycoat}/`
- Clinical+TCR paired Layer-1 outputs: `TRB/result/feature_framework/layered_feature_experiments/layer1/{total,pbmc,buffycoat}/elastic_net/{L1_007_PW,L1_001_D,L1_017_E_PU}/` respectively.
- Summary: `TRB/result/feature_framework/layered_feature_experiments/tcr_only_ablation_summary/TCR_ONLY_VS_AGE_SEX_TCR_SUMMARY.csv`
- Paired repeated metrics: `TRB/result/feature_framework/layered_feature_experiments/tcr_only_ablation_summary/TCR_ONLY_VS_AGE_SEX_TCR_PAIRED_REPEATS.csv`
- Batch status: `TRB/result/feature_framework/layered_feature_experiments/tcr_only_ablation/batch_status.csv`

These result files are intentionally not added to Git. Preserve the neo3 storage separately when delivering the study.
