# TRB V3 179 IGH-aligned hierarchy

## Status

Development branch: `feature/trb-v3-179-igh-aligned-hierarchy`

Base: `feature/trb-v3-179-five-algorithm-adapter`

This branch is an additive development line. Existing 179-sample resolved configurations, repeat-001 outputs, frozen training bundles, split assignments, manifests, and prior experiment roots are read-only and must not be overwritten.

## Frozen scientific model contract

All models predict ILD (positive) versus RA (negative).

| Cohort | Model | Predictors |
|---|---|---|
| Total | M0 | Age + Sex |
| Total | M1 | Age + Sex + Core83 + unweighted Top-500 3-mers + weighted Top-500 3-mers |
| Total | M2 | M1 + eight raw_bilateral public-reference features |
| Total | M3 | M2 + Material |
| PBMC | M0 | Age + Sex |
| PBMC | M1 | Age + Sex + Core83 + unweighted Top-500 3-mers + weighted Top-500 3-mers |
| PBMC | M2 | M1 + eight raw_bilateral public-reference features |
| Buffycoat | M0 | Age + Sex |
| Buffycoat | M1 | Age + Sex + Core83 + unweighted Top-500 3-mers + weighted Top-500 3-mers |
| Buffycoat | M2 | M1 + eight raw_bilateral public-reference features |

PBMC and Buffycoat do not receive M3 because Material is constant within each material-homogeneous cohort.

The eight public-reference predictors are:
- all_ref_public_clone_ratio
- all_ref_public_frequency_sum
- shared_ref_clone_ratio
- shared_ref_frequency_sum
- RA_specific_ref_clone_ratio
- RA_specific_ref_frequency_sum
- ILD_specific_ref_clone_ratio
- ILD_specific_ref_frequency_sum

## Feature and validation contract

- Repertoire source: exactly Top10000 AA clonotypes per retained sample.
- Core static repertoire features: Core83, preserving the existing TRB feature definitions and drop rules.
- 3-mer: shared deterministic ranking for weighted and unweighted representations; Top-500 maximum; min sample count 5; min prevalence 0.05; variance thresholds 1e-12; ranking by prevalence descending, total weighted+unweighted variance descending, then k-mer lexical order.
- 3-mer vocabulary selection must be performed separately inside each inner-training fold. The matching inner-validation samples only receive that fold's selected vocabulary. For final outer evaluation, refit the vocabulary on the complete outer-training partition and apply it to the outer holdout. No inner-validation or outer-holdout sample may contribute to vocabulary selection.
- Public-reference features retain the existing TRB/IGH raw_bilateral thresholds and exact leave-one-out construction for model-fitting samples; validation/holdout features use the current training partition's reference only.
- Enriched dictionary, QC/depth, and other exploratory feature groups are excluded from the primary hierarchy.
- Five algorithms and 100 frozen outer repeated holdouts remain the intended comparison design. Existing algorithm-specific hyperparameter grids and tuning rules are not changed by this hierarchy alignment.

## Required implementation and audit gates

1. Build isolated new experiment configurations; never edit existing resolved configs or frozen bundle inputs.
2. Update shared fold-feature construction so each inner fold has its own 3-mer vocabulary and feature matrix, reused across all five algorithms.
3. Keep the final outer-training 3-mer vocabulary separate from inner-fold vocabularies.
4. Audit exact resolved predictor names/counts for every model and cohort.
5. Add regression tests proving: M0 contains clinical predictors only; M1 has no public features; M2 adds exactly eight public features; M3 adds only Material; inner-validation-only k-mers never enter the fold vocabulary; outer-holdout-only k-mers never enter the outer vocabulary; weighted/unweighted vocabularies are identical.
6. Run syntax checks, unit/regression tests, scheme dry-runs, and a repeat-001 smoke test in new output roots before any full launch.

No model training is launched by this specification document.
