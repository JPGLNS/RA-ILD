# TRB Batch 15 Hotfix 01 — Matplotlib boxplot compatibility

## Problem

The original Batch 15 v1.0.0 visualization used:

```python
Axes.boxplot(..., labels=...)
```

Matplotlib renamed this keyword to `tick_labels` in 3.9 and removed `labels`
in newer releases. On such environments the Batch 15 focused acceptance stopped
at Figure 1 before the installation marker was written.

## Fix

Hotfix 01 adds a small compatibility wrapper that:

1. uses the current `tick_labels` API first;
2. falls back to the historical `labels` API only when `tick_labels` is unavailable;
3. applies the wrapper to both Figure 1 and Figure 4 boxplots.

No modeling, nested-CV, candidate-selection, threshold, aggregation, feature,
material-subset, worker, or coefficient logic is changed.

## Acceptance

The installer runs:

- the focused current/legacy Matplotlib compatibility test;
- the two Linear SVM unit tests;
- the full Batch 15 synthetic acceptance test.

Expected final markers:

```text
TRB_BATCH15_HOTFIX01_MATPLOTLIB_ACCEPTANCE_PASS
TRB_BATCH15_LINEAR_SVM_ACCEPTANCE_PASS
TRB_BATCH15_HOTFIX01_INSTALL_PASS
```
