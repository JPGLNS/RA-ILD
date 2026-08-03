# IGH Batch 10 Hotfix 01 — Matplotlib 3.11 boxplot compatibility

## Problem

Matplotlib 3.9 renamed `Axes.boxplot(labels=...)` to
`Axes.boxplot(tick_labels=...)`. Matplotlib 3.11 removed the old `labels`
keyword, causing Batch 10 acceptance tests to fail with:

```text
TypeError: Axes.boxplot() got an unexpected keyword argument 'labels'
```

## Fix

The visualization module now inspects the active Matplotlib `Axes.boxplot`
signature and uses:

- `tick_labels` on Matplotlib 3.9+ and 3.11+;
- `labels` only on older environments that do not expose `tick_labels`.

The fix is limited to plotting API compatibility. It does not change metrics,
model selection, repeated-holdout handling, thresholds, predictions, or any
training output.

## Validation

Run:

```bash
python3 IGH/set/scripts_v2/test_batch10_repeated_holdout_visualization.py
```

Expected ending:

```text
IGH Batch 10 focused tests: 4/4 PASS
IGH_BATCH10_ACCEPTANCE_PASS
```
