# IGH Upgrade Batch 12 — Probability Metrics and Historical Backfill

Batch 12 adds threshold-independent `log_loss` and `brier_score` (both lower is
better). Future outer tasks write them natively. Existing completed tasks can be
updated from their saved `probability_ILD` values without model refitting.

## Acceptance

```bash
python3 IGH/set/scripts_v2/test_batch12_probability_metrics.py
```

Expected ending:

```text
IGH Batch 12 focused tests: 6/6 PASS
IGH_BATCH12_ACCEPTANCE_PASS
```

## Full-cohort historical backfill

First verify without writing:

```bash
python3 IGH/set/scripts_v2/backfill_probability_metrics.py   --config IGH/set/experiments/igh_scheme_006_a1_a6_repeat100_fullgrid/00_config/resolved_config.yaml   --repository-root /data/users/chenhaisheng/RA-ILD   --dry-run
```

Then write:

```bash
python3 IGH/set/scripts_v2/backfill_probability_metrics.py   --config IGH/set/experiments/igh_scheme_006_a1_a6_repeat100_fullgrid/00_config/resolved_config.yaml   --repository-root /data/users/chenhaisheng/RA-ILD
```

Each original task metric CSV is backed up as
`06_outer_validation_metrics.before_batch12.csv`. The task-root audit is
`09_probability_metric_backfill_audit.csv` and records SHA256 values before and
after the update.

Regenerate the existing `07_*` aggregation:

```bash
python3 IGH/set/scripts_v2/aggregate_repeated_holdout_results.py   --config IGH/set/experiments/igh_scheme_006_a1_a6_repeat100_fullgrid/00_config/resolved_config.yaml   --repository-root /data/users/chenhaisheng/RA-ILD   --overwrite
```

Do not commit task backups, backfill audits, aggregation outputs,
`BATCH12_INSTALLATION.json`, `*.bak.batch12.*`, ZIP files, or extracted package
directories.
