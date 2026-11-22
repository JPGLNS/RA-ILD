# TRB V3 TCR-only clinical ablation (three cohorts)

This follow-up is isolated on `feature/trb-v3-179-tcr-only-ablation`, based on frozen commit `606e8db`.

| Cohort | Existing best age+sex+TCR | New independent TCR-only |
|---|---|---|
| total (179) | Age + Sex + PW | PW |
| pbmc (116) | Age + Sex + D | D |
| buffycoat (63) | Age + Sex + E + PU | E + PU |

Only `experiment.id` and `features.clinical_groups` differ from the parent Layer-1 YAML. The independent TCR-only model retunes Elastic Net with inner 5-fold CV in every repeat (same search space, **not** copied hyperparameters). Both use the identical frozen 100-repeat outer split bank. 3-mer and RA-hit remain OFF. Age/sex may remain in source metadata for split construction but must never enter the predictor matrix.

## Commands (neo3, within ra-ild conda environment)

```bash
cd /data/users/chenhaisheng/RA-ILD
git fetch origin
git switch --track origin/feature/trb-v3-179-tcr-only-ablation
python3 -m py_compile TRB/set/scripts_v2/{audit,run,summarize}_tcr_only_ablation.py
python3 TRB/set/scripts_v2/audit_tcr_only_ablation.py
python3 -m unittest TRB/set/tests/test_tcr_only_ablation.py -v
python3 TRB/set/scripts_v2/run_tcr_only_ablation.py --repeat 1
python3 TRB/set/scripts_v2/run_tcr_only_ablation.py --repeat 1 --max-experiments 3 --execute
```

Inspect logs, smoke metric coverage and `06_run_manifest.json` before running Repeat100. Preflight:

```bash
python3 TRB/set/scripts_v2/run_tcr_only_ablation.py --max-experiments 3
```

For background full100, preserve CPU/memory quotas:

```bash
REPO=/data/users/chenhaisheng/RA-ILD
PYTHON=$(command -v python3)
systemd-run --user --unit=trb-tcr-only-repeat100 \
  -p CPUQuota=5000% -p MemoryHigh=100G -p MemoryMax=120G \
  --working-directory="$REPO" \
  -- "$PYTHON" "$REPO/TRB/set/scripts_v2/run_tcr_only_ablation.py" \
  --max-experiments 3 --execute
```

No overwrite option is provided. Each output is isolated at
`TRB/result/feature_framework/layered_feature_experiments/tcr_only_ablation/<cohort>/`.
Repeat001 smoke results stay in `smoke_repeat_001/` within that cohort.

When finished:

```bash
python3 TRB/set/scripts_v2/summarize_tcr_only_ablation.py
python3 TRB/set/scripts_v2/summarize_tcr_only_ablation.py --apply
```

The summary checks complete Repeat100 outputs, matching holdout sample IDs and labels for every repeat, and the absence of age/sex/material predictor columns. CSVs go to `.../layered_feature_experiments/tcr_only_ablation_summary/`.

**Interpretation**: Report paired differences as TCR-only minus Age+Sex+TCR. Previous features were chosen on the same dataset; inference is exploratory. Existing Layer-1, Layer-2 and five-algorithm files are untouched.
