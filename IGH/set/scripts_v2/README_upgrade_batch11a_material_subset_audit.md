# IGH Upgrade Batch 11A — Material Composition Audit

## Purpose

Batch 11A is a read-only prerequisite for PBMC-only and buffercoat-only IGH
analyses. It validates the complete 169-patient metadata/matrix identity and
reports the actual material composition before Batch 11B freezes any filtered
source cohort.

It does **not**:

- write PBMC-only or buffercoat-only matrices;
- generate repeated-holdout splits;
- build public references or repeat-specific 3-mer vocabularies;
- fit or retune Elastic Net models;
- modify existing experiment outputs;
- read the independent-test result directory;
- select a final model automatically.

## Installed files

```text
IGH/set/src/ra_ild_igh/material_subset_audit.py
IGH/set/scripts_v2/audit_material_subsets.py
IGH/set/scripts_v2/test_batch11a_material_subset_audit.py
IGH/set/scripts_v2/README_upgrade_batch11a_material_subset_audit.md
IGH/set/scripts_v2/BATCH11A_MATERIAL_SUBSET_AUDIT_MANIFEST.json
IGH/set/configs/material_subsets/igh_material_subset_audit_example.yaml
```

## Focused validation

```bash
python3 IGH/set/scripts_v2/test_batch11a_material_subset_audit.py
```

Expected ending:

```text
IGH Batch 11A focused tests: 6/6 PASS
IGH_BATCH11A_ACCEPTANCE_PASS
```

## Real-data dry run

```bash
python3 IGH/set/scripts_v2/audit_material_subsets.py \
  --config IGH/set/configs/material_subsets/igh_material_subset_audit_example.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --dry-run
```

The dry run prints material counts, candidate exact-stratification feasibility,
and preliminary 70/30 train/holdout sizes. It writes nothing.

## Write the audit report

```bash
python3 IGH/set/scripts_v2/audit_material_subsets.py \
  --config IGH/set/configs/material_subsets/igh_material_subset_audit_example.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD
```

Output:

```text
IGH/set/material_subset_audits/igh_full_cohort_material_audit_v1/
├── sample_identity_audit.csv
├── material_summary.csv
├── material_cohort_counts.csv
├── material_cohort_batch_counts.csv
├── material_sex_counts.csv
├── candidate_exact_strata_audit.csv
├── material_split_recommendations.csv
├── audit_summary.json
├── audit_resolved_config.yaml
└── AUDIT_COMPLETE.json
```

Verify hashes:

```bash
python3 IGH/set/scripts_v2/audit_material_subsets.py \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --verify-marker \
  IGH/set/material_subset_audits/igh_full_cohort_material_audit_v1/AUDIT_COMPLETE.json
```

## Batch 11B input

Return or inspect these three files before Batch 11B is generated:

```text
material_summary.csv
material_cohort_batch_counts.csv
material_split_recommendations.csv
```

Batch 11B will use the observed IGH counts and feasible stratification policies;
it will not copy the TRB 111/63 sample counts.

## Git

Only source/config/test/documentation files are committed. The generated audit
folder remains a server-side runtime artifact.
