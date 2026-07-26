# IGH Scheme Upgrade Batch 03

## Title

Configurable, metadata-only repeated-holdout split generation and freezing.

## Scope

Batch 03 adds a patient-level split-set layer. It does not fit models, inspect
repertoire features, read independent-test performance, or select a scientific
winner.

New capabilities:

1. Configure repeat count, train/holdout sizes, exact strata, balancing variables,
   candidate count, random seed, diversity penalty, and overlap limit in YAML.
2. Reconstruct the complete IGH cohort from historical metadata files while
   ignoring their old train/test roles.
3. Generate each repeat from thousands of metadata-only candidate assignments.
4. Preserve exact `cohort × batch` stratification for all splittable strata.
5. Handle the audited IGH batch6 singleton through the explicit
   `singleton_strata_policy: fixed_train` contract.
6. Require the expected singleton sample ID so no patient is silently fixed.
7. Record assignment reasons, balance audits, overlap, holdout frequency, and the
   singleton audit.
8. Freeze the selected assignment set with SHA256 hashes and prohibit overwrite.
9. Validate frozen files and detect later tampering.

## Current development specification

`IGH/set/configs/split_sets/igh_ra_ild_repeat3_v1.yaml` demonstrates:

- complete cohort: 169 patients;
- repeats: 3;
- each split: 120 train and 49 holdout;
- exact strata: cohort + batch;
- categorical balance: material + sex;
- numeric balance: age;
- batch6 singleton `MGI260202N01-31`: fixed in training for all repeats;
- therefore batch6 is never represented in holdout, and repeated-holdout metrics do
  not directly validate generalization to batch6;
- 5,000 candidate assignments per repeat;
- maximum pairwise holdout Jaccard: 0.35.

These are YAML values, not Python constants. A different number of repeats or a
different train/holdout size must use a new split-set ID and produce a separately
frozen assignment set.

## Scientific safeguards

Candidate ranking may use only the configured metadata variables. It must not use:

- immune-repertoire features;
- public-reference features;
- model predictions or coefficients;
- ROC-AUC, PR-AUC, or any downstream metric;
- historical independent-test performance.

A frozen split set cannot be overwritten. Changing any split parameter requires a
new split-set ID.

## Commands

In-memory generation and validation:

```bash
python3 IGH/set/scripts_v2/generate_repeated_holdout_splits.py \
  --spec IGH/set/configs/split_sets/igh_ra_ild_repeat3_v1.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --dry-run
```

Freeze after reviewing the dry-run:

```bash
python3 IGH/set/scripts_v2/generate_repeated_holdout_splits.py \
  --spec IGH/set/configs/split_sets/igh_ra_ild_repeat3_v1.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD
```

Validate the frozen outputs and hashes:

```bash
python3 IGH/set/scripts_v2/validate_repeated_holdout_split_set.py \
  --spec IGH/set/configs/split_sets/igh_ra_ild_repeat3_v1.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --frozen
```

Generated split resources are stored under `IGH/set/split_sets/` and remain outside
Git. The specification, implementation, tests, and documentation are committed.
