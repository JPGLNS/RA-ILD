# RA/RA-ILD IGH Framework V2

**Batch 02 status: configurable models and additional feature-table management**

This directory contains the independent IGH namespace migrated from the validated
TRB framework. Batch 01 established the modular scientific engine and historical
IGH baseline compatibility. Batch 02 completes the scheme-level model and feature
input management layer.

## Installed framework

- `IGH/set/configs/igh_baseline_m2_v1.yaml`
- `IGH/set/configs/schemes/`
- `IGH/set/src/ra_ild_igh/`
- `IGH/set/scripts_v2/`
- `IGH/set/tests/`
- `IGH/set/README_V2.md`

The original `IGH/set/scripts/`, historical train/test resources, historical model
bundle, and independent-test outputs are not modified.

## Scientific contracts retained

- independent test is not read by the nested-CV engine;
- public references are built from the current training partition only;
- samples used for fitting receive exact leave-one-out public features;
- validation samples use only the corresponding training reference;
- category encoding, zero-variance filtering, and scaling are fitted on the
  current training partition;
- alpha/lambda selection uses inner OOF predictions;
- the classification threshold comes from selected-candidate inner OOF
  predictions;
- positive class is `ILD`;
- the model engine remains Elastic Net logistic regression.

## IGH-specific contracts

- receptor must be `IGH`;
- runtime paths must remain under `IGH/`;
- clone-table suffix is `_IGH-without-DJ_CDR3_AA_clone_table.csv`;
- static manifest groups use the `igh_` prefix;
- historical IGH static candidate count is 1083 for regression only;
- somatic-hypermutation NT-to-AA aggregation remains in the original
  `build_01_AA_clone_table.py` and is covered by a synthetic Batch 01 test.

## Batch 02 configurable scheme layer

A scheme YAML may completely replace the `models` mapping. Model names and counts
are not fixed, and every model separately declares:

- `numeric`;
- `categorical`;
- `static_feature_groups`;
- `dynamic_public`.

Use `replacements.models` when inherited models must be deleted. The scheme preparer
automatically recalculates model-dependent nested-CV and aggregation dimensions.
Development schemes use `final_model.status: not_selected`; they do not inherit the
historical locked winner.

## Additional sample-level feature tables

One or more CSV/CSV.GZ tables may be configured under
`data.<partition>.additional_feature_tables`. The base feature matrix remains the
authoritative sample order. Each additional table must pass:

- unique merge-key checks;
- exact sample coverage checks;
- column-collision checks;
- declared numeric/categorical type checks;
- NA/NaN/Inf checks;
- one-to-one merge and row-order checks.

Only ordinary sample-level features belong in these tables. Label-derived,
cross-patient, supervised-screened, or public-reference-derived features must be
constructed inside the relevant CV training partition.

## Historical baseline

The baseline YAML records the already completed 120/49 IGH analysis. These values
are provenance and regression expectations, not defaults for a future full-cohort
repeated-holdout scheme.

## Batch 01 validation

```bash
python3 IGH/set/scripts_v2/check_batch01_igh_installation.py \
  --repository-root /data/users/chenhaisheng/RA-ILD

python3 -m unittest discover -s IGH/set/tests -p 'test_*.py' -v

python3 IGH/set/scripts_v2/run_all_checks.py \
  --config IGH/set/configs/igh_baseline_m2_v1.yaml \
  --quick \
  --skip-outer-orchestration
```

## Batch 02 validation

```bash
python3 IGH/set/scripts_v2/test_batch02_feature_management.py

python3 IGH/set/scripts_v2/prepare_analysis_scheme.py \
  --scheme IGH/set/configs/schemes/igh_scheme_001_baseline_compat.yaml \
  --dry-run

python3 IGH/set/scripts_v2/prepare_analysis_scheme.py \
  --scheme IGH/set/configs/schemes/igh_scheme_002_no_clinical.yaml \
  --dry-run
```

After preparing a scheme, audit its effective feature inputs without fitting:

```bash
python3 IGH/set/scripts_v2/audit_scheme_features.py \
  --config IGH/set/experiments/<scheme_id>/00_config/resolved_config.yaml \
  --output-dir IGH/set/experiments/<scheme_id>/00_config/feature_audit
```

Batch 02 does not launch outer tasks or select a new final model.
