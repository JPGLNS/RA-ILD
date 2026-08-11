# Step 01b — Exact Top-K CDR3-AA repertoire standardization

## Purpose

This step creates a standardized fixed-richness AA repertoire for downstream feature engineering while preserving the original full Step-01 repertoire.

It is **deterministic Top-K truncation**, not random read-level rarefaction.

## Input

Full Step-01 AA clone tables containing at least:

- `sample_id`
- `cdr3_aa`
- `aa_length`
- `nt_clone_number`
- `read_count`
- `read_fraction`
- `input_frequency_sum`
- `input_cell_frequency_sum`
- `frequency`
- `frequency_norm`
- `top_nt_cdr3`

Default discovery pattern:

```text
*_AA_clone_table.csv
```

This matches both the repository canonical name such as
`*_TRB_CDR3_AA_clone_table.csv` and shorter historical names such as
`*_AA_clone_table.csv`.

## Exact Top-K rule

For each sample, AA clonotypes are sorted by:

1. `frequency` descending
2. `read_count` descending
3. `cdr3_aa` ascending

Then the script keeps **exactly the first K rows**. It never uses a cutoff expression such as `frequency >= kth_frequency`, so ties at the K boundary cannot increase the repertoire beyond K.

Default K is 10,000 and can be changed using `--top-k`.

## Preserved versus recomputed columns

The following clone-level values are preserved exactly from the full repertoire:

- `sample_id`
- `cdr3_aa`
- `aa_length`
- `nt_clone_number`
- `read_count`
- `input_frequency_sum`
- `input_cell_frequency_sum`
- `frequency`
- `top_nt_cdr3`

The original normalized values are retained as:

- `read_fraction_full`
- `frequency_norm_full`

The canonical normalized values are recomputed within retained Top-K:

```text
read_fraction  = read_count / sum(read_count in retained Top-K)
frequency_norm = frequency / sum(frequency in retained Top-K)
```

Thus downstream Top-K features should use the new `read_fraction` and `frequency_norm`, while the `_full` columns remain available for audit and retained-mass calculations.

## Safety/QC behavior

Before writing any sample output, the script performs a cohort-wide preflight. Formal writing starts only if every discovered sample passes.

Checks include:

- one sample per input file
- unique `cdr3_aa` within sample
- all required columns present
- finite, non-negative abundance values
- full `frequency_norm` sum approximately 1
- full `read_fraction` sum approximately 1
- full AA clone count >= K
- exact retained clone count = K
- deterministic K-boundary ordering
- Top-K `frequency_norm` sum approximately 1
- Top-K `read_fraction` sum approximately 1
- retained mass from normalized values agrees with mass from raw values
- preserved clone-level fields are unchanged during output construction
- output is read back and revalidated after serialization

Samples with fewer than K clones raise an error instead of being silently kept at a smaller size.

## Outputs

For K=10000:

```text
{sample_id}_TRB_CDR3_AA_clone_table_top10000.csv
01b_top10000_summary.csv
01b_top10000_qc.csv
01b_top10000_configuration.json
```

The summary records full/retained clone counts, retained read/frequency mass, cutoff abundance, K-boundary tie count, and normalization checks.

## Recommended project location

```text
TRB/result/01_AA_clone_table/                # immutable full repertoire
TRB/result/01b_AA_clone_table_topk10000/     # new standardized repertoire
```

## Dry-run

```bash
python3 TRB/set/scripts/build_01b_AA_clone_table_topk.py \
  --input-dir TRB/result/01_AA_clone_table \
  --output-dir TRB/result/01b_AA_clone_table_topk10000 \
  --top-k 10000 \
  --dry-run
```

Dry-run performs the full cohort-wide preflight but writes nothing.

## Formal run

```bash
python3 TRB/set/scripts/build_01b_AA_clone_table_topk.py \
  --input-dir TRB/result/01_AA_clone_table \
  --output-dir TRB/result/01b_AA_clone_table_topk10000 \
  --top-k 10000
```

Existing Top-K sample outputs are not overwritten unless `--overwrite` is explicitly supplied.
