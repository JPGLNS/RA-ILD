# Phase 5 — Advanced Feature Sources

Phase 5 formalizes the experiment-side feature-source contract without changing the legacy biological formulas.

## Capabilities

1. **One repertoire source per experiment**
   - `mode: all`, or
   - `mode: topk` with any positive `top_k`.
   - The existing deterministic Step01b Top-K builder is reused and cached under `TRB/result/01b_AA_clone_table_top<K>`.

2. **3-mer configuration**
   - `representation: unweighted|weighted`.
   - `top_n` maps directly to the existing Step02 `--max-kmers` rule.
   - The existing training-only prevalence/variance filtering and ranking are retained.

3. **Enriched dictionary**
   - The dictionary **must inherit the same all/Top-K repertoire** as the rest of the experiment.
   - Default `T=20%`, `Delta=10 percentage points`; both are configurable.
   - Default predictor is `RA_dict_hit_rate`.
   - `RA_dict_hit_rate = sample RA-dictionary clone hits / RA dictionary size`.
   - No abundance cutoff is applied to dictionary membership.
   - `fit` writes a fixed reference for held-out TEST transform; fit-sample scores are explicitly diagnostic-only.
   - `cv` delegates to the existing `enrich_dictionary_grid.py` optimized repeated-5-fold engine using a temporary source shim, preserving old CV/AUC behavior and workers 1–16.

4. **Experiment runtime controls**
   - `validation.repeats`: arbitrary integer >=1 (e.g. 100, or 50 for expensive XGBoost experiments later).
   - `compute.workers`: default 1; allowed 1–16.
   - model algorithm names are validated now; unified model execution remains a later phase.

## Recommended Phase-5 workflow

1. Copy the template and edit the experiment config.
2. Prepare/validate the chosen repertoire cache with `prepare_advanced_repertoire.py`.
3. Resolve and validate the experiment with `validate_advanced_experiment.py`; this emits a dynamic Phase-2 registry/plan.
4. Run source-aware Step02 using the emitted registry/plan and the printed k-mer arguments.
5. Run Phase3 modules and Phase4 matrix assembly as before.
6. For enriched dictionary:
   - use `--mode cv` for standalone leakage-controlled validation;
   - use `--mode fit` on TRAIN and `--mode transform` on TEST for a frozen independent-test feature.

**Important:** the future unified ML layer must reconstruct learned-reference features inside each model CV training fold. Do not treat full-TRAIN fit scores as CV predictors.
