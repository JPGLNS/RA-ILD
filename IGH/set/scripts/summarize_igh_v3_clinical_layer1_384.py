#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IGH V3 clinical Layer1: 384 checked jobs, paired Age+Sex S000 comparisons.

Requires numpy, pandas, PyYAML and the already deployed, unchanged
run_igh_v3_clinical_layer1_batch.py in this same scripts directory.

Validation reads the immutable 384-YAML plan, config SHA256, frozen split banks,
all 384 complete 100-repeat results, patient/label ID audits, and fit audits,
by REUSING the trusted clinical Layer1 runner's read-only validators.
It writes analysis files ONLY in a new, non-existent directory; never trains.

Caution: all 100 repeated holdouts reuse participants, and selecting 5 best
out of 127 using these holdouts causes optimistic selection bias. Distribution
quantiles are descriptive and must NOT be interpreted as 95% confidence bounds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

# Reuse the same audited preflight checks that already reported PASS=384.
import run_igh_v3_clinical_layer1_batch as runner

METRICS = (
    'roc_auc', 'pr_auc', 'accuracy', 'sensitivity_recall', 'specificity',
    'f1', 'log_loss', 'brier_score',
)
GROUP_CODES = {
    'tcr_diversity': 'D', 'tcr_clonal_expansion': 'E', 'tcr_aa_length': 'L',
    'tcr_aa_composition_unweighted': 'AU', 'tcr_aa_composition_weighted': 'AW',
    'tcr_physicochemical_unweighted': 'PU', 'tcr_physicochemical_weighted': 'PW',
}
REPEATS = set(range(1, 101))
SCOPES = tuple(runner.SCOPES)


def need(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(('true', '1', 'yes'))


def repeats(frame: pd.DataFrame, label: str, unique: bool = False) -> pd.DataFrame:
    need('outer_repeat' in frame.columns, f'{label}: no outer_repeat column')
    frame = frame.copy()
    frame['outer_repeat'] = pd.to_numeric(frame['outer_repeat'], errors='raise').astype(int)
    need(set(frame['outer_repeat']) == REPEATS, f'{label}: repeats not 1..100')
    if unique:
        need(len(frame) == 100 and frame['outer_repeat'].is_unique,
             f'{label}: must contain 100 unique outer repeat rows')
    return frame


def model_data(path: Path, label: str, train_n: int, holdout_n: int,
               clinical_names: set[str] | None) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Read metrics and per-feature selection, not all per-clone inputs.

    Model input identity/fit audit validation is delegated to runner.result_check
    in main() before this function is invoked.
    """
    m = repeats(pd.read_csv(path/'06_outer_metrics.csv'), label+' metrics', True)
    s = repeats(pd.read_csv(path/'06_selected_hyperparameters.csv'), label+' hyperparameters', True)
    p = repeats(pd.read_csv(path/'06_outer_predictions.csv.gz',
                            dtype={'sample_id': str}), label+' predictions')
    c = repeats(pd.read_csv(path/'06_feature_explanations.csv.gz'), label+' coefficients')
    need(len(p) == holdout_n*100, f'{label}: wrong prediction count')
    need({'prediction_score', 'true_cohort', 'sample_id'} <= set(p.columns),
         f'{label}: predictions lack required columns')
    need({'feature_name', 'nonzero', 'absolute_coefficient'} <= set(c.columns),
         f'{label}: coefficient file lacks required columns')
    need({'parameter_alpha', 'parameter_lambda'} <= set(s.columns),
         f'{label}: missing selected parameters')
    need({'n_outer_train', 'n_outer_holdout', 'n_final_predictors', 'fit_converged', *METRICS} <= set(m.columns),
         f'{label}: missing required metric columns')
    need((pd.to_numeric(m['n_outer_train']) == train_n).all()
         and (pd.to_numeric(m['n_outer_holdout']) == holdout_n).all(),
         f'{label}: wrong train/holdout counts')
    need(as_bool(m['fit_converged']).all(), f'{label}: some fits did not converge')
    for field in METRICS:
        need(np.isfinite(pd.to_numeric(m[field], errors='raise').to_numpy(dtype=float)).all(),
             f'{label}: nonfinite {field}')
    need(np.isfinite(pd.to_numeric(p['prediction_score'], errors='raise').to_numpy(dtype=float)).all(),
         f'{label}: nonfinite prediction scores')

    c = c.loc[c['feature_name'].astype(str).ne('__INTERCEPT__')].copy()
    names = set(c['feature_name'].astype(str))
    need(bool(names), f'{label}: no non-intercept coefficient features')
    c['selected'] = as_bool(c['nonzero']).astype(int)
    c['abs_coef'] = pd.to_numeric(c['absolute_coefficient'], errors='raise')
    need(np.isfinite(c['abs_coef'].to_numpy(dtype=float)).all(),
         f'{label}: nonfinite absolute coefficient')
    # S000 determines actual encoded clinical column names rather than guessing
    # that the sex dummy will be spelled 'sex_male' or 'sex=male'.
    if clinical_names is None:
        clinical_names = names
    else:
        need(clinical_names <= names, f'{label}: clinical feature names differ from S000: {sorted(clinical_names-names)}')
    c['feature_type'] = np.where(c['feature_name'].isin(clinical_names), 'clinical_age_sex', 'IGH_static')
    by_rep = c.groupby('outer_repeat')['selected'].sum()
    ign = c[c['feature_type'].eq('IGH_static')].groupby('outer_repeat')['selected'].sum()
    clinical = c[c['feature_type'].eq('clinical_age_sex')].groupby('outer_repeat')['selected'].sum()
    uniq_score = p.groupby('outer_repeat')['prediction_score'].nunique()
    prevalence = p.groupby('outer_repeat')['true_cohort'].apply(
        lambda x: x.astype(str).str.upper().eq('ILD').mean())

    m = m.merge(s[['outer_repeat', 'parameter_alpha', 'parameter_lambda']],
                on='outer_repeat', how='left', validate='one_to_one')
    m['n_nonzero_all'] = m['outer_repeat'].map(by_rep).fillna(0).astype(int)
    m['n_nonzero_IGH'] = m['outer_repeat'].map(ign).fillna(0).astype(int)
    m['n_nonzero_clinical'] = m['outer_repeat'].map(clinical).fillna(0).astype(int)
    m['zero_coefficient_model'] = m['n_nonzero_all'].eq(0)
    m['any_IGH_selected'] = m['n_nonzero_IGH'].gt(0)
    m['constant_score'] = m['outer_repeat'].map(uniq_score).eq(1)
    m['holdout_ILD_prevalence'] = m['outer_repeat'].map(prevalence)
    need(m[['n_nonzero_all', 'n_nonzero_clinical', 'holdout_ILD_prevalence']].notna().all().all(),
         f'{label}: malformed repeat aggregation')
    m['pr_auc_minus_prevalence'] = m['pr_auc'] - m['holdout_ILD_prevalence']
    need((m['n_nonzero_all'] == m['n_nonzero_clinical']+m['n_nonzero_IGH']).all(),
         f'{label}: nonzero partition mismatch')
    coeff_table = (c.groupby(['feature_name', 'feature_type'], as_index=False)
                   .agg(n_selected=('selected', 'sum'), n_observed=('selected', 'size'),
                        mean_abs_coefficient=('abs_coef', 'mean')))
    coeff_table['selection_rate_out_of_100'] = coeff_table['n_selected']/100.0
    return m.set_index('outer_repeat').sort_index(), coeff_table, clinical_names


def summary_record(scope: str, stem: str, feature_groups: list[str], data: pd.DataFrame) -> dict:
    codes = '+'.join(GROUP_CODES[x] for x in feature_groups) if feature_groups else 'none'
    out = {
        'scope': scope, 'stem': stem, 'feature_groups': codes,
        'n_groups': len(feature_groups), 'n_repeats': len(data),
        'n_train': runner.SCOPES[scope][3], 'n_holdout': runner.SCOPES[scope][4],
        'n_final_predictors_median': float(data['n_final_predictors'].median()),
        'zero_coefficient_model_rate': float(data['zero_coefficient_model'].mean()),
        'constant_score_rate': float(data['constant_score'].mean()),
        'any_IGH_selected_rate': float(data['any_IGH_selected'].mean()),
        'n_nonzero_IGH_mean': float(data['n_nonzero_IGH'].mean()),
        'n_nonzero_clinical_mean': float(data['n_nonzero_clinical'].mean()),
        'selected_lambda_median': float(data['parameter_lambda'].median()),
    }
    for metric in METRICS:
        vals = data[metric].astype(float)
        out[f'{metric}_mean'] = float(vals.mean())
        out[f'{metric}_sd'] = float(vals.std(ddof=1))
        out[f'{metric}_median'] = float(vals.median())
        out[f'{metric}_q025'] = float(vals.quantile(.025))
        out[f'{metric}_q975'] = float(vals.quantile(.975))
    return out


def compare_to_baseline(scope: str, stem: str, data: pd.DataFrame,
                        baseline: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    need(data.index.equals(baseline.index), f'{scope}/{stem}: baseline repeats not matched')
    report = {'scope': scope, 'stem': stem, 'n_paired_repeats': 100}
    detail = pd.DataFrame({'scope': scope, 'stem': stem, 'outer_repeat': data.index.to_numpy()})
    for metric in METRICS:
        x = data[metric].to_numpy(dtype=float)
        b = baseline[metric].to_numpy(dtype=float)
        diff = x-b
        detail[f'{metric}_clinical'] = b
        detail[f'{metric}_clinical_IGH'] = x
        detail[f'delta_{metric}'] = diff
        if metric in ('roc_auc', 'pr_auc'):
            report[f'{metric}_baseline_mean'] = float(b.mean())
            report[f'{metric}_joint_mean'] = float(x.mean())
            report[f'delta_{metric}_mean'] = float(diff.mean())
            report[f'delta_{metric}_median'] = float(np.median(diff))
            report[f'delta_{metric}_sd'] = float(diff.std(ddof=1))
            report[f'delta_{metric}_q025'] = float(np.quantile(diff, .025))
            report[f'delta_{metric}_q975'] = float(np.quantile(diff, .975))
            report[f'fraction_{metric}_joint_better'] = float(np.mean(diff > 0))
            report[f'fraction_{metric}_joint_equal'] = float(np.mean(diff == 0))
            report[f'fraction_{metric}_joint_worse'] = float(np.mean(diff < 0))
    for label, frame in [('clinical', baseline), ('joint', data)]:
        for field in ('n_nonzero_all', 'n_nonzero_IGH', 'n_nonzero_clinical', 'zero_coefficient_model',
                      'constant_score', 'any_IGH_selected', 'n_final_predictors'):
            detail[f'{field}_{label}'] = frame[field].to_numpy()
    report['zero_coefficient_rate_joint'] = float(data['zero_coefficient_model'].mean())
    report['any_IGH_selected_rate_joint'] = float(data['any_IGH_selected'].mean())
    report['n_nonzero_IGH_mean_joint'] = float(data['n_nonzero_IGH'].mean())
    report['constant_score_rate_joint'] = float(data['constant_score'].mean())
    return report, detail


def group_addition_effects(scope: str, data: dict[str, pd.DataFrame],
                           groups: dict[str, frozenset[str]]) -> list[dict]:
    group_to_stem = {frozenset(value): stem for stem, value in groups.items()}
    need(len(group_to_stem) == 128, f'{scope}: feature sets are not 128 unique subsets')
    result = []
    for feature in GROUP_CODES:
        diffs_auc, diffs_ap = [], []
        for subset, stem in group_to_stem.items():
            if feature in subset:
                continue
            new_stem = group_to_stem.get(subset | {feature})
            need(new_stem is not None, f'{scope}: no counterpart for {stem}+{feature}')
            a = data[stem].sort_index()
            b = data[new_stem].sort_index()
            need(a.index.equals(b.index), f'{scope}: repeat mismatch in +{feature}')
            diffs_auc.append(float((b['roc_auc']-a['roc_auc']).mean()))
            diffs_ap.append(float((b['pr_auc']-a['pr_auc']).mean()))
        need(len(diffs_auc) == 64, f'{scope}: expected 64 addition contrasts for {feature}')
        vals = np.asarray(diffs_auc)
        aps = np.asarray(diffs_ap)
        result.append({
            'scope': scope, 'added_group': GROUP_CODES[feature],
            'added_group_full': feature, 'n_paired_combinations': len(vals),
            'mean_delta_auc_when_added': float(vals.mean()),
            'median_delta_auc_when_added': float(np.median(vals)),
            'fraction_positive_auc_combinations': float((vals>0).mean()),
            'mean_delta_pr_auc_when_added': float(aps.mean()),
            'fraction_positive_pr_auc_combinations': float((aps>0).mean()),
        })
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='.', help='RA-ILD repo directory (defaults to current working directory)')
    p.add_argument('--output-dir', help='New output directory; must NOT already exist')
    args = p.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    need(repo == runner.REPO, 'Put this script beside run_igh_v3_clinical_layer1_batch.py in IGH/set/scripts; '
         'both must belong to the specified --repo')
    plan_path = repo / 'IGH/set/data_versions/igh_v3_175/analysis/clinical_layer1_384_plan_v3_175.json'
    need(plan_path.is_file(), f'Locked plan not found: {plan_path}')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    need(plan.get('decision') == 'IGH_V3_CLINICAL_LAYER1_AGE_SEX_ALL128'
         and len(plan.get('configs', [])) == 384, 'Locked clinical layer1 plan malformed')
    print('[1/4] Checking frozen split banks, 384 config hashes and plan...', flush=True)
    contexts = {}
    for scope in SCOPES:
        entries = [x for x in plan['configs'] if x['scope'] == scope]
        contexts[scope] = runner.preflight_scope(scope, entries, plan['frozen_split_sha256'][scope])
        need(len(contexts[scope]['jobs']) == 128, f'{scope}: 128 configs not found')

    print('[2/4] Reading all 384 complete clinical model outputs...', flush=True)
    model_summaries, paired_summaries, paired_repeats, feature_sel, additions = [], [], [], [], []
    baseline_summaries = []
    for scope in SCOPES:
        baseline = None
        clinical_names = None
        by_stem = {}
        group_by_stem = {}
        for i, (_, stem, cfg) in enumerate(contexts[scope]['jobs'], 1):
            valid, detail = runner.result_check(scope, stem, contexts[scope])
            need(valid, f'{scope}/{stem}: not PASS: {detail}')
            config = yaml.safe_load(cfg.read_text(encoding='utf-8'))
            group_names = config['features']['static_tcr_groups']
            run = runner.output_dir(scope, stem)
            metrics, coeff, clinical_names = model_data(
                run, f'{scope}/{stem}', contexts[scope]['train_n'], contexts[scope]['holdout_n'],
                clinical_names)
            if stem == 'S000_none':
                need(not group_names and metrics['n_nonzero_IGH'].eq(0).all(),
                     f'{scope}: baseline appears to contain IGH coefficient')
                baseline = metrics
                baseline_summaries.append(summary_record(scope, stem, group_names, metrics))
            else:
                need(baseline is not None, f'{scope}: S000 must be first')
                need(len(group_names) > 0, f'{scope}/{stem}: no IGH feature groups')
            by_stem[stem] = metrics
            group_by_stem[stem] = frozenset(group_names)
            sr = summary_record(scope, stem, group_names, metrics)
            model_summaries.append(sr)
            if stem != 'S000_none':
                pair, reps = compare_to_baseline(scope, stem, metrics, baseline)
                paired_summaries.append(pair)
                paired_repeats.append(reps)
            coeff.insert(0, 'stem', stem)
            coeff.insert(0, 'scope', scope)
            feature_sel.append(coeff)
            if i % 32 == 0:
                print(f'  {scope}: {i}/128 jobs PASS', flush=True)
        need(len(by_stem) == 128, f'{scope}: expected 128 unique experiments')
        additions.extend(group_addition_effects(scope, by_stem, group_by_stem))

    full = pd.DataFrame(model_summaries)
    pair = pd.DataFrame(paired_summaries)
    detail = pd.concat(paired_repeats, ignore_index=True)
    selections = pd.concat(feature_sel, ignore_index=True)
    effects = pd.DataFrame(additions)
    need(len(full) == 384 and len(pair) == 381 and len(detail) == 38100 and len(effects) == 21,
         'Incorrect final summary row counts')
    need(len(full.loc[full.stem == 'S000_none']) == 3, 'Not exactly 3 clinical S000 baselines')
    pair = pair.merge(full[['scope', 'stem', 'feature_groups', 'n_groups', 'roc_auc_mean', 'pr_auc_mean']],
                      on=['scope','stem'], validate='one_to_one')
    for metric in ('roc_auc', 'pr_auc'):
        need(np.allclose(pair[metric+'_joint_mean'], pair[metric+'_mean'], atol=1e-12),
             'Paired mean differs from model mean')
    pair = pair.sort_values(['scope', 'delta_roc_auc_mean', 'delta_pr_auc_mean', 'fraction_roc_auc_joint_better', 'n_groups'],
                            ascending=[True, False, False, False, True]).reset_index(drop=True)
    pair['rank_joint_auc_in_scope'] = pair.groupby('scope').cumcount()+1
    pair['positive_mean_auc_gain'] = pair['delta_roc_auc_mean'] > 0
    pair['positive_mean_pr_auc_gain'] = pair['delta_pr_auc_mean'] > 0
    pair['fraction_auc_wins_gt_half'] = pair['fraction_roc_auc_joint_better'] > .5
    pair['screen_positive_auc_and_ap_and_win_rate'] = (pair['positive_mean_auc_gain'] &
        pair['positive_mean_pr_auc_gain'] & pair['fraction_auc_wins_gt_half'])
    full = full.merge(pair[['scope','stem','rank_joint_auc_in_scope']],
                      on=['scope','stem'], how='left', validate='one_to_one')

    print('[3/4] Reviewing paired ΔAUC and ΔPR-AUC against S000 clinical baselines...', flush=True)
    baselines = pd.DataFrame(baseline_summaries)
    show = ['rank_joint_auc_in_scope', 'stem', 'feature_groups',
            'roc_auc_baseline_mean', 'roc_auc_joint_mean', 'delta_roc_auc_mean',
            'fraction_roc_auc_joint_better', 'delta_pr_auc_mean',
            'any_IGH_selected_rate_joint', 'screen_positive_auc_and_ap_and_win_rate']
    for scope in SCOPES:
        b = baselines.loc[baselines.scope.eq(scope)].iloc[0]
        d = pair.loc[pair.scope.eq(scope)].head(5)
        print(f'\n=== {scope.upper()} S000 Age+Sex: ROC-AUC={b["roc_auc_mean"]:.4f}, '
              f'PR-AUC={b["pr_auc_mean"]:.4f} ===', flush=True)
        print(d[show].to_string(index=False, float_format=lambda x: f'{x:+.4f}'), flush=True)
        print(f'  Models with positive mean ΔAUC: '
              f'{int(pair.loc[pair.scope.eq(scope), "positive_mean_auc_gain"].sum())}/127; '
              f'pass conservative exploratory screen: '
              f'{int(pair.loc[pair.scope.eq(scope), "screen_positive_auc_and_ap_and_win_rate"].sum())}/127',
              flush=True)

    output = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
              repo/'IGH/set/data_versions/igh_v3_175/analysis'/
              ('clinical_layer1_summary_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    need(not output.exists(), f'REFUSE overwrite an existing analysis directory: {output}')
    print('[4/4] Writing analysis-only summary to new directory...', flush=True)
    output.mkdir(parents=True, exist_ok=False)
    full.to_csv(output/'01_experiment_summary_384.csv', index=False)
    pair.to_csv(output/'02_paired_vs_age_sex_381.csv', index=False)
    detail.to_csv(output/'03_paired_per_repeat_38100.csv.gz', index=False, compression='gzip')
    effects.to_csv(output/'04_group_addition_effects.csv', index=False)
    selections.to_csv(output/'05_feature_selection_frequency.csv.gz', index=False, compression='gzip')
    baselines.to_csv(output/'06_clinical_baseline_3.csv', index=False)
    for scope in SCOPES:
        pair.loc[pair.scope.eq(scope)].head(5).to_csv(output/f'ranking_{scope}_top5.csv', index=False)
    audit = {
        'status': 'PASS', 'design': 'IGH_V3_AGE_SEX_STATIC_384',
        'frozen_split_sha256': {scope: runner.SCOPES[scope][1] for scope in SCOPES},
        'locked_plan_sha256': sha256(plan_path),
        'checked_model_outputs': 384, 'checked_outer_repeats': 38400,
        'paired_to_clinical_baseline_experiments': 381,
        'paired_to_clinical_baseline_repeats': 38100,
        'group_addition_comparisons': 21,
        'run_started_utc': datetime.now(timezone.utc).isoformat(),
        'notes': 'Read-only. Ranking is exploratory; repeated splits are correlated; no holdout-based candidate selection inference.',
    }
    (output/'00_validation_report.json').write_text(
        json.dumps(audit, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    readme = [
        '# IGH V3 Clinical Layer1, Age+Sex on all 384 models', '',
        'S000 clinical-only (Age+Sex) and S001..S127 Age+Sex+IGH static features in each of three scopes.',
        'The same original frozen 100 repeated holdout IDs were checked for all model outputs.', '',
        '**Statistical caution:** The 100 repeats share patients; 2.5%/97.5% ranges are descriptive distribution quantiles, not independent-validation CIs.',
        'Ranking the 127 candidate models by mean observed AUC on these holdouts causes winner-selection optimism.',
        'The mean joint AUC ordering exactly equals mean paired ΔAUC ordering within each scope, because S000 is the same baseline.',
        'The screen flag combines positive average ΔAUC, positive average ΔPR-AUC, and AUC win fraction > 0.5; it is exploratory, not a significance test.',
        'Do not claim selected models are externally validated. Consider nested feature-combination selection or an independent validation set.', '',
        '## Generated files', '',
        '- 00_validation_report.json: validations, bank hashes, counting',
        '- 01_experiment_summary_384.csv: 384 raw model summaries (including three clinical S000 baselines)',
        '- 02_paired_vs_age_sex_381.csv: main model increments vs same-scope S000, ranking and screening flags',
        '- 03_paired_per_repeat_38100.csv.gz: per-repeat paired comparison for all 381 non-baseline models',
        '- 04_group_addition_effects.csv: 7 feature-group additions x 3 scopes, 64 combinations each',
        '- 05_feature_selection_frequency.csv.gz: coefficient selection per feature, distinguish clinical vs IGH by S000 names',
        '- 06_clinical_baseline_3.csv: three S000 clinical baselines',
        '- ranking_{scope}_top5.csv: five highest mean joint AUC candidates, not yet the final best model', '',
    ]
    (output/'README_summary.md').write_text('\n'.join(readme)+'\n', encoding='utf-8')
    print('\nIGH V3 CLINICAL LAYER1 SUMMARY PASS: 384 experiments / '
          '38400 holdout repeats / 38100 paired-to-S000 repeats', flush=True)
    print('OUTPUT_DIR:', output, flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
