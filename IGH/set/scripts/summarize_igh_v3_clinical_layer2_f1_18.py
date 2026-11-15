#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize 18 completed IGH V3 clinical Layer2 Top500 AA 3mer (F1) runs.

READ ONLY on existing configs/models/banks. Writes a new timestamped analysis dir.
Paired comparisons on identical frozen outer repeats:
  A) Age+Sex(+static)+F1 versus matched Age+Sex(+static) F0, 18 pairs;
  B) each F1 versus Age+Sex-only S000 (clinical baseline), 18 pairs;
  C) each F1-with-static versus Age+Sex+F1-only S000_F1, 15 pairs.

External requirements (already present in user's IGH/set/scripts/):
  run_igh_v3_clinical_layer1_batch.py
  run_igh_v3_clinical_layer2_f1_batch.py
  summarize_igh_v3_clinical_layer1_384.py
  build_igh_v3_clinical_layer2_f1_18.py
Requires Python >=3.9, numpy, pandas, PyYAML.

The 100 holdouts reuse the same patients; their percentiles are NOT independent
confidence intervals. Layer1 Top5 selection used these splits and is exploratory.
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

import run_igh_v3_clinical_layer1_batch as layer1
import run_igh_v3_clinical_layer2_f1_batch as layer2
import summarize_igh_v3_clinical_layer1_384 as clin1sum

SCOPES = ('total', 'pbmc', 'buffycoat')
METRICS = ('roc_auc', 'pr_auc', 'accuracy', 'sensitivity_recall', 'specificity',
           'f1', 'log_loss', 'brier_score')


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def strue(s):
    return s.astype(str).str.lower().isin(('true', 'yes', '1'))


def get_coeff(path, clinical_set, static_set, label, repeat_set):
    """Provide descriptive coefficient-selection rates, without assuming kmer naming."""
    a = pd.read_csv(path/'06_feature_explanations.csv.gz',
                    usecols=['outer_repeat', 'feature_name', 'nonzero', 'absolute_coefficient'])
    need(len(a)>0 and set(a['outer_repeat'].astype(int))==repeat_set,
         f'{label}: missing coefficient repeats')
    a['outer_repeat'] = a['outer_repeat'].astype(int)
    a = a.loc[a.feature_name.astype(str).ne('__INTERCEPT__')].copy()
    a['feature_name'] = a.feature_name.astype(str)
    a['selected'] = strue(a.nonzero).astype(int)
    a['abs_coefficient'] = pd.to_numeric(a.absolute_coefficient, errors='raise')
    need(np.isfinite(a.abs_coefficient.to_numpy(dtype=float)).all(),
         f'{label}: non-finite coefficients')
    a['feature_type'] = np.select(
        [a.feature_name.isin(clinical_set), a.feature_name.isin(static_set)],
        ['clinical_age_sex', 'IGH_static'], default='3mer_candidate')
    # F0 models must have no 3mer_candidate; F1 models have a fold-fit audit.
    by_rep = (a.groupby(['outer_repeat', 'feature_type'])['selected'].sum().unstack(fill_value=0)
              .reindex(index=sorted(repeat_set), fill_value=0)
              .reindex(columns=['clinical_age_sex','IGH_static','3mer_candidate'], fill_value=0))
    by_rep.columns = ['n_nonzero_clinical', 'n_nonzero_static', 'n_nonzero_3mer']
    cnt = a.groupby(['feature_name','feature_type'], as_index=False).agg(
        n_selected=('selected','sum'), n_observed=('selected','size'),
        mean_abs_coefficient=('abs_coefficient','mean'))
    cnt['selection_rate_out_of_100'] = cnt.n_selected/100.0
    return by_rep, cnt


def load_validated(scope, stem, path, ctx, is_f1, clinical_names, static_names):
    """Delegate full audit to production runners and clinical Layer1 reader."""
    if is_f1:
        ok, why = layer2.result_check(scope, stem, ctx)
    else:
        ok, why = layer1.result_check(scope, stem, ctx)
    need(ok, f'{scope}/{stem}: NOT PASS: {why}')
    mm, _coeff_summary, found_clinical = clin1sum.model_data(
        path, f'{scope}/{stem}', ctx['train_n'], ctx['holdout_n'], clinical_names)
    need(len(mm)==100 and mm.index.is_unique, f'{scope}/{stem}: incorrect repeat rows')
    by_rep, sel = get_coeff(path, clinical_names or found_clinical,
                            static_names, f'{scope}/{stem}', set(range(1,101)))
    need(mm.index.equals(by_rep.index),
         f'{scope}/{stem}: repeat indices mismatch')
    need(mm['n_nonzero_clinical'].equals(by_rep['n_nonzero_clinical']),
         f'{scope}/{stem}: clinical coefficient counts disagree')
    mm = mm.join(
        by_rep[['n_nonzero_static', 'n_nonzero_3mer']],
        how='left'
    )
    need((mm['n_nonzero_IGH'] ==
          mm['n_nonzero_static'] + mm['n_nonzero_3mer']).all(),
         f'{scope}/{stem}: coefficient partition mismatch')
    need(mm[['n_nonzero_clinical','n_nonzero_static','n_nonzero_3mer']].notna().all().all(),
         f'{scope}/{stem}: coefficient counts missing')
    need((mm['n_nonzero_clinical']+mm['n_nonzero_static']+mm['n_nonzero_3mer']
          ==mm['n_nonzero_all']).all(), f'{scope}/{stem}: coefficient partition incorrect')
    if not is_f1:
        need(mm.n_nonzero_3mer.eq(0).all(),f'{scope}/{stem}: F0 has unrecognized 3mer features')
    return mm, sel, found_clinical


def paired(scope, stem, comparison, treatment, control):
    need(treatment.index.equals(control.index) and len(treatment)==100,
         f'{scope}/{stem}/{comparison}: repeats not identically paired')
    row = {'scope':scope, 'stem':stem, 'comparison':comparison, 'n_paired_repeats':100}
    detail = pd.DataFrame({'scope':scope,'stem':stem,'comparison':comparison,
                           'outer_repeat':treatment.index.to_numpy()})
    for metric in METRICS:
        x=treatment[metric].to_numpy(dtype=float)
        y=control[metric].to_numpy(dtype=float)
        d=x-y
        need(np.isfinite(d).all(),f'{scope}/{stem}: nonfinite delta {metric}')
        detail[f'{metric}_treatment']=x
        detail[f'{metric}_control']=y
        detail[f'delta_{metric}']=d
        if metric in ('roc_auc','pr_auc'):
            row[f'{metric}_treatment_mean']=float(x.mean())
            row[f'{metric}_control_mean']=float(y.mean())
            row[f'delta_{metric}_mean']=float(d.mean())
            row[f'delta_{metric}_median']=float(np.median(d))
            row[f'delta_{metric}_sd']=float(d.std(ddof=1))
            row[f'delta_{metric}_q025']=float(np.quantile(d,0.025))
            row[f'delta_{metric}_q975']=float(np.quantile(d,0.975))
            row[f'fraction_{metric}_better']=float((d>1e-12).mean())
            row[f'fraction_{metric}_equal']=float((np.abs(d)<=1e-12).mean())
            row[f'fraction_{metric}_worse']=float((d< -1e-12).mean())
    for t, frame in [('treatment',treatment),('control',control)]:
        for field in ('n_nonzero_all','n_nonzero_clinical','n_nonzero_static',
                      'n_nonzero_3mer','n_final_predictors','constant_score'):
            detail[f'{field}_{t}']=frame[field].to_numpy()
    row['treatment_nonzero_3mer_rate']=float((treatment.n_nonzero_3mer>0).mean())
    row['treatment_nonzero_static_rate']=float((treatment.n_nonzero_static>0).mean())
    row['treatment_zero_all_rate']=float(treatment.zero_coefficient_model.mean())
    row['treatment_constant_score_rate']=float(treatment.constant_score.mean())
    row['treatment_mean_n_3mer_selected']=float(treatment.n_nonzero_3mer.mean())
    row['treatment_mean_n_static_selected']=float(treatment.n_nonzero_static.mean())
    return row,detail


def model_summary(scope, stem, model_type, data, groups):
    r={'scope':scope,'stem':stem,'model_type':model_type,
       'feature_groups':groups,'n_outer_repeats':100,
       'zero_all_rate':float(data.zero_coefficient_model.mean()),
       'any_static_selected_rate':float((data.n_nonzero_static>0).mean()),
       'any_3mer_selected_rate':float((data.n_nonzero_3mer>0).mean()),
       'n_static_selected_mean':float(data.n_nonzero_static.mean()),
       'n_3mer_selected_mean':float(data.n_nonzero_3mer.mean()),
       'constant_score_rate':float(data.constant_score.mean()),
       'selected_lambda_median':float(data.parameter_lambda.median())}
    for metric in METRICS:
        v=data[metric].to_numpy(dtype=float)
        r[f'{metric}_mean']=float(v.mean())
        r[f'{metric}_sd']=float(v.std(ddof=1))
        r[f'{metric}_median']=float(np.median(v))
    return r


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',default='.',help='Root of RA-ILD repository')
    ap.add_argument('--output-dir',help='NEW (non-existing) output directory')
    args=ap.parse_args()
    repo=Path(args.repo).resolve()
    need(repo==layer1.REPO==layer2.REPO,
         'Run script beside both layer1/layer2 batch runners in IGH/set/scripts, with correct --repo')
    print('[1/4] Checking 18-plan, three frozen banks, 18 configurations...',flush=True)
    ctxs,jobs,entries=layer2.plan_check()
    need(len(jobs)==18 and len(entries)==18,'Layer2 plan has wrong experiment count')
    plan2=json.loads(layer2.PLAN.read_text(encoding='utf-8'))
    src_dir=Path(plan2['clinical_layer1_summary_dir'])
    need(src_dir.is_dir(),f'clinical Layer1 summary missing: {src_dir}')
    for fn, dig in plan2['clinical_layer1_summary_file_sha256'].items():
        need(sha(src_dir/fn)==dig,f'clinical Layer1 summary mismatch: {fn}')

    print('[2/4] Validating and reading 18 F1 + 18 matched F0 models...',flush=True)
    all_models=[]; all_pairs=[]; all_details=[]; select=[]
    top5_rows=[]
    for scope in SCOPES:
        ctx=ctxs[scope]
        cohort_jobs=[j for j in jobs if j[0]==scope]
        need(len(cohort_jobs)==6,f'{scope}: expected six F1 models')
        models={}; static_name_map={}
        clinical_names=None
        # S000 F0 must be first so column-level clinical names are derived, not guessed.
        f0_stems=['S000_none']+[stem[:-3] for _,stem,_ in cohort_jobs if stem!='S000_none_F1']
        need(len(f0_stems)==6 and len(set(f0_stems))==6,f'{scope}: matched static groups missing')
        for stem in f0_stems:
            old=layer1.output_dir(scope,stem)
            cfg=layer1.config_dir(scope)/(stem+'.yaml')
            need(cfg.is_file(),f'{scope}/{stem}: missing frozen Layer1 YAML')
            groups=yaml.safe_load(cfg.read_text(encoding='utf-8'))['features']['static_tcr_groups']
            group_label='+'.join(clin1sum.GROUP_CODES[x] for x in groups) or 'none'
            if stem=='S000_none':
                # Reuse actual age/sex column names instead of assuming dummy encoding.
                _,_,clinical_names=clin1sum.model_data(old,f'{scope}/{stem}',ctx['train_n'],ctx['holdout_n'],None)
            need(bool(clinical_names),f'{scope}: cannot determine clinical coefficient names')
            # Use union of model F0 coefficients to identify static predictors in F1.
            old_cols=pd.read_csv(old/'06_feature_explanations.csv.gz',usecols=['feature_name']).feature_name.astype(str)
            static_names=set(old_cols)-set(clinical_names)-{'__INTERCEPT__'}
            static_name_map[stem]=static_names
            f0, co, _=load_validated(scope,stem,old,ctx,False,clinical_names,static_names)
            need(f0.n_nonzero_3mer.eq(0).all(),f'{scope}/{stem}: F0 has unexpected features')
            if stem=='S000_none':
                need(not static_names,f'{scope}: S000 is not clinical only')
            models[(stem,'F0')]=f0
            all_models.append(model_summary(scope,stem,'F0',f0,group_label))
        for _,stem,_cfg in cohort_jobs:
            oldstem=stem[:-3]
            f0=models[(oldstem,'F0')]
            f1path=layer2.output_dir(scope,stem)
            new,fsel,_=load_validated(scope,stem,f1path,ctx,True,clinical_names,static_name_map[oldstem])
            need(set(new.index)==set(f0.index),f'{scope}/{stem}: F0 and F1 not repeat matched')
            # A different ML representation may omit some coefficient rows in particular repeats.
            # New columns not present in the matching F0 matrix are called '3mer_candidate',
            # and 3mer training-local construction was checked by layer2.result_check.
            need(bool((fsel.feature_type=='3mer_candidate').any()),
                 f'{scope}/{stem}: no 3mer candidate coefficient names found')
            groups=next(x for x in all_models if x['scope']==scope and x['stem']==oldstem and x['model_type']=='F0')['feature_groups']
            models[(oldstem,'F1')]=new
            all_models.append(model_summary(scope,stem,'F1',new,groups))
            fsel.insert(0,'stem',stem)
            fsel.insert(0,'scope',scope)
            select.append(fsel)
            specs=[('F1_vs_matched_F0',new,f0),
                   ('F1_vs_AgeSex_S000',new,models[('S000_none','F0')])]
            if oldstem!='S000_none':
                # F1+static against F1-only evaluates incremental STATIC given 3mer.
                specs.append(('F1_static_vs_3mer_only',new,models[('S000_none','F1')]))
            for comparison,treatment,control in specs:
                row,d=paired(scope,stem,comparison,treatment,control)
                row['feature_groups']=groups
                row['layer1_stem']=oldstem
                all_pairs.append(row)
                all_details.append(d)
            print(f'  PASS {scope}/{stem} (100 repeats, three paired contrasts where applicable)',flush=True)
        need(len(cohort_jobs)==6,f'{scope}: wrong completed F1 models')

    model_df=pd.DataFrame(all_models)
    pair_df=pd.DataFrame(all_pairs)
    details=pd.concat(all_details,ignore_index=True)
    selection=pd.concat(select,ignore_index=True)
    need(len(model_df)==36 and len(pair_df)==51 and len(details)==5100,
         f'wrong output row counts: models={len(model_df)} pairs={len(pair_df)} repeats={len(details)}')
    need(len(selection)>0,'empty coefficient selection table')
    print('[3/4] Ranking and reporting paired incremental ROC-AUC / PR-AUC...',flush=True)
    primary=pair_df.loc[pair_df.comparison=='F1_vs_matched_F0'].copy()
    for scope in SCOPES:
        cur=primary.loc[primary.scope==scope].sort_values('roc_auc_treatment_mean',ascending=False)
        print(f'\n=== {scope.upper()} Clinical Layer2 F1 versus corresponding clinical static F0 ===',flush=True)
        display=('stem','feature_groups','roc_auc_control_mean','roc_auc_treatment_mean',
                 'delta_roc_auc_mean','delta_pr_auc_mean','fraction_roc_auc_better',
                 'treatment_nonzero_3mer_rate','treatment_nonzero_static_rate')
        print(cur[list(display)].to_string(index=False,float_format=lambda v:f'{v:+.4f}'),flush=True)
        base=pair_df.loc[(pair_df.scope==scope)&(pair_df.stem=='S000_none_F1')&
                         (pair_df.comparison=='F1_vs_AgeSex_S000')].iloc[0]
        print(f'Age+Sex baseline AUC {base.roc_auc_control_mean:.4f}; '
              f'Age+Sex+3mer AUC {base.roc_auc_treatment_mean:.4f}; '
              f'\u0394AUC {base.delta_roc_auc_mean:+.4f}',flush=True)
        top5=cur.loc[cur.stem.ne('S000_none_F1')].head(5).copy()
        top5.insert(0,'rank_f1_auc',range(1,len(top5)+1))
        top5_rows.append((scope,top5))

    output=(Path(args.output_dir).expanduser().resolve() if args.output_dir else
            layer1.VER/'analysis'/('clinical_layer2_f1_summary_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    need(not output.exists(),f'REFUSE OVERWRITE existing output: {output}')
    print('[4/4] Writing new analysis-only outputs...',flush=True)
    output.mkdir(parents=True,exist_ok=False)
    model_df.to_csv(output/'01_model_summary_36.csv',index=False)
    primary.to_csv(output/'02_f1_vs_matched_f0_18.csv',index=False)
    pair_df.loc[pair_df.comparison=='F1_vs_AgeSex_S000'].to_csv(output/'03_f1_vs_age_sex_18.csv',index=False)
    pair_df.loc[pair_df.comparison=='F1_static_vs_3mer_only'].to_csv(output/'04_static_addition_on_3mer_15.csv',index=False)
    details.to_csv(output/'05_all_paired_repeats_5100.csv.gz',index=False,compression='gzip')
    selection.to_csv(output/'06_f1_feature_selection_frequency.csv.gz',index=False,compression='gzip')
    for scope,rank in top5_rows:
        rank.to_csv(output/f'ranking_{scope}_f1_top5.csv',index=False)
    report={
      'status':'PASS','design':'IGH_V3_CLINICAL_LAYER2_TOP5_PLUS_S000_F1_18',
      'checked_F1_experiments':18,'checked_matched_F0_experiments':18,
      'unique_model_summaries':36,'paired_comparisons':51,
      'matched_repeat_comparisons':5100,
      'layer2_plan_sha256':sha(layer2.PLAN),
      'layer1_plan_sha256':sha(layer1.PLAN),
      'source_clinical_layer1_summary':str(src_dir),
      'frozen_split_sha256':{scope:ctxs[scope]['split_sha'] for scope in SCOPES},
      'cautions':['All split repeats share subjects; quantiles are descriptive, not CIs.',
                  'Layer1 Top5 was selected on same frozen repeat holdouts; apparent gains are selection-biased.',
                  'No external test cohort was used to validate the selected winner.',
                  'Feature classifications based on empirical Layer1 clinical and static coefficient column names.'],
      'summary_utc':datetime.now(timezone.utc).isoformat()
    }
    (output/'00_validation_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    readme = """# IGH V3 Clinical Layer2 F1-only summary

All 18 models use age+sex + Top500 training-fold-fitted unweighted AA 3mer;
15 include the five selected static IGH combinations per scope.

- `01_model_summary_36.csv`: 18 unique F0 and 18 F1 model summaries, each 100 frozen outer repeats.
- `02_f1_vs_matched_f0_18.csv`: **primary** 3mer increment over matched static/clinical model; includes F1-only controls.
- `03_f1_vs_age_sex_18.csv`: each F1 compared with clinical-only Age+Sex S000 (total immunorepertoire increment).
- `04_static_addition_on_3mer_15.csv`: added static feature value beyond Age+Sex+3mer-only.
- `05_all_paired_repeats_5100.csv.gz`: 51 comparisons x 100 matched frozen outer repeats.
- `06_f1_feature_selection_frequency.csv.gz`: feature selection among 18 F1 models, classified relative to F0 column names.
- `ranking_{scope}_f1_top5.csv`: F1 AUC rank of the five static candidate sets; no new model training.

**Important**: Repeated holdouts reuse participants and Top5 selection used these same
holdouts. Quantiles are descriptive, not independent confidence intervals or
selection-adjusted performance. Avoid claims of external generalization or causality.
"""
    (output/'README_summary.md').write_text(readme,encoding='utf-8')
    print('\nIGH V3 CLINICAL LAYER2 F1 SUMMARY PASS: '
          '18 F1 + 18 F0 models / 51 paired comparisons / 5100 matched repeats',flush=True)
    print('OUTPUT_DIR:',output,flush=True)
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f'ERROR: {e}',file=sys.stderr)
        sys.exit(1)
