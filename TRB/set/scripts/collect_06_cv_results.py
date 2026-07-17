#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
# Version 1.0.1: tolerate pre-existing task metadata columns.
import argparse, json, re, sys
from pathlib import Path
import pandas as pd

ROOT=Path('/data/users/chenhaisheng/RA-ILD/TRB')
DEFAULT_INPUT=ROOT/'set/train/result/05_modeling/single_outer_trial_loo'
DEFAULT_OUTPUT=ROOT/'set/train/result/06_cv_result_summary/collected'
MODELS=['M0_clinical','M1_static_tcr','M2_static_tcr_public','M3_static_tcr_public_material']
REQ=['05_trial_configuration.json','05_outer_validation_metrics.csv','05_outer_validation_predictions.csv','05_final_model_coefficients.csv']

def args():
 p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 p.add_argument('--input-root',default=str(DEFAULT_INPUT)); p.add_argument('--output-dir',default=str(DEFAULT_OUTPUT))
 p.add_argument('--expected-repeats',type=int,default=20); p.add_argument('--expected-folds',type=int,default=5); p.add_argument('--expected-samples',type=int,default=123)
 p.add_argument('--allow-incomplete',action='store_true'); p.add_argument('--overwrite',action='store_true'); return p.parse_args()

def b(s): return s if pd.api.types.is_bool_dtype(s) else s.astype(str).str.lower().isin({'true','1','yes'})

def main():
 a=args(); inp=Path(a.input_root).resolve(); out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
 paths={
  'metrics':out/'06_all_outer_metrics.csv','pred':out/'06_all_outer_predictions.csv.gz','coef':out/'06_all_model_coefficients.csv.gz',
  'param':out/'06_all_selected_parameters.csv','qc':out/'06_task_integrity_report.csv','summary':out/'06_collection_summary.md'}
 if not a.overwrite:
  ex=[p for p in paths.values() if p.exists()]
  if ex: raise FileExistsError('Outputs exist; use --overwrite:\n'+'\n'.join(map(str,ex)))
 q=[]; ms=[]; ps=[]; cs=[]; ss=[]
 for r in range(1,a.expected_repeats+1):
  for f in range(1,a.expected_folds+1):
   td=inp/f'repeat_{r:02d}_fold_{f:02d}'; row={'outer_repeat':r,'outer_fold':f,'task_name':td.name,'task_dir':str(td),'status':'FAIL','message':''}
   miss=[x for x in REQ if not (td/x).is_file()]
   if miss: row['message']=f'missing files: {miss}'; q.append(row); continue
   try:
    conf=json.loads((td/'05_trial_configuration.json').read_text()); m=pd.read_csv(td/'05_outer_validation_metrics.csv'); p=pd.read_csv(td/'05_outer_validation_predictions.csv'); c=pd.read_csv(td/'05_final_model_coefficients.csv')
    probs=[]
    if int(conf.get('outer_repeat',-1))!=r or int(conf.get('outer_fold',-1))!=f: probs.append('configuration mismatch')
    if set(m.get('model',[]))!=set(MODELS) or len(m)!=4 or m.duplicated(['model']).any(): probs.append('metrics model rows invalid')
    if 'fit_converged' not in m or not b(m['fit_converged']).all(): probs.append('final fit not converged')
    needp={'model','sample_id','true_label','probability_ILD','predicted_label'}
    if not needp.issubset(p): probs.append(f'predictions missing {sorted(needp-set(p))}')
    else:
     if set(p['model'])!=set(MODELS) or p.duplicated(['model','sample_id']).any(): probs.append('predictions invalid')
     sets=[set(x['sample_id'].astype(str)) for _,x in p.groupby('model')]
     if len(sets)!=4 or any(x!=sets[0] for x in sets[1:]): probs.append('prediction sample sets differ')
    needc={'model','feature_name','coefficient','nonzero'}
    if not needc.issubset(c): probs.append(f'coefficients missing {sorted(needc-set(c))}')
    elif set(c['model'])!=set(MODELS) or c.duplicated(['model','feature_name']).any(): probs.append('coefficients invalid')
    sel=conf.get('selected_parameters',{}); sr=[]
    if set(sel)!=set(MODELS): probs.append('selected parameters invalid')
    else:
     for model in MODELS:
      x=sel[model]; sr.append({'outer_repeat':r,'outer_fold':f,'task_name':td.name,'model':model,'l1_ratio_alpha':x.get('alpha',x.get('l1_ratio')),'lambda':x.get('lambda'),'threshold':x.get('threshold'),'inner_roc_auc':x.get('inner_roc_auc'),'inner_pr_auc':x.get('inner_pr_auc')})
    if probs: row['message']='; '.join(probs); q.append(row); continue
    def add_or_validate_task_columns(df):
     checks=[('outer_repeat',r),('outer_fold',f),('task_name',td.name)]
     for col,val in checks:
      if col in df.columns:
       if col in {'outer_repeat','outer_fold'}:
        observed=pd.to_numeric(df[col],errors='coerce')
        if observed.isna().any() or not (observed.astype(int)==int(val)).all():
         raise ValueError(f'{col} values do not match task directory')
        df[col]=observed.astype(int)
       else:
        if not (df[col].astype(str)==str(val)).all():
         raise ValueError(f'{col} values do not match task directory')
      else:
       df.insert(0,col,val)
     ordered=['outer_repeat','outer_fold','task_name']
     return df[ordered+[x for x in df.columns if x not in ordered]]

    m=add_or_validate_task_columns(m)
    p=add_or_validate_task_columns(p)
    c=add_or_validate_task_columns(c)
    row.update(status='PASS',message='all task checks passed',n_validation_samples=int(p.groupby('model')['sample_id'].nunique().iloc[0]),metric_rows=len(m),prediction_rows=len(p),coefficient_rows=len(c))
    q.append(row); ms.append(m); ps.append(p); cs.append(c); ss.append(pd.DataFrame(sr))
   except Exception as e:
    row['message']=f'read/validation error: {e}'; q.append(row)
 qc=pd.DataFrame(q).sort_values(['outer_repeat','outer_fold'])
 metrics=pd.concat(ms,ignore_index=True) if ms else pd.DataFrame(); pred=pd.concat(ps,ignore_index=True) if ps else pd.DataFrame(); coef=pd.concat(cs,ignore_index=True) if cs else pd.DataFrame(); param=pd.concat(ss,ignore_index=True) if ss else pd.DataFrame()
 if not pred.empty:
  if pred.duplicated(['outer_repeat','outer_fold','model','sample_id']).any(): raise RuntimeError('duplicate global prediction rows')
  counts=pred.groupby(['outer_repeat','model'])['sample_id'].nunique()
  if not a.allow_incomplete and not (counts==a.expected_samples).all(): raise RuntimeError(f'repeat/model OOF counts invalid: {counts[counts!=a.expected_samples].to_dict()}')
 qc.to_csv(paths['qc'],index=False); metrics.to_csv(paths['metrics'],index=False); pred.to_csv(paths['pred'],index=False,compression='gzip'); coef.to_csv(paths['coef'],index=False,compression='gzip'); param.to_csv(paths['param'],index=False)
 passed=int((qc.status=='PASS').sum()); failed=int((qc.status=='FAIL').sum())
 lines=['# 06 CV Result Collection Summary','',f'- Expected tasks: **{a.expected_repeats*a.expected_folds}**',f'- Passed tasks: **{passed}**',f'- Failed tasks: **{failed}**',f'- Metrics rows: **{len(metrics):,}**',f'- Prediction rows: **{len(pred):,}**',f'- Coefficient rows: **{len(coef):,}**',f'- Parameter rows: **{len(param):,}**','','## Failed tasks','']
 bad=qc[qc.status=='FAIL']; lines += ['- None'] if bad.empty else [f"- `{x.task_name}`: {x.message}" for _,x in bad.iterrows()]
 paths['summary'].write_text('\n'.join(lines)+'\n')
 print(f'Passed tasks: {passed}\nFailed tasks: {failed}\nMetrics rows: {len(metrics):,}\nPrediction rows: {len(pred):,}\nCoefficient rows: {len(coef):,}\nParameter rows: {len(param):,}')
 for k,v in paths.items(): print(f'- {k}: {v}')
 return 1 if failed and not a.allow_incomplete else 0
if __name__=='__main__': sys.exit(main())
