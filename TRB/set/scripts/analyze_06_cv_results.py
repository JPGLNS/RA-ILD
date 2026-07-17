#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

ROOT=Path('/data/users/chenhaisheng/RA-ILD/TRB'); COLLECT=ROOT/'set/train/result/06_cv_result_summary/collected'; OUT=ROOT/'set/train/result/06_cv_result_summary/analysis'
MODELS=['M0_clinical','M1_static_tcr','M2_static_tcr_public','M3_static_tcr_public_material']
METRICS=['roc_auc','pr_auc','sensitivity_recall','specificity','f1']

def args():
 p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter); p.add_argument('--predictions',default=str(COLLECT/'06_all_outer_predictions.csv.gz')); p.add_argument('--coefficients',default=str(COLLECT/'06_all_model_coefficients.csv.gz')); p.add_argument('--parameters',default=str(COLLECT/'06_all_selected_parameters.csv')); p.add_argument('--output-dir',default=str(OUT)); p.add_argument('--bootstrap-reps',type=int,default=5000); p.add_argument('--seed',type=int,default=20260711); p.add_argument('--stable-selection-frequency',type=float,default=.50); p.add_argument('--stable-sign-consistency',type=float,default=.80); p.add_argument('--overwrite',action='store_true'); return p.parse_args()

def holm(p):
 p=np.asarray(p,float); order=np.argsort(p); out=np.empty(len(p)); run=0
 for rank,idx in enumerate(order): run=max(run,(len(p)-rank)*p[idx]); out[idx]=min(run,1)
 return out

def wtest(x,y):
 if np.allclose(x-y,0): return 0.,1.
 z=wilcoxon(x,y,zero_method='wilcox',alternative='two-sided'); return float(z.statistic),float(z.pvalue)

def repeat_metrics(pred):
 rows=[]
 for (r,m),x in pred.groupby(['outer_repeat','model']):
  if x.sample_id.duplicated().any(): raise ValueError(f'duplicate OOF sample repeat={r}, model={m}')
  y=x.true_label.astype(int).to_numpy(); pr=x.probability_ILD.astype(float).to_numpy(); cl=x.predicted_label.astype(int).to_numpy(); tn,fp,fn,tp=confusion_matrix(y,cl,labels=[0,1]).ravel()
  rows.append({'outer_repeat':r,'model':m,'n_samples':len(x),'roc_auc':roc_auc_score(y,pr),'pr_auc':average_precision_score(y,pr),'sensitivity_recall':recall_score(y,cl,zero_division=0),'specificity':tn/(tn+fp) if tn+fp else np.nan,'precision':precision_score(y,cl,zero_division=0),'f1':f1_score(y,cl,zero_division=0),'accuracy':accuracy_score(y,cl)})
 return pd.DataFrame(rows).sort_values(['outer_repeat','model'])

def ci(vals,reps,rng):
 vals=np.asarray(vals,float); means=rng.choice(vals,size=(reps,len(vals)),replace=True).mean(axis=1); return np.quantile(means,[.025,.975])

def perf_summary(rep,reps,seed):
 rng=np.random.default_rng(seed); rows=[]
 for m in MODELS:
  for metric in METRICS:
   v=rep.loc[rep.model==m,metric].dropna().to_numpy(float); lo,hi=ci(v,reps,rng); rows.append({'model':m,'metric':metric,'n_repeats':len(v),'mean':v.mean(),'sd':v.std(ddof=1),'median':np.median(v),'q1':np.quantile(v,.25),'q3':np.quantile(v,.75),'min':v.min(),'max':v.max(),'mean_bootstrap_ci95_lower':lo,'mean_bootstrap_ci95_upper':hi})
 return pd.DataFrame(rows)

def pairwise(rep):
 rows=[]
 for metric in METRICS:
  w=rep.pivot(index='outer_repeat',columns='model',values=metric)[MODELS].dropna(); rs=[]
  for a,b in itertools.combinations(MODELS,2):
   st,p=wtest(w[a].to_numpy(),w[b].to_numpy()); d=w[b]-w[a]; rs.append({'metric':metric,'model_a':a,'model_b':b,'n_repeats':len(w),'mean_a':w[a].mean(),'mean_b':w[b].mean(),'mean_difference_b_minus_a':d.mean(),'median_difference_b_minus_a':d.median(),'q1_difference':d.quantile(.25),'q3_difference':d.quantile(.75),'wins_b_gt_a':int((d>0).sum()),'ties':int((d==0).sum()),'losses_b_lt_a':int((d<0).sum()),'wilcoxon_statistic':st,'p_value_raw':p})
  for row,adj in zip(rs,holm([x['p_value_raw'] for x in rs])): row['p_value_holm']=adj; rows.append(row)
 return pd.DataFrame(rows)

def stability(coef,expected):
 c=coef.copy(); c['nonzero']=c['nonzero'] if pd.api.types.is_bool_dtype(c['nonzero']) else c['nonzero'].astype(str).str.lower().isin({'true','1','yes'}); c=c[c.feature_name!='__INTERCEPT__']; c['coefficient']=pd.to_numeric(c.coefficient); c['absolute_coefficient']=c.coefficient.abs(); rows=[]
 for (m,f),x in c.groupby(['model','feature_name']):
  s=x[x.nonzero & (x.absolute_coefficient>1e-12)]; pos=int((s.coefficient>0).sum()); neg=int((s.coefficient<0).sum()); n=len(s); cons=max(pos,neg)/n if n else np.nan; direction='positive' if pos>neg else 'negative' if neg>pos else 'tie_or_none'
  rows.append({'model':m,'feature_name':f,'n_outer_tasks_expected':expected,'n_outer_tasks_observed':x[['outer_repeat','outer_fold']].drop_duplicates().shape[0],'selected_count':n,'selection_frequency':n/expected,'positive_count':pos,'negative_count':neg,'sign_consistency':cons,'predominant_direction':direction,'coefficient_median_selected':s.coefficient.median() if n else 0.,'coefficient_q1_selected':s.coefficient.quantile(.25) if n else 0.,'coefficient_q3_selected':s.coefficient.quantile(.75) if n else 0.,'absolute_coefficient_median_selected':s.absolute_coefficient.median() if n else 0.,'absolute_coefficient_mean_selected':s.absolute_coefficient.mean() if n else 0.})
 return pd.DataFrame(rows).sort_values(['model','selection_frequency','sign_consistency','absolute_coefficient_median_selected'],ascending=[True,False,False,False])

def main():
 a=args(); predp=Path(a.predictions); coefp=Path(a.coefficients); paramp=Path(a.parameters); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 paths={'repeat':out/'06_repeat_level_model_metrics.csv','perf':out/'06_model_performance_summary.csv','pair':out/'06_model_pairwise_comparisons.csv','stab':out/'06_feature_stability_by_model.csv','stable':out/'06_stable_features_filtered.csv','param':out/'06_parameter_selection_summary.csv','summary':out/'06_analysis_summary.md'}
 if not a.overwrite:
  ex=[p for p in paths.values() if p.exists()]
  if ex: raise FileExistsError('Outputs exist; use --overwrite')
 pred=pd.read_csv(predp); coef=pd.read_csv(coefp); param=pd.read_csv(paramp); rep=repeat_metrics(pred); perf=perf_summary(rep,a.bootstrap_reps,a.seed); pair=pairwise(rep); expected=param[['outer_repeat','outer_fold']].drop_duplicates().shape[0]; stab=stability(coef,expected); stable=stab[(stab.selection_frequency>=a.stable_selection_frequency)&(stab.sign_consistency>=a.stable_sign_consistency)].copy(); ps=param.groupby(['model','l1_ratio_alpha','lambda']).size().reset_index(name='selection_count'); totals=param.groupby('model').size().rename('n_tasks').reset_index(); ps=ps.merge(totals,on='model'); ps['selection_frequency']=ps.selection_count/ps.n_tasks; ps=ps.sort_values(['model','selection_count'],ascending=[True,False])
 rep.to_csv(paths['repeat'],index=False); perf.to_csv(paths['perf'],index=False); pair.to_csv(paths['pair'],index=False); stab.to_csv(paths['stab'],index=False); stable.to_csv(paths['stable'],index=False); ps.to_csv(paths['param'],index=False)
 lines=['# 06 Repeated-CV Model and Feature Stability Summary','','## Statistical unit','','- Performance summaries and paired tests use 20 pooled repeat-level OOF results.','- Feature stability uses 100 outer fitted models per model specification.','- Positive coefficients indicate higher predicted probability of ILD; negative coefficients indicate RA direction.','','## Mean repeat-level ROC-AUC','']
 for _,x in perf[perf.metric=='roc_auc'].sort_values('model').iterrows(): lines.append(f"- `{x.model}`: {x['mean']:.4f} (95% bootstrap CI {x.mean_bootstrap_ci95_lower:.4f}–{x.mean_bootstrap_ci95_upper:.4f})")
 lines += ['','## Stable-feature rule','',f'- Selection frequency ≥ **{a.stable_selection_frequency:.2f}**',f'- Sign consistency ≥ **{a.stable_sign_consistency:.2f}**','','## Stable features','']
 lines += ['- None met the prespecified rule.'] if stable.empty else [f"- `{x.model}` / `{x.feature_name}`: selected {x.selection_frequency:.1%}, direction {x.predominant_direction}, sign consistency {x.sign_consistency:.1%}" for _,x in stable.iterrows()]
 paths['summary'].write_text('\n'.join(lines)+'\n')
 print(f'Repeat-level metric rows: {len(rep)}\nFeature stability rows: {len(stab)}\nStable feature rows: {len(stable)}')
 for k,v in paths.items(): print(f'- {k}: {v}')
 return 0
if __name__=='__main__': raise SystemExit(main())
