#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

ROOT=Path('/data/users/chenhaisheng/RA-ILD/TRB'); COLLECT=ROOT/'set/train/result/06_cv_result_summary/collected'; OUT=ROOT/'set/train/result/06_cv_result_summary/visualization'
MODELS=['M0_clinical','M1_static_tcr','M2_static_tcr_public','M3_static_tcr_public_material']
LABELS={'M0_clinical':'M0\nClinical','M1_static_tcr':'M1\nClinical + static TCR','M2_static_tcr_public':'M2\n+ dynamic public','M3_static_tcr_public_material':'M3\n+ material'}
COLORS={'M0_clinical':'#65c8cc','M1_static_tcr':'#f0e94b','M2_static_tcr_public':'#72c15a','M3_static_tcr_public_material':'#f3793b'}
METRICS=[('roc_auc','ROC-AUC'),('pr_auc','PR-AUC'),('sensitivity_recall','Sensitivity'),('specificity','Specificity'),('f1','F1 score')]
PLANNED=[('M0_clinical','M1_static_tcr'),('M1_static_tcr','M2_static_tcr_public'),('M2_static_tcr_public','M3_static_tcr_public_material')]

def args():
 p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter); p.add_argument('--metrics',default=str(COLLECT/'06_all_outer_metrics.csv')); p.add_argument('--predictions',default=str(COLLECT/'06_all_outer_predictions.csv.gz')); p.add_argument('--output-dir',default=str(OUT)); p.add_argument('--dpi',type=int,default=400); p.add_argument('--annotate-pairs',choices=['planned','all'],default='planned'); p.add_argument('--overwrite',action='store_true'); return p.parse_args()

def holm(p):
 p=np.asarray(p,float); order=np.argsort(p); out=np.empty(len(p)); run=0
 for rank,idx in enumerate(order): run=max(run,(len(p)-rank)*p[idx]); out[idx]=min(run,1)
 return out

def wtest(x,y):
 if np.allclose(x-y,0): return 0.,1.
 z=wilcoxon(x,y,zero_method='wilcox',alternative='two-sided',method='auto'); return float(z.statistic),float(z.pvalue)

def repeat_metrics(pred):
 rows=[]
 for (r,m),x in pred.groupby(['outer_repeat','model']):
  if x.sample_id.duplicated().any(): raise ValueError(f'duplicate OOF sample repeat={r}, model={m}')
  y=x.true_label.astype(int).to_numpy(); pr=x.probability_ILD.astype(float).to_numpy(); cl=x.predicted_label.astype(int).to_numpy(); tn,fp,fn,tp=confusion_matrix(y,cl,labels=[0,1]).ravel()
  rows.append({'outer_repeat':r,'model':m,'n_samples':len(x),'roc_auc':roc_auc_score(y,pr),'pr_auc':average_precision_score(y,pr),'sensitivity_recall':recall_score(y,cl,zero_division=0),'specificity':tn/(tn+fp) if tn+fp else np.nan,'precision':precision_score(y,cl,zero_division=0),'f1':f1_score(y,cl,zero_division=0),'accuracy':accuracy_score(y,cl)})
 return pd.DataFrame(rows).sort_values(['outer_repeat','model'])

def tests(rep):
 g=[]; pairs=[]
 for metric,label in METRICS:
  w=rep.pivot(index='outer_repeat',columns='model',values=metric)[MODELS].dropna(); fr=friedmanchisquare(*[w[m].to_numpy() for m in MODELS]); g.append({'metric':metric,'metric_label':label,'n_repeats':len(w),'friedman_statistic':fr.statistic,'friedman_p_value':fr.pvalue})
  rs=[]
  for a,b in itertools.combinations(MODELS,2):
   st,p=wtest(w[a].to_numpy(),w[b].to_numpy()); d=w[b]-w[a]; rs.append({'metric':metric,'metric_label':label,'model_a':a,'model_b':b,'n_repeats':len(w),'median_difference_b_minus_a':d.median(),'mean_difference_b_minus_a':d.mean(),'wins_b_gt_a':int((d>0).sum()),'ties':int((d==0).sum()),'losses_b_lt_a':int((d<0).sum()),'wilcoxon_statistic':st,'p_value_raw':p})
  for row,adj in zip(rs,holm([x['p_value_raw'] for x in rs])): row['p_value_holm']=adj; pairs.append(row)
 return pd.DataFrame(g),pd.DataFrame(pairs)

def fp(p): return f'{p:.1e}' if p<1e-4 else f'{p:.4f}'
def bracket(ax,x1,x2,y,h,text): ax.plot([x1,x1,x2,x2],[y,y+h,y+h,y],lw=.8); ax.text((x1+x2)/2,y+h,text,ha='center',va='bottom',fontsize=7)

def main():
 a=args(); mp=Path(a.metrics); pp=Path(a.predictions); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 paths={'repeat':out/'06_repeat_level_model_metrics.csv','global':out/'06_metric_global_friedman_tests.csv','pair':out/'06_metric_pairwise_wilcoxon_tests.csv','png':out/'06_model_metric_boxplots.png','pdf':out/'06_model_metric_boxplots.pdf'}
 if not a.overwrite:
  ex=[p for p in paths.values() if p.exists()]
  if ex: raise FileExistsError('Outputs exist; use --overwrite')
 fold=pd.read_csv(mp); pred=pd.read_csv(pp); rep=repeat_metrics(pred); glob,pair=tests(rep); rep.to_csv(paths['repeat'],index=False); glob.to_csv(paths['global'],index=False); pair.to_csv(paths['pair'],index=False)
 ann=PLANNED if a.annotate_pairs=='planned' else list(itertools.combinations(MODELS,2))
 fig,axs=plt.subplots(2,3,figsize=(18,10.5)); axs=axs.ravel()
 for i,(metric,title) in enumerate(METRICS):
  ax=axs[i]; arrays=[fold.loc[fold.model==m,metric].dropna().to_numpy() for m in MODELS]
  bp=ax.boxplot(arrays,positions=np.arange(1,5),widths=.58,patch_artist=True,showfliers=False,medianprops={'linewidth':1.4})
  for patch,m in zip(bp['boxes'],MODELS): patch.set_facecolor(COLORS[m]); patch.set_alpha(.85)
  rng=np.random.default_rng(20260711+i)
  for pos,vals in enumerate(arrays,1): ax.scatter(rng.normal(pos,.045,len(vals)),vals,s=7,alpha=.28)
  ax.set_title(title,fontsize=13,fontweight='bold'); ax.set_xticks(range(1,5)); ax.set_xticklabels([LABELS[m] for m in MODELS],fontsize=8); ax.set_ylabel(title); ax.set_ylim(0,1.18); ax.grid(axis='y',alpha=.22)
  gp=float(glob.loc[glob.metric==metric,'friedman_p_value'].iloc[0]); ax.text(.02,.98,f'Friedman p = {fp(gp)}',transform=ax.transAxes,va='top',fontsize=8.5)
  sub=pair[pair.metric==metric]; base=1.005
  for j,(x,y) in enumerate(ann):
   row=sub[(sub.model_a==x)&(sub.model_b==y)]
   if row.empty: row=sub[(sub.model_a==y)&(sub.model_b==x)]
   bracket(ax,MODELS.index(x)+1,MODELS.index(y)+1,base+j*.052,.012,f'p={fp(float(row.p_value_holm.iloc[0]))}')
 axs[5].axis('off'); fig.suptitle('Repeated Nested-CV Performance of RA vs RA-ILD Models',fontsize=17,fontweight='bold',y=.99); fig.text(.5,.015,'Boxes: 100 outer-fold values. P-values: paired tests on 20 pooled repeat-level OOF results; Holm adjusted.',ha='center',fontsize=9); fig.tight_layout(rect=[.02,.04,.98,.965]); fig.savefig(paths['png'],dpi=a.dpi,bbox_inches='tight'); fig.savefig(paths['pdf'],bbox_inches='tight'); plt.close(fig)
 for k,v in paths.items(): print(f'- {k}: {v}')
 return 0
if __name__=='__main__': raise SystemExit(main())
