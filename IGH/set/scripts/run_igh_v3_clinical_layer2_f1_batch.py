#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run ONLY 18 locked IGH V3 clinical Layer2 F1 jobs (Top5+S000 each scope).

Requires build_igh_v3_clinical_layer2_f1_18.py --write first. Never overwrites
training results, ignores SIGHUP, stops dispatch on any failure. Preserves all
existing IGH-only / clinical Layer1 / frozen splits. Python >=3.9; pandas, yaml.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

import pandas as pd
import yaml
import run_igh_v3_clinical_layer1_batch as layer1
import build_igh_v3_clinical_layer2_f1_18 as builder

REPO = layer1.REPO
VER = layer1.VER
PLAN = VER/'analysis/clinical_layer2_f1_18_plan_v3_175.json'
MANAGEMENT = VER/'runs/clinical_layer2_f1_batch_v3_175'
SCOPES = ('total','pbmc','buffycoat')
REQUIRED = layer1.REQUIRED
STOP = False
STOP_AT = None
ACTIVE = {}


def require(ok,msg):
    if not ok:
        raise RuntimeError(msg)


def time_utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def config_dir(scope):
    return layer1.CFG_ROOT/f'igh_clinical_layer2_{scope}_top10000_v3_175'


def output_dir(scope,stem):
    return VER/'runs'/f'clinical_layer2_{scope}_top10000'/stem/'repeat100'


def status_path(scope,stem):
    return MANAGEMENT/'status'/scope/f'{stem}.json'


def log_path(scope,stem):
    return MANAGEMENT/'logs'/scope/f'{stem}.log'


def plan_check():
    require(PLAN.is_file(),f'Locked 18-config clinical Layer2 plan missing; first run builder --write: {PLAN}')
    obj=json.loads(PLAN.read_text(encoding='utf-8'))
    require(obj.get('decision')==builder.SCHEME and len(obj.get('configs',[]))==18,
            'Layer2 plan is not the 18-F1 design')
    require(obj.get('clinical_layer1_plan_sha256')==layer1.sha256(layer1.PLAN),
            'clinical Layer1 locked plan modified')
    summary_dir=Path(obj.get('clinical_layer1_summary_dir',''))
    require(summary_dir.is_dir(),f'locked clinical Layer1 summary missing: {summary_dir}')
    for name,expected in obj['clinical_layer1_summary_file_sha256'].items():
        require(layer1.sha256(summary_dir/name)==expected,
                f'clinical Layer1 summary changed: {summary_dir/name}')
    chosen, hashes, audit=builder.choose(summary_dir)
    require(hashes==obj['clinical_layer1_summary_file_sha256'],
            'source clinical Layer1 summary file mismatch')
    require(audit.get('locked_plan_sha256')==obj['clinical_layer1_plan_sha256'],
            'source clinical Layer1 summary plan mismatch')
    plan1=json.loads(layer1.PLAN.read_text(encoding='utf-8'))
    require(plan1.get('decision')=='IGH_V3_CLINICAL_LAYER1_AGE_SEX_ALL128' and
            len(plan1.get('configs',[]))==384,'source clinical Layer1 plan invalid')
    contexts={}
    jobs=[]
    allentries={}
    for scope in SCOPES:
        ctx=layer1.preflight_scope(scope,[x for x in plan1['configs'] if x['scope']==scope],
                                   plan1['frozen_split_sha256'][scope])
        contexts[scope]=ctx
        require(obj['frozen_split_sha256'][scope]==ctx['split_sha'],f'bank hash differs: {scope}')
        manpath=config_dir(scope)/'CLINICAL_LAYER2_F1_MANIFEST.json'
        require(manpath.is_file(),f'missing F1 manifest: {manpath}')
        saved=json.loads(manpath.read_text(encoding='utf-8'))
        entries=[x for x in obj['configs'] if x['scope']==scope]
        require(len(entries)==len(saved)==6 and entries==saved,
                f'6-config plan/manifest mismatch: {scope}')
        expected_names=['S000_none_F1']+[x+'_F1' for x in chosen[scope]['stem'].tolist()]
        require([e['stem'] for e in entries]==expected_names,
                f'clinical Layer2 Top5 drift: {scope}')
        require({p.stem for p in config_dir(scope).glob('S*.yaml')}==set(expected_names),
                f'unexpected config files: {scope}')
        for e in entries:
            stem=e['stem'];oldstem=e['layer1_stem']
            path=config_dir(scope)/(stem+'.yaml')
            oldpath=layer1.config_dir(scope)/(oldstem+'.yaml')
            require(path.is_file() and oldpath.is_file(),f'YAML absent: {scope}/{stem}')
            require(layer1.sha256(path)==e['config_sha256'] and
                    layer1.sha256(oldpath)==e['clinical_layer1_source_sha256'],
                    f'YAML hash mismatch: {scope}/{stem}')
            config=yaml.safe_load(path.read_text(encoding='utf-8'))
            old=yaml.safe_load(oldpath.read_text(encoding='utf-8'))
            expected=copy.deepcopy(old)
            expected['experiment']['id']=f'igh_clinical_layer2_{scope}_top10000_{stem}_v3_175'
            expected['features']['kmer']['enabled']=True
            require(config==expected,f'Layer2 configuration changed more than experiment.id and 3-mer flag: {scope}/{stem}')
            require(config['features']['clinical_groups']==['clinical_age_sex'] and
                    not config['features']['enriched_dictionary']['enabled'],
                    f'clinical not on or enriched on: {scope}/{stem}')
            require(config['features']['kmer']['representation']=='unweighted' and
                    config['features']['kmer']['top_n']==500,
                    f'not Top500 unweighted 3mer: {scope}/{stem}')
            if oldstem=='S000_none':
                require(not config['features']['static_tcr_groups'],f'3mer-only control has IGH static features: {scope}')
            require(e['frozen_split_sha256']==ctx['split_sha'],f'frozen bank differ: {scope}/{stem}')
            jobs.append((scope,stem,path))
            allentries[(scope,stem)]=e
    require(len(jobs)==18,'not 18 jobs')
    return contexts,jobs,allentries


def result_check(scope,stem,ctx):
    out=output_dir(scope,stem)
    if not out.is_dir():
        return False,'production output directory absent'
    try:
        require(all((out/f).is_file() for f in REQUIRED),
                f'missing outputs: {[f for f in REQUIRED if not (out/f).is_file()]}')
        m=pd.read_csv(out/'06_outer_metrics.csv')
        p=pd.read_csv(out/'06_outer_predictions.csv.gz',dtype={'sample_id':str})
        s=pd.read_csv(out/'06_selected_hyperparameters.csv')
        t=pd.read_csv(out/'06_inner_tuning.csv.gz')
        a=pd.read_csv(out/'06_feature_fit_audit.csv.gz')
        manifest=json.loads((out/'06_run_manifest.json').read_text(encoding='utf-8'))
        require(len(m)==len(s)==100 and len(t)==2400, '100 metric/selected rows or 2400 tuning rows absent')
        require(len(p)==100*ctx['holdout_n'],'wrong number of holdout prediction rows')
        for frame,label in ((m,'metrics'),(s,'selected'),(p,'predictions')):
            reps=layer1.as_int(frame.outer_repeat)
            require(set(reps)==set(range(1,101)),f'repeats incomplete: {label}')
        require((layer1.as_int(m.n_outer_train)==ctx['train_n']).all() and
                (layer1.as_int(m.n_outer_holdout)==ctx['holdout_n']).all(), 'split sample counts wrong')
        require(m.fit_converged.astype(str).str.lower().eq('true').all(), 'unconverged final fit')
        for r,part in p.groupby('outer_repeat'):
            ids=part.sample_id.astype(str)
            require(ids.is_unique and set(ids)==ctx['holdouts'][int(r)],f'holdout ID mismatch: {r}')
            require(part.true_cohort.astype(str).tolist()==[ctx['cohort'][sid] for sid in ids],
                    f'cohort mismatch: repeat={r}')
        exe=manifest.get('execution',{})
        require(exe.get('repeat_count')==100 and exe.get('candidate_count')==24,
                'manifest count mismatch')
        msplit=Path(manifest.get('inputs',{}).get('split_assignments',''))
        require((msplit if msplit.is_absolute() else REPO/msplit).resolve()==ctx['split_path'].resolve(),
                'output not from expected frozen bank')
        require({'outer_repeat','feature_family','fit_policy'}<=set(a.columns), 'audit columns missing')
        k=a[a.feature_family.astype(str)=='3mer']
        require(len(k)>=100 and set(layer1.as_int(k.outer_repeat))==set(range(1,101)),
                'missing 3mer repeated fold fits')
        require(k.fit_policy.astype(str).eq('fit_vocabulary_on_current_training_only').all(),
                '3mer vocabulary fitted outside training partitions')
        require(not a.feature_family.astype(str).eq('enriched_dictionary').any(),
                'unexpected enriched dictionary fit')
        sta=a[a.feature_family.astype(str)=='static_and_clinical']
        require(len(sta)>=100 and set(layer1.as_int(sta.outer_repeat))==set(range(1,101)),
                'static + clinical fitting not audited every repeat')
        return True,'100 repeats, frozen IDs, age+sex, fold-fit 3mer audits, manifest PASS'
    except Exception as exc:
        return False,f'validation error: {exc}'


def stop(signum,_frame):
    global STOP,STOP_AT
    if STOP:return
    STOP=True;STOP_AT=time.monotonic()
    print(f'\nSIGNAL {signum}: stop dispatch; terminate active children',flush=True)
    for j in list(ACTIVE.values()):
        proc=j['proc']
        if proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass


def launch(job,workers):
    scope,stem,cfg=job
    out=output_dir(scope,stem)
    require(not out.exists(),f'refuse overwrite: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    log=log_path(scope,stem)
    log.parent.mkdir(parents=True,exist_ok=True)
    c=layer1.compat_dir(scope)
    cmd=[sys.executable,str(layer1.MODEL),'--config',str(cfg),
         '--train-metadata',str(c/'metadata_train.csv'),
         '--test-metadata',str(c/'metadata_test.csv'),
         '--train-step02-dir',str(c/'train_step02'),
         '--test-step02-dir',str(c/'test_step02'),
         '--registry',str(layer1.REGISTRY),
         '--split-assignments',str(layer1.split_path(scope)),
         '--output-dir',str(out),'--workers',str(workers)]
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        env[key]='1'
    with log.open('w',encoding='utf-8') as f:
        f.write(f'Started UTC: {time_utc()}\nCommand: {json.dumps(cmd,ensure_ascii=False)}\n\n')
        f.flush()
        proc=subprocess.Popen(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,
                              start_new_session=True,env=env)
    atomic_json(status_path(scope,stem),{'state':'RUNNING','scope':scope,'name':stem,
        'pid':proc.pid,'started_utc':time_utc(),'output':str(out),'log':str(log)})
    print(f'START {scope}/{stem} pid={proc.pid} workers={workers} log={log}',flush=True)
    return {'proc':proc,'scope':scope,'stem':stem,'log':log}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scope',choices=['all',*SCOPES],default='all')
    ap.add_argument('--name',action='append',default=[],help='Exact name e.g. S007_PW_F1')
    ap.add_argument('--limit',type=int,default=1,help='Max new jobs; 0=all')
    ap.add_argument('--max-concurrent',type=int,default=1)
    ap.add_argument('--workers-per-experiment',type=int,default=1)
    ap.add_argument('--execute',action='store_true')
    args=ap.parse_args()
    require(args.limit>=0,'--limit must be nonnegative')
    require(1<=args.max_concurrent<=6 and 1<=args.workers_per_experiment<=8 and
            args.max_concurrent*args.workers_per_experiment<=48,
            'parallelism: max-concurrent 1..6, workers per experiment 1..8, total <=48')
    require(layer1.MODEL.is_file() and layer1.REGISTRY.is_file(),'model or registry missing')
    ctxs,alljobs,entries=plan_check()
    scopes=list(SCOPES) if args.scope=='all' else [args.scope]
    jobs=[job for job in alljobs if job[0] in scopes]
    if args.name:
        unknown=set(args.name)-{job[1] for job in jobs}
        require(not unknown,f'unknown --name: {sorted(unknown)}')
        jobs=[job for job in jobs if job[1] in args.name]
    states=[]
    for scope,stem,cfg in jobs:
        out=output_dir(scope,stem)
        if out.exists():
            valid,detail=result_check(scope,stem,ctxs[scope])
            states.append(('PASS' if valid else 'BLOCKED',detail,(scope,stem,cfg)))
        else:
            states.append(('READY','output absent',(scope,stem,cfg)))
    counts=Counter(x[0] for x in states)
    todo=[j for state,_,j in states if state=='READY']
    if args.limit:todo=todo[:args.limit]
    print(f'REPO: {REPO}',flush=True)
    print('CLINICAL LAYER2 F1: Age+Sex always ON; Top5 static+3mer plus S000+3mer controls; no enriched dictionary',flush=True)
    print('Frozen banks: '+', '.join(s+'='+ctxs[s]['split_sha'][:12] for s in scopes),flush=True)
    print(f'Selected configs: {len(jobs)} (18 when --scope all)',flush=True)
    print(f'Status: {dict(counts)}; next jobs: {len(todo)}; mode: {"EXECUTE" if args.execute else "PREFLIGHT ONLY"}',flush=True)
    for state,detail,job in states:
        if state=='BLOCKED':
            print(f'BLOCKED {job[0]}/{job[1]}: {detail}',flush=True)
    for job in todo[:18]:print(f'NEXT {job[0]}/{job[1]}',flush=True)
    if not args.execute:
        print('PREFLIGHT PASS (read-only). Add --execute to train.',flush=True)
        return 0
    if counts.get('BLOCKED',0):
        print('REFUSE EXECUTE: incomplete/corrupt output must be investigated; never overwrite',flush=True)
        return 2
    if not todo:
        print('NOTHING TO DO: already PASS',flush=True)
        return 0
    MANAGEMENT.mkdir(parents=True,exist_ok=True)
    with (MANAGEMENT/'.clinical_layer2_f1.lock').open('a+') as handle:
        try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('REFUSE EXECUTE: concurrent runner',flush=True)
            return 3
        signal.signal(signal.SIGHUP,signal.SIG_IGN)
        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
        queue=[]
        for job in todo:
            sc,st,cf=job
            if output_dir(sc,st).exists():
                valid,detail=result_check(sc,st,ctxs[sc])
                if valid:print(f'SKIP newly PASS {sc}/{st}',flush=True);continue
                print(f'REFUSE newly BLOCKED {sc}/{st}: {detail}',flush=True);return 2
            queue.append(job)
        passed=0;failed=0
        while (queue and not STOP and not failed) or ACTIVE:
            while queue and not STOP and not failed and len(ACTIVE)<args.max_concurrent:
                job=queue.pop(0)
                ACTIVE[job[0]+'/'+job[1]]=launch(job,args.workers_per_experiment)
            time.sleep(1)
            for key,item in list(ACTIVE.items()):
                rc=item['proc'].poll()
                if rc is None:continue
                scope,stem=item['scope'],item['stem']
                valid,detail=(result_check(scope,stem,ctxs[scope]) if rc==0
                              else (False,f'process return code={rc}'))
                state='PASS' if valid and not STOP else ('INTERRUPTED' if STOP else 'FAIL')
                atomic_json(status_path(scope,stem),{'state':state,'scope':scope,'name':stem,
                    'pid':item['proc'].pid,'return_code':rc,'checked_utc':time_utc(),
                    'detail':detail,'output':str(output_dir(scope,stem)),
                    'log':str(item['log'])})
                print(f'{state} {key}: {detail} (rc={rc})',flush=True)
                if state=='PASS':passed+=1
                else:
                    failed+=1
                    print(f'Inspect log: tail -n 50 {item["log"]}',flush=True)
                del ACTIVE[key]
            if STOP and ACTIVE and STOP_AT is not None and time.monotonic()-STOP_AT>20:
                for item in list(ACTIVE.values()):
                    proc=item['proc']
                    if proc.poll() is None:
                        try:os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
        print(f'BATCH FINISHED: PASS={passed}, FAIL/INTERRUPTED={failed}, unstarted={len(queue)}',flush=True)
        return 130 if STOP else (1 if failed else 0)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as ex:
        print(f'ERROR: {ex}',file=sys.stderr,flush=True)
        sys.exit(1)
