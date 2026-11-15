#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safety-first IGH V3 Clinical Layer1 batch: Age+Sex + S000..S127, x3 scopes.

Read-only unless --execute. Requires immutable 384-config plan written by
build_igh_v3_clinical_layer1_384.py. Does not touch IGH-only or Layer2 outputs.
100 repeats per experiment are atomic: incomplete output is BLOCKED, NEVER overwritten.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
VER = REPO / 'IGH/set/data_versions/igh_v3_175'
CFG_ROOT = REPO / 'IGH/set/configs/experiments'
SPLIT_ROOT = REPO / 'IGH/set/split_sets'
REGISTRY = REPO / 'IGH/set/configs/feature_framework/igh_feature_registry_v3_175.yaml'
MODEL = REPO / 'IGH/set/scripts/run_06_unified_ml.py'
PLAN = VER / 'analysis/clinical_layer1_384_plan_v3_175.json'
MANAGEMENT = VER / 'runs/clinical_layer1_batch_v3_175'
SCOPES = {
    'total': ('igh_ra_ild_repeat100_v3_175', '455eba17c74a86667fd2939dc8645ac1c50f0cdb25053ab951dccad0ab8f1788', 175, 122, 53),
    'pbmc': ('igh_ra_ild_pbmc_repeat100_v3_175', '4726d9b48301c4cd535369f9b39f6bdf3c6247c0722b9c9663385a6c041a68b8', 113, 79, 34),
    'buffycoat': ('igh_ra_ild_buffycoat_repeat100_v3_175', 'da1e598559dc734c0e0296bd1f99620351c2d81e903100c85fda23b670047c05', 62, 43, 19),
}
GROUPS = {
    'tcr_diversity', 'tcr_clonal_expansion', 'tcr_aa_length',
    'tcr_aa_composition_unweighted', 'tcr_aa_composition_weighted',
    'tcr_physicochemical_unweighted', 'tcr_physicochemical_weighted',
}
REQUIRED = (
    '06_outer_metrics.csv', '06_outer_predictions.csv.gz', '06_inner_tuning.csv.gz',
    '06_selected_hyperparameters.csv', '06_feature_explanations.csv.gz',
    '06_feature_fit_audit.csv.gz', '06_preprocessing_audit.csv.gz',
    '06_inner_assignment_audit.csv.gz', '06_run_manifest.json', '06_run_summary.md',
)
STOP = False
STOP_AT = None
ACTIVE = {}


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as inp:
        for b in iter(lambda: inp.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def json_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def split_path(scope):
    return SPLIT_ROOT / SCOPES[scope][0] / 'repeated_holdout_assignments.csv'


def old_config_dir(scope):
    return CFG_ROOT / f'igh_layer1_{scope}_top10000_v3_175'


def config_dir(scope):
    return CFG_ROOT / f'igh_clinical_layer1_{scope}_top10000_v3_175'


def compat_dir(scope):
    return VER / 'ml_compat' / scope


def output_dir(scope, stem):
    return VER / 'runs' / f'clinical_layer1_{scope}_top10000' / stem / 'repeat100'


def log_path(scope, stem):
    return MANAGEMENT / 'logs' / scope / f'{stem}.log'


def status_path(scope, stem):
    return MANAGEMENT / 'status' / scope / f'{stem}.json'


def as_int(series):
    return pd.to_numeric(series, errors='raise').astype(int)


def preflight_scope(scope, plan_entries, plan_hash):
    bankname, expected_hash, n, train_n, holdout_n = SCOPES[scope]
    bank = split_path(scope)
    marker = bank.parent / 'SPLITS_FROZEN.json'
    require(bank.is_file() and marker.is_file(), f'frozen files absent: {scope}')
    require(sha256(bank) == expected_hash == plan_hash, f'frozen sha mismatch: {scope}')
    frozen = json.loads(marker.read_text(encoding='utf-8'))
    require(frozen.get('status') == 'FROZEN' and frozen.get('files', {}).get('assignments', {}).get('sha256') == expected_hash,
            f'frozen marker mismatch: {scope}')
    df = pd.read_csv(bank, dtype={'sample_id': str})
    require({'sample_id', 'repeat_index', 'role'} <= set(df.columns), f'split columns absent: {scope}')
    df['repeat_index'] = as_int(df.repeat_index)
    require(set(df.repeat_index) == set(range(1, 101)), f'not 100 splits: {scope}')
    holdouts = {}
    r1 = None
    ids = None
    for rep, part in df.groupby('repeat_index'):
        require(len(part) == n and part.sample_id.is_unique, f'split size/duplicate: {scope} repeat={rep}')
        here = set(part.sample_id.astype(str))
        if ids is None:
            ids = here
        require(ids == here, f'IDs vary by repeat: {scope} repeat={rep}')
        require(part.role.value_counts().to_dict() == {'train': train_n, 'holdout': holdout_n},
                f'train/holdout sizes unexpected: {scope} repeat={rep}')
        holdouts[int(rep)] = set(part.loc[part.role == 'holdout', 'sample_id'])
        if rep == 1:
            r1 = {role: set(part.loc[part.role == role, 'sample_id']) for role in ('train', 'holdout')}
    for sid in ({'total': ['MGI260201R01-24'], 'pbmc': ['MGI260201R01-23', 'MGI260201R01-24'],
                 'buffycoat': ['MGI260202N01-31']}[scope]):
        require(all(sid not in h for h in holdouts.values()), f'singleton in holdout: {scope} {sid}')
    clinical = {}
    compat = compat_dir(scope)
    for part, role in (('train','train'), ('test','holdout')):
        mpath = compat / f'metadata_{part}.csv'
        cpath = compat / f'{part}_step02/02_core_sample_features.csv'
        provenance = compat / f'{part}_step02/02_resolved_feature_source.json'
        require(all(x.is_file() for x in (mpath, cpath, provenance)), f'ml_compat missing: {scope} {part}')
        meta = pd.read_csv(mpath, dtype={'libraryid': str})
        core = pd.read_csv(cpath, usecols=['sample_id'], dtype={'sample_id': str})
        require({'libraryid', 'cohort', 'age', 'sex'} <= set(meta.columns), f'clinical columns missing: {mpath}')
        require(meta.libraryid.is_unique and core.sample_id.is_unique and
                set(meta.libraryid) == set(core.sample_id) == r1[role], f'ml_compat ID mismatch: {scope} {part}')
        require(meta.cohort.isin(['RA', 'ILD']).all(), f'bad cohort: {mpath}')
        ages = pd.to_numeric(meta.age, errors='coerce')
        require(ages.notna().all() and ages.between(1, 119).all(), f'bad ages: {mpath}')
        sex = meta.sex.astype('string').str.strip().str.lower()
        require(sex.notna().all() and sex.isin(['female', 'male']).all(), f'bad sex: {mpath}')
        prov = json.loads(provenance.read_text(encoding='utf-8'))
        require(prov.get('repertoire_input', {}).get('source_id') == 'top10000', f'wrong data origin: {scope} {part}')
        clinical.update(dict(zip(meta.libraryid.astype(str), meta.cohort.astype(str))))
    require(len(clinical) == n and set(clinical) == ids, f'clinical ID coverage failed: {scope}')

    cdir = config_dir(scope)
    require(cdir.is_dir(), f'clinical config dir absent: {cdir}')
    paths = sorted(cdir.glob('S*.yaml'))
    require(len(paths) == 128, f'{scope}: expected 128 configs, got {len(paths)}')
    manifest_file = cdir / 'CLINICAL_LAYER1_MANIFEST.json'
    require(manifest_file.is_file(), f'manifest absent: {manifest_file}')
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    manifests = {m['stem']: m for m in manifest}
    plan_per_scope = {e['stem']: e for e in plan_entries}
    require(len(manifest) == len(manifests) == len(plan_per_scope) == 128, f'manifest/plan entries !=128: {scope}')
    source_ids = set()
    combos = set()
    jobs = []
    for path in paths:
        stem = path.stem
        require(stem in plan_per_scope and stem in manifests, f'not locked in plan: {scope} {stem}')
        entry = plan_per_scope[stem]
        require(entry == manifests[stem], f'plan and manifest differ: {scope} {stem}')
        require(entry['config_sha256'] == sha256(path), f'config hash changed: {scope} {stem}')
        obj = yaml.safe_load(path.read_text(encoding='utf-8'))
        require(obj['experiment']['id'] == f'igh_clinical_layer1_{scope}_top10000_{stem}_v3_175' and
                obj['experiment']['receptor'] == 'IGH' and obj['cohort']['scope'] == scope,
                f'wrong ID/receptor/scope: {path}')
        f = obj['features']
        require(f['clinical_groups'] == ['clinical_age_sex'] and
                f['kmer']['enabled'] is False and f['enriched_dictionary']['enabled'] is False,
                f'not clinical static-only: {path}')
        groups = tuple(f['static_tcr_groups'])
        require(len(groups) == len(set(groups)) and set(groups) <= GROUPS and groups not in combos,
                f'invalid/duplicate groups: {path}')
        combos.add(groups)
        require(list(groups) == entry['feature_groups'], f'manifest group mismatch: {path}')
        require((groups == () and stem == 'S000_none') or (groups != () and stem != 'S000_none'),
                f'baseline/nonbaseline groups wrong: {path}')
        require(obj['repertoire']['mode'] == 'topk' and int(obj['repertoire']['top_k']) == 10000 and
                obj['model']['algorithm'] == 'elastic_net' and obj['model']['positive_label'] == 'ILD' and
                obj['validation']['repeats'] == 100 and obj['validation']['folds'] == 5,
                f'run contract mismatch: {path}')
        grid = obj['model']['parameters']['elastic_net']
        require(len(grid['alpha_grid'])*len(grid['lambda_grid']) == 24, f'24 candidates mismatch: {path}')
        bankval = Path(obj['validation']['split_assignments'])
        require((bankval if bankval.is_absolute() else REPO/bankval).resolve() == bank.resolve(),
                f'YAML frozen bank differs: {path}')
        if stem != 'S000_none':
            old = old_config_dir(scope) / f'{stem}.yaml'
            require(old.is_file() and entry['source_igh_only_config_sha256'] == sha256(old),
                    f'source IGH-only YAML changed: {stem}')
            source = yaml.safe_load(old.read_text(encoding='utf-8'))
            expected = copy.deepcopy(source)
            expected['experiment']['id'] = obj['experiment']['id']
            expected['features']['clinical_groups'] = ['clinical_age_sex']
            require(expected == obj, f'YAML differs beyond clinical+id: {path}')
        jobs.append((scope, stem, path))
    require(len(combos) == 128, f'not 128 distinct combinations: {scope}')
    jobs.sort(key=lambda x: int(x[1][1:4]))
    return {'jobs': jobs, 'split_path': bank, 'split_sha': expected_hash,
            'train_n': train_n, 'holdout_n': holdout_n, 'holdouts': holdouts,
            'cohort': clinical}


def result_check(scope, stem, ctx):
    path = output_dir(scope, stem)
    if not path.is_dir():
        return False, 'output directory absent'
    try:
        missing = [x for x in REQUIRED if not (path/x).is_file()]
        require(not missing, f'missing output files: {missing}')
        m = pd.read_csv(path/'06_outer_metrics.csv')
        p = pd.read_csv(path/'06_outer_predictions.csv.gz', dtype={'sample_id': str})
        t = pd.read_csv(path/'06_inner_tuning.csv.gz')
        s = pd.read_csv(path/'06_selected_hyperparameters.csv')
        a = pd.read_csv(path/'06_feature_fit_audit.csv.gz')
        manifest = json.loads((path/'06_run_manifest.json').read_text(encoding='utf-8'))
        require(len(m) == len(s) == 100 and len(t) == 2400,
                'metrics/selected/tuning row count incorrect')
        require(len(p) == 100*ctx['holdout_n'], 'prediction row count incorrect')
        for frame, label in ((m,'metrics'), (p,'predictions'), (s,'selected')):
            require(set(as_int(frame.outer_repeat)) == set(range(1,101)), f'repeat coverage incorrect: {label}')
        require((as_int(m.n_outer_train) == ctx['train_n']).all() and
                (as_int(m.n_outer_holdout) == ctx['holdout_n']).all(), 'sample counts wrong')
        require(m.fit_converged.astype(str).str.lower().eq('true').all(), 'nonconverged final fit')
        for r, part in p.groupby('outer_repeat'):
            sample_ids = part.sample_id.astype(str)
            require(sample_ids.is_unique and set(sample_ids) == ctx['holdouts'][int(r)],
                    f'Holdout differs from bank at repeat {r}')
            expected = [ctx['cohort'][sid] for sid in sample_ids]
            require(part.true_cohort.astype(str).tolist() == expected, f'cohort labels wrong at repeat {r}')
        exe = manifest.get('execution', {})
        require(exe.get('repeat_count') == 100 and exe.get('candidate_count') == 24,
                'manifest candidate/repeat mismatch')
        saved_split = Path(manifest.get('inputs',{}).get('split_assignments',''))
        require((saved_split if saved_split.is_absolute() else REPO/saved_split).resolve() == ctx['split_path'].resolve(),
                'output manifest frozen path incorrect')
        require({'outer_repeat','feature_family','fit_policy'} <= set(a.columns), 'feature audit incomplete')
        require(not a.feature_family.astype(str).isin(['3mer', 'enriched_dictionary']).any(),
                'unexpected dynamic feature in clinical Layer1')
        st = a[a.feature_family.astype(str) == 'static_and_clinical']
        require(len(st) >= 100 and set(as_int(st.outer_repeat)) == set(range(1,101)),
                'static/clinical fold audits incomplete')
        return True, '100 repeats, frozen IDs, clinical/static fit audits, manifest PASS'
    except Exception as exc:
        return False, f'validation error: {exc}'


def request_stop(signum, _frame):
    global STOP, STOP_AT
    if STOP:
        return
    STOP, STOP_AT = True, time.monotonic()
    print(f'\nSIGNAL {signum}: stop new jobs; terminate active child groups', flush=True)
    for j in list(ACTIVE.values()):
        p = j['proc']
        if p.poll() is None:
            try: os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError: pass


def launch(job, workers):
    scope, stem, config = job
    out = output_dir(scope, stem)
    require(not out.exists(), f'refuse overwrite {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    log = log_path(scope, stem)
    status = status_path(scope, stem)
    log.parent.mkdir(parents=True, exist_ok=True)
    compat = compat_dir(scope)
    cmd = [sys.executable, str(MODEL),
           '--config', str(config),
           '--train-metadata', str(compat/'metadata_train.csv'),
           '--test-metadata', str(compat/'metadata_test.csv'),
           '--train-step02-dir', str(compat/'train_step02'),
           '--test-step02-dir', str(compat/'test_step02'),
           '--registry', str(REGISTRY),
           '--split-assignments', str(split_path(scope)),
           '--output-dir', str(out),
           '--workers', str(workers)]
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    with log.open('w', encoding='utf-8') as f:
        f.write(f'UTC started: {utcnow()}\nCommand: {json.dumps(cmd, ensure_ascii=False)}\n\n')
        f.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
    json_atomic(status, {'state':'RUNNING','scope':scope,'name':stem,'pid':proc.pid,
                         'started_utc':utcnow(),'output':str(out),'log':str(log)})
    print(f'START {scope}/{stem} pid={proc.pid} workers={workers} log={log}',flush=True)
    return {'proc':proc,'scope':scope,'stem':stem,'status':status,'log':log}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['all', *SCOPES], default='all')
    parser.add_argument('--name', action='append', default=[], help='Exact config stem e.g. S000_none; repeatable')
    parser.add_argument('--limit', type=int, default=1, help='Number of new jobs to start; 0=all')
    parser.add_argument('--max-concurrent', type=int, default=1)
    parser.add_argument('--workers-per-experiment', type=int, default=1)
    parser.add_argument('--execute', action='store_true', help='Train selected jobs; absent means read-only')
    args = parser.parse_args()
    require(args.limit >= 0, '--limit must be >=0')
    require(1 <= args.max_concurrent <= 6, '--max-concurrent must be 1..6')
    require(1 <= args.workers_per_experiment <= 8, '--workers-per-experiment must be 1..8')
    require(args.max_concurrent*args.workers_per_experiment <= 48, 'total workers must be <=48')
    require(PLAN.is_file(), f'missing immutable 384-config plan, run builder --write: {PLAN}')
    require(MODEL.is_file() and REGISTRY.is_file(), 'missing model engine / V3 registry')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    require(plan.get('decision') == 'IGH_V3_CLINICAL_LAYER1_AGE_SEX_ALL128' and
            len(plan.get('configs',[])) == 384, 'incorrect locked clinical Layer1 plan')
    ctxs = {}
    for scope in SCOPES:
        entries = [x for x in plan['configs'] if x['scope'] == scope]
        ctxs[scope] = preflight_scope(scope, entries, plan['frozen_split_sha256'][scope])
    scopes = list(SCOPES) if args.scope == 'all' else [args.scope]
    jobs = [job for scope in scopes for job in ctxs[scope]['jobs']]
    if args.name:
        unknown = set(args.name)-{j[1] for j in jobs}
        require(not unknown, f'unknown --name for selected scope: {sorted(unknown)}')
        jobs = [j for j in jobs if j[1] in set(args.name)]
    states = []
    for scope, stem, cfg in jobs:
        if output_dir(scope, stem).exists():
            valid, detail = result_check(scope, stem, ctxs[scope])
            state = 'PASS' if valid else 'BLOCKED'
        else:
            state, detail = 'READY', 'no production output'
        states.append((state,detail,(scope,stem,cfg)))
    counts = Counter(s for s,_,_ in states)
    queue = [j for state,_,j in states if state == 'READY']
    if args.limit:
        queue = queue[:args.limit]
    print(f'REPO: {REPO}',flush=True)
    print(f'CLINICAL IGH LAYER1: Age+Sex always ON; S000 + 127 static combos per scope',flush=True)
    print(f'Scopes: {", ".join(scopes)}; selected configs: {len(jobs)}',flush=True)
    print('Frozen banks: '+', '.join(s+'='+ctxs[s]['split_sha'][:12] for s in scopes),flush=True)
    print(f'Status: {dict(counts)}; next jobs: {len(queue)}; mode: {"EXECUTE" if args.execute else "PREFLIGHT ONLY"}',flush=True)
    for state,detail,job in states:
        if state=='BLOCKED':
            print(f'BLOCKED {job[0]}/{job[1]}: {detail}; inspect {output_dir(job[0],job[1])}',flush=True)
    for job in queue[:15]:
        print(f'NEXT {job[0]}/{job[1]}',flush=True)
    if not args.execute:
        print('PREFLIGHT PASS (read-only). To train, add --execute.',flush=True)
        return 0
    if counts.get('BLOCKED',0):
        print('REFUSE EXECUTE: existing incomplete results; never overwrite',flush=True)
        return 2
    if not queue:
        print('NOTHING TO DO: selected experiments already PASS',flush=True)
        return 0
    MANAGEMENT.mkdir(parents=True,exist_ok=True)
    with (MANAGEMENT/'.clinical_batch_runner.lock').open('a+') as f:
        try:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('REFUSE EXECUTE: another clinical Layer1 runner holds the lock',flush=True)
            return 3
        signal.signal(signal.SIGHUP,signal.SIG_IGN)  # crucial for nohup SSH disconnect
        signal.signal(signal.SIGTERM,request_stop)
        signal.signal(signal.SIGINT,request_stop)
        remaining=[]
        for item in queue:
            scope,stem,cfg=item
            if output_dir(scope,stem).exists():
                valid,detail=result_check(scope,stem,ctxs[scope])
                if valid:
                    print(f'SKIP newly PASS {scope}/{stem} after lock',flush=True)
                    continue
                print(f'REFUSE newly BLOCKED {scope}/{stem}: {detail}',flush=True)
                return 2
            remaining.append(item)
        queue=remaining
        passed=0
        failed=0
        while (queue and not STOP and not failed) or ACTIVE:
            while queue and len(ACTIVE)<args.max_concurrent and not STOP and not failed:
                item=queue.pop(0)
                ACTIVE[item[0]+'/'+item[1]]=launch(item,args.workers_per_experiment)
            time.sleep(1)
            for key,item in list(ACTIVE.items()):
                rc=item['proc'].poll()
                if rc is None:
                    continue
                scope,stem=item['scope'],item['stem']
                valid,detail=(result_check(scope,stem,ctxs[scope]) if rc==0 else (False,f'exit code={rc}'))
                state='PASS' if valid and not STOP else ('INTERRUPTED' if STOP else 'FAIL')
                json_atomic(item['status'], {'state':state,'scope':scope,'name':stem,
                    'pid':item['proc'].pid,'return_code':rc,'checked_utc':utcnow(),
                    'detail':detail,'output':str(output_dir(scope,stem)),'log':str(item['log'])})
                print(f'{state} {key}: {detail} (rc={rc})',flush=True)
                if state=='PASS':
                    passed+=1
                else:
                    failed+=1
                    print(f'Inspect log: tail -n 50 {item["log"]}',flush=True)
                del ACTIVE[key]
            if STOP and ACTIVE and STOP_AT is not None and time.monotonic()-STOP_AT>20:
                for item in list(ACTIVE.values()):
                    p=item['proc']
                    if p.poll() is None:
                        try: os.killpg(p.pid,signal.SIGKILL)
                        except ProcessLookupError: pass
        print(f'BATCH FINISHED: PASS={passed}, FAIL/INTERRUPTED={failed}, unstarted={len(queue)}',flush=True)
        return 130 if STOP else (1 if failed else 0)

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
