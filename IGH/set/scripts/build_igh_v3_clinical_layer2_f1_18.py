#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create IGH V3 Age+Sex clinical Layer2: per scope Top5 static+F1 and S000+F1.

Only the 3mer (F1: Top500 unweighted) is added; enriched dictionary remains off.
Top5 is read from the actual clinical Layer1 381-pair summary, NOT IGH-only ranks.
Read-only by default; --write creates three new config directories and one frozen
experiment plan. Never modifies existing Layer1, IGH-only, or earlier Layer2.

Requires successful clinical Layer1 (384 configs and 100-repeated outputs).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import yaml

import run_igh_v3_clinical_layer1_batch as layer1

SCOPES = ('total', 'pbmc', 'buffycoat')
SCHEME = 'IGH_V3_CLINICAL_LAYER2_F1_TOP5_PLUS_S000_18'
SUMMARY_FILES = ('01_experiment_summary_384.csv', '02_paired_vs_age_sex_381.csv')


def ensure(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha(path.read_bytes())


def ybytes(obj):
    data = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True).encode('utf-8')
    ensure(yaml.safe_load(data) == obj, 'YAML roundtrip failed')
    return data


def choose(summary_dir):
    full_path, pair_path = (summary_dir / p for p in SUMMARY_FILES)
    ensure(full_path.is_file() and pair_path.is_file(), 'missing clinical Layer1 summary csv(s)')
    validation = summary_dir / '00_validation_report.json'
    ensure(validation.is_file(), f'missing validated clinical Layer1 audit: {validation}')
    audit = json.loads(validation.read_text(encoding='utf-8'))
    ensure(audit.get('status') == 'PASS' and
           audit.get('checked_model_outputs') == 384 and
           audit.get('paired_to_clinical_baseline_repeats') == 38100,
           'clinical Layer1 summary validation is not complete')
    full = pd.read_csv(full_path, dtype={'stem': str})
    pair = pd.read_csv(pair_path, dtype={'stem': str})
    ensure(len(full) == 384 and len(pair) == 381, 'summary row counts are not 384/381')
    ensure(full[['scope','stem']].drop_duplicates().shape[0] == 384 and
           pair[['scope','stem']].drop_duplicates().shape[0] == 381, 'duplicate scope/stem')
    mandatory = {'scope','stem','delta_roc_auc_mean','delta_pr_auc_mean',
                 'roc_auc_joint_mean','roc_auc_baseline_mean','rank_joint_auc_in_scope'}
    ensure(mandatory <= set(pair.columns), f'paired summary missing columns: {mandatory-set(pair.columns)}')
    chosen = {}
    for scope in SCOPES:
        p = pair.loc[pair.scope == scope].copy()
        f = full.loc[full.scope == scope].copy()
        ensure(len(p) == 127 and len(f) == 128 and
               set(f.stem) == set(p.stem) | {'S000_none'}, f'incomplete cohort: {scope}')
        ensure(pd.to_numeric(p.delta_roc_auc_mean,errors='coerce').notna().all(),
               f'invalid delta AUC: {scope}')
        ensure(p.rank_joint_auc_in_scope.notna().all() and
               set(p.rank_joint_auc_in_scope.astype(int)) == set(range(1,128)),
               f'bad rank values: {scope}')
        baserow = f.loc[f.stem == 'S000_none'].iloc[0]
        ensure(abs(float(baserow.roc_auc_mean) - float(p.roc_auc_baseline_mean.iloc[0])) < 1e-9,
               f'baseline AUC differs between CSVs: {scope}')
        # Reproduce audited ranking; ties use rank from saved summary.
        p = p.sort_values('rank_joint_auc_in_scope')
        ensure((p.delta_roc_auc_mean.diff().dropna() <= 1e-9).all(),
               f'nonmonotonic clinical Layer1 ranking: {scope}')
        ensure((abs(p.roc_auc_joint_mean - (p.roc_auc_baseline_mean + p.delta_roc_auc_mean)) < 1e-9).all(),
               f'mean AUC arithmetic mismatch: {scope}')
        chosen[scope] = p.head(5)
        names = ', '.join(f'{r.stem}({r.delta_roc_auc_mean:+.4f})'
                          for r in chosen[scope].itertuples(index=False))
        print(f'[{scope.upper()}] clinical Layer1 Top5 ΔAUC: {names}', flush=True)
    return chosen, {p:sha_file(summary_dir / p) for p in SUMMARY_FILES}, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path.cwd())
    ap.add_argument('--summary-dir', type=Path, required=True,
                    help='Existing clinical_layer1_summary_* directory, as printed by the 384 summary script')
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    repo = args.repo.expanduser().resolve()
    ensure(repo == layer1.REPO, 'Place script beside run_igh_v3_clinical_layer1_batch.py in IGH/set/scripts and run from the repository root')
    summary_dir = args.summary_dir.expanduser()
    if not summary_dir.is_absolute():
        summary_dir = repo / summary_dir
    summary_dir = summary_dir.resolve()
    ensure(summary_dir.is_dir(), f'not a summary directory: {summary_dir}')
    plan1_path = layer1.PLAN
    ensure(plan1_path.is_file(), 'clinical Layer1 locked plan absent')
    plan1 = json.loads(plan1_path.read_text(encoding='utf-8'))
    ensure(plan1.get('decision') == 'IGH_V3_CLINICAL_LAYER1_AGE_SEX_ALL128' and
           len(plan1.get('configs', [])) == 384, 'Layer1 plan mismatch')
    chosen, source_summary_shas, summary_audit = choose(summary_dir)
    ensure(summary_audit.get('locked_plan_sha256') == layer1.sha256(plan1_path),
           'clinical Layer1 summary and locked plan hashes differ')

    newplan_path = layer1.VER / 'analysis/clinical_layer2_f1_18_plan_v3_175.json'
    ensure(not newplan_path.exists(), f'REFUSE OVERWRITE existing plan: {newplan_path}')
    old_root = layer1.CFG_ROOT
    outputs = {}
    entries = []
    for scope in SCOPES:
        planentries = [x for x in plan1['configs'] if x['scope'] == scope]
        context = layer1.preflight_scope(scope, planentries, plan1['frozen_split_sha256'][scope])
        ensure(len(context['jobs']) == 128, f'{scope} source not 128 configs')
        ensure(summary_audit['frozen_split_sha256'][scope] == context['split_sha'],
               f'split hash mismatch with summary: {scope}')
        # Verify 6 source results are indeed complete, with same frozen bank.
        selected = [('S000_none',0,None)]
        selected += [(r.stem, int(r.rank_joint_auc_in_scope), float(r.delta_roc_auc_mean))
                     for r in chosen[scope].itertuples(index=False)]
        for stem,_,_ in selected:
            valid,detail = layer1.result_check(scope,stem,context)
            ensure(valid, f'Layer1 source not PASS: {scope}/{stem}: {detail}')
        dest = old_root / f'igh_clinical_layer2_{scope}_top10000_v3_175'
        ensure(not dest.exists(), f'REFUSE OVERWRITE config directory: {dest}')
        config_bytes = {}
        manifest = []
        for oldstem,rank,delta in selected:
            oldpath = layer1.config_dir(scope) / f'{oldstem}.yaml'
            old = yaml.safe_load(oldpath.read_text(encoding='utf-8'))
            ensure(old['features']['clinical_groups'] == ['clinical_age_sex'] and
                   not old['features']['kmer']['enabled'] and
                   not old['features']['enriched_dictionary']['enabled'],
                   f'not clinical static-only YAML: {oldpath}')
            name = oldstem + '_F1'
            obj = copy.deepcopy(old)
            obj['experiment']['id'] = f'igh_clinical_layer2_{scope}_top10000_{name}_v3_175'
            obj['features']['kmer']['enabled'] = True
            k = obj['features']['kmer']
            ensure(k.get('k') == 3 and k.get('representation') == 'unweighted' and
                   k.get('top_n') == 500 and k.get('min_sample_count') == 5 and
                   abs(float(k.get('min_prevalence'))-.05) < 1e-12,
                   f'wrong 3-mer settings: {scope}/{name}')
            ensure(not obj['features']['enriched_dictionary']['enabled'],
                   f'enriched unexpectedly enabled: {scope}/{name}')
            data = ybytes(obj)
            config_bytes[f'{name}.yaml'] = data
            entry = {
                'scope':scope, 'stem':name, 'layer1_stem':oldstem, 'clinical_layer1_rank':rank,
                'layer1_delta_auc_vs_age_sex':delta,
                'feature_groups':old['features']['static_tcr_groups'],
                'config_sha256':sha(data), 'clinical_layer1_source_sha256':sha_file(oldpath),
                'frozen_split_sha256':context['split_sha'],
            }
            entries.append(entry);manifest.append(entry)
        ensure(len(config_bytes) == len(manifest) == 6, f'{scope}: expected six configs')
        outputs[scope] = (dest,config_bytes,manifest)
    ensure(len(entries) == 18, 'expected 18 jobs')
    plan = {
        'schema_version':1, 'decision':SCHEME,
        'meaning':'Age+Sex always ON; clinical Layer1 Top5 static (+ S000 no-static control); add fold-fit unweighted AA 3mer Top500 only.',
        'model':'Elastic Net; 24 candidates; 5-fold inner CV; 100 original frozen holdouts',
        'kmer':'F1 only; no enriched dictionary (F2/F3 excluded)',
        'selection':'Top5 per scope by clinical Layer1 100-repeat mean AUC, exploratory/selection-biased; S000+F1 added as direct clinical plus 3-mer control',
        'clinical_layer1_plan_sha256':sha_file(plan1_path),
        'clinical_layer1_summary_dir':str(summary_dir),
        'clinical_layer1_summary_file_sha256':source_summary_shas,
        'frozen_split_sha256':plan1['frozen_split_sha256'],
        'configs':entries,
    }
    if not args.write:
        print('\nCLINICAL LAYER2 F1 PREFLIGHT PASS: 18 YAMLs selected, no files written', flush=True)
        return
    with tempfile.TemporaryDirectory(prefix='.igh_clinical_layer2_f1_stage_',dir=old_root) as tmp:
        temp = Path(tmp)
        for scope,(dest,files,manifest) in outputs.items():
            d = temp / dest.name
            d.mkdir()
            for name,bytes_ in files.items():
                (d/name).write_bytes(bytes_)
            (d/'CLINICAL_LAYER2_F1_MANIFEST.json').write_text(
                json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        for scope,(dest,files,manifest) in outputs.items():
            ensure(not dest.exists(), f'config dir suddenly appeared: {dest}')
            (temp/dest.name).rename(dest)
            print(f'[WRITE PASS] {scope}: six YAMLs at {dest}',flush=True)
    newplan_path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with newplan_path.open('x',encoding='utf-8') as out:
            json.dump(plan,out,ensure_ascii=False,indent=2)
            out.write('\n')
    except FileExistsError:
        raise RuntimeError(f'plan appeared during write: {newplan_path}')
    print(f'CLINICAL LAYER2 F1 WRITE PASS: 18 YAMLs; PLAN={newplan_path}; no model fits',flush=True)

if __name__ == '__main__':
    main()
