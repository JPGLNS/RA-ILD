#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build separate IGH V3 Age+Sex clinical Layer1: S000 + S001..S127, three scopes.

Read-only preflight unless --write. Never edit IGH-only/Layer2/final-six configs.
Copies validated frozen V3 IGH-only S001..S127 YAMLs, changing only experiment.id
and features.clinical_groups; S000 is age+sex and no static IGH predictors.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import yaml

SCOPE_INFO = {
    'total': ('igh_ra_ild_repeat100_v3_175', '455eba17c74a86667fd2939dc8645ac1c50f0cdb25053ab951dccad0ab8f1788', 175),
    'pbmc': ('igh_ra_ild_pbmc_repeat100_v3_175', '4726d9b48301c4cd535369f9b39f6bdf3c6247c0722b9c9663385a6c041a68b8', 113),
    'buffycoat': ('igh_ra_ild_buffycoat_repeat100_v3_175', 'da1e598559dc734c0e0296bd1f99620351c2d81e903100c85fda23b670047c05', 62),
}
GROUPS = {
    'tcr_diversity', 'tcr_clonal_expansion', 'tcr_aa_length',
    'tcr_aa_composition_unweighted', 'tcr_aa_composition_weighted',
    'tcr_physicochemical_unweighted', 'tcr_physicochemical_weighted',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(path.read_bytes())


def yaml_bytes(obj):
    b = yaml.safe_dump(obj, allow_unicode=True, sort_keys=False).encode('utf-8')
    require(yaml.safe_load(b) == obj, 'YAML roundtrip failure')
    return b


def verify_scope(root, scope):
    bank_name, expected_sha, n = SCOPE_INFO[scope]
    srcdir = root / 'IGH/set/configs/experiments' / f'igh_layer1_{scope}_top10000_v3_175'
    destdir = root / 'IGH/set/configs/experiments' / f'igh_clinical_layer1_{scope}_top10000_v3_175'
    require(srcdir.is_dir(), f'missing IGH-only Layer1: {srcdir}')
    require(not destdir.exists(), f'REFUSE OVERWRITE: {destdir}')
    bank = root / 'IGH/set/split_sets' / bank_name / 'repeated_holdout_assignments.csv'
    marker = bank.parent / 'SPLITS_FROZEN.json'
    require(bank.is_file() and marker.is_file(), f'missing frozen bank/marker: {bank}')
    require(sha256_file(bank) == expected_sha, f'FROZEN SPLIT HASH MISMATCH: {scope}')
    freeze = json.loads(marker.read_text(encoding='utf-8'))
    require(freeze.get('status') == 'FROZEN' and
            freeze.get('files', {}).get('assignments', {}).get('sha256') == expected_sha,
            f'freeze marker mismatch: {scope}')
    compat = root / 'IGH/set/data_versions/igh_v3_175/ml_compat' / scope
    metas = []
    for part in ('train', 'test'):
        mp = compat / f'metadata_{part}.csv'
        require(mp.is_file(), f'missing metadata: {mp}')
        df = pd.read_csv(mp, dtype={'libraryid': str})
        require({'libraryid', 'cohort', 'age', 'sex'} <= set(df.columns), f'clinical columns absent: {mp}')
        require(df.libraryid.notna().all() and df.libraryid.astype(str).str.strip().ne('').all(), f'blank IDs {mp}')
        require(df.cohort.isin(['RA', 'ILD']).all(), f'invalid cohorts: {mp}')
        a = pd.to_numeric(df.age, errors='coerce')
        require(a.notna().all() and a.between(1, 119).all(), f'bad age: {mp}')
        s = df.sex.astype('string').str.strip().str.lower()
        require(s.notna().all() and s.isin(['female', 'male']).all(), f'bad sex: {mp}')
        metas.append(df)
    combined = pd.concat(metas, ignore_index=True)
    require(len(combined) == n and combined.libraryid.is_unique, f'{scope}: metadata expected {n} unique IDs')
    files = sorted(srcdir.glob('S*.yaml'))
    require(len(files) == 127 and len({p.stem for p in files}) == 127,
            f'{scope}: expected exactly 127 IGH-only YAMLs')
    require(all(p.name.startswith('S') and p.stem[1:4].isdigit() for p in files), f'{scope}: unexpected filenames')
    stems = {int(p.stem[1:4]) for p in files}
    require(stems == set(range(1, 128)), f'{scope}: missing S001..S127')
    configs, manifest = {}, []
    combos = set()
    sample = None
    base_model = None
    for path in files:
        old = yaml.safe_load(path.read_text(encoding='utf-8'))
        stem = path.stem
        require(isinstance(old, dict), f'invalid yaml: {path}')
        require(old['experiment']['receptor'] == 'IGH' and
                old['experiment']['id'] == f'igh_layer1_{scope}_top10000_{stem}_v3_175',
                f'unexpected source experiment: {path}')
        require(old['cohort']['scope'] == scope and
                old['repertoire']['mode'] == 'topk' and int(old['repertoire']['top_k']) == 10000,
                f'unexpected source cohort/repertoire: {path}')
        f = old['features']
        group_tuple = tuple(f['static_tcr_groups'])
        require(1 <= len(group_tuple) <= 7 and len(set(group_tuple)) == len(group_tuple)
                and set(group_tuple) <= GROUPS and group_tuple not in combos,
                f'duplicate/invalid feature combination: {path}')
        combos.add(group_tuple)
        require(f['clinical_groups'] == [] and
                f['kmer']['enabled'] is False and f['enriched_dictionary']['enabled'] is False,
                f'source contains clinical/dynamic features: {path}')
        require(old['model']['algorithm'] == 'elastic_net' and old['model']['positive_label'] == 'ILD',
                f'wrong model: {path}')
        require(old['validation']['repeats'] == 100 and old['validation']['folds'] == 5,
                f'wrong validation: {path}')
        grid = old['model']['parameters']['elastic_net']
        require(len(grid['alpha_grid'])*len(grid['lambda_grid']) == 24, f'wrong grid: {path}')
        refbank = Path(old['validation']['split_assignments'])
        require((refbank if refbank.is_absolute() else root / refbank).resolve() == bank.resolve(),
                f'wrong bank in YAML: {path}')
        if base_model is None:
            base_model = old['model']
            sample = old
        require(old['model'] == base_model, f'inconsistent model across Layer1: {path}')
        new = copy.deepcopy(old)
        new['experiment']['id'] = f'igh_clinical_layer1_{scope}_top10000_{stem}_v3_175'
        new['features']['clinical_groups'] = ['clinical_age_sex']
        content = yaml_bytes(new)
        configs[f'{stem}.yaml'] = content
        manifest.append({'scope': scope, 'stem': stem, 'feature_groups': list(group_tuple),
                         'config_sha256': sha256_bytes(content), 'source_igh_only_config_sha256': sha256_file(path)})
    require(len(combos) == 127, f'{scope}: not 127 distinct feature subsets')
    baseline = copy.deepcopy(sample)
    baseline['experiment']['id'] = f'igh_clinical_layer1_{scope}_top10000_S000_none_v3_175'
    baseline['features']['clinical_groups'] = ['clinical_age_sex']
    baseline['features']['static_tcr_groups'] = []
    s000 = yaml_bytes(baseline)
    configs['S000_none.yaml'] = s000
    manifest.insert(0, {'scope': scope, 'stem': 'S000_none', 'feature_groups': [],
                        'config_sha256': sha256_bytes(s000), 'source_igh_only_config_sha256': None})
    require(len(configs) == len(manifest) == 128, f'config count wrong: {scope}')
    print(f'[{scope.upper()}] 128 configs (S000 clinical-only + 127 clinical+IGH); '
          f'n={n}; split_SHA={expected_sha[:12]}', flush=True)
    return destdir, configs, manifest, expected_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    root = args.repo.resolve()
    require((root / 'IGH/set').is_dir(), f'not RA-ILD root: {root}')
    registry = root / 'IGH/set/configs/feature_framework/igh_feature_registry_v3_175.yaml'
    require(registry.is_file(), f'no V3 registry: {registry}')
    obj = yaml.safe_load(registry.read_text(encoding='utf-8'))
    group = obj.get('feature_groups', {}).get('clinical_age_sex', {})
    require(group.get('family') == 'clinical' and
            {'age', 'sex'} <= set(group.get('selector', {}).get('exact', [])),
            'registry clinical_age_sex does not select age+sex')
    planpath = root / 'IGH/set/data_versions/igh_v3_175/analysis/clinical_layer1_384_plan_v3_175.json'
    require(not planpath.exists(), f'REFUSE OVERWRITE plan: {planpath}')
    items, entries, bank_hashes = {}, [], {}
    for scope in SCOPE_INFO:
        dest, files, manifest, sha = verify_scope(root, scope)
        items[scope] = (dest, files, manifest)
        entries.extend(manifest)
        bank_hashes[scope] = sha
    require(len(entries) == 384, 'expected 384 configs')
    plan = {
        'schema_version': 1,
        'decision': 'IGH_V3_CLINICAL_LAYER1_AGE_SEX_ALL128',
        'model': 'Elastic Net, inner 5-fold CV, 24 candidates, outer 100 frozen repeats',
        'features': 'clinical_age_sex is ALWAYS ON; S000 contains zero IGH predictors',
        'dynamic': '3mer=False; enriched_dictionary=False',
        'source': 'Original verified IGH V3 IGH-only 127 YAMLs per scope',
        'prior_IGH_only': 'retain all previous results and configs unchanged',
        'frozen_split_sha256': bank_hashes,
        'configs': entries,
    }
    if not args.write:
        print('\nCLINICAL LAYER1 PREFLIGHT PASS: 384 configs ready; NO FILES WRITTEN', flush=True)
        return
    cfg_root = root / 'IGH/set/configs/experiments'
    with tempfile.TemporaryDirectory(prefix='.igh_clinical_layer1_384_stage_', dir=cfg_root) as tmp:
        stage = Path(tmp)
        for scope, (dest, files, manifest) in items.items():
            folder = stage / dest.name
            folder.mkdir()
            for filename, b in files.items():
                (folder / filename).write_bytes(b)
            (folder / 'CLINICAL_LAYER1_MANIFEST.json').write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        for scope, (dest, files, manifest) in items.items():
            require(not dest.exists(), f'REFUSE OVERWRITE: {dest}')
            os.rename(stage / dest.name, dest)
            print(f'[WRITE PASS] {scope}: {dest}', flush=True)
    planpath.parent.mkdir(parents=True, exist_ok=True)
    with planpath.open('x', encoding='utf-8') as out:
        json.dump(plan, out, indent=2, ensure_ascii=False)
        out.write('\n')
    print(f'CLINICAL LAYER1 WRITE PASS: 384 YAMLs; PLAN={planpath}; NO TRAINING EXECUTED', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        raise SystemExit(f'ERROR: {exc}')
