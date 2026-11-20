#!/usr/bin/env python3
"""Audit Layer-1 Repeat001 smoke outputs for Total, PBMC and Buffycoat."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'TRB/set/src'))
from ra_ild_trb.feature_framework import load_feature_registry
from ra_ild_trb.fold_features import select_static_features
EXPECTED = {'total': (179, 125, 54), 'pbmc': (116, 81, 35), 'buffycoat': (63, 44, 19)}
OUTPUT_FILES = [
    '06_outer_metrics.csv', '06_outer_predictions.csv.gz', '06_inner_tuning.csv.gz',
    '06_selected_hyperparameters.csv', '06_feature_explanations.csv.gz',
    '06_feature_fit_audit.csv.gz', '06_preprocessing_audit.csv.gz',
    '06_inner_assignment_audit.csv.gz', '06_run_manifest.json', '06_run_summary.md',
]
def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
def truth(value) -> bool:
    return str(value).strip().lower() in {'true', '1', 'yes'}
def audit_one(root: Path, cohort: str, entry: dict, repeat: int, metadata_cohort: dict) -> dict:
    algorithm, combo = entry['algorithm'], entry['combination_id']
    require(algorithm == 'elastic_net', f'{cohort}: unexpected algorithm {algorithm}')
    tag = f'smoke_repeat_{repeat:03d}'
    out = root / 'TRB/result/feature_framework/layered_feature_experiments/layer1' / cohort / algorithm / combo / tag
    status_path = root / 'TRB/result/feature_framework/layered_feature_experiments/status/layer1' / cohort / algorithm / f'{combo}.{tag}.json'
    require(status_path.is_file(), f'{cohort}: missing runner status {status_path}')
    status = json.loads(status_path.read_text(encoding='utf-8'))
    require(status.get('status') == 'COMPLETE' and status.get('returncode') == 0, f'{cohort}: runner status is not COMPLETE: {status}')
    require(Path(status.get('output', '')).resolve() == out.resolve(), f'{cohort}: runner output path mismatch')
    for name in OUTPUT_FILES:
        file = out / name
        require(file.is_file() and file.stat().st_size > 0, f'{cohort}: missing/empty output {file}')
    cfg_path = root / entry['config']
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    require(cfg['cohort']['scope'] == cohort, f'{cohort}: config cohort mismatch')
    require(cfg['model']['algorithm'] == algorithm, f'{cohort}: config algorithm mismatch')
    require(cfg['repertoire'] == {'mode': 'topk', 'top_k': 10000}, f'{cohort}: config repertoire mismatch')
    require(cfg['features']['clinical_groups'] == ['clinical_age_sex'], f'{cohort}: age/sex adjustment missing')
    require(cfg['features']['static_tcr_groups'] == entry['feature_groups'], f'{cohort}: static feature groups mismatch')
    require(cfg['features']['kmer']['enabled'] is False, f'{cohort}: Layer-1 3-mer must be disabled')
    require(cfg['features']['enriched_dictionary']['enabled'] is False, f'{cohort}: Layer-1 RA-hit must be disabled')
    require(cfg['validation']['repeats'] == 100 and cfg['validation']['folds'] == 5, f'{cohort}: configured CV contract mismatch')
    split_path = root / cfg['validation']['split_assignments']
    require(split_path.is_file(), f'{cohort}: split bank missing {split_path}')
    split = pd.read_csv(split_path)
    if 'sample_id' not in split.columns and 'libraryid' in split.columns:
        split = split.rename(columns={'libraryid': 'sample_id'})
    require({'sample_id', 'repeat_index', 'role'}.issubset(split.columns), f'{cohort}: split bank columns invalid')
    split['sample_id'] = split['sample_id'].astype(str)
    split['repeat_index'] = pd.to_numeric(split['repeat_index'], errors='raise').astype(int)
    split['role'] = split['role'].astype(str).str.lower().str.strip()
    part = split.loc[split['repeat_index'] == repeat].copy()
    require(len(part) == part['sample_id'].nunique(), f'{cohort}: duplicate sample in repeat assignment')
    train_ids = set(part.loc[part['role'] == 'train', 'sample_id'])
    holdout_ids = set(part.loc[part['role'] == 'holdout', 'sample_id'])
    n_total, n_train, n_holdout = EXPECTED[cohort]
    require(len(train_ids) == n_train and len(holdout_ids) == n_holdout, f'{cohort}: split sizes {len(train_ids)}/{len(holdout_ids)} != {n_train}/{n_holdout}')
    require(not (train_ids & holdout_ids) and len(train_ids | holdout_ids) == n_total, f'{cohort}: split coverage/separation invalid')
    metrics = pd.read_csv(out / '06_outer_metrics.csv')
    require(len(metrics) == 1, f'{cohort}: Repeat{repeat:03d} must have exactly one outer metric row')
    m = metrics.iloc[0]
    require(int(m['outer_repeat']) == repeat, f'{cohort}: metrics repeat ID mismatch')
    require(int(m['n_outer_train']) == n_train and int(m['n_outer_holdout']) == n_holdout, f'{cohort}: metrics train/holdout counts mismatch')
    require(truth(m['fit_converged']), f'{cohort}: final Elastic-Net model did not converge')
    require(np.isfinite([float(m['roc_auc']), float(m['pr_auc'])]).all(), f'{cohort}: non-finite ROC-AUC/AP')
    require(int(m['n_final_predictors']) > 0, f'{cohort}: no final predictors survived preprocessing')
    pred = pd.read_csv(out / '06_outer_predictions.csv.gz')
    require(len(pred) == n_holdout, f'{cohort}: prediction row count {len(pred)} != {n_holdout}')
    require(pred['sample_id'].astype(str).is_unique, f'{cohort}: duplicate prediction sample IDs')
    require(set(pred['sample_id'].astype(str)) == holdout_ids, f'{cohort}: predictions do not exactly match frozen holdout IDs')
    require(set(pred['true_cohort'].astype(str).str.upper()) == {'RA', 'ILD'}, f'{cohort}: holdout predictions lack one class')
    require((pd.to_numeric(pred['true_label'], errors='raise').to_numpy() == (pred['true_cohort'].astype(str).str.upper() == 'ILD').astype(int).to_numpy()).all(), f'{cohort}: true label/cohort encoding mismatch')
    require(np.isfinite(pd.to_numeric(pred['prediction_score'], errors='raise')).all(), f'{cohort}: non-finite prediction scores')
    scores = pd.to_numeric(pred['prediction_score'], errors='raise')
    require(((scores >= 0) & (scores <= 1)).all(), f'{cohort}: Elastic-Net probabilities outside [0,1]')
    require(set(pred['score_type'].astype(str)) == {'probability'}, f'{cohort}: unexpected score type')
    observed_truth = pred['sample_id'].astype(str).map(metadata_cohort)
    require(observed_truth.notna().all(), f'{cohort}: prediction ID absent from compatibility metadata')
    require((observed_truth.astype(str).str.upper().to_numpy() == pred['true_cohort'].astype(str).str.upper().to_numpy()).all(), f'{cohort}: prediction labels disagree with metadata')
    selected = pd.read_csv(out / '06_selected_hyperparameters.csv')
    require(len(selected) == 1 and int(selected.iloc[0]['outer_repeat']) == repeat, f'{cohort}: selected hyperparameter record invalid')
    require(str(selected.iloc[0]['candidate_id']) == str(m['selected_candidate_id']), f'{cohort}: selected candidate mismatch')
    tuning = pd.read_csv(out / '06_inner_tuning.csv.gz')
    require(not tuning.empty, f'{cohort}: inner tuning table is empty')
    inner = pd.read_csv(out / '06_inner_assignment_audit.csv.gz')
    require(len(inner) == 5 and set(pd.to_numeric(inner['inner_fold']).astype(int)) == {1, 2, 3, 4, 5}, f'{cohort}: inner 5-fold audit invalid')
    require(int(pd.to_numeric(inner['validation_samples']).sum()) == n_train, f'{cohort}: inner validation folds do not cover outer training samples')
    prep = pd.read_csv(out / '06_preprocessing_audit.csv.gz')
    require({'feature_name', 'training_mean', 'training_sd', 'kept_after_zero_variance_filter', 'stage', 'inner_fold'}.issubset(prep.columns), f'{cohort}: preprocessing audit schema invalid')
    outer_prep = prep.loc[prep['stage'].astype(str) == 'outer_final'].copy()
    require(not outer_prep.empty, f'{cohort}: outer preprocessing audit missing')
    kept = outer_prep['kept_after_zero_variance_filter'].map(truth)
    require(int(kept.sum()) == int(m['n_final_predictors']), f'{cohort}: final predictor count disagrees with preprocessing audit')
    means = pd.to_numeric(outer_prep['training_mean'], errors='raise').to_numpy(float)
    sds = pd.to_numeric(outer_prep['training_sd'], errors='raise').to_numpy(float)
    require(np.isfinite(means).all() and np.isfinite(sds).all(), f'{cohort}: non-finite preprocessing parameters')
    require((sds[kept.to_numpy()] > 0).all(), f'{cohort}: kept predictor has zero/non-positive training SD')
    inner_prep = prep.loc[prep['stage'].astype(str) == 'inner']
    require(set(pd.to_numeric(inner_prep['inner_fold'].dropna()).astype(int)) == {1, 2, 3, 4, 5}, f'{cohort}: inner-fold preprocessing audit incomplete')
    feature_audit_path = out / '06_feature_fit_audit.csv.gz'
    try:
        feature_audit = pd.read_csv(feature_audit_path)
    except pd.errors.EmptyDataError:
        feature_audit = pd.DataFrame()
    if not feature_audit.empty and 'feature_family' in feature_audit.columns:
        families = set(feature_audit['feature_family'].astype(str))
        require(not ({'3mer', 'enriched_dictionary'} & families), f'{cohort}: unexpected learned Layer-2 feature in Layer-1: {families}')
    run_manifest = json.loads((out / '06_run_manifest.json').read_text(encoding='utf-8'))
    execution = run_manifest['execution']
    require(execution['repeat_ids'] == [repeat] and execution['repeat_count'] == 1, f'{cohort}: run manifest repeat scope mismatch')
    require(execution['static_tcr_predictor_count'] > 0 and execution['candidate_count'] > 0, f'{cohort}: run manifest predictor/candidate count invalid')
    require(run_manifest['inputs']['repertoire_source']['source_id'] == 'top10000', f'{cohort}: run manifest repertoire source mismatch')
    leakage = run_manifest['leakage_contract']
    require(leakage['preprocessing'] == 'current_training_only' and leakage['threshold_selection'] == 'selected_candidate_pooled_inner_oof_youden', f'{cohort}: leakage-control manifest mismatch')
    return {'cohort': cohort, 'samples': n_total, 'train': n_train, 'holdout': n_holdout, 'static_predictors': execution['static_tcr_predictor_count'], 'final_predictors': int(m['n_final_predictors']), 'roc_auc': float(m['roc_auc']), 'pr_auc': float(m['pr_auc']), 'status': 'PASS'}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root', type=Path, default=ROOT)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--cohort', choices=['total', 'pbmc', 'buffycoat', 'all'], default='all')
    args = parser.parse_args()
    root = args.repository_root.expanduser().resolve()
    manifest_path = root / 'TRB/set/configs/layered_feature_experiments/generated/layer1/LAYER1_CONFIG_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = manifest['entries']
    meta_root = root / 'TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total'
    meta = pd.concat([pd.read_csv(meta_root / 'metadata_train_70.csv', dtype=str), pd.read_csv(meta_root / 'metadata_test_30.csv', dtype=str)], ignore_index=True)
    require(meta['libraryid'].astype(str).is_unique, 'compatibility metadata contains duplicate library IDs')
    metadata_cohort = meta.set_index(meta['libraryid'].astype(str))['cohort'].astype(str).str.upper().to_dict()
    registry_path = root / 'TRB/set/configs/feature_framework/trb_feature_registry_v3_179_top10000.yaml'
    registry = load_feature_registry(registry_path)
    cache_columns = pd.read_csv(root / 'TRB/result/feature_framework/cache/step02/top10000/train/02_core_sample_features.csv', nrows=0).columns
    selected_cohorts = list(EXPECTED) if args.cohort == 'all' else [args.cohort]
    results = []
    for cohort in selected_cohorts:
        matched = [e for e in entries if e['cohort'] == cohort and e['algorithm'] == 'elastic_net' and e['combination_id'] == 'L1_001_D']
        require(len(matched) == 1, f'{cohort}: expected exactly one L1_001_D manifest entry')
        entry = matched[0]
        static_names = select_static_features(registry, cache_columns, entry['feature_groups'])
        result = audit_one(root, cohort, entry, args.repeat, metadata_cohort)
        require(result['static_predictors'] == len(static_names), f'{cohort}: manifest static predictor count disagrees with registry ({result["static_predictors"]} != {len(static_names)})')
        out = root / 'TRB/result/feature_framework/layered_feature_experiments/layer1' / cohort / 'elastic_net' / 'L1_001_D' / f'smoke_repeat_{args.repeat:03d}'
        prep = pd.read_csv(out / '06_preprocessing_audit.csv.gz')
        outer_names = set(prep.loc[prep['stage'].astype(str) == 'outer_final', 'feature_name'].astype(str))
        require(set(static_names).issubset(outer_names), f'{cohort}: one or more selected static predictors absent from outer preprocessing audit')
        results.append(result)
    print('TRB V3 LAYER-1 REPEAT SMOKE AUDIT')
    for row in results:
        print(f"[PASS] {row['cohort']}: N={row['samples']} train/holdout={row['train']}/{row['holdout']} static={row['static_predictors']} final={row['final_predictors']} ROC-AUC={row['roc_auc']:.4f} PR-AUC={row['pr_auc']:.4f}")
    print(f'ALL SMOKE OUTPUT AUDITS PASSED: {len(results)} cohort(s), repeat={args.repeat}')
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'AUDIT FAIL: {exc}', file=sys.stderr)
        raise SystemExit(1)
