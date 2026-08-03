#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused acceptance tests for IGH Batch 11B."""
from __future__ import annotations
import json, shutil, sys, tempfile
from pathlib import Path
import pandas as pd
import yaml
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from ra_ild_igh.cohort_subset import (
    CohortSubsetError, load_cohort_subset_spec, prepare_cohort_subset,
    verify_cohort_subset_marker, write_cohort_subset,
)
ROOT = Path(__file__).resolve().parents[3]

def _make_repo(root: Path, *, mismatch=False, duplicate_patient=False):
    (root/'data').mkdir(parents=True)
    rows=[]
    for i in range(1,15):
        material='PBMC' if i<=8 else 'buffercoat'
        cohort='RA' if i in {1,2,3,4,9,10,11} else 'ILD'
        rows.append({'libraryid':f'S{i:02d}','patient':('P01' if duplicate_patient and i==2 else f'P{i:02d}'),
                     'cohort':cohort,'material':material,'batch':'b1' if i%2 else 'b2','sex':'female' if i%3 else 'male','age':40+i})
    meta=pd.DataFrame(rows)
    train=meta.iloc[:9].copy(); test=meta.iloc[9:].copy()
    train.to_csv(root/'data/train_meta.csv',index=False); test.to_csv(root/'data/test_meta.csv',index=False)
    def matrix(frame):
        x=frame.rename(columns={'libraryid':'sample_id'}).copy(); x['aa_clone_number']=100; x['igh_feature']=range(len(x)); return x
    tx=matrix(train); vx=matrix(test)
    if mismatch: tx.loc[tx.index[0],'material']='WRONG'
    tx.to_csv(root/'data/train_matrix.csv',index=False); vx.to_csv(root/'data/test_matrix.csv',index=False)
    spec={'material_subset_version':'1.0','subset':{'id':'synthetic_pbmc','description':'x','output_root':'IGH/set/cohort_subsets/synthetic_pbmc'},
          'source':{'metadata_files':['data/train_meta.csv','data/test_meta.csv'],'train_base_matrix':'data/train_matrix.csv','test_final_matrix':'data/test_matrix.csv',
                    'metadata_sample_id_column':'libraryid','matrix_sample_id_column':'sample_id','patient_id_column':'patient','label_column':'cohort','require_one_sample_per_patient':True},
          'filter':{'column':'material','include_values':['PBMC'],'case_insensitive':True,'strip_whitespace':True},
          'expected':{'samples':8,'patients':8,'label_counts':{'RA':4,'ILD':4},'filter_counts':{'PBMC':8}}}
    p=root/'spec.yaml'; p.write_text(yaml.safe_dump(spec,sort_keys=False),encoding='utf-8'); return p

def test_prepare_and_freeze():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td); spec=load_cohort_subset_spec(_make_repo(r),repository_root=r); result=prepare_cohort_subset(spec)
        assert len(result.metadata)==8 and len(result.train_matrix)+len(result.test_matrix)==8
        out=write_cohort_subset(spec,result); marker=verify_cohort_subset_marker(out/'SUBSET_FROZEN.json',repository_root=r)
        assert marker['summary']['label_counts']=={'RA':4,'ILD':4}
def test_case_insensitive_filter():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td); p=_make_repo(r); raw=yaml.safe_load(p.read_text()); raw['filter']['include_values']=[' pbmc ']; p.write_text(yaml.safe_dump(raw,sort_keys=False))
        result=prepare_cohort_subset(load_cohort_subset_spec(p,repository_root=r)); assert len(result.metadata)==8
def test_identity_mismatch_rejected():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td); spec=load_cohort_subset_spec(_make_repo(r,mismatch=True),repository_root=r)
        try: prepare_cohort_subset(spec)
        except CohortSubsetError: return
        raise AssertionError('identity mismatch accepted')
def test_duplicate_patient_rejected():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td); spec=load_cohort_subset_spec(_make_repo(r,duplicate_patient=True),repository_root=r)
        try: prepare_cohort_subset(spec)
        except CohortSubsetError: return
        raise AssertionError('duplicate patient accepted')
def test_frozen_subset_is_immutable():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td); spec=load_cohort_subset_spec(_make_repo(r),repository_root=r); result=prepare_cohort_subset(spec); write_cohort_subset(spec,result)
        try: write_cohort_subset(spec,result,replace_incomplete=True)
        except CohortSubsetError: return
        raise AssertionError('frozen subset overwritten')
def test_marker_detects_tampering():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td); spec=load_cohort_subset_spec(_make_repo(r),repository_root=r); out=write_cohort_subset(spec,prepare_cohort_subset(spec));
        (out/'subset_metadata.csv').write_text('tampered\n')
        try: verify_cohort_subset_marker(out/'SUBSET_FROZEN.json',repository_root=r)
        except CohortSubsetError: return
        raise AssertionError('tampering not detected')
def test_real_configs_and_counts():
    p=yaml.safe_load((ROOT/'IGH/set/configs/material_subsets/igh_pbmc_only_v1.yaml').read_text()); b=yaml.safe_load((ROOT/'IGH/set/configs/material_subsets/igh_buffercoat_only_v1.yaml').read_text())
    assert p['expected']['samples']==108 and p['expected']['label_counts']=={'RA':63,'ILD':45}
    assert b['expected']['samples']==61 and b['expected']['label_counts']=={'RA':39,'ILD':22}
    ps=yaml.safe_load((ROOT/'IGH/set/configs/split_sets/igh_ra_ild_pbmc_repeat100_v1.yaml').read_text()); bs=yaml.safe_load((ROOT/'IGH/set/configs/split_sets/igh_ra_ild_buffercoat_repeat100_v1.yaml').read_text())
    assert ps['split']['exact_strata']==['cohort','batch'] and ps['split']['balance_categorical']==['sex']
    assert bs['split']['exact_strata']==['cohort'] and bs['split']['balance_categorical']==['batch','sex']
    assert ps['split']['train_size']==76 and ps['split']['holdout_size']==32
    assert bs['split']['train_size']==43 and bs['split']['holdout_size']==18
def test_material_not_a_predictor():
    template=yaml.safe_load((ROOT/'IGH/set/configs/schemes/igh_scheme_material_homogeneous_no_clinical_template.yaml').read_text())
    assert set(template['replacements']['models'])=={'M1_static_igh','M2_static_igh_public'}
    for path in [ROOT/'IGH/set/configs/schemes/igh_scheme_pbmc_a1_a6_repeat100_v1.yaml',ROOT/'IGH/set/configs/schemes/igh_scheme_buffercoat_a1_a6_repeat100_v1.yaml']:
        cfg=yaml.safe_load(path.read_text()); assert len(cfg['models'])==14
        for model in cfg['models'].values(): assert 'material' not in model.get('numeric',[]) and 'material' not in model.get('categorical',[])

def main():
    tests=[test_prepare_and_freeze,test_case_insensitive_filter,test_identity_mismatch_rejected,test_duplicate_patient_rejected,test_frozen_subset_is_immutable,test_marker_detects_tampering,test_real_configs_and_counts,test_material_not_a_predictor]
    for t in tests: t(); print('PASS',t.__name__)
    print(f'IGH Batch 11B focused tests: {len(tests)}/{len(tests)} PASS')
    print('IGH_BATCH11B_ACCEPTANCE_PASS'); return 0
if __name__=='__main__': raise SystemExit(main())
