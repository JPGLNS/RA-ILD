#!/usr/bin/env python3
"""Standalone synthetic regression checks for IGH V3 3-mer both adapter.

No patient data and no model training required.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
# Works when copied into IGH/set/tests/ OR run from /mnt/data with --repo unavailable;
# preferred location: IGH/set/tests/test_igh_v3_both_3mer.py
REPO = HERE.parents[3]
SRC = REPO / 'IGH/set/src'
if not SRC.exists():
    raise SystemExit(f'Copy test to IGH/set/tests/ first: missing {SRC}')
sys.path.insert(0, str(SRC))
from ra_ild_igh.experiment_spec import KMER_REPRESENTATIONS
from ra_ild_igh.fold_features import FoldFeatureInputs, build_fold_features


TRAIN_IDS = ('A', 'B', 'C', 'D', 'E', 'F')
VALID_IDS = ('V1', 'V2')
IDS = TRAIN_IDS + VALID_IDS


def spec(representation):
    kmer = SimpleNamespace(
        enabled=True, k=3, representation=representation, top_n=2,
        min_sample_count=1, min_prevalence=0.0,
        min_unweighted_variance=0.0, min_weighted_variance=0.0,
    )
    enrich = SimpleNamespace(enabled=False)
    return SimpleNamespace(base=SimpleNamespace(features=SimpleNamespace(kmer=kmer, enriched_dictionary=enrich)))


def inputs():
    md = pd.DataFrame({
        'sample_id': IDS,
        'cohort': ['RA', 'ILD', 'RA', 'ILD', 'RA', 'ILD', 'RA', 'ILD'],
        'age': [30, 40, 50, 60, 70, 80, 45, 65],
        'sex': ['F'] * len(IDS),
    })
    u = {
        'A': {'AAA': 2, 'BBB': 1},
        'B': {'AAA': 1, 'BBB': 1},
        'C': {'AAA': 1},
        'D': {'AAA': 3, 'CCC': 1},
        'E': {'BBB': 1, 'CCC': 1},
        'F': {'BBB': 2, 'CCC': 2},
        'V1': {'AAA': 10, 'ZZZ': 300},
        'V2': {'CCC': 1, 'ZZZ': 5},
    }
    w = {sid: {k: float(v) / 100 for k, v in vals.items()} for sid, vals in u.items()}
    return FoldFeatureInputs(
        metadata=md,
        static_values=pd.DataFrame({'sample_id': IDS}),
        static_feature_names=tuple(),
        clinical_numeric=('age',),
        clinical_categorical=('sex',),
        source=SimpleNamespace(source_id='top10000'),
        kmer_unweighted=u,
        kmer_weighted=w,
        enrich_repertoires={},
    )


class TestBothRepresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inp = inputs()

    def build(self, rep):
        from types import SimpleNamespace
        fixture = spec(rep)
        fixture.base.features.public_reference = SimpleNamespace(enabled=False)
        return build_fold_features(self.inp, fixture, TRAIN_IDS, VALID_IDS, context='synthetic')

    def test_config_accepts_both(self):
        self.assertIn('both', KMER_REPRESENTATIONS)

    def test_shared_vocabulary_and_2k_columns(self):
        result = self.build('both')
        selected = result.kmer_vocabulary.loc[result.kmer_vocabulary['keep'], 'kmer'].tolist()
        self.assertEqual(len(selected), 2)
        self.assertNotIn('ZZZ', selected)
        prefixes = [c for c in result.numeric_columns if c.endswith('3mer_'+selected[0])]
        self.assertEqual(set(prefixes), {'unweighted_3mer_'+selected[0], 'weighted_3mer_'+selected[0]})
        self.assertEqual(len(result.numeric_columns), 1 + 4)
        self.assertEqual(set(result.train['sample_id']), set(TRAIN_IDS))
        self.assertEqual(set(result.valid['sample_id']), set(VALID_IDS))
        self.assertEqual(len(result.audit[result.audit.feature_family == '3mer']), 1)
        self.assertEqual(int(result.audit.loc[result.audit.feature_family == '3mer', 'feature_count'].iloc[0]), 4)

    def test_preserves_old_single_modes(self):
        both = self.build('both')
        for rep in ('unweighted', 'weighted'):
            single = self.build(rep)
            self.assertEqual(
                both.kmer_vocabulary['kmer'].tolist(),
                single.kmer_vocabulary['kmer'].tolist(),
            )
            for part in ('train', 'valid'):
                columns = [c for c in getattr(single, part).columns if c.startswith(rep + '_3mer_')]
                self.assertEqual(len(columns), 2)
                np.testing.assert_allclose(getattr(both, part)[columns], getattr(single, part)[columns])

    def test_outer_holdout_labels_do_not_select_vocabulary(self):
        one = self.build('both')
        changed = inputs()
        # Only holdout clonotypes change; fold vocabulary and training features remain identical.
        changed.kmer_unweighted['V1'] = {'ZZZ': 999999.0}
        changed.kmer_weighted['V1'] = {'ZZZ': 999999.0}
        changed_spec = spec('both')
        changed_spec.base.features.public_reference = SimpleNamespace(enabled=False)
        two = build_fold_features(changed, changed_spec, TRAIN_IDS, VALID_IDS, context='synthetic')
        pd.testing.assert_frame_equal(one.kmer_vocabulary, two.kmer_vocabulary)
        pd.testing.assert_frame_equal(one.train, two.train)


if __name__ == '__main__':
    unittest.main(verbosity=2)
