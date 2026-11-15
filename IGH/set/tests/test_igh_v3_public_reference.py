#!/usr/bin/env python3
"""Phase 2B: V2 Public-reference adapter scientific/compatibility tests."""
from __future__ import annotations

import dataclasses
import pathlib
import sys
import tempfile
import types
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ra_ild_igh.dynamic_repertoire import DynamicRepertoireSource
from ra_ild_igh.experiment_spec import PublicReferenceSpec, load_advanced_experiment
from ra_ild_igh.fold_features import FoldFeatureInputs, build_fold_features
from ra_ild_igh.public_reference import (
    apply_reference,
    build_reference_from_presence,
    public_feature_columns,
)
from ra_ild_igh.v3_public_adapter import (
    V2_PUBLIC_COLUMNS,
    V3PublicAdapterError,
    build_public_fold,
    load_public_input,
)


class TestV3PublicAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = pathlib.Path(self.tmp.name)
        self.ids = tuple(f"P{i:02d}" for i in range(14))
        self.labels = np.asarray(["RA"] * 7 + ["ILD"] * 7)
        alphabet = "ACDEFGHIKLMNPQ"
        common1 = "CASSLGQETQYF"
        common2 = "CASSMGQETQYF"
        ra = "CASSRGQETQYF"
        ild = "CASSPGQETQYF"
        for i, (sid, label) in enumerate(zip(self.ids, self.labels)):
            uniq = f"CASS{alphabet[i]}TDTQYF"
            frame = pd.DataFrame({
                "cdr3_aa": [common1, common2, ra if label == "RA" else ild, uniq],
                "frequency_norm": [0.4, 0.3, 0.2, 0.1],
            })
            frame.to_csv(d / f"{sid}.csv", index=False)
        self.source = DynamicRepertoireSource(
            source_id="top4", representation="topk", top_k=4,
            aa_dir=d, summary_path=None,
            sample_filename_template="{sample_id}.csv",
            weight_column="frequency_norm",
        )
        self.cache = load_public_input(self.ids, self.labels, self.source)
        self.train_ids = tuple(self.ids[i] for i in (list(range(6)) + list(range(7, 13))))
        self.valid_ids = (self.ids[6], self.ids[13])
        self.config = PublicReferenceSpec(enabled=True)

    def test_eight_columns_finite_and_loo(self):
        result = build_public_fold(
            self.cache, self.config, self.train_ids, self.valid_ids, context="inner0",
        )
        self.assertEqual(V2_PUBLIC_COLUMNS, public_feature_columns("raw_bilateral"))
        self.assertEqual(tuple(result.train.columns[1:]), V2_PUBLIC_COLUMNS)
        self.assertEqual(tuple(result.valid.columns[1:]), V2_PUBLIC_COLUMNS)
        self.assertEqual(len(result.train), 12)
        self.assertEqual(len(result.valid), 2)
        self.assertTrue(np.isfinite(result.train.loc[:, list(V2_PUBLIC_COLUMNS)].to_numpy(float)).all())
        self.assertTrue(np.isfinite(result.valid.loc[:, list(V2_PUBLIC_COLUMNS)].to_numpy(float)).all())
        self.assertEqual(result.audit["fit_sample_count"], 12)
        self.assertEqual(result.audit["validation_sample_count"], 2)
        self.assertEqual(result.audit["feature_count"], 8)
        self.assertEqual(result.audit["fit_policy"], "current_train_exact_LOO;validation_training_reference_only")

        # Strong LOO check: first train sample must equal an independently fitted
        # reference that omits this sample entirely (not a cached global ref).
        first = self.train_ids[0]
        row = self.cache.sample_ids.index(first)
        remaining = [self.cache.sample_ids.index(s) for s in self.train_ids[1:]]
        from ra_ild_igh.public_reference import ThresholdScheme
        scheme = ThresholdScheme(
            "test", self.config.global_min_prevalence, self.config.global_min_count,
            self.config.group_min_prevalence, self.config.group_min_count,
            self.config.specific_prevalence_delta,
        )
        reference = build_reference_from_presence(
            self.cache.presence, remaining, self.cache.labels, scheme,
        )
        expected = apply_reference(
            self.cache.presence, self.cache.frequency, [row],
            self.cache.clone_numbers, reference, self.config.epsilon,
        ).loc[row, list(V2_PUBLIC_COLUMNS)].to_numpy(float)
        observed = result.train.loc[result.train["sample_id"] == first, list(V2_PUBLIC_COLUMNS)].iloc[0].to_numpy(float)
        np.testing.assert_allclose(expected, observed, atol=1e-12, rtol=1e-12)

    def test_validation_label_flips_do_not_change_features(self):
        first = build_public_fold(self.cache, self.config, self.train_ids, self.valid_ids, context="outer")
        mutated = self.cache.labels.copy()
        for sid in self.valid_ids:
            row = self.cache.sample_ids.index(sid)
            mutated[row] = "RA" if mutated[row] == "ILD" else "ILD"
        second = build_public_fold(
            dataclasses.replace(self.cache, labels=mutated),
            self.config, self.train_ids, self.valid_ids, context="outer",
        )
        pd.testing.assert_frame_equal(first.valid, second.valid)
        pd.testing.assert_frame_equal(first.train, second.train)
        self.assertEqual(first.audit["reference_hash"], second.audit["reference_hash"])

    def test_validation_only_sequence_not_in_training_reference(self):
        first = build_public_fold(self.cache, self.config, self.train_ids, self.valid_ids, context="outer")
        test_sid = self.valid_ids[0]
        file = self.source.sample_path(test_sid)
        f = pd.read_csv(file)
        f.loc[3, "cdr3_aa"] = "CASSVVVVVVQYF"  # new valid-only sequence
        f.to_csv(file, index=False)
        new_cache = load_public_input(self.ids, self.labels, self.source)
        second = build_public_fold(new_cache, self.config, self.train_ids, self.valid_ids, context="outer")
        pd.testing.assert_frame_equal(first.train, second.train)
        # The catalog may change, but the reference behavior is train-only.
        self.assertEqual(first.audit["reference_size"], second.audit["reference_size"])

    def test_split_overlap_fails(self):
        with self.assertRaises(V3PublicAdapterError):
            build_public_fold(
                self.cache, self.config, self.train_ids, (self.train_ids[0],), context="invalid",
            )

    def test_build_fold_features_integration_and_m0_isolation(self):
        metadata = pd.DataFrame({
            "sample_id": self.ids,
            "cohort": self.labels,
            "age": [40 + i for i in range(len(self.ids))],
            "sex": ["M" if i % 2 else "F" for i in range(len(self.ids))],
        })
        inp = FoldFeatureInputs(
            metadata=metadata,
            static_values=pd.DataFrame({"sample_id": self.ids, "Shannon": np.linspace(1, 2, len(self.ids))}),
            static_feature_names=("Shannon",),
            clinical_numeric=("age",),
            clinical_categorical=("sex",),
            source=self.source,
            kmer_unweighted={}, kmer_weighted={}, enrich_repertoires={},
            public_input=self.cache,
        )
        # Each model has its own feature spec; no hidden defaults should add public to M0.
        def fake_spec(enabled):
            feat = types.SimpleNamespace(
                kmer=types.SimpleNamespace(enabled=False),
                enriched_dictionary=types.SimpleNamespace(enabled=False),
                public_reference=PublicReferenceSpec(enabled=enabled),
            )
            return types.SimpleNamespace(base=types.SimpleNamespace(features=feat))
        m2 = build_fold_features(inp, fake_spec(True), self.train_ids, self.valid_ids, context="m2")
        self.assertEqual(tuple(m2.numeric_columns), ("age", "Shannon", *V2_PUBLIC_COLUMNS))
        self.assertEqual(tuple(m2.categorical_columns), ("sex",))
        self.assertEqual(m2.audit["feature_family"].tolist().count("public_reference"), 1)
        self.assertTrue(np.isfinite(m2.valid[list(V2_PUBLIC_COLUMNS)].to_numpy(float)).all())
        m0 = build_fold_features(inp, fake_spec(False), self.train_ids, self.valid_ids, context="m0")
        self.assertEqual(m0.numeric_columns, ("age", "Shannon"))
        self.assertTrue(set(V2_PUBLIC_COLUMNS).isdisjoint(m0.train.columns))

    def test_yaml_public_config_and_source_override_blocked(self):
        path = pathlib.Path(self.tmp.name) / "test_config.yaml"
        doc = {
            "schema_version": "1.0", "experiment": {"id": "public_test", "receptor": "IGH"},
            "cohort": {"scope": "total"}, "repertoire": {"mode": "topk", "top_k": 4},
            "features": {
                "clinical_groups": ["clinical_age_sex"], "static_tcr_groups": [],
                "kmer": {"enabled": False}, "enriched_dictionary": {"enabled": False},
                "public_reference": {"enabled": True, "feature_set": "raw_bilateral"},
            },
            "model": {"algorithm": "ridge"},
        }
        path.write_text(yaml.safe_dump(doc), encoding="utf-8")
        spec = load_advanced_experiment(path)
        self.assertTrue(spec.features.public_reference.enabled)
        self.assertEqual(spec.features.public_reference.feature_set, "raw_bilateral")
        self.assertEqual(spec.as_dict()["features"]["public_reference"]["repertoire_source"], "top4")
        doc["features"]["public_reference"]["source"] = "full"
        path.write_text(yaml.safe_dump(doc), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_advanced_experiment(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
