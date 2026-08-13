#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

THIS = Path(__file__).resolve()
SRC = THIS.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ra_ild_trb.dynamic_repertoire import (
    DynamicRepertoireSource,
    read_and_filter_metadata,
    render_dynamic_registry,
    render_phase2_plan,
    resolve_dynamic_repertoire,
    validate_dynamic_source,
)
from ra_ild_trb.enriched_dictionary import (
    fit_enriched_dictionary,
    load_reference,
    read_sample_repertoire,
    run_existing_grid_cv_via_source_shim,
    score_sample,
    transform_repertoires,
    write_reference,
)
from ra_ild_trb.experiment_spec import ExperimentSpecError, load_advanced_experiment


class Phase5AdvancedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="phase5_unit_")
        self.root = Path(self.tmp.name)
        self.registry = self.root / "registry.yaml"
        self.registry.write_text(
            yaml.safe_dump({
                "schema_version": "1.0",
                "registry": {"id": "r", "receptor": "TRB", "stage": "x", "default_repertoire_source": "top10000", "description": "x"},
                "repertoire_sources": {
                    "all": {
                        "status": "available", "representation": "full", "source_layer": "01_AA_clone_table",
                        "default_path": "TRB/result/01_AA_clone_table", "file_glob": "*_AA_clone_table.csv",
                        "sample_filename_template": "{sample_id}_TRB_CDR3_AA_clone_table.csv", "summary_kind": "step01_full",
                        "summary_path": "TRB/result/01_AA_clone_table/01_AA_clone_table_summary.csv", "parent_source": None,
                        "required_columns": ["sample_id", "cdr3_aa"], "weight_column": "frequency_norm", "top_k": None,
                    },
                    "top10000": {
                        "status": "available", "representation": "topk", "source_layer": "01b_AA_clone_table_top10000",
                        "default_path": "TRB/result/01b_AA_clone_table_top10000", "file_glob": "*_top10000.csv",
                        "sample_filename_template": "{sample_id}_TRB_CDR3_AA_clone_table_top10000.csv", "summary_kind": "step01b_topk",
                        "summary_path": "TRB/result/01b_AA_clone_table_top10000/01b_top10000_summary.csv", "parent_source": "all",
                        "required_columns": ["sample_id", "cdr3_aa"], "weight_column": "frequency_norm", "top_k": 10000,
                    },
                },
                "feature_groups": {
                    "clinical_age_sex": {
                        "family": "clinical", "status": "defined", "module_type": "metadata_static", "fit_scope": "none",
                        "leakage_policy": "metadata_only", "repertoire_dependency": "none", "supported_repertoire_sources": [],
                        "producer": "metadata", "model_role": "primary_predictor", "selector": {"exact": ["age","sex"], "prefixes": []},
                        "reference_drop_columns": [], "notes": "",
                    },
                    "tcr_diversity": {
                        "family": "tcr_stat", "status": "defined", "module_type": "sample_static", "fit_scope": "per_sample",
                        "leakage_policy": "x", "repertoire_dependency": "repertoire", "supported_repertoire_sources": ["all","top10000"],
                        "producer": "step02_core", "model_role": "candidate_predictor", "selector": {"exact": ["Shannon"], "prefixes": []},
                        "reference_drop_columns": [], "notes": "",
                    },
                    "tcr_3mer_unweighted": {
                        "family": "kmer", "status": "defined", "module_type": "training_vocabulary", "fit_scope": "training_only",
                        "leakage_policy": "x", "repertoire_dependency": "repertoire", "supported_repertoire_sources": ["all","top10000"],
                        "producer": "step02", "model_role": "candidate_predictor", "selector": {"exact": [], "prefixes": ["unweighted_3mer_"]},
                        "reference_drop_columns": [], "notes": "",
                    },
                },
            }, sort_keys=False), encoding="utf-8"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def write_config(self, **changes):
        raw = {
            "schema_version": "1.0",
            "experiment": {"id": "x", "receptor": "TRB"},
            "cohort": {"scope": "total"},
            "repertoire": {"mode": "topk", "top_k": 3},
            "features": {
                "clinical_groups": ["clinical_age_sex"],
                "static_tcr_groups": ["tcr_diversity"],
                "kmer": {"enabled": True, "k": 3, "representation": "unweighted", "top_n": 500},
                "enriched_dictionary": {"enabled": True, "threshold_pct": 20, "delta_pct": 10, "features": ["RA_dict_hit_rate"]},
            },
            "model": {"algorithm": "elastic_net"},
            "validation": {"folds": 5, "repeats": 100, "split_candidates": 300, "seed": 7},
            "compute": {},
        }
        for section, value in changes.items():
            raw[section] = value
        path = self.root / f"config_{len(list(self.root.glob('config_*.yaml')))}.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return path

    def test_defaults_workers_one_and_repeat_configurable(self):
        spec = load_advanced_experiment(self.write_config())
        self.assertEqual(spec.compute.workers, 1)
        self.assertEqual(spec.validation.repeats, 100)
        raw = yaml.safe_load(spec.source_path.read_text())
        raw["validation"]["repeats"] = 50
        raw["compute"]["workers"] = 8
        p = self.root / "repeat50.yaml"
        p.write_text(yaml.safe_dump(raw), encoding="utf-8")
        s2 = load_advanced_experiment(p)
        self.assertEqual(s2.validation.repeats, 50)
        self.assertEqual(s2.compute.workers, 8)

    def test_workers_max_16(self):
        raw = yaml.safe_load(self.write_config().read_text())
        raw["compute"]["workers"] = 16
        p = self.root / "w16.yaml"; p.write_text(yaml.safe_dump(raw), encoding="utf-8")
        self.assertEqual(load_advanced_experiment(p).compute.workers, 16)
        raw["compute"]["workers"] = 17
        p2 = self.root / "w17.yaml"; p2.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with self.assertRaises(ExperimentSpecError): load_advanced_experiment(p2)

    def test_enrich_source_override_rejected(self):
        raw = yaml.safe_load(self.write_config().read_text())
        raw["features"]["enriched_dictionary"]["repertoire_source"] = "all"
        p = self.root / "badsource.yaml"; p.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with self.assertRaises(ExperimentSpecError): load_advanced_experiment(p)

    def test_dynamic_topk_resolution(self):
        spec = load_advanced_experiment(self.write_config())
        src = resolve_dynamic_repertoire(spec, self.root)
        self.assertEqual(src.source_id, "top3")
        self.assertEqual(src.top_k, 3)
        self.assertTrue(str(src.aa_dir).endswith("01b_AA_clone_table_top3"))
        self.assertTrue(src.sample_filename("S1").endswith("_top3.csv"))

    def test_render_dynamic_registry_and_phase2_plan(self):
        spec = load_advanced_experiment(self.write_config())
        registry = render_dynamic_registry(self.registry, spec)
        self.assertIn("top3", registry["repertoire_sources"])
        self.assertIn("top3", registry["feature_groups"]["tcr_diversity"]["supported_repertoire_sources"])
        plan = render_phase2_plan(spec)
        self.assertEqual(plan["plan"]["repertoire_source"], "top3")
        self.assertIn("tcr_3mer_unweighted", plan["plan"]["feature_groups"])

    def test_scope_filter_pbmc(self):
        m = pd.DataFrame({"libraryid":["A","B","C","D"], "cohort":["RA","ILD","RA","ILD"], "material":["PBMC","PBMC","buffercoat","buffercoat"]})
        p = self.root / "meta.csv"; m.to_csv(p,index=False)
        out = read_and_filter_metadata(p, "pbmc")
        self.assertEqual(out["libraryid"].tolist(), ["A","B"])

    def _make_repertoires(self):
        d = self.root / "aa"; d.mkdir(exist_ok=True)
        samples = {
            "R1": [("A", .4),("B",.3),("C",1e-12)],
            "R2": [("A", .5),("B",.5)],
            "I1": [("A", .4),("D",.6)],
            "I2": [("D", .5),("E",.5)],
            "T":  [("A", .25),("B",.25),("D",.25),("F",.25)],
        }
        reps = {}
        for sid, rows in samples.items():
            p = d / f"{sid}.csv"
            pd.DataFrame(rows, columns=["cdr3_aa","read_fraction"]).to_csv(p,index=False)
            reps[sid] = read_sample_repertoire(p, sid)
        return reps

    def test_dictionary_formula_and_no_abundance_filter(self):
        reps = self._make_repertoires()
        meta = pd.DataFrame({"libraryid":["R1","R2","I1","I2"], "cohort":["RA","RA","ILD","ILD"]})
        ref = fit_enriched_dictionary(meta, reps, source_id="all", threshold_pct=50, delta_pct=25)
        self.assertEqual(set(ref.ra_clones), {"A","B","C"})  # C survives despite 1e-12 abundance.
        self.assertEqual(set(ref.ild_clones), {"D","E"})
        score = score_sample(reps["T"], ref)
        self.assertAlmostEqual(score["RA_dict_hit_rate"], 2/3)
        self.assertAlmostEqual(score["RA_hit_fraction_of_sample"], 2/4)
        self.assertAlmostEqual(score["ILD_dict_hit_rate"], 1/2)

    def test_source_choice_changes_dictionary(self):
        reps = self._make_repertoires()
        meta = pd.DataFrame({"libraryid":["R1","R2","I1","I2"], "cohort":["RA","RA","ILD","ILD"]})
        full = fit_enriched_dictionary(meta, reps, source_id="all", threshold_pct=50, delta_pct=25)
        # Simulate Top-2 by removing C/E from the sample repertoires.
        top = dict(reps)
        for sid in ["R1","R2","I1","I2"]:
            keep = top[sid].clones[:2]
            top[sid] = type(top[sid])(sid, keep, {k: top[sid].read_fraction[k] for k in keep})
        cut = fit_enriched_dictionary(meta, top, source_id="top2", threshold_pct=50, delta_pct=25)
        self.assertNotEqual(full.ra_clones, cut.ra_clones)
        self.assertEqual(cut.source_id, "top2")

    def test_reference_roundtrip_and_transform(self):
        reps = self._make_repertoires()
        meta = pd.DataFrame({"libraryid":["R1","R2","I1","I2"], "cohort":["RA","RA","ILD","ILD"]})
        ref = fit_enriched_dictionary(meta, reps, source_id="all", threshold_pct=50, delta_pct=25)
        out = self.root / "ref"
        write_reference(ref, out)
        loaded = load_reference(out)
        self.assertEqual(loaded.ra_clones, ref.ra_clones)
        test = pd.DataFrame({"libraryid":["T"], "cohort":["RA"]})
        score = transform_repertoires(test, reps, loaded)
        self.assertAlmostEqual(score.loc[0,"RA_dict_hit_rate"], 2/3)

    def test_validate_topk_cache(self):
        src = DynamicRepertoireSource("top3", "topk", 3, self.root / "top3", self.root / "top3/01b_top3_summary.csv", "{sample_id}_TRB_CDR3_AA_clone_table_top3.csv")
        src.aa_dir.mkdir()
        for sid in ["S1","S2"]:
            pd.DataFrame({"sample_id":[sid]*3,"cdr3_aa":["A","B","C"]}).to_csv(src.sample_path(sid), index=False)
        pd.DataFrame({"sample_id":["S1","S2"],"top_k":[3,3],"retained_aa_clone_number":[3,3],"qc_pass":[True,True]}).to_csv(src.summary_path,index=False)
        audit = validate_dynamic_source(src,["S1","S2"],deep=True)
        self.assertEqual(audit["status"],"PASS")

    def test_all_mode_rejects_top_k(self):
        raw = yaml.safe_load(self.write_config().read_text())
        raw["repertoire"] = {"mode":"all", "top_k":10000}
        p=self.root/"badall.yaml"; p.write_text(yaml.safe_dump(raw),encoding="utf-8")
        with self.assertRaises(ExperimentSpecError): load_advanced_experiment(p)

    def test_delta_cannot_exceed_threshold(self):
        raw = yaml.safe_load(self.write_config().read_text())
        raw["features"]["enriched_dictionary"]["threshold_pct"] = 10
        raw["features"]["enriched_dictionary"]["delta_pct"] = 20
        p=self.root/"baddelta.yaml"; p.write_text(yaml.safe_dump(raw),encoding="utf-8")
        with self.assertRaises(ExperimentSpecError): load_advanced_experiment(p)

    def test_model_algorithm_contracts(self):
        for algo in ["elastic_net","ridge","lasso","linear_svm","xgboost"]:
            raw = yaml.safe_load(self.write_config().read_text())
            raw["model"]["algorithm"] = algo
            p=self.root/f"{algo}.yaml"; p.write_text(yaml.safe_dump(raw),encoding="utf-8")
            self.assertEqual(load_advanced_experiment(p).model.algorithm, algo)

    def test_cv_shim_topk_resolve_keeps_legacy_filename_and_workers(self):
        reps = self._make_repertoires()
        source_dir = self.root / "source_top3"; source_dir.mkdir()
        for sid in ["R1","R2","I1","I2"]:
            # Reproduce a dynamic Top-K source filename such as the real
            # ``*_TRB_CDR3_AA_clone_table_top10000.csv`` inputs.
            original = self.root / "aa" / f"{sid}.csv"
            (source_dir / f"{sid}_TRB_CDR3_AA_clone_table_top3.csv").write_bytes(original.read_bytes())
        source = DynamicRepertoireSource(
            "top3", "topk", 3, source_dir, None,
            "{sample_id}_TRB_CDR3_AA_clone_table_top3.csv",
        )
        meta = pd.DataFrame({"libraryid":["R1","R2","I1","I2"],"cohort":["RA","RA","ILD","ILD"],"material":["PBMC"]*4})
        called = {}
        stub = types.ModuleType("ra_ild_trb.enrich_dictionary_grid")
        def fake(**kwargs):
            called.update(kwargs)
            shim = Path(kwargs["project_root"])
            self.assertTrue((shim/"TRB/metadata.csv").is_file())
            legacy_files = sorted((shim/"TRB/result/01_AA_clone_table").glob("*_AA_clone_table.csv"))
            self.assertEqual(len(legacy_files), 4)
            for path in legacy_files:
                self.assertFalse(path.is_symlink())
                # This is the exact operation that exposed the real bug in the
                # legacy loader.  The resolved basename must stay legacy-safe.
                self.assertTrue(path.resolve().name.endswith("_AA_clone_table.csv"))
            return {"ok": True, "workers": kwargs["workers"]}
        stub.run_project_grid_analysis = fake
        sys.modules["ra_ild_trb.enrich_dictionary_grid"] = stub
        try:
            out = run_existing_grid_cv_via_source_shim(
                repository_root=self.root, metadata=meta, source=source, output_dir=self.root/"cv",
                threshold_pct=20, delta_pct=10, folds=5, repeats=50, split_candidates=300,
                seed=7, workers=8, overwrite=False,
            )
        finally:
            sys.modules.pop("ra_ild_trb.enrich_dictionary_grid", None)
        self.assertEqual(out["workers"],8)
        self.assertEqual(called["repeats"],50)
        self.assertEqual(called["threshold_values"],[20.0])
        self.assertEqual(called["delta_values"],[10.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
