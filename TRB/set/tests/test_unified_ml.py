#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ra_ild_trb.enriched_dictionary import (
    SampleRepertoire,
    fit_enriched_dictionary,
    score_sample,
)
from ra_ild_trb.feature_framework import load_feature_registry
from ra_ild_trb.fold_features import (
    FoldFeatureInputs,
    exact_loo_enriched_scores,
    rank_training_kmers,
    select_static_features,
)
from ra_ild_trb.dynamic_repertoire import DynamicRepertoireSource
from ra_ild_trb.unified_cv import load_repeated_holdout_bank, run_repeat, run_repeats
from ra_ild_trb.unified_experiment import load_unified_experiment
from ra_ild_trb.unified_modeling import (
    build_candidates,
    fit_candidate,
    fit_preprocessor,
)
from ra_ild_trb.fold_features import load_fold_feature_inputs


DEFAULT_STATIC = [
    "tcr_diversity",
    "tcr_clonal_expansion",
    "tcr_aa_length",
    "tcr_aa_composition_unweighted",
    "tcr_aa_composition_weighted",
    "tcr_physicochemical_unweighted",
    "tcr_physicochemical_weighted",
]


def write_config(path: Path, *, algorithm="elastic_net", folds=2, repeats=1, top_n=2, static=None):
    data = {
        "schema_version": "1.0",
        "experiment": {"id": "test_unified", "receptor": "TRB"},
        "cohort": {"scope": "total"},
        "repertoire": {"mode": "all", "top_k": None},
        "features": {
            "clinical_groups": ["clinical_age_sex"],
            "static_tcr_groups": static or ["tcr_diversity"],
            "kmer": {
                "enabled": True, "k": 3, "representation": "unweighted", "top_n": top_n,
                "min_sample_count": 1, "min_prevalence": 0.0,
                "min_unweighted_variance": 0.0, "min_weighted_variance": 0.0,
            },
            "enriched_dictionary": {
                "enabled": True, "threshold_pct": 20, "delta_pct": 10,
                "features": ["RA_dict_hit_rate"],
            },
        },
        "model": {
            "algorithm": algorithm,
            "positive_label": "ILD",
            "negative_label": "RA",
            "tuning_primary_metric": "roc_auc",
            "parameters": {
                "elastic_net": {"alpha_grid": [0.5], "lambda_grid": [1.0]},
                "ridge": {"alpha_grid": [0.0], "lambda_grid": [1.0]},
                "lasso": {"alpha_grid": [1.0], "lambda_grid": [1.0]},
                "linear_svm": {"C_grid": [0.1]},
                "xgboost": {
                    "fixed_parameters": {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "verbosity": 0},
                    "hyperparameters": {"max_depth": [1], "n_estimators": [5], "learning_rate": [0.1]},
                    "search_n_iter": 1,
                    "search_random_state": 1,
                    "n_jobs_per_fit": 1,
                },
            },
        },
        "validation": {
            "folds": folds, "repeats": repeats, "split_candidates": 3, "seed": 11,
            "inner_split_candidates": 3, "inner_split_seed": 17,
        },
        "compute": {"workers": 1},
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def simple_repertoires(ids, labels):
    out = {}
    for index, (sid, label) in enumerate(zip(ids, labels)):
        clones = ["CASSTTT", "CASSRAA" if label == "RA" else "CASSILD"]
        # add one valid unique-ish clone drawn from a deterministic set
        extras = ["CASSAAA", "CASSCCC", "CASSDDD", "CASSEEE", "CASSFFF", "CASSGGG"]
        clones.append(extras[index % len(extras)])
        rf = {clone: 1.0 / len(clones) for clone in clones}
        out[sid] = SampleRepertoire(sid, tuple(clones), rf)
    return out


class UnifiedMLTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_candidate_contracts(self):
        template = Path(__file__).resolve().parents[1] / "configs/feature_framework/trb_experiment_unified_ml_template_v1.yaml"
        raw = yaml.safe_load(template.read_text())
        expected = {
            "elastic_net": 24,
            "ridge": 8,
            "lasso": 8,
            "linear_svm": 8,
            "xgboost": 100,
        }
        for algorithm, count in expected.items():
            copy = json.loads(json.dumps(raw))
            copy["model"]["algorithm"] = algorithm
            path = self.root / f"{algorithm}.yaml"
            path.write_text(yaml.safe_dump(copy, sort_keys=False))
            spec = load_unified_experiment(path)
            self.assertEqual(len(build_candidates(spec)), count, algorithm)
            self.assertEqual(spec.workers, 8)
            self.assertEqual(spec.cv.repeats, 100)
        self.assertEqual(load_unified_experiment(self.root / "ridge.yaml").model.ridge.alpha_grid, (0.0,))
        self.assertEqual(load_unified_experiment(self.root / "lasso.yaml").model.lasso.alpha_grid, (1.0,))

    def test_static_default_is_83(self):
        registry_path = Path(__file__).resolve().parents[1] / "configs/feature_framework/trb_feature_registry_v1.yaml"
        registry = load_feature_registry(registry_path)
        columns = ["sample_id"]
        for group_id in DEFAULT_STATIC:
            group = registry.feature_groups[group_id]
            columns.extend(group.selector.exact)
        chosen = select_static_features(registry, columns, DEFAULT_STATIC)
        self.assertEqual(len(chosen), 83)
        self.assertEqual(len(chosen), len(set(chosen)))
        for group_id in DEFAULT_STATIC:
            for drop in registry.feature_groups[group_id].reference_drop_columns:
                self.assertNotIn(drop, chosen)

    def test_kmer_vocabulary_uses_training_only(self):
        config = self.root / "c.yaml"
        write_config(config)
        spec = load_unified_experiment(config)
        ids = ("R1", "R2", "I1", "I2", "V1")
        meta = pd.DataFrame({
            "sample_id": ids,
            "cohort": ["RA", "RA", "ILD", "ILD", "RA"],
            "age": [1,2,3,4,5], "sex": ["F","M","F","M","F"],
        })
        static = pd.DataFrame({"sample_id": ids, "Shannon": range(5)})
        u = {sid: {"AAA": 0.5, "CCC": 0.2} for sid in ids}
        w = {sid: {"AAA": 0.5, "CCC": 0.2} for sid in ids}
        u["V1"] = {"VVV": 1.0}
        w["V1"] = {"VVV": 1.0}
        inputs = FoldFeatureInputs(
            metadata=meta, static_values=static, static_feature_names=("Shannon",),
            clinical_numeric=("age",), clinical_categorical=("sex",),
            source=DynamicRepertoireSource("all", "full", None, self.root, None, "{sample_id}.csv"),
            kmer_unweighted=u, kmer_weighted=w, enrich_repertoires={},
        )
        vocab = rank_training_kmers(("R1","R2","I1","I2"), inputs, spec)
        self.assertNotIn("VVV", vocab["kmer"].tolist())
        self.assertEqual(len(vocab.loc[vocab["keep"]]), 2)

    def test_fast_loo_enrich_exactly_matches_naive(self):
        ids = [f"R{i}" for i in range(4)] + [f"I{i}" for i in range(4)]
        labels = ["RA"] * 4 + ["ILD"] * 4
        metadata = pd.DataFrame({"sample_id": ids, "cohort": labels})
        reps = simple_repertoires(ids, labels)
        fast = exact_loo_enriched_scores(
            metadata, ids, reps, threshold_pct=20.0, delta_pct=10.0
        ).set_index("sample_id")
        for sid in ids:
            train = metadata.loc[metadata["sample_id"] != sid].rename(columns={"sample_id": "libraryid"})
            ref = fit_enriched_dictionary(
                train, reps, source_id="all", threshold_pct=20.0, delta_pct=10.0,
                id_col="libraryid", cohort_col="cohort"
            )
            expected = score_sample(reps[sid], ref)
            for key, value in expected.items():
                got = float(fast.loc[sid, key])
                if np.isnan(value):
                    self.assertTrue(np.isnan(got), (sid, key))
                else:
                    self.assertAlmostEqual(got, float(value), places=12, msg=f"{sid}/{key}")

    def test_preprocessing_is_training_only(self):
        train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "sex": ["F","M","F","M"]})
        valid = pd.DataFrame({"x": [1000000.0], "sex": ["F"]})
        pre = fit_preprocessor(train, ["x"], ["sex"], zero_sd_tolerance=1e-12)
        self.assertAlmostEqual(float(pre.means[0]), 1.5)
        self.assertAlmostEqual(float(pre.sds[0]), float(np.std([0,1,2,3], ddof=0)))
        X, _ = pre.transform(valid)
        self.assertTrue(np.isfinite(X).all())

    def test_model_smoke_all_five_algorithms(self):
        X = np.array([[-2,-1],[-1,-1],[-1,0],[1,0],[1,1],[2,1]], float)
        y = np.array([0,0,0,1,1,1], int)
        for algorithm in ("elastic_net", "ridge", "lasso", "linear_svm", "xgboost"):
            config = self.root / f"{algorithm}.yaml"
            write_config(config, algorithm=algorithm)
            spec = load_unified_experiment(config)
            candidate = build_candidates(spec)[0]
            fit = fit_candidate(candidate, X, y, spec, random_state=1)
            score = fit.score(X)
            self.assertEqual(score.shape, (6,))
            self.assertTrue(np.isfinite(score).all())

    def test_validation_labels_do_not_change_learned_feature_values(self):
        config = self.root / "leak.yaml"
        write_config(config, top_n=1)
        spec = load_unified_experiment(config)
        ids = ["R1","R2","R3","I1","I2","I3","VRA","VILD"]
        labels = ["RA","RA","RA","ILD","ILD","ILD","RA","ILD"]
        metadata = pd.DataFrame({
            "sample_id": ids, "cohort": labels,
            "age": np.arange(len(ids))+40,
            "sex": ["F","M"]*4,
        })
        reps = simple_repertoires(ids, labels)
        static = pd.DataFrame({"sample_id": ids, "Shannon": np.arange(len(ids), dtype=float)})
        u = {sid: {"AAA": 0.5, "CCC": 0.2} for sid in ids}
        w = {sid: {"AAA": 0.5, "CCC": 0.2} for sid in ids}
        inputs = FoldFeatureInputs(
            metadata=metadata, static_values=static, static_feature_names=("Shannon",),
            clinical_numeric=("age",), clinical_categorical=("sex",),
            source=DynamicRepertoireSource("all", "full", None, self.root, None, "{sample_id}.csv"),
            kmer_unweighted=u, kmer_weighted=w, enrich_repertoires=reps,
        )
        from ra_ild_trb.fold_features import build_fold_features
        train_ids = ids[:6]
        valid_ids = ids[6:]
        a = build_fold_features(inputs, spec, train_ids, valid_ids, context="a")
        mutated_meta = metadata.copy()
        mutated_meta.loc[mutated_meta.sample_id == "VRA", "cohort"] = "ILD"
        mutated_meta.loc[mutated_meta.sample_id == "VILD", "cohort"] = "RA"
        inputs2 = FoldFeatureInputs(
            metadata=mutated_meta, static_values=static, static_feature_names=("Shannon",),
            clinical_numeric=("age",), clinical_categorical=("sex",),
            source=inputs.source, kmer_unweighted=u, kmer_weighted=w, enrich_repertoires=reps,
        )
        b = build_fold_features(inputs2, spec, train_ids, valid_ids, context="b")
        av = a.valid.set_index("sample_id")["RA_dict_hit_rate"].sort_index()
        bv = b.valid.set_index("sample_id")["RA_dict_hit_rate"].sort_index()
        np.testing.assert_allclose(av.to_numpy(), bv.to_numpy(), rtol=0, atol=0)

    def _build_end_to_end_repo(self):
        root = self.root / "repo"
        (root / "TRB/result/01_AA_clone_table").mkdir(parents=True)
        (root / "TRB/set/train").mkdir(parents=True)
        (root / "TRB/set/test").mkdir(parents=True)
        (root / "TRB/set/split_sets/ra_ild_repeat100_v1").mkdir(parents=True)
        # registry copied from installed test environment
        reg_src = Path(__file__).resolve().parents[1] / "configs/feature_framework/trb_feature_registry_v1.yaml"
        reg_dst = root / "TRB/set/configs/feature_framework/trb_feature_registry_v1.yaml"
        reg_dst.parent.mkdir(parents=True)
        reg_dst.write_text(reg_src.read_text())

        ids = [f"R{i:02d}" for i in range(10)] + [f"I{i:02d}" for i in range(10)]
        cohorts = ["RA"]*10 + ["ILD"]*10
        metadata = pd.DataFrame({
            "patient": [f"P{i:02d}" for i in range(20)],
            "libraryid": ids,
            "batch": ["B1","B2"]*10,
            "sex": ["F","M"]*10,
            "age": np.arange(40,60),
            "cohort": cohorts,
            "material": ["PBMC"]*20,
        })
        # original 70/30 partition only supplies inputs; repeated holdout later recombines them
        train_meta = pd.concat([metadata.iloc[:7], metadata.iloc[10:17]], ignore_index=True)
        test_meta = pd.concat([metadata.iloc[7:10], metadata.iloc[17:20]], ignore_index=True)
        train_meta.to_csv(root / "TRB/set/train/metadata_train_70.csv", index=False)
        test_meta.to_csv(root / "TRB/set/test/metadata_test_30.csv", index=False)

        train_step = root / "TRB/set/train/result/step02"
        test_step = root / "TRB/set/test/result/step02"
        train_step.mkdir(parents=True); test_step.mkdir(parents=True)
        core = pd.DataFrame({
            "sample_id": ids,
            "Shannon": np.linspace(1.0, 2.0, 20),
            "Simpson": np.linspace(0.1, 0.2, 20),
            "inverse_Simpson": np.linspace(10, 5, 20),
            "Gini": np.linspace(0.3, 0.5, 20),
            "clonality": np.linspace(0.1, 0.3, 20),
        })
        core.set_index("sample_id").loc[train_meta.libraryid].reset_index().to_csv(train_step / "02_core_sample_features.csv", index=False)
        core.set_index("sample_id").loc[test_meta.libraryid].reset_index().to_csv(test_step / "02_core_sample_features.csv", index=False)

        reps = simple_repertoires(ids, cohorts)
        for sid in ids:
            rep = reps[sid]
            n = len(rep.clones)
            pd.DataFrame({
                "sample_id": [sid]*n,
                "cdr3_aa": rep.clones,
                "frequency_norm": [1/n]*n,
                "read_fraction": [1/n]*n,
            }).to_csv(root / f"TRB/result/01_AA_clone_table/{sid}_TRB_CDR3_AA_clone_table.csv", index=False)

        # repeated holdout: 8 RA + 8 ILD train; 2+2 holdout
        holdout = {"R08","R09","I08","I09"}
        rows = []
        for sid in ids:
            rows.append({"sample_id": sid, "repeat_index": 1, "role": "holdout" if sid in holdout else "train", "split_id": "repeat_001"})
        pd.DataFrame(rows).to_csv(root / "TRB/set/split_sets/ra_ild_repeat100_v1/repeated_holdout_assignments.csv", index=False)
        return root, train_step, test_step

    def test_end_to_end_one_repeat(self):
        root, train_step, test_step = self._build_end_to_end_repo()
        config = root / "config.yaml"
        write_config(config, algorithm="elastic_net", folds=2, repeats=1, top_n=2)
        spec = load_unified_experiment(config)
        inputs = load_fold_feature_inputs(
            spec,
            repository_root=root,
            train_metadata=root / "TRB/set/train/metadata_train_70.csv",
            test_metadata=root / "TRB/set/test/metadata_test_30.csv",
            train_step02_dir=train_step,
            test_step02_dir=test_step,
            registry_path=root / "TRB/set/configs/feature_framework/trb_feature_registry_v1.yaml",
        )
        self.assertEqual(len(inputs.sample_ids), 20)
        bank = load_repeated_holdout_bank(
            root / "TRB/set/split_sets/ra_ild_repeat100_v1/repeated_holdout_assignments.csv",
            inputs.sample_ids,
            requested_repeats=1,
        )
        result = run_repeat(inputs, spec, bank, repeat=1)
        self.assertEqual(result.metrics.shape[0], 1)
        self.assertEqual(result.predictions.shape[0], 4)
        self.assertEqual(int(result.metrics.iloc[0]["n_outer_train"]), 16)
        self.assertEqual(int(result.metrics.iloc[0]["n_outer_holdout"]), 4)
        self.assertTrue(np.isfinite(float(result.metrics.iloc[0]["roc_auc"])))
        policies = set(result.feature_audit.loc[
            result.feature_audit["feature_family"] == "enriched_dictionary", "fit_policy"
        ])
        self.assertEqual(policies, {"training_exact_LOO;validation_training_reference_only"})


    def test_concurrent_workers_two_repeats(self):
        root, train_step, test_step = self._build_end_to_end_repo()
        assignment_path = root / "TRB/set/split_sets/ra_ild_repeat100_v1/repeated_holdout_assignments.csv"
        one = pd.read_csv(assignment_path)
        two = one.copy()
        two["repeat_index"] = 2
        two["split_id"] = "repeat_002"
        pd.concat([one, two], ignore_index=True).to_csv(assignment_path, index=False)
        config = root / "config2.yaml"
        write_config(config, algorithm="ridge", folds=2, repeats=2, top_n=2)
        spec = load_unified_experiment(config)
        inputs = load_fold_feature_inputs(
            spec, repository_root=root,
            train_metadata=root / "TRB/set/train/metadata_train_70.csv",
            test_metadata=root / "TRB/set/test/metadata_test_30.csv",
            train_step02_dir=train_step, test_step02_dir=test_step,
            registry_path=root / "TRB/set/configs/feature_framework/trb_feature_registry_v1.yaml",
        )
        bank = load_repeated_holdout_bank(assignment_path, inputs.sample_ids, requested_repeats=2)
        results = run_repeats(inputs, spec, bank, [1, 2], workers=2)
        self.assertEqual([x.repeat for x in results], [1, 2])
        self.assertTrue(all(x.metrics.shape[0] == 1 for x in results))

    def test_cli_writes_one_repeat_outputs(self):
        root, train_step, test_step = self._build_end_to_end_repo()
        config = root / "config.yaml"
        write_config(config, algorithm="elastic_net", folds=2, repeats=1, top_n=2)
        installed_root = Path(__file__).resolve().parents[1]
        src_from = installed_root / "src/ra_ild_trb"
        src_to = root / "TRB/set/src/ra_ild_trb"
        src_to.mkdir(parents=True, exist_ok=True)
        for source in src_from.glob("*.py"):
            shutil.copy2(source, src_to / source.name)
        scripts_to = root / "TRB/set/scripts"
        scripts_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(installed_root / "scripts/run_06_unified_ml.py", scripts_to / "run_06_unified_ml.py")
        out = root / "TRB/result/phase6_cli"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "TRB/set/src")
        command = [
            "python3", str(scripts_to / "run_06_unified_ml.py"),
            "--config", str(config),
            "--train-step02-dir", str(train_step),
            "--test-step02-dir", str(test_step),
            "--output-dir", str(out),
            "--train-metadata", "TRB/set/train/metadata_train_70.csv",
            "--test-metadata", "TRB/set/test/metadata_test_30.csv",
            "--repeat", "1", "--workers", "1",
        ]
        completed = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
        if completed.returncode != 0:
            self.fail(f"CLI failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
        self.assertIn("WRITE PASS", completed.stdout)
        metrics = pd.read_csv(out / "06_outer_metrics.csv")
        self.assertEqual(metrics.shape[0], 1)
        self.assertTrue((out / "06_run_manifest.json").is_file())
        self.assertTrue((out / "06_run_summary.md").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
