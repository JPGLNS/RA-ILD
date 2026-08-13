#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

THIS = Path(__file__).resolve()
REPO = THIS.parents[3]
SRC = REPO / "TRB/set/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ra_ild_trb.orchestrator import (  # noqa:E402
    OrchestratorError,
    build_orchestrator_plan,
    ml_command,
    remove_step02_cache,
    resolve_orchestrator_paths,
    source_builder_command,
    step02_commands,
    validate_step02_cache,
    write_resolved_contract,
)
from ra_ild_trb.unified_experiment import load_unified_experiment  # noqa:E402


DIV = ["Shannon", "Simpson", "inverse_Simpson", "Gini", "clonality"]


def _metadata_rows():
    rows = []
    for cohort, prefix in (("RA", "RA"), ("ILD", "ILD")):
        for i in range(1, 7):
            rows.append({
                "patient": f"P_{prefix}{i}",
                "libraryid": f"{prefix}{i}",
                "batch": "B1" if i <= 3 else "B2",
                "sex": "female" if i % 2 else "male",
                "age": 40 + i + (5 if cohort == "ILD" else 0),
                "cohort": cohort,
                "material": "PBMC" if i % 2 else "buffercoat",
            })
    return rows


def _write_synthetic_fixture(root: Path) -> dict:
    train_ids = ["RA1", "RA2", "RA3", "RA4", "ILD1", "ILD2", "ILD3", "ILD4"]
    test_ids = ["RA5", "RA6", "ILD5", "ILD6"]
    all_rows = {row["libraryid"]: row for row in _metadata_rows()}
    # IMPORTANT: synthetic fixtures must never overwrite canonical project metadata.
    fixture_root = root / "TRB/test_phase7_fixture"
    train_meta = fixture_root / "metadata_train_70.csv"
    test_meta = fixture_root / "metadata_test_30.csv"
    train_meta.parent.mkdir(parents=True, exist_ok=True)
    test_meta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([all_rows[x] for x in train_ids]).to_csv(train_meta, index=False)
    pd.DataFrame([all_rows[x] for x in test_ids]).to_csv(test_meta, index=False)

    source_dir = root / "TRB/result/01b_AA_clone_table_top3"
    source_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for sid in train_ids + test_ids:
        cohort = "RA" if sid.startswith("RA") else "ILD"
        specific = "CASSAAA" if cohort == "RA" else "CASSDDD"
        unique = "CASSAAG" if cohort == "RA" else "CASSDDG"
        frame = pd.DataFrame({
            "sample_id": [sid] * 3,
            "topk_rank": [1, 2, 3],
            "cdr3_aa": [specific, "CASSTTT", unique],
            "frequency_norm": [0.5, 0.3, 0.2],
            "read_fraction": [0.5, 0.3, 0.2],
        })
        frame.to_csv(source_dir / f"{sid}_TRB_CDR3_AA_clone_table_top3.csv", index=False)
        summary.append({
            "sample_id": sid,
            "top_k": 3,
            "retained_aa_clone_number": 3,
            "qc_pass": True,
        })
    pd.DataFrame(summary).to_csv(source_dir / "01b_top3_summary.csv", index=False)

    step_train = root / "TRB/test_phase7_cache/train"
    step_test = root / "TRB/test_phase7_cache/test"
    for mode, ids, path in (("train", train_ids, step_train), ("test", test_ids, step_test)):
        path.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, sid in enumerate(ids):
            is_ild = sid.startswith("ILD")
            rows.append({
                "sample_id": sid,
                "Shannon": 2.0 + 0.05 * idx + (0.15 if is_ild else 0),
                "Simpson": 0.7 + 0.01 * idx,
                "inverse_Simpson": 3.0 + 0.1 * idx,
                "Gini": 0.4 + 0.01 * idx,
                "clonality": 0.3 + (0.05 if is_ild else 0),
            })
        pd.DataFrame(rows).to_csv(path / "02_core_sample_features.csv", index=False)
        (path / "02_resolved_feature_source.json").write_text(json.dumps({
            "mode": mode,
            "sample_count": len(ids),
            "repertoire_input": {"source_id": "top3"},
        }), encoding="utf-8")

    split = root / "TRB/set/split_sets/phase7_synth/repeated_holdout_assignments.csv"
    split.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"sample_id": sid, "repeat_index": 1, "role": "train" if sid in train_ids else "holdout"}
        for sid in train_ids + test_ids
    ]).to_csv(split, index=False)

    config = root / "TRB/set/configs/feature_framework/phase7_synth.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "schema_version": "1.0",
        "experiment": {"id": "phase7_synth", "receptor": "TRB"},
        "cohort": {"scope": "total"},
        "repertoire": {"mode": "topk", "top_k": 3},
        "features": {
            "clinical_groups": ["clinical_age_sex"],
            "static_tcr_groups": ["tcr_diversity"],
            "kmer": {
                "enabled": True,
                "k": 3,
                "representation": "unweighted",
                "top_n": 2,
                "min_sample_count": 1,
                "min_prevalence": 0.0,
                "min_unweighted_variance": 0.0,
                "min_weighted_variance": 0.0,
            },
            "enriched_dictionary": {
                "enabled": True,
                "threshold_pct": 20,
                "delta_pct": 10,
                "features": ["RA_dict_hit_rate"],
            },
        },
        "model": {
            "algorithm": "elastic_net",
            "positive_label": "ILD",
            "negative_label": "RA",
            "tuning_primary_metric": "roc_auc",
            "parameters": {
                "elastic_net": {
                    "alpha_grid": [0.5],
                    "lambda_grid": [0.1],
                    "class_weight": "balanced",
                    "max_iter": 5000,
                    "tolerance": 1.0e-4,
                }
            },
        },
        "validation": {
            "folds": 2,
            "repeats": 1,
            "split_candidates": 2,
            "seed": 123,
            "inner_split_candidates": 5,
            "inner_split_seed": 321,
            "split_assignments": "TRB/set/split_sets/phase7_synth/repeated_holdout_assignments.csv",
        },
        "compute": {"workers": 1},
    }
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return {
        "config": config,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "step_train": step_train,
        "step_test": step_test,
    }


class Phase7OrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = _write_synthetic_fixture(REPO)

    @classmethod
    def tearDownClass(cls):
        for path in [
            REPO / "TRB/test_phase7_cache",
            REPO / "TRB/test_phase7_fixture",
            REPO / "TRB/result/01b_AA_clone_table_top3",
            REPO / "TRB/set/split_sets/phase7_synth",
            REPO / "TRB/set/configs/feature_framework/phase7_synth.yaml",
            REPO / "TRB/result/feature_framework/experiments/phase7_synth",
        ]:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    def _plan(self, **kwargs):
        return build_orchestrator_plan(
            self.fixture["config"],
            repository_root=REPO,
            train_metadata=self.fixture["train_meta"],
            test_metadata=self.fixture["test_meta"],
            train_step02_override=self.fixture["step_train"],
            test_step02_override=self.fixture["step_test"],
            **kwargs,
        )


    def test_synthetic_fixture_never_touches_canonical_metadata(self):
        with tempfile.TemporaryDirectory(prefix="phase7_metadata_isolation_") as tempdir:
            root = Path(tempdir)
            train = root / "TRB/set/train/metadata_train_70.csv"
            test = root / "TRB/set/test/metadata_test_30.csv"
            train.parent.mkdir(parents=True, exist_ok=True)
            test.parent.mkdir(parents=True, exist_ok=True)
            train.write_text("sentinel_train\n", encoding="utf-8")
            test.write_text("sentinel_test\n", encoding="utf-8")
            fixture = _write_synthetic_fixture(root)
            self.assertEqual(train.read_text(encoding="utf-8"), "sentinel_train\n")
            self.assertEqual(test.read_text(encoding="utf-8"), "sentinel_test\n")
            self.assertNotEqual(fixture["train_meta"].resolve(), train.resolve())
            self.assertNotEqual(fixture["test_meta"].resolve(), test.resolve())

    def test_paths_are_deterministic(self):
        spec = load_unified_experiment(self.fixture["config"])
        paths = resolve_orchestrator_paths(spec, REPO, repeat_override=1)
        self.assertTrue(str(paths.experiment_root).endswith("experiments/phase7_synth"))
        self.assertTrue(str(paths.step02_cache_root).endswith("cache/step02/top3"))
        self.assertTrue(str(paths.ml_output_dir).endswith("06_unified_ml/smoke_repeat1"))

    def test_step02_cache_contract(self):
        audit = validate_step02_cache(
            self.fixture["step_train"],
            expected_ids=self.fixture["train_ids"],
            expected_source_id="top3",
            mode="train",
        )
        self.assertEqual(audit.sample_count, 8)
        with self.assertRaises(OrchestratorError):
            validate_step02_cache(
                self.fixture["step_train"],
                expected_ids=list(reversed(self.fixture["train_ids"])),
                expected_source_id="top3",
                mode="train",
            )

    def test_ready_source_has_no_builder_command(self):
        plan = self._plan(repeat_override=1)
        self.assertTrue(plan.source_ready)
        self.assertTrue(plan.step02_ready)
        self.assertIsNone(source_builder_command(plan))

    def test_step02_builder_is_minimal_scaffold(self):
        plan = self._plan(repeat_override=1)
        resolved = write_resolved_contract(plan)
        train, test = step02_commands(plan, resolved)
        self.assertIn("--max-kmers", train)
        self.assertEqual(train[train.index("--max-kmers") + 1], "1")
        self.assertIn("--vocabulary-file", test)

    def test_ml_command_passes_repeat_and_workers(self):
        plan = self._plan(repeat_override=1, workers_override=2)
        cmd = ml_command(plan, config_path=self.fixture["config"])
        self.assertEqual(cmd[cmd.index("--repeat") + 1], "1")
        self.assertEqual(cmd[cmd.index("--workers") + 1], "2")
        self.assertIn(str(self.fixture["step_train"]), cmd)

    def test_rebuild_refuses_arbitrary_override(self):
        plan = self._plan(repeat_override=1)
        with self.assertRaises(OrchestratorError):
            remove_step02_cache(
                self.fixture["step_train"],
                canonical_cache_root=plan.paths.step02_cache_root,
            )

    def test_cli_dry_run_writes_nothing(self):
        workspace = REPO / "TRB/result/feature_framework/experiments/phase7_synth_dry"
        if workspace.exists():
            shutil.rmtree(workspace)
        script = REPO / "TRB/set/scripts/run_trb_experiment.py"
        result = subprocess.run([
            sys.executable, str(script),
            "--config", str(self.fixture["config"]),
            "--train-step02-dir", str(self.fixture["step_train"]),
            "--test-step02-dir", str(self.fixture["step_test"]),
            "--train-metadata", str(self.fixture["train_meta"]),
            "--test-metadata", str(self.fixture["test_meta"]),
            "--workspace-root", str(workspace),
            "--repeat", "1",
            "--dry-run",
        ], cwd=REPO, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("DRY-RUN PASS", result.stdout)
        self.assertFalse(workspace.exists())

    def test_end_to_end_one_repeat_real_phase6(self):
        workspace = REPO / "TRB/result/feature_framework/experiments/phase7_synth"
        if workspace.exists():
            shutil.rmtree(workspace)
        script = REPO / "TRB/set/scripts/run_trb_experiment.py"
        result = subprocess.run([
            sys.executable, str(script),
            "--config", str(self.fixture["config"]),
            "--train-step02-dir", str(self.fixture["step_train"]),
            "--test-step02-dir", str(self.fixture["step_test"]),
            "--train-metadata", str(self.fixture["train_meta"]),
            "--test-metadata", str(self.fixture["test_meta"]),
            "--workspace-root", str(workspace),
            "--repeat", "1",
            "--workers", "1",
        ], cwd=REPO, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("PHASE 7 RUN PASS", result.stdout)
        ml = workspace / "06_unified_ml/smoke_repeat1"
        metrics = pd.read_csv(ml / "06_outer_metrics.csv")
        predictions = pd.read_csv(ml / "06_outer_predictions.csv.gz")
        tuning = pd.read_csv(ml / "06_inner_tuning.csv.gz")
        self.assertEqual(metrics.shape[0], 1)
        self.assertEqual(predictions.shape[0], 4)
        self.assertEqual(tuning.shape[0], 1)
        manifest = json.loads((workspace / "07_orchestrator_manifest.json").read_text())
        self.assertEqual(manifest["actions"]["repertoire"], "cache_hit")
        self.assertEqual(manifest["actions"]["step02_static_core"], "cache_hit")
        self.assertEqual(manifest["actions"]["unified_ml"], "completed")
        self.assertFalse(manifest["modeling_leakage_policy"]["precomputed_step02_kmer_used_for_ml"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
