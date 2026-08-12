#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SET_ROOT = HERE.parent
SRC_DIR = SET_ROOT / "src"
CONFIG_DIR = SET_ROOT / "configs" / "feature_framework"
LEGACY_SCRIPT = SET_ROOT / "scripts" / "build_02_sample_level_features.py"
SOURCE_AWARE_SCRIPT = SET_ROOT / "scripts" / "build_02_sample_level_features_source_aware.py"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.feature_framework import (  # noqa: E402
    FeatureRegistry,
    load_feature_plan,
    load_feature_registry,
)
from ra_ild_trb.repertoire_input import (  # noqa: E402
    RepertoireInputError,
    adapt_summary_for_legacy_step02,
    load_summary,
    resolve_repertoire_input,
)

REGISTRY_PATH = CONFIG_DIR / "trb_feature_registry_v1.yaml"
ALL_PLAN_PATH = CONFIG_DIR / "trb_feature_plan_all_compat_v1.yaml"


def load_legacy_module():
    spec = importlib.util.spec_from_file_location("test_legacy_step02", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load legacy Step02")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_source_aware_module():
    spec = importlib.util.spec_from_file_location("test_source_aware_step02", SOURCE_AWARE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load source-aware Step02")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_full_summary(sample_id="S1"):
    return pd.Series(
        {
            "sample_id": sample_id,
            "input_nt_rows": 5,
            "valid_nt_rows": 4,
            "valid_nt_row_ratio": 0.8,
            "total_reads_all_nt": 110,
            "total_reads_valid_nt": 100,
            "unique_valid_nt_clones": 4,
            "aa_clone_number": 3,
        }
    )


def write_full_summary(path: Path, sample_id="S1"):
    pd.DataFrame([dict(make_full_summary(sample_id))]).to_csv(path, index=False)


def write_topk_table(path: Path, sample_id="S1"):
    pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "topk_rank": 1,
                "cdr3_aa": "CASSAAA",
                "aa_length": 7,
                "nt_clone_number": 2,
                "read_count": 50,
                "read_fraction_full": 0.5,
                "read_fraction": 0.625,
                "input_frequency_sum": 5.0,
                "input_cell_frequency_sum": 5.0,
                "frequency": 5.0,
                "frequency_norm_full": 0.5,
                "frequency_norm": 0.625,
                "top_nt_cdr3": "TGTGCCAGCAGTGCTGCTGCT",
            },
            {
                "sample_id": sample_id,
                "topk_rank": 2,
                "cdr3_aa": "CASSBBB".replace("B", "G"),
                "aa_length": 7,
                "nt_clone_number": 1,
                "read_count": 30,
                "read_fraction_full": 0.3,
                "read_fraction": 0.375,
                "input_frequency_sum": 3.0,
                "input_cell_frequency_sum": 3.0,
                "frequency": 3.0,
                "frequency_norm_full": 0.3,
                "frequency_norm": 0.375,
                "top_nt_cdr3": "TGTGCCAGCAGTGGCGGCGGC",
            },
        ]
    ).to_csv(path, index=False)


def write_topk_summary(path: Path, sample_id="S1"):
    pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "top_k": 2,
                "full_aa_clone_number": 3,
                "retained_aa_clone_number": 2,
                "retained_read_count": 80,
                "retained_read_mass": 0.8,
                "full_read_count": 100,
                "full_frequency_sum": 10.0,
                "retained_frequency_sum": 8.0,
                "retained_frequency_mass": 0.8,
                "frequency_norm_sum_topk": 1.0,
                "read_fraction_sum_topk": 1.0,
                "qc_pass": True,
                "qc_warning": "boundary_frequency_tie:1",
            }
        ]
    ).to_csv(path, index=False)


def make_top2_registry(registry: FeatureRegistry) -> FeatureRegistry:
    top = registry.repertoire_sources["top10000"]
    top2 = dataclasses.replace(top, top_k=2)
    sources = dict(registry.repertoire_sources)
    sources["top10000"] = top2
    return dataclasses.replace(registry, repertoire_sources=sources)


class TestRepertoireInput(unittest.TestCase):
    def test_extended_registry_source_io_contract(self):
        registry = load_feature_registry(REGISTRY_PATH)
        full = registry.repertoire_sources["all"]
        top = registry.repertoire_sources["top10000"]
        self.assertEqual(full.summary_kind, "step01_full")
        self.assertIsNone(full.parent_source)
        self.assertEqual(
            full.sample_filename("ABC"), "ABC_TRB_CDR3_AA_clone_table.csv"
        )
        self.assertEqual(top.summary_kind, "step01b_topk")
        self.assertEqual(top.parent_source, "all")
        self.assertEqual(
            top.sample_filename("ABC"),
            "ABC_TRB_CDR3_AA_clone_table_top10000.csv",
        )

    def test_all_compat_plan_is_ready_without_code_change(self):
        registry = load_feature_registry(REGISTRY_PATH)
        plan = load_feature_plan(ALL_PLAN_PATH)
        resolved = registry.resolve_plan(plan)
        self.assertEqual(resolved.repertoire_source.id, "all")
        self.assertEqual(len(resolved.feature_groups), 9)

    def test_split_routing_all_uses_train_step01(self):
        source_aware = load_source_aware_module()
        registry = load_feature_registry(REGISTRY_PATH)
        plan = load_feature_plan(ALL_PLAN_PATH)
        resolved_plan = registry.resolve_plan(plan)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "TRB" / "set" / "train"
            metadata = split_dir / "metadata_train.csv"
            layer = split_dir / "result" / "01_AA_clone_table"
            layer.mkdir(parents=True)
            metadata.write_text("libraryid\nS1\n", encoding="utf-8")
            write_full_summary(layer / "01_AA_clone_table_summary.csv")
            args = argparse.Namespace(
                mode="train", aa_dir=None, source_summary_file=None, parent_summary_file=None
            )
            aa, summary, parent, notes = source_aware._split_aware_overrides(
                args, registry, resolved_plan, metadata, root
            )
            self.assertEqual(Path(aa), layer)
            self.assertEqual(Path(summary), layer / "01_AA_clone_table_summary.csv")
            self.assertIsNone(parent)
            self.assertEqual(len(notes), 2)

    def test_split_routing_topk_keeps_global_source_and_uses_train_parent(self):
        source_aware = load_source_aware_module()
        registry = load_feature_registry(REGISTRY_PATH)
        top_plan = load_feature_plan(CONFIG_DIR / "trb_feature_plan_template_v1.yaml")
        resolved_plan = registry.resolve_plan(top_plan)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "TRB" / "set" / "train"
            metadata = split_dir / "metadata_train.csv"
            parent_dir = split_dir / "result" / "01_AA_clone_table"
            parent_dir.mkdir(parents=True)
            metadata.write_text("libraryid\nS1\n", encoding="utf-8")
            parent_summary = parent_dir / "01_AA_clone_table_summary.csv"
            write_full_summary(parent_summary)
            args = argparse.Namespace(
                mode="train", aa_dir=None, source_summary_file=None, parent_summary_file=None
            )
            aa, summary, parent, notes = source_aware._split_aware_overrides(
                args, registry, resolved_plan, metadata, root
            )
            self.assertIsNone(aa)
            self.assertIsNone(summary)
            self.assertEqual(Path(parent), parent_summary)
            self.assertEqual(len(notes), 1)

    def test_split_routing_explicit_overrides_win(self):
        source_aware = load_source_aware_module()
        registry = load_feature_registry(REGISTRY_PATH)
        plan = load_feature_plan(ALL_PLAN_PATH)
        resolved_plan = registry.resolve_plan(plan)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "TRB" / "set" / "train"
            metadata = split_dir / "metadata_train.csv"
            layer = split_dir / "result" / "01_AA_clone_table"
            layer.mkdir(parents=True)
            metadata.write_text("libraryid\nS1\n", encoding="utf-8")
            write_full_summary(layer / "01_AA_clone_table_summary.csv")
            explicit_aa = root / "custom_aa"
            explicit_summary = root / "custom_summary.csv"
            args = argparse.Namespace(
                mode="train",
                aa_dir=str(explicit_aa),
                source_summary_file=str(explicit_summary),
                parent_summary_file=None,
            )
            aa, summary, parent, notes = source_aware._split_aware_overrides(
                args, registry, resolved_plan, metadata, root
            )
            self.assertEqual(aa, str(explicit_aa))
            self.assertEqual(summary, str(explicit_summary))
            self.assertIsNone(parent)
            self.assertEqual(notes, [])

    def test_split_routing_rejects_mode_metadata_mismatch(self):
        source_aware = load_source_aware_module()
        registry = load_feature_registry(REGISTRY_PATH)
        plan = load_feature_plan(ALL_PLAN_PATH)
        resolved_plan = registry.resolve_plan(plan)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_dir = root / "TRB" / "set" / "test"
            test_dir.mkdir(parents=True)
            metadata = test_dir / "metadata_test.csv"
            metadata.write_text("libraryid\nS1\n", encoding="utf-8")
            args = argparse.Namespace(
                mode="train", aa_dir=None, source_summary_file=None, parent_summary_file=None
            )
            with self.assertRaisesRegex(ValueError, "metadata belongs to split"):
                source_aware._split_aware_overrides(
                    args, registry, resolved_plan, metadata, root
                )

    def test_full_summary_loader_and_adapter_preserve_contract(self):
        registry = load_feature_registry(REGISTRY_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aa_dir = root / "aa"
            aa_dir.mkdir()
            summary = root / "full.csv"
            write_full_summary(summary)
            resolved = resolve_repertoire_input(
                registry,
                "all",
                root,
                aa_dir_override=aa_dir,
                summary_override=summary,
                must_exist=True,
            )
            frame = load_summary(summary, "step01_full")
            adapted, audit = adapt_summary_for_legacy_step02(
                resolved, "S1", frame.loc["S1"]
            )
            original = frame.loc["S1"]
            for name in (
                "input_nt_rows",
                "valid_nt_rows",
                "valid_nt_row_ratio",
                "total_reads_all_nt",
                "total_reads_valid_nt",
                "unique_valid_nt_clones",
                "aa_clone_number",
            ):
                self.assertEqual(adapted[name], original[name])
            self.assertEqual(audit["retained_read_mass"], 1.0)
            self.assertEqual(audit["analysis_aa_clone_number"], 3)

    def test_topk_adapter_uses_retained_counts_not_full_equality(self):
        registry = make_top2_registry(load_feature_registry(REGISTRY_PATH))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aa_dir = root / "top"
            aa_dir.mkdir()
            top_file = aa_dir / registry.repertoire_sources["top10000"].sample_filename("S1")
            write_topk_table(top_file)
            top_summary = root / "top_summary.csv"
            parent_summary = root / "full_summary.csv"
            write_topk_summary(top_summary)
            write_full_summary(parent_summary)
            resolved = resolve_repertoire_input(
                registry,
                "top10000",
                root,
                aa_dir_override=aa_dir,
                summary_override=top_summary,
                parent_summary_override=parent_summary,
                must_exist=True,
            )
            source = load_summary(top_summary, "step01b_topk")
            parent = load_summary(parent_summary, "step01_full")
            adapted, audit = adapt_summary_for_legacy_step02(
                resolved,
                "S1",
                source.loc["S1"],
                parent_summary_row=parent.loc["S1"],
            )
            self.assertEqual(int(adapted["aa_clone_number"]), 2)
            self.assertEqual(int(adapted["total_reads_valid_nt"]), 80)
            self.assertEqual(int(adapted["unique_valid_nt_clones"]), 3)
            self.assertAlmostEqual(float(adapted["valid_nt_row_ratio"]), 0.6)
            self.assertEqual(int(adapted["total_reads_all_nt"]), 110)
            self.assertAlmostEqual(float(audit["retained_read_mass"]), 0.8)
            self.assertAlmostEqual(float(audit["retained_nt_clone_fraction"]), 0.75)

    def test_topk_rank_contract_is_strict(self):
        registry = make_top2_registry(load_feature_registry(REGISTRY_PATH))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aa_dir = root / "top"
            aa_dir.mkdir()
            top_file = aa_dir / registry.repertoire_sources["top10000"].sample_filename("S1")
            write_topk_table(top_file)
            frame = pd.read_csv(top_file)
            frame.loc[1, "topk_rank"] = 3
            frame.to_csv(top_file, index=False)
            top_summary = root / "top_summary.csv"
            parent_summary = root / "full_summary.csv"
            write_topk_summary(top_summary)
            write_full_summary(parent_summary)
            resolved = resolve_repertoire_input(
                registry,
                "top10000",
                root,
                aa_dir_override=aa_dir,
                summary_override=top_summary,
                parent_summary_override=parent_summary,
                must_exist=True,
            )
            source = load_summary(top_summary, "step01b_topk")
            parent = load_summary(parent_summary, "step01_full")
            with self.assertRaisesRegex(RepertoireInputError, "topk_rank"):
                adapt_summary_for_legacy_step02(
                    resolved,
                    "S1",
                    source.loc["S1"],
                    parent_summary_row=parent.loc["S1"],
                )

    @unittest.skipUnless(LEGACY_SCRIPT.is_file(), "legacy Step02 exists only after installation")
    def test_all_adapter_is_numerically_identical_to_legacy_kernel(self):
        legacy = load_legacy_module()
        registry = load_feature_registry(REGISTRY_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aa_dir = root / "aa"
            aa_dir.mkdir()
            path = aa_dir / registry.repertoire_sources["all"].sample_filename("S1")
            pd.DataFrame(
                [
                    {"sample_id":"S1","cdr3_aa":"CASSAAA","aa_length":7,"nt_clone_number":2,"read_count":50,"read_fraction":0.5,"input_frequency_sum":5.0,"input_cell_frequency_sum":5.0,"frequency":5.0,"frequency_norm":0.5,"top_nt_cdr3":"AAA"},
                    {"sample_id":"S1","cdr3_aa":"CASSGGG","aa_length":7,"nt_clone_number":1,"read_count":30,"read_fraction":0.3,"input_frequency_sum":3.0,"input_cell_frequency_sum":3.0,"frequency":3.0,"frequency_norm":0.3,"top_nt_cdr3":"GGG"},
                    {"sample_id":"S1","cdr3_aa":"CASSTTT","aa_length":7,"nt_clone_number":1,"read_count":20,"read_fraction":0.2,"input_frequency_sum":2.0,"input_cell_frequency_sum":2.0,"frequency":2.0,"frequency_norm":0.2,"top_nt_cdr3":"TTT"},
                ]
            ).to_csv(path, index=False)
            summary = root / "full.csv"
            write_full_summary(summary)
            resolved = resolve_repertoire_input(
                registry, "all", root, aa_dir_override=aa_dir, summary_override=summary
            )
            frame = load_summary(summary, "step01_full")
            adapted, _ = adapt_summary_for_legacy_step02(resolved, "S1", frame.loc["S1"])
            args = argparse.Namespace(
                invalid_aa_policy="error",
                norm_tolerance=1e-8,
                summary_mismatch_policy="error",
            )
            direct = legacy.process_one_sample(
                "S1", path, frame.loc["S1"], None, True, args
            )
            routed = legacy.process_one_sample(
                "S1", path, adapted, None, True, args
            )
            self.assertEqual(direct[0].keys(), routed[0].keys())
            self.assertEqual(len(direct[0]), 97)  # sample_id + 96 core feature columns
            self.assertEqual(len(direct[0]) - 1, 96)
            for key in direct[0]:
                if key == "sample_id":
                    self.assertEqual(direct[0][key], routed[0][key])
                else:
                    self.assertAlmostEqual(float(direct[0][key]), float(routed[0][key]), places=14)
            self.assertEqual(direct[1], routed[1])
            self.assertEqual(direct[2], routed[2])

    @unittest.skipUnless(LEGACY_SCRIPT.is_file(), "legacy Step02 exists only after installation")
    def test_topk_adapter_removes_false_full_summary_mismatch(self):
        legacy = load_legacy_module()
        registry = make_top2_registry(load_feature_registry(REGISTRY_PATH))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aa_dir = root / "top"
            aa_dir.mkdir()
            path = aa_dir / registry.repertoire_sources["top10000"].sample_filename("S1")
            write_topk_table(path)
            top_summary = root / "top_summary.csv"
            parent_summary = root / "full_summary.csv"
            write_topk_summary(top_summary)
            write_full_summary(parent_summary)
            resolved = resolve_repertoire_input(
                registry,
                "top10000",
                root,
                aa_dir_override=aa_dir,
                summary_override=top_summary,
                parent_summary_override=parent_summary,
            )
            source = load_summary(top_summary, "step01b_topk")
            parent = load_summary(parent_summary, "step01_full")
            adapted, _ = adapt_summary_for_legacy_step02(
                resolved, "S1", source.loc["S1"], parent_summary_row=parent.loc["S1"]
            )
            args = argparse.Namespace(
                invalid_aa_policy="error",
                norm_tolerance=1e-8,
                summary_mismatch_policy="error",
            )
            with self.assertRaisesRegex(ValueError, "summary"):
                legacy.process_one_sample("S1", path, parent.loc["S1"], None, True, args)
            core, _, _, log = legacy.process_one_sample(
                "S1", path, adapted, None, True, args
            )
            self.assertEqual(log.summary_mismatch_count, 0)
            self.assertEqual(int(core["aa_clone_number"]), 2)
            self.assertAlmostEqual(float(core["Shannon"]), -(0.625*__import__('math').log(0.625)+0.375*__import__('math').log(0.375)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
