#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import importlib.util
import json
import math
import tempfile
import unittest
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
SCRIPT = HERE.parent.parent / "scripts" / "build_01b_AA_clone_table_topk.py"
spec = importlib.util.spec_from_file_location("build_01b_AA_clone_table_topk", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)


def write_full_table(path: Path, rows):
    total_reads = sum(r["read_count"] for r in rows)
    total_freq = sum(r["frequency"] for r in rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=mod.REQUIRED_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "sample_id": r["sample_id"],
                    "cdr3_aa": r["cdr3_aa"],
                    "aa_length": len(r["cdr3_aa"]),
                    "nt_clone_number": r.get("nt_clone_number", 1),
                    "read_count": r["read_count"],
                    "read_fraction": format(r["read_count"] / total_reads, ".17g"),
                    "input_frequency_sum": r["frequency"],
                    "input_cell_frequency_sum": r["frequency"],
                    "frequency": r["frequency"],
                    "frequency_norm": format(r["frequency"] / total_freq, ".17g"),
                    "top_nt_cdr3": r.get("top_nt_cdr3", "TGC"),
                }
            )


class TestTopK(unittest.TestCase):
    def make_rows(self, sample_id="S1"):
        return [
            {"sample_id": sample_id, "cdr3_aa": "CAA", "read_count": 100, "frequency": 10.0},
            {"sample_id": sample_id, "cdr3_aa": "CAB", "read_count": 90, "frequency": 9.0},
            {"sample_id": sample_id, "cdr3_aa": "CAC", "read_count": 80, "frequency": 8.0},
            {"sample_id": sample_id, "cdr3_aa": "CAD", "read_count": 70, "frequency": 7.0},
            {"sample_id": sample_id, "cdr3_aa": "CAE", "read_count": 60, "frequency": 6.0},
            {"sample_id": sample_id, "cdr3_aa": "CAF", "read_count": 50, "frequency": 5.0},
        ]

    def test_exact_k_and_renormalization(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "S1_AA_clone_table.csv"
            write_full_table(src, self.make_rows())
            plan = mod.build_sample_plan(src, root / "out", 4, 1e-6)
            self.assertEqual([r.cdr3_aa for r in plan.selected], ["CAA", "CAB", "CAC", "CAD"])
            self.assertEqual(plan.summary["retained_aa_clone_number"], 4)
            self.assertTrue(plan.qc["qc_pass"])
            mod.write_sample(plan, overwrite=False)
            with plan.output_path.open("r", encoding="utf-8", newline="") as handle:
                out = list(csv.DictReader(handle))
            self.assertEqual(len(out), 4)
            self.assertEqual([int(r["topk_rank"]) for r in out], [1, 2, 3, 4])
            self.assertAlmostEqual(sum(float(r["frequency_norm"]) for r in out), 1.0, places=12)
            self.assertAlmostEqual(sum(float(r["read_fraction"]) for r in out), 1.0, places=12)
            self.assertEqual(out[0]["frequency"], "10.0")
            self.assertEqual(out[0]["read_count"], "100")
            self.assertNotEqual(out[0]["frequency_norm_full"], out[0]["frequency_norm"])

    def test_tie_breaking_is_deterministic(self):
        rows = [
            {"sample_id": "S1", "cdr3_aa": "CCC", "read_count": 10, "frequency": 1.0},
            {"sample_id": "S1", "cdr3_aa": "CAA", "read_count": 20, "frequency": 1.0},
            {"sample_id": "S1", "cdr3_aa": "CAB", "read_count": 20, "frequency": 1.0},
            {"sample_id": "S1", "cdr3_aa": "CZZ", "read_count": 100, "frequency": 0.5},
        ]
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "S1_AA_clone_table.csv"
            write_full_table(src, rows)
            plan = mod.build_sample_plan(src, Path(td) / "out", 3, 1e-6)
            self.assertEqual([r.cdr3_aa for r in plan.selected], ["CAA", "CAB", "CCC"])
            self.assertEqual(plan.summary["boundary_same_frequency_clone_number"], 3)

    def test_sub_k_is_error(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "S1_AA_clone_table.csv"
            write_full_table(src, self.make_rows())
            with self.assertRaises(mod.TopKError):
                mod.build_sample_plan(src, Path(td) / "out", 7, 1e-6)

    def test_duplicate_cdr3_is_error(self):
        rows = self.make_rows()
        rows[-1]["cdr3_aa"] = rows[0]["cdr3_aa"]
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "S1_AA_clone_table.csv"
            write_full_table(src, rows)
            with self.assertRaises(mod.TopKError):
                mod.build_sample_plan(src, Path(td) / "out", 4, 1e-6)

    def test_invalid_full_normalization_is_error(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "S1_AA_clone_table.csv"
            write_full_table(src, self.make_rows())
            records = []
            with src.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                records = list(reader)
            records[0]["frequency_norm"] = "0.9"
            with src.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=mod.REQUIRED_COLUMNS)
                writer.writeheader()
                writer.writerows(records)
            with self.assertRaises(mod.TopKError):
                mod.build_sample_plan(src, Path(td) / "out", 4, 1e-6)

    def test_realistic_19000_to_exact_10000(self):
        rows = []
        for i in range(19000):
            rows.append(
                {
                    "sample_id": "S19K",
                    "cdr3_aa": f"C{i:05d}F",
                    "read_count": 20000 - i,
                    "frequency": float(19000 - i),
                }
            )
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "S19K_AA_clone_table.csv"
            write_full_table(src, rows)
            plan = mod.build_sample_plan(src, Path(td) / "out", 10000, 1e-6)
            self.assertEqual(len(plan.selected), 10000)
            self.assertEqual(plan.selected[0].cdr3_aa, "C00000F")
            self.assertEqual(plan.selected[-1].cdr3_aa, "C09999F")
            self.assertEqual(plan.summary["full_aa_clone_number"], 19000)
            self.assertEqual(plan.summary["retained_aa_clone_number"], 10000)
            self.assertTrue(plan.qc["qc_pass"])

    def test_integration_writes_summary_qc_and_config(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "input"
            out = root / "output"
            inp.mkdir()
            write_full_table(inp / "S1_AA_clone_table.csv", self.make_rows("S1"))
            write_full_table(inp / "S2_AA_clone_table.csv", self.make_rows("S2"))
            args = argparse.Namespace(
                input_dir=str(inp),
                output_dir=str(out),
                top_k=4,
                input_glob="*_AA_clone_table.csv",
                normalization_tol=1e-6,
                dry_run=False,
                overwrite=False,
            )
            plans = mod.run_pipeline(args)
            self.assertEqual(len(plans), 2)
            summary = out / "01b_top4_summary.csv"
            qc = out / "01b_top4_qc.csv"
            config = out / "01b_top4_configuration.json"
            self.assertTrue(summary.exists())
            self.assertTrue(qc.exists())
            self.assertTrue(config.exists())
            with config.open("r", encoding="utf-8") as handle:
                cfg = json.load(handle)
            self.assertEqual(cfg["top_k"], 4)
            self.assertTrue(cfg["all_samples_qc_pass"])

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "input"
            out = root / "output"
            inp.mkdir()
            write_full_table(inp / "S1_AA_clone_table.csv", self.make_rows("S1"))
            args = argparse.Namespace(
                input_dir=str(inp),
                output_dir=str(out),
                top_k=4,
                input_glob="*_AA_clone_table.csv",
                normalization_tol=1e-6,
                dry_run=True,
                overwrite=False,
            )
            mod.run_pipeline(args)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
