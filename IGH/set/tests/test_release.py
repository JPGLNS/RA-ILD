#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.release import (  # noqa: E402
    FRAMEWORK_VERSION,
    CheckResult,
    build_check_plan,
    build_completion_payload,
    write_completion_marker,
)


class TestRelease(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("/tmp/example_repo").resolve()
        self.config = self.root / "IGH/set/configs/example.yaml"

    def test_framework_version_is_release(self) -> None:
        self.assertEqual(FRAMEWORK_VERSION, "1.0.0")

    def test_quick_plan_has_expected_order(self) -> None:
        plan = build_check_plan(
            self.root,
            self.config,
            mode="quick",
            python_executable="/env/bin/python3",
        )
        self.assertEqual(
            [item.name for item in plan],
            [
                "configuration",
                "configured_paths",
                "unit_tests",
                "v1_baseline",
                "public_reference",
                "preprocessing",
                "modeling",
                "specifications",
                "nested_cv_quick",
                "outer_orchestration",
            ],
        )
        self.assertTrue(all(item.command[0] == "/env/bin/python3" for item in plan))
        self.assertFalse(plan[8].expensive)

    def test_full_plan_marks_nested_check_expensive(self) -> None:
        plan = build_check_plan(self.root, self.config, mode="full")
        nested = next(item for item in plan if item.name == "nested_cv_full")
        self.assertTrue(nested.expensive)
        self.assertIn("--full", nested.command)

    def test_outer_check_can_be_skipped_for_active_background_task(self) -> None:
        plan = build_check_plan(
            self.root,
            self.config,
            include_outer_orchestration=False,
        )
        self.assertNotIn("outer_orchestration", [item.name for item in plan])
        self.assertEqual(len(plan), 9)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_check_plan(self.root, self.config, mode="slow")

    def test_completion_payload_records_policy_and_locked_model(self) -> None:
        result = CheckResult(
            name="unit_tests",
            category="unit",
            command=("python3", "-m", "unittest"),
            returncode=0,
            duration_seconds=1.0,
            stdout_tail="Ran 118 tests\nOK",
            stderr_tail="",
        )
        payload = build_completion_payload(
            repository_root=self.root,
            config_path=self.config,
            mode="quick",
            results=[result],
            unit_test_count=118,
            locked_model={
                "model": "M2_static_igh_public",
                "alpha": 0.9,
                "lambda": 10.0,
                "threshold": 0.4568009623694209,
            },
        )
        self.assertEqual(payload["status"], "complete")
        self.assertTrue(payload["all_checks_passed"])
        self.assertEqual(payload["unit_test_count"], 118)
        self.assertEqual(payload["locked_model"]["model"], "M2_static_igh_public")
        self.assertFalse(
            payload["independent_test_policy"][
                "independent_test_refit_or_reprediction_in_v2"
            ]
        )
        self.assertFalse(
            payload["outer_task_status"][
                "full_v2_recomputation_required_for_release"
            ]
        )

    def test_failed_payload_cannot_be_written_as_complete(self) -> None:
        result = CheckResult(
            name="configuration",
            category="configuration",
            command=("python3", "check.py"),
            returncode=1,
            duration_seconds=0.1,
            stdout_tail="",
            stderr_tail="failed",
        )
        payload = build_completion_payload(
            repository_root=self.root,
            config_path=self.config,
            mode="quick",
            results=[result],
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_completion_marker(Path(tmp) / "marker.json", payload)

    def test_completion_marker_is_written_as_json(self) -> None:
        payload = {
            "status": "complete",
            "all_checks_passed": True,
            "framework_version": "1.0.0",
        }
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "FRAMEWORK_V2_COMPLETE.json"
            written = write_completion_marker(destination, payload)
            self.assertEqual(written, destination.resolve())
            self.assertEqual(json.loads(destination.read_text()), payload)
            self.assertFalse(destination.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
