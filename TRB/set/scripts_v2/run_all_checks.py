#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the complete TRB V2 validation stack through one command."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb import __version__  # noqa: E402
from ra_ild_trb.config import load_experiment_config  # noqa: E402
from ra_ild_trb.paths import find_repository_root  # noqa: E402
from ra_ild_trb.release import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_MARKER,
    CheckResult,
    build_check_plan,
    build_completion_payload,
    write_completion_marker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all TRB V2 configuration, unit, regression, and orchestration "
            "checks with the current Python interpreter."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--repository-root", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Use the inexpensive nested-CV regression (default).",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Rerun all 480 inner fits for the frozen nested-CV regression task.",
    )
    parser.add_argument(
        "--skip-outer-orchestration",
        action="store_true",
        help=(
            "Skip the task-tree check while an optional background outer task is "
            "actively writing. This option cannot create a completion marker."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed command instead of collecting all failures.",
    )
    parser.add_argument(
        "--write-completion-marker",
        action="store_true",
        help="Write FRAMEWORK_V2_COMPLETE.json only when every planned check passes.",
    )
    parser.add_argument(
        "--marker-path",
        default=DEFAULT_MARKER,
        help="Repository-relative or absolute completion-marker path.",
    )
    args = parser.parse_args()
    if args.write_completion_marker and args.skip_outer_orchestration:
        parser.error(
            "--write-completion-marker cannot be combined with "
            "--skip-outer-orchestration"
        )
    return args


def _tail(text: str, lines: int = 20) -> str:
    values = text.rstrip().splitlines()
    return "\n".join(values[-lines:])


def _parse_unit_test_count(result: CheckResult) -> Optional[int]:
    if result.name != "unit_tests":
        return None
    combined = result.stdout_tail + "\n" + result.stderr_tail
    match = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    return int(match.group(1)) if match else None


def run_check(command: Sequence[str], repository_root: Path) -> tuple[int, float, str, str]:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=repository_root,
        capture_output=True,
        text=True,
    )
    duration = time.perf_counter() - started
    return completed.returncode, duration, completed.stdout, completed.stderr


def main() -> int:
    args = parse_args()
    mode = "full" if args.full else "quick"
    root = (
        Path(args.repository_root).expanduser().resolve()
        if args.repository_root
        else find_repository_root(SCRIPT_DIR)
    )
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()

    plan = build_check_plan(
        root,
        config_path,
        mode=mode,
        python_executable=sys.executable,
        include_outer_orchestration=not args.skip_outer_orchestration,
    )
    print(f"TRB V2 unified validation: mode={mode}")
    print(f"Framework version: {__version__}")
    print(f"Python executable: {sys.executable}")
    print(f"Repository root: {root}")
    print(f"Configuration: {config_path}")
    print(f"Checks planned: {len(plan)}")

    results: List[CheckResult] = []
    for index, spec in enumerate(plan, start=1):
        print("\n" + "=" * 78)
        print(f"[{index}/{len(plan)}] {spec.name}")
        print("$ " + shlex.join(spec.command))
        returncode, duration, stdout, stderr = run_check(spec.command, root)
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
        result = CheckResult(
            name=spec.name,
            category=spec.category,
            command=tuple(spec.command),
            returncode=int(returncode),
            duration_seconds=float(duration),
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
        )
        results.append(result)
        print(f"[{'PASS' if result.passed else 'FAIL'}] {spec.name} ({duration:.2f} s)")
        if not result.passed and args.fail_fast:
            break

    passed = sum(result.passed for result in results)
    print("\n" + "=" * 78)
    print(
        f"Unified validation summary: mode={mode} "
        f"checks={len(results)} PASS={passed} FAIL={len(results) - passed}"
    )
    for result in results:
        print(
            f"[{'PASS' if result.passed else 'FAIL'}] "
            f"{result.name}: {result.duration_seconds:.2f} s"
        )

    all_passed = len(results) == len(plan) and passed == len(plan)
    if args.write_completion_marker:
        if not all_passed:
            print(
                "Completion marker was not written because one or more checks failed.",
                file=sys.stderr,
            )
        else:
            config = load_experiment_config(config_path, repository_root=root)
            final_model = config.section("final_model")
            unit_test_count = next(
                (
                    count
                    for result in results
                    if (count := _parse_unit_test_count(result)) is not None
                ),
                None,
            )
            payload = build_completion_payload(
                repository_root=root,
                config_path=config_path,
                mode=mode,
                results=results,
                framework_version=__version__,
                unit_test_count=unit_test_count,
                locked_model={
                    "model": final_model["selected_model"],
                    "alpha": float(final_model["selected_alpha"]),
                    "lambda": float(final_model["selected_lambda"]),
                    "threshold": float(final_model["locked_threshold"]),
                },
            )
            marker_path = Path(args.marker_path).expanduser()
            if not marker_path.is_absolute():
                marker_path = root / marker_path
            written = write_completion_marker(marker_path, payload)
            print(f"Completion marker: {written}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
