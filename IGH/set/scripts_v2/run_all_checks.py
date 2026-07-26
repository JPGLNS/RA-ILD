#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the complete IGH V2 Batch 06 validation stack."""
from __future__ import annotations
import argparse, re, shlex, subprocess, sys, time
from pathlib import Path
from typing import List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh import __version__
from ra_ild_igh.paths import find_repository_root
from ra_ild_igh.release import (
    DEFAULT_BASELINE_CONFIG, DEFAULT_CONFIG, DEFAULT_MARKER, CheckResult,
    build_check_plan, build_completion_payload, write_completion_marker,
)

def parse_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--baseline-config", default=DEFAULT_BASELINE_CONFIG)
    p.add_argument("--repository-root", default=None)
    mode=p.add_mutually_exclusive_group(); mode.add_argument("--quick",action="store_true"); mode.add_argument("--full",action="store_true")
    p.add_argument("--allow-dirty-source", action="store_true", help="Acceptance-only mode before Batch 06 source commit")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--write-completion-marker", action="store_true")
    p.add_argument("--marker-path", default=DEFAULT_MARKER)
    args=p.parse_args()
    if args.write_completion_marker and not args.full:
        p.error("--write-completion-marker requires --full")
    if args.write_completion_marker and args.allow_dirty_source:
        p.error("--write-completion-marker cannot use --allow-dirty-source")
    return args

def tail(text: str, lines: int=20) -> str:
    return "\n".join(text.rstrip().splitlines()[-lines:])

def parse_unit_count(result: CheckResult) -> Optional[int]:
    if result.name != "unit_tests": return None
    match=re.search(r"Ran\s+(\d+)\s+tests?", result.stdout_tail+"\n"+result.stderr_tail)
    return int(match.group(1)) if match else None

def run(command: Sequence[str], root: Path):
    started=time.perf_counter(); done=subprocess.run(list(command),cwd=root,text=True,capture_output=True)
    return done.returncode,time.perf_counter()-started,done.stdout,done.stderr

def main():
    args=parse_args(); mode="full" if args.full else "quick"
    root=Path(args.repository_root).expanduser().resolve() if args.repository_root else find_repository_root(SCRIPT_DIR)
    def resolve(value):
        path=Path(value).expanduser(); return (root/path).resolve() if not path.is_absolute() else path.resolve()
    config=resolve(args.config); baseline=resolve(args.baseline_config)
    plan=build_check_plan(root,config,baseline,mode=mode,python_executable=sys.executable,allow_dirty_source=args.allow_dirty_source)
    print(f"IGH V2 unified validation: mode={mode}")
    print(f"Framework version: {__version__}")
    print(f"Python executable: {sys.executable}")
    print(f"Repository root: {root}")
    print(f"Configuration: {config}")
    print(f"Historical baseline: {baseline}")
    print(f"Checks planned: {len(plan)}")
    results: List[CheckResult]=[]
    for i,spec in enumerate(plan,1):
        print("\n"+"="*78); print(f"[{i}/{len(plan)}] {spec.name}"); print("$ "+shlex.join(spec.command))
        rc,duration,out,err=run(spec.command,root)
        if out: print(out,end="" if out.endswith("\n") else "\n")
        if err: print(err,file=sys.stderr,end="" if err.endswith("\n") else "\n")
        result=CheckResult(spec.name,spec.category,tuple(spec.command),int(rc),float(duration),tail(out),tail(err)); results.append(result)
        print(f"[{'PASS' if result.passed else 'FAIL'}] {spec.name} ({duration:.2f} s)")
        if not result.passed and args.fail_fast: break
    passed=sum(x.passed for x in results)
    print("\n"+"="*78)
    print(f"Unified validation summary: mode={mode} checks={len(results)} PASS={passed} FAIL={len(results)-passed}")
    for result in results: print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.duration_seconds:.2f} s")
    all_passed=len(results)==len(plan) and passed==len(plan)
    if args.write_completion_marker:
        if not all_passed: print("Completion marker not written because checks failed.",file=sys.stderr)
        else:
            unit_count=next((n for r in results if (n:=parse_unit_count(r)) is not None),None)
            payload=build_completion_payload(repository_root=root,config_path=config,baseline_config_path=baseline,mode=mode,results=results,unit_test_count=unit_count)
            marker=resolve(args.marker_path); written=write_completion_marker(marker,payload); print(f"Completion marker: {written}")
    return 0 if all_passed else 1

if __name__=="__main__": raise SystemExit(main())
