#!/usr/bin/env python3
"""IGH V3 Phase 2A: safely enable weighted+unweighted 3-mer with one fold-local vocabulary.

Run from any directory:
    python phase2a_enable_both_3mer.py --repo /data/users/chenhaisheng/RA-ILD --check
    python phase2a_enable_both_3mer.py --repo /data/users/chenhaisheng/RA-ILD --apply

This tool edits ONLY experiment_spec.py and fold_features.py on the target branch.
It does not create experiments, train models, stage, commit or push.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import shutil
import subprocess
from pathlib import Path

ROOT_REL = Path('IGH/set/src/ra_ild_igh')
BRANCH = 'feature/igh-v3-hierarchical-5alg'
CONSTANT_BEFORE = 'KMER_REPRESENTATIONS = {"unweighted", "weighted"}'
CONSTANT_AFTER = 'KMER_REPRESENTATIONS = {"unweighted", "weighted", "both"}'

FOLD_BEFORE = '''        rep = spec.base.features.kmer.representation
        train_kmer = _kmer_feature_frame(train_ids, selected, inputs, rep)
        valid_kmer = _kmer_feature_frame(valid_ids, selected, inputs, rep)
        train = train.merge(train_kmer, on="sample_id", how="left", validate="one_to_one")
        valid = valid.merge(valid_kmer, on="sample_id", how="left", validate="one_to_one")
        kcols = [c for c in train_kmer.columns if c != "sample_id"]
        numeric.extend(kcols)
'''
FOLD_AFTER = '''        rep = spec.base.features.kmer.representation
        # With 'both' the SAME fold-local Top-K vocabulary is represented twice.
        # Validation/holdout samples never contribute to vocabulary selection.
        representations = ("unweighted", "weighted") if rep == "both" else (rep,)
        if any(r not in {"unweighted", "weighted"} for r in representations):
            raise FoldFeatureError(f"unsupported 3-mer representation: {rep!r}")
        kcols = []
        for representation in representations:
            train_kmer = _kmer_feature_frame(train_ids, selected, inputs, representation)
            valid_kmer = _kmer_feature_frame(valid_ids, selected, inputs, representation)
            train = train.merge(train_kmer, on="sample_id", how="left", validate="one_to_one")
            valid = valid.merge(valid_kmer, on="sample_id", how="left", validate="one_to_one")
            kcols.extend(c for c in train_kmer.columns if c != "sample_id")
        if len(kcols) != len(set(kcols)):
            raise FoldFeatureError("duplicate weighted/unweighted 3-mer feature names")
        numeric.extend(kcols)
'''


def sh(*args: str) -> str:
    p = subprocess.run(args, capture_output=True, text=True, check=True)
    return p.stdout.strip()


def plan(old: str, before: str, after: str, label: str) -> str:
    count = old.count(before)
    if count == 1:
        return old.replace(before, after, 1)
    if count == 0 and old.count(after) == 1:
        print(f'[ALREADY PATCHED] {label}')
        return old
    raise RuntimeError(f'{label}: expected exactly one unmodified target (got {count}); stop to avoid corrupting source')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    repo = args.repo.resolve()
    if not (repo / '.git').exists():
        raise SystemExit(f'Not a Git working tree: {repo}')
    branch = sh('git', '-C', str(repo), 'branch', '--show-current')
    if branch != BRANCH:
        raise SystemExit(f'Wrong branch {branch!r}, expected {BRANCH!r}')
    paths = [repo / ROOT_REL / 'experiment_spec.py', repo / ROOT_REL / 'fold_features.py']
    old_text = [p.read_text(encoding='utf-8') for p in paths]
    new_text = [
        plan(old_text[0], CONSTANT_BEFORE, CONSTANT_AFTER, paths[0].name),
        plan(old_text[1], FOLD_BEFORE, FOLD_AFTER, paths[1].name),
    ]
    for p, old, new in zip(paths, old_text, new_text):
        print(f'[TARGET] {p.relative_to(repo)} change={old != new}; old_sha256={hashlib.sha256(old.encode()).hexdigest()[:12]}')
        compile(new, str(p), 'exec')
    print('[SYNTAX PASS] both edited source files compile')
    if args.check:
        print('[CHECK ONLY] No files changed; use --apply when ready')
        return
    changes = [(p, old, new) for p, old, new in zip(paths, old_text, new_text) if old != new]
    if not changes:
        print('[NOOP] All edits already present')
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup_root = Path('/tmp') / f'igh_v3_phase2a_backup_{stamp}'
    backup_root.mkdir(parents=True, exist_ok=False)
    for p, old, _ in changes:
        dst = backup_root / p.name
        dst.write_text(old, encoding='utf-8')
    for p, _, new in changes:
        p.write_text(new, encoding='utf-8')
    print(f'[PATCH PASS] Modified {len(changes)} files; previous copies at {backup_root}')
    print('[NEXT] Run: python IGH/set/tests/test_igh_v3_both_3mer.py')
    print('[NEXT] Review: git diff -- IGH/set/src/ra_ild_igh/experiment_spec.py IGH/set/src/ra_ild_igh/fold_features.py')


if __name__ == '__main__':
    main()
