#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create fixed repeated nested cross-validation splits for RA/RA-ILD IGH.

Only metadata columns from the fixed IGH TRAIN base matrix are used. The
independent test set and IGH feature values are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.1.0-IGH"
RECEPTOR = "IGH"
ROOT = Path("/data/users/chenhaisheng/RA-ILD/IGH")
DEFAULT_INPUT = ROOT / "set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv"
DEFAULT_OUTPUT = ROOT / "set/train/result/05_modeling/cv_splits"
BALANCE_WEIGHTS = {"batch": 1.0, "material": 0.75, "sex": 0.5}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate fixed repeated outer/inner CV assignments for IGH.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--label-col", default="cohort")
    p.add_argument("--patient-col", default="patient")
    p.add_argument("--balance-cols", default="batch,material,sex")
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--outer-repeats", type=int, default=20)
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--outer-candidates", type=int, default=300)
    p.add_argument("--inner-candidates", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260711)
    p.add_argument("--expected-samples", type=int, default=120)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    a.balance_cols = [x.strip() for x in a.balance_cols.split(",") if x.strip()]
    if a.outer_folds < 2 or a.inner_folds < 2:
        p.error("Fold counts must be >= 2.")
    if a.outer_repeats < 1:
        p.error("--outer-repeats must be >= 1.")
    if a.outer_candidates < 1 or a.inner_candidates < 1:
        p.error("Candidate counts must be >= 1.")
    return a


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def derived_seed(base: int, *parts: int) -> int:
    text = ":".join(str(x) for x in (base, *parts)).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32 - 1)


def paths(out: Path) -> Dict[str, Path]:
    return {
        "outer_assignments": out / "05_outer_fold_assignments.csv",
        "outer_tasks": out / "05_outer_tasks_long.csv.gz",
        "inner_assignments": out / "05_inner_fold_assignments.csv.gz",
        "outer_balance": out / "05_outer_fold_balance.csv",
        "inner_balance": out / "05_inner_fold_balance.csv",
        "configuration": out / "05_cv_split_configuration.json",
        "summary": out / "05_cv_split_summary.md",
    }


def check_overwrite(output_paths: Iterable[Path], overwrite: bool) -> None:
    existing = [p for p in output_paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Outputs already exist; add --overwrite:\n" +
            "\n".join(f"  - {p}" for p in existing)
        )


def load_metadata(a: argparse.Namespace, input_path: Path) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    df = pd.read_csv(input_path, dtype=str)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")], errors="ignore")
    required = [a.sample_id_col, a.patient_col, a.label_col, *a.balance_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df[required].copy()
    for c in required:
        out[c] = out[c].astype(str).str.strip()
        if out[c].eq("").any() or out[c].isna().any():
            raise ValueError(f"Empty values in {c}")
    if out[a.sample_id_col].duplicated().any():
        raise ValueError("Duplicated sample_id values detected.")
    if out[a.patient_col].duplicated().any():
        d = out.loc[out[a.patient_col].duplicated(keep=False), a.patient_col].unique()[:10]
        raise ValueError(
            "Multiple rows per patient detected; grouped CV is required. "
            f"Examples: {d.tolist()}"
        )
    out[a.label_col] = out[a.label_col].str.upper()
    if set(out[a.label_col]) != {"RA", "ILD"}:
        raise ValueError(f"Expected labels RA/ILD; got {sorted(set(out[a.label_col]))}")
    if a.expected_samples > 0 and len(out) != a.expected_samples:
        raise ValueError(f"Expected {a.expected_samples} samples; got {len(out)}")
    out = out.reset_index(drop=True)
    out.insert(0, "row_index", np.arange(len(out), dtype=int))
    return out


def stratified_candidate(labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Assign each cohort as evenly as possible while also balancing total fold size."""
    assign = np.full(len(labels), -1, dtype=np.int16)
    current_sizes = np.zeros(k, dtype=int)

    # Process larger cohorts first; ties are deterministic by label name.
    label_order = sorted(
        np.unique(labels),
        key=lambda x: (-int(np.sum(labels == x)), str(x)),
    )
    for label in label_order:
        idx = rng.permutation(np.flatnonzero(labels == label))
        base, remainder = divmod(len(idx), k)
        counts = np.full(k, base, dtype=int)

        # Give remainder samples to currently smallest folds, randomly breaking ties.
        if remainder:
            jitter = rng.random(k) * 1e-6
            chosen = np.argsort(current_sizes + jitter)[:remainder]
            counts[chosen] += 1

        fold_ids = np.concatenate([
            np.full(counts[f], f + 1, dtype=np.int16)
            for f in range(k)
        ])
        fold_ids = rng.permutation(fold_ids)
        assign[idx] = fold_ids
        current_sizes += counts

    if np.any(assign < 1):
        raise RuntimeError("Incomplete fold assignment")
    if current_sizes.max() - current_sizes.min() > 1:
        raise RuntimeError("Unable to balance total fold sizes within one sample")
    return assign


def category_score(values: np.ndarray, folds: np.ndarray, k: int) -> float:
    score = 0.0
    for category in sorted(np.unique(values)):
        mask = values == category
        expected = mask.sum() / k
        observed = np.array([(mask & (folds == f)).sum() for f in range(1, k + 1)], float)
        score += float(np.sum(((observed - expected) / max(expected, 1.0)) ** 2))
    return score


def split_score(df: pd.DataFrame, folds: np.ndarray, a: argparse.Namespace, k: int) -> float:
    expected_size = len(df) / k
    sizes = np.array([(folds == f).sum() for f in range(1, k + 1)], float)
    score = 0.25 * float(np.sum(((sizes - expected_size) / max(expected_size, 1.0)) ** 2))
    for col in a.balance_cols:
        weight = BALANCE_WEIGHTS.get(col, 0.5)
        vals = df[col].astype(str).to_numpy()
        score += weight * category_score(vals, folds, k)
        joint = (df[a.label_col].astype(str) + "|" + df[col].astype(str)).to_numpy()
        score += 1.5 * weight * category_score(joint, folds, k)
    return score


def signature(ids: Sequence[str], folds: np.ndarray) -> str:
    text = "\n".join(f"{s}\t{int(f)}" for s, f in sorted(zip(ids, folds)))
    return hashlib.sha256(text.encode()).hexdigest()


def optimize_split(
    df: pd.DataFrame,
    k: int,
    candidates: int,
    base_seed: int,
    a: argparse.Namespace,
    forbidden: Optional[set[str]] = None,
) -> Tuple[np.ndarray, float, int, str]:
    labels = df[a.label_col].to_numpy()
    ids = df[a.sample_id_col].tolist()
    best = None
    fallback = None
    for candidate in range(1, candidates + 1):
        seed = derived_seed(base_seed, candidate)
        folds = stratified_candidate(labels, k, np.random.default_rng(seed))
        score = split_score(df, folds, a, k)
        sig = signature(ids, folds)
        rec = (score, seed, folds.copy(), sig)
        if fallback is None or score < fallback[0]:
            fallback = rec
        if forbidden and sig in forbidden:
            continue
        if best is None or score < best[0]:
            best = rec
    chosen = best or fallback
    if chosen is None:
        raise RuntimeError("No candidate split generated")
    score, seed, folds, sig = chosen
    return folds, float(score), int(seed), sig


def balance_table(
    df: pd.DataFrame,
    folds: np.ndarray,
    k: int,
    variables: Sequence[str],
    level: str,
    repeat: int,
    outer_fold: Optional[int] = None,
) -> pd.DataFrame:
    rows: List[dict] = []
    for fold in range(1, k + 1):
        sub = df.loc[folds == fold]
        base = {
            "level": level,
            "outer_repeat": repeat,
            "outer_fold": "" if outer_fold is None else outer_fold,
            "fold": fold,
            "fold_size": len(sub),
        }
        rows.append({**base, "variable": "__fold__", "category": "all", "count": len(sub), "proportion_within_fold": 1.0})
        for variable in variables:
            all_categories = sorted(df[variable].astype(str).unique())
            vc = sub[variable].astype(str).value_counts()
            for cat in all_categories:
                count = int(vc.get(cat, 0))
                rows.append({
                    **base,
                    "variable": variable,
                    "category": cat,
                    "count": count,
                    "proportion_within_fold": count / len(sub) if len(sub) else np.nan,
                })
    return pd.DataFrame(rows)


def validate_outer(df: pd.DataFrame, out: pd.DataFrame, a: argparse.Namespace) -> List[str]:
    expected = len(df) * a.outer_repeats
    if len(out) != expected:
        raise RuntimeError(f"Outer rows expected {expected}; got {len(out)}")
    signatures = []
    for r in range(1, a.outer_repeats + 1):
        x = out[out.outer_repeat == r]
        if len(x) != len(df) or x[a.sample_id_col].duplicated().any():
            raise RuntimeError(f"Invalid outer repeat {r}")
        if set(x.outer_fold) != set(range(1, a.outer_folds + 1)):
            raise RuntimeError(f"Missing fold in outer repeat {r}")
        signatures.append(signature(x[a.sample_id_col], x.outer_fold.to_numpy()))
        for label in ["RA", "ILD"]:
            counts = x.loc[x[a.label_col] == label, "outer_fold"].value_counts().reindex(range(1, a.outer_folds + 1), fill_value=0)
            if counts.max() - counts.min() > 1:
                raise RuntimeError(f"Outer cohort imbalance >1: repeat={r}, cohort={label}")
    if len(set(signatures)) != a.outer_repeats:
        raise RuntimeError("Duplicate outer partitions detected")
    return [
        "Every outer repeat contains each training sample exactly once.",
        "All outer repeat partitions are unique.",
        "Within every outer repeat, RA and ILD counts differ by at most one across folds.",
    ]


def validate_inner(df: pd.DataFrame, outer: pd.DataFrame, inner: pd.DataFrame, a: argparse.Namespace) -> List[str]:
    expected = len(df) * a.outer_repeats * (a.outer_folds - 1)
    if len(inner) != expected:
        raise RuntimeError(f"Inner rows expected {expected}; got {len(inner)}")
    for r in range(1, a.outer_repeats + 1):
        out_r = outer[outer.outer_repeat == r]
        for f in range(1, a.outer_folds + 1):
            train_ids = set(out_r.loc[out_r.outer_fold != f, a.sample_id_col])
            valid_ids = set(out_r.loc[out_r.outer_fold == f, a.sample_id_col])
            x = inner[(inner.outer_repeat == r) & (inner.outer_fold == f)]
            if set(x[a.sample_id_col]) != train_ids:
                raise RuntimeError(f"Inner IDs mismatch: repeat={r}, outer_fold={f}")
            if set(x[a.sample_id_col]) & valid_ids:
                raise RuntimeError(f"Outer validation leakage: repeat={r}, outer_fold={f}")
            if x[a.sample_id_col].duplicated().any():
                raise RuntimeError(f"Duplicated inner sample: repeat={r}, outer_fold={f}")
            if set(x.inner_fold) != set(range(1, a.inner_folds + 1)):
                raise RuntimeError(f"Missing inner fold: repeat={r}, outer_fold={f}")
            for label in ["RA", "ILD"]:
                counts = x.loc[x[a.label_col] == label, "inner_fold"].value_counts().reindex(range(1, a.inner_folds + 1), fill_value=0)
                if counts.max() - counts.min() > 1:
                    raise RuntimeError(f"Inner cohort imbalance >1: repeat={r}, outer_fold={f}, cohort={label}")
    return [
        "Every inner task contains exactly its corresponding outer-training samples.",
        "No outer-validation sample appears in the corresponding inner CV.",
        "Within every inner task, RA and ILD counts differ by at most one across folds.",
    ]


def write_summary(
    path: Path,
    input_path: Path,
    df: pd.DataFrame,
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    a: argparse.Namespace,
    outer_scores: Sequence[float],
    inner_scores: Sequence[float],
    checks: Sequence[str],
    runtime: float,
) -> None:
    outer_sizes = outer.groupby(["outer_repeat", "outer_fold"]).size()
    inner_sizes = inner.groupby(["outer_repeat", "outer_fold", "inner_fold"]).size()
    counts = df[a.label_col].value_counts()
    lines = [
        "# 05 Repeated Nested Cross-Validation Split Summary",
        "",
        "## Scope",
        "",
        "- Only the fixed training cohort was used.",
        "- The independent test set was not read.",
        "- No IGH feature values were used to optimize partitions.",
        "- Cohort was hard-stratified; batch, material and sex were secondary balance targets.",
        "",
        "## Configuration",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Receptor: `{RECEPTOR}`",
        f"- Input: `{input_path}`",
        f"- Training samples: **{len(df)}**",
        f"- RA: **{int(counts.get('RA', 0))}**",
        f"- ILD: **{int(counts.get('ILD', 0))}**",
        f"- Outer CV: **{a.outer_folds} folds × {a.outer_repeats} repeats**",
        f"- Outer tasks: **{a.outer_folds * a.outer_repeats}**",
        f"- Inner CV: **{a.inner_folds} folds per outer task**",
        f"- Seed: **{a.seed}**",
        "",
        "## Output dimensions",
        "",
        f"- Outer assignment rows: **{len(outer)}**",
        f"- Inner assignment rows: **{len(inner)}**",
        f"- Outer validation fold sizes: **{outer_sizes.min()}–{outer_sizes.max()}**",
        f"- Inner validation fold sizes: **{inner_sizes.min()}–{inner_sizes.max()}**",
        "",
        "## Optimization scores",
        "",
        f"- Outer median: **{np.median(outer_scores):.6f}**",
        f"- Outer range: **{min(outer_scores):.6f}–{max(outer_scores):.6f}**",
        f"- Inner median: **{np.median(inner_scores):.6f}**",
        f"- Inner range: **{min(inner_scores):.6f}–{max(inner_scores):.6f}**",
        "",
        "## Integrity checks",
        "",
        *[f"- {x}" for x in checks],
        "",
        "## Use",
        "",
        "- In outer task `(outer_repeat=r, outer_fold=f)`, samples with `outer_fold == f` are validation samples; all others are training samples.",
        "- Use the matching rows in `05_inner_fold_assignments.csv.gz` only to tune alpha/lambda within that outer-training subset.",
        "- All model variants must reuse these same partitions.",
        "- Dynamic reference-public sets must be rebuilt inside each relevant training partition.",
        "",
        f"- Runtime: **{runtime:.2f} seconds**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    a = parse_args()
    started = time.time()
    input_path = Path(a.input).expanduser().resolve()
    output_dir = Path(a.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = paths(output_dir)
    check_overwrite(out_paths.values(), a.overwrite)
    df = load_metadata(a, input_path)

    print("=" * 80)
    print("05 IGH Repeated Nested Cross-Validation Split Builder")
    print("=" * 80)
    print(f"Training samples: {len(df)}")
    print(df[a.label_col].value_counts().to_string())
    print(f"Outer: {a.outer_folds} folds × {a.outer_repeats} repeats")
    print(f"Inner: {a.inner_folds} folds per outer task")
    print(f"Secondary balance: {a.balance_cols}")
    print("Independent test set: NOT READ")
    print()

    outer_parts: List[pd.DataFrame] = []
    outer_task_parts: List[pd.DataFrame] = []
    inner_parts: List[pd.DataFrame] = []
    outer_balance_parts: List[pd.DataFrame] = []
    inner_balance_parts: List[pd.DataFrame] = []
    outer_scores: List[float] = []
    inner_scores: List[float] = []
    outer_meta: List[dict] = []
    inner_meta: List[dict] = []
    used_outer: set[str] = set()

    for repeat in range(1, a.outer_repeats + 1):
        folds, score, seed, sig = optimize_split(
            df, a.outer_folds, a.outer_candidates,
            derived_seed(a.seed, 1, repeat), a, used_outer
        )
        used_outer.add(sig)
        outer_scores.append(score)
        outer_meta.append({"outer_repeat": repeat, "score": score, "chosen_seed": seed, "signature": sig})

        assign = df.copy()
        assign.insert(1, "outer_repeat", repeat)
        assign.insert(2, "outer_fold", folds.astype(int))
        outer_parts.append(assign)
        outer_balance_parts.append(balance_table(df, folds, a.outer_folds, [a.label_col, *a.balance_cols], "outer", repeat))

        for task_fold in range(1, a.outer_folds + 1):
            task = assign.copy()
            task["outer_task_fold"] = task_fold
            task["outer_role"] = np.where(task.outer_fold == task_fold, "validation", "training")
            outer_task_parts.append(task)

            train_df = df.loc[folds != task_fold].reset_index(drop=True)
            in_folds, in_score, in_seed, in_sig = optimize_split(
                train_df, a.inner_folds, a.inner_candidates,
                derived_seed(a.seed, 2, repeat, task_fold), a
            )
            inner_scores.append(in_score)
            inner_meta.append({
                "outer_repeat": repeat,
                "outer_fold": task_fold,
                "score": in_score,
                "chosen_seed": in_seed,
                "signature": in_sig,
            })
            inner = train_df.copy()
            inner.insert(1, "outer_repeat", repeat)
            inner.insert(2, "outer_fold", task_fold)
            inner.insert(3, "inner_fold", in_folds.astype(int))
            inner_parts.append(inner)
            inner_balance_parts.append(balance_table(train_df, in_folds, a.inner_folds, [a.label_col, *a.balance_cols], "inner", repeat, task_fold))

        if repeat == 1 or repeat % 5 == 0 or repeat == a.outer_repeats:
            sizes = [(folds == f).sum() for f in range(1, a.outer_folds + 1)]
            print(f"[repeat {repeat:>2}/{a.outer_repeats}] score={score:.6f} | fold sizes={sizes}")

    outer = pd.concat(outer_parts, ignore_index=True)
    tasks = pd.concat(outer_task_parts, ignore_index=True)
    inner = pd.concat(inner_parts, ignore_index=True)
    outer_balance = pd.concat(outer_balance_parts, ignore_index=True)
    inner_balance = pd.concat(inner_balance_parts, ignore_index=True)

    checks = validate_outer(df, outer, a) + validate_inner(df, outer, inner, a)

    meta_cols = [a.sample_id_col, a.patient_col, a.label_col, *a.balance_cols]
    outer = outer[["outer_repeat", "outer_fold", *meta_cols, "row_index"]].sort_values(["outer_repeat", "outer_fold", a.sample_id_col])
    tasks = tasks[["outer_repeat", "outer_task_fold", "outer_role", "outer_fold", *meta_cols, "row_index"]].sort_values(["outer_repeat", "outer_task_fold", "outer_role", a.sample_id_col])
    inner = inner[["outer_repeat", "outer_fold", "inner_fold", *meta_cols, "row_index"]].sort_values(["outer_repeat", "outer_fold", "inner_fold", a.sample_id_col])

    outer.to_csv(out_paths["outer_assignments"], index=False)
    tasks.to_csv(out_paths["outer_tasks"], index=False, compression="gzip")
    inner.to_csv(out_paths["inner_assignments"], index=False, compression="gzip")
    outer_balance.to_csv(out_paths["outer_balance"], index=False)
    inner_balance.to_csv(out_paths["inner_balance"], index=False)

    config = {
        "script_version": SCRIPT_VERSION,
        "receptor": RECEPTOR,
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output_dir": str(output_dir),
        "sample_id_col": a.sample_id_col,
        "patient_col": a.patient_col,
        "label_col": a.label_col,
        "balance_cols": a.balance_cols,
        "balance_weights": BALANCE_WEIGHTS,
        "outer_folds": a.outer_folds,
        "outer_repeats": a.outer_repeats,
        "inner_folds": a.inner_folds,
        "outer_candidates": a.outer_candidates,
        "inner_candidates": a.inner_candidates,
        "seed": a.seed,
        "outer_partitions": outer_meta,
        "inner_partitions": inner_meta,
    }
    out_paths["configuration"].write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    runtime = time.time() - started
    write_summary(out_paths["summary"], input_path, df, outer, inner, a, outer_scores, inner_scores, checks, runtime)

    print()
    print("[Completed]")
    print(f"Outer assignment rows: {len(outer):,}")
    print(f"Outer task rows:       {len(tasks):,}")
    print(f"Inner assignment rows: {len(inner):,}")
    print("[Integrity checks]")
    for item in checks:
        print(f"- PASS: {item}")
    print("[Output files]")
    for name, path in out_paths.items():
        print(f"- {name}: {path}")
    print(f"Runtime: {runtime:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
