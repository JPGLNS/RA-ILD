#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate and resolve the additive TRB Feature Framework v1 contracts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SET_ROOT = SCRIPT_DIR.parent
SRC_DIR = SET_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.feature_framework import (  # noqa: E402
    FeatureFrameworkError,
    load_feature_plan,
    load_feature_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate TRB Feature Framework v1 registry/plan contracts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--registry", required=True, help="Feature registry YAML.")
    parser.add_argument("--plan", default=None, help="Optional feature plan YAML to resolve.")
    parser.add_argument(
        "--columns-csv",
        default=None,
        help="Optional CSV whose header will be classified by registry selectors.",
    )
    parser.add_argument(
        "--resolved-output",
        default=None,
        help="Optional JSON path for the resolved plan. No output is written without this option.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate completely but do not write --resolved-output.",
    )
    return parser.parse_args()


def read_header(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"columns CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"columns CSV is empty: {path}") from exc
    if not header:
        raise ValueError(f"columns CSV has an empty header: {path}")
    return [str(item).strip() for item in header]


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry).expanduser().resolve()
    registry = load_feature_registry(registry_path)

    print(f"Registry PASS: {registry.id}")
    print(f"Receptor:      {registry.receptor}")
    print(f"Sources:       {', '.join(registry.repertoire_sources)}")
    print(f"Feature groups:{len(registry.feature_groups):>4}")

    resolved = None
    if args.plan:
        plan_path = Path(args.plan).expanduser().resolve()
        plan = load_feature_plan(plan_path)
        resolved = registry.resolve_plan(plan)
        print("")
        print(f"Plan PASS:     {resolved.plan_id}")
        print(f"Repertoire:    {resolved.repertoire_source.id}")
        print(f"Groups:        {len(resolved.feature_groups)}")
        print(f"Training fit:  {resolved.requires_training_fit}")
        for group in resolved.feature_groups:
            print(
                f"  - {group.id}: family={group.family}; type={group.module_type}; "
                f"fit_scope={group.fit_scope}"
            )
        for warning in resolved.warnings:
            print(f"WARNING: {warning}")

    if args.columns_csv:
        header = read_header(Path(args.columns_csv).expanduser().resolve())
        groups, unmatched = registry.classify_columns(header)
        print("")
        print("Column classification:")
        for group_id, columns in groups.items():
            if columns:
                print(f"  - {group_id}: {len(columns)}")
        print(f"  - unmatched: {len(unmatched)}")
        if unmatched:
            print("    preview: " + ", ".join(unmatched[:20]))

    if args.resolved_output:
        if resolved is None:
            raise FeatureFrameworkError("--resolved-output requires --plan")
        out = Path(args.resolved_output).expanduser().resolve()
        if args.dry_run:
            print(f"DRY-RUN: resolved output not written: {out}")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            temp = out.with_suffix(out.suffix + ".tmp")
            temp.write_text(json.dumps(resolved.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temp.replace(out)
            print(f"Resolved plan written: {out}")
    elif args.dry_run:
        print("DRY-RUN PASS: validation only; no files written.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
