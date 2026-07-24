#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare one isolated RA-ILD TRB analysis scheme from a compact YAML definition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.paths import find_repository_root  # noqa: E402
from ra_ild_trb.scheme_management import (  # noqa: E402
    build_prepared_scheme,
    write_prepared_scheme,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve and prepare an isolated RA-ILD TRB analysis scheme.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scheme", required=True, help="Scheme YAML definition")
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-inputs", action="store_true")
    parser.add_argument(
        "--refresh-config-only",
        action="store_true",
        help="Refresh snapshots only when no scientific result file exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        scheme_path = Path(args.scheme).expanduser().resolve()
        repository_root = (
            Path(args.repository_root).expanduser().resolve()
            if args.repository_root
            else find_repository_root(scheme_path)
        )
        prepared = build_prepared_scheme(scheme_path, repository_root)
        missing = [row for row in prepared.input_manifest if not bool(row["exists"])]

        print("RA-ILD TRB scheme preparation")
        print(f"Scheme ID:       {prepared.scheme_id}")
        print(f"Base config:     {prepared.base_config_path}")
        print(f"Output root:     {prepared.output_root}")
        print(f"Resolved config: {prepared.resolved_config_path}")
        models = list(prepared.resolved_config.get("models", {}))
        additional_tables = [
            row for row in prepared.input_manifest
            if "additional_feature_tables" in str(row.get("config_key", ""))
        ]
        print(f"Models:          {len(models)} ({', '.join(models)})")
        print(f"Input paths:     {len(prepared.input_manifest)}")
        print(f"Additional tables: {len(additional_tables)}")
        print(f"Missing inputs:  {len(missing)}")

        if args.dry_run:
            print("Dry run only; no directory or snapshot was written.")
            if missing:
                print("Missing input details:")
                for row in missing:
                    print(f"  - {row['config_key']}: {row['resolved_path']}")
            return 0 if (not missing or args.allow_missing_inputs) else 1

        metadata = write_prepared_scheme(
            prepared,
            repository_root,
            allow_missing_inputs=args.allow_missing_inputs,
            refresh_config_only=args.refresh_config_only,
        )
        print("Scheme preparation: COMPLETE")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Scheme preparation: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
