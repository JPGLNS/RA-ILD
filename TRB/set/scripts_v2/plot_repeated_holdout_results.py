#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate configurable PDF figures from completed repeated-holdout results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.repeated_holdout_visualization import (
    generate_visualizations,
    load_visualization_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-ready PDF figures from native frozen repeated-holdout "
            "aggregation results without rerunning model training."
        )
    )
    parser.add_argument("--config", required=True, help="Visualization YAML configuration")
    parser.add_argument(
        "--repository-root",
        default=None,
        help="Repository root used to resolve relative paths (default: current directory)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a documented rerun to replace existing visualization outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_visualization_config(
            Path(args.config),
            repository_root=Path(args.repository_root) if args.repository_root else None,
        )
        result = generate_visualizations(
            config,
            overwrite_override=True if args.overwrite else None,
        )
        print("TRB repeated-holdout visualization: COMPLETE")
        print(f"Visualization ID: {config.visualization_id}")
        print(f"Output directory: {result.output_dir}")
        for key, path in result.generated_files.items():
            print(f"  {key:24s} {path}")
        return 0
    except Exception as exc:
        print(f"TRB repeated-holdout visualization: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
