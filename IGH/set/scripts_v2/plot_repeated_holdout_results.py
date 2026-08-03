#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate configurable PDF figures from completed IGH repeated holdouts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.repeated_holdout_visualization import (  # noqa: E402
    generate_visualizations,
    load_visualization_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create read-only PDF figures from frozen native IGH repeated-holdout "
            "aggregation results without rerunning model training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
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
        print("IGH repeated-holdout visualization: COMPLETE")
        print(f"Visualization ID: {config.visualization_id}")
        print(f"Output directory: {result.output_dir}")
        for key, path in result.generated_files.items():
            print(f"  {key:26s} {path}")
        print("IGH_BATCH10_VISUALIZATION_PASS")
        return 0
    except Exception as exc:
        print(f"IGH repeated-holdout visualization: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
