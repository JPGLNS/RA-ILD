#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render four standard visualizations from aggregated TRB Linear SVM results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.linear_svm_visualization import create_visualizations  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot aggregated TRB Linear SVM repeated-holdout results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = create_visualizations(Path(args.config), overwrite=args.overwrite)
        print("TRB Linear SVM visualization: COMPLETE")
        for key in ("figure1", "figure2", "figure3", "figure4"):
            print(f"{key}: {paths[key]}")
        return 0
    except Exception as exc:
        print(f"TRB Linear SVM visualization: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
