#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate an IGH split specification in memory or validate its frozen output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.paths import find_repository_root  # noqa: E402
from ra_ild_igh.repeated_holdout import (  # noqa: E402
    generate_repeated_holdout,
    load_combined_metadata,
    load_repeated_holdout_spec,
    validate_frozen_split_set,
    validate_generation_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an IGH repeated-holdout split set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        spec_path = Path(args.spec).expanduser().resolve()
        root = (
            Path(args.repository_root).expanduser().resolve()
            if args.repository_root
            else find_repository_root(spec_path)
        )
        spec = load_repeated_holdout_spec(spec_path, repository_root=root)
        if args.frozen:
            report = validate_frozen_split_set(spec)
            mode = "frozen"
        else:
            metadata = load_combined_metadata(spec)
            result = generate_repeated_holdout(metadata, spec)
            report = validate_generation_result(result, spec)
            mode = "in_memory"

        if args.json_output:
            path = Path(args.json_output).expanduser()
            if not path.is_absolute():
                path = root / path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print("RA-ILD IGH repeated holdout validation: PASS")
        print(f"Mode:              {mode}")
        print(f"Split set:         {report['split_set_id']}")
        print(f"Repeats:           {report['repeats']}")
        print(
            f"Per split:         train={report['train_size']}, "
            f"holdout={report['holdout_size']}"
        )
        print(
            "Fixed train IDs:   "
            + (", ".join(report['fixed_train_sample_ids']) or "none")
        )
        if "assignment_sha256" in report:
            print(f"Assignment SHA256: {report['assignment_sha256']}")
        print("IGH_REPEATED_HOLDOUT_VALIDATION_PASS")
        return 0
    except Exception as exc:
        print(f"RA-ILD IGH repeated holdout validation: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
