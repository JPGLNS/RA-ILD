#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit effective IGH feature inputs and model definitions without fitting models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.config import load_experiment_config  # noqa: E402
from ra_ild_igh.feature_inputs import load_partition_feature_matrix  # noqa: E402
from ra_ild_igh.specifications import select_static_igh_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit one prepared RA-ILD IGH scheme's effective model features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    duplicated = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicated:
        raise ValueError(f"{label} has duplicate columns: {duplicated[:20]}")
    return frame


def main() -> int:
    args = parse_args()
    try:
        config = load_experiment_config(
            Path(args.config),
            repository_root=Path(args.repository_root) if args.repository_root else None,
        )
        base, input_audit = load_partition_feature_matrix(config, "train")
        manifest = _read_csv(
            config.path("data.train.feature_manifest", must_exist=True, expect="file"),
            "feature manifest",
        )
        static_features = select_static_igh_features(
            manifest,
            available_columns=base.columns,
            require_all_available=True,
        )
        selection = config.section("model_selection")
        static_group_name = str(selection["static_feature_group"])
        allowed_dynamic = set(config.section("public_reference")["dynamic_features"])

        feature_rows: List[Dict[str, object]] = []
        summary_rows: List[Dict[str, object]] = []
        for model_name, spec in config.section("models").items():
            numeric = [str(x) for x in spec.get("numeric", [])]
            categorical = [str(x) for x in spec.get("categorical", [])]
            groups = [str(x) for x in spec.get("static_feature_groups", [])]
            dynamic = [str(x) for x in spec.get("dynamic_public", [])]

            missing_explicit = sorted(
                set(numeric + categorical) - set(map(str, base.columns))
            )
            if missing_explicit:
                raise ValueError(
                    f"{model_name} references columns absent from merged training matrix: "
                    f"{missing_explicit}"
                )
            for column in numeric:
                converted = pd.to_numeric(base[column], errors="coerce")
                if converted.isna().any() or not np.isfinite(converted.to_numpy(float)).all():
                    raise ValueError(
                        f"{model_name} numeric feature {column!r} contains missing, "
                        "non-numeric, or infinite values"
                    )
            for column in categorical:
                if base[column].isna().any():
                    raise ValueError(
                        f"{model_name} categorical feature {column!r} contains missing values"
                    )
            unsupported_dynamic = sorted(set(dynamic) - allowed_dynamic)
            if unsupported_dynamic:
                raise ValueError(
                    f"{model_name} uses unsupported dynamic public features: "
                    f"{unsupported_dynamic}"
                )
            unsupported_groups = sorted(set(groups) - {static_group_name})
            if unsupported_groups:
                raise ValueError(
                    f"{model_name} uses unsupported static feature groups: "
                    f"{unsupported_groups}"
                )

            ordered = []
            for source_type, names in (
                ("explicit_numeric", numeric),
                (
                    "static_igh",
                    list(static_features) if static_group_name in groups else [],
                ),
                ("dynamic_public", dynamic),
                ("categorical_raw", categorical),
            ):
                for feature_name in names:
                    if any(row[0] == feature_name for row in ordered):
                        raise ValueError(
                            f"{model_name} contains duplicate effective feature {feature_name!r}"
                        )
                    ordered.append((feature_name, source_type))

            for order, (feature_name, source_type) in enumerate(ordered, start=1):
                feature_rows.append(
                    {
                        "model": model_name,
                        "feature_order": order,
                        "feature_name": feature_name,
                        "source_type": source_type,
                        "present_in_merged_training_matrix": source_type != "dynamic_public",
                    }
                )
            summary_rows.append(
                {
                    "model": model_name,
                    "explicit_numeric_count": len(numeric),
                    "static_feature_count": len(static_features)
                    if static_group_name in groups
                    else 0,
                    "dynamic_public_count": len(dynamic),
                    "categorical_raw_count": len(categorical),
                    "effective_raw_feature_count": len(ordered),
                }
            )

        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
        else:
            output_dir = Path(args.config).expanduser().resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / "feature_input_audit.csv"
        features_path = output_dir / "resolved_model_features.csv"
        summary_path = output_dir / "resolved_model_summary.csv"
        json_path = output_dir / "feature_audit_summary.json"
        input_audit.to_csv(input_path, index=False)
        pd.DataFrame(feature_rows).to_csv(features_path, index=False)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        summary = {
            "status": "PASS",
            "experiment_id": config.experiment_id,
            "merged_training_samples": int(len(base)),
            "merged_training_columns": int(len(base.columns)),
            "additional_feature_tables": int(
                (input_audit["source_type"] == "additional_feature_table").sum()
            ),
            "static_feature_count": int(len(static_features)),
            "models": list(config.section("models")),
            "outputs": {
                "feature_input_audit": str(input_path),
                "resolved_model_features": str(features_path),
                "resolved_model_summary": str(summary_path),
            },
        }
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("RA-ILD IGH feature audit: PASS")
        print(f"Experiment:       {config.experiment_id}")
        print(f"Samples:          {len(base)}")
        print(f"Merged columns:   {len(base.columns)}")
        print(f"Static features:  {len(static_features)}")
        print(f"Models:           {len(config.section('models'))}")
        print(f"Additional tables: {summary['additional_feature_tables']}")
        print(f"Output:           {output_dir}")
        return 0
    except Exception as exc:
        print(f"RA-ILD IGH feature audit: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
