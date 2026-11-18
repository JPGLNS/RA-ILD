#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create immutable Batch-09 repeat-specific 3-mer scheme copies for frozen V3 bundles.

The script never edits a Training Bundle, its frozen marker, its split assignments,
or the original generated scheme. It creates one new scheme copy per cohort.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.paths import find_repository_root  # noqa: E402


DEFAULT_BUNDLE_CONFIGS = {
    "total": "TRB/set/configs/repeated_holdout_training/trb_v3_179_top10000_total_training_v1.yaml",
    "pbmc": "TRB/set/configs/repeated_holdout_training/trb_v3_179_top10000_pbmc_training_v1.yaml",
    "buffycoat": "TRB/set/configs/repeated_holdout_training/trb_v3_179_top10000_buffycoat_training_v1.yaml",
}
DEFAULT_OUTPUT_DIR = "TRB/set/configs/schemes/generated/repeat3mer"
SCHEME_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")

REPEAT_3MER_OPTIONS = {
    "enabled": True,
    "selection_scope": "outer_repeat_train_only",
    "ranking_rule": "prevalence_then_total_variance",
    "sequence_column": "cdr3_aa",
    "min_sample_count": 5,
    "min_prevalence": 0.05,
    "min_unweighted_variance": 1.0e-12,
    "min_weighted_variance": 1.0e-12,
    "max_kmers": 500,
    "top_k_values": [50, 100, 200, 500],
    "expected_core_feature_count": 83,
    "core_group_name": "core83",
    "legacy_group_name": "static_tcr_candidate_predictors",
}


class Repeat3merBundleSchemeError(ValueError):
    """Raised when a frozen bundle cannot safely produce a 3-mer scheme."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Repeat3merBundleSchemeError(f"{label} root must be a mapping: {path}")
    return copy.deepcopy(dict(value))


def _resolve(value: object, root: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise Repeat3merBundleSchemeError(f"{context} must be a non-empty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _repo_relative(path: Path, root: Path, context: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise Repeat3merBundleSchemeError(
            f"{context} must be inside repository root: {path}"
        ) from exc


def _require_hash_record(
    marker: Mapping[str, Any],
    name: str,
    path: Path,
    root: Path,
) -> Mapping[str, Any]:
    files = marker.get("files")
    if not isinstance(files, Mapping) or not isinstance(files.get(name), Mapping):
        raise Repeat3merBundleSchemeError(
            f"Frozen bundle marker is missing files.{name}"
        )
    record = files[name]
    expected_path = _repo_relative(path, root, f"files.{name}.path")
    if record.get("path") != expected_path:
        raise Repeat3merBundleSchemeError(
            f"Frozen files.{name}.path mismatch: {record.get('path')!r} != {expected_path!r}"
        )
    observed_hash = _sha256(path)
    if record.get("sha256") != observed_hash:
        raise Repeat3merBundleSchemeError(
            f"Frozen files.{name}.sha256 mismatch for {path}"
        )
    return record


def _read_aa_manifest(path: Path, expected_samples: int) -> Tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "sample_id" not in reader.fieldnames:
            raise Repeat3merBundleSchemeError(
                f"AA manifest must contain sample_id: {path}"
            )
        if "path" not in reader.fieldnames or "frequency_column" not in reader.fieldnames:
            raise Repeat3merBundleSchemeError(
                f"AA manifest must contain path and frequency_column: {path}"
            )
        rows = list(reader)

    sample_ids = tuple(str(row["sample_id"]).strip() for row in rows)
    if any(not value for value in sample_ids):
        raise Repeat3merBundleSchemeError("AA manifest contains an empty sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise Repeat3merBundleSchemeError("AA manifest contains duplicate sample_id")
    if len(sample_ids) != int(expected_samples):
        raise Repeat3merBundleSchemeError(
            f"AA manifest rows={len(sample_ids)} but frozen bundle samples={expected_samples}"
        )

    missing_tables = []
    for row in rows:
        table_path = Path(str(row["path"])).expanduser()
        if not table_path.is_absolute():
            table_path = path.parent.parent.parent / table_path
        if not table_path.is_file():
            missing_tables.append(str(table_path))
            if len(missing_tables) >= 10:
                break
    if missing_tables:
        raise Repeat3merBundleSchemeError(
            "AA manifest points to missing clone tables:\n"
            + "\n".join(f"  - {value}" for value in missing_tables)
        )
    return sample_ids


def _validate_assignments(
    path: Path,
    expected_sample_ids: Sequence[str],
    marker: Mapping[str, Any],
) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise Repeat3merBundleSchemeError("Split assignments have no header")
        id_column = "sample_id" if "sample_id" in reader.fieldnames else "libraryid"
        required = {id_column, "repeat_index", "role"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise Repeat3merBundleSchemeError(
                f"Split assignments missing columns: {missing}"
            )
        rows = list(reader)

    by_repeat: Dict[int, list[Tuple[str, str]]] = {}
    for row in rows:
        sample_id = str(row[id_column]).strip()
        if not sample_id:
            raise Repeat3merBundleSchemeError("Split assignments contain empty sample IDs")
        try:
            repeat = int(row["repeat_index"])
        except (TypeError, ValueError) as exc:
            raise Repeat3merBundleSchemeError(
                f"Invalid repeat_index: {row.get('repeat_index')!r}"
            ) from exc
        role = str(row["role"]).strip().lower()
        if role not in {"train", "holdout"}:
            raise Repeat3merBundleSchemeError(
                f"Unsupported split role for repeat {repeat}: {role!r}"
            )
        by_repeat.setdefault(repeat, []).append((sample_id, role))

    repeats = sorted(by_repeat)
    expected_count = int(marker.get("split_count", 0))
    if not expected_count or repeats != list(range(1, expected_count + 1)):
        raise Repeat3merBundleSchemeError(
            f"Split repeat coverage mismatch: observed={repeats[:10]}, expected=1..{expected_count}"
        )

    expected = set(map(str, expected_sample_ids))
    train_sizes = []
    holdout_sizes = []
    for repeat in repeats:
        values = by_repeat[repeat]
        ids = [sample_id for sample_id, _ in values]
        if len(ids) != len(set(ids)):
            raise Repeat3merBundleSchemeError(
                f"Repeat {repeat} contains duplicate sample IDs"
            )
        if set(ids) != expected:
            missing_ids = sorted(expected - set(ids))[:10]
            extra_ids = sorted(set(ids) - expected)[:10]
            raise Repeat3merBundleSchemeError(
                f"Repeat {repeat} sample coverage mismatch; "
                f"missing={missing_ids}, extra={extra_ids}"
            )
        train_n = sum(role == "train" for _, role in values)
        holdout_n = sum(role == "holdout" for _, role in values)
        if train_n < 5 or holdout_n < 1:
            raise Repeat3merBundleSchemeError(
                f"Repeat {repeat} has invalid train/holdout sizes: {train_n}/{holdout_n}"
            )
        train_sizes.append(train_n)
        holdout_sizes.append(holdout_n)

    frozen_train = [int(value) for value in marker.get("train_sizes", [])]
    frozen_holdout = [int(value) for value in marker.get("holdout_sizes", [])]
    if train_sizes != frozen_train or holdout_sizes != frozen_holdout:
        raise Repeat3merBundleSchemeError(
            "Split sizes do not match the frozen Training Bundle marker"
        )
    if len(set(train_sizes)) != 1 or len(set(holdout_sizes)) != 1:
        raise Repeat3merBundleSchemeError(
            "This scheme adapter requires constant train/holdout sizes across repeats"
        )
    return {
        "repeat_count": len(repeats),
        "train_size": train_sizes[0],
        "holdout_size": holdout_sizes[0],
        "train_sizes": train_sizes,
        "holdout_sizes": holdout_sizes,
    }


def prepare_one(
    cohort: str,
    bundle_config_path: Path,
    repository_root: Path,
    output_dir: Path,
) -> Tuple[Dict[str, Any], str]:
    """Validate one frozen bundle and return its isolated repeat-3mer scheme copy."""

    spec = _load_yaml(bundle_config_path, f"{cohort} bundle config")
    bundle = spec.get("bundle")
    split_set = spec.get("split_set")
    scheme_spec = spec.get("scheme")
    aa_tables = spec.get("aa_tables")
    for label, value in (
        ("bundle", bundle),
        ("split_set", split_set),
        ("scheme", scheme_spec),
        ("aa_tables", aa_tables),
    ):
        if not isinstance(value, Mapping):
            raise Repeat3merBundleSchemeError(f"{cohort} config section {label} must be a mapping")

    bundle_root = _resolve(bundle.get("output_root"), repository_root, f"{cohort}.bundle.output_root")
    bundle_marker_path = bundle_root / "TRAINING_BUNDLE_FROZEN.json"
    if not bundle_marker_path.is_file():
        raise FileNotFoundError(
            f"{cohort} frozen Training Bundle marker not found: {bundle_marker_path}"
        )
    marker = json.loads(bundle_marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "FROZEN":
        raise Repeat3merBundleSchemeError(
            f"{cohort} Training Bundle status is not FROZEN"
        )
    if marker.get("bundle_id") != bundle.get("id"):
        raise Repeat3merBundleSchemeError(
            f"{cohort} bundle_id does not match its bundle config"
        )

    aa_manifest_path = bundle_root / "aa_clone_table_manifest.csv"
    _require_hash_record(marker, "aa_manifest", aa_manifest_path, repository_root)
    sample_ids = _read_aa_manifest(aa_manifest_path, int(marker.get("samples", 0)))

    assignments_path = _resolve(
        split_set.get("assignments"), repository_root, f"{cohort}.split_set.assignments"
    )
    frozen_split_path = _resolve(
        split_set.get("frozen_marker"), repository_root, f"{cohort}.split_set.frozen_marker"
    )
    if not frozen_split_path.is_file():
        raise FileNotFoundError(f"{cohort} frozen split marker not found: {frozen_split_path}")
    if marker.get("split_frozen_marker_sha256") != _sha256(frozen_split_path):
        raise Repeat3merBundleSchemeError(
            f"{cohort} frozen split marker hash does not match Training Bundle"
        )
    split_audit = _validate_assignments(assignments_path, sample_ids, marker)

    generated_scheme_path = _resolve(
        scheme_spec.get("generated_scheme"),
        repository_root,
        f"{cohort}.scheme.generated_scheme",
    )
    if marker.get("generated_scheme") != _repo_relative(
        generated_scheme_path, repository_root, f"{cohort}.generated_scheme"
    ):
        raise Repeat3merBundleSchemeError(
            f"{cohort} generated scheme path does not match frozen bundle marker"
        )
    if marker.get("generated_scheme_sha256") != _sha256(generated_scheme_path):
        raise Repeat3merBundleSchemeError(
            f"{cohort} generated scheme hash does not match frozen bundle marker"
        )

    source_scheme = _load_yaml(generated_scheme_path, f"{cohort} generated scheme")
    source_scheme_section = source_scheme.get("scheme")
    if not isinstance(source_scheme_section, Mapping):
        raise Repeat3merBundleSchemeError(
            f"{cohort} generated scheme has no scheme mapping"
        )
    source_id = str(scheme_spec.get("id", "")).strip()
    if source_scheme_section.get("id") != source_id:
        raise Repeat3merBundleSchemeError(
            f"{cohort} generated scheme ID does not match bundle config"
        )
    if source_scheme_section.get("output_root") != scheme_spec.get("output_root"):
        raise Repeat3merBundleSchemeError(
            f"{cohort} generated scheme output_root does not match bundle config"
        )
    if "repeat_3mer_features" in source_scheme:
        raise Repeat3merBundleSchemeError(
            f"{cohort} source scheme already contains repeat_3mer_features; refusing to layer twice"
        )

    new_id = f"{source_id}_repeat3mer"
    if not SCHEME_ID_PATTERN.fullmatch(new_id):
        raise Repeat3merBundleSchemeError(
            f"Generated scheme ID is invalid or too long: {new_id}"
        )
    new_output_root = f"TRB/set/experiments/{new_id}"
    new_output_path = (repository_root / new_output_root).resolve()
    if new_output_path.exists() and any(new_output_path.iterdir()):
        raise FileExistsError(
            f"{cohort} repeat-3mer experiment output already contains files: {new_output_path}"
        )

    scheme_copy = copy.deepcopy(source_scheme)
    scheme_copy["scheme"]["id"] = new_id
    scheme_copy["scheme"]["description"] = (
        str(source_scheme_section.get("description", "")).strip()
        + " Repeat-specific 3-mer vocabulary is fitted using outer-training samples only."
    ).strip()
    scheme_copy["scheme"]["output_root"] = new_output_root

    repeat_options = copy.deepcopy(REPEAT_3MER_OPTIONS)
    repeat_options["aa_manifest"] = _repo_relative(
        aa_manifest_path, repository_root, f"{cohort}.aa_manifest"
    )
    repeat_options["sequence_column"] = str(aa_tables.get("sequence_column", "cdr3_aa"))
    scheme_copy["repeat_3mer_features"] = repeat_options

    output_path = output_dir / f"{cohort}_{new_id}.yaml"
    payload = yaml.safe_dump(
        scheme_copy, allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    record = {
        "cohort": cohort,
        "bundle_id": marker["bundle_id"],
        "bundle_marker": _repo_relative(bundle_marker_path, repository_root, "bundle marker"),
        "bundle_marker_sha256": _sha256(bundle_marker_path),
        "sample_count": len(sample_ids),
        "repeat_count": split_audit["repeat_count"],
        "train_size": split_audit["train_size"],
        "holdout_size": split_audit["holdout_size"],
        "split_assignments": _repo_relative(assignments_path, repository_root, "split assignments"),
        "split_assignments_sha256": _sha256(assignments_path),
        "aa_manifest": _repo_relative(aa_manifest_path, repository_root, "AA manifest"),
        "aa_manifest_sha256": _sha256(aa_manifest_path),
        "source_generated_scheme": _repo_relative(
            generated_scheme_path, repository_root, "source generated scheme"
        ),
        "source_generated_scheme_sha256": _sha256(generated_scheme_path),
        "scheme_id": new_id,
        "scheme_path": _repo_relative(output_path, repository_root, "output scheme"),
        "scheme_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "repeat_3mer_features": repeat_options,
    }
    return record, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare isolated repeat-specific 3-mer scheme copies from frozen V3 bundles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repository-root", default=None)
    parser.add_argument(
        "--cohorts",
        nargs="+",
        choices=tuple(DEFAULT_BUNDLE_CONFIGS),
        default=list(DEFAULT_BUNDLE_CONFIGS),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write scheme copies and a manifest. Without this flag, run validation only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing scheme copies/manifest; never modifies frozen bundles.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = (
        Path(args.repository_root).expanduser().resolve()
        if args.repository_root
        else find_repository_root(SCRIPT_DIR)
    )
    output_dir = _resolve(args.output_dir, repository_root, "output_dir")
    records = []
    payloads: Dict[Path, str] = {}

    for cohort in args.cohorts:
        config_path = _resolve(
            DEFAULT_BUNDLE_CONFIGS[cohort], repository_root, f"{cohort}.bundle_config"
        )
        record, payload = prepare_one(
            cohort, config_path, repository_root, output_dir
        )
        output_path = repository_root / record["scheme_path"]
        if output_path.exists():
            existing_hash = _sha256(output_path)
            if existing_hash != record["scheme_sha256"] and not args.overwrite:
                raise FileExistsError(
                    f"Scheme copy exists with different content: {output_path}; "
                    "use --overwrite only after reviewing it."
                )
        payloads[output_path] = payload
        records.append(record)

    manifest = {
        "status": "PREPARED" if args.apply else "DRY_RUN_PASS",
        "adapter_version": "1.0",
        "cohorts": records,
        "frozen_bundle_policy": (
            "read_only: no Training Bundle, frozen marker, split assignment, "
            "or original generated scheme is modified"
        ),
        "selection_policy": "outer_repeat_train_only",
    }
    manifest_path = output_dir / "REPEAT3MER_BUNDLE_SCHEME_MANIFEST.json"
    manifest_payload = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=False
    ) + "\n"

    print("TRB V3 repeat-specific 3-mer scheme preparation")
    print(f"Repository root: {repository_root}")
    print(f"Output directory: {output_dir}")
    for record in records:
        print(
            f"[PASS] {record['cohort']}: samples={record['sample_count']}; "
            f"repeats={record['repeat_count']}; "
            f"train/holdout={record['train_size']}/{record['holdout_size']}; "
            f"scheme={record['scheme_id']}"
        )
        print(f"       AA manifest SHA256: {record['aa_manifest_sha256']}")
        print(f"       Split SHA256:       {record['split_assignments_sha256']}")
    print(f"3-mer: Top {REPEAT_3MER_OPTIONS['max_kmers']}; "
          f"min_count={REPEAT_3MER_OPTIONS['min_sample_count']}; "
          f"min_prevalence={REPEAT_3MER_OPTIONS['min_prevalence']}")
    print("Original frozen bundles and generated schemes: READ-ONLY")

    if not args.apply:
        print("DRY-RUN PASS: no files were written. Add --apply to create scheme copies.")
        return 0

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in payloads.items():
        if path.exists() and not args.overwrite:
            if _sha256(path) == hashlib.sha256(payload.encode("utf-8")).hexdigest():
                print(f"[SKIP] identical scheme already exists: {path}")
                continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        print(f"[WRITE] {path}")

    if manifest_path.exists() and not args.overwrite:
        if _sha256(manifest_path) != hashlib.sha256(manifest_payload.encode("utf-8")).hexdigest():
            raise FileExistsError(
                f"Manifest exists with different content: {manifest_path}; use --overwrite."
            )
    manifest_path.write_text(manifest_payload, encoding="utf-8")
    print(f"[WRITE] {manifest_path}")
    print("SCHEME PREPARATION PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SCHEME PREPARATION FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
