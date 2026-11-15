#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe/strict patch for IGH V3 eight-column fold-local V2 Public reference.

--check inspects, compiles and makes no project changes.
--apply edits only experiment_spec.py, fold_features.py and adds v3_public_adapter.py.
No model training, YAML generation, Git operations or dataset writes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import shutil
import subprocess
from pathlib import Path

EXPECTED_BRANCH = "feature/igh-v3-hierarchical-5alg"
REL = Path("IGH/set/src/ra_ild_igh")

# Restrict replacements to unique, audited anchors. Phase2A already applied.
SPEC_PATCHES = [
    (
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass, field\n",
        "spec dataclass import",
    ),
    (
        "@dataclass(frozen=True)\nclass FeatureSpec:\n",
        '''@dataclass(frozen=True)
class PublicReferenceSpec:
    enabled: bool = False
    feature_set: str = "raw_bilateral"
    global_min_prevalence: float = 0.02
    global_min_count: int = 3
    group_min_prevalence: float = 0.05
    group_min_count: int = 3
    specific_prevalence_delta: float = 0.05
    epsilon: float = 1.0e-8


@dataclass(frozen=True)
class FeatureSpec:
''',
        "spec PublicReferenceSpec",
    ),
    (
        "    enriched_dictionary: EnrichedDictionarySpec\n\n    @property",
        "    enriched_dictionary: EnrichedDictionarySpec\n    public_reference: PublicReferenceSpec = field(default_factory=PublicReferenceSpec)\n\n    @property",
        "spec FeatureSpec field",
    ),
    (
        '        if self.kmer.enabled:\n            groups.append(f"tcr_3mer_{self.kmer.representation}")\n        return tuple(groups)',
        '        if self.kmer.enabled:\n            groups.append(f"tcr_3mer_{self.kmer.representation}")\n        if self.public_reference.enabled:\n            groups.append("public_reference_raw_bilateral")\n        return tuple(groups)',
        "spec group introspection",
    ),
    (
        '''                "enriched_dictionary": {
                    "enabled": self.features.enriched_dictionary.enabled,
                    "threshold_pct": self.features.enriched_dictionary.threshold_pct,
                    "delta_pct": self.features.enriched_dictionary.delta_pct,
                    "features": list(self.features.enriched_dictionary.features),
                    "repertoire_source": self.repertoire.source_id,
                    "source_policy": "inherit_main_repertoire",
                },
''',
        '''                "enriched_dictionary": {
                    "enabled": self.features.enriched_dictionary.enabled,
                    "threshold_pct": self.features.enriched_dictionary.threshold_pct,
                    "delta_pct": self.features.enriched_dictionary.delta_pct,
                    "features": list(self.features.enriched_dictionary.features),
                    "repertoire_source": self.repertoire.source_id,
                    "source_policy": "inherit_main_repertoire",
                },
                "public_reference": {
                    "enabled": self.features.public_reference.enabled,
                    "feature_set": self.features.public_reference.feature_set,
                    "global_min_prevalence": self.features.public_reference.global_min_prevalence,
                    "global_min_count": self.features.public_reference.global_min_count,
                    "group_min_prevalence": self.features.public_reference.group_min_prevalence,
                    "group_min_count": self.features.public_reference.group_min_count,
                    "specific_prevalence_delta": self.features.public_reference.specific_prevalence_delta,
                    "epsilon": self.features.public_reference.epsilon,
                    "repertoire_source": self.repertoire.source_id,
                    "reference_policy": "train_only_exact_LOO",
                },
''',
        "spec as_dict",
    ),
    (
        '''    unknown = sorted(set(e_features) - ENRICH_FEATURES)
    if unknown:
        raise ExperimentSpecError(f"unknown enriched_dictionary features: {unknown}")

    model_raw = _mapping''',
        '''    unknown = sorted(set(e_features) - ENRICH_FEATURES)
    if unknown:
        raise ExperimentSpecError(f"unknown enriched_dictionary features: {unknown}")

    praw = _mapping(features_raw.get("public_reference", {}), "features.public_reference")
    for forbidden in ("source", "repertoire_source", "source_id"):
        if forbidden in praw:
            raise ExperimentSpecError(
                f"features.public_reference.{forbidden} is forbidden; source inherits selected repertoire"
            )
    public_enabled = _bool(praw.get("enabled", False), "features.public_reference.enabled")
    public_feature_set = str(praw.get("feature_set", "raw_bilateral")).strip().lower()
    if public_feature_set != "raw_bilateral":
        raise ExperimentSpecError("V3 Public currently supports only raw_bilateral (8 features)")
    public_global_prev = _float(praw.get("global_min_prevalence", 0.02), "public.global_min_prevalence", 0.0)
    public_group_prev = _float(praw.get("group_min_prevalence", 0.05), "public.group_min_prevalence", 0.0)
    public_delta = _float(praw.get("specific_prevalence_delta", 0.05), "public.specific_prevalence_delta", 0.0)
    if max(public_global_prev, public_group_prev, public_delta) > 1.0:
        raise ExperimentSpecError("Public prevalence and delta must be <= 1")
    public_global_min = _int(praw.get("global_min_count", 3), "public.global_min_count", 1)
    public_group_min = _int(praw.get("group_min_count", 3), "public.group_min_count", 1)
    public_epsilon = _float(praw.get("epsilon", 1.0e-8), "public.epsilon", 0.0)
    if public_epsilon <= 0:
        raise ExperimentSpecError("features.public_reference.epsilon must be > 0")

    model_raw = _mapping''',
        "spec config parsing",
    ),
    (
        '''            enriched_dictionary=EnrichedDictionarySpec(
                enabled=e_enabled,
                threshold_pct=threshold,
                delta_pct=delta,
                features=e_features,
            ),
''',
        '''            enriched_dictionary=EnrichedDictionarySpec(
                enabled=e_enabled,
                threshold_pct=threshold,
                delta_pct=delta,
                features=e_features,
            ),
            public_reference=PublicReferenceSpec(
                enabled=public_enabled,
                feature_set=public_feature_set,
                global_min_prevalence=public_global_prev,
                global_min_count=public_global_min,
                group_min_prevalence=public_group_prev,
                group_min_count=public_group_min,
                specific_prevalence_delta=public_delta,
                epsilon=public_epsilon,
            ),
''',
        "spec instance construction",
    ),
]

FOLD_PATCHES = [
    (
        "from .feature_framework import FeatureRegistry, load_feature_registry\n",
        "from .feature_framework import FeatureRegistry, load_feature_registry\n"
        "from .v3_public_adapter import (\n"
        "    PublicInput, V2_PUBLIC_COLUMNS, build_public_fold, load_public_input,\n"
        ")\n",
        "fold import",
    ),
    (
        "    enrich_repertoires: Mapping[str, SampleRepertoire]\n\n    @property",
        "    enrich_repertoires: Mapping[str, SampleRepertoire]\n"
        "    public_input: Optional[PublicInput] = None\n\n    @property",
        "fold data class",
    ),
    (
        '''    if spec.base.features.enriched_dictionary.enabled:
        enrich_meta = metadata.rename(columns={"sample_id": "libraryid"})
        enrich = load_repertoires(enrich_meta, source, id_col="libraryid")

    return FoldFeatureInputs(''',
        '''    if spec.base.features.enriched_dictionary.enabled:
        enrich_meta = metadata.rename(columns={"sample_id": "libraryid"})
        enrich = load_repertoires(enrich_meta, source, id_col="libraryid")

    public_input = None
    if spec.base.features.public_reference.enabled:
        public_input = load_public_input(
            ids, metadata["cohort"].astype(str).tolist(), source,
        )

    return FoldFeatureInputs(''',
        "fold load inputs",
    ),
    (
        '''        enrich_repertoires=enrich,
    )
''',
        '''        enrich_repertoires=enrich,
        public_input=public_input,
    )
''',
        "fold return inputs",
    ),
    (
        '''    required = [*numeric, *categorical]
    if len(required) != len(set(required)):
''',
        '''    public_spec = spec.base.features.public_reference
    if public_spec.enabled:
        if inputs.public_input is None:
            raise FoldFeatureError("Public-reference feature inputs were not loaded")
        public_fold = build_public_fold(
            inputs.public_input, public_spec, train_ids, valid_ids, context=context,
        )
        public_columns = list(V2_PUBLIC_COLUMNS)
        if set(public_columns) & (set(numeric) | set(categorical)):
            raise FoldFeatureError("Public feature names conflict with existing predictors")
        train = train.merge(
            public_fold.train, on="sample_id", how="left", validate="one_to_one",
        )
        valid = valid.merge(
            public_fold.valid, on="sample_id", how="left", validate="one_to_one",
        )
        numeric.extend(public_columns)
        audit_rows.append(dict(public_fold.audit))

    required = [*numeric, *categorical]
    if len(required) != len(set(required)):
''',
        "fold feature assembly",
    ),
]


def replace_once(source: str, before: str, after: str, label: str) -> str:
    n = source.count(before)
    if n == 1:
        return source.replace(before, after, 1)
    if n == 0 and source.count(after) == 1:
        print(f"[ALREADY PATCHED] {label}")
        return source
    raise RuntimeError(f"Refuse ambiguous source patch {label}: original occurrences={n}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = p.parse_args()
    repo = args.repo.resolve()
    branch = subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip()
    if branch != EXPECTED_BRANCH:
        raise SystemExit(f"Wrong Git branch: {branch!r}; expected {EXPECTED_BRANCH!r}")
    base = repo / REL
    adapter_from = Path(__file__).resolve().with_name("v3_public_adapter.py")
    if not adapter_from.is_file():
        raise SystemExit(f"Missing bundled adapter: {adapter_from}")
    if not (base / "public_reference.py").is_file():
        raise SystemExit("V2 public_reference.py is missing from the target repository")
    modifications = []
    for name, patches in (("experiment_spec.py", SPEC_PATCHES), ("fold_features.py", FOLD_PATCHES)):
        path = base / name
        old = path.read_text(encoding="utf-8")
        new = old
        for before, after, label in patches:
            if after in new:
                print(f"[ALREADY PATCHED] {label}")
                continue
            new = replace_once(new, before, after, label)
        compile(new, str(path), "exec")
        modifications.append((path, old, new))
        print(f"[PATCH CHECK] {name}: changed={old != new}; before_SHA256={hashlib.sha256(old.encode()).hexdigest()[:12]}")
    new_adapter = adapter_from.read_text(encoding="utf-8")
    compile(new_adapter, str(base / adapter_from.name), "exec")
    current_adapter_path = base / adapter_from.name
    if current_adapter_path.is_file():
        existing = current_adapter_path.read_text(encoding="utf-8")
        if existing != new_adapter:
            raise SystemExit("Existing v3_public_adapter.py differs; abort instead of overwriting")
    else:
        modifications.append((current_adapter_path, None, new_adapter))
    print("[SYNTAX PASS] experiment_spec.py, fold_features.py, v3_public_adapter.py")
    if args.check:
        print("[CHECK ONLY] No repository changes made")
        return
    delta = [(path, old, new) for path, old, new in modifications if old != new]
    if not delta:
        print("[NOOP] Already applied; no files changed")
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = Path("/tmp") / f"igh_v3_phase2b_backup_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    for path, old, _ in delta:
        if old is not None:
            (backup / path.name).write_text(old, encoding="utf-8")
    for path, _, new in delta:
        path.write_text(new, encoding="utf-8")
    print(f"[PATCH PASS] {len(delta)} files applied; backups: {backup}")
    print("[NEXT] python IGH/set/tests/test_igh_v3_public_reference.py")
    print("[NEXT] git diff --check")


if __name__ == "__main__":
    main()
