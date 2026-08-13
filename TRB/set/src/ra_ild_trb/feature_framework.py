#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TRB feature-framework contracts for exploratory repertoire modeling.

Phase-1 scope
-------------
This module does not calculate biological features and does not fit models. It
provides a strict, auditable contract for:

* logical repertoire sources (for example ``all`` and ``top10000``);
* modular feature groups;
* training-derived versus per-sample feature definitions;
* validation/resolution of experiment feature plans;
* deterministic classification of existing matrix columns by registry rules.

The goal is to decouple feature definitions from the legacy fixed Step02/03/04
pipeline before those steps are migrated in later upgrades.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for the TRB feature framework") from exc


SCHEMA_VERSION = "1.0"
ALLOWED_SOURCE_STATUS = {"available", "planned"}
ALLOWED_GROUP_STATUS = {"defined", "legacy_optional", "planned"}
ALLOWED_MODULE_TYPES = {
    "metadata_static",
    "sample_static",
    "training_vocabulary",
    "learned_reference",
    "descriptive_legacy",
}
ALLOWED_FIT_SCOPES = {"none", "per_sample", "training_only"}
ALLOWED_REPERTOIRE_DEPENDENCY = {"none", "repertoire"}
ALLOWED_SUMMARY_KINDS = {"step01_full", "step01b_topk"}


class FeatureFrameworkError(ValueError):
    """Raised when a registry or feature plan violates the framework contract."""


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeatureFrameworkError(f"{context} must be a mapping")
    return value


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeatureFrameworkError(f"{context} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any, context: str, *, allow_empty: bool = True) -> Tuple[str, ...]:
    if value is None and allow_empty:
        return tuple()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FeatureFrameworkError(f"{context} must be a list of strings")
    out: List[str] = []
    for index, item in enumerate(value):
        out.append(_nonempty_string(item, f"{context}[{index}]"))
    if not allow_empty and not out:
        raise FeatureFrameworkError(f"{context} must not be empty")
    if len(out) != len(set(out)):
        raise FeatureFrameworkError(f"{context} contains duplicates")
    return tuple(out)


def _bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise FeatureFrameworkError(f"{context} must be true or false")
    return value


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _mapping(raw, label)


@dataclass(frozen=True)
class ColumnSelector:
    exact: Tuple[str, ...]
    prefixes: Tuple[str, ...]

    def matches(self, column: str) -> bool:
        return column in self.exact or any(column.startswith(prefix) for prefix in self.prefixes)


@dataclass(frozen=True)
class RepertoireSource:
    id: str
    status: str
    representation: str
    source_layer: str
    default_path: str
    file_glob: str
    sample_filename_template: str
    summary_kind: str
    summary_path: str
    parent_source: Optional[str]
    required_columns: Tuple[str, ...]
    weight_column: str
    top_k: Optional[int]

    def sample_filename(self, sample_id: str) -> str:
        sample_id = _nonempty_string(sample_id, "sample_id")
        try:
            rendered = self.sample_filename_template.format(sample_id=sample_id)
        except (KeyError, ValueError) as exc:
            raise FeatureFrameworkError(
                f"invalid sample_filename_template for source {self.id!r}: "
                f"{self.sample_filename_template!r}"
            ) from exc
        if not rendered or Path(rendered).name != rendered:
            raise FeatureFrameworkError(
                f"source {self.id!r} sample filename must render to a file name, got {rendered!r}"
            )
        return rendered


@dataclass(frozen=True)
class FeatureGroup:
    id: str
    family: str
    status: str
    module_type: str
    fit_scope: str
    leakage_policy: str
    repertoire_dependency: str
    supported_repertoire_sources: Tuple[str, ...]
    producer: str
    model_role: str
    selector: ColumnSelector
    reference_drop_columns: Tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class FeaturePlan:
    id: str
    receptor: str
    repertoire_source: str
    feature_groups: Tuple[str, ...]
    allow_legacy: bool
    allow_planned: bool


@dataclass(frozen=True)
class ResolvedFeaturePlan:
    plan_id: str
    receptor: str
    repertoire_source: RepertoireSource
    feature_groups: Tuple[FeatureGroup, ...]
    warnings: Tuple[str, ...]

    @property
    def requires_training_fit(self) -> bool:
        return any(group.fit_scope == "training_only" for group in self.feature_groups)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "receptor": self.receptor,
            "repertoire_source": {
                "id": self.repertoire_source.id,
                "representation": self.repertoire_source.representation,
                "source_layer": self.repertoire_source.source_layer,
                "default_path": self.repertoire_source.default_path,
                "file_glob": self.repertoire_source.file_glob,
                "sample_filename_template": self.repertoire_source.sample_filename_template,
                "summary_kind": self.repertoire_source.summary_kind,
                "summary_path": self.repertoire_source.summary_path,
                "parent_source": self.repertoire_source.parent_source,
                "weight_column": self.repertoire_source.weight_column,
                "top_k": self.repertoire_source.top_k,
            },
            "feature_groups": [
                {
                    "id": group.id,
                    "family": group.family,
                    "status": group.status,
                    "module_type": group.module_type,
                    "fit_scope": group.fit_scope,
                    "leakage_policy": group.leakage_policy,
                    "producer": group.producer,
                    "model_role": group.model_role,
                }
                for group in self.feature_groups
            ],
            "requires_training_fit": self.requires_training_fit,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class FeatureRegistry:
    id: str
    receptor: str
    default_repertoire_source: str
    repertoire_sources: Mapping[str, RepertoireSource]
    feature_groups: Mapping[str, FeatureGroup]

    def resolve_plan(self, plan: FeaturePlan) -> ResolvedFeaturePlan:
        if plan.receptor != self.receptor:
            raise FeatureFrameworkError(
                f"plan receptor {plan.receptor!r} does not match registry receptor {self.receptor!r}"
            )
        if plan.repertoire_source not in self.repertoire_sources:
            raise FeatureFrameworkError(
                f"unknown repertoire source {plan.repertoire_source!r}; "
                f"available={sorted(self.repertoire_sources)}"
            )
        source = self.repertoire_sources[plan.repertoire_source]
        if source.status != "available":
            raise FeatureFrameworkError(
                f"repertoire source {source.id!r} is not available (status={source.status})"
            )

        resolved: List[FeatureGroup] = []
        warnings: List[str] = []
        for group_id in plan.feature_groups:
            if group_id not in self.feature_groups:
                raise FeatureFrameworkError(
                    f"unknown feature group {group_id!r}; available={sorted(self.feature_groups)}"
                )
            group = self.feature_groups[group_id]
            if group.status == "legacy_optional":
                if not plan.allow_legacy:
                    raise FeatureFrameworkError(
                        f"feature group {group.id!r} is legacy_optional; set allow_legacy: true to select it"
                    )
                warnings.append(f"legacy feature group selected: {group.id}")
            elif group.status == "planned":
                if not plan.allow_planned:
                    raise FeatureFrameworkError(
                        f"feature group {group.id!r} is planned but not implemented; "
                        "set allow_planned: true only for planning/contract validation"
                    )
                warnings.append(f"planned feature group selected: {group.id}")

            if group.repertoire_dependency == "repertoire":
                if source.id not in group.supported_repertoire_sources:
                    raise FeatureFrameworkError(
                        f"feature group {group.id!r} does not support repertoire source {source.id!r}; "
                        f"supported={list(group.supported_repertoire_sources)}"
                    )
            resolved.append(group)

        return ResolvedFeaturePlan(
            plan_id=plan.id,
            receptor=plan.receptor,
            repertoire_source=source,
            feature_groups=tuple(resolved),
            warnings=tuple(warnings),
        )

    def classify_columns(
        self,
        columns: Iterable[str],
        *,
        group_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """Classify feature columns using registry selectors.

        Each matched column must resolve to exactly one selected group. Ambiguous
        selectors are rejected so the registry cannot silently assign a feature to
        two modules.
        """
        if group_ids is None:
            groups = list(self.feature_groups.values())
        else:
            requested = _string_tuple(group_ids, "group_ids", allow_empty=False)
            unknown = [name for name in requested if name not in self.feature_groups]
            if unknown:
                raise FeatureFrameworkError(f"unknown group_ids: {unknown}")
            groups = [self.feature_groups[name] for name in requested]

        result: Dict[str, List[str]] = {group.id: [] for group in groups}
        unmatched: List[str] = []
        seen_columns = set()
        for raw in columns:
            column = _nonempty_string(str(raw), "column")
            if column in seen_columns:
                raise FeatureFrameworkError(f"duplicate input column: {column}")
            seen_columns.add(column)
            matches = [group for group in groups if group.selector.matches(column)]
            if len(matches) > 1:
                raise FeatureFrameworkError(
                    f"column {column!r} matches multiple groups: {[group.id for group in matches]}"
                )
            if not matches:
                unmatched.append(column)
            else:
                result[matches[0].id].append(column)
        return result, unmatched


def _parse_selector(raw: Any, context: str) -> ColumnSelector:
    mapping = _mapping(raw, context)
    exact = _string_tuple(mapping.get("exact", []), f"{context}.exact")
    prefixes = _string_tuple(mapping.get("prefixes", []), f"{context}.prefixes")
    if not exact and not prefixes:
        raise FeatureFrameworkError(f"{context} must define exact and/or prefixes")
    return ColumnSelector(exact=exact, prefixes=prefixes)


def load_feature_registry(path: Union[Path, str]) -> FeatureRegistry:
    path = Path(path).expanduser().resolve()
    raw = _load_yaml(path, "feature registry")
    if str(raw.get("schema_version")) != SCHEMA_VERSION:
        raise FeatureFrameworkError(f"feature registry schema_version must be {SCHEMA_VERSION!r}")

    registry_raw = _mapping(raw.get("registry"), "registry")
    registry_id = _nonempty_string(registry_raw.get("id"), "registry.id")
    receptor = _nonempty_string(registry_raw.get("receptor"), "registry.receptor")
    if receptor != "TRB":
        raise FeatureFrameworkError("registry.receptor must be TRB")
    default_source = _nonempty_string(
        registry_raw.get("default_repertoire_source"), "registry.default_repertoire_source"
    )

    sources_raw = _mapping(raw.get("repertoire_sources"), "repertoire_sources")
    if not sources_raw:
        raise FeatureFrameworkError("repertoire_sources must not be empty")
    sources: Dict[str, RepertoireSource] = {}
    for source_id, item in sources_raw.items():
        source_id = _nonempty_string(source_id, "repertoire source id")
        item = _mapping(item, f"repertoire_sources.{source_id}")
        status = _nonempty_string(item.get("status"), f"repertoire_sources.{source_id}.status")
        if status not in ALLOWED_SOURCE_STATUS:
            raise FeatureFrameworkError(
                f"repertoire_sources.{source_id}.status must be one of {sorted(ALLOWED_SOURCE_STATUS)}"
            )
        representation = _nonempty_string(
            item.get("representation"), f"repertoire_sources.{source_id}.representation"
        )
        top_k = item.get("top_k")
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
                raise FeatureFrameworkError(
                    f"repertoire_sources.{source_id}.top_k must be an integer >= 1 or null"
                )
        if representation == "topk" and top_k is None:
            raise FeatureFrameworkError(f"topk repertoire source {source_id!r} requires top_k")
        if representation == "full" and top_k is not None:
            raise FeatureFrameworkError(f"full repertoire source {source_id!r} must not define top_k")
        summary_kind = _nonempty_string(
            item.get("summary_kind"), f"repertoire_sources.{source_id}.summary_kind"
        )
        if summary_kind not in ALLOWED_SUMMARY_KINDS:
            raise FeatureFrameworkError(
                f"repertoire_sources.{source_id}.summary_kind must be one of "
                f"{sorted(ALLOWED_SUMMARY_KINDS)}"
            )
        parent_source = item.get("parent_source")
        if parent_source is not None:
            parent_source = _nonempty_string(
                parent_source, f"repertoire_sources.{source_id}.parent_source"
            )
            if parent_source == source_id:
                raise FeatureFrameworkError(
                    f"repertoire source {source_id!r} cannot be its own parent"
                )
        if representation == "full" and summary_kind != "step01_full":
            raise FeatureFrameworkError(
                f"full repertoire source {source_id!r} must use summary_kind=step01_full"
            )
        if representation == "topk" and summary_kind != "step01b_topk":
            raise FeatureFrameworkError(
                f"topk repertoire source {source_id!r} must use summary_kind=step01b_topk"
            )
        if representation == "full" and parent_source is not None:
            raise FeatureFrameworkError(
                f"full repertoire source {source_id!r} must not define parent_source"
            )
        if representation == "topk" and parent_source is None:
            raise FeatureFrameworkError(
                f"topk repertoire source {source_id!r} requires parent_source"
            )
        sources[source_id] = RepertoireSource(
            id=source_id,
            status=status,
            representation=representation,
            source_layer=_nonempty_string(
                item.get("source_layer"), f"repertoire_sources.{source_id}.source_layer"
            ),
            default_path=_nonempty_string(
                item.get("default_path"), f"repertoire_sources.{source_id}.default_path"
            ),
            file_glob=_nonempty_string(
                item.get("file_glob"), f"repertoire_sources.{source_id}.file_glob"
            ),
            sample_filename_template=_nonempty_string(
                item.get("sample_filename_template"),
                f"repertoire_sources.{source_id}.sample_filename_template",
            ),
            summary_kind=summary_kind,
            summary_path=_nonempty_string(
                item.get("summary_path"), f"repertoire_sources.{source_id}.summary_path"
            ),
            parent_source=parent_source,
            required_columns=_string_tuple(
                item.get("required_columns"),
                f"repertoire_sources.{source_id}.required_columns",
                allow_empty=False,
            ),
            weight_column=_nonempty_string(
                item.get("weight_column"), f"repertoire_sources.{source_id}.weight_column"
            ),
            top_k=top_k,
        )
    if default_source not in sources:
        raise FeatureFrameworkError(
            f"registry.default_repertoire_source {default_source!r} is not defined"
        )
    for source in sources.values():
        if source.parent_source is not None and source.parent_source not in sources:
            raise FeatureFrameworkError(
                f"repertoire source {source.id!r} references unknown parent_source "
                f"{source.parent_source!r}"
            )
        # Rendering one deterministic probe catches missing/invalid format fields early.
        source.sample_filename("__SAMPLE__")

    groups_raw = _mapping(raw.get("feature_groups"), "feature_groups")
    if not groups_raw:
        raise FeatureFrameworkError("feature_groups must not be empty")
    groups: Dict[str, FeatureGroup] = {}
    for group_id, item in groups_raw.items():
        group_id = _nonempty_string(group_id, "feature group id")
        item = _mapping(item, f"feature_groups.{group_id}")
        status = _nonempty_string(item.get("status"), f"feature_groups.{group_id}.status")
        if status not in ALLOWED_GROUP_STATUS:
            raise FeatureFrameworkError(
                f"feature_groups.{group_id}.status must be one of {sorted(ALLOWED_GROUP_STATUS)}"
            )
        module_type = _nonempty_string(
            item.get("module_type"), f"feature_groups.{group_id}.module_type"
        )
        if module_type not in ALLOWED_MODULE_TYPES:
            raise FeatureFrameworkError(
                f"feature_groups.{group_id}.module_type must be one of {sorted(ALLOWED_MODULE_TYPES)}"
            )
        fit_scope = _nonempty_string(
            item.get("fit_scope"), f"feature_groups.{group_id}.fit_scope"
        )
        if fit_scope not in ALLOWED_FIT_SCOPES:
            raise FeatureFrameworkError(
                f"feature_groups.{group_id}.fit_scope must be one of {sorted(ALLOWED_FIT_SCOPES)}"
            )
        repertoire_dependency = _nonempty_string(
            item.get("repertoire_dependency"),
            f"feature_groups.{group_id}.repertoire_dependency",
        )
        if repertoire_dependency not in ALLOWED_REPERTOIRE_DEPENDENCY:
            raise FeatureFrameworkError(
                f"feature_groups.{group_id}.repertoire_dependency must be one of "
                f"{sorted(ALLOWED_REPERTOIRE_DEPENDENCY)}"
            )
        supported_sources = _string_tuple(
            item.get("supported_repertoire_sources", []),
            f"feature_groups.{group_id}.supported_repertoire_sources",
        )
        unknown_sources = sorted(set(supported_sources) - set(sources))
        if unknown_sources:
            raise FeatureFrameworkError(
                f"feature_groups.{group_id} references unknown repertoire sources: {unknown_sources}"
            )
        if repertoire_dependency == "repertoire" and not supported_sources:
            raise FeatureFrameworkError(
                f"feature_groups.{group_id} depends on repertoire but supports no sources"
            )
        if repertoire_dependency == "none" and supported_sources:
            raise FeatureFrameworkError(
                f"feature_groups.{group_id} has repertoire_dependency=none but lists repertoire sources"
            )
        if fit_scope == "training_only" and module_type not in {
            "training_vocabulary", "learned_reference"
        }:
            raise FeatureFrameworkError(
                f"feature_groups.{group_id}: training_only fit_scope requires training_vocabulary or learned_reference"
            )
        groups[group_id] = FeatureGroup(
            id=group_id,
            family=_nonempty_string(item.get("family"), f"feature_groups.{group_id}.family"),
            status=status,
            module_type=module_type,
            fit_scope=fit_scope,
            leakage_policy=_nonempty_string(
                item.get("leakage_policy"), f"feature_groups.{group_id}.leakage_policy"
            ),
            repertoire_dependency=repertoire_dependency,
            supported_repertoire_sources=supported_sources,
            producer=_nonempty_string(item.get("producer"), f"feature_groups.{group_id}.producer"),
            model_role=_nonempty_string(
                item.get("model_role"), f"feature_groups.{group_id}.model_role"
            ),
            selector=_parse_selector(item.get("selector"), f"feature_groups.{group_id}.selector"),
            reference_drop_columns=_string_tuple(
                item.get("reference_drop_columns", []),
                f"feature_groups.{group_id}.reference_drop_columns",
            ),
            notes=str(item.get("notes", "")).strip(),
        )

    # Detect selector collisions using all exact labels and representative prefixes.
    exact_owners: Dict[str, str] = {}
    for group in groups.values():
        for column in group.selector.exact:
            if column in exact_owners:
                raise FeatureFrameworkError(
                    f"exact selector {column!r} is shared by {exact_owners[column]!r} and {group.id!r}"
                )
            exact_owners[column] = group.id
    prefix_pairs = []
    for group in groups.values():
        for prefix in group.selector.prefixes:
            prefix_pairs.append((prefix, group.id))
    for i, (prefix_a, group_a) in enumerate(prefix_pairs):
        for prefix_b, group_b in prefix_pairs[i + 1:]:
            if group_a != group_b and (prefix_a.startswith(prefix_b) or prefix_b.startswith(prefix_a)):
                raise FeatureFrameworkError(
                    f"prefix selectors overlap: {prefix_a!r} ({group_a}) vs {prefix_b!r} ({group_b})"
                )

    return FeatureRegistry(
        id=registry_id,
        receptor=receptor,
        default_repertoire_source=default_source,
        repertoire_sources=sources,
        feature_groups=groups,
    )


def load_feature_plan(path: Union[Path, str]) -> FeaturePlan:
    path = Path(path).expanduser().resolve()
    raw = _load_yaml(path, "feature plan")
    if str(raw.get("schema_version")) != SCHEMA_VERSION:
        raise FeatureFrameworkError(f"feature plan schema_version must be {SCHEMA_VERSION!r}")
    plan = _mapping(raw.get("plan"), "plan")
    return FeaturePlan(
        id=_nonempty_string(plan.get("id"), "plan.id"),
        receptor=_nonempty_string(plan.get("receptor"), "plan.receptor"),
        repertoire_source=_nonempty_string(
            plan.get("repertoire_source"), "plan.repertoire_source"
        ),
        feature_groups=_string_tuple(
            plan.get("feature_groups"), "plan.feature_groups", allow_empty=False
        ),
        allow_legacy=_bool(plan.get("allow_legacy", False), "plan.allow_legacy"),
        allow_planned=_bool(plan.get("allow_planned", False), "plan.allow_planned"),
    )


def resolve_feature_plan(
    registry_path: Union[Path, str],
    plan_path: Union[Path, str],
) -> ResolvedFeaturePlan:
    registry = load_feature_registry(registry_path)
    plan = load_feature_plan(plan_path)
    return registry.resolve_plan(plan)
