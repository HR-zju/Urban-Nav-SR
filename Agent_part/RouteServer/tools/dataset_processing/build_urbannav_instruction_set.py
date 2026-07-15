#!/usr/bin/env python3
r"""Build the full UrbanNav benchmark instruction set from taxonomy and quotas.

This script compiles the paper-level benchmark definition into auditable
artifacts. Taxonomy templates define canonical semantics; the default output
uses case-level rendering so the final JSONL does not simply cycle through a
small fixed template pool. The compilation objective is:

\[
\sum_{c \in C} N_c = N,\qquad
N_s = \operatorname{round}\left(N_c \frac{w_s}{\sum_{j \in S_c}w_j}\right)
\]

where \(N_c\) is the category quota, \(w_s\) is the taxonomy quota weight, and
\(N_s\) is the planned number of cases for one subcategory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
from typing import Any, Iterable


ROUTE_SERVER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY = ROUTE_SERVER_ROOT / "config" / "yelp_demand_taxonomy_mined_v2.json"
DEFAULT_DISTRIBUTION = ROUTE_SERVER_ROOT / "config" / "urbannav_10k_distribution_plan.json"
DEFAULT_PROVIDER_MAP = ROUTE_SERVER_ROOT / "config" / "urbannav_target_type_provider_map_mined_v2.json"
DEFAULT_REGIONS = ROUTE_SERVER_ROOT / "config" / "urbannav_global_gt_collection_full_en.json"
DEFAULT_OUTPUT_ROOT = ROUTE_SERVER_ROOT / "data" / "benchmark" / "urbannav_instruction_set_full_en_v2"
BUILDER_VERSION = "urbannav_instruction_set_builder_v2"
NEAREST_ACCEPTABLE_ROUTE_RATIO = 1.15
NEARBY_ACCEPTABLE_ROUTE_RATIO = 1.75
NEARBY_ACCEPTABLE_ROUTE_LIMIT_M = 1500.0
IMPLICIT_ACCEPTABLE_ROUTE_RATIO = 1.75
IMPLICIT_ACCEPTABLE_ROUTE_LIMIT_M = 1500.0

IMPLICIT_ROUTE_RATIO_OVERRIDES = {
    "restroom": 1.5,
    "not_feeling_well": 1.5,
    "charge_phone": 1.75,
    "hungry": 1.75,
    "thirsty": 1.75,
    "buy_daily_goods": 1.75,
    "buy_fruit": 1.75,
    "quick_rest": 2.0,
    "work_with_wifi": 2.0,
    "family_kids": 2.0,
    "parking_lot": 2.0,
    "cheap_budget": 2.0,
}


@dataclass(frozen=True)
class CompiledTask:
    r"""One taxonomy task compiled for benchmark planning.

    Args:
        task_id: Stable task id.
        category: Normalized benchmark category.
        source_category: Original taxonomy category.
        subcategory: Fine-grained task label.
        templates: Instruction templates in the selected language.
        target_types: Unordered target types for single-destination tasks.
        ordered_target_types: Ordered target groups for multi-stop tasks.
        ordered_goal_labels: Canonical, distinct user-facing labels for ordered
            multi-stop goals.
        query_terms: Provider-neutral text queries derived from target types.
        osm_tags: OSM tag constraints derived from target types.
        tomtom_category_set: TomTom category ids derived from target types.
        semantic_tags: Optional semantic preference tags.
        route_update_action: Optional route-update action type.
        quota_weight: Taxonomy quota weight.
        planned_count: Planned number of cases.

    The task definition represents a distribution cell:

    \[
    \tau=(c,s,T,Q)
    \]

    where \(c\) is the task family, \(s\) is the subcategory, \(T\) is the
    target-type set, and \(Q\) is the instruction-template set.
    """

    task_id: str
    category: str
    source_category: str
    subcategory: str
    templates: list[str]
    target_types: list[str]
    ordered_target_types: list[list[str]]
    ordered_goal_labels: list[str]
    query_terms: list[str]
    osm_tags: list[str]
    tomtom_category_set: list[str]
    semantic_tags: list[str]
    route_update_action: str | None
    quota_weight: float
    planned_count: int

    def to_task_record(self) -> dict[str, Any]:
        """Serialize a collector-compatible task record.

        Returns:
            JSON-ready task dictionary.
        """

        acceptance = _acceptance_settings(
            category=self.category,
            subcategory=self.subcategory,
            instruction="",
        )
        return {
            "task_id": self.task_id,
            "category": self.category,
            "source_category": self.source_category,
            "subcategory": self.subcategory,
            "templates": list(self.templates),
            "target_types": list(self.target_types),
            "ordered_target_types": [list(group) for group in self.ordered_target_types],
            "ordered_goal_labels": list(self.ordered_goal_labels),
            "queries_by_locale": {"default": list(self.query_terms), "en": list(self.query_terms)},
            "osm_tags": list(self.osm_tags),
            "provider_category_map": {
                "osm": list(self.osm_tags),
                "tomtom": list(self.tomtom_category_set),
            },
            "semantic_tags": list(self.semantic_tags),
            "route_update_action": self.route_update_action,
            "quota_weight": self.quota_weight,
            "planned_count": self.planned_count,
            "allow_alternatives": acceptance["allow_alternatives"],
            "acceptance_policy": acceptance["acceptance_policy"],
            "acceptable_route_ratio": acceptance["acceptable_route_ratio"],
            "acceptable_route_limit_m": acceptance["acceptable_route_limit_m"],
        }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY, help="Demand taxonomy JSON.")
    parser.add_argument("--distribution", type=Path, default=DEFAULT_DISTRIBUTION, help="Benchmark distribution plan JSON.")
    parser.add_argument("--provider-map", type=Path, default=DEFAULT_PROVIDER_MAP, help="Target-type provider ontology JSON.")
    parser.add_argument("--regions-config", type=Path, default=DEFAULT_REGIONS, help="Region config used to populate collection config.")
    parser.add_argument("--quality-policy", type=Path, default=None, help="Optional v2 benchmark quality policy JSON.")
    parser.add_argument(
        "--city-plan-mode",
        choices=["available_equal", "exact"],
        default="available_equal",
        help="Use legacy equal allocation over available regions or require the explicit city distribution plan.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output directory.")
    parser.add_argument("--language", choices=["en", "zh"], default="en", help="Instruction template language.")
    parser.add_argument(
        "--instruction-mode",
        choices=["case_variant", "template_cycle"],
        default="case_variant",
        help="Use case-level instruction rendering or raw taxonomy template cycling.",
    )
    parser.add_argument("--seed", type=int, default=20260629, help="Deterministic assignment seed.")
    parser.add_argument("--include-semantic-analysis", action="store_true", help="Include semantic_preference as an analysis split when quotas exist.")
    parser.add_argument("--allow-missing-target-types", action="store_true", help="Do not fail when a target type has no provider mapping.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def main() -> None:
    """Build and write instruction-set artifacts.

    Returns:
        None.
    """

    args = parse_args()
    artifacts = build_instruction_set(
        taxonomy_path=args.taxonomy,
        distribution_path=args.distribution,
        provider_map_path=args.provider_map,
        regions_config_path=args.regions_config,
        quality_policy_path=args.quality_policy,
        city_plan_mode=args.city_plan_mode,
        output_root=args.output_root,
        language=args.language,
        instruction_mode=args.instruction_mode,
        seed=args.seed,
        include_semantic_analysis=args.include_semantic_analysis,
        allow_missing_target_types=args.allow_missing_target_types,
        overwrite=args.overwrite,
    )
    print(json.dumps(artifacts["manifest"], ensure_ascii=False, indent=2))


def build_instruction_set(
    *,
    taxonomy_path: Path,
    distribution_path: Path,
    provider_map_path: Path,
    regions_config_path: Path | None,
    quality_policy_path: Path | None = None,
    city_plan_mode: str = "available_equal",
    output_root: Path,
    language: str,
    instruction_mode: str,
    seed: int,
    include_semantic_analysis: bool,
    allow_missing_target_types: bool,
    overwrite: bool,
) -> dict[str, Any]:
    r"""Compile taxonomy, city quotas, and task quotas into benchmark artifacts.

    Args:
        taxonomy_path: Path to the demand taxonomy.
        distribution_path: Path to the global distribution plan.
        provider_map_path: Path to the target-type provider map.
        regions_config_path: Optional region config copied into collection config.
        quality_policy_path: Optional v2 quality policy with multi-stop goal
            labels and target-group overrides.
        city_plan_mode: ``"exact"`` requires every configured city quota to
            match an available collection city. ``"available_equal"`` keeps
            the legacy behavior.
        output_root: Output artifact directory.
        language: Template language.
        instruction_mode: Instruction rendering mode.
        seed: Deterministic assignment seed.
        include_semantic_analysis: Whether to include semantic preference splits
            when a quota is present.
        allow_missing_target_types: If true, missing provider ontology entries
            are emitted as warnings instead of errors.
        overwrite: Whether to overwrite existing artifacts.

    Returns:
        Dictionary with the manifest and output paths.
    """

    output_root.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_root)
    if not overwrite and any(path.exists() for path in paths.values()):
        raise RuntimeError(f"{output_root} already contains instruction-set artifacts; use --overwrite")

    taxonomy = _load_json(taxonomy_path)
    distribution = _load_json(distribution_path)
    provider_map = _load_json(provider_map_path)
    regions_config = _load_json(regions_config_path) if regions_config_path and regions_config_path.exists() else {}
    quality_policy = _load_json(quality_policy_path) if quality_policy_path and quality_policy_path.exists() else {}

    task_quotas = _task_quotas(distribution, include_semantic_analysis=include_semantic_analysis)
    route_update_action_quotas = _route_update_action_quotas(distribution)
    city_slots = _city_slots(distribution, regions_config=regions_config, seed=seed, mode=city_plan_mode)
    compiled = compile_tasks(
        taxonomy=taxonomy,
        provider_map=provider_map,
        task_quotas=task_quotas,
        route_update_action_quotas=route_update_action_quotas,
        language=language,
        allow_missing_target_types=allow_missing_target_types,
    )
    compiled = _apply_quality_policy(compiled, quality_policy=quality_policy)
    instruction_rows = build_instruction_rows(
        compiled_tasks=compiled,
        city_slots=city_slots,
        case_id_prefix=str(distribution.get("case_id_prefix") or "urbannav"),
        seed=seed,
        instruction_mode=instruction_mode,
    )
    collection_config = _collection_config(regions_config=regions_config, compiled_tasks=compiled, distribution=distribution)
    manifest = _manifest(
        taxonomy_path=taxonomy_path,
        distribution_path=distribution_path,
        provider_map_path=provider_map_path,
        regions_config_path=regions_config_path,
        quality_policy_path=quality_policy_path,
        city_plan_mode=city_plan_mode,
        output_root=output_root,
        language=language,
        instruction_mode=instruction_mode,
        compiled_tasks=compiled,
        instruction_rows=instruction_rows,
        distribution=distribution,
        collection_config=collection_config,
    )

    _write_jsonl(paths["instruction_set"], instruction_rows)
    _write_json(paths["task_catalog"], [task.to_task_record() for task in compiled])
    _write_json(paths["collection_config"], collection_config)
    _write_json(paths["manifest"], manifest)
    return {"manifest": manifest, "paths": {key: str(path) for key, path in paths.items()}}


def compile_tasks(
    *,
    taxonomy: dict[str, Any],
    provider_map: dict[str, Any],
    task_quotas: dict[str, int],
    route_update_action_quotas: dict[str, int] | None = None,
    language: str,
    allow_missing_target_types: bool,
) -> list[CompiledTask]:
    """Compile taxonomy specs into quota-bearing task records.

    Args:
        taxonomy: Raw taxonomy payload.
        provider_map: Target-type provider ontology.
        task_quotas: Category quota map.
        route_update_action_quotas: Optional exact route-update action quotas.
        language: Template language.
        allow_missing_target_types: Whether missing provider mappings are
            warnings or errors.

    Returns:
        Compiled tasks with nonzero planned counts.
    """

    specs_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in taxonomy.get("specs", []):
        category = _normalize_category(str(spec.get("category") or ""))
        if category in task_quotas:
            specs_by_category[category].append(dict(spec))

    compiled: list[CompiledTask] = []
    for category, quota in task_quotas.items():
        specs = specs_by_category.get(category, [])
        if not specs:
            raise ValueError(f"No taxonomy specs found for planned category: {category}")
        if category == "route_update_add_via" and route_update_action_quotas:
            planned_specs = _allocate_route_update_specs(
                specs,
                action_quotas=route_update_action_quotas,
                category_quota=quota,
            )
        else:
            planned_specs = list(zip(specs, _allocate_by_weight(specs, total=quota)))
        for spec, planned_count in planned_specs:
            if planned_count <= 0:
                continue
            compiled.append(
                _compile_one_task(
                    spec=spec,
                    planned_count=planned_count,
                    provider_map=provider_map,
                    taxonomy=taxonomy,
                    language=language,
                    allow_missing_target_types=allow_missing_target_types,
                )
            )
    return compiled


def _allocate_route_update_specs(
    specs: list[dict[str, Any]],
    *,
    action_quotas: dict[str, int],
    category_quota: int,
) -> list[tuple[dict[str, Any], int]]:
    """Allocate route-update taxonomy specs under exact action quotas.

    Args:
        specs: Route-update taxonomy specs.
        action_quotas: Desired count by update action.
        category_quota: Total route-update category quota.

    Returns:
        ``(spec, count)`` pairs in taxonomy order.

    Raises:
        ValueError: If action quotas do not match the category total or an
            action has no taxonomy specs.
    """

    if sum(action_quotas.values()) != category_quota:
        raise ValueError(
            f"Route-update action quotas sum to {sum(action_quotas.values())}, expected {category_quota}"
        )
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        by_action[str(spec.get("route_update_action") or "add_via")].append(spec)
    counts_by_id: dict[int, int] = {}
    for action, action_quota in action_quotas.items():
        action_specs = by_action.get(action) or []
        if action_quota > 0 and not action_specs:
            raise ValueError(f"No route-update taxonomy specs for action={action}")
        for spec, count in zip(action_specs, _allocate_by_weight(action_specs, total=action_quota)):
            counts_by_id[id(spec)] = count
    return [(spec, counts_by_id.get(id(spec), 0)) for spec in specs]


def _apply_quality_policy(
    tasks: list[CompiledTask],
    *,
    quality_policy: dict[str, Any],
) -> list[CompiledTask]:
    """Apply explicit v2 multi-stop semantics to compiled tasks.

    Args:
        tasks: Compiled taxonomy tasks.
        quality_policy: Optional quality policy payload.

    Returns:
        Tasks with configured target-group overrides and canonical goal labels.

    Raises:
        ValueError: If a strict multi-stop policy contains repeated labels,
            overlapping target groups, or mismatched group/label counts.
    """

    multi_policy = quality_policy.get("multi_stop") if isinstance(quality_policy.get("multi_stop"), dict) else {}
    subcategory_policy = multi_policy.get("subcategories") if isinstance(multi_policy.get("subcategories"), dict) else {}
    require_disjoint = bool(multi_policy.get("require_disjoint_target_types", False))
    output: list[CompiledTask] = []
    for task in tasks:
        if task.category != "multi_stop_route":
            output.append(task)
            continue
        override = subcategory_policy.get(task.subcategory) if isinstance(subcategory_policy.get(task.subcategory), dict) else {}
        groups = [
            [str(item) for item in group]
            for group in (override.get("ordered_target_types") or task.ordered_target_types)
        ]
        labels = [
            str(item).strip()
            for item in (override.get("ordered_goal_labels") or task.ordered_goal_labels)
            if str(item).strip()
        ]
        if not labels:
            labels = [group[0].replace("_", " ") for group in groups if group]
        updated = replace(task, ordered_target_types=groups, ordered_goal_labels=labels)
        _validate_multi_stop_task(updated, require_disjoint=require_disjoint)
        output.append(updated)
    return output


def _validate_multi_stop_task(task: CompiledTask, *, require_disjoint: bool) -> None:
    """Validate one structured multi-stop task definition.

    Args:
        task: Multi-stop task.
        require_disjoint: Whether target-type groups must be pairwise disjoint.

    Returns:
        None.

    Raises:
        ValueError: If the task violates ordered-goal invariants.
    """

    groups = [{str(item).strip().lower() for item in group if str(item).strip()} for group in task.ordered_target_types]
    labels = [" ".join(label.lower().split()) for label in task.ordered_goal_labels]
    if len(groups) < 2 or any(not group for group in groups):
        raise ValueError(f"Multi-stop task requires at least two non-empty target groups: {task.task_id}")
    if len(labels) != len(groups):
        raise ValueError(f"Multi-stop labels/groups mismatch for {task.task_id}: labels={len(labels)} groups={len(groups)}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Repeated multi-stop goal labels for {task.task_id}: {labels}")
    if require_disjoint:
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                overlap = groups[left].intersection(groups[right])
                if overlap:
                    raise ValueError(f"Overlapping target groups for {task.task_id}: {sorted(overlap)}")


def build_instruction_rows(
    *,
    compiled_tasks: list[CompiledTask],
    city_slots: list[dict[str, Any]],
    case_id_prefix: str = "urbannav",
    seed: int,
    instruction_mode: str,
) -> list[dict[str, Any]]:
    """Build one planned instruction row per case.

    Args:
        compiled_tasks: Compiled task definitions.
        city_slots: City quota slots.
        case_id_prefix: Stable namespace for generated case ids.
        seed: Deterministic assignment seed.
        instruction_mode: Instruction rendering mode.

    Returns:
        Planned instruction rows.
    """

    rng = random.Random(seed)
    tasks = list(compiled_tasks)
    rows: list[dict[str, Any]] = []
    category_offsets: Counter[str] = Counter()
    task_offsets: Counter[str] = Counter()
    seen_instructions: set[str] = set()
    city_slots = list(city_slots)
    rng.shuffle(city_slots)

    for task in tasks:
        for _ in range(task.planned_count):
            city = city_slots[len(rows) % len(city_slots)] if city_slots else {}
            template_index = task_offsets[task.task_id] % max(1, len(task.templates))
            canonical_instruction = task.templates[template_index]
            if instruction_mode == "case_variant":
                instruction = _case_variant_instruction(
                    task=task,
                    canonical_instruction=canonical_instruction,
                    city=city,
                    variant_index=task_offsets[task.task_id],
                    global_index=len(rows),
                    seen=seen_instructions,
                )
            else:
                instruction = canonical_instruction
            seen_instructions.add(instruction)
            task_offsets[task.task_id] += 1
            category_offsets[task.category] += 1
            acceptance = _acceptance_settings(
                category=task.category,
                subcategory=task.subcategory,
                instruction=instruction,
            )
            rows.append(
                {
                    "instruction_id": f"{case_id_prefix}-{task.category}-{category_offsets[task.category]:05d}",
                    "builder_version": BUILDER_VERSION,
                    "planned_city": dict(city),
                    "category": task.category,
                    "source_category": task.source_category,
                    "subcategory": task.subcategory,
                    "task_id": task.task_id,
                    "instruction": instruction,
                    "canonical_instruction": canonical_instruction,
                    "instruction_generation_mode": instruction_mode,
                    "template_index": template_index,
                    "target_types": list(task.target_types),
                    "ordered_target_types": [list(group) for group in task.ordered_target_types],
                    "ordered_goal_labels": list(task.ordered_goal_labels),
                    "query_terms": list(task.query_terms),
                    "osm_tags": list(task.osm_tags),
                    "provider_category_map": {
                        "osm": list(task.osm_tags),
                        "tomtom": list(task.tomtom_category_set),
                    },
                    "acceptance_policy": acceptance["acceptance_policy"],
                    "allow_alternatives": acceptance["allow_alternatives"],
                    "acceptable_route_ratio": acceptance["acceptable_route_ratio"],
                    "acceptable_route_limit_m": acceptance["acceptable_route_limit_m"],
                    "route_update_action": task.route_update_action,
                }
            )

    rng.shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["global_index"] = index
    return rows


def _compile_one_task(
    *,
    spec: dict[str, Any],
    planned_count: int,
    provider_map: dict[str, Any],
    taxonomy: dict[str, Any],
    language: str,
    allow_missing_target_types: bool,
) -> CompiledTask:
    """Compile one taxonomy spec.

    Args:
        spec: Taxonomy spec.
        planned_count: Planned case count.
        provider_map: Provider ontology.
        taxonomy: Raw taxonomy.
        language: Template language.
        allow_missing_target_types: Whether to allow missing target maps.

    Returns:
        Compiled task.
    """

    category = _normalize_category(str(spec.get("category") or ""))
    source_category = str(spec.get("category") or category)
    subcategory = str(spec.get("subcategory") or spec.get("demand_group") or "unknown")
    task_id = f"{category}:{subcategory}"
    target_types = [str(item) for item in spec.get("target_types") or []]
    ordered_target_types = [[str(item) for item in group] for group in spec.get("ordered_target_types") or []]
    ordered_goal_labels = [str(item).strip() for item in spec.get("ordered_goal_labels") or [] if str(item).strip()]
    route_update_action = _str_or_none(spec.get("route_update_action"))
    if category == "route_update_add_via" and route_update_action is None:
        route_update_action = "add_via"
    templates = _templates_for_spec(spec=spec, taxonomy=taxonomy, language=language)
    target_groups = ordered_target_types or ([target_types] if target_types else [])
    ontology = _provider_ontology_for_groups(
        target_groups=target_groups,
        provider_map=provider_map,
        allow_missing_target_types=allow_missing_target_types,
    )
    return CompiledTask(
        task_id=task_id,
        category=category,
        source_category=source_category,
        subcategory=subcategory,
        templates=templates,
        target_types=target_types,
        ordered_target_types=ordered_target_types,
        ordered_goal_labels=ordered_goal_labels,
        query_terms=ontology["query_terms"],
        osm_tags=ontology["osm_tags"],
        tomtom_category_set=ontology["tomtom_category_set"],
        semantic_tags=[str(item) for item in spec.get("semantic_tags") or []],
        route_update_action=route_update_action,
        quota_weight=float(spec.get("quota_weight") or 1.0),
        planned_count=planned_count,
    )


def _provider_ontology_for_groups(
    *,
    target_groups: list[list[str]],
    provider_map: dict[str, Any],
    allow_missing_target_types: bool,
) -> dict[str, list[str]]:
    """Resolve target types into provider query terms and tags.

    Args:
        target_groups: Target-type groups.
        provider_map: Provider ontology.
        allow_missing_target_types: Whether missing entries are allowed.

    Returns:
        Dictionary with query terms, OSM tags, and TomTom categories.
    """

    target_type_map = provider_map.get("target_types") or {}
    query_terms: list[str] = []
    osm_tags: list[str] = []
    tomtom_category_set: list[str] = []
    missing: list[str] = []
    for target_type in _flatten(target_groups):
        entry = target_type_map.get(target_type)
        if not entry:
            missing.append(target_type)
            query_terms.append(target_type.replace("_", " "))
            continue
        query_terms.extend(str(item) for item in entry.get("query_terms") or [])
        osm_tags.extend(str(item) for item in entry.get("osm_tags") or [])
        tomtom_category_set.extend(str(item) for item in entry.get("tomtom_category_set") or [])
    if missing and not allow_missing_target_types:
        raise ValueError(f"Missing provider ontology for target types: {sorted(set(missing))}")
    return {
        "query_terms": _dedupe(query_terms),
        "osm_tags": _dedupe(osm_tags),
        "tomtom_category_set": _dedupe(tomtom_category_set),
    }


def _templates_for_spec(*, spec: dict[str, Any], taxonomy: dict[str, Any], language: str) -> list[str]:
    """Collect templates for one spec from taxonomy-owned fields.

    Args:
        spec: Taxonomy spec.
        taxonomy: Raw taxonomy.
        language: Template language.

    Returns:
        Non-empty template list.
    """

    templates = [str(item).strip() for item in spec.get(f"templates_{language}") or [] if str(item).strip()]
    extra_sources = []
    category = str(spec.get("category") or "")
    demand_group = str(spec.get("demand_group") or "")
    subcategory = str(spec.get("subcategory") or "")
    if category == "multi_stop_route":
        extra_sources.append(taxonomy.get("multi_stop_extra") or {})
    elif category == "route_update_add_via":
        extra_sources.append(taxonomy.get("route_update_extra") or {})
    extra_sources.append(taxonomy.get("extra_templates") or {})
    for source in extra_sources:
        for key in {demand_group, subcategory}:
            payload = source.get(key) if key else None
            if isinstance(payload, dict):
                templates.extend(str(item).strip() for item in payload.get(language) or [] if str(item).strip())
            elif isinstance(payload, list):
                templates.extend(str(item).strip() for item in payload if str(item).strip())
    templates = _dedupe(templates)
    if not templates:
        raise ValueError(f"No templates found for {category}:{subcategory} language={language}")
    return templates


def _task_quotas(distribution: dict[str, Any], *, include_semantic_analysis: bool) -> dict[str, int]:
    """Read task quotas from the distribution plan.

    Args:
        distribution: Raw distribution plan.
        include_semantic_analysis: Whether semantic preference quotas are kept.

    Returns:
        Normalized category quotas.
    """

    quotas: dict[str, int] = {}
    for item in distribution.get("task_distribution", []):
        category = _normalize_category(str(item.get("category") or ""))
        if category == "semantic_preference" and not include_semantic_analysis:
            continue
        count = int(item.get("count") or 0)
        if count > 0:
            quotas[category] = quotas.get(category, 0) + count
    if not quotas:
        raise ValueError("Distribution plan has no nonzero task quotas")
    return quotas


def _route_update_action_quotas(distribution: dict[str, Any]) -> dict[str, int]:
    """Read optional route-update action quotas.

    Args:
        distribution: Raw distribution plan.

    Returns:
        Count by ``replace_destination``, ``add_via``, or
        ``insert_before_final``.
    """

    quotas: dict[str, int] = {}
    for item in distribution.get("route_update_action_distribution", []):
        action = str(item.get("action") or "").strip()
        count = int(item.get("count") or 0)
        if action and count > 0:
            quotas[action] = quotas.get(action, 0) + count
    return quotas


def _city_slots(
    distribution: dict[str, Any],
    *,
    regions_config: dict[str, Any],
    seed: int,
    mode: str = "available_equal",
) -> list[dict[str, Any]]:
    """Expand city quotas into one slot per planned case.

    Args:
        distribution: Raw distribution plan.
        regions_config: Region config that defines actually collectable cities.
        seed: Deterministic assignment seed.
        mode: City quota policy. ``"exact"`` honors the distribution plan and
            rejects unavailable cities.

    Returns:
        City slot records.
    """

    available = _collectable_city_records(regions_config)
    if mode == "exact":
        available_by_key = {_normalize_city_name(item.get("city")): item for item in available}
        slots: list[dict[str, Any]] = []
        missing: list[str] = []
        for item in distribution.get("city_distribution", []):
            count = int(item.get("count") or 0)
            key = _normalize_city_name(item.get("city"))
            city = available_by_key.get(key)
            if count > 0 and city is None:
                missing.append(str(item.get("city") or key))
                continue
            for _ in range(max(0, count)):
                slots.append(dict(city or item))
        if missing:
            raise ValueError(f"Distribution cities are unavailable in regions config: {sorted(missing)}")
        target = int(distribution.get("target_cases") or len(slots))
        if len(slots) != target:
            raise ValueError(f"Exact city quotas sum to {len(slots)}, expected target_cases={target}")
        random.Random(seed).shuffle(slots)
        return slots
    if available:
        target = int(distribution.get("target_cases") or 0)
        if target <= 0:
            target = sum(int(item.get("count") or 0) for item in distribution.get("city_distribution", []))
        base = target // len(available)
        remainder = target - base * len(available)
        slots = []
        for index, city in enumerate(available):
            count = base + (1 if index < remainder else 0)
            for _ in range(max(0, count)):
                slots.append(dict(city))
        random.Random(seed).shuffle(slots)
        return slots

    slots: list[dict[str, Any]] = []
    for item in distribution.get("city_distribution", []):
        count = int(item.get("count") or 0)
        for _ in range(max(0, count)):
            slots.append({"country": item.get("country"), "city": item.get("city")})
    random.Random(seed).shuffle(slots)
    return slots


def _normalize_city_name(value: Any) -> str:
    """Normalize a city name for distribution/region matching.

    Args:
        value: Raw city value.

    Returns:
        Lowercase underscore-delimited city key.
    """

    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _collectable_city_records(regions_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return unique cities that have collection regions.

    Args:
        regions_config: Region config.

    Returns:
        Stable list of collectable city records.
    """

    records: dict[str, dict[str, Any]] = {}
    for region in regions_config.get("regions") or []:
        city = str(region.get("city") or "").strip()
        if not city:
            continue
        key = city.lower().replace(" ", "_").replace("-", "_")
        records[key] = {"country": region.get("country"), "city": city}
    return [records[key] for key in sorted(records)]


def _allocate_by_weight(specs: list[dict[str, Any]], *, total: int) -> list[int]:
    """Allocate an integer quota by taxonomy weights.

    Args:
        specs: Taxonomy specs.
        total: Total quota.

    Returns:
        Integer counts with exact sum equal to total.
    """

    weights = [max(float(spec.get("quota_weight") or 1.0), 0.0) for spec in specs]
    weight_sum = sum(weights) or float(len(specs))
    raw = [total * weight / weight_sum for weight in weights]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(specs)), key=lambda idx: (raw[idx] - counts[idx], weights[idx]), reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def _collection_config(
    *,
    regions_config: dict[str, Any],
    compiled_tasks: list[CompiledTask],
    distribution: dict[str, Any],
) -> dict[str, Any]:
    """Build a collector config from compiled single-stop tasks.

    Args:
        regions_config: Existing region/provider config.
        compiled_tasks: Compiled task definitions.
        distribution: Raw distribution plan.

    Returns:
        Collector-compatible configuration.
    """

    tasks = [
        task.to_task_record()
        for task in compiled_tasks
        if task.category in {"poi_navigation", "implicit_need_navigation", "semantic_preference"}
    ]
    config = dict(regions_config)
    config["version"] = "urbannav_full_instruction_collection_config_v1"
    config["source_distribution"] = distribution.get("version")
    config["instruction_set_builder_version"] = BUILDER_VERSION
    config["acceptance"] = {
        "nearest_route_ratio": NEAREST_ACCEPTABLE_ROUTE_RATIO,
        "nearby_route_ratio": NEARBY_ACCEPTABLE_ROUTE_RATIO,
        "nearby_route_limit_m": NEARBY_ACCEPTABLE_ROUTE_LIMIT_M,
        "implicit_route_ratio": IMPLICIT_ACCEPTABLE_ROUTE_RATIO,
        "implicit_route_limit_m": IMPLICIT_ACCEPTABLE_ROUTE_LIMIT_M,
        "implicit_route_ratio_overrides": dict(sorted(IMPLICIT_ROUTE_RATIO_OVERRIDES.items())),
    }
    config["tasks"] = tasks
    config["planned_single_stop_cases"] = sum(int(task.get("planned_count") or 0) for task in tasks)
    config["planned_complex_cases"] = sum(
        task.planned_count for task in compiled_tasks if task.category in {"multi_stop_route", "route_update_add_via"}
    )
    return config


def _case_variant_instruction(
    *,
    task: CompiledTask,
    canonical_instruction: str,
    city: dict[str, Any],
    variant_index: int,
    global_index: int,
    seen: set[str],
) -> str:
    """Render one case-level instruction with deterministic language variation.

    Args:
        task: Compiled task.
        canonical_instruction: Taxonomy-owned canonical instruction.
        city: Planned city slot.
        variant_index: Index within the task.
        global_index: Index within the full instruction set.
        seen: Previously emitted instruction strings.

    Returns:
        A natural-language instruction.
    """

    city_name = str(city.get("city") or "this area")
    for attempt in range(96):
        idx = global_index * 31 + variant_index * 17 + attempt * 13
        if task.category == "poi_navigation":
            text = _render_poi_instruction(task, idx=idx, city_name=city_name)
        elif task.category == "implicit_need_navigation":
            text = _render_implicit_instruction(task, idx=idx, city_name=city_name)
        elif task.category == "multi_stop_route":
            text = _render_multi_stop_instruction(task, idx=idx, city_name=city_name)
        elif task.category == "route_update_add_via":
            text = _render_route_update_instruction(task, idx=idx, city_name=city_name)
        else:
            text = _compose_with_context(canonical_instruction, idx=idx, city_name=city_name)
        text = _normalize_sentence(text)
        if text and text not in seen:
            return text
    return _normalize_sentence(_compose_with_context(canonical_instruction, idx=global_index + variant_index, city_name=city_name))


def _render_poi_instruction(task: CompiledTask, *, idx: int, city_name: str) -> str:
    """Render one explicit POI instruction.

    Args:
        task: Compiled task.
        idx: Deterministic variant index.
        city_name: Planned city name.

    Returns:
        Instruction string.
    """

    target = _target_phrase(task, idx=idx)
    article_target = _with_article(target)
    near = _pick(_NEAR_PHRASES, idx)
    opener = _pick(_OPENERS, idx // 3)
    purpose = _pick(_POI_PURPOSES, idx // 5)
    modifier = _pick(_REQUEST_MODIFIERS, idx // 11)
    patterns = [
        "{opener} find {article_target} {near}. {modifier}",
        "{opener} route me to {article_target} {near}. {modifier}",
        "I need {article_target} {near}; {purpose}. {modifier}",
        "Where can I find {article_target} {near}? {modifier}",
        "From my current position, take me to {article_target} {near}. {modifier}",
        "Look for {article_target} that is practical to reach from here. {modifier}",
        "Around here, find {article_target} for me. {modifier}",
        "While I am in {city_name}, find {article_target} close to my starting point. {modifier}",
        "Help me get to {article_target} without going far from here. {modifier}",
        "Find a walkable option for {target} from this location. {modifier}",
        "Find the closest {target} by walking distance from here. {modifier}",
        "Take me to the nearest routeable {target} on foot. {modifier}",
    ]
    return _pick(patterns, idx).format(
        opener=opener,
        article_target=article_target,
        target=target,
        near=near,
        purpose=purpose,
        city_name=city_name,
        modifier=modifier,
    )


def _render_implicit_instruction(task: CompiledTask, *, idx: int, city_name: str) -> str:
    """Render one implicit-need instruction.

    Args:
        task: Compiled task.
        idx: Deterministic variant index.
        city_name: Planned city name.

    Returns:
        Instruction string.
    """

    need = _need_phrase(task, idx=idx)
    target_hint = _target_phrase(task, idx=idx)
    near = _pick(_NEAR_PHRASES, idx // 2)
    opener = _pick(_OPENERS, idx // 7)
    modifier = _pick(_REQUEST_MODIFIERS, idx // 13)
    patterns = [
        "{opener} {need}; find a suitable place {near}. {modifier}",
        "{need}. Show me a practical nearby option. {modifier}",
        "I need help because {need_lower}; choose a place I can reach on foot. {modifier}",
        "From here, find somewhere for this need: {need_lower}. {modifier}",
        "I am looking for a place that can handle this: {need_lower}. {modifier}",
        "Help me with this nearby need: {need_lower}. {modifier}",
        "Around my current location, find a good option for {need_short}. {modifier}",
        "In {city_name}, from this starting point, find somewhere for {need_short}. {modifier}",
        "{opener} look for a {target_hint} or similar place that fits this need. {modifier}",
        "Find a walkable place for {need_short}, not just any random POI. {modifier}",
    ]
    return _pick(patterns, idx).format(
        opener=opener,
        need=need,
        need_lower=_lower_first(need),
        need_short=_need_short(task, idx=idx),
        target_hint=target_hint,
        near=near,
        city_name=city_name,
        modifier=modifier,
    )


def _render_multi_stop_instruction(task: CompiledTask, *, idx: int, city_name: str) -> str:
    """Render one ordered multi-stop instruction.

    Args:
        task: Compiled task.
        idx: Deterministic variant index.
        city_name: Planned city name.

    Returns:
        Instruction string.
    """

    stops = list(task.ordered_goal_labels) or [
        _group_phrase(group, idx=idx + offset)
        for offset, group in enumerate(task.ordered_target_types)
    ]
    sequence = _sequence_phrase(stops)
    compact = " -> ".join(stops)
    modifier = _pick(_MULTI_STOP_MODIFIERS, idx // 11)
    patterns = [
        "Plan a route where I go to {sequence}. {modifier}",
        "I need these stops in order: {compact}. {modifier}",
        "From my current location, arrange {sequence}. {modifier}",
        "Build a walkable route with this order: {compact}. {modifier}",
        "In {city_name}, starting here, take me to {sequence}. {modifier}",
        "Do not skip the intermediate stop: route me through {sequence}. {modifier}",
        "I want an ordered trip: {compact}. {modifier}",
        "Make the route respect this sequence: {compact}. {modifier}",
    ]
    return _pick(patterns, idx).format(sequence=sequence, compact=compact, city_name=city_name, modifier=modifier)


def _render_route_update_instruction(task: CompiledTask, *, idx: int, city_name: str) -> str:
    """Render one route-update instruction.

    Args:
        task: Compiled task.
        idx: Deterministic variant index.
        city_name: Planned city name.

    Returns:
        Instruction string.
    """

    target = _target_phrase(task, idx=idx)
    article_target = _with_article(target)
    action = task.route_update_action or "add_via"
    modifier = _pick(_ROUTE_UPDATE_MODIFIERS, idx // 7)
    if action == "replace_destination":
        patterns = [
            "Change the destination to {article_target} instead. {modifier}",
            "Cancel the old destination and route me to {article_target}. {modifier}",
            "Update the route: I want {article_target} as the new final stop. {modifier}",
            "Replace the final stop with {article_target}. {modifier}",
            "I changed my mind; take me to {article_target} from here. {modifier}",
        ]
    elif action == "insert_before_final":
        patterns = [
            "Insert {article_target} before the final destination. {modifier}",
            "Before the final stop, add {article_target}. {modifier}",
            "Keep the original destination, but stop at {article_target} first. {modifier}",
            "Update the current route so it goes through {article_target} before finishing. {modifier}",
            "Add {article_target} right before I arrive at the final stop. {modifier}",
        ]
    else:
        patterns = [
            "Add {article_target} as a stop on the way. {modifier}",
            "Insert {article_target} into my current route. {modifier}",
            "Detour through {article_target} before continuing. {modifier}",
            "Keep going to the destination, but add {article_target} first. {modifier}",
            "Update the route with a nearby stop for {target}. {modifier}",
        ]
    return _pick(patterns, idx).format(article_target=article_target, target=target, city_name=city_name, modifier=modifier)


def _compose_with_context(canonical_instruction: str, *, idx: int, city_name: str) -> str:
    """Compose a canonical instruction with generic context.

    Args:
        canonical_instruction: Source instruction.
        idx: Deterministic variant index.
        city_name: Planned city name.

    Returns:
        Rendered instruction.
    """

    prefix = _pick(_CONTEXT_PREFIXES, idx)
    suffix = _pick(_CONTEXT_SUFFIXES, idx // 3)
    city_clause = _pick(["", "", "", f" I am starting in {city_name}."], idx // 5)
    return f"{prefix}{_lower_first(canonical_instruction) if prefix else canonical_instruction}{city_clause} {suffix}"


def _target_phrase(task: CompiledTask, *, idx: int) -> str:
    terms = [term for term in task.query_terms if len(term) <= 36]
    if not terms:
        terms = [item.replace("_", " ") for item in task.target_types]
    if not terms:
        terms = [task.subcategory.replace("_", " ")]
    return _pick(terms, idx).replace("nearby ", "").strip()


def _group_phrase(group: list[str], *, idx: int) -> str:
    options = [item.replace("_", " ") for item in group]
    return _pick(options, idx)


def _sequence_phrase(stops: list[str]) -> str:
    if len(stops) == 1:
        return stops[0]
    if len(stops) == 2:
        return f"{stops[0]} first, then {stops[1]}"
    return ", ".join(stops[:-1]) + f", and finally {stops[-1]}"


def _need_phrase(task: CompiledTask, *, idx: int) -> str:
    phrases = _NEED_PHRASES.get(task.subcategory) or _NEED_PHRASES.get(task.task_id) or []
    if not phrases:
        target = _target_phrase(task, idx=idx)
        phrases = [f"I need a place related to {target}", f"I have a need that can be solved by {target}"]
    return _pick(phrases, idx)


def _need_short(task: CompiledTask, *, idx: int) -> str:
    phrases = _NEED_SHORT.get(task.subcategory) or []
    if not phrases:
        return _target_phrase(task, idx=idx)
    return _pick(phrases, idx)


def _with_article(phrase: str) -> str:
    phrase = phrase.strip()
    if not phrase:
        return "a suitable place"
    if phrase.lower().split(" ", 1)[0] in {"a", "an", "the", "some"}:
        return phrase
    article = "an" if phrase[0].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {phrase}"


def _pick(items: list[str], idx: int) -> str:
    if not items:
        return ""
    return items[idx % len(items)]


def _lower_first(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    if text.startswith("I ") or text.startswith("I'm") or text.startswith("I need") or text.startswith("I want"):
        return text
    return text[:1].lower() + text[1:]


def _normalize_sentence(text: str) -> str:
    text = " ".join(str(text or "").split())
    text = text.replace("..", ".").replace("?.", "?").replace("!.", "!")
    return text.strip()


_OPENERS = [
    "Please",
    "Could you",
    "Can you",
    "Help me",
    "I would like you to",
    "For this trip,",
    "From here,",
    "Starting at my current spot,",
    "Before I continue,",
    "For the next stop,",
]


_NEAR_PHRASES = [
    "near me",
    "close to my current location",
    "within walking distance",
    "around here",
    "that is easy to reach from here",
    "not too far from this starting point",
    "in the surrounding area",
    "along a practical walking route",
    "near this origin",
    "that I can get to without a long detour",
]


_POI_PURPOSES = [
    "keep the route practical",
    "prefer the walkable option",
    "do not choose something far away",
    "make the destination easy to reach",
    "use the best nearby match",
    "prioritize a reachable stop",
]


_CONTEXT_PREFIXES = [
    "",
    "From my current location, ",
    "Starting here, ",
    "Before I keep moving, ",
    "For this navigation request, ",
    "Around my current position, ",
    "Using this as the origin, ",
    "For the next walking route, ",
]


_CONTEXT_SUFFIXES = [
    "",
    "Keep it practical from this origin.",
    "A walkable option is preferred.",
    "Avoid choosing a place that is unnecessarily far.",
    "Use a destination that fits the request.",
    "The route should make sense on foot.",
]


_REQUEST_MODIFIERS = [
    "",
    "Prefer a walkable choice.",
    "Do not make me walk farther than needed.",
    "Choose something practical from this origin.",
    "Use a place that clearly satisfies the request.",
    "The destination should make sense for this need.",
    "Avoid a random match if a better nearby option exists.",
    "Keep the route simple if possible.",
    "Prioritize a place that is easy to reach.",
    "Use the current location as the starting point.",
    "I care more about a suitable destination than a scenic detour.",
    "Pick a reasonable nearby option, not just a famous one.",
    "The result should be useful for a pedestrian.",
    "Try to keep the stop within a normal walking range.",
    "I want the destination to match the intent, not only the keyword.",
    "If there are several choices, use the most sensible one.",
]


_MULTI_STOP_MODIFIERS = [
    "",
    "Keep the order exactly as requested.",
    "Each stop should satisfy its own purpose.",
    "Do not collapse this into a single final destination.",
    "The intermediate stop matters.",
    "Make the combined route practical on foot.",
    "Avoid skipping any requested stop.",
    "Choose each stop based on where I am after the previous one.",
    "The sequence is more important than choosing a famous place.",
    "Use reachable stops for all legs.",
    "Keep unnecessary backtracking low.",
    "The route should still be walkable after adding all stops.",
]


_ROUTE_UPDATE_MODIFIERS = [
    "",
    "Apply this to the route I am already following.",
    "Use my current position as the new start.",
    "Do not restart the entire trip from the original origin.",
    "Keep the update consistent with the existing route state.",
    "Make the edited route practical from where I am now.",
    "Only change what the update asks for.",
    "Respect whether this is an added stop or a replacement.",
    "The current route context matters.",
    "Avoid dropping the final destination unless I asked to replace it.",
]


_NEED_PHRASES = {
    "hungry": ["I am hungry", "I need a meal soon", "I want something to eat", "I need a quick food option"],
    "thirsty": ["I am thirsty", "I need something to drink", "I want a beverage", "I need water, tea, coffee, or another drink"],
    "coffee_shop": ["I want coffee", "I need a cafe break", "I want a place for coffee", "I need caffeine before continuing"],
    "dessert_snack": ["I want something sweet", "I need a snack", "I want dessert", "I feel like cake, pastry, or ice cream"],
    "work_with_wifi": ["I need a place to work with Wi-Fi", "I need a laptop-friendly spot", "I want somewhere quiet for remote work", "I need seating and internet"],
    "family_kids": ["I am with children", "I need a kid-friendly place", "I want somewhere suitable for a family", "I need a place where kids would be comfortable"],
    "parking_lot": ["I need convenient parking", "I am driving and need parking handled", "I want a place where parking is easy", "I need a stop with a lot or garage nearby"],
    "cheap_budget": ["I am on a budget", "I want an affordable option", "I need something not too expensive", "I want good value nearby"],
    "charge_phone": ["my phone battery is low", "I need to charge my phone", "I need a plug or power outlet", "I want somewhere I can sit and recharge my phone"],
    "buy_daily_goods": ["I need daily goods", "I need basic supplies", "I want to buy household essentials", "I need a quick stop for everyday items"],
    "buy_fruit": ["I want to buy fruit", "I need fresh produce", "I want vegetables or fruit", "I need a grocery or market stop for produce"],
    "restroom": ["I need to use a restroom", "I need a bathroom soon", "I need a public toilet", "I need somewhere with a usable washroom"],
    "quick_rest": ["I need a short rest", "I want to sit down for a bit", "I need a calm place to pause", "I want a nearby break spot"],
    "print_document": ["I need to print a document", "I need to copy some papers", "I need a printing service", "I want somewhere that can handle document printing"],
}


_NEED_SHORT = {
    "hungry": ["food", "a meal", "a quick bite"],
    "thirsty": ["a drink", "a beverage", "something to drink"],
    "coffee_shop": ["coffee", "a cafe stop"],
    "dessert_snack": ["dessert", "a snack", "something sweet"],
    "work_with_wifi": ["work with Wi-Fi", "a laptop-friendly place"],
    "family_kids": ["a kid-friendly stop", "a family-friendly place"],
    "parking_lot": ["parking", "easy parking"],
    "cheap_budget": ["an affordable option", "a budget-friendly place"],
    "charge_phone": ["phone charging", "a power outlet"],
    "buy_daily_goods": ["daily goods", "basic supplies"],
    "buy_fruit": ["fruit or produce", "fresh groceries"],
    "restroom": ["a restroom", "a bathroom"],
    "quick_rest": ["a short rest", "a place to sit"],
    "print_document": ["document printing", "printing or copying"],
}


def _manifest(
    *,
    taxonomy_path: Path,
    distribution_path: Path,
    provider_map_path: Path,
    regions_config_path: Path | None,
    quality_policy_path: Path | None,
    city_plan_mode: str,
    output_root: Path,
    language: str,
    compiled_tasks: list[CompiledTask],
    instruction_rows: list[dict[str, Any]],
    distribution: dict[str, Any],
    collection_config: dict[str, Any],
    instruction_mode: str,
) -> dict[str, Any]:
    """Build instruction-set manifest.

    Args:
        taxonomy_path: Taxonomy path.
        distribution_path: Distribution path.
        provider_map_path: Provider map path.
        regions_config_path: Region config path.
        quality_policy_path: Optional v2 quality policy path.
        city_plan_mode: City quota resolution policy.
        output_root: Output root.
        language: Template language.
        compiled_tasks: Compiled tasks.
        instruction_rows: Planned rows.
        distribution: Raw distribution plan.
        collection_config: Generated collection config.
        instruction_mode: Instruction rendering mode.

    Returns:
        Manifest dictionary.
    """

    by_category = Counter(row["category"] for row in instruction_rows)
    by_subcategory = Counter(f"{row['category']}:{row['subcategory']}" for row in instruction_rows)
    by_city = Counter(str((row.get("planned_city") or {}).get("city") or "") for row in instruction_rows)
    return {
        "version": "urbannav_instruction_set_manifest_v1",
        "builder_version": BUILDER_VERSION,
        "language": language,
        "instruction_mode": instruction_mode,
        "output_root": str(output_root),
        "taxonomy": str(taxonomy_path),
        "distribution": str(distribution_path),
        "provider_map": str(provider_map_path),
        "regions_config": str(regions_config_path) if regions_config_path else None,
        "quality_policy": str(quality_policy_path) if quality_policy_path else None,
        "city_plan_mode": city_plan_mode,
        "target_cases": int(distribution.get("target_cases") or len(instruction_rows)),
        "instruction_rows": len(instruction_rows),
        "unique_instruction_count": len({str(row.get("instruction") or "") for row in instruction_rows}),
        "task_count": len(compiled_tasks),
        "template_count": sum(len(task.templates) for task in compiled_tasks),
        "single_stop_collection_cases": int(collection_config.get("planned_single_stop_cases") or 0),
        "complex_collection_cases": int(collection_config.get("planned_complex_cases") or 0),
        "category_distribution": dict(sorted(by_category.items())),
        "top_subcategory_distribution": dict(by_subcategory.most_common(64)),
        "city_distribution": dict(sorted(by_city.items())),
        "complex_task_note": (
            "multi_stop_route and route_update_add_via rows are generated in instruction_set.jsonl; "
            "their live GT construction must use the multi-stop and route-update builders, not the single-stop collector."
        ),
        "outputs": {
            "instruction_set": "instruction_set.jsonl",
            "task_catalog": "task_catalog.json",
            "collection_config": "collection_config_full_single_stop.json",
            "manifest": "manifest.json",
        },
    }


def _acceptance_settings(*, category: str, subcategory: str, instruction: str) -> dict[str, Any]:
    """Return instruction-level destination acceptance settings.

    Args:
        category: Normalized task category.
        subcategory: Task subcategory.
        instruction: Rendered instruction text.

    Returns:
        Policy fields consumed by GT builders.
    """

    policy = _acceptance_policy(category=category, subcategory=subcategory, instruction=instruction)
    if policy == "nearest_provider_route":
        return {
            "acceptance_policy": policy,
            "allow_alternatives": True,
            "acceptable_route_ratio": NEAREST_ACCEPTABLE_ROUTE_RATIO,
            "acceptable_route_limit_m": None,
        }
    if policy == "semantic_nearby_with_route_efficiency":
        ratio = IMPLICIT_ROUTE_RATIO_OVERRIDES.get(subcategory, IMPLICIT_ACCEPTABLE_ROUTE_RATIO)
        return {
            "acceptance_policy": policy,
            "allow_alternatives": True,
            "acceptable_route_ratio": ratio,
            "acceptable_route_limit_m": IMPLICIT_ACCEPTABLE_ROUTE_LIMIT_M,
        }
    if policy == "semantic_nearby_poi":
        return {
            "acceptance_policy": policy,
            "allow_alternatives": True,
            "acceptable_route_ratio": NEARBY_ACCEPTABLE_ROUTE_RATIO,
            "acceptable_route_limit_m": NEARBY_ACCEPTABLE_ROUTE_LIMIT_M,
        }
    return {
        "acceptance_policy": policy,
        "allow_alternatives": True,
        "acceptable_route_ratio": NEAREST_ACCEPTABLE_ROUTE_RATIO,
        "acceptable_route_limit_m": None,
    }


def _acceptance_policy(*, category: str, subcategory: str, instruction: str) -> str:
    """Return the target acceptance policy name.

    Args:
        category: Normalized task category.
        subcategory: Task subcategory.
        instruction: Rendered instruction text.

    Returns:
        Policy label.
    """

    if category == "poi_navigation":
        return "nearest_provider_route" if _is_explicit_nearest_instruction(instruction) else "semantic_nearby_poi"
    if category == "implicit_need_navigation":
        return "semantic_nearby_with_route_efficiency"
    if category == "multi_stop_route":
        return "ordered_goals_segmentwise_optimal"
    if category == "route_update_add_via":
        return "route_session_update"
    return f"analysis:{subcategory}"


def _is_explicit_nearest_instruction(instruction: str) -> bool:
    """Detect whether an instruction explicitly asks for the nearest option.

    Args:
        instruction: Natural-language instruction.

    Returns:
        ``True`` when the instruction should be evaluated as nearest-target
        selection instead of generic nearby satisfaction.
    """

    text = f" {str(instruction or '').lower()} "
    nearest_markers = (
        " closest ",
        " nearest ",
        " shortest ",
        " shortest-route ",
        " shortest route ",
        " minimum walking ",
        " least walking ",
        " as near as possible ",
        " nearest-route ",
    )
    return any(marker in text for marker in nearest_markers)


def _normalize_category(category: str) -> str:
    if category == "public_poi_navigation":
        return "poi_navigation"
    return category


def _flatten(groups: Iterable[Iterable[str]]) -> list[str]:
    return [item for group in groups for item in group]


def _dedupe(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _output_paths(root: Path) -> dict[str, Path]:
    return {
        "instruction_set": root / "instruction_set.jsonl",
        "task_catalog": root / "task_catalog.json",
        "collection_config": root / "collection_config_full_single_stop.json",
        "manifest": root / "manifest.json",
    }


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
