#!/usr/bin/env python3
r"""Mine Yelp-supported demand groups for the UrbanNav benchmark.

The script implements the demand-mining loop used before GT construction:

\[
e_g=(B_g,C_g,R_g,T_g,M_g)
\]

where \(B_g\) is the number of Yelp businesses matching candidate demand group
\(g\), \(C_g\) is city coverage, \(R_g\) and \(T_g\) are review/tip evidence
counts, and \(M_g\) indicates whether every target type has a provider mapping.
A group is promoted only when:

\[
M_g=1 \land C_g \ge C_{\min} \land
\left(B_g \ge B_{\min} \lor R_g+T_g \ge E_{\min}\right).
\]

The output is intentionally split into auditable artifacts instead of mutating
the production taxonomy in place.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


ROUTE_SERVER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YELP_ROOT = Path("/media/ubuntu/data2/SR-Nav Dataset/archive/yelp_dataset")
DEFAULT_RULES = ROUTE_SERVER_ROOT / "config" / "urbannav_yelp_demand_mining_rules.json"
DEFAULT_TAXONOMY = ROUTE_SERVER_ROOT / "config" / "yelp_demand_taxonomy.json"
DEFAULT_PROVIDER_MAP = ROUTE_SERVER_ROOT / "config" / "urbannav_target_type_provider_map.json"
DEFAULT_DISTRIBUTION = ROUTE_SERVER_ROOT / "config" / "urbannav_10k_distribution_plan.json"
DEFAULT_REGIONS = ROUTE_SERVER_ROOT / "config" / "urbannav_global_gt_collection_full_en.json"
DEFAULT_OUTPUT_ROOT = ROUTE_SERVER_ROOT / "data" / "benchmark" / "urbannav_yelp_demand_mining_v1"
DEFAULT_TAXONOMY_OUTPUT = ROUTE_SERVER_ROOT / "config" / "yelp_demand_taxonomy_mined_v2.json"
DEFAULT_PROVIDER_MAP_OUTPUT = ROUTE_SERVER_ROOT / "config" / "urbannav_target_type_provider_map_mined_v2.json"
DEFAULT_INSTRUCTION_OUTPUT = ROUTE_SERVER_ROOT / "data" / "benchmark" / "urbannav_instruction_set_full_en_v2"
MINER_VERSION = "urbannav_yelp_demand_miner_v1"


@dataclass(frozen=True)
class DemandRule:
    r"""One candidate demand rule.

    Args:
        demand_group: Stable demand-group id.
        category: UrbanNav task family.
        subcategory: Reporting subcategory.
        target_types: Target types for single-stop tasks.
        ordered_target_types: Ordered target groups for multi-stop tasks.
        yelp_categories_any: Yelp business categories used as evidence.
        attribute_any: Yelp attributes used as optional semantic evidence.
        text_pattern: Review/tip regex.
        semantic_tags: Semantic tags retained for audit and analysis splits.
        templates_en: English instruction templates.
        templates_zh: Chinese instruction templates.
        review_by_default: Whether this should remain an analysis split by
            default because it relies on attributes or subjective reviews.
        quota_weight: Relative weight when compiled into an instruction set.
        route_update_action: Optional route-update operation label.

    The rule is a hypothesis:

    \[
    h_g = (K_g, A_g, P_g)
    \]

    with Yelp category constraints \(K_g\), optional attributes \(A_g\), and
    text evidence pattern \(P_g\).
    """

    demand_group: str
    category: str
    subcategory: str
    target_types: list[str]
    ordered_target_types: list[list[str]]
    yelp_categories_any: list[str]
    attribute_any: list[str]
    text_pattern: str
    semantic_tags: list[str]
    templates_en: list[str]
    templates_zh: list[str]
    review_by_default: bool
    quota_weight: float
    route_update_action: str | None = None

    @property
    def target_type_groups(self) -> list[list[str]]:
        """Return target groups that require provider mappings.

        Returns:
            Ordered target groups for multi-stop rules, otherwise one group
            containing ``target_types``.
        """

        if self.ordered_target_types:
            return [list(group) for group in self.ordered_target_types]
        return [list(self.target_types)] if self.target_types else []


@dataclass
class DemandEvidence:
    """Accumulated evidence for one demand rule.

    Args:
        demand_group: Rule id.
        category: Task family.
        subcategory: Reporting subcategory.
        business_count: Number of matching Yelp businesses.
        business_city_counts: Matching business city distribution.
        business_category_counts: Matching Yelp category distribution.
        attribute_business_count: Businesses with one of the requested
            attributes.
        review_evidence_count: Matching Yelp review count.
        tip_evidence_count: Matching Yelp tip count.
        text_city_counts: Review/tip city distribution.
        examples: Short evidence examples.
        missing_target_types: Target types without provider mapping.
        decision: Promotion decision.
        decision_reason: Human-readable reason.

    Returns:
        None.
    """

    demand_group: str
    category: str
    subcategory: str
    business_count: int = 0
    business_city_counts: Counter[str] = field(default_factory=Counter)
    business_category_counts: Counter[str] = field(default_factory=Counter)
    attribute_business_count: int = 0
    review_evidence_count: int = 0
    tip_evidence_count: int = 0
    text_city_counts: Counter[str] = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)
    missing_target_types: list[str] = field(default_factory=list)
    decision: str = "unscored"
    decision_reason: str = ""

    @property
    def city_count(self) -> int:
        """Return city coverage.

        Returns:
            Number of distinct non-empty Yelp cities represented by business or
            text evidence.
        """

        return len({city for city in [*self.business_city_counts, *self.text_city_counts] if city})

    @property
    def text_evidence_count(self) -> int:
        """Return combined review and tip evidence count.

        Returns:
            ``review_evidence_count + tip_evidence_count``.
        """

        return self.review_evidence_count + self.tip_evidence_count

    def to_record(self) -> dict[str, Any]:
        """Serialize evidence.

        Returns:
            JSON-ready evidence record.
        """

        return {
            "demand_group": self.demand_group,
            "category": self.category,
            "subcategory": self.subcategory,
            "business_count": self.business_count,
            "city_count": self.city_count,
            "attribute_business_count": self.attribute_business_count,
            "review_evidence_count": self.review_evidence_count,
            "tip_evidence_count": self.tip_evidence_count,
            "text_evidence_count": self.text_evidence_count,
            "missing_target_types": list(self.missing_target_types),
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "top_business_cities": dict(self.business_city_counts.most_common(20)),
            "top_text_cities": dict(self.text_city_counts.most_common(20)),
            "top_yelp_categories": dict(self.business_category_counts.most_common(20)),
            "examples": list(self.examples),
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yelp-root", type=Path, default=DEFAULT_YELP_ROOT, help="Yelp dataset directory.")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES, help="Demand-mining rule config.")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY, help="Existing UrbanNav taxonomy.")
    parser.add_argument("--provider-map", type=Path, default=DEFAULT_PROVIDER_MAP, help="Existing provider ontology.")
    parser.add_argument("--distribution", type=Path, default=DEFAULT_DISTRIBUTION, help="Instruction-set distribution plan.")
    parser.add_argument("--regions-config", type=Path, default=DEFAULT_REGIONS, help="Region config for instruction-set compilation.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Mining report output directory.")
    parser.add_argument("--taxonomy-output", type=Path, default=DEFAULT_TAXONOMY_OUTPUT, help="Merged mined taxonomy output path.")
    parser.add_argument("--provider-map-output", type=Path, default=DEFAULT_PROVIDER_MAP_OUTPUT, help="Merged mined provider-map output path.")
    parser.add_argument("--instruction-output-root", type=Path, default=DEFAULT_INSTRUCTION_OUTPUT, help="Instruction-set v2 output root.")
    parser.add_argument("--business-limit", type=int, default=None, help="Optional business scan limit.")
    parser.add_argument("--review-limit", type=int, default=500_000, help="Optional review scan limit.")
    parser.add_argument("--tip-limit", type=int, default=500_000, help="Optional tip scan limit.")
    parser.add_argument("--build-instruction-set", action="store_true", help="Compile a v2 instruction set from the mined taxonomy.")
    parser.add_argument("--build-review-html", action="store_true", help="Build HTML review for the v2 instruction set.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def main() -> None:
    """Run Yelp demand mining.

    Returns:
        None.
    """

    args = parse_args()
    result = mine_yelp_demands(
        yelp_root=args.yelp_root,
        rules_path=args.rules,
        taxonomy_path=args.taxonomy,
        provider_map_path=args.provider_map,
        distribution_path=args.distribution,
        regions_config_path=args.regions_config,
        output_root=args.output_root,
        taxonomy_output=args.taxonomy_output,
        provider_map_output=args.provider_map_output,
        instruction_output_root=args.instruction_output_root,
        business_limit=args.business_limit,
        review_limit=args.review_limit,
        tip_limit=args.tip_limit,
        build_instruction_set=args.build_instruction_set,
        build_review_html=args.build_review_html,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def mine_yelp_demands(
    *,
    yelp_root: Path,
    rules_path: Path,
    taxonomy_path: Path,
    provider_map_path: Path,
    distribution_path: Path,
    regions_config_path: Path,
    output_root: Path,
    taxonomy_output: Path,
    provider_map_output: Path,
    instruction_output_root: Path,
    business_limit: int | None,
    review_limit: int | None,
    tip_limit: int | None,
    build_instruction_set: bool,
    build_review_html: bool,
    overwrite: bool,
) -> dict[str, Any]:
    r"""Mine, filter, and optionally compile Yelp-supported demands.

    Args:
        yelp_root: Directory containing Yelp JSONL files.
        rules_path: Candidate-rule config.
        taxonomy_path: Existing taxonomy.
        provider_map_path: Existing provider map.
        distribution_path: Instruction-set distribution plan.
        regions_config_path: Region config.
        output_root: Mining report directory.
        taxonomy_output: Merged taxonomy output path.
        provider_map_output: Merged provider-map output path.
        instruction_output_root: Optional v2 instruction-set output root.
        business_limit: Optional business scan limit.
        review_limit: Optional review scan limit.
        tip_limit: Optional tip scan limit.
        build_instruction_set: Whether to compile v2 instructions.
        build_review_html: Whether to build v2 review HTML.
        overwrite: Whether to overwrite existing artifacts.

    Returns:
        Summary of output paths and accepted/rejected demand counts.
    """

    output_paths = _output_paths(output_root)
    _check_outputs(output_paths=output_paths, taxonomy_output=taxonomy_output, provider_map_output=provider_map_output, overwrite=overwrite)

    rules_payload = _load_json(rules_path)
    taxonomy = _load_json(taxonomy_path)
    provider_map = _load_json(provider_map_path)
    rules = _load_rules(rules_payload)
    thresholds = dict(rules_payload.get("thresholds") or {})
    target_type_additions = dict(rules_payload.get("target_type_additions") or {})
    merged_provider_map = _merged_provider_map(provider_map, target_type_additions)

    business_path = yelp_root / "yelp_academic_dataset_business.json"
    review_path = yelp_root / "yelp_academic_dataset_review.json"
    tip_path = yelp_root / "yelp_academic_dataset_tip.json"
    _require_files([business_path, review_path, tip_path])

    evidence = {rule.demand_group: DemandEvidence(rule.demand_group, rule.category, rule.subcategory) for rule in rules}
    business_meta = _scan_businesses(
        business_path=business_path,
        rules=rules,
        evidence=evidence,
        limit=business_limit,
    )
    _scan_text(
        path=review_path,
        rules=rules,
        business_meta=business_meta,
        evidence=evidence,
        source="review",
        limit=review_limit,
    )
    _scan_text(
        path=tip_path,
        rules=rules,
        business_meta=business_meta,
        evidence=evidence,
        source="tip",
        limit=tip_limit,
    )

    existing_groups = {str(item.get("demand_group")) for item in taxonomy.get("specs") or []}
    accepted_rules: list[DemandRule] = []
    accepted_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    for rule in rules:
        record = evidence[rule.demand_group]
        record.missing_target_types = _missing_target_types(rule, merged_provider_map)
        _score_evidence(record, rule=rule, thresholds=thresholds, existing_groups=existing_groups)
        serialized = record.to_record()
        serialized["rule"] = _rule_record(rule)
        if record.decision.startswith("accept"):
            accepted_rules.append(rule)
            accepted_records.append(serialized)
        else:
            rejected_records.append(serialized)

    new_specs = [_rule_to_taxonomy_spec(rule) for rule in accepted_rules if rule.demand_group not in existing_groups]
    existing_supported = [item for item in accepted_records if item["demand_group"] in existing_groups]
    taxonomy_v2 = _merged_taxonomy(
        taxonomy=taxonomy,
        new_specs=new_specs,
        mining_metadata={
            "miner_version": MINER_VERSION,
            "rules": str(rules_path),
            "business_limit": business_limit,
            "review_limit": review_limit,
            "tip_limit": tip_limit,
            "accepted_new_specs": len(new_specs),
            "accepted_existing_specs": len(existing_supported),
        },
    )

    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_paths["candidates_jsonl"], [evidence[rule.demand_group].to_record() | {"rule": _rule_record(rule)} for rule in rules])
    _write_json(output_paths["taxonomy_patch"], {"version": "urbannav_yelp_taxonomy_patch_v1", "specs": new_specs, "accepted_existing": existing_supported})
    _write_json(output_paths["provider_map_patch"], {"version": "urbannav_provider_map_patch_v1", "target_types": target_type_additions})
    _write_json(output_paths["report_json"], _report_payload(rules, evidence, thresholds, yelp_root))
    _write_text(output_paths["report_md"], _render_markdown_report(rules, evidence))
    _write_json(taxonomy_output, taxonomy_v2)
    _write_json(provider_map_output, merged_provider_map)

    instruction_result: dict[str, Any] | None = None
    review_result: dict[str, Any] | None = None
    if build_instruction_set:
        instruction_result = _build_instruction_set(
            taxonomy_output=taxonomy_output,
            provider_map_output=provider_map_output,
            distribution_path=distribution_path,
            regions_config_path=regions_config_path,
            instruction_output_root=instruction_output_root,
            overwrite=overwrite,
        )
    if build_review_html:
        if not build_instruction_set:
            raise ValueError("--build-review-html requires --build-instruction-set in this script.")
        review_result = _build_review_html(instruction_output_root=instruction_output_root)

    summary = {
        "miner_version": MINER_VERSION,
        "rule_count": len(rules),
        "accepted_count": len(accepted_records),
        "accepted_new_specs": len(new_specs),
        "accepted_existing_specs": len(existing_supported),
        "rejected_count": len(rejected_records),
        "taxonomy_output": str(taxonomy_output),
        "provider_map_output": str(provider_map_output),
        "output_root": str(output_root),
        "report_json": str(output_paths["report_json"]),
        "report_md": str(output_paths["report_md"]),
        "instruction_result": instruction_result,
        "review_result": review_result,
    }
    _write_json(output_paths["manifest"], summary)
    return summary


def _scan_businesses(
    *,
    business_path: Path,
    rules: list[DemandRule],
    evidence: dict[str, DemandEvidence],
    limit: int | None,
) -> dict[str, dict[str, Any]]:
    """Scan Yelp businesses and update category/attribute evidence.

    Args:
        business_path: Yelp business JSONL path.
        rules: Candidate rules.
        evidence: Evidence accumulator by demand group.
        limit: Optional row limit.

    Returns:
        Business metadata keyed by Yelp business id.
    """

    business_meta: dict[str, dict[str, Any]] = {}
    for index, business in enumerate(_iter_jsonl(business_path), 1):
        if limit is not None and index > limit:
            break
        categories = _categories(business)
        attributes = business.get("attributes") or {}
        city = str(business.get("city") or "")
        meta = {
            "name": business.get("name"),
            "city": city,
            "state": business.get("state"),
            "categories": categories,
            "attributes": attributes,
            "stars": business.get("stars"),
            "review_count": business.get("review_count"),
        }
        business_meta[str(business.get("business_id") or "")] = meta
        for rule in rules:
            if not _business_matches_rule(meta, rule):
                continue
            item = evidence[rule.demand_group]
            item.business_count += 1
            item.business_city_counts[city] += 1
            for category in categories:
                item.business_category_counts[category] += 1
            if _has_any_attribute(attributes, rule.attribute_any):
                item.attribute_business_count += 1
    return business_meta


def _scan_text(
    *,
    path: Path,
    rules: list[DemandRule],
    business_meta: dict[str, dict[str, Any]],
    evidence: dict[str, DemandEvidence],
    source: str,
    limit: int | None,
) -> None:
    """Scan Yelp reviews or tips for demand evidence.

    Args:
        path: Yelp review/tip JSONL path.
        rules: Candidate rules.
        business_meta: Business metadata.
        evidence: Evidence accumulator.
        source: ``review`` or ``tip``.
        limit: Optional row limit.

    Returns:
        None.
    """

    compiled = {rule.demand_group: re.compile(rule.text_pattern, re.IGNORECASE) for rule in rules if rule.text_pattern}
    rules_by_category: dict[str, list[DemandRule]] = defaultdict(list)
    categoryless_rules: list[DemandRule] = []
    for rule in rules:
        if not rule.yelp_categories_any:
            categoryless_rules.append(rule)
            continue
        for category in rule.yelp_categories_any:
            rules_by_category[_norm(category)].append(rule)
    for index, row in enumerate(_iter_jsonl(path), 1):
        if limit is not None and index > limit:
            break
        text = str(row.get("text") or "")
        business = business_meta.get(str(row.get("business_id") or ""), {})
        candidate_rules = _rules_for_business(meta=business, rules_by_category=rules_by_category, categoryless_rules=categoryless_rules)
        for rule in candidate_rules:
            pattern = compiled.get(rule.demand_group)
            if pattern is None or not pattern.search(text):
                continue
            item = evidence[rule.demand_group]
            if source == "review":
                item.review_evidence_count += 1
            else:
                item.tip_evidence_count += 1
            city = str(business.get("city") or "")
            item.text_city_counts[city] += 1
            if len(item.examples) < 5:
                item.examples.append(
                    {
                        "source": source,
                        "business": business.get("name"),
                        "city": city,
                        "text_excerpt": text[:240].replace("\n", " "),
                    }
                )


def _rules_for_business(
    *,
    meta: dict[str, Any],
    rules_by_category: dict[str, list[DemandRule]],
    categoryless_rules: list[DemandRule],
) -> list[DemandRule]:
    """Return candidate rules relevant to one business.

    Args:
        meta: Business metadata. Empty metadata falls back to categoryless rules.
        rules_by_category: Precomputed rule index by Yelp category.
        categoryless_rules: Rules without Yelp category constraints.

    Returns:
        Deduplicated rules whose category constraints overlap the business.
    """

    if not meta:
        return list(categoryless_rules)
    selected: dict[str, DemandRule] = {rule.demand_group: rule for rule in categoryless_rules}
    for category in meta.get("categories") or []:
        for rule in rules_by_category.get(_norm(category), []):
            selected[rule.demand_group] = rule
    return list(selected.values())


def _score_evidence(
    record: DemandEvidence,
    *,
    rule: DemandRule,
    thresholds: dict[str, Any],
    existing_groups: set[str],
) -> None:
    """Apply provider and evidence thresholds to one candidate.

    Args:
        record: Evidence record to mutate.
        rule: Demand rule.
        thresholds: Threshold config.
        existing_groups: Demand groups already in the base taxonomy.

    Returns:
        None.
    """

    if record.missing_target_types:
        record.decision = "reject"
        record.decision_reason = "missing_provider_mapping:" + ",".join(record.missing_target_types)
        return

    if rule.category == "semantic_preference" or rule.review_by_default:
        min_businesses = int(thresholds.get("analysis_min_businesses", 120))
        min_cities = int(thresholds.get("analysis_min_cities", 4))
        min_text = int(thresholds.get("analysis_min_text_evidence", 120))
        decision = "accept_analysis"
    else:
        min_businesses = int(thresholds.get("main_min_businesses", 300))
        min_cities = int(thresholds.get("main_min_cities", 6))
        min_text = int(thresholds.get("main_min_text_evidence", 300))
        decision = "accept_main"

    has_business = record.business_count >= min_businesses
    has_text = record.text_evidence_count >= min_text
    has_city = record.city_count >= min_cities
    if has_city and (has_business or has_text):
        record.decision = decision + ("_existing" if rule.demand_group in existing_groups else "")
        record.decision_reason = f"city_count={record.city_count}, business_count={record.business_count}, text_evidence={record.text_evidence_count}"
        return
    record.decision = "reject"
    record.decision_reason = (
        f"low_evidence: city_count={record.city_count}/{min_cities}, "
        f"business_count={record.business_count}/{min_businesses}, "
        f"text_evidence={record.text_evidence_count}/{min_text}"
    )


def _load_rules(payload: dict[str, Any]) -> list[DemandRule]:
    """Load mining rules from JSON.

    Args:
        payload: Rule config payload.

    Returns:
        Candidate rules.
    """

    rules: list[DemandRule] = []
    seen: set[str] = set()
    for item in payload.get("candidate_rules") or []:
        demand_group = str(item.get("demand_group") or "")
        if not demand_group:
            raise ValueError("candidate rule is missing demand_group")
        if demand_group in seen:
            raise ValueError(f"duplicate candidate demand_group: {demand_group}")
        seen.add(demand_group)
        rules.append(
            DemandRule(
                demand_group=demand_group,
                category=str(item.get("category") or ""),
                subcategory=str(item.get("subcategory") or demand_group),
                target_types=_str_list(item.get("target_types")),
                ordered_target_types=[_str_list(group) for group in item.get("ordered_target_types") or []],
                yelp_categories_any=_str_list(item.get("yelp_categories_any")),
                attribute_any=_str_list(item.get("attribute_any")),
                text_pattern=str(item.get("text_pattern") or ""),
                semantic_tags=_str_list(item.get("semantic_tags")),
                templates_en=_str_list(item.get("templates_en")),
                templates_zh=_str_list(item.get("templates_zh")),
                review_by_default=bool(item.get("review_by_default")),
                quota_weight=float(item.get("quota_weight") or 100),
                route_update_action=item.get("route_update_action"),
            )
        )
    if not rules:
        raise ValueError("No candidate_rules found in mining config.")
    return rules


def _missing_target_types(rule: DemandRule, provider_map: dict[str, Any]) -> list[str]:
    """Find target types missing from provider ontology.

    Args:
        rule: Candidate rule.
        provider_map: Provider ontology.

    Returns:
        Sorted missing target types.
    """

    known = provider_map.get("target_types") or {}
    missing = sorted({target for group in rule.target_type_groups for target in group if target not in known})
    return missing


def _rule_to_taxonomy_spec(rule: DemandRule) -> dict[str, Any]:
    """Convert an accepted rule into a taxonomy spec.

    Args:
        rule: Accepted demand rule.

    Returns:
        Taxonomy spec dictionary.
    """

    return {
        "demand_group": rule.demand_group,
        "category": rule.category,
        "category_zh": _category_zh(rule.category),
        "subcategory": rule.subcategory,
        "target_types": list(rule.target_types),
        "semantic_tags": list(rule.semantic_tags),
        "templates_en": list(rule.templates_en),
        "templates_zh": list(rule.templates_zh),
        "yelp_categories_any": list(rule.yelp_categories_any),
        "attribute_any": list(rule.attribute_any),
        "text_pattern": rule.text_pattern,
        "ordered_target_types": [list(group) for group in rule.ordered_target_types] if rule.ordered_target_types else None,
        "review_by_default": bool(rule.review_by_default),
        "route_update_action": rule.route_update_action,
        "quota_weight": rule.quota_weight,
    }


def _merged_taxonomy(*, taxonomy: dict[str, Any], new_specs: list[dict[str, Any]], mining_metadata: dict[str, Any]) -> dict[str, Any]:
    """Build mined taxonomy v2.

    Args:
        taxonomy: Base taxonomy.
        new_specs: New accepted specs.
        mining_metadata: Audit metadata.

    Returns:
        Merged taxonomy payload.
    """

    merged = dict(taxonomy)
    merged["version"] = "yelp_demand_taxonomy_mined_v2"
    merged["description"] = (
        "Yelp-supported UrbanNav demand taxonomy with mined, provider-mapped demand groups. "
        "Yelp is used as demand evidence; final GT is still built by map providers."
    )
    merged["mining_metadata"] = mining_metadata
    merged["specs"] = [*list(taxonomy.get("specs") or []), *new_specs]
    return merged


def _merged_provider_map(provider_map: dict[str, Any], additions: dict[str, Any]) -> dict[str, Any]:
    """Merge provider-map additions.

    Args:
        provider_map: Base provider map.
        additions: Target-type additions.

    Returns:
        Merged provider map.
    """

    merged = dict(provider_map)
    target_types = dict(provider_map.get("target_types") or {})
    for key, value in additions.items():
        target_types.setdefault(key, value)
    merged["version"] = "urbannav_target_type_provider_map_mined_v2"
    merged["target_types"] = target_types
    return merged


def _build_instruction_set(
    *,
    taxonomy_output: Path,
    provider_map_output: Path,
    distribution_path: Path,
    regions_config_path: Path,
    instruction_output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Compile v2 instruction-set artifacts.

    Args:
        taxonomy_output: Mined taxonomy path.
        provider_map_output: Mined provider-map path.
        distribution_path: Distribution plan.
        regions_config_path: Region config.
        instruction_output_root: Output root.
        overwrite: Whether to overwrite existing instruction artifacts.

    Returns:
        Instruction-set builder result.
    """

    builder_path = Path(__file__).with_name("build_urbannav_instruction_set.py")
    module = _load_module(builder_path, "build_urbannav_instruction_set_for_mining")
    return module.build_instruction_set(
        taxonomy_path=taxonomy_output,
        distribution_path=distribution_path,
        provider_map_path=provider_map_output,
        regions_config_path=regions_config_path,
        output_root=instruction_output_root,
        language="en",
        instruction_mode="case_variant",
        seed=20260629,
        include_semantic_analysis=False,
        allow_missing_target_types=False,
        overwrite=overwrite,
    )


def _build_review_html(*, instruction_output_root: Path) -> dict[str, Any]:
    """Build review HTML for the v2 instruction set.

    Args:
        instruction_output_root: Instruction-set output root.

    Returns:
        Review builder result.
    """

    review_path = ROUTE_SERVER_ROOT / "tools" / "evaluation" / "build_urbannav_instruction_review_html.py"
    module = _load_module(review_path, "build_urbannav_instruction_review_html_for_mining")
    return module.build_instruction_review_html(
        instruction_set=instruction_output_root / "instruction_set.jsonl",
        manifest_path=instruction_output_root / "manifest.json",
        output_dir=instruction_output_root / "review",
        max_samples=360,
        samples_per_subcategory=3,
        seed=20260629,
        title="UrbanNav Yelp-Mined Instruction Set Review",
    )


def _load_module(path: Path, module_name: str):
    """Load a Python module from a path.

    Args:
        path: Module file path.
        module_name: Temporary import name.

    Returns:
        Loaded module.
    """

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _business_matches_rule(meta: dict[str, Any], rule: DemandRule) -> bool:
    """Return whether a Yelp business matches a rule's category evidence.

    Args:
        meta: Business metadata.
        rule: Candidate rule.

    Returns:
        True when category constraints match.
    """

    wanted = {_norm(category) for category in rule.yelp_categories_any}
    if not wanted:
        return True
    business_categories = {_norm(category) for category in meta.get("categories") or []}
    return bool(wanted & business_categories)


def _has_any_attribute(attributes: dict[str, Any], keys: list[str]) -> bool:
    """Return whether any requested Yelp attribute exists with a non-empty value.

    Args:
        attributes: Yelp attribute dictionary.
        keys: Attribute keys.

    Returns:
        True when at least one attribute is present.
    """

    for key in keys:
        value = attributes.get(key)
        if value not in (None, "", "False", False):
            return True
    return False


def _categories(row: dict[str, Any]) -> list[str]:
    """Parse Yelp category string.

    Args:
        row: Yelp business row.

    Returns:
        Category list.
    """

    return [item.strip() for item in str(row.get("categories") or "").split(",") if item.strip()]


def _norm(value: str) -> str:
    """Normalize category labels.

    Args:
        value: Raw value.

    Returns:
        Lowercase normalized value.
    """

    return re.sub(r"\s+", " ", str(value).strip().lower())


def _rule_record(rule: DemandRule) -> dict[str, Any]:
    """Serialize a rule.

    Args:
        rule: Demand rule.

    Returns:
        JSON-ready rule record.
    """

    return {
        "demand_group": rule.demand_group,
        "category": rule.category,
        "subcategory": rule.subcategory,
        "target_types": list(rule.target_types),
        "ordered_target_types": [list(group) for group in rule.ordered_target_types],
        "yelp_categories_any": list(rule.yelp_categories_any),
        "attribute_any": list(rule.attribute_any),
        "semantic_tags": list(rule.semantic_tags),
        "review_by_default": rule.review_by_default,
        "route_update_action": rule.route_update_action,
        "quota_weight": rule.quota_weight,
    }


def _report_payload(
    rules: list[DemandRule],
    evidence: dict[str, DemandEvidence],
    thresholds: dict[str, Any],
    yelp_root: Path,
) -> dict[str, Any]:
    """Build report payload.

    Args:
        rules: Candidate rules.
        evidence: Evidence by rule.
        thresholds: Threshold config.
        yelp_root: Yelp root.

    Returns:
        Report dictionary.
    """

    records = [evidence[rule.demand_group].to_record() | {"rule": _rule_record(rule)} for rule in rules]
    decisions = Counter(record["decision"] for record in records)
    categories = Counter(record["category"] for record in records)
    return {
        "version": "urbannav_yelp_demand_mining_report_v1",
        "miner_version": MINER_VERSION,
        "yelp_root": str(yelp_root),
        "thresholds": thresholds,
        "rule_count": len(rules),
        "decision_distribution": dict(decisions),
        "category_distribution": dict(categories),
        "accepted": [record for record in records if str(record["decision"]).startswith("accept")],
        "rejected": [record for record in records if record["decision"] == "reject"],
    }


def _render_markdown_report(rules: list[DemandRule], evidence: dict[str, DemandEvidence]) -> str:
    """Render a compact Markdown report.

    Args:
        rules: Candidate rules.
        evidence: Evidence by rule.

    Returns:
        Markdown report.
    """

    lines = [
        "# UrbanNav Yelp Demand Mining Report",
        "",
        "| Demand | Category | Decision | Businesses | Cities | Text Evidence | Missing Target Types |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for rule in sorted(rules, key=lambda item: (evidence[item.demand_group].decision, item.category, item.subcategory)):
        record = evidence[rule.demand_group]
        missing = ", ".join(record.missing_target_types) or "-"
        lines.append(
            f"| `{rule.demand_group}` | `{rule.category}` | `{record.decision}` | "
            f"{record.business_count} | {record.city_count} | {record.text_evidence_count} | {missing} |"
        )
    lines.append("")
    return "\n".join(lines)


def _category_zh(category: str) -> str:
    """Return Chinese category label.

    Args:
        category: Category id.

    Returns:
        Chinese label.
    """

    return {
        "public_poi_navigation": "公共场所导航",
        "poi_navigation": "公共场所导航",
        "implicit_need_navigation": "隐式需求导航",
        "multi_stop_route": "多目标导航",
        "route_update_add_via": "中途增加途经点",
        "semantic_preference": "语义偏好导航",
    }.get(category, category)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Iterate JSONL records.

    Args:
        path: JSONL path.

    Returns:
        Parsed JSON dictionaries.
    """

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _str_list(value: Any) -> list[str]:
    """Normalize a value into a list of strings.

    Args:
        value: Raw list-like value.

    Returns:
        String list.
    """

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON file.

    Args:
        path: JSON path.

    Returns:
        Parsed JSON dictionary.
    """

    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON.

    Args:
        path: Output path.
        payload: JSON-ready payload.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL rows.

    Args:
        path: Output path.
        rows: Row dictionaries.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_text(path: Path, text: str) -> None:
    """Write text file.

    Args:
        path: Output path.
        text: Text content.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _require_files(paths: list[Path]) -> None:
    """Validate required files.

    Args:
        paths: Required paths.

    Returns:
        None.
    """

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Yelp files: " + ", ".join(missing))


def _output_paths(output_root: Path) -> dict[str, Path]:
    """Return mining output paths.

    Args:
        output_root: Output root.

    Returns:
        Mapping of artifact names to paths.
    """

    return {
        "candidates_jsonl": output_root / "demand_candidates.jsonl",
        "taxonomy_patch": output_root / "accepted_taxonomy_patch.json",
        "provider_map_patch": output_root / "provider_map_patch.json",
        "report_json": output_root / "demand_mining_report.json",
        "report_md": output_root / "demand_mining_report.md",
        "manifest": output_root / "manifest.json",
    }


def _check_outputs(
    *,
    output_paths: dict[str, Path],
    taxonomy_output: Path,
    provider_map_output: Path,
    overwrite: bool,
) -> None:
    """Prevent accidental overwrite.

    Args:
        output_paths: Mining output paths.
        taxonomy_output: Merged taxonomy output path.
        provider_map_output: Merged provider map output path.
        overwrite: Whether overwrite is allowed.

    Returns:
        None.
    """

    if overwrite:
        return
    existing = [path for path in [*output_paths.values(), taxonomy_output, provider_map_output] if path.exists()]
    if existing:
        raise RuntimeError("Output already exists; use --overwrite: " + ", ".join(str(path) for path in existing[:8]))


if __name__ == "__main__":
    main()
