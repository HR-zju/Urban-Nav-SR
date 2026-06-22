"""Validation helpers for frozen UrbanNav benchmark packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from mcp_server.eval.urbannav_io import iter_json_records, load_urbannav_cases
from mcp_server.eval.urbannav_metrics import haversine_m, route_distance_m
from mcp_server.eval.urbannav_schema import UrbanNavCase, UrbanNavDestination, UrbanNavPoint


@dataclass(frozen=True)
class UrbanNavValidationConfig:
    """Configuration for UrbanNav GT validation.

    Args:
        origin_tolerance_m: Maximum allowed distance between ``origin`` and the
            first point of ``gt_route``.
        destination_tolerance_m: Maximum allowed distance between
            ``gt_destination`` and the last point of ``gt_route``.
        require_cache_keys: Whether each case must contain ``cache_keys``.
        require_ranked_destinations: Whether each case must contain
            ``ranked_destinations``.
    """

    origin_tolerance_m: float = 30.0
    destination_tolerance_m: float = 30.0
    require_cache_keys: bool = True
    require_ranked_destinations: bool = True


@dataclass
class UrbanNavValidationReport:
    """Validation report for one benchmark package.

    Args:
        total_cases: Number of loaded GT cases.
        valid_cases: Number of cases without validation errors.
        invalid_cases: Number of cases with at least one validation error.
        errors_by_case: Mapping from case id to validation errors.
        reject_count: Number of reject rows, if a reject file was supplied.
        reject_reason_counts: Count by reject reason.
        category_counts: Case count by category.
        cache_missing_count: Number of referenced cache keys not found.
    """

    total_cases: int = 0
    valid_cases: int = 0
    invalid_cases: int = 0
    errors_by_case: dict[str, list[str]] = field(default_factory=dict)
    reject_count: int = 0
    reject_reason_counts: dict[str, int] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    cache_missing_count: int = 0

    @property
    def valid(self) -> bool:
        """Return whether all GT cases passed validation.

        Returns:
            ``True`` when ``invalid_cases == 0`` and no referenced cache keys
            are missing.
        """

        return self.invalid_cases == 0 and self.cache_missing_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the validation report.

        Returns:
            JSON-ready report dictionary.
        """

        return {
            "valid": self.valid,
            "total_cases": self.total_cases,
            "valid_cases": self.valid_cases,
            "invalid_cases": self.invalid_cases,
            "errors_by_case": self.errors_by_case,
            "reject_count": self.reject_count,
            "reject_reason_counts": self.reject_reason_counts,
            "category_counts": self.category_counts,
            "cache_missing_count": self.cache_missing_count,
        }


def validate_urbannav_cases(
    cases: Iterable[UrbanNavCase],
    *,
    poi_cache_keys: set[str] | None = None,
    route_cache_keys: set[str] | None = None,
    rejects: Iterable[dict[str, Any]] | None = None,
    config: UrbanNavValidationConfig | None = None,
) -> UrbanNavValidationReport:
    r"""Validate normalized UrbanNav cases.

    Core GT invariant:

    \[
    gt\_destination = ranked\_destinations[0]
    \]

    Route endpoint invariant:

    \[
    d(endpoint(gt\_route), gt\_destination) \le \epsilon_{dest}
    \]

    Args:
        cases: Normalized GT cases.
        poi_cache_keys: Available POI cache keys. If supplied, referenced POI
            cache keys must exist in this set.
        route_cache_keys: Available route cache keys. If supplied, referenced
            route cache keys must exist in this set.
        rejects: Optional reject records for reject reason statistics.
        config: Optional validation configuration.

    Returns:
        ``UrbanNavValidationReport`` with per-case errors and aggregate counts.
    """

    cfg = config or UrbanNavValidationConfig()
    report = UrbanNavValidationReport()
    seen_ids: set[str] = set()

    for case in cases:
        report.total_cases += 1
        report.category_counts[case.category or "unknown"] = report.category_counts.get(case.category or "unknown", 0) + 1
        errors = _validate_case(case, seen_ids, cfg)
        errors.extend(_validate_cache_keys(case, poi_cache_keys=poi_cache_keys, route_cache_keys=route_cache_keys, cfg=cfg))
        if errors:
            report.errors_by_case[case.case_id] = errors
            report.invalid_cases += 1
            report.cache_missing_count += sum(1 for item in errors if item.startswith("missing_cache_key:"))
        else:
            report.valid_cases += 1
        seen_ids.add(case.case_id)

    if rejects is not None:
        for reject in rejects:
            report.reject_count += 1
            reason = str(reject.get("reject_reason") or "unknown")
            report.reject_reason_counts[reason] = report.reject_reason_counts.get(reason, 0) + 1

    return report


def validate_urbannav_package(
    *,
    gt_path: str,
    poi_cache_path: str | None = None,
    route_cache_path: str | None = None,
    rejects_path: str | None = None,
    config: UrbanNavValidationConfig | None = None,
) -> UrbanNavValidationReport:
    """Validate a package from JSONL files.

    Args:
        gt_path: Path to ``task_gt.jsonl``.
        poi_cache_path: Optional path to ``poi_cache.jsonl``.
        route_cache_path: Optional path to ``route_cache.jsonl``.
        rejects_path: Optional path to ``task_rejects.jsonl``.
        config: Optional validation configuration.

    Returns:
        ``UrbanNavValidationReport``.
    """

    cases = load_urbannav_cases(gt_path)
    poi_keys = load_cache_keys(poi_cache_path) if poi_cache_path else None
    route_keys = load_cache_keys(route_cache_path) if route_cache_path else None
    rejects = list(iter_json_records(rejects_path)) if rejects_path else None
    return validate_urbannav_cases(
        cases,
        poi_cache_keys=poi_keys,
        route_cache_keys=route_keys,
        rejects=rejects,
        config=config,
    )


def load_cache_keys(path: str | None) -> set[str]:
    """Load replay cache keys from a JSON/JSONL cache file.

    Args:
        path: Cache file path. Records may use ``cache_key`` or ``key``.

    Returns:
        Set of cache key strings.
    """

    if not path:
        return set()
    keys: set[str] = set()
    for record in iter_json_records(path):
        key = record.get("cache_key") or record.get("key")
        if key:
            keys.add(str(key))
    return keys


def _validate_case(case: UrbanNavCase, seen_ids: set[str], cfg: UrbanNavValidationConfig) -> list[str]:
    """Validate one normalized case.

    Args:
        case: Case to validate.
        seen_ids: Previously seen case ids for duplicate detection.
        cfg: Validation configuration.

    Returns:
        List of validation error codes.
    """

    errors: list[str] = []
    if not case.case_id:
        errors.append("missing_id")
    if case.case_id in seen_ids:
        errors.append("duplicate_id")
    if not case.instruction:
        errors.append("missing_instruction")
    if case.origin is None:
        errors.append("missing_origin")
    if case.gt_destination is None:
        errors.append("missing_gt_destination")
    if not case.acceptable_destinations:
        errors.append("missing_acceptable_destinations")
    if cfg.require_ranked_destinations and not case.ranked_destinations:
        errors.append("missing_ranked_destinations")
    if case.gt_route and len(case.gt_route) < 2:
        errors.append("gt_route_too_short")
    if not case.gt_route:
        errors.append("missing_gt_route")
    if case.gt_distance_m is not None and case.gt_distance_m <= 0:
        errors.append("non_positive_gt_distance")

    errors.extend(_validate_route_alignment(case, cfg))
    errors.extend(_validate_ranked_destinations(case))
    errors.extend(_validate_acceptable_destinations(case))
    return errors


def _validate_route_alignment(case: UrbanNavCase, cfg: UrbanNavValidationConfig) -> list[str]:
    """Validate route start/end alignment.

    Args:
        case: Case to validate.
        cfg: Validation configuration.

    Returns:
        Route alignment error codes.
    """

    errors: list[str] = []
    if not case.gt_route:
        return errors
    if case.origin is not None:
        origin_error = haversine_m(case.origin, case.gt_route[0])
        if origin_error > cfg.origin_tolerance_m:
            errors.append(f"route_origin_far:{origin_error:.3f}")
    if case.gt_destination is not None and case.gt_destination.point is not None:
        endpoint_error = haversine_m(case.gt_route[-1], case.gt_destination.point)
        if endpoint_error > cfg.destination_tolerance_m:
            errors.append(f"route_endpoint_far:{endpoint_error:.3f}")
    route_distance = route_distance_m(case.gt_route)
    if case.gt_distance_m is None and route_distance is None:
        errors.append("missing_gt_distance")
    return errors


def _validate_ranked_destinations(case: UrbanNavCase) -> list[str]:
    """Validate ranked destination invariants.

    Args:
        case: Case to validate.

    Returns:
        Ranked destination error codes.
    """

    if not case.ranked_destinations:
        return []
    errors: list[str] = []
    ranks = [item.rank for item in case.ranked_destinations if item.rank is not None]
    if ranks and ranks != sorted(ranks):
        errors.append("ranked_destinations_rank_not_sorted")
    distances = [_rank_distance(item) for item in case.ranked_destinations]
    numeric_distances = [item for item in distances if item is not None]
    if len(numeric_distances) >= 2 and numeric_distances != sorted(numeric_distances):
        errors.append("ranked_destinations_distance_not_sorted")
    first = case.ranked_destinations[0]
    if case.gt_destination is not None and not _same_destination(first, case.gt_destination):
        errors.append("gt_destination_not_rank1")
    return errors


def _validate_acceptable_destinations(case: UrbanNavCase) -> list[str]:
    """Validate acceptable destination reference routes and distances.

    Args:
        case: Case to validate.

    Returns:
        Acceptable destination error codes.
    """

    errors: list[str] = []
    for index, destination in enumerate(case.acceptable_destinations):
        if destination.reference_route_distance_m is None or destination.reference_route_distance_m <= 0:
            errors.append(f"acceptable_missing_reference_distance:{index}")
        if not destination.reference_route:
            errors.append(f"acceptable_missing_reference_route:{index}")
    return errors


def _validate_cache_keys(
    case: UrbanNavCase,
    *,
    poi_cache_keys: set[str] | None,
    route_cache_keys: set[str] | None,
    cfg: UrbanNavValidationConfig,
) -> list[str]:
    """Validate replay cache key references.

    Args:
        case: Case to validate.
        poi_cache_keys: Available POI cache keys, or ``None`` to skip lookup.
        route_cache_keys: Available route cache keys, or ``None`` to skip lookup.
        cfg: Validation configuration.

    Returns:
        Cache key error codes.
    """

    errors: list[str] = []
    if cfg.require_cache_keys and not case.cache_keys:
        return ["missing_cache_keys"]
    for key in _as_str_list(case.cache_keys.get("poi_search")):
        if poi_cache_keys is not None and key not in poi_cache_keys:
            errors.append(f"missing_cache_key:poi:{key}")
    for key in _as_str_list(case.cache_keys.get("routes")):
        if route_cache_keys is not None and key not in route_cache_keys:
            errors.append(f"missing_cache_key:route:{key}")
    return errors


def _same_destination(a: UrbanNavDestination, b: UrbanNavDestination) -> bool:
    """Return whether two destinations represent the same POI.

    Args:
        a: First destination.
        b: Second destination.

    Returns:
        ``True`` if stable ids overlap or coordinates are exactly equal.
    """

    if a.stable_ids and b.stable_ids and a.stable_ids & b.stable_ids:
        return True
    return a.lat == b.lat and a.lng == b.lng and a.name == b.name


def _rank_distance(destination: UrbanNavDestination) -> float | None:
    """Return the numeric rank distance for sorting validation.

    Args:
        destination: Ranked destination.

    Returns:
        Route distance from ranked field or reference distance.
    """

    return destination.route_distance_m or destination.reference_route_distance_m


def _as_str_list(value: Any) -> list[str]:
    """Normalize cache key field to a string list.

    Args:
        value: Raw cache key field.

    Returns:
        List of string cache keys.
    """

    if isinstance(value, list):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return []
