"""Regression tests for UrbanNav final-25K v2 invariants."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from mcp_server.core.replay_map_backend import ReplayRouteResult
from mcp_server.eval.urbannav.urbannav_metrics import evaluate_multi_stop_case
from mcp_server.eval.urbannav.urbannav_runners import _best_origin_matched_route
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavCase, UrbanNavDestination, UrbanNavPoint, UrbanNavPrediction


ROUTE_SERVER_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROUTE_SERVER_ROOT / "tools" / "dataset_processing" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_capacity_limited_quota_allocation_is_simultaneous() -> None:
    finalizer = _load_script("finalize_urbannav_benchmark_package")
    quotas = finalizer._allocate_quotas(
        available={
            "poi_navigation": 10_000,
            "implicit_need_navigation": 10_000,
            "multi_stop_route": 10_000,
            "route_update_add_via": 587,
        },
        target_cases=15_000,
        weights=finalizer.DEFAULT_CATEGORY_WEIGHTS,
    )
    assert quotas == {
        "implicit_need_navigation": 5605,
        "multi_stop_route": 2402,
        "poi_navigation": 6406,
        "route_update_add_via": 587,
    }


def test_v2_distribution_and_quality_policy_compile_exactly() -> None:
    builder = _load_script("build_urbannav_instruction_set")
    distribution = json.loads((ROUTE_SERVER_ROOT / "config" / "urbannav_25k_v2_distribution_plan.json").read_text())
    regions = json.loads((ROUTE_SERVER_ROOT / "config" / "urbannav_global_gt_collection_full_en.json").read_text())
    quality = json.loads((ROUTE_SERVER_ROOT / "config" / "urbannav_25k_v2_quality_policy.json").read_text())
    slots = builder._city_slots(distribution, regions_config=regions, seed=7, mode="exact")
    assert len(slots) == 25_000
    assert sum(item["city"] == "amsterdam" for item in slots) == 1471
    tasks = builder.compile_tasks(
        taxonomy=json.loads((ROUTE_SERVER_ROOT / "config" / "yelp_demand_taxonomy_mined_v2.json").read_text()),
        provider_map=json.loads((ROUTE_SERVER_ROOT / "config" / "urbannav_target_type_provider_map_mined_v2.json").read_text()),
        task_quotas=builder._task_quotas(distribution, include_semantic_analysis=False),
        route_update_action_quotas=builder._route_update_action_quotas(distribution),
        language="en",
        allow_missing_target_types=False,
    )
    tasks = builder._apply_quality_policy(tasks, quality_policy=quality)
    multi = [task for task in tasks if task.category == "multi_stop_route"]
    assert sum(task.planned_count for task in multi) == 3750
    assert all(len(task.ordered_goal_labels) == len(task.ordered_target_types) for task in multi)
    assert all(len(set(task.ordered_goal_labels)) == len(task.ordered_goal_labels) for task in multi)
    for task in multi:
        groups = [set(group) for group in task.ordered_target_types]
        assert all(not groups[left].intersection(groups[right]) for left in range(len(groups)) for right in range(left + 1, len(groups)))
    update_actions = {
        action: sum(task.planned_count for task in tasks if task.route_update_action == action)
        for action in {"replace_destination", "add_via", "insert_before_final"}
    }
    assert update_actions == {"replace_destination": 1000, "add_via": 1000, "insert_before_final": 500}


def test_with_rank_preserves_provider_scope_metadata() -> None:
    collector = _load_script("collect_urbannav_global_gt")
    destination = UrbanNavDestination(
        lat=30.0,
        lng=120.0,
        provider_poi_id="osm:node/1",
        raw={"osm_local": {"scope": "hangzhou"}},
    )
    ranked = collector._with_rank(destination, rank=2)
    assert ranked.rank == 2
    assert ranked.raw == destination.raw


def test_replay_route_rejects_wrong_origin() -> None:
    destination = UrbanNavDestination(lat=30.0, lng=120.01, provider_poi_id="poi-1")
    record = ReplayRouteResult(
        cache_key="route-1",
        request={"origin": {"lat": 31.0, "lng": 121.0}, "destination": destination.to_dict()},
        polyline=[UrbanNavPoint(lat=31.0, lng=121.0), UrbanNavPoint(lat=30.0, lng=120.01)],
        distance_m=100.0,
    )
    assert _best_origin_matched_route(
        [record],
        origin=UrbanNavPoint(lat=30.0, lng=120.0),
        origin_tolerance_m=5.0,
    ) is None


def test_disconnected_multistop_legs_fail_route_metric() -> None:
    case = UrbanNavCase.from_dict(
        {
            "id": "disconnected",
            "category": "multi_stop_route",
            "instruction": "Visit food, then a bank.",
            "origin": {"lat": 30.0, "lng": 120.0},
            "gt_destination": {"provider_poi_id": "bank", "lat": 30.0, "lng": 120.002},
            "gt_destination_sequence": [
                {"provider_poi_id": "food", "lat": 30.0, "lng": 120.001},
                {"provider_poi_id": "bank", "lat": 30.0, "lng": 120.002},
            ],
            "gt_route_legs": [
                {"leg_index": 1, "gt_destination": {"provider_poi_id": "food", "lat": 30.0, "lng": 120.001, "reference_route_distance_m": 100.0}, "acceptable_destinations": [{"provider_poi_id": "food", "lat": 30.0, "lng": 120.001, "reference_route_distance_m": 100.0}], "distance_m": 100.0},
                {"leg_index": 2, "gt_destination": {"provider_poi_id": "bank", "lat": 30.0, "lng": 120.002, "reference_route_distance_m": 100.0}, "acceptable_destinations": [{"provider_poi_id": "bank", "lat": 30.0, "lng": 120.002, "reference_route_distance_m": 100.0}], "distance_m": 100.0},
            ],
            "gt_route": [[120.0, 30.0], [120.001, 30.0], [120.002, 30.0]],
            "gt_distance_m": 200.0,
        }
    )
    prediction = UrbanNavPrediction.from_dict(
        {
            "id": "disconnected",
            "chosen_destination": {"provider_poi_id": "bank", "lat": 30.0, "lng": 120.002},
            "pred_destination_sequence": [
                {"provider_poi_id": "food", "lat": 30.0, "lng": 120.001},
                {"provider_poi_id": "bank", "lat": 30.0, "lng": 120.002},
            ],
            "pred_route_legs": [
                {"route": [[120.0, 30.0], [120.001, 30.0]], "route_summary": {"distance_m": 100.0}},
                {"origin": {"lat": 30.0, "lng": 120.01}, "route": [[120.01, 30.0], [120.002, 30.0]], "route_summary": {"distance_m": 100.0}},
            ],
            "route_summary": {"distance_m": 200.0},
        }
    )
    metrics = evaluate_multi_stop_case(case, prediction)
    assert metrics["sequence_success"]
    assert not metrics["leg_continuity_success"]
    assert not metrics["leg_route_success"]
    assert metrics["multi_stop_spl"] == 0.0
