#!/usr/bin/env python3
"""Build SideWalk Agentic Stage2 review traces and ms-swift rows.

输入:
    ``--cases`` 指向 ``sidewalk_agent_stage2_case_v1`` JSONL 或
    ``--phase1-archive-jsonl`` 指向 Phase1 archive/pre-archive JSONL。

输出:
    ``--traces-out``: ``sidewalk_agent_trace_v2`` JSONL。
    ``--ms-swift-out``: ``sidewalk_agentic_stage2_sft_v1`` JSONL。
    ``--report-out``: case/trace/export row 的 validator report。

说明:
    默认不执行真实工具，使用 deterministic synthetic tool_response。传
    ``--execute-tools`` 后会调用 ``ToolExecutor``。``--tool-context fake`` 用本地
    假地图客户端跑真实 executor 代码路径，适合 smoke ``provider`` 会创建真实
    RouteServer map provider 可能依赖网络/API key。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_server.eval.cases import load_stage2_cases
from mcp_server.eval.exporters import (
    trace_to_ms_swift_stage2_row,
    write_ms_swift_stage2_jsonl,
    write_traces_jsonl,
)
from mcp_server.eval.phase1_archive_adapter import load_phase1_archive_stage2_cases
from mcp_server.eval.runners import RouteServerTeacherRunner, run_cases
from mcp_server.eval.stage2_validator import (
    validate_ms_swift_stage2_row,
    validate_stage2_case,
    validate_stage2_trace,
)


def _metric_average(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 6)


def _validation_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(items),
        "valid": sum(1 for item in items if item.get("valid")),
        "invalid": sum(1 for item in items if not item.get("valid")),
    }


class FakeStage2MapClient:
    """Small in-process map client for ToolExecutor smoke tests.

    输入/输出接口与 ``MapProviderClient`` 的常用方法一致，但完全不访问网络。
    它不是 synthetic tool_response：调用链仍然经过 ``ToolExecutor``、
    ``map_ops`` 和 ``route_session_ops``。
    """

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def geocode(self, query: str) -> list[dict[str, Any]]:
        lon, lat = _stable_location(query)
        return [{"name": query, "lat": lat, "lon": lon, "source": "fake_stage2_map"}]

    async def reverse_geocode(self, lat: float, lon: float) -> dict[str, Any]:
        return {"address": f"Fake address near {lat:.6f},{lon:.6f}", "lat": lat, "lon": lon}

    async def search_semantic_pois(
        self,
        location: str,
        keywords: str,
        radius: int = 1000,
        sortrule: str = "distance",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        lon, lat = _parse_location_or_default(location)
        count = max(1, min(int(limit or 1), 3))
        return [
            {
                "id": f"fake-poi-{index}",
                "name": f"{keywords or 'sidewalk'} candidate {index + 1}",
                "location": f"{lon + index * 0.0001:.6f},{lat + index * 0.0001:.6f}",
                "distance_m": 60 + index * 45,
                "source": "fake_stage2_map",
                "sortrule": sortrule,
                "radius": radius,
            }
            for index in range(count)
        ]

    async def get_coarse_waypoints(self, origin: str, destination: str, mode: str = "foot") -> dict[str, Any]:
        origin_lon, origin_lat = _parse_location_or_default(origin)
        dest_lon, dest_lat = _parse_location_or_default(destination, fallback=(origin_lon + 0.001, origin_lat + 0.001))
        distance_m = _haversine_m(origin_lat, origin_lon, dest_lat, dest_lon)
        duration_s = int(round(distance_m / 1.25))
        return {
            "provider": "fake_stage2_map",
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "route_summary": {
                "distance_m": round(distance_m, 1),
                "duration_s": duration_s,
                "duration_min": round(duration_s / 60.0, 1),
                "step_count": 1,
            },
            "coarse_waypoints": [
                {
                    "instruction": f"Proceed to {destination}",
                    "action": "arrive",
                    "location": {"longitude": dest_lon, "latitude": dest_lat},
                    "distance_m": round(distance_m, 1),
                    "duration_s": duration_s,
                }
            ],
            "route_polyline": [[origin_lon, origin_lat], [dest_lon, dest_lat]],
        }

    async def get_route(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        mode: str = "car",
        steps: bool = False,
        overview: str = "overview",
        annotations: bool = True,
    ) -> dict[str, Any]:
        distance_m = _haversine_m(from_lat, from_lon, to_lat, to_lon)
        return {
            "routes": [
                {
                    "distance": distance_m,
                    "duration": distance_m / 1.25,
                    "geometry": {"coordinates": [[from_lon, from_lat], [to_lon, to_lat]]},
                    "legs": [
                        {
                            "steps": [
                                {
                                    "maneuver": {
                                        "instruction": "Proceed to the destination.",
                                        "location": [to_lon, to_lat],
                                    },
                                    "distance": distance_m,
                                    "duration": distance_m / 1.25,
                                    "name": "fake route",
                                }
                            ]
                            if steps
                            else []
                        }
                    ],
                }
            ],
            "waypoints": [
                {"location": [from_lon, from_lat], "name": "origin"},
                {"location": [to_lon, to_lat], "name": "destination"},
            ],
            "overview": overview,
            "annotations": annotations,
            "mode": mode,
        }

    async def get_nearby_pois(
        self,
        lat: float,
        lon: float,
        radius: float = 1000,
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        keyword = (categories or ["poi"])[0]
        return await self.search_semantic_pois(location=f"{lon},{lat}", keywords=keyword, radius=int(radius), limit=3)

    async def search_features_by_category(
        self,
        bbox: tuple[float, float, float, float],
        category: str,
        subcategories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        min_lon, min_lat, max_lon, max_lat = bbox
        return [
            {
                "id": "fake-feature-0",
                "type": "node",
                "lat": (min_lat + max_lat) / 2.0,
                "lon": (min_lon + max_lon) / 2.0,
                "tags": {"name": f"fake {category}", category: (subcategories or [category])[0]},
            }
        ]


def _parse_location_or_default(value: Any, *, fallback: tuple[float, float] = (120.0, 30.0)) -> tuple[float, float]:
    text = str(value or "").strip()
    if "," not in text:
        return fallback
    try:
        lon_text, lat_text = text.split(",", 1)
        return float(lon_text), float(lat_text)
    except Exception:
        return fallback


def _stable_location(seed: Any) -> tuple[float, float]:
    text = str(seed or "stage2")
    offset = (sum(ord(ch) for ch in text) % 1000) / 1_000_000.0
    return 120.0 + offset, 30.0 + offset


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(a))


async def _build_app_context(args: argparse.Namespace):
    if not args.execute_tools or args.tool_context == "none":
        return None
    if args.tool_context == "fake":
        os.environ.setdefault("ROUTE_SESSION_STORE", "memory")
        return SimpleNamespace(provider="fake_stage2_map", osm_client=FakeStage2MapClient())
    if args.tool_context == "provider":
        from mcp_server.map_providers import create_map_client

        client, provider = create_map_client()
        await client.connect()
        return SimpleNamespace(provider=provider, osm_client=client)
    raise ValueError(f"Unsupported tool context: {args.tool_context}")


async def _close_app_context(app_context: Any | None) -> None:
    client = getattr(app_context, "osm_client", None)
    disconnect = getattr(client, "disconnect", None)
    if disconnect is not None:
        result = disconnect()
        if asyncio.iscoroutine(result):
            await result


def _load_cases_from_args(args: argparse.Namespace):
    if args.phase1_archive_jsonl:
        return load_phase1_archive_stage2_cases(
            args.phase1_archive_jsonl,
            limit=args.limit,
            case_id_prefix=args.phase1_case_id_prefix,
            resolve_relative_paths=not args.preserve_phase1_relative_paths,
        )
    cases = load_stage2_cases(args.cases)
    if args.limit is not None:
        cases = cases[: int(args.limit)]
    return cases


async def _run(args: argparse.Namespace) -> None:
    cases = _load_cases_from_args(args)
    case_validations = [{"case_id": case.case_id, **validate_stage2_case(case).to_dict()} for case in cases]

    app_context = await _build_app_context(args)
    try:
        runner = RouteServerTeacherRunner(
            use_llm=bool(args.use_llm),
            execute_tools=bool(args.execute_tools),
            enable_visual_prompt=bool(args.enable_visual_prompt),
            enable_recommendation=bool(args.enable_recommendation),
            app_context=app_context,
            tool_artifact_dir=args.tool_artifact_dir,
        )
        traces = await run_cases(runner, cases)
    finally:
        await _close_app_context(app_context)

    cases_by_id = {case.case_id: case for case in cases}
    rows = [trace_to_ms_swift_stage2_row(trace, case=cases_by_id.get(trace.case_id)) for trace in traces]

    trace_validations = [
        {"case_id": trace.case_id, **validate_stage2_trace(trace, case=cases_by_id.get(trace.case_id)).to_dict()}
        for trace in traces
    ]
    row_validations = [
        {"case_id": row.get("sidewalk_meta", {}).get("case_id"), **validate_ms_swift_stage2_row(row).to_dict()}
        for row in rows
    ]

    write_traces_jsonl(traces, args.traces_out)
    write_ms_swift_stage2_jsonl(traces, args.ms_swift_out, cases_by_id=cases_by_id)

    metric_names = sorted({key for trace in traces for key in trace.metrics.keys()})
    report = {
        "schema_version": "sidewalk_agent_stage2_review_report_v1",
        "input_mode": "phase1_archive" if args.phase1_archive_jsonl else "stage2_cases",
        "input_path": str(args.phase1_archive_jsonl or args.cases),
        "case_count": len(cases),
        "trace_count": len(traces),
        "runner": "routeserver_teacher_tool_executor" if args.execute_tools else "routeserver_teacher_offline_synthetic_tool_response",
        "tool_context": args.tool_context if args.execute_tools else "synthetic",
        "use_llm": bool(args.use_llm),
        "enable_visual_prompt": bool(args.enable_visual_prompt),
        "enable_recommendation": bool(args.enable_recommendation),
        "outputs": {
            "traces_out": str(args.traces_out),
            "ms_swift_out": str(args.ms_swift_out),
            "report_out": str(args.report_out),
            "tool_artifact_dir": str(args.tool_artifact_dir) if args.tool_artifact_dir else None,
        },
        "metrics": {
            name: _metric_average([trace.metrics.get(name) for trace in traces])
            for name in metric_names
        },
        "validation": {
            "cases": _validation_counts(case_validations),
            "traces": _validation_counts(trace_validations),
            "ms_swift_rows": _validation_counts(row_validations),
        },
        "cases": [
            {
                "case_id": trace.case_id,
                "metrics": trace.metrics,
                "case_validation": case_validations[index],
                "trace_validation": trace_validations[index],
                "row_validation": row_validations[index],
                "selected_tool_call": (
                    trace.metadata.get("teacher_internal", {}).get("selected_student_tool_call")
                    if isinstance(trace.metadata, dict)
                    else None
                ),
                "tool_execution_warning": (
                    trace.metadata.get("tool_execution_warning") if isinstance(trace.metadata, dict) else None
                ),
            }
            for index, trace in enumerate(traces)
        ],
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    invalid_count = (
        report["validation"]["cases"]["invalid"]
        + report["validation"]["traces"]["invalid"]
        + report["validation"]["ms_swift_rows"]["invalid"]
    )
    if invalid_count and args.fail_on_invalid:
        raise SystemExit(f"Stage2 review build found {invalid_count} invalid validation entries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default=str(ROOT / "examples" / "agent_eval" / "sidewalk_agent_stage2_cases_review.jsonl"),
    )
    parser.add_argument(
        "--phase1-archive-jsonl",
        default=None,
        help="Optional Phase1 archive/pre-archive JSONL to adapt into Stage2 cases.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--phase1-case-id-prefix", default="phase1-stage2")
    parser.add_argument(
        "--preserve-phase1-relative-paths",
        action="store_true",
        help="Keep Phase1 archive relative image/sidecar paths instead of resolving against the manifest directory.",
    )
    parser.add_argument(
        "--traces-out",
        default=str(ROOT / "examples" / "agent_eval" / "sidewalk_agent_stage2_traces_review.jsonl"),
    )
    parser.add_argument(
        "--ms-swift-out",
        default=str(ROOT / "examples" / "agent_eval" / "sidewalk_agentic_stage2_ms_swift_review.jsonl"),
    )
    parser.add_argument(
        "--report-out",
        default=str(ROOT / "examples" / "agent_eval" / "sidewalk_agent_stage2_report_review.json"),
    )
    parser.add_argument("--execute-tools", action="store_true", help="Run selected tool_call through ToolExecutor.")
    parser.add_argument(
        "--tool-context",
        choices=["none", "fake", "provider"],
        default="none",
        help="ToolExecutor app_context source. fake is deterministic local smoke; provider uses configured RouteServer provider.",
    )
    parser.add_argument(
        "--tool-artifact-dir",
        default=str(ROOT / "examples" / "agent_eval" / "artifacts"),
        help="Directory for image_tool artifacts when --execute-tools is used.",
    )
    parser.add_argument("--use-llm", action="store_true", help="Allow RouteServer teacher LLM intent/decomposition calls.")
    parser.add_argument("--enable-visual-prompt", action="store_true")
    parser.add_argument("--enable-recommendation", action="store_true")
    parser.add_argument("--no-fail-on-invalid", dest="fail_on_invalid", action="store_false")
    parser.set_defaults(fail_on_invalid=True)
    args = parser.parse_args()
    if args.phase1_archive_jsonl and args.cases != parser.get_default("cases"):
        raise SystemExit("Use either --cases or --phase1-archive-jsonl, not both.")
    if args.tool_context == "provider" and not args.execute_tools:
        raise SystemExit("--tool-context provider requires --execute-tools")
    if args.tool_artifact_dir:
        args.tool_artifact_dir = str(Path(args.tool_artifact_dir).expanduser())
    return args


def main() -> None:
    asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    main()
