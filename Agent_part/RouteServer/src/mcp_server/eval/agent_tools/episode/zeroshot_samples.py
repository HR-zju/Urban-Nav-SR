"""Canonical event-driven zero-shot rows generated from RouteServer contracts."""

from __future__ import annotations

import copy
import json
from typing import Any

from mcp_server.agent.stage2 import append_system1_readout
from mcp_server.agent.tool_schema import get_tool_definitions


def build_event_driven_zeroshot_rows() -> list[dict[str, Any]]:
    """Build three contract-grounded System2 smoke rows.

    Returns:
        Route update, ambiguous instruction and resume scenarios. Tool schemas
        come directly from ``ToolSpecRegistry`` rather than a copied fixture.
    """

    tools = [
        definition
        for definition in get_tool_definitions()
        if definition.get("function", {}).get("name") in {"map_tool", "user_interaction_tool"}
    ]
    scenarios = [
        {
            "case_id": "zeroshot-modify-route",
            "episode_id": "zeroshot-episode-1",
            "system": (
                "You are UrbanNav System2. Use the supplied runtime state and tools. Never continue a stale "
                "action. Output exactly one canonical <tool_call>{...}</tool_call> when external state must change, "
                "or one <finish> with decision_type=navigate followed by the [SYS1_0] ... [SYS1_15] readout "
                "for a local navigation decision."
            ),
            "history": [
                {
                    "role": "user",
                    "content": '<environment_observation>{"fpv_summary":"clear sidewalk","instruction":"Walk to the cafe"}</environment_observation>',
                },
                {
                    "role": "assistant",
                    "content": _navigate_target(route_action="continue_navigation", intent="go_straight"),
                },
            ],
            "event": {
                "event_id": "event:update:1",
                "event_type": "modify_task",
                "content": {"active_instruction": "Go to the library and add a pharmacy as a via stop"},
                "expected_task_revision": 0,
            },
            "runtime": {
                "active_task": {
                    "active_instruction": "Go to the library and add a pharmacy as a via stop",
                    "task_revision": 1,
                    "route_session_id": "route-1",
                    "route_revision": 0,
                    "status": "active",
                },
                "dialogue": {"pending_interaction": None},
                "execution": {"control_epoch": 1, "status": "idle"},
            },
            "observation": {"fpv_summary": "robot stopped on a clear sidewalk", "observation_id": "obs-1"},
            "target": '<tool_call>{"name":"map_tool","arguments":{"op":"search_destination","query":"pharmacy"}}</tool_call>',
        },
        {
            "case_id": "zeroshot-ambiguous-update",
            "episode_id": "zeroshot-episode-2",
            "system": (
                "You are UrbanNav System2. Use the supplied runtime state and tools. Never guess an ambiguous "
                "destination. Output exactly one canonical <tool_call>{...}</tool_call>, or a decision_type=navigate "
                "<finish> followed by the [SYS1_0] ... [SYS1_15] readout."
            ),
            "history": [
                {
                    "role": "user",
                    "content": '<environment_observation>{"fpv_summary":"sidewalk junction","instruction":"Walk to the station"}</environment_observation>',
                },
                {
                    "role": "assistant",
                    "content": _navigate_target(route_action="continue_navigation", intent="go_straight"),
                },
            ],
            "event": {
                "event_id": "event:update:2",
                "event_type": "new_instruction",
                "content": "Go to the other entrance instead",
                "expected_task_revision": 0,
            },
            "runtime": {
                "active_task": {
                    "active_instruction": "Go to the other entrance instead",
                    "task_revision": 1,
                    "route_session_id": "route-1",
                    "route_revision": 0,
                    "status": "active",
                },
                "dialogue": {"pending_interaction": None},
                "execution": {"control_epoch": 1, "status": "idle"},
            },
            "observation": {
                "fpv_summary": "two entrances are not visually distinguishable",
                "observation_id": "obs-2",
            },
            "target": '<tool_call>{"name":"user_interaction_tool","arguments":{"op":"request_clarification","question":"Which entrance do you mean?","requires_user_response":true}}</tool_call>',
        },
        {
            "case_id": "zeroshot-resume-navigation",
            "episode_id": "zeroshot-episode-3",
            "system": (
                "You are UrbanNav System2. Use the supplied runtime state and tools. A resume invalidates the old "
                "action chunk; reason from the latest observation. Output exactly one canonical "
                "<tool_call>{...}</tool_call>, or a decision_type=navigate <finish> followed by the "
                "[SYS1_0] ... [SYS1_15] readout."
            ),
            "history": [],
            "event": {
                "event_id": "event:resume:3",
                "event_type": "resume",
                "content": None,
                "expected_task_revision": 0,
            },
            "runtime": {
                "active_task": {
                    "active_instruction": "Continue to the station",
                    "task_revision": 0,
                    "route_session_id": "route-1",
                    "route_revision": 0,
                    "status": "active",
                },
                "dialogue": {"pending_interaction": None},
                "execution": {"control_epoch": 2, "status": "idle"},
            },
            "observation": {"fpv_summary": "clear sidewalk toward the station", "observation_id": "obs-new"},
            "target": _navigate_target(
                route_action="continue_navigation",
                intent="go_straight",
                confidence_level="medium",
            ),
        },
    ]
    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        context = (
            "<ExternalEvent>"
            + _compact_json(scenario["event"])
            + "</ExternalEvent>\n<RuntimeState>"
            + _compact_json(scenario["runtime"])
            + "</RuntimeState>\n<environment_observation>"
            + _compact_json(scenario["observation"])
            + "</environment_observation>"
        )
        rows.append({
            "case_id": scenario["case_id"],
            "episode_id": scenario["episode_id"],
            "messages": [
                {"role": "system", "content": scenario["system"]},
                *copy.deepcopy(scenario["history"]),
                {"role": "user", "content": context},
                {"role": "assistant", "content": scenario["target"]},
            ],
            "tools": copy.deepcopy(tools),
            "images": [],
            "sidewalk_meta": {
                "prompt_template": "agentic_stage2_cot_trajectory",
                "episode_training_view": "urbannav_episode_per_decision_v3",
                "decision_id": f"decision:{scenario['episode_id']}:{index + 1}",
            },
        })
    return rows


def _compact_json(value: Any) -> str:
    """Serialize deterministic model-visible context without presentation whitespace."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _navigate_target(**fields: Any) -> str:
    """Build one canonical navigation decision with the masked System1 readout suffix."""

    finish = {"decision_type": "navigate", **fields}
    return append_system1_readout(f"<finish>{_compact_json(finish)}</finish>")
