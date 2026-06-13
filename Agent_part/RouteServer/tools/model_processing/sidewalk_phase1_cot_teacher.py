#!/usr/bin/env python3
"""Generate validated Phase1 structured CoT targets for ms-swift.

This script is the second offline stage after ``sidewalk_phase1_dataset.py``.
It calls the Agent repository's configured VLM endpoint to produce a single
assistant answer with strict tags:

    <think>
    <Scene>{...}</Scene>
    <FPVScene>{...}</FPVScene>
    <BEVContext>{...}</BEVContext>
    <CrossView>{...}</CrossView>
    <Plan>{...}</Plan>
    </think>
    <finish>{...}</finish>

Geometry targets remain authoritative.  The VLM may explain the sample, but it
must not change ``intent_gt``, target direction, or ``pixel_trajectory``.
Rows that fail validation are retried; after that they are dropped, rejected, or
explicitly downgraded to a deterministic template according to ``--fallback-policy``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
THINK_SECTION_TAGS = ["Scene", "FPVScene", "BEVContext", "CrossView", "Plan"]
REQUIRED_TAGS = ["think", *THINK_SECTION_TAGS, "finish"]
COT_SCHEMA_VERSION = "sidewalk_phase1_cot_teacher_v5"
COT_PROMPT_TEMPLATE = "phase1_structured_cot_v5"
COT_BEV_VISUAL_CONVENTION = (
    "BEV visual convention:\n"
    "- Use the BEV image as static top-down route context for direction and topology.\n"
    "- Ground local obstacle avoidance, sidewalk boundaries, and spatial maneuvering in the FPV image.\n"
    "- Blue polyline = planned route; green circle = route start; red square = route end; orange circle = current ego pose.\n"
)
COT_TRAJECTORY_TYPE_ENUM = (
    "straight_left|straight_center|straight_right|"
    "slight_left_left|slight_left_center|slight_left_right|"
    "slight_right_left|slight_right_center|slight_right_right|"
    "stop_hold|unknown"
)
NO_SEG_FORBIDDEN_PATTERNS = [
    r"\br\d{2,}\b",
    r"\bregion_id\b",
    r"\bmask_id\b",
    r"\bforbidden_overlap\b",
    r"\bsupport_ratio\b",
    r"\bmiou\b",
    r"\bsegmentation\s+mask\b",
]


@dataclass(frozen=True)
class TeacherConfig:
    expected_points: int = 16
    max_retries: int = 2
    trajectory_tolerance: float = 0.025
    theta_tolerance_deg: float = 2.0
    fallback_policy: str = "drop"
    image_detail: str = "auto"


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]
    sections: dict[str, Any]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: str | None, report: dict[str, Any]) -> None:
    if not path:
        return
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def telemetry_llm_metadata(telemetry: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(telemetry, dict):
        return {}
    llm_telemetry = telemetry.get("llm_telemetry")
    if isinstance(llm_telemetry, dict):
        return llm_telemetry
    return telemetry


def telemetry_usage(telemetry: dict[str, Any] | None) -> dict[str, Any]:
    llm_telemetry = telemetry_llm_metadata(telemetry)
    response = llm_telemetry.get("response")
    if isinstance(response, dict) and isinstance(response.get("usage"), dict):
        return response["usage"]
    usage = llm_telemetry.get("usage")
    return usage if isinstance(usage, dict) else {}


def telemetry_cached_tokens(telemetry: dict[str, Any] | None) -> int | None:
    llm_telemetry = telemetry_llm_metadata(telemetry)
    for value in (
        llm_telemetry.get("prompt_cached_tokens"),
        llm_telemetry.get("cached_tokens"),
    ):
        parsed = optional_int(value)
        if parsed is not None:
            return parsed
    response = llm_telemetry.get("response")
    if isinstance(response, dict):
        parsed = optional_int(response.get("cached_tokens"))
        if parsed is not None:
            return parsed
    return None


def add_usage_to_report(report: dict[str, Any], telemetry: dict[str, Any] | None) -> None:
    usage = telemetry_usage(telemetry)
    if not usage:
        return
    totals = report.setdefault(
        "usage_totals",
        {
            "calls_with_usage": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        },
    )
    prompt_tokens = optional_int(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("prompt_token_count")
    )
    completion_tokens = optional_int(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("completion_token_count")
    )
    total_tokens = optional_int(usage.get("total_tokens") or usage.get("total_token_count"))
    cached_tokens = telemetry_cached_tokens(telemetry)
    totals["calls_with_usage"] = int(totals.get("calls_with_usage") or 0) + 1
    if prompt_tokens is not None:
        totals["prompt_tokens"] = int(totals.get("prompt_tokens") or 0) + prompt_tokens
    if completion_tokens is not None:
        totals["completion_tokens"] = int(totals.get("completion_tokens") or 0) + completion_tokens
    if total_tokens is not None:
        totals["total_tokens"] = int(totals.get("total_tokens") or 0) + total_tokens
    if cached_tokens is not None:
        totals["cached_tokens"] = int(totals.get("cached_tokens") or 0) + cached_tokens


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def load_evidence(row: dict[str, Any]) -> dict[str, Any]:
    """Load geometry evidence produced by the Phase1 exporter."""

    label = load_json(row["trajectory_label_path"])
    bev_meta = load_json(row["bev_patch_meta_path"])
    trajectory = np.asarray(row.get("pixel_trajectory") or label.get("trajectory_norm"), dtype=np.float32).reshape(-1, 2)
    return {
        "route_context": row.get("route_context") or {},
        "trajectory_label": label,
        "bev_patch_meta": bev_meta,
        "target_trajectory": trajectory.astype(float).tolist(),
        "target_trajectory_source": label.get("trajectory_source") or (row.get("sidewalk_meta") or {}).get("trajectory_source"),
    }


def build_teacher_system_prompt() -> str:
    return (
        "You are a careful sidewalk navigation dataset teacher. "
        "Generate one structured chain-of-thought annotation for supervised fine-tuning. "
        "Use the provided FPV image, BEV image, and geometry evidence. "
        "Fully use both FPV and BEV environmental information and understand their cross-view relationship. "
        "The geometry evidence is ground truth: do not alter intent, target direction, or trajectory. "
        "Return only the required XML-like tags. Put structured observation and planning JSON sections inside <think>; "
        "put only the executable trajectory JSON inside <finish>."
    )


def build_teacher_question(row: dict[str, Any], evidence: dict[str, Any], *, previous_answer: str | None = None, errors: list[str] | None = None) -> str:
    """Build the VLM teacher prompt.

    This prompt is never used for ms-swift training.  It may contain ground
    truth evidence because its sole role is producing an offline teacher answer.
    """

    route_context = evidence["route_context"]
    bev_meta = evidence["bev_patch_meta"]
    label = evidence["trajectory_label"]
    target_traj = round_trajectory(evidence["target_trajectory"], digits=5)
    repair_block = ""
    if previous_answer is not None:
        repair_block = (
            "\n\nPrevious invalid answer:\n"
            f"{previous_answer}\n\n"
            "Validation errors:\n"
            f"{json.dumps(errors or [], ensure_ascii=False)}\n"
            "Repair the answer while preserving all ground-truth fields."
        )
    payload = {
        "sample_id": row.get("id"),
        "instruction": row.get("instruction"),
        "image_size": {"width": row.get("image_width"), "height": row.get("image_height")},
        "route_context": route_context,
        "bev_patch_meta": {
            "intent_gt": bev_meta.get("intent_gt"),
            "target_direction": bev_meta.get("target_direction"),
            "confidence_level_gt": bev_meta.get("confidence_level_gt"),
        },
        "trajectory_label": {
            "trajectory_source": label.get("trajectory_source"),
            "intent_gt": label.get("intent_gt"),
            "anchor": label.get("anchor"),
            "trajectory_norm": target_traj,
        },
        "phase1_limitations": {
            "has_segmentation": False,
            "allowed": "Describe visible walkable support, risk objects, BEV topology, cross-view alignment, and a conservative plan.",
            "forbidden": "Do not claim region ids, segmentation masks, mIoU, support_ratio, or mask overlap.",
        },
    }
    return (
        "Create a structured CoT assistant target for this sidewalk navigation sample.\n"
        "Output exactly this nested tag structure in this order. Prefer compact machine-learnable labels over decorative prose:\n"
        "<think>\n"
        "<Scene>{\"summary\":\"...\",\"visibility\":\"clear|partial_occlusion|low_light|crowded|uncertain\",\"occlusion_sources\":[\"parked_vehicle|tree|fence|pedestrian|vehicle|none|unknown\"],\"uncertainty\":\"...\"}</Scene>\n"
        "<FPVScene>{\"navigable_support\":[\"stable semantic names for visible safe navigation support\"],\"conditional_support\":[\"stable semantic names for conditionally usable connectors\"],\"risk_objects\":[{\"name\":\"stable object name\",\"side\":\"left|center|right|front|unknown\",\"risk\":\"boundary|avoid|occlusion|dynamic|low\"}],\"visibility\":\"clear|partial_occlusion|low_light|crowded|uncertain\",\"local_risk\":\"low|medium|high|unknown\"}</FPVScene>\n"
        "<BEVContext>{\"scene_summary\":\"local BEV route context\",\"route_summary\":\"route continues from ego toward target\",\"route_direction_bin\":\"front|slight_left|slight_right|left|right|stop|unknown\",\"intent\":\"go_straight|turn_left|turn_right|stop\",\"target_direction\":\"front\",\"bev_anchor\":\"front route segment\"}</BEVContext>\n"
        "<CrossView>{\"bev_anchor\":\"...\",\"fpv_anchor\":\"...\",\"near_reference\":\"nearest stable visible reference\",\"blind_area_context\":{\"left\":\"occluded/visible risk statement\",\"right\":\"occluded/visible risk statement\",\"front\":\"visible support or occlusion statement\"},\"fpv_route_region\":\"left|center|right|unknown\",\"route_to_fpv_direction\":\"forward_left|forward_center|forward_right|stop|unknown\",\"safe_corridor\":\"left_corridor|center_corridor|right_corridor|stop|unknown\",\"alignment\":\"aligned|partially_aligned|uncertain\",\"confidence\":\"high|medium|low|unknown\"}</CrossView>\n"
        "<Plan>{\"motion_primitive\":\"go_straight|turn_left|turn_right|stop|unknown\",\"lateral_preference\":\"left|center|right|hold|unknown\",\"avoidance_side\":\"left|right|both|none|unknown\",\"speed_policy\":\"normal|slow|stop|unknown\",\"primary_route\":\"concise semantic route\",\"safety_policy\":\"concise safety policy\",\"contingency\":\"concise fallback behavior\"}</Plan>\n"
        "</think>\n"
        f"<finish>{{\"trajectory\":[[0.5,0.9]],\"intent\":\"go_straight|turn_left|turn_right|stop|unknown\",\"confidence_level\":\"normal|conflict|low\",\"trajectory_type\":\"{COT_TRAJECTORY_TYPE_ENUM}\",\"trajectory_validity\":\"valid|invalid\"}}</finish>\n\n"
        "Rules:\n"
        "- Do not write narrative prose inside <think>; it must contain only the five structured JSON sections.\n"
        "- <FPVScene>.navigable_support must list visible safe ground/support semantics, not route goals or distant background.\n"
        "- <FPVScene>.risk_objects must list visible obstacles, occlusions, non-ground surfaces, or regions to avoid.\n"
        "- The <finish>.trajectory must exactly copy trajectory_label.trajectory_norm.\n"
        "- <BEVContext>.intent and <finish>.intent must equal intent_gt.\n"
        "- Use normalized FPV coordinates in [0, 1].\n"
        "- The trajectory must be supported by <FPVScene>.navigable_support or conservative conditional_support.\n"
        "- Do not route through <FPVScene>.risk_objects or into sky/buildings/fences/trees/far background.\n"
        "- Do not add markdown or text outside the tags.\n\n"
        f"{COT_BEV_VISUAL_CONVENTION}\n"
        "Evidence JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        f"{repair_block}"
    )


def build_training_prompt(row: dict[str, Any]) -> str:
    """Build the ms-swift user prompt without leaking GT evidence."""

    num_points = len(row.get("pixel_trajectory") or [])
    image_tokens = "<image><image>" if row.get("bev_image_path") else "<image>"
    bev_line = (
        "Input 2 (BEV): static topological map aligned to the FPV time for route direction and context\n"
        if row.get("bev_image_path")
        else ""
    )
    image_width = row.get("image_width")
    image_height = row.get("image_height")
    image_size_line = ""
    if image_width and image_height:
        image_size_line = (
            f"FPV image size: {int(image_width)}x{int(image_height)} px. "
            "The model must output normalized coordinates; pixel conversion is x_px=x*width, y_px=y*height.\n"
        )
    return (
        f"{image_tokens}\n"
        "Task: Sidewalk navigation structured reasoning and trajectory prediction\n"
        "Input 1 (FPV): real-time robot camera view with visible walkable support, local boundaries, and dynamic obstacles\n"
        f"{bev_line}"
        f"{image_size_line}"
        f"Instruction: {row.get('instruction') or 'Navigate safely on the sidewalk.'}\n\n"
        f"{COT_BEV_VISUAL_CONVENTION}\n"
        "Output exactly this nested structured CoT format. Every section must be valid JSON:\n"
        "<think>\n"
        "<Scene>{\"summary\":\"urban sidewalk scene\",\"visibility\":\"clear|partial_occlusion|low_light|crowded|uncertain\",\"occlusion_sources\":[\"parked_vehicle\"],\"uncertainty\":\"low\"}</Scene>\n"
        "<FPVScene>{\"navigable_support\":[\"sidewalk\"],\"conditional_support\":[\"curb_cut\"],\"risk_objects\":[{\"name\":\"pedestrian\",\"side\":\"left|center|right|front|unknown\",\"risk\":\"boundary|avoid|occlusion|dynamic|low\"}],\"visibility\":\"clear|partial_occlusion|low_light|crowded|uncertain\",\"local_risk\":\"low|medium|high|unknown\"}</FPVScene>\n"
        "<BEVContext>{\"scene_summary\":\"local BEV route context\",\"route_summary\":\"route continues from ego toward target\",\"route_direction_bin\":\"front|slight_left|slight_right|left|right|stop|unknown\",\"intent\":\"go_straight|turn_left|turn_right|stop|unknown\",\"target_direction\":\"left_front|right_front|front|stop|unknown\",\"bev_anchor\":\"front route segment\"}</BEVContext>\n"
        "<CrossView>{\"bev_anchor\":\"...\",\"fpv_anchor\":\"...\",\"near_reference\":\"nearest stable visible reference\",\"blind_area_context\":{\"left\":\"...\",\"right\":\"...\",\"front\":\"...\"},\"fpv_route_region\":\"left|center|right|unknown\",\"route_to_fpv_direction\":\"forward_left|forward_center|forward_right|stop|unknown\",\"safe_corridor\":\"left_corridor|center_corridor|right_corridor|stop|unknown\",\"alignment\":\"aligned|partially_aligned|uncertain\",\"confidence\":\"high|medium|low|unknown\"}</CrossView>\n"
        "<Plan>{\"motion_primitive\":\"go_straight|turn_left|turn_right|stop|unknown\",\"lateral_preference\":\"left|center|right|hold|unknown\",\"avoidance_side\":\"left|right|both|none|unknown\",\"speed_policy\":\"normal|slow|stop|unknown\",\"primary_route\":\"...\",\"safety_policy\":\"...\",\"contingency\":\"...\"}</Plan>\n"
        "</think>\n"
        f"<finish>{{\"trajectory\":[[0.5,0.9]],\"intent\":\"go_straight|turn_left|turn_right|stop|unknown\",\"confidence_level\":\"normal|conflict|low\",\"trajectory_type\":\"{COT_TRAJECTORY_TYPE_ENUM}\",\"trajectory_validity\":\"valid|invalid\"}}</finish>\n\n"
        "Trajectory rules:\n"
        f"- Use exactly {num_points} normalized FPV trajectory points in [0, 1]. Do not output pixel coordinates.\n"
        "- Origin (0,0) is the top-left corner; x is horizontal, y is vertical. Convert to pixels only after parsing.\n"
        "- First identify safe visible navigation support in <FPVScene>.navigable_support, conditional connectors in <FPVScene>.conditional_support, and avoid regions in <FPVScene>.risk_objects.\n"
        "- The first point should be near the robot-side bottom-center walking support, usually around x=0.5 and y=0.85-0.95.\n"
        "- The path is a short local visual prompt, not a route-to-horizon polyline. Stop before occlusion, sky, building facades, fences, walls, vehicles, pedestrians, or uncertain support.\n"
        "- The final point should still be on near-field visible ground support. In normal sidewalk frames it should usually remain in the lower half of the FPV image; do not extend into distant background just because the BEV route continues forward.\n"
        "- Prefer a shorter safe stop/hold trajectory over a long straight line when the foreground sidewalk is narrow, cluttered, or partially blocked.\n"
        "- Every trajectory point must lie on visible walkable ground support named in <FPVScene>.navigable_support, such as sidewalk, pedestrian paving, curb cut, or crosswalk.\n"
        "- Use conditional_support only as a short connector, and never through risk_objects.\n"
        "- Do not place any point in sky, trees, building walls, fences, vertical objects, pedestrians, vehicles, clutter, or far background above the visible ground corridor.\n"
        "- Use BEV only for route topology and intent. If BEV intent conflicts with FPV safety, FPV safety wins and the trajectory should stop or shorten.\n"
        "- If the immediate sidewalk is blocked or unsafe, output a short stop/hold trajectory and set intent or confidence_level accordingly."
    )



def route_direction_bin(theta_deg: float, intent: str, target_direction: str) -> str:
    intent = str(intent or "unknown")
    if intent == "stop" or str(target_direction or "") == "stop":
        return "stop"
    theta = float(theta_deg)
    if theta <= -15.0:
        return "left"
    if theta < -5.0:
        return "slight_left"
    if theta <= 5.0:
        return "front"
    if theta <= 15.0:
        return "slight_right"
    return "right"



def trajectory_region_and_type(trajectory: list[list[float]]) -> dict[str, str]:
    arr = np.asarray(trajectory or [], dtype=np.float32).reshape(-1, 2)
    if arr.size == 0:
        return {
            "fpv_route_region": "unknown",
            "route_to_fpv_direction": "unknown",
            "safe_corridor": "unknown",
            "lateral_preference": "center",
            "trajectory_type": "unknown",
            "speed_policy": "normal",
        }
    mean_x = float(np.mean(arr[:, 0]))
    delta_x = float(arr[-1, 0] - arr[0, 0]) if len(arr) > 1 else 0.0
    y_span = float(np.max(arr[:, 1]) - np.min(arr[:, 1])) if len(arr) > 1 else 0.0
    if mean_x < 0.42:
        region = "left"
    elif mean_x > 0.58:
        region = "right"
    else:
        region = "center"
    if abs(delta_x) < 0.035:
        turn = "straight"
    elif delta_x < 0.0:
        turn = "slight_left"
    else:
        turn = "slight_right"
    if y_span < 0.08:
        trajectory_type = "stop_hold"
        speed_policy = "stop"
    else:
        trajectory_type = f"{turn}_{region}"
        speed_policy = "normal" if y_span > 0.18 else "slow"
    route_to_fpv = "forward_center" if region == "center" else f"forward_{region}"
    return {
        "fpv_route_region": region,
        "route_to_fpv_direction": route_to_fpv,
        "safe_corridor": f"{region}_corridor",
        "lateral_preference": region,
        "trajectory_type": trajectory_type,
        "speed_policy": speed_policy,
    }



def render_template_answer(row: dict[str, Any], evidence: dict[str, Any]) -> str:
    """Deterministic fallback used only after an attempted teacher failure."""

    bev_meta = evidence["bev_patch_meta"]
    label = evidence["trajectory_label"]
    route_theta = float(label.get("route_theta_deg") or bev_meta.get("route_theta_deg") or 0.0)
    intent = str(label.get("intent_gt") or bev_meta.get("intent_gt") or "unknown")
    anchor = str(label.get("anchor") or bev_meta.get("target_direction") or "front")
    trajectory = round_trajectory(evidence["target_trajectory"], digits=5)
    traj_labels = trajectory_region_and_type(trajectory)
    direction_bin = route_direction_bin(route_theta, intent, anchor)
    sections = {
        "Scene": {
            "summary": "The FPV and BEV inputs provide local sidewalk navigation context.",
            "visibility": "uncertain",
            "occlusion_sources": ["unknown"],
            "uncertainty": "Phase1 has no segmentation or temporal risk evidence.",
        },
        "FPVScene": {
            "navigable_support": ["visible sidewalk or pedestrian paving"],
            "conditional_support": [],
            "risk_objects": [],
            "visibility": "uncertain",
            "local_risk": "unknown",
        },
        "BEVContext": {
            "scene_summary": "The BEV shows the local top-down sidewalk context around the robot.",
            "route_summary": f"local route context indicates movement toward {anchor}.",
            "route_direction_bin": direction_bin,
            "intent": intent,
            "target_direction": anchor,
            "bev_anchor": anchor
        },
        "CrossView": {
            "bev_anchor": anchor,
            "fpv_anchor": "the forward navigable corridor in the FPV image",
            "near_reference": "nearest visible sidewalk edge, curb, building line, or open corridor boundary",
            "blind_area_context": {"left": "unknown without dense labels", "right": "unknown without dense labels", "front": "route continues beyond visible support"},
            "fpv_route_region": traj_labels["fpv_route_region"],
            "route_to_fpv_direction": traj_labels["route_to_fpv_direction"],
            "safe_corridor": traj_labels["safe_corridor"],
            "alignment": "aligned",
            "confidence": "medium",
        },
        "Plan": {
            "motion_primitive": intent,
            "lateral_preference": traj_labels["lateral_preference"],
            "avoidance_side": "none",
            "speed_policy": traj_labels["speed_policy"],
            "primary_route": f"follow the {anchor} route while staying on visible walkable support",
            "safety_policy": "prefer sidewalk support, keep clearance from visible obstacles, and avoid uncertain areas",
            "contingency": "slow down or stop if the visible support becomes occluded or unsafe",
        },
        "finish": {
            "trajectory": trajectory,
            "intent": intent,
            "confidence_level": bev_meta.get("confidence_level_gt") or "normal",
            "trajectory_type": traj_labels["trajectory_type"],
            "trajectory_validity": "valid",
        },
    }
    return "\n".join(
        [
            "<think>",
            f"<Scene>{json.dumps(sections['Scene'], ensure_ascii=False, separators=(',', ':'))}</Scene>",
            f"<FPVScene>{json.dumps(sections['FPVScene'], ensure_ascii=False, separators=(',', ':'))}</FPVScene>",
            f"<BEVContext>{json.dumps(sections['BEVContext'], ensure_ascii=False, separators=(',', ':'))}</BEVContext>",
            f"<CrossView>{json.dumps(sections['CrossView'], ensure_ascii=False, separators=(',', ':'))}</CrossView>",
            f"<Plan>{json.dumps(sections['Plan'], ensure_ascii=False, separators=(',', ':'))}</Plan>",
            "</think>",
            f"<finish>{json.dumps(sections['finish'], ensure_ascii=False, separators=(',', ':'))}</finish>",
        ]
    )


def validate_teacher_answer(answer: str, row: dict[str, Any], evidence: dict[str, Any], config: TeacherConfig) -> ValidationResult:
    errors: list[str] = []
    sections: dict[str, Any] = {}
    if not str(answer or "").strip():
        return ValidationResult(False, ["empty_answer"], {})

    errors.extend(validate_tag_layout(answer))
    think_body = extract_tag(answer, "think")
    for tag in REQUIRED_TAGS:
        if tag in THINK_SECTION_TAGS:
            body = extract_tag(think_body or "", tag)
        else:
            body = extract_tag(answer, tag)
        if body is None:
            errors.append(f"missing_tag:{tag}")
            continue
        if tag == "think":
            if not body.strip() or not any(extract_tag(body, section) for section in THINK_SECTION_TAGS):
                errors.append("empty_think")
            sections[tag] = body.strip()
            continue
        parsed = parse_json_object(body)
        if parsed is None:
            errors.append(f"invalid_json:{tag}")
        else:
            sections[tag] = parsed

    if errors:
        return ValidationResult(False, errors, sections)

    route_context = evidence["route_context"]
    bev_meta = evidence["bev_patch_meta"]
    label = evidence["trajectory_label"]
    row_meta = row.get("sidewalk_meta") or {}
    intent_gt = str(label.get("intent_gt") or bev_meta.get("intent_gt") or route_context.get("intent_gt") or "")
    theta_gt = optional_float(label.get("route_theta_deg") or bev_meta.get("route_theta_deg") or route_context.get("route_theta_deg"))
    try:
        target_traj = np.asarray(evidence["target_trajectory"], dtype=np.float32).reshape(-1, 2)
    except Exception:
        errors.append("target_trajectory_invalid")
        target_traj = np.empty((0, 2), dtype=np.float32)

    fpv_scene = sections.get("FPVScene") or {}
    bev = sections.get("BEVContext") or {}
    cross_view = sections.get("CrossView") or {}
    finish = sections.get("finish") or {}
    scene = sections.get("Scene") or {}
    for field in ("summary", "visibility", "occlusion_sources", "uncertainty"):
        if field not in scene:
            errors.append(f"missing_scene_field:{field}")
    for field in ("navigable_support", "conditional_support", "risk_objects", "visibility", "local_risk"):
        if field not in fpv_scene:
            errors.append(f"missing_fpv_scene_field:{field}")
    for field in ("scene_summary", "route_summary", "route_direction_bin", "bev_anchor"):
        if field not in bev:
            errors.append(f"missing_bev_context_field:{field}")
    for field in ("bev_anchor", "fpv_anchor", "near_reference", "blind_area_context", "fpv_route_region", "route_to_fpv_direction", "safe_corridor"):
        if field not in cross_view:
            errors.append(f"missing_cross_view_field:{field}")
    plan = sections.get("Plan") or {}
    for field in ("motion_primitive", "lateral_preference", "avoidance_side", "speed_policy", "primary_route", "safety_policy", "contingency"):
        if field not in plan:
            errors.append(f"missing_plan_field:{field}")
    for field in ("trajectory_type", "trajectory_validity"):
        if field not in finish:
            errors.append(f"missing_finish_field:{field}")
    risk_objects = fpv_scene.get("risk_objects")
    if isinstance(risk_objects, list):
        for index, item in enumerate(risk_objects):
            if not isinstance(item, dict):
                errors.append(f"risk_object_not_structured:{index}")
                continue
            for field in ("name", "side", "risk"):
                if field not in item:
                    errors.append(f"risk_object_missing_{field}:{index}")
    elif risk_objects is not None:
        errors.append("risk_objects_not_list")
    if str(bev.get("intent") or "") != intent_gt:
        errors.append(f"bev_intent_mismatch:{bev.get('intent')}!={intent_gt}")
    if str(finish.get("intent") or "") != intent_gt:
        errors.append(f"finish_intent_mismatch:{finish.get('intent')}!={intent_gt}")

    if theta_gt is not None:
        expected_direction_bin = route_direction_bin(theta_gt, intent_gt, str(bev_meta.get("target_direction") or route_context.get("target_direction") or "front"))
        if str(bev.get("route_direction_bin") or "") != expected_direction_bin:
            errors.append(f"route_direction_bin_mismatch:{bev.get('route_direction_bin')}!={expected_direction_bin}")
    scene_visibility = str(scene.get("visibility") or "")
    fpv_visibility = str(fpv_scene.get("visibility") or "")
    local_risk = str(fpv_scene.get("local_risk") or "")
    allow_unknown_stop = (
        scene_visibility in {"uncertain", "partial_occlusion", "low_light", "crowded"}
        or fpv_visibility in {"uncertain", "partial_occlusion", "low_light", "crowded"}
        or local_risk in {"unknown", "high"}
    )

    try:
        traj = np.asarray(finish.get("trajectory") or [], dtype=np.float32).reshape(-1, 2)
    except Exception:
        errors.append("trajectory_invalid")
        traj = np.empty((0, 2), dtype=np.float32)
    if traj.shape != target_traj.shape:
        errors.append(f"trajectory_shape:{list(traj.shape)}!={list(target_traj.shape)}")
    elif target_traj.shape[0] != int(config.expected_points):
        errors.append(f"target_points:{target_traj.shape[0]}!={config.expected_points}")
    else:
        if not np.all(np.isfinite(traj)) or float(np.min(traj)) < -1e-4 or float(np.max(traj)) > 1.0001:
            errors.append("trajectory_out_of_bounds")
        max_delta = float(np.max(np.abs(traj - target_traj))) if traj.size else math.inf
        if max_delta > float(config.trajectory_tolerance):
            errors.append(f"trajectory_changed:max_delta={max_delta:.6f}")
        derived = trajectory_region_and_type(round_trajectory(target_traj.astype(float).tolist(), digits=5))
        is_stop_like = intent_gt == "stop" or derived["trajectory_type"] == "stop_hold"
        if is_stop_like:
            allowed_cross_values = {
                "fpv_route_region": {derived["fpv_route_region"], "center", "stop"},
                "route_to_fpv_direction": {derived["route_to_fpv_direction"], "forward_center", "stop"},
                "safe_corridor": {derived["safe_corridor"], "center_corridor", "stop", "hold"},
            }
            if allow_unknown_stop:
                for allowed_values in allowed_cross_values.values():
                    allowed_values.add("unknown")
            for field, allowed_values in allowed_cross_values.items():
                if str(cross_view.get(field) or "") not in allowed_values:
                    errors.append(f"crossview_{field}_mismatch:{cross_view.get(field)}!={derived[field]}")
            allowed_lateral = {derived["lateral_preference"], "center", "hold"}
            if allow_unknown_stop:
                allowed_lateral.add("unknown")
            if str((sections.get("Plan") or {}).get("lateral_preference") or "") not in allowed_lateral:
                errors.append(f"plan_lateral_preference_mismatch:{(sections.get('Plan') or {}).get('lateral_preference')}!={derived['lateral_preference']}")
        else:
            for field in ("fpv_route_region", "route_to_fpv_direction", "safe_corridor"):
                if str(cross_view.get(field) or "") != derived[field]:
                    errors.append(f"crossview_{field}_mismatch:{cross_view.get(field)}!={derived[field]}")
            if str((sections.get("Plan") or {}).get("lateral_preference") or "") != derived["lateral_preference"]:
                errors.append(f"plan_lateral_preference_mismatch:{(sections.get('Plan') or {}).get('lateral_preference')}!={derived['lateral_preference']}")
        allowed_speed = {derived["speed_policy"]}
        if is_stop_like:
            allowed_speed.update({"stop", "slow"})
            if allow_unknown_stop:
                allowed_speed.add("unknown")
        if str((sections.get("Plan") or {}).get("speed_policy") or "") not in allowed_speed:
            errors.append(f"plan_speed_policy_mismatch:{(sections.get('Plan') or {}).get('speed_policy')}!={derived['speed_policy']}")
        finish_trajectory_type = str(finish.get("trajectory_type") or "")
        reviewed_fixed_type = str(row_meta.get("fixed_trajectory_type") or "")
        allow_reviewed_type_fix = (
            bool(row_meta.get("readded_after_review"))
            and reviewed_fixed_type
            and finish_trajectory_type == reviewed_fixed_type
            and intent_gt in {"turn_left", "turn_right"}
            and str(row_meta.get("old_trajectory_type") or "") == "stop_hold"
        )
        if finish_trajectory_type != derived["trajectory_type"] and not allow_reviewed_type_fix:
            errors.append(f"trajectory_type_mismatch:{finish.get('trajectory_type')}!={derived['trajectory_type']}")
        if str(finish.get("trajectory_validity") or "") != "valid":
            errors.append(f"trajectory_validity_not_valid:{finish.get('trajectory_validity')}")

    if not has_segmentation_context(row, evidence):
        lower_answer = answer.lower()
        for pattern in NO_SEG_FORBIDDEN_PATTERNS:
            if re.search(pattern, lower_answer):
                errors.append(f"segmentation_claim_without_seg:{pattern}")

    return ValidationResult(not errors, errors, sections)


def build_cot_manifest_row(
    row: dict[str, Any],
    *,
    answer: str,
    validation: ValidationResult,
    teacher_source: str,
    teacher_model: str | None = None,
    teacher_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    out["prompt"] = build_training_prompt(row)
    out["prompt_template"] = COT_PROMPT_TEMPLATE
    out["response"] = answer
    cot_teacher = {
        "schema_version": COT_SCHEMA_VERSION,
        "teacher_source": teacher_source,
        "teacher_model": teacher_model,
        "sections": sorted(validation.sections.keys()),
    }
    if validation.errors:
        cot_teacher["validation_errors"] = validation.errors
    if teacher_usage:
        cot_teacher["usage"] = teacher_usage
    out["cot_teacher"] = cot_teacher
    meta = out.setdefault("sidewalk_meta", {})
    meta["cot_teacher_source"] = teacher_source
    meta["cot_schema_version"] = COT_SCHEMA_VERSION
    return out


async def call_llm_teacher(row: dict[str, Any], evidence: dict[str, Any], config: TeacherConfig, *, previous_answer: str | None = None, errors: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    VisionAnalyzer, local_image_path_to_data_url = load_agent_llm_helpers()
    analyzer = VisionAnalyzer()
    if not analyzer.is_available():
        raise RuntimeError("LLM/VLM teacher is not configured. Set ARK_API_KEY or LLM_API_KEY.")

    fpv_url = local_image_path_to_data_url(row["fpv_image_path"])
    extra_urls: list[str] = []
    if row.get("bev_image_path"):
        extra_urls.append(local_image_path_to_data_url(row["bev_image_path"]))
    result = await analyzer.analyze(
        image_url=fpv_url,
        extra_image_urls=extra_urls,
        question=build_teacher_question(row, evidence, previous_answer=previous_answer, errors=errors),
        system_prompt=build_teacher_system_prompt(),
        detail=config.image_detail,
    )
    return str(result.get("answer") or "").strip(), result


def load_agent_llm_helpers():
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from mcp_server.llm import VisionAnalyzer, local_image_path_to_data_url

    return VisionAnalyzer, local_image_path_to_data_url


async def generate_cot_dataset(
    source_jsonl: str,
    output_jsonl: str,
    *,
    report_json: str | None = None,
    limit: int | None = None,
    config: TeacherConfig | None = None,
    progress_interval: int = 25,
) -> dict[str, Any]:
    cfg = config or TeacherConfig()
    rows = list(read_jsonl(Path(source_jsonl)))
    if limit is not None:
        rows = rows[: int(limit)]

    output_path = Path(output_jsonl)
    existing_rows = count_jsonl_rows(output_path)
    if existing_rows > len(rows):
        raise RuntimeError(f"Existing output has more rows than source: {output_jsonl}")

    report: dict[str, Any] = {
        "schema_version": f"{COT_SCHEMA_VERSION}_report",
        "source_jsonl": str(source_jsonl),
        "output_jsonl": str(output_jsonl),
        "input_rows": len(rows),
        "written_rows": existing_rows,
        "dropped_rows": 0,
        "fallback_rows": 0,
        "processed_rows": existing_rows,
        "resumed_rows": existing_rows,
        "status": "running",
        "error_counts": {},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    progress_every = max(1, int(progress_interval))
    print(
        f"[cot-teacher] starting source={source_jsonl} rows={len(rows)} "
        f"resume={existing_rows} output={output_jsonl}",
        flush=True,
    )

    if existing_rows == len(rows):
        report["status"] = "done"
        report["eta_seconds"] = 0.0
        write_report(report_json, report)
        return report

    with output_path.open("a" if existing_rows else "w", encoding="utf-8", buffering=1) as out_f:
        for index, row in enumerate(rows[existing_rows:], start=existing_rows):
            evidence = load_evidence(row)
            answer = ""
            validation = ValidationResult(False, ["not_started"], {})
            telemetry: dict[str, Any] | None = None
            for attempt in range(int(cfg.max_retries) + 1):
                try:
                    answer, telemetry = await call_llm_teacher(
                        row,
                        evidence,
                        cfg,
                        previous_answer=answer if attempt else None,
                        errors=validation.errors if attempt else None,
                    )
                    add_usage_to_report(report, telemetry)
                except RuntimeError as exc:
                    if "not configured" in str(exc).lower():
                        raise
                    validation = ValidationResult(False, [f"teacher_call_failed:{type(exc).__name__}:{str(exc)[:200]}"], {})
                    continue
                except Exception as exc:
                    validation = ValidationResult(False, [f"teacher_call_failed:{type(exc).__name__}:{str(exc)[:200]}"], {})
                    continue
                validation = validate_teacher_answer(answer, row, evidence, cfg)
                if validation.valid:
                    break

            teacher_model = None
            teacher_usage = telemetry_usage(telemetry)
            if isinstance(telemetry, dict):
                llm_telemetry = telemetry_llm_metadata(telemetry)
                teacher_model = str(llm_telemetry.get("model") or telemetry.get("model") or "") or None

            if validation.valid:
                out_row = build_cot_manifest_row(
                    row,
                    answer=answer,
                    validation=validation,
                    teacher_source="llm_teacher",
                    teacher_model=teacher_model,
                    teacher_usage=teacher_usage,
                )
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                report["written_rows"] += 1
                report["processed_rows"] = index + 1
                if (index + 1) % progress_every == 0 or index + 1 == len(rows):
                    elapsed = max(time.monotonic() - started_at, 1e-6)
                    current_run_rows = max(int(report["processed_rows"]) - int(report["resumed_rows"]), 0)
                    rate = float(current_run_rows) / elapsed
                    remaining = max(len(rows) - int(report["processed_rows"]), 0)
                    eta_seconds = remaining / rate if rate > 0 else None
                    report["rate_rows_per_hour"] = rate * 3600.0
                    report["eta_seconds"] = eta_seconds
                    write_report(report_json, report)
                    eta_text = f"{eta_seconds / 3600.0:.2f}h" if eta_seconds is not None else "unknown"
                    print(
                        "[cot-teacher] progress "
                        f"{report['processed_rows']}/{len(rows)} "
                        f"written={report['written_rows']} "
                        f"dropped={report['dropped_rows']} "
                        f"fallback={report['fallback_rows']} "
                        f"rate={report['rate_rows_per_hour']:.1f}/h "
                        f"eta={eta_text}",
                        flush=True,
                    )
                continue

            for error in validation.errors:
                _inc(report["error_counts"], error.split(":", 1)[0])

            if cfg.fallback_policy == "template":
                template_answer = render_template_answer(row, evidence)
                template_validation = validate_teacher_answer(template_answer, row, evidence, cfg)
                if not template_validation.valid:
                    raise RuntimeError(f"Template fallback failed for {row.get('id')}: {template_validation.errors}")
                out_row = build_cot_manifest_row(
                    row,
                    answer=template_answer,
                    validation=template_validation,
                    teacher_source="template_after_llm_failure",
                    teacher_model=teacher_model,
                    teacher_usage=teacher_usage,
                )
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                report["written_rows"] += 1
                report["fallback_rows"] += 1
            elif cfg.fallback_policy == "error":
                raise RuntimeError(f"Teacher validation failed for row {index} ({row.get('id')}): {validation.errors}")
            else:
                report["dropped_rows"] += 1

            report["processed_rows"] = index + 1
            if (index + 1) % progress_every == 0 or index + 1 == len(rows):
                elapsed = max(time.monotonic() - started_at, 1e-6)
                current_run_rows = max(int(report["processed_rows"]) - int(report["resumed_rows"]), 0)
                rate = float(current_run_rows) / elapsed
                remaining = max(len(rows) - int(report["processed_rows"]), 0)
                eta_seconds = remaining / rate if rate > 0 else None
                report["rate_rows_per_hour"] = rate * 3600.0
                report["eta_seconds"] = eta_seconds
                write_report(report_json, report)
                eta_text = f"{eta_seconds / 3600.0:.2f}h" if eta_seconds is not None else "unknown"
                print(
                    "[cot-teacher] progress "
                    f"{report['processed_rows']}/{len(rows)} "
                    f"written={report['written_rows']} "
                    f"dropped={report['dropped_rows']} "
                    f"fallback={report['fallback_rows']} "
                    f"rate={report['rate_rows_per_hour']:.1f}/h "
                    f"eta={eta_text}",
                    flush=True,
                )

    report["status"] = "done"
    report["eta_seconds"] = 0.0
    write_report(report_json, report)
    return report


def extract_tag(text: str, tag: str) -> str | None:
    pattern = re.compile(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(str(text or ""))
    return match.group(1).strip() if match else None


def validate_tag_layout(text: str) -> list[str]:
    """Reject markdown wrappers, reordered tags, and extra prose.

    The training target is meant to teach a stable agent grammar.  Being strict
    here makes invalid teacher generations visible before they enter SFT data.
    """

    raw = str(text or "").strip()
    errors: list[str] = []
    think_match = re.match(r"^<think>\s*([\s\S]*?)\s*</think>\s*", raw, flags=re.IGNORECASE)
    if not think_match:
        errors.append("outer_layout_missing_think_first")
        return errors

    finish_tail = raw[think_match.end():]
    finish_match = re.match(r"^<finish>\s*[\s\S]*?\s*</finish>\s*$", finish_tail, flags=re.IGNORECASE)
    if not finish_match:
        errors.append("outer_layout_missing_finish_after_think")

    think_body = think_match.group(1)
    pos = 0
    for tag in THINK_SECTION_TAGS:
        match = re.search(rf"<{re.escape(tag)}>\s*[\s\S]*?\s*</{re.escape(tag)}>", think_body[pos:], re.IGNORECASE)
        if not match:
            errors.append(f"missing_nested_tag:{tag}")
            continue
        if think_body[pos:pos + match.start()].strip():
            errors.append(f"text_before_nested_tag:{tag}")
        pos += match.end()
    if think_body[pos:].strip():
        errors.append("text_after_nested_tags")
    return errors


def parse_json_object(text: str) -> dict[str, Any] | None:
    raw = strip_json_fence(str(text or "").strip())
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def strip_json_fence(text: str) -> str:
    match = re.match(r"```(?:json)?\s*([\s\S]*?)\s*```$", text.strip(), flags=re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def round_trajectory(points: Any, *, digits: int) -> list[list[float]]:
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return [[round(float(x), digits), round(float(y), digits)] for x, y in arr]


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def has_segmentation_context(row: dict[str, Any], evidence: dict[str, Any]) -> bool:
    if row.get("semantic_segmentation_path") or row.get("semantic_mask_path"):
        return True
    qa_path = row.get("qa_bundle_path")
    if qa_path and Path(str(qa_path)).is_file():
        try:
            qa = load_json(qa_path)
            return bool(qa.get("segmentation_context_ref") or qa.get("som_context"))
        except Exception:
            return False
    return False


def _inc(counter: dict[str, int], key: str) -> None:
    counter[key] = int(counter.get(key, 0)) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM-teacher structured CoT JSONL for SideWalk Phase1")
    parser.add_argument("--source-jsonl", required=True, help="Geometry source JSONL from sidewalk_phase1_dataset.py")
    parser.add_argument("--output-jsonl", required=True, help="Validated CoT source JSONL for ms-swift dataset_builder")
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expected-points", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--trajectory-tolerance", type=float, default=0.025)
    parser.add_argument("--theta-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--image-detail", default="auto", choices=["auto", "low", "high"])
    parser.add_argument("--fallback-policy", default="drop", choices=["drop", "error", "template"])
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TeacherConfig(
        expected_points=int(args.expected_points),
        max_retries=int(args.max_retries),
        trajectory_tolerance=float(args.trajectory_tolerance),
        theta_tolerance_deg=float(args.theta_tolerance_deg),
        fallback_policy=str(args.fallback_policy),
        image_detail=str(args.image_detail),
    )
    report = asyncio.run(
        generate_cot_dataset(
            args.source_jsonl,
            args.output_jsonl,
            report_json=args.report_json,
            limit=args.limit,
            config=config,
            progress_interval=int(args.progress_interval),
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
