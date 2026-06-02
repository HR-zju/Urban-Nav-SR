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
must not change ``intent_gt``, ``route_theta_deg``, or ``pixel_trajectory``.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
THINK_SECTION_TAGS = ["Scene", "FPVScene", "BEVContext", "CrossView", "Plan"]
REQUIRED_TAGS = ["think", *THINK_SECTION_TAGS, "finish"]
COT_SCHEMA_VERSION = "sidewalk_phase1_cot_teacher_v3"
COT_PROMPT_TEMPLATE = "phase1_structured_cot_v3"
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
        "The geometry evidence is ground truth: do not alter intent, route_theta_deg, target cells, or trajectory. "
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
            "grid_size": bev_meta.get("grid_size"),
            "ego_cell_gt": bev_meta.get("ego_cell_gt"),
            "target_cell_gt": bev_meta.get("target_cell_gt"),
            "route_polyline_cells": bev_meta.get("route_polyline_cells"),
            "route_theta_deg": bev_meta.get("route_theta_deg"),
            "intent_gt": bev_meta.get("intent_gt"),
            "target_direction": bev_meta.get("target_direction"),
            "confidence_level_gt": bev_meta.get("confidence_level_gt"),
        },
        "trajectory_label": {
            "trajectory_source": label.get("trajectory_source"),
            "route_theta_deg": label.get("route_theta_deg"),
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
        "Create a Phase1 single-shot structured CoT assistant target for this sidewalk navigation sample.\n"
        "Output exactly this nested tag structure in this order:\n"
        "<think>\n"
        "<Scene>{\"summary\":\"...\",\"visibility\":\"visible|partially_visible|occluded|unknown\",\"uncertainty\":\"...\"}</Scene>\n"
        "<FPVScene>{\"navigable_support\":[\"stable semantic names for visible safe navigation support\"],\"conditional_support\":[\"stable semantic names for conditionally usable connectors\"],\"risk_objects\":[\"stable semantic names for visible regions to avoid\"],\"visibility\":\"visible|partially_visible|occluded|unknown\",\"local_risk\":\"low|medium|high|unknown\"}</FPVScene>\n"
        "<BEVContext>{\"grid_size\":[M,N],\"ego_cell\":[c,r],\"target_cell\":[c,r],\"route_theta_deg\":0.0,\"intent\":\"go_straight|turn_left|turn_right|stop\",\"target_direction\":\"...\"}</BEVContext>\n"
        "<CrossView>{\"bev_anchor\":\"...\",\"fpv_anchor\":\"...\",\"alignment\":\"aligned|partially_aligned|uncertain\",\"confidence\":\"high|medium|low|unknown\"}</CrossView>\n"
        "<Plan>{\"primary_route\":\"...\",\"safety_policy\":\"...\",\"contingency\":\"...\"}</Plan>\n"
        "</think>\n"
        "<finish>{\"trajectory\":[[x,y],...],\"intent\":\"...\",\"route_theta_deg\":0.0,\"confidence_level\":\"normal|conflict|low\"}</finish>\n\n"
        "Rules:\n"
        "- Do not write narrative prose inside <think>; it must contain only the five structured JSON sections.\n"
        "- <FPVScene>.navigable_support must list visible safe ground/support semantics, not route goals or distant background.\n"
        "- <FPVScene>.risk_objects must list visible obstacles, occlusions, non-ground surfaces, or regions to avoid.\n"
        "- <BEVContext>.ego_cell and <BEVContext>.target_cell must exactly copy ego_cell_gt and target_cell_gt.\n"
        "- The <finish>.trajectory must exactly copy trajectory_label.trajectory_norm.\n"
        "- <BEVContext>.intent and <finish>.intent must equal intent_gt.\n"
        "- route_theta_deg must equal the evidence route_theta_deg.\n"
        "- Use normalized FPV coordinates in [0, 1].\n"
        "- The trajectory must be supported by <FPVScene>.navigable_support or conservative conditional_support.\n"
        "- Do not route through <FPVScene>.risk_objects or into sky/buildings/fences/trees/far background.\n"
        "- Do not add markdown or text outside the tags.\n\n"
        "Evidence JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        f"{repair_block}"
    )


def build_training_prompt(row: dict[str, Any]) -> str:
    """Build the ms-swift user prompt without leaking GT evidence."""

    num_points = len(row.get("pixel_trajectory") or [])
    image_tokens = "<image><image>" if row.get("bev_image_path") else "<image>"
    bev_line = "Input 2: BEV map aligned to the FPV time\n" if row.get("bev_image_path") else ""
    return (
        f"{image_tokens}\n"
        "Task: Sidewalk navigation structured reasoning and trajectory prediction\n"
        "Input 1: First-person view (FPV) from the robot\n"
        f"{bev_line}"
        f"Instruction: {row.get('instruction') or 'Navigate safely on the sidewalk.'}\n\n"
        "Output exactly this nested structured CoT format. Every section must be valid JSON:\n"
        "<think>\n"
        "<Scene>{\"summary\":\"...\",\"visibility\":\"visible|partially_visible|occluded|unknown\",\"uncertainty\":\"...\"}</Scene>\n"
        "<FPVScene>{\"navigable_support\":[\"sidewalk\",...],\"conditional_support\":[\"curb_cut\",...],\"risk_objects\":[\"pedestrian\",...],\"visibility\":\"visible|partially_visible|occluded|unknown\",\"local_risk\":\"low|medium|high|unknown\"}</FPVScene>\n"
        "<BEVContext>{\"grid_size\":[M,N],\"ego_cell\":[c,r],\"target_cell\":[c,r],\"route_theta_deg\":...,\"intent\":\"go_straight|turn_left|turn_right|stop|unknown\",\"target_direction\":\"left_front|right_front|front|stop|unknown\"}</BEVContext>\n"
        "<CrossView>{\"bev_anchor\":\"...\",\"fpv_anchor\":\"...\",\"alignment\":\"aligned|partially_aligned|uncertain\",\"confidence\":\"high|medium|low|unknown\"}</CrossView>\n"
        "<Plan>{\"primary_route\":\"...\",\"safety_policy\":\"...\",\"contingency\":\"...\"}</Plan>\n"
        "</think>\n"
        "<finish>{\"trajectory\":[[x,y],...],\"intent\":\"...\",\"route_theta_deg\":...,\"confidence_level\":\"...\"}</finish>\n\n"
        "Trajectory rules:\n"
        f"- Use exactly {num_points} normalized FPV trajectory points in [0, 1]. Do not output pixel coordinates.\n"
        "- Origin (0,0) is the top-left corner; x is horizontal, y is vertical.\n"
        "- First identify safe visible navigation support in <FPVScene>.navigable_support, conditional connectors in <FPVScene>.conditional_support, and avoid regions in <FPVScene>.risk_objects.\n"
        "- The first point should be near the robot-side bottom-center walking support, usually around x=0.5 and y=0.85-0.95.\n"
        "- The path is a short local visual prompt, not a route-to-horizon polyline. Stop before occlusion, sky, building facades, fences, walls, vehicles, pedestrians, or uncertain support.\n"
        "- Every trajectory point must lie on visible walkable ground support named in <FPVScene>.navigable_support, such as sidewalk, pedestrian paving, curb cut, or crosswalk.\n"
        "- Use conditional_support only as a short connector, and never through risk_objects.\n"
        "- Do not place any point in sky, trees, building walls, fences, vertical objects, pedestrians, vehicles, clutter, or far background above the visible ground corridor.\n"
        "- Use BEV only for route topology and intent. If BEV intent conflicts with FPV safety, FPV safety wins and the trajectory should stop or shorten.\n"
        "- If the immediate sidewalk is blocked or unsafe, output a short stop/hold trajectory and set intent or confidence_level accordingly."
    )


def render_template_answer(row: dict[str, Any], evidence: dict[str, Any]) -> str:
    """Deterministic fallback used only after an attempted teacher failure."""

    bev_meta = evidence["bev_patch_meta"]
    label = evidence["trajectory_label"]
    route_theta = float(label.get("route_theta_deg") or bev_meta.get("route_theta_deg") or 0.0)
    intent = str(label.get("intent_gt") or bev_meta.get("intent_gt") or "unknown")
    anchor = str(label.get("anchor") or bev_meta.get("target_direction") or "front")
    trajectory = round_trajectory(evidence["target_trajectory"], digits=5)
    sections = {
        "Scene": {
            "summary": "The FPV and BEV inputs provide local sidewalk navigation context.",
            "visibility": "unknown",
            "uncertainty": "Phase1 has no segmentation or temporal risk evidence.",
        },
        "FPVScene": {
            "navigable_support": ["visible sidewalk or pedestrian paving"],
            "conditional_support": [],
            "risk_objects": [],
            "visibility": "unknown",
            "local_risk": "unknown",
        },
        "BEVContext": {
            "grid_size": bev_meta.get("grid_size"),
            "ego_cell": bev_meta.get("ego_cell_gt"),
            "target_cell": bev_meta.get("target_cell_gt"),
            "route_theta_deg": round(route_theta, 3),
            "intent": intent,
            "target_direction": anchor,
        },
        "CrossView": {
            "bev_anchor": anchor,
            "fpv_anchor": "the forward navigable corridor in the FPV image",
            "alignment": "aligned",
            "confidence": "medium",
        },
        "Plan": {
            "primary_route": f"follow the {anchor} route while staying on visible walkable support",
            "safety_policy": "prefer sidewalk support, keep clearance from visible obstacles, and avoid uncertain areas",
            "contingency": "slow down or stop if the visible support becomes occluded or unsafe",
        },
        "finish": {
            "trajectory": trajectory,
            "intent": intent,
            "route_theta_deg": round(route_theta, 3),
            "confidence_level": bev_meta.get("confidence_level_gt") or "normal",
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
    intent_gt = str(label.get("intent_gt") or bev_meta.get("intent_gt") or route_context.get("intent_gt") or "")
    theta_gt = optional_float(label.get("route_theta_deg") or bev_meta.get("route_theta_deg") or route_context.get("route_theta_deg"))
    try:
        target_traj = np.asarray(evidence["target_trajectory"], dtype=np.float32).reshape(-1, 2)
    except Exception:
        errors.append("target_trajectory_invalid")
        target_traj = np.empty((0, 2), dtype=np.float32)

    fpv_scene = sections.get("FPVScene") or {}
    bev = sections.get("BEVContext") or {}
    finish = sections.get("finish") or {}
    for field in ("navigable_support", "conditional_support", "risk_objects"):
        if field not in fpv_scene:
            errors.append(f"missing_fpv_scene_field:{field}")
    if _normalize_cell(bev.get("ego_cell")) != _normalize_cell(bev_meta.get("ego_cell_gt")):
        errors.append(f"ego_cell_mismatch:{bev.get('ego_cell')}!={bev_meta.get('ego_cell_gt')}")
    if _normalize_cell(bev.get("target_cell")) != _normalize_cell(bev_meta.get("target_cell_gt")):
        errors.append(f"target_cell_mismatch:{bev.get('target_cell')}!={bev_meta.get('target_cell_gt')}")
    if str(bev.get("intent") or "") != intent_gt:
        errors.append(f"bev_intent_mismatch:{bev.get('intent')}!={intent_gt}")
    if str(finish.get("intent") or "") != intent_gt:
        errors.append(f"finish_intent_mismatch:{finish.get('intent')}!={intent_gt}")

    for section_name, payload in (("BEVContext", bev), ("finish", finish)):
        theta = optional_float(payload.get("route_theta_deg"))
        if theta_gt is not None and theta is not None and abs(theta - theta_gt) > float(config.theta_tolerance_deg):
            errors.append(f"theta_mismatch:{section_name}:{theta}!={theta_gt}")
        elif theta_gt is not None and theta is None:
            errors.append(f"missing_theta:{section_name}")

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
) -> dict[str, Any]:
    cfg = config or TeacherConfig()
    rows = list(read_jsonl(Path(source_jsonl)))
    if limit is not None:
        rows = rows[: int(limit)]

    output_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": f"{COT_SCHEMA_VERSION}_report",
        "source_jsonl": str(source_jsonl),
        "output_jsonl": str(output_jsonl),
        "input_rows": len(rows),
        "written_rows": 0,
        "dropped_rows": 0,
        "fallback_rows": 0,
        "error_counts": {},
    }

    for index, row in enumerate(rows):
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
        if isinstance(telemetry, dict):
            teacher_model = str(telemetry.get("model") or "") or None

        if validation.valid:
            output_rows.append(
                build_cot_manifest_row(
                    row,
                    answer=answer,
                    validation=validation,
                    teacher_source="llm_teacher",
                    teacher_model=teacher_model,
                )
            )
            continue

        for error in validation.errors:
            _inc(report["error_counts"], error.split(":", 1)[0])

        if cfg.fallback_policy == "template":
            template_answer = render_template_answer(row, evidence)
            template_validation = validate_teacher_answer(template_answer, row, evidence, cfg)
            if not template_validation.valid:
                raise RuntimeError(f"Template fallback failed for {row.get('id')}: {template_validation.errors}")
            output_rows.append(
                build_cot_manifest_row(
                    row,
                    answer=template_answer,
                    validation=template_validation,
                    teacher_source="template_after_llm_failure",
                    teacher_model=teacher_model,
                )
            )
            report["fallback_rows"] += 1
        elif cfg.fallback_policy == "error":
            raise RuntimeError(f"Teacher validation failed for row {index} ({row.get('id')}): {validation.errors}")
        else:
            report["dropped_rows"] += 1

    write_jsonl(Path(output_jsonl), output_rows)
    report["written_rows"] = len(output_rows)
    if report_json:
        path = Path(report_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
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


def _normalize_cell(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [int(value[0]), int(value[1])]
    except Exception:
        return None


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
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
