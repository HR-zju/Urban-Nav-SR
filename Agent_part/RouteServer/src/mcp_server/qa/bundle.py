from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .pov_candidates import PoVPromptCandidateBundle, pov_candidate_bundle_to_dict
from .schema import QASample, to_plain_dict
from .tasks import TASK_REGISTRY
from . import teachers


def _sample(
    *,
    sample_id: str,
    task_type: str,
    inputs: dict[str, Any],
    teacher_answer: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> QASample:
    spec = TASK_REGISTRY[task_type]
    return QASample(
        sample_id=f"{sample_id}:{task_type}",
        task_family=spec.task_family,
        task_type=spec.task_type,
        answer_type=spec.answer_type,
        inputs=to_plain_dict(inputs),
        teacher_answer=to_plain_dict(teacher_answer),
        metadata=metadata or {},
    )


def _visual_regions_from_candidate_bundle(candidate_bundle: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(candidate_bundle, dict):
        return []
    regions: list[dict[str, Any]] = []
    for item in candidate_bundle.get("fpv_candidates") or []:
        center = item.get("center_uv")
        if isinstance(center, dict):
            regions.append(
                {
                    "region_id": item.get("prompt_id"),
                    "region_type": item.get("prompt_type"),
                    "score": item.get("fpv_support_score"),
                    "center_uv": [center.get("u"), center.get("v")],
                }
            )
    return regions


def build_visual_prompt_qa_bundle(
    *,
    route_context: dict[str, Any],
    local_bev_context: dict[str, Any],
    image_width: int,
    image_height: int,
    fpv_segmentation_summary: dict[str, Any] | None,
    segmentation_validation: dict[str, Any] | None,
    rendered_prompt: dict[str, Any] | None,
    arbiter_decision: dict[str, Any],
    candidate_bundle: PoVPromptCandidateBundle | dict[str, Any] | None,
    som_summary: dict[str, Any] | None = None,
    sample_id: str = "visual_prompt",
) -> dict[str, Any]:
    candidate_bundle_dict = (
        pov_candidate_bundle_to_dict(candidate_bundle)
        if isinstance(candidate_bundle, PoVPromptCandidateBundle)
        else candidate_bundle
    )
    samples: list[QASample] = []

    samples.append(
        _sample(
            sample_id=sample_id,
            task_type="bev_route_theta",
            inputs={
                "local_segment_bev": local_bev_context.get("local_segment_bev") or [],
                "route_polyline_local": local_bev_context.get("route_polyline_local") or [],
                "lookahead_m": local_bev_context.get("lookahead_m"),
            },
            teacher_answer=teachers.teacher_bev_route_theta(local_bev_context),
        )
    )
    for component_type in ["connector", "crossing", "entrance"]:
        samples.append(
            _sample(
                sample_id=f"{sample_id}:{component_type}",
                task_type="bev_component_exists",
                inputs={
                    "component_type": component_type,
                    "connector_bev": local_bev_context.get("connector_bev") or [],
                    "semantic_bev_layers": local_bev_context.get("semantic_bev_layers") or {},
                },
                teacher_answer=teachers.teacher_bev_component_exists(local_bev_context, component_type),
            )
        )
        samples.append(
            _sample(
                sample_id=f"{sample_id}:{component_type}",
                task_type="bev_component_theta_range",
                inputs={
                    "component_type": component_type,
                    "component_points_bev": local_bev_context.get("connector_bev") if component_type == "connector" else [],
                    "semantic_bev_layers": local_bev_context.get("semantic_bev_layers") or {},
                },
                teacher_answer=teachers.teacher_bev_component_theta_range(local_bev_context, component_type),
            )
        )

    samples.append(
        _sample(
            sample_id=sample_id,
            task_type="bev_navigation_stage",
            inputs={
                "progress": route_context.get("progress") or {},
                "local_semantics": route_context.get("local_semantics") or {},
            },
            teacher_answer=teachers.teacher_bev_navigation_stage(route_context),
        )
    )

    samples.append(
        _sample(
            sample_id=sample_id,
            task_type="fpv_corridor_theta",
            inputs={
                "image_width": image_width,
                "image_height": image_height,
                "fpv_segmentation_summary": fpv_segmentation_summary,
                "rendered_prompt": rendered_prompt,
            },
            teacher_answer=teachers.teacher_fpv_corridor_theta(rendered_prompt, image_width),
        )
    )
    candidate_regions = _visual_regions_from_candidate_bundle(candidate_bundle_dict)
    samples.append(
        _sample(
            sample_id=sample_id,
            task_type="fpv_main_region_selection",
            inputs={
                "candidate_regions": candidate_regions,
                "fpv_segmentation_summary": fpv_segmentation_summary,
                "som_summary": som_summary,
            },
            teacher_answer=teachers.teacher_fpv_main_region_selection(candidate_regions),
        )
    )
    samples.append(
        _sample(
            sample_id=sample_id,
            task_type="fpv_scene_stage",
            inputs={
                "fpv_segmentation_summary": fpv_segmentation_summary,
                "som_summary": som_summary,
                "image_width": image_width,
                "image_height": image_height,
            },
            teacher_answer=teachers.teacher_fpv_scene_stage(fpv_segmentation_summary),
        )
    )
    samples.append(
        _sample(
            sample_id=sample_id,
            task_type="fpv_reject_all",
            inputs={
                "fpv_segmentation_summary": fpv_segmentation_summary,
                "som_summary": som_summary,
                "rendered_prompt": rendered_prompt,
                "segmentation_validation": segmentation_validation,
            },
            teacher_answer=teachers.teacher_fpv_reject_all(rendered_prompt, segmentation_validation),
        )
    )

    samples.append(
        _sample(
            sample_id=sample_id,
            task_type="crossview_theta_estimation",
            inputs={
                "route_context": route_context,
                "local_bev_context": local_bev_context,
                "candidate_bundle": candidate_bundle_dict or {},
                "image_width": image_width,
                "image_height": image_height,
            },
            teacher_answer=teachers.teacher_crossview_theta(candidate_bundle_dict or {}),
        )
    )
    samples.append(
        _sample(
            sample_id=sample_id,
            task_type="crossview_prompt_ranking",
            inputs={"candidate_bundle": candidate_bundle_dict or {}},
            teacher_answer=teachers.teacher_crossview_prompt_ranking(candidate_bundle_dict or {}),
        )
    )
    samples.append(
        _sample(
            sample_id=sample_id,
            task_type="crossview_reject_all",
            inputs={
                "candidate_bundle": candidate_bundle_dict or {},
                "fpv_segmentation_summary": fpv_segmentation_summary,
                "som_summary": som_summary,
            },
            teacher_answer=teachers.teacher_crossview_reject_all(candidate_bundle_dict or {}),
        )
    )

    samples.append(
        _sample(
            sample_id=sample_id,
            task_type="point_prompt",
            inputs={
                "rendered_prompt": rendered_prompt,
                "image_width": image_width,
                "image_height": image_height,
            },
            teacher_answer=teachers.teacher_point_prompt(rendered_prompt, image_width),
        )
    )
    samples.append(
        _sample(
            sample_id=sample_id,
            task_type="step_instruction",
            inputs={"route_context": route_context, "rendered_prompt": rendered_prompt},
            teacher_answer=teachers.teacher_step_instruction(route_context, rendered_prompt),
        )
    )
    samples.append(
        _sample(
            sample_id=sample_id,
            task_type="prompt_confidence",
            inputs={
                "route_context": route_context,
                "rendered_prompt": rendered_prompt,
                "segmentation_validation": segmentation_validation,
                "candidate_bundle": candidate_bundle_dict,
            },
            teacher_answer=teachers.teacher_prompt_confidence(
                route_context,
                rendered_prompt,
                segmentation_validation,
                candidate_bundle_dict,
            ),
        )
    )
    samples.append(
        _sample(
            sample_id=sample_id,
            task_type="reject_all",
            inputs={
                "rendered_prompt": rendered_prompt,
                "segmentation_validation": segmentation_validation,
                "arbiter_decision": arbiter_decision,
            },
            teacher_answer=teachers.teacher_grounded_reject(rendered_prompt, segmentation_validation, arbiter_decision),
        )
    )

    sample_dicts = [asdict(sample) for sample in samples]
    task_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for sample in sample_dicts:
        task_counts[sample["task_type"]] = task_counts.get(sample["task_type"], 0) + 1
        family_counts[sample["task_family"]] = family_counts.get(sample["task_family"], 0) + 1
    return {
        "sample_id": sample_id,
        "schema_version": "visual_prompt_qa_v1",
        "theta_definition": {
            "bev": "ego_aligned_atan2_x_y_right_positive",
            "fpv": "linear_u_mapping_right_positive",
            "range_deg": [-90.0, 90.0],
        },
        "task_counts": task_counts,
        "family_counts": family_counts,
        "samples": sample_dicts,
    }
