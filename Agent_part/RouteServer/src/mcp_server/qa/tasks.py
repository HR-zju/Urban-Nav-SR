from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import schema
from . import teachers


TeacherFn = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class QATaskSpec:
    task_family: str
    task_type: str
    answer_type: str
    input_dataclass: type
    teacher_dataclass: type
    output_dataclass: type
    teacher_fn: TeacherFn | None
    enabled_by_default: bool = True


TASK_REGISTRY: dict[str, QATaskSpec] = {
    "bev_route_theta": QATaskSpec(
        task_family="A_bev",
        task_type="bev_route_theta",
        answer_type="angle_deg_json",
        input_dataclass=schema.BEVRouteThetaInput,
        teacher_dataclass=schema.BEVRouteThetaTeacher,
        output_dataclass=schema.BEVRouteThetaOutput,
        teacher_fn=teachers.teacher_bev_route_theta,
    ),
    "bev_component_exists": QATaskSpec(
        task_family="A_bev",
        task_type="bev_component_exists",
        answer_type="yes_no",
        input_dataclass=schema.BEVComponentExistsInput,
        teacher_dataclass=schema.BEVComponentExistsTeacher,
        output_dataclass=schema.BEVComponentExistsOutput,
        teacher_fn=teachers.teacher_bev_component_exists,
    ),
    "bev_component_theta_range": QATaskSpec(
        task_family="A_bev",
        task_type="bev_component_theta_range",
        answer_type="component_geometry_json",
        input_dataclass=schema.BEVComponentThetaRangeInput,
        teacher_dataclass=schema.BEVComponentThetaRangeTeacher,
        output_dataclass=schema.BEVComponentThetaRangeOutput,
        teacher_fn=teachers.teacher_bev_component_theta_range,
    ),
    "bev_navigation_stage": QATaskSpec(
        task_family="A_bev",
        task_type="bev_navigation_stage",
        answer_type="mcq",
        input_dataclass=schema.BEVNavigationStageInput,
        teacher_dataclass=schema.BEVNavigationStageTeacher,
        output_dataclass=schema.BEVNavigationStageOutput,
        teacher_fn=teachers.teacher_bev_navigation_stage,
    ),
    "fpv_corridor_theta": QATaskSpec(
        task_family="F_fpv",
        task_type="fpv_corridor_theta",
        answer_type="angle_deg_json",
        input_dataclass=schema.FPVCorridorThetaInput,
        teacher_dataclass=schema.FPVCorridorThetaTeacher,
        output_dataclass=schema.FPVCorridorThetaOutput,
        teacher_fn=teachers.teacher_fpv_corridor_theta,
    ),
    "fpv_main_region_selection": QATaskSpec(
        task_family="F_fpv",
        task_type="fpv_main_region_selection",
        answer_type="region_id",
        input_dataclass=schema.FPVMainRegionSelectionInput,
        teacher_dataclass=schema.FPVMainRegionSelectionTeacher,
        output_dataclass=schema.FPVMainRegionSelectionOutput,
        teacher_fn=teachers.teacher_fpv_main_region_selection,
    ),
    "fpv_scene_stage": QATaskSpec(
        task_family="F_fpv",
        task_type="fpv_scene_stage",
        answer_type="mcq",
        input_dataclass=schema.FPVSceneStageInput,
        teacher_dataclass=schema.FPVSceneStageTeacher,
        output_dataclass=schema.FPVSceneStageOutput,
        teacher_fn=teachers.teacher_fpv_scene_stage,
    ),
    "fpv_reject_all": QATaskSpec(
        task_family="F_fpv",
        task_type="fpv_reject_all",
        answer_type="yes_no",
        input_dataclass=schema.FPVRejectAllInput,
        teacher_dataclass=schema.FPVRejectAllTeacher,
        output_dataclass=schema.FPVRejectAllOutput,
        teacher_fn=teachers.teacher_fpv_reject_all,
    ),
    "crossview_theta_estimation": QATaskSpec(
        task_family="B_crossview",
        task_type="crossview_theta_estimation",
        answer_type="angle_deg_json",
        input_dataclass=schema.CrossViewThetaInput,
        teacher_dataclass=schema.CrossViewThetaTeacher,
        output_dataclass=schema.CrossViewThetaOutput,
        teacher_fn=teachers.teacher_crossview_theta,
    ),
    "crossview_prompt_ranking": QATaskSpec(
        task_family="B_crossview",
        task_type="crossview_prompt_ranking",
        answer_type="prompt_id",
        input_dataclass=schema.CrossViewPromptRankingInput,
        teacher_dataclass=schema.CrossViewPromptRankingTeacher,
        output_dataclass=schema.CrossViewPromptRankingOutput,
        teacher_fn=teachers.teacher_crossview_prompt_ranking,
    ),
    "crossview_reject_all": QATaskSpec(
        task_family="B_crossview",
        task_type="crossview_reject_all",
        answer_type="yes_no",
        input_dataclass=schema.CrossViewRejectAllInput,
        teacher_dataclass=schema.CrossViewRejectAllTeacher,
        output_dataclass=schema.CrossViewRejectAllOutput,
        teacher_fn=teachers.teacher_crossview_reject_all,
    ),
    "point_prompt": QATaskSpec(
        task_family="C_grounded",
        task_type="point_prompt",
        answer_type="point_with_theta_json",
        input_dataclass=schema.PointPromptInput,
        teacher_dataclass=schema.PointPromptTeacher,
        output_dataclass=schema.PointPromptOutput,
        teacher_fn=teachers.teacher_point_prompt,
    ),
    "step_instruction": QATaskSpec(
        task_family="C_grounded",
        task_type="step_instruction",
        answer_type="text",
        input_dataclass=schema.StepInstructionInput,
        teacher_dataclass=schema.StepInstructionTeacher,
        output_dataclass=schema.StepInstructionOutput,
        teacher_fn=teachers.teacher_step_instruction,
    ),
    "prompt_confidence": QATaskSpec(
        task_family="C_grounded",
        task_type="prompt_confidence",
        answer_type="confidence_json",
        input_dataclass=schema.PromptConfidenceInput,
        teacher_dataclass=schema.PromptConfidenceTeacher,
        output_dataclass=schema.PromptConfidenceOutput,
        teacher_fn=teachers.teacher_prompt_confidence,
    ),
    "reject_all": QATaskSpec(
        task_family="C_grounded",
        task_type="reject_all",
        answer_type="yes_no",
        input_dataclass=schema.GroundedRejectInput,
        teacher_dataclass=schema.GroundedRejectTeacher,
        output_dataclass=schema.GroundedRejectOutput,
        teacher_fn=teachers.teacher_grounded_reject,
    ),
    "arrow_prompt": QATaskSpec(
        task_family="C_grounded",
        task_type="arrow_prompt",
        answer_type="arrow_json",
        input_dataclass=schema.ArrowPromptInput,
        teacher_dataclass=schema.ArrowPromptTeacher,
        output_dataclass=schema.ArrowPromptOutput,
        teacher_fn=None,
        enabled_by_default=False,
    ),
    "mask_prompt": QATaskSpec(
        task_family="C_grounded",
        task_type="mask_prompt",
        answer_type="mask_json",
        input_dataclass=schema.MaskPromptInput,
        teacher_dataclass=schema.MaskPromptTeacher,
        output_dataclass=schema.MaskPromptOutput,
        teacher_fn=None,
        enabled_by_default=False,
    ),
    "heatmap_prompt": QATaskSpec(
        task_family="C_grounded",
        task_type="heatmap_prompt",
        answer_type="heatmap_json",
        input_dataclass=schema.HeatmapPromptInput,
        teacher_dataclass=schema.HeatmapPromptTeacher,
        output_dataclass=schema.HeatmapPromptOutput,
        teacher_fn=None,
        enabled_by_default=False,
    ),
}


BASELINE_TASK_NAMES = [name for name, spec in TASK_REGISTRY.items() if spec.enabled_by_default]


def get_task_spec(task_type: str) -> QATaskSpec:
    return TASK_REGISTRY[task_type]
