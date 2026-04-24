from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


@dataclass
class ThetaLabel:
    theta_deg: float
    coarse_bin: str
    bin_index: int
    residual_deg: float
    confidence: float
    source: str


@dataclass
class PointUV:
    u: float
    v: float


@dataclass
class RangeM:
    min_m: float
    max_m: float


@dataclass
class QASample:
    sample_id: str
    task_family: str
    task_type: str
    answer_type: str
    inputs: dict[str, Any]
    teacher_answer: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class BEVRouteThetaInput:
    local_segment_bev: list[dict[str, float]]
    route_polyline_local: list[dict[str, float]]
    lookahead_m: float


@dataclass
class BEVRouteThetaTeacher:
    theta: ThetaLabel


@dataclass
class BEVRouteThetaOutput:
    theta: ThetaLabel


@dataclass
class BEVComponentExistsInput:
    component_type: str
    connector_bev: list[dict[str, float]]
    semantic_bev_layers: dict[str, Any]


@dataclass
class BEVComponentExistsTeacher:
    exists: bool
    confidence: float
    source: str


@dataclass
class BEVComponentExistsOutput:
    exists: bool
    confidence: float


@dataclass
class BEVComponentThetaRangeInput:
    component_type: str
    component_points_bev: list[dict[str, float]]
    semantic_bev_layers: dict[str, Any]


@dataclass
class BEVComponentThetaRangeTeacher:
    exists: bool
    theta: ThetaLabel | None
    range_m: RangeM | None
    span_deg: float | None


@dataclass
class BEVComponentThetaRangeOutput:
    exists: bool
    theta: ThetaLabel | None
    range_m: RangeM | None
    span_deg: float | None


@dataclass
class BEVNavigationStageInput:
    progress: dict[str, Any]
    local_semantics: dict[str, Any]


@dataclass
class BEVNavigationStageTeacher:
    stage: str
    confidence: float
    source: str


@dataclass
class BEVNavigationStageOutput:
    stage: str
    confidence: float


@dataclass
class FPVCorridorThetaInput:
    image_width: int
    image_height: int
    fpv_segmentation_summary: dict[str, Any] | None
    rendered_prompt: dict[str, Any] | None


@dataclass
class FPVCorridorThetaTeacher:
    theta: ThetaLabel | None
    reject_all: bool
    source: str


@dataclass
class FPVCorridorThetaOutput:
    theta: ThetaLabel | None
    reject_all: bool


@dataclass
class FPVMainRegionSelectionInput:
    candidate_regions: list[dict[str, Any]]
    fpv_segmentation_summary: dict[str, Any] | None


@dataclass
class FPVMainRegionSelectionTeacher:
    selected_region_id: str | None
    reject_all: bool
    confidence: float


@dataclass
class FPVMainRegionSelectionOutput:
    selected_region_id: str | None
    reject_all: bool
    confidence: float


@dataclass
class FPVSceneStageInput:
    fpv_segmentation_summary: dict[str, Any] | None
    image_width: int
    image_height: int


@dataclass
class FPVSceneStageTeacher:
    scene_stage: str
    confidence: float
    source: str


@dataclass
class FPVSceneStageOutput:
    scene_stage: str
    confidence: float


@dataclass
class FPVRejectAllInput:
    fpv_segmentation_summary: dict[str, Any] | None
    rendered_prompt: dict[str, Any] | None
    segmentation_validation: dict[str, Any] | None


@dataclass
class FPVRejectAllTeacher:
    reject_all: bool
    reason: str
    confidence: float


@dataclass
class FPVRejectAllOutput:
    reject_all: bool
    reason: str
    confidence: float


@dataclass
class CrossViewThetaInput:
    route_context: dict[str, Any]
    local_bev_context: dict[str, Any]
    candidate_bundle: dict[str, Any]
    image_width: int
    image_height: int


@dataclass
class CrossViewThetaTeacher:
    theta: ThetaLabel | None
    reject_all: bool
    source: str


@dataclass
class CrossViewThetaOutput:
    theta: ThetaLabel | None
    reject_all: bool


@dataclass
class CrossViewPromptRankingInput:
    candidate_bundle: dict[str, Any]


@dataclass
class CrossViewPromptRankingTeacher:
    selected_prompt_id: str | None
    delta_theta_deg: float | None
    reject_all: bool
    confidence: float


@dataclass
class CrossViewPromptRankingOutput:
    selected_prompt_id: str | None
    delta_theta_deg: float | None
    reject_all: bool


@dataclass
class CrossViewRejectAllInput:
    candidate_bundle: dict[str, Any]
    fpv_segmentation_summary: dict[str, Any] | None


@dataclass
class CrossViewRejectAllTeacher:
    reject_all: bool
    reason: str
    confidence: float


@dataclass
class CrossViewRejectAllOutput:
    reject_all: bool
    reason: str
    confidence: float


@dataclass
class PointPromptInput:
    rendered_prompt: dict[str, Any] | None
    image_width: int
    image_height: int


@dataclass
class PointPromptTeacher:
    target_point_uv: PointUV | None
    theta: ThetaLabel | None
    reject_all: bool
    source: str


@dataclass
class PointPromptOutput:
    target_point_uv: PointUV | None
    theta: ThetaLabel | None
    reject_all: bool


@dataclass
class StepInstructionInput:
    route_context: dict[str, Any]
    rendered_prompt: dict[str, Any] | None


@dataclass
class StepInstructionTeacher:
    instruction: str
    stage: str
    confidence: float


@dataclass
class StepInstructionOutput:
    instruction: str
    confidence: float


@dataclass
class PromptConfidenceInput:
    route_context: dict[str, Any]
    rendered_prompt: dict[str, Any] | None
    segmentation_validation: dict[str, Any] | None
    candidate_bundle: dict[str, Any] | None


@dataclass
class PromptConfidenceTeacher:
    confidence: float
    factors: dict[str, float]


@dataclass
class PromptConfidenceOutput:
    confidence: float


@dataclass
class GroundedRejectInput:
    rendered_prompt: dict[str, Any] | None
    segmentation_validation: dict[str, Any] | None
    arbiter_decision: dict[str, Any]


@dataclass
class GroundedRejectTeacher:
    reject_all: bool
    reason: str
    confidence: float


@dataclass
class GroundedRejectOutput:
    reject_all: bool
    reason: str
    confidence: float


@dataclass
class ArrowPromptInput:
    rendered_prompt: dict[str, Any] | None


@dataclass
class ArrowPromptTeacher:
    enabled: bool = False


@dataclass
class ArrowPromptOutput:
    enabled: bool = False


@dataclass
class MaskPromptInput:
    rendered_prompt: dict[str, Any] | None


@dataclass
class MaskPromptTeacher:
    enabled: bool = False


@dataclass
class MaskPromptOutput:
    enabled: bool = False


@dataclass
class HeatmapPromptInput:
    rendered_prompt: dict[str, Any] | None


@dataclass
class HeatmapPromptTeacher:
    enabled: bool = False


@dataclass
class HeatmapPromptOutput:
    enabled: bool = False


def to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    return value
