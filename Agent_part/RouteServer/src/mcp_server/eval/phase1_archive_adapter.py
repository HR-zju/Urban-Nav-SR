"""Adapt real SideWalk Phase1 archive rows into Agentic Stage2 cases.

Phase1 archive manifests are already ms-swift-ready rows:

``messages/images/pixel_trajectory/sidewalk_meta`` plus optional trajectory
and segmentation supervision.  Stage2 must not reinterpret those labels.  This
adapter only changes the dialogue wrapper: the original assistant CoT becomes
``ground_truth.phase1_response`` and remains LM supervision for the final
assistant turn after tool interaction.

数学/因果约束:
    Phase1 的轨迹监督是固定目标 ``Y``。Stage2 只在输入序列中插入工具交互上下文
    ``C_tool``，然后在 ``C_tool`` 之后放置 ``[TRAJ_i]`` readout token。因此适配器
    必须原样保留 ``Y``，不能因为工具调用而重采样或重写轨迹标签。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from mcp_server.agent.schemas import AgentStage2Case
from mcp_server.agent.tool_schema import get_tool_definitions


PHASE1_ARCHIVE_STAGE2_ADAPTER_VERSION = "sidewalk_phase1_archive_to_stage2_case_v1"

SUPERVISION_KEYS = (
    "pixel_trajectory",
    "trajectory_fpv_t",
    "trajectory_fpv_tv",
    "trajectory_ego_t",
    "trajectory_policy",
    "semantic_segmentation_path",
    "semantic_mask_path",
)

SIDECAR_PATH_KEYS = (
    "trajectory_label_path",
    "bev_patch_meta_path",
    "qa_bundle_path",
    "cot_sections_path",
    "semantic_segmentation_path",
    "semantic_mask_path",
    "segmentation_path",
    "mask_path",
)


def phase1_archive_row_to_stage2_case(
    row: dict[str, Any],
    *,
    index: int = 0,
    source_path: str | Path | None = None,
    case_id_prefix: str = "phase1-stage2",
    resolve_relative_paths: bool = True,
) -> AgentStage2Case:
    """Convert one Phase1 archive/pre-archive row to ``AgentStage2Case``.

    输入:
        ``row`` 可以是 ``sidewalk_phase1_archive/manifests/train.jsonl`` 的一行，
        也可以是归档前 ``phase1_cot_*.jsonl`` 的一行。

    输出:
        ``AgentStage2Case``。其中：
        - ``ground_truth.phase1_response`` 保存原 Phase1 assistant CoT。
        - ``ground_truth.pixel_trajectory`` 等监督字段原样透传。
        - ``observation.images`` 以 ``{"fpv": ..., "bev": ...}`` 形式保存。
    """

    base_dir = Path(source_path).expanduser().resolve().parent if source_path and resolve_relative_paths else None
    messages = _messages(row)
    system_prompt = _first_message_content(messages, "system")
    user_prompt = _first_message_content(messages, "user") or str(row.get("prompt") or "")
    assistant_response = _last_message_content(messages, "assistant") or str(row.get("response") or "")
    sidewalk_meta = dict(row.get("sidewalk_meta") or {})

    images = _extract_images(row, base_dir=base_dir)
    observation = _build_observation(row, sidewalk_meta=sidewalk_meta, images=images, base_dir=base_dir)
    ground_truth = _build_ground_truth(row, sidewalk_meta=sidewalk_meta, assistant_response=assistant_response, base_dir=base_dir)
    metadata = _build_metadata(
        row,
        sidewalk_meta=sidewalk_meta,
        source_path=source_path,
        index=index,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        assistant_response=assistant_response,
    )

    return AgentStage2Case(
        case_id=_case_id(row, sidewalk_meta=sidewalk_meta, index=index, prefix=case_id_prefix),
        user_request=_extract_user_request(row, user_prompt),
        observation=observation,
        dialogue_history=messages,
        tools=get_tool_definitions(),
        ground_truth=ground_truth,
        split=str(row.get("split") or sidewalk_meta.get("split") or "train"),
        locale=str(row.get("locale") or sidewalk_meta.get("locale") or "zh-CN"),
        metadata=metadata,
    )


def iter_phase1_archive_stage2_cases(
    path: str | Path,
    *,
    limit: int | None = None,
    case_id_prefix: str = "phase1-stage2",
    resolve_relative_paths: bool = True,
) -> Iterator[AgentStage2Case]:
    """Yield Stage2 cases from a Phase1 archive JSONL manifest."""

    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= int(limit):
                break
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                continue
            yield phase1_archive_row_to_stage2_case(
                row,
                index=index,
                source_path=source,
                case_id_prefix=case_id_prefix,
                resolve_relative_paths=resolve_relative_paths,
            )


def load_phase1_archive_stage2_cases(
    path: str | Path,
    *,
    limit: int | None = None,
    case_id_prefix: str = "phase1-stage2",
    resolve_relative_paths: bool = True,
) -> list[AgentStage2Case]:
    """Load a Phase1 archive manifest as a list of Stage2 cases."""

    return list(
        iter_phase1_archive_stage2_cases(
            path,
            limit=limit,
            case_id_prefix=case_id_prefix,
            resolve_relative_paths=resolve_relative_paths,
        )
    )


def write_phase1_archive_stage2_cases(
    cases: Iterable[AgentStage2Case],
    path: str | Path,
) -> None:
    """Write adapted Stage2 cases for manual review/debugging."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_messages = row.get("messages") or []
    if not isinstance(raw_messages, list):
        return []
    messages: list[dict[str, Any]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        if not role:
            continue
        messages.append({"role": role, "content": item.get("content")})
    return messages


def _first_message_content(messages: list[dict[str, Any]], role: str) -> str:
    for message in messages:
        if message.get("role") == role and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


def _last_message_content(messages: list[dict[str, Any]], role: str) -> str:
    for message in reversed(messages):
        if message.get("role") == role and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


def _extract_user_request(row: dict[str, Any], user_prompt: str) -> str:
    instruction = str(row.get("instruction") or "").strip()
    if instruction:
        return instruction
    match = re.search(r"^Instruction:\s*(.+?)\s*$", user_prompt, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    stripped = _strip_image_tokens(user_prompt)
    if stripped:
        return stripped
    return "Navigate safely on the sidewalk."


def _strip_image_tokens(text: str) -> str:
    value = str(text or "").strip()
    while value.startswith("<image>"):
        value = value[len("<image>") :].lstrip()
    return value


def _extract_images(row: dict[str, Any], *, base_dir: Path | None) -> list[str]:
    raw_images = row.get("images")
    if isinstance(raw_images, list):
        return [_resolve_path(item, base_dir=base_dir) for item in raw_images if item]
    images = []
    if row.get("fpv_image_path"):
        images.append(_resolve_path(row["fpv_image_path"], base_dir=base_dir))
    if row.get("bev_image_path"):
        images.append(_resolve_path(row["bev_image_path"], base_dir=base_dir))
    return images


def _build_observation(
    row: dict[str, Any],
    *,
    sidewalk_meta: dict[str, Any],
    images: list[str],
    base_dir: Path | None,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "images": _image_dict(images),
        "route_context": dict(row.get("route_context") or sidewalk_meta.get("route_context") or {}),
        "local_bev_context": dict(row.get("local_bev_context") or sidewalk_meta.get("local_bev_context") or {}),
    }
    for key in (
        "clip_id",
        "frame_idx",
        "frame_name",
        "timestamp",
        "heading_deg",
        "route_theta_deg",
        "intent_gt",
        "trajectory_source",
        "image_width",
        "image_height",
    ):
        value = row.get(key, sidewalk_meta.get(key))
        if value is not None:
            observation[key] = value

    location = _current_location(row, sidewalk_meta)
    if location:
        observation["current_location"] = location
        observation["gps"] = _gps_dict(row, sidewalk_meta)
    for key in SIDECAR_PATH_KEYS:
        value = row.get(key) or sidewalk_meta.get(key)
        if value:
            observation[key] = _resolve_path(value, base_dir=base_dir)
    return _drop_empty(observation)


def _build_ground_truth(
    row: dict[str, Any],
    *,
    sidewalk_meta: dict[str, Any],
    assistant_response: str,
    base_dir: Path | None,
) -> dict[str, Any]:
    ground_truth: dict[str, Any] = {}
    for key in SUPERVISION_KEYS:
        value = row.get(key)
        if value is not None:
            ground_truth[key] = _resolve_path(value, base_dir=base_dir) if key.endswith("_path") else value
    for key in ("intent_gt", "route_theta_deg", "trajectory_source"):
        value = row.get(key, sidewalk_meta.get(key))
        if value is not None:
            ground_truth[key] = value
    for key in ("intent", "confidence_level", "trajectory_type", "trajectory_validity"):
        value = row.get(key) or sidewalk_meta.get(key)
        if value is not None:
            ground_truth[key] = value
    if assistant_response.strip():
        ground_truth["phase1_response"] = assistant_response
    for key in ("semantic_segmentation_path", "semantic_mask_path", "segmentation_path", "mask_path"):
        value = row.get(key) or sidewalk_meta.get(key)
        if value:
            target_key = "semantic_mask_path" if "mask" in key else "semantic_segmentation_path"
            ground_truth.setdefault(target_key, _resolve_path(value, base_dir=base_dir))
    return ground_truth


def _build_metadata(
    row: dict[str, Any],
    *,
    sidewalk_meta: dict[str, Any],
    source_path: str | Path | None,
    index: int,
    system_prompt: str,
    user_prompt: str,
    assistant_response: str,
) -> dict[str, Any]:
    return _drop_empty(
        {
            "adapter_schema_version": PHASE1_ARCHIVE_STAGE2_ADAPTER_VERSION,
            "source_dataset": "sidewalk_phase1_archive",
            "source_jsonl": str(source_path) if source_path else None,
            "source_jsonl_index": index,
            "phase1_id": row.get("id") or sidewalk_meta.get("source_id"),
            "phase1_archive_schema_version": sidewalk_meta.get("archive_schema_version"),
            "phase1_cot_schema_version": sidewalk_meta.get("cot_schema_version"),
            "phase1_prompt_template": row.get("prompt_template"),
            "phase1_system_prompt": system_prompt,
            "phase1_user_prompt": user_prompt,
            "phase1_response_chars": len(assistant_response),
            "phase1_sidewalk_meta": sidewalk_meta,
        }
    )


def _image_dict(images: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    if images:
        result["fpv"] = images[0]
    if len(images) > 1:
        result["bev"] = images[1]
    for index, image in enumerate(images[2:], start=2):
        result[f"image_{index}"] = image
    return result


def _current_location(row: dict[str, Any], sidewalk_meta: dict[str, Any]) -> str:
    for payload in (row, sidewalk_meta):
        for key in ("current_location", "origin", "location"):
            value = payload.get(key)
            if value:
                return _location_text(value)
        lng = payload.get("lng", payload.get("longitude"))
        lat = payload.get("lat", payload.get("latitude"))
        if lng is not None and lat is not None:
            return f"{float(lng)},{float(lat)}"
    return ""


def _gps_dict(row: dict[str, Any], sidewalk_meta: dict[str, Any]) -> dict[str, float] | None:
    for payload in (row, sidewalk_meta):
        lng = payload.get("lng", payload.get("longitude"))
        lat = payload.get("lat", payload.get("latitude"))
        if lng is not None and lat is not None:
            return {"longitude": float(lng), "latitude": float(lat)}
    return None


def _location_text(value: Any) -> str:
    if isinstance(value, dict):
        location = value.get("location")
        if isinstance(location, str) and location.strip():
            return location.strip()
        lng = value.get("lng", value.get("lon", value.get("longitude")))
        lat = value.get("lat", value.get("latitude"))
        if lng is not None and lat is not None:
            return f"{float(lng)},{float(lat)}"
    return str(value or "").strip()


def _resolve_path(value: Any, *, base_dir: Path | None) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    path = Path(text).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return str(path.resolve() if path.exists() else path)


def _case_id(row: dict[str, Any], *, sidewalk_meta: dict[str, Any], index: int, prefix: str) -> str:
    raw = (
        row.get("case_id")
        or row.get("id")
        or sidewalk_meta.get("source_id")
        or "_".join(
            str(item)
            for item in (
                row.get("clip_id") or sidewalk_meta.get("clip_id") or sidewalk_meta.get("archive_clip_id"),
                row.get("frame_name") or sidewalk_meta.get("frame_name") or sidewalk_meta.get("archive_frame_name"),
            )
            if item not in (None, "")
        )
        or f"row-{index:06d}"
    )
    return f"{prefix}-{_safe_id(raw)}"


def _safe_id(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z_.:-]+", "-", text)
    return text.strip("-") or "sample"


def _drop_empty(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}
