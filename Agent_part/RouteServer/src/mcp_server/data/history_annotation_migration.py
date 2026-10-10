"""Plan section reuse between immutable canonical source revisions.

Media identity is supplied by a provenance index, not inferred from basenames.
Planning is offline. Annotation explicitly invokes the existing teacher runner
and leaves original annotation files intact.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

from mcp_server.contracts.reasoning_dataset_v4 import (
    AnnotationStageRecordV4, CandidateStructureRecordV4, GroundingOutputsV4,
    ReasoningAnnotationV4, ReasoningSourceV4,
)
from mcp_server.contracts.stage_i_sft import StageIFullCoTViewV2
from mcp_server.data.canonical.reasoning_v4_composer import REASONING_V4_COMPOSER_VERSION
from mcp_server.data.canonical.reasoning_v4_source import build_history_reasoning_source_v4
from mcp_server.data.canonical.reasoning_v4_task_canonicalization import (
    canonicalize_task_instruction_v4, task_instruction_fingerprint_v4,
)
from mcp_server.data.canonical.reasoning_v4_views import export_reasoning_training_views_v4
from mcp_server.data.io.jsonl import append_jsonl_row, iter_jsonl, iter_jsonl_with_offsets, write_jsonl_atomic
from mcp_server.data.history_label_reuse import anchor_label_evidence, same_evidence_image
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.stage_i.views import project_stage_i_reasoning_view_with_audit_v2
from mcp_server.teacher.reasoning_v4_batch import (
    ReasoningV4BatchConfig, ReasoningV4TeacherBackend, annotate_reasoning_source_v4,
)
from mcp_server.teacher.reasoning_v4_image_policy import get_reasoning_v4_section_image_policy
from mcp_server.teacher.reasoning_v4_profiles import get_reasoning_v4_teacher_profiles
from mcp_server.teacher.reasoning_v4_teacher import build_reasoning_v4_section_question


SECTION_FIELDS = {
    "TaskGrounding": "task_grounding",
    "SceneGrounding": "scene_grounding",
    "LocalTraversability": "local_traversability",
    "NavigationContextGrounding": "navigation_context_grounding",
    "InteractionConstraintGrounding": "interaction_constraint_grounding",
}


def build_migration_sources(dataset_root: Path, dataset: str, destination: Path,
                            *, limit: int | None = None,
                            bev_tolerance_ns: int | None = None) -> dict[str, Any]:
    """Stream packaged history observations through the existing source builder.

    Args:
        dataset_root: Extracted portable dataset root.
        dataset: Dataset family named in the portable receipt.
        destination: New canonical source JSONL, created exclusively.
        limit: Optional bounded subset for initial real-data verification.
        bev_tolerance_ns: Explicit original generation budget for legacy
            packages that omitted it. Obtain this from the frozen clip plans.

    Returns:
        Number of canonical sources written and exact input/output paths.

    Raises:
        ValueError: If clip identity or anchor is inconsistent.
        FileExistsError: If the destination already exists.
    """
    family = dataset_root / dataset
    clip_map = {}
    for item in iter_jsonl(family / "packages/clips.jsonl"):
        if item["status"] == "materialized":
            clip = item["clip_id"]
            if clip in clip_map:
                raise ValueError(f"duplicate clip identity: {clip}")
            clip_map[clip] = family / "packages" / item["package_directory"]
    budgets = {}
    source_path = family / "manifests" / f"{dataset}_observations_direction12.jsonl"
    count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for row in iter_jsonl(source_path):
            if limit is not None and count >= limit:
                break
            parent = row["source_parent_id"]
            prefix, clip = row["split_group_id"].split(":", 1)
            original = row["source_row"]
            if dataset == "sidewalk":
                parent_clip, anchor = parent.removeprefix(dataset + ":").rsplit(":", 1)
                if parent_clip != clip:
                    raise ValueError(f"SideWalk parent clip mismatch: {parent}")
            else:
                if original["clip_id"] != clip:
                    raise ValueError(f"source clip mismatch: {parent}")
                anchor = original["frame_idx"]
            if prefix != dataset or int(anchor) != row["observation"]["anchor_row"]:
                raise ValueError(f"history anchor identity mismatch: {parent}")
            package = clip_map[clip]
            if clip not in budgets:
                receipt = json.loads((package / "manifests/stage_ii_direct_action_q16_report.json").read_text())
                budgets[clip] = receipt["max_lag_ns"]
            row = copy.deepcopy(row)
            observation = row["observation"]
            bev = observation["bev"]
            if bev is not None:
                # 旧媒体包漏写预算时，从已核实的生成合同显式补回；不以 FPV lag 替代 BEV age。
                if "bev_tolerance_ns" not in bev:
                    if bev_tolerance_ns is None:
                        raise ValueError(f"{parent}: supply the original --bev-tolerance-ns budget")
                    bev["bev_tolerance_ns"] = bev_tolerance_ns
                elif bev_tolerance_ns is not None and bev["bev_tolerance_ns"] != bev_tolerance_ns:
                    raise ValueError(f"{parent}: supplied BEV budget conflicts with the package")
                age_ns = observation["source_ns"][-1] - bev["timestamp"]
                if age_ns != bev["age_ns"] or not 0 <= age_ns <= bev["bev_tolerance_ns"]:
                    raise ValueError(f"{parent}: BEV age violates the original generation budget")
            package_relative = package.relative_to(dataset_root)
            paths = [*observation["fpv_paths"]]
            if observation["bev"] is not None:
                paths.append(observation["bev"]["bev_image_path"])
            portable = {}
            for path in paths:
                relative = (package_relative / path).as_posix()
                if Path(path).is_absolute() or ".." in Path(path).parts:
                    raise ValueError(f"expected package-relative media path: {path}")
                if not (dataset_root / relative).is_file():
                    raise FileNotFoundError(dataset_root / relative)
                portable[path] = relative
            observation["fpv_paths"] = [portable[path] for path in observation["fpv_paths"]]
            if observation["bev"] is not None:
                observation["bev"]["bev_image_path"] = portable[observation["bev"]["bev_image_path"]]
            # 指纹只绑定可迁移路径；根目录仅参与运行时文件读取，迁移机器不改变 source 身份。
            source = build_history_reasoning_source_v4(
                row, index=count,
                config=ReasoningSourceBuildConfig(dataset_source=dataset,
                    media_root=None, max_fpv_frames=9),
                max_lag_ns=budgets[clip],
            )
            handle.write(json.dumps(source.model_dump(mode="json"), ensure_ascii=False) + "\n")
            count += 1
    return {"source_rows": count, "input": source_path.relative_to(dataset_root).as_posix(),
            "output": destination.name, "media_path_base": "dataset_root",
            "legacy_bev_tolerance_ns": bev_tolerance_ns}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def normalize_media_references(value: Any, identities: Mapping[str, str]) -> Any:
    """Replace media references with proven frame identities for comparison.

    Args:
        value: JSON-compatible teacher input payload.
        identities: Exact media-reference to source-frame identity mapping.

    Returns:
        A new payload with matching string references replaced. Other values
        remain unchanged, including timestamps and direction conditions.
    """
    if isinstance(value, str):
        return identities.get(value, value)
    if isinstance(value, list):
        return [normalize_media_references(item, identities) for item in value]
    if isinstance(value, dict):
        return {key: normalize_media_references(item, identities) for key, item in value.items()}
    return value


def decide_section_reuse(
    old_source: ReasoningSourceV4,
    new_source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
    section: str,
    *,
    image_policy_id: str,
    direction_registry_version: str,
    media_identities: Mapping[str, str],
    reuse_policy: str = "exact_input",
    anchor_records: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare a grounding section's real teacher inputs and producer policy.

    Args:
        old_source: Frozen source that produced the old annotation.
        new_source: Destination source revision.
        annotation: Old annotation with its original producer records.
        section: One of the five canonical grounding names.
        image_policy_id: Destination section routing policy.
        direction_registry_version: Destination direction registry.
        media_identities: Path to proven source-frame identity mapping.
        reuse_policy: Historical exact_input or preferred anchor_label_v1.
        anchor_records: Optional legacy sampling evidence keyed by source fingerprint.

    Returns:
        Disposition, reason, and compared input digests. A changed input is a
        rerun candidate, not proof that the old answer is semantically wrong.

    Raises:
        ValueError: If the old annotation belongs to a different source.
    """
    if (annotation.source_group_id != old_source.observation.source_group_id
            or annotation.source_fingerprint != old_source.source_fingerprint):
        raise ValueError("annotation must match its original source revision")
    record = annotation.stage_records.get(section)
    value = getattr(annotation.grounding_outputs, SECTION_FIELDS[section])
    if record is None or record.status != "accepted" or value is None:
        return {"disposition": "missing", "reason": "accepted_section_missing"}
    if section == "TaskGrounding" and canonicalize_task_instruction_v4(new_source.observation.instruction) is not None:
        return {"disposition": "rebuild_deterministic", "reason": "task_registry"}
    if section == "TaskGrounding" and new_source.observation.instruction is None:
        return {"disposition": "rerun_teacher", "reason": "task_instruction_missing"}
    if record.producer_kind == "deterministic":
        return {"disposition": "rebuild_deterministic", "reason": "task_registry"}
    if reuse_policy == "anchor_label_v1":
        if section == "TaskGrounding":
            equal = task_instruction_fingerprint_v4(old_source.observation.instruction) == \
                task_instruction_fingerprint_v4(new_source.observation.instruction)
            return {"disposition": "reuse_instruction" if equal else "pending",
                    "reason": "task_instruction_equal" if equal else "task_instruction_changed"}
        return _anchor_section_decision(old_source, new_source, section, value,
                                        media_identities, anchor_records)
    if reuse_policy != "exact_input":
        raise ValueError(f"unsupported reuse policy: {reuse_policy}")
    profiles = get_reasoning_v4_teacher_profiles(
        image_policy_id, direction_registry_version=direction_registry_version)
    profile = profiles[section]
    if (record.producer_profile_id != profile.profile_id
            or record.prompt_policy_digest != profile.policy_digest
            or record.derivation_version != profile.derivation_version):
        return {"disposition": "rerun_teacher", "reason": "producer_contract_changed"}
    if section == "TaskGrounding":
        # 沿用 batch runner 的 instruction-keyed taxonomy cache 合同；
        # Task 的图像上下文变化不重新定义指令分类，原 producer 记录继续保留。
        old_instruction = task_instruction_fingerprint_v4(old_source.observation.instruction)
        new_instruction = task_instruction_fingerprint_v4(new_source.observation.instruction)
        same_instruction = old_instruction == new_instruction
        return {
            "disposition": "reuse_instruction" if same_instruction else "rerun_teacher",
            "reason": "task_instruction_equal" if same_instruction else "task_instruction_changed",
            "old_input_digest": old_instruction, "new_input_digest": new_instruction,
            "comparison_scope": "task_instruction_fingerprint_v4",
        }
    old_policy = get_reasoning_v4_section_image_policy(
        annotation.metadata["section_image_policy_id"])
    new_policy = get_reasoning_v4_section_image_policy(image_policy_id)
    inputs = []
    for source, policy in ((old_source, old_policy), (new_source, new_policy)):
        observation = policy.project_observation(source.observation, section)
        # 源帧身份由索引证明；路径迁移不改变图像证据，文件名相同不能证明同一帧。
        unresolved = [path for path in observation.images if path not in media_identities]
        if unresolved:
            return {"disposition": "rerun_teacher", "reason": "media_identity_unresolved",
                    "unresolved_media": unresolved}
        question = build_reasoning_v4_section_question(observation, profile)
        payload = json.loads(question.split("\n\n", 1)[1])
        inputs.append({
            "prompt": normalize_media_references(payload, media_identities),
            "images": [media_identities[path] for path in observation.images],
        })
    fingerprints = [_digest(item) for item in inputs]
    old_visible = inputs[0]["prompt"]["section_visible_input"]
    new_visible = inputs[1]["prompt"]["section_visible_input"]
    changed = [key for key in sorted(old_visible.keys() | new_visible.keys())
               if old_visible.get(key) != new_visible.get(key)]
    if inputs[0]["images"] != inputs[1]["images"]:
        changed.append("image_frame_identities")
    same = fingerprints[0] == fingerprints[1]
    return {
        "disposition": "reuse_exact" if same else "rerun_teacher",
        "reason": "section_input_equal" if same else "section_input_changed",
        "old_input_digest": fingerprints[0],
        "new_input_digest": fingerprints[1],
        "changed_inputs": changed,
    }


def _anchor_section_decision(old, new, section, value, identities, anchor_records):
    """Reuse current-state fields while retaining their original producer provenance."""
    same, reason, evidence = anchor_label_evidence(
        old, new, media_identities=identities, anchor_records=anchor_records,
    )
    decision = {"anchor_evidence": evidence, "comparison_scope": "anchor_label_v1"}
    if not same:
        return {**decision, "disposition": "pending", "reason": reason}
    before, after = old.observation, new.observation
    if before.travel_mode != after.travel_mode:
        return {**decision, "disposition": "pending", "reason": "travel_mode_changed"}
    # Scene 四个字段属于当前环境摘要，不将历史图片列表或 producer digest 当作标签目标。
    if section == "LocalTraversability":
        keys = ("affordance_probe_profile_id", "affordance_probe_profile_version")
        if any(getattr(before.direction_context, key) != getattr(after.direction_context, key) for key in keys):
            return {**decision, "disposition": "pending", "reason": "affordance_direction_contract_changed"}
    if section in {"NavigationContextGrounding", "InteractionConstraintGrounding"}:
        if before.instruction != after.instruction:
            return {**decision, "disposition": "pending", "reason": "instruction_changed"}
        uses_direction = (section == "NavigationContextGrounding" or
                          any(item.affected_direction_labels for item in value.constraints))
        keys = ("action_profile_id", "action_profile_version")
        if uses_direction and any(getattr(before.direction_context, key) != getattr(after.direction_context, key)
                                  for key in keys):
            return {**decision, "disposition": "pending", "reason": "action_direction_contract_changed"}
    if section == "NavigationContextGrounding":
        if before.model_visible_context != after.model_visible_context:
            return {**decision, "disposition": "pending", "reason": "navigation_context_changed"}
        if "bev.current" in before.image_roles or "bev.current" in after.image_roles:
            if not same_evidence_image(before, after, "bev.current", identities):
                return {**decision, "disposition": "pending", "reason": "bev_alignment_unresolved"}
            contracts = [next(item.bev_route_evidence for item in obs.evidence_catalog
                              if str(item.ref) == "bev.current") for obs in (before, after)]
            if contracts[0] != contracts[1]:
                return {**decision, "disposition": "pending", "reason": "bev_interpretation_changed"}
    return {**decision, "disposition": "reuse_anchor", "reason": "current_state_label_equal"}


def plan_canonical_migration(config: Mapping[str, Any], output: Path,
                             *, input_root: Path) -> dict[str, Any]:
    """Write a reproducible migration plan for already built canonical sources.

    Args:
        config: New sources, ordered old runs, media index, and teacher policy.
            Later old runs have higher priority for accepted sections.
        output: New run directory, created exclusively.
        input_root: Runtime root for config-relative input artifact paths.

    Returns:
        Counts of section dispositions and source coverage.

    Raises:
        ValueError: For ambiguous sources or annotation lineage mismatches.
        FileExistsError: If the destination already exists.
    """
    def input_path(value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"configuration path must be input-root-relative: {value}")
        return input_root / path

    identities: dict[str, str] = {}
    anchor_records = ({row["source_fingerprint"]: row for row in
                       iter_jsonl(input_path(config["anchor_records_jsonl"]))}
                      if "anchor_records_jsonl" in config else {})
    for row in iter_jsonl(input_path(config["media_identity_jsonl"])):
        path, identity = row["path"], row["identity"]
        if path in identities and identities[path] != identity:
            raise ValueError(f"conflicting media identity: {path}")
        identities[path] = identity
    candidates: dict[str, list[dict[str, Any]]] = {}
    for run in config["old_runs"]:
        sources = {}
        for record_index, (offset, row) in enumerate(iter_jsonl_with_offsets(input_path(run["sources"]))):
            source = ReasoningSourceV4.model_validate(row)
            key = source.observation.source_group_id
            if key in sources:
                raise ValueError(f"duplicate old source: {key}")
            sources[key] = (source.source_fingerprint, record_index, offset)
        # append-only 中失败的重试不会覆盖此前已经成功的独立 section。
        for record_index, (offset, row) in enumerate(iter_jsonl_with_offsets(input_path(run["annotations"]))):
            annotation = ReasoningAnnotationV4.model_validate(row)
            fingerprint, source_index, source_offset = sources[annotation.source_group_id]
            if annotation.source_fingerprint != fingerprint:
                raise ValueError(f"old source revision mismatch: {annotation.source_group_id}")
            # 序号与 iter_jsonl 一致：从零开始，只计非空记录；路径相对于 input_root。
            # 保留文件位置使补标可直接取回选中的旧结果，digest 仅负责版本核对。
            origin = {
                "sources": run["sources"], "source_record_index": source_index,
                "annotations": run["annotations"], "annotation_record_index": record_index,
                "source_byte_offset": source_offset, "annotation_byte_offset": offset,
            }
            candidates.setdefault(annotation.source_group_id, []).append(origin)
    output.mkdir(parents=True, exist_ok=False)
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    with ExitStack() as inputs, (output / "sources.jsonl").open("x") as source_file, (output / "migration_plan.jsonl").open("x") as plan_file:
        handles = {}

        def read_at(path, offset):
            if path not in handles:
                handles[path] = inputs.enter_context(input_path(path).open("rb"))
            handle = handles[path]
            handle.seek(offset)
            return json.loads(handle.readline())

        for raw in iter_jsonl(input_path(config["new_sources"])):
            new = ReasoningSourceV4.model_validate(raw)
            key = new.observation.source_group_id
            if key in seen:
                raise ValueError(f"duplicate new source: {key}")
            seen.add(key)
            # 全局只保存位置；同一 anchor 的候选按需解析，用完即可释放文本和模型对象。
            parsed = [(
                ReasoningSourceV4.model_validate(read_at(origin["sources"], origin["source_byte_offset"])),
                ReasoningAnnotationV4.model_validate(read_at(origin["annotations"], origin["annotation_byte_offset"])),
                origin,
            ) for origin in reversed(candidates.get(key, []))]
            sections = {}
            for name in SECTION_FIELDS:
                result = {"disposition": "missing", "reason": (
                    "accepted_section_missing" if parsed else "old_annotation_missing")}
                for old, annotation, origin in parsed:
                    candidate = decide_section_reuse(
                        old, new, annotation, name,
                        image_policy_id=config["section_image_policy_id"],
                        direction_registry_version=config["direction_registry_version"],
                        media_identities=identities,
                        reuse_policy=config.get("reuse_policy", "exact_input"), anchor_records=anchor_records,
                    )
                    if candidate["disposition"] == "missing":
                        continue
                    candidate["origin"] = origin
                    candidate["old_source_fingerprint"] = old.source_fingerprint
                    candidate["old_annotation_digest"] = _digest(annotation.model_dump(mode="json"))
                    # 优先级较高但尚待对齐的记录，不阻挡较早、已满足复用条件的记录。
                    if result["disposition"] == "missing":
                        result = candidate
                    if candidate["disposition"] in {
                        "reuse_exact", "reuse_anchor", "reuse_instruction", "rebuild_deterministic",
                    }:
                        result = candidate
                        break
                sections[name] = result
                counts[f"{name}:{result['disposition']}"] += 1
            counts["sources"] += 1
            source_file.write(json.dumps(raw, ensure_ascii=False) + "\n")
            plan_file.write(json.dumps({"source_group_id": key,
                "source_fingerprint": new.source_fingerprint, "sections": sections}, ensure_ascii=False) + "\n")
    report = {"status": "planned", "scope": config.get("reuse_policy", "exact_input"),
              "counts": dict(counts), "config": dict(config),
              "pending": ["semantic_static_reuse", "decision", "vqa", "annotate", "export"]}
    (output / "migration_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def export_canonical_migration(run_root: Path, *, modality_profile_id: str) -> dict[str, Any]:
    """Export formal reasoning and Stage I views through existing validators.

    Args:
        run_root: Migration directory containing canonical sources and results.
        modality_profile_id: Existing Stage I modality selection to retain.

    Returns:
        Export counts, projection drops, and run-relative artifact paths.

    Raises:
        ValueError: If an annotation/source join or existing schema is invalid.
        RuntimeError: If sources still lack annotations.
    """
    report_path = run_root / "migration_report.json"
    migration = json.loads(report_path.read_text())
    views = run_root / "views"
    views.mkdir(exist_ok=True)
    formal_path = views / "reasoning_formal.jsonl"
    report = export_reasoning_training_views_v4(
        run_root / "sources.jsonl", run_root / "annotations.jsonl", formal_path,
        release_track="formal_training",
        direction_registry_version=migration["config"]["direction_registry_version"],
    )
    # exporter 的运行时路径仅用于读取；保存的报告统一以迁移目录为基准。
    for key in ("source_jsonl", "annotation_jsonl", "output_jsonl"):
        report[key] = Path(report[key]).relative_to(run_root).as_posix()
    if report["counts"].get("missing_annotation", 0):
        report["formal_release_ready"] = False
        (views / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        raise RuntimeError(f"export requires remaining annotations: {report['counts']['missing_annotation']}")

    counts: Counter[str] = Counter()
    drops_path = views / "stage_i_projection_drops.jsonl"
    with drops_path.open("w", encoding="utf-8") as drops:
        def projected_rows():
            for raw in iter_jsonl(formal_path):
                projection = project_stage_i_reasoning_view_with_audit_v2(
                    raw, modality_profile_id=modality_profile_id,
                    instruction_training_role="primary" if str(raw.get("instruction") or "").strip() else "not_present",
                )
                payload = projection.model_dump(mode="json", exclude_none=True)
                if isinstance(projection, StageIFullCoTViewV2):
                    counts["accepted"] += 1
                    yield payload
                else:
                    counts["dropped"] += 1
                    counts[f"drop:{payload['reason_code']}"] += 1
                    append_jsonl_row(drops, payload)

        write_jsonl_atomic(views / "stage_i_full_cot.jsonl", projected_rows())
    report.update(
        modality_profile_id=modality_profile_id, stage_i_counts=dict(counts),
        stage_i_output="views/stage_i_full_cot.jsonl", stage_i_drops="views/stage_i_projection_drops.jsonl",
        stage_i_has_rows=counts["accepted"] > 0,
    )
    (views / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    migration["stage_i_export"] = {
        "report": "views/export_report.json", "accepted": counts["accepted"], "dropped": counts["dropped"],
    }
    report_path.write_text(json.dumps(migration, ensure_ascii=False, indent=2) + "\n")
    return report


async def annotate_canonical_migration(
    run_root: Path, *, input_root: Path, backend: ReasoningV4TeacherBackend,
    limit: int | None = None,
) -> dict[str, Any]:
    """Execute a saved grounding plan with the existing annotation runner.

    Args:
        run_root: Directory containing sources, plan, and migration report.
        input_root: Runtime root for the plan's relative old annotation paths.
        backend: Existing teacher backend, configured with the media root.
        limit: Maximum source rows to visit, including already completed rows.

    Returns:
        Updated report with this invocation's processed and resumed counts.

    Raises:
        ValueError: If source revisions or selected old annotations changed.
        RuntimeError: If annotation fails; the failed result is saved first.
        OSError: If an input or output artifact cannot be accessed.
    """
    report_path = run_root / "migration_report.json"
    report = json.loads(report_path.read_text())
    config = report["config"]
    cfg = ReasoningV4BatchConfig(
        section_image_policy_id=config["section_image_policy_id"],
        direction_registry_version=config["direction_registry_version"],
    )
    annotations_path = run_root / "annotations.jsonl"
    completed = {}
    if annotations_path.exists():
        for raw in iter_jsonl(annotations_path):
            previous = ReasoningAnnotationV4.model_validate(raw)
            if previous.all_sections_accepted and previous.candidate_target is not None:
                completed[previous.source_group_id] = previous.source_fingerprint

    progress_path = run_root / "section_progress.jsonl"
    progress = {}
    if progress_path.exists():
        for row in iter_jsonl(progress_path):
            key = row["source_group_id"]
            if key not in completed and row["record"]["status"] == "accepted":
                progress.setdefault(key, {})[row["section_name"]] = row
    profiles = get_reasoning_v4_teacher_profiles(
        cfg.section_image_policy_id, direction_registry_version=cfg.direction_registry_version,
    )

    counts: Counter[str] = Counter()
    # 索引只保留记录的字节位置；读取选中结果时才解析 annotation，避免驻留全部历史文本。
    with ExitStack() as stack:
        indexed_files = {}

        def read_old(path: str, byte_offset: int) -> ReasoningAnnotationV4:
            if path not in indexed_files:
                indexed_files[path] = stack.enter_context((input_root / path).open("rb"))
            handle = indexed_files[path]
            handle.seek(byte_offset)
            return ReasoningAnnotationV4.model_validate_json(handle.readline())

        output = stack.enter_context(annotations_path.open("a", encoding="utf-8"))
        checkpoints = stack.enter_context(progress_path.open("a", encoding="utf-8"))
        rows = zip(iter_jsonl(run_root / "sources.jsonl"),
                   iter_jsonl(run_root / "migration_plan.jsonl"), strict=True)
        current_key = None
        try:
            for index, (raw, plan) in enumerate(rows):
                if limit is not None and index >= limit:
                    break
                source = ReasoningSourceV4.model_validate(raw)
                current_key = source.observation.source_group_id
                if (plan["source_group_id"] != current_key
                        or plan["source_fingerprint"] != source.source_fingerprint):
                    raise ValueError(f"plan/source revision mismatch: {current_key}")
                if current_key in completed:
                    if completed[current_key] != source.source_fingerprint:
                        raise ValueError(f"resume source revision mismatch: {current_key}")
                    counts["already_completed"] += 1
                    continue
                reused = {}
                selected = {}
                for name, decision in plan["sections"].items():
                    if decision["disposition"] not in {"reuse_exact", "reuse_instruction", "reuse_anchor"}:
                        continue
                    origin = decision["origin"]
                    location = (origin["annotations"], origin["annotation_byte_offset"])
                    if location not in selected:
                        selected[location] = read_old(*location)
                    old = selected[location]
                    if (old.source_fingerprint != decision["old_source_fingerprint"]
                            or _digest(old.model_dump(mode="json")) != decision["old_annotation_digest"]):
                        raise ValueError(f"selected old annotation changed: {current_key}/{name}")
                    reused[name] = old
                resumed_decision = None
                for name, row in progress.get(current_key, {}).items():
                    record = AnnotationStageRecordV4.model_validate(row["record"])
                    profile = profiles[name]
                    if (row["source_fingerprint"] != source.source_fingerprint
                            or record.producer_profile_id != profile.profile_id
                            or record.prompt_policy_digest != profile.policy_digest
                            or record.derivation_version != profile.derivation_version):
                        raise ValueError(f"section checkpoint revision mismatch: {current_key}/{name}")
                    # journal 保存独立 section；用现有 annotation schema 表达尚未完成的样本。
                    restored = ReasoningAnnotationV4(
                        source_group_id=current_key, source_fingerprint=source.source_fingerprint,
                        record_status="error",
                        grounding_outputs=GroundingOutputsV4.model_validate(
                            row["provenance"]["groundings"] if name == "Decision"
                            else {SECTION_FIELDS[name]: row["section"]}),
                        decision_proposal=row["section"] if name == "Decision" else None,
                        candidate_structure_record=CandidateStructureRecordV4(
                            composer_version=REASONING_V4_COMPOSER_VERSION, status="not_available",
                            errors=["section_checkpoint_pending_composition"],
                        ),
                        stage_records={name: record},
                        metadata=row["provenance"],
                    )
                    if name == "Decision":
                        resumed_decision = restored
                    else:
                        reused[name] = restored
                        counts["resumed_groundings"] += 1

                def save_section(name, section, record, provenance):
                    # 每个结果独立 flush；后续 teacher 失败不会丢失已完成的 grounding。
                    append_jsonl_row(checkpoints, {
                        "source_group_id": current_key,
                        "source_fingerprint": source.source_fingerprint,
                        "section_name": name,
                        "section": section.model_dump(mode="json") if section is not None else None,
                        "record": record.model_dump(mode="json"),
                        "provenance": dict(provenance),
                    })

                annotation = await annotate_reasoning_source_v4(
                    source, backend=backend, config=cfg, reused_groundings=reused,
                    reused_decision=resumed_decision, on_section_result=save_section,
                )
                append_jsonl_row(output, annotation.model_dump(mode="json"))
                counts["annotated"] += 1
                counts["reused_groundings"] += len(reused)
                counts["resumed_decisions"] += int("reused_decision_source_fingerprint" in annotation.metadata)
                if not annotation.all_sections_accepted or annotation.candidate_target is None:
                    raise RuntimeError(f"annotation failed for {current_key}: {annotation.errors}")
        except Exception as exc:
            # 保存失败位置后原样传播异常；已写入的结果保留，续跑时跳过完整样本。
            report.update(status="annotation_failed", annotation_counts=dict(counts),
                          failure={"source_group_id": current_key, "error": str(exc)})
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            raise
    report.update(status="annotation_pass_finished", annotation_counts=dict(counts),
                  annotation_limit=limit)
    report.pop("failure", None)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
