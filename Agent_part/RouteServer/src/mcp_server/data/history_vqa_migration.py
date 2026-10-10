"""Compare question-scoped VQA inputs and rebind proven unchanged answers."""

from __future__ import annotations

import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

from mcp_server.contracts.reasoning_dataset_v4 import reasoning_payload_digest_v4
from mcp_server.contracts.stage_i_sft import StageIVQAViewV2
from mcp_server.contracts.vqa_dataset_v2 import (
    VQAAnnotationV2, VQACapabilityConfigV2, VQASourceV2, annotation_context_digest_v2,
)
from mcp_server.contracts.vqa_v2 import VQAQuestionV2
from mcp_server.data.history_annotation_migration import normalize_media_references
from mcp_server.data.history_label_reuse import anchor_label_evidence, history_observation, same_evidence_image
from mcp_server.data.io.jsonl import append_jsonl_row, iter_jsonl, iter_jsonl_with_offsets, write_jsonl_atomic
from mcp_server.data.stage_i.views import project_stage_i_vqa_view_with_audit_v2
from mcp_server.data.vqa.eligibility import load_vqa_capabilities_v2
from mcp_server.data.vqa.questions import build_vqa_questions_v2
from mcp_server.data.vqa.source import build_vqa_source_v2
from mcp_server.data.vqa.producers import (
    CanonicalDemonstratedDirectionProducerV2, produce_external_interface_annotation_v2,
)
from mcp_server.teacher.vqa_v2_batch import LVLMWeakLabelProducerV2, VQATeacherBackendV2
from mcp_server.teacher.vqa_v2_profiles import (
    LVLM_VQA_PRODUCER_ID, LVLM_VQA_PRODUCER_VERSION, VQA_V2_TEACHER_PROFILES,
)
from mcp_server.teacher.vqa_v2_teacher import build_vqa_teacher_question_v2


def _offline_plan_eligible(plan: Mapping[str, Any]) -> bool:
    """Keep proven reuse and canonical E0 separate from teacher-generated data."""
    return plan["disposition"] in {"reuse_exact", "reuse_anchor"} or (
        plan["disposition"] == "rebuild_label"
        and plan["question"]["task_id"] == "e0_demonstrated_direction_selection"
    )


def _anchor_vqa_decision(old, new, old_question, question, identities, anchor_records):
    """Compare the annotation target rather than the original teacher request."""
    # template id、历史角色编号和 prompt 元数据不定义监督答案；题意和方向条件仍定义答案。
    excluded = {"template_id", "evidence_refs"}
    before = old_question.model_dump(mode="json", exclude=excluded)
    after = question.model_dump(mode="json", exclude=excluded)
    decision = {"old_input_digest": reasoning_payload_digest_v4(before),
                "new_input_digest": reasoning_payload_digest_v4(after)}
    if before != after:
        return {**decision, "disposition": "pending", "reason": "label_question_or_direction_changed"}
    same, reason, evidence = anchor_label_evidence(
        old, new, media_identities=identities, anchor_records=anchor_records,
    )
    decision["anchor_evidence"] = evidence
    if not same:
        return {**decision, "disposition": "pending", "reason": reason}
    old_obs, new_obs = old.observation, new.observation
    if old_obs.travel_mode != new_obs.travel_mode:
        return {**decision, "disposition": "pending", "reason": "travel_mode_changed"}
    if any(ref.startswith("instruction.") for ref in question.evidence_refs):
        if old_obs.instruction != new_obs.instruction:
            return {**decision, "disposition": "pending", "reason": "instruction_changed"}
    if question.target_refs and old_obs.model_visible_context != new_obs.model_visible_context:
        return {**decision, "disposition": "pending", "reason": "target_context_changed"}
    for role in question.evidence_refs:
        if role in new_obs.image_roles and not role.startswith("fpv."):
            if not same_evidence_image(old_obs, new_obs, role, identities):
                return {**decision, "disposition": "pending", "reason": "required_context_image_unresolved"}
            if role.startswith("bev."):
                contracts = [next(item.bev_route_evidence for item in obs.evidence_catalog
                                  if str(item.ref) == role) for obs in (old_obs, new_obs)]
                if contracts[0] != contracts[1]:
                    return {**decision, "disposition": "pending", "reason": "bev_interpretation_changed"}
    if question.task_id == "d3_temporal_interaction_relation":
        old_history, new_history = history_observation(old_obs), history_observation(new_obs)
        if not old_history or not new_history:
            return {**decision, "disposition": "pending", "reason": "temporal_target_interval_unresolved"}
        before_times = old_history.get("target_ns", old_history["source_ns"])
        after_times = new_history.get("target_ns", new_history["source_ns"])
        if (before_times[0], before_times[-1]) != (after_times[0], after_times[-1]):
            return {**decision, "disposition": "pending", "reason": "temporal_target_interval_changed"}
    return {**decision, "disposition": "reuse_anchor", "reason": "anchor_label_semantics_equal"}


def migrate_vqa_answer(
    old_source: VQASourceV2, new_source: VQASourceV2,
    annotation: VQAAnnotationV2, question: VQAQuestionV2,
    *, media_identities: Mapping[str, str], reuse_policy: str = "exact_input",
    anchor_records: Mapping[str, Any] | None = None,
) -> tuple[VQAAnnotationV2 | None, dict[str, Any]]:
    """Reuse one answer when its actual question-scoped teacher inputs match.

    Args:
        old_source: Original source that produced the annotation.
        new_source: New history source at the same anchor.
        annotation: Accepted original typed answer.
        question: New question built by the existing question builder.
        media_identities: Proven media-reference to source-frame identities.
        reuse_policy: exact_input for historical cache comparison, or
            anchor_label_v1 for label migration across expanded history.
        anchor_records: Optional legacy timestamp/frame records keyed by source fingerprint.

    Returns:
        Rebound annotation for exact reuse, otherwise None; plus a disposition
        and explicit reason for the migration plan. Strong labels are routed
        back to their existing deterministic/interface producers.

    Raises:
        ValueError: If the source/annotation lineage or anchor is inconsistent.
    """
    if (annotation.source_group_id != old_source.observation.source_group_id
            or annotation.source_fingerprint != old_source.source_fingerprint
            or new_source.observation.source_group_id != annotation.source_group_id):
        raise ValueError("VQA migration requires matching original lineage and anchor")
    old_context = annotation_context_digest_v2(
        source=old_source, question=annotation.question,
        producer_id=annotation.producer_id, producer_version=annotation.producer_version,
    )
    if annotation.annotation_context_digest != old_context:
        raise ValueError("VQA annotation context does not match its original source")
    if annotation.label_source != "lvlm_weak":
        return None, {"disposition": "rebuild_label", "reason": "use_existing_label_producer"}
    if annotation.question.task_id != question.task_id:
        return None, {"disposition": "rerun_teacher", "reason": "question_task_changed"}
    profile = VQA_V2_TEACHER_PROFILES[question.task_id]
    if reuse_policy == "anchor_label_v1":
        decision = _anchor_vqa_decision(old_source, new_source, annotation.question, question,
                                        media_identities, anchor_records)
        if decision["disposition"] != "reuse_anchor":
            return None, decision
        return _rebind_vqa_answer(old_source, new_source, annotation, question,
                                  migration_id="history_anchor_label_reuse_v1"), decision
    if reuse_policy != "exact_input":
        raise ValueError(f"unsupported reuse policy: {reuse_policy}")
    if (annotation.producer_id != LVLM_VQA_PRODUCER_ID
            or annotation.producer_version != LVLM_VQA_PRODUCER_VERSION
            or annotation.prompt_profile_id != profile.profile_id
            or annotation.prompt_policy_digest != profile.policy_digest):
        return None, {"disposition": "rerun_teacher", "reason": "producer_contract_changed"}

    digests = []
    for source, item in ((old_source, annotation.question), (new_source, question)):
        # 使用 runner 相同的 evidence_refs 投影；当前帧问答不绑定无关历史图片的序号。
        images = [image for image, role in zip(source.observation.images, source.observation.image_roles)
                  if role in item.evidence_refs]
        if any(image not in media_identities for image in images):
            return None, {"disposition": "rerun_teacher", "reason": "media_identity_unresolved"}
        prompt = build_vqa_teacher_question_v2(source, item, profile)
        payload = json.loads(prompt.split("\n\n", 1)[1])
        digests.append(reasoning_payload_digest_v4({
            "prompt": normalize_media_references(payload, media_identities),
            "images": [media_identities[image] for image in images],
        }))
    decision = {"old_input_digest": digests[0], "new_input_digest": digests[1]}
    if digests[0] != digests[1]:
        return None, {**decision, "disposition": "rerun_teacher", "reason": "question_input_changed"}

    return _rebind_vqa_answer(old_source, new_source, annotation, question,
                              migration_id="history_question_exact_reuse_v1"), {
        **decision, "disposition": "reuse_exact", "reason": "question_input_equal",
    }


def _rebind_vqa_answer(old_source, new_source, annotation, question, *, migration_id):
    """Preserve the original answer/producer while assigning destination lineage."""
    context = annotation_context_digest_v2(
        source=new_source, question=question,
        producer_id=annotation.producer_id, producer_version=annotation.producer_version,
    )
    payload = annotation.model_dump(mode="json")
    payload.update(
        annotation_id=f"{annotation.source_group_id}:vqa_v2:{question.task_id}:history:{context[:20]}",
        source_fingerprint=new_source.source_fingerprint,
        input_fingerprint=new_source.source_fingerprint,
        annotation_context_digest=context, question=question.model_dump(mode="json"),
        evidence_refs=list(question.evidence_refs),
        metadata={**annotation.metadata, "migration_id": migration_id,
                  "migrated_from_annotation_id": annotation.annotation_id,
                  "migrated_from_source_fingerprint": old_source.source_fingerprint},
    )
    return VQAAnnotationV2.model_validate(payload)


def plan_vqa_migration(config: Mapping[str, Any], output: Path, *, input_root: Path) -> dict[str, Any]:
    """Plan question-level migration without calling a teacher.

    Args:
        config: New reasoning sources, media identities, optional capability
            file, ordered old VQA runs, and optional task_ids selection.
            Later runs/records take precedence.
        output: New VQA migration directory, created exclusively.
        input_root: Runtime root for all configuration-relative inputs.

    Returns:
        Counts, frozen capability policy, and relative input configuration.

    Raises:
        ValueError: If identities, paths, or source lineage are inconsistent.
        FileExistsError: If the output directory already exists.
    """
    def input_path(value: str) -> Path:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"configuration path must be input-root-relative: {value}")
        return input_root / relative

    capabilities = load_vqa_capabilities_v2(
        input_path(config["capability_json"]) if "capability_json" in config else None,
    )
    identities = {}
    anchor_records = ({row["source_fingerprint"]: row for row in
                       iter_jsonl(input_path(config["anchor_records_jsonl"]))}
                      if "anchor_records_jsonl" in config else {})
    for row in iter_jsonl(input_path(config["media_identity_jsonl"])):
        path, identity = row["path"], row["identity"]
        if path in identities and identities[path] != identity:
            raise ValueError(f"conflicting media identity: {path}")
        identities[path] = identity
    candidates = {}
    for run in config["old_runs"]:
        sources = {}
        for offset, raw in iter_jsonl_with_offsets(input_path(run["sources"])):
            source = VQASourceV2.model_validate(raw)
            key = source.observation.source_group_id
            if key in sources:
                raise ValueError(f"duplicate old VQA source: {key}")
            sources[key] = (offset, source.source_fingerprint)
        for offset, raw in iter_jsonl_with_offsets(input_path(run["annotations"])):
            annotation = VQAAnnotationV2.model_validate(raw)
            source_offset, fingerprint = sources[annotation.source_group_id]
            if annotation.source_fingerprint != fingerprint:
                raise ValueError(f"old VQA source revision mismatch: {annotation.source_group_id}")
            question = annotation.question
            key = (annotation.source_group_id, question.task_id, question.conditioned_direction_label)
            # VQAAnnotationV2 只承载 accepted 答案；按显式 run 顺序选最新记录。
            candidates[key] = {
                "sources": run["sources"], "source_byte_offset": source_offset,
                "annotations": run["annotations"], "annotation_byte_offset": offset,
                "annotation_id": annotation.annotation_id,
            }

    output.mkdir(parents=True, exist_ok=False)
    counts: Counter[str] = Counter()
    seen = set()
    with ExitStack() as stack:
        source_file = stack.enter_context((output / "sources.jsonl").open("x", encoding="utf-8"))
        plan_file = stack.enter_context((output / "migration_plan.jsonl").open("x", encoding="utf-8"))
        handles = {}

        def read_at(path, offset):
            if path not in handles:
                handles[path] = stack.enter_context(input_path(path).open("rb"))
            handle = handles[path]
            handle.seek(offset)
            return json.loads(handle.readline())

        for raw in iter_jsonl(input_path(config["new_sources"])):
            source = build_vqa_source_v2(raw)
            group = source.observation.source_group_id
            if group in seen:
                raise ValueError(f"duplicate new VQA source: {group}")
            seen.add(group)
            append_jsonl_row(source_file, source.model_dump(mode="json"))
            counts["sources"] += 1
            build = build_vqa_questions_v2(source, capabilities)
            for task, unavailable in build.unavailable.items():
                counts[f"unavailable:{task}:{unavailable.reason}"] += 1
            for question in build.questions:
                if "task_ids" in config and question.task_id not in config["task_ids"]:
                    continue
                origin = candidates.get((group, question.task_id, question.conditioned_direction_label))
                decision = {"disposition": ("pending" if config.get("reuse_policy") == "anchor_label_v1"
                                             else "rerun_teacher"), "reason": "old_answer_missing"}
                if question.task_id not in VQA_V2_TEACHER_PROFILES:
                    decision = {"disposition": "rebuild_label", "reason": "use_existing_label_producer"}
                elif origin is not None:
                    old = VQASourceV2.model_validate(read_at(origin["sources"], origin["source_byte_offset"]))
                    annotation = VQAAnnotationV2.model_validate(read_at(origin["annotations"], origin["annotation_byte_offset"]))
                    _, decision = migrate_vqa_answer(
                        old, source, annotation, question, media_identities=identities,
                        reuse_policy=config.get("reuse_policy", "exact_input"), anchor_records=anchor_records,
                    )
                append_jsonl_row(plan_file, {
                    "source_group_id": group, "source_fingerprint": source.source_fingerprint,
                    "question": question.model_dump(mode="json"), "origin": origin, **decision,
                })
                counts[f"{question.task_id}:{decision['disposition']}"] += 1
                counts["questions"] += 1
    report = {
        "status": "planned", "scope": config.get("reuse_policy", "exact_input"),
        "config": dict(config), "capabilities": capabilities.model_dump(mode="json"),
        "counts": dict(counts), "pending": ["annotate", "export", "real_data_validation"],
    }
    (output / "migration_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


async def annotate_vqa_migration(
    run_root: Path, *, input_root: Path, backend: VQATeacherBackendV2 | None,
    limit: int | None = None, offline_only: bool = False,
) -> dict[str, Any]:
    """Execute a frozen VQA plan and flush each accepted question result.

    Args:
        run_root: VQA plan directory.
        input_root: Runtime root for old artifacts and media identity inputs.
        backend: Existing VQA teacher backend, configured with the media root.
        limit: Maximum questions to visit, including completed questions.
        offline_only: Produce exact reuse and deterministic E0 only; keep other
            questions pending without creating a teacher backend.

    Returns:
        Report with this pass's reuse, generation, and resume counts.

    Raises:
        ValueError: If the plan or old annotation revision changed.
        RuntimeError: If a producer fails; previous accepted answers remain.
    """
    report_path = run_root / "migration_report.json"
    report = json.loads(report_path.read_text())
    capabilities = VQACapabilityConfigV2.model_validate(report["capabilities"])
    identities = {row["path"]: row["identity"] for row in
                  iter_jsonl(input_root / report["config"]["media_identity_jsonl"])}
    anchor_records = ({row["source_fingerprint"]: row for row in iter_jsonl(
        input_root / report["config"]["anchor_records_jsonl"])}
        if "anchor_records_jsonl" in report["config"] else {})
    source_offsets = {raw["observation"]["source_group_id"]: offset for offset, raw in
                      iter_jsonl_with_offsets(run_root / "sources.jsonl")}
    annotations_path = run_root / "annotations.jsonl"
    completed = set()
    if annotations_path.exists():
        for raw in iter_jsonl(annotations_path):
            annotation = VQAAnnotationV2.model_validate(raw)
            completed.add((annotation.source_fingerprint,
                           reasoning_payload_digest_v4(annotation.question.model_dump(mode="json"))))
    weak = LVLMWeakLabelProducerV2(backend) if backend is not None else None
    direction = CanonicalDemonstratedDirectionProducerV2()
    counts: Counter[str] = Counter()
    current = None
    with ExitStack() as stack:
        handles = {}

        def read_at(path, offset):
            if path not in handles:
                handles[path] = stack.enter_context(path.open("rb"))
            handle = handles[path]
            handle.seek(offset)
            return json.loads(handle.readline())

        output = stack.enter_context(annotations_path.open("a", encoding="utf-8"))
        try:
            for index, plan in enumerate(iter_jsonl(run_root / "migration_plan.jsonl")):
                if limit is not None and index >= limit:
                    break
                current = {"source_group_id": plan["source_group_id"], "question": plan["question"]}
                question = VQAQuestionV2.model_validate(plan["question"])
                # 离线迁移只接受已证明的复用与既有方向真值，待补标题保留在完整计划中。
                if offline_only and not _offline_plan_eligible(plan):
                    counts["pending"] += 1
                    continue
                key = (plan["source_fingerprint"], reasoning_payload_digest_v4(question.model_dump(mode="json")))
                if key in completed:
                    counts["already_completed"] += 1
                    continue
                source = VQASourceV2.model_validate(read_at(
                    run_root / "sources.jsonl", source_offsets[plan["source_group_id"]],
                ))
                if source.source_fingerprint != plan["source_fingerprint"]:
                    raise ValueError(f"VQA plan/source revision mismatch: {plan['source_group_id']}")
                disposition = plan["disposition"]
                if disposition in {"reuse_exact", "reuse_anchor"}:
                    origin = plan["origin"]
                    old = VQASourceV2.model_validate(read_at(input_root / origin["sources"], origin["source_byte_offset"]))
                    previous = VQAAnnotationV2.model_validate(read_at(input_root / origin["annotations"], origin["annotation_byte_offset"]))
                    annotation, decision = migrate_vqa_answer(
                        old, source, previous, question, media_identities=identities,
                        reuse_policy=report["config"].get("reuse_policy", "exact_input"),
                        anchor_records=anchor_records,
                    )
                    if (annotation is None or previous.annotation_id != origin["annotation_id"]
                            or decision["new_input_digest"] != plan["new_input_digest"]
                            or decision["old_input_digest"] != plan["old_input_digest"]
                            or decision.get("anchor_evidence") != plan.get("anchor_evidence")):
                        raise ValueError(f"planned VQA reuse changed: {plan['source_group_id']}/{question.task_id}")
                else:
                    if disposition == "rerun_teacher":
                        if weak is None:
                            raise ValueError("teacher execution requires an explicit backend")
                        result = await weak.produce(source, question, capabilities)
                    elif disposition == "rebuild_label":
                        result = (direction.produce(source, question, capabilities)
                                  if question.task_id == "e0_demonstrated_direction_selection"
                                  else produce_external_interface_annotation_v2(source, question, capabilities))
                    else:
                        raise ValueError(f"unsupported VQA disposition: {disposition}")
                    annotation = result.annotation
                    if annotation is None:
                        raise RuntimeError(f"VQA producer failed: {current}: {result.errors or [result.reason]}")
                # 单题落盘后才更新完成集合；下个问题失败时保留已经成功的答案。
                append_jsonl_row(output, annotation.model_dump(mode="json"))
                completed.add(key)
                counts[disposition] += 1
        except Exception as exc:
            report.update(status="annotation_failed", annotation_counts=dict(counts),
                          failure={"question": current, "error": str(exc)})
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            raise
    report.update(status="annotation_pass_finished", annotation_counts=dict(counts),
                  annotation_limit=limit, offline_only=offline_only)
    report.pop("failure", None)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def export_vqa_migration(
    run_root: Path, *, modality_profile_id: str, accepted_only: bool = False,
) -> dict[str, Any]:
    """Export completed planned questions through the existing Stage I projector.

    Args:
        run_root: Directory containing the frozen VQA plan and accepted answers.
        modality_profile_id: Registered student input profile to retain.
        accepted_only: Export the accepted subset and retain missing questions
            in a separate pending manifest. Full-plan completion stays explicit.

    Returns:
        Completion counts, projection drops, and package-relative output paths.

    Raises:
        RuntimeError: If planned questions still need accepted answers.
        ValueError: If an annotation or source fails its typed contract.
    """
    migration = json.loads((run_root / "migration_report.json").read_text())
    capabilities = VQACapabilityConfigV2.model_validate(migration["capabilities"])
    source_offsets = {raw["observation"]["source_group_id"]: offset for offset, raw in
                      iter_jsonl_with_offsets(run_root / "sources.jsonl")}
    answer_offsets = {}
    for offset, raw in iter_jsonl_with_offsets(run_root / "annotations.jsonl"):
        annotation = VQAAnnotationV2.model_validate(raw)
        key = (annotation.source_fingerprint,
               reasoning_payload_digest_v4(annotation.question.model_dump(mode="json")))
        answer_offsets[key] = offset
    counts: Counter[str] = Counter()
    for plan in iter_jsonl(run_root / "migration_plan.jsonl"):
        key = (plan["source_fingerprint"], reasoning_payload_digest_v4(plan["question"]))
        counts["planned"] += 1
        counts["missing"] += (key not in answer_offsets or (
            migration.get("offline_only", False) and not _offline_plan_eligible(plan)
        ))
    views = run_root / "views"
    views.mkdir(exist_ok=True)
    report_path = views / "export_report.json"
    report = {"export_complete": False, "modality_profile_id": modality_profile_id,
              "counts": dict(counts)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if counts["missing"] and not accepted_only:
        raise RuntimeError(f"VQA export requires remaining annotations: {counts['missing']}")

    # 只索引文件位置，逐题读取；多 GB 的旧答案无需整体驻留内存。
    with (run_root / "sources.jsonl").open("rb") as sources, \
            (run_root / "annotations.jsonl").open("rb") as answers, \
            (views / "stage_i_projection_drops.jsonl").open("w", encoding="utf-8") as drops, \
            (views / "pending_questions.jsonl").open("w", encoding="utf-8") as pending:
        def projected_rows():
            for plan in iter_jsonl(run_root / "migration_plan.jsonl"):
                key = (plan["source_fingerprint"], reasoning_payload_digest_v4(plan["question"]))
                if key not in answer_offsets or (
                    migration.get("offline_only", False) and not _offline_plan_eligible(plan)
                ):
                    append_jsonl_row(pending, plan)
                    continue
                sources.seek(source_offsets[plan["source_group_id"]])
                source = VQASourceV2.model_validate(json.loads(sources.readline()))
                answers.seek(answer_offsets[key])
                annotation = VQAAnnotationV2.model_validate(json.loads(answers.readline()))
                projection = project_stage_i_vqa_view_with_audit_v2(
                    source, annotation, capabilities=capabilities,
                    modality_profile_id=modality_profile_id,
                    instruction_training_role="primary" if (source.observation.instruction or "").strip() else "not_present",
                )
                payload = projection.model_dump(mode="json", exclude_none=True)
                if isinstance(projection, StageIVQAViewV2):
                    counts["accepted"] += 1
                    counts[f"task:{annotation.question.task_id}"] += 1
                    counts[f"provenance:{plan['disposition']}"] += 1
                    yield payload
                else:
                    counts["dropped"] += 1
                    counts[f"drop:{payload['reason_code']}"] += 1
                    append_jsonl_row(drops, payload)

        write_jsonl_atomic(views / "stage_i_vqa.jsonl", projected_rows())
    report.update(export_complete=True, counts=dict(counts),
                  full_plan_complete=counts["missing"] == 0,
                  export_scope="accepted_subset" if accepted_only else "complete_plan",
                  pending_output="views/pending_questions.jsonl",
                  stage_i_has_rows=counts["accepted"] > 0,
                  stage_i_output="views/stage_i_vqa.jsonl",
                  stage_i_drops="views/stage_i_projection_drops.jsonl")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
