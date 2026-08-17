"""Export canonical DF-OPD samples into auditable ms-swift training rows.

This module owns the offline physical boundary only.  It never creates prompt
text, tokenizes media, or trains a model.  Before a private Teacher cue is
allowed into a training row, its on-disk bytes must match the digest frozen in
the canonical Teacher view.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Mapping

from mcp_server.contracts.df_opd import (
    DFOPDCanonicalSampleV1,
    DFOPDTeacherConditionV1,
    DF_OPD_REPLAY_SCHEMA_VERSION,
    df_opd_digest_v1,
)
from mcp_server.contracts.stage_i_sft import StageIFullCoTViewV2, parse_stage_i_training_view_v2
from mcp_server.data.io.jsonl import file_sha256, iter_jsonl, write_jsonl_atomic
from mcp_server.data.df_opd.dtot_prompt import (
    DF_OPD_DTOT_PROMPT_POLICY_DIGEST,
    DF_OPD_DTOT_PROMPT_POLICY_ID,
    build_df_opd_dtot_student_messages,
)


DF_OPD_TRAINABLE_MANIFEST_REPORT_SCHEMA_V1 = "urbannav_df_opd_trainable_manifest_report_v1"
DF_OPD_DTOT_MANIFEST_REPORT_SCHEMA_V1 = "urbannav_df_opd_dtot_manifest_report_v1"
DF_OPD_TEACHER_EVAL_MANIFEST_REPORT_SCHEMA_V1 = "urbannav_df_opd_teacher_eval_manifest_report_v1"
DF_OPD_STUDENT_MANIFEST_REPORT_SCHEMA_V1 = "urbannav_df_opd_student_manifest_report_v1"

def build_df_opd_trainable_manifest_v1(
    canonical_jsonl: str | Path,
    messages_jsonl: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    *,
    teacher_condition: Literal["same_input_t0", "correct_privilege_tc"],
    cue_artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build one strict trainable manifest from frozen canonical rows.

    ``messages_jsonl`` is deliberately separate from the canonical payload:
    each row contains only ``sample_id`` and a Student-visible ``messages``
    list.  The function requires an exact one-to-one ID match, preserves
    canonical input order, rejects evaluation-only TS/TO rows, and verifies
    the digest of each selected TC cue before writing the output.

    Args:
        canonical_jsonl: Frozen canonical ``DFOPDCanonicalSampleV1`` rows.
        messages_jsonl: Student prompt rows keyed by canonical ``sample_id``.
        output_jsonl: Destination physical training-view JSONL.
        report_json: Destination compact lineage/integrity report JSON.
        teacher_condition: Selected trainable T0 or TC view for every row.
        cue_artifact_root: Base directory for relative Teacher cue paths.
            Defaults to the canonical manifest's parent directory.

    Returns:
        Deterministic report containing only paths, digests and aggregate
        counts; it intentionally excludes prompts, exact labels and targets.

    Raises:
        ValueError: If IDs, canonical schema, cue bytes, or private-boundary
            invariants are invalid.
    """

    canonical_path = Path(canonical_jsonl).expanduser().resolve()
    messages_path = Path(messages_jsonl).expanduser().resolve()
    output_path = Path(output_jsonl).expanduser().resolve()
    report_path = Path(report_json).expanduser().resolve()
    cue_root = (
        Path(cue_artifact_root).expanduser().resolve()
        if cue_artifact_root is not None
        else canonical_path.parent
    )
    messages_by_sample_id = _read_messages_by_sample_id(messages_path)
    outputs: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    counts: Counter[str] = Counter()
    canonical_digests: list[str] = []

    for line_number, payload in enumerate(iter_jsonl(canonical_path), start=1):
        try:
            sample = DFOPDCanonicalSampleV1.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid DF-OPD canonical row at line {line_number}: {exc}") from exc
        if sample.sample_id in seen_sample_ids:
            raise ValueError(f"duplicate DF-OPD canonical sample_id: {sample.sample_id!r}")
        seen_sample_ids.add(sample.sample_id)
        try:
            messages = messages_by_sample_id.pop(sample.sample_id)
        except KeyError as exc:
            raise ValueError(f"missing Student messages for canonical sample {sample.sample_id!r}") from exc
        _verify_private_cue_artifact(sample, cue_root, teacher_condition)
        outputs.append(
            sample.to_trainable_physical_row(
                messages=messages,
                teacher_condition=teacher_condition,
            )
        )
        counts["output_rows"] += 1
        counts[f"profile:{sample.profile.profile_id}"] += 1
        counts[f"teacher_condition:{teacher_condition}"] += 1
        canonical_digests.append(sample.canonical_digest)

    if messages_by_sample_id:
        orphaned = sorted(messages_by_sample_id)
        preview = ", ".join(repr(value) for value in orphaned[:3])
        raise ValueError(f"messages JSONL contains unknown canonical sample_id values: {preview}")
    if not outputs:
        raise ValueError("DF-OPD trainable export requires at least one canonical row")

    write_jsonl_atomic(output_path, outputs)
    report = {
        "schema_version": DF_OPD_TRAINABLE_MANIFEST_REPORT_SCHEMA_V1,
        "canonical_jsonl": str(canonical_path),
        "canonical_jsonl_sha256": file_sha256(canonical_path),
        "messages_jsonl": str(messages_path),
        "messages_jsonl_sha256": file_sha256(messages_path),
        "cue_artifact_root": str(cue_root),
        "teacher_condition": teacher_condition,
        "output_jsonl": str(output_path),
        "output_jsonl_sha256": file_sha256(output_path),
        "canonical_digest_set_sha256": df_opd_digest_v1({"digests": sorted(canonical_digests)}),
        "counts": dict(sorted(counts.items())),
    }
    _write_json_atomic(report_path, report)
    return report


def build_df_opd_student_manifest_v1(
    canonical_jsonl: str | Path,
    messages_jsonl: str | Path,
    stage_i_full_cot_jsonl: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    *,
    cue_artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the Student DF-OPD manifest with auditable replay sources.

    This is the only trainable export that carries both the privileged TC
    transport required for on-policy OPD and the public FullCoT replay target
    required by Reasoning Dropout.  FullCoT text is never reconstructed in
    ms-swift: every row is joined to one existing Stage-I V2 ``full_cot``
    projection and its canonical Decision must equal the DF-OPD Decision.

    The join key intentionally includes source/split/dataset identity and the
    complete Student media sequence.  A caller with multiple Stage-I views for
    one source must pre-filter them rather than allowing an arbitrary match.

    Args:
        canonical_jsonl: Frozen canonical ``DFOPDCanonicalSampleV1`` rows.
        messages_jsonl: Student prompt rows keyed by canonical ``sample_id``.
        stage_i_full_cot_jsonl: Filtered Stage-I V2 FullCoT views used as the
            only FullCoT replay authority.
        output_jsonl: Destination private Student-training JSONL.
        report_json: Destination compact lineage/integrity receipt.
        cue_artifact_root: Base directory for relative TC cue paths. Defaults
            to the canonical manifest's parent directory.

    Returns:
        Deterministic receipt with source digests and aggregate counts only.

    Raises:
        ValueError: If a FullCoT source is missing, ambiguous, mismatched, or
            cannot be safely joined to the canonical DF-OPD sample.
    """

    canonical_path = Path(canonical_jsonl).expanduser().resolve()
    messages_path = Path(messages_jsonl).expanduser().resolve()
    full_cot_path = Path(stage_i_full_cot_jsonl).expanduser().resolve()
    output_path = Path(output_jsonl).expanduser().resolve()
    report_path = Path(report_json).expanduser().resolve()
    cue_root = (
        Path(cue_artifact_root).expanduser().resolve()
        if cue_artifact_root is not None
        else canonical_path.parent
    )
    messages_by_sample_id = _read_messages_by_sample_id(messages_path)
    full_cot_by_join_key = _read_stage_i_full_cot_by_join_key(full_cot_path)
    outputs: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    canonical_digests: list[str] = []
    full_cot_projection_digests: list[str] = []
    counts: Counter[str] = Counter()

    for line_number, payload in enumerate(iter_jsonl(canonical_path), start=1):
        try:
            sample = DFOPDCanonicalSampleV1.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid DF-OPD canonical row at line {line_number}: {exc}") from exc
        if sample.sample_id in seen_sample_ids:
            raise ValueError(f"duplicate DF-OPD canonical sample_id: {sample.sample_id!r}")
        seen_sample_ids.add(sample.sample_id)
        try:
            messages = messages_by_sample_id.pop(sample.sample_id)
        except KeyError as exc:
            raise ValueError(f"missing Student messages for canonical sample {sample.sample_id!r}") from exc
        join_key = _stage_i_join_key(
            source_group_id=sample.source_group_id,
            split_group_id=sample.split_group_id,
            dataset_source=sample.dataset_source,
            images=sample.images,
            image_roles=sample.image_roles,
        )
        try:
            full_cot = full_cot_by_join_key.pop(join_key)
        except KeyError as exc:
            raise ValueError(
                "DF-OPD canonical sample has no unique matching Stage-I V2 FullCoT replay source: "
                f"{sample.sample_id!r}"
            ) from exc
        _verify_private_cue_artifact(sample, cue_root, "correct_privilege_tc")
        row = sample.to_trainable_physical_row(
            messages=messages,
            teacher_condition="correct_privilege_tc",
        )
        _attach_full_cot_replay(row, sample=sample, full_cot=full_cot)
        _assert_df_opd_student_training_row(row)
        outputs.append(row)
        counts["output_rows"] += 1
        counts[f"profile:{sample.profile.profile_id}"] += 1
        counts["teacher_condition:correct_privilege_tc"] += 1
        canonical_digests.append(sample.canonical_digest)
        full_cot_projection_digests.append(full_cot.projection_digest)

    if messages_by_sample_id:
        orphaned = sorted(messages_by_sample_id)
        preview = ", ".join(repr(value) for value in orphaned[:3])
        raise ValueError(f"messages JSONL contains unknown canonical sample_id values: {preview}")
    if full_cot_by_join_key:
        raise ValueError("Stage-I FullCoT JSONL contains rows with no matching DF-OPD canonical sample")
    if not outputs:
        raise ValueError("DF-OPD Student manifest requires at least one canonical row")

    write_jsonl_atomic(output_path, outputs)
    report = {
        "schema_version": DF_OPD_STUDENT_MANIFEST_REPORT_SCHEMA_V1,
        "replay_schema_version": DF_OPD_REPLAY_SCHEMA_VERSION,
        "canonical_jsonl": str(canonical_path),
        "canonical_jsonl_sha256": file_sha256(canonical_path),
        "messages_jsonl": str(messages_path),
        "messages_jsonl_sha256": file_sha256(messages_path),
        "stage_i_full_cot_jsonl": str(full_cot_path),
        "stage_i_full_cot_jsonl_sha256": file_sha256(full_cot_path),
        "cue_artifact_root": str(cue_root),
        "teacher_condition": "correct_privilege_tc",
        "output_jsonl": str(output_path),
        "output_jsonl_sha256": file_sha256(output_path),
        "canonical_digest_set_sha256": df_opd_digest_v1({"digests": sorted(canonical_digests)}),
        "full_cot_projection_digest_set_sha256": df_opd_digest_v1(
            {"digests": sorted(full_cot_projection_digests)}
        ),
        "counts": dict(sorted(counts.items())),
    }
    _write_json_atomic(report_path, report)
    return report


def build_df_opd_dtot_manifest_v1(
    canonical_jsonl: str | Path,
    messages_jsonl: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
) -> dict[str, Any]:
    """Build the explicitly Student-only manifest for the no-OPD D-to-T baseline.

    Unlike :func:`build_df_opd_trainable_manifest_v1`, this export never
    selects or persists a Teacher view.  It uses the canonical T0 validation
    path only to retain the existing Student-message privacy checks, then
    removes every Teacher/privilege field before the atomic JSONL write.  The
    resulting rows are the only supported input to the D-to-T SFT command.

    Args:
        canonical_jsonl: Frozen canonical ``DFOPDCanonicalSampleV1`` rows.
        messages_jsonl: Student-visible prompt rows keyed by ``sample_id``.
        output_jsonl: Destination Student-only D-to-T training-view JSONL.
        report_json: Destination compact lineage/integrity report JSON.

    Returns:
        Deterministic report with paths, digests and aggregate counts only.

    Raises:
        ValueError: If IDs, canonical rows, or Student-message privacy
            invariants are invalid.
    """

    canonical_path = Path(canonical_jsonl).expanduser().resolve()
    messages_path = Path(messages_jsonl).expanduser().resolve()
    output_path = Path(output_jsonl).expanduser().resolve()
    report_path = Path(report_json).expanduser().resolve()
    messages_by_sample_id = _read_messages_by_sample_id(messages_path)
    outputs: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    counts: Counter[str] = Counter()
    canonical_digests: list[str] = []

    for line_number, payload in enumerate(iter_jsonl(canonical_path), start=1):
        try:
            sample = DFOPDCanonicalSampleV1.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid DF-OPD canonical row at line {line_number}: {exc}") from exc
        if sample.sample_id in seen_sample_ids:
            raise ValueError(f"duplicate DF-OPD canonical sample_id: {sample.sample_id!r}")
        seen_sample_ids.add(sample.sample_id)
        try:
            raw_messages = messages_by_sample_id.pop(sample.sample_id)
        except KeyError as exc:
            raise ValueError(f"missing Student messages for canonical sample {sample.sample_id!r}") from exc
        _reject_dtot_private_key({"messages": raw_messages})
        messages = build_df_opd_dtot_student_messages(raw_messages)

        # Reuse the canonical Student-message validator through T0, then
        # construct a fresh row whose serialized schema has no Teacher alias.
        t0_row = sample.to_trainable_physical_row(
            messages=messages,
            teacher_condition="same_input_t0",
        )
        student_row = {
            key: value for key, value in t0_row.items()
            if key not in _DF_OPD_PRIVATE_PHYSICAL_KEYS
        }
        student_row["df_opd_dtot_baseline"] = True
        _assert_dtot_student_only_row(student_row)
        outputs.append(student_row)
        counts["output_rows"] += 1
        counts[f"profile:{sample.profile.profile_id}"] += 1
        canonical_digests.append(sample.canonical_digest)

    if messages_by_sample_id:
        orphaned = sorted(messages_by_sample_id)
        preview = ", ".join(repr(value) for value in orphaned[:3])
        raise ValueError(f"messages JSONL contains unknown canonical sample_id values: {preview}")
    if not outputs:
        raise ValueError("DF-OPD D-to-T export requires at least one canonical row")

    write_jsonl_atomic(output_path, outputs)
    report = {
        "schema_version": DF_OPD_DTOT_MANIFEST_REPORT_SCHEMA_V1,
        "canonical_jsonl": str(canonical_path),
        "canonical_jsonl_sha256": file_sha256(canonical_path),
        "messages_jsonl": str(messages_path),
        "messages_jsonl_sha256": file_sha256(messages_path),
        "student_only": True,
        "prompt_policy_id": DF_OPD_DTOT_PROMPT_POLICY_ID,
        "prompt_policy_digest": DF_OPD_DTOT_PROMPT_POLICY_DIGEST,
        "output_jsonl": str(output_path),
        "output_jsonl_sha256": file_sha256(output_path),
        "canonical_digest_set_sha256": df_opd_digest_v1({"digests": sorted(canonical_digests)}),
        "counts": dict(sorted(counts.items())),
    }
    _write_json_atomic(report_path, report)
    return report


def build_df_opd_teacher_eval_manifest_v1(
    canonical_jsonl: str | Path,
    messages_jsonl: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    *,
    teacher_condition: DFOPDTeacherConditionV1,
    cue_artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a private, GT-free Teacher evaluation manifest for one condition.

    This is not a training view: it permits T0/TC/TS/TO, emits only the
    chosen Teacher media as normal template ``images``, and rejects writing
    any Decision/trajectory target.  The manifest is intentionally private;
    it is only valid for ``infer_df_opd_teacher`` and score/review plumbing.

    Args:
        canonical_jsonl: Frozen canonical ``DFOPDCanonicalSampleV1`` rows.
        messages_jsonl: Student-visible prompt rows keyed by ``sample_id``.
        output_jsonl: Destination private Teacher evaluation JSONL.
        report_json: Destination compact private-artifact receipt.
        teacher_condition: One frozen T0/TC/TS/TO view for every output row.
        cue_artifact_root: Base directory for relative private cue paths.

    Returns:
        Deterministic private receipt with aggregate counts and file digests.

    Raises:
        ValueError: If IDs, canonical rows, cue bytes, or private text
            boundaries are invalid.
    """

    canonical_path = Path(canonical_jsonl).expanduser().resolve()
    messages_path = Path(messages_jsonl).expanduser().resolve()
    output_path = Path(output_jsonl).expanduser().resolve()
    report_path = Path(report_json).expanduser().resolve()
    cue_root = (
        Path(cue_artifact_root).expanduser().resolve()
        if cue_artifact_root is not None
        else canonical_path.parent
    )
    messages_by_sample_id = _read_messages_by_sample_id(messages_path)
    outputs: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    canonical_digests: list[str] = []
    counts: Counter[str] = Counter()

    for line_number, payload in enumerate(iter_jsonl(canonical_path), start=1):
        try:
            sample = DFOPDCanonicalSampleV1.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid DF-OPD canonical row at line {line_number}: {exc}") from exc
        if sample.sample_id in seen_sample_ids:
            raise ValueError(f"duplicate DF-OPD canonical sample_id: {sample.sample_id!r}")
        seen_sample_ids.add(sample.sample_id)
        try:
            messages = messages_by_sample_id.pop(sample.sample_id)
        except KeyError as exc:
            raise ValueError(f"missing Student messages for canonical sample {sample.sample_id!r}") from exc
        # Teacher evaluation follows the same Decision -> trajectory path as
        # D-to-T.  Frozen Stage-II source rows may carry their training query
        # suffix; compile a Decision-visible prompt before inference, which
        # appends isolated ``[TRAJ_i]`` queries after the Decision cache.
        messages = build_df_opd_dtot_student_messages(messages)
        _verify_private_cue_artifact(sample, cue_root, teacher_condition)
        row = sample.to_teacher_evaluation_physical_row(
            messages=messages,
            teacher_condition=teacher_condition,
        )
        _assert_teacher_evaluation_row(row)
        outputs.append(row)
        counts["output_rows"] += 1
        counts[f"profile:{sample.profile.profile_id}"] += 1
        canonical_digests.append(sample.canonical_digest)

    if messages_by_sample_id:
        orphaned = sorted(messages_by_sample_id)
        preview = ", ".join(repr(value) for value in orphaned[:3])
        raise ValueError(f"messages JSONL contains unknown canonical sample_id values: {preview}")
    if not outputs:
        raise ValueError("DF-OPD Teacher evaluation export requires at least one canonical row")

    write_jsonl_atomic(output_path, outputs)
    report = {
        "schema_version": DF_OPD_TEACHER_EVAL_MANIFEST_REPORT_SCHEMA_V1,
        "private_artifact": True,
        "evaluation_only": True,
        "teacher_condition": teacher_condition,
        "canonical_jsonl": str(canonical_path),
        "canonical_jsonl_sha256": file_sha256(canonical_path),
        "messages_jsonl": str(messages_path),
        "messages_jsonl_sha256": file_sha256(messages_path),
        "cue_artifact_root": str(cue_root),
        "output_jsonl": str(output_path),
        "output_jsonl_sha256": file_sha256(output_path),
        "canonical_digest_set_sha256": df_opd_digest_v1({"digests": sorted(canonical_digests)}),
        "counts": dict(sorted(counts.items())),
    }
    _write_json_atomic(report_path, report)
    return report


_DF_OPD_PRIVATE_PHYSICAL_KEYS = {
    "teacher_messages",
    "teacher_images",
    "teacher_image_roles",
    "teacher_prompt",
    "df_opd_teacher_condition",
    "df_opd_privilege_policy_digest",
    "df_opd_teacher_cue_sha256",
}


def _read_stage_i_full_cot_by_join_key(path: Path) -> dict[tuple[Any, ...], StageIFullCoTViewV2]:
    """Load one exact V2 FullCoT replay source for every DF-OPD join key."""

    result: dict[tuple[Any, ...], StageIFullCoTViewV2] = {}
    for line_number, payload in enumerate(iter_jsonl(path), start=1):
        try:
            view = parse_stage_i_training_view_v2(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid Stage-I V2 replay row at line {line_number}: {exc}") from exc
        if not isinstance(view, StageIFullCoTViewV2):
            raise ValueError(f"Stage-I replay row at line {line_number} must have task_type=full_cot")
        if any(span.scale != 1.0 for span in view.loss_spans):
            raise ValueError(f"Stage-I FullCoT replay row at line {line_number} has a non-unit loss span")
        join_key = _stage_i_join_key(
            source_group_id=view.source_group_id,
            split_group_id=view.split_group_id,
            dataset_source=view.dataset_source,
            images=view.images,
            image_roles=view.image_roles,
        )
        if join_key in result:
            raise ValueError(
                "Stage-I FullCoT replay join is ambiguous; pre-filter to exactly one V2 FullCoT view per "
                "source/split/dataset/media key"
            )
        result[join_key] = view
    if not result:
        raise ValueError("DF-OPD Student manifest requires at least one Stage-I V2 FullCoT replay row")
    return result


def _stage_i_join_key(
    *,
    source_group_id: str,
    split_group_id: str,
    dataset_source: str,
    images: Any,
    image_roles: Any,
) -> tuple[Any, ...]:
    """Return the strict public identity used to bind FullCoT replay input."""

    return (
        str(source_group_id),
        str(split_group_id),
        str(dataset_source),
        tuple(str(item) for item in images),
        tuple(str(item) for item in image_roles),
    )


def _attach_full_cot_replay(
    row: dict[str, Any],
    *,
    sample: DFOPDCanonicalSampleV1,
    full_cot: StageIFullCoTViewV2,
) -> None:
    """Attach an existing public FullCoT target after Decision identity checks."""

    decision = full_cot.reasoning_target.reasoning.decision.model_dump(mode="json")
    rendered = "<Decision>" + json.dumps(decision, ensure_ascii=False, separators=(",", ":")) + "</Decision>"
    if rendered != sample.decision_target:
        raise ValueError(
            "Stage-I FullCoT Decision does not equal the canonical DF-OPD decision_target for "
            f"{sample.sample_id!r}"
        )
    target = full_cot.assistant_text
    if "[TRAJ_" in target or "teacher" in target.lower() or "oracle" in target.lower():
        raise ValueError("Stage-I FullCoT replay target contains forbidden DF-OPD transport content")
    row.update({
        "df_opd_replay_schema_version": DF_OPD_REPLAY_SCHEMA_VERSION,
        "df_opd_full_cot_target": target,
        "df_opd_full_cot_source_sample_id": full_cot.sample_id,
        "df_opd_full_cot_projection_digest": full_cot.projection_digest,
        "df_opd_full_cot_assistant_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
    })


def _assert_df_opd_student_training_row(row: Mapping[str, Any]) -> None:
    """Require the combined on-policy and public-replay trainable view."""

    required = {
        "df_opd_replay_schema_version",
        "df_opd_full_cot_target",
        "df_opd_full_cot_source_sample_id",
        "df_opd_full_cot_projection_digest",
        "df_opd_full_cot_assistant_sha256",
    }
    missing = sorted(key for key in required if not row.get(key))
    if missing:
        raise ValueError(f"DF-OPD Student manifest is missing replay fields: {missing}")
    if row.get("df_opd_replay_schema_version") != DF_OPD_REPLAY_SCHEMA_VERSION:
        raise ValueError("DF-OPD Student manifest replay schema version mismatch")
    target = row.get("df_opd_full_cot_target")
    if not isinstance(target, str) or not target:
        raise ValueError("DF-OPD Student manifest FullCoT target must be non-empty text")
    recorded_digest = row.get("df_opd_full_cot_assistant_sha256")
    if hashlib.sha256(target.encode("utf-8")).hexdigest() != recorded_digest:
        raise ValueError("DF-OPD Student manifest FullCoT target digest mismatch")


def _assert_dtot_student_only_row(row: Mapping[str, Any]) -> None:
    """Fail closed if a D-to-T row exposes a private field at any nesting level."""

    _reject_dtot_private_key(row)


def _assert_teacher_evaluation_row(row: Mapping[str, Any]) -> None:
    """Reject accidental replay targets from a private Teacher eval row."""

    forbidden = {
        "decision_target",
        "trajectory_ego_t",
        "action_ego_t",
        "action_mask",
        "canonical_full_cot",
        "teacher_images",
        "teacher_image_roles",
        "teacher_messages",
    }
    found = _find_forbidden_key(row, forbidden)
    if found is not None:
        raise ValueError(f"Teacher evaluation manifest retained forbidden field: {found}")
    if row.get("df_opd_evaluation_only") is not True or row.get("private_artifact") is not True:
        raise ValueError("Teacher evaluation manifest must be explicitly private and evaluation-only")


def _reject_dtot_private_key(value: Any) -> None:
    """Recursively reject serialized Teacher/oracle/privilege field names."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            lower = str(key).lower()
            if "teacher" in lower or "oracle" in lower or "privilege" in lower:
                raise ValueError(f"D-to-T manifest retained private field: {key}")
            _reject_dtot_private_key(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_dtot_private_key(item)


def _find_forbidden_key(value: Any, forbidden: set[str]) -> str | None:
    """Return the first recursively retained field from a forbidden set."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in forbidden:
                return str(key)
            nested = _find_forbidden_key(item, forbidden)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _find_forbidden_key(item, forbidden)
            if nested is not None:
                return nested
    return None


def _read_messages_by_sample_id(path: Path) -> dict[str, list[Mapping[str, Any]]]:
    """Load a one-to-one Student-message sidecar without retaining extra keys."""

    result: dict[str, list[Mapping[str, Any]]] = {}
    for line_number, row in enumerate(iter_jsonl(path), start=1):
        sample_id = row.get("sample_id")
        messages = row.get("messages")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"messages JSONL line {line_number} requires non-empty sample_id")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"messages JSONL line {line_number} requires a non-empty messages list")
        if not all(isinstance(item, Mapping) for item in messages):
            raise ValueError(f"messages JSONL line {line_number} messages must contain objects")
        if sample_id in result:
            raise ValueError(f"duplicate messages JSONL sample_id: {sample_id!r}")
        result[sample_id] = [dict(item) for item in messages]
    return result


def _verify_private_cue_artifact(
    sample: DFOPDCanonicalSampleV1,
    cue_root: Path,
    teacher_condition: Literal["same_input_t0", "correct_privilege_tc"],
) -> None:
    """Bind a privileged cue path to its frozen canonical digest before export."""

    view = sample.teacher_view_for_condition(teacher_condition)
    if view.condition == "same_input_t0":
        return
    expected_digest = view.cue_artifact_sha256
    if expected_digest is None:
        raise ValueError("privileged Teacher view is missing cue_artifact_sha256")
    recorded_path = Path(view.images[-1]).expanduser()
    cue_path = recorded_path if recorded_path.is_absolute() else cue_root / recorded_path
    if not cue_path.is_file():
        raise ValueError(f"private Teacher cue file does not exist: {cue_path}")
    if file_sha256(cue_path) != expected_digest:
        raise ValueError(f"private Teacher cue digest mismatch: {cue_path}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist a compact non-sensitive export report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "DF_OPD_DTOT_MANIFEST_REPORT_SCHEMA_V1",
    "DF_OPD_REPLAY_SCHEMA_VERSION",
    "DF_OPD_STUDENT_MANIFEST_REPORT_SCHEMA_V1",
    "DF_OPD_TEACHER_EVAL_MANIFEST_REPORT_SCHEMA_V1",
    "DF_OPD_TRAINABLE_MANIFEST_REPORT_SCHEMA_V1",
    "build_df_opd_dtot_manifest_v1",
    "build_df_opd_student_manifest_v1",
    "build_df_opd_teacher_eval_manifest_v1",
    "build_df_opd_trainable_manifest_v1",
]
