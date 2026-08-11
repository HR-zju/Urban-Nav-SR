"""CPU tests for the canonical-to-physical DF-OPD export boundary."""

from __future__ import annotations

import json

import pytest

from mcp_server.agent.reasoning_v4_prompt_policy import get_prompt_profile_v4
from mcp_server.agent.reasoning_v4_text import render_reasoning_target_v4
from mcp_server.contracts.df_opd import (
    DFOPDCanonicalSampleV1,
    DFOPDProfileV1,
    DFOPDTeacherViewV1,
    DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION,
    DF_OPD_ORACLE_CUE_ROLE,
    DF_OPD_PROFILE_2S16,
    DF_OPD_TEACHER_CUE_ROLE,
    df_opd_digest_v1,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionContextV4,
    InputModalitiesV4,
    ReasoningTrainingViewV4,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
    DirectionAffordanceV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    ReasoningTargetV4,
    ReasoningTraceV4,
    SceneGroundingV4,
    TaskDirectionV4,
    TaskGroundingV4,
)
from mcp_server.data.df_opd.export import (
    build_df_opd_dtot_manifest_v1,
    build_df_opd_student_manifest_v1,
    build_df_opd_teacher_eval_manifest_v1,
    build_df_opd_trainable_manifest_v1,
)
from mcp_server.data.io.jsonl import file_sha256, iter_jsonl, write_jsonl_atomic
from mcp_server.data.stage_i_views import project_stage_i_reasoning_view_v2


_POLICY_DIGEST = "a" * 64


def _canonical_sample(cue_name: str, cue_digest: str) -> DFOPDCanonicalSampleV1:
    """Build one minimum valid TC fixture with an externally rendered cue."""

    profile = DFOPDProfileV1(
        profile_id=DF_OPD_PROFILE_2S16,
        target_horizon_steps=16,
    )
    payload = {
        "schema_version": DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION,
        "sample_id": "sample-1",
        "source_group_id": "source-1",
        "split_group_id": "split-1",
        "dataset_source": "sidewalk",
        "profile": profile,
        "images": ("current.jpg",),
        "image_roles": ("fpv.current",),
        "decision_target": (
            '<Decision>{"selected_action_label":"straight","speed":"normal",'
            '"response":"proceed"}</Decision>'
        ),
        "trajectory_ego_t": tuple((float(index), 0.0, 0.0) for index in range(16)),
        "action_mask": (True,) * 16,
        "teacher_views": (
            DFOPDTeacherViewV1(
                condition="same_input_t0",
                images=("current.jpg",),
                image_roles=("fpv.current",),
                privilege_policy_digest=_POLICY_DIGEST,
            ),
            DFOPDTeacherViewV1(
                condition="correct_privilege_tc",
                images=("current.jpg", cue_name),
                image_roles=("fpv.current", DF_OPD_TEACHER_CUE_ROLE),
                cue_class="go_straight",
                privilege_policy_digest=_POLICY_DIGEST,
                cue_artifact_sha256=cue_digest,
            ),
            DFOPDTeacherViewV1(
                condition="shuffled_privilege_ts",
                images=("current.jpg", "ts-cue.png"),
                image_roles=("fpv.current", DF_OPD_TEACHER_CUE_ROLE),
                cue_class="turn_left",
                privilege_policy_digest=_POLICY_DIGEST,
                cue_artifact_sha256=cue_digest,
            ),
            DFOPDTeacherViewV1(
                condition="exact_future_oracle_to",
                images=("current.jpg", "oracle.png"),
                image_roles=("fpv.current", DF_OPD_ORACLE_CUE_ROLE),
                cue_class="oracle",
                privilege_policy_digest=_POLICY_DIGEST,
                cue_artifact_sha256=cue_digest,
            ),
        ),
    }
    serialized = {
        key: value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        for key, value in payload.items()
    }
    serialized["teacher_views"] = [view.model_dump(mode="json") for view in payload["teacher_views"]]
    return DFOPDCanonicalSampleV1(
        **payload,
        canonical_digest=df_opd_digest_v1(serialized),
    )


def _stage_i_full_cot(sample: DFOPDCanonicalSampleV1):
    """Build one real V2 FullCoT view matching the compact DF-OPD fixture."""

    target = ReasoningTargetV4(reasoning=ReasoningTraceV4(
        task_grounding=TaskGroundingV4(objective="follow_route", requirements=["route_adherence"]),
        scene_grounding=SceneGroundingV4(
            scene_type="urban_sidewalk",
            walkable_support="paved_sidewalk",
            key_entities=[],
            traffic_context="none_visible",
        ),
        local_traversability=LocalTraversabilityV4(direction_affordances=[
            DirectionAffordanceV4(
                direction_label="straight",
                affordance="direct",
                clearance="sufficient",
            )
        ]),
        navigation_context_grounding=NavigationContextGroundingV4(
            task_direction=TaskDirectionV4(status="not_visible", source="none"),
        ),
        interaction_constraint_grounding=InteractionConstraintGroundingV4(),
        decision=DecisionV4(selected_action_label="straight", speed="normal", response="proceed"),
    ))
    profile = get_prompt_profile_v4("urbannav_reasoning_v4_6_compact_full_cot_sft_v1")
    source = ReasoningTrainingViewV4.model_validate({
        "sample_id": "stage-i:sample-1",
        "source_group_id": sample.source_group_id,
        "split_group_id": sample.split_group_id,
        "dataset_source": sample.dataset_source,
        "release_track": "formal_training",
        "audit_id": "b" * 64,
        "review_status": "not_required",
        "artifact_status": "training_ready",
        "artifact_training_eligible": True,
        "formal_training_eligible": True,
        "images": list(sample.images),
        "image_roles": list(sample.image_roles),
        "instruction": "Go straight.",
        "model_visible_context": {},
        "evidence_catalog": [
            {"ref": "instruction.active", "modality": "instruction"},
            {"ref": "fpv.current", "modality": "fpv"},
        ],
        "input_modalities": InputModalitiesV4(instruction=True).model_dump(mode="json"),
        "direction_context": DirectionContextV4(
            action_profile_id="fpv_degree10_terminal_centroid_v2",
            action_profile_version="2.0",
            affordance_probe_profile_id="fpv_degree10_affordance7_v1",
            affordance_probe_profile_version="1.0",
        ).model_dump(mode="json"),
        "assistant_text": render_reasoning_target_v4(target),
        "reasoning_target": target.model_dump(mode="json", exclude_none=True),
        "prompt_profile_id": profile.profile_id,
        "prompt_policy_digest": profile.policy_digest,
        "direction_supervision": {
            "profile_id": "fpv_degree10_terminal_centroid_v2",
            "profile_version": "2.0",
            "label_index": 9,
            "label_name": "straight",
            "choice": "J",
            "angle_deg": 0.0,
            "source": "pixel_trajectory_terminal_centroid_rule_v2",
            "bin_mode": "degree10_with_stop",
            "derivation_version": "pixel_trajectory_terminal_centroid_rule_v2",
        },
        "metadata": {
            "source_fingerprint": "a" * 64,
            "action_reconciliation_audit_id": "b" * 64,
        },
    })
    return project_stage_i_reasoning_view_v2(
        source,
        task_type="full_cot",
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )


def test_export_binds_private_cue_bytes_and_keeps_report_non_sensitive(tmp_path):
    cue_path = tmp_path / "cue.png"
    cue_path.write_bytes(b"private-cue-bytes")
    canonical_path = tmp_path / "canonical.jsonl"
    messages_path = tmp_path / "messages.jsonl"
    output_path = tmp_path / "trainable.jsonl"
    report_path = tmp_path / "trainable.report.json"
    sample = _canonical_sample(cue_path.name, file_sha256(cue_path))
    write_jsonl_atomic(
        canonical_path,
        [sample.model_dump(mode="json")],
    )
    write_jsonl_atomic(
        messages_path,
        [{"sample_id": sample.sample_id, "messages": [{"role": "user", "content": "go"}]}],
    )

    report = build_df_opd_trainable_manifest_v1(
        canonical_path,
        messages_path,
        output_path,
        report_path,
        teacher_condition="correct_privilege_tc",
    )

    rows = list(iter_jsonl(output_path))
    assert rows[0]["df_opd_teacher_cue_sha256"] == file_sha256(cue_path)
    assert rows[0]["teacher_images"][-1] == cue_path.name
    assert report["output_jsonl_sha256"] == file_sha256(output_path)
    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "decision_target" not in persisted_report
    assert "messages" not in persisted_report


def test_student_export_binds_existing_full_cot_replay_to_the_same_canonical_decision(tmp_path):
    """Keep Reasoning Dropout replay tied to a real Stage-I V2 FullCoT parent."""

    cue_path = tmp_path / "cue.png"
    cue_path.write_bytes(b"private-cue-bytes")
    sample = _canonical_sample(cue_path.name, file_sha256(cue_path))
    canonical_path = tmp_path / "canonical.jsonl"
    messages_path = tmp_path / "messages.jsonl"
    full_cot_path = tmp_path / "stage-i-full-cot.jsonl"
    output_path = tmp_path / "student.jsonl"
    report_path = tmp_path / "student.report.json"
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(
        messages_path,
        [{"sample_id": sample.sample_id, "messages": [{"role": "user", "content": "go"}]}],
    )
    full_cot = _stage_i_full_cot(sample)
    write_jsonl_atomic(full_cot_path, [full_cot.model_dump(mode="json", exclude_none=True)])

    report = build_df_opd_student_manifest_v1(
        canonical_path,
        messages_path,
        full_cot_path,
        output_path,
        report_path,
    )

    row = next(iter_jsonl(output_path))
    assert row["df_opd_replay_schema_version"] == "urbannav_df_opd_replay_v1"
    assert row["df_opd_full_cot_source_sample_id"] == full_cot.sample_id
    assert row["df_opd_full_cot_target"] == full_cot.assistant_text
    assert row["decision_target"] in row["df_opd_full_cot_target"]
    assert row["df_opd_teacher_condition"] == "correct_privilege_tc"
    assert report["counts"]["output_rows"] == 1
    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "df_opd_full_cot_target" not in persisted_report
    assert "messages" not in persisted_report


def test_student_export_rejects_full_cot_decision_drift_before_writing(tmp_path):
    """Do not let a Stage-I source with another action join a DF-OPD row."""

    cue_path = tmp_path / "cue.png"
    cue_path.write_bytes(b"private-cue-bytes")
    sample = _canonical_sample(cue_path.name, file_sha256(cue_path))
    canonical_path = tmp_path / "canonical.jsonl"
    messages_path = tmp_path / "messages.jsonl"
    full_cot_path = tmp_path / "stage-i-full-cot.jsonl"
    output_path = tmp_path / "student.jsonl"
    report_path = tmp_path / "student.report.json"
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(
        messages_path,
        [{"sample_id": sample.sample_id, "messages": [{"role": "user", "content": "go"}]}],
    )
    full_cot = _stage_i_full_cot(sample)
    payload = full_cot.model_dump(mode="json", exclude_none=True)
    payload["reasoning_target"]["reasoning"]["decision"]["selected_action_label"] = "stop"
    payload["reasoning_target"]["reasoning"]["decision"]["speed"] = "stop"
    payload["reasoning_target"]["reasoning"]["decision"]["response"] = "stop"
    payload["direction_supervision"].update({
        "label_index": 19,
        "label_name": "stop",
        "choice": "T",
        "angle_deg": None,
        "source": "finish_stop_rule",
    })
    # The immutable Stage-I projection remains valid only when its target and
    # digest agree, so derive a fresh, valid-but-different FullCoT source.
    drift_target = ReasoningTargetV4(reasoning=ReasoningTraceV4(
        task_grounding=TaskGroundingV4(objective="follow_route", requirements=["route_adherence"]),
        scene_grounding=SceneGroundingV4(
            scene_type="urban_sidewalk", walkable_support="paved_sidewalk", key_entities=[], traffic_context="none_visible"
        ),
        local_traversability=LocalTraversabilityV4(direction_affordances=[
            DirectionAffordanceV4(direction_label="straight", affordance="direct", clearance="sufficient")
        ]),
        navigation_context_grounding=NavigationContextGroundingV4(
            task_direction=TaskDirectionV4(status="not_visible", source="none")
        ),
        interaction_constraint_grounding=InteractionConstraintGroundingV4(),
        decision=DecisionV4(selected_action_label="stop", speed="stop", response="stop"),
    ))
    payload["assistant_text"] = render_reasoning_target_v4(drift_target)
    payload["reasoning_target"] = drift_target.model_dump(mode="json", exclude_none=True)
    payload["loss_spans"] = [{
        "start": 0,
        "end": len(payload["assistant_text"]),
        "scale": 1.0,
        "channel": "full_cot",
    }]
    payload["projection_digest"] = ""
    from mcp_server.contracts.stage_i_sft import stage_i_projection_digest_v2

    payload["projection_digest"] = stage_i_projection_digest_v2(
        {key: value for key, value in payload.items() if key != "projection_digest"}
    )
    write_jsonl_atomic(full_cot_path, [payload])

    with pytest.raises(ValueError, match="does not equal"):
        build_df_opd_student_manifest_v1(
            canonical_path,
            messages_path,
            full_cot_path,
            output_path,
            report_path,
        )
    assert not output_path.exists()
    assert not report_path.exists()


def test_export_fails_before_writing_when_private_cue_bytes_drift(tmp_path):
    cue_path = tmp_path / "cue.png"
    cue_path.write_bytes(b"changed-cue-bytes")
    canonical_path = tmp_path / "canonical.jsonl"
    messages_path = tmp_path / "messages.jsonl"
    output_path = tmp_path / "trainable.jsonl"
    report_path = tmp_path / "trainable.report.json"
    sample = _canonical_sample(cue_path.name, "b" * 64)
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(
        messages_path,
        [{"sample_id": sample.sample_id, "messages": [{"role": "user", "content": "go"}]}],
    )

    with pytest.raises(ValueError, match="cue digest mismatch"):
        build_df_opd_trainable_manifest_v1(
            canonical_path,
            messages_path,
            output_path,
            report_path,
            teacher_condition="correct_privilege_tc",
        )
    assert not output_path.exists()
    assert not report_path.exists()


def test_dtot_export_is_student_only_and_never_reads_private_cue_bytes(tmp_path):
    cue_path = tmp_path / "cue.png"
    cue_path.write_bytes(b"private-cue-bytes")
    canonical_path = tmp_path / "canonical.jsonl"
    messages_path = tmp_path / "messages.jsonl"
    output_path = tmp_path / "dtot.jsonl"
    report_path = tmp_path / "dtot.report.json"
    sample = _canonical_sample(cue_path.name, file_sha256(cue_path))
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    cue_path.unlink()
    write_jsonl_atomic(
        messages_path,
        [{"sample_id": sample.sample_id, "messages": [{"role": "user", "content": "go"}]}],
    )

    report = build_df_opd_dtot_manifest_v1(
        canonical_path,
        messages_path,
        output_path,
        report_path,
    )

    row = list(iter_jsonl(output_path))[0]
    assert row["df_opd_dtot_baseline"] is True
    assert not any(
        "teacher" in key.lower() or "oracle" in key.lower() or "privilege" in key.lower()
        for key in row
    )
    assert "cue_artifact_root" not in report
    assert report["student_only"] is True


def test_dtot_export_moves_direct_latent_queries_out_of_visible_prompt(tmp_path):
    canonical_path = tmp_path / "canonical.jsonl"
    messages_path = tmp_path / "messages.jsonl"
    output_path = tmp_path / "dtot.jsonl"
    report_path = tmp_path / "dtot.report.json"
    sample = _canonical_sample("unused.png", "0" * 64)
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(
        messages_path,
        [{
            "sample_id": sample.sample_id,
            "messages": [
                {"role": "system", "content": "Pass T has no language answer."},
                {"role": "user", "content": "<TASK_TRAJECTORY_CFM_DIRECT_LATENT> go [TRAJ_0] [TRAJ_15]"},
            ],
        }],
    )

    build_df_opd_dtot_manifest_v1(canonical_path, messages_path, output_path, report_path)
    row = list(iter_jsonl(output_path))[0]
    assert "<Decision>" in row["messages"][0]["content"]
    assert "[TRAJ_" not in row["messages"][0]["content"]
    assert row["messages"][1]["content"].startswith("<TASK_DECISION>")


def test_dtot_export_rejects_nested_private_field_before_writing(tmp_path):
    cue_path = tmp_path / "cue.png"
    cue_path.write_bytes(b"private-cue-bytes")
    canonical_path = tmp_path / "canonical.jsonl"
    messages_path = tmp_path / "messages.jsonl"
    output_path = tmp_path / "dtot.jsonl"
    report_path = tmp_path / "dtot.report.json"
    sample = _canonical_sample(cue_path.name, file_sha256(cue_path))
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(
        messages_path,
        [{
            "sample_id": sample.sample_id,
            "messages": [{"role": "user", "content": {"teacher_hint": "private"}}],
        }],
    )

    with pytest.raises(ValueError, match="private field"):
        build_df_opd_dtot_manifest_v1(
            canonical_path,
            messages_path,
            output_path,
            report_path,
        )
    assert not output_path.exists()
    assert not report_path.exists()


def test_teacher_evaluation_export_allows_ts_to_and_omits_replay_targets(tmp_path):
    """Keep four-condition audit input separate from trainable physical rows."""

    cue_path = tmp_path / "cue.png"
    ts_path = tmp_path / "ts-cue.png"
    oracle_path = tmp_path / "oracle.png"
    for path in (cue_path, ts_path, oracle_path):
        path.write_bytes(b"private-cue-bytes")
    canonical_path = tmp_path / "canonical.jsonl"
    messages_path = tmp_path / "messages.jsonl"
    output_path = tmp_path / "teacher-to.jsonl"
    report_path = tmp_path / "teacher-to.report.json"
    sample = _canonical_sample(cue_path.name, file_sha256(cue_path))
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(
        messages_path,
        [{"sample_id": sample.sample_id, "messages": [{"role": "user", "content": "go"}]}],
    )

    report = build_df_opd_teacher_eval_manifest_v1(
        canonical_path,
        messages_path,
        output_path,
        report_path,
        teacher_condition="exact_future_oracle_to",
    )

    row = list(iter_jsonl(output_path))[0]
    assert row["df_opd_teacher_condition"] == "exact_future_oracle_to"
    assert row["images"] == ["current.jpg", "oracle.png"]
    assert row["image_roles"][-1] == DF_OPD_ORACLE_CUE_ROLE
    assert row["private_artifact"] is True
    assert row["df_opd_evaluation_only"] is True
    assert all(key not in row for key in ("decision_target", "trajectory_ego_t", "action_mask"))
    assert report["private_artifact"] is True
    assert report["evaluation_only"] is True
