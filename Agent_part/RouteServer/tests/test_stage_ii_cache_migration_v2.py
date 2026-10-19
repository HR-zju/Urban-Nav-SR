from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_server.agent.reasoning_v4_text import render_reasoning_section_v4
from mcp_server.contracts.stage_i_sft import (
    StageIFullCoTViewV1,
    parse_stage_i_training_view_v2,
    stage_i_projection_digest_v1,
)
from mcp_server.contracts.stage_ii_cfm import (
    StageIIContextCacheV1,
    StageIIContextProducerV1,
    stage_ii_digest_v1,
)
from mcp_server.data.stage_i.migration_v2 import upgrade_stage_i_full_cot_manifest_v2
from mcp_server.data.stage_i.views import project_stage_i_reasoning_view_v1
from mcp_server.data.stage_ii.cache_migration_v2 import rebind_stage_ii_context_cache_manifest_v2
from mcp_server.data.stage_ii.context import build_stage_ii_context_cache_v1, stage_ii_projected_input_digest_v1

from test_stage_i_sft import _formal_reasoning_view


def _producer(kind: str) -> StageIIContextProducerV1:
    """Build one deterministic cache producer fixture."""

    generation = {"temperature": 0.0, "max_tokens": 1024}
    parser_id = {
        "reasoning_grounding": "reasoning_v4_6_full_target_strict_v1",
        "decision": "reasoning_v4_6_decision_section_strict_v1",
    }[kind]
    return StageIIContextProducerV1(
        source="model_prediction",
        base_model_id="test/qwen",
        checkpoint_id="checkpoint-test",
        checkpoint_digest="1" * 64,
        prompt_profile_id=f"test_{kind}_v1",
        prompt_digest="2" * 64,
        parser_contract_id=parser_id,
        parser_contract_digest="3" * 64,
        generation_config=generation,
        generation_config_digest=stage_ii_digest_v1(generation),
    )


def _legacy_full_with_stale_image(
        tmp_path: Path) -> tuple[StageIFullCoTViewV1, Path]:
    """Create one valid V1 parent whose media requires a path-prefix map."""

    full = project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(),
        task_type="full_cot",
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )
    mapped_image = tmp_path / "mounted" / "current.jpg"
    mapped_image.parent.mkdir()
    mapped_image.write_bytes(b"same-transported-image")
    payload = full.model_dump(mode="json", exclude_none=True)
    payload["images"] = ["/retired/stage-i/current.jpg"]
    payload.pop("projection_digest")
    payload["projection_digest"] = stage_i_projection_digest_v1(payload)
    return StageIFullCoTViewV1.model_validate(payload), mapped_image


def _legacy_cache(
    full: StageIFullCoTViewV1,
    *,
    kind: str,
    raw_generation: str,
    projected_input_digest: str | None = None,
) -> StageIIContextCacheV1:
    """Seal one cache against the exact V1 observation."""

    digest = projected_input_digest or stage_ii_projected_input_digest_v1(
        images=list(full.images),
        image_roles=list(full.image_roles),
        instruction=full.instruction,
        model_visible_context=full.model_visible_context,
        evidence_catalog=[
            item.model_dump(mode="json", exclude_none=True)
            for item in full.evidence_catalog
        ],
        input_modalities=full.input_modalities.model_dump(mode="json"),
    )
    return build_stage_ii_context_cache_v1(
        sample_id=full.sample_id,
        source_group_id=full.source_group_id,
        split_group_id=full.split_group_id,
        dataset_source=full.dataset_source,
        modality_profile_id=full.modality_profile_id,
        parent_source_fingerprint=full.parent_source_fingerprint,
        projected_input_digest=digest,
        context_kind=kind,
        producer=_producer(kind),
        raw_generation=raw_generation,
    )


def _write_jsonl(path: Path, rows) -> None:
    """Write compact fixture rows."""

    path.write_text(
        "".join(
            json.dumps(row.model_dump(mode="json", exclude_none=True)) + "\n"
            for row in rows),
        encoding="utf-8",
    )


def test_cache_rebind_proves_transport_only_and_preserves_parser_output(
    tmp_path: Path, ) -> None:
    """A path-only migration may reseal but never reinterpret Pass R/D."""

    full, mapped_image = _legacy_full_with_stale_image(tmp_path)
    v1_path = tmp_path / "stage_i_v1.jsonl"
    _write_jsonl(v1_path, [full])
    v2_path = tmp_path / "stage_i_v2.jsonl"
    migration_report = tmp_path / "stage_i_migration.json"
    upgrade_stage_i_full_cot_manifest_v2(
        v1_path,
        v2_path,
        report_json=migration_report,
        image_prefix_maps=[("/retired/stage-i", mapped_image.parent)],
    )
    decision_text = render_reasoning_section_v4(
        "Decision",
        full.reasoning_target.reasoning.decision,
    )
    old_caches = [
        _legacy_cache(
            full,
            kind="reasoning_grounding",
            raw_generation=full.assistant_text,
        ),
        _legacy_cache(full, kind="decision", raw_generation=decision_text),
    ]
    old_path = tmp_path / "old_caches.jsonl"
    _write_jsonl(old_path, old_caches)
    output_path = tmp_path / "rebound_caches.jsonl"
    report = rebind_stage_ii_context_cache_manifest_v2(
        stage_i_v1_jsonl=v1_path,
        stage_i_v2_jsonl=v2_path,
        migration_report_json=migration_report,
        input_cache_jsonl=old_path,
        output_cache_jsonl=output_path,
        report_json=tmp_path / "cache_rebind_report.json",
    )

    upgraded = parse_stage_i_training_view_v2(
        json.loads(v2_path.read_text(encoding="utf-8")))
    rebound = [
        StageIIContextCacheV1.model_validate(json.loads(line))
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert upgraded.images == [str(mapped_image.resolve())]
    assert report["counts"] == {
        "cache_digest_changed": 2,
        "content_sha256_verified_media_pairs": 0,
        "context_kind:decision": 1,
        "context_kind:reasoning_grounding": 1,
        "input_caches": 2,
        "old_cache_binding:stage_i_v1_direct": 2,
        "output_caches": 2,
        "parse_status:accepted": 2,
        "projected_input_digest_changed": 2,
    }
    assert [item.canonical_text for item in rebound
            ] == [item.canonical_text for item in old_caches]
    assert all(new.projected_input_digest != old.projected_input_digest
               for new, old in zip(rebound, old_caches))


def test_cache_rebind_accepts_historical_case_only_after_content_match(
        tmp_path: Path) -> None:
    """A copied-media inference case may bind cache identity by file SHA."""

    full, mapped_image = _legacy_full_with_stale_image(tmp_path)
    v1_path = tmp_path / "stage_i_v1.jsonl"
    _write_jsonl(v1_path, [full])
    v2_path = tmp_path / "stage_i_v2.jsonl"
    migration_report = tmp_path / "stage_i_migration.json"
    upgrade_stage_i_full_cot_manifest_v2(
        v1_path,
        v2_path,
        report_json=migration_report,
        image_prefix_maps=[("/retired/stage-i", mapped_image.parent)],
    )
    case_image = tmp_path / "historical-run" / "copied.jpg"
    case_image.parent.mkdir()
    case_image.write_bytes(mapped_image.read_bytes())
    case_digest = stage_ii_projected_input_digest_v1(
        images=[str(case_image)],
        image_roles=list(full.image_roles),
        instruction=full.instruction,
        model_visible_context=full.model_visible_context,
        evidence_catalog=[
            item.model_dump(mode="json", exclude_none=True)
            for item in full.evidence_catalog
        ],
        input_modalities=full.input_modalities.model_dump(mode="json"),
    )
    old_cache = _legacy_cache(
        full,
        kind="decision",
        raw_generation=render_reasoning_section_v4(
            "Decision",
            full.reasoning_target.reasoning.decision,
        ),
        projected_input_digest=case_digest,
    )
    cache_path = tmp_path / "historical_cache.jsonl"
    _write_jsonl(cache_path, [old_cache])
    case = {
        "sample_id": f"{full.sample_id}:stage_ii_cfm",
        "source_group_id": full.source_group_id,
        "split_group_id": full.split_group_id,
        "dataset_source": full.dataset_source,
        "modality_profile_id": str(full.modality_profile_id),
        "images": [str(case_image)],
        "image_roles": list(full.image_roles),
        "public_context": {
            "parent_stage_i_sample_id": full.sample_id,
            "parent_source_fingerprint": full.parent_source_fingerprint,
            "projected_input_digest": case_digest,
            "input_modalities": full.input_modalities.model_dump(mode="json"),
        },
    }
    case_path = tmp_path / "historical_cases.jsonl"
    case_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    report = rebind_stage_ii_context_cache_manifest_v2(
        stage_i_v1_jsonl=v1_path,
        stage_i_v2_jsonl=v2_path,
        migration_report_json=migration_report,
        input_cache_jsonl=cache_path,
        output_cache_jsonl=tmp_path / "rebound.jsonl",
        cache_input_case_jsonl=case_path,
    )

    assert report["counts"][
        "old_cache_binding:historical_case_content_equivalent"] == 1
    assert report["counts"]["content_sha256_verified_media_pairs"] == 1

    case_image.write_bytes(b"different-image-content")
    with pytest.raises(ValueError, match="media content does not match"):
        rebind_stage_ii_context_cache_manifest_v2(
            stage_i_v1_jsonl=v1_path,
            stage_i_v2_jsonl=v2_path,
            migration_report_json=migration_report,
            input_cache_jsonl=cache_path,
            output_cache_jsonl=tmp_path / "rejected_rebind.jsonl",
            cache_input_case_jsonl=case_path,
        )


def test_cache_rebind_rejects_cache_not_bound_to_v1_parent(
        tmp_path: Path) -> None:
    """A valid cache envelope with the wrong observation digest fails closed."""

    full, mapped_image = _legacy_full_with_stale_image(tmp_path)
    v1_path = tmp_path / "stage_i_v1.jsonl"
    _write_jsonl(v1_path, [full])
    v2_path = tmp_path / "stage_i_v2.jsonl"
    migration_report = tmp_path / "stage_i_migration.json"
    upgrade_stage_i_full_cot_manifest_v2(
        v1_path,
        v2_path,
        report_json=migration_report,
        image_prefix_maps=[("/retired/stage-i", mapped_image.parent)],
    )
    invalid_cache = _legacy_cache(
        full,
        kind="decision",
        raw_generation=render_reasoning_section_v4(
            "Decision",
            full.reasoning_target.reasoning.decision,
        ),
        projected_input_digest="9" * 64,
    )
    cache_path = tmp_path / "invalid_cache.jsonl"
    _write_jsonl(cache_path, [invalid_cache])

    with pytest.raises(ValueError, match="not bound to the V1 parent"):
        rebind_stage_ii_context_cache_manifest_v2(
            stage_i_v1_jsonl=v1_path,
            stage_i_v2_jsonl=v2_path,
            migration_report_json=migration_report,
            input_cache_jsonl=cache_path,
            output_cache_jsonl=tmp_path / "should_not_exist.jsonl",
        )
