"""Explicit instruction-provenance registry for Stage I projection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from mcp_server.contracts.stage_i_sft import (
    STAGE_I_INSTRUCTION_PROVENANCE_SCHEMA_VERSION,
    StageIInstructionProvenanceV1,
    StageIInstructionTrainingRoleV1,
    stage_i_projection_digest_v1,
)
from mcp_server.data.io.jsonl import iter_jsonl


def instruction_sha256_v1(instruction: str) -> str:
    """Hash exact visible instruction text without normalization."""

    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()


def build_stage_i_instruction_provenance_v1(
    *,
    source_group_id: str,
    source_fingerprint: str,
    instruction: str | None,
    training_role: StageIInstructionTrainingRoleV1,
    provenance_source: str,
    policy_id: str,
    policy_version: str,
) -> StageIInstructionProvenanceV1:
    """Build one digest-bound entry from an explicit policy decision."""

    payload = {
        "schema_version": STAGE_I_INSTRUCTION_PROVENANCE_SCHEMA_VERSION,
        "source_group_id": source_group_id,
        "source_fingerprint": source_fingerprint,
        "instruction_sha256": (
            instruction_sha256_v1(instruction)
            if instruction is not None
            else None
        ),
        "training_role": training_role,
        "provenance_source": provenance_source,
        "policy_id": policy_id,
        "policy_version": policy_version,
    }
    return StageIInstructionProvenanceV1(
        **payload,
        entry_digest=stage_i_projection_digest_v1(payload),
    )


def load_stage_i_instruction_provenance_v1(
    path: str | Path,
) -> dict[tuple[str, str], StageIInstructionProvenanceV1]:
    """Load a registry and reject duplicate source-revision assignments."""

    result: dict[tuple[str, str], StageIInstructionProvenanceV1] = {}
    for raw in iter_jsonl(Path(path).expanduser()):
        entry = StageIInstructionProvenanceV1.model_validate(raw)
        key = (entry.source_group_id, entry.source_fingerprint)
        if key in result:
            raise ValueError(
                f"duplicate Stage I instruction provenance: {key}"
            )
        result[key] = entry
    return result


def resolve_stage_i_instruction_role_v1(
    *,
    source_group_id: str,
    source_fingerprint: str,
    instruction: str | None,
    registry: dict[
        tuple[str, str],
        StageIInstructionProvenanceV1,
    ],
) -> StageIInstructionTrainingRoleV1:
    """Resolve an exact source revision or fail closed for present text."""

    entry = registry.get((source_group_id, source_fingerprint))
    if instruction is None:
        if entry is not None and entry.training_role != "not_present":
            raise ValueError(
                "instruction provenance says present but source has none"
            )
        return "not_present"
    if entry is None:
        raise ValueError(
            "instruction provenance is missing for source revision: "
            f"{source_group_id}/{source_fingerprint}"
        )
    if entry.instruction_sha256 != instruction_sha256_v1(instruction):
        raise ValueError(
            "instruction provenance text digest does not match source"
        )
    if entry.training_role == "not_present":
        raise ValueError(
            "instruction provenance says absent but source has text"
        )
    return entry.training_role


__all__ = [
    "build_stage_i_instruction_provenance_v1",
    "instruction_sha256_v1",
    "load_stage_i_instruction_provenance_v1",
    "resolve_stage_i_instruction_role_v1",
]
