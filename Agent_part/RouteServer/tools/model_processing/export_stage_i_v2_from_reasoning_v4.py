#!/usr/bin/env python3
"""Export formal Reasoning V4.6 parents into the Stage I V2 FullCoT contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from mcp_server.contracts.stage_i_sft import StageIFullCoTViewV2
from mcp_server.data.stage_i.views import project_stage_i_reasoning_view_with_audit_v2


def _iter_jsonl(paths: Iterable[Path]):
    """Yield source payloads with their originating path and line number."""
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    yield path, line_number, json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    """Write deterministic JSONL and return its SHA256 digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            handle.buffer.write(payload)
            digest.update(payload)
    return digest.hexdigest()


def export(args: argparse.Namespace) -> dict[str, Any]:
    """Project formal V4.6 rows and persist accepted rows plus drop audit."""
    accepted: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    seen: set[str] = set()
    reasons: Counter[str] = Counter()

    for path, line_number, raw in _iter_jsonl(args.input):
        sample_id = str(raw.get("sample_id") or f"{path}:{line_number}")
        if raw.get("formal_training_eligible") is not True:
            reasons["not_formal_training_eligible"] += 1
            drops.append({"sample_id": sample_id, "reason": "not_formal_training_eligible"})
            continue
        if sample_id in seen:
            reasons["duplicate_sample_id"] += 1
            drops.append({"sample_id": sample_id, "reason": "duplicate_sample_id"})
            continue
        role = "primary" if str(raw.get("instruction") or "").strip() else "not_present"
        result = project_stage_i_reasoning_view_with_audit_v2(
            raw,
            modality_profile_id=args.modality_profile_id,
            instruction_training_role=role,
        )
        if not isinstance(result, StageIFullCoTViewV2):
            payload = result.model_dump(mode="json", exclude_none=True)
            reason = str(payload.get("reason_code") or "projection_drop")
            reasons[reason] += 1
            drops.append(payload)
            continue
        seen.add(sample_id)
        accepted.append(result.model_dump(mode="json", exclude_none=True))

    output_sha = _write_jsonl(args.output, accepted)
    drop_sha = _write_jsonl(args.drop, drops)
    report = {
        "schema_version": "urbannav_stage_i_v2_from_reasoning_v4_export_v1",
        "modality_profile_id": args.modality_profile_id,
        "inputs": [str(path) for path in args.input],
        "outputs": {
            "accepted": {"path": str(args.output), "rows": len(accepted), "sha256": output_sha},
            "dropped": {"path": str(args.drop), "rows": len(drops), "sha256": drop_sha},
        },
        "drop_reasons": dict(sorted(reasons.items())),
        "unique_parent_ids": len(seen),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    """Parse the formal-view export CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--drop", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--modality-profile-id", default="fpv_temporal_instruction_bev_v1")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(export(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
