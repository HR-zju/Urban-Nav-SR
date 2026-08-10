"""Deterministic stratified source sampling for quality-filter smoke reviews.

The sampler never assigns visual labels. It can use existing direction and
motion sidecars plus source-visible text metadata to make a useful small review
set. Ambiguous textual matches are recorded as selection hints, not training
facts; the LVLM profiles and human review remain authoritative for scene type.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4
from mcp_server.data.io.jsonl import file_sha256, iter_jsonl, write_jsonl_atomic
from mcp_server.data.quality.trainability import summarize_reasoning_source_motion_v1


NAVIGATION_QUALITY_SMOKE_SCHEMA_VERSION_V1 = "urbannav_navigation_quality_smoke_v1"
NAVIGATION_QUALITY_SMOKE_STRATA_V1 = (
    "indoor_or_transit",
    "stop",
    "high_speed",
    "dense_crowd",
    "ordinary_sidewalk",
    "turn",
)
_INDOOR_OR_TRANSIT = re.compile(r"\b(metro|subway|train|station|platform|indoor|mall|corridor|elevator)\b", re.I)
_DENSE_CROWD = re.compile(r"\b(crowd|crowded|dense\s+pedestrian|many\s+pedestrian|queue)\b", re.I)


@dataclass(frozen=True)
class NavigationQualitySmokeConfigV1:
    """Configure fixed-size, no-replacement quality smoke selection.

    Args:
        target_per_stratum: Requested count for each named stratum.
        max_rows: Upper bound for one dataset smoke manifest.
        high_speed_mps: Motion-summary threshold for high-speed audit samples.
        seed: Deterministic ranking seed independent of manifest order.
    """

    target_per_stratum: int = 24
    max_rows: int = 160
    high_speed_mps: float = 3.0
    seed: str = "urbannav-navigation-quality-smoke-v1"


def build_navigation_quality_smoke_manifest_v1(
    *,
    source_jsonl: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    dataset_source: str | None = None,
    config: NavigationQualitySmokeConfigV1 | None = None,
) -> dict[str, Any]:
    """Build one deterministic raw-source smoke view with strata receipts.

    Args:
        source_jsonl: Canonical V4 source ledger.
        output_jsonl: New source-only smoke manifest destination.
        report_json: Selection receipt and stratum-shortfall report.
        dataset_source: Optional exact dataset filter for combined ledgers.
        config: Fixed selection policy.

    Returns:
        Counts, shortfalls, and content digests for the smoke view.

    Raises:
        ValueError: If canonical source parents are duplicated.
    """

    cfg = config or NavigationQualitySmokeConfigV1()
    if cfg.target_per_stratum < 1 or cfg.max_rows < 1 or cfg.high_speed_mps <= 0.0:
        raise ValueError("navigation_quality_smoke_config_requires_positive_bounds")
    sources = _load_sources_v1(source_jsonl, dataset_source=dataset_source)
    buckets: dict[str, list[ReasoningSourceV4]] = defaultdict(list)
    source_hints: dict[str, tuple[str, ...]] = {}
    for source in sources:
        hints = _source_strata_hints_v1(source, high_speed_mps=cfg.high_speed_mps)
        source_hints[source.observation.source_group_id] = hints
        for hint in hints:
            buckets[hint].append(source)
    for values in buckets.values():
        values.sort(key=lambda item: _stable_rank_v1(item, seed=cfg.seed))

    chosen: list[ReasoningSourceV4] = []
    chosen_ids: set[str] = set()
    selected_counts: Counter[str] = Counter()
    available_counts = {stratum: len(buckets[stratum]) for stratum in NAVIGATION_QUALITY_SMOKE_STRATA_V1}
    for stratum in NAVIGATION_QUALITY_SMOKE_STRATA_V1:
        for source in buckets[stratum]:
            if len(chosen) >= cfg.max_rows or selected_counts[stratum] >= cfg.target_per_stratum:
                break
            parent_id = source.observation.source_group_id
            if parent_id in chosen_ids:
                continue
            chosen.append(source)
            chosen_ids.add(parent_id)
            selected_counts[stratum] += 1
    # 小类不足时以稳定全局顺序填充，保留 shortfall 而不是无证据复制样本。
    for source in sorted(sources, key=lambda item: _stable_rank_v1(item, seed=cfg.seed)):
        if len(chosen) >= cfg.max_rows:
            break
        parent_id = source.observation.source_group_id
        if parent_id in chosen_ids:
            continue
        chosen.append(source)
        chosen_ids.add(parent_id)
    write_jsonl_atomic(output_jsonl, [source.model_dump(mode="json") for source in chosen])
    selected_hint_counts = Counter(
        hint for source in chosen for hint in source_hints[source.observation.source_group_id]
    )
    report = {
        "schema_version": NAVIGATION_QUALITY_SMOKE_SCHEMA_VERSION_V1,
        "source_jsonl": str(Path(source_jsonl).expanduser().resolve()),
        "source_jsonl_sha256": file_sha256(source_jsonl),
        "output_jsonl": str(Path(output_jsonl).expanduser().resolve()),
        "dataset_source_filter": dataset_source,
        "input_source_count": len(sources),
        "selected_count": len(chosen),
        "target_per_stratum": cfg.target_per_stratum,
        "max_rows": cfg.max_rows,
        "high_speed_mps": cfg.high_speed_mps,
        "seed": cfg.seed,
        "available_hint_counts": available_counts,
        "selected_primary_counts": dict(selected_counts),
        "selected_all_hint_counts": dict(selected_hint_counts),
        "shortfalls": {
            stratum: max(0, cfg.target_per_stratum - selected_counts[stratum])
            for stratum in NAVIGATION_QUALITY_SMOKE_STRATA_V1
        },
        "stratum_provenance": {
            "indoor_or_transit": "visible instruction/context/warning keyword hint only",
            "stop": "existing strong direction label or deterministic stationary motion",
            "high_speed": "teacher-only motion summary above configured threshold",
            "dense_crowd": "visible instruction/context/warning keyword hint only",
            "ordinary_sidewalk": "SideWalk source without another selected hint",
            "turn": "existing strong direction label or retained continuous geometry",
        },
        "no_media_mutation": True,
        "no_visual_label_claim": True,
    }
    report_path = Path(report_json).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _load_sources_v1(path: str | Path, *, dataset_source: str | None) -> list[ReasoningSourceV4]:
    """Load one V4 source per parent, optionally for one exact dataset family."""

    rows: list[ReasoningSourceV4] = []
    seen: set[str] = set()
    for payload in iter_jsonl(path):
        source = ReasoningSourceV4.model_validate(payload)
        if dataset_source is not None and source.observation.dataset_source != dataset_source:
            continue
        parent_id = source.observation.source_group_id
        if parent_id in seen:
            raise ValueError(f"duplicate_source_parent_sample_id:{parent_id}")
        seen.add(parent_id)
        rows.append(source)
    return rows


def _source_strata_hints_v1(source: ReasoningSourceV4, *, high_speed_mps: float) -> tuple[str, ...]:
    """Return bounded source-side selection hints, never visual annotations."""

    hints: list[str] = []
    text = _visible_source_text_v1(source)
    if _INDOOR_OR_TRANSIT.search(text):
        hints.append("indoor_or_transit")
    supervision = source.direction_supervision
    label_name = str(supervision.label_name) if supervision is not None else ""
    summary = summarize_reasoning_source_motion_v1(source)
    if label_name == "stop" or summary.recorded_motion_state == "stationary":
        hints.append("stop")
    if summary.available and (summary.max_speed_mps or 0.0) >= high_speed_mps:
        hints.append("high_speed")
    if _DENSE_CROWD.search(text):
        hints.append("dense_crowd")
    if label_name.startswith(("left_", "right_", "turn_left", "turn_right")):
        hints.append("turn")
    # 较早的 V4.6 source 尚未写入 optional direction_geometry。分层
    # 不能因此拒绝旧 source；已有强 direction label 仍是首选依据。
    direction_geometry = getattr(source, "direction_geometry", None)
    if (
        "turn" not in hints
        and direction_geometry is not None
        and abs(direction_geometry.raw_angle_deg or 0.0) >= 8.0
    ):
        hints.append("turn")
    if source.observation.dataset_source == "sidewalk" and not hints:
        hints.append("ordinary_sidewalk")
    return tuple(hints)


def _visible_source_text_v1(source: ReasoningSourceV4) -> str:
    """Serialize only visible instruction/context/warnings for hint matching."""

    pieces = [source.observation.instruction or ""]
    pieces.append(json.dumps(source.observation.model_visible_context, ensure_ascii=False, sort_keys=True))
    pieces.extend(source.warnings)
    return "\n".join(pieces)


def _stable_rank_v1(source: ReasoningSourceV4, *, seed: str) -> str:
    """Derive a manifest-order-independent deterministic selection rank."""

    value = f"{seed}\n{source.observation.source_group_id}\n{source.source_fingerprint}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "NAVIGATION_QUALITY_SMOKE_SCHEMA_VERSION_V1",
    "NAVIGATION_QUALITY_SMOKE_STRATA_V1",
    "NavigationQualitySmokeConfigV1",
    "build_navigation_quality_smoke_manifest_v1",
]
