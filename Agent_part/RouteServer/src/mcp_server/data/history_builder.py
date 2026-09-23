"""Join observation plans to existing parent identities and trajectory sidecars."""

from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from mcp_server.data.history_observation import BaseHistoryProcessor, CausalBevIndex, SampleRejected
from mcp_server.data.stage_ii.sources import (
    stage_ii_assign_split_v1,
    stage_ii_source_row_identity_v1,
)


def build_clip_observations(
    rows: Iterable[dict], *, source: Path, processor: BaseHistoryProcessor,
    spec, registry, sidecars: Mapping[str, dict], bev: CausalBevIndex | None,
) -> tuple[list[dict], list[dict], dict]:
    """Plan one clip and retain original supervision through canonical parent IDs.

    Args:
        rows: Original rows for one clip; label content is preserved verbatim.
        source: Raw clip directory or EgoWalk parquet file.
        processor: Dataset-specific source-clock processor.
        spec: Existing StageIITrajectorySourceSpecV1 for this dataset.
        registry: Existing registry carrying the frozen split policy.
        sidecars: Original trajectory sidecars keyed by canonical sample_id.
        bev: Recorded or rebuilt BEV index for SideWalk/Foxglove; None for EgoWalk.
    Returns:
        Accepted source/observation/sidecar records, expected rejections, and
        unique-media and split counts. Counts describe observation eligibility;
        semantic-label compatibility requires subsequent admission.
    Raises:
        ValueError: If identities repeat or sidecar identity/split disagree.
        OSError: If source IO fails. Only SampleRejected becomes an exclusion.
    """
    if spec.dataset_source in ("sidewalk", "foxglove") and bev is None:
        raise ValueError(f"{source}: BEV index required for {spec.dataset_source}")
    accepted, rejected = [], []
    parent_ids, unique_ordinals, unique_bev = set(), set(), set()
    split_counts, sidecar_counts, reasons = Counter(), Counter(), Counter()
    for row in rows:
        parent_id, group_id = stage_ii_source_row_identity_v1(row, spec)
        if parent_id in parent_ids:
            raise ValueError(f"{source}: duplicate parent identity {parent_id}")
        parent_ids.add(parent_id)
        split = stage_ii_assign_split_v1(group_id, registry)
        sidecar = sidecars.get(parent_id)
        if sidecar is not None:
            if (sidecar["sample_id"] != parent_id or sidecar["split_group_id"] != group_id
                    or sidecar["split"] != split or sidecar["dataset_source"] != spec.dataset_source):
                raise ValueError(f"{source}: sidecar identity/split mismatch: {parent_id}")
        try:
            observation = processor.plan(row, source, source_parent_id=parent_id)
            observation["bev"] = bev.select(observation["source_ns"][-1]) if bev else None
        except SampleRejected as error:
            reasons[error.reason] += 1
            rejected.append({"source_parent_id": parent_id, "reason": error.reason})
            continue
        # 保留原 source 和 sidecar 的身份及监督；观测字典只携带历史采样信息。
        accepted.append({
            "source_parent_id": parent_id,
            "split_group_id": group_id,
            "split": split,
            "observation": observation,
            "source_row": row,
            "trajectory_sidecar": sidecar,
        })
        unique_ordinals.update(observation["video_ordinals"])
        if observation["bev"] is not None:
            unique_bev.add(observation["bev"]["bev_image_path"])
        split_counts[split] += 1
        if sidecar is not None:
            sidecar_counts[split] += 1
    return accepted, rejected, {
        "source_rows": len(parent_ids),
        "history_eligible": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": dict(reasons),
        "unique_fpv": len(unique_ordinals),
        "unique_bev": len(unique_bev),
        "history_eligible_by_split": dict(split_counts),
        "history_with_sidecar_by_split": dict(sidecar_counts),
    }
