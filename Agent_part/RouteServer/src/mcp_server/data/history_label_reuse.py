"""Compare label anchors independently of the size of the observation window."""

from typing import Any, Mapping


def history_observation(observation: Any) -> dict[str, Any]:
    """Return the recorded FPV time mapping, or an empty mapping for legacy data.

    Args:
        observation: Canonical navigation observation.

    Returns:
        Existing prepared-history metadata without inventing legacy timestamps.
    """
    for item in observation.evidence_catalog:
        if str(item.ref) == "fpv.current":
            return item.attributes.get("history_observation", {})
    return {}


def anchor_label_evidence(
    old_source: Any, new_source: Any, *, media_identities: Mapping[str, str],
    anchor_records: Mapping[str, Any] | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Resolve equal clip/time anchors from recorded sampling or frame evidence.

    Args:
        old_source: Original canonical source.
        new_source: Expanded-history canonical source at the same stable anchor.
        media_identities: Proven frame identities; JPEG inequality is inconclusive.
        anchor_records: Optional source-fingerprint keyed legacy sampling records.
            Each contains clip_id, timestamp_ns, current_frame_id and provenance.

    Returns:
        Equality, diagnostic reason and the evidence used. Missing evidence is
        an alignment task, not a request to run a teacher.

    Raises:
        ValueError: If an external alignment record has no provenance.
    """
    old, new = old_source.observation, new_source.observation
    if old.dataset_source != new.dataset_source or old.source_group_id != new.source_group_id:
        return False, "anchor_identity_changed", {}

    def record(source):
        observation = source.observation
        supplied = (anchor_records or {}).get(source.source_fingerprint)
        if supplied is not None:
            if not supplied["provenance"]:
                raise ValueError("anchor sampling record requires provenance")
            return supplied
        history = history_observation(observation)
        if not history:
            return None
        return {
            "clip_id": observation.split_group_id.removeprefix(f"{observation.dataset_source}:"),
            "timestamp_ns": history["source_ns"][-1],
            "current_frame_id": f"video_ordinal:{history['video_ordinals'][-1]}",
            "provenance": "canonical.history_observation",
        }

    before, after = record(old_source), record(new_source)
    evidence = {"old_anchor": before, "new_anchor": after}
    if before is not None and after is not None:
        fields = ("clip_id", "timestamp_ns", "current_frame_id")
        if all(before[key] == after[key] for key in fields):
            return True, "anchor_time_and_frame_equal", evidence
        return False, "anchor_time_or_frame_changed", evidence

    old_image = old.images[old.image_roles.index("fpv.current")]
    new_image = new.images[new.image_roles.index("fpv.current")]
    old_id, new_id = media_identities.get(old_image), media_identities.get(new_image)
    # 同一稳定 anchor 的相同画面可以复用；JPEG 重编码差异本身不能证明时刻变化。
    if old_id is not None and old_id == new_id:
        evidence["current_frame_identity"] = old_id
        return True, "anchor_current_frame_equal", evidence
    return False, "anchor_alignment_unresolved", evidence


def same_evidence_image(old: Any, new: Any, role: str, identities: Mapping[str, str]) -> bool:
    """Compare a non-FPV image actually required by a label.

    Args:
        old: Original observation.
        new: Destination observation.
        role: Exact role, such as bev.current.
        identities: Proven media identities.

    Returns:
        Whether both observations supply the same proven image for that role.
    """
    if role not in old.image_roles or role not in new.image_roles:
        return False
    before = identities.get(old.images[old.image_roles.index(role)])
    after = identities.get(new.images[new.image_roles.index(role)])
    return before is not None and before == after
