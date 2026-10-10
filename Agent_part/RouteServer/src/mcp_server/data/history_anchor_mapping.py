"""Bridge retained prepared timestamps to expanded-history anchors."""

import json
from collections import Counter
from pathlib import Path

from mcp_server.data.io.jsonl import iter_jsonl


def build_anchor_map(manifest: Path, dataset: str, output: Path, *,
                     old_sources: list[Path], overrides: Path | None = None,
                     tolerance_ns: int = 100_000_000) -> dict:
    """Write anchor mappings and legacy source-fingerprint alignment records.

    Args:
        manifest: Portable observations manifest retaining the original source_row.
        dataset: sidewalk, foxglove or egowalk; determines the recorded clock unit.
        output: New output directory.
        old_sources: Original canonical sources, including VQA source revisions.
        overrides: Verified physical-frame records keyed by source_fingerprint.
        tolerance_ns: Inclusive bound abs(t_old - t_new) <= tolerance_ns.

    Returns:
        Counts of mapped anchors, exclusions and linked old source revisions.

    Raises:
        ValueError: If input identities conflict or a duplicate anchor is present.
        FileExistsError: If an output file already exists.

    Prepared timestamps establish annotation-anchor alignment. They do not prove
    raw sensor synchronization or the decoded identity of every legacy JPEG.
    """
    if dataset not in {"sidewalk", "foxglove", "egowalk"} or tolerance_ns < 0:
        raise ValueError("supported dataset and nonnegative tolerance required")
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    anchors = {}
    with (output / "anchor_map.jsonl").open("x") as stream:
        for row in iter_jsonl(manifest):
            key = row["source_parent_id"]
            if key in anchors:
                raise ValueError(f"duplicate anchor: {key}")
            original = row["source_row"]
            meta = original if dataset == "egowalk" else original["sidewalk_meta"]
            clip = meta["clip_id"]
            if row["split_group_id"] != f"{dataset}:{clip}":
                raise ValueError(f"clip identity conflict: {key}")
            old_time = meta.get("timestamp")
            old_ns = None if old_time is None else int(old_time) * (1_000_000 if dataset == "egowalk" else 1)
            new_ns = int(row["observation"]["source_ns"][-1])
            delta = None if old_ns is None else abs(old_ns - new_ns)
            status = ("missing_original_timestamp" if delta is None else
                      "mapped" if delta <= tolerance_ns else "outside_tolerance")
            mapped = {
                "source_group_id": key, "clip_id": clip, "split": row["split"],
                "old_timestamp_ns": old_ns, "new_timestamp_ns": new_ns,
                "old_frame_idx": meta["frame_idx"],
                "new_video_ordinal": row["observation"]["video_ordinals"][-1],
                "delta_ns": delta, "tolerance_ns": tolerance_ns, "status": status,
                "alignment_level": "prepared_sampling_timestamps",
                "provenance": {"manifest": manifest.name,
                               "path_base": "input_manifest_directory",
                               "field": "source_row.timestamp" if dataset == "egowalk"
                               else "source_row.sidewalk_meta.timestamp"},
            }
            anchors[key] = mapped
            counts[status] += 1
            stream.write(json.dumps(mapped, ensure_ascii=False) + "\n")
    verified = ({r["source_fingerprint"]: r for r in iter_jsonl(overrides)} if overrides else {})
    seen = set()
    with (output / "anchor_records.jsonl").open("x") as stream:
        for path in old_sources:
            for source in iter_jsonl(path):
                fingerprint = source["source_fingerprint"]
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                observation = source["observation"]
                mapping = anchors.get(observation["source_group_id"])
                if mapping is None or mapping["status"] != "mapped":
                    counts["old_source_unmapped"] += 1
                    continue
                if (observation["dataset_source"] != dataset or
                        observation["split_group_id"].removeprefix(f"{dataset}:") != mapping["clip_id"]):
                    raise ValueError("old source clip identity conflicts with retained sampling metadata")
                record = {
                    "source_fingerprint": fingerprint, "clip_id": mapping["clip_id"],
                    "timestamp_ns": mapping["old_timestamp_ns"],
                    "current_frame_id": f"prepared_frame:{mapping['old_frame_idx']}",
                    "provenance": mapping["provenance"],
                    "alignment_level": mapping["alignment_level"],
                }
                # 个别已核实的物理帧映射覆盖名义采样时刻，禁止推广全局帧偏移。
                if fingerprint in verified:
                    record = verified[fingerprint]
                    if record["clip_id"] != mapping["clip_id"]:
                        raise ValueError("verified frame record belongs to a different clip")
                    counts["verified_frame_overrides"] += 1
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["old_sources_linked"] += 1
    report = {"dataset": dataset, "tolerance_ns": tolerance_ns, "counts": dict(counts),
              "scope": "prepared sampling alignment; raw sensor synchronization remains unverified"}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
