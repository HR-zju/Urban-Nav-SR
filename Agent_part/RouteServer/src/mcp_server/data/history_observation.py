"""Source-clock history selection and clip-local BEV loading for dataset rebuilds."""

from abc import ABC, abstractmethod
from functools import lru_cache
import json
from pathlib import Path

import numpy as np


class SampleRejected(ValueError):
    """An expected observation exclusion, distinct from corrupt source data.

    Args:
        reason: Stable reason code used by the sampling report.
        source: Source clip or parquet path identifying the exclusion.
    """

    def __init__(self, reason: str, source: Path):
        self.reason = reason
        self.source = source
        super().__init__(f"{source}: {reason}")


class CausalBevIndex:
    """Select latest-causal timestamps while preserving recorded BEV content provenance.

    Args:
        path: Recorded or rebuilt bev_index.json belonging to the selected clip.
        clip_id: Expected source clip identity.
        source_ns: Physical source timestamps for checking frame identity.
        max_age_ns: Explicit maximum BEV age at the observation anchor.

    Raises:
        ValueError: If profile, clip, timestamps or image paths disagree.
        OSError: If the index or a referenced image cannot be read.
    """

    def __init__(self, path: Path, *, clip_id: str, source_ns: np.ndarray, max_age_ns: int):
        index = json.loads(path.read_text(encoding="utf-8"))
        profile = index.get("input_profile", "recorded_bev_v1")
        if index["clip_id"] != clip_id:
            raise ValueError(f"{path}: BEV clip identity mismatch")
        if profile not in ("causal_map_v1", "recorded_bev_v1"):
            raise ValueError(f"{path}: unsupported BEV profile {profile}")
        if (profile == "causal_map_v1"
                and (index["bev_route_scope"] != "none" or index["draw_trajectory"])):
            raise ValueError(f"{path}: expected clip-owned causal_map_v1 without recorded route")
        if max_age_ns < 0:
            raise ValueError("max_age_ns must be nonnegative")
        self.path, self.max_age_ns = path, max_age_ns
        self.entries = [dict(entry) for entry in index["entries"]]
        self.timestamps = np.array([entry["timestamp"] for entry in self.entries], dtype=np.int64)
        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError(f"{path}: BEV timestamps must strictly increase")
        for entry in self.entries:
            frame = entry["frame_idx"]
            if not 0 <= frame < len(source_ns) or source_ns[frame] != entry["timestamp"]:
                raise ValueError(f"{path}: BEV source timestamp/frame mismatch: {frame}")
            image = Path(entry["bev_image_path"])
            if profile == "recorded_bev_v1":
                # 旧绝对盘符不决定媒体身份；已核验 clip 内的稳定文件名决定当前路径。
                image = Path("bev_images") / image.name
                entry["bev_image_path"] = image.as_posix()
            if image.is_absolute() or ".." in image.parts:
                raise ValueError(f"{path}: rebuilt BEV paths must be package-relative")
            if not (path.parent / image).is_file():
                raise FileNotFoundError(path.parent / image)
        self.provenance = {key: index[key] for key in (
            "draw_trajectory", "bev_route_scope", "bev_sample_fps", "bev_alignment",
            "bev_route_lookahead_s",
        ) if key in index}
        self.provenance.update({
            "input_profile": profile,
            "causality_scope": index.get("causality_scope", "prepared_frame_timestamps_only"),
            "sensor_sync_causality": index.get("sensor_sync_causality", "unverified"),
            # The compiler must carry the configured BEV age budget to the
            # model row; FPV history lag and BEV age are independent clocks.
            "bev_tolerance_ns": self.max_age_ns,
        })

    def select(self, anchor_ns: int) -> dict:
        """Choose max{i: BEV_time[i] <= anchor}, preserving causality scope.

        Args:
            anchor_ns: Observation anchor in source-clock nanoseconds.
        Returns:
            Selected entry, age and recorded provenance. Sensor synchronization
            marked unverified stays unverified after temporal selection.
        Raises:
            SampleRejected: If no causal BEV exists or its age exceeds the limit.
        """
        index = int(np.searchsorted(self.timestamps, anchor_ns, side="right")) - 1
        if index < 0:
            raise SampleRejected("causal_bev_missing", self.path)
        age = int(anchor_ns - self.timestamps[index])
        if age > self.max_age_ns:
            raise SampleRejected("bev_age_exceeded", self.path)
        return {**self.entries[index], **self.provenance, "age_ns": age}


class BaseHistoryProcessor(ABC):
    """Share the fixed 4-second, 2-Hz causal selection algorithm.

    Args:
        max_lag_ns: Explicit source-to-target lag limit in nanoseconds.
    """

    def __init__(self, max_lag_ns: int):
        if max_lag_ns < 0:
            raise ValueError("max_lag_ns must be nonnegative")
        self.max_lag_ns = max_lag_ns
        # 每个处理器只保留当前 clip，避免逐锚点重复读取 parquet/NPY。
        self._timeline = lru_cache(maxsize=1)(self._read_timeline)

    def _read_timeline(self, source: Path) -> tuple[np.ndarray, np.ndarray]:
        timestamps, ordinals = self.load_timeline(source)
        if timestamps.ndim != 1 or timestamps.size == 0:
            raise ValueError(f"{source}: source timestamps must be a nonempty vector")
        if timestamps.shape != ordinals.shape:
            raise ValueError(f"{source}: timestamps and video ordinals must have matching shape")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"{source}: source timestamps must strictly increase")
        if np.any(ordinals < 0) or np.any(np.diff(ordinals) <= 0):
            raise ValueError(f"{source}: video ordinals must be nonnegative and strictly increase")
        return timestamps, ordinals

    @abstractmethod
    def load_timeline(self, source: Path) -> tuple[np.ndarray, np.ndarray]:
        """Read source timestamps in ns and corresponding video ordinals.

        Args:
            source: Dataset-specific clip directory or parquet file.
        Returns:
            Timestamp and ordinal arrays with matching row order.
        """

    @abstractmethod
    def anchor_index(self, row: dict, source: Path) -> int:
        """Locate an existing source anchor using its recorded time and frame.

        Args:
            row: Original source-manifest row, with labels left unchanged.
            source: Clip directory or parquet file.
        Returns:
            Anchor row in the physical source timeline.
        Raises:
            ValueError: If the recorded anchor does not match the source.
        """

    def plan(self, row: dict, source: Path, *, source_parent_id: str) -> dict:
        """Build a JSON-compatible FPV observation from the original anchor.

        Args:
            row: Existing source row; future targets stay outside the observation.
            source: Dataset-specific timeline location.
            source_parent_id: Identity supplied by the existing source registry.
        Returns:
            Parent identity, profile and nine target/source timestamps and ordinals.
        Raises:
            SampleRejected: If a complete causal history cannot be selected.
            ValueError: If the external source clock or anchor disagrees.
        """
        anchor = self.anchor_index(row, source)
        selected = self.select_history(source, anchor)
        return {
            "source_parent_id": source_parent_id,
            "observation_profile": "history_4s_2hz_9f_v1",
            "anchor_row": anchor,
            **{name: values.tolist() for name, values in selected.items()},
        }

    def select_history(self, source: Path, anchor_row: int) -> dict:
        """Select i_k=max{i:s_i<=t-4s+k*0.5s}, k=0..8.

        Args:
            source: Clip directory or parquet file.
            anchor_row: Existing anchor's row in the source timeline.
        Returns:
            Target timestamps, actual timestamps and video ordinal arrays.
        Raises:
            SampleRejected: If history is incomplete or exceeds the sampling lag.
            ValueError: If the source clock or ordinal mapping is invalid.
            IndexError: If the anchor is outside the source timeline.
        """
        timestamps, ordinals = self._timeline(source)
        if not 0 <= anchor_row < len(timestamps):
            raise IndexError(f"{source}: anchor row {anchor_row} outside source timeline")
        targets = timestamps[anchor_row] + np.arange(-8, 1, dtype=np.int64) * 500_000_000
        # right-1 保证每个槽位只读目标时刻及之前的源帧，保留真实时差。
        rows = np.searchsorted(timestamps, targets, side="right") - 1
        if np.any(rows < 0):
            raise SampleRejected("insufficient_history", source)
        selected = ordinals[rows]
        if len(np.unique(selected)) != 9:
            raise SampleRejected("repeated_source_frame", source)
        if np.any(targets - timestamps[rows] > self.max_lag_ns):
            raise SampleRejected("history_lag_exceeded", source)
        return {"target_ns": targets, "source_ns": timestamps[rows], "video_ordinals": selected}


class SidewalkHistoryProcessor(BaseHistoryProcessor):
    """Read prepared SideWalk clips with clip-owned BEV media.

    Args:
        max_lag_ns: Maximum permitted source sampling lag.
    """

    def load_timeline(self, source: Path) -> tuple[np.ndarray, np.ndarray]:
        """Read nanosecond timestamps and prepared video row ordinals.

        Args:
            source: Prepared SideWalk clip directory.
        Returns:
            int64 source timestamps and zero-based video ordinals.
        """
        timestamps = np.load(source / "frame_times.npy").astype(np.int64)
        return timestamps, np.arange(len(timestamps), dtype=np.int64)

    def anchor_index(self, row: dict, source: Path) -> int:
        """Resolve SideWalk/Foxglove metadata against its clip's source clock.

        Args:
            row: Source row containing sidewalk_meta.
            source: Prepared clip directory whose name is the clip identity.
        Returns:
            Verified source row index, distinct from a mapped video ordinal.
        Raises:
            ValueError: If clip, timestamp or frame index disagree.
        """
        meta = row["sidewalk_meta"]
        timestamps, _ = self._timeline(source)
        index = int(meta["frame_idx"])
        if meta["clip_id"] != source.name:
            raise ValueError(f"{source}: source anchor clip mismatch")
        if not 0 <= index < len(timestamps) or timestamps[index] != int(meta["timestamp"]):
            raise ValueError(f"{source}: source anchor timestamp/frame mismatch: {index}")
        return index

    def load_bev_index(self, clip: Path) -> list[dict]:
        """Resolve BEV within its clip and verify source-frame timestamps.

        Args:
            clip: Current raw_root/chunk/clip directory.
        Returns:
            Entries with clip-relative paths; original index stays unchanged.
        Raises:
            FileNotFoundError: If the index or a referenced clip-local image is missing.
            ValueError: If an index belongs to another clip or its timestamps disagree.
        """
        index = json.loads((clip / "bev_index.json").read_text(encoding="utf-8"))
        if index["clip_id"] != clip.name:
            raise ValueError(f"{clip}: BEV clip identity mismatch")
        timestamps, _ = self._timeline(clip)
        result = []
        for item in index["entries"]:
            path = Path("bev_images") / Path(item["bev_image_path"]).name
            if not (clip / path).is_file():
                raise FileNotFoundError(clip / path)
            if int(timestamps[item["frame_idx"]]) != int(item["timestamp"]):
                raise ValueError(f"{clip}: BEV timestamp mismatch at frame {item['frame_idx']}")
            result.append({**item, "bev_image_path": path.as_posix()})
        return result


class FoxgloveHistoryProcessor(SidewalkHistoryProcessor):
    """Read Foxglove's prepared NPY layout using a verified row-to-video map.

    Args:
        max_lag_ns: Maximum permitted source sampling lag.
        video_ordinals: Audited source-row to video-ordinal array for this clip.
    """

    def __init__(self, max_lag_ns: int, video_ordinals: np.ndarray):
        super().__init__(max_lag_ns)
        self.video_ordinals = np.asarray(video_ordinals, dtype=np.int64)

    def load_timeline(self, source: Path) -> tuple[np.ndarray, np.ndarray]:
        """Bind source timestamps to the supplied, clip-specific audited map.

        Args:
            source: Prepared Foxglove clip directory.
        Returns:
            Source timestamps and verified video ordinals.
        Raises:
            ValueError: If map length differs from source row count.
        """
        timestamps, _ = super().load_timeline(source)
        if len(timestamps) != len(self.video_ordinals):
            raise ValueError(f"{source}: video map length differs from source timeline")
        return timestamps, self.video_ordinals


class EgoWalkHistoryProcessor(BaseHistoryProcessor):
    """Read EgoWalk source-clock parquet without previously extracted media.

    Args:
        max_lag_ns: Maximum permitted source sampling lag.
    """

    def load_timeline(self, source: Path) -> tuple[np.ndarray, np.ndarray]:
        """Read physical timestamps and video ordinals from parquet.

        Args:
            source: EgoWalk data parquet file.
        Returns:
            Millisecond timestamps converted to ns, and the original frame column.
        """
        import pyarrow.parquet as pq
        table = pq.read_table(source, columns=["timestamp", "frame"])
        # 容器播放时间不同于采集时间：timestamp 定义物理时钟，frame 仅定义视频 ordinal。
        timestamps = np.asarray(table["timestamp"], dtype=np.int64) * 1_000_000
        return timestamps, np.asarray(table["frame"], dtype=np.int64)

    def anchor_index(self, row: dict, source: Path) -> int:
        """Match the original EgoWalk anchor by timestamp and physical frame ID.

        Args:
            row: EgoWalk source row with clip_id, timestamp (ms) and frame_idx.
            source: Original clip parquet, named by clip_id.
        Returns:
            Matching parquet row index.
        Raises:
            ValueError: If the source clip or anchor identity disagrees.
        """
        if row["clip_id"] != source.stem:
            raise ValueError(f"{source}: source anchor clip mismatch")
        timestamps, ordinals = self._timeline(source)
        timestamp_ns = int(row["timestamp"]) * 1_000_000
        index = int(np.searchsorted(timestamps, timestamp_ns))
        if (index == len(timestamps) or timestamps[index] != timestamp_ns
                or ordinals[index] != int(row["frame_idx"])):
            raise ValueError(f"{source}: source anchor timestamp/frame mismatch")
        return index
