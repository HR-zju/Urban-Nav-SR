"""Sequential clip planning over an explicit, source-clock corpus inventory."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np

from mcp_server.data.history_observation import (
    CausalBevIndex, EgoWalkHistoryProcessor, FoxgloveHistoryProcessor, SidewalkHistoryProcessor,
)
from mcp_server.data.history_media import MediaDecodeError
from mcp_server.data.history_package import archive_history_package, materialize_clip_plan, read_jsonl, write_clip_plan


def prepare_history_clip(*, dataset: str, clip_id: str, source: Path, max_lag_ns: int,
                         bev_index: Path | None, max_bev_age_ns: int | None, video_map: Path | None):
    """Construct the dataset processor and its explicit map/time inputs.

    Args:
        dataset: Registered sidewalk, foxglove or egowalk source.
        clip_id: Exact source clip identity.
        source: Raw timeline directory or parquet file.
        max_lag_ns: Explicit source sampling lag budget.
        bev_index: Recorded or rebuilt clip-owned BEV index for SideWalk/Foxglove.
        max_bev_age_ns: Explicit BEV age budget for mapped sources.
        video_map: Audited Foxglove ordinal mapping with clip identity/evidence.
    Returns:
        Dataset processor and optional causal BEV index.
    Raises:
        ValueError: If the selected dataset's required inputs disagree.
        OSError: If a required source/index file cannot be read.
    """
    if dataset == "egowalk":
        if bev_index is not None or max_bev_age_ns is not None:
            raise ValueError("EgoWalk uses the FPV-only observation profile")
        return EgoWalkHistoryProcessor(max_lag_ns), None
    if dataset == "sidewalk":
        processor = SidewalkHistoryProcessor(max_lag_ns)
    elif dataset == "foxglove":
        if video_map is None:
            raise ValueError("Foxglove requires a clip-specific video map")
        mapping = json.loads(video_map.read_text(encoding="utf-8"))
        if mapping["clip_id"] != clip_id or not mapping["evidence"]:
            raise ValueError(f"{video_map}: clip identity and mapping evidence required")
        processor = FoxgloveHistoryProcessor(max_lag_ns, np.array(mapping["video_ordinals"]))
    else:
        raise ValueError(f"Unsupported history dataset: {dataset}")
    if bev_index is None or max_bev_age_ns is None:
        raise ValueError("SideWalk/Foxglove require a BEV index and an explicit age budget")
    timestamps, _ = processor.load_timeline(source)
    return processor, CausalBevIndex(bev_index, clip_id=clip_id, source_ns=timestamps, max_age_ns=max_bev_age_ns)


def plan_history_corpus(*, jobs: list[dict], source_index: Path, spec, registry, output: Path,
                        max_lag_ns: int, max_bev_age_ns: int | None, fpv_bytes_per_frame: int | None = None,
                        bev_bytes_per_frame: int | None = None) -> dict:
    """Plan declared clips once each, aggregating observations without decoding.

    Args:
        jobs: Ordered clip_id/source/video records, with optional bev_index/video_map paths.
            Foxglove jobs without an audited video_map are recorded as excluded clips.
        source_index: Completed source-occurrence SQLite index for this dataset.
        spec: Canonical dataset source specification.
        registry: Frozen identity/split policy.
        output: New corpus-plan directory containing numbered clip plans.
        max_lag_ns: Explicit physical sampling lag budget.
        max_bev_age_ns: BEV age budget, or None for EgoWalk.
        fpv_bytes_per_frame: Optional measured mean JPEG size for capacity estimation.
        bev_bytes_per_frame: Optional measured mean BEV size for capacity estimation.
    Returns:
        Aggregate counts and optional media-only byte estimate, saved last as summary.json.
    Raises:
        ValueError: If jobs repeat, size assumptions are invalid or a clip contract fails.
        OSError: If source IO or output creation fails; completed clip plans remain inspectable.
    """
    if not jobs or len({job["clip_id"] for job in jobs}) != len(jobs):
        raise ValueError("Corpus planning requires nonempty, unique clip jobs")
    for size in (fpv_bytes_per_frame, bev_bytes_per_frame):
        if size is not None and size <= 0:
            raise ValueError("Measured bytes per frame must be positive")
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    maps = {key: Counter() for key in (
        "rejection_reasons", "history_eligible_by_split", "history_with_sidecar_by_split")}
    with (output / "clips.jsonl").open("x", encoding="utf-8") as progress:
        for index, job in enumerate(jobs):
            if spec.dataset_source == "foxglove" and not job.get("video_map"):
                # 映射缺证据是显式准入排除；已提供但损坏的映射仍由读取/校验直接报错。
                counts["excluded_clips"] += 1
                progress.write(json.dumps({"clip_id": job["clip_id"], "status": "excluded",
                                           "reason": "video_mapping_unverified"}) + "\n")
                progress.flush()
                continue
            source, video = Path(job["source"]), Path(job["video"])
            if not video.is_file():
                raise FileNotFoundError(video)
            processor, bev = prepare_history_clip(
                dataset=spec.dataset_source, clip_id=job["clip_id"], source=source, max_lag_ns=max_lag_ns,
                bev_index=Path(job["bev_index"]) if job.get("bev_index") else None,
                max_bev_age_ns=max_bev_age_ns,
                video_map=Path(job["video_map"]) if job.get("video_map") else None)
            directory = f"clips/{index:06d}"
            summary = write_clip_plan(
                manifests=[], sidecar_paths=[], source_index=source_index, clip_id=job["clip_id"],
                source=source, video=video, processor=processor, spec=spec, registry=registry, bev=bev,
                output=output / directory, max_samples=None)
            for key in ("source_rows", "history_eligible", "rejected", "unique_fpv", "unique_bev",
                        "multiple_annotation_parents"):
                counts[key] += summary[key]
            for key, counter in maps.items():
                counter.update(summary[key])
            counts["planned_clips"] += 1
            progress.write(json.dumps({"plan_directory": directory, **summary}, ensure_ascii=False) + "\n")
            progress.flush()
    result = {"status": "planned", "dataset_source": spec.dataset_source, "clips": len(jobs),
              "source_index": str(source_index.resolve()), **dict(counts),
              **{key: dict(counter) for key, counter in maps.items()}, "semantic_label_admission": "pending"}
    if fpv_bytes_per_frame is not None and (counts["unique_bev"] == 0 or bev_bytes_per_frame is not None):
        # 片内去重计数乘实测均值；不包括原视频、manifest、文件系统分配开销及临时副本。
        result["capacity_estimate"] = {
            "fpv_bytes_per_frame": fpv_bytes_per_frame, "bev_bytes_per_frame": bev_bytes_per_frame,
            "media_bytes": counts["unique_fpv"] * fpv_bytes_per_frame
            + counts["unique_bev"] * (bev_bytes_per_frame or 0),
            "scope": "encoded_media_only",
        }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def materialize_history_corpus(plan: Path, output: Path, *, jpeg_quality: int, archive: bool = False) -> dict:
    """Materialize completed clip plans sequentially, retaining explicit exclusions.

    Args:
        plan: Completed plan-batch root containing summary.json and clips.jsonl.
        output: New destination for clip packages and completion receipts.
        jpeg_quality: Explicit encoding quality passed to the existing media writer.
        archive: Also create one tar per completed clip package.
    Returns:
        Counts and actual media/archive bytes, written after all clips complete.
    Raises:
        ValueError: If the corpus plan has no completion status.
        OSError: If reading, decoding or writing fails; prior clip receipts remain available.
    """
    summary = json.loads((plan / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "planned":
        raise ValueError("Batch materialization requires a completed corpus plan")
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    with (output / "clips.jsonl").open("x", encoding="utf-8") as progress:
        for job in read_jsonl(plan / "clips.jsonl"):
            if job.get("status") == "excluded":
                result = job
                counts["excluded_clips"] += 1
            elif job["history_eligible"] == 0:
                result = {"clip_id": job["clip_id"], "status": "excluded", "reason": "no_eligible_history"}
                counts["excluded_clips"] += 1
            else:
                directory = job["plan_directory"]
                receipt = materialize_clip_plan(plan / directory, output / directory, jpeg_quality=jpeg_quality)
                result = {"clip_id": job["clip_id"], "package_directory": directory, **receipt}
                counts["materialized_clips"] += 1
                counts.update({key: receipt[key] for key in ("samples", "unique_fpv", "unique_bev", "media_bytes")})
                if archive:
                    result["archive"] = archive_history_package(output / directory, output / f"{directory}.tar")
                    counts["archive_bytes"] += result["archive"]["archive_bytes"]
            # 每个成功 clip 即时落盘；错误直接传播，整批完成记录只在循环结束后写入。
            progress.write(json.dumps(result, ensure_ascii=False) + "\n")
            progress.flush()
    result = {"status": "materialized", "plan": str(plan.resolve()), **dict(counts)}
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def materialize_history_corpus_parallel(
    plan: Path,
    output: Path,
    *,
    jpeg_quality: int,
    max_workers: int,
    archive: bool = False,
    resume: bool = False,
) -> dict:
    """Materialize independent clip packages with bounded worker concurrency.

    Args:
        plan: Completed plan-batch directory.
        output: Package root containing one directory per clip.
        jpeg_quality: Explicit JPEG encoding quality.
        max_workers: Number of concurrent clip workers.
        archive: Create a tar beside each completed package.
        resume: Reuse package receipts and append only unfinished clips.
    Returns:
        Materialization summary written after every planned clip has a result.
    Raises:
        ValueError: If the plan, worker count, or existing partial package state is invalid.
        OSError: If a worker cannot decode or write its assigned clip.
    """
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    summary = json.loads((plan / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "planned":
        raise ValueError("Batch materialization requires a completed corpus plan")
    if output.exists() and not resume:
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    if not output.exists():
        output.mkdir(parents=True)
    summary_path = output / "summary.json"
    if resume and summary_path.is_file():
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        if result.get("status") == "materialized":
            return result

    plan_rows = list(read_jsonl(plan / "clips.jsonl"))
    progress_path = output / "clips.jsonl"
    records = {}
    def record_key(record: dict) -> str:
        return str(record.get("package_directory") or f"excluded:{record['clip_id']}")

    if progress_path.is_file():
        for record in read_jsonl(progress_path):
            records[record_key(record)] = record
    else:
        progress_path.touch()

    def append_record(record: dict) -> None:
        with progress_path.open("a", encoding="utf-8") as progress:
            progress.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.flush()
        records[record_key(record)] = record

    tasks = []
    for row in plan_rows:
        if row.get("status") == "excluded":
            if row["clip_id"] not in {record.get("clip_id") for record in records.values()}:
                append_record({"clip_id": row["clip_id"], "status": "excluded", "reason": row["reason"]})
            continue
        directory = row["plan_directory"]
        package = output / directory
        receipt_path = package / "receipt.json"
        if directory in records:
            continue
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            append_record({"clip_id": row["clip_id"], "package_directory": directory, **receipt})
            continue
        if package.exists():
            if resume:
                package.rename(Path(str(package) + ".partial"))
            else:
                raise ValueError(f"{package}: partial package has no completion receipt")
        tasks.append(row)

    def materialize_one(row: dict) -> dict:
        directory = row["plan_directory"]
        package = output / directory
        try:
            receipt = materialize_clip_plan(plan / directory, package, jpeg_quality=jpeg_quality)
        except MediaDecodeError as exc:
            # 源视频损坏是数据质量结果；保留部分输出并继续其他独立 clip。
            corrupt = Path(str(package) + ".corrupt")
            if corrupt.exists():
                raise FileExistsError(f"{corrupt}: corrupt package quarantine already exists") from exc
            if package.exists():
                package.rename(corrupt)
            return {"clip_id": row["clip_id"], "package_directory": directory,
                    "status": "excluded", "reason": "corrupt_video",
                    "video": row.get("video"), "error": str(exc)}
        result = {"clip_id": row["clip_id"], "package_directory": directory, **receipt}
        if archive:
            result["archive"] = archive_history_package(package, output / f"{directory}.tar")
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(materialize_one, row) for row in tasks]
        for future in as_completed(futures):
            append_record(future.result())

    expected = {row["plan_directory"] for row in plan_rows if row.get("status") != "excluded"}
    actual = {key for key in records if key in expected}
    if actual != expected:
        raise RuntimeError(f"materialization result does not cover plan: missing={sorted(expected - actual)}")

    counts = Counter()
    archive_bytes = 0
    for row in plan_rows:
        if row.get("status") == "excluded":
            counts["excluded_clips"] += 1
            continue
        record = records[row["plan_directory"]]
        if record.get("status") == "excluded":
            counts["excluded_clips"] += 1
            counts[f"excluded:{record['reason']}"] += 1
            continue
        counts["materialized_clips"] += 1
        counts.update({key: record[key] for key in ("samples", "unique_fpv", "unique_bev", "media_bytes")})
        if "archive" in record:
            archive_bytes += record["archive"]["archive_bytes"]
    if archive_bytes:
        counts["archive_bytes"] = archive_bytes
    result = {"status": "materialized", "plan": str(plan.resolve()), **dict(counts)}
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
