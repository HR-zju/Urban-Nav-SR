"""File orchestration for clip-sized history plans and portable media shards."""

import json
import shutil
import sqlite3
import tarfile
from contextlib import closing
from pathlib import Path
from typing import Iterable

from mcp_server.data.history_builder import build_clip_observations
from mcp_server.data.history_media import extract_history_frames
from mcp_server.data.history_review import write_history_review
from mcp_server.data.stage_ii.sources import stage_ii_source_row_identity_v1


def read_jsonl(path: Path) -> Iterable[dict]:
    """Read manifest rows, surfacing parse errors with file and line context.

    Args:
        path: UTF-8 JSONL input.
    Returns:
        Iterator of parsed rows in source order.
    Raises:
        ValueError: If a row has invalid JSON.
        OSError: If the file cannot be read.
    """
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: {error}") from error


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def index_history_manifests(*, manifests: list[Path], sidecar_paths: list[Path], spec, output: Path) -> dict:
    """Index original manifests once for bounded-memory, clip-local planning.

    Args:
        manifests: Ordered source JSONL files for one dataset.
        sidecar_paths: Original accepted trajectory JSONL files.
        spec: Existing canonical dataset identity specification.
        output: New SQLite file; a completion record is committed last.
    Returns:
        Source/linked-sidecar counts per clip and exact input locations.
    Raises:
        OSError: If an input is unreadable or the output already exists.
        ValueError: If JSON or canonical identity fields are invalid.
        sqlite3.IntegrityError: If a source occurrence or linked GT repeats.
    """
    # SQLite 仅用于一次索引、多 clip 查询，避免每片扫描全部 GT 或将全量 JSON 常驻内存。
    output.touch(exist_ok=False)
    with closing(sqlite3.connect(output)) as db, db:
        db.executescript('''
            CREATE TABLE sources (ordinal INTEGER PRIMARY KEY, parent TEXT UNIQUE, clip TEXT, payload TEXT,
                                  manifest TEXT, line INTEGER);
            CREATE INDEX source_clip ON sources(clip, ordinal);
            CREATE TABLE variants (parent TEXT, manifest TEXT, line INTEGER, payload TEXT,
                                   UNIQUE(manifest, line));
            CREATE INDEX variant_parent ON variants(parent);
            CREATE TABLE sidecars (parent TEXT PRIMARY KEY, payload TEXT);
            CREATE TABLE completion (payload TEXT);
        ''')
        ordinal = 0
        for path in manifests:
            for line, row in enumerate(read_jsonl(path), 1):
                parent, _ = stage_ii_source_row_identity_v1(row, spec)
                meta = row if spec.dataset_source == "egowalk" else row["sidewalk_meta"]
                payload = json.dumps(row, ensure_ascii=False)
                manifest = str(path.resolve())
                previous = db.execute("SELECT payload, manifest, line FROM sources WHERE parent=?",
                                      (parent,)).fetchone()
                if previous is None:
                    db.execute("INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)",
                               (ordinal, parent, meta["clip_id"], payload, manifest, line))
                else:
                    old = json.loads(previous[0])
                    old_meta = old if spec.dataset_source == "egowalk" else old["sidewalk_meta"]
                    old_messages = [m for m in old.get("messages", []) if m["role"] != "assistant"]
                    messages = [m for m in row.get("messages", []) if m["role"] != "assistant"]
                    if (old_meta["timestamp"] != meta["timestamp"] or old_meta["clip_id"] != meta["clip_id"]
                            or (spec.dataset_source == "egowalk" and old_meta["frame_idx"] != meta["frame_idx"])
                            or old.get("images") != row.get("images") or old_messages != messages
                            or old.get("instruction") != row.get("instruction")):
                        raise ValueError(f"{path}:{line}: conflicting observation for parent {parent}")
                    if not db.execute("SELECT 1 FROM variants WHERE parent=?", (parent,)).fetchone():
                        db.execute("INSERT INTO variants VALUES (?, ?, ?, ?)",
                                   (parent, previous[1], previous[2], previous[0]))
                    db.execute("INSERT INTO variants VALUES (?, ?, ?, ?)", (parent, manifest, line, payload))
                ordinal += 1
        for path in sidecar_paths:
            for row in read_jsonl(path):
                parent = row["sample_id"]
                if db.execute("SELECT 1 FROM sources WHERE parent=?", (parent,)).fetchone():
                    db.execute("INSERT INTO sidecars VALUES (?, ?)", (parent, json.dumps(row, ensure_ascii=False)))
                    variants = db.execute("SELECT manifest, line, payload FROM variants WHERE parent=?",
                                          (parent,)).fetchall()
                    if variants:
                        matches = [v for v in variants if v[1] == row["source_line"]
                                   and f"{spec.dataset_source}/{Path(v[0]).name}" == row["source_manifest_name"]]
                        if len(matches) != 1:
                            raise ValueError(f"{parent}: GT provenance must identify one source occurrence")
                        # 只按原 GT 来源绑定 source；所有语义答案版本另存待审，不产生重复训练 parent。
                        manifest, line, payload = matches[0]
                        db.execute("UPDATE sources SET payload=?, manifest=?, line=? WHERE parent=?",
                                   (payload, manifest, line, parent))
        clips = [{"clip_id": clip, "source_rows": count, "linked_sidecars": linked}
                 for clip, count, linked in db.execute('''
                     SELECT clip, count(*), count(sidecars.parent) FROM sources
                     LEFT JOIN sidecars USING(parent) GROUP BY clip ORDER BY min(ordinal)
                 ''')]
        summary = {
            "schema_version": "history_source_index_v2",
            "dataset_source": spec.dataset_source, "source_rows": ordinal, "clips": clips,
            "unique_parents": sum(clip["source_rows"] for clip in clips),
            "multiple_annotation_parents": db.execute("SELECT count(DISTINCT parent) FROM variants").fetchone()[0],
            "source_manifests": [str(path.resolve()) for path in manifests],
            "sidecar_paths": [str(path.resolve()) for path in sidecar_paths],
            "status": "indexed", "observation_eligibility": "pending",
        }
        db.execute("INSERT INTO completion VALUES (?)", (json.dumps(summary, ensure_ascii=False),))
    return summary


def _read_indexed_clip(path: Path, *, clip_id: str, dataset_source: str, max_samples: int | None):
    """Read one completed index without creating or mutating the SQLite file."""
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        completed = db.execute("SELECT payload FROM completion").fetchone()
        if completed is None:
            raise ValueError(f"{path}: history source index is incomplete")
        summary = json.loads(completed[0])
        if summary.get("schema_version") != "history_source_index_v2":
            raise ValueError(f"{path}: rebuild index with source-occurrence provenance")
        if summary["dataset_source"] != dataset_source:
            raise ValueError(f"{path}: history index dataset mismatch")
        records = db.execute('''
            SELECT sources.parent, sources.payload, sidecars.payload FROM sources LEFT JOIN sidecars USING(parent)
            WHERE clip=? ORDER BY ordinal LIMIT ?
        ''', (clip_id, max_samples if max_samples is not None else -1))
        rows, sidecars, variants = [], {}, {}
        for parent, source_json, sidecar_json in records:
            rows.append(json.loads(source_json))
            occurrences = [{"source_manifest": manifest, "source_line": line, "source_row": json.loads(payload)}
                           for manifest, line, payload in db.execute(
                               "SELECT manifest, line, payload FROM variants WHERE parent=? ORDER BY rowid", (parent,))]
            if occurrences:
                variants[parent] = occurrences
            if sidecar_json is not None:
                sidecar = json.loads(sidecar_json)
                sidecars[sidecar["sample_id"]] = sidecar
    return rows, sidecars, summary, variants


def write_clip_plan(
    *, manifests: list[Path], sidecar_paths: list[Path], clip_id: str,
    source: Path, video: Path, processor, spec, registry, bev,
    output: Path, max_samples: int | None, source_index: Path | None = None,
) -> dict:
    """Join existing source and sidecar files into one auditable clip plan.

    Args:
        manifests: Original source manifests, scanned in declared order.
        sidecar_paths: Original accepted GT sidecars; only selected parents load.
        clip_id: Exact clip to select; original anchors stay unchanged.
        source: Dataset-specific source timeline path.
        video: Explicit source video, used only during materialization.
        processor: Dataset history processor.
        spec: Frozen dataset source specification.
        registry: Existing identity/split registry.
        bev: Rebuilt BEV index, or None for EgoWalk.
        output: New plan directory.
        max_samples: Optional positive source-row limit for a representative run.
        source_index: Completed dataset index instead of rescanning manifests.
    Returns:
        Plan counts and local source locations, also saved as summary.json.
    Raises:
        ValueError: If no anchors match, sidecars repeat or sample limit is invalid.
        OSError: If input reading or output creation fails.
    """
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    variants = {}
    if source_index is not None:
        if manifests or sidecar_paths:
            raise ValueError("Use either source_index or original manifest/sidecar paths")
        rows, sidecars, indexed_summary, variants = _read_indexed_clip(
            source_index, clip_id=clip_id, dataset_source=spec.dataset_source, max_samples=max_samples)
        manifests = [Path(path) for path in indexed_summary["source_manifests"]]
        sidecar_paths = [Path(path) for path in indexed_summary["sidecar_paths"]]
    else:
        rows, sidecars = _read_clip_manifests(manifests, sidecar_paths, clip_id, spec, max_samples)
    if not rows:
        raise ValueError(f"No source anchors for clip {clip_id}")
    accepted, rejected, summary = build_clip_observations(
        rows, source=source, processor=processor, spec=spec, registry=registry,
        sidecars=sidecars, bev=bev)
    for record in accepted:
        parent = record["source_parent_id"]
        if parent in variants:
            record["source_variants"] = variants[parent]
            record["semantic_label_selection"] = "pending"
    summary["multiple_annotation_parents"] = sum(record["source_parent_id"] in variants for record in accepted)
    summary.update({
        "dataset_source": spec.dataset_source, "clip_id": clip_id,
        "source": str(source.resolve()), "video": str(video.resolve()),
        "source_manifests": [str(path.resolve()) for path in manifests],
        "sidecar_paths": [str(path.resolve()) for path in sidecar_paths],
        "bev_index": str(bev.path.resolve()) if bev else None,
        "max_lag_ns": processor.max_lag_ns,
        "max_bev_age_ns": bev.max_age_ns if bev else None,
    })
    if source_index is not None:
        summary["source_index"] = str(source_index.resolve())
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "observation_plan.jsonl", accepted)
    _write_jsonl(output / "rejected.jsonl", rejected)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _read_clip_manifests(manifests, sidecar_paths, clip_id, spec, max_samples):
    """Retain direct single-clip planning for small, explicitly selected runs."""
    rows = []
    for manifest in manifests:
        for row in read_jsonl(manifest):
            meta = row if spec.dataset_source == "egowalk" else row["sidewalk_meta"]
            if meta["clip_id"] == clip_id:
                rows.append(row)
                if max_samples is not None and len(rows) >= max_samples:
                    break
        if max_samples is not None and len(rows) >= max_samples:
            break
    if not rows:
        raise ValueError(f"No source anchors for clip {clip_id}")
    parents = {stage_ii_source_row_identity_v1(row, spec)[0] for row in rows}
    sidecars = {}
    for path in sidecar_paths:
        for sidecar in read_jsonl(path):
            parent = sidecar["sample_id"]
            if parent in parents:
                if parent in sidecars:
                    raise ValueError(f"{path}: duplicate sidecar parent {parent}")
                sidecars[parent] = sidecar
    return rows, sidecars


def materialize_clip_plan(plan: Path, output: Path, *, jpeg_quality: int) -> dict:
    """Extract a clip shard with portable observation paths and preserved GT.

    Args:
        plan: Directory written by write_clip_plan.
        output: New media shard directory; partial shards remain for diagnosis.
        jpeg_quality: Explicit JPEG quality used by the sequential extractor.
    Returns:
        Completion receipt containing actual counts and encoded media bytes.
    Raises:
        OSError: If input/output, decoding or media copying fails.
        ValueError: If serialized JSON or extraction parameters are invalid.
    """
    summary = json.loads((plan / "summary.json").read_text(encoding="utf-8"))
    rows = list(read_jsonl(plan / "observation_plan.jsonl"))
    ordinals = {i for row in rows for i in row["observation"]["video_ordinals"]}
    output.mkdir(parents=True, exist_ok=False)
    frames = extract_history_frames(Path(summary["video"]), ordinals,
                                    output / "media/fpv", jpeg_quality=jpeg_quality)
    bev_paths = {}
    for row in rows:
        observation = row["observation"]
        observation["fpv_paths"] = [frames[i].relative_to(output).as_posix()
                                    for i in observation["video_ordinals"]]
        if observation["bev"] is not None:
            old_path = observation["bev"]["bev_image_path"]
            if old_path not in bev_paths:
                # 每片只包含一个 clip；源 frame_idx 决定稳定文件名，保留实际 BEV 时刻。
                relative = Path("media/bev") / f"{observation['bev']['frame_idx']:06d}.jpg"
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(summary["bev_index"]).parent / old_path, target)
                bev_paths[old_path] = relative.as_posix()
            observation["bev"]["bev_image_path"] = bev_paths[old_path]
    (output / "manifests").mkdir()
    # 原 source_row 仅用于追溯；消费者读取 observation 的相对路径，而非旧图像列表。
    _write_jsonl(output / "manifests/observations.jsonl", rows)
    review_path = write_history_review(rows, output)
    receipt = {
        "status": "materialized", "samples": len(rows), "unique_fpv": len(frames),
        "unique_bev": len(bev_paths), "jpeg_quality": jpeg_quality,
        "media_bytes": sum(path.stat().st_size for path in frames.values())
        + sum((output / path).stat().st_size for path in bev_paths.values()),
        "consumer_validation": "pending", "semantic_label_review": "pending",
        "review": review_path.relative_to(output).as_posix(),
    }
    # 完成记录最后写入；存在散图只能说明部分输出，不能代替片级完成证据。
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def archive_history_package(package: Path, destination: Path) -> dict:
    """Archive one completed clip shard, preserving portable media references.

    Args:
        package: Materialized shard with manifests, media, review and receipt.
        destination: New uncompressed .tar path, normally on the transfer disk.
    Returns:
        Member count and actual archive size; also saved next to the archive.
    Raises:
        ValueError: If the shard is incomplete, paths escape it, or media is missing.
        OSError: If reading or exclusive archive/receipt creation fails.
    """
    root = package.resolve()
    receipt = json.loads((root / 'receipt.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'materialized':
        raise ValueError(f'{root}: materialized receipt required')
    files = {root / 'receipt.json', root / 'manifests/observations.jsonl', root / 'review/index.html'}
    samples = 0
    for row in read_jsonl(root / 'manifests/observations.jsonl'):
        samples += 1
        observation = row['observation']
        paths = list(observation['fpv_paths'])
        if observation['bev'] is not None:
            paths.append(observation['bev']['bev_image_path'])
        for relative in paths:
            path = root / relative
            if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink():
                raise ValueError(f'{root}: archive requires package-owned regular media: {relative}')
            if not path.resolve().is_relative_to(root):
                raise ValueError(f'{root}: media resolves outside package: {relative}')
            files.add(path)
    if samples != receipt['samples']:
        raise ValueError(f'{root}: sample count differs from materialized receipt')
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'{root}: package file missing or not regular: {path}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # JPEG 已压缩；tar 聚合小文件，避免 exFAT 大 allocation unit 的散图空间浪费。
    with tarfile.open(destination, 'x') as archive:
        for path in sorted(files):
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    result = {'status': 'archived', 'samples': samples, 'members': len(files),
              'archive_bytes': destination.stat().st_size, 'archive': str(destination.resolve())}
    with Path(str(destination) + '.receipt.json').open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
    return result
