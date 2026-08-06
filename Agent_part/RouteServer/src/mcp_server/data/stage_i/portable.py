"""Relocate canonical Stage I exchange views into portable artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.contracts.stage_i_sft import (
    parse_stage_i_training_view_v1,
    stage_i_projection_digest_v1,
)
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic


STAGE_I_PORTABLE_RELOCATION_SCHEMA_VERSION = (
    "urbannav_stage_i_portable_relocation_v1"
)


def relocate_stage_i_training_views_v1(
    input_jsonl: str | Path,
    output_root: str | Path,
    *,
    output_name: str | None = None,
    root_image_dir: str | Path | None = None,
    source_prefix_maps: Sequence[tuple[str | Path, str | Path]] = (),
) -> dict[str, Any]:
    """Copy media and rewrite canonical Stage I views as artifact-relative rows.

    Every input row is parsed before relocation. Image bytes are copied under a
    content-addressed ``media/`` path, ``images`` is rewritten relative to the
    output artifact root, and ``projection_digest`` is recomputed before the
    relocated row is parsed again. A stale absolute source path may be repaired
    only by an explicit anchored prefix map; basename searching is prohibited.

    Args:
        input_jsonl: Canonical Stage I exchange-view JSONL.
        output_root: Destination artifact root containing ``manifests/`` and
            ``media/``.
        output_name: Optional output manifest filename. Defaults to the input
            filename and must end in ``.jsonl``.
        root_image_dir: Optional root for resolving relative input paths.
        source_prefix_maps: Ordered ``(old_prefix, new_prefix)`` mappings used
            only when the recorded source path does not exist.

    Returns:
        JSON-compatible relocation report including artifact paths, counts, and
        SHA-256 identities. ``root_image_dir_for_consumers`` is the exact root
        that downstream loaders must use for the relative image paths.

    Raises:
        FileNotFoundError: If an image cannot be resolved without guessing.
        ValueError: If a row, output name, or prefix mapping is invalid.
    """

    input_path = Path(input_jsonl).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    manifest_name = output_name or input_path.name
    if Path(manifest_name).name != manifest_name or not manifest_name.endswith(
        ".jsonl"
    ):
        raise ValueError("output_name must be one .jsonl filename")
    image_root = (
        Path(root_image_dir).expanduser().resolve()
        if root_image_dir is not None
        else None
    )
    prefix_maps = _normalize_prefix_maps(source_prefix_maps)
    output_path = destination / "manifests" / manifest_name
    relocated_rows: list[dict[str, Any]] = []
    source_files: set[Path] = set()
    media_inventory: dict[str, dict[str, Any]] = {}
    image_references = 0
    prefix_repaired_references = 0

    for line_number, raw_row in enumerate(iter_jsonl(input_path), start=1):
        try:
            parsed = parse_stage_i_training_view_v1(raw_row)
        except Exception as exc:
            raise ValueError(
                f"invalid canonical Stage I row at {input_path}:{line_number}"
            ) from exc
        payload = parsed.model_dump(mode="json", exclude_none=True)
        relocated_images: list[str] = []
        for image_value in payload["images"]:
            source, prefix_repaired = _resolve_image_path(
                str(image_value),
                input_manifest_dir=input_path.parent,
                root_image_dir=image_root,
                source_prefix_maps=prefix_maps,
            )
            source_files.add(source)
            image_references += 1
            prefix_repaired_references += int(prefix_repaired)
            content_digest = _sha256_file(source)
            suffix = source.suffix.lower() or ".bin"
            relative_path = Path("media") / content_digest[:2] / (
                content_digest + suffix
            )
            target = destination / relative_path
            _copy_verified(source, target, expected_digest=content_digest)
            relocated_images.append(relative_path.as_posix())
            media_inventory[relative_path.as_posix()] = {
                "path": relative_path.as_posix(),
                "sha256": content_digest,
                "size_bytes": source.stat().st_size,
            }
        payload["images"] = relocated_images
        payload.pop("projection_digest", None)
        payload["projection_digest"] = stage_i_projection_digest_v1(payload)
        relocated = parse_stage_i_training_view_v1(payload)
        relocated_rows.append(
            relocated.model_dump(mode="json", exclude_none=True)
        )

    if not relocated_rows:
        raise ValueError("Stage I portable relocation requires at least one row")
    write_jsonl_atomic(output_path, relocated_rows)
    inventory = sorted(media_inventory.values(), key=lambda item: item["path"])
    inventory_digest = _json_digest(inventory)
    report = {
        "schema_version": STAGE_I_PORTABLE_RELOCATION_SCHEMA_VERSION,
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": _sha256_file(input_path),
        "output_root": str(destination),
        "output_jsonl": str(output_path),
        "output_jsonl_sha256": _sha256_file(output_path),
        "root_image_dir_for_consumers": str(destination),
        "path_mode": "artifact_root_relative",
        "row_count": len(relocated_rows),
        "image_reference_count": image_references,
        "unique_source_file_count": len(source_files),
        "unique_media_blob_count": len(inventory),
        "prefix_repaired_reference_count": prefix_repaired_references,
        "source_prefix_map_digest": _json_digest(
            [[str(old), str(new)] for old, new in prefix_maps]
        ),
        "media_inventory_digest": inventory_digest,
    }
    report_path = destination / "manifests" / (
        f"{Path(manifest_name).stem}.relocation.json"
    )
    report["report_json"] = str(report_path)
    _write_json_atomic(report_path, report)
    return report


def _normalize_prefix_maps(
    mappings: Sequence[tuple[str | Path, str | Path]],
) -> list[tuple[Path, Path]]:
    """Normalize and reject ambiguous duplicate relocation prefixes."""

    normalized: list[tuple[Path, Path]] = []
    old_prefixes: set[Path] = set()
    for old_value, new_value in mappings:
        old_prefix = Path(old_value).expanduser()
        new_prefix = Path(new_value).expanduser().resolve()
        if not old_prefix.is_absolute():
            raise ValueError("source prefix-map keys must be absolute paths")
        if old_prefix in old_prefixes:
            raise ValueError(f"duplicate source prefix-map key: {old_prefix}")
        old_prefixes.add(old_prefix)
        normalized.append((old_prefix, new_prefix))
    return sorted(normalized, key=lambda item: len(item[0].parts), reverse=True)


def _resolve_image_path(
    image_value: str,
    *,
    input_manifest_dir: Path,
    root_image_dir: Path | None,
    source_prefix_maps: Sequence[tuple[Path, Path]],
) -> tuple[Path, bool]:
    """Resolve an image with explicit roots or an anchored prefix rewrite."""

    recorded = Path(image_value).expanduser()
    candidates: list[tuple[Path, bool]] = []
    if recorded.is_absolute():
        candidates.append((recorded, False))
        for old_prefix, new_prefix in source_prefix_maps:
            try:
                relative = recorded.relative_to(old_prefix)
            except ValueError:
                continue
            candidates.append((new_prefix / relative, True))
    else:
        if root_image_dir is not None:
            candidates.append((root_image_dir / recorded, False))
        candidates.append((input_manifest_dir / recorded, False))
    for candidate, repaired in candidates:
        if candidate.is_file():
            return candidate.resolve(), repaired
    rendered = ", ".join(str(candidate) for candidate, _ in candidates)
    raise FileNotFoundError(
        f"cannot resolve Stage I image path {image_value!r}; checked: {rendered}"
    )


def _copy_verified(source: Path, target: Path, *, expected_digest: str) -> None:
    """Copy one content-addressed file and verify any existing target."""

    if target.is_file():
        if _sha256_file(target) != expected_digest:
            raise ValueError(f"portable media digest collision: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copy2(source, temporary)
    if _sha256_file(temporary) != expected_digest:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"portable media copy verification failed: {source}")
    temporary.replace(target)


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    """Return canonical SHA-256 for one JSON-compatible value."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one formatted JSON object through an adjacent temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "STAGE_I_PORTABLE_RELOCATION_SCHEMA_VERSION",
    "relocate_stage_i_training_views_v1",
]
