"""Build deterministic ClearML experiment report media manifests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


REPORT_MANIFEST_SCHEMA_V1 = "clearml_experiment_report_manifest_v1"
_MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".mp4", ".webm", ".mov", ".mkv", ".mp3", ".wav", ".ogg", ".html"}
_SECTION_BY_DIRECTORY = {
    "charts": "主要指标",
    "success": "成功案例",
    "successes": "成功案例",
    "failure": "失败案例与人工校阅",
    "failures": "失败案例与人工校阅",
    "review": "失败案例与人工校阅",
    "reviews": "失败案例与人工校阅",
}
_MEDIA_DIRECTORIES = ("charts", "videos", "images", "media")


def build_conventional_report_manifest(
    experiment_root: str | Path,
    *,
    default_section: str | None = None,
) -> dict[str, Any]:
    """Build a report manifest from a convention-based experiment directory.

    Media under charts maps to the metrics section. Other media is classified by
    a path component such as success or failure. An unclassified file
    requires default_section; this makes the semantic fallback explicit.

    Args:
        experiment_root: Directory containing report Markdown and media folders.
        default_section: Explicit section for otherwise unclassified media.

    Returns:
        A JSON-serializable ClearML report manifest.

    Raises:
        FileNotFoundError: If the experiment root is absent.
        ValueError: If media cannot be assigned to a section.
    """

    root = Path(experiment_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if default_section is not None and not default_section.strip():
        raise ValueError("default_section must be non-empty")
    sections: dict[str, list[dict[str, str]]] = {}
    unclassified: list[str] = []
    for directory in _MEDIA_DIRECTORIES:
        media_root = root / directory
        if not media_root.is_dir():
            continue
        for path in sorted(media_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _MEDIA_SUFFIXES:
                continue
            relative = path.relative_to(root).as_posix()
            section = _section_for_path(path.relative_to(root).parts, default_section)
            if section is None:
                unclassified.append(relative)
                continue
            sections.setdefault(section, []).append({
                "title": path.stem.replace("_", " ").replace("-", " ").strip() or path.name,
                "path": relative,
            })
    if unclassified:
        preview = ", ".join(unclassified[:8])
        raise ValueError(
            "Unclassified report media; place it under success/failure/review or pass "
            f"--default-section: {preview}"
        )
    return {
        "schema_version": REPORT_MANIFEST_SCHEMA_V1,
        "sections": {name: sections[name] for name in sorted(sections)},
    }


def write_report_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Atomically write one report manifest without replacing source media."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent), text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return output


def _section_for_path(parts: tuple[str, ...], default_section: str | None) -> str | None:
    """Return the first explicit semantic section in a relative media path."""

    for part in parts[:-1]:
        section = _SECTION_BY_DIRECTORY.get(part.lower())
        if section is not None:
            return section
    if parts and parts[0] == "charts":
        return "主要指标"
    return default_section.strip() if default_section is not None else None


__all__ = ["REPORT_MANIFEST_SCHEMA_V1", "build_conventional_report_manifest", "write_report_manifest"]
