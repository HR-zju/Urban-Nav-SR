#!/usr/bin/env python3
"""Download OSM PBF sources for UrbanNav local benchmark construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


DEFAULT_SOURCE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "urbannav_osm_extract_sources_full.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG, help="OSM source config JSON.")
    parser.add_argument("--output-root", type=Path, default=Path("data/osm"), help="OSM asset root containing raw/ files.")
    parser.add_argument("--retry", type=int, default=5, help="Retry count passed to wget.")
    parser.add_argument("--timeout-s", type=int, default=180, help="Network timeout seconds passed to wget.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing raw files before downloading.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue downloading other sources when one URL fails.")
    return parser.parse_args()


def main() -> None:
    """Download configured OSM raw PBF files.

    Returns:
        None.
    """

    args = parse_args()
    sources = _load_sources(args.source_config)
    raw_root = args.output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    downloader = _select_downloader()
    for filename, url in sorted(sources.items()):
        output = raw_root / filename
        if args.overwrite and output.exists():
            output.unlink()
        if _looks_like_pbf(output):
            print(json.dumps({"status": "skip_existing", "path": str(output), "url": url}, ensure_ascii=False), flush=True)
            continue
        try:
            if downloader == "wget":
                _download_with_wget(url=url, output=output, retry=args.retry, timeout_s=args.timeout_s)
            else:
                _download_with_curl(url=url, output=output, retry=args.retry, timeout_s=args.timeout_s)
            if not _looks_like_pbf(output):
                raise RuntimeError(f"Downloaded file is not a valid-looking OSM PBF: {output}")
        except Exception as exc:
            if output.exists() and not _looks_like_pbf(output):
                output.unlink()
            if not args.continue_on_error:
                raise
            print(
                json.dumps(
                    {"status": "download_failed", "path": str(output), "url": url, "error": str(exc)},
                    ensure_ascii=False,
                ),
                flush=True,
            )


def _load_sources(path: Path) -> dict[str, str]:
    """Load unique raw filenames and URLs from a source config.

    Args:
        path: Source config path.

    Returns:
        Mapping from raw filename to source URL.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    sources: dict[str, str] = {}
    for item in (payload.get("sources") or {}).values():
        filename = str(item["raw_filename"])
        url = str(item["url"])
        previous = sources.get(filename)
        if previous is not None and previous != url:
            raise ValueError(f"Conflicting URL for {filename}: {previous} vs {url}")
        sources[filename] = url
    return sources


def _select_downloader() -> str:
    """Select an available resumable downloader.

    Returns:
        ``wget`` or ``curl``.
    """

    if shutil.which("wget"):
        return "wget"
    if shutil.which("curl"):
        return "curl"
    raise RuntimeError("Either wget or curl is required for resumable OSM downloads.")


def _download_with_wget(*, url: str, output: Path, retry: int, timeout_s: int) -> None:
    """Download one file with wget.

    Args:
        url: Source URL.
        output: Output PBF path.
        retry: Retry count.
        timeout_s: Network timeout in seconds.

    Returns:
        None.
    """

    print(json.dumps({"status": "download", "tool": "wget", "path": str(output), "url": url}, ensure_ascii=False), flush=True)
    cmd = [
        "wget",
        "-c",
        "--tries",
        str(max(1, retry)),
        "--timeout",
        str(max(30, timeout_s)),
        "--read-timeout",
        str(max(30, timeout_s)),
        "-O",
        str(output),
        url,
    ]
    subprocess.run(cmd, check=True)


def _download_with_curl(*, url: str, output: Path, retry: int, timeout_s: int) -> None:
    """Download one file with curl.

    Args:
        url: Source URL.
        output: Output PBF path.
        retry: Retry count.
        timeout_s: Network timeout in seconds.

    Returns:
        None.
    """

    print(json.dumps({"status": "download", "tool": "curl", "path": str(output), "url": url}, ensure_ascii=False), flush=True)
    cmd = [
        "curl",
        "-L",
        "-C",
        "-",
        "--retry",
        str(max(1, retry)),
        "--connect-timeout",
        str(max(30, timeout_s)),
        "--output",
        str(output),
        url,
    ]
    subprocess.run(cmd, check=True)


def _looks_like_pbf(path: Path) -> bool:
    """Check whether a file looks like an OSM PBF.

    Args:
        path: Candidate file path.

    Returns:
        True when the file exists and has an OSM PBF header marker.
    """

    if not path.exists() or path.stat().st_size < 32:
        return False
    with path.open("rb") as handle:
        return b"OSMHeader" in handle.read(128)


if __name__ == "__main__":
    main()
