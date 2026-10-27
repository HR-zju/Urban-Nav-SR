#!/usr/bin/env python3
"""Upload a portable history view in batches below ModelScope's file limit.

ModelScope rejects a folder when its recursive file count exceeds 100,000.
This uploader keeps the remote dataset layout stable and uploads metadata first,
then groups materialized clip packages by file count.  Temporary staging uses
hard links, so it does not duplicate the media bytes on the same filesystem.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
from pathlib import Path
import shutil
from typing import Any

from modelscope.hub.api import HubApi


DATASETS = ("egowalk", "sidewalk", "foxglove")
MAX_FILES_PER_UPLOAD = 80_000


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the portable view, target repository, and resumable paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--staging-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args(argv)


def _write_json(path: Path, value: Any) -> None:
    """Write a small progress receipt atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read an existing receipt or return an empty resumable state."""

    if not path.is_file():
        return {"control_uploaded": False, "completed_batches": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _link_tree(source: Path, destination: Path) -> int:
    """Hard-link one clip package into a temporary upload staging directory."""

    files = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            raise ValueError(f"unsupported package entry: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(path, target)
        files += 1
    return files


def _materialized_clips(dataset_root: Path) -> list[tuple[str, int]]:
    """Return materialized clip directories and their file counts."""

    result = []
    with (dataset_root / "packages/clips.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") != "materialized":
                continue
            package = dataset_root / "packages" / row["package_directory"]
            count = sum(1 for path in package.rglob("*") if path.is_file())
            if count > MAX_FILES_PER_UPLOAD:
                raise ValueError(f"{package}: {count} files exceed upload batch limit")
            result.append((row["package_directory"], count))
    return result


def _build_batches(dataset_root: Path) -> list[list[str]]:
    """Group clips deterministically without crossing the upload file limit."""

    batches: list[list[str]] = []
    current: list[str] = []
    current_files = 0
    for package_directory, file_count in _materialized_clips(dataset_root):
        if current and current_files + file_count > MAX_FILES_PER_UPLOAD:
            batches.append(current)
            current = []
            current_files = 0
        current.append(package_directory)
        current_files += file_count
    if current:
        batches.append(current)
    return batches


def _prepare_control_view(dataset_root: Path, staging: Path) -> None:
    """Stage README, metadata, manifests, and package indexes."""

    files = [dataset_root / "README.md"]
    files.extend(sorted((dataset_root / "metadata").glob("*")))
    for dataset in DATASETS:
        files.extend(sorted((dataset_root / dataset / "manifests").glob("*.jsonl")))
        files.extend([
            dataset_root / dataset / "packages/clips.jsonl",
            dataset_root / dataset / "packages/summary.json",
        ])
    for source in files:
        if not source.is_file():
            raise FileNotFoundError(source)
        target = staging / source.relative_to(dataset_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)


def _upload_folder(api: HubApi, folder: Path, *, repo_id: str, path_in_repo: str,
                   message: str, max_workers: int, skip_errors: bool = False,
                   ignore_patterns: list[str] | None = None) -> list[dict[str, str]]:
    """Upload a batch; optionally return the SDK's final per-file failures."""

    errors = io.StringIO()
    handler = logging.StreamHandler(errors)
    handler.setLevel(logging.ERROR)
    logger = logging.getLogger("modelscope")
    logger.addHandler(handler)
    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=folder,
            path_in_repo=path_in_repo,
            repo_type="dataset",
            commit_message=message,
            max_workers=max_workers,
            ignore_patterns=ignore_patterns,
        )
    except RuntimeError as exc:
        # 仅跳过 SDK 重试耗尽后明确列出的逐文件失败；其他错误继续抛出。
        count = re.match(r"ERROR - (\d+) file\(s\) failed to upload\.", str(exc))
        failures = [
            {"path": path, "error": error}
            for path, error in re.findall(r"^  - (.+?): (.+)$", errors.getvalue(), re.MULTILINE)
        ]
        if not skip_errors or not count or len(failures) != int(count.group(1)):
            raise
        return failures
    finally:
        logger.removeHandler(handler)
    return []


def upload(args: argparse.Namespace) -> dict[str, Any]:
    """Upload the portable dataset in bounded resumable batches."""

    root = args.dataset_root.resolve()
    staging_root = args.staging_root.resolve()
    receipt_path = args.receipt.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    for dataset in DATASETS:
        if not (root / dataset / "packages/clips.jsonl").is_file():
            raise FileNotFoundError(root / dataset / "packages/clips.jsonl")

    state = _read_json(receipt_path)
    state.update({
        "schema_version": "urbannav_history_modelscope_upload_v1",
        "repo_id": args.repo_id,
        "dataset_root_name": root.name,
        "max_files_per_upload": MAX_FILES_PER_UPLOAD,
        "status": "uploading",
    })
    api = HubApi()
    api.get_cookies(cookies_required=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    if not state.get("control_uploaded"):
        control = staging_root / "control"
        if control.exists():
            raise FileExistsError(f"staging directory exists: {control}")
        _prepare_control_view(root, control)
        _upload_folder(
            api,
            control,
            repo_id=args.repo_id,
            path_in_repo="",
            message="Upload UrbanNav history metadata and manifests",
            max_workers=args.max_workers,
        )
        state["control_uploaded"] = True
        _write_json(receipt_path, state)
        shutil.rmtree(control)

    completed = set(state.get("completed_batches", []))
    batch_count = 0
    for dataset in DATASETS:
        batches = _build_batches(root / dataset)
        batch_count += len(batches)
        for index, package_directories in enumerate(batches):
            batch_id = f"{dataset}:{index:04d}"
            if batch_id in completed:
                continue
            staging = staging_root / f"{dataset}_{index:04d}"
            staging.mkdir(parents=True, exist_ok=True)
            for package_directory in package_directories:
                # package_directory already starts with ``clips/``; stage it
                # directly under the batch root so each clip remains its own
                # directory in the ModelScope upload tree.
                _link_tree(root / dataset / "packages" / package_directory,
                           staging / package_directory)
            print(
                f"uploading {batch_id} ({len(package_directories)} clips) "
                f"to {dataset}/packages",
                flush=True,
            )
            failures = _upload_folder(
                api,
                # SDK 对每个子目录递归统计 50,000 项限制；以 clips 为
                # 上传根，使每个 clip 独立受检，并保持远端路径一致。
                staging / "clips",
                repo_id=args.repo_id,
                path_in_repo=f"{dataset}/packages/clips",
                message=f"Upload {dataset} clip batch {index:04d}",
                max_workers=args.max_workers,
                skip_errors=True,
                ignore_patterns=[
                    item["path"].removeprefix(f"{dataset}/packages/clips/")
                    for item in state.get("skipped_files", [])
                    if item["path"].startswith(f"{dataset}/packages/clips/")
                ],
            )
            state.setdefault("skipped_files", []).extend(
                dict(item, batch_id=batch_id) for item in failures
            )
            completed.add(batch_id)
            state["completed_batches"] = sorted(completed)
            state["batch_count"] = batch_count
            state["uploaded_batch_count"] = len(completed)
            _write_json(receipt_path, state)
            shutil.rmtree(staging)

    state.update({
        "status": "uploaded_with_skips" if state.get("skipped_files") else "uploaded",
        "batch_count": batch_count,
        "uploaded_batch_count": len(completed),
    })
    _write_json(receipt_path, state)
    return state


def main(argv: list[str] | None = None) -> None:
    """Run the resumable ModelScope upload and print its final receipt."""

    result = upload(_parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
