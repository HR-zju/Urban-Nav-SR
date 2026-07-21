#!/usr/bin/env python3
"""Prepare and run four resumable Qwen reasoning-label workers."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


DEFAULT_WORKSPACE = Path("/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation")
DEFAULT_DATA = Path("/lpai/volumes/mind-eb-bd-ga/suh/dataset/urbannav-sidewalk-agent-data-v1")
DEFAULT_PYTHON = Path("/lpai/volumes/mind-eb-bd-ga/suh/miniconda3/envs/qwen35_vllm_cuda128/bin/python")
DATASETS = (
    "sidewalk_main",
    "sidewalk_extra",
    "frodobots",
    "foxglove",
    "egowalk_part_a",
    "egowalk_part_b",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse prepare/worker arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker"))
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--shard-index", type=int, choices=range(4), default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-fpv-frames", type=int, default=5)
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="low")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare normalized sources or run one teacher/export worker."""

    args = parse_args(argv)
    workspace = args.workspace_root.resolve()
    source_run = (
        args.source_run_root.resolve()
        if args.source_run_root
        else workspace / "runs/qwen35_122b_a10b_20260716/behavior_filter"
    )
    run_root = (
        args.run_root.resolve()
        if args.run_root
        else workspace / "runs/reasoning_v2_qwen122_interesting_multiframe_20260719"
    )
    code_root = workspace / "Urban-Nav-SR/Agent_part/RouteServer"
    ms_swift_root = workspace / "ms-swift"
    for name in ("sources", "shards", "annotations", "views", "reports", "logs"):
        (run_root / name).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    pythonpath = f"{code_root / 'src'}:{ms_swift_root}"
    if env.get("PYTHONPATH"):
        pythonpath += f":{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    env["NO_PROXY"] = _append_no_proxy(env.get("NO_PROXY"))
    env["no_proxy"] = env["NO_PROXY"]

    if args.action == "prepare":
        prepare_sources(
            python_bin=args.python_bin,
            code_root=code_root,
            data_root=args.data_root,
            source_run_root=source_run,
            run_root=run_root,
            max_fpv_frames=max(1, args.max_fpv_frames),
            env=env,
        )
        return

    if args.shard_index is None or not args.api_base or not args.model:
        raise SystemExit("worker requires --shard-index, --api-base, and --model")
    run_worker(
        python_bin=args.python_bin,
        code_root=code_root,
        data_root=args.data_root,
        run_root=run_root,
        shard_index=args.shard_index,
        api_base=args.api_base,
        model=args.model,
        concurrency=max(1, args.concurrency),
        max_retries=max(0, args.max_retries),
        image_detail=args.image_detail,
        env=env,
    )


def prepare_sources(
    *,
    python_bin: Path,
    code_root: Path,
    data_root: Path,
    source_run_root: Path,
    run_root: Path,
    max_fpv_frames: int,
    env: dict[str, str],
) -> None:
    """Normalize six interesting manifests and split them into four line shards."""

    source_paths = []
    for dataset in DATASETS:
        input_path = source_run_root / dataset / "interesting.jsonl"
        output_path = run_root / "sources" / f"{dataset}.jsonl"
        report_path = run_root / "reports" / f"source_{dataset}.json"
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if not (output_path.is_file() and output_path.stat().st_size and report_path.is_file()):
            _run(
                [
                    python_bin,
                    code_root / "tools/model_processing/build_reasoning_v2_sources.py",
                    "--input-jsonl",
                    input_path,
                    "--output-jsonl",
                    output_path,
                    "--report-json",
                    report_path,
                    "--dataset-source",
                    dataset,
                    "--max-fpv-frames",
                    str(max_fpv_frames),
                ],
                cwd=data_root,
                env=env,
            )
        _assert_exact_fpv_frames(output_path, expected=max_fpv_frames)
        source_paths.append(output_path)

    merged = run_root / "sources/interesting_all.multiframe_fpv_current_bev.jsonl"
    temp = merged.with_suffix(f"{merged.suffix}.tmp.{os.getpid()}")
    with temp.open("wb") as target:
        for source in source_paths:
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, target, length=8 * 1024 * 1024)
    os.replace(temp, merged)
    _assert_unique_source_group_ids(merged)

    total_rows = _line_count(merged)
    chunk_rows = int(math.ceil(total_rows / 4.0))
    shard_paths = [run_root / "shards" / f"shard_{index}.jsonl" for index in range(4)]
    handles = [path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}").open("wb") for path in shard_paths]
    try:
        with merged.open("rb") as source:
            for row_index, line in enumerate(source):
                shard_index = min(row_index // chunk_rows, 3)
                handles[shard_index].write(line)
    finally:
        for handle in handles:
            handle.close()
    for shard in shard_paths:
        os.replace(shard.with_suffix(f"{shard.suffix}.tmp.{os.getpid()}"), shard)
    counts = [_line_count(path) for path in shard_paths]
    if sum(counts) != total_rows or any(count <= 0 for count in counts):
        raise RuntimeError(f"invalid shard split total={total_rows} counts={counts}")
    (run_root / ".prepared").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    print({"total_rows": total_rows, "shard_rows": counts})


def _assert_exact_fpv_frames(path: Path, *, expected: int) -> None:
    """Reject a nominal multiframe source containing incomplete FPV windows."""

    invalid: list[tuple[int, int, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            roles = row.get("image_roles") or []
            actual = sum(str(role).startswith("fpv.") for role in roles)
            if actual != expected:
                invalid.append((line_number, actual, str(row.get("source_record_id") or "")))
                if len(invalid) >= 20:
                    break
    if invalid:
        raise RuntimeError(
            f"strict multiframe QC failed path={path} expected_fpv_frames={expected} "
            f"examples={invalid}"
        )


def _assert_unique_source_group_ids(path: Path) -> None:
    """Reject merged sources that would make resume and source joins ambiguous."""

    seen: set[str] = set()
    duplicates: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            source_id = str(json.loads(line).get("source_group_id") or "")
            if not source_id or source_id in seen:
                duplicates.append((line_number, source_id))
                if len(duplicates) >= 20:
                    break
            seen.add(source_id)
    if duplicates:
        raise RuntimeError(f"duplicate source_group_id path={path} examples={duplicates}")


def run_worker(
    *,
    python_bin: Path,
    code_root: Path,
    data_root: Path,
    run_root: Path,
    shard_index: int,
    api_base: str,
    model: str,
    concurrency: int,
    max_retries: int,
    image_detail: str,
    env: dict[str, str],
) -> None:
    """Label one source shard and deterministically export full-CoT plus VQA views."""

    shard = run_root / "shards" / f"shard_{shard_index}.jsonl"
    if not (run_root / ".prepared").is_file() or not shard.is_file():
        raise RuntimeError(f"sources are not prepared: {shard}")
    annotation = run_root / "annotations" / f"shard_{shard_index}.jsonl"
    env.update(
        {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": model,
            "LLM_API_BASE_URL": api_base,
            "LLM_API_STYLE": "chat_completions",
            "LLM_API_KEY": "dummy",
            "LLM_TEMPERATURE": "0",
            "LLM_TIMEOUT_S": "120",
            "LLM_VISION_TIMEOUT_S": "300",
            "LLM_MAX_RETRIES": "1",
            "LLM_CHAT_TEMPLATE_ENABLE_THINKING": "0",
        }
    )
    _run(
        [
            python_bin,
            code_root / "tools/model_processing/label_reasoning_v2.py",
            "--source-jsonl",
            shard,
            "--annotation-jsonl",
            annotation,
            "--report-json",
            run_root / "reports" / f"annotation_shard_{shard_index}.json",
            "--concurrency",
            str(concurrency),
            "--max-retries",
            str(max_retries),
            "--image-detail",
            image_detail,
            "--retry-failed-on-resume",
        ],
        cwd=data_root,
        env=env,
    )
    _run(
        [
            python_bin,
            code_root / "tools/model_processing/export_reasoning_v2_views.py",
            "--source-jsonl",
            shard,
            "--annotation-jsonl",
            annotation,
            "--output-jsonl",
            run_root / "views" / f"shard_{shard_index}.jsonl",
            "--report-json",
            run_root / "reports" / f"views_shard_{shard_index}.json",
            "--minimum-qa-confidence",
            "0.7",
            "--skip-errors",
        ],
        cwd=data_root,
        env=env,
    )
    (run_root / f".worker_{shard_index}_done").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )


def _run(command: Sequence[object], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run([str(item) for item in command], cwd=cwd, env=env, check=True)


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _append_no_proxy(value: str | None) -> str:
    tokens = [item.strip() for item in str(value or "").split(",") if item.strip()]
    for item in ("127.0.0.1", "localhost"):
        if item not in tokens:
            tokens.append(item)
    return ",".join(tokens)


if __name__ == "__main__":
    main()
