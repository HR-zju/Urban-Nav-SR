#!/usr/bin/env python3
"""Parallel BEV sampling runner for converted FrodoBots SideWalk clips."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROUTESERVER_ROOT = Path(__file__).resolve().parents[2]
PROCESS_SCRIPT = ROUTESERVER_ROOT / "tools" / "dataset_processing" / "process_sidewalk_dataset.py"


@dataclass(frozen=True)
class Job:
    index: int
    sequence_name: str
    clip_dir: Path
    frame_count: int
    duration_s: float | None


def load_jobs(report_path: Path, *, percent: float, max_clips: int | None) -> list[Job]:
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    selected = report.get("selected") or report.get("converted") or []
    if not isinstance(selected, list) or not selected:
        raise RuntimeError(f"No selected/converted clips found in {report_path}")
    selected = sorted(selected, key=lambda item: int(item.get("frame_count") or 0), reverse=True)
    count = max(1, int(math.ceil(len(selected) * float(percent) / 100.0)))
    if max_clips is not None:
        count = min(count, int(max_clips))
    jobs: list[Job] = []
    for idx, item in enumerate(selected[:count]):
        clip_dir = Path(str(item["clip_dir"])).expanduser()
        jobs.append(
            Job(
                index=idx,
                sequence_name=str(item.get("sequence_name") or clip_dir.name),
                clip_dir=clip_dir,
                frame_count=int(item.get("frame_count") or 0),
                duration_s=(float(item["duration_s"]) if item.get("duration_s") is not None else None),
            )
        )
    return jobs


def bev_index_matches(clip_dir: Path, *, sample_fps: float) -> bool:
    index_path = clip_dir / "bev_index.json"
    if not index_path.is_file():
        return False
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list) or not entries:
        return False
    first = entries[0]
    if not isinstance(first, dict):
        return False
    existing_fps = first.get("bev_sample_fps")
    if existing_fps is not None:
        try:
            if abs(float(existing_fps) - float(sample_fps)) > 1e-6:
                return False
        except Exception:
            return False
    for item in entries[: min(5, len(entries))]:
        raw_path = item.get("bev_image_path") or item.get("path")
        if raw_path and not Path(str(raw_path)).expanduser().is_file():
            return False
    return True


def run_job(job: Job, args: argparse.Namespace, log_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    log_path = log_dir / f"{job.index:06d}_{job.clip_dir.name}.log"
    if bool(args.skip_existing) and bev_index_matches(job.clip_dir, sample_fps=float(args.bev_sample_fps)):
        return {
            "index": job.index,
            "sequence_name": job.sequence_name,
            "clip_dir": str(job.clip_dir),
            "frame_count": job.frame_count,
            "status": "skipped_existing",
            "elapsed_s": 0.0,
            "log_path": str(log_path),
        }

    cmd = [
        sys.executable,
        str(PROCESS_SCRIPT),
        "--sidewalk-root",
        str(job.clip_dir),
        "--bev-sample-fps",
        str(args.bev_sample_fps),
        "--map-provider",
        str(args.map_provider),
    ]
    if bool(args.use_pose_ekf):
        cmd.append("--use-pose-ekf")

    env = os.environ.copy()
    env["BEV_MAP_TILE_OFFLINE"] = "1" if bool(args.offline_tiles) else env.get("BEV_MAP_TILE_OFFLINE", "0")
    log_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("$ " + " ".join(cmd) + "\n")
        log_f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROUTESERVER_ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.monotonic() - started
    return {
        "index": job.index,
        "sequence_name": job.sequence_name,
        "clip_dir": str(job.clip_dir),
        "frame_count": job.frame_count,
        "status": "done" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 3),
        "log_path": str(log_path),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel sample BEV artifacts for FrodoBots SideWalk clips")
    parser.add_argument("--subset-report", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--percent", type=float, default=10.0)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bev-sample-fps", type=float, default=0.5)
    parser.add_argument("--map-provider", default="osm")
    parser.add_argument("--offline-tiles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-pose-ekf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-interval", type=int, default=10)
    args = parser.parse_args()

    subset_report = Path(args.subset_report).expanduser()
    output_root = Path(args.output_root).expanduser()
    log_dir = output_root / "logs"
    report_path = output_root / f"sampling_report_top{float(args.percent):g}pct.json"
    jobs = load_jobs(subset_report, percent=float(args.percent), max_clips=args.max_clips)
    total_frames = sum(job.frame_count for job in jobs)
    estimated_bev_samples = int(round(total_frames / 20.0))
    report: dict[str, Any] = {
        "schema_version": "frodobots_bev_sampling_batch_v1",
        "subset_report": str(subset_report),
        "output_root": str(output_root),
        "percent": float(args.percent),
        "num_workers": int(args.num_workers),
        "bev_sample_fps": float(args.bev_sample_fps),
        "map_provider": str(args.map_provider),
        "offline_tiles": bool(args.offline_tiles),
        "use_pose_ekf": bool(args.use_pose_ekf),
        "skip_existing": bool(args.skip_existing),
        "selected_count": len(jobs),
        "selected_total_frames": int(total_frames),
        "estimated_bev_samples": estimated_bev_samples,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "running",
        "results": [],
    }
    write_report(report_path, report)
    print(
        f"[frodobots-sampling] jobs={len(jobs)} frames={total_frames} "
        f"est_bev={estimated_bev_samples} workers={args.num_workers} report={report_path}",
        flush=True,
    )

    started = time.monotonic()
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(args.num_workers))) as executor:
        futures = [executor.submit(run_job, job, args, log_dir) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            report["results"].append(result)
            completed += 1
            done = sum(1 for item in report["results"] if item.get("status") in {"done", "skipped_existing"})
            errors = sum(1 for item in report["results"] if item.get("status") == "error")
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = completed / elapsed
            remaining = len(jobs) - completed
            report["completed_count"] = completed
            report["done_count"] = done
            report["error_count"] = errors
            report["elapsed_s"] = round(elapsed, 3)
            report["eta_s"] = round(remaining / rate, 3) if rate > 0 else None
            if completed % max(1, int(args.report_interval)) == 0 or completed == len(jobs) or errors:
                write_report(report_path, report)
                eta_h = (float(report["eta_s"]) / 3600.0) if report.get("eta_s") is not None else None
                print(
                    f"[frodobots-sampling] progress {completed}/{len(jobs)} "
                    f"done={done} errors={errors} eta_h={eta_h:.2f}" if eta_h is not None else
                    f"[frodobots-sampling] progress {completed}/{len(jobs)} done={done} errors={errors}",
                    flush=True,
                )

    report["status"] = "done" if int(report.get("error_count", 0)) == 0 else "done_with_errors"
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_report(report_path, report)
    print(json.dumps({k: report.get(k) for k in ("status", "selected_count", "done_count", "error_count", "elapsed_s")}, indent=2))
    print(f"[frodobots-sampling] report: {report_path}")


if __name__ == "__main__":
    main()
