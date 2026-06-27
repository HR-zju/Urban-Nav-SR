"""Urban-side benchmark packaging for SideWalk Agentic Stage2.

This module does not implement model inference.  Instead, it treats ms-swift
benchmark outputs as replay data and packages them into RouteServer review
artifacts: per-case alignment, summary reports, error rows, and artifact
indexes.  The only executable wrapper here is a thin subprocess orchestrator
for the existing ms-swift Stage2 pipeline.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


RUN_NAMES = ("base_zero_shot", "stage2a", "stage2b")
RUN_LABELS = {
    "base_zero_shot": "base",
    "stage2a": "stage2a",
    "stage2b": "stage2b",
}
BENCHMARK_REF_FILES = {
    "decision": "sidewalk_agentic_stage2_decision_refs.jsonl",
    "final": "sidewalk_agentic_stage2_final_refs.jsonl",
}


@dataclass(frozen=True)
class AgenticStage2BenchmarkConfig:
    """Configuration for one Urban-side Agentic Stage2 benchmark package.

    Args:
        run_id: Stable run identifier written to every report.
        heldout_manifest: Canonical Stage2 held-out ms-swift JSONL used by the
            benchmark.  It is recorded for provenance; collection reads the
            prepared reference files under ``ms_swift_output_root``.
        ms_swift_output_root: Output root produced by
            ``run_agentic_stage2_pipeline.sh``.
        report_output_dir: Destination directory for Urban review artifacts.
        base_model: Base model path or id used for benchmark inference.
        stage2a_checkpoint: Optional Stage2-A adapter/checkpoint path.
        stage2b_checkpoint: Optional Stage2-B adapter/checkpoint path.
        ms_swift_root: Optional ms-swift repository root for subprocess runs.
        python_bin: Python executable used by the ms-swift pipeline.
        docker_container: Optional Docker container name.  When set,
            ``run_ms_swift_stage2_pipeline`` executes commands through
            ``docker exec``.
        container_ms_swift_root: ms-swift path inside the Docker container.
        env: Extra environment variables for subprocess pipeline calls.
    """

    run_id: str
    heldout_manifest: str | Path
    ms_swift_output_root: str | Path
    report_output_dir: str | Path
    base_model: str
    stage2a_checkpoint: str | None = None
    stage2b_checkpoint: str | None = None
    ms_swift_root: str | Path | None = None
    python_bin: str = "python"
    docker_container: str | None = None
    container_ms_swift_root: str | Path | None = None
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize benchmark configuration.

        Returns:
            JSON-serializable configuration dictionary.
        """

        return {
            "run_id": self.run_id,
            "heldout_manifest": str(self.heldout_manifest),
            "ms_swift_output_root": str(self.ms_swift_output_root),
            "report_output_dir": str(self.report_output_dir),
            "base_model": self.base_model,
            "stage2a_checkpoint": self.stage2a_checkpoint,
            "stage2b_checkpoint": self.stage2b_checkpoint,
            "ms_swift_root": str(self.ms_swift_root) if self.ms_swift_root else None,
            "python_bin": self.python_bin,
            "docker_container": self.docker_container,
            "container_ms_swift_root": str(self.container_ms_swift_root) if self.container_ms_swift_root else None,
            "env": dict(sorted(self.env.items())),
        }


@dataclass(frozen=True)
class PipelineCommandResult:
    """Result for one ms-swift pipeline subprocess call.

    Args:
        action: Pipeline action name, for example ``benchmark_base``.
        command: Executed command tokens.
        returncode: Process return code.
        stdout_path: File path containing captured stdout.
        stderr_path: File path containing captured stderr.
    """

    action: str
    command: list[str]
    returncode: int
    stdout_path: str
    stderr_path: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize subprocess result.

        Returns:
            JSON-ready result dictionary.
        """

        return {
            "action": self.action,
            "command": list(self.command),
            "returncode": int(self.returncode),
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
        }


def run_ms_swift_stage2_pipeline(
    config: AgenticStage2BenchmarkConfig,
    *,
    actions: Sequence[str],
    dry_run: bool = False,
) -> list[PipelineCommandResult]:
    """Run selected ms-swift Stage2 pipeline actions.

    Args:
        config: Benchmark configuration.
        actions: Ordered actions passed to
            ``examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh``.
        dry_run: If true, write command logs without executing subprocesses.

    Returns:
        List of command results in action order.

    Raises:
        ValueError: If required ms-swift paths are missing.
        subprocess.CalledProcessError: If an executed action exits non-zero.
    """

    if not config.ms_swift_root and not config.container_ms_swift_root:
        raise ValueError("ms_swift_root or container_ms_swift_root is required to run pipeline actions.")
    report_dir = Path(config.report_output_dir)
    logs_dir = report_dir / "pipeline_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results: list[PipelineCommandResult] = []
    for action in actions:
        # Urban 侧只编排 ms-swift 已有 pipeline；stdout/stderr 全量保留，便于复现失败。
        command = _pipeline_command(config, action)
        stdout_path = logs_dir / f"{action}.stdout.log"
        stderr_path = logs_dir / f"{action}.stderr.log"
        if dry_run:
            stdout_path.write_text("DRY_RUN\n" + shlex.join(command) + "\n", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            result = PipelineCommandResult(action, command, 0, str(stdout_path), str(stderr_path))
        else:
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    env=_pipeline_env(config),
                )
            result = PipelineCommandResult(action, command, completed.returncode, str(stdout_path), str(stderr_path))
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, command)
        results.append(result)
    _write_json(
        {
            "run_id": config.run_id,
            "commands": [item.to_dict() for item in results],
        },
        report_dir / "pipeline_commands.json",
    )
    return results


def collect_agentic_stage2_benchmark(config: AgenticStage2BenchmarkConfig) -> dict[str, Any]:
    """Collect ms-swift benchmark outputs into Urban review artifacts.

    Args:
        config: Benchmark configuration pointing at a completed ms-swift output
            root.

    Returns:
        Summary dictionary including written artifact paths and key metrics.

    Raises:
        FileNotFoundError: If required reference files are missing.
        ValueError: If prediction/reference row counts cannot be aligned.
    """

    output_root = Path(config.ms_swift_output_root)
    report_dir = Path(config.report_output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # prepared refs 是 held-out 对齐基准；base/stage2a/stage2b 都按同一行号比较。
    decision_refs = _read_required_jsonl(output_root / "benchmark" / BENCHMARK_REF_FILES["decision"])
    final_refs = _read_required_jsonl(output_root / "benchmark" / BENCHMARK_REF_FILES["final"])
    if len(decision_refs) != len(final_refs):
        raise ValueError(f"decision/final reference mismatch: {len(decision_refs)} != {len(final_refs)}")

    # 三个 run 目录可以缺少 trajectory-head 输出，但不能缺少核心文本 prediction。
    runs = {
        run_name: _load_run_outputs(output_root, run_name, expected_rows=len(final_refs))
        for run_name in RUN_NAMES
    }
    # per_case_report 把同一个 case_id 的 base / Stage2-A / Stage2-B 输出放在一行。
    per_case = [
        _build_case_report(
            config=config,
            index=index,
            decision_ref=decision_ref,
            final_ref=final_ref,
            runs=runs,
            output_root=output_root,
        )
        for index, (decision_ref, final_ref) in enumerate(zip(decision_refs, final_refs))
    ]
    # error_report 是 review 队列：只收集 parse、tool、trajectory 等可操作失败。
    errors = _build_error_rows(per_case)
    # artifact_index 只记录原始文件位置，不复制大文件，避免 benchmark 目录膨胀。
    artifact_index = {
        "schema_version": "sidewalk_agentic_stage2_benchmark_artifact_index_v1",
        "run_id": config.run_id,
        "ms_swift_output_root": str(output_root),
        "report_output_dir": str(report_dir),
        "cases": [item["artifacts"] for item in per_case],
    }
    summary = _build_summary(config, per_case, runs, errors)

    _write_json(config.to_dict(), report_dir / "config.json")
    write_jsonl_records(per_case, report_dir / "per_case_report.jsonl")
    write_jsonl_records(errors, report_dir / "error_report.jsonl")
    _write_json(artifact_index, report_dir / "artifact_index.json")
    _write_json(summary, report_dir / "summary_report.json")
    (report_dir / "summary_report.md").write_text(_summary_markdown(summary), encoding="utf-8")

    return {
        "run_id": config.run_id,
        "num_cases": len(per_case),
        "num_errors": len(errors),
        "report_output_dir": str(report_dir),
        "artifacts": {
            "config": str(report_dir / "config.json"),
            "per_case_report": str(report_dir / "per_case_report.jsonl"),
            "error_report": str(report_dir / "error_report.jsonl"),
            "artifact_index": str(report_dir / "artifact_index.json"),
            "summary_json": str(report_dir / "summary_report.json"),
            "summary_md": str(report_dir / "summary_report.md"),
        },
    }


def write_jsonl_records(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write JSONL records.

    Args:
        records: JSON-serializable records.
        path: Destination JSONL path.

    Returns:
        ``None``.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _pipeline_command(config: AgenticStage2BenchmarkConfig, action: str) -> list[str]:
    script = "examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh"
    if config.docker_container:
        root = str(config.container_ms_swift_root or config.ms_swift_root or "/workspace/ms-swift")
        shell_cmd = f"cd {shlex.quote(root)} && bash {shlex.quote(script)} {shlex.quote(action)}"
        return ["docker", "exec", str(config.docker_container), "bash", "-lc", shell_cmd]
    root = Path(config.ms_swift_root or ".")
    return ["bash", str(root / script), action]


def _pipeline_env(config: AgenticStage2BenchmarkConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "STAGE2_MANIFEST": str(config.heldout_manifest),
            "BASE_MODEL": str(config.base_model),
            "OUTPUT_ROOT": str(config.ms_swift_output_root),
            "PYTHON_BIN": str(config.python_bin),
        }
    )
    if config.stage2a_checkpoint:
        env["STAGE2A_CHECKPOINT"] = str(config.stage2a_checkpoint)
    if config.stage2b_checkpoint:
        env["STAGE2B_CHECKPOINT"] = str(config.stage2b_checkpoint)
    env.update(config.env)
    return env


def _read_required_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return _read_jsonl(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _load_run_outputs(output_root: Path, run_name: str, *, expected_rows: int) -> dict[str, Any]:
    run_dir = output_root / run_name
    # final_head_* 是 Stage2-B 相关可选产物；文本/agentic prediction 是必需对齐项。
    files = {
        "decision_predictions": run_dir / "decision_predictions.jsonl",
        "decision_metrics": run_dir / "decision_agentic_metrics.jsonl",
        "decision_summary": run_dir / "decision_agentic_summary.json",
        "final_predictions": run_dir / "final_text_predictions.jsonl",
        "final_agentic_metrics": run_dir / "final_agentic_metrics.jsonl",
        "final_agentic_summary": run_dir / "final_agentic_summary.json",
        "final_text_traj_metrics": run_dir / "final_text_traj_metrics.jsonl",
        "final_text_traj_summary": run_dir / "final_text_traj_summary.json",
        "final_head_predictions": run_dir / "final_head_predictions.jsonl",
        "final_head_traj_metrics": run_dir / "final_head_traj_metrics.jsonl",
        "final_head_traj_summary": run_dir / "final_head_traj_summary.json",
    }
    rows = {name: _read_jsonl(path) for name, path in files.items() if path.suffix == ".jsonl"}
    summaries = {name: _read_json(path) for name, path in files.items() if path.suffix == ".json"}
    for name in ("decision_predictions", "final_predictions"):
        # 非空 prediction 文件必须与 held-out refs 等长，防止错位比较。
        if rows.get(name) and len(rows[name]) != expected_rows:
            raise ValueError(f"{run_name}/{name} row count mismatch: {len(rows[name])} != {expected_rows}")
    return {
        "run_name": run_name,
        "run_label": RUN_LABELS[run_name],
        "run_dir": str(run_dir),
        "files": {name: str(path) for name, path in files.items()},
        "rows": rows,
        "summaries": summaries,
    }


def _build_case_report(
    *,
    config: AgenticStage2BenchmarkConfig,
    index: int,
    decision_ref: dict[str, Any],
    final_ref: dict[str, Any],
    runs: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    case_id = _case_id(final_ref, index)
    expected = _expected_from_ref(final_ref)
    run_reports = {
        run_name: _run_case_report(run_name, run, index)
        for run_name, run in runs.items()
    }
    return {
        "schema_version": "sidewalk_agentic_stage2_benchmark_case_report_v1",
        "run_id": config.run_id,
        "case_id": case_id,
        "index": index,
        "identity": _identity(final_ref, index),
        "tool_policy": expected["tool_policy"],
        "expected": expected,
        "runs": run_reports,
        "artifacts": _case_artifacts(case_id, index, final_ref, runs, output_root),
    }


def _run_case_report(run_name: str, run: dict[str, Any], index: int) -> dict[str, Any]:
    rows = run["rows"]
    return {
        "run_label": RUN_LABELS[run_name],
        "decision_prediction": _row_at(rows.get("decision_predictions"), index),
        "decision_metrics": _row_at(rows.get("decision_metrics"), index),
        "final_prediction": _row_at(rows.get("final_predictions"), index),
        "final_agentic_metrics": _row_at(rows.get("final_agentic_metrics"), index),
        "final_text_traj_metrics": _row_at(rows.get("final_text_traj_metrics"), index),
        "final_head_prediction": _row_at(rows.get("final_head_predictions"), index),
        "final_head_traj_metrics": _row_at(rows.get("final_head_traj_metrics"), index),
        "errors": _run_case_errors(run_name, run, index),
    }


def _row_at(rows: list[dict[str, Any]] | None, index: int) -> dict[str, Any] | None:
    if not rows or index >= len(rows):
        return None
    return rows[index]


def _run_case_errors(run_name: str, run: dict[str, Any], index: int) -> list[str]:
    label = RUN_LABELS[run_name]
    errors: list[str] = []
    rows = run["rows"]
    decision_metrics = _row_at(rows.get("decision_metrics"), index)
    final_metrics = _row_at(rows.get("final_agentic_metrics"), index)
    final_traj_metrics = _row_at(rows.get("final_text_traj_metrics"), index)
    head_metrics = _row_at(rows.get("final_head_traj_metrics"), index)
    # 这里不直接判定模型好坏，只把会阻断人工 review 的协议/解析问题列出来。
    if _row_at(rows.get("decision_predictions"), index) is None:
        errors.append(f"{label}:missing_decision_prediction")
    if _row_at(rows.get("final_predictions"), index) is None:
        errors.append(f"{label}:missing_final_prediction")
    if decision_metrics and _metric_value(decision_metrics, "protocol_parse_success") == 0.0:
        errors.append(f"{label}:decision_protocol_parse_failed")
    if final_metrics and _metric_value(final_metrics, "protocol_parse_success") == 0.0:
        errors.append(f"{label}:final_protocol_parse_failed")
    if final_metrics and _metric_value(final_metrics, "final_cot_tag_validity") == 0.0:
        errors.append(f"{label}:final_cot_tags_invalid")
    if final_traj_metrics and _metric_value(final_traj_metrics, "parse_success") == 0.0:
        errors.append(f"{label}:text_trajectory_parse_failed")
    if head_metrics and _metric_value(head_metrics, "parse_success") == 0.0:
        errors.append(f"{label}:head_trajectory_parse_failed")
    return errors


def _metric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_error_rows(per_case: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in per_case:
        for run_name, run in (case.get("runs") or {}).items():
            for error in run.get("errors") or []:
                rows.append(
                    {
                        "run_id": case["run_id"],
                        "case_id": case["case_id"],
                        "index": case["index"],
                        "run": RUN_LABELS.get(run_name, run_name),
                        "error": error,
                        "tool_policy": case.get("tool_policy"),
                        "expected_tool": (case.get("expected") or {}).get("tool"),
                        "expected_op": (case.get("expected") or {}).get("op"),
                    }
                )
    return rows


def _case_artifacts(
    case_id: str,
    index: int,
    final_ref: dict[str, Any],
    runs: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "index": index,
        "reference": {
            "benchmark_dir": str(output_root / "benchmark"),
            "decision_refs": str(output_root / "benchmark" / BENCHMARK_REF_FILES["decision"]),
            "final_refs": str(output_root / "benchmark" / BENCHMARK_REF_FILES["final"]),
            "row_index": index,
            "images": list(final_ref.get("images") or []),
        },
        "runs": {
            RUN_LABELS[run_name]: {
                "run_dir": run["run_dir"],
                "row_index": index,
                "files": run["files"],
            }
            for run_name, run in runs.items()
        },
    }


def _build_summary(
    config: AgenticStage2BenchmarkConfig,
    per_case: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    tool_counts: dict[str, int] = {}
    for case in per_case:
        policy = str(case.get("tool_policy") or "unknown")
        tool_counts[policy] = tool_counts.get(policy, 0) + 1
    return {
        "schema_version": "sidewalk_agentic_stage2_benchmark_summary_v1",
        "run_id": config.run_id,
        "num_cases": len(per_case),
        "tool_policy_counts": dict(sorted(tool_counts.items())),
        "num_errors": len(errors),
        "run_summaries": {
            RUN_LABELS[run_name]: _compact_run_summary(run)
            for run_name, run in runs.items()
        },
        "caveats": [
            "Tool-call traces are diagnostics unless promoted by the benchmark contract.",
            "Text and trajectory-head metrics are collected separately.",
        ],
    }


def _compact_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    summaries = run.get("summaries") or {}
    return {
        "decision_agentic": _summary_metrics(summaries.get("decision_summary")),
        "final_agentic": _summary_metrics(summaries.get("final_agentic_summary")),
        "final_text_traj": _summary_metrics(summaries.get("final_text_traj_summary")),
        "final_head_traj": _summary_metrics(summaries.get("final_head_traj_summary")),
    }


def _summary_metrics(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else summary
    keys = [
        "protocol_parse_success",
        "tool_decision_present",
        "tool_op_match",
        "final_cot_tag_validity",
        "tool_result_use_valid",
        "trajectory_parse_success",
        "parse_success",
        "strict_success",
        "ade",
        "fde",
        "pixel_ade",
        "pixel_fde",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# SideWalk Agentic Stage2 Benchmark: {summary['run_id']}",
        "",
        f"- Cases: {summary['num_cases']}",
        f"- Errors: {summary['num_errors']}",
        f"- Tool policy counts: `{json.dumps(summary['tool_policy_counts'], sort_keys=True)}`",
        "",
        "## Run Summaries",
        "",
    ]
    for run_label, run_summary in (summary.get("run_summaries") or {}).items():
        lines.append(f"### {run_label}")
        for metric_group, metrics in (run_summary or {}).items():
            lines.append(f"- {metric_group}: `{json.dumps(metrics, allow_nan=True, sort_keys=True)}`")
        lines.append("")
    lines.extend(["## Caveats", ""])
    for caveat in summary.get("caveats") or []:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def _expected_from_ref(row: dict[str, Any]) -> dict[str, Any]:
    meta = _meta(row)
    return {
        "tool_policy": str(meta.get("stage2_tool_policy") or "unknown"),
        "should_call_tool": bool(meta.get("stage2_expected_should_call_tool")),
        "tool": meta.get("stage2_expected_tool"),
        "op": meta.get("stage2_expected_op"),
        "source": meta.get("source"),
        "urbannav_case_id": meta.get("urbannav_case_id"),
        "images": list(row.get("images") or []),
    }


def _identity(row: dict[str, Any], index: int) -> dict[str, Any]:
    meta = _meta(row)
    return {
        "case_id": _case_id(row, index),
        "source_id": meta.get("source_id"),
        "clip_id": meta.get("clip_id"),
        "frame_idx": meta.get("frame_idx"),
        "urbannav_case_id": meta.get("urbannav_case_id"),
        "city": meta.get("city"),
        "scene_id": meta.get("scene_id"),
    }


def _case_id(row: dict[str, Any], index: int) -> str:
    meta = _meta(row)
    for key in ("case_id", "source_id", "urbannav_case_id"):
        value = meta.get(key)
        if value:
            return str(value)
    # 老数据缺 stable id 时才退回行号；正式 benchmark 应优先固定 case_id。
    return f"row-{index:06d}"


def _meta(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("sidewalk_meta")
    return meta if isinstance(meta, dict) else {}
