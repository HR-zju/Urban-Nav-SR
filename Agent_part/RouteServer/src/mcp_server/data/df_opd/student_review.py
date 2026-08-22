"""Build a private, static Student-versus-GT DF-OPD review bundle."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.data.df_opd.review import _image_html, _trajectory_svg
from mcp_server.data.io.jsonl import file_sha256, iter_jsonl
from mcp_server.eval.df_opd_student import (
    _find_forbidden_field,
    _parse_valid_trajectory,
    _prefix_ade,
    _prefix_fde,
)
from mcp_server.contracts.df_opd import DFOPDCanonicalSampleV1


DF_OPD_STUDENT_REVIEW_MANIFEST_SCHEMA_V1 = "urbannav_df_opd_student_review_manifest_v1"


def build_df_opd_student_review_bundle_v1(
    canonical_jsonl: str | Path,
    prediction_jsonls: Sequence[str | Path],
    output_html: str | Path,
    output_manifest_json: str | Path,
    *,
    review_selection_jsonl: str | Path | None = None,
    media_root: str | Path | None = None,
    max_samples: int = 50,
    visualization_index_json: str | Path | None = None,
    scorecard_json: str | Path | None = None,
    run_id: str | None = None,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    """Render a local Student-versus-GT review page.

    Args:
        canonical_jsonl: Canonical samples containing private GT and media.
        prediction_jsonls: Student prediction/failure ledgers.
        output_html: Destination HTML review file.
        output_manifest_json: Destination lineage and integrity receipt.
        review_selection_jsonl: Optional ordered sample-ID selection.
        media_root: Base directory for relative Student media paths.
        max_samples: Maximum number of rendered samples.
        visualization_index_json: Optional finished visualization index path.
        scorecard_json: Optional scorecard included in the finished index.
        run_id: Stable experiment run identity for the visualization index.
        run_root: Durable experiment root for the visualization index.

    Returns:
        A private review receipt with input/output digests and sample IDs.

    Raises:
        ValueError: If identity, coverage, media, or trajectory invariants fail.
        FileNotFoundError: If an input ledger or media file is missing.
    """

    if int(max_samples) <= 0:
        raise ValueError("max_samples must be positive")
    canonical_path = Path(canonical_jsonl).expanduser().resolve()
    prediction_paths = [Path(path).expanduser().resolve() for path in prediction_jsonls]
    if not prediction_paths or len(set(prediction_paths)) != len(prediction_paths):
        raise ValueError("Student review requires unique prediction JSONLs")
    for path in prediction_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    html_path = Path(output_html).expanduser().resolve()
    manifest_path = Path(output_manifest_json).expanduser().resolve()
    resolved_media_root = (
        Path(media_root).expanduser().resolve()
        if media_root is not None
        else canonical_path.parent
    )

    samples = _load_samples(canonical_path)
    predictions = _load_predictions(prediction_paths, samples)
    selected_ids = _load_selection_ids(review_selection_jsonl) if review_selection_jsonl else list(samples)
    selected_ids = selected_ids[:int(max_samples)]
    if not selected_ids:
        raise ValueError("Student review requires at least one selected sample")
    unknown_ids = sorted(set(selected_ids) - set(samples))
    if unknown_ids:
        raise ValueError("review selection contains unknown sample_id: " + ", ".join(unknown_ids[:8]))

    cards = [
        _render_case(samples[sample_id], predictions[sample_id], resolved_media_root)
        for sample_id in selected_ids
    ]
    _write_text_atomic(html_path, _render_html(cards))
    receipt = {
        "schema_version": DF_OPD_STUDENT_REVIEW_MANIFEST_SCHEMA_V1,
        "private_artifact": True,
        "canonical_jsonl": str(canonical_path),
        "canonical_jsonl_sha256": file_sha256(canonical_path),
        "prediction_jsonls": [str(path) for path in prediction_paths],
        "prediction_jsonl_sha256": [file_sha256(path) for path in prediction_paths],
        "media_root": str(resolved_media_root),
        "output_html": str(html_path),
        "output_html_sha256": file_sha256(html_path),
        "sample_ids": selected_ids,
    }
    _write_json_atomic(manifest_path, receipt)
    if visualization_index_json is not None:
        if not run_id or run_root is None:
            raise ValueError("visualization index requires run_id and run_root")
        index_path = Path(visualization_index_json).expanduser().resolve()
        index = {
            "schema_version": "research_visualization_manifest_v1",
            "run_id": run_id,
            "run_kind": "evaluation",
            "phase": "finished",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_root": str(Path(run_root).expanduser().resolve()),
            "tensorboard_logdir": None,
            "dashboard_url": None,
            "planned_outputs": [],
            "artifacts": [
                _describe_artifact("scorecard", Path(scorecard_json).expanduser().resolve())
                if scorecard_json is not None
                else None,
                _describe_artifact("student_review_html", html_path),
            ],
        }
        index["artifacts"] = [item for item in index["artifacts"] if item is not None]
        _write_json_atomic(index_path, index)
        receipt["visualization_index"] = str(index_path)
        receipt["visualization_index_sha256"] = file_sha256(index_path)
        _write_json_atomic(manifest_path, receipt)
    return receipt


def _describe_artifact(name: str, path: Path) -> dict[str, Any]:
    """Describe one finished visualization artifact with its content digest."""

    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "id": name,
        "path": str(path.resolve()),
        "exists": True,
        "kind": "file",
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _load_samples(path: Path) -> dict[str, DFOPDCanonicalSampleV1]:
    """Load unique canonical samples."""

    samples: dict[str, DFOPDCanonicalSampleV1] = {}
    for row in iter_jsonl(path):
        sample = DFOPDCanonicalSampleV1.model_validate(row)
        if sample.sample_id in samples:
            raise ValueError(f"duplicate canonical sample_id: {sample.sample_id!r}")
        samples[sample.sample_id] = sample
    if not samples:
        raise ValueError("Student review requires canonical samples")
    return samples


def _load_predictions(
    paths: Sequence[Path],
    samples: Mapping[str, DFOPDCanonicalSampleV1],
) -> dict[str, Mapping[str, Any]]:
    """Load one Student row per canonical sample and preserve no private fields."""

    predictions: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        for row in iter_jsonl(path):
            forbidden = _find_forbidden_field(row)
            if forbidden is not None:
                raise ValueError(f"Student review ledger leaked forbidden field {forbidden!r}: {path}")
            sample_id = row.get("sample_id")
            sample = samples.get(sample_id)
            if sample is None:
                raise ValueError(f"Student review row has unknown sample_id: {sample_id!r}")
            if (
                row.get("canonical_digest") != sample.canonical_digest
                or row.get("df_opd_profile_id") != sample.profile.profile_id
                or row.get("df_opd_profile_digest") != sample.profile.profile_digest
            ):
                raise ValueError(f"Student review identity/profile drift: {sample_id!r}")
            if sample_id in predictions:
                raise ValueError(f"duplicate Student review row: {sample_id!r}")
            predictions[sample_id] = row
    missing = sorted(set(samples) - set(predictions))
    if missing:
        raise ValueError("Student review lacks prediction rows: " + ", ".join(missing[:8]))
    return predictions


def _load_selection_ids(path: str | Path) -> list[str]:
    """Load an ordered, unique review selection."""

    values: list[str] = []
    seen: set[str] = set()
    for row in iter_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("review selection requires non-empty sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate review selection sample_id: {sample_id!r}")
        seen.add(sample_id)
        values.append(sample_id)
    return values


def _render_case(
    sample: DFOPDCanonicalSampleV1,
    prediction: Mapping[str, Any],
    media_root: Path,
) -> str:
    """Render one Student-visible case with GT and predicted trajectory."""

    predicted_trajectory = _parse_valid_trajectory(prediction.get("trajectory_ego_t"), sample)
    trajectories = [sample.trajectory_ego_t]
    if predicted_trajectory is not None:
        trajectories.append(predicted_trajectory)
    images = "".join(
        _image_html(path, media_root, role)
        for path, role in zip(sample.images, sample.image_roles)
    )
    decision = prediction.get("decision") or "(no Decision)"
    failure = prediction.get("failure_code") or "none"
    horizon_steps = sample.profile.target_horizon_steps
    metrics = (
        (
            f"{_prefix_ade(sample, predicted_trajectory, horizon_steps):.4f}",
            f"{_prefix_fde(sample, predicted_trajectory, horizon_steps):.4f}",
        )
        if predicted_trajectory is not None and all(sample.action_mask[:horizon_steps])
        else ("—", "—")
    )
    return (
        "<section class='case'>"
        f"<h2>{html.escape(sample.sample_id)}</h2>"
        f"<p>canonical={html.escape(sample.canonical_digest)} · profile={html.escape(sample.profile.profile_id)}</p>"
        f"<h3>Student-visible images</h3><div class='images'>{images}</div>"
        f"<h3>GT (green) / Student (blue)</h3>{_trajectory_svg(trajectories)}"
        "<p class='formula'>ADE_T = \u2211_t ||p_t \u2212 g_t||_2 / T; "
        "FDE_T = ||p_T \u2212 g_T||_2（仅在完整有效轨迹上计算）</p>"
        "<table><thead><tr><th>field</th><th>value</th></tr></thead><tbody>"
        f"<tr><th>GT Decision</th><td><pre>{html.escape(sample.decision_target)}</pre></td></tr>"
        f"<tr><th>Student Decision</th><td><pre>{html.escape(str(decision))}</pre></td></tr>"
        f"<tr><th>failure_code</th><td>{html.escape(str(failure))}</td></tr>"
        f"<tr><th>ADE (m)</th><td>{metrics[0]}</td></tr>"
        f"<tr><th>FDE (m)</th><td>{metrics[1]}</td></tr>"
        f"<tr><th>trajectory_points</th><td>{len(predicted_trajectory) if predicted_trajectory else 0} / {sample.profile.target_horizon_steps}</td></tr>"
        "</tbody></table></section>"
    )


__all__ = [
    "DF_OPD_STUDENT_REVIEW_MANIFEST_SCHEMA_V1",
    "build_df_opd_student_review_bundle_v1",
]


def _render_html(cards: list[str]) -> str:
    """Assemble a light, local-only Student review page."""

    return """<!doctype html><html><head><meta charset='utf-8'>
<title>Private DF-OPD Student review</title><meta name='robots' content='noindex'><style>
body{font-family:system-ui,sans-serif;background:#f8fafc;color:#0f172a;margin:24px}
.warning{background:#fee2e2;padding:12px;border-radius:8px}
.case{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin:16px 0}
.images{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0;width:180px}
img{max-width:180px;max-height:140px;object-fit:contain;background:#e2e8f0}
figcaption{font-size:12px;overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{border:1px solid #cbd5e1;padding:6px;vertical-align:top}
pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0}
.formula{font-family:serif;color:#475569;background:#f8fafc;padding:8px}
svg{width:300px;height:300px;background:#f8fafc}polyline{fill:none;stroke-width:2}
line{stroke:#64748b;stroke-width:1}.gt{background:#f1f5f9}</style></head><body>
<h1>Private DF-OPD Student review</h1><p class='warning'>
Contains private GT and raw Student outputs. Keep this file local.</p>
""" + "".join(cards) + "</body></html>"


def _write_text_atomic(path: Path, value: str) -> None:
    """Write text atomically so a review URL never sees a partial HTML file."""

    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one deterministic receipt atomically."""

    _write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
    )
