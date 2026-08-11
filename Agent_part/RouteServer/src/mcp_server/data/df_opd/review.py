"""Build private, static four-view DF-OPD human-review HTML bundles."""

from __future__ import annotations

import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from mcp_server.contracts.df_opd import DFOPDCanonicalSampleV1
from mcp_server.contracts.df_opd_certification import DFOPDTeacherCertificationScoreV1
from mcp_server.contracts.df_opd_review import DFOPDTeacherReviewPredictionV1
from mcp_server.data.io.jsonl import file_sha256, iter_jsonl


DF_OPD_PRIVATE_REVIEW_MANIFEST_SCHEMA_V1 = "urbannav_df_opd_private_review_manifest_v1"
_CONDITIONS = (
    "same_input_t0",
    "correct_privilege_tc",
    "shuffled_privilege_ts",
    "exact_future_oracle_to",
)
_CONDITION_LABELS = {
    "same_input_t0": "T0: same input",
    "correct_privilege_tc": "TC: correct cue",
    "shuffled_privilege_ts": "TS: shuffled cue",
    "exact_future_oracle_to": "TO: exact oracle",
}
_TRAJECTORY_COLORS = ("#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#a855f7")


def build_df_opd_private_review_bundle_v1(
    canonical_jsonl: str | Path,
    score_jsonl: str | Path,
    prediction_jsonl: str | Path,
    output_html: str | Path,
    output_manifest_json: str | Path,
    *,
    review_selection_jsonl: str | Path | None = None,
    media_root: str | Path | None = None,
    max_samples: int = 50,
) -> dict[str, Any]:
    """Render a local-only review page for complete T0/TC/TS/TO samples.

    The output deliberately references local media through ``file:`` URIs and
    never copies, uploads, or base64-embeds images.  It includes private GT and
    predictions and must therefore remain outside public/deployment artifacts.

    Args:
        canonical_jsonl: Canonical samples containing all four Teacher views.
        score_jsonl: Frozen condition score rows from certification inference.
        prediction_jsonl: Private raw Decision/trajectory review sidecars.
        output_html: Destination local review HTML.
        output_manifest_json: Destination integrity/lineage receipt.
        review_selection_jsonl: Optional ordered sample-ID selection, normally
            the certification high-conflict selection output.
        media_root: Base directory for relative image/cue paths.
        max_samples: Hard upper bound on rendered cases.

    Returns:
        A non-sensitive receipt listing input/output digests and sample IDs.

    Raises:
        ValueError: If four-view, score, prediction, identity, or local-media
            invariants are missing or inconsistent.
    """

    if int(max_samples) <= 0:
        raise ValueError("max_samples must be positive")
    canonical_path = Path(canonical_jsonl).expanduser().resolve()
    score_path = Path(score_jsonl).expanduser().resolve()
    prediction_path = Path(prediction_jsonl).expanduser().resolve()
    html_path = Path(output_html).expanduser().resolve()
    manifest_path = Path(output_manifest_json).expanduser().resolve()
    resolved_media_root = (
        Path(media_root).expanduser().resolve()
        if media_root is not None
        else canonical_path.parent
    )
    samples = _load_canonical_samples(canonical_path)
    scores = _load_condition_rows(score_path, DFOPDTeacherCertificationScoreV1)
    predictions = _load_condition_rows(prediction_path, DFOPDTeacherReviewPredictionV1)
    requested_ids = (
        _load_selection_ids(review_selection_jsonl)
        if review_selection_jsonl is not None
        else list(samples)
    )
    selected_ids = requested_ids[:int(max_samples)]
    if not selected_ids:
        raise ValueError("DF-OPD private review bundle requires at least one selected sample")
    cards = []
    for sample_id in selected_ids:
        try:
            sample = samples[sample_id]
        except KeyError as exc:
            raise ValueError(f"review selection references unknown canonical sample_id: {sample_id!r}") from exc
        cards.append(_render_case(sample, scores, predictions, resolved_media_root))
    page = _render_html(cards)
    _write_text_atomic(html_path, page)
    receipt = {
        "schema_version": DF_OPD_PRIVATE_REVIEW_MANIFEST_SCHEMA_V1,
        "private_artifact": True,
        "canonical_jsonl": str(canonical_path),
        "canonical_jsonl_sha256": file_sha256(canonical_path),
        "score_jsonl": str(score_path),
        "score_jsonl_sha256": file_sha256(score_path),
        "prediction_jsonl": str(prediction_path),
        "prediction_jsonl_sha256": file_sha256(prediction_path),
        "review_selection_jsonl": (
            str(Path(review_selection_jsonl).expanduser().resolve())
            if review_selection_jsonl is not None
            else None
        ),
        "media_root": str(resolved_media_root),
        "output_html": str(html_path),
        "output_html_sha256": file_sha256(html_path),
        "sample_ids": selected_ids,
    }
    _write_json_atomic(manifest_path, receipt)
    return receipt


def _load_canonical_samples(path: Path) -> dict[str, DFOPDCanonicalSampleV1]:
    """Load unique canonical samples without allowing silent replacement."""

    samples = {}
    for row in iter_jsonl(path):
        sample = DFOPDCanonicalSampleV1.model_validate(row)
        if sample.sample_id in samples:
            raise ValueError(f"duplicate canonical sample_id in review input: {sample.sample_id!r}")
        samples[sample.sample_id] = sample
    return samples


def _load_condition_rows(path: Path, model_type):
    """Load unique four-condition rows keyed by ``(sample_id, condition)``."""

    rows = {}
    for raw in iter_jsonl(path):
        row = model_type.model_validate(raw)
        key = (row.sample_id, row.condition)
        if key in rows:
            raise ValueError(f"duplicate review row for sample/condition {key!r}")
        rows[key] = row
    return rows


def _load_selection_ids(path: str | Path) -> list[str]:
    """Load ordered, unique sample IDs from a certification selection JSONL."""

    values = []
    seen = set()
    for row in iter_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("review selection rows require non-empty sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate review selection sample_id: {sample_id!r}")
        seen.add(sample_id)
        values.append(sample_id)
    return values


def _render_case(
    sample: DFOPDCanonicalSampleV1,
    scores: Mapping[tuple[str, str], DFOPDTeacherCertificationScoreV1],
    predictions: Mapping[tuple[str, str], DFOPDTeacherReviewPredictionV1],
    media_root: Path,
) -> str:
    """Render one fully paired private review card after identity validation."""

    condition_rows = []
    for condition in _CONDITIONS:
        score = scores.get((sample.sample_id, condition))
        prediction = predictions.get((sample.sample_id, condition))
        if score is None or prediction is None:
            raise ValueError(f"review requires score and prediction for {sample.sample_id!r}/{condition!r}")
        if score.canonical_digest != sample.canonical_digest or prediction.canonical_digest != sample.canonical_digest:
            raise ValueError(f"review canonical digest mismatch for {sample.sample_id!r}/{condition!r}")
        if (
            score.profile_digest != sample.profile.profile_digest
            or prediction.profile_digest != sample.profile.profile_digest
        ):
            raise ValueError(f"review profile digest mismatch for {sample.sample_id!r}/{condition!r}")
        view = sample.teacher_view_for_condition(condition)
        cue_html = "—"
        if condition != "same_input_t0":
            cue_html = _image_html(
                view.images[-1],
                media_root,
                "private Teacher cue",
                expected_sha256=view.cue_artifact_sha256,
            )
        condition_rows.append(
            "<tr>"
            f"<th>{html.escape(_CONDITION_LABELS[condition])}</th>"
            f"<td>{cue_html}</td>"
            f"<td>{html.escape(str(score.decision_strict_parse))}</td>"
            f"<td>{html.escape(str(score.decision_correct))}</td>"
            f"<td>{_number(score.trajectory_ade_m)}</td>"
            f"<td>{_number(score.trajectory_fde_m)}</td>"
            f"<td><pre>{html.escape(prediction.decision_text or '(no Decision)')}</pre></td>"
            "</tr>"
        )
    student_html = "".join(
        _image_html(path, media_root, role)
        for path, role in zip(sample.images, sample.image_roles)
    )
    trajectories = [sample.trajectory_ego_t]
    for condition in _CONDITIONS:
        prediction = predictions[(sample.sample_id, condition)]
        if prediction.trajectory_ego_t is not None:
            trajectories.append(prediction.trajectory_ego_t)
    return (
        "<section class='case'>"
        f"<h2>{html.escape(sample.sample_id)}</h2>"
        f"<p>canonical={html.escape(sample.canonical_digest)} · profile={html.escape(sample.profile.profile_id)}</p>"
        "<h3>Student-visible images</h3><div class='images'>"
        f"{student_html}</div>"
        "<h3>GT and predicted trajectories</h3>"
        f"{_trajectory_svg(trajectories)}"
        f"<pre class='gt'>{html.escape(sample.decision_target)}</pre>"
        "<table><thead><tr><th>condition</th><th>private cue</th><th>parse</th><th>Decision correct</th>"
        "<th>ADE m</th><th>FDE m</th><th>predicted Decision</th></tr></thead><tbody>"
        f"{''.join(condition_rows)}</tbody></table></section>"
    )


def _image_html(
    recorded_path: str,
    media_root: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Render a checked local-media reference without copying private bytes."""

    path = Path(recorded_path).expanduser()
    path = path if path.is_absolute() else media_root / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"review media file does not exist: {path}")
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError(f"review Teacher cue digest mismatch: {path}")
    return (
        "<figure><img loading='lazy' src='"
        f"{html.escape(path.as_uri(), quote=True)}' alt='{html.escape(label, quote=True)}'>"
        f"<figcaption>{html.escape(label)}: {html.escape(path.name)}</figcaption></figure>"
    )


def _trajectory_svg(trajectories) -> str:
    """Render metric FLU XY polylines without any external plotting dependency."""

    points = [(float(point[0]), float(point[1])) for trajectory in trajectories for point in trajectory]
    min_x, max_x = min(value[0] for value in points), max(value[0] for value in points)
    min_y, max_y = min(value[1] for value in points), max(value[1] for value in points)
    span = max(max_x - min_x, max_y - min_y, 1e-6)

    def render_points(trajectory):
        return " ".join(
            f"{20 + (float(x) - min_x) * 260 / span:.1f},{280 - (float(y) - min_y) * 260 / span:.1f}"
            for x, y, _ in trajectory
        )

    polylines = "".join(
        f"<polyline points='{render_points(trajectory)}' stroke='{_TRAJECTORY_COLORS[index]}'/>"
        for index, trajectory in enumerate(trajectories)
    )
    return (
        "<svg viewBox='0 0 300 300' role='img'>"
        "<line x1='20' y1='280' x2='280' y2='280'/>"
        "<line x1='20' y1='20' x2='20' y2='280'/>"
        f"{polylines}</svg>"
    )


def _number(value: float | None) -> str:
    """Render optional numeric metrics without inventing unavailable values."""

    return "—" if value is None else f"{value:.4f}"


def _render_html(cards: list[str]) -> str:
    """Assemble a static private review document with no remote dependencies."""

    page_prefix = """<!doctype html><html><head><meta charset='utf-8'>
<title>Private DF-OPD review</title><meta name='robots' content='noindex'><style>
body{font-family:system-ui,sans-serif;background:#f8fafc;color:#0f172a;margin:24px}
.warning{background:#fee2e2;padding:12px;border-radius:8px}
.case{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin:16px 0}
.images{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0;width:180px}
img{max-width:180px;max-height:140px;object-fit:contain;background:#e2e8f0}
figcaption{font-size:12px;overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{border:1px solid #cbd5e1;padding:6px;vertical-align:top}
pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0}
svg{width:300px;height:300px;background:#f8fafc}polyline{fill:none;stroke-width:2}
line{stroke:#64748b;stroke-width:1}.gt{background:#f1f5f9;padding:8px}</style></head><body>
<h1>Private DF-OPD four-view review</h1><p class='warning'>
Contains private Teacher media, GT and raw predictions. Keep this file local;
do not publish or place it in a Student deployment bundle.</p>
"""
    return page_prefix + "".join(cards) + "</body></html>"


def _write_text_atomic(path: Path, value: str) -> None:
    """Atomically write one private HTML artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
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
    """Atomically write a non-sensitive private-review receipt."""

    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


__all__ = [
    "DF_OPD_PRIVATE_REVIEW_MANIFEST_SCHEMA_V1",
    "build_df_opd_private_review_bundle_v1",
]
