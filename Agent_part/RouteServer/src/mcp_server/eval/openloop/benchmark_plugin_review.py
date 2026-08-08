"""Portable HTML review for frozen pluggable open-loop benchmarks.

The generic renderer deliberately separates model-visible inputs, adapter
provenance, and evaluator-only geometry.  It is therefore safe for both the
UrbanNav-model-on-CityWalker direction and a future native-CityWalker-model-on-
UrbanNav direction without changing a benchmark scorecard.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from mcp_server.contracts.benchmark_plugin import BenchmarkInputV1
from mcp_server.eval.openloop.benchmark_plugin_base import BenchmarkJoinedRowV1
from mcp_server.eval.openloop.benchmark_plugin_review_base import (
    BenchmarkNativeAdapterTraceV1,
    BenchmarkReviewRendererBaseV1,
    BenchmarkReviewTimeSplitV1,
    summarize_benchmark_native_adapter_funnel_v1,
)


def write_benchmark_plugin_review_html_v1(
    *,
    path: str | Path,
    rows: Iterable[BenchmarkJoinedRowV1],
    inputs: Mapping[str, BenchmarkInputV1],
    horizon_s: float,
    title: str,
    media_href_resolver: Callable[[str], str] | None = None,
    review_annotations: Mapping[str, str] | None = None,
    embed_media: bool = False,
    adapter_traces: Mapping[str, BenchmarkNativeAdapterTraceV1] | None = None,
    adapter_trace_funnel: Mapping[str, object] | None = None,
    coverage_horizons_s: Sequence[float] | None = None,
) -> None:
    """Write a local-review page linked to stable IDs, media, and xy overlays.

    A supplied adapter trace is an independently emitted public-only sidecar.
    It augments the review page but is never written into a frozen input,
    reference, prediction, or scorecard.  With no sidecar this renders the
    usual two-column model-input/evaluator-geometry page.

    Args:
        path: Output HTML location.
        rows: Full joined benchmark rows in frozen reference order.
        inputs: Model-visible cases keyed by stable sample id.
        horizon_s: One physical horizon rendered in each trajectory overlay.
        title: Human-readable review title.
        media_href_resolver: Optional pure path-to-link renderer.  Supplying a
            resolver allows a review bundle to use relative copied media; the
            default remains the original local ``file://`` source path.
        review_annotations: Optional evaluator-owned concise annotation keyed
            by stable sample id, for example a deterministic selection stratum.
        embed_media: Whether to render the ordered FPV images as local review
            thumbnails in addition to their links.
        adapter_traces: Optional public-only native adapter traces keyed by
            stable sample id.
        adapter_trace_funnel: Optional precomputed full-denominator trace
            funnel.  It must not be computed from only selected cards.  This
            helper derives one only when ``rows`` exactly equals ``inputs``.
        coverage_horizons_s: Ordered score horizons used only when this helper
            needs to build an omitted ``adapter_trace_funnel``.

    Returns:
        ``None``.

    Raises:
        ValueError: If a joined reference has no corresponding visible input.
    """

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized_rows = list(rows)
    normalized_inputs = dict(inputs)
    normalized_traces = None if adapter_traces is None else dict(
        adapter_traces)
    geometry_renderer = BenchmarkReviewRendererBaseV1(horizon_s=horizon_s)
    row_ids = {row.reference.sample_id for row in materialized_rows}
    if (normalized_traces is not None and adapter_trace_funnel is None
            and row_ids == set(normalized_inputs)):
        adapter_trace_funnel = summarize_benchmark_native_adapter_funnel_v1(
            rows=materialized_rows,
            inputs=normalized_inputs,
            traces=normalized_traces,
            horizons_s=tuple(coverage_horizons_s or (horizon_s, )),
        )
    if adapter_trace_funnel is not None:
        reported_denominator = adapter_trace_funnel.get("denominator_count")
        if reported_denominator != len(normalized_inputs):
            raise ValueError(
                "adapter trace funnel denominator must equal frozen input count"
            )
    cards: list[str] = []
    for row in materialized_rows:
        input_row = normalized_inputs.get(row.reference.sample_id)
        if input_row is None:
            raise ValueError(
                f"review input is missing: {row.reference.sample_id}")
        trace = (None if normalized_traces is None else normalized_traces.get(
            row.reference.sample_id))
        cards.append(
            _render_case_v1(
                row=row,
                input_row=input_row,
                geometry_renderer=geometry_renderer,
                media_href_resolver=media_href_resolver,
                review_annotation=None if review_annotations is None else
                review_annotations.get(row.reference.sample_id),
                embed_media=embed_media,
                adapter_trace=trace,
            ))
    destination.write_text(
        _render_page_v1(
            title=title,
            cards=cards,
            adapter_trace_funnel=adapter_trace_funnel,
        ),
        encoding="utf-8",
    )


def _render_case_v1(
    *,
    row: BenchmarkJoinedRowV1,
    input_row: BenchmarkInputV1,
    geometry_renderer: BenchmarkReviewRendererBaseV1,
    media_href_resolver: Callable[[str], str] | None,
    review_annotation: str | None,
    embed_media: bool,
    adapter_trace: BenchmarkNativeAdapterTraceV1 | None,
) -> str:
    """Render one stable-ID review card with strictly separated evidence."""

    reference = row.reference
    case_geometry = geometry_renderer.prepare_case(row)
    reference_curve = case_geometry.reference_curve
    prediction_curve = case_geometry.prediction_curve
    history = _public_xy_series_v1(
        input_row.public_context.get("history_xy_m"))
    point_goal = _public_xy_point_v1(
        input_row.public_context.get("target_xy_m"))
    coordinate_frame = str(
        input_row.public_context.get("coordinate_frame",
                                     reference.coordinate_frame))
    visible_input = _render_visible_input_panel_v1(
        input_row=input_row,
        media_href_resolver=media_href_resolver,
        embed_media=embed_media,
        adapter_trace=adapter_trace,
        history=history,
        point_goal=point_goal,
    )
    geometry_panel = _render_evaluator_geometry_panel_v1(
        history=history,
        point_goal=point_goal,
        reference_curve=reference_curve,
        prediction_curve=prediction_curve,
        horizon_s=geometry_renderer.horizon_s,
        coordinate_frame=coordinate_frame,
    )
    adapter_panel = ("" if adapter_trace is None else
                     _render_adapter_audit_panel_v1(adapter_trace))
    failure = ""
    if row.failure is not None:
        failure = html.escape(
            json.dumps(row.failure, ensure_ascii=False, sort_keys=True))
    prompt_json = html.escape(
        json.dumps(input_row.model_dump(mode="json"),
                   ensure_ascii=False,
                   indent=2))
    time_summary = _time_summary_v1(
        reference_curve=reference_curve,
        prediction_curve=prediction_curve,
        horizon_s=geometry_renderer.horizon_s,
    )
    layout_class = "layout three-column" if adapter_trace is not None else "layout"
    return f'''<article class="case" id="{html.escape(reference.sample_id)}">
<h2>{html.escape(reference.sample_id)}</h2>
<p><b>dataset:</b> {html.escape(reference.dataset_source)} &nbsp; <b>split:</b> {html.escape(reference.split)}
&nbsp; <b>scenarios:</b> {html.escape(', '.join(reference.scenario_labels) or 'none')}</p>
<p><b>score horizon:</b> {geometry_renderer.horizon_s:g}s &nbsp; <b>L2 @{geometry_renderer.horizon_s:g}s:</b> {_format_float(case_geometry.l2_at_horizon_m)}
&nbsp; <b>prediction failure:</b> {failure or 'none'}</p>
<p class="hint">{html.escape(time_summary)}</p>
{'' if review_annotation is None else f'<p><b>audit selection:</b> {html.escape(review_annotation)}</p>'}
<div class="{layout_class}">{visible_input}{adapter_panel}{geometry_panel}</div>
<details><summary>Visible prompt and context</summary><pre>{prompt_json}</pre></details>
</article>'''


def _render_visible_input_panel_v1(
    *,
    input_row: BenchmarkInputV1,
    media_href_resolver: Callable[[str], str] | None,
    embed_media: bool,
    adapter_trace: BenchmarkNativeAdapterTraceV1 | None,
    history: list[tuple[float, float]],
    point_goal: tuple[float, float] | None,
) -> str:
    """Render model-visible images and only public history/goal context."""

    selected_indices = (None if adapter_trace is None else
                        adapter_trace.selected_image_indices)
    media = _render_media_v1(
        input_row.images,
        media_href_resolver=media_href_resolver,
        embed_media=embed_media,
        selected_image_indices=selected_indices,
    )
    native_hint = ""
    if adapter_trace is not None:
        native_hint = (
            "blue border = native selected; gray = UrbanNav-visible but unused"
        )
    public_history = _format_public_series_v1(history)
    public_goal = ("not declared in public_context"
                   if point_goal is None else _format_xy_v1(point_goal))
    return f'''<section class="fpv-panel"><b>1. Model-visible UrbanNav input</b>
<span class="hint">dark border = current input frame{'; ' + native_hint if native_hint else ''}</span>
{media}
<dl class="public-context"><dt>public history</dt><dd>{html.escape(public_history)}</dd>
<dt>public point goal</dt><dd>{html.escape(public_goal)}</dd></dl></section>'''


def _render_adapter_audit_panel_v1(
    trace: BenchmarkNativeAdapterTraceV1, ) -> str:
    """Render public-only native adapter provenance without evaluator geometry."""

    rows = (
        ("status", trace.eligibility_status),
        ("reject reason", trace.reject_reason or "none"),
        ("benchmark input digest",
         _short_digest_v1(trace.benchmark_input_digest)),
        ("adapter",
         f"{trace.native_adapter_id} / {trace.native_adapter_version}"),
        ("checkpoint digest",
         _short_digest_v1(trace.native_checkpoint_digest)),
        ("selected input frames",
         _format_selection_v1(trace.selected_image_indices,
                              trace.selected_image_roles)),
        ("public history transform", trace.public_history_transform_id
         or "N/A"),
        ("public scale estimator", trace.public_scale_estimator_id or "N/A"),
        ("public scale value", _format_scale_v1(trace.public_scale_value)),
        ("input projection", trace.input_coordinate_projection_id or "N/A"),
        ("output projection", trace.output_coordinate_projection_id or "N/A"),
        ("native output schedule", trace.native_output_schedule_id or "N/A"),
        ("native output count", str(trace.native_output_count)),
    )
    table = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in rows)
    return f'''<section class="adapter-panel"><b>2. Native-adapter audit</b>
<span class="hint">public adapter trace only; no future GT, hidden reference time, or evaluator L2</span>
<table class="audit-table">{table}</table></section>'''


def _render_evaluator_geometry_panel_v1(
    *,
    history: list[tuple[float, float]],
    point_goal: tuple[float, float] | None,
    reference_curve: BenchmarkReviewTimeSplitV1,
    prediction_curve: BenchmarkReviewTimeSplitV1,
    horizon_s: float,
    coordinate_frame: str,
) -> str:
    """Render evaluator-owned GT/prediction geometry distinct from input audit."""

    overlay = _trajectory_svg_v1(
        history=history,
        point_goal=point_goal,
        reference_curve=reference_curve,
        prediction_curve=prediction_curve,
        horizon_s=horizon_s,
        coordinate_frame=coordinate_frame,
    )
    return f'''<section class="geometry-panel"><b>3. Evaluator-only geometry</b>
<span class="hint">{html.escape(coordinate_frame)}</span>
<svg viewBox="0 0 360 360" aria-label="local trajectory overlay">{overlay}</svg>
<p class="legend"><span class="history">●</span> visible history
<span class="goal">★</span> public goal <span class="ground-truth">●</span> hidden GT
<span class="prediction">●</span> prediction <span class="current">●</span> current origin<br>
solid = score interval; pale dashed = beyond score horizon; filled markers = @{horizon_s:g}s;
hollow markers = available terminal</p></section>'''


def _trajectory_svg_v1(
    *,
    history: list[tuple[float, float]],
    point_goal: tuple[float, float] | None,
    reference_curve: BenchmarkReviewTimeSplitV1,
    prediction_curve: BenchmarkReviewTimeSplitV1,
    horizon_s: float,
    coordinate_frame: str,
) -> str:
    """Render an equal-axis local plot with score and terminal semantics split.

    The solid reference/prediction prefixes end exactly at the requested score
    horizon.  Future available points are pale dashed geometry only.  This
    avoids implying that a filled ``GT @ 2s`` marker is the five-waypoint
    terminal target in CityWalker-derived references.
    """

    all_points = [
        (0.0, 0.0),
        *history,
        *reference_curve.scored_prefix,
        *reference_curve.future_suffix,
        *prediction_curve.scored_prefix,
        *prediction_curve.future_suffix,
    ]
    local_radius = max(
        1.0,
        max((max(abs(x), abs(y)) for x, y in all_points), default=1.0),
    )
    axis_radius = local_radius * 1.18

    def project(point: tuple[float, float]) -> tuple[float, float]:
        return (
            180.0 + 136.0 * point[0] / axis_radius,
            180.0 - 136.0 * point[1] / axis_radius,
        )

    grid = _grid_svg_v1()
    history_line = _polyline_v1(history, "#5771DB", project, width=3.5)
    reference_prefix = _polyline_v1(reference_curve.scored_prefix,
                                    "#4F9A26",
                                    project,
                                    width=4.0)
    reference_tail = _polyline_v1(
        reference_curve.future_suffix,
        "#92DB58",
        project,
        width=3.2,
        dash="8 6",
        opacity=0.58,
    )
    prediction_prefix = _polyline_v1(prediction_curve.scored_prefix,
                                     "#C6423B",
                                     project,
                                     width=4.0)
    prediction_tail = _polyline_v1(
        prediction_curve.future_suffix,
        "#DB6057",
        project,
        width=3.2,
        dash="8 6",
        opacity=0.58,
    )
    reference_mark = _score_marker_v1(
        reference_curve.at_horizon,
        color="#4F9A26",
        label=f"GT @ {horizon_s:g}s",
        project=project,
        label_direction="left",
    )
    prediction_mark = _score_marker_v1(
        prediction_curve.at_horizon,
        color="#C6423B",
        label=f"Pred @ {horizon_s:g}s",
        project=project,
        label_direction="right",
    )
    reference_terminal = _terminal_marker_v1(
        reference_curve,
        color="#4F9A26",
        label="GT terminal",
        project=project,
        label_direction="left",
    )
    prediction_terminal = _terminal_marker_v1(
        prediction_curve,
        color="#C6423B",
        label="Pred terminal",
        project=project,
        label_direction="right",
    )
    origin = _circle_v1(project((0.0, 0.0)), "#1C2938")
    goal = _goal_svg_v1(point_goal=point_goal,
                        axis_radius=axis_radius,
                        project=project)
    x_label, y_label = _axis_labels_v1(coordinate_frame)
    return (
        grid + history_line + reference_prefix + reference_tail +
        prediction_prefix + prediction_tail + goal + origin + reference_mark +
        prediction_mark + reference_terminal + prediction_terminal +
        f'<text x="180" y="348" text-anchor="middle" class="axis-label">{html.escape(x_label)} →</text>'
        +
        f'<text x="14" y="24" class="axis-label">{html.escape(y_label)} ↑</text>'
        +
        f'<text x="14" y="344" class="scale-label">local scale ±{axis_radius:.1f} m</text>'
    )


def _grid_svg_v1() -> str:
    """Render deterministic equal-axis grid lines for one local xy panel."""

    result: list[str] = []
    for fraction in (-0.5, 0.0, 0.5):
        offset = 180.0 + 136.0 * fraction
        color = "#AAB8C5" if fraction == 0.0 else "#DCE4EC"
        width = "1.5" if fraction == 0.0 else "1"
        result.append(
            f'<line x1="44" y1="{offset:.2f}" x2="316" y2="{offset:.2f}" stroke="{color}" stroke-width="{width}"/>'
        )
        result.append(
            f'<line x1="{offset:.2f}" y1="44" x2="{offset:.2f}" y2="316" stroke="{color}" stroke-width="{width}"/>'
        )
    return "".join(result)


def _polyline_v1(
    points: Sequence[tuple[float, float]],
    color: str,
    project: Callable[[tuple[float, float]], tuple[float, float]],
    *,
    width: float,
    dash: str | None = None,
    opacity: float | None = None,
) -> str:
    """Return a styled SVG polyline, omitting zero- and one-point curves."""

    if len(points) < 2:
        return ""
    coordinates = " ".join(f"{x:.2f},{y:.2f}"
                           for x, y in (project(point) for point in points))
    dash_attribute = "" if dash is None else f' stroke-dasharray="{dash}"'
    opacity_attribute = (""
                         if opacity is None else f' opacity="{opacity:.3f}"')
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width:.1f}"'
        f'{dash_attribute}{opacity_attribute} points="{coordinates}"/>')


def _score_marker_v1(
    point: tuple[float, float] | None,
    *,
    color: str,
    label: str,
    project: Callable[[tuple[float, float]], tuple[float, float]],
    label_direction: str,
) -> str:
    """Render a filled evaluator horizon marker and a compact label."""

    if point is None:
        return ""
    x_value, y_value = project(point)
    offset = -9.0 if label_direction == "left" else 9.0
    anchor = "end" if label_direction == "left" else "start"
    y_label = y_value - 8.0 if y_value > 54.0 else y_value + 16.0
    return (
        _circle_v1((x_value, y_value), color) +
        f'<text x="{x_value + offset:.2f}" y="{y_label:.2f}" text-anchor="{anchor}" class="score-label">{html.escape(label)}</text>'
    )


def _terminal_marker_v1(
    curve: BenchmarkReviewTimeSplitV1,
    *,
    color: str,
    label: str,
    project: Callable[[tuple[float, float]], tuple[float, float]],
    label_direction: str,
) -> str:
    """Render a hollow terminal only when it lies beyond a scored endpoint."""

    if curve.terminal is None or curve.terminal_time_s is None:
        return ""
    if curve.at_horizon is not None and curve.terminal == curve.at_horizon:
        return ""
    x_value, y_value = project(curve.terminal)
    y_label = y_value + 16.0 if y_value < 56.0 else y_value - 8.0
    offset = -8.0 if label_direction == "left" else 8.0
    anchor = "end" if label_direction == "left" else "start"
    return (
        f'<circle cx="{x_value:.2f}" cy="{y_value:.2f}" r="5.4" fill="#FFFFFF" stroke="{color}" stroke-width="2.2"/>'
        f'<text x="{x_value + offset:.2f}" y="{y_label:.2f}" text-anchor="{anchor}" class="terminal-label">{html.escape(label)} @ {curve.terminal_time_s:g}s</text>'
    )


def _axis_labels_v1(coordinate_frame: str) -> tuple[str, str]:
    """Choose non-misleading local axes from a versioned frame identifier."""

    lowered = coordinate_frame.lower()
    if "x_right_y_front" in lowered:
        return "x_right (m)", "y_front (m)"
    if "x_forward_y_left" in lowered:
        return "x_forward (m)", "y_left (m)"
    return "x_local (m)", "y_local (m)"


def _goal_svg_v1(
    *,
    point_goal: tuple[float, float] | None,
    axis_radius: float,
    project: Callable[[tuple[float, float]], tuple[float, float]],
) -> str:
    """Render a public goal star in-frame or a directional clipped indicator."""

    if point_goal is None:
        return ""
    radius = math.hypot(*point_goal)
    displayed = point_goal
    suffix = ""
    if radius > axis_radius * 0.92:
        ratio = axis_radius * 0.88 / radius
        displayed = point_goal[0] * ratio, point_goal[1] * ratio
        suffix = f" (clipped; {radius:.1f} m)"
    x_value, y_value = project(displayed)
    text_on_left = x_value > 240.0
    label_x = x_value - 10.0 if text_on_left else x_value + 10.0
    label_y = y_value + 16.0 if y_value < 56.0 else y_value - 8.0
    anchor = "end" if text_on_left else "start"
    return (f'<path d="{_star_path_v1(x_value, y_value)}" fill="#A157DB" '
            'stroke="#5F278B" stroke-width="1.2"/>'
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" '
            f'text-anchor="{anchor}" class="goal-label">goal{suffix}</text>')


def _star_path_v1(
    x_value: float,
    y_value: float,
    *,
    outer_radius: float = 8.0,
    inner_radius: float = 3.4,
) -> str:
    """Return an SVG five-point-star path centered at a projected local point."""

    points: list[tuple[float, float]] = []
    for index in range(10):
        angle = -math.pi / 2.0 + math.pi * index / 5.0
        radius = outer_radius if index % 2 == 0 else inner_radius
        points.append((x_value + radius * math.cos(angle),
                       y_value + radius * math.sin(angle)))
    return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points) + " Z"


def _circle_v1(point: tuple[float, float], color: str) -> str:
    """Render one colored trajectory marker."""

    return f'<circle cx="{point[0]:.2f}" cy="{point[1]:.2f}" r="5" fill="{color}"/>'


def _media_href_v1(image: str, resolver: Callable[[str], str] | None) -> str:
    """Return one safe review media link while retaining legacy local paths."""

    return f"file://{image}" if resolver is None else resolver(image)


def _render_media_v1(
    images: Sequence[str],
    *,
    media_href_resolver: Callable[[str], str] | None,
    embed_media: bool,
    selected_image_indices: Sequence[int] | None,
) -> str:
    """Render ordered input media and explicitly mark native frame selection."""

    if not images:
        return "No images"
    selected = None if selected_image_indices is None else set(
        selected_image_indices)
    blocks: list[str] = []
    for index, image in enumerate(images):
        href = html.escape(_media_href_v1(image, media_href_resolver),
                           quote=True)
        name = html.escape(Path(image).name)
        if embed_media:
            classes = ["fpv"]
            if index == len(images) - 1:
                classes.append("current")
            if selected is not None:
                classes.append("native-selected" if index in
                               selected else "native-unused")
            class_name = " ".join(classes)
            blocks.append(
                f'<a href="{href}" target="_blank"><img class="{class_name}" src="{href}" alt="{name}" title="image {index}: {name}"></a>'
            )
        else:
            status = ""
            if selected is not None:
                status = " [native selected]" if index in selected else " [not native selected]"
            blocks.append(
                f'<a href="{href}" target="_blank">{name}{html.escape(status)}</a>'
            )
    return '<div class="media-strip">' + "".join(blocks) + "</div>"


def _public_xy_series_v1(value: object) -> list[tuple[float, float]]:
    """Parse a finite public xy history without accepting malformed context."""

    if not isinstance(value, list):
        return []
    result: list[tuple[float, float]] = []
    for item in value:
        point = _public_xy_point_v1(item)
        if point is None:
            return []
        result.append(point)
    return result


def _public_xy_point_v1(value: object) -> tuple[float, float] | None:
    """Parse one finite public two-dimensional point for review-only display."""

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        point = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(item) for item in point) else None


def _time_summary_v1(
    *,
    reference_curve: BenchmarkReviewTimeSplitV1,
    prediction_curve: BenchmarkReviewTimeSplitV1,
    horizon_s: float,
) -> str:
    """Describe time coverage without claiming a terminal is scoreable."""

    return (
        f"GT span {_format_time_span_v1(reference_curve)}; prediction span "
        f"{_format_time_span_v1(prediction_curve)}; scored only at {horizon_s:g}s."
    )


def _format_time_span_v1(curve: BenchmarkReviewTimeSplitV1) -> str:
    """Render the observed time span or an explicit unavailable state."""

    if curve.initial_time_s is None or curve.terminal_time_s is None:
        return "unavailable"
    return f"{curve.initial_time_s:g}–{curve.terminal_time_s:g}s"


def _format_public_series_v1(history: Sequence[tuple[float, float]]) -> str:
    """Compactly render public history without accidentally implying GT."""

    if not history:
        return "not declared in public_context"
    return ", ".join(_format_xy_v1(point) for point in history)


def _format_xy_v1(point: tuple[float, float]) -> str:
    """Render one local point for a human audit table."""

    return f"({point[0]:.3f}, {point[1]:.3f}) m"


def _format_selection_v1(
    indices: Sequence[int],
    roles: Sequence[str],
) -> str:
    """Render the explicit native image-index/role selection."""

    return ("none" if not indices else ", ".join(
        f"{index}:{role}" for index, role in zip(indices, roles)))


def _format_scale_v1(value: float | None) -> str:
    """Render public scale with an explicit unavailable state."""

    return "N/A" if value is None else f"{value:.6g}"


def _short_digest_v1(value: str) -> str:
    """Keep review tables legible while retaining a checkpoint identity prefix."""

    return value if len(value) <= 16 else value[:16] + "…"


def _format_float(value: float | None) -> str:
    """Render unavailable review values without converting them into zeros."""

    return "N/A" if value is None else f"{value:.3f} m"


def _render_page_v1(
    *,
    title: str,
    cards: Sequence[str],
    adapter_trace_funnel: Mapping[str, object] | None,
) -> str:
    """Render one self-contained local review document and optional funnel."""

    funnel = ("" if adapter_trace_funnel is None else
              _render_adapter_trace_funnel_v1(adapter_trace_funnel))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#f5f7fa;color:#17202a}}main{{max-width:1580px;margin:auto}}
.case,.funnel{{background:white;border:1px solid #d6dde5;border-radius:10px;padding:16px;margin:16px 0}}.layout{{display:grid;grid-template-columns:minmax(420px,1.15fr) minmax(380px,1fr);gap:18px;align-items:start}}.layout.three-column{{grid-template-columns:minmax(300px,1fr) minmax(285px,.86fr) minmax(360px,1fr)}}
.fpv-panel,.geometry-panel,.adapter-panel{{min-width:0}}.hint{{display:block;color:#5e6c79;font-size:.88rem;margin:4px 0 8px}}svg{{width:100%;max-width:420px;aspect-ratio:1;background:#fbfcfd;border:1px solid #d6dde5;border-radius:6px}}a{{margin-right:10px}}pre{{white-space:pre-wrap;overflow:auto;background:#16202a;color:#eaf0f5;padding:12px}}
.media-strip{{display:flex;flex-wrap:wrap;gap:7px;margin:8px 0}}.media-strip a{{margin:0}}img.fpv{{width:calc(20% - 6px);min-width:104px;aspect-ratio:1.56;object-fit:cover;border:2px solid #d6dde5;border-radius:4px}}img.fpv.current{{box-shadow:0 0 0 2px #1c2938}}img.fpv.native-selected{{border-color:#4f7dcc;box-shadow:0 0 0 2px #c9d8f7}}img.fpv.native-unused{{opacity:.36;filter:saturate(.18)}}
.legend{{font-size:.9rem;line-height:1.8;margin:6px 0}}.legend span{{font-weight:700;margin-left:7px}}.history{{color:#5771DB}}.goal{{color:#A157DB}}.ground-truth{{color:#4F9A26}}.prediction{{color:#C6423B}}.current{{color:#1C2938}}.axis-label{{font:12px system-ui,sans-serif;fill:#314152}}.scale-label{{font:11px system-ui,sans-serif;fill:#5e6c79}}.goal-label,.score-label,.terminal-label{{font:10px system-ui,sans-serif;paint-order:stroke;stroke:#fbfcfd;stroke-width:3;stroke-linejoin:round}}.goal-label{{fill:#5F278B}}.score-label{{fill:#1f2a35;font-weight:600}}.terminal-label{{fill:#5e6c79}}
.public-context,.audit-table{{font-size:.86rem;margin:10px 0;border-collapse:collapse;width:100%}}.public-context dt{{font-weight:650;margin-top:8px}}.public-context dd{{margin:1px 0;color:#425161;word-break:break-word}}.audit-table th,.audit-table td{{border-bottom:1px solid #e4e9ee;padding:5px 4px;text-align:left;vertical-align:top;word-break:break-word}}.audit-table th{{color:#465767;width:43%;font-weight:650}}.funnel table{{border-collapse:collapse;width:100%;max-width:920px}}.funnel th,.funnel td{{border-bottom:1px solid #dfe6ec;padding:6px 8px;text-align:left}}.funnel th{{color:#465767}}
@media(max-width:1120px){{.layout.three-column{{grid-template-columns:1fr 1fr}}.geometry-panel{{grid-column:1 / -1}}}}@media(max-width:900px){{.layout,.layout.three-column{{grid-template-columns:1fr}}.geometry-panel{{grid-column:auto}}img.fpv{{width:calc(33.333% - 6px)}}}}
</style></head><body><main><h1>{html.escape(title)}</h1><p>Review only. Solid geometry ends at the metric horizon; pale dashed geometry is available but not part of that horizon. It does not alter the canonical scorecard.</p>{funnel}{''.join(cards)}</main></body></html>'''


def _render_adapter_trace_funnel_v1(funnel: Mapping[str, object]) -> str:
    """Render full-denominator adapter coverage without hiding failed inputs."""

    denominator = _funnel_integer_v1(funnel.get("denominator_count"))
    trace_present = _funnel_integer_v1(funnel.get("trace_present_count"))
    trace_missing = _funnel_integer_v1(funnel.get("trace_missing_count"))
    statuses = funnel.get("adapter_status_counts")
    status_map = statuses if isinstance(statuses, Mapping) else {}
    reject_reasons = funnel.get("adapter_reject_reason_counts")
    reject_reason_map = (reject_reasons
                         if isinstance(reject_reasons, Mapping) else {})
    input_gates = funnel.get("public_input_gate_counts")
    input_gate_map = input_gates if isinstance(input_gates, Mapping) else {}
    successful = _funnel_integer_v1(funnel.get("successful_prediction_count"))
    rows = [
        ("UrbanNav frozen denominator", denominator),
        ("native adapter trace present", trace_present),
        ("native adapter trace missing", trace_missing),
        ("sufficient public FPV",
         _funnel_integer_v1(input_gate_map.get("public_fpv_available_count"))),
        ("public history and goal available",
         _funnel_integer_v1(
             input_gate_map.get("public_history_goal_available_count"))),
        ("public scale valid",
         _funnel_integer_v1(input_gate_map.get("public_scale_valid_count"))),
        ("input eligible", _funnel_integer_v1(status_map.get("eligible"))),
        ("input ineligible",
         _funnel_integer_v1(status_map.get("input_ineligible"))),
        ("first reject: missing public history",
         _funnel_integer_v1(reject_reason_map.get("missing_public_history"))),
        ("first reject: invalid scale",
         _funnel_integer_v1(reject_reason_map.get("invalid_scale"))),
        ("native model error",
         _funnel_integer_v1(status_map.get("native_model_error"))),
        ("native output contract error",
         _funnel_integer_v1(status_map.get("output_contract_error"))),
        ("native inference yielded prediction", successful),
    ]
    horizon_coverage = funnel.get("horizon_coverage")
    if isinstance(horizon_coverage, Mapping):
        for label, coverage in horizon_coverage.items():
            if not isinstance(coverage, Mapping):
                continue
            rows.append(
                (f"jointly scoreable @ {label}",
                 _funnel_integer_v1(coverage.get("joint_covered_count"))))
    table = "".join(f"<tr><th>{html.escape(label)}</th><td>{value}</td></tr>"
                    for label, value in rows)
    return f'''<section class="funnel"><h2>Adapter coverage funnel (review denominator)</h2>
<p class="hint">Diagnostic coverage only; canonical metrics still use the frozen benchmark denominator.</p>
<table>{table}</table></section>'''


def _funnel_integer_v1(value: object) -> int:
    """Display a malformed optional funnel field as a conservative zero."""

    return value if isinstance(value, int) and value >= 0 else 0


__all__ = ["write_benchmark_plugin_review_html_v1"]
