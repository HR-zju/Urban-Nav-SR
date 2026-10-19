#!/usr/bin/env python3
"""Build a self-contained review page for Reasoning V2 annotations."""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import mimetypes
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mcp_server.data.io.jsonl import iter_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse review-page arguments.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--views-jsonl", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--title", default="UrbanNav Reasoning V2 Review")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument(
        "--link-images",
        action="store_true",
        help="Link local files instead of embedding them as data URLs.",
    )
    return parser.parse_args(argv)


def build_reasoning_v2_review_html(
    *,
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    views_jsonl: str | Path,
    output_html: str | Path,
    title: str = "UrbanNav Reasoning V2 Review",
    max_samples: int = 100,
    embed_images: bool = True,
) -> dict[str, Any]:
    """Build one review page by joining the three frozen V2 artifacts.

    Args:
        source_jsonl: Frozen ``ReasoningSourceV2`` rows.
        annotation_jsonl: Teacher ``ReasoningAnnotationV2`` rows.
        views_jsonl: Exported ``ReasoningTrainingViewV2`` rows.
        output_html: Destination HTML path.
        title: Browser and page title.
        max_samples: Maximum source groups rendered.
        embed_images: Embed local images as data URLs when true.

    Returns:
        Review artifact path and aggregate counts.

    Raises:
        ValueError: If duplicate source ids are found.
    """

    sources = _index_unique(iter_jsonl(source_jsonl), key="source_group_id", artifact="source")
    annotations = _index_latest(iter_jsonl(annotation_jsonl), key="source_group_id")
    views_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for view in iter_jsonl(views_jsonl):
        views_by_source[str(view.get("source_group_id") or "")].append(view)

    selected_ids = list(sources)[: max(0, int(max_samples))]
    cards = [
        _render_sample(
            source=sources[source_id],
            annotation=annotations.get(source_id),
            views=views_by_source.get(source_id, []),
            embed_images=embed_images,
        )
        for source_id in selected_ids
    ]
    status_counts = Counter(
        str((annotations.get(source_id) or {}).get("status") or "missing")
        for source_id in selected_ids
    )
    rendered_views = [view for source_id in selected_ids for view in views_by_source.get(source_id, [])]
    summary = {
        "sources": len(selected_ids),
        "accepted": status_counts.get("accepted", 0),
        "rejected": status_counts.get("rejected", 0),
        "errors": status_counts.get("error", 0),
        "missing_annotations": status_counts.get("missing", 0),
        "full_cot_views": sum(view.get("view_type") == "full_cot" for view in rendered_views),
        "system1_views": sum(view.get("trajectory_ego_t") is not None for view in rendered_views),
    }
    page = _render_page(
        title=title,
        cards=cards,
        summary=summary,
        source_jsonl=str(source_jsonl),
        annotation_jsonl=str(annotation_jsonl),
        views_jsonl=str(views_jsonl),
    )
    output = Path(output_html).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return {"output_html": str(output), **summary}


def main(argv: Sequence[str] | None = None) -> None:
    """Build the requested page and print a JSON summary.

    Args:
        argv: Optional command-line arguments.

    Returns:
        ``None``.
    """

    args = parse_args(argv)
    result = build_reasoning_v2_review_html(
        source_jsonl=args.source_jsonl,
        annotation_jsonl=args.annotation_jsonl,
        views_jsonl=args.views_jsonl,
        output_html=args.output_html,
        title=args.title,
        max_samples=args.max_samples,
        embed_images=not args.link_images,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _index_unique(rows: Iterable[dict[str, Any]], *, key: str, artifact: str) -> dict[str, dict[str, Any]]:
    """Index rows and reject duplicate stable ids."""

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "").strip()
        if not value:
            raise ValueError(f"{artifact} row is missing {key}")
        if value in result:
            raise ValueError(f"duplicate {artifact} {key}: {value}")
        result[value] = row
    return result


def _index_latest(rows: Iterable[dict[str, Any]], *, key: str) -> dict[str, dict[str, Any]]:
    """Index append-only rows, keeping the last record for each id."""

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "").strip()
        if value:
            result[value] = row
    return result


def _render_page(
    *,
    title: str,
    cards: list[str],
    summary: Mapping[str, Any],
    source_jsonl: str,
    annotation_jsonl: str,
    views_jsonl: str,
) -> str:
    """Render the complete standalone HTML document."""

    metrics = "".join(
        f"<div class='metric'><strong>{_e(value)}</strong><span>{_e(label)}</span></div>"
        for label, value in (
            ("Sources", summary["sources"]),
            ("Accepted", summary["accepted"]),
            ("Full CoT", summary["full_cot_views"]),
            ("System1", summary["system1_views"]),
        )
    )
    provenance = _e(
        f"source={source_jsonl} | annotation={annotation_jsonl} | views={views_jsonl}"
    )
    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{_e(title)}</title>
<style>
:root{{--ink:#18212b;--muted:#64707d;--line:#d8dee5;--soft:#f4f6f8;--blue:#1769aa;--green:#18794e;--amber:#9a6700;--red:#c62828;--panel:#fff}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);background:#eef1f4;font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.97);border-bottom:1px solid var(--line);padding:14px 24px}}
.header-row{{max-width:1500px;margin:auto;display:flex;align-items:center;gap:18px;justify-content:space-between}} h1{{font-size:20px;margin:0}} .sub{{color:var(--muted);font-size:12px;overflow-wrap:anywhere}}
.controls{{display:flex;gap:8px;min-width:360px}} input,select{{height:34px;border:1px solid #aeb8c2;background:#fff;padding:0 10px;font:inherit}} input{{flex:1}}
main{{max-width:1500px;margin:20px auto;padding:0 20px 60px}} .metrics{{display:grid;grid-template-columns:repeat(4,minmax(110px,1fr));gap:8px;margin-bottom:14px}}
.metric{{background:#fff;border:1px solid var(--line);padding:11px 13px;border-radius:6px}} .metric strong{{display:block;font-size:21px}} .metric span{{color:var(--muted);font-size:12px}}
.sample{{background:var(--panel);border:1px solid #cbd3dc;border-radius:6px;margin:0 0 18px;overflow:hidden}} .sample-head{{display:flex;justify-content:space-between;gap:18px;padding:14px 18px;background:#f8fafb;border-bottom:1px solid var(--line)}}
.sample h2{{font-size:16px;margin:0 0 3px}} .badges{{display:flex;gap:6px;align-items:flex-start;flex-wrap:wrap}} .badge{{border:1px solid;padding:2px 7px;font-size:11px;font-weight:650;background:#fff}}
.accepted{{color:var(--green);border-color:#8bc9ad;background:#edf8f2}} .rejected,.error{{color:var(--red);border-color:#e2a4a4;background:#fff1f1}} .missing{{color:var(--amber);border-color:#dbc27b;background:#fff8df}}
.sample-body{{padding:16px 18px}} .image-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;margin:0 0 16px}} figure{{margin:0;border:1px solid var(--line);background:#111}} figure img{{display:block;width:100%;height:330px;object-fit:contain}} figcaption{{padding:7px 9px;color:#fff;background:#202832;font-size:12px}}
.decision{{display:grid;grid-template-columns:repeat(5,minmax(110px,1fr));border:1px solid var(--line);margin:12px 0 16px}} .decision div{{padding:8px 10px;border-right:1px solid var(--line)}} .decision div:last-child{{border:0}} .decision small{{display:block;color:var(--muted)}}
.two-col{{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(340px,.65fr);gap:16px}} .section{{border-top:1px solid var(--line);padding:13px 0}} .section:first-child{{border-top:0;padding-top:0}} h3{{font-size:13px;margin:0 0 7px;color:#34404b;text-transform:uppercase}}
pre{{margin:0;padding:10px 12px;background:var(--soft);border:1px solid #e0e5ea;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}}
table{{width:100%;border-collapse:collapse;font-size:12px}} th,td{{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:7px 8px}} th{{background:var(--soft);color:#4d5965}}
.trajectory{{border:1px solid var(--line);background:#fbfcfd;padding:8px}} .trajectory svg{{display:block;width:100%;height:auto}} details{{margin-top:9px}} summary{{cursor:pointer;color:var(--blue);font-weight:650}} .notice{{padding:9px 11px;border-left:3px solid var(--amber);background:#fff8df;margin:8px 0}}
.empty{{color:var(--muted);font-style:italic}} .hidden{{display:none!important}}
@media(max-width:900px){{header{{position:static}}.header-row{{display:block}}.controls{{min-width:0;margin-top:10px}}.two-col{{grid-template-columns:1fr}}.metrics,.decision{{grid-template-columns:repeat(2,1fr)}}figure img{{height:auto;max-height:420px}}}}
</style>
</head>
<body>
<header><div class='header-row'><div><h1>{_e(title)}</h1><div class='sub'>{provenance}</div></div><div class='controls'><input id='search' type='search' placeholder='Filter id, task, evidence...'><select id='status'><option value=''>All status</option><option>accepted</option><option>rejected</option><option>error</option><option>missing</option></select></div></div></header>
<main><div class='metrics'>{metrics}</div>{''.join(cards) if cards else "<p class='empty'>No rows rendered.</p>"}</main>
<script>
const search=document.getElementById('search');const status=document.getElementById('status');
function applyFilter(){{const q=search.value.trim().toLowerCase();const s=status.value;document.querySelectorAll('.sample').forEach(el=>{{const show=(!q||el.dataset.search.includes(q))&&(!s||el.dataset.status===s);el.classList.toggle('hidden',!show)}})}}
search.addEventListener('input',applyFilter);status.addEventListener('change',applyFilter);
</script>
</body></html>"""


def _render_sample(
    *,
    source: Mapping[str, Any],
    annotation: Mapping[str, Any] | None,
    views: list[dict[str, Any]],
    embed_images: bool,
) -> str:
    """Render one joined source/annotation/view sample section."""

    source_id = str(source.get("source_group_id") or "unknown")
    status = str((annotation or {}).get("status") or "missing")
    target = (annotation or {}).get("target") if isinstance((annotation or {}).get("target"), Mapping) else {}
    reasoning = target.get("reasoning") if isinstance(target.get("reasoning"), Mapping) else {}
    finish = target.get("finish") if isinstance(target.get("finish"), Mapping) else {}
    full_view = next((view for view in views if view.get("view_type") == "full_cot"), None)
    images = _render_images(source, embed_images=embed_images)
    decision = _render_decision(finish=finish, reasoning=reasoning)
    reasoning_sections = "".join(
        _render_json_section(label, reasoning[key])
        for label, key in (
            ("Task Grounding", "task_grounding"),
            ("Evidence Grounding", "evidence_grounding"),
            ("Cross-modal Alignment", "cross_modal_alignment"),
            ("Constraint Applicability", "constraint_applicability"),
            ("Decision", "decision"),
        )
        if key in reasoning
    )
    if finish:
        reasoning_sections += _render_json_section("Finish", finish)
    evidence = _render_evidence_catalog(source.get("evidence_catalog"))
    trajectory = _render_trajectory(full_view or {})
    validation = _render_validation(annotation)
    searchable = " ".join(
        [source_id, status, json.dumps(source.get("available_modalities") or []), json.dumps(target, ensure_ascii=False), json.dumps(views, ensure_ascii=False)]
    ).lower()
    raw = {
        "source": source,
        "annotation": annotation,
        "views": views,
    }
    return f"""<article class='sample' data-status='{_e(status)}' data-search='{_e(searchable)}'>
<div class='sample-head'><div><h2>{_e(source_id)}</h2><div class='sub'>{_e(source.get('dataset_source'))} · {_e(source.get('source_record_id'))} · fingerprint {_e(str(source.get('source_fingerprint') or '')[:12])}</div></div><div class='badges'><span class='badge {_e(status)}'>{_e(status)}</span><span class='badge'>{_e((annotation or {}).get('teacher_provider') or 'no provider')}</span><span class='badge'>{_e((annotation or {}).get('teacher_model') or 'no model')}</span><span class='badge'>attempts {_e((annotation or {}).get('attempts') or 0)}</span></div></div>
<div class='sample-body'>
<div class='section'><h3>Instruction</h3><div>{_e(source.get('instruction'))}</div><div class='sub'>travel_mode={_e(source.get('travel_mode'))} · modalities={_e(', '.join(source.get('available_modalities') or []))}</div></div>
{images}{decision}
<div class='two-col'><div><div class='section'><h3>ReasoningTraceV2</h3>{reasoning_sections or "<div class='empty'>No accepted target.</div>"}</div></div><aside><div class='section'><h3>System1 Trajectory</h3>{trajectory}</div><div class='section'><h3>Evidence Catalog</h3>{evidence}</div><div class='section'><h3>Validation & Repair</h3>{validation}</div></aside></div>
<details><summary>Raw joined artifacts</summary>{_json_pre(raw)}</details>
</div></article>"""


def _render_images(source: Mapping[str, Any], *, embed_images: bool) -> str:
    """Render ordered image inputs with their explicit roles."""

    figures: list[str] = []
    images = source.get("images") if isinstance(source.get("images"), list) else []
    roles = source.get("image_roles") if isinstance(source.get("image_roles"), list) else []
    for index, value in enumerate(images):
        role = str(roles[index]) if index < len(roles) else f"image.{index}"
        src = _image_src(value, embed=embed_images)
        if src:
            figures.append(f"<figure><img src='{_e(src)}' alt='{_e(role)}'><figcaption>{_e(role)} · {_e(value)}</figcaption></figure>")
        else:
            figures.append(f"<figure><div class='notice'>Image unavailable: {_e(value)}</div><figcaption>{_e(role)}</figcaption></figure>")
    return f"<div class='image-grid'>{''.join(figures)}</div>" if figures else ""


def _render_decision(*, finish: Mapping[str, Any], reasoning: Mapping[str, Any]) -> str:
    """Render the compact symbolic action strip."""

    decision = reasoning.get("decision") if isinstance(reasoning.get("decision"), Mapping) else {}
    fields = (
        ("Decision", finish.get("decision_type")),
        ("Motion", finish.get("motion") or decision.get("motion")),
        ("Speed", finish.get("speed") or decision.get("speed")),
        ("Social", finish.get("social_action") or decision.get("social_action")),
        ("Confidence", finish.get("confidence_level") or decision.get("confidence")),
    )
    return "<div class='decision'>" + "".join(
        f"<div><small>{_e(label)}</small><strong>{_e(value or 'unknown')}</strong></div>" for label, value in fields
    ) + "</div>"


def _render_json_section(label: str, value: Any) -> str:
    """Render one structured reasoning section."""

    return f"<div class='section'><h3>{_e(label)}</h3>{_json_pre(value)}</div>"


def _render_evidence_catalog(value: Any) -> str:
    """Render addressable evidence refs as a compact table."""

    if not isinstance(value, list) or not value:
        return "<div class='empty'>No evidence catalog.</div>"
    rows = "".join(
        f"<tr><td><code>{_e(item.get('ref'))}</code></td><td>{_e(item.get('modality'))}</td><td>{_e(item.get('summary'))}</td></tr>"
        for item in value
        if isinstance(item, Mapping)
    )
    return f"<table><thead><tr><th>Ref</th><th>Modality</th><th>Summary</th></tr></thead><tbody>{rows}</tbody></table>"


def _render_validation(annotation: Mapping[str, Any] | None) -> str:
    """Render hard errors, warnings, telemetry, and repair attempts."""

    if not annotation:
        return "<div class='notice'>Annotation missing.</div>"
    errors = annotation.get("errors") or []
    warnings = annotation.get("warnings") or []
    metadata = annotation.get("metadata") if isinstance(annotation.get("metadata"), Mapping) else {}
    telemetry = annotation.get("telemetry") if isinstance(annotation.get("telemetry"), Mapping) else {}
    compact = {
        "status": annotation.get("status"),
        "attempts": annotation.get("attempts"),
        "errors": errors,
        "warnings": warnings,
        "fallback_used": metadata.get("fallback_used"),
        "repair_history": metadata.get("repair_history") or [],
        "roundtrip_ms": telemetry.get("roundtrip_ms"),
        "usage": (telemetry.get("response") or {}).get("usage") if isinstance(telemetry.get("response"), Mapping) else None,
    }
    return _json_pre(compact)


def _render_trajectory(view: Mapping[str, Any]) -> str:
    """Render the cumulative current-ego FLU trajectory and contract."""

    points = view.get("trajectory_ego_t") if isinstance(view.get("trajectory_ego_t"), list) else []
    valid: list[tuple[float, float, float]] = []
    for item in points:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        try:
            point = (float(item[0]), float(item[1]), float(item[2]))
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in point):
            valid.append(point)
    if not valid:
        return "<div class='empty'>No System1 trajectory for this view.</div>"

    max_forward = max(1.0, max(point[0] for point in valid) * 1.15)
    max_lateral = max(0.5, max(abs(point[1]) for point in valid) * 1.25)
    origin_x, origin_y = 260.0, 218.0

    def project(point: tuple[float, float, float]) -> tuple[float, float]:
        forward, left, _ = point
        return origin_x - (left / max_lateral) * 205.0, origin_y - (forward / max_forward) * 176.0

    projected = [(origin_x, origin_y), *(project(point) for point in valid)]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in projected)
    circles = "".join(
        f"<circle cx='{x:.2f}' cy='{y:.2f}' r='5' fill='{('#1769aa' if index < len(projected) - 1 else '#18794e')}'><title>step {index}: {valid[index - 1] if index else (0, 0, 0)}</title></circle>"
        for index, (x, y) in enumerate(projected)
    )
    svg = f"""<div class='trajectory'><svg viewBox='0 0 520 250' role='img' aria-label='System1 FLU trajectory'>
<rect x='0' y='0' width='520' height='250' fill='#fbfcfd'/><line x1='260' y1='18' x2='260' y2='228' stroke='#aeb8c2'/><line x1='30' y1='218' x2='490' y2='218' stroke='#aeb8c2'/>
<text x='270' y='28' font-size='12' fill='#64707d'>forward +x</text><text x='34' y='210' font-size='12' fill='#64707d'>left +y</text><text x='438' y='210' font-size='12' fill='#64707d'>right -y</text>
<polyline points='{polyline}' fill='none' stroke='#1769aa' stroke-width='4' stroke-linecap='round' stroke-linejoin='round'/>{circles}</svg></div>"""
    policy = view.get("trajectory_policy") or {}
    mask = view.get("action_mask")
    return svg + _json_pre({"trajectory_ego_t": points, "trajectory_policy": policy, "action_mask": mask})


def _image_src(value: Any, *, embed: bool) -> str:
    """Resolve HTTP/data URLs or encode one local image."""

    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith(("http://", "https://", "data:image/")):
        return text
    path = Path(text).expanduser()
    if not path.is_file():
        return ""
    if not embed:
        return path.resolve().as_uri()
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _json_pre(value: Any) -> str:
    """Render JSON-compatible data in an escaped pre block."""

    return _text_pre(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False))


def _text_pre(value: Any) -> str:
    """Render escaped preformatted text."""

    return f"<pre>{_e(value)}</pre>"


def _e(value: Any) -> str:
    """Escape one value for HTML text or attribute use."""

    return html.escape(str(value if value is not None else ""), quote=True)


if __name__ == "__main__":
    main()
