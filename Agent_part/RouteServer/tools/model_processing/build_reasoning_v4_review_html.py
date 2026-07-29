#!/usr/bin/env python3
"""Build a self-contained Reasoning V4.6 human-review page."""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.contracts.reasoning_dataset_v4 import (
    ReasoningAnnotationV4,
    ReasoningSourceV4,
)
from mcp_server.data.jsonl_io import iter_jsonl
from mcp_server.data.reasoning_v4_reconciliation import (
    action_reconciliation_review_priority_v4,
)
from mcp_server.data.reasoning_v4_state import (
    derive_reasoning_release_state_v4,
    load_latest_action_resolutions_v4,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse review-builder arguments.

    Args:
        argv: Optional explicit CLI arguments.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-jsonl",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--annotation-jsonl",
        required=True,
        type=Path,
    )
    parser.add_argument("--views-jsonl", type=Path)
    parser.add_argument("--resolution-jsonl", type=Path)
    parser.add_argument(
        "--output-html",
        required=True,
        type=Path,
    )
    parser.add_argument("--report-json", type=Path)
    parser.add_argument(
        "--title",
        default=(
            "UrbanNav Reasoning V4.6 "
            "Action Reconciliation Review"
        ),
    )
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument(
        "--link-images",
        action="store_true",
        help="Link local files instead of embedding image bytes.",
    )
    return parser.parse_args(argv)


def build_reasoning_v4_review_html(
    *,
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    output_html: str | Path,
    views_jsonl: str | Path | None = None,
    resolution_jsonl: str | Path | None = None,
    report_json: str | Path | None = None,
    title: str = (
        "UrbanNav Reasoning V4.6 "
        "Action Reconciliation Review"
    ),
    max_samples: int = 100,
    embed_images: bool = True,
) -> dict[str, Any]:
    """Join sources, annotations, resolutions, and optional views.

    Args:
        source_jsonl: Frozen V4 source manifest.
        annotation_jsonl: Append-only annotation manifest.
        output_html: Self-contained review-page destination.
        views_jsonl: Optional exported preview/formal views.
        resolution_jsonl: Optional append-only human resolutions.
        report_json: Optional machine-readable summary destination.
        title: Page title.
        max_samples: Maximum source cards.
        embed_images: Embed local media bytes when true.

    Returns:
        Exact review summary.
    """

    sources = _source_index(source_jsonl)
    annotations = _annotation_index(annotation_jsonl)
    resolutions = load_latest_action_resolutions_v4(
        resolution_jsonl
    )
    views_by_source: dict[str, list[dict[str, Any]]] = (
        defaultdict(list)
    )
    if (
        views_jsonl is not None
        and Path(views_jsonl).expanduser().is_file()
    ):
        for view in iter_jsonl(views_jsonl):
            source_id = str(
                view.get("source_group_id") or ""
            )
            if source_id:
                views_by_source[source_id].append(view)

    metrics: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    cards: list[str] = []
    selected_ids = list(sources)[
        : max(0, int(max_samples))
    ]
    for source_id in selected_ids:
        source = sources[source_id]
        annotation = annotations.get(source_id)
        metrics["sources"] += 1
        if annotation is None:
            counts["missing_annotation"] += 1
            cards.append(
                _render_missing_card(source, embed_images)
            )
            continue
        audit = annotation.action_reconciliation
        resolution = (
            resolutions.get(audit.audit_id)
            if audit is not None
            else None
        )
        state = derive_reasoning_release_state_v4(
            source,
            annotation,
            resolution=resolution,
        )
        metrics["record_complete"] += int(
            state.record_complete
        )
        metrics["all_sections_accepted"] += int(
            state.all_sections_accepted
        )
        metrics["candidate_structurally_valid"] += int(
            state.candidate_structurally_valid
        )
        metrics["pending_human"] += int(
            state.review_status == "pending_human"
        )
        metrics["formal_training_eligible"] += int(
            state.formal_training_eligible
        )
        counts[
            f"review_status:{state.review_status}"
        ] += 1
        counts[
            "candidate_structure:"
            f"{annotation.candidate_structure_record.status}"
        ] += 1
        if audit is not None:
            counts[
                "review_priority:"
                f"{action_reconciliation_review_priority_v4(audit)}"
            ] += 1
        cards.append(
            _render_sample_card(
                source,
                annotation,
                state=state,
                resolution=(
                    resolution.model_dump(
                        mode="json",
                        exclude_none=True,
                    )
                    if resolution is not None
                    else None
                ),
                views=views_by_source.get(source_id, []),
                embed_images=embed_images,
            )
        )

    summary = {
        "schema_version": (
            "urbannav_reasoning_v4_6_review_report_v1"
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "views_jsonl": (
            str(views_jsonl)
            if views_jsonl is not None
            else None
        ),
        "resolution_jsonl": (
            str(resolution_jsonl)
            if resolution_jsonl is not None
            else None
        ),
        "output_html": str(output_html),
        "metrics": {
            name: int(metrics[name])
            for name in (
                "sources",
                "record_complete",
                "all_sections_accepted",
                "candidate_structurally_valid",
                "pending_human",
                "formal_training_eligible",
            )
        },
        "counts": dict(counts),
        "sample_cards": len(cards),
        "view_rows": sum(
            len(views_by_source[source_id])
            for source_id in selected_ids
        ),
    }
    document = _render_document(
        title=title,
        summary=summary,
        cards=cards,
    )
    destination = Path(output_html).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    if report_json is not None:
        report_path = Path(report_json).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return summary


def _source_index(
    path: str | Path,
) -> dict[str, ReasoningSourceV4]:
    """Load unique sources in manifest order."""

    result: dict[str, ReasoningSourceV4] = {}
    for row in iter_jsonl(path):
        source = ReasoningSourceV4.model_validate(row)
        source_id = source.observation.source_group_id
        if source_id in result:
            raise ValueError(
                f"duplicate source_group_id: {source_id}"
            )
        result[source_id] = source
    return result


def _annotation_index(
    path: str | Path,
) -> dict[str, ReasoningAnnotationV4]:
    """Load latest append-only annotation per source group."""

    result: dict[str, ReasoningAnnotationV4] = {}
    for row in iter_jsonl(path):
        annotation = ReasoningAnnotationV4.model_validate(
            row
        )
        result[annotation.source_group_id] = annotation
    return result


def _render_document(
    *,
    title: str,
    summary: Mapping[str, Any],
    cards: list[str],
) -> str:
    """Render one standalone review document."""

    metrics = summary["metrics"]
    labels = (
        ("Sources", "sources"),
        ("Record Complete", "record_complete"),
        ("All Sections Accepted", "all_sections_accepted"),
        (
            "Candidate Structurally Valid",
            "candidate_structurally_valid",
        ),
        ("Pending Human", "pending_human"),
        ("Formal Training Eligible", "formal_training_eligible"),
    )
    metric_html = "".join(
        "<div class='metric'><span>"
        f"{html.escape(label)}</span><strong>{metrics[key]}</strong>"
        "</div>"
        for label, key in labels
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#eef2f7;--card:#fff;--ink:#182235;--muted:#66758c;
--line:#d8e0eb;--good:#16794b;--bad:#b33b32;--warn:#aa6b00}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 system-ui,sans-serif}} main{{max-width:1500px;margin:auto;padding:24px}}
h1{{margin:0 0 14px}} .metrics{{display:grid;grid-template-columns:repeat(6,1fr);
gap:10px;margin:14px 0 22px}} .metric,.card{{background:var(--card);
border:1px solid var(--line);border-radius:12px}} .metric{{padding:12px}}
.metric span{{display:block;color:var(--muted);font-size:12px}}
.metric strong{{font-size:24px}} .card{{padding:16px;margin:14px 0}}
.bad{{color:var(--bad)}} .good{{color:var(--good)}} .warn{{color:var(--warn)}}
.badges{{display:flex;gap:7px;flex-wrap:wrap}} .badge{{border:1px solid var(--line);
border-radius:99px;padding:3px 9px;background:#f8fafc}} .images{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px;margin:12px 0}}
.images img{{width:100%;max-height:260px;object-fit:contain;background:#111;
border-radius:8px}} .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
gap:10px}} details{{border:1px solid var(--line);border-radius:8px;padding:8px}}
summary{{cursor:pointer;font-weight:650}} pre{{white-space:pre-wrap;word-break:break-word;
font:12px/1.45 ui-monospace,monospace;background:#f6f8fb;padding:10px;border-radius:7px}}
@media(max-width:900px){{.metrics{{grid-template-columns:repeat(2,1fr)}}
.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<h1>{html.escape(title)}</h1>
<div class="metrics">{metric_html}</div>
{''.join(cards)}
</main></body></html>"""


def _render_missing_card(
    source: ReasoningSourceV4,
    embed_images: bool,
) -> str:
    """Render one source without annotation."""

    return (
        "<article class='card'><h2>"
        f"{html.escape(source.observation.source_group_id)}"
        "</h2><p class='bad'>Missing annotation</p>"
        f"{_render_images(source, embed_images)}"
        "</article>"
    )


def _render_sample_card(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
    *,
    state: Any,
    resolution: dict[str, Any] | None,
    views: list[dict[str, Any]],
    embed_images: bool,
) -> str:
    """Render one joined review card."""

    proposal = (
        annotation.decision_proposal.model_dump(
            mode="json",
            exclude_none=True,
        )
        if annotation.decision_proposal is not None
        else None
    )
    strong = (
        source.direction_supervision.model_dump(
            mode="json",
            exclude_none=True,
        )
        if source.direction_supervision is not None
        else None
    )
    candidate = (
        annotation.composer_candidate.model_dump(
            mode="json",
            exclude_none=True,
        )
        if annotation.composer_candidate is not None
        else None
    )
    audit = (
        annotation.action_reconciliation.model_dump(
            mode="json",
            exclude_none=True,
        )
        if annotation.action_reconciliation is not None
        else None
    )
    badges = (
        ("record", annotation.record_status),
        (
            "sections",
            "accepted"
            if state.all_sections_accepted
            else "blocked",
        ),
        (
            "candidate_structure",
            annotation.candidate_structure_record.status,
        ),
        ("review", state.review_status),
        (
            "formal_training_eligible",
            state.formal_training_eligible,
        ),
    )
    badge_html = "".join(
        "<span class='badge'>"
        f"{html.escape(name)}={html.escape(str(value))}"
        "</span>"
        for name, value in badges
    )
    payloads = (
        (
            "Grounding outputs",
            annotation.grounding_outputs.model_dump(
                mode="json",
                exclude_none=True,
            ),
        ),
        (
            "Proposal / strong / candidate",
            {
                "teacher_proposal": proposal,
                "strong_direction": strong,
                "composer_candidate": candidate,
            },
        ),
        ("Action audit", audit),
        (
            "Candidate structure and blockers",
            {
                "candidate_structure_record": (
                    annotation.candidate_structure_record.model_dump(
                        mode="json",
                        exclude_none=True,
                    )
                ),
                "release_blockers": list(state.blockers),
                "errors": annotation.errors,
                "warnings": annotation.warnings,
            },
        ),
        ("Human resolution", resolution),
        ("Exported views", views),
    )
    details = "".join(
        "<details><summary>"
        f"{html.escape(label)}</summary>"
        f"<pre>{_json_html(payload)}</pre></details>"
        for label, payload in payloads
    )
    return (
        "<article class='card'><h2>"
        f"{html.escape(source.observation.source_group_id)}"
        "</h2><div class='badges'>"
        f"{badge_html}</div>"
        f"{_render_images(source, embed_images)}"
        f"<div class='grid'>{details}</div></article>"
    )


def _render_images(
    source: ReasoningSourceV4,
    embed_images: bool,
) -> str:
    """Render ordered source images."""

    items: list[str] = []
    for image_path, role in zip(
        source.observation.images,
        source.observation.image_roles,
    ):
        src = _image_src(image_path, embed_images)
        items.append(
            "<figure><img loading='lazy' src='"
            f"{html.escape(src, quote=True)}' alt='"
            f"{html.escape(role, quote=True)}'>"
            f"<figcaption>{html.escape(role)}</figcaption></figure>"
        )
    return f"<div class='images'>{''.join(items)}</div>"


def _image_src(path: str, embed: bool) -> str:
    """Return embedded data URI or escaped local path."""

    value = Path(path).expanduser()
    if not embed or not value.is_file():
        return str(value)
    mime = (
        mimetypes.guess_type(value.name)[0]
        or "application/octet-stream"
    )
    encoded = base64.b64encode(value.read_bytes()).decode(
        "ascii"
    )
    return f"data:{mime};base64,{encoded}"


def _json_html(value: Any) -> str:
    """Render escaped pretty JSON."""

    return html.escape(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Build the review page.

    Args:
        argv: Optional explicit CLI arguments.
    """

    args = parse_args(argv)
    report = build_reasoning_v4_review_html(
        source_jsonl=args.source_jsonl,
        annotation_jsonl=args.annotation_jsonl,
        views_jsonl=args.views_jsonl,
        resolution_jsonl=args.resolution_jsonl,
        output_html=args.output_html,
        report_json=args.report_json,
        title=args.title,
        max_samples=args.max_samples,
        embed_images=not args.link_images,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
