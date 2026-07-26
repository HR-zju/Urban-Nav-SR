#!/usr/bin/env python3
"""Build a self-contained human-review page for Navigation VQA V2."""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mcp_server.contracts.vqa_dataset_v2 import (
    VQAAnnotationV2,
    VQASourceV2,
    VQATrainingViewV2,
)
from mcp_server.data.jsonl_io import iter_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse Navigation VQA V2 review-builder arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--views-jsonl", type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--title", default="UrbanNav Navigation VQA V2 Review")
    parser.add_argument("--max-annotations", type=int, default=100)
    parser.add_argument(
        "--link-images",
        action="store_true",
        help="Link local files instead of embedding image bytes.",
    )
    return parser.parse_args(argv)


def build_vqa_v2_review_html(
    *,
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    output_html: str | Path,
    views_jsonl: str | Path | None = None,
    report_json: str | Path | None = None,
    title: str = "UrbanNav Navigation VQA V2 Review",
    max_annotations: int = 100,
    embed_images: bool = True,
) -> dict[str, Any]:
    """Join typed VQA sources, annotations, and optional training views.

    Args:
        source_jsonl: Frozen VQA source rows.
        annotation_jsonl: Accepted typed annotation rows.
        output_html: Destination standalone HTML file.
        views_jsonl: Optional exported training-view rows.
        report_json: Optional machine-readable review summary.
        title: Review page title.
        max_annotations: Maximum accepted annotations to render.
        embed_images: Embed local image bytes when true.

    Returns:
        Counts and provenance for the generated review page.

    Raises:
        ValueError: If source, annotation, or view identities are invalid.
    """

    sources = _source_index(iter_jsonl(source_jsonl))
    annotations = [
        VQAAnnotationV2.model_validate(row)
        for row in iter_jsonl(annotation_jsonl)
    ][: max(0, int(max_annotations))]
    views = _view_index(
        iter_jsonl(views_jsonl)
        if views_jsonl is not None
        and Path(views_jsonl).expanduser().is_file()
        else ()
    )
    task_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    rendered_source_ids: set[str] = set()
    cards: list[str] = []
    rendered_images = 0
    rendered_views = 0
    seen_annotations: set[str] = set()
    for annotation in annotations:
        if annotation.annotation_id in seen_annotations:
            raise ValueError(
                f"duplicate annotation id: {annotation.annotation_id}"
            )
        seen_annotations.add(annotation.annotation_id)
        try:
            source = sources[annotation.source_group_id]
        except KeyError as exc:
            raise ValueError(
                "annotation source_group_id is absent from sources: "
                f"{annotation.source_group_id}"
            ) from exc
        if annotation.source_fingerprint != source.source_fingerprint:
            raise ValueError("annotation/source fingerprint mismatch")
        view = views.get(annotation.annotation_id)
        if view is not None and (
            view.source_group_id != annotation.source_group_id
            or view.question.task_id != annotation.question.task_id
        ):
            raise ValueError("annotation/training-view identity mismatch")
        task_counts[annotation.question.task_id] += 1
        label_counts[annotation.label_source] += 1
        rendered_source_ids.add(annotation.source_group_id)
        rendered_images += len(source.observation.images)
        rendered_views += int(view is not None)
        cards.append(
            _render_annotation(
                source=source,
                annotation=annotation,
                view=view,
                embed_images=embed_images,
            )
        )
    summary: dict[str, Any] = {
        "schema_version": "urbannav_navigation_vqa_v2_review_report_v1",
        "source_rows": len(sources),
        "rendered_source_groups": len(rendered_source_ids),
        "rendered_annotations": len(annotations),
        "rendered_images": rendered_images,
        "rendered_views": rendered_views,
        "task_counts": dict(sorted(task_counts.items())),
        "label_source_counts": dict(sorted(label_counts.items())),
        "embedded_images": bool(embed_images),
        "output_html": str(Path(output_html).expanduser()),
    }
    page = _render_page(
        title=title,
        cards=cards,
        summary=summary,
        provenance={
            "source_jsonl": str(source_jsonl),
            "annotation_jsonl": str(annotation_jsonl),
            "views_jsonl": str(views_jsonl) if views_jsonl else None,
        },
    )
    destination = Path(output_html).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    if report_json is not None:
        report_path = Path(report_json).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Build a standalone VQA V2 review page and print its summary."""

    args = parse_args(argv)
    result = build_vqa_v2_review_html(
        source_jsonl=args.source_jsonl,
        annotation_jsonl=args.annotation_jsonl,
        views_jsonl=args.views_jsonl,
        output_html=args.output_html,
        report_json=args.report_json,
        title=args.title,
        max_annotations=args.max_annotations,
        embed_images=not args.link_images,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _source_index(
    rows: Iterable[dict[str, Any]],
) -> dict[str, VQASourceV2]:
    """Load unique validated sources by source-group identity."""

    result: dict[str, VQASourceV2] = {}
    for row in rows:
        source = VQASourceV2.model_validate(row)
        key = source.observation.source_group_id
        if key in result:
            raise ValueError(f"duplicate VQA source id: {key}")
        result[key] = source
    return result


def _view_index(
    rows: Iterable[dict[str, Any]],
) -> dict[str, VQATrainingViewV2]:
    """Load unique validated training views by annotation identity."""

    result: dict[str, VQATrainingViewV2] = {}
    for row in rows:
        view = VQATrainingViewV2.model_validate(row)
        if view.annotation_id in result:
            raise ValueError(
                f"duplicate VQA training view: {view.annotation_id}"
            )
        result[view.annotation_id] = view
    return result


def _render_page(
    *,
    title: str,
    cards: list[str],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    """Render the complete standalone review document."""

    metrics = (
        ("Annotations", summary["rendered_annotations"]),
        ("Source groups", summary["rendered_source_groups"]),
        ("Tasks", len(summary["task_counts"])),
        ("Derived GT", summary["label_source_counts"].get("derived_gt", 0)),
        ("LVLM weak", summary["label_source_counts"].get("lvlm_weak", 0)),
    )
    metric_html = "".join(
        "<div class='metric'><strong>"
        f"{_escape(value)}</strong><span>{_escape(label)}</span></div>"
        for label, value in metrics
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape(title)}</title>
<style>
:root{{--ink:#17212b;--muted:#64717e;--line:#d6dde5;--soft:#f3f6f8;--gt:#18794e;--weak:#7c3aed}}
*{{box-sizing:border-box}}body{{margin:0;background:#edf1f4;color:var(--ink);font:14px/1.5 Inter,system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:4;padding:14px 22px;background:#fffffff5;border-bottom:1px solid var(--line)}}
.head{{max-width:1500px;margin:auto;display:flex;gap:20px;justify-content:space-between;align-items:center}}h1{{font-size:20px;margin:0}}
.muted{{color:var(--muted);font-size:12px;overflow-wrap:anywhere}}input,select{{height:34px;border:1px solid #aab6c2;background:white;padding:0 10px}}
main{{max-width:1500px;margin:18px auto;padding:0 18px 60px}}.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:14px}}
.metric,.sample{{background:white;border:1px solid var(--line);border-radius:7px}}.metric{{padding:10px 13px}}.metric strong{{display:block;font-size:21px}}.metric span{{font-size:12px;color:var(--muted)}}
.sample{{margin-bottom:18px;overflow:hidden}}.sample-head{{padding:13px 16px;background:#f8fafb;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:18px}}
h2{{font-size:16px;margin:0 0 3px}}.badge{{display:inline-block;border:1px solid;padding:2px 7px;font-size:11px;font-weight:700;background:#fff}}
.derived_gt{{color:var(--gt);border-color:#8bc9ad;background:#edf8f2}}.lvlm_weak{{color:var(--weak);border-color:#c4b5fd;background:#f5f3ff}}
.body{{padding:15px 16px}}.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}}
figure{{margin:0;background:#10161c;border:1px solid var(--line)}}figure img{{width:100%;height:300px;display:block;object-fit:contain}}figcaption{{padding:6px 8px;color:#fff;font-size:11px;overflow-wrap:anywhere}}
.grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(360px,.75fr);gap:14px;margin-top:14px}}section{{border-top:1px solid var(--line);padding:12px 0}}section:first-child{{border-top:0}}
h3{{margin:0 0 7px;font-size:12px;text-transform:uppercase;color:#3d4a57}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--soft);border:1px solid #e0e6eb;padding:9px 11px;margin:0;font:12px/1.45 ui-monospace,monospace}}
details{{margin-top:8px}}summary{{cursor:pointer;font-weight:700;color:#1769aa}}.hidden{{display:none!important}}
@media(max-width:900px){{header{{position:static}}.head,.grid{{display:block}}.metrics{{grid-template-columns:repeat(2,1fr)}}figure img{{height:auto;max-height:420px}}}}
</style></head><body>
<header><div class="head"><div><h1>{_escape(title)}</h1><div class="muted">{_escape(json.dumps(provenance, ensure_ascii=False))}</div></div>
<div><input id="q" type="search" placeholder="Filter task, question, answer"><select id="s"><option value="">All label sources</option><option>derived_gt</option><option>lvlm_weak</option><option>external_gt</option></select></div></div></header>
<main><div class="metrics">{metric_html}</div>{''.join(cards) or '<p>No accepted annotations.</p>'}</main>
<script>const q=document.querySelector('#q'),s=document.querySelector('#s');function f(){{const a=q.value.toLowerCase(),b=s.value;document.querySelectorAll('.sample').forEach(x=>x.classList.toggle('hidden',(a&&!x.dataset.search.includes(a))||(b&&x.dataset.source!==b)))}}q.addEventListener('input',f);s.addEventListener('change',f);</script>
</body></html>"""


def _render_annotation(
    *,
    source: VQASourceV2,
    annotation: VQAAnnotationV2,
    view: VQATrainingViewV2 | None,
    embed_images: bool,
) -> str:
    """Render one accepted question-answer record and its provenance."""

    observation = source.observation
    figures = _render_images(
        observation.images,
        observation.image_roles,
        embed=embed_images,
    )
    question = annotation.question.model_dump(
        mode="json",
        exclude_none=True,
    )
    answer = annotation.answer.payload.model_dump(
        mode="json",
        exclude_none=True,
    )
    visible = {
        "instruction": observation.instruction,
        "model_visible_context": observation.model_visible_context,
        "input_modalities": observation.input_modalities.model_dump(
            mode="json"
        ),
        "evidence_catalog": [
            item.model_dump(mode="json", exclude_none=True)
            for item in observation.evidence_catalog
        ],
    }
    provenance = {
        "label_source": annotation.label_source,
        "source_kind": annotation.source_kind,
        "evidence_refs": annotation.evidence_refs,
        "source_field_refs": annotation.source_field_refs,
        "producer_id": annotation.producer_id,
        "producer_version": annotation.producer_version,
        "teacher_provider": annotation.teacher_provider,
        "teacher_model": annotation.teacher_model,
        "prompt_profile_id": annotation.prompt_profile_id,
        "prompt_policy_digest": annotation.prompt_policy_digest,
    }
    export = (
        {
            "sample_id": view.sample_id,
            "assistant_text": view.assistant_text,
            "prompt_profile_id": view.prompt_profile_id,
            "capability_config_digest": (
                view.capability_config_digest
            ),
        }
        if view is not None
        else {"status": "not_exported"}
    )
    searchable = json.dumps(
        {
            "annotation_id": annotation.annotation_id,
            "task_id": annotation.question.task_id,
            "question": question,
            "answer": answer,
            "provenance": provenance,
        },
        ensure_ascii=False,
    ).lower()
    return f"""<article class="sample" data-source="{_escape(annotation.label_source)}" data-search="{_escape(searchable)}">
<div class="sample-head"><div><h2>{_escape(annotation.question.task_id)}</h2><div class="muted">{_escape(annotation.annotation_id)} · {_escape(observation.dataset_source)} · {_escape(annotation.source_fingerprint[:12])}</div></div>
<div><span class="badge {_escape(annotation.label_source)}">{_escape(annotation.label_source)}</span> <span class="badge">{_escape(annotation.producer_id)}</span></div></div>
<div class="body">{figures}<div class="grid"><div><section><h3>Answer-independent question</h3>{_json_pre(question)}</section><section><h3>Minimal target payload</h3>{_json_pre(answer)}</section><section><h3>Complete student-visible input</h3>{_json_pre(visible)}</section></div>
<aside><section><h3>Label provenance</h3>{_json_pre(provenance)}</section><section><h3>Training export</h3>{_json_pre(export)}</section><section><h3>Validation</h3>{_json_pre({'typed_contract': 'accepted', 'source_fingerprint_match': True, 'private_interfaces_exported': False})}</section></aside></div>
<details><summary>Raw accepted annotation</summary>{_json_pre(annotation.model_dump(mode='json', exclude_none=True))}</details></div></article>"""


def _render_images(
    images: Sequence[str],
    roles: Sequence[str],
    *,
    embed: bool,
) -> str:
    """Render ordered prompt-visible images with explicit roles."""

    figures: list[str] = []
    for index, value in enumerate(images):
        role = roles[index] if index < len(roles) else f"image.{index}"
        source = _image_source(value, embed=embed)
        body = (
            f"<img src='{_escape(source)}' alt='{_escape(role)}'>"
            if source
            else f"<pre>Unavailable image: {_escape(value)}</pre>"
        )
        figures.append(
            "<figure>"
            f"{body}<figcaption>{_escape(role)} · "
            f"{_escape(value)}</figcaption></figure>"
        )
    return f"<div class='images'>{''.join(figures)}</div>"


def _image_source(value: Any, *, embed: bool) -> str:
    """Resolve an image reference as a data URL or local file URI."""

    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("data:image/"):
        return text
    if text.startswith(("https://", "http://")):
        return ""
    path = Path(text).expanduser()
    if not path.is_file():
        return ""
    if not embed:
        return path.resolve().as_uri()
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _json_pre(value: Any) -> str:
    """Serialize and escape one JSON-compatible value."""

    return (
        "<pre>"
        f"{_escape(json.dumps(value, ensure_ascii=False, indent=2))}"
        "</pre>"
    )


def _escape(value: Any) -> str:
    """Escape HTML text and attribute values."""

    return html.escape(
        str(value if value is not None else ""),
        quote=True,
    )


if __name__ == "__main__":
    main()
