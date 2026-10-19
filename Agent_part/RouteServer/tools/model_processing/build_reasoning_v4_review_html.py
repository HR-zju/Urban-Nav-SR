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
    ActionAdjudicationV4,
    ActionReviewDiagnosisRecordV4,
    ReasoningAnnotationV4,
    ReasoningSourceV4,
)
from mcp_server.data.io.jsonl import iter_jsonl
from mcp_server.data.canonical.reasoning_v4_reconciliation import action_reconciliation_review_priority_v4
from mcp_server.data.canonical.reasoning_v4_state import derive_reasoning_release_state_v4, load_latest_action_adjudications_v4
from mcp_server.data.canonical.reasoning_v4_review import load_latest_action_review_diagnoses_v4

_REVIEW_VERDICTS_V4 = (
    "teacher_proposal_error",
    "grounding_error",
    "strong_label_error",
    "unresolved",
)
_REPAIR_ACTIONS_V4 = (
    "keep_without_change",
    "rerun_teacher_proposal",
    "relabel_grounding",
    "rebuild_strong_sidecar",
    "quarantine",
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
    parser.add_argument("--diagnosis-jsonl", type=Path)
    parser.add_argument("--adjudication-jsonl", type=Path)
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
    diagnosis_jsonl: str | Path | None = None,
    adjudication_jsonl: str | Path | None = None,
    report_json: str | Path | None = None,
    title: str = (
        "UrbanNav Reasoning V4.6 "
        "Action Reconciliation Review"
    ),
    max_samples: int = 100,
    embed_images: bool = True,
) -> dict[str, Any]:
    """Join sources, annotations, diagnoses, adjudications, and views.

    Args:
        source_jsonl: Frozen V4 source manifest.
        annotation_jsonl: Append-only annotation manifest.
        output_html: Self-contained review-page destination.
        views_jsonl: Optional exported preview/formal views.
        diagnosis_jsonl: Optional append-only two-stage LVLM diagnoses.
        adjudication_jsonl: Optional append-only human adjudications.
        report_json: Optional machine-readable summary destination.
        title: Page title.
        max_samples: Maximum source cards.
        embed_images: Embed local media bytes when true.

    Returns:
        Exact review summary.
    """

    sources = _source_index(source_jsonl)
    annotations = _annotation_index(annotation_jsonl)
    diagnoses = load_latest_action_review_diagnoses_v4(
        diagnosis_jsonl
    )
    adjudications = load_latest_action_adjudications_v4(
        adjudication_jsonl
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
        diagnosis = (
            diagnoses.get(audit.audit_id)
            if audit is not None
            else None
        )
        adjudication = (
            adjudications.get(audit.audit_id)
            if audit is not None
            else None
        )
        state = derive_reasoning_release_state_v4(
            source,
            annotation,
            diagnosis=diagnosis,
            adjudication=adjudication,
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
                diagnosis=diagnosis,
                adjudication=adjudication,
                views=views_by_source.get(source_id, []),
                embed_images=embed_images,
            )
        )

    summary = {
        "schema_version": (
            "urbannav_reasoning_v4_6_review_report_v2"
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "views_jsonl": (
            str(views_jsonl)
            if views_jsonl is not None
            else None
        ),
        "diagnosis_jsonl": (
            str(diagnosis_jsonl)
            if diagnosis_jsonl is not None
            else None
        ),
        "adjudication_jsonl": (
            str(adjudication_jsonl)
            if adjudication_jsonl is not None
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
.toolbar{{position:sticky;top:0;z-index:4;background:#eef2f7e8;backdrop-filter:blur(8px);
padding:10px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
button,input,select,textarea{{font:inherit;border:1px solid var(--line);border-radius:7px;
padding:7px;background:#fff}} button{{cursor:pointer;background:#1f5ca8;color:#fff}}
.review-form{{margin-top:12px;border:2px solid #b7c9e2;border-radius:10px;padding:12px}}
.review-form label{{display:block;color:var(--muted);font-size:12px;margin-top:8px}}
.review-form input[type=text],.review-form textarea,.review-form select{{width:100%}}
.components{{display:flex;gap:10px;flex-wrap:wrap}} .components label{{color:var(--ink)}}
.saved{{color:var(--good);font-weight:650}}
@media(max-width:900px){{.metrics{{grid-template-columns:repeat(2,1fr)}}
.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<h1>{html.escape(title)}</h1>
<div class="metrics">{metric_html}</div>
<div class="toolbar">
<label>Reviewer <input id="reviewer" type="text" placeholder="name or stable ID"></label>
<label>Status <select id="status-filter"><option value="">all</option>
<option>pending_human</option><option>resolved</option><option>rejected</option>
<option>not_required</option></select></label>
<label>LVLM verdict <select id="verdict-filter"><option value="">all</option>
<option>teacher_proposal_error</option><option>grounding_error</option>
<option>strong_label_error</option><option>unresolved</option>
<option value="not_run">not_run</option></select></label>
<button id="export-jsonl" type="button">Export adjudications JSONL</button>
<span id="save-status" class="saved"></span>
</div>
{''.join(cards)}
</main>
<script>
const storageKey="urbannav-reasoning-v46-adjudication-drafts";
const reviewerInput=document.getElementById("reviewer");
const forms=[...document.querySelectorAll(".review-form")];
function splitValues(value){{return [...new Set(value.split(",").map(x=>x.trim()).filter(Boolean))].sort();}}
function payload(form,requireReviewer=false){{
  const reviewer=reviewerInput.value.trim();
  if(requireReviewer && !reviewer) throw new Error("Reviewer is required");
  const components=[...form.querySelectorAll("input[name=component]:checked")].map(x=>x.value).sort();
  const row={{
    source_group_id:form.dataset.sourceId,
    audit_id:form.dataset.auditId,
    diagnosis_id:form.dataset.diagnosisId,
    reviewer:reviewer,
    reviewed_at:new Date().toISOString(),
    diagnosis_disposition:form.querySelector("[name=diagnosis_disposition]").value,
    final_verdict:form.querySelector("[name=final_verdict]").value,
    affected_components:components,
    reason_codes:splitValues(form.querySelector("[name=reason_codes]").value),
    repair_action:form.querySelector("[name=repair_action]").value,
    fields_to_inspect:splitValues(form.querySelector("[name=fields_to_inspect]").value)
  }};
  const subtype=form.querySelector("[name=error_subtype]").value.trim();
  const note=form.querySelector("[name=note]").value.trim();
  if(subtype) row.error_subtype=subtype;
  if(note) row.note=note;
  return row;
}}
function saveDrafts(){{
  const saved={{reviewer:reviewerInput.value,rows:{{}}}};
  forms.forEach(form=>saved.rows[form.dataset.auditId]=payload(form));
  localStorage.setItem(storageKey,JSON.stringify(saved));
  document.getElementById("save-status").textContent="draft saved locally";
}}
function restoreDrafts(){{
  const raw=localStorage.getItem(storageKey); if(!raw) return;
  const saved=JSON.parse(raw); reviewerInput.value=saved.reviewer||"";
  forms.forEach(form=>{{
    const row=saved.rows?.[form.dataset.auditId]; if(!row) return;
    for(const name of ["diagnosis_disposition","final_verdict",
      "error_subtype","reason_codes","repair_action","fields_to_inspect",
      "note"]){{
      const el=form.querySelector(`[name=${{name}}]`); if(!el) continue;
      const value=row[name]; el.value=Array.isArray(value)?value.join(","):value||"";
    }}
    form.querySelectorAll("input[name=component]").forEach(
      el=>el.checked=(row.affected_components||[]).includes(el.value));
  }});
}}
function applyFilters(){{
  const status=document.getElementById("status-filter").value;
  const verdict=document.getElementById("verdict-filter").value;
  document.querySelectorAll("article.card[data-review-status]").forEach(card=>{{
    card.hidden=!!((status&&card.dataset.reviewStatus!==status)||
      (verdict&&card.dataset.diagnosisVerdict!==verdict));
  }});
}}
forms.forEach(form=>form.addEventListener("change",saveDrafts));
forms.forEach(form=>form.addEventListener("input",saveDrafts));
reviewerInput.addEventListener("input",saveDrafts);
document.getElementById("status-filter").addEventListener("change",applyFilters);
document.getElementById("verdict-filter").addEventListener("change",applyFilters);
document.getElementById("export-jsonl").addEventListener("click",()=>{{
  try{{
    const rows=forms.map(form=>payload(form,true));
    for(const row of rows){{
      if(!row.diagnosis_id) throw new Error(`Missing diagnosis for ${{row.source_group_id}}`);
      if(!row.affected_components.length||!row.reason_codes.length)
        throw new Error(`Components and reason codes are required for ${{row.source_group_id}}`);
    }}
    const body=rows.map(row=>JSON.stringify(row)).join("\\n")+"\\n";
    const url=URL.createObjectURL(new Blob([body],{{type:"application/jsonl"}}));
    const link=document.createElement("a"); link.href=url;
    link.download="reasoning_v46_adjudications.unbound.jsonl"; link.click();
    URL.revokeObjectURL(url);
  }}catch(error){{alert(error.message);}}
}});
restoreDrafts(); applyFilters();
</script>
</body></html>"""


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
    diagnosis: ActionReviewDiagnosisRecordV4 | None,
    adjudication: ActionAdjudicationV4 | None,
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
    diagnosis_payload = (
        diagnosis.model_dump(mode="json", exclude_none=True)
        if diagnosis is not None
        else None
    )
    adjudication_payload = (
        adjudication.model_dump(mode="json", exclude_none=True)
        if adjudication is not None
        else None
    )
    diagnosis_verdict = (
        diagnosis.diagnosis.verdict
        if diagnosis is not None
        and diagnosis.diagnosis is not None
        else "not_run"
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
        (
            "Blind LVLM assessment",
            (
                diagnosis.blind_assessment.model_dump(
                    mode="json",
                    exclude_none=True,
                )
                if diagnosis is not None
                and diagnosis.blind_assessment is not None
                else None
            ),
        ),
        (
            "Reveal: proposal / strong / candidate / audit",
            {
                "teacher_proposal": proposal,
                "strong_direction": strong,
                "composer_candidate": candidate,
                "action_audit": audit,
            },
        ),
        ("Comparative LVLM diagnosis", diagnosis_payload),
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
        ("Human adjudication", adjudication_payload),
        ("Exported views", views),
    )
    details = "".join(
        "<details><summary>"
        f"{html.escape(label)}</summary>"
        f"<pre>{_json_html(payload)}</pre></details>"
        for label, payload in payloads
    )
    review_form = _render_adjudication_form_v4(
        source,
        annotation,
        diagnosis=diagnosis,
        adjudication=adjudication,
    )
    return (
        "<article class='card' data-review-status='"
        f"{html.escape(state.review_status, quote=True)}' "
        "data-diagnosis-verdict='"
        f"{html.escape(str(diagnosis_verdict), quote=True)}'><h2>"
        f"{html.escape(source.observation.source_group_id)}"
        "</h2><div class='badges'>"
        f"{badge_html}</div>"
        f"{_render_images(source, embed_images)}"
        f"<div class='grid'>{details}</div>{review_form}</article>"
    )


def _render_adjudication_form_v4(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
    *,
    diagnosis: ActionReviewDiagnosisRecordV4 | None,
    adjudication: ActionAdjudicationV4 | None,
) -> str:
    """Render one machine-exportable human-adjudication form."""

    audit = annotation.action_reconciliation
    if (
        audit is None
        or diagnosis is None
        or diagnosis.record_status != "complete"
        or diagnosis.diagnosis is None
    ):
        return (
            "<p class='warn'>Human form unavailable until the two-stage "
            "LVLM diagnosis is complete for this audit.</p>"
        )
    current = (
        adjudication.model_dump(mode="json", exclude_none=True)
        if adjudication is not None
        else {}
    )
    suggested = diagnosis.diagnosis
    values = {
        "diagnosis_disposition": current.get(
            "diagnosis_disposition",
            "confirmed",
        ),
        "final_verdict": current.get(
            "final_verdict",
            suggested.verdict,
        ),
        "error_subtype": current.get(
            "error_subtype",
            suggested.error_subtype or "",
        ),
        "affected_components": current.get(
            "affected_components",
            list(suggested.affected_components),
        ),
        "reason_codes": current.get(
            "reason_codes",
            list(suggested.reason_codes),
        ),
        "repair_action": current.get(
            "repair_action",
            suggested.repair_action,
        ),
        "fields_to_inspect": current.get(
            "fields_to_inspect",
            list(suggested.fields_to_inspect),
        ),
        "note": current.get("note", ""),
    }
    components = (
        "DecisionProposal",
        "TaskGrounding",
        "SceneGrounding",
        "LocalTraversability",
        "NavigationContextGrounding",
        "InteractionConstraintGrounding",
        "StrongDirectionSidecar",
    )
    component_html = "".join(
        "<label><input type='checkbox' name='component' value='"
        f"{item}'{' checked' if item in values['affected_components'] else ''}>"
        f"{item}</label>"
        for item in components
    )
    return (
        "<section class='review-form' data-source-id='"
        f"{html.escape(source.observation.source_group_id, quote=True)}' "
        f"data-audit-id='{audit.audit_id}' "
        f"data-diagnosis-id='{diagnosis.diagnosis.diagnosis_id}'>"
        "<h3>Human adjudication</h3>"
        "<label>LVLM disposition<select name='diagnosis_disposition'>"
        f"{_options_html(('confirmed', 'overridden', 'deferred'), values['diagnosis_disposition'])}"
        "</select></label>"
        "<label>Final four-way verdict<select name='final_verdict'>"
        f"{_options_html(_REVIEW_VERDICTS_V4, values['final_verdict'])}"
        "</select></label>"
        "<label>Error subtype<input type='text' name='error_subtype' value='"
        f"{html.escape(str(values['error_subtype']), quote=True)}'></label>"
        f"<label>Affected components</label><div class='components'>{component_html}</div>"
        "<label>Reason codes, comma separated<input type='text' name='reason_codes' value='"
        f"{html.escape(','.join(values['reason_codes']), quote=True)}'></label>"
        "<label>Repair action<select name='repair_action'>"
        f"{_options_html(_REPAIR_ACTIONS_V4, values['repair_action'])}"
        "</select></label>"
        "<label>Fields to inspect, comma separated<input type='text' name='fields_to_inspect' value='"
        f"{html.escape(','.join(values['fields_to_inspect']), quote=True)}'></label>"
        "<label>Human note<textarea name='note'>"
        f"{html.escape(str(values['note']))}</textarea></label></section>"
    )


def _options_html(
    options: Sequence[str],
    selected: str,
) -> str:
    """Render escaped select options with one current value."""

    return "".join(
        "<option"
        f"{' selected' if item == selected else ''}>"
        f"{html.escape(item)}</option>"
        for item in options
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
        diagnosis_jsonl=args.diagnosis_jsonl,
        adjudication_jsonl=args.adjudication_jsonl,
        output_html=args.output_html,
        report_json=args.report_json,
        title=args.title,
        max_samples=args.max_samples,
        embed_images=not args.link_images,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
