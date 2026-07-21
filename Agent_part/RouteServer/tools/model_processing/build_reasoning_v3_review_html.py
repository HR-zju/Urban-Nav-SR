#!/usr/bin/env python3
"""Build a self-contained multimodal Reasoning V3 annotation review page."""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningAnnotationV3, ReasoningSourceV3
from mcp_server.data.jsonl_io import iter_jsonl


def build_reasoning_v3_review_html(
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    output_html: str | Path,
    *,
    per_stratum: int = 3,
    seed: int = 20260720,
    media_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a self-contained review of LVLM observations and composed targets.

    Args:
        source_jsonl: Frozen Reasoning V3.3 source manifest.
        annotation_jsonl: Append-only annotation manifest.
        output_html: Destination HTML file.
        per_stratum: Maximum samples selected per dataset/modality stratum.
        seed: Deterministic sampling seed.
        media_root: Optional root used to resolve relative image paths.

    Returns:
        Review report with selection, stratum, and missing-image counts.

    Raises:
        ValueError: If a source or annotation row violates the V3.3 schema.
    """

    annotations: dict[str, ReasoningAnnotationV3] = {}
    for row in iter_jsonl(annotation_jsonl):
        annotation = ReasoningAnnotationV3.model_validate(row)
        annotations[annotation.source_group_id] = annotation
    strata: dict[tuple[str, str, str], list[tuple[ReasoningSourceV3, ReasoningAnnotationV3]]] = defaultdict(list)
    for row in iter_jsonl(source_jsonl):
        source = ReasoningSourceV3.model_validate(row)
        annotation = annotations.get(source.model_input.source_group_id)
        if annotation is None:
            continue
        dataset = _dataset_family(source.model_input.dataset_source)
        modalities = "+".join(sorted(source.model_input.available_modalities))
        quality = _quality_stratum(annotation)
        strata[(dataset, modalities, quality)].append((source, annotation))
    rng = random.Random(int(seed))
    selected: list[tuple[ReasoningSourceV3, ReasoningAnnotationV3, tuple[str, str, str]]] = []
    for stratum in sorted(strata):
        rows = strata[stratum]
        chosen = rows if len(rows) <= per_stratum else rng.sample(rows, per_stratum)
        selected.extend((source, annotation, stratum) for source, annotation in chosen)
    destination = Path(output_html)
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = Path(media_root).expanduser() if media_root is not None else None
    missing_images = 0
    cards: list[str] = []
    for card_index, (source, annotation, stratum) in enumerate(selected):
        image_blocks: list[str] = []
        for image_path, role in zip(source.model_input.images, source.model_input.image_roles):
            data_url = _image_data_url(image_path, media_root=root)
            if data_url is None:
                missing_images += 1
                media = f'<div class="missing">missing: {html.escape(image_path)}</div>'
            else:
                media = f'<img loading="lazy" src="{data_url}" alt="{html.escape(role)}">'
            image_blocks.append(
                f'<figure><figcaption>{html.escape(role)}</figcaption>{media}</figure>'
            )
        target = (
            annotation.target.model_dump(mode="json", exclude_none=True)
            if annotation.target is not None
            else None
        )
        observation = (
            annotation.observation.model_dump(mode="json", exclude_none=True)
            if annotation.observation is not None
            else None
        )
        rationale = (
            annotation.decision_rationale.model_dump(mode="json", exclude_none=True)
            if annotation.decision_rationale is not None
            else None
        )
        fixed = source.teacher_supervision_context.get("fixed_labels") or {}
        provenance = {
            **{key: value.model_dump(mode="json") for key, value in source.label_provenance.items()},
            **{key: value.model_dump(mode="json") for key, value in annotation.label_provenance.items()},
        }
        optional_context_refs = [
            item.ref
            for item in source.model_input.evidence_catalog
            if item.modality in {"bev", "goal", "goal_image", "waypoint", "route", "tool", "tool_route"}
        ]
        modality_ledger = {
            "expected_optional_context_refs": optional_context_refs,
            "scene_context_observations": (
                observation.get("scene_grounding", {}).get("context_observations", [])
                if observation is not None
                else []
            ),
            "cross_modal_relations": (
                observation.get("scene_grounding", {}).get("cross_modal_relations", [])
                if observation is not None
                else []
            ),
            "navigation_source_groundings": (
                observation.get("navigation_context_grounding", {}).get("source_groundings", [])
                if observation is not None
                else []
            ),
        }
        if target is not None:
            target_details = (
                f'<details open><summary>TaskGrounding</summary><pre>{_pretty(target["reasoning"]["task_grounding"])}</pre></details>'
                f'<details open><summary>SceneGrounding</summary><pre>{_pretty(target["reasoning"]["scene_grounding"])}</pre></details>'
                f'<details open><summary>LocalTraversability</summary><pre>{_pretty(target["reasoning"]["local_traversability"])}</pre></details>'
                f'<details open><summary>NavigationContextGrounding</summary><pre>{_pretty(target["reasoning"]["navigation_context_grounding"])}</pre></details>'
                f'<details open><summary>InteractionAndRuleGrounding</summary><pre>{_pretty(target["reasoning"]["interaction_and_rule_grounding"])}</pre></details>'
                f'<details open><summary>Decision / finish</summary><pre>{_pretty({"decision": target["reasoning"]["decision"], "finish": target["finish"]})}</pre></details>'
                f'<details><summary>Full target</summary><pre>{_pretty(target)}</pre></details>'
            )
        else:
            target_details = (
                '<details open><summary>Rejected / error audit</summary>'
                f'<pre>{_pretty({"status": annotation.status, "errors": annotation.errors, "warnings": annotation.warnings, "attempts": annotation.attempts, "metadata": annotation.metadata, "telemetry": annotation.telemetry})}</pre></details>'
            )
        cards.append(
            '<article class="sample" '
            f'data-dataset="{html.escape(stratum[0])}" data-modalities="{html.escape(stratum[1])}" data-quality="{html.escape(stratum[2])}" data-status="{html.escape(annotation.status)}">'
            f'<header><span>#{card_index}</span><b>{html.escape(stratum[0])}</b>'
            f'<code>{html.escape(stratum[1])}</code><b class="status {html.escape(annotation.status)}">{html.escape(annotation.status)}</b><b class="quality">{html.escape(stratum[2])}</b><span>{html.escape(source.model_input.source_group_id)}</span></header>'
            f'<p class="instruction">{html.escape(source.model_input.instruction)}</p>'
            f'<div class="images">{"".join(image_blocks)}</div>'
            '<div class="details">'
            f'<details open><summary>Modality accounting</summary><pre>{_pretty(modality_ledger)}</pre></details>'
            f'<details open><summary>LVLM observation</summary><pre>{_pretty(observation)}</pre></details>'
            f'<details open><summary>GT-constrained decision rationale</summary><pre>{_pretty(rationale)}</pre></details>'
            f'{target_details}'
            f'<details><summary>Raw teacher answer</summary><pre>{_pretty(annotation.raw_answer)}</pre></details>'
            f'<details><summary>Fixed supervision / provenance / mask</summary><pre>{_pretty({"fixed_labels": fixed, "provenance": provenance, "section_loss_mask": annotation.section_loss_mask.model_dump(mode="json") if annotation.section_loss_mask is not None else None, "metadata": annotation.metadata})}</pre></details>'
            '</div></article>'
        )
    datasets = sorted({key[0] for key in strata})
    options = ''.join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in datasets)
    qualities = sorted({key[2] for key in strata})
    quality_options = ''.join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>'
        for value in qualities
    )
    statuses = sorted({annotation.status for _, annotation, _ in selected})
    status_options = ''.join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>'
        for value in statuses
    )
    page = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>Reasoning V3.3 Review</title>
<style>
body{{margin:0;background:#f3f5f7;color:#17202a;font:14px Arial,sans-serif}}header.toolbar{{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #cbd3da;padding:12px 20px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}main{{padding:18px;max-width:1600px;margin:auto}}.sample{{background:#fff;border:1px solid #cbd3da;border-radius:6px;margin:0 0 18px;overflow:hidden}}.sample>header{{display:flex;gap:12px;align-items:center;padding:10px 14px;border-bottom:1px solid #dce2e7;flex-wrap:wrap}}code{{background:#eef2f5;padding:3px 6px}}.quality,.status{{padding:3px 6px;border-radius:3px}}.quality,.status.accepted{{background:#dff3e4;color:#145c2e}}.status.rejected,.status.error{{background:#fbe2df;color:#8a231c}}.instruction{{padding:0 14px;font-size:16px}}.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px;padding:0 14px}}figure{{margin:0}}figcaption{{font-weight:bold;margin:4px 0}}img,.missing{{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}}.missing{{display:grid;place-items:center;color:#fff;overflow-wrap:anywhere}}.details{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:8px;padding:14px}}details{{border:1px solid #dce2e7;padding:8px;min-width:0}}summary{{font-weight:bold;cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0 0;font-size:12px}}@media(max-width:600px){{.details{{grid-template-columns:1fr}}}}
</style></head><body>
<header class="toolbar"><b>UrbanNav Reasoning V3.3 Review</b><label>Dataset <select id="dataset"><option value="all">all</option>{options}</select></label><label>Status <select id="status"><option value="all">all</option>{status_options}</select></label><label>Quality <select id="quality"><option value="all">all</option>{quality_options}</select></label><span>{len(selected)} samples, {len(strata)} strata, missing images {missing_images}</span></header>
<main>{''.join(cards)}</main>
<script>const apply=()=>{{const d=document.getElementById('dataset').value,s=document.getElementById('status').value,q=document.getElementById('quality').value;document.querySelectorAll('.sample').forEach(x=>x.hidden=(d!=='all'&&x.dataset.dataset!==d)||(s!=='all'&&x.dataset.status!==s)||(q!=='all'&&x.dataset.quality!==q))}};document.getElementById('dataset').addEventListener('change',apply);document.getElementById('status').addEventListener('change',apply);document.getElementById('quality').addEventListener('change',apply);</script>
</body></html>'''
    destination.write_text(page, encoding="utf-8")
    return {
        "schema_version": "urbannav_reasoning_v3_3_review_report_v1",
        "output_html": str(destination),
        "selected_rows": len(selected),
        "strata": {"|".join(key): len(value) for key, value in sorted(strata.items())},
        "quality_strata": {
            quality: sum(len(rows) for key, rows in strata.items() if key[2] == quality)
            for quality in qualities
        },
        "status_counts": {
            status: sum(1 for _, annotation, _ in selected if annotation.status == status)
            for status in statuses
        },
        "missing_images": missing_images,
    }


def _quality_stratum(annotation: ReasoningAnnotationV3) -> str:
    """Return the highest-priority manual-review category for one target."""

    if annotation.status != "accepted":
        errors = " ".join(annotation.errors).lower()
        if annotation.status == "error":
            return "error"
        if "composition_failed" in errors:
            return "rejected_composition"
        if "teacher_schema" in errors:
            return "rejected_teacher_schema"
        return "rejected_validation"
    if annotation.target is None:
        return "missing_target"
    trace = annotation.target.reasoning
    local = trace.local_traversability
    grounding = trace.navigation_context_grounding
    selected = trace.decision.selected_corridor_ref
    if trace.decision.rationale_status == "unsupported":
        return "unsupported_decision_rationale"
    if trace.decision.rationale_status == "partial":
        return "partial_decision_rationale"
    if grounding.source_consistency == "conflict" or grounding.fused_execution_status == "conflict":
        return "context_conflict"
    if grounding.fused_execution_status == "insufficient_evidence":
        return "insufficient_evidence"
    if grounding.fused_target_corridor_ref is not None and grounding.fused_target_corridor_ref != selected:
        return "fallback"
    # 优先抽取会改变可执行策略的区域类别，再补普通 clear 样本。
    states = {item.state for item in local.regions}
    if grounding.fused_execution_status == "locally_blocked" or "blocked" in states:
        return "blocked"
    if "conditional" in states:
        return "conditional"
    if trace.interaction_and_rule_grounding.interactions:
        return "social_interaction"
    if trace.interaction_and_rule_grounding.traffic_rules:
        return "traffic_rule"
    if (
        local.visibility == "unknown"
        or local.local_risk == "unknown"
        or local.confidence == "unknown"
    ):
        return "critical_unknown"
    if grounding.applicable and grounding.fused_execution_status == "clear":
        return "context_clear"
    return "direct_traversable"



def _image_data_url(value: str, *, media_root: Path | None) -> str | None:
    if str(value).startswith("data:image/"):
        return str(value)
    path = Path(value).expanduser()
    if not path.is_absolute() and media_root is not None:
        path = media_root / path
    if not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _pretty(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def _dataset_family(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    for family in ("sidewalk", "foxglove", "frodobots", "egowalk"):
        if normalized.startswith(family):
            return family
    return normalized or "unknown"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse review CLI arguments.

    Args:
        argv: Optional argument vector for tests or programmatic use.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--per-stratum", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--media-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate the review page and optional JSON report.

    Args:
        argv: Optional argument vector for tests or programmatic use.
    """

    args = parse_args(argv)
    report = build_reasoning_v3_review_html(
        args.source_jsonl,
        args.annotation_jsonl,
        args.output_html,
        per_stratum=max(1, args.per_stratum),
        seed=args.seed,
        media_root=args.media_root,
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
