"""Self-contained human-review renderers for V1 quality sidecars.

The generated HTML is an audit artifact. It embeds source images plus sidecar
labels only, never mutates canonical V4.6 JSONL, legacy language-goal manifests,
or source media.
"""

from __future__ import annotations

import base64
import html
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_server.contracts.local_traversability_geometry import LocalTraversabilityGeometryV1
from mcp_server.contracts.navigation_trainability import NavigationTrainabilityFilterV1
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4
from mcp_server.data.io.jsonl import file_sha256, iter_jsonl


TRAVERSABILITY_GEOMETRY_REVIEW_SCHEMA_VERSION_V1 = "urbannav_traversability_geometry_review_v1"
NAVIGATION_TRAINABILITY_REVIEW_SCHEMA_VERSION_V1 = "urbannav_navigation_trainability_review_v1"


@dataclass(frozen=True)
class NavigationQualityReviewConfigV1:
    """Configure deterministic portable quality-review output.

    Args:
        sample_offset: Zero-based offset into the deterministic review order.
        max_samples: Maximum deterministic number of sidecars to render.
        thumbnail_max_width_px: Maximum embedded image width.
        jpeg_quality: Embedded JPEG quality.
        max_temporal_frames: Maximum ordered FPV frames on suitability cards.
    """

    sample_offset: int = 0
    max_samples: int = 200
    thumbnail_max_width_px: int = 960
    jpeg_quality: int = 88
    max_temporal_frames: int = 5


def write_traversability_geometry_review_html_v1(
    *,
    source_jsonl: str | Path,
    geometry_jsonl: str | Path,
    geometry_decisions_jsonl: str | Path | None = None,
    output_html: str | Path,
    output_manifest_json: str | Path | None = None,
    config: NavigationQualityReviewConfigV1 | None = None,
    title: str = "UrbanNav Local Traversability Geometry Review",
) -> dict[str, Any]:
    """Render current-FPV quadrilateral overlays for human geometry review.

    Args:
        source_jsonl: Canonical Reasoning V4 source ledger.
        geometry_jsonl: Direct-LVLM typed geometry sidecar ledger.
        geometry_decisions_jsonl: Optional append-only decision ledger.  When
            present, parser failures and rejected responses are rendered with
            their raw answer and retry diagnostics alongside accepted polygons.
        output_html: Self-contained HTML destination.
        output_manifest_json: Optional machine-readable review manifest path.
        config: Sampling and thumbnail policy.
        title: Browser title.

    Returns:
        Artifact paths and rendered distribution report.
    """

    cfg = _validate_config_v1(config)
    sources = _load_sources_v1(source_jsonl)
    annotations = _load_geometry_v1(geometry_jsonl, sources=sources)
    annotations_by_source = {
        (item.parent_sample_id, item.source_fingerprint, item.prompt_profile_digest): item
        for item in annotations
    }
    decisions = _load_geometry_decisions_v1(geometry_decisions_jsonl, sources=sources)
    # A decision ledger is the authoritative review population because it
    # includes failures that never produced a typed geometry sidecar.
    review_rows = decisions or [
        {
            "parent_sample_id": item.parent_sample_id,
            "source_fingerprint": item.source_fingerprint,
            "profile_digest": item.prompt_profile_digest,
            "status": item.validator_status,
            "attempts": [],
            "raw_answer": None,
        }
        for item in annotations
    ]
    samples: list[dict[str, Any]] = []
    image_errors: Counter[str] = Counter()
    for decision in review_rows[cfg.sample_offset : cfg.sample_offset + cfg.max_samples]:
        parent_sample_id = str(decision["parent_sample_id"])
        source_fingerprint = str(decision["source_fingerprint"])
        source = _match_source_v1(sources, parent_sample_id, source_fingerprint)
        annotation = annotations_by_source.get(
            (parent_sample_id, source_fingerprint, str(decision.get("profile_digest", "")))
        )
        image_data_uri, image_error = _embed_image_v1(
            _current_fpv_path_v1(source),
            max_width_px=cfg.thumbnail_max_width_px,
            jpeg_quality=cfg.jpeg_quality,
        )
        if image_error:
            image_errors[image_error] += 1
        samples.append(
            {
                "parent_sample_id": parent_sample_id,
                "dataset_source": source.observation.dataset_source,
                "split_group_id": source.observation.split_group_id,
                "instruction": source.observation.instruction,
                "image_data_uri": image_data_uri,
                "image_error": image_error,
                "regions": (
                    [region.model_dump(mode="json") for region in annotation.regions]
                    if annotation is not None
                    else []
                ),
                "selected_corridor_region_id": (
                    annotation.selected_corridor_region_id if annotation is not None else None
                ),
                "validator_status": str(decision.get("status", "unknown")),
                "confidence": annotation.confidence if annotation is not None else None,
                "warnings": list(annotation.warnings) if annotation is not None else [],
                "teacher_model": annotation.teacher_model if annotation is not None else None,
                "prompt_profile_id": annotation.prompt_profile_id if annotation is not None else None,
                "raw_answer": decision.get("raw_answer"),
                "attempts": decision.get("attempts", []),
                "review_errors": _geometry_decision_errors_v1(decision),
            }
        )
    input_paths: dict[str, str | Path] = {
        "source_jsonl": source_jsonl,
        "geometry_jsonl": geometry_jsonl,
    }
    if geometry_decisions_jsonl is not None:
        input_paths["geometry_decisions_jsonl"] = geometry_decisions_jsonl
    return _write_review_v1(
        schema_version=TRAVERSABILITY_GEOMETRY_REVIEW_SCHEMA_VERSION_V1,
        kind="geometry",
        title=title,
        samples=samples,
        input_paths=input_paths,
        output_html=output_html,
        output_manifest_json=output_manifest_json,
        extra_manifest={
            "source_count": len(sources),
            "geometry_annotation_count": len(annotations),
            "geometry_decision_count": len(review_rows),
            "geometry_count": len(samples),
            "sample_offset": cfg.sample_offset,
            "sample_limit": cfg.max_samples,
            "geometry_status_counts": dict(Counter(item["validator_status"] for item in samples)),
            "geometry_error_counts": dict(
                Counter(error for item in samples for error in item["review_errors"])
            ),
            "region_role_counts": dict(
                Counter(region["role"] for item in samples for region in item["regions"])
            ),
            "image_error_counts": dict(image_errors),
        },
    )


def write_navigation_trainability_review_html_v1(
    *,
    source_jsonl: str | Path,
    filter_jsonl: str | Path,
    output_html: str | Path,
    output_manifest_json: str | Path | None = None,
    config: NavigationQualityReviewConfigV1 | None = None,
    title: str = "UrbanNav Navigation Dataset Suitability Review",
) -> dict[str, Any]:
    """Render scene/motion filter outcomes and source FPV evidence.

    Args:
        source_jsonl: Canonical Reasoning V4 source ledger.
        filter_jsonl: Trainability filter sidecar ledger.
        output_html: Self-contained HTML destination.
        output_manifest_json: Optional machine-readable review manifest path.
        config: Sampling and thumbnail policy.
        title: Browser title.

    Returns:
        Artifact paths and rendered distribution report.
    """

    cfg = _validate_config_v1(config)
    sources = _load_sources_v1(source_jsonl)
    filters = _load_filters_v1(filter_jsonl, sources=sources)[
        cfg.sample_offset : cfg.sample_offset + cfg.max_samples
    ]
    samples: list[dict[str, Any]] = []
    image_errors: Counter[str] = Counter()
    for result in filters:
        source = _match_source_v1(sources, result.parent_sample_id, result.source_fingerprint)
        temporal_images: list[dict[str, Any]] = []
        for role, path in _temporal_fpv_paths_v1(source, max_frames=cfg.max_temporal_frames):
            image_data_uri, image_error = _embed_image_v1(
                path,
                max_width_px=cfg.thumbnail_max_width_px,
                jpeg_quality=cfg.jpeg_quality,
            )
            if image_error:
                image_errors[image_error] += 1
            temporal_images.append(
                {"role": role, "image_data_uri": image_data_uri, "image_error": image_error}
            )
        samples.append(
            {
                "parent_sample_id": result.parent_sample_id,
                "dataset_source": source.observation.dataset_source,
                "split_group_id": source.observation.split_group_id,
                "instruction": source.observation.instruction,
                "temporal_images": temporal_images,
                "status": result.status,
                "reason_codes": list(result.reason_codes),
                "warnings": list(result.warnings),
                "motion_summary": result.motion_summary.model_dump(mode="json"),
                "scene_assessment": result.scene_assessment.model_dump(mode="json") if result.scene_assessment else None,
                "motion_scene_assessment": (
                    result.motion_scene_assessment.model_dump(mode="json")
                    if result.motion_scene_assessment
                    else None
                ),
                "eligibility": result.eligibility.model_dump(mode="json"),
            }
        )
    return _write_review_v1(
        schema_version=NAVIGATION_TRAINABILITY_REVIEW_SCHEMA_VERSION_V1,
        kind="suitability",
        title=title,
        samples=samples,
        input_paths={"source_jsonl": source_jsonl, "filter_jsonl": filter_jsonl},
        output_html=output_html,
        output_manifest_json=output_manifest_json,
        extra_manifest={
            "source_count": len(sources),
            "filter_count": len(filters),
            "sample_offset": cfg.sample_offset,
            "sample_limit": cfg.max_samples,
            "status_counts": dict(Counter(item["status"] for item in samples)),
            "scene_class_counts": dict(
                Counter(
                    item["scene_assessment"]["scene_class"]
                    for item in samples
                    if item["scene_assessment"] is not None
                )
            ),
            "motion_consistency_counts": dict(
                Counter(
                    item["motion_scene_assessment"]["consistency"]
                    for item in samples
                    if item["motion_scene_assessment"] is not None
                )
            ),
            "image_error_counts": dict(image_errors),
        },
    )


def _validate_config_v1(config: NavigationQualityReviewConfigV1 | None) -> NavigationQualityReviewConfigV1:
    """Validate bounded browser artifact settings."""

    cfg = config or NavigationQualityReviewConfigV1()
    if (
        cfg.sample_offset < 0
        or cfg.max_samples < 1
        or cfg.thumbnail_max_width_px < 32
        or cfg.max_temporal_frames < 1
    ):
        raise ValueError("review_config_requires_positive_bounds")
    return cfg


def _load_sources_v1(path: str | Path) -> dict[str, ReasoningSourceV4]:
    """Load one source revision per parent id for exact sidecar matching."""

    rows: dict[str, ReasoningSourceV4] = {}
    for payload in iter_jsonl(path):
        source = ReasoningSourceV4.model_validate(payload)
        parent = source.observation.source_group_id
        if parent in rows:
            raise ValueError(f"duplicate_source_parent_sample_id:{parent}")
        rows[parent] = source
    return rows


def _load_geometry_v1(
    path: str | Path,
    *,
    sources: dict[str, ReasoningSourceV4],
) -> list[LocalTraversabilityGeometryV1]:
    """Load latest geometry rows that match the currently reviewed source.

    Geometry results from prior source or prompt revisions remain in the ledger,
    but cannot silently appear in a current-source human review.
    """

    latest: dict[tuple[str, str, str], LocalTraversabilityGeometryV1] = {}
    for payload in iter_jsonl(path):
        annotation = LocalTraversabilityGeometryV1.model_validate(payload)
        key = (
            annotation.parent_sample_id,
            annotation.source_fingerprint,
            annotation.prompt_profile_digest,
        )
        latest[key] = annotation
    rows = [
        annotation
        for (parent_id, fingerprint, _digest), annotation in latest.items()
        if parent_id in sources and sources[parent_id].source_fingerprint == fingerprint
    ]
    return sorted(rows, key=lambda item: (item.parent_sample_id, item.geometry_id))


def _load_geometry_decisions_v1(
    path: str | Path | None,
    *,
    sources: dict[str, ReasoningSourceV4],
) -> list[dict[str, Any]]:
    """Load final geometry attempts for the exact reviewed source revision.

    The decision ledger deliberately remains loosely parsed here: an invalid
    teacher answer is the object under review and must not be discarded merely
    because it cannot instantiate ``LocalTraversabilityGeometryV1``.
    """

    if path is None:
        return []
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for payload in iter_jsonl(path):
        if payload.get("kind") != "traversability_geometry":
            continue
        parent_sample_id = payload.get("parent_sample_id")
        source_fingerprint = payload.get("source_fingerprint")
        profile_digest = payload.get("profile_digest")
        if not all(isinstance(value, str) and value for value in (parent_sample_id, source_fingerprint, profile_digest)):
            continue
        latest[(parent_sample_id, source_fingerprint, profile_digest)] = payload
    rows = [
        payload
        for (parent_sample_id, source_fingerprint, _profile_digest), payload in latest.items()
        if parent_sample_id in sources
        and sources[parent_sample_id].source_fingerprint == source_fingerprint
    ]
    return sorted(rows, key=lambda item: (str(item["parent_sample_id"]), str(item.get("profile_digest", ""))))


def _geometry_decision_errors_v1(decision: dict[str, Any]) -> list[str]:
    """Flatten bounded parser/provider errors for the browser review payload."""

    errors: list[str] = []
    for attempt in decision.get("attempts", []):
        if not isinstance(attempt, dict):
            continue
        for error in attempt.get("errors", []):
            if isinstance(error, str) and error:
                errors.append(error)
    return errors


def _load_filters_v1(
    path: str | Path,
    *,
    sources: dict[str, ReasoningSourceV4],
) -> list[NavigationTrainabilityFilterV1]:
    """Load final filter rows that match the current source revision only."""

    latest: dict[tuple[str, str], NavigationTrainabilityFilterV1] = {}
    for payload in iter_jsonl(path):
        result = NavigationTrainabilityFilterV1.model_validate(payload)
        latest[(result.parent_sample_id, result.source_fingerprint)] = result
    rows = [
        result
        for (parent_id, fingerprint), result in latest.items()
        if parent_id in sources and sources[parent_id].source_fingerprint == fingerprint
    ]
    return sorted(rows, key=lambda item: (item.parent_sample_id, item.decision_id))


def _match_source_v1(
    sources: dict[str, ReasoningSourceV4],
    parent_sample_id: str,
    source_fingerprint: str,
) -> ReasoningSourceV4:
    """Require review rows to target the exact frozen source revision."""

    source = sources.get(parent_sample_id)
    if source is None:
        raise ValueError(f"review_source_missing:{parent_sample_id}")
    if source.source_fingerprint != source_fingerprint:
        raise ValueError(f"review_source_fingerprint_mismatch:{parent_sample_id}")
    return source


def _current_fpv_path_v1(source: ReasoningSourceV4) -> Path:
    """Return current FPV media by canonical image role."""

    for image, role in zip(source.observation.images, source.observation.image_roles):
        if role == "fpv.current":
            return Path(image)
    raise ValueError(f"source_has_no_fpv_current:{source.observation.source_group_id}")


def _temporal_fpv_paths_v1(source: ReasoningSourceV4, *, max_frames: int) -> list[tuple[str, Path]]:
    """Select ordered history/current FPV frames while preserving source order."""

    selected = [
        (role, Path(image))
        for image, role in zip(source.observation.images, source.observation.image_roles)
        if role == "fpv.current" or role.startswith("fpv.history_")
    ]
    if not any(role == "fpv.current" for role, _ in selected):
        raise ValueError(f"source_has_no_fpv_current:{source.observation.source_group_id}")
    return selected[-max_frames:]


def _embed_image_v1(path: Path, *, max_width_px: int, jpeg_quality: int) -> tuple[str | None, str | None]:
    """Return a compact data URI while retaining media errors in the review."""

    if not path.is_file():
        return None, "image_missing"
    try:
        from PIL import Image

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if rgb.width > max_width_px:
                rgb.thumbnail((max_width_px, max(1, int(rgb.height * max_width_px / rgb.width))))
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    except Exception as exc:  # pragma: no cover - decoder errors depend on Pillow build.
        return None, f"image_decode_failed:{type(exc).__name__}"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded, None


def _write_review_v1(
    *,
    schema_version: str,
    kind: str,
    title: str,
    samples: list[dict[str, Any]],
    input_paths: dict[str, str | Path],
    output_html: str | Path,
    output_manifest_json: str | Path | None,
    extra_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Write portable HTML and its immutable artifact receipt."""

    output = Path(output_html).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(samples, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    output.write_text(_render_html_v1(title=title, kind=kind, payload=payload), encoding="utf-8")
    manifest = {
        "schema_version": schema_version,
        "kind": kind,
        "output_html": str(output),
        "rendered_count": len(samples),
        "input_sha256": {name: file_sha256(path) for name, path in input_paths.items()},
        "input_paths": {name: str(Path(path).expanduser().resolve()) for name, path in input_paths.items()},
        "review_actions": ["accept", "quarantine", "reject", "needs_retry", "uncertain"],
        **extra_manifest,
    }
    manifest_path = Path(output_manifest_json).expanduser().resolve() if output_manifest_json else output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest | {"output_manifest_json": str(manifest_path)}


def _render_html_v1(*, title: str, kind: str, payload: str) -> str:
    """Render a dependency-free UI with viewport-aware quadrilateral overlays."""

    template = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#14213d;background:#f5f7fb}body{margin:0}header{position:sticky;top:0;z-index:2;padding:14px 22px;background:#fff;border-bottom:1px solid #d9e0ec;display:flex;gap:12px;align-items:center;flex-wrap:wrap}h1{font-size:20px;margin:0 12px 0 0}button,select,input{font:inherit;border:1px solid #a9b7ca;border-radius:5px;padding:7px 10px;background:#fff}main{max-width:1520px;margin:0 auto;padding:20px}.card{background:#fff;border:1px solid #d9e0ec;border-radius:8px;padding:16px;margin-bottom:18px}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px;margin-bottom:14px}.tag{display:inline-block;padding:2px 8px;border-radius:12px;background:#e7eef8;margin:1px 3px 1px 0;font-size:13px}.image-wrap{position:relative;display:inline-block;max-width:100%;background:#111}.image-wrap img{display:block;max-width:100%;height:auto}.overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.regions{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px;margin-top:12px}pre{white-space:pre-wrap;word-break:break-word;background:#f4f6fa;padding:12px;border-radius:5px;margin:10px 0;max-height:340px;overflow:auto}.temporal{display:flex;gap:10px;overflow:auto}.temporal figure{margin:0;min-width:240px}.temporal img{max-width:480px;width:100%;background:#111}.error{color:#b91c1c;font-weight:600}label{font-size:13px;color:#425466;display:block}textarea{box-sizing:border-box;width:100%;min-height:64px;font:inherit;border:1px solid #a9b7ca;border-radius:5px;padding:7px}.review{display:grid;grid-template-columns:170px 1fr 110px;gap:10px;align-items:end;margin-top:14px}@media(max-width:720px){.review{grid-template-columns:1fr}}</style></head>
<body><header><h1>__TITLE__</h1><button id="prev">Prev</button><button id="next">Next</button><label>数据集<select id="dataset"><option value="all">all</option></select></label><label>状态<select id="status"><option value="all">all</option></select></label><label>索引<input id="index" type="number" value="0" min="0"></label><button id="download">导出人工结论</button><span id="count"></span></header><main id="root"></main><script>
const KIND=__KIND_JSON__,DATA=__PAYLOAD__,decisions={};let visible=[],pos=0;const COLORS={direct_traversable:'#10b981',conditional_traversable:'#f59e0b',blocked:'#ea580c',forbidden:'#dc2626'};
function options(id,vals){const s=document.getElementById(id);[...new Set(vals)].sort().forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;s.appendChild(o)})}options('dataset',DATA.map(x=>x.dataset_source));options('status',DATA.map(x=>x.validator_status||x.status||'unknown'));
function esc(v){return String(v??'').replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}function tags(xs){return(xs||[]).map(x=>'<span class="tag">'+esc(x)+'</span>').join('')}function filtered(){const d=document.getElementById('dataset').value,s=document.getElementById('status').value;return DATA.filter(x=>(d==='all'||x.dataset_source===d)&&(s==='all'||(x.validator_status||x.status||'unknown')===s))}
function overlay(regions,selected){if(!regions?.length)return'';const p=regions.map(r=>{const q=r.quadrilateral_uv,pts=[q.near_left,q.near_right,q.far_right,q.far_left].map(x=>(x.u*100)+' '+(x.v*100)).join(' L '),c=COLORS[r.role]||'#64748b';return '<path d="M '+pts+' Z" fill="'+c+'22" stroke="'+c+'" stroke-width="'+(r.region_id===selected?0.8:0.45)+'"/><text x="'+(q.far_left.u*100)+'" y="'+Math.max(3,q.far_left.v*100-1)+'" fill="'+c+'" stroke="white" stroke-width="0.2" paint-order="stroke" font-size="3.4">'+esc(r.region_id)+': '+esc(r.role)+'</text>'}).join('');return '<svg class="overlay" viewBox="0 0 100 100" preserveAspectRatio="none">'+p+'</svg>'}
function reviewFields(id){const old=decisions[id]||{action:'',comment:''};const opts=['accept','quarantine','reject','needs_retry','uncertain'].map(v=>'<option value="'+v+'" '+(old.action===v?'selected':'')+'>'+v+'</option>').join('');return '<div class="review"><label>人工结论<select data-action="'+esc(id)+'"><option value="">未选择</option>'+opts+'</select></label><label>备注<textarea data-comment="'+esc(id)+'">'+esc(old.comment)+'</textarea></label><button data-save="'+esc(id)+'">保存</button></div>'}
function geometry(x){const im=x.image_data_uri?'<div class="image-wrap"><img src="'+x.image_data_uri+'" alt="fpv">'+overlay(x.regions,x.selected_corridor_region_id)+'</div>':'<p class="error">'+esc(x.image_error||'image unavailable')+'</p>';const rs=(x.regions||[]).map(r=>'<div class="card"><b>'+esc(r.region_id)+'</b>'+tags([r.role,r.surface_type])+'<div>'+tags(r.limiting_factors)+tags(r.evidence_refs)+'</div><pre>'+esc(JSON.stringify(r.quadrilateral_uv,null,2))+'</pre></div>').join('');const diag=(x.review_errors?.length||x.raw_answer||x.attempts?.length)?'<div class="regions"><div><h3>解析 / 重试诊断</h3><pre>'+esc(JSON.stringify({errors:x.review_errors||[],attempts:x.attempts||[]},null,2))+'</pre></div><div><h3>教师原始回答</h3><pre>'+esc(x.raw_answer||'none')+'</pre></div></div>':'';return '<section class="card"><div class="meta"><div><b>样本</b><br>'+esc(x.parent_sample_id)+'</div><div><b>数据集 / split</b><br>'+esc(x.dataset_source)+' / '+esc(x.split_group_id)+'</div><div><b>状态 / 置信度</b><br>'+tags([x.validator_status,x.confidence])+'</div><div><b>选择走廊</b><br>'+esc(x.selected_corridor_region_id||'none')+'</div></div><div><b>可见 instruction</b><br>'+esc(x.instruction||'none')+'</div>'+im+'<div class="regions">'+rs+'</div>'+diag+reviewFields(x.parent_sample_id)+'</section>'}
function suitability(x){const ims=(x.temporal_images||[]).map(i=>'<figure><figcaption>'+esc(i.role)+'</figcaption>'+(i.image_data_uri?'<img src="'+i.image_data_uri+'" alt="'+esc(i.role)+'">':'<p class="error">'+esc(i.image_error||'image unavailable')+'</p>')+'</figure>').join('');const elig=Object.entries(x.eligibility||{}).filter(e=>e[1]).map(e=>e[0]);return '<section class="card"><div class="meta"><div><b>样本</b><br>'+esc(x.parent_sample_id)+'</div><div><b>数据集 / split</b><br>'+esc(x.dataset_source)+' / '+esc(x.split_group_id)+'</div><div><b>过滤状态</b><br>'+tags([x.status])+tags(x.reason_codes)+'</div><div><b>训练资格</b><br>'+tags(elig)+'</div></div><div><b>可见 instruction</b><br>'+esc(x.instruction||'none')+'</div><div class="temporal">'+ims+'</div><div class="regions"><div><h3>场景评估</h3><pre>'+esc(JSON.stringify(x.scene_assessment,null,2))+'</pre></div><div><h3>运动-场景一致性</h3><pre>'+esc(JSON.stringify(x.motion_scene_assessment,null,2))+'</pre></div><div><h3>记录运动摘要（teacher-only）</h3><pre>'+esc(JSON.stringify(x.motion_summary,null,2))+'</pre></div></div>'+reviewFields(x.parent_sample_id)+'</section>'}
function render(){visible=filtered();if(!visible.length){document.getElementById('root').innerHTML='<section class="card">没有满足当前筛选的样本。</section>';document.getElementById('count').textContent='0 / 0';return}pos=Math.max(0,Math.min(pos,visible.length-1));const x=visible[pos];document.getElementById('index').value=pos;document.getElementById('count').textContent=(pos+1)+' / '+visible.length;document.getElementById('root').innerHTML=KIND==='geometry'?geometry(x):suitability(x);const save=document.querySelector('[data-save]');if(save)save.addEventListener('click',e=>{const id=e.target.dataset.save;decisions[id]={action:document.querySelector('[data-action="'+id+'"]').value,comment:document.querySelector('[data-comment="'+id+'"]').value}})}
document.getElementById('prev').onclick=()=>{pos--;render()};document.getElementById('next').onclick=()=>{pos++;render()};document.getElementById('dataset').onchange=()=>{pos=0;render()};document.getElementById('status').onchange=()=>{pos=0;render()};document.getElementById('index').onchange=e=>{pos=Number(e.target.value)||0;render()};document.getElementById('download').onclick=()=>{const rows=Object.entries(decisions).map(([parent_sample_id,value])=>({parent_sample_id,...value}));const b=new Blob([JSON.stringify(rows,null,2)+'\\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=KIND+'_human_review.json';a.click();URL.revokeObjectURL(a.href)};render();
</script></body></html>"""
    return (
        template.replace("__TITLE__", html.escape(title))
        .replace("__KIND_JSON__", json.dumps(kind))
        .replace("__PAYLOAD__", payload)
    )


__all__ = [
    "NAVIGATION_TRAINABILITY_REVIEW_SCHEMA_VERSION_V1",
    "TRAVERSABILITY_GEOMETRY_REVIEW_SCHEMA_VERSION_V1",
    "NavigationQualityReviewConfigV1",
    "write_navigation_trainability_review_html_v1",
    "write_traversability_geometry_review_html_v1",
]
