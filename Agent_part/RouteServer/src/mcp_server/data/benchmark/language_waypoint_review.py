"""Offline human review artifacts for visible language-goal annotations.

The review payload deliberately contains only student-visible observations and
the resulting language annotation.  Teacher-only future coordinates remain in
the candidate ledger and are never serialized into the HTML page.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_server.contracts.language_waypoint_benchmark import LanguageGoalAnnotationV1
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4
from mcp_server.data.io.jsonl import iter_jsonl
from mcp_server.eval.openloop.benchmark_plugin_package import artifact_sha256_v1


LANGUAGE_WAYPOINT_REVIEW_SCHEMA_VERSION_V1 = "urbannav_language_waypoint_review_v1"


@dataclass(frozen=True)
class LanguageWaypointReviewConfigV1:
    """Configure a portable visible-language annotation review.

    Args:
        max_samples: Maximum deterministically ordered annotations to render.
        thumbnail_max_width_px: Maximum embedded FPV thumbnail width.  This
            affects review artifact size only, never the annotation itself.
        jpeg_quality: JPEG quality used for embedded review thumbnails.
    """

    max_samples: int = 100
    thumbnail_max_width_px: int = 960
    jpeg_quality: int = 88


def write_language_waypoint_review_html_v1(
    *,
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    output_html: str | Path,
    output_manifest_json: str | Path | None = None,
    config: LanguageWaypointReviewConfigV1 | None = None,
    title: str = "UrbanNav Visible Language Goal Review",
) -> dict[str, Any]:
    """Write a self-contained review page without teacher-only geometry.

    Args:
        source_jsonl: Canonical Reasoning V4 source ledger.
        annotation_jsonl: Typed language-goal annotation ledger.
        output_html: Self-contained review page destination.
        output_manifest_json: Optional machine-readable review manifest.
        config: Deterministic sample and thumbnail policy.
        title: Browser title for this review artifact.

    Returns:
        Artifact paths, rendered counts, distributions and input digests.

    Raises:
        ValueError: If accepted annotations do not match a canonical source.
    """

    cfg = config or LanguageWaypointReviewConfigV1()
    if cfg.max_samples < 1:
        raise ValueError("max_samples must be positive")
    sources = _load_sources_v1(source_jsonl)
    annotations = _load_annotations_v1(annotation_jsonl)
    selected = annotations[: cfg.max_samples]
    samples: list[dict[str, Any]] = []
    image_errors: Counter[str] = Counter()
    for annotation in selected:
        source = sources.get(annotation.parent_sample_id)
        if source is None:
            raise ValueError(f"language annotation source missing: {annotation.parent_sample_id}")
        image_path = _current_fpv_path_v1(source)
        image_data_uri, image_error = _embed_thumbnail_v1(
            image_path,
            max_width_px=cfg.thumbnail_max_width_px,
            jpeg_quality=cfg.jpeg_quality,
        )
        if image_error is not None:
            image_errors[image_error] += 1
        # 只投影学生可见内容，避免 candidate_xy 等 teacher-only 几何泄漏到人工 review。
        samples.append(
            {
                "review_id": language_goal_review_id_v1(annotation),
                "parent_sample_id": annotation.parent_sample_id,
                "split_group_id": source.observation.split_group_id,
                "dataset_source": source.observation.dataset_source,
                "image_data_uri": image_data_uri,
                "image_error": image_error,
                "existing_dataset_text": source.observation.instruction,
                "instruction_text": annotation.instruction_text,
                "instruction_type": annotation.instruction_type,
                "target_ref": annotation.target_ref,
                "target_bbox_anchor_xyxy": annotation.target_bbox_anchor_xyxy,
                "spatial_relation": annotation.spatial_relation,
                "visibility_at_anchor": annotation.visibility_at_anchor,
                "evidence_refs": list(annotation.evidence_refs),
                "confidence": annotation.confidence,
                "validator_status": annotation.validator_status,
                "human_review": annotation.human_review,
                "teacher_model": annotation.teacher_model,
                "prompt_version": annotation.prompt_version,
            }
        )
    output = Path(output_html).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(samples, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    output.write_text(_render_html_v1(title=title, payload=payload), encoding="utf-8")
    manifest = {
        "schema_version": LANGUAGE_WAYPOINT_REVIEW_SCHEMA_VERSION_V1,
        "output_html": str(output),
        "source_jsonl": str(Path(source_jsonl).expanduser().resolve()),
        "annotation_jsonl": str(Path(annotation_jsonl).expanduser().resolve()),
        "source_jsonl_sha256": artifact_sha256_v1(source_jsonl),
        "annotation_jsonl_sha256": artifact_sha256_v1(annotation_jsonl),
        "source_count": len(sources),
        "annotation_count": len(annotations),
        "rendered_count": len(samples),
        "instruction_type_counts": dict(Counter(item["instruction_type"] for item in samples)),
        "validator_status_counts": dict(Counter(item["validator_status"] for item in samples)),
        "image_error_counts": dict(image_errors),
        "teacher_only_geometry_embedded": False,
        "review_actions": [
            "accept",
            "reject_not_visible",
            "reject_wrong_relation",
            "reject_generic",
            "needs_rewrite",
            "uncertain",
        ],
    }
    manifest_path = Path(output_manifest_json).expanduser().resolve() if output_manifest_json else output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest | {"output_manifest_json": str(manifest_path)}


def _load_sources_v1(path: str | Path) -> dict[str, ReasoningSourceV4]:
    """Load canonical sources keyed by stable parent id."""

    rows: dict[str, ReasoningSourceV4] = {}
    for payload in iter_jsonl(path):
        source = ReasoningSourceV4.model_validate(payload)
        parent_id = source.observation.source_group_id
        if parent_id in rows:
            raise ValueError(f"duplicate source_group_id: {parent_id}")
        rows[parent_id] = source
    return rows


def _load_annotations_v1(path: str | Path) -> list[LanguageGoalAnnotationV1]:
    """Load deterministically ordered typed annotation rows."""

    rows = [LanguageGoalAnnotationV1.model_validate(payload) for payload in iter_jsonl(path)]
    return sorted(rows, key=lambda item: (item.parent_sample_id, item.annotation_id))


def _current_fpv_path_v1(source: ReasoningSourceV4) -> Path:
    """Return the current FPV file selected by its canonical image role."""

    for image, role in zip(source.observation.images, source.observation.image_roles):
        if role == "fpv.current":
            return Path(image)
    raise ValueError(f"source has no fpv.current image: {source.observation.source_group_id}")


def _embed_thumbnail_v1(
    path: Path,
    *,
    max_width_px: int,
    jpeg_quality: int,
) -> tuple[str | None, str | None]:
    """Encode one portable review thumbnail, retaining any decode failure."""

    if not path.is_file():
        return None, "fpv_current_missing"
    try:
        from PIL import Image

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if rgb.width > max_width_px:
                rgb.thumbnail((max_width_px, int(rgb.height * max_width_px / rgb.width)))
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    except Exception as exc:  # pragma: no cover - exact decoder exceptions are backend-dependent.
        return None, f"fpv_current_decode_failed:{type(exc).__name__}"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", None


def language_goal_review_id_v1(annotation: LanguageGoalAnnotationV1) -> str:
    """Derive a stable manual-review key from annotation identity.

    Args:
        annotation: Typed language annotation represented by one review card.

    Returns:
        Stable identifier exported by the browser review page.
    """

    digest = hashlib.sha256(annotation.annotation_id.encode("utf-8")).hexdigest()[:16]
    return f"language_goal_review:{digest}"


def _render_html_v1(*, title: str, payload: str) -> str:
    """Render a portable browser UI with optional, non-forced adjudication."""

    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>{safe_title}</title>
<style>
:root{{--bg:#eef2f7;--panel:#fff;--ink:#172033;--muted:#5b6b80;--line:#cfd8e3;--accent:#0f766e;--warn:#b45309;--bad:#b91c1c}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}} header{{padding:18px 28px;background:#102033;color:#fff}}h1{{font-size:22px;margin:0}}main{{max-width:1500px;margin:auto;padding:18px}}.toolbar,.meta,.actions{{display:flex;flex-wrap:wrap;align-items:center;gap:8px}}.toolbar{{margin-bottom:14px}}button,select,input,textarea{{font:inherit;border:1px solid var(--line);border-radius:6px;padding:7px 9px;background:#fff}}button{{cursor:pointer}}button.primary{{background:var(--accent);color:#fff;border-color:var(--accent)}}.layout{{display:grid;grid-template-columns:minmax(360px,1.7fr) minmax(360px,1fr);gap:16px}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}}.frame{{position:relative;background:#101820;min-height:240px;display:grid;place-items:center;overflow:hidden}}.frame img{{display:block;max-width:100%;max-height:72vh}}.bbox{{position:absolute;border:3px solid #ef4444;box-shadow:0 0 0 1px #fff inset;pointer-events:none}}.label{{color:var(--muted);font-size:12px;text-transform:uppercase;font-weight:700;letter-spacing:.04em}}.instruction{{font-size:19px;font-weight:650;margin:5px 0 16px}}.pill{{display:inline-block;border-radius:999px;padding:3px 9px;background:#e8f5f1;color:#065f46;font-size:13px}}.pill.warn{{background:#fff4e5;color:#9a3412}}.pill.bad{{background:#feecec;color:#991b1b}}.kv{{padding:8px 0;border-top:1px solid var(--line)}}.kv:first-of-type{{border-top:0}}textarea{{width:100%;min-height:74px;resize:vertical}}.muted{{color:var(--muted)}}.empty{{padding:20px;color:var(--muted)}}@media(max-width:900px){{main{{padding:10px}}header{{padding:14px}}.layout{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>{safe_title}</h1><div class=\"muted\">仅含当前 FPV 与模型可见语言标注；不含 future trajectory 或候选坐标。</div></header>
<main><div class=\"toolbar\"><button id=\"prev\">上一条</button><button id=\"next\">下一条</button><label>序号 <input id=\"index\" type=\"number\" min=\"1\" value=\"1\"></label><label>类型 <select id=\"type\"><option value=\"\">全部</option></select></label><span id=\"counter\"></span><button id=\"export\" class=\"primary\">导出已裁决样本</button></div><div id=\"root\"></div></main>
<script>const allSamples={payload};let filtered=allSamples.slice(),cursor=0;const decisions=new Map();const labels={{accept:'接受',reject_not_visible:'拒绝：目标不可见',reject_wrong_relation:'拒绝：方向/关系错误',reject_generic:'拒绝：过度泛化',needs_rewrite:'需要改写',uncertain:'不确定'}};const esc=(v)=>String(v??'').replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));const type=document.querySelector('#type');for(const v of [...new Set(allSamples.map(x=>x.instruction_type||'unknown'))].sort()){{type.insertAdjacentHTML('beforeend',`<option value="${{esc(v)}}">${{esc(v)}}</option>`);}}function render(){{if(!filtered.length){{document.querySelector('#root').innerHTML='<div class="panel empty">没有满足筛选条件的样本。</div>';return;}}cursor=Math.max(0,Math.min(cursor,filtered.length-1));const s=filtered[cursor],d=decisions.get(s.review_id)||{{}};document.querySelector('#index').value=cursor+1;document.querySelector('#counter').textContent=`${{cursor+1}} / ${{filtered.length}}`;const b=s.target_bbox_anchor_xyxy;const bbox=b?`<div class="bbox" style="left:${{b[0]*100}}%;top:${{b[1]*100}}%;width:${{(b[2]-b[0])*100}}%;height:${{(b[3]-b[1])*100}}%"></div>`:'';const image=s.image_data_uri?`<div class="frame"><img src="${{s.image_data_uri}}" alt="fpv.current">${{bbox}}</div>`:`<div class="frame empty">${{esc(s.image_error||'图像不可用')}}</div>`;document.querySelector('#root').innerHTML=`<div class="layout"><section class="panel">${{image}}<p class="muted">红框表示 LVLM 提交的可选 anchor 区域；走廊或过街区域也可能使用较大的框。</p></section><section class="panel"><div class="meta"><span class="pill">${{esc(s.validator_status)}}</span><span class="pill">${{esc(s.confidence)}}</span><span class="pill warn">${{esc(s.instruction_type)}}</span></div><div class="label">生成的导航指令</div><div class="instruction">${{esc(s.instruction_text)}}</div><div class="kv"><b>已有数据文本</b><br>${{esc(s.existing_dataset_text||'(无)')}}</div><div class="kv"><b>可见证据绑定</b><br>target=${{esc(s.target_ref)}}; visible=${{esc(s.visibility_at_anchor)}}; evidence=${{esc((s.evidence_refs||[]).join(', '))}}</div><div class="kv"><b>空间关系</b><br>${{esc(s.spatial_relation||'(无)')}}</div><div class="kv"><b>样本 ID</b><br><span class="muted">${{esc(s.parent_sample_id)}}</span></div><div class="kv"><b>教师配置</b><br><span class="muted">${{esc(s.teacher_model)}} / ${{esc(s.prompt_version)}}</span></div><div class="kv"><div class="label">人工裁决（可选）</div><div class="actions">${{Object.entries(labels).map(([v,label])=>`<label><input type="radio" name="decision" value="${{v}}" ${{d.decision===v?'checked':''}}> ${{label}}</label>`).join('')}}</div></div><div class="kv"><label class="label" for="note">备注</label><textarea id="note" placeholder="可选的人工校阅备注">${{esc(d.note||'')}}</textarea></div></section></div>`;document.querySelectorAll('input[name="decision"]').forEach(el=>el.addEventListener('change',()=>{{decisions.set(s.review_id,{{...d,review_id:s.review_id,parent_sample_id:s.parent_sample_id,decision:el.value,note:document.querySelector('#note').value}});}}));document.querySelector('#note').addEventListener('input',el=>{{const prior=decisions.get(s.review_id)||{{review_id:s.review_id,parent_sample_id:s.parent_sample_id,decision:null}};decisions.set(s.review_id,{{...prior,note:el.target.value}});}});}}document.querySelector('#prev').onclick=()=>{{cursor--;render();}};document.querySelector('#next').onclick=()=>{{cursor++;render();}};document.querySelector('#index').onchange=e=>{{cursor=Number(e.target.value)-1;render();}};type.onchange=e=>{{filtered=allSamples.filter(s=>!e.target.value||s.instruction_type===e.target.value);cursor=0;render();}};document.querySelector('#export').onclick=()=>{{const rows=[...decisions.values()].filter(x=>x.decision);const blob=new Blob([JSON.stringify(rows,null,2)+'\\n'],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='language_goal_human_adjudications.json';a.click();URL.revokeObjectURL(a.href);}};render();</script></body></html>"""


__all__ = [
    "LANGUAGE_WAYPOINT_REVIEW_SCHEMA_VERSION_V1",
    "LanguageWaypointReviewConfigV1",
    "language_goal_review_id_v1",
    "write_language_waypoint_review_html_v1",
]
