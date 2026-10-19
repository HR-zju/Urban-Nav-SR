#!/usr/bin/env python3
"""Build a portable, stratified human-review page for Navigation VQA V2."""

from __future__ import annotations

import argparse
import base64
import hashlib
import heapq
import html
import io
import json
import mimetypes
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.agent.vqa_v2_text import render_vqa_answer_v2
from mcp_server.contracts.vqa_dataset_v2 import (
    VQAAnnotationV2,
    VQASourceV2,
    VQATrainingViewV2,
)
from mcp_server.data.io.jsonl import iter_jsonl
from mcp_server.data.vqa.review import VQA_REVIEW_DECISION_V2_SCHEMA_VERSION


_REVIEW_REPORT_SCHEMA_VERSION = (
    "urbannav_navigation_vqa_v2_review_report_v2"
)
_REVIEW_ISSUE_CODES = (
    "question_not_grounded",
    "answer_incorrect",
    "answer_unsupported",
    "modality_mismatch",
    "evidence_reference_error",
    "task_not_applicable",
    "unsafe_navigation_judgment",
    "schema_or_format_error",
    "duplicate_or_leakage",
    "other",
)


@dataclass(frozen=True)
class VQAReviewDatasetInput:
    """Describe one dataset's immutable VQA review artifacts.

    Args:
        dataset_name: Stable review filter name.
        source_jsonl: VQA source manifest.
        annotation_jsonl: Accepted annotation manifest.
        views_jsonl: Optional training-view manifest.
    """

    dataset_name: str
    source_jsonl: Path
    annotation_jsonl: Path
    views_jsonl: Path | None = None


@dataclass(frozen=True)
class _SelectedAnnotation:
    dataset_name: str
    annotation: VQAAnnotationV2
    score: int


class _ImageAssetStore:
    """Embed each local image once as a shared portable review asset."""

    def __init__(
        self,
        *,
        embed: bool,
        max_edge: int,
        quality: int,
    ) -> None:
        self.embed = embed
        self.max_edge = max_edge
        self.quality = quality
        self.assets: dict[str, str] = {}
        self.failures: Counter[str] = Counter()

    def add(self, value: Any) -> str | None:
        """Resolve one image reference and return its shared asset key."""

        text = str(value or "").strip()
        if not text:
            self.failures["empty"] += 1
            return None
        key = "image-" + hashlib.sha256(
            (
                f"{text}\0{self.embed}\0{self.max_edge}\0"
                f"{self.quality}"
            ).encode("utf-8")
        ).hexdigest()[:20]
        if key in self.assets:
            return key
        source = _image_source(
            text,
            embed=self.embed,
            max_edge=self.max_edge,
            quality=self.quality,
        )
        if not source:
            self.failures["unavailable"] += 1
            return None
        self.assets[key] = source
        return key


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse Navigation VQA V2 review-builder arguments.

    Args:
        argv: Optional explicit CLI arguments.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--annotation-jsonl", type=Path)
    parser.add_argument("--views-jsonl", type=Path)
    parser.add_argument(
        "--dataset-name",
        default="dataset",
        help="Filter name for the single-manifest mode.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help=(
            "Run root containing datasets/<name>/vqa_sources.jsonl and "
            "datasets/<name>/annotations/vqa_annotations.jsonl."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=("foxglove", "sidewalk", "egowalk"),
    )
    parser.add_argument(
        "--include-training-views",
        action="store_true",
        help="Also scan selected rows from each multi-dataset view manifest.",
    )
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument(
        "--title",
        default="UrbanNav Navigation VQA V2 Human Review",
    )
    parser.add_argument("--samples-per-stratum", type=int, default=3)
    parser.add_argument("--sampling-seed", type=int, default=20260804)
    parser.add_argument("--max-annotations", type=int, default=180)
    parser.add_argument("--image-max-edge", type=int, default=960)
    parser.add_argument("--image-quality", type=int, default=78)
    parser.add_argument(
        "--link-images",
        action="store_true",
        help="Link local files instead of embedding portable thumbnails.",
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
    """Build a backward-compatible single-dataset VQA review page.

    Args:
        source_jsonl: Frozen VQA source rows.
        annotation_jsonl: Accepted typed annotation rows.
        output_html: Destination standalone HTML file.
        views_jsonl: Optional exported training-view rows.
        report_json: Optional machine-readable review summary.
        title: Review page title.
        max_annotations: Maximum accepted annotations to render.
        embed_images: Embed local image thumbnails when true.

    Returns:
        Counts and provenance for the generated review page.
    """

    return build_vqa_v2_multi_dataset_review_html(
        datasets=(
            VQAReviewDatasetInput(
                dataset_name="dataset",
                source_jsonl=Path(source_jsonl).expanduser(),
                annotation_jsonl=Path(annotation_jsonl).expanduser(),
                views_jsonl=(
                    Path(views_jsonl).expanduser()
                    if views_jsonl is not None
                    else None
                ),
            ),
        ),
        output_html=output_html,
        report_json=report_json,
        title=title,
        samples_per_stratum=max(1, int(max_annotations)),
        sampling_seed=0,
        max_annotations=max_annotations,
        embed_images=embed_images,
    )


def build_vqa_v2_multi_dataset_review_html(
    *,
    datasets: Sequence[VQAReviewDatasetInput],
    output_html: str | Path,
    report_json: str | Path | None = None,
    title: str = "UrbanNav Navigation VQA V2 Human Review",
    samples_per_stratum: int = 3,
    sampling_seed: int = 20260804,
    max_annotations: int = 180,
    embed_images: bool = True,
    image_max_edge: int = 960,
    image_quality: int = 78,
) -> dict[str, Any]:
    """Build one deterministic, portable review page across datasets.

    Args:
        datasets: Dataset-specific source, annotation, and optional view paths.
        output_html: Destination standalone HTML file.
        report_json: Optional machine-readable review summary.
        title: Review page title.
        samples_per_stratum: Maximum rows per dataset/task/label-source cell.
        sampling_seed: Stable hash-sampling seed.
        max_annotations: Global review-card limit after stratification.
        embed_images: Embed shared local image thumbnails when true.
        image_max_edge: Longest embedded thumbnail edge in pixels.
        image_quality: JPEG thumbnail quality from 1 through 95.

    Returns:
        Exact scan, sampling, rendering, and provenance summary.

    Raises:
        ValueError: If paths, identities, or sampling settings are invalid.
    """

    if not datasets:
        raise ValueError("at least one VQA review dataset is required")
    if samples_per_stratum <= 0:
        raise ValueError("samples_per_stratum must be positive")
    if max_annotations < 0:
        raise ValueError("max_annotations cannot be negative")
    if image_max_edge <= 0:
        raise ValueError("image_max_edge must be positive")
    if not 1 <= image_quality <= 95:
        raise ValueError("image_quality must be in [1, 95]")
    names = [item.dataset_name for item in datasets]
    if len(set(names)) != len(names):
        raise ValueError("review dataset names must be unique")
    for item in datasets:
        if not item.source_jsonl.is_file():
            raise ValueError(f"VQA source is absent: {item.source_jsonl}")
        if not item.annotation_jsonl.is_file():
            raise ValueError(
                f"VQA annotation is absent: {item.annotation_jsonl}"
            )

    selected, scan_summary = _select_stratified_annotations(
        datasets=datasets,
        samples_per_stratum=samples_per_stratum,
        sampling_seed=sampling_seed,
        max_annotations=max_annotations,
    )
    review_set_id = _review_set_id(
        selected,
        samples_per_stratum=samples_per_stratum,
        sampling_seed=sampling_seed,
    )
    sources, source_scan_counts = _load_selected_sources(
        datasets,
        selected,
    )
    views, view_scan_counts = _load_selected_views(datasets, selected)
    image_store = _ImageAssetStore(
        embed=embed_images,
        max_edge=image_max_edge,
        quality=image_quality,
    )
    task_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    rendered_source_ids: set[tuple[str, str]] = set()
    cards: list[str] = []
    rendered_images = 0
    rendered_views = 0
    seen_annotations: set[str] = set()
    for item in selected:
        annotation = item.annotation
        if annotation.annotation_id in seen_annotations:
            raise ValueError(
                f"duplicate selected annotation id: {annotation.annotation_id}"
            )
        seen_annotations.add(annotation.annotation_id)
        source_key = (item.dataset_name, annotation.source_group_id)
        try:
            source = sources[source_key]
        except KeyError as exc:
            raise ValueError(
                "selected annotation source is absent: "
                f"{source_key}"
            ) from exc
        if annotation.source_fingerprint != source.source_fingerprint:
            raise ValueError("annotation/source fingerprint mismatch")
        view = views.get(annotation.annotation_id)
        if view is not None and (
            view.source_group_id != annotation.source_group_id
            or view.question.task_id != annotation.question.task_id
            or view.label_source != annotation.label_source
        ):
            raise ValueError("annotation/training-view identity mismatch")
        task_counts[annotation.question.task_id] += 1
        label_counts[annotation.label_source] += 1
        dataset_counts[item.dataset_name] += 1
        rendered_source_ids.add(source_key)
        rendered_images += len(source.observation.images)
        rendered_views += int(view is not None)
        cards.append(
            _render_annotation(
                dataset_name=item.dataset_name,
                review_set_id=review_set_id,
                source=source,
                annotation=annotation,
                view=view,
                image_store=image_store,
            )
        )

    summary: dict[str, Any] = {
        "schema_version": _REVIEW_REPORT_SCHEMA_VERSION,
        "review_decision_schema_version": (
            VQA_REVIEW_DECISION_V2_SCHEMA_VERSION
        ),
        "review_set_id": review_set_id,
        "sampling_strategy": "sha256_top_k_per_dataset_task_label_source",
        "sampling_seed": sampling_seed,
        "samples_per_stratum": samples_per_stratum,
        "max_annotations": max_annotations,
        "annotation_rows_scanned": scan_summary[
            "annotation_rows_scanned"
        ],
        "available_stratum_counts": scan_summary[
            "available_stratum_counts"
        ],
        "source_rows_scanned": dict(sorted(source_scan_counts.items())),
        "view_rows_scanned": dict(sorted(view_scan_counts.items())),
        "rendered_source_groups": len(rendered_source_ids),
        "rendered_annotations": len(selected),
        "rendered_images": rendered_images,
        "unique_embedded_image_assets": len(image_store.assets),
        "image_failures": dict(sorted(image_store.failures.items())),
        "rendered_views": rendered_views,
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "label_source_counts": dict(sorted(label_counts.items())),
        "selected_stratum_counts": scan_summary[
            "selected_stratum_counts"
        ],
        "embedded_images": bool(embed_images),
        "image_max_edge": image_max_edge,
        "image_quality": image_quality,
        "output_html": str(Path(output_html).expanduser()),
        "inputs": [
            {
                "dataset_name": item.dataset_name,
                "source_jsonl": str(item.source_jsonl),
                "annotation_jsonl": str(item.annotation_jsonl),
                "views_jsonl": (
                    str(item.views_jsonl)
                    if item.views_jsonl is not None
                    else None
                ),
            }
            for item in datasets
        ],
    }
    page = _render_page(
        title=title,
        cards=cards,
        summary=summary,
        image_assets=image_store.assets,
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
    """Build a standalone VQA V2 review page and print its summary.

    Args:
        argv: Optional explicit CLI arguments.

    Returns:
        ``None``.
    """

    args = parse_args(argv)
    if args.dataset_root is not None:
        datasets = _discover_dataset_inputs(
            args.dataset_root,
            args.datasets,
            include_training_views=args.include_training_views,
        )
        result = build_vqa_v2_multi_dataset_review_html(
            datasets=datasets,
            output_html=args.output_html,
            report_json=args.report_json,
            title=args.title,
            samples_per_stratum=args.samples_per_stratum,
            sampling_seed=args.sampling_seed,
            max_annotations=args.max_annotations,
            embed_images=not args.link_images,
            image_max_edge=args.image_max_edge,
            image_quality=args.image_quality,
        )
    else:
        if args.source_jsonl is None or args.annotation_jsonl is None:
            raise ValueError(
                "single mode requires --source-jsonl and --annotation-jsonl"
            )
        result = build_vqa_v2_multi_dataset_review_html(
            datasets=(
                VQAReviewDatasetInput(
                    dataset_name=args.dataset_name,
                    source_jsonl=args.source_jsonl.expanduser(),
                    annotation_jsonl=args.annotation_jsonl.expanduser(),
                    views_jsonl=(
                        args.views_jsonl.expanduser()
                        if args.views_jsonl is not None
                        else None
                    ),
                ),
            ),
            output_html=args.output_html,
            report_json=args.report_json,
            title=args.title,
            samples_per_stratum=args.samples_per_stratum,
            sampling_seed=args.sampling_seed,
            max_annotations=args.max_annotations,
            embed_images=not args.link_images,
            image_max_edge=args.image_max_edge,
            image_quality=args.image_quality,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _discover_dataset_inputs(
    dataset_root: Path,
    dataset_names: Sequence[str],
    *,
    include_training_views: bool,
) -> tuple[VQAReviewDatasetInput, ...]:
    """Resolve standard VQA run-root paths for requested datasets."""

    root = dataset_root.expanduser()
    result: list[VQAReviewDatasetInput] = []
    for name in dataset_names:
        base = root / "datasets" / name
        view_path = base / "views" / "vqa_training_views.jsonl"
        result.append(
            VQAReviewDatasetInput(
                dataset_name=name,
                source_jsonl=base / "vqa_sources.jsonl",
                annotation_jsonl=(
                    base / "annotations" / "vqa_annotations.jsonl"
                ),
                views_jsonl=(
                    view_path if include_training_views else None
                ),
            )
        )
    return tuple(result)


def _select_stratified_annotations(
    *,
    datasets: Sequence[VQAReviewDatasetInput],
    samples_per_stratum: int,
    sampling_seed: int,
    max_annotations: int,
) -> tuple[list[_SelectedAnnotation], dict[str, Any]]:
    """Select deterministic top-k hashes per dataset/task/source stratum."""

    heaps: dict[
        tuple[str, str, str],
        list[tuple[int, str, int, VQAAnnotationV2]],
    ] = defaultdict(list)
    available: Counter[tuple[str, str, str]] = Counter()
    scanned: Counter[str] = Counter()
    serial = 0
    for dataset in datasets:
        for row in iter_jsonl(dataset.annotation_jsonl):
            annotation = VQAAnnotationV2.model_validate(row)
            key = (
                dataset.dataset_name,
                annotation.question.task_id,
                annotation.label_source,
            )
            available[key] += 1
            scanned[dataset.dataset_name] += 1
            score = _sample_score(
                sampling_seed,
                dataset.dataset_name,
                annotation.annotation_id,
            )
            serial += 1
            entry = (-score, annotation.annotation_id, serial, annotation)
            heap = heaps[key]
            if len(heap) < samples_per_stratum:
                heapq.heappush(heap, entry)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, entry)

    by_stratum: dict[tuple[str, str, str], list[_SelectedAnnotation]] = {}
    for key, heap in heaps.items():
        by_stratum[key] = sorted(
            (
                _SelectedAnnotation(
                    dataset_name=key[0],
                    annotation=entry[3],
                    score=-entry[0],
                )
                for entry in heap
            ),
            key=lambda item: (item.score, item.annotation.annotation_id),
        )
    selected: list[_SelectedAnnotation] = []
    offsets: Counter[tuple[str, str, str]] = Counter()
    keys = sorted(by_stratum)
    while len(selected) < max_annotations:
        added = False
        for key in keys:
            offset = offsets[key]
            values = by_stratum[key]
            if offset >= len(values):
                continue
            selected.append(values[offset])
            offsets[key] += 1
            added = True
            if len(selected) >= max_annotations:
                break
        if not added:
            break
    selected_counts = Counter(
        (
            item.dataset_name,
            item.annotation.question.task_id,
            item.annotation.label_source,
        )
        for item in selected
    )
    return selected, {
        "annotation_rows_scanned": dict(sorted(scanned.items())),
        "available_stratum_counts": {
            _stratum_name(key): value
            for key, value in sorted(available.items())
        },
        "selected_stratum_counts": {
            _stratum_name(key): value
            for key, value in sorted(selected_counts.items())
        },
    }


def _load_selected_sources(
    datasets: Sequence[VQAReviewDatasetInput],
    selected: Sequence[_SelectedAnnotation],
) -> tuple[dict[tuple[str, str], VQASourceV2], Counter[str]]:
    """Stream source manifests and retain only sampled source identities."""

    wanted: dict[str, set[str]] = defaultdict(set)
    for item in selected:
        wanted[item.dataset_name].add(item.annotation.source_group_id)
    result: dict[tuple[str, str], VQASourceV2] = {}
    scanned: Counter[str] = Counter()
    for dataset in datasets:
        remaining = set(wanted[dataset.dataset_name])
        if not remaining:
            continue
        for row in iter_jsonl(dataset.source_jsonl):
            scanned[dataset.dataset_name] += 1
            raw_id = str(
                (row.get("observation") or {}).get("source_group_id") or ""
            )
            if raw_id not in remaining:
                continue
            source = VQASourceV2.model_validate(row)
            key = (dataset.dataset_name, raw_id)
            if key in result:
                raise ValueError(f"duplicate selected VQA source: {key}")
            result[key] = source
            remaining.remove(raw_id)
            if not remaining:
                break
        if remaining:
            raise ValueError(
                "selected VQA sources are absent from manifest "
                f"{dataset.dataset_name}: {sorted(remaining)[:5]}"
            )
    return result, scanned


def _load_selected_views(
    datasets: Sequence[VQAReviewDatasetInput],
    selected: Sequence[_SelectedAnnotation],
) -> tuple[dict[str, VQATrainingViewV2], Counter[str]]:
    """Stream optional training views and retain only sampled annotations."""

    wanted: dict[str, set[str]] = defaultdict(set)
    for item in selected:
        wanted[item.dataset_name].add(item.annotation.annotation_id)
    result: dict[str, VQATrainingViewV2] = {}
    scanned: Counter[str] = Counter()
    for dataset in datasets:
        if dataset.views_jsonl is None:
            continue
        if not dataset.views_jsonl.is_file():
            raise ValueError(
                f"VQA training view is absent: {dataset.views_jsonl}"
            )
        remaining = set(wanted[dataset.dataset_name])
        for row in iter_jsonl(dataset.views_jsonl):
            scanned[dataset.dataset_name] += 1
            annotation_id = str(row.get("annotation_id") or "")
            if annotation_id not in remaining:
                continue
            view = VQATrainingViewV2.model_validate(row)
            if annotation_id in result:
                raise ValueError(
                    f"duplicate selected VQA training view: {annotation_id}"
                )
            result[annotation_id] = view
            remaining.remove(annotation_id)
            if not remaining:
                break
    return result, scanned


def _render_page(
    *,
    title: str,
    cards: list[str],
    summary: Mapping[str, Any],
    image_assets: Mapping[str, str],
) -> str:
    """Render the complete standalone review document."""

    metrics = (
        ("Review rows", summary["rendered_annotations"]),
        ("Source groups", summary["rendered_source_groups"]),
        ("Datasets", len(summary["dataset_counts"])),
        ("Tasks", len(summary["task_counts"])),
        ("Derived GT", summary["label_source_counts"].get("derived_gt", 0)),
        ("LVLM weak", summary["label_source_counts"].get("lvlm_weak", 0)),
    )
    metric_html = "".join(
        "<div class='metric'><strong>"
        f"{_escape(value)}</strong><span>{_escape(label)}</span></div>"
        for label, value in metrics
    )
    dataset_options = "".join(
        f"<option>{_escape(value)}</option>"
        for value in sorted(summary["dataset_counts"])
    )
    task_options = "".join(
        f"<option>{_escape(value)}</option>"
        for value in sorted(summary["task_counts"])
    )
    assets_json = json.dumps(
        image_assets,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape(title)}</title>
<style>
:root{{--ink:#182330;--muted:#657484;--line:#ccd6df;--soft:#f3f6f8;--bg:#eaf0f4;--accent:#135ea8;--good:#177549;--bad:#ad3028;--warn:#9a6200}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:5;background:#fffffffa;border-bottom:1px solid var(--line);padding:12px 18px}}
.head{{max-width:1580px;margin:auto;display:flex;gap:18px;justify-content:space-between;align-items:center}}h1{{font-size:20px;margin:0}}
.muted{{color:var(--muted);font-size:12px;overflow-wrap:anywhere}}main{{max-width:1580px;margin:16px auto;padding:0 16px 60px}}
.toolbar{{display:flex;gap:8px;align-items:end;flex-wrap:wrap;margin:12px 0}}label.control{{display:grid;gap:3px;color:var(--muted);font-size:11px}}
input,select,textarea,button{{font:inherit;border:1px solid #9eacba;border-radius:6px;background:#fff;padding:7px 9px}}input,select{{height:36px}}button{{cursor:pointer;background:var(--accent);border-color:var(--accent);color:#fff;font-weight:650}}button.secondary{{background:#fff;color:var(--accent)}}
.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:12px 0 16px}}.metric,.sample{{background:#fff;border:1px solid var(--line);border-radius:7px}}.metric{{padding:10px 13px}}.metric strong{{display:block;font-size:21px}}.metric span{{font-size:12px;color:var(--muted)}}
.sample{{margin-bottom:16px;overflow:hidden}}.sample-head{{padding:12px 15px;background:#f8fafb;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:14px}}h2{{font-size:15px;margin:0 0 3px}}.badges{{display:flex;gap:5px;align-items:flex-start;flex-wrap:wrap}}.badge{{display:inline-block;border:1px solid #a9b5c0;padding:2px 7px;font-size:11px;font-weight:700;background:#fff}}.derived_gt{{color:var(--good);border-color:#83c4a5;background:#ecf8f1}}.lvlm_weak{{color:#7040aa;border-color:#c2a8e5;background:#f6f0fc}}
.body{{padding:14px 15px}}.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:9px}}figure{{margin:0;background:#10161c;border:1px solid var(--line)}}figure img{{width:100%;height:280px;display:block;object-fit:contain}}figcaption{{padding:6px 8px;color:#fff;font-size:11px;overflow-wrap:anywhere}}
.grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(360px,.72fr);gap:14px;margin-top:12px}}section{{border-top:1px solid var(--line);padding:11px 0}}section:first-child{{border-top:0}}h3{{margin:0 0 6px;font-size:12px;text-transform:uppercase;color:#3c4b5a}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--soft);border:1px solid #dfe6ec;padding:9px 10px;margin:0;font:12px/1.45 ui-monospace,monospace}}
details{{margin-top:8px}}summary{{cursor:pointer;font-weight:700;color:var(--accent)}}.review-form{{margin-top:13px;border-top:2px solid #9bb8d5;padding-top:12px}}.review-row{{display:grid;grid-template-columns:210px 1fr;gap:12px}}.review-form textarea{{width:100%;min-height:74px;resize:vertical}}fieldset{{border:1px solid var(--line);border-radius:6px;padding:8px;margin:0}}fieldset legend{{font-size:11px;color:var(--muted)}}.issues{{display:flex;gap:8px 14px;flex-wrap:wrap}}.issues label{{font-size:12px}}.issues input{{height:auto;padding:0}}.review-state{{font-size:12px;font-weight:700}}.reviewed{{color:var(--good)}}.unreviewed{{color:var(--warn)}}.hidden{{display:none!important}}
@media(max-width:950px){{header{{position:static}}.head,.grid,.review-row{{display:block}}.metrics{{grid-template-columns:repeat(2,1fr)}}figure img{{height:auto;max-height:420px}}}}
</style></head><body>
<header><div class="head"><div><h1>{_escape(title)}</h1><div class="muted">review_set={_escape(summary['review_set_id'])} · sampled {_escape(summary['rendered_annotations'])} from {_escape(sum(summary['annotation_rows_scanned'].values()))} annotations</div></div><div id="progress" class="review-state unreviewed">Reviewed 0 / {_escape(summary['rendered_annotations'])}</div></div></header>
<main><div class="toolbar">
<label class="control">Reviewer<input id="reviewer" type="text" placeholder="name or stable ID"></label>
<label class="control">Search<input id="query" type="search" placeholder="task, question, answer"></label>
<label class="control">Dataset<select id="dataset-filter"><option value="">all</option>{dataset_options}</select></label>
<label class="control">Task<select id="task-filter"><option value="">all</option>{task_options}</select></label>
<label class="control">Label source<select id="source-filter"><option value="">all</option><option>derived_gt</option><option>lvlm_weak</option><option>external_gt</option></select></label>
<label class="control">Review verdict<select id="verdict-filter"><option value="">all</option><option>unreviewed</option><option>keep</option><option>reject</option><option>needs_fix</option></select></label>
<button id="save-draft" class="secondary" type="button">Save draft</button>
<button id="export-jsonl" type="button">Export reviewed JSONL</button>
<label class="control">Import decisions<input id="import-jsonl" type="file" accept=".jsonl,.json"></label>
</div><div id="status" class="muted"></div><div class="metrics">{metric_html}</div>{''.join(cards) or '<p>No accepted annotations.</p>'}</main>
<script type="application/json" id="image-assets">{assets_json}</script>
<script>
const schemaVersion={json.dumps(VQA_REVIEW_DECISION_V2_SCHEMA_VERSION)};
const reviewSetId={json.dumps(summary['review_set_id'])};
const storageKey=`urbannav-vqa-v2-review-${{reviewSetId}}`;
const forms=[...document.querySelectorAll('.review-form')];
const reviewer=document.getElementById('reviewer');
const assets=JSON.parse(document.getElementById('image-assets').textContent||'{{}}');
document.querySelectorAll('img[data-image-key]').forEach(img=>{{img.src=assets[img.dataset.imageKey]||'';}});
function issueValues(form){{return [...form.querySelectorAll('input[name=issue]:checked')].map(x=>x.value).sort();}}
function setIssues(form,values){{const set=new Set(values||[]);form.querySelectorAll('input[name=issue]').forEach(x=>x.checked=set.has(x.value));}}
function draftRow(form){{return {{verdict:form.querySelector('[name=verdict]').value,issue_codes:issueValues(form),note:form.querySelector('[name=note]').value}};}}
function decisionRow(form,requireReviewer=false){{const verdict=form.querySelector('[name=verdict]').value;if(verdict==='unreviewed')return null;const person=reviewer.value.trim();if(requireReviewer&&!person)throw new Error('Reviewer is required before export');const issues=issueValues(form);if(verdict!=='keep'&&!issues.length)throw new Error(`${{form.dataset.annotationId}}: reject/needs_fix requires an issue code`);const row={{schema_version:schemaVersion,review_set_id:reviewSetId,annotation_id:form.dataset.annotationId,source_group_id:form.dataset.sourceGroupId,source_fingerprint:form.dataset.sourceFingerprint,task_id:form.dataset.taskId,label_source:form.dataset.labelSource,reviewer:person,reviewed_at:new Date().toISOString(),verdict:verdict,issue_codes:issues}};const note=form.querySelector('[name=note]').value.trim();if(note)row.note=note;return row;}}
function updateCard(form){{const verdict=form.querySelector('[name=verdict]').value;const card=form.closest('.sample');card.dataset.verdict=verdict;const state=form.querySelector('.review-state');state.textContent=verdict;state.className=`review-state ${{verdict==='unreviewed'?'unreviewed':'reviewed'}}`;}}
function updateProgress(){{const reviewed=forms.filter(f=>f.querySelector('[name=verdict]').value!=='unreviewed').length;const el=document.getElementById('progress');el.textContent=`Reviewed ${{reviewed}} / ${{forms.length}}`;el.className=`review-state ${{reviewed?'reviewed':'unreviewed'}}`;}}
function saveDraft(){{const value={{reviewer:reviewer.value,rows:{{}}}};forms.forEach(form=>value.rows[form.dataset.annotationId]=draftRow(form));localStorage.setItem(storageKey,JSON.stringify(value));document.getElementById('status').textContent='Draft saved in this browser.';updateProgress();applyFilters();}}
function restoreDraft(){{const raw=localStorage.getItem(storageKey);if(!raw)return;const value=JSON.parse(raw);reviewer.value=value.reviewer||'';forms.forEach(form=>{{const row=value.rows?.[form.dataset.annotationId];if(!row)return;form.querySelector('[name=verdict]').value=row.verdict||'unreviewed';setIssues(form,row.issue_codes);form.querySelector('[name=note]').value=row.note||'';updateCard(form);}});}}
function applyFilters(){{const query=document.getElementById('query').value.toLowerCase();const dataset=document.getElementById('dataset-filter').value;const task=document.getElementById('task-filter').value;const source=document.getElementById('source-filter').value;const verdict=document.getElementById('verdict-filter').value;document.querySelectorAll('.sample').forEach(card=>{{card.classList.toggle('hidden',!!((query&&!card.dataset.search.includes(query))||(dataset&&card.dataset.dataset!==dataset)||(task&&card.dataset.task!==task)||(source&&card.dataset.source!==source)||(verdict&&card.dataset.verdict!==verdict)));}});}}
forms.forEach(form=>{{form.addEventListener('change',()=>{{updateCard(form);saveDraft();}});form.addEventListener('input',saveDraft);updateCard(form);}});
reviewer.addEventListener('input',saveDraft);document.getElementById('save-draft').addEventListener('click',saveDraft);
for(const id of ['query','dataset-filter','task-filter','source-filter','verdict-filter'])document.getElementById(id).addEventListener(id==='query'?'input':'change',applyFilters);
document.getElementById('export-jsonl').addEventListener('click',()=>{{try{{const rows=forms.map(f=>decisionRow(f,true)).filter(Boolean);if(!rows.length)throw new Error('No reviewed rows to export');const body=rows.map(row=>JSON.stringify(row)).join('\\n')+'\\n';const url=URL.createObjectURL(new Blob([body],{{type:'application/jsonl'}}));const link=document.createElement('a');link.href=url;link.download=`vqa_v2_review_decisions_${{reviewSetId}}.jsonl`;link.click();URL.revokeObjectURL(url);document.getElementById('status').textContent=`Exported ${{rows.length}} reviewed decisions.`;}}catch(error){{alert(error.message);}}}});
document.getElementById('import-jsonl').addEventListener('change',async event=>{{try{{const text=await event.target.files[0].text();const rows=text.split(/\\r?\\n/).filter(x=>x.trim()).map(x=>JSON.parse(x));const byId=new Map(forms.map(f=>[f.dataset.annotationId,f]));let imported=0;for(const row of rows){{if(row.schema_version!==schemaVersion)throw new Error('Decision schema mismatch');const form=byId.get(row.annotation_id);if(!form)continue;if(row.source_group_id!==form.dataset.sourceGroupId||row.source_fingerprint!==form.dataset.sourceFingerprint||row.task_id!==form.dataset.taskId||row.label_source!==form.dataset.labelSource)throw new Error(`Identity mismatch for ${{row.annotation_id}}`);form.querySelector('[name=verdict]').value=row.verdict;setIssues(form,row.issue_codes);form.querySelector('[name=note]').value=row.note||'';if(!reviewer.value&&row.reviewer)reviewer.value=row.reviewer;updateCard(form);imported++;}}saveDraft();document.getElementById('status').textContent=`Imported ${{imported}} matching decisions.`;}}catch(error){{alert(error.message);}}}});
restoreDraft();updateProgress();applyFilters();
</script></body></html>"""


def _render_annotation(
    *,
    dataset_name: str,
    review_set_id: str,
    source: VQASourceV2,
    annotation: VQAAnnotationV2,
    view: VQATrainingViewV2 | None,
    image_store: _ImageAssetStore,
) -> str:
    """Render one accepted question-answer record and review form."""

    observation = source.observation
    figures = _render_images(
        observation.images,
        observation.image_roles,
        image_store=image_store,
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
    canonical_target = render_vqa_answer_v2(annotation.answer)
    export = {
        "status": "training_view" if view is not None else "canonical_reconstructed",
        "assistant_text": (
            view.assistant_text if view is not None else canonical_target
        ),
        "sample_id": view.sample_id if view is not None else None,
        "prompt_profile_id": (
            view.prompt_profile_id
            if view is not None
            else annotation.prompt_profile_id
        ),
        "capability_config_digest": (
            view.capability_config_digest if view is not None else None
        ),
    }
    searchable = json.dumps(
        {
            "dataset": dataset_name,
            "annotation_id": annotation.annotation_id,
            "task_id": annotation.question.task_id,
            "question": question,
            "answer": answer,
            "provenance": provenance,
        },
        ensure_ascii=False,
    ).lower()
    issue_controls = "".join(
        "<label><input type='checkbox' name='issue' value='"
        f"{_escape(code)}'> {_escape(code)}</label>"
        for code in _REVIEW_ISSUE_CODES
    )
    return f"""<article class="sample" data-dataset="{_escape(dataset_name)}" data-task="{_escape(annotation.question.task_id)}" data-source="{_escape(annotation.label_source)}" data-verdict="unreviewed" data-search="{_escape(searchable)}">
<div class="sample-head"><div><h2>{_escape(annotation.question.task_id)}</h2><div class="muted">{_escape(annotation.annotation_id)} · {_escape(dataset_name)} · {_escape(annotation.source_fingerprint[:12])}</div></div><div class="badges"><span class="badge {_escape(annotation.label_source)}">{_escape(annotation.label_source)}</span><span class="badge">{_escape(annotation.producer_id)}</span><span class="review-state unreviewed">unreviewed</span></div></div>
<div class="body">{figures}<div class="grid"><div><section><h3>Answer-independent question</h3>{_json_pre(question)}</section><section><h3>Typed target payload</h3>{_json_pre(answer)}</section><section><h3>Canonical assistant target</h3>{_json_pre(canonical_target)}</section><section><h3>Complete student-visible input</h3>{_json_pre(visible)}</section></div>
<aside><section><h3>Label provenance</h3>{_json_pre(provenance)}</section><section><h3>Training export</h3>{_json_pre(export)}</section><section><h3>Validation</h3>{_json_pre({'typed_contract': 'accepted', 'source_fingerprint_match': True, 'private_interfaces_exported': False})}</section></aside></div>
<form class="review-form" data-review-set-id="{_escape(review_set_id)}" data-annotation-id="{_escape(annotation.annotation_id)}" data-source-group-id="{_escape(annotation.source_group_id)}" data-source-fingerprint="{_escape(annotation.source_fingerprint)}" data-task-id="{_escape(annotation.question.task_id)}" data-label-source="{_escape(annotation.label_source)}"><div class="review-row"><label class="control">Human verdict<select name="verdict"><option>unreviewed</option><option>keep</option><option>reject</option><option>needs_fix</option></select></label><fieldset><legend>Issue codes (required for reject / needs_fix)</legend><div class="issues">{issue_controls}</div></fieldset></div><label class="control">Review note<textarea name="note" placeholder="Describe visible evidence and the correction needed."></textarea></label></form>
<details><summary>Raw accepted annotation</summary>{_json_pre(annotation.model_dump(mode='json', exclude_none=True))}</details></div></article>"""


def _render_images(
    images: Sequence[str],
    roles: Sequence[str],
    *,
    image_store: _ImageAssetStore,
) -> str:
    """Render ordered prompt-visible images with explicit roles."""

    figures: list[str] = []
    for index, value in enumerate(images):
        role = roles[index] if index < len(roles) else f"image.{index}"
        key = image_store.add(value)
        body = (
            f"<img data-image-key='{_escape(key)}' alt='{_escape(role)}' "
            "loading='lazy'>"
            if key is not None
            else f"<pre>Unavailable image: {_escape(value)}</pre>"
        )
        figures.append(
            "<figure>"
            f"{body}<figcaption>{_escape(role)} · "
            f"{_escape(value)}</figcaption></figure>"
        )
    return f"<div class='images'>{''.join(figures)}</div>"


def _image_source(
    value: str,
    *,
    embed: bool,
    max_edge: int,
    quality: int,
) -> str:
    """Resolve one image as a thumbnail data URL or local file URI."""

    if value.startswith("data:image/"):
        return value
    if value.startswith(("https://", "http://")):
        return ""
    path = Path(value).expanduser()
    if not path.is_file():
        return ""
    if not embed:
        return path.resolve().as_uri()
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((max_edge, max_edge))
            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
            )
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"


def _sample_score(seed: int, dataset: str, annotation_id: str) -> int:
    """Return a deterministic hash score for one annotation."""

    digest = hashlib.sha256(
        f"{seed}\0{dataset}\0{annotation_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _review_set_id(
    selected: Sequence[_SelectedAnnotation],
    *,
    samples_per_stratum: int,
    sampling_seed: int,
) -> str:
    """Bind the exact sampled identities to a stable review-set ID."""

    payload = {
        "schema_version": _REVIEW_REPORT_SCHEMA_VERSION,
        "samples_per_stratum": samples_per_stratum,
        "sampling_seed": sampling_seed,
        "annotations": [
            [item.dataset_name, item.annotation.annotation_id]
            for item in selected
        ],
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"vqa-v2-{digest[:20]}"


def _stratum_name(key: tuple[str, str, str]) -> str:
    """Serialize a review stratum for reports."""

    return "::".join(key)


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
