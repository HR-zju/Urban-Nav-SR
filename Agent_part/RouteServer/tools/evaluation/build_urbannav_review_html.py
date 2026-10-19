#!/usr/bin/env python3
"""Build an HTML review page for UrbanNav benchmark runs."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from mcp_server.eval.urbannav.urbannav_io import iter_json_records, load_urbannav_cases, load_urbannav_predictions
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavCase, UrbanNavDestination, UrbanNavPoint, UrbanNavPrediction
from mcp_server.map_providers import (
    build_amap_staticmap_url,
    encode_polyline_text,
    format_location,
    sample_route_coordinates,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path, help="UrbanNav task_gt.jsonl")
    parser.add_argument("--predictions", type=Path, default=None, help="Optional prediction JSONL")
    parser.add_argument("--eval-rows", type=Path, default=None, help="Optional evaluator per-case JSONL")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output review directory")
    parser.add_argument("--title", default="UrbanNav Real API Review", help="HTML title")
    parser.add_argument("--max-cases", type=int, default=50, help="Maximum cases to render")
    parser.add_argument("--download-staticmaps", action="store_true", help="Download AMap staticmap images")
    parser.add_argument("--staticmap-size", default="800*600", help="AMap staticmap size")
    parser.add_argument("--staticmap-zoom", type=int, default=None, help="Optional AMap staticmap zoom")
    return parser.parse_args()


def main() -> None:
    """Build the review page.

    Returns:
        None.
    """

    args = parse_args()
    build_review_html(
        gt_path=args.ground_truth,
        output_dir=args.output_dir,
        predictions_path=args.predictions,
        eval_rows_path=args.eval_rows,
        title=args.title,
        max_cases=args.max_cases,
        download_staticmaps=args.download_staticmaps,
        staticmap_size=args.staticmap_size,
        staticmap_zoom=args.staticmap_zoom,
    )


def build_review_html(
    *,
    gt_path: str | Path,
    output_dir: str | Path,
    predictions_path: str | Path | None = None,
    eval_rows_path: str | Path | None = None,
    title: str = "UrbanNav Review",
    max_cases: int = 50,
    download_staticmaps: bool = False,
    staticmap_size: str = "800*600",
    staticmap_zoom: int | None = None,
) -> dict[str, Any]:
    """Build an UrbanNav HTML review artifact.

    Args:
        gt_path: Ground-truth JSONL path.
        output_dir: Output directory. The generated page is ``index.html``.
        predictions_path: Optional prediction JSONL path.
        eval_rows_path: Optional evaluator rows JSONL path.
        title: HTML page title.
        max_cases: Maximum cases to render.
        download_staticmaps: Whether to download local AMap static map images.
        staticmap_size: AMap staticmap size, such as ``"800*600"``.
        staticmap_zoom: Optional staticmap zoom level.

    Returns:
        Summary dictionary containing output paths and rendered counts.
    """

    cases = load_urbannav_cases(gt_path)[: max(0, max_cases)]
    predictions = _load_predictions_by_id(predictions_path)
    eval_rows = _load_eval_rows_by_id(eval_rows_path)
    root = Path(output_dir)
    image_dir = root / "images"
    root.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    cards: list[str] = []
    manifest: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        prediction = predictions.get(case.case_id)
        eval_row = eval_rows.get(case.case_id, {})
        gt_sequence = _destination_sequence(case)
        gt_image = _maybe_render_route_image(
            case=case,
            route=case.gt_route,
            output_path=image_dir / f"{_safe_name(case.case_id)}_gt.png",
            label="gt",
            download_staticmaps=download_staticmaps,
            staticmap_size=staticmap_size,
            staticmap_zoom=staticmap_zoom,
            via_points=[destination for destination in gt_sequence[:-1] if destination.point is not None],
        )
        pred_image = None
        if prediction is not None and prediction.route_polyline:
            pred_sequence = _prediction_destination_sequence(prediction)
            pred_image = _maybe_render_route_image(
                case=case,
                route=prediction.route_polyline,
                output_path=image_dir / f"{_safe_name(case.case_id)}_pred.png",
                label="pred",
                download_staticmaps=download_staticmaps,
                staticmap_size=staticmap_size,
                staticmap_zoom=staticmap_zoom,
                via_points=[destination for destination in pred_sequence[:-1] if destination.point is not None],
            )
        cards.append(_case_card(index=index, case=case, prediction=prediction, eval_row=eval_row, gt_image=gt_image, pred_image=pred_image))
        manifest.append(
            {
                "case_id": case.case_id,
                "gt_image": gt_image,
                "pred_image": pred_image,
                "has_prediction": prediction is not None,
                "eval": eval_row,
            }
        )

    html_text = _html_page(title=title, cards=cards, case_count=len(cases), gt_path=str(gt_path), predictions_path=str(predictions_path or ""))
    index_path = root / "index.html"
    index_path.write_text(html_text, encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"index_html": str(index_path), "manifest": str(manifest_path), "case_count": len(cases)}


def _load_predictions_by_id(path: str | Path | None) -> dict[str, UrbanNavPrediction]:
    """Load predictions keyed by case id.

    Args:
        path: Optional prediction JSONL path.

    Returns:
        Mapping from case id to prediction.
    """

    if not path:
        return {}
    return {prediction.case_id: prediction for prediction in load_urbannav_predictions(path)}


def _load_eval_rows_by_id(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load evaluator rows keyed by case id.

    Args:
        path: Optional evaluator rows JSONL path.

    Returns:
        Mapping from case id to raw row dictionary.
    """

    if not path:
        return {}
    return {str(row.get("case_id") or row.get("id")): row for row in iter_json_records(path)}


def _maybe_render_route_image(
    *,
    case: UrbanNavCase,
    route: list[UrbanNavPoint],
    output_path: Path,
    label: str,
    download_staticmaps: bool,
    staticmap_size: str,
    staticmap_zoom: int | None,
    via_points: list[UrbanNavDestination],
) -> str | None:
    """Render one route image through AMap staticmap when requested.

    Args:
        case: UrbanNav case.
        route: Route polyline.
        output_path: Local image path.
        label: Image label for error messages.
        download_staticmaps: Whether to download the staticmap.
        staticmap_size: Staticmap size.
        staticmap_zoom: Optional staticmap zoom.
        via_points: Ordered intermediate destinations to mark on the map.

    Returns:
        Relative image path for HTML, or ``None`` when no image was produced.
    """

    if not download_staticmaps or len(route) < 2:
        return None
    api_key = os.getenv("AMAP_KEY", "").strip()
    if not api_key:
        return None
    origin = case.origin or route[0]
    destination = route[-1]
    path_coords = [[point.lng, point.lat] for point in route]
    if via_points:
        url = _build_amap_multistop_staticmap_url(
            api_key=api_key,
            path_coords=path_coords,
            origin=origin,
            destination=destination,
            via_points=via_points,
            size=staticmap_size,
            zoom=staticmap_zoom,
        )
    else:
        url = build_amap_staticmap_url(
            api_key=api_key,
            path_coords=path_coords,
            origin=format_location(origin.lng, origin.lat),
            destination=format_location(destination.lng, destination.lat),
            size=staticmap_size,
            zoom=staticmap_zoom,
        )
    try:
        _download_binary(url, output_path)
    except Exception as exc:
        error_path = output_path.with_suffix(f".{label}.error.txt")
        error_path.write_text(str(exc), encoding="utf-8")
        return None
    return f"images/{output_path.name}"


def _build_amap_multistop_staticmap_url(
    *,
    api_key: str,
    path_coords: list[list[float]],
    origin: UrbanNavPoint,
    destination: UrbanNavPoint,
    via_points: list[UrbanNavDestination],
    size: str,
    zoom: int | None,
) -> str:
    r"""Build an AMap staticmap URL with ordered stop markers.

    The full route is a concatenation of ordered route legs:

    \[
    R = R_1 \oplus R_2 \oplus \cdots \oplus R_K
    \]

    Args:
        api_key: AMap Web Service key.
        path_coords: Full route coordinates in ``[lng, lat]`` order.
        origin: Route start.
        destination: Final destination.
        via_points: Ordered intermediate stops.
        size: Staticmap size, such as ``"800*600"``.
        zoom: Optional zoom level.

    Returns:
        Staticmap URL with ``S``, numbered via markers, and ``D``.
    """

    sampled_coords = sample_route_coordinates(
        path_coords,
        max_points=int(os.getenv("AMAP_STATIC_ROUTE_MAX_POINTS", os.getenv("STATIC_ROUTE_MAX_POINTS", "240")) or 240),
    )
    params: list[tuple[str, str]] = [
        ("key", api_key),
        ("size", size),
        ("scale", "2"),
        ("traffic", "0"),
        ("paths", f"6,0x2E8B57,1,,:{encode_polyline_text(sampled_coords)}"),
    ]
    if zoom is not None:
        params.append(("zoom", str(int(zoom))))

    marker_parts = [f"mid,0x2E86FF,S:{format_location(origin.lng, origin.lat)}"]
    for index, destination_item in enumerate(via_points, start=1):
        point = destination_item.point
        if point is None:
            continue
        marker_parts.append(f"mid,0xFFAA00,{index}:{format_location(point.lng, point.lat)}")
    marker_parts.append(f"mid,0xFF5533,D:{format_location(destination.lng, destination.lat)}")
    params.append(("markers", "|".join(marker_parts)))
    query = "&".join(f"{key}={value}" for key, value in params)
    return f"https://restapi.amap.com/v3/staticmap?{query}"


def _download_binary(url: str, output_path: Path) -> None:
    """Download binary content.

    Args:
        url: Source URL.
        output_path: Output file path.

    Returns:
        None.

    Raises:
        RuntimeError: If the response looks like a JSON API error.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeout_s = float(os.getenv("AMAP_STATICMAP_TIMEOUT_S", "30") or 30)
    with urlopen(url, timeout=timeout_s) as response:
        content = response.read()
    stripped = content[:64].lstrip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        raise RuntimeError(content.decode("utf-8", errors="replace"))
    output_path.write_bytes(content)


def _case_card(
    *,
    index: int,
    case: UrbanNavCase,
    prediction: UrbanNavPrediction | None,
    eval_row: dict[str, Any],
    gt_image: str | None,
    pred_image: str | None,
) -> str:
    """Build one HTML case card.

    Args:
        index: One-based case index.
        case: UrbanNav case.
        prediction: Optional prediction.
        eval_row: Optional evaluator row.
        gt_image: Relative GT image path.
        pred_image: Relative prediction image path.

    Returns:
        HTML fragment.
    """

    gt_dest = case.gt_destination.to_dict() if case.gt_destination else {}
    pred_dest = prediction.chosen_destination.to_dict() if prediction and prediction.chosen_destination else {}
    metrics = _metric_badges(eval_row)
    route_update = case.metadata.get("route_update") if isinstance(case.metadata.get("route_update"), dict) else {}
    multi_stop = _multi_stop_block(case)
    return f"""
    <article class="case-card">
      <header>
        <div>
          <div class="case-index">#{index} {_esc(case.case_id)}</div>
          <h2>{_esc(case.instruction)}</h2>
        </div>
        <div class="badges">{metrics}</div>
      </header>
      <section class="meta-grid">
        <div><strong>Category</strong><span>{_esc(case.category)} / {_esc(case.subcategory)}</span></div>
        <div><strong>Scene</strong><span>{_esc(case.city)} / {_esc(case.scene_id)}</span></div>
        <div><strong>Origin</strong><span>{_point_text(case.origin)}</span></div>
        <div><strong>GT Distance</strong><span>{_num(case.gt_distance_m)} m</span></div>
      </section>
      {_route_update_block(route_update)}
      {multi_stop}
      <section class="maps">
        {_image_block("GT route", gt_image)}
        {_image_block("Pred route", pred_image)}
      </section>
      <section class="details">
        <div>
          <h3>GT destination</h3>
          <pre>{_esc(json.dumps(gt_dest, ensure_ascii=False, indent=2))}</pre>
        </div>
        <div>
          <h3>Pred destination</h3>
          <pre>{_esc(json.dumps(pred_dest, ensure_ascii=False, indent=2))}</pre>
        </div>
      </section>
      {_pred_multi_stop_block(prediction)}
      <section class="details">
        <div>
          <h3>Acceptable destinations</h3>
          <pre>{_esc(json.dumps([item.to_dict() for item in case.acceptable_destinations], ensure_ascii=False, indent=2))}</pre>
        </div>
        <div>
          <h3>Eval row</h3>
          <pre>{_esc(json.dumps(eval_row, ensure_ascii=False, indent=2))}</pre>
        </div>
      </section>
    </article>
    """


def _html_page(*, title: str, cards: list[str], case_count: int, gt_path: str, predictions_path: str) -> str:
    """Build the full HTML document.

    Args:
        title: Page title.
        cards: Case card fragments.
        case_count: Number of rendered cases.
        gt_path: Ground-truth path.
        predictions_path: Prediction path.

    Returns:
        HTML document.
    """

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f7f8; color: #172026; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    .page-head {{ margin-bottom: 20px; }}
    .page-head h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .page-head p {{ margin: 4px 0; color: #5c6972; font-size: 14px; }}
    .case-card {{ background: #fff; border: 1px solid #dce3e8; border-radius: 8px; margin: 18px 0; padding: 18px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
    .case-card header {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; border-bottom: 1px solid #edf1f3; padding-bottom: 12px; }}
    .case-index {{ color: #60717d; font-size: 13px; font-weight: 700; }}
    h2 {{ margin: 6px 0 0; font-size: 20px; line-height: 1.35; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .badges {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; min-width: 260px; }}
    .badge {{ border-radius: 999px; padding: 5px 9px; font-size: 12px; font-weight: 700; background: #edf1f3; color: #34444f; }}
    .badge.ok {{ background: #dff3e7; color: #17633a; }}
    .badge.fail {{ background: #ffe3df; color: #9a2f21; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 14px 0; }}
    .meta-grid div {{ background: #f8fafb; border: 1px solid #e5ebef; border-radius: 6px; padding: 10px; }}
    .meta-grid strong {{ display: block; font-size: 12px; color: #60717d; margin-bottom: 5px; }}
    .meta-grid span {{ font-size: 14px; }}
    .maps {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 14px 0; }}
    .map-panel {{ border: 1px solid #dce3e8; border-radius: 6px; overflow: hidden; background: #fbfcfd; }}
    .map-panel h3 {{ padding: 10px; border-bottom: 1px solid #e5ebef; }}
    .map-panel img {{ width: 100%; display: block; background: #eef3f5; }}
    .empty-map {{ min-height: 220px; display: flex; align-items: center; justify-content: center; color: #687781; font-size: 14px; }}
    .details {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }}
    pre {{ margin: 0; max-height: 340px; overflow: auto; background: #0f1720; color: #e7edf3; border-radius: 6px; padding: 12px; font-size: 12px; line-height: 1.45; }}
    .route-update {{ border-left: 4px solid #d38b00; background: #fff8e8; padding: 10px 12px; margin: 12px 0; border-radius: 4px; }}
    .multi-stop {{ border-left: 4px solid #267a55; background: #eef9f3; padding: 10px 12px; margin: 12px 0; border-radius: 4px; }}
    @media (max-width: 900px) {{ .meta-grid, .maps, .details {{ grid-template-columns: 1fr; }} .case-card header {{ flex-direction: column; }} }}
  </style>
</head>
<body>
  <main>
    <section class="page-head">
      <h1>{_esc(title)}</h1>
      <p>Cases: {case_count}</p>
      <p>GT: {_esc(gt_path)}</p>
      <p>Predictions: {_esc(predictions_path or "not provided")}</p>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def _metric_badges(row: dict[str, Any]) -> str:
    """Build metric badges.

    Args:
        row: Evaluator row.

    Returns:
        HTML badges.
    """

    if not row:
        return '<span class="badge">no eval</span>'
    items = [
        ("DS", row.get("destination_success")),
        ("Hit@5", row.get("hit@5")),
        ("RS", row.get("route_success")),
        ("SPL", row.get("spl")),
        ("Ratio", row.get("route_ratio")),
    ]
    parts = []
    for label, value in items:
        cls = "ok" if value is True or (isinstance(value, (int, float)) and value >= 0.999) else "fail" if value is False else ""
        parts.append(f'<span class="badge {cls}">{_esc(label)}: {_esc(_value_text(value))}</span>')
    return "".join(parts)


def _image_block(title: str, image_path: str | None) -> str:
    """Build one image panel.

    Args:
        title: Panel title.
        image_path: Optional relative image path.

    Returns:
        HTML fragment.
    """

    if not image_path:
        return f'<div class="map-panel"><h3>{_esc(title)}</h3><div class="empty-map">No local staticmap image</div></div>'
    return f'<div class="map-panel"><h3>{_esc(title)}</h3><img src="{_esc(image_path)}" alt="{_esc(title)}"></div>'


def _route_update_block(payload: dict[str, Any]) -> str:
    """Build route-update metadata block.

    Args:
        payload: Route-update metadata.

    Returns:
        HTML fragment.
    """

    if not payload:
        return ""
    return f'<section class="route-update"><strong>Route update</strong><pre>{_esc(json.dumps(payload, ensure_ascii=False, indent=2))}</pre></section>'


def _destination_sequence(case: UrbanNavCase) -> list[UrbanNavDestination]:
    """Return the ordered GT destination sequence when present.

    Args:
        case: UrbanNav case.

    Returns:
        Ordered destination sequence. Empty when the case is not multi-stop.
    """

    raw_sequence = case.raw.get("gt_destination_sequence")
    if not isinstance(raw_sequence, list):
        return []
    return [UrbanNavDestination.from_dict(item) for item in raw_sequence if isinstance(item, dict)]


def _prediction_destination_sequence(prediction: UrbanNavPrediction) -> list[UrbanNavDestination]:
    """Return the ordered predicted destination sequence when present.

    Args:
        prediction: UrbanNav prediction.

    Returns:
        Ordered predicted destinations. Empty for single-destination outputs.
    """

    if prediction.destination_sequence:
        return list(prediction.destination_sequence)
    raw_sequence = prediction.raw.get("pred_destination_sequence")
    if not isinstance(raw_sequence, list):
        return []
    return [UrbanNavDestination.from_dict(item) for item in raw_sequence if isinstance(item, dict)]


def _multi_stop_block(case: UrbanNavCase) -> str:
    """Build a multi-stop audit block for review pages.

    Args:
        case: UrbanNav case.

    Returns:
        HTML fragment listing ordered stops and route legs.
    """

    sequence = _destination_sequence(case)
    if not sequence:
        return ""
    payload = {
        "gt_destination_sequence": [destination.to_dict() for destination in sequence],
        "gt_route_legs": case.raw.get("gt_route_legs") or [],
    }
    return f'<section class="multi-stop"><strong>Multi-stop sequence</strong><pre>{_esc(json.dumps(payload, ensure_ascii=False, indent=2))}</pre></section>'


def _pred_multi_stop_block(prediction: UrbanNavPrediction | None) -> str:
    """Build a predicted multi-stop audit block.

    Args:
        prediction: Optional UrbanNav prediction.

    Returns:
        HTML fragment listing predicted stops and route legs.
    """

    if prediction is None or not prediction.destination_sequence:
        return ""
    payload = {
        "pred_destination_sequence": [destination.to_dict() for destination in prediction.destination_sequence],
        "pred_route_legs": prediction.route_legs,
    }
    return f'<section class="multi-stop"><strong>Pred multi-stop sequence</strong><pre>{_esc(json.dumps(payload, ensure_ascii=False, indent=2))}</pre></section>'


def _safe_name(value: str) -> str:
    """Convert text to a filesystem-safe stem.

    Args:
        value: Raw text.

    Returns:
        Safe filename stem.
    """

    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))[:120]


def _point_text(point: UrbanNavPoint | None) -> str:
    """Format a point.

    Args:
        point: Optional point.

    Returns:
        Human-readable point string.
    """

    if point is None:
        return "n/a"
    return f"{point.lat:.6f}, {point.lng:.6f}"


def _num(value: Any) -> str:
    """Format a numeric value.

    Args:
        value: Raw value.

    Returns:
        Compact number string.
    """

    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "n/a"


def _value_text(value: Any) -> str:
    """Format a metric value.

    Args:
        value: Raw value.

    Returns:
        Display text.
    """

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    if value is None:
        return "null"
    return str(value)


def _esc(value: Any) -> str:
    """HTML-escape a value.

    Args:
        value: Raw value.

    Returns:
        Escaped text.
    """

    return html.escape(str(value if value is not None else ""))


if __name__ == "__main__":
    main()
