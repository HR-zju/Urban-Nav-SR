#!/usr/bin/env python3
"""Build an interactive Leaflet review page for UrbanNav GT routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mcp_server.eval.urbannav.urbannav_io import load_urbannav_cases
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavCase, UrbanNavPoint


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path, help="UrbanNav task_gt.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output review directory")
    parser.add_argument("--title", default="UrbanNav Leaflet Review", help="Page title")
    parser.add_argument("--max-cases", type=int, default=150, help="Maximum cases to render")
    return parser.parse_args()


def main() -> None:
    """Build the review page.

    Returns:
        None.
    """

    args = parse_args()
    build_leaflet_review_html(
        gt_path=args.ground_truth,
        output_dir=args.output_dir,
        title=args.title,
        max_cases=args.max_cases,
    )


def build_leaflet_review_html(
    *,
    gt_path: str | Path,
    output_dir: str | Path,
    title: str,
    max_cases: int,
) -> dict[str, Any]:
    """Build a Leaflet route review page.

    Args:
        gt_path: Ground-truth JSONL.
        output_dir: Output directory.
        title: Page title.
        max_cases: Maximum rendered cases.

    Returns:
        Summary containing output paths and rendered count.
    """

    cases = load_urbannav_cases(gt_path)[: max(0, max_cases)]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cards = [_case_card(index=index, case=case) for index, case in enumerate(cases, start=1)]
    index_path = root / "index.html"
    index_path.write_text(_html_page(title=title, gt_path=str(gt_path), cards=cards, case_count=len(cases)), encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps([{"case_id": case.case_id, "city": case.city, "category": case.category} for case in cases], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"index_html": str(index_path), "manifest": str(manifest_path), "case_count": len(cases)}


def _case_card(*, index: int, case: UrbanNavCase) -> str:
    """Build one case card.

    Args:
        index: One-based index.
        case: UrbanNav case.

    Returns:
        HTML fragment.
    """

    map_id = f"map_{index:04d}"
    route = [_point_pair(point) for point in case.gt_route]
    origin = _point_pair(case.origin) if case.origin else (route[0] if route else None)
    destination = _point_pair(case.gt_route[-1]) if case.gt_route else None
    markers = _review_markers(case)
    old_route = _route_update_old_route(case)
    gt_dest = case.gt_destination.to_dict() if case.gt_destination else {}
    acceptable = [destination_item.to_dict() for destination_item in case.acceptable_destinations]
    metadata = dict(case.metadata or {})
    return f"""
    <article class="case-card">
      <header>
        <div>
          <div class="case-index">#{index} {_esc(case.case_id)}</div>
          <h2>{_esc(case.instruction)}</h2>
        </div>
        <div class="badges">
          <span class="badge">{_esc(case.city or '')}</span>
          <span class="badge">{_esc(case.category or '')}</span>
          <span class="badge">{_esc(case.difficulty or '')}</span>
          <span class="badge">{_num(case.gt_distance_m)} m</span>
        </div>
      </header>
      <section class="meta-grid">
        <div><strong>Scene</strong><span>{_esc(case.scene_id)}</span></div>
        <div><strong>Origin</strong><span>{_point_text(case.origin)}</span></div>
        <div><strong>Destination</strong><span>{_esc(gt_dest.get("name"))}</span></div>
        <div><strong>Acceptable</strong><span>{len(case.acceptable_destinations)}</span></div>
      </section>
      <div class="legend">
        <span><i class="dot start"></i>Start/current origin</span>
        <span><i class="dot stop"></i>Waypoint</span>
        <span><i class="dot via"></i>Route update via</span>
        <span><i class="dot end"></i>Final destination</span>
        <span><i class="line current"></i>GT route</span>
        <span><i class="line old"></i>Old route</span>
      </div>
      <div id="{map_id}" class="map"></div>
      <section class="details">
        <div>
          <h3>GT destination</h3>
          <pre>{_esc(json.dumps(gt_dest, ensure_ascii=False, indent=2))}</pre>
        </div>
        <div>
          <h3>Metadata</h3>
          <pre>{_esc(json.dumps(metadata, ensure_ascii=False, indent=2))}</pre>
        </div>
      </section>
      <section class="details">
        <div>
          <h3>Acceptable destinations</h3>
          <pre>{_esc(json.dumps(acceptable, ensure_ascii=False, indent=2))}</pre>
        </div>
        <div>
          <h3>Ranked destinations</h3>
          <pre>{_esc(json.dumps([item.to_dict() for item in case.ranked_destinations], ensure_ascii=False, indent=2))}</pre>
        </div>
      </section>
      <script>
        renderRouteMap("{map_id}", {json.dumps(route)}, {json.dumps(origin)}, {json.dumps(destination)}, {json.dumps(markers)}, {json.dumps(old_route)});
      </script>
    </article>
    """


def _html_page(*, title: str, gt_path: str, cards: list[str], case_count: int) -> str:
    """Build the full HTML page.

    Args:
        title: Page title.
        gt_path: GT path.
        cards: Case card HTML.
        case_count: Rendered case count.

    Returns:
        HTML document.
    """

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f4f6f7; color: #172026; }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 24px; }}
    .page-head {{ margin-bottom: 20px; }}
    .page-head h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .page-head p {{ margin: 4px 0; color: #5c6972; font-size: 14px; }}
    .case-card {{ background: #fff; border: 1px solid #dce3e8; border-radius: 8px; margin: 18px 0; padding: 18px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
    .case-card header {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; border-bottom: 1px solid #edf1f3; padding-bottom: 12px; }}
    .case-index {{ color: #60717d; font-size: 13px; font-weight: 700; }}
    h2 {{ margin: 6px 0 0; font-size: 20px; line-height: 1.35; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .badges {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; min-width: 280px; }}
    .badge {{ border-radius: 999px; padding: 5px 9px; font-size: 12px; font-weight: 700; background: #edf1f3; color: #34444f; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 14px 0; }}
    .meta-grid div {{ background: #f8fafb; border: 1px solid #e5ebef; border-radius: 6px; padding: 10px; }}
    .meta-grid strong {{ display: block; font-size: 12px; color: #60717d; margin-bottom: 5px; }}
    .meta-grid span {{ font-size: 14px; }}
    .map {{ height: 420px; border: 1px solid #dce3e8; border-radius: 6px; margin: 14px 0; background: #eef3f5; }}
    .map-error {{ display: grid; place-items: center; color: #8a3b12; background: #fff4e8; font-size: 13px; padding: 12px; text-align: center; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 10px 0 0; color: #44545f; font-size: 13px; }}
    .legend span {{ display: inline-flex; gap: 6px; align-items: center; }}
    .dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; border: 2px solid #fff; box-shadow: 0 0 0 1px rgba(0,0,0,.22); }}
    .dot.start {{ background: #1b8a5a; }}
    .dot.stop {{ background: #1769aa; }}
    .dot.via {{ background: #7c3aed; }}
    .dot.end {{ background: #c43c39; }}
    .line {{ width: 22px; height: 0; display: inline-block; border-top: 4px solid #1769aa; }}
    .line.old {{ border-top: 3px dashed #6b7280; }}
    .details {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }}
    pre {{ margin: 0; max-height: 340px; overflow: auto; background: #0f1720; color: #e7edf3; border-radius: 6px; padding: 12px; font-size: 12px; line-height: 1.45; }}
    @media (max-width: 900px) {{ .meta-grid, .details {{ grid-template-columns: 1fr; }} .case-card header {{ flex-direction: column; }} }}
  </style>
  <script>
    function renderRouteMap(id, route, origin, destination, markers, oldRoute) {{
      const container = document.getElementById(id);
      if (!container) return;
      if (!window.L) {{
        container.classList.add('map-error');
        container.textContent = 'Leaflet 未加载。请检查浏览器是否能访问 https://unpkg.com。';
        return;
      }}
      const map = L.map(id);
      const tiles = L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors'
      }}).addTo(map);
      tiles.on('tileerror', function() {{
        container.classList.add('map-error');
        if (!container.textContent) {{
          container.textContent = 'OSM 瓦片加载失败。路线数据已写入 HTML，但当前网络无法访问 tile.openstreetmap.org。';
        }}
      }});
      if (!route || route.length === 0) {{
        map.setView([0, 0], 2);
        return;
      }}
      if (oldRoute && oldRoute.length > 1) {{
        L.polyline(oldRoute, {{color: '#6b7280', weight: 4, opacity: 0.72, dashArray: '7,8'}}).addTo(map);
      }}
      const line = L.polyline(route, {{color: '#1769aa', weight: 5, opacity: 0.9}}).addTo(map);
      const colorByKind = {{
        start: '#1b8a5a',
        current: '#f59e0b',
        stop: '#1769aa',
        via: '#7c3aed',
        previous: '#6b7280',
        final: '#c43c39',
        end: '#c43c39'
      }};
      const labelByKind = {{
        start: 'S',
        current: 'C',
        stop: 'W',
        via: 'V',
        previous: 'P',
        final: 'D',
        end: 'D'
      }};
      function addMarker(point, kind, label) {{
        if (!point) return;
        const color = colorByKind[kind] || '#1769aa';
        L.circleMarker(point, {{
          radius: 8,
          color: '#ffffff',
          weight: 2,
          fillColor: color,
          fillOpacity: 0.96
        }}).addTo(map).bindTooltip(label || labelByKind[kind] || kind, {{
          permanent: true,
          direction: 'top',
          offset: [0, -8],
          className: 'marker-label'
        }}).bindPopup(label || kind);
      }}
      if (markers && markers.length) {{
        markers.forEach(function(item) {{ addMarker(item.point, item.kind, item.label); }});
      }} else {{
        addMarker(origin, 'start', 'S');
        addMarker(destination, 'end', 'D');
      }}
      map.fitBounds(line.getBounds(), {{padding: [24, 24]}});
      setTimeout(() => map.invalidateSize(), 100);
    }}
  </script>
</head>
<body>
  <main>
    <section class="page-head">
      <h1>{_esc(title)}</h1>
      <p>Cases: {case_count}</p>
      <p>GT: {_esc(gt_path)}</p>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def _point_pair(point: UrbanNavPoint | None) -> list[float] | None:
    """Convert point to Leaflet ``[lat, lng]``.

    Args:
        point: Optional point.

    Returns:
        Pair or ``None``.
    """

    if point is None:
        return None
    return [point.lat, point.lng]


def _review_markers(case: UrbanNavCase) -> list[dict[str, Any]]:
    """Build colored review markers for one case.

    Args:
        case: UrbanNav case.

    Returns:
        Marker dictionaries consumed by the Leaflet renderer.
    """

    markers: list[dict[str, Any]] = []
    if case.origin:
        markers.append({"point": _point_pair(case.origin), "kind": "start", "label": "S"})

    sequence = case.raw.get("gt_destination_sequence")
    if isinstance(sequence, list) and sequence:
        for index, item in enumerate(sequence, start=1):
            point = _point_from_payload(item)
            if point is None:
                continue
            kind = "end" if index == len(sequence) else "stop"
            label = "D" if kind == "end" else f"W{index}"
            markers.append({"point": _point_pair(point), "kind": kind, "label": label})
        return markers

    route_update = case.metadata.get("route_update") if isinstance(case.metadata, dict) else None
    if isinstance(route_update, dict):
        original = _point_from_payload(route_update.get("original_origin"))
        current = _point_from_payload(route_update.get("current_origin"))
        previous = _point_from_payload(route_update.get("previous_destination"))
        via = _point_from_payload(route_update.get("via_destination"))
        final = _point_from_payload(route_update.get("final_destination"))
        if original:
            markers.append({"point": _point_pair(original), "kind": "start", "label": "S0"})
        if current:
            markers.append({"point": _point_pair(current), "kind": "current", "label": "C"})
        if previous:
            markers.append({"point": _point_pair(previous), "kind": "previous", "label": "OLD"})
        if via:
            markers.append({"point": _point_pair(via), "kind": "via", "label": "V"})
        if final:
            markers.append({"point": _point_pair(final), "kind": "final", "label": "D"})
        if markers:
            return markers

    if case.gt_destination and case.gt_destination.point:
        markers.append({"point": _point_pair(case.gt_destination.point), "kind": "end", "label": "D"})
    elif case.gt_route:
        markers.append({"point": _point_pair(case.gt_route[-1]), "kind": "end", "label": "D"})
    return markers


def _route_update_old_route(case: UrbanNavCase) -> list[list[float]]:
    """Return an old route polyline for route-update review cases.

    Args:
        case: UrbanNav case.

    Returns:
        Leaflet coordinate pairs.
    """

    route_update = case.metadata.get("route_update") if isinstance(case.metadata, dict) else None
    if not isinstance(route_update, dict):
        return []
    route = route_update.get("old_route")
    if not isinstance(route, list):
        return []
    points: list[list[float]] = []
    for item in route:
        point = _point_from_payload(item)
        if point:
            points.append(_point_pair(point))
    return points


def _point_from_payload(value: Any) -> UrbanNavPoint | None:
    """Parse a point from a raw payload.

    Args:
        value: Raw point or destination payload.

    Returns:
        Parsed point, or ``None``.
    """

    if not isinstance(value, dict):
        return None
    lat = value.get("lat") or value.get("latitude")
    lng = value.get("lng") or value.get("lon") or value.get("longitude")
    try:
        if lat is None or lng is None:
            return None
        return UrbanNavPoint(lat=float(lat), lng=float(lng))
    except Exception:
        return None


def _point_text(point: UrbanNavPoint | None) -> str:
    """Format point text.

    Args:
        point: Optional point.

    Returns:
        Human-readable coordinate.
    """

    if point is None:
        return "-"
    return f"{point.lat:.6f}, {point.lng:.6f}"


def _num(value: Any) -> str:
    """Format numeric value.

    Args:
        value: Raw value.

    Returns:
        Formatted value.
    """

    try:
        return f"{float(value):.1f}"
    except Exception:
        return "-"


def _esc(value: Any) -> str:
    """Escape HTML.

    Args:
        value: Raw value.

    Returns:
        Escaped string.
    """

    import html

    return html.escape(str(value if value is not None else ""))


if __name__ == "__main__":
    main()
