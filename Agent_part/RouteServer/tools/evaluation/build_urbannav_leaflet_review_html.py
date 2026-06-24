#!/usr/bin/env python3
"""Build an interactive Leaflet review page for UrbanNav GT routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mcp_server.eval.urbannav_io import load_urbannav_cases
from mcp_server.eval.urbannav_schema import UrbanNavCase, UrbanNavPoint


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
        renderRouteMap("{map_id}", {json.dumps(route)}, {json.dumps(origin)}, {json.dumps(destination)});
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
    .details {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }}
    pre {{ margin: 0; max-height: 340px; overflow: auto; background: #0f1720; color: #e7edf3; border-radius: 6px; padding: 12px; font-size: 12px; line-height: 1.45; }}
    @media (max-width: 900px) {{ .meta-grid, .details {{ grid-template-columns: 1fr; }} .case-card header {{ flex-direction: column; }} }}
  </style>
  <script>
    function renderRouteMap(id, route, origin, destination) {{
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
      const line = L.polyline(route, {{color: '#1769aa', weight: 5, opacity: 0.9}}).addTo(map);
      if (origin) L.marker(origin).addTo(map).bindPopup('S');
      if (destination) L.marker(destination).addTo(map).bindPopup('D');
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
