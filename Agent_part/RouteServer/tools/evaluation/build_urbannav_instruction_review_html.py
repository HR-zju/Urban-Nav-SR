#!/usr/bin/env python3
"""Build an offline HTML review page for an UrbanNav instruction set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import random
from typing import Any, Iterable


DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "benchmark" / "urbannav_instruction_set_full_en_v1"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction-set", type=Path, default=DEFAULT_ROOT / "instruction_set.jsonl")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "review")
    parser.add_argument("--max-samples", type=int, default=260)
    parser.add_argument("--samples-per-subcategory", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--title", default="UrbanNav Full Instruction Set Review")
    return parser.parse_args()


def main() -> None:
    """Build the review HTML.

    Returns:
        None.
    """

    args = parse_args()
    result = build_instruction_review_html(
        instruction_set=args.instruction_set,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        samples_per_subcategory=args.samples_per_subcategory,
        seed=args.seed,
        title=args.title,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_instruction_review_html(
    *,
    instruction_set: Path,
    manifest_path: Path,
    output_dir: Path,
    max_samples: int,
    samples_per_subcategory: int,
    seed: int,
    title: str,
) -> dict[str, Any]:
    """Render an instruction-set review page.

    Args:
        instruction_set: JSONL instruction-set path.
        manifest_path: Manifest path.
        output_dir: Output directory.
        max_samples: Maximum instruction cards rendered.
        samples_per_subcategory: Per-subcategory sampling cap before the global
            cap is applied.
        seed: Deterministic sample seed.
        title: HTML title.

    Returns:
        Summary with output paths and sample counts.
    """

    rows = list(_load_jsonl(instruction_set))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    samples = _pick_samples(rows, max_samples=max_samples, samples_per_subcategory=samples_per_subcategory, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_text = _render_page(title=title, rows=rows, samples=samples, manifest=manifest)
    index_path = output_dir / "index.html"
    index_path.write_text(html_text, encoding="utf-8")
    review_manifest = {
        "instruction_set": str(instruction_set),
        "manifest": str(manifest_path),
        "index_html": str(index_path),
        "total_rows": len(rows),
        "rendered_samples": len(samples),
        "category_distribution": dict(Counter(row.get("category") for row in rows)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(review_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return review_manifest


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Load JSONL rows.

    Args:
        path: JSONL path.

    Returns:
        Iterable of parsed dictionaries.
    """

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _pick_samples(
    rows: list[dict[str, Any]],
    *,
    max_samples: int,
    samples_per_subcategory: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Pick stratified review samples.

    Args:
        rows: Instruction rows.
        max_samples: Global sample cap.
        samples_per_subcategory: Per-subcategory cap.
        seed: Deterministic sample seed.

    Returns:
        Sampled rows.
    """

    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("category") or ""), str(row.get("subcategory") or ""))].append(row)
    samples: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        samples.extend(group[: max(1, samples_per_subcategory)])
    rng.shuffle(samples)
    return samples[: max(1, max_samples)]


def _render_page(*, title: str, rows: list[dict[str, Any]], samples: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    """Render the HTML document.

    Args:
        title: Page title.
        rows: All instruction rows.
        samples: Sampled instruction rows.
        manifest: Instruction-set manifest.

    Returns:
        HTML string.
    """

    category_counts = Counter(str(row.get("category") or "") for row in rows)
    city_counts = Counter(str((row.get("planned_city") or {}).get("city") or "") for row in rows)
    subcategory_counts = Counter(f"{row.get('category')}:{row.get('subcategory')}" for row in rows)
    template_count = len({str(row.get("instruction") or "") for row in rows})
    nearby_count = sum(1 for row in rows if "nearby" in str(row.get("instruction") or "").lower())
    target_type_count = len({target for row in rows for target in row.get("target_types") or []})

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(title)}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1d2430;
      --muted: #607083;
      --line: #d9e0e8;
      --blue: #2563eb;
      --green: #0f766e;
      --orange: #b45309;
      --purple: #7c3aed;
    }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ padding: 28px 36px 16px; background: #101827; color: white; }}
    header h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }}
    header p {{ margin: 0; color: #cbd5e1; max-width: 980px; }}
    main {{ padding: 24px 36px 48px; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    .grid {{ display: grid; gap: 14px; }}
    .summary {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
    .metric .value {{ font-size: 24px; font-weight: 700; }}
    .metric .label {{ color: var(--muted); }}
    .two {{ grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr); }}
    @media (max-width: 900px) {{ .two {{ grid-template-columns: 1fr; }} main {{ padding: 18px; }} header {{ padding: 22px 18px 14px; }} }}
    .bar-row {{ display: grid; grid-template-columns: 220px 1fr 72px; align-items: center; gap: 10px; margin: 8px 0; }}
    .bar-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .bar-track {{ background: #edf2f7; border-radius: 4px; height: 12px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: var(--blue); }}
    .bar-row:nth-child(4n+1) .bar-fill {{ background: var(--green); }}
    .bar-row:nth-child(4n+2) .bar-fill {{ background: var(--orange); }}
    .bar-row:nth-child(4n+3) .bar-fill {{ background: var(--purple); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: #edf2f7; color: #334155; }}
    tr:last-child td {{ border-bottom: 0; }}
    .samples {{ grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }}
    .sample h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .instruction {{ font-size: 16px; margin: 10px 0; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }}
    .tag {{ background: #eef2ff; color: #3730a3; border-radius: 999px; padding: 2px 8px; font-size: 12px; }}
    .tag.green {{ background: #ecfdf5; color: #065f46; }}
    .tag.orange {{ background: #fff7ed; color: #9a3412; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; color: #334155; }}
    .muted {{ color: var(--muted); }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 16px; }}
    input, select {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; background: white; min-width: 180px; }}
  </style>
</head>
<body>
  <header>
    <h1>{_e(title)}</h1>
    <p>This page reviews the generated instruction plan before live GT construction. It verifies task quotas, city quotas, template diversity, target-type coverage, and stratified instruction examples.</p>
  </header>
  <main>
    <section class="grid summary">
      {_metric("Rows", len(rows))}
      {_metric("Unique Templates", template_count)}
      {_metric("Cities", len(city_counts))}
      {_metric("Target Types", target_type_count)}
      {_metric("Nearby Wording", nearby_count)}
      {_metric("Rendered Samples", len(samples))}
    </section>

    <section class="grid two">
      <div class="card">
        <h2>Task Family Distribution</h2>
        {_bars(category_counts)}
      </div>
      <div class="card">
        <h2>City Distribution</h2>
        {_bars(city_counts)}
      </div>
    </section>

    <section>
      <h2>Subcategory Distribution</h2>
      {_table_subcategories(subcategory_counts)}
    </section>

    <section>
      <h2>Instruction Samples</h2>
      <div class="toolbar">
        <input id="search" placeholder="Search instruction, city, type">
        <select id="category">
          <option value="">All categories</option>
          {_category_options(category_counts)}
        </select>
      </div>
      <div class="grid samples" id="samples">
        {''.join(_sample_card(row) for row in samples)}
      </div>
    </section>
  </main>
  <script>
    const search = document.getElementById('search');
    const category = document.getElementById('category');
    const cards = Array.from(document.querySelectorAll('.sample'));
    function applyFilter() {{
      const q = search.value.toLowerCase();
      const c = category.value;
      for (const card of cards) {{
        const text = card.dataset.text;
        const okSearch = !q || text.includes(q);
        const okCat = !c || card.dataset.category === c;
        card.style.display = okSearch && okCat ? '' : 'none';
      }}
    }}
    search.addEventListener('input', applyFilter);
    category.addEventListener('change', applyFilter);
  </script>
</body>
</html>"""


def _metric(label: str, value: Any) -> str:
    return f'<div class="card metric"><div class="value">{_e(value)}</div><div class="label">{_e(label)}</div></div>'


def _bars(counts: Counter[str], *, limit: int = 24) -> str:
    if not counts:
        return '<p class="muted">No records.</p>'
    maximum = max(counts.values())
    rows = []
    for label, count in counts.most_common(limit):
        width = 100.0 * count / maximum if maximum else 0.0
        rows.append(
            f'<div class="bar-row"><div class="bar-label" title="{_e(label)}">{_e(label)}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>'
            f'<div class="mono">{count}</div></div>'
        )
    return "\n".join(rows)


def _table_subcategories(counts: Counter[str]) -> str:
    rows = []
    for key, count in counts.most_common():
        category, _, subcategory = key.partition(":")
        rows.append(f"<tr><td>{_e(category)}</td><td>{_e(subcategory)}</td><td class=\"mono\">{count}</td></tr>")
    return "<table><thead><tr><th>Category</th><th>Subcategory</th><th>Count</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _category_options(counts: Counter[str]) -> str:
    return "".join(f'<option value="{_e(category)}">{_e(category)}</option>' for category in sorted(counts))


def _sample_card(row: dict[str, Any]) -> str:
    city = row.get("planned_city") or {}
    target_types = row.get("target_types") or []
    ordered_target_types = row.get("ordered_target_types") or []
    osm_tags = row.get("osm_tags") or []
    searchable = " ".join(
        [
            str(row.get("instruction") or ""),
            str(row.get("category") or ""),
            str(row.get("subcategory") or ""),
            str(city.get("city") or ""),
            str(city.get("country") or ""),
            " ".join(target_types),
            " ".join(_flatten(ordered_target_types)),
        ]
    ).lower()
    ordered = ""
    if ordered_target_types:
        ordered = "<div><strong>Ordered goals:</strong> " + _e(" -> ".join("/".join(group) for group in ordered_target_types)) + "</div>"
    update_action = row.get("route_update_action")
    update = f"<div><strong>Route update action:</strong> {_e(update_action)}</div>" if update_action else ""
    return f"""
<article class="card sample" data-category="{_e(row.get('category') or '')}" data-text="{_e(searchable)}">
  <h3>{_e(row.get('global_index'))}. {_e(city.get('city'))}, {_e(city.get('country'))}</h3>
  <div class="meta">
    <span class="tag">{_e(row.get('category'))}</span>
    <span class="tag green">{_e(row.get('subcategory'))}</span>
    <span class="tag orange">{_e(row.get('acceptance_policy'))}</span>
  </div>
  <div class="instruction">{_e(row.get('instruction'))}</div>
  <div class="mono">{_e(row.get('instruction_id'))}</div>
  <div><strong>Target types:</strong> {_e(', '.join(target_types) or '-')}</div>
  {ordered}
  {update}
  <div><strong>OSM tags:</strong> {_e(', '.join(osm_tags[:8]) or '-')}</div>
</article>"""


def _flatten(groups: Iterable[Iterable[str]]) -> list[str]:
    return [str(item) for group in groups for item in group]


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


if __name__ == "__main__":
    main()
