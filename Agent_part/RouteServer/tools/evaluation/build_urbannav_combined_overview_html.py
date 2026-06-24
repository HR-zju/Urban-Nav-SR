#!/usr/bin/env python3
"""Build a combined UrbanNav provider/country review and 10K plan overview."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

from mcp_server.core.osm_local_index import LocalOsmPoiIndex
from mcp_server.eval.urbannav_io import load_urbannav_cases
from mcp_server.eval.urbannav_schema import UrbanNavCase, UrbanNavDestination, UrbanNavPoint


DEFAULT_PLAN = Path(__file__).resolve().parents[2] / "config" / "urbannav_10k_distribution_plan.json"
DEFAULT_MULTI_STOP_TASKS = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "urbannav_yuquan_multistop_tasks.jsonl"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="Output overview directory.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="10K distribution plan JSON.")
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="Review package in 'label=task_gt.jsonl' form. Can be repeated.",
    )
    parser.add_argument("--max-samples", type=int, default=160, help="Maximum route samples rendered.")
    parser.add_argument(
        "--multi-stop-tasks",
        type=Path,
        default=DEFAULT_MULTI_STOP_TASKS,
        help="Optional JSONL file with complex multi-stop instruction examples.",
    )
    parser.add_argument(
        "--osm-local-index-root",
        type=Path,
        default=None,
        help="Optional local OSM POI index root rendered as provider review examples.",
    )
    parser.add_argument("--title", default="UrbanNav Benchmark Overview", help="HTML title.")
    return parser.parse_args()


def main() -> None:
    """Build the combined overview.

    Returns:
        None.
    """

    args = parse_args()
    packages = _default_packages() if not args.package else [_parse_package_arg(item) for item in args.package]
    build_combined_overview_html(
        output_dir=args.output_dir,
        plan_path=args.plan,
        packages=packages,
        max_samples=args.max_samples,
        multi_stop_tasks_path=args.multi_stop_tasks,
        osm_local_index_root=args.osm_local_index_root,
        title=args.title,
    )


def build_combined_overview_html(
    *,
    output_dir: str | Path,
    plan_path: str | Path,
    packages: list[tuple[str, Path]],
    max_samples: int,
    multi_stop_tasks_path: str | Path | None = DEFAULT_MULTI_STOP_TASKS,
    osm_local_index_root: str | Path | None = None,
    title: str,
) -> dict[str, Any]:
    """Build a combined provider/country review page.

    Args:
        output_dir: Output directory.
        plan_path: 10K distribution plan JSON.
        packages: ``(label, task_gt_path)`` package list.
        max_samples: Maximum rendered route samples.
        multi_stop_tasks_path: Optional JSONL file with complex multi-stop
            instruction examples for human review.
        osm_local_index_root: Optional local OSM POI index root.
        title: Page title.

    Returns:
        Summary with output paths and counts.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    records = _load_records(packages)
    samples = _balanced_samples(records, max_samples=max_samples)
    summary = _actual_summary(records)
    multi_stop_tasks = _load_jsonl_dicts(Path(multi_stop_tasks_path)) if multi_stop_tasks_path else []
    osm_local_examples = _load_osm_local_examples(Path(osm_local_index_root)) if osm_local_index_root else []
    cards = [_case_card(index=index, item=item) for index, item in enumerate(samples, start=1)]
    html = _html_page(
        title=title,
        plan=plan,
        summary=summary,
        records=records,
        cards=cards,
        sample_count=len(samples),
        package_count=len(packages),
        multi_stop_tasks=multi_stop_tasks,
        osm_local_examples=osm_local_examples,
    )
    index_path = root / "index.html"
    index_path.write_text(html, encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "plan": str(plan_path),
                "packages": [{"label": label, "path": str(path)} for label, path in packages],
                "total_records": len(records),
                "rendered_samples": len(samples),
                "osm_local_examples": len(osm_local_examples),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"index_html": str(index_path), "manifest": str(manifest_path), "rendered_samples": len(samples)}


def _default_packages() -> list[tuple[str, Path]]:
    """Return default local packages when no package is specified.

    Returns:
        Default package list.
    """

    root = Path(__file__).resolve().parents[2] / "data" / "benchmark"
    return [
        ("AMap China", root / "urbannav_live_v1" / "task_gt.jsonl"),
        ("AMap Yuquan Multi-stop", root / "urbannav_yuquan_multistop_live_v1" / "task_gt.jsonl"),
        ("AMap Yuquan Route-update", root / "urbannav_yuquan_route_update_live_v1" / "task_gt.jsonl"),
        ("OSM Global", root / "urbannav_global_osm_v1" / "task_gt.jsonl"),
        ("Complex GT Examples", Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "urbannav_complex_gt_6.jsonl"),
    ]


def _parse_package_arg(value: str) -> tuple[str, Path]:
    """Parse one ``label=path`` package argument.

    Args:
        value: Raw argument.

    Returns:
        Label and path.

    Raises:
        ValueError: If the argument is malformed.
    """

    if "=" not in value:
        raise ValueError("--package must be in 'label=task_gt.jsonl' form")
    label, path = value.split("=", 1)
    return label.strip(), Path(path.strip())


def _load_records(packages: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    """Load cases from all package paths.

    Args:
        packages: Package labels and paths.

    Returns:
        Case records with package metadata.
    """

    records: list[dict[str, Any]] = []
    for label, path in packages:
        if not path.exists():
            continue
        for case in load_urbannav_cases(path):
            provider = _provider_name(case)
            country = _country_name(case)
            records.append({"package": label, "provider": provider, "country": country, "case": case})
    return records


def _load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file as dictionaries.

    Args:
        path: JSONL path.

    Returns:
        Parsed dictionary records. Missing files and malformed lines are
        ignored because this file is optional review context.
    """

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _balanced_samples(records: list[dict[str, Any]], *, max_samples: int) -> list[dict[str, Any]]:
    """Sample diverse records across provider, country, city, and task type.

    Args:
        records: Loaded records.
        max_samples: Maximum samples.

    Returns:
        Diverse sample records with repeated instruction templates suppressed.
    """

    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        case: UrbanNavCase = record["case"]
        key = (record["provider"], record["country"], str(case.city or ""), str(case.category or ""))
        buckets[key].append(record)

    for bucket in buckets.values():
        bucket.sort(key=_sample_priority)

    ordered_keys = sorted(buckets, key=lambda key: (_bucket_complexity(key, buckets[key]), key))
    samples: list[dict[str, Any]] = []
    seen_instructions: set[str] = set()
    instruction_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    seen_city_category: Counter[tuple[str, str]] = Counter()
    cursor = 0
    while len(samples) < max(0, max_samples) and ordered_keys:
        ordered_keys = [
            item
            for item in ordered_keys
            if buckets[item] and _bucket_has_available_instruction(buckets[item], instruction_counts, max_per_instruction=2)
            and _bucket_has_available_intent(buckets[item], intent_counts, max_per_intent=3)
        ]
        if not ordered_keys:
            break
        key = ordered_keys[cursor % len(ordered_keys)]
        bucket = buckets[key]
        if bucket:
            selected_index = _select_diverse_record_index(
                bucket,
                seen_instructions=seen_instructions,
                instruction_counts=instruction_counts,
                intent_counts=intent_counts,
                seen_city_category=seen_city_category,
            )
            if selected_index is None:
                buckets[key] = []
                ordered_keys = [item for item in ordered_keys if buckets[item]]
                continue
            record = bucket.pop(selected_index)
            case: UrbanNavCase = record["case"]
            signature = _instruction_signature(case.instruction)
            intent = _instruction_intent(case.instruction, case.category)
            seen_instructions.add(signature)
            instruction_counts[signature] += 1
            intent_counts[intent] += 1
            seen_city_category[(str(case.city or ""), str(case.category or ""))] += 1
            samples.append(record)
        ordered_keys = [item for item in ordered_keys if buckets[item]]
        ordered_keys.sort(key=lambda item: (seen_city_category[(item[2], item[3])], _bucket_complexity(item, buckets[item]), item))
        cursor += 1
    return samples


def _bucket_has_available_instruction(
    bucket: list[dict[str, Any]],
    instruction_counts: Counter[str],
    *,
    max_per_instruction: int,
) -> bool:
    """Check whether a bucket still contains an underrepresented instruction.

    Args:
        bucket: Candidate records.
        instruction_counts: Current rendered counts by instruction signature.
        max_per_instruction: Maximum rendered cards per signature.

    Returns:
        ``True`` when at least one candidate can still be rendered.
    """

    for record in bucket:
        case: UrbanNavCase = record["case"]
        if instruction_counts[_instruction_signature(case.instruction)] < max_per_instruction:
            return True
    return False


def _bucket_has_available_intent(
    bucket: list[dict[str, Any]],
    intent_counts: Counter[str],
    *,
    max_per_intent: int,
) -> bool:
    """Check whether a bucket still has an underrepresented intent.

    Args:
        bucket: Candidate records.
        intent_counts: Current rendered counts by semantic intent.
        max_per_intent: Maximum rendered cards per intent.

    Returns:
        ``True`` if at least one candidate is still allowed.
    """

    for record in bucket:
        case: UrbanNavCase = record["case"]
        intent = _instruction_intent(case.instruction, case.category)
        if _is_complex_category(case.category) or intent_counts[intent] < max_per_intent:
            return True
    return False


def _sample_priority(record: dict[str, Any]) -> tuple[int, int, str]:
    """Return priority for a review sample.

    Args:
        record: Loaded case record.

    Returns:
        Sort key. Lower values are rendered earlier.
    """

    case: UrbanNavCase = record["case"]
    category = str(case.category or "")
    complexity = 0
    if category.startswith("multi_stop"):
        complexity -= 80
    if category.startswith("route_update"):
        complexity -= 70
    if case.metadata.get("acceptable_destinations"):
        complexity -= 20
    if len(str(case.instruction or "")) >= 28:
        complexity -= 10
    if case.gt_route and len(case.gt_route) >= 8:
        complexity -= 5
    return (complexity, len(_instruction_signature(case.instruction)), str(case.case_id))


def _bucket_complexity(key: tuple[str, str, str, str], bucket: list[dict[str, Any]]) -> int:
    """Return bucket priority for diverse review rendering.

    Args:
        key: Bucket key.
        bucket: Records in the bucket.

    Returns:
        Numeric priority.
    """

    category = key[3]
    score = 0
    if category.startswith("multi_stop"):
        score -= 80
    if category.startswith("route_update"):
        score -= 70
    score -= min(len(bucket), 20)
    return score


def _select_diverse_record_index(
    bucket: list[dict[str, Any]],
    *,
    seen_instructions: set[str],
    instruction_counts: Counter[str],
    intent_counts: Counter[str],
    seen_city_category: Counter[tuple[str, str]],
) -> int | None:
    """Select a record index while suppressing repeated instructions.

    Args:
        bucket: Candidate records.
        seen_instructions: Normalized instruction signatures already rendered.
        instruction_counts: Render count by normalized instruction signature.
        intent_counts: Render count by semantic intent.
        seen_city_category: Coverage counter by city/category.

    Returns:
        Selected list index, or ``None`` when no under-cap candidate remains.
    """

    best_index: int | None = None
    best_score: tuple[int, int, int, str] | None = None
    candidates = [
        (index, record)
        for index, record in enumerate(bucket)
        if instruction_counts[_instruction_signature(record["case"].instruction)] < 2
        and (_is_complex_category(record["case"].category) or intent_counts[_instruction_intent(record["case"].instruction, record["case"].category)] < 3)
    ]
    for index, record in candidates[:240]:
        case: UrbanNavCase = record["case"]
        signature = _instruction_signature(case.instruction)
        duplicate_penalty = 500 if signature in seen_instructions else 0
        if instruction_counts[signature] >= 2:
            duplicate_penalty += 2000
        intent = _instruction_intent(case.instruction, case.category)
        intent_penalty = intent_counts[intent] * 120
        if not _is_complex_category(case.category) and intent_counts[intent] >= 3:
            intent_penalty += 2000
        city_category = (str(case.city or ""), str(case.category or ""))
        coverage_penalty = seen_city_category[city_category] * 6
        priority = _sample_priority(record)
        score = (duplicate_penalty + intent_penalty + coverage_penalty + priority[0], priority[1], index, str(case.case_id))
        if best_score is None or score < best_score:
            best_score = score
            best_index = index
    return best_index


def _is_complex_category(category: str | None) -> bool:
    """Return whether a task category is inherently complex.

    Args:
        category: Task category.

    Returns:
        ``True`` for multi-stop and route-update tasks.
    """

    text = str(category or "")
    return text.startswith("multi_stop") or text.startswith("route_update")


def _instruction_intent(instruction: str | None, category: str | None = None) -> str:
    """Map an instruction into a coarse review intent.

    Args:
        instruction: Raw instruction.
        category: Task category.

    Returns:
        Coarse intent label used only for preview de-duplication.
    """

    text = str(instruction or "").lower()
    category_text = str(category or "")
    if _is_complex_category(category_text):
        return category_text
    patterns = [
        ("medical", ["hospital", "healthcare", "medical", "clinic", "pharmacy", "药", "医疗", "医院", "诊所", "身体"]),
        ("cash_bank", ["atm", "bank", "cash", "银行", "取钱", "现金", "自助机"]),
        ("food", ["restaurant", "food", "hungry", "eat", "kfc", "mcdonald", "饭", "餐", "吃", "小吃"]),
        ("drink_cafe", ["coffee", "cafe", "drink", "thirsty", "starbucks", "咖啡", "饮料", "渴"]),
        ("work_study", ["work", "wi-fi", "wifi", "read", "book", "library", "办公", "看书", "安静"]),
        ("daily_goods", ["convenience", "grocery", "supermarket", "fruit", "vegetable", "日用品", "便利", "超市", "水果", "蔬菜"]),
        ("restroom", ["restroom", "toilet", "bathroom", "厕所", "洗手间"]),
        ("hotel", ["hotel", "hostel", "酒店", "宾馆"]),
        ("shopping", ["shopping", "mall", "store", "购物", "商场"]),
        ("child_play", ["child", "play", "park", "孩子", "儿童", "公园"]),
        ("cinema", ["cinema", "movie", "theater", "影院", "电影院"]),
        ("charging", ["charge", "charging", "充电"]),
    ]
    for label, keywords in patterns:
        if any(keyword in text for keyword in keywords):
            return label
    return _instruction_signature(instruction)[:32] or "unknown"


def _instruction_signature(instruction: str | None) -> str:
    """Normalize an instruction for review de-duplication.

    Args:
        instruction: Raw navigation instruction.

    Returns:
        Compact signature.
    """

    text = str(instruction or "").strip().lower()
    for token in ("附近", "最近", "帮我", "请", "一个", "一家", "the", "a ", "an "):
        text = text.replace(token, "")
    return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")[:80]


def _actual_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize loaded packages.

    Args:
        records: Loaded records.

    Returns:
        Summary dictionary.
    """

    provider_counts: Counter[str] = Counter()
    country_counts: Counter[str] = Counter()
    city_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    for record in records:
        case: UrbanNavCase = record["case"]
        provider_counts[record["provider"]] += 1
        country_counts[record["country"]] += 1
        city_counts[str(case.city or "unknown")] += 1
        task_counts[str(case.category or "unknown")] += 1
    return {
        "total_cases": len(records),
        "provider_counts": dict(provider_counts),
        "country_counts": dict(country_counts),
        "city_counts": dict(city_counts),
        "task_counts": dict(task_counts),
    }


def _case_card(*, index: int, item: dict[str, Any]) -> str:
    """Build one route sample card.

    Args:
        index: One-based sample index.
        item: Sample record.

    Returns:
        HTML fragment.
    """

    case: UrbanNavCase = item["case"]
    map_id = f"map_{index:04d}"
    route = [_point_pair(point) for point in case.gt_route]
    origin = _point_pair(case.origin) if case.origin else (route[0] if route else None)
    destination = _point_pair(case.gt_route[-1]) if case.gt_route else None
    gt_dest = case.gt_destination.to_dict() if case.gt_destination else {}
    return f"""
    <article class="case-card">
      <header>
        <div>
          <div class="case-index">#{index} {_esc(case.case_id)}</div>
          <h2>{_esc(case.instruction)}</h2>
        </div>
        <div class="badges">
          <span class="badge provider">{_esc(item["provider"])}</span>
          <span class="badge">{_esc(item["country"])}</span>
          <span class="badge">{_esc(case.city or "")}</span>
          <span class="badge">{_esc(case.category or "")}</span>
          <span class="badge">{_num(case.gt_distance_m)} m</span>
        </div>
      </header>
      <section class="meta-grid">
        <div><strong>Package</strong><span>{_esc(item["package"])}</span></div>
        <div><strong>Scene</strong><span>{_esc(case.scene_id)}</span></div>
        <div><strong>Origin</strong><span>{_point_text(case.origin)}</span></div>
        <div><strong>Destination</strong><span>{_esc(gt_dest.get("name"))}</span></div>
      </section>
      <div id="{map_id}" class="map"></div>
      <section class="details">
        <div>
          <h3>GT destination</h3>
          <pre>{_esc(json.dumps(gt_dest, ensure_ascii=False, indent=2))}</pre>
        </div>
        <div>
          <h3>Metadata</h3>
          <pre>{_esc(json.dumps(case.metadata, ensure_ascii=False, indent=2))}</pre>
        </div>
      </section>
      <script>
        renderRouteMap("{map_id}", {json.dumps(route)}, {json.dumps(origin)}, {json.dumps(destination)});
      </script>
    </article>
    """


def _html_page(
    *,
    title: str,
    plan: dict[str, Any],
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    cards: list[str],
    sample_count: int,
    package_count: int,
    multi_stop_tasks: list[dict[str, Any]],
    osm_local_examples: list[dict[str, Any]],
) -> str:
    """Build the full HTML page.

    Args:
        title: Page title.
        plan: 10K distribution plan.
        summary: Actual loaded data summary.
        records: Loaded case records.
        cards: Sample card fragments.
        sample_count: Rendered sample count.
        package_count: Number of packages.
        multi_stop_tasks: Complex multi-stop task examples.
        osm_local_examples: Local OSM POI examples.

    Returns:
        HTML document.
    """

    city_plan = [(f"{item['city']}", int(item["count"])) for item in plan.get("city_distribution", [])]
    task_plan = [(str(item.get("label") or item.get("category")), int(item["count"])) for item in plan.get("task_distribution", [])]
    provider_actual = sorted((key, int(value)) for key, value in summary.get("provider_counts", {}).items())
    country_actual = sorted((key, int(value)) for key, value in summary.get("country_counts", {}).items())
    city_table = _distribution_table(plan.get("city_distribution", []), label_key="city", extra_key="country")
    task_table = _distribution_table(plan.get("task_distribution", []), label_key="label", extra_key="category")
    multi_stop_section = _multi_stop_examples_section(multi_stop_tasks)
    route_update_section = _route_update_examples_section(records)
    osm_local_section = _osm_local_examples_section(osm_local_examples)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['\\\\(', '\\\\)']],
        displayMath: [['\\\\[', '\\\\]']]
      }},
      svg: {{fontCache: 'global'}}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f4f6f7; color: #172026; }}
    main {{ max-width: 1560px; margin: 0 auto; padding: 24px; }}
    .page-head {{ margin-bottom: 18px; }}
    .page-head h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .page-head p {{ margin: 4px 0; color: #5c6972; font-size: 14px; }}
    .panel {{ background: #fff; border: 1px solid #dce3e8; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .mini-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .chart-row {{ display: grid; grid-template-columns: 260px minmax(0, 1fr); gap: 16px; align-items: center; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .metric-card {{ border: 1px solid #e5ebef; border-radius: 6px; padding: 14px; background: #fbfcfd; }}
    .metric-card h4 {{ margin: 0 0 8px; font-size: 15px; color: #172026; }}
    .metric-card p {{ margin: 0 0 10px; color: #53636e; font-size: 13px; line-height: 1.55; }}
    .formula {{ margin: 0; padding: 10px; overflow-x: auto; background: #fff; border: 1px solid #e5ebef; border-radius: 6px; color: #172026; font-size: 13px; }}
    .math-note {{ margin: 8px 0 0; color: #8a3b12; font-size: 12px; }}
    .section-note {{ margin: 0 0 14px; color: #60717d; font-size: 13px; line-height: 1.55; }}
    .task-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .task-card {{ border: 1px solid #e5ebef; border-radius: 6px; padding: 14px; background: #fbfcfd; }}
    .task-card h4 {{ margin: 0 0 8px; font-size: 15px; line-height: 1.4; }}
    .task-card .sub {{ color: #60717d; font-size: 12px; margin-bottom: 10px; }}
    .goal-list {{ display: flex; flex-direction: column; gap: 8px; }}
    .goal-item {{ border-left: 3px solid #2a9d8f; background: #fff; padding: 8px 9px; font-size: 13px; }}
    .goal-item strong {{ display: block; margin-bottom: 3px; }}
    .flow-vis {{ display: flex; align-items: stretch; gap: 8px; margin: 12px 0; overflow-x: auto; padding-bottom: 4px; }}
    .flow-node {{ min-width: 108px; border: 1px solid #dce3e8; border-radius: 6px; background: #fff; padding: 9px; }}
    .flow-node strong {{ display: block; font-size: 12px; color: #60717d; margin-bottom: 4px; }}
    .flow-node span {{ font-size: 13px; font-weight: 700; }}
    .flow-arrow {{ display: grid; place-items: center; color: #1769aa; font-weight: 800; min-width: 18px; }}
    .mini-map {{ height: 240px; border: 1px solid #dce3e8; border-radius: 6px; margin: 10px 0; background: #eef3f5; }}
    .update-map {{ height: 340px; border: 1px solid #dce3e8; border-radius: 6px; margin: 12px 0; background: #eef3f5; }}
    .waypoint-marker {{ display: grid; place-items: center; width: 26px; height: 26px; border-radius: 50%; background: #e76f51; color: #fff; border: 2px solid #fff; box-shadow: 0 1px 4px rgba(0,0,0,.35); font-size: 12px; font-weight: 800; }}
    .waypoint-marker.start {{ background: #2a9d8f; }}
    .waypoint-marker.finish {{ background: #1769aa; }}
    .legend {{ columns: 2; font-size: 13px; line-height: 1.7; }}
    .legend span {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5ebef; padding: 7px 8px; text-align: left; }}
    th {{ color: #60717d; font-size: 12px; }}
    .case-card {{ background: #fff; border: 1px solid #dce3e8; border-radius: 8px; margin: 18px 0; padding: 18px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
    .case-card header {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; border-bottom: 1px solid #edf1f3; padding-bottom: 12px; }}
    .case-index {{ color: #60717d; font-size: 13px; font-weight: 700; }}
    h2 {{ margin: 6px 0 0; font-size: 20px; line-height: 1.35; }}
    h3 {{ margin: 0 0 10px; font-size: 16px; }}
    .badges {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; min-width: 320px; }}
    .badge {{ border-radius: 999px; padding: 5px 9px; font-size: 12px; font-weight: 700; background: #edf1f3; color: #34444f; }}
    .badge.provider {{ background: #e2f0f0; color: #0f5f5c; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 14px 0; }}
    .meta-grid div {{ background: #f8fafb; border: 1px solid #e5ebef; border-radius: 6px; padding: 10px; }}
    .meta-grid strong {{ display: block; font-size: 12px; color: #60717d; margin-bottom: 5px; }}
    .meta-grid span {{ font-size: 14px; }}
    .map {{ height: 400px; border: 1px solid #dce3e8; border-radius: 6px; margin: 14px 0; background: #eef3f5; }}
    .map-error {{ display: grid; place-items: center; color: #8a3b12; background: #fff4e8; font-size: 13px; padding: 12px; text-align: center; }}
    .details {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }}
    pre {{ margin: 0; max-height: 280px; overflow: auto; background: #0f1720; color: #e7edf3; border-radius: 6px; padding: 12px; font-size: 12px; line-height: 1.45; }}
    @media (max-width: 960px) {{ .chart-grid, .mini-grid, .metric-grid, .task-grid, .details, .meta-grid {{ grid-template-columns: 1fr; }} .chart-row {{ grid-template-columns: 1fr; }} .case-card header {{ flex-direction: column; }} }}
  </style>
  <script>
    function waypointIcon(label, className) {{
      return L.divIcon({{
        html: `<div class="waypoint-marker ${{className || ''}}">${{label}}</div>`,
        className: '',
        iconSize: [26, 26],
        iconAnchor: [13, 13]
      }});
    }}
    function renderRouteMap(id, route, origin, destination, markers) {{
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
        if (!container.textContent) container.textContent = 'OSM 瓦片加载失败；路线坐标仍已写入 HTML。';
      }});
      if (!route || route.length === 0) {{
        map.setView([0, 0], 2);
        return;
      }}
      const line = L.polyline(route, {{color: '#1769aa', weight: 5, opacity: 0.9}}).addTo(map);
      if (markers && markers.length) {{
        markers.forEach(function(marker) {{
          if (!marker || !marker.point) return;
          L.marker(marker.point, {{icon: waypointIcon(marker.label || '', marker.kind || '')}})
            .addTo(map)
            .bindPopup(marker.popup || marker.label || '');
        }});
      }} else {{
        if (origin) L.marker(origin, {{icon: waypointIcon('S', 'start')}}).addTo(map).bindPopup('S');
        if (destination) L.marker(destination, {{icon: waypointIcon('D', 'finish')}}).addTo(map).bindPopup('D');
      }}
      map.fitBounds(line.getBounds(), {{padding: [24, 24]}});
      setTimeout(() => map.invalidateSize(), 100);
    }}
    function renderRouteUpdateMap(id, oldRoute, updatedRoute, markers) {{
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
        if (!container.textContent) container.textContent = 'OSM 瓦片加载失败；路线坐标仍已写入 HTML。';
      }});
      const layers = [];
      if (oldRoute && oldRoute.length) {{
        layers.push(L.polyline(oldRoute, {{color: '#8a8f98', weight: 4, opacity: 0.75, dashArray: '8 8'}}).addTo(map));
      }}
      if (updatedRoute && updatedRoute.length) {{
        layers.push(L.polyline(updatedRoute, {{color: '#e76f51', weight: 5, opacity: 0.9}}).addTo(map));
      }}
      if (markers && markers.length) {{
        markers.forEach(function(marker) {{
          if (!marker || !marker.point) return;
          L.marker(marker.point, {{icon: waypointIcon(marker.label || '', marker.kind || '')}})
            .addTo(map)
            .bindPopup(marker.popup || marker.label || '');
        }});
      }}
      if (layers.length) {{
        const group = L.featureGroup(layers);
        map.fitBounds(group.getBounds(), {{padding: [24, 24]}});
      }} else {{
        map.setView([0, 0], 2);
      }}
      setTimeout(() => map.invalidateSize(), 100);
    }}
    function renderPointMap(id, markers) {{
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
        if (!container.textContent) container.textContent = 'OSM 瓦片加载失败；候选点坐标仍已写入 HTML。';
      }});
      const layers = [];
      if (markers && markers.length) {{
        markers.forEach(function(marker) {{
          if (!marker || !marker.point) return;
          layers.push(L.marker(marker.point, {{icon: waypointIcon(marker.label || '', marker.kind || '')}})
            .addTo(map)
            .bindPopup(marker.popup || marker.label || ''));
        }});
      }}
      if (layers.length) {{
        map.fitBounds(L.featureGroup(layers).getBounds(), {{padding: [24, 24]}});
      }} else {{
        map.setView([0, 0], 2);
      }}
      setTimeout(() => map.invalidateSize(), 100);
    }}
  </script>
</head>
<body>
  <main>
    <section class="page-head">
      <h1>{_esc(title)}</h1>
      <p>Loaded packages: {package_count}; loaded cases: {summary.get("total_cases", 0)}; rendered route samples: {sample_count}</p>
    </section>
    <section class="panel">
      <h3>10K Planned Distribution</h3>
      <div class="chart-grid">
        <div class="chart-row">{_pie_svg(city_plan, title="Cities")}<div>{_legend(city_plan)}</div></div>
        <div class="chart-row">{_pie_svg(task_plan, title="Tasks")}<div>{_legend(task_plan)}</div></div>
      </div>
      <div class="mini-grid">
        <div>{city_table}</div>
        <div>{task_table}</div>
      </div>
    </section>
    <section class="panel">
      <h3>Loaded Sample Coverage</h3>
      <div class="chart-grid">
        <div class="chart-row">{_pie_svg(provider_actual, title="Providers")}<div>{_legend(provider_actual)}</div></div>
        <div class="chart-row">{_pie_svg(country_actual, title="Countries")}<div>{_legend(country_actual)}</div></div>
      </div>
    </section>
    {_metrics_section()}
    {multi_stop_section}
    {route_update_section}
    {osm_local_section}
    {''.join(cards)}
  </main>
</body>
</html>
"""


def _metrics_section() -> str:
    """Build the core metric definition section.

    Returns:
        HTML section with readable LaTeX formulas.
    """

    cards = [
        _metric_card(
            "目的地匹配",
            "优先使用 provider/place 稳定 ID；缺失 ID 时只允许名称、类型和坐标同时兼容的 fallback 匹配。",
            r"""\operatorname{match}(\hat d,d)=\mathbb{1}[\operatorname{id}(\hat d)=\operatorname{id}(d)]

\operatorname{match}_{fb}(\hat d,d)=\mathbb{1}[\operatorname{name}(\hat d)\sim\operatorname{name}(d)\land \operatorname{type}(\hat d)\sim\operatorname{type}(d)\land \operatorname{dist}(\hat d,d)\le \tau_d]""",
        ),
        _metric_card(
            "目的地成功与召回",
            "Top-1 判断是否命中任一可接受目的地；Hit@5 判断候选前五是否覆盖任一可接受目的地。",
            r"""DS=\mathbb{1}[\exists d_j\in D_{acc},\operatorname{match}(\hat d,d_j)=1]

\mathrm{Hit@5}=\mathbb{1}[\exists \hat d_i\in \hat C_{1:5},\exists d_j\in D_{acc},\operatorname{match}(\hat d_i,d_j)=1]""",
        ),
        _metric_card(
            "路线成功",
            "只有目的地已匹配、预测路线非空，并且路线终点到达 matched destination 的情况下才算路线成功。",
            r"""RS=DS \land |\hat P|>0 \land \operatorname{dist}(\operatorname{end}(\hat P),d_{matched})\le \tau_e""",
        ),
        _metric_card(
            "路线效率",
            "对 matched acceptable destination 使用其冻结参考路线长度作为分母，避免多个合理目标被错误地按单一 GT 惩罚。",
            r"""\mathrm{route\_ratio}=\frac{L(\hat P)}{L(P^*_{d_{matched}})}

SPL=RS\cdot\frac{L(P^*_{d_{matched}})}{\max(L(P^*_{d_{matched}}),L(\hat P))}""",
        ),
        _metric_card(
            "多途经点",
            "按指令给定顺序逐段匹配 stop，既报告 stop 覆盖，也报告完整有序序列和分段路线成功。",
            r"""\mathrm{stop\_recall}=\frac{1}{K}\sum_{i=1}^{K}\mathbb{1}[\exists d\in G_i,\operatorname{match}(\hat d_i,d)=1]

\mathrm{sequence\_success}=\mathbb{1}[\forall i,\exists d\in G_i,\operatorname{match}(\hat d_i,d)=1]

\mathrm{multi\_stop\_SPL}=\mathrm{sequence\_success}\cdot\mathrm{leg\_route\_success}\cdot\frac{L(P^*)}{\max(L(P^*),L(\hat P))}""",
        ),
        _metric_card(
            "Route Update",
            "评估更新后的导航状态：是否从当前位置重规划、是否避开被替换的旧目标、是否到达更新后的目标。",
            r"""\mathrm{route\_update\_success}=\mathrm{updated\_route\_success}\land \mathrm{current\_origin\_used}\land \mathrm{previous\_destination\_avoided}

\mathrm{current\_origin\_used}=\mathbb{1}[\operatorname{dist}(\hat P_1,o_t)\le \tau_e]""",
        ),
    ]
    return f"""
    <section class="panel">
      <h3>Core Evaluation Metrics</h3>
      <div class="metric-grid">{''.join(cards)}</div>
    </section>
    """


def _multi_stop_examples_section(tasks: list[dict[str, Any]]) -> str:
    """Build a human-review section for complex multi-stop instructions.

    Args:
        tasks: Multi-stop task seed records.

    Returns:
        HTML section, or an empty string when no tasks are available.
    """

    if not tasks:
        return ""
    has_real_routes = any(_task_route_points(task) for task in tasks)
    note = (
        "以下多途经点样例已由真实地图 API 生成 GT，每张卡片包含 ordered stops 和完整步行路线可视化。"
        if has_real_routes
        else "这些是多途经点任务 seed 的流程可视化；seed 本身没有真实 POI 坐标。若输入记录包含 gt_route 或 gt_route_legs，本节会自动渲染真实路线地图。"
    )
    cards = []
    for index, task in enumerate(tasks[:9], start=1):
        goals = _task_goals(task)
        map_id = f"multistop_map_{index:04d}"
        route = _task_route_points(task)
        origin = route[0] if route else None
        destination = route[-1] if route else None
        markers = _task_waypoint_markers(task, origin=origin, destination=destination)
        map_html = ""
        map_script = ""
        if route:
            map_html = f'<div id="{map_id}" class="mini-map"></div>'
            map_script = (
                f'<script>renderRouteMap("{map_id}", {json.dumps(route)}, '
                f'{json.dumps(origin)}, {json.dumps(destination)}, {json.dumps(markers, ensure_ascii=False)});</script>'
            )
        goal_items = []
        for step_index, goal in enumerate(goals, start=1):
            if not isinstance(goal, dict):
                continue
            goal_items.append(
                f"""
                <div class="goal-item">
                  <strong>Stop {step_index}: {_esc(goal.get("label") or "")}</strong>
                  <span>{_esc(goal.get("keywords") or "")}</span>
                </div>
                """
            )
        flow_nodes = _multi_stop_flow_nodes(goals)
        cards.append(
            f"""
            <article class="task-card">
              <div class="sub">#{index} {_esc(task.get("subcategory") or "")} / {_esc(task.get("difficulty") or "")}</div>
              <h4>{_esc(task.get("instruction") or "")}</h4>
              {flow_nodes}
              {map_html}
              <div class="goal-list">{''.join(goal_items)}</div>
              {map_script}
            </article>
            """
        )
    return f"""
    <section class="panel">
      <h3>Complex Multi-Stop Instruction Examples</h3>
      <p class="section-note">{_esc(note)}</p>
      <div class="task-grid">{''.join(cards)}</div>
    </section>
    """


def _route_update_examples_section(records: list[dict[str, Any]]) -> str:
    """Build route-update old-vs-updated route visualizations.

    Args:
        records: Loaded case records.

    Returns:
        HTML section, or an empty string when no route-update cases exist.
    """

    cards = []
    route_update_items = []
    for record in records:
        case: UrbanNavCase = record["case"]
        if not str(case.category or "").startswith("route_update"):
            continue
        update = case.metadata.get("route_update") if isinstance(case.metadata.get("route_update"), dict) else {}
        if update:
            route_update_items.append((record, case, update))
    for index, (record, case, update) in enumerate(route_update_items[:8], start=1):
        map_id = f"route_update_map_{index:04d}"
        old_route = _parse_route_like(update.get("old_route") or [])
        updated_route = _parse_route_like(update.get("updated_route") or case.raw.get("gt_route") or case.gt_route)
        markers = _route_update_markers(update)
        updated_destination = update.get("via_destination") or case.raw.get("gt_destination") or (case.gt_destination.to_dict() if case.gt_destination else {})
        cards.append(
            f"""
            <article class="task-card">
              <div class="sub">#{index} {_esc(record["package"])} / {_esc(update.get("update_type") or case.category)}</div>
              <h4>{_esc(case.instruction)}</h4>
              <div id="{map_id}" class="update-map"></div>
              <div class="goal-list">
                <div class="goal-item"><strong>Previous</strong><span>{_esc(_destination_name(update.get("previous_destination")))}</span></div>
                <div class="goal-item"><strong>Updated/Via</strong><span>{_esc(_destination_name(updated_destination))}</span></div>
                <div class="goal-item"><strong>Final</strong><span>{_esc(_destination_name(update.get("final_destination")))}</span></div>
              </div>
              <script>renderRouteUpdateMap("{map_id}", {json.dumps(old_route)}, {json.dumps(updated_route)}, {json.dumps(markers, ensure_ascii=False)});</script>
            </article>
            """
        )
    if not cards:
        return ""
    return f"""
    <section class="panel">
      <h3>Route Update Examples</h3>
      <p class="section-note">灰色虚线为更新前旧路线，红色实线为更新后路线；地图标注原起点、当前位置、旧目标、新增/更新目标和最终目标。</p>
      <div class="task-grid">{''.join(cards)}</div>
    </section>
    """


def _route_update_markers(update: dict[str, Any]) -> list[dict[str, Any]]:
    """Build route-update map markers.

    Args:
        update: Route-update metadata.

    Returns:
        Marker dictionaries.
    """

    marker_specs = [
        ("O", "start", update.get("original_origin"), "Original origin"),
        ("C", "stop", update.get("current_origin"), "Current origin"),
        ("P", "stop", update.get("previous_destination"), f"Previous: {_destination_name(update.get('previous_destination'))}"),
        ("U", "stop", update.get("via_destination"), f"Updated/Via: {_destination_name(update.get('via_destination'))}"),
        ("F", "finish", update.get("final_destination"), f"Final: {_destination_name(update.get('final_destination'))}"),
    ]
    markers: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float]] = set()
    for label, kind, payload, popup in marker_specs:
        point = _point_pair_from_dict(payload)
        if not point:
            continue
        key = (label, round(point[0], 7), round(point[1], 7))
        if key in seen:
            continue
        seen.add(key)
        markers.append({"label": label, "kind": kind, "point": point, "popup": popup})
    return markers


def _load_osm_local_examples(index_root: Path, *, per_city: int = 6) -> list[dict[str, Any]]:
    """Load diverse local OSM POI examples from JSONL indexes.

    Args:
        index_root: Local OSM POI index root.
        per_city: Maximum examples per city/region.

    Returns:
        Example dictionaries for review rendering.
    """

    if not index_root.exists():
        return []
    preferred_categories = [
        "amenity:cafe",
        "amenity:restaurant",
        "amenity:fast_food",
        "amenity:bank",
        "amenity:atm",
        "amenity:pharmacy",
        "shop:convenience",
        "shop:supermarket",
        "leisure:park",
        "tourism:hotel",
        "amenity:library",
    ]
    examples: list[dict[str, Any]] = []
    for path in sorted(index_root.glob("*.jsonl")):
        try:
            index = LocalOsmPoiIndex.from_path(path, source_id=path.stem)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        by_category: dict[str, list[Any]] = defaultdict(list)
        for record in index.records:
            category = str(record.category or "")
            if category in preferred_categories:
                by_category[category].append(record)
        selected = []
        for category in preferred_categories:
            candidates = sorted(by_category.get(category, []), key=lambda item: (_poi_name_quality(item.name), item.name, item.osm_id))
            for record in candidates:
                if _poi_name_quality(record.name) < 0:
                    continue
                selected.append(record)
                break
            if len(selected) >= per_city:
                break
        if not selected:
            selected = sorted(index.records, key=lambda item: (_poi_name_quality(item.name), item.name, item.osm_id))[:per_city]
        for record in selected[:per_city]:
            examples.append(
                {
                    "region_id": record.region_id or path.stem,
                    "city": record.city or path.stem,
                    "country": record.country or "",
                    "name": record.name,
                    "category": record.category,
                    "tags": record.tags,
                    "point": [record.lat, record.lng],
                    "provider_poi_id": f"osm:{record.osm_id}",
                }
            )
    return examples


def _poi_name_quality(name: str | None) -> int:
    """Score display quality of a POI name.

    Args:
        name: POI name.

    Returns:
        Higher means better.
    """

    text = str(name or "").strip()
    if not text or text.lower().startswith("unnamed"):
        return -1
    score = min(len(text), 40)
    if any(ch.isdigit() for ch in text):
        score -= 2
    return score


def _osm_local_examples_section(examples: list[dict[str, Any]]) -> str:
    """Build local OSM POI review examples.

    Args:
        examples: Local OSM examples.

    Returns:
        HTML section.
    """

    if not examples:
        return ""
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in examples:
        by_region[str(item.get("region_id") or item.get("city") or "unknown")].append(item)
    cards = []
    for index, (region_id, rows) in enumerate(sorted(by_region.items()), start=1):
        map_id = f"osm_local_map_{index:04d}"
        markers = [
            {
                "label": str(i),
                "kind": "stop",
                "point": row["point"],
                "popup": f"{row.get('name')} / {row.get('category')}",
            }
            for i, row in enumerate(rows[:8], start=1)
        ]
        table_rows = []
        for i, row in enumerate(rows[:8], start=1):
            tags = row.get("tags") if isinstance(row.get("tags"), dict) else {}
            tag_text = ", ".join(f"{key}={value}" for key, value in list(tags.items())[:5])
            table_rows.append(
                f"<tr><td>{i}</td><td>{_esc(row.get('name'))}</td><td>{_esc(row.get('category'))}</td><td>{_esc(tag_text)}</td></tr>"
            )
        city = rows[0].get("city") if rows else region_id
        country = rows[0].get("country") if rows else ""
        cards.append(
            f"""
            <article class="task-card">
              <div class="sub">#{index} {_esc(country)} / {_esc(region_id)}</div>
              <h4>{_esc(city)} local OSM POI candidates</h4>
              <div id="{map_id}" class="mini-map"></div>
              <table><thead><tr><th>#</th><th>Name</th><th>Category</th><th>Tags</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
              <script>renderPointMap("{map_id}", {json.dumps(markers, ensure_ascii=False)});</script>
            </article>
            """
        )
    return f"""
    <section class="panel">
      <h3>Local OSM POI Index Examples</h3>
      <p class="section-note">这些样例来自本地 OSM PBF 抽取后的 POI index，用于检查跨城市、跨类别的候选目的地覆盖；这里展示的是候选 POI，不代表已经完成路线 GT。</p>
      <div class="task-grid">{''.join(cards)}</div>
    </section>
    """


def _destination_name(value: Any) -> str:
    """Return a display name from a destination-like payload.

    Args:
        value: Destination dictionary or object.

    Returns:
        Destination display string.
    """

    if isinstance(value, UrbanNavDestination):
        return str(value.name or value.provider_poi_id or "")
    if isinstance(value, dict):
        return str(value.get("name") or value.get("provider_poi_id") or "")
    return ""


def _task_goals(task: dict[str, Any]) -> list[Any]:
    """Return ordered goals from a seed task or built multi-stop GT case.

    Args:
        task: Seed or GT-like multi-stop task dictionary.

    Returns:
        Ordered goal dictionaries.
    """

    goals = task.get("goals")
    if isinstance(goals, list):
        return goals
    metadata = task.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("goals"), list):
        return list(metadata["goals"])
    legs = task.get("gt_route_legs")
    if isinstance(legs, list):
        inferred = []
        for leg in legs:
            if isinstance(leg, dict):
                inferred.append({"label": leg.get("goal_label"), "keywords": leg.get("keywords")})
        return inferred
    return []


def _multi_stop_flow_nodes(goals: list[Any]) -> str:
    """Build a visual origin-to-stop flow for a multi-stop seed task.

    Args:
        goals: Ordered goal dictionaries.

    Returns:
        HTML flow diagram.
    """

    pieces = [
        """
        <div class="flow-node">
          <strong>Origin</strong>
          <span>起点</span>
        </div>
        """
    ]
    for index, goal in enumerate(goals, start=1):
        if not isinstance(goal, dict):
            continue
        pieces.append('<div class="flow-arrow">→</div>')
        pieces.append(
            f"""
            <div class="flow-node">
              <strong>Stop {index}</strong>
              <span>{_esc(goal.get("label") or "")}</span>
            </div>
            """
        )
    return f'<div class="flow-vis">{"".join(pieces)}</div>'


def _task_route_points(task: dict[str, Any]) -> list[list[float]]:
    """Extract a review route from a multi-stop task if coordinates exist.

    Args:
        task: Task or GT-like multi-stop dictionary.

    Returns:
        Leaflet coordinate pairs. Empty for seed-only tasks.
    """

    route = _parse_route_like(task.get("gt_route") or task.get("route") or [])
    if route:
        return route
    points: list[list[float]] = []
    origin = _point_pair_from_dict(task.get("origin"))
    if origin:
        points.append(origin)
    legs = task.get("gt_route_legs")
    if isinstance(legs, list):
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            leg_route = _parse_route_like(leg.get("route") or leg.get("gt_route") or leg.get("reference_route") or [])
            if leg_route:
                if points and leg_route and points[-1] == leg_route[0]:
                    points.extend(leg_route[1:])
                else:
                    points.extend(leg_route)
                continue
            destination = _point_pair_from_dict(leg.get("gt_destination"))
            if destination:
                points.append(destination)
    return points if len(points) >= 2 else []


def _task_waypoint_markers(
    task: dict[str, Any],
    *,
    origin: list[float] | None,
    destination: list[float] | None,
) -> list[dict[str, Any]]:
    """Extract map markers for start, intermediate stops, and destination.

    Args:
        task: Multi-stop GT-like dictionary.
        origin: Route origin point.
        destination: Route final point.

    Returns:
        Marker dictionaries consumed by the Leaflet renderer.
    """

    markers: list[dict[str, Any]] = []
    if origin:
        markers.append({"label": "S", "kind": "start", "point": origin, "popup": "Start"})

    stops = _task_stop_destinations(task)
    for index, stop in enumerate(stops, start=1):
        point = _point_pair_from_dict(stop)
        if not point:
            continue
        name = str(stop.get("name") or stop.get("label") or f"Stop {index}")
        markers.append({"label": str(index), "kind": "stop", "point": point, "popup": f"Stop {index}: {name}"})

    if destination:
        if not markers or markers[-1].get("point") != destination:
            markers.append({"label": "D", "kind": "finish", "point": destination, "popup": "Destination"})
        else:
            markers[-1]["kind"] = "finish"
            markers[-1]["popup"] = f"{markers[-1].get('popup', 'Destination')} / Final destination"
    return markers


def _task_stop_destinations(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ordered stop destinations from a multi-stop GT task.

    Args:
        task: Multi-stop GT-like dictionary.

    Returns:
        Ordered destination dictionaries.
    """

    sequence = task.get("gt_destination_sequence")
    if isinstance(sequence, list):
        return [dict(item) for item in sequence if isinstance(item, dict)]
    legs = task.get("gt_route_legs")
    if isinstance(legs, list):
        stops = []
        for leg in legs:
            if isinstance(leg, dict) and isinstance(leg.get("gt_destination"), dict):
                stops.append(dict(leg["gt_destination"]))
        return stops
    return []


def _metric_card(title: str, text: str, formula: str) -> str:
    """Build one metric definition card.

    Args:
        title: Metric title.
        text: Metric explanation.
        formula: LaTeX formula text.

    Returns:
        HTML card.
    """

    formula_html = _formula_html(formula)
    return f"""
    <div class="metric-card">
      <h4>{_esc(title)}</h4>
      <p>{_esc(text)}</p>
      {formula_html}
      <div class="math-note">如果这里仍显示 LaTeX 源码，请检查浏览器是否能访问 MathJax CDN。</div>
    </div>
    """


def _formula_html(formula: str) -> str:
    """Build MathJax display blocks from one or more LaTeX formulas.

    Args:
        formula: Formula text. Blank lines separate independent display blocks.

    Returns:
        HTML display math blocks.
    """

    blocks = []
    for raw_block in formula.strip().split("\n\n"):
        lines = [line.rstrip() for line in raw_block.splitlines() if line.strip()]
        if not lines:
            continue
        body = r" \\".join(lines)
        blocks.append(f"<div class='formula'>\\[\\begin{{gathered}}{_esc(body)}\\end{{gathered}}\\]</div>")
    return "".join(blocks)


def _pie_svg(items: list[tuple[str, int]], *, title: str) -> str:
    """Build an inline SVG pie chart.

    Args:
        items: Label/count pairs.
        title: Chart title.

    Returns:
        SVG string.
    """

    total = sum(value for _, value in items)
    if total <= 0:
        return "<svg width='240' height='240'></svg>"
    cx = cy = 120
    radius = 88
    start = -math.pi / 2
    paths = []
    for index, (label, value) in enumerate(items):
        angle = (value / total) * math.tau
        end = start + angle
        large = 1 if angle > math.pi else 0
        x1 = cx + radius * math.cos(start)
        y1 = cy + radius * math.sin(start)
        x2 = cx + radius * math.cos(end)
        y2 = cy + radius * math.sin(end)
        color = _color(index)
        paths.append(
            f"<path d='M {cx:.2f} {cy:.2f} L {x1:.2f} {y1:.2f} A {radius} {radius} 0 {large} 1 {x2:.2f} {y2:.2f} Z' fill='{color}'><title>{_esc(label)}: {value}</title></path>"
        )
        start = end
    return f"<svg width='240' height='240' viewBox='0 0 240 240' role='img' aria-label='{_esc(title)} pie chart'>{''.join(paths)}<circle cx='120' cy='120' r='48' fill='white'/><text x='120' y='116' text-anchor='middle' font-size='14' font-weight='700'>{_esc(title)}</text><text x='120' y='136' text-anchor='middle' font-size='12'>{total}</text></svg>"


def _legend(items: list[tuple[str, int]]) -> str:
    """Build a chart legend.

    Args:
        items: Label/count pairs.

    Returns:
        HTML legend.
    """

    total = sum(value for _, value in items) or 1
    rows = []
    for index, (label, value) in enumerate(items):
        percent = value / total * 100
        rows.append(f"<div><span style='background:{_color(index)}'></span>{_esc(label)}: {value} ({percent:.1f}%)</div>")
    return f"<div class='legend'>{''.join(rows)}</div>"


def _distribution_table(rows: list[dict[str, Any]], *, label_key: str, extra_key: str) -> str:
    """Build a distribution table.

    Args:
        rows: Distribution rows.
        label_key: Main label key.
        extra_key: Secondary label key.

    Returns:
        HTML table.
    """

    total = sum(int(row.get("count") or 0) for row in rows) or 1
    body = []
    for row in rows:
        count = int(row.get("count") or 0)
        label = row.get(label_key) or row.get("category") or row.get("city")
        extra = row.get(extra_key) or ""
        body.append(f"<tr><td>{_esc(label)}</td><td>{_esc(extra)}</td><td>{count}</td><td>{count / total * 100:.1f}%</td></tr>")
    return f"<table><thead><tr><th>Name</th><th>Group</th><th>Count</th><th>Share</th></tr></thead><tbody>{''.join(body)}</tbody></table>"


def _provider_name(case: UrbanNavCase) -> str:
    """Infer provider name from a case.

    Args:
        case: UrbanNav case.

    Returns:
        Provider label.
    """

    provider = case.optimality_scope.poi_provider or case.optimality_scope.route_provider
    if provider:
        return str(provider)
    metadata_provider = case.metadata.get("provider")
    if isinstance(metadata_provider, dict) and metadata_provider.get("provider_name"):
        return str(metadata_provider["provider_name"])
    return "unknown_provider"


def _country_name(case: UrbanNavCase) -> str:
    """Infer country name from metadata or city/provider.

    Args:
        case: UrbanNav case.

    Returns:
        Country label.
    """

    country = case.metadata.get("country")
    if country:
        return str(country)
    if str(case.optimality_scope.poi_provider).lower() == "amap":
        return "china"
    return "unknown_country"


def _point_pair(point: UrbanNavPoint | None) -> list[float] | None:
    """Convert a point to Leaflet coordinate pair.

    Args:
        point: Optional point.

    Returns:
        Coordinate pair or ``None``.
    """

    if point is None:
        return None
    return [point.lat, point.lng]


def _point_pair_from_dict(value: Any) -> list[float] | None:
    """Parse a coordinate dictionary into a Leaflet coordinate pair.

    Args:
        value: Coordinate-like dictionary.

    Returns:
        ``[lat, lng]`` pair, or ``None`` when unavailable.
    """

    if not isinstance(value, dict):
        return None
    try:
        point = UrbanNavPoint.from_dict(value)
    except (TypeError, ValueError):
        return None
    return [point.lat, point.lng]


def _parse_route_like(value: Any) -> list[list[float]]:
    """Parse common route representations into Leaflet coordinate pairs.

    Args:
        value: Route as ``[{lat,lng}, ...]`` or ``[[lng, lat], ...]``.

    Returns:
        List of ``[lat, lng]`` pairs.
    """

    if not isinstance(value, list):
        return []
    points: list[list[float]] = []
    for item in value:
        if isinstance(item, dict):
            pair = _point_pair_from_dict(item)
            if pair:
                points.append(pair)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                lng = float(item[0])
                lat = float(item[1])
            except (TypeError, ValueError):
                continue
            points.append([lat, lng])
    return points


def _point_text(point: UrbanNavPoint | None) -> str:
    """Format a coordinate.

    Args:
        point: Optional point.

    Returns:
        Human-readable coordinate.
    """

    if point is None:
        return "-"
    return f"{point.lat:.6f}, {point.lng:.6f}"


def _num(value: Any) -> str:
    """Format a numeric value.

    Args:
        value: Raw value.

    Returns:
        Formatted number.
    """

    try:
        return f"{float(value):.1f}"
    except Exception:
        return "-"


def _color(index: int) -> str:
    """Return chart color.

    Args:
        index: Color index.

    Returns:
        CSS color.
    """

    palette = [
        "#1769aa",
        "#2a9d8f",
        "#e76f51",
        "#8a5cf6",
        "#f4a261",
        "#3d5a80",
        "#d62828",
        "#457b9d",
        "#6a994e",
        "#bc6c25",
        "#7b2cbf",
        "#118ab2",
        "#ef476f",
        "#06d6a0",
        "#8338ec",
        "#fb5607",
        "#4d908e",
        "#577590",
        "#9d4edd",
        "#2b9348",
    ]
    return palette[index % len(palette)]


def _esc(value: Any) -> str:
    """Escape HTML text.

    Args:
        value: Raw value.

    Returns:
        Escaped string.
    """

    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
