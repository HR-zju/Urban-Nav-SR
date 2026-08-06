"""Deterministic JSON, CSV, and self-contained HTML scorecard export."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


def write_openloop_scorecard_v1(
    report: dict[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
    html_path: str | Path,
    title: str,
) -> None:
    """Write one scorecard in machine- and human-readable formats.

    Args:
        report: JSON-compatible evaluator output.
        json_path: Pretty JSON destination.
        csv_path: Flattened scalar metric destination.
        html_path: Self-contained review HTML destination.
        title: Human-readable report title.
    """

    json_destination = Path(json_path).expanduser()
    csv_destination = Path(csv_path).expanduser()
    html_destination = Path(html_path).expanduser()
    for destination in (
        json_destination,
        csv_destination,
        html_destination,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    json_destination.write_text(rendered + "\n", encoding="utf-8")
    scalar_rows = list(_flatten_scalars(report))
    with csv_destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerows(
            {"metric": metric, "value": value}
            for metric, value in scalar_rows
        )
    html_destination.write_text(
        _render_html(title=title, report=report, scalar_rows=scalar_rows),
        encoding="utf-8",
    )


def _flatten_scalars(
    value: Any,
    *,
    prefix: str = "",
):
    """Yield dotted paths for scalar JSON leaves."""

    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_scalars(item, prefix=path)
    elif isinstance(value, list):
        return
    elif value is None or isinstance(value, (str, int, float, bool)):
        yield prefix, value


def _render_html(
    *,
    title: str,
    report: dict[str, Any],
    scalar_rows: list[tuple[str, Any]],
) -> str:
    """Render one dependency-free local scorecard page."""

    table_rows = "\n".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(
            html.escape(metric),
            html.escape(json.dumps(value, ensure_ascii=False)),
        )
        for metric, value in scalar_rows
    )
    raw_json = html.escape(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#17202a;background:#f7f9fb}}
main{{max-width:1200px;margin:auto}}
table{{border-collapse:collapse;width:100%;background:white}}
th,td{{border:1px solid #d9e1e8;padding:7px 9px;text-align:left;vertical-align:top}}
th{{background:#eaf0f5;position:sticky;top:0}}
details{{margin-top:20px}}
pre{{white-space:pre-wrap;background:#101820;color:#e8eef2;padding:16px;overflow:auto}}
</style>
</head>
<body><main>
<h1>{html.escape(title)}</h1>
<p>This report preserves the evaluator JSON exactly and lists scalar leaves for review.</p>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
{table_rows}
</tbody></table>
<details><summary>Complete scorecard JSON</summary><pre>{raw_json}</pre></details>
</main></body></html>
"""


__all__ = ["write_openloop_scorecard_v1"]
