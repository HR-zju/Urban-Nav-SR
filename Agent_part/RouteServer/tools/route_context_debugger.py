from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>RouteContext Debugger</title>
  <style>
    body {{
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #0f172a;
      color: #e2e8f0;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 12px;
      padding: 12px;
    }}
    .panel {{
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 12px;
    }}
    .stack {{
      display: grid;
      gap: 12px;
    }}
    h1, h2, h3 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    h3 {{
      font-size: 14px;
    }}
    #bev {{
      width: 100%;
      height: 440px;
      background: #020617;
      border-radius: 8px;
      border: 1px solid #1e293b;
    }}
    #localPatch {{
      width: 100%;
      height: 420px;
      background: #020617;
      border-radius: 8px;
      border: 1px solid #1e293b;
    }}
    #charts {{
      width: 100%;
      height: 280px;
      background: #020617;
      border-radius: 8px;
      border: 1px solid #1e293b;
    }}
    .row {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }}
    .meta {{
      color: #94a3b8;
      font-size: 12px;
    }}
    .tag {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: #1e293b;
      font-size: 12px;
    }}
    button {{
      background: #1d4ed8;
      color: white;
      border: 0;
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
      font-family: inherit;
    }}
    button.secondary {{
      background: #334155;
    }}
    button:disabled {{
      opacity: 0.5;
      cursor: default;
    }}
    input[type=range] {{
      width: 100%;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
    }}
    .legend {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 12px;
      color: #cbd5e1;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 12px;
      height: 3px;
      margin-right: 6px;
      vertical-align: middle;
      border-radius: 999px;
    }}
    .route::before {{ background: #22c55e; }}
    .local::before {{ background: #3b82f6; }}
    .connector::before {{ background: #fb923c; }}
    .heading::before {{ background: #f43f5e; }}
    .tangent::before {{ background: #facc15; }}
    .patchcorridor::before {{ background: rgba(96,165,250,0.8); }}
    .patchturn::before {{ background: rgba(245,158,11,0.8); }}
    .controls {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
      margin-bottom: 10px;
    }}
    .toggles {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      font-size: 12px;
      color: #cbd5e1;
    }}
    .field-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      font-size: 12px;
      margin-top: 10px;
    }}
    .field {{
      background: #0b1220;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 8px;
    }}
    .field .label {{
      color: #94a3b8;
      margin-bottom: 4px;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="stack">
      <div class="panel">
        <div class="row">
          <h1>RouteContext Debugger</h1>
          <span class="tag" id="frameTag">frame 0</span>
          <span class="tag" id="turnTag">turn</span>
          <span class="tag" id="corridorTag">corridor</span>
          <span class="tag" id="offRouteTag">on_route</span>
        </div>
        <div class="meta" id="metaLine"></div>
        <div class="controls">
          <button id="prevBtn" class="secondary">Prev</button>
          <button id="playBtn">Play</button>
          <button id="nextBtn" class="secondary">Next</button>
        </div>
        <div style="margin-bottom:12px;">
          <input id="frameSlider" type="range" min="0" max="0" value="0" />
        </div>
        <div class="legend" style="margin:10px 0;">
          <span class="route">full route</span>
          <span class="local">local segment</span>
          <span class="connector">connector</span>
          <span class="heading">heading</span>
          <span class="tangent">route tangent</span>
        </div>
        <svg id="bev" viewBox="0 0 900 440"></svg>
      </div>
      <div class="panel">
        <h2>Time Series</h2>
        <svg id="charts" viewBox="0 0 900 280"></svg>
      </div>
    </div>
    <div class="stack">
      <div class="panel">
        <div class="row">
          <h2>Local BEV Context</h2>
          <span class="tag" id="patchTurnTag">turn</span>
          <span class="tag" id="patchCorridorTag">corridor</span>
        </div>
        <div class="toggles">
          <label><input type="checkbox" id="togglePose" checked /> current pose</label>
          <label><input type="checkbox" id="toggleHeading" checked /> current heading</label>
          <label><input type="checkbox" id="toggleLabels" checked /> C / M / W / T</label>
          <label><input type="checkbox" id="toggleOffRoute" checked /> off-route mark</label>
        </div>
        <div class="legend" style="margin:10px 0;">
          <span class="route">route polyline local</span>
          <span class="local">local segment</span>
          <span class="connector">connector</span>
          <span class="patchcorridor">corridor prior</span>
          <span class="patchturn">turn prior</span>
        </div>
        <svg id="localPatch" viewBox="0 0 680 420"></svg>
        <div class="field-grid">
          <div class="field"><div class="label">matched_segment_index</div><div id="fieldSegment">-</div></div>
          <div class="field"><div class="label">cross_track_error_m</div><div id="fieldCrossTrack">-</div></div>
          <div class="field"><div class="label">distance_to_next_turn_m</div><div id="fieldDistanceToTurn">-</div></div>
          <div class="field"><div class="label">turn_type</div><div id="fieldTurnType">-</div></div>
          <div class="field"><div class="label">corridor_type</div><div id="fieldCorridorType">-</div></div>
          <div class="field"><div class="label">off_route</div><div id="fieldOffRoute">-</div></div>
        </div>
      </div>
      <div class="panel">
        <h3>Current RouteContext JSON</h3>
        <pre id="jsonView"></pre>
      </div>
    </div>
  </div>
  <script>
    const CAPTURE = __CAPTURE_JSON__;
    let currentFrameIndex = 0;
    let isPlaying = false;
    let playTimer = null;

    function corridorValue(name) {{
      const mapping = {{
        "right_front": -2,
        "slight_right_front": -1,
        "front": 0,
        "slight_left_front": 1,
        "left_front": 2,
        "uncertain": 3,
      }};
      return Object.prototype.hasOwnProperty.call(mapping, name) ? mapping[name] : 0;
    }}

    function flattenPoints(frame) {{
      const pose = frame.route_context.pose_match || {{}};
      const geom = frame.route_context.geometry || {{}};
      const points = [];
      const pushPoint = (p) => {{
        if (!p || typeof p.lat !== "number" || typeof p.lng !== "number") return;
        points.push(p);
      }};
      (CAPTURE.meta.route_polyline || []).forEach((p) => pushPoint({{lat: p.latitude ?? p.lat, lng: p.longitude ?? p.lng}}));
      (geom.local_segment_polyline || []).forEach(pushPoint);
      (geom.connector_polyline || []).forEach(pushPoint);
      pushPoint(frame.gps);
      pushPoint(pose.matched_point);
      return points;
    }}

    function computeViewport(frame) {{
      const points = flattenPoints(frame);
      let minLat = Infinity, maxLat = -Infinity, minLng = Infinity, maxLng = -Infinity;
      points.forEach((p) => {{
        minLat = Math.min(minLat, p.lat);
        maxLat = Math.max(maxLat, p.lat);
        minLng = Math.min(minLng, p.lng);
        maxLng = Math.max(maxLng, p.lng);
      }});
      if (!isFinite(minLat)) {{
        minLat = 0; maxLat = 1; minLng = 0; maxLng = 1;
      }}
      const latPad = Math.max((maxLat - minLat) * 0.15, 0.00035);
      const lngPad = Math.max((maxLng - minLng) * 0.15, 0.00035);
      return {{
        minLat: minLat - latPad,
        maxLat: maxLat + latPad,
        minLng: minLng - lngPad,
        maxLng: maxLng + lngPad,
      }};
    }}

    function projectPoint(point, viewport, width, height) {{
      const x = ((point.lng - viewport.minLng) / Math.max(1e-9, viewport.maxLng - viewport.minLng)) * width;
      const y = height - ((point.lat - viewport.minLat) / Math.max(1e-9, viewport.maxLat - viewport.minLat)) * height;
      return {{ x, y }};
    }}

    function svgPolyline(points, viewport, width, height, color, strokeWidth, dasharray="") {{
      const coords = points
        .map((p) => {{
          const lat = p.latitude ?? p.lat;
          const lng = p.longitude ?? p.lng;
          if (typeof lat !== "number" || typeof lng !== "number") return null;
          const q = projectPoint({{lat, lng}}, viewport, width, height);
          return `${{q.x.toFixed(1)}},${{q.y.toFixed(1)}}`;
        }})
        .filter(Boolean)
        .join(" ");
      if (!coords) return "";
      const dashAttr = dasharray ? ` stroke-dasharray="${{dasharray}}"` : "";
      return `<polyline fill="none" stroke="${{color}}" stroke-width="${{strokeWidth}}" stroke-linecap="round" stroke-linejoin="round"${{dashAttr}} points="${{coords}}" />`;
    }}

    function svgCircle(point, viewport, width, height, color, radius) {{
      if (!point || typeof point.lat !== "number" || typeof point.lng !== "number") return "";
      const p = projectPoint(point, viewport, width, height);
      return `<circle cx="${{p.x.toFixed(1)}}" cy="${{p.y.toFixed(1)}}" r="${{radius}}" fill="${{color}}" />`;
    }}

    function svgText(point, viewport, width, height, text, color) {{
      if (!point || typeof point.lat !== "number" || typeof point.lng !== "number") return "";
      const p = projectPoint(point, viewport, width, height);
      return `<text x="${{(p.x + 8).toFixed(1)}}" y="${{(p.y - 8).toFixed(1)}}" fill="${{color}}" font-size="12">${{text}}</text>`;
    }}

    function vectorEndpoint(origin, headingDeg, lengthPx, viewport, width, height) {{
      const base = projectPoint(origin, viewport, width, height);
      const rad = (headingDeg - 90) * Math.PI / 180;
      return {{ x: base.x + Math.cos(rad) * lengthPx, y: base.y + Math.sin(rad) * lengthPx }};
    }}

    function lineFromProjected(origin, endpoint, viewport, width, height, color, strokeWidth) {{
      const a = projectPoint(origin, viewport, width, height);
      return `<line x1="${{a.x.toFixed(1)}}" y1="${{a.y.toFixed(1)}}" x2="${{endpoint.x.toFixed(1)}}" y2="${{endpoint.y.toFixed(1)}}" stroke="${{color}}" stroke-width="${{strokeWidth}}" stroke-linecap="round" />`;
    }}

    function localTangentDeg(localSegment) {{
      if (!Array.isArray(localSegment) || localSegment.length < 2) return null;
      const start = localSegment[0];
      for (let i = 1; i < localSegment.length; i += 1) {{
        const candidate = localSegment[i];
        if (Math.abs(candidate.lat - start.lat) > 1e-9 || Math.abs(candidate.lng - start.lng) > 1e-9) {{
          const dy = candidate.lat - start.lat;
          const dx = candidate.lng - start.lng;
          return (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360;
        }}
      }}
      return null;
    }}

    function renderBEV(frameIndex) {{
      const frame = CAPTURE.frames[frameIndex];
      const svg = document.getElementById("bev");
      const width = 900;
      const height = 440;
      const viewport = computeViewport(frame);
      const pose = frame.route_context.pose_match || {{}};
      const geom = frame.route_context.geometry || {{}};
      const current = frame.gps;
      const matched = pose.matched_point;
      const nextWaypointIndex = frame.route_context.progress?.next_waypoint_index;
      let nextWaypoint = null;
      (CAPTURE.meta.coarse_waypoints || []).forEach((wp, idx) => {{
        if (idx === nextWaypointIndex) {{
          const c = wp.coordinate || {{}};
          nextWaypoint = {{ lat: c.latitude ?? wp.latitude, lng: c.longitude ?? wp.longitude }};
        }}
      }});

      let html = `<rect x="0" y="0" width="${{width}}" height="${{height}}" fill="#020617" />`;
      html += svgPolyline(CAPTURE.meta.route_polyline || [], viewport, width, height, "#22c55e", 4);
      html += svgPolyline(geom.local_segment_polyline || [], viewport, width, height, "#3b82f6", 5);
      html += svgPolyline(geom.connector_polyline || [], viewport, width, height, "#fb923c", 4, "8 6");
      html += svgCircle(current, viewport, width, height, "#3b82f6", 6);
      html += svgCircle(matched, viewport, width, height, "#10b981", 5);
      html += svgCircle(nextWaypoint, viewport, width, height, "#facc15", 5);
      html += svgText(current, viewport, width, height, "C", "#93c5fd");
      html += svgText(matched, viewport, width, height, "M", "#6ee7b7");
      html += svgText(nextWaypoint, viewport, width, height, "W", "#fde68a");

      if (typeof frame.heading_deg === "number") {{
        html += lineFromProjected(current, vectorEndpoint(current, frame.heading_deg, 48, viewport, width, height), viewport, width, height, "#f43f5e", 3);
      }}
      const tangentDeg = localTangentDeg(geom.local_segment_polyline || []);
      if (typeof tangentDeg === "number") {{
        const anchor = matched || current;
        html += lineFromProjected(anchor, vectorEndpoint(anchor, tangentDeg, 42, viewport, width, height), viewport, width, height, "#facc15", 3);
      }}
      if (pose.off_route) {{
        html += `<text x="18" y="28" fill="#fb7185" font-size="13">OFF ROUTE</text>`;
      }}
      svg.innerHTML = html;
    }}

    function buildSeries() {{
      return CAPTURE.frames.map((frame, index) => {{
        const debug = frame.debug || {{}};
        return {{
          index,
          segment: typeof debug.matched_segment_index === "number" ? debug.matched_segment_index : 0,
          crossTrack: typeof debug.cross_track_error_m === "number" ? debug.cross_track_error_m : 0,
          headingDelta: typeof debug.heading_delta_deg === "number" ? debug.heading_delta_deg : 0,
          distanceToTurn: typeof debug.distance_to_next_turn_m === "number" ? debug.distance_to_next_turn_m : 0,
          corridor: corridorValue(debug.corridor_type || "front"),
        }};
      }});
    }}

    function drawSeries(frameIndex) {{
      const svg = document.getElementById("charts");
      const width = 900;
      const height = 280;
      const margin = 28;
      const series = buildSeries();
      const plots = [
        {{ key: "segment", color: "#22c55e", label: "segment index" }},
        {{ key: "crossTrack", color: "#fb923c", label: "cross-track" }},
        {{ key: "headingDelta", color: "#f43f5e", label: "heading delta" }},
        {{ key: "distanceToTurn", color: "#3b82f6", label: "distance to turn" }},
        {{ key: "corridor", color: "#a855f7", label: "corridor" }},
      ];
      const maxValue = Math.max(1, ...plots.flatMap((plot) => series.map((item) => Math.abs(item[plot.key]))));
      const stepX = (width - margin * 2) / Math.max(1, series.length - 1);
      let html = `<rect x="0" y="0" width="${{width}}" height="${{height}}" fill="#020617" />`;
      html += `<line x1="${{margin}}" y1="${{height - margin}}" x2="${{width - margin}}" y2="${{height - margin}}" stroke="#334155" />`;
      html += `<line x1="${{margin}}" y1="${{margin}}" x2="${{margin}}" y2="${{height - margin}}" stroke="#334155" />`;
      plots.forEach((plot, plotIndex) => {{
        const points = series.map((item, idx) => {{
          const x = margin + idx * stepX;
          const normalized = item[plot.key] / maxValue;
          const y = height - margin - normalized * (height - margin * 2);
          return `${{x.toFixed(1)}},${{y.toFixed(1)}}`;
        }}).join(" ");
        html += `<polyline fill="none" stroke="${{plot.color}}" stroke-width="2" points="${{points}}" />`;
        html += `<text x="${{margin + 8}}" y="${{18 + plotIndex * 14}}" fill="${{plot.color}}" font-size="11">${{plot.label}}</text>`;
      }});
      const cursorX = margin + frameIndex * stepX;
      html += `<line x1="${{cursorX.toFixed(1)}}" y1="${{margin}}" x2="${{cursorX.toFixed(1)}}" y2="${{height - margin}}" stroke="#e2e8f0" stroke-dasharray="4 4" />`;
      svg.innerHTML = html;
    }}

    function projectPatchPoint(point, bbox, width, height, padding=24) {{
      if (!point || typeof point.x_m !== "number" || typeof point.y_m !== "number") return null;
      const x = padding + (point.x_m - bbox.x_min_m) / Math.max(bbox.x_max_m - bbox.x_min_m, 1e-6) * (width - padding * 2);
      const y = padding + (bbox.y_max_m - point.y_m) / Math.max(bbox.y_max_m - bbox.y_min_m, 1e-6) * (height - padding * 2);
      return {{ x, y }};
    }}

    function patchPolyline(points, bbox, width, height, color, strokeWidth, dasharray="") {{
      const coords = (points || []).map((p) => projectPatchPoint(p, bbox, width, height)).filter(Boolean).map((p) => `${{p.x.toFixed(1)}},${{p.y.toFixed(1)}}`).join(" ");
      if (!coords) return "";
      const dashAttr = dasharray ? ` stroke-dasharray="${{dasharray}}"` : "";
      return `<polyline fill="none" stroke="${{color}}" stroke-width="${{strokeWidth}}" stroke-linecap="round" stroke-linejoin="round"${{dashAttr}} points="${{coords}}" />`;
    }}

    function patchCircle(point, bbox, width, height, fill, radius) {{
      const p = projectPatchPoint(point, bbox, width, height);
      if (!p) return "";
      return `<circle cx="${{p.x.toFixed(1)}}" cy="${{p.y.toFixed(1)}}" r="${{radius}}" fill="${{fill}}" />`;
    }}

    function patchText(point, bbox, width, height, text, fill) {{
      const p = projectPatchPoint(point, bbox, width, height);
      if (!p) return "";
      return `<text x="${{(p.x + 8).toFixed(1)}}" y="${{(p.y - 8).toFixed(1)}}" fill="${{fill}}" font-size="12">${{text}}</text>`;
    }}

    function patchVector(point, yawDeg, bbox, width, height, color) {{
      const p = projectPatchPoint(point, bbox, width, height);
      if (!p) return "";
      const rad = (yawDeg - 90) * Math.PI / 180;
      const x2 = p.x + Math.cos(rad) * 42;
      const y2 = p.y + Math.sin(rad) * 42;
      return `<line x1="${{p.x.toFixed(1)}}" y1="${{p.y.toFixed(1)}}" x2="${{x2.toFixed(1)}}" y2="${{y2.toFixed(1)}}" stroke="${{color}}" stroke-width="3" stroke-linecap="round" />`;
    }}

    function patchSectorPath(bbox, width, height, centerDeg, spanDeg, radiusM) {{
      const origin = projectPatchPoint({{x_m: 0, y_m: 0}}, bbox, width, height);
      if (!origin) return "";
      const points = [`${{origin.x.toFixed(1)}},${{origin.y.toFixed(1)}}`];
      const startDeg = centerDeg - spanDeg / 2;
      const endDeg = centerDeg + spanDeg / 2;
      const samples = 12;
      for (let i = 0; i <= samples; i += 1) {{
        const deg = startDeg + (endDeg - startDeg) * i / samples;
        const rad = deg * Math.PI / 180;
        const pt = projectPatchPoint({{x_m: Math.sin(rad) * radiusM, y_m: Math.cos(rad) * radiusM}}, bbox, width, height);
        if (pt) points.push(`${{pt.x.toFixed(1)}},${{pt.y.toFixed(1)}}`);
      }}
      return points.join(" ");
    }}

    function renderLocalPatch(frameIndex) {{
      const frame = CAPTURE.frames[frameIndex];
      const patch = document.getElementById("localPatch");
      const width = 680;
      const height = 420;
      const local = frame.local_bev_context || {{}};
      const bbox = local.bev_bbox || {{ x_min_m: -15, x_max_m: 15, y_min_m: -10, y_max_m: 30 }};
      const toggles = {{
        pose: document.getElementById("togglePose").checked,
        heading: document.getElementById("toggleHeading").checked,
        labels: document.getElementById("toggleLabels").checked,
        offRoute: document.getElementById("toggleOffRoute").checked,
      }};

      let html = `<rect x="0" y="0" width="${{width}}" height="${{height}}" fill="#020617" />`;
      html += `<line x1="24" y1="${{height - 24}}" x2="120" y2="${{height - 24}}" stroke="#334155" /><line x1="24" y1="${{height - 24}}" x2="24" y2="${{height - 120}}" stroke="#334155" />`;
      html += `<text x="126" y="${{height - 20}}" fill="#64748b" font-size="11">+x</text><text x="8" y="${{height - 126}}" fill="#64748b" font-size="11">+y</text>`;

      const corridorSector = local.corridor_sector_deg || {{ center_deg: 0, span_deg: 60 }};
      const turnSector = local.turn_sector_deg || {{ center_deg: 0, span_deg: 35 }};
      html += `<polygon points="${{patchSectorPath(bbox, width, height, corridorSector.center_deg || 0, corridorSector.span_deg || 60, Math.max(local.lookahead_m || 20, 12))}}" fill="#60a5fa" fill-opacity="0.10" stroke="none" />`;
      html += `<polygon points="${{patchSectorPath(bbox, width, height, turnSector.center_deg || 0, turnSector.span_deg || 35, Math.max((local.lookahead_m || 20) * 0.85, 10))}}" fill="#f59e0b" fill-opacity="0.10" stroke="none" />`;

      html += patchPolyline(local.route_polyline_local || [], bbox, width, height, "#22c55e", 4);
      html += patchPolyline(local.local_segment_bev || [], bbox, width, height, "#3b82f6", 5);
      html += patchPolyline(local.connector_bev || [], bbox, width, height, "#fb923c", 4, "8 6");

      if (toggles.pose) {{
        html += patchCircle(local.current_pose_bev, bbox, width, height, "#3b82f6", 6);
      }}
      html += patchCircle(local.matched_point_bev, bbox, width, height, "#10b981", 5);
      html += patchCircle(local.next_waypoint_bev, bbox, width, height, "#facc15", 5);
      html += patchCircle(local.next_turn_point_bev, bbox, width, height, "#f97316", 5);

      if (toggles.heading && local.current_pose_bev && typeof local.current_pose_bev.yaw_deg === "number") {{
        html += patchVector(local.current_pose_bev, local.current_pose_bev.yaw_deg, bbox, width, height, "#f43f5e");
      }}

      if (toggles.labels) {{
        html += patchText(local.current_pose_bev, bbox, width, height, "C", "#93c5fd");
        html += patchText(local.matched_point_bev, bbox, width, height, "M", "#6ee7b7");
        html += patchText(local.next_waypoint_bev, bbox, width, height, "W", "#fde68a");
        html += patchText(local.next_turn_point_bev, bbox, width, height, "T", "#fdba74");
      }}

      if (toggles.offRoute && frame.route_context?.pose_match?.off_route) {{
        html += `<text x="18" y="28" fill="#fb7185" font-size="13">OFF ROUTE</text>`;
      }}

      patch.innerHTML = html;
    }}

    function setFields(frameIndex) {{
      const frame = CAPTURE.frames[frameIndex];
      const debug = frame.debug || {{}};
      document.getElementById("fieldSegment").textContent = String(debug.matched_segment_index ?? "-");
      document.getElementById("fieldCrossTrack").textContent = debug.cross_track_error_m ?? "-";
      document.getElementById("fieldDistanceToTurn").textContent = debug.distance_to_next_turn_m ?? "-";
      document.getElementById("fieldTurnType").textContent = debug.turn_type || "-";
      document.getElementById("fieldCorridorType").textContent = debug.corridor_type || "-";
      document.getElementById("fieldOffRoute").textContent = String(Boolean(debug.off_route));
    }}

    function renderFrame(frameIndex) {{
      currentFrameIndex = frameIndex;
      const frame = CAPTURE.frames[frameIndex];
      const debug = frame.debug || {{}};
      document.getElementById("frameTag").textContent = `frame ${{frameIndex}}`;
      document.getElementById("turnTag").textContent = debug.turn_semantics || "unknown";
      document.getElementById("corridorTag").textContent = debug.corridor_type || "unknown";
      document.getElementById("offRouteTag").textContent = debug.off_route ? "off_route" : "on_route";
      document.getElementById("patchTurnTag").textContent = frame.local_bev_context?.turn_semantics || "unknown";
      document.getElementById("patchCorridorTag").textContent = frame.local_bev_context?.corridor_type || "unknown";
      document.getElementById("metaLine").textContent =
        `gps=(${{frame.gps.lat.toFixed(6)}}, ${{frame.gps.lng.toFixed(6)}}) heading=${{frame.heading_deg ?? "n/a"}} cross_track=${{debug.cross_track_error_m ?? "n/a"}}m next_turn=${{debug.distance_to_next_turn_m ?? "n/a"}}m`;
      document.getElementById("jsonView").textContent = JSON.stringify({{
        route_context: frame.route_context,
        local_bev_context: frame.local_bev_context
      }}, null, 2);
      document.getElementById("frameSlider").value = String(frameIndex);
      setFields(frameIndex);
      renderBEV(frameIndex);
      renderLocalPatch(frameIndex);
      drawSeries(frameIndex);
    }}

    function nextFrame() {{
      renderFrame(Math.min(currentFrameIndex + 1, CAPTURE.frames.length - 1));
    }}

    function prevFrame() {{
      renderFrame(Math.max(currentFrameIndex - 1, 0));
    }}

    function setPlaying(playing) {{
      isPlaying = playing;
      const playBtn = document.getElementById("playBtn");
      playBtn.textContent = playing ? "Pause" : "Play";
      if (playTimer) {{
        clearInterval(playTimer);
        playTimer = null;
      }}
      if (playing) {{
        playTimer = setInterval(() => {{
          if (currentFrameIndex >= CAPTURE.frames.length - 1) {{
            setPlaying(false);
            return;
          }}
          nextFrame();
        }}, 700);
      }}
    }}

    function init() {{
      const slider = document.getElementById("frameSlider");
      slider.max = String(Math.max(CAPTURE.frames.length - 1, 0));
      slider.addEventListener("input", (event) => renderFrame(Number(event.target.value)));
      document.getElementById("prevBtn").addEventListener("click", () => prevFrame());
      document.getElementById("nextBtn").addEventListener("click", () => nextFrame());
      document.getElementById("playBtn").addEventListener("click", () => setPlaying(!isPlaying));
      ["togglePose", "toggleHeading", "toggleLabels", "toggleOffRoute"].forEach((id) => {{
        document.getElementById(id).addEventListener("change", () => renderLocalPatch(currentFrameIndex));
      }});
      renderFrame(0);
    }}

    init();
  </script>
</body>
</html>
"""


def load_capture(path: str) -> dict[str, Any]:
    capture_path = Path(path)
    if not capture_path.exists():
        raise FileNotFoundError(
            f"Capture file not found: {path}. "
            "Run tools/route_context_capture.py first to generate a capture JSON."
        )
    return json.loads(capture_path.read_text(encoding="utf-8"))


def render_html(capture: dict[str, Any]) -> str:
    return HTML_TEMPLATE.replace("__CAPTURE_JSON__", json.dumps(capture, ensure_ascii=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an HTML RouteContext debugger replay from a capture JSON file")
    parser.add_argument("--capture", required=True, help="Path to RouteContext capture JSON")
    parser.add_argument("--output", required=True, help="Output HTML path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture = load_capture(args.capture)
    Path(args.output).write_text(render_html(capture), encoding="utf-8")
    print(f"Wrote RouteContext debugger HTML to {args.output}")


if __name__ == "__main__":
    main()
