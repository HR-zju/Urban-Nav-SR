#!/usr/bin/env python3
"""Fetch static FPV street-view/panorama images for RouteServer pipelines.

This tool reuses the Street View Static API URL pattern used by
Sparsely-Grounded-Long-Range-Navigation, but writes an offline dataset contract
that this repository already consumes:

    output_dir/
      images/000001.jpg
      trajectory.json
      frames_manifest.json
      streetview_manifest.json
      metadata/*.json

Supported providers:
  - google: Google Street View Static API + optional metadata lookup/signing
  - baidu: Baidu Map Panorama static image API

AMap and OSM are route/map providers in this repository, but they do not expose
an equivalent first-person panorama static-image API through the current code.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


GOOGLE_STREETVIEW_ENDPOINT = "https://maps.googleapis.com/maps/api/streetview"
GOOGLE_STREETVIEW_METADATA_ENDPOINT = "https://maps.googleapis.com/maps/api/streetview/metadata"
BAIDU_PANORAMA_ENDPOINT = "https://api.map.baidu.com/panorama/v2"


@dataclass(frozen=True)
class SourcePoint:
    index: int
    lat: float
    lng: float
    heading_deg: float
    pitch_deg: float | None = None
    pano_id: str | None = None
    frame_id: str | None = None
    timestamp: int | float | str | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class ViewRequest:
    source: SourcePoint
    heading_deg: float
    pitch_deg: float
    fov_deg: float
    width: int
    height: int
    view_index: int


def _finite_float(value: Any, *, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _required_float(value: Any, name: str) -> float:
    parsed = _finite_float(value)
    if parsed is None:
        raise ValueError(f"Missing or invalid {name}: {value!r}")
    return parsed


def _heading(value: Any, fallback: float = 0.0) -> float:
    parsed = _finite_float(value, default=fallback)
    assert parsed is not None
    return parsed % 360.0


def _pick(raw: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in raw and raw[name] not in (None, ""):
            return raw[name]
    return None


def load_points_from_trajectory(path: Path) -> list[SourcePoint]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError(f"trajectory-json must contain a list, got {type(raw)}")

    points: list[SourcePoint] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        lat = _pick(item, "lat", "latitude")
        lng = _pick(item, "lng", "lon", "longitude")
        if lat is None or lng is None:
            continue
        image_path = str(item.get("image_path") or "").strip()
        frame_id = str(item.get("frame_id") or item.get("frame_name") or "").strip() or None
        if frame_id is None and image_path:
            frame_id = Path(image_path).stem
        points.append(
            SourcePoint(
                index=len(points),
                lat=_required_float(lat, "lat"),
                lng=_required_float(lng, "lng"),
                heading_deg=_heading(_pick(item, "heading_deg", "heading", "yaw_deg"), 0.0),
                pitch_deg=_finite_float(_pick(item, "pitch_deg", "pitch")),
                pano_id=str(_pick(item, "pano_id", "panoid", "panorama_id") or "").strip() or None,
                frame_id=frame_id,
                timestamp=_pick(item, "timestamp", "timestamp_ns", "time"),
                raw=item,
            )
        )
    return points


def load_points_from_csv(path: Path) -> list[SourcePoint]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    points: list[SourcePoint] = []
    for row in rows:
        lat = _pick(row, "lat", "latitude")
        lng = _pick(row, "lng", "lon", "longitude")
        if lat is None or lng is None:
            continue
        points.append(
            SourcePoint(
                index=len(points),
                lat=_required_float(lat, "lat"),
                lng=_required_float(lng, "lng"),
                heading_deg=_heading(_pick(row, "heading_deg", "heading", "yaw_deg"), 0.0),
                pitch_deg=_finite_float(_pick(row, "pitch_deg", "pitch")),
                pano_id=str(_pick(row, "pano_id", "panoid", "panorama_id") or "").strip() or None,
                frame_id=str(_pick(row, "frame_id", "frame_name", "id") or "").strip() or None,
                timestamp=_pick(row, "timestamp", "timestamp_ns", "time"),
                raw=dict(row),
            )
        )
    return points


def select_points(points: list[SourcePoint], *, start: int | None, end: int | None, step: int) -> list[SourcePoint]:
    start_i = 0 if start is None else max(0, int(start))
    end_i = len(points) if end is None else min(len(points), int(end))
    step_i = max(1, int(step))
    return points[start_i:end_i:step_i]


def _url_with_params(base_url: str, params: Iterable[tuple[str, Any]]) -> str:
    clean = [(key, str(value)) for key, value in params if value is not None and str(value) != ""]
    return f"{base_url}?{urllib.parse.urlencode(clean)}"


def sign_google_streetview_url(input_url: str, secret: str) -> str:
    parsed = urllib.parse.urlparse(input_url)
    url_to_sign = f"{parsed.path}?{parsed.query}"
    padded_secret = secret + ("=" * (-len(secret) % 4))
    decoded_key = base64.urlsafe_b64decode(padded_secret)
    signature = hmac.new(decoded_key, url_to_sign.encode("utf-8"), hashlib.sha1)
    encoded_signature = base64.urlsafe_b64encode(signature.digest()).decode("utf-8")
    return f"{input_url}&signature={encoded_signature}"


def fetch_bytes(url: str, *, timeout_s: float, user_agent: str) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
            return int(response.status), headers, response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), {str(k).lower(): str(v) for k, v in exc.headers.items()}, exc.read()


def fetch_json(url: str, *, timeout_s: float, user_agent: str) -> tuple[int, dict[str, str], dict[str, Any]]:
    status, headers, body = fetch_bytes(url, timeout_s=timeout_s, user_agent=user_agent)
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        data = {"raw_text": body.decode("utf-8", errors="replace")}
    return status, headers, data


def google_metadata_url(point: SourcePoint, *, api_key: str, radius_m: int) -> str:
    params: list[tuple[str, Any]] = [("key", api_key)]
    if point.pano_id:
        params.append(("pano", point.pano_id))
    else:
        params.extend(
            [
                ("location", f"{point.lat:.8f},{point.lng:.8f}"),
                ("radius", int(radius_m)),
            ]
        )
    return _url_with_params(GOOGLE_STREETVIEW_METADATA_ENDPOINT, params)


def google_image_url(
    view: ViewRequest,
    *,
    api_key: str,
    pano_id: str | None,
    return_error_code: bool,
) -> str:
    params: list[tuple[str, Any]] = [
        ("size", f"{view.width}x{view.height}"),
        ("heading", round(view.heading_deg % 360.0, 6)),
        ("pitch", round(view.pitch_deg, 6)),
        ("fov", round(view.fov_deg, 6)),
        ("key", api_key),
    ]
    if pano_id:
        params.insert(1, ("pano", pano_id))
    else:
        params.insert(1, ("location", f"{view.source.lat:.8f},{view.source.lng:.8f}"))
    if return_error_code:
        params.append(("return_error_code", "true"))
    return _url_with_params(GOOGLE_STREETVIEW_ENDPOINT, params)


def baidu_image_url(view: ViewRequest, *, ak: str, pano_id: str | None, coordtype: str) -> str:
    params: list[tuple[str, Any]] = [
        ("ak", ak),
        ("width", int(view.width)),
        ("height", int(view.height)),
        ("fov", round(view.fov_deg, 6)),
        ("heading", round(view.heading_deg % 360.0, 6)),
        ("pitch", round(view.pitch_deg, 6)),
    ]
    if pano_id:
        params.append(("panoid", pano_id))
    else:
        params.append(("location", f"{view.source.lng:.8f},{view.source.lat:.8f}"))
        if coordtype:
            params.append(("coordtype", coordtype))
    return _url_with_params(BAIDU_PANORAMA_ENDPOINT, params)


def _rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def pano_band_limits(top_trim_deg: float, bottom_trim_deg: float) -> tuple[float, float]:
    lam_min = -math.pi / 2 + _rad(bottom_trim_deg)
    lam_max = math.pi / 2 - _rad(top_trim_deg)
    if not lam_min < lam_max:
        raise ValueError("Invalid trims: top + bottom trim must be less than 180 degrees.")
    return lam_min, lam_max


def pano_centers(
    x_count: int,
    y_count: int,
    top_trim_deg: float,
    bottom_trim_deg: float,
    *,
    equal_area: bool,
    yaw0_deg: float,
) -> tuple[list[float], list[float]]:
    lam_min, lam_max = pano_band_limits(top_trim_deg, bottom_trim_deg)
    d_yaw = 360.0 / x_count
    headings = [((yaw0_deg % 360.0) + (i + 0.5) * d_yaw) % 360.0 for i in range(x_count)]

    if not equal_area:
        d_lam = (lam_max - lam_min) / y_count
        lats = [lam_min + (j + 0.5) * d_lam for j in range(y_count)]
    else:
        smin, smax = math.sin(lam_min), math.sin(lam_max)
        ds = (smax - smin) / y_count
        lats = [math.asin(smin + (j + 0.5) * ds) for j in range(y_count)]
    return headings, [_deg(lam) for lam in lats]


def tile_hfov_deg(
    x_count: int,
    y_count: int,
    top_trim_deg: float,
    bottom_trim_deg: float,
    *,
    size: tuple[int, int],
    equal_area: bool,
    overlap_frac: float,
) -> float:
    width, height = size
    aspect = width / height
    dlon = (360.0 / x_count) * (1.0 + max(0.0, overlap_frac))
    lam_min, lam_max = pano_band_limits(top_trim_deg, bottom_trim_deg)

    if not equal_area:
        dlam_deg = _deg((lam_max - lam_min) / y_count)
    else:
        smin, smax = math.sin(lam_min), math.sin(lam_max)
        ds = (smax - smin) / y_count
        dlam_max = 0.0
        for j in range(y_count):
            lo = max(-1.0, min(1.0, smin + j * ds))
            hi = max(-1.0, min(1.0, smin + (j + 1) * ds))
            dlam_max = max(dlam_max, math.asin(hi) - math.asin(lo))
        dlam_deg = _deg(dlam_max)

    h_for_v = _deg(2.0 * math.atan(aspect * math.tan(_rad(dlam_deg) / 2.0)))
    return min(120.0, max(10.0, max(dlon, h_for_v)))


def apply_grid_selection(items: list[ViewRequest], x_count: int, y_count: int, selection: str) -> list[ViewRequest]:
    if selection == "all":
        return items
    if selection == "middle_row":
        row = y_count // 2
        return items[row * x_count : (row + 1) * x_count]
    if selection == "first_row":
        return items[:x_count]
    if selection == "last_row":
        return items[(y_count - 1) * x_count : y_count * x_count]
    if selection:
        selected: list[ViewRequest] = []
        for token in selection.split(","):
            try:
                idx = int(token.strip())
            except Exception:
                continue
            if 0 <= idx < len(items):
                selected.append(items[idx])
        if selected:
            return selected
    return items


def build_views_for_point(point: SourcePoint, args: argparse.Namespace) -> list[ViewRequest]:
    width, height = args.size
    if args.grid_x <= 1 and args.grid_y <= 1:
        return [
            ViewRequest(
                source=point,
                heading_deg=(point.heading_deg + args.heading_offset_deg) % 360.0,
                pitch_deg=float(point.pitch_deg if point.pitch_deg is not None else args.pitch_deg),
                fov_deg=float(args.fov_deg),
                width=width,
                height=height,
                view_index=0,
            )
        ]

    yaw0 = float(args.grid_yaw0_deg) if args.grid_yaw0_deg is not None else float(point.heading_deg) - 180.0 / args.grid_x
    headings, pitches = pano_centers(
        args.grid_x,
        args.grid_y,
        args.grid_top_trim_deg,
        args.grid_bottom_trim_deg,
        equal_area=args.grid_equal_area,
        yaw0_deg=yaw0,
    )
    hfov = tile_hfov_deg(
        args.grid_x,
        args.grid_y,
        args.grid_top_trim_deg,
        args.grid_bottom_trim_deg,
        size=(width, height),
        equal_area=args.grid_equal_area,
        overlap_frac=args.grid_overlap_frac,
    )
    views: list[ViewRequest] = []
    view_index = 0
    for pitch in pitches:
        for heading in headings:
            views.append(
                ViewRequest(
                    source=point,
                    heading_deg=heading,
                    pitch_deg=max(-90.0, min(90.0, pitch)),
                    fov_deg=hfov,
                    width=width,
                    height=height,
                    view_index=view_index,
                )
            )
            view_index += 1
    return apply_grid_selection(views, args.grid_x, args.grid_y, args.grid_selection)


def resolve_provider(raw_provider: str) -> str:
    provider = str(raw_provider or "auto").strip().lower()
    if provider == "auto":
        provider = os.getenv("MAP_PROVIDER", "google").strip().lower()
    aliases = {
        "google_streetview": "google",
        "streetview": "google",
        "baidu_panorama": "baidu",
        "panorama": "baidu",
    }
    provider = aliases.get(provider, provider)
    if provider not in {"google", "baidu"}:
        raise RuntimeError(
            f"Provider {provider!r} cannot fetch FPV panorama images. "
            "Use --provider google or --provider baidu. Existing osm/amap providers are for maps/routes, "
            "not first-person street-view static images."
        )
    return provider


def resolve_api_key(provider: str, args: argparse.Namespace) -> str:
    if provider == "google":
        key = args.google_api_key or os.getenv("GOOGLE_STREETVIEW_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
        if not key:
            raise RuntimeError("Google Street View requires --google-api-key or GOOGLE_MAPS_API_KEY.")
        return key
    ak = args.baidu_ak or os.getenv("BAIDU_AK")
    if not ak:
        raise RuntimeError("Baidu panorama requires --baidu-ak or BAIDU_AK.")
    return ak


def load_source_points(args: argparse.Namespace) -> list[SourcePoint]:
    sources = [bool(args.trajectory_json), bool(args.locations_csv), args.lat is not None or args.lng is not None]
    if sum(1 for item in sources if item) != 1:
        raise RuntimeError("Provide exactly one input: --trajectory-json, --locations-csv, or --lat/--lng.")

    if args.trajectory_json:
        points = load_points_from_trajectory(Path(args.trajectory_json).expanduser())
    elif args.locations_csv:
        points = load_points_from_csv(Path(args.locations_csv).expanduser())
    else:
        if args.lat is None or args.lng is None:
            raise RuntimeError("Both --lat and --lng are required for single-point mode.")
        points = [
            SourcePoint(
                index=0,
                lat=float(args.lat),
                lng=float(args.lng),
                heading_deg=_heading(args.heading_deg, 0.0),
                pitch_deg=args.pitch_deg,
                pano_id=args.pano_id,
                frame_id=args.frame_id,
                timestamp=None,
                raw=None,
            )
        ]

    points = select_points(points, start=args.frame_start, end=args.frame_end, step=args.frame_step)
    if args.limit:
        points = points[: max(0, int(args.limit))]
    if not points:
        raise RuntimeError("No source points selected.")
    return points


def _safe_log_url(url: str, *, provider: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    secret_keys = {"key", "ak", "signature"}
    redacted = [(key, "<redacted>" if key.lower() in secret_keys else value) for key, value in query]
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(redacted)))


def _is_image_response(headers: dict[str, str], body: bytes) -> bool:
    content_type = headers.get("content-type", "").lower()
    if content_type.startswith("image/"):
        return True
    return body.startswith(b"\xff\xd8\xff") or body.startswith(b"\x89PNG")


def _frame_name(seq: int) -> str:
    return f"{seq:06d}"


def _metadata_sidecar_path(metadata_dir: Path, frame_name: str) -> Path:
    return metadata_dir / f"{frame_name}.json"


def fetch_dataset(args: argparse.Namespace) -> dict[str, Any]:
    provider = resolve_provider(args.provider)
    api_key = resolve_api_key(provider, args)
    points = load_source_points(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    trajectory_rows: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    sequence = 0
    for point in points:
        google_metadata: dict[str, Any] | None = None
        resolved_pano_id = point.pano_id
        resolved_lat = point.lat
        resolved_lng = point.lng

        if provider == "google" and args.google_metadata:
            metadata_url = google_metadata_url(point, api_key=api_key, radius_m=args.google_radius_m)
            try:
                status, _headers, google_metadata = fetch_json(
                    metadata_url,
                    timeout_s=args.timeout_s,
                    user_agent=args.user_agent,
                )
                if status == 200 and str(google_metadata.get("status") or "") == "OK":
                    resolved_pano_id = str(google_metadata.get("pano_id") or resolved_pano_id or "").strip() or None
                    loc = google_metadata.get("location") or {}
                    resolved_lat = float(loc.get("lat", resolved_lat))
                    resolved_lng = float(loc.get("lng", resolved_lng))
                elif not args.allow_missing_google_metadata:
                    failures.append(
                        {
                            "point_index": point.index,
                            "provider": provider,
                            "status": "metadata_failed",
                            "http_status": status,
                            "metadata": google_metadata,
                            "metadata_url": _safe_log_url(metadata_url, provider=provider),
                        }
                    )
                    continue
            except Exception as exc:
                if not args.allow_missing_google_metadata:
                    failures.append(
                        {
                            "point_index": point.index,
                            "provider": provider,
                            "status": "metadata_error",
                            "error": str(exc),
                        }
                    )
                    continue

        resolved_point = SourcePoint(
            index=point.index,
            lat=resolved_lat,
            lng=resolved_lng,
            heading_deg=point.heading_deg,
            pitch_deg=point.pitch_deg,
            pano_id=resolved_pano_id,
            frame_id=point.frame_id,
            timestamp=point.timestamp,
            raw=point.raw,
        )

        for view in build_views_for_point(resolved_point, args):
            sequence += 1
            frame_name = _frame_name(sequence)
            image_path = images_dir / f"{frame_name}.jpg"
            sidecar_path = _metadata_sidecar_path(metadata_dir, frame_name)

            if provider == "google":
                url = google_image_url(
                    view,
                    api_key=api_key,
                    pano_id=resolved_pano_id,
                    return_error_code=args.google_return_error_code,
                )
                signing_secret = args.google_signing_secret or os.getenv("GOOGLE_STREETVIEW_SIGNING_SECRET") or ""
                if args.use_signed_google_url and signing_secret:
                    fetch_url = sign_google_streetview_url(url, signing_secret)
                else:
                    fetch_url = url
            else:
                url = baidu_image_url(
                    view,
                    ak=api_key,
                    pano_id=resolved_pano_id,
                    coordtype=args.baidu_coordtype,
                )
                fetch_url = url

            row: dict[str, Any] = {
                "frame_name": frame_name,
                "source_point_index": point.index,
                "source_frame_id": point.frame_id,
                "provider": provider,
                "coordinate_system": args.coordinate_system,
                "lat": float(resolved_point.lat),
                "lng": float(resolved_point.lng),
                "pano_id": resolved_pano_id,
                "heading_deg": float(view.heading_deg % 360.0),
                "pitch_deg": float(view.pitch_deg),
                "fov_deg": float(view.fov_deg),
                "width": int(view.width),
                "height": int(view.height),
                "image_path": str(image_path),
                "metadata_path": str(sidecar_path),
                "request_url_without_secret": _safe_log_url(url, provider=provider),
                "timestamp": point.timestamp,
                "google_metadata": google_metadata,
            }

            try:
                status, headers, body = fetch_bytes(fetch_url, timeout_s=args.timeout_s, user_agent=args.user_agent)
                row["http_status"] = status
                row["response_content_type"] = headers.get("content-type", "")
                row["response_bytes"] = len(body)
                if status >= 400 or not _is_image_response(headers, body):
                    row["status"] = "image_failed"
                    row["response_preview"] = body[:300].decode("utf-8", errors="replace")
                    failures.append(row)
                    sidecar_path.write_text(json.dumps(row, indent=2, ensure_ascii=True), encoding="utf-8")
                    continue

                image_path.write_bytes(body)
                row["status"] = "ok"
                sidecar_path.write_text(json.dumps(row, indent=2, ensure_ascii=True), encoding="utf-8")

                trajectory_item = {
                    "timestamp": point.timestamp if point.timestamp is not None else sequence,
                    "lat": float(resolved_point.lat),
                    "lng": float(resolved_point.lng),
                    "heading_deg": float(view.heading_deg % 360.0),
                    "pitch_deg": float(view.pitch_deg),
                    "fov_deg": float(view.fov_deg),
                    "image_path": str(image_path),
                    "frame_name": frame_name,
                    "provider": provider,
                    "coordinate_system": args.coordinate_system,
                    "pano_id": resolved_pano_id,
                    "source_point_index": point.index,
                    "source_frame_id": point.frame_id,
                }
                trajectory_rows.append(trajectory_item)
                frame_records.append(
                    {
                        "frame_index": sequence,
                        "frame_name": frame_name,
                        "image_path": str(image_path),
                        "timestamp": trajectory_item["timestamp"],
                        "lat": trajectory_item["lat"],
                        "lng": trajectory_item["lng"],
                        "heading_deg": trajectory_item["heading_deg"],
                        "provider": provider,
                        "pano_id": resolved_pano_id,
                    }
                )
                manifest_rows.append(row)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
                failures.append(row)
                sidecar_path.write_text(json.dumps(row, indent=2, ensure_ascii=True), encoding="utf-8")

            if args.sleep_s > 0:
                time.sleep(float(args.sleep_s))

    trajectory_path = output_dir / "trajectory.json"
    frames_manifest_path = output_dir / "frames_manifest.json"
    streetview_manifest_path = output_dir / "streetview_manifest.json"
    failure_path = output_dir / "streetview_failures.json"

    trajectory_path.write_text(json.dumps(trajectory_rows, indent=2, ensure_ascii=True), encoding="utf-8")
    frames_manifest_path.write_text(json.dumps(frame_records, indent=2, ensure_ascii=True), encoding="utf-8")
    streetview_manifest_path.write_text(json.dumps(manifest_rows, indent=2, ensure_ascii=True), encoding="utf-8")
    failure_path.write_text(json.dumps(failures, indent=2, ensure_ascii=True), encoding="utf-8")

    return {
        "provider": provider,
        "source_points": len(points),
        "success_frames": len(trajectory_rows),
        "failed_frames": len(failures),
        "output_dir": str(output_dir),
        "images_dir": str(images_dir),
        "trajectory_json": str(trajectory_path),
        "frames_manifest": str(frames_manifest_path),
        "streetview_manifest": str(streetview_manifest_path),
        "failures_json": str(failure_path),
    }


def parse_size(raw: str) -> tuple[int, int]:
    text = str(raw or "").lower().replace("*", "x")
    try:
        w_raw, h_raw = text.split("x", 1)
        width = int(w_raw)
        height = int(h_raw)
    except Exception as exc:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT, e.g. 512x512") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size values must be positive")
    return width, height


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Google Street View or Baidu panorama static images into RouteServer trajectory format."
    )
    input_group = parser.add_argument_group("input")
    input_group.add_argument("--trajectory-json", help="Existing trajectory JSON with lat/lng/heading rows.")
    input_group.add_argument("--locations-csv", help="CSV with lat,lng,heading_deg[,pano_id,frame_id].")
    input_group.add_argument("--lat", type=float, help="Single-point latitude.")
    input_group.add_argument("--lng", type=float, help="Single-point longitude.")
    input_group.add_argument("--heading-deg", type=float, default=0.0, help="Single-point heading when --lat/--lng are used.")
    input_group.add_argument("--pano-id", help="Single-point pano id.")
    input_group.add_argument("--frame-id", help="Single-point frame id.")

    parser.add_argument("--output-dir", required=True, help="Output directory for images/ and manifests.")
    parser.add_argument(
        "--provider",
        default="auto",
        help="Image provider: google, baidu, or auto. auto reads MAP_PROVIDER and supports only google/baidu.",
    )
    parser.add_argument("--google-api-key", help="Google Maps API key. Defaults to GOOGLE_STREETVIEW_API_KEY or GOOGLE_MAPS_API_KEY.")
    parser.add_argument("--baidu-ak", help="Baidu Map AK. Defaults to BAIDU_AK.")
    parser.add_argument("--coordinate-system", default="WGS84", help="Coordinate system of input lat/lng, recorded in manifest.")
    parser.add_argument("--size", type=parse_size, default=(512, 512), help="Image size WIDTHxHEIGHT. Default: 512x512.")
    parser.add_argument("--pitch-deg", type=float, default=0.0, help="Default pitch for non-grid single-view requests.")
    parser.add_argument("--fov-deg", type=float, default=90.0, help="Default FOV for non-grid single-view requests.")
    parser.add_argument("--heading-offset-deg", type=float, default=0.0, help="Add offset to source heading for single-view requests.")
    parser.add_argument("--frame-start", type=int, help="Start source point index, inclusive.")
    parser.add_argument("--frame-end", type=int, help="End source point index, exclusive.")
    parser.add_argument("--frame-step", type=int, default=1, help="Source point step. Default: 1.")
    parser.add_argument("--limit", type=int, help="Maximum number of source points after slicing.")

    google = parser.add_argument_group("google")
    google.add_argument("--google-metadata", action=argparse.BooleanOptionalAction, default=True, help="Query Street View metadata before image download.")
    google.add_argument("--google-radius-m", type=int, default=50, help="Google metadata search radius for location requests.")
    google.add_argument("--allow-missing-google-metadata", action="store_true", help="Try image request even if metadata lookup fails.")
    google.add_argument("--google-return-error-code", action=argparse.BooleanOptionalAction, default=True, help="Ask Google to return HTTP errors instead of placeholder image.")
    google.add_argument("--use-signed-google-url", action="store_true", help="Sign Google Street View URL.")
    google.add_argument("--google-signing-secret", help="Google URL signing secret. Defaults to GOOGLE_STREETVIEW_SIGNING_SECRET.")

    baidu = parser.add_argument_group("baidu")
    baidu.add_argument("--baidu-coordtype", default="wgs84ll", help="Baidu panorama coordtype for location requests. Default: wgs84ll.")

    grid = parser.add_argument_group("pano grid")
    grid.add_argument("--grid-x", type=int, default=1, help="Number of horizontal pano tiles. 1 means single-view mode.")
    grid.add_argument("--grid-y", type=int, default=1, help="Number of vertical pano tiles. 1 means single-view mode.")
    grid.add_argument("--grid-top-trim-deg", type=float, default=15.0, help="Top polar trim for pano grid.")
    grid.add_argument("--grid-bottom-trim-deg", type=float, default=15.0, help="Bottom polar trim for pano grid.")
    grid.add_argument("--grid-equal-area", action="store_true", help="Use equal-area pitch rows for pano grid.")
    grid.add_argument("--grid-yaw0-deg", type=float, help="Absolute grid yaw start. Default: source heading.")
    grid.add_argument("--grid-overlap-frac", type=float, default=0.02, help="Horizontal grid overlap fraction.")
    grid.add_argument(
        "--grid-selection",
        default="all",
        help="Grid subset: all, middle_row, first_row, last_row, or comma-separated tile indices.",
    )

    net = parser.add_argument_group("network")
    net.add_argument("--timeout-s", type=float, default=20.0, help="HTTP timeout seconds.")
    net.add_argument("--sleep-s", type=float, default=0.0, help="Sleep between image requests.")
    net.add_argument("--user-agent", default="UrbanNavRouteServer/0.1 streetview-fetcher", help="HTTP User-Agent.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        summary = fetch_dataset(args)
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from None
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
