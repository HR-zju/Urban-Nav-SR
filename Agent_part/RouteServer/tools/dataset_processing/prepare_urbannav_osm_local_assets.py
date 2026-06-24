#!/usr/bin/env python3
"""Prepare local OSM assets for reproducible UrbanNav global GT collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib.request import Request, urlopen


DEFAULT_GLOBAL_CONFIG = Path(__file__).resolve().parents[2] / "config" / "urbannav_global_gt_collection.json"
DEFAULT_SOURCE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "urbannav_osm_extract_sources.json"
DEFAULT_ROUTER_CONFIG = Path(__file__).resolve().parents[2] / "config" / "urbannav_osm_local_sources.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-config", type=Path, default=DEFAULT_GLOBAL_CONFIG, help="UrbanNav global region config.")
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG, help="OSM source URL config.")
    parser.add_argument("--output-root", type=Path, default=Path("data/osm"), help="Output root for raw/city/index/osrm assets.")
    parser.add_argument("--router-config-output", type=Path, default=DEFAULT_ROUTER_CONFIG, help="Router config JSON to write.")
    parser.add_argument("--bbox-buffer-m", type=float, default=2500.0, help="Additional bbox buffer around each region radius.")
    parser.add_argument("--download", action="store_true", help="Download raw PBF files.")
    parser.add_argument("--extract", action="store_true", help="Run osmium extract for city/region PBFs.")
    parser.add_argument("--build-poi-index", action="store_true", help="Build local POI JSONL indexes from extracted PBFs.")
    parser.add_argument("--write-osrm-scripts", action="store_true", help="Write OSRM build/run shell scripts.")
    parser.add_argument("--write-valhalla-scripts", action="store_true", help="Write Valhalla build/run shell scripts.")
    parser.add_argument("--write-router-config", action="store_true", help="Write osm_local router config JSON.")
    parser.add_argument("--router-type", choices=["osrm", "valhalla"], default="osrm", help="Router type written to generated config.")
    parser.add_argument("--osrm-profile", default="foot", help="OSRM profile name used by routed endpoints.")
    parser.add_argument("--osrm-profile-path", default="profiles/foot.lua", help="Path to OSRM profile for generated build scripts.")
    parser.add_argument("--valhalla-costing", default="pedestrian", help="Valhalla costing model written to generated config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated files when possible.")
    return parser.parse_args()


def main() -> None:
    """Prepare local OSM assets.

    Returns:
        None.
    """

    args = parse_args()
    global_config = _load_json(args.global_config)
    source_config = _load_json(args.source_config)
    plan = _build_asset_plan(global_config=global_config, source_config=source_config, output_root=args.output_root, buffer_m=args.bbox_buffer_m)

    _ensure_dirs(args.output_root)
    # Write deterministic plans and helper scripts before expensive live steps.
    # This keeps the operator-visible next commands available even when a
    # machine is missing osmium/OSRM/Valhalla dependencies.
    if args.write_osrm_scripts:
        _write_osrm_scripts(plan=plan, profile_path=args.osrm_profile_path, profile=args.osrm_profile, overwrite=args.overwrite)
    if args.write_valhalla_scripts:
        _write_valhalla_scripts(plan=plan, costing=args.valhalla_costing, overwrite=args.overwrite)
    if args.write_router_config:
        _write_router_config(
            plan=plan,
            output=args.router_config_output,
            router_type=args.router_type,
            osrm_profile=args.osrm_profile,
            valhalla_costing=args.valhalla_costing,
            overwrite=args.overwrite,
        )

    _write_plan(plan=plan, output_root=args.output_root, overwrite=True)
    if args.download:
        _download_raw_sources(plan=plan, overwrite=args.overwrite)
    if args.extract:
        _run_extracts(plan=plan, overwrite=args.overwrite)
    if args.build_poi_index:
        _build_poi_indexes(plan=plan, overwrite=args.overwrite)
    print(json.dumps(_plan_summary(plan), ensure_ascii=False, indent=2))


def _build_asset_plan(*, global_config: dict[str, Any], source_config: dict[str, Any], output_root: Path, buffer_m: float) -> list[dict[str, Any]]:
    """Build the deterministic local OSM asset plan.

    Args:
        global_config: UrbanNav global benchmark config.
        source_config: OSM extract source config.
        output_root: Root directory for generated assets.
        buffer_m: Extra meters added to each region radius.

    Returns:
        One asset-plan dictionary per configured region.
    """

    source_map = source_config.get("sources") or {}
    plan: list[dict[str, Any]] = []
    for region in global_config.get("regions", []):
        region_id = str(region["region_id"])
        source = source_map.get(region_id)
        if not source:
            raise ValueError(f"Missing OSM source config for region {region_id}")
        bbox = _bbox_for_region(region=region, buffer_m=buffer_m)
        raw_path = output_root / "raw" / str(source["raw_filename"])
        city_pbf_path = output_root / "cities" / f"{region_id}.osm.pbf"
        poi_index_path = output_root / "index" / f"{region_id}.jsonl"
        osrm_base = output_root / "osrm" / region_id / region_id
        plan.append(
            {
                "region_id": region_id,
                "country": region.get("country"),
                "city": region.get("city"),
                "label": region.get("label"),
                "locale": region.get("locale"),
                "center": {"lat": region.get("lat"), "lng": region.get("lng")},
                "bbox": bbox,
                "source_url": source["url"],
                "raw_pbf": str(raw_path),
                "city_pbf": str(city_pbf_path),
                "poi_index": str(poi_index_path),
                "osrm_base": str(osrm_base),
                "osrm_url": f"http://127.0.0.1:{int(source.get('osrm_port', 5000))}",
                "osrm_port": int(source.get("osrm_port", 5000)),
                "valhalla_url": f"http://127.0.0.1:{int(source.get('valhalla_port', int(source.get('osrm_port', 5000)) + 1000))}",
                "valhalla_port": int(source.get("valhalla_port", int(source.get("osrm_port", 5000)) + 1000)),
            }
        )
    return plan


def _download_raw_sources(*, plan: list[dict[str, Any]], overwrite: bool) -> None:
    """Download each raw OSM PBF source once.

    Args:
        plan: Asset plan.
        overwrite: Whether to overwrite existing raw PBF files.

    Returns:
        None.
    """

    seen: set[str] = set()
    for item in plan:
        raw_path = Path(item["raw_pbf"])
        if str(raw_path) in seen:
            continue
        seen.add(str(raw_path))
        if raw_path.exists() and not overwrite:
            continue
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        _download_file(url=str(item["source_url"]), output=raw_path)
        item["raw_pbf_sha256"] = _sha256(raw_path)


def _run_extracts(*, plan: list[dict[str, Any]], overwrite: bool) -> None:
    """Run ``osmium extract`` for each city/region.

    Args:
        plan: Asset plan.
        overwrite: Whether to overwrite extracted city PBFs.

    Returns:
        None.
    """

    osmium = shutil.which("osmium")
    if not osmium:
        raise RuntimeError("osmium is required for --extract. Install osmium-tool or run generated commands manually.")
    for item in plan:
        output = Path(item["city_pbf"])
        if output.exists() and not overwrite:
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        bbox = ",".join(f"{value:.7f}" for value in item["bbox"])
        cmd = [osmium, "extract", "--bbox", bbox, "-o", str(output), str(item["raw_pbf"])]
        _run(cmd)


def _build_poi_indexes(*, plan: list[dict[str, Any]], overwrite: bool) -> None:
    """Build JSONL POI indexes from extracted city PBFs.

    Args:
        plan: Asset plan.
        overwrite: Whether to overwrite existing POI indexes.

    Returns:
        None.
    """

    script = Path(__file__).resolve().with_name("build_osm_poi_index.py")
    for item in plan:
        output = Path(item["poi_index"])
        if output.exists() and not overwrite:
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        input_pbf = Path(item["city_pbf"]) if Path(item["city_pbf"]).exists() else Path(item["raw_pbf"])
        cmd = [
            sys.executable,
            str(script),
            "--input-pbf",
            str(input_pbf),
            "--output-jsonl",
            str(output),
            "--country",
            str(item["country"]),
            "--city",
            str(item["city"]),
            "--region-id",
            str(item["region_id"]),
        ]
        if not Path(item["city_pbf"]).exists():
            cmd.append("--bbox=" + ",".join(str(value) for value in item["bbox"]))
        if overwrite:
            cmd.append("--overwrite")
        _run(cmd)


def _write_osrm_scripts(*, plan: list[dict[str, Any]], profile_path: str, profile: str, overwrite: bool) -> None:
    """Write shell scripts for building and serving OSRM graphs.

    Args:
        plan: Asset plan.
        profile_path: OSRM Lua profile path.
        profile: OSRM route profile name.
        overwrite: Whether to overwrite generated scripts.

    Returns:
        None.
    """

    script_dir = Path(plan[0]["osrm_base"]).parents[1] if plan else Path("data/osm/osrm")
    script_dir.mkdir(parents=True, exist_ok=True)
    build_script = script_dir / "build_osrm_all.sh"
    run_script = script_dir / "run_osrm_all.sh"
    if build_script.exists() and not overwrite:
        raise FileExistsError(f"{build_script} exists; pass --overwrite")
    if run_script.exists() and not overwrite:
        raise FileExistsError(f"{run_script} exists; pass --overwrite")

    build_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    run_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for item in plan:
        city_pbf = shlex.quote(str(item["city_pbf"]))
        osrm_base = shlex.quote(str(item["osrm_base"]))
        osrm_dir = shlex.quote(str(Path(item["osrm_base"]).parent))
        profile_arg = shlex.quote(profile_path)
        build_lines.extend(
            [
                f"mkdir -p {osrm_dir}",
                f"osrm-extract -p {profile_arg} -o {osrm_base}.osrm {city_pbf}",
                f"osrm-partition {osrm_base}.osrm",
                f"osrm-customize {osrm_base}.osrm",
                "",
            ]
        )
        run_lines.append(f"osrm-routed --algorithm mld --port {int(item['osrm_port'])} {osrm_base}.osrm &")
    run_lines.append("wait")
    build_script.write_text("\n".join(build_lines) + "\n", encoding="utf-8")
    run_script.write_text("\n".join(run_lines) + "\n", encoding="utf-8")
    build_script.chmod(0o755)
    run_script.chmod(0o755)


def _write_valhalla_scripts(*, plan: list[dict[str, Any]], costing: str, overwrite: bool) -> None:
    """Write shell scripts for building and serving Valhalla tiles.

    Args:
        plan: Asset plan.
        costing: Valhalla costing model recorded in script comments.
        overwrite: Whether to overwrite generated scripts.

    Returns:
        None.
    """

    script_dir = Path(plan[0]["osrm_base"]).parents[1] if plan else Path("data/osm/osrm")
    script_dir.mkdir(parents=True, exist_ok=True)
    build_script = script_dir / "build_valhalla_all.sh"
    run_script = script_dir / "run_valhalla_all.sh"
    if build_script.exists() and not overwrite:
        raise FileExistsError(f"{build_script} exists; pass --overwrite")
    if run_script.exists() and not overwrite:
        raise FileExistsError(f"{run_script} exists; pass --overwrite")

    build_lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"# Valhalla costing used by UrbanNav: {costing}", ""]
    run_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for item in plan:
        city_pbf = shlex.quote(str(item["city_pbf"]))
        region_dir = Path(item["osrm_base"]).parent / "valhalla"
        config_path = region_dir / "valhalla.json"
        tile_dir = region_dir / "tiles"
        tile_tar = region_dir / "tiles.tar"
        region_dir_q = shlex.quote(str(region_dir))
        config_q = shlex.quote(str(config_path))
        tile_dir_q = shlex.quote(str(tile_dir))
        tile_tar_q = shlex.quote(str(tile_tar))
        port = int(item["valhalla_port"])
        build_lines.extend(
            [
                f"mkdir -p {region_dir_q} {tile_dir_q}",
                f"valhalla_build_config --mjolnir-tile-dir {tile_dir_q} --mjolnir-tile-extract {tile_tar_q} > {config_q}",
                f"python - {config_q} {port} <<'PY'",
                "import json, sys",
                "path = sys.argv[1]",
                "port = int(sys.argv[2])",
                "with open(path, 'r', encoding='utf-8') as handle:",
                "    config = json.load(handle)",
                "httpd = config.setdefault('httpd', {})",
                "service = httpd.setdefault('service', {})",
                "service['listen'] = '0.0.0.0'",
                "service['port'] = port",
                "with open(path, 'w', encoding='utf-8') as handle:",
                "    json.dump(config, handle, indent=2)",
                "    handle.write('\\n')",
                "PY",
                f"valhalla_build_tiles -c {config_q} {city_pbf}",
                "",
            ]
        )
        run_lines.append(f"valhalla_service {config_q} 1 &")
    run_lines.append("wait")
    build_script.write_text("\n".join(build_lines) + "\n", encoding="utf-8")
    run_script.write_text("\n".join(run_lines) + "\n", encoding="utf-8")
    build_script.chmod(0o755)
    run_script.chmod(0o755)


def _write_router_config(
    *,
    plan: list[dict[str, Any]],
    output: Path,
    router_type: str,
    osrm_profile: str,
    valhalla_costing: str,
    overwrite: bool,
) -> None:
    """Write local OSM provider router config.

    Args:
        plan: Asset plan.
        output: Config path.
        router_type: Default router type, ``osrm`` or ``valhalla``.
        osrm_profile: OSRM profile name.
        valhalla_costing: Valhalla costing model.
        overwrite: Whether to overwrite output.

    Returns:
        None.
    """

    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite")
    indexes = {item["region_id"]: str(Path(item["poi_index"]).name) for item in plan}
    if router_type == "valhalla":
        routers = {
            item["region_id"]: {"router_type": "valhalla", "valhalla_url": item["valhalla_url"], "costing": valhalla_costing}
            for item in plan
        }
        default_router = {"router_type": "valhalla", "valhalla_url": "http://127.0.0.1:8002", "costing": valhalla_costing}
    else:
        routers = {
            item["region_id"]: {"router_type": "osrm", "osrm_url": item["osrm_url"], "profile": osrm_profile}
            for item in plan
        }
        default_router = {"router_type": "osrm", "osrm_url": "http://127.0.0.1:5000", "profile": osrm_profile}
    payload = {
        "description": "Generated local OSM router config for UrbanNav.",
        "poi_index_root": "data/osm/index",
        "default_router_type": router_type,
        "default_osrm_url": "http://127.0.0.1:5000",
        "default_osrm_profile": osrm_profile,
        "default_valhalla_url": "http://127.0.0.1:8002",
        "default_valhalla_costing": valhalla_costing,
        "indexes": {"default": "all.jsonl", **indexes},
        "routers": {"default": default_router, **routers},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_plan(*, plan: list[dict[str, Any]], output_root: Path, overwrite: bool) -> None:
    """Write the asset plan JSON.

    Args:
        plan: Asset plan.
        output_root: Asset output root.
        overwrite: Whether to overwrite an existing plan.

    Returns:
        None.
    """

    output = output_root / "urbannav_osm_asset_plan.json"
    if output.exists() and not overwrite:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"regions": plan}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bbox_for_region(*, region: dict[str, Any], buffer_m: float) -> list[float]:
    """Compute a region bbox with an additional buffer.

    Args:
        region: Region config with center and radius.
        buffer_m: Extra buffer in meters.

    Returns:
        Bounding box as ``[min_lng, min_lat, max_lng, max_lat]``.
    """

    center_lat = float(region["lat"])
    center_lng = float(region["lng"])
    radius = float(region.get("radius_m", 900)) + max(0.0, buffer_m)
    lat_delta = radius / 111_320.0
    lng_delta = radius / (111_320.0 * max(0.2, math.cos(math.radians(center_lat))))
    return [
        round(center_lng - lng_delta, 7),
        round(center_lat - lat_delta, 7),
        round(center_lng + lng_delta, 7),
        round(center_lat + lat_delta, 7),
    ]


def _download_file(*, url: str, output: Path) -> None:
    """Download one URL to disk with bounded retries.

    Args:
        url: Download URL.
        output: Output file.

    Returns:
        None.
    """

    request = Request(url, headers={"User-Agent": "UrbanNavBenchmark/0.1"})
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    last_error: BaseException | None = None
    for attempt in range(1, 4):
        try:
            print(f"Downloading {url} -> {output} (attempt {attempt}/3)", flush=True)
            if tmp_output.exists():
                tmp_output.unlink()
            with urlopen(request, timeout=120) as response, tmp_output.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            _validate_pbf_file(tmp_output, url=url)
            tmp_output.replace(output)
            return
        except Exception as exc:  # pragma: no cover - live network failure path.
            last_error = exc
            if tmp_output.exists():
                tmp_output.unlink()
            if attempt < 3:
                time.sleep(float(attempt * 5))
    raise RuntimeError(f"Failed to download {url} after 3 attempts") from last_error


def _validate_pbf_file(path: Path, *, url: str) -> None:
    """Validate that a downloaded file looks like an OSM PBF.

    Args:
        path: Downloaded file path.
        url: Source URL, used in error messages.

    Returns:
        None.

    Raises:
        RuntimeError: If the file is empty or appears to be an HTML/error page.
    """

    if not path.exists() or path.stat().st_size < 32:
        raise RuntimeError(f"Downloaded OSM PBF is too small: {path} from {url}")
    with path.open("rb") as handle:
        head = handle.read(64)
    if b"OSMHeader" not in head:
        preview = head[:40].decode("utf-8", errors="replace").replace("\n", " ")
        raise RuntimeError(f"Downloaded file is not an OSM PBF: {path} from {url}; first bytes={preview!r}")


def _sha256(path: Path) -> str:
    """Compute a file SHA256 digest.

    Args:
        path: File path.

    Returns:
        Hex SHA256 digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(cmd: list[str]) -> None:
    """Run a subprocess command.

    Args:
        cmd: Command argv.

    Returns:
        None.
    """

    subprocess.run(cmd, check=True)


def _ensure_dirs(output_root: Path) -> None:
    """Create standard local OSM asset directories.

    Args:
        output_root: Asset output root.

    Returns:
        None.
    """

    for name in ("raw", "cities", "index", "osrm"):
        (output_root / name).mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object.

    Args:
        path: JSON path.

    Returns:
        Decoded JSON object.
    """

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _plan_summary(plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a compact asset plan summary.

    Args:
        plan: Asset plan.

    Returns:
        JSON-ready summary.
    """

    return {
        "region_count": len(plan),
        "raw_sources": sorted({item["raw_pbf"] for item in plan}),
        "regions": [
            {
                "region_id": item["region_id"],
                "bbox": item["bbox"],
                "raw_pbf": item["raw_pbf"],
                "city_pbf": item["city_pbf"],
                "poi_index": item["poi_index"],
                "osrm_url": item["osrm_url"],
            }
            for item in plan
        ],
    }


if __name__ == "__main__":
    main()
