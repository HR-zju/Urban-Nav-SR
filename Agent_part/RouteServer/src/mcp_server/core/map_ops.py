"""Deterministic map operations shared by HTTP, MCP, and Agent evaluators."""

from __future__ import annotations

from typing import Any


async def geocode(*, app_context: Any, address: str) -> list[dict[str, Any]]:
    """Resolve an address/place name through the active map provider.

    Args:
        app_context: RouteServer app context with ``osm_client`` and
            ``provider`` attributes.
        address: Free-form address, POI name, or landmark query.

    Returns:
        Provider geocode results.  Results with ``lat/lon`` get a normalized
        ``coordinates`` field for downstream tools and APP consumers.
    """

    results = await app_context.osm_client.geocode(address)
    for result in results:
        if "lat" in result and "lon" in result:
            result["coordinates"] = {
                "latitude": float(result["lat"]),
                "longitude": float(result["lon"]),
            }
    return results


async def reverse_geocode(*, app_context: Any, latitude: float, longitude: float) -> dict[str, Any]:
    """Resolve coordinates to provider address metadata."""

    return await app_context.osm_client.reverse_geocode(latitude, longitude)


async def search_poi(
    *,
    app_context: Any,
    location: str,
    keywords: str,
    radius: int = 1000,
    sortrule: str = "distance",
    limit: int = 10,
) -> dict[str, Any]:
    """Search provider POIs with semantic keywords near a location."""

    results = await app_context.osm_client.search_semantic_pois(
        location=location,
        keywords=keywords,
        radius=radius,
        sortrule=sortrule,
        limit=limit,
    )
    return {
        "provider": app_context.provider,
        "query": {
            "location": location,
            "keywords": keywords,
            "radius": radius,
            "sortrule": sortrule,
            "limit": limit,
        },
        "results": results,
        "count": len(results),
    }


async def get_route(
    *,
    app_context: Any,
    from_latitude: float,
    from_longitude: float,
    to_latitude: float,
    to_longitude: float,
    mode: str = "car",
    steps: bool = False,
    overview: str = "simplified",
    annotations: bool = False,
) -> dict[str, Any]:
    """Calculate route directions and return the existing compact API payload."""

    valid_modes = {"car", "bike", "foot"}
    if mode not in valid_modes:
        mode = "car"
    route_data = await app_context.osm_client.get_route(
        from_latitude,
        from_longitude,
        to_latitude,
        to_longitude,
        mode,
        steps=steps,
        overview=overview,
        annotations=annotations,
    )
    if "routes" not in route_data or not route_data["routes"]:
        raise RuntimeError("No route found")

    route = route_data["routes"][0]
    steps_list: list[dict[str, Any]] = []
    if "legs" in route:
        for leg in route["legs"]:
            for step in leg.get("steps", []):
                maneuver = step.get("maneuver", {})
                waypoint = maneuver.get("location")
                if not waypoint and step.get("geometry", {}).get("coordinates"):
                    waypoint = step["geometry"]["coordinates"][-1]
                steps_list.append(
                    {
                        "instruction": maneuver.get("instruction", ""),
                        "distance": step.get("distance"),
                        "duration": step.get("duration"),
                        "name": step.get("name", ""),
                        "waypoint": waypoint,
                    }
                )
    return {
        "summary": {
            "distance": route.get("distance"),
            "duration": route.get("duration"),
            "mode": mode,
        },
        "directions": steps_list,
        "geometry": route.get("geometry"),
        "waypoints": route_data.get("waypoints", []),
    }


async def search_category(
    *,
    app_context: Any,
    category: str,
    min_latitude: float,
    min_longitude: float,
    max_latitude: float,
    max_longitude: float,
    subcategories: list[str] | None = None,
) -> dict[str, Any]:
    """Search categorized map features inside a bounding box."""

    bbox = (min_longitude, min_latitude, max_longitude, max_latitude)
    features = await app_context.osm_client.search_features_by_category(bbox, category, subcategories)
    results: list[dict[str, Any]] = []
    for feature in features:
        tags = feature.get("tags", {})
        coords = {}
        if feature.get("type") == "node":
            coords = {"latitude": feature.get("lat"), "longitude": feature.get("lon")}
        elif "center" in feature:
            coords = {
                "latitude": feature.get("center", {}).get("lat"),
                "longitude": feature.get("center", {}).get("lon"),
            }
        if coords:
            results.append(
                {
                    "id": feature.get("id"),
                    "type": feature.get("type"),
                    "name": tags.get("name", "Unnamed"),
                    "coordinates": coords,
                    "category": category,
                    "subcategory": tags.get(category),
                    "tags": tags,
                }
            )
    return {
        "query": {
            "category": category,
            "subcategories": subcategories,
            "bbox": {
                "min_latitude": min_latitude,
                "min_longitude": min_longitude,
                "max_latitude": max_latitude,
                "max_longitude": max_longitude,
            },
        },
        "results": results,
        "count": len(results),
    }
