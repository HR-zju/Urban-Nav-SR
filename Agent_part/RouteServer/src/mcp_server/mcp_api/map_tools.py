"""MCP endpoints for map tools."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List
import math
from mcp.server.fastmcp import Context
from mcp_server.macos_location import get_current_location
from mcp_server.core.map_ops import geocode as core_geocode, get_route as core_get_route, reverse_geocode as core_reverse_geocode, search_category as core_search_category, search_poi as core_search_poi
from mcp_server.mcp_api.app import mcp

@mcp.tool()
async def get_current_location_macos(
    ctx: Context,
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    """
    Get the current device location from macOS CoreLocation.

    This tool is intended for local deployment on macOS so the MCP server can
    source the robot or laptop position directly from the host operating system.

    Args:
        timeout_s: Maximum seconds to wait for a location fix

    Returns:
        Current coordinates and accuracy metadata
    """
    await ctx.info(f"Requesting current macOS location with timeout={timeout_s:.1f}s")
    location = await asyncio.to_thread(get_current_location, timeout_s)
    return {
        "provider": "macos_corelocation",
        "location": f"{location['longitude']},{location['latitude']}",
        "coordinates": {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
        },
        "accuracy_m": location.get("accuracy_m"),
        "altitude_m": location.get("altitude_m"),
        "timestamp": location.get("timestamp"),
        "source": location.get("source"),
        "platform": location.get("platform"),
    }


@mcp.tool()
async def geocode_address(address: str, ctx: Context) -> List[Dict]:
    """
    Convert an address or place name to geographic coordinates with detailed location information.

    This tool takes a text description of a location (such as an address, landmark name, or
    place of interest) and returns its precise geographic coordinates along with rich metadata.
    The results can be used for mapping, navigation, location-based analysis, and as input to
    other geospatial tools.

    Args:
        address: The address, place name, landmark, or description to geocode (e.g., "Empire State Building",
                "123 Main St, Springfield", "Golden Gate Park, San Francisco")

    Returns:
        List of matching locations with:
        - Geographic coordinates (latitude/longitude)
        - Formatted address
        - Administrative boundaries (city, state, country)
        - OSM type and ID
        - Bounding box (if applicable)
        - Importance ranking
    """
    return await core_geocode(
        app_context=ctx.request_context.lifespan_context,
        address=address,
    )


@mcp.tool()
async def reverse_geocode(latitude: float, longitude: float, ctx: Context) -> Dict:
    """
    Convert geographic coordinates to a detailed address and location description.

    This tool takes a specific point on Earth (latitude and longitude) and returns
    comprehensive information about that location, including its address, nearby landmarks,
    administrative boundaries, and other contextual information. Useful for translating
    GPS coordinates into human-readable locations.

    Args:
        latitude: The latitude coordinate (decimal degrees, WGS84)
        longitude: The longitude coordinate (decimal degrees, WGS84)

    Returns:
        Detailed address and location information including:
        - Formatted address
        - Building, street, city, state, country
        - Administrative hierarchy
        - OSM metadata
        - Postal code and other relevant identifiers
    """
    return await core_reverse_geocode(
        app_context=ctx.request_context.lifespan_context,
        latitude=latitude,
        longitude=longitude,
    )


@mcp.tool()
async def search_nearby_pois(
    location: str,
    keywords: str,
    ctx: Context,
    radius: int = 1000,
    sortrule: str = "distance",
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search nearby POIs using semantic keywords against the active map provider.

    This tool is designed for domestic-map-first workflows where the LLM first
    converts user intent into provider-friendly search keywords such as
    "bookstore|cafe" or "convenience store|tea shop", then retrieves cleaned nearby candidates.

    Args:
        location: Current position in "longitude,latitude" format
        keywords: Provider-facing semantic keywords, optionally pipe-delimited
        radius: Search radius in meters
        sortrule: Sorting rule, typically "distance"
        limit: Maximum number of cleaned POI candidates to return

    Returns:
        Cleaned POI candidates with provider, coordinates, distance, type, and address
    """
    app_context = ctx.request_context.lifespan_context
    await ctx.info(f"Semantic POI search: provider={app_context.provider}, keywords={keywords}, radius={radius}m")
    return await core_search_poi(
        app_context=app_context,
        location=location,
        keywords=keywords,
        radius=radius,
        sortrule=sortrule,
        limit=limit,
    )


@mcp.tool()
async def find_nearby_places(
    latitude: float,
    longitude: float,
    ctx: Context,
    radius: float = 1000,  # meters
    categories: List[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Discover points of interest and amenities near a specific location.

    This tool performs a comprehensive search around a geographic point to identify
    nearby establishments, amenities, and points of interest. Results are organized by
    category and subcategory, making it easy to find specific types of places. Essential
    for location-based recommendations, neighborhood analysis, and proximity-based decision making.

    Args:
        latitude: Center point latitude (decimal degrees)
        longitude: Center point longitude (decimal degrees)
        radius: Search radius in meters (defaults to 1000m/1km)
        categories: List of OSM categories to search for (e.g., ["amenity", "shop", "tourism"]).
                   If omitted, searches common categories.
        limit: Maximum number of total results to return

    Returns:
        Structured dictionary containing:
        - Original query parameters
        - Total count of places found
        - Results grouped by category and subcategory
        - Each place includes name, coordinates, and associated tags
    """
    osm_client = ctx.request_context.lifespan_context.osm_client

    # Set default categories if not provided
    if not categories:
        categories = ["amenity", "shop", "tourism", "leisure"]

    await ctx.info(f"Searching for places within {radius}m of ({latitude}, {longitude})")
    places = await osm_client.get_nearby_pois(latitude, longitude, radius, categories)

    # Group results into category -> subcategory -> list[place] for easier LLM summarization.
    results_by_category = {}

    for place in places[:limit]:
        tags = place.get("tags", {})

        # Find the matching category
        for category in categories:
            if category in tags:
                subcategory = tags[category]
                if category not in results_by_category:
                    results_by_category[category] = {}

                if subcategory not in results_by_category[category]:
                    results_by_category[category][subcategory] = []

                # Add place to appropriate category and subcategory
                place_info = {
                    "id": place.get("id"),
                    "name": tags.get("name", "Unnamed"),
                    "latitude": place.get("lat"),
                    "longitude": place.get("lon"),
                    "tags": tags
                }

                results_by_category[category][subcategory].append(place_info)

    # Calculate total count
    total_count = sum(
        len(places)
        for category_data in results_by_category.values()
        for places in category_data.values()
    )

    return {
        "query": {
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius
        },
        "categories": results_by_category,
        "total_count": total_count
    }


@mcp.tool()
async def get_route_directions(
    from_latitude: float,
    from_longitude: float,
    to_latitude: float,
    to_longitude: float,
    ctx: Context,
    mode: str = "car",
    steps: bool = False,
    overview: str = "simplified",
    annotations: bool = False
) -> Dict[str, Any]:
    """
    Calculate detailed route directions between two geographic points.

    This tool provides comprehensive routing information between two locations using OpenStreetMap/OSRM.
    The output can be minimized using the steps, overview, and annotations parameters to reduce the response size.

    Args:
        from_latitude: Starting point latitude (decimal degrees)
        from_longitude: Starting point longitude (decimal degrees)
        to_latitude: Destination latitude (decimal degrees)
        to_longitude: Destination longitude (decimal degrees)
        ctx: Context (provided internally by MCP)
        mode: Transportation mode ("car", "bike", "foot")
        steps: Turn-by-turn instructions (True/False, Default: False)
        overview: Geometry output ("full", "simplified", "false"; Default: "simplified")
        annotations: Additional segment info (True/False, Default: False)

    Returns:
        Dictionary with routing information (summary, directions, geometry, waypoints)

    Example:
        {
          "from_latitude": 51.3334193,
          "from_longitude": 9.4540423,
          "to_latitude": 51.3295516,
          "to_longitude": 9.4576721,
          "mode": "car",
          "steps": false,
          "overview": "simplified",
          "annotations": false
        }
    """
    # Validate transportation mode
    valid_modes = ["car", "bike", "foot"]
    if mode not in valid_modes:
        await ctx.warning(f"Invalid mode '{mode}'. Using 'car' instead.")
        mode = "car"

    await ctx.info(f"Calculating {mode} route from ({from_latitude}, {from_longitude}) to ({to_latitude}, {to_longitude})")

    return await core_get_route(
        app_context=ctx.request_context.lifespan_context,
        from_latitude=from_latitude,
        from_longitude=from_longitude,
        to_latitude=to_latitude,
        to_longitude=to_longitude,
        mode=mode,
        steps=steps,
        overview=overview,
        annotations=annotations,
    )


@mcp.tool()
async def search_category(
    category: str,
    min_latitude: float,
    min_longitude: float,
    max_latitude: float,
    max_longitude: float,
    ctx: Context,
    subcategories: List[str] = None
) -> Dict[str, Any]:
    """
    Search for specific types of places within a defined geographic area.

    This tool allows targeted searches for places matching specific categories within
    a rectangular geographic region. It's particularly useful for filtering places by type
    (restaurants, schools, parks, etc.) within a neighborhood or city district. Results include
    complete location details and metadata about each matching place.

    Args:
        category: Main OSM category to search for (e.g., "amenity", "shop", "tourism", "building")
        min_latitude: Southern boundary of search area (decimal degrees)
        min_longitude: Western boundary of search area (decimal degrees)
        max_latitude: Northern boundary of search area (decimal degrees)
        max_longitude: Eastern boundary of search area (decimal degrees)
        subcategories: Optional list of specific subcategories to filter by (e.g., ["restaurant", "cafe"])

    Returns:
        Structured results including:
        - Query parameters
        - Count of matching places
        - List of matching places with coordinates, names, and metadata
    """
    await ctx.info(f"Searching for {category} in bounding box")
    return await core_search_category(
        app_context=ctx.request_context.lifespan_context,
        category=category,
        min_latitude=min_latitude,
        min_longitude=min_longitude,
        max_latitude=max_latitude,
        max_longitude=max_longitude,
        subcategories=subcategories,
    )


@mcp.tool()
async def suggest_meeting_point(
    locations: List[Dict[str, float]],
    ctx: Context,
    venue_type: str = "cafe"
) -> Dict[str, Any]:
    """
    Find the optimal meeting place for multiple people coming from different locations.

    This tool calculates a central meeting point based on the locations of multiple individuals,
    then recommends suitable venues near that central point. Ideal for planning social gatherings,
    business meetings, or any situation where multiple people need to converge from different
    starting points.

    Args:
        locations: List of dictionaries, each containing the latitude and longitude of a person's location
                  Example: [{"latitude": 37.7749, "longitude": -122.4194}, {"latitude": 37.3352, "longitude": -121.8811}]
        venue_type: Type of venue to suggest as a meeting point. Options include:
                   "cafe", "restaurant", "bar", "library", "park", etc.

    Returns:
        Meeting point recommendations including:
        - Calculated center point coordinates
        - List of suggested venues with names and details
        - Total number of matching venues in the area
    """
    osm_client = ctx.request_context.lifespan_context.osm_client

    if len(locations) < 2:
        raise ValueError("Need at least two locations to suggest a meeting point")

    # Calculate the center point (simple average)
    avg_lat = sum(loc.get("latitude", 0) for loc in locations) / len(locations)
    avg_lon = sum(loc.get("longitude", 0) for loc in locations) / len(locations)

    await ctx.info(f"Calculating center point for {len(locations)} locations: ({avg_lat}, {avg_lon})")

    # Search for venues around this center point
    venues = await osm_client.get_nearby_pois(
        avg_lat, avg_lon,
        radius=500,  # Search within 500m of center
        categories=["amenity"]
    )

    # Filter venues by type
    matching_venues = []
    for venue in venues:
        tags = venue.get("tags", {})
        if tags.get("amenity") == venue_type:
            matching_venues.append({
                "id": venue.get("id"),
                "name": tags.get("name", "Unnamed Venue"),
                "latitude": venue.get("lat"),
                "longitude": venue.get("lon"),
                "tags": tags
            })

    # If no venues found, expand search
    if not matching_venues:
        await ctx.info(f"No {venue_type} found within 500m, expanding search to 1000m")
        venues = await osm_client.get_nearby_pois(
            avg_lat, avg_lon,
            radius=1000,
            categories=["amenity"]
        )

        for venue in venues:
            tags = venue.get("tags", {})
            if tags.get("amenity") == venue_type:
                matching_venues.append({
                    "id": venue.get("id"),
                    "name": tags.get("name", "Unnamed Venue"),
                    "latitude": venue.get("lat"),
                    "longitude": venue.get("lon"),
                    "tags": tags
                })

    # Return the result
    return {
        "center_point": {
            "latitude": avg_lat,
            "longitude": avg_lon
        },
        "suggested_venues": matching_venues[:5],  # Top 5 venues
        "venue_type": venue_type,
        "total_options": len(matching_venues)
    }


@mcp.tool()
async def explore_area(
    latitude: float,
    longitude: float,
    ctx: Context,
    radius: float = 500
) -> Dict[str, Any]:
    """
    Generate a comprehensive profile of an area including all amenities and features.

    This powerful analysis tool creates a detailed overview of a neighborhood or area by
    identifying and categorizing all geographic features, amenities, and points of interest.
    Results are organized by category for easy analysis. Excellent for neighborhood research,
    area comparisons, and location-based decision making.

    Args:
        latitude: Center point latitude (decimal degrees)
        longitude: Center point longitude (decimal degrees)
        radius: Search radius in meters (defaults to 500m)

    Returns:
        In-depth area profile including:
        - Address and location context
        - Total feature count
        - Features organized by category and subcategory
        - Each feature includes name, coordinates, and detailed metadata
    """
    osm_client = ctx.request_context.lifespan_context.osm_client

    # Categories to search for
    categories = [
        "amenity", "shop", "tourism", "leisure",
        "natural", "historic", "public_transport"
    ]

    results = {}
    for i, category in enumerate(categories):
        await ctx.report_progress(i, len(categories))
        await ctx.info(f"Exploring {category} features...")

        try:
            # Convert radius to bounding box
            lat_delta = radius / 111000
            lon_delta = radius / (111000 * math.cos(math.radians(latitude)))

            bbox = (
                longitude - lon_delta,
                latitude - lat_delta,
                longitude + lon_delta,
                latitude + lat_delta
            )

            features = await osm_client.search_features_by_category(bbox, category)

            # Group by subcategory
            subcategories = {}
            for feature in features:
                tags = feature.get("tags", {})
                subcategory = tags.get(category)

                if subcategory:
                    if subcategory not in subcategories:
                        subcategories[subcategory] = []

                    # Get coordinates based on feature type
                    coords = {}
                    if feature.get("type") == "node":
                        coords = {
                            "latitude": feature.get("lat"),
                            "longitude": feature.get("lon")
                        }
                    elif "center" in feature:
                        coords = {
                            "latitude": feature.get("center", {}).get("lat"),
                            "longitude": feature.get("center", {}).get("lon")
                        }

                    subcategories[subcategory].append({
                        "id": feature.get("id"),
                        "name": tags.get("name", "Unnamed"),
                        "coordinates": coords,
                        "type": feature.get("type"),
                        "tags": tags
                    })

            results[category] = subcategories

        except Exception as e:
            await ctx.warning(f"Error fetching {category} features: {str(e)}")
            results[category] = {}

    # Get address information for the center point
    try:
        address_info = await osm_client.reverse_geocode(latitude, longitude)
    except Exception:
        address_info = {"error": "Could not retrieve address information"}

    # Report completion
    await ctx.report_progress(len(categories), len(categories))

    # Count total features
    total_features = sum(
        len(places)
        for category_data in results.values()
        for places in category_data.values()
    )

    return {
        "query": {
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius
        },
        "address": address_info,
        "categories": results,
        "total_features": total_features,
        "timestamp": datetime.now().isoformat()
    }
