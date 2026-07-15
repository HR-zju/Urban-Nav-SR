"""Shared semantic policies for UrbanNav benchmark collection."""

from __future__ import annotations

from typing import Any


def previous_target_types_for_update(row: dict[str, Any]) -> list[str]:
    """Infer the original destination types for a route-update instruction.

    Args:
        row: Route-update instruction row.

    Returns:
        Target types used to construct the pre-update route state.
    """

    subcategory = str(row.get("subcategory") or "")
    targets = {str(item) for item in row.get("target_types") or []}
    if subcategory.startswith("replace_destination_restaurant"):
        return ["pharmacy", "library"]
    if subcategory.startswith("replace_destination_pharmacy"):
        return ["restaurant", "cafe"]
    if subcategory.startswith("insert_before_final"):
        return ["library", "cafe"]
    if "parking" in targets or "parking_lot" in targets:
        return ["restaurant", "shopping_mall"]
    if "restaurant" in targets:
        return ["cafe", "library"]
    return ["restaurant", "cafe"]


def instruction_target_groups(row: dict[str, Any]) -> list[list[str]]:
    """Return ordered OR-groups of target types required by one instruction.

    Args:
        row: Compiled instruction row.

    Returns:
        A list of target groups. Values inside one group are alternatives;
        every group in the returned list must be supported by one collection
        hotspot.
    """

    category = str(row.get("category") or "")
    if category == "multi_stop_route":
        return [
            [str(item) for item in group if str(item).strip()]
            for group in row.get("ordered_target_types") or []
            if isinstance(group, list)
        ]
    targets = [str(item) for item in row.get("target_types") or [] if str(item).strip()]
    if category == "route_update_add_via":
        return [previous_target_types_for_update(row), targets]
    return [targets]
