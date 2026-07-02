#!/usr/bin/env python3
"""Extract a routable OSM PBF bbox with pyosmium.

This is a data-preparation utility for local OSRM/Valhalla benchmarks. It does
not implement routing. It only crops a raw PBF into a smaller region PBF so that
the real routing backend can build a pedestrian graph.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import osmium


@dataclass(frozen=True)
class BBox:
    """Longitude/latitude bbox.

    Args:
        min_lng: Minimum longitude.
        min_lat: Minimum latitude.
        max_lng: Maximum longitude.
        max_lat: Maximum latitude.
    """

    min_lng: float
    min_lat: float
    max_lng: float
    max_lat: float

    def contains(self, *, lat: float, lng: float) -> bool:
        """Check whether a coordinate is inside the bbox.

        Args:
            lat: Latitude.
            lng: Longitude.

        Returns:
            ``True`` when the coordinate is inside the bbox.
        """

        return self.min_lat <= lat <= self.max_lat and self.min_lng <= lng <= self.max_lng


class CandidateWayCollector(osmium.SimpleHandler):
    """Collect ways that touch the bbox and all nodes needed by those ways.

    Args:
        bbox: Region bbox.
    """

    def __init__(self, bbox: BBox) -> None:
        super().__init__()
        self.bbox = bbox
        self.nodes_in_bbox: set[int] = set()
        self.selected_way_ids: set[int] = set()
        self.needed_node_ids: set[int] = set()

    def node(self, node: osmium.osm.Node) -> None:
        """Collect node ids inside the bbox.

        Args:
            node: OSM node.

        Returns:
            None.
        """

        if node.location.valid() and self.bbox.contains(lat=node.location.lat, lng=node.location.lon):
            self.nodes_in_bbox.add(int(node.id))

    def way(self, way: osmium.osm.Way) -> None:
        """Collect ways that include at least one bbox node.

        Args:
            way: OSM way.

        Returns:
            None.
        """

        refs = [int(node.ref) for node in way.nodes]
        if any(ref in self.nodes_in_bbox for ref in refs):
            self.selected_way_ids.add(int(way.id))
            self.needed_node_ids.update(refs)


class BBoxWriter(osmium.SimpleHandler):
    """Write selected nodes and ways to a PBF.

    Args:
        output: Output PBF path.
        needed_node_ids: Node ids referenced by selected ways.
        selected_way_ids: Way ids to write.
    """

    def __init__(self, output: Path, *, needed_node_ids: set[int], selected_way_ids: set[int]) -> None:
        super().__init__()
        self.writer = osmium.SimpleWriter(str(output))
        self.needed_node_ids = needed_node_ids
        self.selected_way_ids = selected_way_ids

    def close(self) -> None:
        """Close the underlying writer.

        Returns:
            None.
        """

        self.writer.close()

    def node(self, node: osmium.osm.Node) -> None:
        """Write required nodes.

        Args:
            node: OSM node.

        Returns:
            None.
        """

        if int(node.id) in self.needed_node_ids:
            self.writer.add_node(node)

    def way(self, way: osmium.osm.Way) -> None:
        """Write selected ways.

        Args:
            way: OSM way.

        Returns:
            None.
        """

        if int(way.id) in self.selected_way_ids:
            self.writer.add_way(way)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw source .osm.pbf.")
    parser.add_argument("--output", required=True, type=Path, help="Output cropped .osm.pbf.")
    parser.add_argument(
        "--bbox",
        required=True,
        help="Bounding box as min_lng,min_lat,max_lng,max_lat.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    return parser.parse_args()


def main() -> None:
    """Run bbox extraction.

    Returns:
        None.
    """

    args = parse_args()
    bbox = _parse_bbox(args.bbox)
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    collector = CandidateWayCollector(bbox)
    collector.apply_file(str(args.input), locations=True)
    if not collector.selected_way_ids:
        raise RuntimeError(f"No ways selected inside bbox {args.bbox} from {args.input}")

    writer = BBoxWriter(
        args.output,
        needed_node_ids=collector.needed_node_ids,
        selected_way_ids=collector.selected_way_ids,
    )
    try:
        writer.apply_file(str(args.input), locations=True)
    finally:
        writer.close()
    print(
        {
            "input": str(args.input),
            "output": str(args.output),
            "nodes": len(collector.needed_node_ids),
            "ways": len(collector.selected_way_ids),
        }
    )


def _parse_bbox(text: str) -> BBox:
    """Parse bbox text.

    Args:
        text: BBox as comma-separated numeric values.

    Returns:
        Parsed bbox.
    """

    values = [float(item) for item in str(text).split(",")]
    if len(values) != 4:
        raise ValueError(f"bbox must contain four values: {text}")
    return BBox(min_lng=values[0], min_lat=values[1], max_lng=values[2], max_lat=values[3])


if __name__ == "__main__":
    main()
