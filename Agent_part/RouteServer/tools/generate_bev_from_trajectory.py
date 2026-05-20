#!/usr/bin/env python3
"""Generate BEV maps from trajectory.json files."""

import sys
import os
from pathlib import Path

# Use compass coordinate system (default)
# Compass: 0 deg = north, 90 deg = east, clockwise-positive
os.environ["BEV_MAP_HEADING_CONVENTION"] = "compass"

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from PIL import Image

from frodobot_bev_enhanced import (
    render_bev_map,
    MAP_PROVIDERS,
    ProcessOptions,
)


def load_trajectory(trajectory_path: Path) -> List[Dict[str, Any]]:
    """Load trajectory from JSON file."""
    with open(trajectory_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_trajectory(
    trajectory: List[Dict[str, Any]],
    output_dir: Path,
    map_provider: str = "osm",
):
    """Process a trajectory and generate BEV maps."""
    bev_images_dir = output_dir / "bev_images"
    bev_images_dir.mkdir(parents=True, exist_ok=True)

    provider_config = MAP_PROVIDERS.get(map_provider, MAP_PROVIDERS["osm"])
    options = ProcessOptions(
        map_provider=map_provider,
        output_size=512,
        zoom=19,
        cache_dir=Path("/tmp/bev_cache"),
        draw_trajectory=True,
    )

    # Extract global trajectory (all GPS points in this ride)
    global_trajectory = [(frame["lat"], frame["lng"]) for frame in trajectory]

    samples = []

    for i, frame in enumerate(tqdm(trajectory, desc=f"Generating BEV for {output_dir.name}")):
        lat = frame["lat"]
        lng = frame["lng"]
        heading_deg = frame["heading_deg"]
        fpv_image_path = Path(frame["image_path"])

        # Generate BEV map with global trajectory
        try:
            bev_image = render_bev_map(
                lat=lat,
                lng=lng,
                heading_deg=heading_deg,
                navigation_path_gps=global_trajectory,
                options=options,
                provider_config=provider_config,
            )

            # Save BEV image
            bev_filename = f"{fpv_image_path.stem}_bev.jpg"
            bev_path = bev_images_dir / bev_filename
            bev_image.save(bev_path, quality=95)

            # Create sample entry
            sample = {
                "ride_id": output_dir.name,
                "source": "gps_only",
                "fpv_image_path": str(fpv_image_path),
                "bev_image_path": str(bev_path),
                "metadata": {
                    "lat": lat,
                    "lng": lng,
                    "timestamp": frame["timestamp"],
                    "heading_deg": heading_deg,
                },
                "instruction": "",
                "action": "",
                "plan": "",
                "think": "",
            }
            samples.append(sample)
        except Exception as e:
            print(f"  [WARN] Failed to generate BEV for frame {i}: {e}")

    return samples


def process_ride(
    ride_dir: Path,
    output_dir: Path,
    map_provider: str = "osm",
):
    """Process a single ride directory."""
    trajectory_path = ride_dir / "trajectory.json"
    if not trajectory_path.exists():
        raise RuntimeError(f"trajectory.json not found in {ride_dir}")

    print(f"Processing {ride_dir.name}...")

    trajectory = load_trajectory(trajectory_path)
    print(f"Loaded {len(trajectory)} frames")

    # Create output directory
    ride_output_dir = output_dir / ride_dir.name
    ride_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate BEV maps
    samples = process_trajectory(trajectory, ride_output_dir, map_provider)

    # Save samples.json
    samples_path = ride_output_dir / "samples.json"
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Done. Output saved to {ride_output_dir}")
    print(f"  Samples: {samples_path}")
    print(f"  BEV images: {ride_output_dir / 'bev_images'}")
    print(f"  Total samples: {len(samples)}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate BEV maps from trajectory.json")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default="/media/ubuntu/data1/sr_nav_output",
        help="Input directory with ride_* subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="/media/ubuntu/data1/sr_nav_final",
        help="Output directory for final dataset",
    )
    parser.add_argument(
        "--map-provider",
        type=str,
        default="osm",
        choices=["osm", "carto", "carto-dark", "osm-toner", "osm-terrain"],
        help="Map provider to use",
    )
    args = parser.parse_args()

    # Find all ride directories
    ride_dirs = sorted([d for d in args.input_dir.iterdir() if d.is_dir() and d.name.startswith("ride_")])

    if not ride_dirs:
        print(f"No ride directories found in {args.input_dir}")
        return

    print(f"Found {len(ride_dirs)} ride directories\n")

    all_samples = []
    for ride_dir in ride_dirs:
        try:
            samples = process_ride(ride_dir, args.output_dir, args.map_provider)
            all_samples.extend(samples)
            print()
        except Exception as e:
            print(f"Error processing {ride_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Save all samples combined
    if all_samples:
        combined_path = args.output_dir / "all_samples.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        print(f"Combined samples saved to {combined_path}")
        print(f"Total samples: {len(all_samples)}")


if __name__ == "__main__":
    main()
