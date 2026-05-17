#!/usr/bin/env python3
"""Test multiple map providers with a single location."""
from __future__ import annotations

import sys
from pathlib import Path

# Add the src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from frodobot_bev_enhanced import (
    MAP_PROVIDERS,
    ProcessOptions,
    render_bev_map,
)


def main():
    # Test location (from FrodoBot dataset)
    lat = 35.716705
    lng = 139.854324
    heading = 90.0

    output_dir = Path("/tmp/frodobot_multiple_providers")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testing {len(MAP_PROVIDERS)} map providers...")
    print(f"Location: lat={lat:.6f}, lng={lng:.6f}, heading={heading:.1f}°")

    for provider_key, provider_config in MAP_PROVIDERS.items():
        print(f"\nRendering with {provider_config['name']}...")

        options = ProcessOptions(
            map_provider=provider_key,
            output_size=640,
            zoom=19,
            draw_trajectory=False,
            add_attribution=True,
        )

        try:
            img = render_bev_map(
                lat=lat,
                lng=lng,
                heading_deg=heading,
                navigation_path_gps=None,
                options=options,
                provider_config=provider_config,
            )

            output_path = output_dir / f"bev_{provider_key}.jpg"
            img.save(output_path, quality=95)
            print(f"  Saved to: {output_path}")
        except Exception as e:
            print(f"  [ERROR] Failed: {e}")

    print(f"\n[DONE] Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
