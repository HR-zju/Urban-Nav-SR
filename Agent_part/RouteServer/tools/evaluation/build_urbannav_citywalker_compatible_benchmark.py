#!/usr/bin/env python3
"""Audit and materialize a public-only native CityWalker UrbanNav subset."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.benchmark.urbannav_citywalker_compatibility import (
    build_urbannav_citywalker_compatible_package_v1,
    urbannav_citywalker_native_default_profile_v1,
)


def main() -> int:
    """Parse source-package arguments and print a deterministic build report.

    Returns:
        Zero after writing the audit and optional immutable derived package.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--package-id",
                        default="urbannav_citywalker_native_openloop_v1")
    parser.add_argument("--profile-id",
                        default="urbannav_citywalker_native_openloop_v1")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    report = build_urbannav_citywalker_compatible_package_v1(
        source_package_json=args.source_package_json,
        output_root=args.output_root,
        package_id=args.package_id,
        profile=urbannav_citywalker_native_default_profile_v1(
            profile_id=args.profile_id),
        audit_only=args.audit_only,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
