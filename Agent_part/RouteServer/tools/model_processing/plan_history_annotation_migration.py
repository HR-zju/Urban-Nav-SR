#!/usr/bin/env python3
"""Build portable sources, plan reuse, and explicitly run missing annotations."""

import argparse
import asyncio
import json
from pathlib import Path

from mcp_server.data.history_annotation_migration import (
    annotate_canonical_migration, build_migration_sources, export_canonical_migration, plan_canonical_migration,
)
from mcp_server.teacher.reasoning_v4_batch import VisionReasoningV4TeacherBackend
from mcp_server.data.history_vqa_migration import annotate_vqa_migration, export_vqa_migration, plan_vqa_migration
from mcp_server.teacher.vqa_v2_batch import VisionVQATeacherBackendV2


def main() -> None:
    """Read explicit old-run priority and write a new migration plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sources = commands.add_parser("sources")
    sources.add_argument("--dataset-root", type=Path, required=True)
    sources.add_argument("--dataset", choices=("egowalk", "sidewalk", "foxglove"), required=True)
    sources.add_argument("--output", type=Path, required=True)
    sources.add_argument("--limit", type=int)
    sources.add_argument("--bev-tolerance-ns", type=int,
                         help="Original generation budget for legacy packages missing BEV tolerance")
    plan = commands.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--input-root", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    vqa_plan = commands.add_parser("vqa-plan")
    vqa_plan.add_argument("--config", type=Path, required=True)
    vqa_plan.add_argument("--input-root", type=Path, required=True)
    vqa_plan.add_argument("--output", type=Path, required=True)
    vqa_annotate = commands.add_parser("vqa-annotate")
    vqa_annotate.add_argument("--run-root", type=Path, required=True)
    vqa_annotate.add_argument("--input-root", type=Path, required=True)
    vqa_annotate.add_argument("--dataset-root", type=Path, required=True)
    vqa_annotate.add_argument("--limit", type=int)
    vqa_annotate.add_argument("--offline-only", action="store_true")
    annotate = commands.add_parser("annotate")
    annotate.add_argument("--run-root", type=Path, required=True)
    annotate.add_argument("--input-root", type=Path, required=True)
    annotate.add_argument("--dataset-root", type=Path, required=True)
    annotate.add_argument("--limit", type=int)
    export = commands.add_parser("export")
    export.add_argument("--run-root", type=Path, required=True)
    export.add_argument("--modality-profile-id", required=True)
    vqa_export = commands.add_parser("vqa-export")
    vqa_export.add_argument("--run-root", type=Path, required=True)
    vqa_export.add_argument("--modality-profile-id", required=True)
    vqa_export.add_argument("--accepted-only", action="store_true")
    args = parser.parse_args()
    if args.command == "sources":
        report = build_migration_sources(args.dataset_root, args.dataset, args.output, limit=args.limit,
                                         bev_tolerance_ns=args.bev_tolerance_ns)
    elif args.command == "plan":
        report = plan_canonical_migration(json.loads(args.config.read_text()), args.output,
                                         input_root=args.input_root)
    elif args.command == "annotate":
        report = asyncio.run(annotate_canonical_migration(
            args.run_root, input_root=args.input_root, limit=args.limit,
            backend=VisionReasoningV4TeacherBackend(media_root=args.dataset_root),
        ))
    elif args.command == "vqa-plan":
        report = plan_vqa_migration(json.loads(args.config.read_text()), args.output, input_root=args.input_root)
    elif args.command == "vqa-annotate":
        report = asyncio.run(annotate_vqa_migration(
            args.run_root, input_root=args.input_root, limit=args.limit,
            backend=None if args.offline_only else VisionVQATeacherBackendV2(media_root=args.dataset_root),
            offline_only=args.offline_only,
        ))
    elif args.command == "vqa-export":
        report = export_vqa_migration(args.run_root, modality_profile_id=args.modality_profile_id,
                                      accepted_only=args.accepted_only)
    else:
        report = export_canonical_migration(args.run_root, modality_profile_id=args.modality_profile_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
