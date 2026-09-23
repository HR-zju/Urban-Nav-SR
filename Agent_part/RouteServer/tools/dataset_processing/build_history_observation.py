"""Plan and materialize clip shards for the 4s/2Hz/nine-frame observation view."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mcp_server.data.history_corpus import materialize_history_corpus, plan_history_corpus, prepare_history_clip
from mcp_server.data.history_package import (
    archive_history_package, index_history_manifests, materialize_clip_plan, write_clip_plan,
)
from mcp_server.data.stage_ii.sources import load_stage_ii_trajectory_source_registry_v1


def main() -> None:
    """Parse explicit source paths and run one local plan or materialize step."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--dataset", choices=("sidewalk", "foxglove", "egowalk"), required=True)
    sources = plan.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source-manifests", type=Path, nargs="+", default=[])
    sources.add_argument("--source-index", type=Path)
    plan.add_argument("--sidecars", type=Path, nargs="+", default=[])
    plan.add_argument("--registry", type=Path, default=Path(__file__).resolve().parents[2]
                      / "config/stage_ii_cfm/trajectory_source_registry_v1.json")
    plan.add_argument("--source", type=Path, required=True)
    plan.add_argument("--video", type=Path, required=True)
    plan.add_argument("--clip-id", required=True)
    plan.add_argument("--max-lag-ns", type=int, required=True)
    plan.add_argument("--bev-index", type=Path)
    plan.add_argument("--max-bev-age-ns", type=int)
    plan.add_argument("--video-map", type=Path, help="Foxglove JSON: clip_id, video_ordinals, evidence")
    plan.add_argument("--max-samples", type=int)
    plan.add_argument("--output", type=Path, required=True)
    index = commands.add_parser("index", help="Scan source/GT once, then plan clips through --source-index")
    index.add_argument("--dataset", choices=("sidewalk", "foxglove", "egowalk"), required=True)
    index.add_argument("--source-manifests", type=Path, nargs="+", required=True)
    index.add_argument("--sidecars", type=Path, nargs="+", required=True)
    index.add_argument("--registry", type=Path, default=plan.get_default("registry"))
    index.add_argument("--output", type=Path, required=True)
    batch = commands.add_parser("plan-batch", help="Plan explicit clip jobs without decoding media")
    batch.add_argument("--dataset", choices=("sidewalk", "foxglove", "egowalk"), required=True)
    batch.add_argument("--source-index", type=Path, required=True)
    batch.add_argument("--jobs", type=Path, required=True, help="JSON list of clip_id/source/video and map paths")
    batch.add_argument("--registry", type=Path, default=plan.get_default("registry"))
    batch.add_argument("--max-lag-ns", type=int, required=True)
    batch.add_argument("--max-bev-age-ns", type=int)
    batch.add_argument("--fpv-bytes-per-frame", type=int)
    batch.add_argument("--bev-bytes-per-frame", type=int)
    batch.add_argument("--output", type=Path, required=True)
    media = commands.add_parser("materialize")
    media.add_argument("--plan", type=Path, required=True)
    media.add_argument("--output", type=Path, required=True)
    media.add_argument("--jpeg-quality", type=int, required=True)
    media.add_argument("--tar-output", type=Path, help="Optionally archive this completed clip shard")
    batch_media = commands.add_parser("materialize-batch", help="Materialize completed plan-batch clips sequentially")
    batch_media.add_argument("--plan", type=Path, required=True)
    batch_media.add_argument("--output", type=Path, required=True)
    batch_media.add_argument("--jpeg-quality", type=int, required=True)
    batch_media.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if args.command == "materialize-batch":
        result = materialize_history_corpus(args.plan, args.output, jpeg_quality=args.jpeg_quality, archive=args.archive)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "plan-batch":
        registry = load_stage_ii_trajectory_source_registry_v1(args.registry)
        spec = next(item for item in registry.datasets if item.dataset_source == args.dataset)
        result = plan_history_corpus(
            jobs=json.loads(args.jobs.read_text(encoding="utf-8")), source_index=args.source_index,
            spec=spec, registry=registry, output=args.output, max_lag_ns=args.max_lag_ns,
            max_bev_age_ns=args.max_bev_age_ns, fpv_bytes_per_frame=args.fpv_bytes_per_frame,
            bev_bytes_per_frame=args.bev_bytes_per_frame)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "index":
        registry = load_stage_ii_trajectory_source_registry_v1(args.registry)
        spec = next(item for item in registry.datasets if item.dataset_source == args.dataset)
        result = index_history_manifests(manifests=args.source_manifests, sidecar_paths=args.sidecars,
                                         spec=spec, output=args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "materialize":
        result = materialize_clip_plan(args.plan, args.output, jpeg_quality=args.jpeg_quality)
        if args.tar_output is not None:
            result['archive'] = archive_history_package(args.output, args.tar_output)
    else:
        if args.source_index is None and not args.sidecars:
            parser.error("Direct manifest planning requires --sidecars")
        registry = load_stage_ii_trajectory_source_registry_v1(args.registry)
        spec = next(item for item in registry.datasets if item.dataset_source == args.dataset)
        processor, bev = prepare_history_clip(
            dataset=args.dataset, clip_id=args.clip_id, source=args.source, max_lag_ns=args.max_lag_ns,
            bev_index=args.bev_index, max_bev_age_ns=args.max_bev_age_ns, video_map=args.video_map)
        result = write_clip_plan(
            manifests=args.source_manifests, sidecar_paths=args.sidecars,
            clip_id=args.clip_id, source=args.source, video=args.video, processor=processor,
            spec=spec, registry=registry, bev=bev, output=args.output, max_samples=args.max_samples,
            source_index=args.source_index)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
