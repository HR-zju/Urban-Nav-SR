"""Re-annotate affected sections into a separate revision; preserve original labels."""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from mcp_server.contracts.reasoning_dataset_v4 import ReasoningAnnotationV4, ReasoningSourceV4
from mcp_server.teacher.reasoning_v4_batch import (
    ReasoningV4BatchConfig, VisionReasoningV4TeacherBackend, annotate_reasoning_source_v4,
)


def repair_sections(row: ReasoningAnnotationV4) -> set[str]:
    """Select the two audited failure classes, keeping historical QC advisory.

    Args:
        row: Original annotation, including rejected raw answers.
    Returns:
        Grounding sections to re-run; Decision is recomputed from their outputs.
    """
    sections = set()
    if row.stage_records['NavigationContextGrounding'].status != 'accepted':
        sections.add('NavigationContextGrounding')
    if 'candidate_construction_failed:strong_moving_action_conflicts_with_stop_constraint' in row.errors:
        sections.add('InteractionConstraintGrounding')
    return sections


async def main(args: argparse.Namespace) -> None:
    """Run bounded section repair, writing an append-only independent revision.

    Args:
        args: CLI source, annotation, media-root and output locations.
    Returns:
        None; writes the repair plan, revised annotations and summary report.
    Raises:
        ValueError: Identity/fingerprint mismatch or output overlapping original.
        Exception: Provider and output IO failures retain their original exception.
    """
    if args.output.resolve() == args.annotations.resolve():
        raise ValueError('repair output must be separate from original annotations')
    sources = {}
    for line in args.sources.open():
        source = ReasoningSourceV4.model_validate_json(line)
        sources[source.observation.source_group_id] = source
    selected = []
    for line in args.annotations.open():
        original = ReasoningAnnotationV4.model_validate_json(line)
        sections = repair_sections(original)
        if sections:
            source = sources[original.source_group_id]
            if original.source_fingerprint != source.source_fingerprint:
                raise ValueError(f'input changed: {original.source_group_id}')
            selected.append((source, original, sections))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan = [{'source_group_id': old.source_group_id, 'rerun': sorted(sections | {'Decision'})}
            for _, old, sections in selected]
    plan_path = args.output.with_suffix('.plan.json')
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('frozen repair plan changed')
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    print(json.dumps({'selected': len(selected), 'output': str(args.output)}), flush=True)
    if args.prepare_only:
        return
    completed = set()
    if args.output.exists():
        for line in args.output.open():
            completed.add(json.loads(line)['source_group_id'])
    queue = asyncio.Queue()
    for item in selected:
        if item[1].source_group_id not in completed:
            queue.put_nowait(item)
    for _ in range(args.workers):
        queue.put_nowait(None)
    fields = ('TaskGrounding', 'SceneGrounding', 'LocalTraversability',
              'NavigationContextGrounding', 'InteractionConstraintGrounding')
    config = ReasoningV4BatchConfig(max_retries=2, image_detail='low',
        navigation_context_image_detail='high',
        section_image_policy_id='reasoning_v4_section_images_routed_visual_v1',
        direction_registry_version='urbannav_direction_profile_registry_v2')
    # 每次只替换具名 section；保留旧 producer 记录，GT 和其余输出逐字复用。
    with args.output.open('a', buffering=1) as output:
        async def worker() -> None:
            backend = VisionReasoningV4TeacherBackend(api_key='local-vllm', media_root=args.media_root)
            while (item := await queue.get()) is not None:
                source, original, sections = item
                reused = {name: original for name in fields if name not in sections}
                repaired = await annotate_reasoning_source_v4(
                    source, backend=backend, config=config, reused_groundings=reused)
                repaired.metadata['section_repair'] = {
                    'original_annotations': str(args.annotations),
                    'original_source_fingerprint': original.source_fingerprint,
                    'rerun': sorted(sections | {'Decision'}),
                    'final_action_policy': 'pending_user_decision',
                }
                output.write(repaired.model_dump_json() + '\n')
                completed.add(repaired.source_group_id)
                print(f'[repair] {len(completed)}/{len(selected)} {repaired.source_group_id}', flush=True)
        await asyncio.gather(*(worker() for _ in range(args.workers)))
    counts = Counter()
    for line in args.output.open():
        row = ReasoningAnnotationV4.model_validate_json(line)
        counts['rows'] += 1
        counts['candidate_' + row.candidate_structure_record.status] += 1
        for name, record in row.stage_records.items():
            counts[name + ':' + record.status] += 1
    args.output.with_suffix('.report.json').write_text(json.dumps(dict(counts), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--media-root', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    asyncio.run(main(parser.parse_args()))
