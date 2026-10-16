"""Review the bounded Decision exceptions against ordered visual inputs.

Original proposals and groundings remain immutable. This tool emits a teacher
review overlay, not a formal FullCoT eligibility decision.
"""
import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from mcp_server.contracts.reasoning_v4 import (
    DecisionProposalV4, InteractionConstraintGroundingV4, LocalTraversabilityV4,
    NavigationContextGroundingV4, SceneGroundingV4,
)
from mcp_server.teacher.reasoning_v4_batch import VisionReasoningV4TeacherBackend
from tools.model_processing.fill_history_decisions import LABELS

SECTIONS = {
    'SceneGrounding': ('scene_grounding', SceneGroundingV4),
    'LocalTraversability': ('local_traversability', LocalTraversabilityV4),
    'NavigationContextGrounding': ('navigation_context_grounding', NavigationContextGroundingV4),
    'InteractionConstraintGrounding': ('interaction_constraint_grounding', InteractionConstraintGroundingV4),
}
SYSTEM = """Review navigation labels using the supplied chronological FPV history,
current anchor, current BEV, instruction, and original context. The previous
teacher note is a hypothesis to check, not ground truth. Future GT is unavailable.
Judge current claims at fpv.current; temporal claims use the ordered history.
An earlier obstacle or signal need not remain applicable now. Identify which
signal applies to the pedestrian/robot path. Camera motion alone is insufficient
to establish another actor's velocity. Preserve supported original statements.
Review exactly requested_sections. Each section result must contain:
verdict (consistent, corrected, insufficient_evidence, alignment_issue),
reason, evidence_frames (supplied image roles), and replacement (full section
JSON for corrected, null otherwise). Correct only the affected section, following
its supplied schema. Preserve its other fields. Use insufficient_evidence for
unresolved claims; avoid fabricating certainty from the prior answer.
Return a Decision for the current anchor consistent with visible evidence and
reviewed context, or null if unresolved. Use supplied action_labels exclusively.
Negative angles are left, positive right; 50plus includes angles >=50 degrees,
so left_90 is represented by left_50plus. Stop uses stop/stop/stop; moving actions
use normal or slow speed and a moving response. Responses yield, detour and slow
require speed=slow. This is an output contract, not independent visual evidence.
Return JSON: {"decision": {...} or null, "sections": {section_name: {...}},
"decision_evidence_frames": [...], "decision_reason": "..."}.
"""


def read_rows(path):
    """Read JSONL rows from path; propagate input errors."""
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_cases(inputs, proposals):
    """Join only failed proposals or explicit context hints by stable ID.

    Args:
        inputs: Frozen visible inputs from the completed Decision run.
        proposals: Completed run outputs, including raw failed responses.
    Returns:
        Bounded review cases preserving original proposals and inputs.
    """
    by_id = {row['source_group_id']: row for row in inputs}
    cases = []
    for proposal in proposals:
        if proposal['status'] == 'generated' and not proposal['context_sections_to_review']:
            continue
        raw = json.loads(proposal['raw_answer'])
        sections = raw['context_sections_to_review']
        if set(sections) - SECTIONS.keys():
            raise ValueError(f'unknown context sections: {sections}')
        cases.append({'input': by_id[proposal['source_group_id']], 'original': proposal,
                      'requested_sections': sections})
    return cases


def validate_review(raw, case):
    """Check review coverage, concrete frame references and typed replacements.

    Args:
        raw: Teacher response text.
        case: Frozen review case.
    Returns:
        Parsed review and explicit resolved/pending status.
    Raises:
        ValueError: Invalid schema, evidence roles, or unrequested edits.
    """
    result = json.loads(raw)
    if set(result['sections']) != set(case['requested_sections']):
        raise ValueError('review must cover exactly requested sections')
    roles = set(case['input']['image_roles'])
    pending = result['decision'] is None
    for section, review in result['sections'].items():
        verdict = review['verdict']
        if verdict not in {'consistent', 'corrected', 'insufficient_evidence', 'alignment_issue'}:
            raise ValueError(f'unknown verdict: {verdict}')
        if not review['reason'] or set(review['evidence_frames']) - roles:
            raise ValueError('section evidence must use supplied frame roles')
        if verdict in {'consistent', 'corrected'} and not review['evidence_frames']:
            raise ValueError('resolved section requires visual evidence references')
        if verdict == 'corrected':
            SECTIONS[section][1].model_validate(review['replacement'])
        elif review['replacement'] is not None:
            raise ValueError('replacement requires corrected verdict')
        pending |= verdict in {'insufficient_evidence', 'alignment_issue'}
    if result['decision'] is not None:
        decision = DecisionProposalV4.model_validate(result['decision'])
        if decision.selected_action_label not in LABELS:
            raise ValueError('Decision must use ego-metric 12-way')
        if not result['decision_evidence_frames'] or set(result['decision_evidence_frames']) - roles:
            raise ValueError('Decision requires supplied evidence frames')
    if not result['decision_reason']:
        raise ValueError('Decision reason is required')
    return result, 'pending' if pending else 'resolved'


async def review_cases(base, source, output, workers):
    """Run only frozen exceptions; persist every attempt and resume by ID.

    Args:
        base: Shared suh root for relative media resolution.
        source: Completed Decision run directory.
        output: Separate review directory.
        workers: Concurrent requests to the four-GPU teacher.
    Raises:
        Exception: Provider/IO failure propagates, preserving flushed results.
    """
    cases = select_cases(read_rows(source/'inputs.jsonl'), read_rows(source/'decisions.jsonl'))
    output.mkdir(parents=True, exist_ok=True)
    plan = {'ids': [c['input']['source_group_id'] for c in cases], 'system': SYSTEM}
    plan_path = output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('frozen review plan changed')
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    target = output/'reviews.jsonl'
    done = {r['source_group_id'] for r in read_rows(target)} if target.exists() else set()
    queue = asyncio.Queue()
    for case in cases:
        if case['input']['source_group_id'] not in done:
            queue.put_nowait(case)
    for _ in range(workers):
        queue.put_nowait(None)
    with target.open('a', buffering=1) as stream:
        async def worker():
            backend = VisionReasoningV4TeacherBackend(api_key='local-vllm', media_root=base/'dataset/urbannav_history_4s_2hz_v1')
            while (case := await queue.get()) is not None:
                row = case['input']
                visible = {key: row[key] for key in ('instruction', 'grounding_outputs', 'raw_context')}
                visible.update(previous_answer=case['original']['raw_answer'],
                    requested_sections=case['requested_sections'], action_labels=LABELS,
                    section_schemas={s: SECTIONS[s][1].model_json_schema() for s in case['requested_sections']},
                    decision_schema=DecisionProposalV4.model_json_schema(),
                    image_order=[{'role': role, 'relative_time_sec': t} for role, t in
                                 zip(row['image_roles'], row['relative_times_sec'])])
                attempts = []
                # 仅对当前回答的合同错误重试一次，原始响应仍逐条留存。
                for attempt in range(2):
                    generation = await backend.generate(SimpleNamespace(images=row['images'], image_roles=row['image_roles']),
                        question=json.dumps(visible, ensure_ascii=False), system_prompt=SYSTEM, image_detail='low')
                    attempts.append({'raw_answer': generation.answer, 'telemetry': generation.telemetry,
                                     'model': generation.model})
                    try:
                        parsed, status = validate_review(generation.answer, case)
                        break
                    except (ValueError, KeyError, TypeError) as error:
                        attempts[-1]['validation_error'] = str(error)
                        parsed, status = None, 'validation_error'
                        visible['previous_validation_error'] = str(error)
                        visible['previous_review_response'] = generation.answer
                result = {'source_group_id': row['source_group_id'], 'status': status,
                          'review': parsed, 'attempts': attempts, 'original': case['original'],
                          'requested_sections': case['requested_sections']}
                stream.write(json.dumps(result, ensure_ascii=False)+'\n')
                done.add(row['source_group_id'])
                print(f'[review] {len(done)}/{len(cases)} {row["source_group_id"]} {status}', flush=True)
        async with asyncio.TaskGroup() as group:
            for _ in range(workers):
                group.create_task(worker())
    report = {'selected': len(cases), 'statuses': dict(Counter(r['status'] for r in read_rows(target))),
              'formal_training_exported': False}
    (output/'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    asyncio.run(review_cases(args.base, args.source, args.output, args.workers))
