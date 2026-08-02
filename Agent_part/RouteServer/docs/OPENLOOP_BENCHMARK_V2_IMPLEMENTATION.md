# UrbanNav SideWalk Open-loop Benchmark V2 implementation

This document maps the Stage I SFT and Stage II CFM benchmark design to the
implemented RouteServer and ms-swift entry points. It does not replace the
task/data contracts, and it does not claim closed-loop navigation quality.

## Ownership and trust boundary

RouteServer owns the canonical schemas, hidden references, package/run lineage,
metrics, aggregation, and reports. ms-swift owns target-free prompt projection,
checkpoint loading, model inference, and prediction/failure artifacts.

The exchange is split into four disjoint row types:

| Artifact | Model-visible | Contains hidden truth | Owner |
| --- | --- | --- | --- |
| `OpenLoopInputV1` | yes | no | RouteServer schema; ms-swift projector |
| `OpenLoopReferenceV1` | no | yes | RouteServer |
| `OpenLoopPredictionV1` | model output only | no | ms-swift |
| `OpenLoopFailureV1` | failure identity only | no | ms-swift |

The canonical contract is
`src/mcp_server/contracts/openloop_benchmark.py`. Generated JSON Schemas live in
`schemas/openloop_benchmark/`; ms-swift consumes the generated Python mirror and
the same schema artifacts. `contract_digest.json` is the cross-repository parity
gate.

## Frozen policies

`config/evaluation/stage_i_openloop_v1.json` and
`config/evaluation/stage_ii_cfm_openloop_v1.json` freeze:

- the reasoning and Decision renderer identities and source digests;
- schema/list/null rendering rules;
- `max_context_tokens=32768` and overflow `fail_closed`;
- one primary aggregation per scorecard;
- Stage II horizons `0.5/1.0/2.0 s` at `0.125 s` per step;
- the 50plus direction range policy and physical-motion N/A status.

Changing these values requires a new benchmark policy version. They are not
free runtime knobs for an already frozen package.

## Stage I flow

1. ms-swift validates a canonical Stage I exchange view and rebuilds its
   inference prompt with
   `swift.sidewalk_nav.data.stage_i_eval_manifest`. The output contains no
   assistant target, loss span, strong direction label, trajectory target, or
   annotation sidecar.
2. RouteServer builds the hidden reference with
   `tools/evaluation/build_openloop_references.py --stage stage_i`.
3. `tools/evaluation/build_stage_i_openloop_benchmark.py` validates exact
   input/reference identity and writes an immutable package manifest.
4. `swift.sidewalk_nav.cli.infer_stage_i_openloop` writes one prediction or
   failure for every attempted row. It keeps raw response text and disables
   legacy task parsing; strict canonical parsing belongs to the evaluator.
5. `tools/evaluation/evaluate_stage_i_openloop.py` validates package/run,
   checkpoint exposure, and per-output input digests before producing JSON,
   CSV, and self-contained HTML scorecards.

The FullCoT, VQA, and standalone Decision scorecards remain independent. The
FullCoT embedded action is diagnostic rather than a second direction headline.
VQA uses the complete reference denominator and reports deterministic/human and
LVLM-weak provenance separately. Decision is aggregated within each native
direction profile before profile macro aggregation. 50plus labels are ranges,
not exact 50-degree point targets.

## Stage II flow

1. RouteServer creates independent Pass-R and Pass-D cache records and the
   canonical Stage II exchange view. This is still the training-data owner
   boundary defined by `STAGE_II_CFM_DATA_CONTRACT.md`.
2. RouteServer emits hidden references and public eligible identities as soon
   as the metric trajectory target passes QC, before reading R/D caches.
   ms-swift joins those identities with the FullCoT/Decision inference prompts
   through `swift.sidewalk_nav.data.stage_ii_case_manifest`. These full-denominator
   cases are the canonical package inputs.
3. Stage I raw R/D outputs are reconciled by
   `swift.sidewalk_nav.data.stage_ii_prepass`. Missing, failed, or unparseable R/D
   outputs become canonical Stage II `pass_r`/`pass_d` failures; accepted outputs
   become raw records for the existing strict context-cache builder.
4. ms-swift projects only the R/D survivors to physical Pass-T rows with
   `swift.sidewalk_nav.data.stage_ii_eval_manifest`, but binds every survivor to
   its pre-cache benchmark case. Production requires predicted R/D context and
   equal R/D/T checkpoint digests. Frozen reference context is an explicitly
   named ablation only.
5. `swift.sidewalk_nav.cli.infer_stage_ii_cfm` writes exactly one prediction or
   failure ledger entry per parent. The run freezes the canonical input artifact
   digest separately from the physical adapter-manifest digest, plus R/D cache,
   normalization, flow-step, and seed identities.
6. `tools/evaluation/merge_openloop_failure_ledgers.py` rejects duplicate case
   identities while merging R/D and T failures. Then
   `tools/evaluation/evaluate_stage_ii_cfm_openloop.py` requires exactly 16
   finite FLU `[x_forward_m, y_left_m, yaw_rad]` points. Short or non-finite
   predictions are invalid at every horizon; reference masks alone determine
   horizon eligibility.

The formal Stage II output reports Geometry Coverage and conditional ADE/FDE at
each fixed horizon, moving-only yaw diagnostics, eligible dataset counts,
per-dataset metrics, three-pass reliability, and DecisionStopDrift. Collision,
route progress, physical FalseMove/FalseFreeze without a frozen policy, and
closed-loop success are N/A.

## Lineage and split safety

`tools/evaluation/build_checkpoint_exposure_registry.py` consumes an explicit
JSON mapping for every train/replay manifest. Field names are never guessed.
The evaluator rejects a checkpoint when any evaluation `parent_sample_id` or
`split_group_id` appears in its lineage registry.

`tools/evaluation/build_openloop_run_manifest.py` validates or assembles a run
manifest containing both repository states, checkpoint/tokenizer/processor
identity, benchmark digest, renderer/metric policies, batch/precision, and
Stage II context identity. Dirty repository states require a captured diff
digest.

Predictions and failures carry the digest of their exact canonical input row.
The formal evaluator compares it with the frozen package before semantic
scoring. Missing outputs remain in the full reference denominator; extra and
duplicate identities fail closed.

## Current validation level

Focused contract, projector, parser, metric, failure-ledger, exposure, and
cross-repository parity tests are implemented. A CPU data dry-run may validate
real view projection, package construction, full-reference denominators, and
report export, but it is not model-quality evidence. Formal release still
requires a frozen validation/test package, a real checkpoint exposure registry,
actual Stage I generation, predicted Pass-R/Pass-D caches, Stage II Pass-T GPU
inference, and reviewed scorecards.
