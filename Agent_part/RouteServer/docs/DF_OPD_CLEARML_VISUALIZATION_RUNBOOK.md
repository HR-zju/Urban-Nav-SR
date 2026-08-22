# DF-OPD ClearML visualization runbook

This runbook describes the CPU/offline evaluation bundle implemented in this
repository. It does not claim that a training process has online diagnostics.

## Ownership

- RouteServer owns canonical DF-OPD data contracts, offline evaluation, and the
  local review HTML.
- ResearchWorkflow owns the research_visualization_manifest_v1 lifecycle
  schema.
- The ms-swift ClearML reporter owns Task creation and post-run upload.
- ClearML Server stores the Task and selected artifacts; it does not define
  UrbanNav metric semantics.

## Run directory

A convention-based evaluation root uses:

    run/
    |-- report.md
    |-- report.html                # derived from report.md; safe ClearML report
    |-- student_review.html        # private GT/Student review; local only
    |-- report_manifest.json
    |-- scorecard.json
    |-- receipt.json
    |-- visualization_manifest.json
    |-- visualization_index.json
    |-- charts/
    |-- images/
    |-- videos/
    +-- media/

report.md remains the human-editable canonical document. report.html is a
derived browser artifact. The reporter command below writes it explicitly so
the same public HTML can be inspected locally and archived as a ClearML
artifact.

## Lifecycle

Create the planned visualization manifest before evaluation:

    python tools/evaluation/prepare_df_opd_student_visualization.py \
      --research-workflow-root /path/to/ResearchWorkflow \
      --out /path/to/run/visualization_manifest.json \
      --run-id urbannav-eval-001 \
      --run-root /path/to/run \
      --planned-output scorecard=/path/to/run/scorecard.json \
      --planned-output student_review_html=/path/to/run/student_review.html \
      --planned-output receipt=/path/to/run/receipt.json \
      --planned-output visualization_index=/path/to/run/visualization_index.json

Run the evaluator with convention-based outputs:

    python tools/evaluation/evaluate_df_opd_student.py \
      --canonical-jsonl /path/to/canonical.jsonl \
      --inference-jsonl /path/to/predictions.jsonl \
      --dataset-fingerprint DATA_SHA256 \
      --checkpoint-digest CHECKPOINT_SHA256 \
      --seed 7 \
      --experiment-root /path/to/run \
      --run-id urbannav-eval-001 \
      --build-report-manifest \
      --default-report-section 失败案例与人工校阅

The evaluator refuses a mismatched or missing planned manifest. With
``--experiment-root`` it also checks that the planned ``scorecard``,
``student_review_html``, ``receipt`` and ``visualization_index`` paths exactly
match the outputs it will write. It writes the scorecard, private Student
review HTML, receipt, and finished visualization index. The finished index
binds the scorecard and private Student review HTML by path, size, and SHA256.
The private review HTML is not an auto-discovered ClearML artifact.

## Media section mapping

build_report_manifest.py classifies media deterministically:

    charts/                 -> metrics
    images/success/         -> success_cases
    videos/failure/         -> failure_review
    images/review/          -> failure_review

Media with no semantic path component requires an explicit default section.
The fallback value must exactly match the corresponding ``##`` heading in
``report.md``; the example above uses the Chinese heading used by the template.
Existing report_manifest.json files are never replaced by the evaluator.

## ClearML upload

The existing reporter discovers the bundle without per-file flags. The
canonical human-edited ``report.md`` must already exist in the run root; the
reporter renders it to HTML for ClearML and does not invent or overwrite it:

    python /path/to/ms-swift/scripts/sidewalk_nav/clearml_report_eval.py \
      --profile /path/to/ResearchWorkflow/integrations/srnav-clearml/project-profile.yaml \
      --run-id urbannav-eval-001 \
      --experiment-root /path/to/run \
      --experiment-record-html /path/to/run/report.html

For the current local compose, the verified SDK endpoints are:

    API:   http://127.0.0.1:8008
    Files: http://127.0.0.1:8081
    Web:   http://127.0.0.1:8082

Credentials belong in a local ClearML config or environment and must not be
committed.

## Evidence status

Verified:

- 46 focused RouteServer tests pass.
- Planned and finished visualization manifests pass CPU smoke checks.
- The public Markdown renderer smoke inserted an image and a video after the
  matching Chinese ``##`` headings.
- ClearML reporter dry-run discovers the convention-based bundle and excludes
  the private ``student_review.html`` from automatic artifact selection.
- A real synthetic CPU Task was uploaded and completed:
  70441c0eb88e42a29d895ecf832f0c33.
- After planned-output binding was tightened, a second synthetic CPU Task was
  uploaded and completed:
  bc4d046176ce471481babae3b561757a.
- The current Student review HTML, including per-case ADE/FDE and their
  definitions, was uploaded and completed as Task
  71c23e2db572405599856c761658fec8.
- A privacy-boundary Task was uploaded and completed as
  ``afdf2dc38c3a4e19be004a4327733ef8``; its artifact list contains no
  ``student_review`` entry.
- Explicit ``--experiment-record-html`` persistence was verified with Task
  ``0ae1ae35e5b945c69f0086a024c2b976``; both ``report.html`` and the private
  ``student_review.html`` existed locally, while only the rendered public
  report artifact was archived.

Not verified by this runbook:

- real Qwen checkpoint intermediate-state diagnostics;
- online training metrics from ms-swift;
- GPU Agent or ClearML Pipeline scheduling;
- benchmark or training quality claims.
