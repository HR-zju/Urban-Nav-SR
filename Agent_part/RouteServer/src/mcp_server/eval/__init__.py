"""Evaluation helpers for SideWalk Agentic Stage2 traces."""

from mcp_server.eval.cases import (
    iter_eval_cases,
    iter_stage2_cases,
    load_eval_cases,
    load_stage2_cases,
    write_eval_cases,
    write_stage2_cases,
)
from mcp_server.eval.exporters import (
    trace_to_ms_swift_row,
    trace_to_ms_swift_stage2_row,
    write_ms_swift_jsonl,
    write_ms_swift_stage2_jsonl,
    write_traces_jsonl,
)
from mcp_server.eval.metrics import compute_trace_metrics
from mcp_server.eval.online_agent_runner import (
    CallableModelAdapter,
    ModelAdapter,
    MsSwiftInferenceAdapter,
    MsSwiftInferenceConfig,
    OnlineAgentRunResult,
    OnlineAgentRunner,
    OnlineAgentRunnerConfig,
    build_initial_model_messages,
)
from mcp_server.eval.phase1_archive_adapter import (
    PHASE1_ARCHIVE_STAGE2_ADAPTER_VERSION,
    iter_phase1_archive_stage2_cases,
    load_phase1_archive_stage2_cases,
    phase1_archive_row_to_stage2_case,
    write_phase1_archive_stage2_cases,
)
from mcp_server.eval.runners import AgentRunner, RouteServerTeacherRunner, RuleRunner
from mcp_server.eval.stage2_validator import (
    Stage2ValidationResult,
    validate_ms_swift_stage2_row,
    validate_stage2_case,
    validate_stage2_trace,
)

__all__ = [
    "AgentRunner",
    "RouteServerTeacherRunner",
    "RuleRunner",
    "PHASE1_ARCHIVE_STAGE2_ADAPTER_VERSION",
    "Stage2ValidationResult",
    "compute_trace_metrics",
    "CallableModelAdapter",
    "iter_eval_cases",
    "iter_phase1_archive_stage2_cases",
    "iter_stage2_cases",
    "load_eval_cases",
    "load_phase1_archive_stage2_cases",
    "load_stage2_cases",
    "ModelAdapter",
    "MsSwiftInferenceAdapter",
    "MsSwiftInferenceConfig",
    "OnlineAgentRunResult",
    "OnlineAgentRunner",
    "OnlineAgentRunnerConfig",
    "phase1_archive_row_to_stage2_case",
    "trace_to_ms_swift_row",
    "trace_to_ms_swift_stage2_row",
    "validate_ms_swift_stage2_row",
    "validate_stage2_case",
    "validate_stage2_trace",
    "write_eval_cases",
    "write_phase1_archive_stage2_cases",
    "write_stage2_cases",
    "write_ms_swift_jsonl",
    "write_ms_swift_stage2_jsonl",
    "write_traces_jsonl",
    "build_initial_model_messages",
]
