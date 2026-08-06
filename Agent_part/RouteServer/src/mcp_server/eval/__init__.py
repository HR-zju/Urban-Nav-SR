"""Compatibility facade for the split evaluation packages.

New code should import from ``agent_tools``, ``urbannav`` or ``openloop``.
The facade remains intentionally stable for backend and legacy callers.
"""

from mcp_server.eval.agent_tools.cases import (
    iter_eval_cases,
    iter_stage2_cases,
    load_eval_cases,
    load_stage2_cases,
    write_eval_cases,
    write_stage2_cases,
)
from mcp_server.eval.agent_tools.exporters import (
    trace_to_ms_swift_row,
    trace_to_ms_swift_stage2_row,
    write_ms_swift_jsonl,
    write_ms_swift_stage2_jsonl,
    write_traces_jsonl,
)
from mcp_server.eval.agent_tools.metrics import compute_trace_metrics
from mcp_server.eval.agent_tools.online_agent_runner import (
    CallableModelAdapter,
    ModelAdapter,
    MsSwiftInferenceAdapter,
    MsSwiftInferenceConfig,
    OpenAICompatibleInferenceConfig,
    OpenAICompatibleStage2Adapter,
    OnlineAgentRunResult,
    OnlineAgentRunner,
    OnlineAgentRunnerConfig,
    build_initial_model_messages,
)
from mcp_server.eval.agent_tools.phase1_archive_adapter import (
    PHASE1_ARCHIVE_STAGE2_ADAPTER_VERSION,
    iter_phase1_archive_stage2_cases,
    load_phase1_archive_stage2_cases,
    phase1_archive_row_to_stage2_case,
    write_phase1_archive_stage2_cases,
)
from mcp_server.eval.agent_tools.runners import AgentRunner, RouteServerTeacherRunner, RuleRunner
from mcp_server.eval.agent_tools.stage2_validator import (
    Stage2ValidationResult,
    validate_ms_swift_stage2_row,
    validate_stage2_case,
    validate_stage2_trace,
)

_LAZY_EXPORTS = {
    "URBANNAV_MODULE1_BENCHMARK_VERSION": "mcp_server.eval.urbannav.urbannav_module1_benchmark",
    "UrbanNavModule1BenchmarkConfig": "mcp_server.eval.urbannav.urbannav_module1_benchmark",
    "UrbanNavModule1BenchmarkResult": "mcp_server.eval.urbannav.urbannav_module1_benchmark",
    "module1_prediction_contract": "mcp_server.eval.urbannav.urbannav_module1_benchmark",
    "run_module1_benchmark": "mcp_server.eval.urbannav.urbannav_module1_benchmark",
    "MODULE1_COMPARISON_VERSION": "mcp_server.eval.urbannav.urbannav_module1_compare",
    "Module1ComparisonConfig": "mcp_server.eval.urbannav.urbannav_module1_compare",
    "Module1RunSpec": "mcp_server.eval.urbannav.urbannav_module1_compare",
    "compare_module1_runs": "mcp_server.eval.urbannav.urbannav_module1_compare",
    "comparison_to_markdown": "mcp_server.eval.urbannav.urbannav_module1_compare",
}


def __getattr__(name: str):
    """Resolve heavy UrbanNav exports lazily.

    Args:
        name: Attribute requested from ``mcp_server.eval``.

    Returns:
        Exported object from the owning submodule.

    Raises:
        AttributeError: If the name is not a package export.
    """

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "AgentRunner",
    "RouteServerTeacherRunner",
    "RuleRunner",
    "PHASE1_ARCHIVE_STAGE2_ADAPTER_VERSION",
    "Stage2ValidationResult",
    "compute_trace_metrics",
    "URBANNAV_MODULE1_BENCHMARK_VERSION",
    "UrbanNavModule1BenchmarkConfig",
    "UrbanNavModule1BenchmarkResult",
    "module1_prediction_contract",
    "run_module1_benchmark",
    "MODULE1_COMPARISON_VERSION",
    "Module1ComparisonConfig",
    "Module1RunSpec",
    "compare_module1_runs",
    "comparison_to_markdown",
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
    "OpenAICompatibleInferenceConfig",
    "OpenAICompatibleStage2Adapter",
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
