"""Ensure evaluation package moves retain stable legacy imports."""

from mcp_server.eval.agent_tools.episode.runner import FrozenEpisodeBenchmarkRunner
from mcp_server.eval.agent_tools.exporters import trace_to_ms_swift_stage2_row
from mcp_server.eval.agent_tools.runners import RuleRunner
from mcp_server.eval.agent_tools.stage2_validator import validate_stage2_trace
from mcp_server.eval.openloop.stage_i_openloop import evaluate_stage_i_openloop_v1
from mcp_server.eval.urbannav.urbannav_metrics import haversine_m
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavPoint

from mcp_server.eval.episode.runner import FrozenEpisodeBenchmarkRunner as LegacyEpisodeRunner
from mcp_server.eval.exporters import trace_to_ms_swift_stage2_row as legacy_trace_exporter
from mcp_server.eval.runners import RuleRunner as LegacyRuleRunner
from mcp_server.eval.stage2_validator import validate_stage2_trace as legacy_trace_validator
from mcp_server.eval.stage_i_openloop import evaluate_stage_i_openloop_v1 as legacy_openloop_evaluator
from mcp_server.eval.urbannav_metrics import haversine_m as legacy_haversine
from mcp_server.eval.urbannav_schema import UrbanNavPoint as LegacyUrbanNavPoint


def test_legacy_eval_imports_are_identity_compatible():
    assert FrozenEpisodeBenchmarkRunner is LegacyEpisodeRunner
    assert trace_to_ms_swift_stage2_row is legacy_trace_exporter
    assert RuleRunner is LegacyRuleRunner
    assert validate_stage2_trace is legacy_trace_validator
    assert evaluate_stage_i_openloop_v1 is legacy_openloop_evaluator
    assert haversine_m is legacy_haversine
    assert UrbanNavPoint is LegacyUrbanNavPoint
