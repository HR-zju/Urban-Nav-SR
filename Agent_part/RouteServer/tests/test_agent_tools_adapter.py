"""Focused contract test for the standalone AgentTools RouteServer adapter."""

from mcp_server.agent.schemas import AgentStage2Case, AgentTrace
from mcp_server.eval.agent_tools.urbannav_agent_tools_adapter import export_route_trace_to_training_row


def test_route_trace_exports_through_standalone_agent_tools() -> None:
    case = AgentStage2Case(
        case_id='adapter-fixture',
        user_request='go forward',
        observation={'images': {'fpv': 'front.jpg'}},
        tools=[{'name': 'look'}],
    )
    trace = AgentTrace.from_case(case)
    trace.add_turn('assistant', 'continue')

    row = export_route_trace_to_training_row(trace, case=case)

    assert row['agent_meta']['case_id'] == 'adapter-fixture'
    assert row['images'] == ['front.jpg']
    assert row['tools']
    assert len(row['messages']) == 3
