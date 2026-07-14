from __future__ import annotations

import unittest

from mcp_server.agent.schemas import AgentTrace
from mcp_server.contracts.episode import EpisodeEventType, EpisodeStatus, EpisodeTraceV2
from mcp_server.contracts.trace_adapter import agent_trace_to_episode_trace


class EpisodeTraceV2Test(unittest.TestCase):
    """Tests for causal episode events and legacy trace conversion."""

    def test_append_builds_causal_chain(self) -> None:
        """Each appended event should point to the preceding event."""

        trace = EpisodeTraceV2(episode_id="episode-1")
        first = trace.append(EpisodeEventType.EPISODE_STARTED, tick_id=0, source="test")
        second = trace.append(
            EpisodeEventType.OBSERVATION_RECEIVED,
            tick_id=0,
            source="test",
            payload={"observation": {"frame": "fpv.jpg"}},
        )
        trace.finish(status=EpisodeStatus.SUCCEEDED, stop_reason="arrival", tick_id=0, source="test")

        self.assertEqual(second.parent_event_id, first.event_id)
        self.assertEqual(trace.status, EpisodeStatus.SUCCEEDED)
        self.assertEqual(trace.events[-1].event_type, EpisodeEventType.EPISODE_FINISHED)
        EpisodeTraceV2.model_validate(trace.model_dump(mode="json"))

    def test_legacy_tool_trace_converts_to_common_events(self) -> None:
        """Legacy Stage2 roles should map to the canonical event vocabulary."""

        trace = AgentTrace(case_id="case-1")
        trace.add_turn("user", "request", metadata={"stage2_step": "initial_request"})
        trace.add_turn("assistant", "decision")
        trace.add_turn("tool_call", {"name": "map_tool", "arguments": {"op": "search_poi"}})
        trace.add_turn("tool_response", {"tool": "map_tool", "success": True, "result": {}})
        trace.add_turn("user", "[SYS1_0]", metadata={"stage2_step": "trajectory_readout_prompt"})
        trace.add_turn("assistant", "<finish>{}</finish>")

        converted = agent_trace_to_episode_trace(trace)
        event_types = [event.event_type for event in converted.events]

        self.assertIn(EpisodeEventType.TOOL_CALL, event_types)
        self.assertIn(EpisodeEventType.TOOL_RESPONSE, event_types)
        self.assertIn(EpisodeEventType.READOUT_REQUESTED, event_types)
        self.assertEqual(converted.status, EpisodeStatus.SUCCEEDED)


if __name__ == "__main__":
    unittest.main()
