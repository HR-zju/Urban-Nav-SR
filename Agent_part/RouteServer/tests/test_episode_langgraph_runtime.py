from __future__ import annotations

import unittest

from langgraph.checkpoint.memory import InMemorySaver

from mcp_server.contracts.episode import EpisodeEventType, EpisodeStatus, EpisodeTraceV2
from mcp_server.contracts.tool import ToolCallV2, ToolResponseV2
from mcp_server.runtime import (
    EnvironmentStepResult,
    EpisodeRuntime,
    EpisodeRuntimeConfig,
    ObservationResult,
    SafetyResult,
    System1Output,
    System2Action,
    System2ActionType,
    build_episode_graph,
)


class EpisodeLangGraphRuntimeTest(unittest.IsolatedAsyncioTestCase):
    """Closed-loop graph tests with deterministic policies and environment."""

    async def test_multi_tool_system1_safety_and_environment_loop(self) -> None:
        """One graph should execute multiple tools before an accepted System1 chunk."""

        policy = _SequenceSystem2Policy(
            [
                _tool_action("search_destination", call_id="call-1", query="coffee"),
                _tool_action("plan_route", call_id="call-2", destination_id="poi-1"),
                System2Action(action_type=System2ActionType.NAVIGATE),
            ]
        )
        graph = build_episode_graph(
            system2_policy=policy,
            system1_policy=_System1Policy(),
            tool_executor=_ToolExecutor(),
            safety_validator=_SafetyValidator([True]),
            environment=_Environment(arrival_after_steps=1),
            config=EpisodeRuntimeConfig(max_ticks=4, max_tool_calls=4),
        )

        state = await EpisodeRuntime(graph).run(episode_id="episode-tools", initial_context={"request": "coffee"})
        trace = EpisodeTraceV2.model_validate(state["episode_trace"])
        event_types = [event.event_type for event in trace.events]

        self.assertTrue(state["done"])
        self.assertEqual(state["stop_reason"], "arrival")
        self.assertEqual(event_types.count(EpisodeEventType.TOOL_CALL), 2)
        self.assertIn(EpisodeEventType.SYSTEM1_OUTPUT, event_types)
        self.assertIn(EpisodeEventType.SAFETY_VALIDATION, event_types)
        self.assertIn(EpisodeEventType.ENVIRONMENT_STEP, event_types)
        self.assertEqual(trace.status, EpisodeStatus.SUCCEEDED)

    async def test_safety_rejection_returns_to_system2_recovery(self) -> None:
        """Rejected chunks should be visible to System2 before a retry."""

        policy = _SequenceSystem2Policy(
            [
                System2Action(action_type=System2ActionType.NAVIGATE),
                System2Action(action_type=System2ActionType.NAVIGATE),
            ]
        )
        graph = build_episode_graph(
            system2_policy=policy,
            system1_policy=_System1Policy(),
            tool_executor=_ToolExecutor(),
            safety_validator=_SafetyValidator([False, True]),
            environment=_Environment(arrival_after_steps=1),
            config=EpisodeRuntimeConfig(max_safety_retries=2),
        )

        state = await EpisodeRuntime(graph).run(episode_id="episode-recovery", initial_context={})
        trace = EpisodeTraceV2.model_validate(state["episode_trace"])

        self.assertEqual(policy.calls, 2)
        self.assertEqual(
            sum(event.event_type == EpisodeEventType.SAFETY_VALIDATION for event in trace.events),
            2,
        )
        self.assertEqual(state["stop_reason"], "arrival")

    async def test_hitl_interrupt_resumes_same_checkpoint_thread(self) -> None:
        """User clarification should pause and resume through LangGraph Command."""

        policy = _HitlSystem2Policy()
        graph = build_episode_graph(
            system2_policy=policy,
            system1_policy=_System1Policy(),
            tool_executor=_ToolExecutor(),
            safety_validator=_SafetyValidator([True]),
            environment=_Environment(arrival_after_steps=1),
            checkpointer=InMemorySaver(),
        )
        runtime = EpisodeRuntime(graph)

        interrupted = await runtime.run(
            episode_id="episode-hitl",
            initial_context={},
            thread_id="thread-hitl",
        )
        resumed = await runtime.resume(thread_id="thread-hitl", user_response={"option_id": "entrance-a"})
        trace = EpisodeTraceV2.model_validate(resumed["episode_trace"])
        event_types = [event.event_type for event in trace.events]

        self.assertIn("__interrupt__", interrupted)
        self.assertIn(EpisodeEventType.USER_INTERRUPT, event_types)
        self.assertIn(EpisodeEventType.USER_RESUME, event_types)
        self.assertEqual(resumed["stop_reason"], "arrival")


class _SequenceSystem2Policy:
    def __init__(self, actions: list[System2Action]) -> None:
        self.actions = list(actions)
        self.calls = 0

    async def decide(self, state):
        action = self.actions[self.calls]
        self.calls += 1
        return action


class _HitlSystem2Policy:
    async def decide(self, state):
        if not state.get("user_response"):
            return System2Action(
                action_type=System2ActionType.ASK_USER,
                user_request={"question": "Which entrance?", "options": ["entrance-a", "entrance-b"]},
            )
        return System2Action(action_type=System2ActionType.NAVIGATE)


class _System1Policy:
    async def predict(self, state):
        return System1Output(trajectory_ego_t=[[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])


class _SafetyValidator:
    def __init__(self, decisions: list[bool]) -> None:
        self.decisions = list(decisions)

    async def validate(self, output, state):
        accepted = self.decisions.pop(0)
        return SafetyResult(accepted=accepted, reason="clear" if accepted else "blocked")


class _Environment:
    def __init__(self, *, arrival_after_steps: int) -> None:
        self.arrival_after_steps = arrival_after_steps
        self.steps = 0

    async def reset(self, initial_context):
        return ObservationResult(observation={"fpv": "frame-0.jpg"}, replay_key="obs-0")

    async def step(self, output, state):
        self.steps += 1
        done = self.steps >= self.arrival_after_steps
        return EnvironmentStepResult(
            observation={"fpv": f"frame-{self.steps}.jpg"},
            reward=1.0 if done else 0.0,
            done=done,
            info={"arrival_success": done, "stop_reason": "arrival" if done else None},
            replay_key=f"transition-{self.steps}",
        )


class _ToolExecutor:
    async def execute(self, call, state):
        return ToolResponseV2(
            tool=call.tool,
            op=call.op,
            success=True,
            result={"op": call.op, "candidate_id": "poi-1"},
            call_id=call.call_id,
            executor="test_replay",
            replay_key=f"replay:{call.call_id}",
        )


def _tool_action(op: str, *, call_id: str, **arguments) -> System2Action:
    return System2Action(
        action_type=System2ActionType.TOOL_CALL,
        tool_call=ToolCallV2(
            tool="map_tool",
            arguments={"op": op, **arguments},
            call_id=call_id,
        ),
    )


if __name__ == "__main__":
    unittest.main()
