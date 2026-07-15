from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mcp_server.api_http import create_app
from mcp_server.contracts.episode import EpisodeEventType, EpisodeStatus, EpisodeTraceV2
from mcp_server.contracts.tool import ToolResponseV2
from mcp_server.eval.episode import (
    FrozenEpisodeBenchmarkRunner,
    FrozenEpisodeCase,
    FrozenEpisodeTransition,
    FrozenExternalEpisodeService,
    compute_episode_metrics,
    load_frozen_episode_cases,
    write_frozen_episode_cases,
)
from mcp_server.runtime import SafetyResult, System1Output, System2Action, System2ActionType


class FrozenEpisodeBenchmarkTest(unittest.IsolatedAsyncioTestCase):
    """Offline deterministic episode benchmark tests."""

    async def test_runner_scores_arrival_efficiency_and_replay_digest(self) -> None:
        """Frozen transitions should produce stable main metrics and digest."""

        case = _case()
        runner = _runner([True, True])

        first = await runner.run_case(case)
        second = await runner.run_case(case)

        self.assertEqual(first.metrics.arrival_success, 1.0)
        self.assertEqual(first.metrics.collision_free_success, 1.0)
        self.assertEqual(first.metrics.normalized_efficiency, 1.0)
        self.assertEqual(first.metrics.total_reward, 1.0)
        self.assertEqual(first.replay_digest, second.replay_digest)

    async def test_recovery_metric_requires_rejection_then_arrival(self) -> None:
        """Recovery case should succeed after one rejected chunk and arrival."""

        case = FrozenEpisodeCase(
            episode_id="recovery-1",
            instruction="Avoid the blockage and continue.",
            initial_observation={"fpv": "frame-0.jpg"},
            initial_replay_key="obs-0",
            transitions=[
                FrozenEpisodeTransition(
                    replay_key="transition-0",
                    observation={"fpv": "frame-1.jpg"},
                    reward=1.0,
                    done=True,
                    info={"arrival_success": True, "stop_reason": "arrival"},
                    expected_action_shape=(2, 3),
                )
            ],
            optimal_steps=1,
            requires_recovery=True,
        )
        result = await _runner([False, True]).run_case(case)

        self.assertEqual(result.metrics.recovery_success, 1.0)
        self.assertEqual(result.metrics.diagnostics["safety_rejections"], 1)

    def test_case_jsonl_roundtrip(self) -> None:
        """Frozen case IO should preserve the strict Pydantic schema."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "episodes.jsonl"
            count = write_frozen_episode_cases(path, [_case()])
            loaded = load_frozen_episode_cases(path)

        self.assertEqual(count, 1)
        self.assertEqual(loaded, [_case()])

    def test_repository_smoke_case_is_valid(self) -> None:
        """The checked-in offline smoke manifest should satisfy the frozen schema."""

        path = Path(__file__).parents[1] / "examples" / "frozen_episode_v2_smoke.jsonl"
        cases = load_frozen_episode_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].transitions[0].expected_tool_op, "search_destination")

    def test_collision_free_success_requires_arrival(self) -> None:
        """A pre-execution failure is not a collision-free navigation success."""

        trace = EpisodeTraceV2(episode_id="episode-1")
        trace.append(EpisodeEventType.EPISODE_STARTED, tick_id=0, source="test", payload={})
        trace.finish(status=EpisodeStatus.FAILED, stop_reason="invalid_action", tick_id=0, source="test")

        metrics = compute_episode_metrics(trace, _case())

        self.assertEqual(metrics.arrival_success, 0.0)
        self.assertEqual(metrics.collision_free_success, 0.0)

    async def test_external_service_replays_typed_tool_and_finish_actions(self) -> None:
        """ms-swift environment actions should advance only matching transitions."""

        case = FrozenEpisodeCase(
            episode_id="external-1",
            instruction="Find the cafe and finish.",
            initial_observation={"prompt": "Find a cafe."},
            initial_replay_key="reset-1",
            transitions=[
                FrozenEpisodeTransition(
                    replay_key="tool-1",
                    observation={"tool_response": {"candidate_id": "poi-1"}},
                    reward=0.0,
                    done=False,
                    expected_action_type="tool_call",
                    expected_tool_op="search_destination",
                ),
                FrozenEpisodeTransition(
                    replay_key="finish-1",
                    observation={"prompt": "Episode complete."},
                    reward=1.0,
                    done=True,
                    info={"arrival_success": True, "reward_components": {"arrival": 1.0}},
                    expected_action_type="finish",
                ),
            ],
            optimal_steps=2,
        )
        service = FrozenExternalEpisodeService([case])

        reset = await service.reset({"episode": {"case_id": "external-1"}})
        tool = await service.step(
            "external-1",
            {"action": {"role": "assistant", "content": '<tool_call>{"name":"map_tool","arguments":{"op":"search_destination","query":"cafe"}}</tool_call>'}},
        )
        finish = await service.step(
            "external-1",
            {"action": {"role": "assistant", "content": '<finish>{"route_id":"route-1"}</finish>'}},
        )

        self.assertEqual(reset.replay_key, "reset-1")
        self.assertEqual(tool.replay_key, "tool-1")
        self.assertFalse(tool.done)
        self.assertTrue(finish.done)
        self.assertEqual(finish.reward_components["arrival"], 1.0)

    def test_episode_only_http_app_does_not_expose_map_routes(self) -> None:
        """Frozen benchmark serving must not initialize or expose live map APIs."""

        app = create_app(episode_service=FrozenExternalEpisodeService([_case()]), episode_only=True)
        routes = {route.resource.canonical for route in app.router.routes()}

        self.assertIn("/v1/episodes/reset", routes)
        self.assertIn("/v1/episodes/{episode_id}/step", routes)
        self.assertNotIn("/api/routes/plan", routes)


class _System2Policy:
    async def decide(self, state):
        return System2Action(action_type=System2ActionType.NAVIGATE)


class _System1Policy:
    async def predict(self, state):
        return System1Output(trajectory_ego_t=[[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])


class _SafetyValidator:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    async def validate(self, output, state):
        accepted = self.decisions.pop(0)
        return SafetyResult(accepted=accepted, reason="clear" if accepted else "blocked")


class _UnusedToolExecutor:
    async def execute(self, call, state):
        return ToolResponseV2(tool=call.tool, op=call.op, success=False, error={"code": "unused"})


def _runner(safety_decisions):
    return FrozenEpisodeBenchmarkRunner(
        system2_policy_factory=lambda case: _System2Policy(),
        system1_policy_factory=lambda case: _System1Policy(),
        tool_executor_factory=lambda case: _UnusedToolExecutor(),
        safety_validator_factory=lambda case: _SafetyValidator(safety_decisions),
    )


def _case() -> FrozenEpisodeCase:
    return FrozenEpisodeCase(
        episode_id="episode-1",
        instruction="Walk to the entrance.",
        initial_observation={"fpv": "frame-0.jpg"},
        initial_replay_key="obs-0",
        transitions=[
            FrozenEpisodeTransition(
                replay_key="transition-0",
                observation={"fpv": "frame-1.jpg"},
                reward=0.0,
                done=False,
                info={"collision": False},
                expected_action_shape=(2, 3),
            ),
            FrozenEpisodeTransition(
                replay_key="transition-1",
                observation={"fpv": "frame-2.jpg"},
                reward=1.0,
                done=True,
                info={"arrival_success": True, "collision": False, "stop_reason": "arrival"},
                expected_action_shape=(2, 3),
            ),
        ],
        optimal_steps=2,
    )


if __name__ == "__main__":
    unittest.main()
