"""Offline runner for frozen closed-loop episode cases."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from mcp_server.contracts.episode import EpisodeTraceV2
from mcp_server.eval.episode.environment import FrozenEpisodeEnvironment
from mcp_server.eval.episode.metrics import EpisodeMetricResult, compute_episode_metrics
from mcp_server.eval.episode.schema import FrozenEpisodeCase
from mcp_server.runtime.graph import EpisodeRuntime, EpisodeRuntimeConfig, build_episode_graph


class FrozenEpisodeBenchmarkResult(BaseModel):
    """One deterministic frozen episode benchmark result.

    Args:
        case: Frozen source case.
        trace: Canonical runtime trace.
        metrics: Main outcome metrics.
        replay_digest: SHA-256 digest of the immutable case and trace events.
    """

    model_config = ConfigDict(extra="forbid")

    case: FrozenEpisodeCase
    trace: EpisodeTraceV2
    metrics: EpisodeMetricResult
    replay_digest: str


class FrozenEpisodeBenchmarkRunner:
    """Run injected deterministic policies against frozen episode transitions.

    Args:
        system2_policy_factory: Factory receiving one case and returning a
            deterministic System2 policy.
        system1_policy_factory: Factory receiving one case and returning a
            deterministic System1 policy.
        tool_executor_factory: Factory receiving one case and returning a
            frozen tool executor.
        safety_validator_factory: Factory receiving one case and returning a
            deterministic validator.
        runtime_config: Closed-loop execution budgets.

    This evaluator never constructs a live provider or model client. A caller
    that needs model inference must generate predictions before offline scoring.
    """

    def __init__(
        self,
        *,
        system2_policy_factory: Callable[[FrozenEpisodeCase], Any],
        system1_policy_factory: Callable[[FrozenEpisodeCase], Any],
        tool_executor_factory: Callable[[FrozenEpisodeCase], Any],
        safety_validator_factory: Callable[[FrozenEpisodeCase], Any],
        runtime_config: EpisodeRuntimeConfig | None = None,
    ) -> None:
        self.system2_policy_factory = system2_policy_factory
        self.system1_policy_factory = system1_policy_factory
        self.tool_executor_factory = tool_executor_factory
        self.safety_validator_factory = safety_validator_factory
        self.runtime_config = runtime_config or EpisodeRuntimeConfig()

    async def run_case(self, case: FrozenEpisodeCase) -> FrozenEpisodeBenchmarkResult:
        """Execute and score one frozen episode.

        Args:
            case: Frozen benchmark case.

        Returns:
            Trace, metrics and deterministic replay digest.
        """

        environment = FrozenEpisodeEnvironment(case)
        graph = build_episode_graph(
            system2_policy=self.system2_policy_factory(case),
            system1_policy=self.system1_policy_factory(case),
            tool_executor=self.tool_executor_factory(case),
            safety_validator=self.safety_validator_factory(case),
            environment=environment,
            config=self.runtime_config,
        )
        state = await EpisodeRuntime(graph).run(
            episode_id=case.episode_id,
            session_id=str(case.metadata.get("session_id") or "") or None,
            initial_context={"instruction": case.instruction},
        )
        trace = EpisodeTraceV2.model_validate(state["episode_trace"])
        metrics = compute_episode_metrics(trace, case)
        return FrozenEpisodeBenchmarkResult(
            case=case,
            trace=trace,
            metrics=metrics,
            replay_digest=_replay_digest(case, trace),
        )


def _replay_digest(case: FrozenEpisodeCase, trace: EpisodeTraceV2) -> str:
    """Return a deterministic digest excluding wall-clock event timestamps."""

    events = [
        {
            "tick_id": event.tick_id,
            "event_type": event.event_type.value,
            "payload": event.payload,
            "source": event.source,
        }
        for event in trace.events
    ]
    payload = {"case": case.model_dump(mode="json"), "events": events}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
