"""Frozen deterministic environment implementing the episode runtime protocol."""

from __future__ import annotations

from mcp_server.eval.agent_tools.episode.schema import FrozenEpisodeCase
from mcp_server.runtime.interfaces import EnvironmentStepResult, ObservationResult, System1Output


class FrozenEpisodeEnvironment:
    """Replay immutable observations, rewards and termination labels.

    Args:
        case: Validated frozen episode case.

    The environment never calls map APIs, simulators or mutable services. It
    validates action tensor shape when the frozen transition declares one.
    """

    def __init__(self, case: FrozenEpisodeCase) -> None:
        self.case = case
        self.transition_index = 0
        self.preempt_reasons: list[str] = []
        self.halt_reason: str | None = None

    async def reset(self, initial_context: dict) -> ObservationResult:
        """Reset to the case's frozen initial observation.

        Args:
            initial_context: Runtime reset metadata; ignored by frozen replay.

        Returns:
            First immutable observation.
        """

        self.transition_index = 0
        self.preempt_reasons.clear()
        self.halt_reason = None
        return ObservationResult(
            observation=dict(self.case.initial_observation),
            info={"instruction": self.case.instruction, "case_metadata": self.case.metadata},
            replay_key=self.case.initial_replay_key,
            external_events=[event.model_dump(mode="json") for event in self.case.initial_external_events],
        )

    async def step(self, output: System1Output, state: dict) -> EnvironmentStepResult:
        """Consume one frozen transition after validating action shape.

        Args:
            output: Proposed ego-frame System1 chunk.
            state: Current runtime state, used only for audit context.

        Returns:
            Next immutable transition result.

        Raises:
            RuntimeError: If the episode is stepped past its terminal record.
            ValueError: If the action shape violates the frozen contract.
        """

        if self.transition_index >= len(self.case.transitions):
            raise RuntimeError("frozen episode stepped after terminal transition")
        transition = self.case.transitions[self.transition_index]
        if transition.expected_action_shape is not None:
            actual_shape = (len(output.trajectory_ego_t), len(output.trajectory_ego_t[0]))
            if actual_shape != transition.expected_action_shape:
                raise ValueError(
                    f"System1 action shape mismatch: expected={transition.expected_action_shape}, actual={actual_shape}"
                )
        self.transition_index += 1
        return EnvironmentStepResult(
            observation=dict(transition.observation),
            reward=transition.reward,
            done=transition.done,
            info=dict(transition.info),
            replay_key=transition.replay_key,
            external_events=[event.model_dump(mode="json") for event in transition.external_events],
        )

    async def preempt(self, *, reason: str) -> None:
        """Record deterministic cancellation of an in-flight replay action.

        Args:
            reason: Stable preemption reason.

        Returns:
            None.
        """

        self.preempt_reasons.append(reason)

    async def halt(self, *, reason: str) -> None:
        """Record the idempotent terminal stop without consuming a transition.

        Args:
            reason: Stable terminal reason.

        Returns:
            None.
        """

        if self.halt_reason is None:
            self.halt_reason = reason
