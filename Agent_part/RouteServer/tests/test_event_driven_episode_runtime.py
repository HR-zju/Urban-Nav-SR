from __future__ import annotations

import asyncio
import unittest

from langgraph.checkpoint.memory import InMemorySaver

from mcp_server.contracts.episode import EpisodeEventType, EpisodeTraceV2
from mcp_server.contracts.task import (
    ControlState,
    DialogueState,
    ExternalUserEvent,
    ExternalUserEventType,
    PendingInteraction,
    TaskState,
    TaskStatus,
)
from mcp_server.contracts.tool import ToolCallV2, ToolResponseV2
from mcp_server.runtime import (
    EnvironmentStepResult,
    EnvironmentExecutionManager,
    EpisodeRuntime,
    EpisodeRuntimeConfig,
    EpisodeRuntimeEventService,
    InMemoryExternalEventStore,
    ObservationResult,
    SafetyResult,
    System1Output,
    System2Action,
    System2ActionType,
    TerminalDecision,
    TerminalOutcome,
    build_episode_graph,
    reduce_external_event,
)


class ExternalEventReducerTest(unittest.TestCase):
    """Pure task revision and interaction-correlation tests."""

    def test_revision_conflict_does_not_mutate_task(self) -> None:
        event = ExternalUserEvent(
            episode_id="episode-1",
            event_type=ExternalUserEventType.MODIFY_TASK,
            content={"constraints": {"speed": "slow"}},
            expected_task_revision=2,
        )

        result = reduce_external_event(
            event,
            task_state=TaskState(task_id="task-1", task_revision=1),
            dialogue_state=DialogueState(),
            control_state=ControlState(),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "task_revision_conflict")
        self.assertEqual(result.task_state.task_revision, 1)
        self.assertEqual(result.control_state.control_epoch, 0)

    def test_correlated_response_clears_only_matching_interaction(self) -> None:
        dialogue = DialogueState(
            pending_interaction=PendingInteraction(
                interaction_id="question-1",
                request={"question": "Which entrance?"},
            )
        )
        event = ExternalUserEvent(
            episode_id="episode-1",
            event_type=ExternalUserEventType.USER_RESPONSE,
            interaction_id="question-1",
            content={"option_id": "north"},
        )

        result = reduce_external_event(
            event,
            task_state=TaskState(task_id="task-1"),
            dialogue_state=dialogue,
            control_state=ControlState(),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "interaction_resolved")
        self.assertIsNone(result.dialogue_state.pending_interaction)
        self.assertEqual(result.task_state.task_revision, 0)

    def test_stale_response_becomes_new_instruction(self) -> None:
        dialogue = DialogueState(
            pending_interaction=PendingInteraction(
                interaction_id="question-current",
                request={"question": "Which entrance?"},
            )
        )
        event = ExternalUserEvent(
            episode_id="episode-1",
            event_type=ExternalUserEventType.USER_RESPONSE,
            interaction_id="question-stale",
            content="Go to the library instead.",
        )

        result = reduce_external_event(
            event,
            task_state=TaskState(task_id="task-1", active_instruction="Go to the station."),
            dialogue_state=dialogue,
            control_state=ControlState(),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "new_instruction")
        self.assertEqual(result.task_state.active_instruction, "Go to the library instead.")
        self.assertEqual(result.task_state.task_revision, 1)
        self.assertIsNone(result.dialogue_state.pending_interaction)


class ExecutionContractTest(unittest.IsolatedAsyncioTestCase):
    """Stale chunks and live event ingress must fail or resume explicitly."""

    async def test_stale_system1_output_is_rejected_before_environment_step(self) -> None:
        environment = _CaptureEnvironment()
        manager = EnvironmentExecutionManager(environment, execute_steps=2)
        output = System1Output(
            trajectory_ego_t=[[0.5, 0.0, 0.0]],
            task_revision=0,
            route_revision=0,
            control_epoch=0,
            observation_id="obs-old",
        )
        state = {
            "episode_id": "episode-stale",
            "observation_id": "obs-new",
            "task_state": TaskState(task_id="task-stale", task_revision=1).model_dump(mode="json"),
            "control_state": ControlState(control_epoch=1).model_dump(mode="json"),
        }

        with self.assertRaisesRegex(ValueError, "stale System1 output"):
            await manager.dispatch(output, state)
        self.assertEqual(environment.executed, [])

    async def test_preempt_before_wait_returns_same_feedback(self) -> None:
        environment = _BlockingThenCaptureEnvironment()
        manager = EnvironmentExecutionManager(environment, execute_steps=1)
        output = System1Output(
            trajectory_ego_t=[[0.5, 0.0, 0.0]],
            task_revision=0,
            route_revision=0,
            control_epoch=0,
            observation_id="obs-current",
        )
        state = {
            "episode_id": "episode-preempt-first",
            "observation_id": "obs-current",
            "task_state": TaskState(task_id="task-preempt").model_dump(mode="json"),
            "control_state": ControlState().model_dump(mode="json"),
        }

        lease = await manager.dispatch(output, state)
        await asyncio.wait_for(environment.started.wait(), timeout=2.0)
        immediate = await manager.preempt("episode-preempt-first", reason="pause")
        waited = await manager.wait(lease)

        self.assertIsNotNone(immediate)
        self.assertTrue(waited.preempted)
        self.assertEqual(waited.reason, "pause")
        self.assertEqual(waited.lease.action_id, lease.action_id)

    async def test_halt_without_active_lease_is_idempotent(self) -> None:
        environment = _CaptureEnvironment()
        manager = EnvironmentExecutionManager(environment, execute_steps=1)

        first = await manager.halt("episode-halt", reason="unsafe_to_continue")
        second = await manager.halt("episode-halt", reason="ignored-retry-reason")

        self.assertFalse(first.had_active_action)
        self.assertEqual(second, first)
        self.assertEqual(environment.halt_reasons, ["unsafe_to_continue"])

    async def test_halt_cancels_active_lease_and_resolves_waiter(self) -> None:
        environment = _BlockingThenCaptureEnvironment()
        manager = EnvironmentExecutionManager(environment, execute_steps=1)
        output = System1Output(
            trajectory_ego_t=[[0.5, 0.0, 0.0]],
            task_revision=0,
            route_revision=0,
            control_epoch=0,
            observation_id="obs-current",
        )
        state = {
            "episode_id": "episode-active-halt",
            "observation_id": "obs-current",
            "task_state": TaskState(task_id="task-active-halt").model_dump(mode="json"),
            "control_state": ControlState().model_dump(mode="json"),
        }

        lease = await manager.dispatch(output, state)
        await asyncio.wait_for(environment.started.wait(), timeout=2.0)
        halted = await manager.halt("episode-active-halt", reason="controller_fault")
        waited = await manager.wait(lease)

        self.assertTrue(halted.had_active_action)
        self.assertEqual(halted.action_id, lease.action_id)
        self.assertTrue(waited.preempted)
        self.assertEqual(waited.reason, "controller_fault")
        self.assertEqual(environment.halt_reasons, ["controller_fault"])

    async def test_legacy_terminal_actions_normalize_but_canonical_abort_requires_reason(self) -> None:
        legacy_finish = System2Action.model_validate({"action_type": "finish", "finish": {}})
        legacy_stop = System2Action.model_validate({"action_type": "stop", "finish": {}})

        self.assertEqual(legacy_finish.action_type, System2ActionType.COMPLETE_TASK)
        self.assertEqual(legacy_finish.terminal.reason_code, "legacy_system2_finish")
        self.assertEqual(legacy_stop.action_type, System2ActionType.ABORT_TASK)
        self.assertEqual(legacy_stop.terminal.reason_code, "legacy_system2_stop")
        with self.assertRaisesRegex(ValueError, "requires terminal payload"):
            System2Action(action_type=System2ActionType.ABORT_TASK)

    async def test_registered_runtime_event_service_resumes_waiting_graph(self) -> None:
        runtime = _RegisteredRuntimeFake()
        service = EpisodeRuntimeEventService()
        service.register(episode_id="episode-live", runtime=runtime, thread_id="thread-live")

        response = await service.submit_event(
            "episode-live",
            {
                "event": {
                    "episode_id": "episode-live",
                    "event_type": "resume",
                    "event_id": "event:resume:live",
                },
                "resume_if_waiting": True,
            },
        )

        self.assertTrue(response.accepted)
        self.assertTrue(response.info["resumed"])
        self.assertEqual(runtime.resume_thread_ids, ["thread-live"])
        self.assertEqual(runtime.events[0].event_id, "event:resume:live")

        with self.assertRaisesRegex(ValueError, "episode_id mismatch"):
            await service.submit_event(
                "episode-live",
                {"event": {"episode_id": "another-episode", "event_type": "pause"}},
            )

    def test_tool_response_preserves_revision_zero(self) -> None:
        response = ToolResponseV2.from_legacy(
            {
                "tool": "map_tool",
                "success": True,
                "result": {"state_revision": 0},
            },
            op="create_route_session",
        )

        self.assertEqual(response.state_revision, 0)


class EventDrivenEpisodeRuntimeTest(unittest.IsolatedAsyncioTestCase):
    """Closed-loop tests for external events, repeated HITL and preemption."""

    async def test_complete_task_requires_authoritative_arrival_evidence(self) -> None:
        system1 = _NeverSystem1Policy()
        environment = _ArrivalReadyEnvironment()
        graph = build_episode_graph(
            system2_policy=_SingleActionPolicy(_complete_action()),
            system1_policy=system1,
            tool_executor=_UnusedToolExecutor(),
            safety_validator=_AcceptSafety(),
            environment=environment,
        )

        state = await EpisodeRuntime(graph).run(
            episode_id="episode-complete",
            initial_context={"instruction": "Go to the goal."},
        )
        trace = EpisodeTraceV2.model_validate(state["episode_trace"])

        self.assertEqual(state["terminal_status"], "succeeded")
        self.assertEqual(TaskState.model_validate(state["task_state"]).status, TaskStatus.COMPLETED)
        self.assertEqual(system1.calls, 0)
        self.assertEqual(environment.halt_reasons, ["arrival_confirmed"])
        validation = next(
            event for event in trace.events if event.event_type == EpisodeEventType.TERMINAL_VALIDATION
        )
        self.assertTrue(validation.payload["result"]["accepted"])

    async def test_unverified_complete_reenters_system2_without_false_success(self) -> None:
        policy = _ActionSequencePolicy([_complete_action(), System2Action(action_type=System2ActionType.NAVIGATE)])
        environment = _CaptureEnvironment()
        graph = build_episode_graph(
            system2_policy=policy,
            system1_policy=_LongSystem1Policy(),
            tool_executor=_UnusedToolExecutor(),
            safety_validator=_AcceptSafety(),
            environment=environment,
        )

        state = await EpisodeRuntime(graph).run(
            episode_id="episode-false-complete",
            initial_context={"instruction": "Continue."},
        )
        trace = EpisodeTraceV2.model_validate(state["episode_trace"])
        validations = [
            event for event in trace.events if event.event_type == EpisodeEventType.TERMINAL_VALIDATION
        ]

        self.assertEqual(policy.calls, 2)
        self.assertEqual(len(environment.executed), 1)
        self.assertFalse(validations[0].payload["result"]["accepted"])
        self.assertEqual(state["terminal_status"], "succeeded")

    async def test_repeated_unverified_complete_fails_after_budget(self) -> None:
        environment = _CaptureEnvironment()
        graph = build_episode_graph(
            system2_policy=_SingleActionPolicy(_complete_action()),
            system1_policy=_NeverSystem1Policy(),
            tool_executor=_UnusedToolExecutor(),
            safety_validator=_AcceptSafety(),
            environment=environment,
            config=EpisodeRuntimeConfig(max_terminal_rejections=1),
        )

        state = await EpisodeRuntime(graph).run(
            episode_id="episode-complete-budget",
            initial_context={"instruction": "Continue."},
        )
        trace = EpisodeTraceV2.model_validate(state["episode_trace"])

        self.assertEqual(state["terminal_status"], "failed")
        self.assertEqual(state["stop_reason"], "terminal_validation_retry_exhausted")
        self.assertEqual(
            sum(event.event_type == EpisodeEventType.TERMINAL_VALIDATION for event in trace.events),
            2,
        )
        self.assertEqual(environment.halt_reasons, ["terminal_validation_retry_exhausted"])

    async def test_abort_task_fails_and_halts_without_calling_system1(self) -> None:
        system1 = _NeverSystem1Policy()
        environment = _CaptureEnvironment()
        action = System2Action(
            action_type=System2ActionType.ABORT_TASK,
            terminal=TerminalDecision(
                outcome=TerminalOutcome.FAILED,
                reason_code="unsafe_to_continue",
                reason="The pedestrian corridor is blocked.",
            ),
        )
        graph = build_episode_graph(
            system2_policy=_SingleActionPolicy(action),
            system1_policy=system1,
            tool_executor=_UnusedToolExecutor(),
            safety_validator=_AcceptSafety(),
            environment=environment,
        )

        state = await EpisodeRuntime(graph).run(
            episode_id="episode-abort",
            initial_context={"instruction": "Continue."},
        )

        self.assertEqual(state["terminal_status"], "failed")
        self.assertEqual(state["stop_reason"], "unsafe_to_continue")
        self.assertEqual(TaskState.model_validate(state["task_state"]).status, TaskStatus.FAILED)
        self.assertEqual(system1.calls, 0)
        self.assertEqual(environment.halt_reasons, ["unsafe_to_continue"])

    async def test_navigate_stop_hold_still_calls_system1(self) -> None:
        system1 = _NeverSystem1Policy(allow_predict=True)
        graph = build_episode_graph(
            system2_policy=_SingleActionPolicy(
                System2Action(
                    action_type=System2ActionType.NAVIGATE,
                    finish={"intent": "stop", "trajectory_type": "stop_hold"},
                )
            ),
            system1_policy=system1,
            tool_executor=_UnusedToolExecutor(),
            safety_validator=_AcceptSafety(),
            environment=_CaptureEnvironment(),
        )

        state = await EpisodeRuntime(graph).run(
            episode_id="episode-stop-hold",
            initial_context={"instruction": "Yield briefly."},
        )

        self.assertEqual(system1.calls, 1)
        self.assertEqual(state["terminal_status"], "succeeded")

    async def test_initial_task_update_binds_system1_revisions(self) -> None:
        event_store = InMemoryExternalEventStore()
        event = ExternalUserEvent(
            episode_id="episode-update",
            event_type=ExternalUserEventType.NEW_INSTRUCTION,
            content="Go to the library instead.",
            expected_task_revision=0,
            event_id="event:update:1",
        )
        await event_store.append(event)
        policy = _RecordingNavigatePolicy()
        environment = _CaptureEnvironment()
        graph = build_episode_graph(
            system2_policy=policy,
            system1_policy=_LongSystem1Policy(),
            tool_executor=_UnusedToolExecutor(),
            safety_validator=_AcceptSafety(),
            environment=environment,
            external_event_store=event_store,
            config=EpisodeRuntimeConfig(execute_steps=2),
        )

        state = await EpisodeRuntime(graph).run(
            episode_id="episode-update",
            initial_context={"instruction": "Go to the cafe."},
        )
        trace = EpisodeTraceV2.model_validate(state["episode_trace"])
        system1_event = next(event for event in trace.events if event.event_type == EpisodeEventType.SYSTEM1_OUTPUT)
        output = system1_event.payload["output"]

        self.assertEqual(policy.instructions, ["Go to the library instead."])
        self.assertEqual(output["task_revision"], 1)
        self.assertEqual(output["control_epoch"], 1)
        self.assertEqual(len(environment.executed[0].trajectory_ego_t), 2)
        self.assertIn(EpisodeEventType.TASK_STATE_UPDATED, [event.event_type for event in trace.events])

    async def test_repeated_hitl_records_wait_before_resume_and_clears_response(self) -> None:
        policy = _TwoQuestionPolicy()
        graph = build_episode_graph(
            system2_policy=policy,
            system1_policy=_LongSystem1Policy(),
            tool_executor=_UnusedToolExecutor(),
            safety_validator=_AcceptSafety(),
            environment=_CaptureEnvironment(),
            checkpointer=InMemorySaver(),
        )
        runtime = EpisodeRuntime(graph)

        first = await runtime.run(
            episode_id="episode-dialogue",
            initial_context={"instruction": "Go to the station."},
            thread_id="thread-dialogue",
        )
        first_trace = EpisodeTraceV2.model_validate(first["episode_trace"])
        second = await runtime.resume(thread_id="thread-dialogue", user_response={"option_id": "north"})
        third = await runtime.resume(thread_id="thread-dialogue", user_response={"option_id": "stairs"})
        final_trace = EpisodeTraceV2.model_validate(third["episode_trace"])

        self.assertIn("__interrupt__", first)
        self.assertIn(EpisodeEventType.USER_INTERRUPT, [event.event_type for event in first_trace.events])
        self.assertIn("__interrupt__", second)
        self.assertEqual(
            sum(event.event_type == EpisodeEventType.INTERACTION_RESOLVED for event in final_trace.events),
            2,
        )
        self.assertIsNone(third.get("user_response"))
        self.assertEqual(third["stop_reason"], "arrival")

    async def test_pause_preempts_active_chunk_and_resume_reobserves(self) -> None:
        environment = _BlockingThenCaptureEnvironment()
        graph = build_episode_graph(
            system2_policy=_RecordingNavigatePolicy(),
            system1_policy=_LongSystem1Policy(),
            tool_executor=_UnusedToolExecutor(),
            safety_validator=_AcceptSafety(),
            environment=environment,
            checkpointer=InMemorySaver(),
        )
        runtime = EpisodeRuntime(graph)
        run_task = asyncio.create_task(
            runtime.run(
                episode_id="episode-pause",
                initial_context={"instruction": "Continue."},
                thread_id="thread-pause",
            )
        )
        await asyncio.wait_for(environment.started.wait(), timeout=2.0)
        await runtime.submit_event(
            ExternalUserEvent(
                episode_id="episode-pause",
                event_type=ExternalUserEventType.PAUSE,
                event_id="event:pause:1",
            )
        )
        paused = await asyncio.wait_for(run_task, timeout=2.0)
        resumed = await runtime.resume_with_event(
            thread_id="thread-pause",
            event=ExternalUserEvent(
                episode_id="episode-pause",
                event_type=ExternalUserEventType.RESUME,
                event_id="event:resume:1",
            ),
        )
        trace = EpisodeTraceV2.model_validate(resumed["episode_trace"])

        self.assertIn("__interrupt__", paused)
        self.assertEqual(
            sum(event.event_type == EpisodeEventType.ACTION_PREEMPTED for event in trace.events),
            1,
        )
        self.assertEqual(resumed["stop_reason"], "arrival")
        self.assertEqual(TaskState.model_validate(resumed["task_state"]).task_revision, 0)
        # pause、resume 和最终 terminal halt 各自推进一次 control generation。
        self.assertEqual(ControlState.model_validate(resumed["control_state"]).control_epoch, 3)

    async def test_safety_intervention_is_the_executed_chunk(self) -> None:
        environment = _CaptureEnvironment()
        graph = build_episode_graph(
            system2_policy=_RecordingNavigatePolicy(),
            system1_policy=_LongSystem1Policy(),
            tool_executor=_UnusedToolExecutor(),
            safety_validator=_StopInterventionSafety(),
            environment=environment,
        )

        state = await EpisodeRuntime(graph).run(
            episode_id="episode-intervention",
            initial_context={"instruction": "Continue."},
        )
        trace = EpisodeTraceV2.model_validate(state["episode_trace"])

        self.assertEqual(environment.executed[0].trajectory_ego_t, [[0.0, 0.0, 0.0]])
        self.assertIn(EpisodeEventType.SAFETY_INTERVENTION, [event.event_type for event in trace.events])

    async def test_pause_during_tool_call_is_reduced_before_next_decision(self) -> None:
        tool = _BlockingToolExecutor()
        environment = _CaptureEnvironment()
        graph = build_episode_graph(
            system2_policy=_ToolThenNavigatePolicy(),
            system1_policy=_LongSystem1Policy(),
            tool_executor=tool,
            safety_validator=_AcceptSafety(),
            environment=environment,
            checkpointer=InMemorySaver(),
        )
        runtime = EpisodeRuntime(graph)
        run_task = asyncio.create_task(
            runtime.run(
                episode_id="episode-tool-pause",
                initial_context={"instruction": "Find a cafe."},
                thread_id="thread-tool-pause",
            )
        )
        await asyncio.wait_for(tool.started.wait(), timeout=2.0)
        await runtime.submit_event(
            ExternalUserEvent(
                episode_id="episode-tool-pause",
                event_type=ExternalUserEventType.PAUSE,
                event_id="event:tool-pause:1",
            )
        )
        tool.release.set()
        state = await asyncio.wait_for(run_task, timeout=2.0)
        trace = EpisodeTraceV2.model_validate(state["episode_trace"])

        self.assertIn("__interrupt__", state)
        self.assertEqual(environment.executed, [])
        self.assertIn(EpisodeEventType.TOOL_RESPONSE, [event.event_type for event in trace.events])
        self.assertIn(EpisodeEventType.EXTERNAL_EVENT_RECEIVED, [event.event_type for event in trace.events])

    async def test_cancel_during_system1_inference_never_dispatches_output(self) -> None:
        system1 = _BlockingSystem1Policy()
        environment = _CaptureEnvironment()
        graph = build_episode_graph(
            system2_policy=_RecordingNavigatePolicy(),
            system1_policy=system1,
            tool_executor=_UnusedToolExecutor(),
            safety_validator=_AcceptSafety(),
            environment=environment,
        )
        runtime = EpisodeRuntime(graph)
        run_task = asyncio.create_task(
            runtime.run(
                episode_id="episode-cfm-cancel",
                initial_context={"instruction": "Continue."},
            )
        )
        await asyncio.wait_for(system1.started.wait(), timeout=2.0)
        await runtime.submit_event(
            ExternalUserEvent(
                episode_id="episode-cfm-cancel",
                event_type=ExternalUserEventType.CANCEL,
                event_id="event:cfm-cancel:1",
            )
        )
        system1.release.set()
        state = await asyncio.wait_for(run_task, timeout=2.0)
        trace = EpisodeTraceV2.model_validate(state["episode_trace"])

        self.assertEqual(state["terminal_status"], "cancelled")
        self.assertEqual(environment.executed, [])
        self.assertNotIn(EpisodeEventType.ACTION_DISPATCHED, [event.event_type for event in trace.events])


class _RecordingNavigatePolicy:
    def __init__(self) -> None:
        self.instructions: list[str] = []

    async def decide(self, state):
        self.instructions.append(TaskState.model_validate(state["task_state"]).active_instruction)
        return System2Action(action_type=System2ActionType.NAVIGATE)


class _SingleActionPolicy:
    def __init__(self, action: System2Action) -> None:
        self.action = action

    async def decide(self, state):
        return self.action


class _ActionSequencePolicy:
    def __init__(self, actions: list[System2Action]) -> None:
        self.actions = actions
        self.calls = 0

    async def decide(self, state):
        action = self.actions[self.calls]
        self.calls += 1
        return action


class _TwoQuestionPolicy:
    async def decide(self, state):
        dialogue = DialogueState.model_validate(state["dialogue_state"])
        answers = [turn for turn in dialogue.recent_turns if turn.role == "user"]
        if len(answers) < 2:
            return System2Action(
                action_type=System2ActionType.ASK_USER,
                user_request={"question": f"Question {len(answers) + 1}?"},
            )
        return System2Action(action_type=System2ActionType.NAVIGATE)


class _LongSystem1Policy:
    async def predict(self, state):
        return System1Output(
            trajectory_ego_t=[[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
        )


class _NeverSystem1Policy:
    def __init__(self, *, allow_predict: bool = False) -> None:
        self.allow_predict = allow_predict
        self.calls = 0

    async def predict(self, state):
        self.calls += 1
        if not self.allow_predict:
            raise AssertionError("System1 must not run for a terminal action")
        return System1Output(trajectory_ego_t=[[0.0, 0.0, 0.0]])


class _BlockingSystem1Policy:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def predict(self, state):
        self.started.set()
        await self.release.wait()
        return System1Output(trajectory_ego_t=[[0.5, 0.0, 0.0]])


class _AcceptSafety:
    async def validate(self, output, state):
        return SafetyResult(accepted=True, reason="clear")


class _StopInterventionSafety:
    async def validate(self, output, state):
        return SafetyResult(
            accepted=False,
            reason="obstacle",
            intervention={"trajectory_ego_t": [[0.0, 0.0, 0.0]]},
        )


class _CaptureEnvironment:
    def __init__(self) -> None:
        self.executed: list[System1Output] = []
        self.preempt_reasons: list[str] = []
        self.halt_reasons: list[str] = []

    async def reset(self, initial_context):
        return ObservationResult(observation={"fpv": "frame-0.jpg"}, replay_key="obs-0")

    async def step(self, output, state):
        self.executed.append(output)
        return EnvironmentStepResult(
            observation={"fpv": "frame-1.jpg"},
            reward=1.0,
            done=True,
            info={"arrival_success": True, "stop_reason": "arrival"},
            replay_key="transition-1",
        )

    async def preempt(self, *, reason):
        self.preempt_reasons.append(reason)

    async def halt(self, *, reason):
        self.halt_reasons.append(reason)


class _ArrivalReadyEnvironment(_CaptureEnvironment):
    async def reset(self, initial_context):
        return ObservationResult(
            observation={"fpv": "arrival-frame.jpg"},
            info={"arrival_success": True},
            replay_key="arrival-observation",
        )


class _BlockingThenCaptureEnvironment(_CaptureEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.calls = 0

    async def step(self, output, state):
        self.calls += 1
        self.executed.append(output)
        if self.calls == 1:
            self.started.set()
            await asyncio.Event().wait()
        return EnvironmentStepResult(
            observation={"fpv": f"frame-{self.calls}.jpg"},
            reward=1.0,
            done=True,
            info={"arrival_success": True, "stop_reason": "arrival"},
            replay_key=f"transition-{self.calls}",
        )


class _UnusedToolExecutor:
    async def execute(self, call, state):
        return ToolResponseV2(tool=call.tool, op=call.op, success=False, error={"code": "unused"})


class _ToolThenNavigatePolicy:
    async def decide(self, state):
        if not state.get("last_tool_response"):
            return System2Action(
                action_type=System2ActionType.TOOL_CALL,
                tool_call=ToolCallV2(
                    tool="map_tool",
                    arguments={"op": "search_destination", "query": "cafe"},
                ),
            )
        return System2Action(action_type=System2ActionType.NAVIGATE)


class _BlockingToolExecutor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, call, state):
        self.started.set()
        await self.release.wait()
        return ToolResponseV2(
            tool=call.tool,
            op=call.op,
            success=True,
            result={"candidates": []},
        )


class _RegisteredRuntimeFake:
    def __init__(self) -> None:
        self.events: list[ExternalUserEvent] = []
        self.resume_thread_ids: list[str] = []

    async def submit_event(self, event):
        self.events.append(ExternalUserEvent.model_validate(event))
        return True

    async def resume(self, *, thread_id, user_response):
        self.resume_thread_ids.append(thread_id)
        self.user_response = user_response
        return {"terminal_status": "running"}


def _complete_action() -> System2Action:
    return System2Action(
        action_type=System2ActionType.COMPLETE_TASK,
        terminal=TerminalDecision(
            outcome=TerminalOutcome.SUCCEEDED,
            reason_code="arrival_confirmed",
            evidence_refs=["observation:latest"],
        ),
    )


if __name__ == "__main__":
    unittest.main()
