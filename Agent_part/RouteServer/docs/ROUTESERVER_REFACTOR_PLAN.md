# RouteServer Refactor Plan

## Non-Breaking Contract

The APP-facing HTTP/WebSocket contract is frozen during the refactor:

```text
GET  /api/health
POST /api/recommendations
POST /api/routes/plan
POST /api/route-sessions
GET  /api/route-sessions/{session_id}
POST /api/route-sessions/{session_id}/via-points
POST /api/route-sessions/{session_id}/replan
POST /api/waypoints/push
GET  /ws/route-sessions/{session_id}
```

MCP tool names and arguments also stay compatible.  Refactors must move logic
behind these entry points, not rename or remove the entry points.

## Target Layers

```text
src/mcp_server/
  server.py                 # MCP entry layer
  api_http.py               # APP HTTP/WebSocket entry layer

  core/
    map_ops.py              # deterministic geocode/search/category/route ops
    route_session_ops.py    # create/get/add/remove/replace/replan session ops
    route_context_ops.py    # route context/local BEV/route state ops
    schemas.py              # shared core result schemas

  teacher/
    intent_teacher.py
    recommendation_teacher.py
    visual_prompt_teacher.py
    trace_builder.py

  eval/
    cases.py
    runners.py
    metrics.py
    exporters.py

  agent/
    schemas.py              # AgentEvalCase / AgentTrace / ToolCall
    tool_schema.py          # student-visible meta tools
    tool_parser.py          # parse <tool_call> / <finish>
    tool_executor.py        # dispatch meta tools to core ops
```

## Current Refactor Increment

- Added `core/` package.
- Added deterministic `map_ops.py` wrappers.
- Added `route_session_ops.py` wrappers for route session mutation.
- Switched APP HTTP route-session handlers to `core.route_session_ops`.
- Switched MCP route-session tools to `core.route_session_ops`.
- Switched deterministic MCP map tools to `core.map_ops`.
- Added `agent/` protocol modules for the student-visible meta tools:
  `map_tool`, `image_tool`, and `user_interaction_tool`.
- Added `eval/` modules for the first closed loop:
  `AgentEvalCase -> AgentRunner -> AgentTrace -> metrics -> ms-swift JSONL`.
- Added a contract/core smoke test that locks APP route paths and route-session
  payload shape.

## Next Increments

1. Move route context helpers behind `core/route_context_ops.py`.
2. Add `teacher/` wrappers for the old LLM/LVLM flows:
   `IntentParser`, `GoalDecomposer`, recommendation rerank, and visual prompt
   teacher.
3. Add external LVLM / local checkpoint runners on top of the existing
   `AgentRunner` interface.
4. Add `core/route_context_ops.py` only if online APP and Agent runtime need
   shared route progress/local BEV context.
5. Export high-quality teacher/human traces to ms-swift Agent JSONL without
   exposing teacher internal LLM calls as student-visible tools.
