# Agent Part

`Agent_part` contains the navigation-side agent services for UrbanNav.

## RouteServer

The active backend is:

- [RouteServer](RouteServer/)

RouteServer provides:

- map provider access
- natural-language destination recommendation
- itinerary decomposition
- persistent route sessions
- dynamic via-point updates
- RouteContext / LocalBEVContext
- FPV image understanding
- zero-shot visual prompt generation

Start with:

- [RouteServer README](RouteServer/README.md)

For the current main workflows, use these RouteServer examples:

- `examples/image_understanding_client.py`
- `examples/route_context_debug_client.py`
- `examples/zero_shot_visual_prompt_agent_client.py`
- `examples/route_session_client.py`
- `examples/realtime_instruction_session_client.py`
- `examples/itinerary_client.py`
- `examples/macos_prompt_recommendation_client.py`
