# Agent-tools teacher

本目录只包含未启用的 agent-tools/EpisodeTrace teacher。它与当前非 agent-tools E2E 的
Reasoning FullCoT、VQA teacher 分开维护。

旧入口 `mcp_server.teacher.trace_builder` 和
`mcp_server.teacher.agentic_stage2_trace_builder` 暂时保留为 compatibility shim；新代码
应使用 `mcp_server.teacher.agent_tools`。删除旧 shim 前必须完成全仓 import parity 和
focused regression。
