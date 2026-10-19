# Agent-tools teacher

本目录只包含未启用的 agent-tools/EpisodeTrace teacher。它与当前非 agent-tools E2E 的
Reasoning FullCoT、VQA teacher 分开维护。

旧顶层 `teacher.trace_builder` 和 `teacher.agentic_stage2_trace_builder` 转发已移除。调用方应直接从本子包的责任模块导入，映射见 `docs/retired_module_paths.json`。
