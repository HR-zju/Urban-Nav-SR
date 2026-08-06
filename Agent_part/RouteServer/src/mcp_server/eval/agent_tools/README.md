# Agent-tools 评测

这里维护 agent-tools / Stage2 的在线运行、trace、replay、validation、episode
benchmark 和训练样本导出。它依赖 RouteServer runtime，但不拥有 backend runtime。

旧的 `mcp_server.eval.*` 模块路径暂时由兼容 shim 保留；新代码应从
`mcp_server.eval.agent_tools.*` 导入。

