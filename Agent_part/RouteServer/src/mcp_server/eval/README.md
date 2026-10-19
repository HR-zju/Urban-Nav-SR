# 评测代码目录

`eval` 现在按评测责任拆分为三个物理子包：

| 子包 | 责任 | 典型内容 |
| --- | --- | --- |
| `agent_tools/` | agent-tools / Stage2 在线与 trace 评测 | runner、replay、episode、validation、训练样本导出 |
| `urbannav/` | UrbanNav benchmark | schema、JSONL IO、metrics、collection、Module 1 |
| `openloop/` | E2E open-loop | Stage I、Stage II CFM、VQA、package、registry、reporting |

顶层旧模块及 `eval.episode` 兼容转发已移除。所有调用方直接从责任子包导入；路径映射见 `docs/retired_module_paths.json`。

评测代码不负责修改 runtime/backend 状态。UrbanNav 和 open-loop evaluator 应保持
离线、确定性、稳定 ID 对齐；任何真实服务依赖必须由 agent-tools runtime 侧明确管理。
