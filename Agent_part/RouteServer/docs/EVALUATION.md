# Destination / Route Evaluation：导航入口

原文按协议与结果分开归档，原路径保留为入口。

| 内容 | 位置 |
| --- | --- |
| 任务定义与 benchmark 设计 | [技术设计](evaluation/URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md) |
| 数据格式、指标、replay 和运行协议 | [评测协议](evaluation/PROTOCOL.md) |
| 原 Module 1 历史结果表与运行记录 | [历史结果](experiments/MODULE1_RESULTS_HISTORY.md) |
| A6 距离基线与 InternVL 配对核验 | [实验归档入口](experiments/README.md) |
| SideWalk Agent 评测 | [Stage2 协议](evaluation/SIDEWALK_AGENT_EVAL_PROTOCOL.md) |
| 全部文档 | [文档索引](README.md) |

历史结果保留当时的口径和待测占位，不自动视为当前论文结果。A6 使用冻结 final-15K、v2 scorer 和固定分母：距离基线／InternVL3.5-14B 的 Task SPL 为 81.03%／76.33%。原历史表中的 81.13%／77.69% 属于原归档口径，不能与 A6 混用。详细解释、逐 case 结果和可复核证据见实验归档。
