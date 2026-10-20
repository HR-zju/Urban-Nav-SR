# 实验与结果

历史运行记录、结果表与离线核验入口。实验记录保留原有结论和适用时间；当前采用哪份结果，必须绑定具体数据与评分版本。

[返回总索引](../README.md)

## 文档

| 文档 | 内容 |
| --- | --- |
| [HISTORY_CANONICAL_GENERATION_20261009.md](HISTORY_CANONICAL_GENERATION_20261009.md) | 实验记录：真实 4s/2Hz canonical source 全量生成 |
| [HISTORY_STRICT_TRAINING_20261010.md](HISTORY_STRICT_TRAINING_20261010.md) | 实验记录：严格复用首批训练数据导出 |
| [HISTORY_TIME_ONLY_FULL_MIGRATION_20261010.md](HISTORY_TIME_ONLY_FULL_MIGRATION_20261010.md) | 实验记录：全量 observation 旧标注时间关联迁移 |
| [MODULE1_RESULTS_HISTORY.md](MODULE1_RESULTS_HISTORY.md) | Module 1 历史结果与运行记录 |
| [QWEN122B_HISTORY_SMOKE_20261011.md](QWEN122B_HISTORY_SMOKE_20261011.md) | 实验记录：Qwen122B 新窗口单样本推理 |
| [VLA_GAP_CLIP_20261011.md](VLA_GAP_CLIP_20261011.md) | 实验记录：新窗口单 clip 缺失 reasoning 补标 |
| [VLA_GAP_FULL_20261011.md](VLA_GAP_FULL_20261011.md) | 实验记录：新窗口全量缺失 reasoning 补标 |
| [VLA_SECTION_REPAIR_20261013.md](VLA_SECTION_REPAIR_20261013.md) | 实验记录：视觉目标方向与 stop 约束局部修复 |

## A6：距离基线与 InternVL 配对核验

2026-10-20 已完成：同一 final-15K 的四任务配对统计、候选／缓存／scorer 核验。完整记录保留在组合层，避免复制成两份可漂移的实验记录。

- [原理、配置、结果和限制](../../../../../ResearchWorkflow/evidence/agent-distance-paired-audit-20261020/REPORT.md)
- [论文 LaTeX 草稿](../../../../../ResearchWorkflow/evidence/agent-distance-paired-audit-20261020/PAPER_READY.tex)
- [四任务汇总 CSV](../../../../../benchresults/agent_distance_paired_audit_20261020/summary.csv)
- [15K 逐 case 配对 CSV](../../../../../benchresults/agent_distance_paired_audit_20261020/paired_cases.csv)
- [输入哈希和统计证据](../../../../../benchresults/agent_distance_paired_audit_20261020/audit.json)
- [离线 scorer 重算核验](../../../../../benchresults/agent_distance_paired_audit_20261020/scorer_recheck.json)

历史 Module 1 表与 A6 v2 严格评分口径不同，不能混用：A6 的距离基线 Task SPL 为 81.03%，InternVL3.5-14B 为 76.33%。A6 比较两套归档工作流，不构成 reranker 独立消融。
