# RouteServer 文档索引

按主题归档，实验记录与评测协议分开维护。新增文档放入对应目录并更新索引。此处是旧 RouteServer 来源仓的文档；导航新实现的归属遵循工作区 UrbanNav-SR 迁移决策，归档不改变实现责任。

| 目录 | 用途 |
| --- | --- |
| [experiments/](experiments/README.md) | 实验与结果：历史运行记录、结果表与离线核验入口。实验记录保留原有结论和适用时间；当前采用哪份结果，必须绑定具体数据与评分版本。 |
| [pipeline/](pipeline/README.md) | 数据流水线：数据构建、标注、质量筛选、迁移和发布流程。带日期或状态的段落是留存快照，执行前核验产物。 |
| [evaluation/](evaluation/README.md) | 评测协议：任务定义、预测与评分协议、benchmark 实现说明。结果表见 experiments/。 |
| [contracts/](contracts/README.md) | 数据与工具合同：系统接口、观测与动作、训练投影和轨迹数据合同。 |
| [design/](design/README.md) | 架构与设计：系统架构和设计背景；每份文档自身标明的历史适用范围仍然有效。 |
| [operations/](operations/README.md) | 运行与排错：命令与调试说明；文中相对命令默认在 RouteServer 根目录执行。 |
| [archive/](archive/README.md) | 历史参考：历史变更、阶段 checklist 和本次文档路径迁移表。 |

## 常用入口

- [评测导航](EVALUATION.md)
- [架构](design/architecture.md)
- [评测协议](evaluation/PROTOCOL.md)
- [实验结果与 A6 核验](experiments/README.md)
- [SideWalk 数据流水线](pipeline/SIDEWALK_PHASE1_DATA_PIPELINE.md)
- [标注状态](pipeline/DATASET_ANNOTATION_STATUS.md)
- [文档旧路径映射](archive/DOCUMENT_RELOCATION.json)

历史路径引用可通过迁移表查找；不改写外部历史 manifest 或迁移来源哈希。`retired_module_paths.json` 保留在原位置，供现有工具读取。

## 已合并或删除

- `DATASET_TASK_DESIGN.md`：早期 CitySeeker 迁移和 GT 构造草稿，核心内容已合并到 `URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md`。
- `INSTRUCTION_REVIEW_EXAMPLES.md`：早期指令 taxonomy review 表，当前任务类型和样例已由 benchmark 技术设计与评测协议覆盖。
- `TASK_INSTRUCTION_ROUTE_EVAL_DESIGN.md`：早期分层 benchmark 草稿，正式口径已合并到 `URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md` 和 `EVALUATION.md`。
- `URBANNAV_FAKE_EVAL_REVIEW.md`：fake evaluator smoke 记录，核心指标语义已合并到 `EVALUATION.md`。
- `URBANNAV_COMPLEX_EVAL_EXAMPLES.md`：复杂样例记录，multi-stop 与 route-update 语义已合并到正式评测文档。
- `URBANNAV_REAL_API_REVIEW.md`：真实 API review 命令草稿，当前正式采集流程以 `EVALUATION.md` 和工具脚本为准。
- `FRODOBOT_SYSTEM2_PLAN.md`：早期计划错误地把近似 FPV trajectory 当作可训练监督，已由当前 FrodoBots trust policy 取代。
- `BEV_GENERATOR_FINAL_SUMMARY.md`：一次性 BEV 生成阶段报告，现行转换链路由 `SIDEWALK_PHASE1_DATA_PIPELINE.md` 和代码维护。
- `UrbanNav Reasoning V4 Canonical Contract.pdf`：`REASONING_V4_DATA_ANNOTATION_DESIGN.md` 的重复导出，已删除以避免双源漂移。
- `coco script.pdf`：与 RouteServer 当前合同无直接关系的二进制附件，已删除。
