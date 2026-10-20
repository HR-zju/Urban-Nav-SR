# 实验记录：视觉目标方向与 stop 约束局部修复

experiment_record_version: 1

## 实验元数据

- ID：`vla-section-repair-20261013`；负责人：UrbanNav annotation；状态：running。
- 原结果：`/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation/annotation_runs/vla_gap_full_20261011`，保留原始 JSONL。
- 新结果：`/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation/annotation_runs/vla_section_repair_20261013`。
- 代码：Urban-Nav-SR 基于 `081bd24e1c491796c058e0f452d85611ca04daf0`，本轮合同、Prompt、validator 和修复 CLI 为 dirty；UrbanNav-Contracts 使用 canonical exporter 同步生成物。服务器保存隔离源码。
- 验证范围：review-subset，固定 1,029 条问题记录，属于局部修复，非全量语义验收。

## 实验的目的

保留全部旧标签，修复 742 条 NavigationContext 被拒绝/Decision 缺失及 287 条移动 GT 与 stop 冲突。只重做受影响 section，不把历史 accepted 或修复后结构有效当成语义正确。

## 实验的 setting

- 输入沿用原 sources.jsonl 的 9 帧 FPV、可用 BEV 和自身 instruction；媒体相对原 history dataset root；逐项确认原 annotation/source fingerprint 一致。
- 新增来源 `visual_goal`：图像定位 instruction 指定的目标；文字明确方向仍使用 `instruction`。证据 policy v9，新增或修订的 section profile 追加 `visual_goal_stop_v1`，policy digest 随实际内容生成。
- NavigationContext 问题重做 NavigationContext+Decision，复用 Task/Scene/Local/Interaction；stop 问题重做 Interaction+Decision，复用 Task/Scene/Local/NavigationContext。原 producer、原回答与原路径保持可追溯。
- stop 按当前主体/路径/信号适用性确认，保留有证据的 stop；真实 stop 与移动 GT 冲突继续保留，不改 GT。
- 模型/环境：原 Qwen3.5-122B-A10B `26-02-24-1554`；原 `qwen35_vllm_cuda128`，TP=4，最多四卡、四 workers、32k、10 图、temperature=0、section retry=2。沿用模型默认 seed，无 bitwise 确定性声明。
- 命令：`bash code/.artifacts/vla-section-repair-20261013/run.sh`；32011 上 GPU 4–7，tmux 使用 vla 前缀。原 31041 卡全部被占用，保持其他任务原样。
- 修复输出单独 append，已有修复 ID resume；原始输入不覆盖。teacher/composer 继续并存，最终动作来源待用户确认，暂不导出训练视图。

### 可视化与导出计划

新 root 保存 `visualization_manifest.json`、started/finished receipt、controller.log；各数据集保存 repair.plan.json、repair.jsonl、repair.report.json。结束后生成 `review.html` 与 `visualization_index.json`，按 source ID 展示新旧 section/动作、输入媒体路径；文件仅通过受信 SSH 读取，模型只监听 loopback。TensorBoard N/A：仅补标，无训练。benchmark 图表 N/A：此次仅字段修复。

## 记录实验结果

全量旧结果检查：19,494 条与 source 匹配，重复/漏项/fingerprint 不一致均为零。EgoWalk 742 条 NavigationContext 受拒并跳过 Decision；Foxglove 57 条、EgoWalk 230 条存在 stop 冲突。

本地 focused tests 4 passed（39 deselected）：`visual_goal` 通过且既有 instruction 来源规则保留，仅请求 NavigationContext/Decision 时四个其他 section 保持原记录，真实 stop 与移动 GT 的冲突继续保留。独立 Contracts 包导入和完整 bundle freshness check 通过，shell 语法和 diff check 通过。

2026-10-13 14:59:46 已启动 `vla-section-repair-20261013`，15:00:21 创建 `vla-eval-1`。冻结计划为 Foxglove 57、EgoWalk 972（742 NavigationContext + 230 stop），合计 1,029 条；32011 GPU 4–7 空闲检查通过。15:02 模型处于四卡初始化阶段，尚未生成修复回答。

portfolio doctor 仍报告 1 个历史产物缺失和 11 项快照漂移；保留 CityWalker 及其他无关 dirty changes。未做全仓测试，未 commit/push。

### 可视化与产物

原始质量案例位于工作区 `.artifacts/vla-quality-20261013`；启动前创建 visual manifest。本轮真实输出、review 和完成 receipt 仅在执行后记录。

## 分析实验结果，可能的原因

视觉目标方向原合同只能声明 instruction/BEV，导致图像定位目标的回答无法表达其真实来源；新增来源解决表达缺口，仍需 teacher 重新看图确认。stop 案例存在信号主体混淆及 section 不一致的证据，修订 Prompt 后效果待验证。原 287 条可能含真实停止情形，零冲突不是成功指标。

## 用户讨论与决策记录

| 时间 | 用户讨论/关注点 | 最终决定 | 对实验的影响 | 未解决问题 |
| --- | --- | --- | --- | --- |
| 2026-10-13 | 对有问题部分完成修复，保留旧标签 | 授权局部修复 | 仅 1,029 条受影响 section；其余复用 | 最终 teacher/GT 动作政策仍待确认 |

## 接下来要做的实验

| 优先级 | 下一实验 | 假设 | 唯一主要变量 | 所需证据 | 通过/停止条件 | 依赖或授权 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 受影响字段修复 | 新来源合同与信号适用性提示能解决已定位问题 | section policy | 前后回答、未改 section 一致性、残留原因 | 保留不确定项；服务/IO 异常原样暴露 | 已授权，最多四卡 |
| P1 | 导出训练视图 | 明确的动作来源可避免监督混淆 | final action policy | 用户决策和修复核对结果 | 决策后导出 | 本轮暂不执行 |
