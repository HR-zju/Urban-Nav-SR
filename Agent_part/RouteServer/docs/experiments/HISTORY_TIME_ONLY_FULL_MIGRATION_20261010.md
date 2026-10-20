# 实验记录：全量 observation 旧标注时间关联迁移

experiment_record_version: 1

## 实验的目的

将全部原始 observation 与现有标注按同数据集、同 clip、≤100ms 关联。保留旧原文和来源，
旧 QC 仅作为历史诊断。本任务不调用 LVLM，不导出正式训练集，不宣称标签正确率。

## 实验的 setting

- Urban-Nav-SR 基线 b6ba1c2，已有迁移 dirty changes 保留；本轮增加磁盘索引和流式批量入口。
- 本地原始清单：SideWalk 34,256、Foxglove 44,036、EgoWalk 281,190，共 359,482 observation。
  不使用 direction12 或 Stage II sidecar 筛选后的清单。
- 服务器：经 lpai-jump 连接 32011，hostname 为 lsm-a100-copy-copy-0。
- 输入根：`/lpai/volumes/mind-eb-bd-ga/suh/dataset/urbannav_history_4s_2hz_v1`。
- 标注根：`/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation/annotation_runs`。
- 已现场核实 SideWalk navigation repair 只有预检报告，无 annotations/views；main baseline 为
  此盘可用汇总来源。extra 使用有完整 annotation 的 v2，Foxglove 使用 routed_visual full v1。
- EgoWalk 使用 12-way 的活动 service_0/service_1，按源 ID 取同 v2 合同的最后记录；
  旧版本与合同不符记录另存来源，不混入当前候选。排除显式 quarantine 和 20-way 归档目录。
- EgoWalk 曾于 2026-08-05 为 Stage I 训练抢占，不能把旧 running report 当作仍在运行或全部标注完成。
- VQA 使用各数据集 consolidated vqa_annotations；保留现有 interesting 选样范围。
- 未知旧 anchor 的时间进入 unresolved 清单，不从文件名推算时间、不伪造补齐。
- 代码入口：`src/mcp_server/data/history_candidate_batch.py`，只依赖 Python 标准库。
- 配置本地：`.artifacts/history-strict-20261010/full_migration_server.json`。
- 服务器产物计划：`/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation/annotation_migrations/history_time_only_20261010_r1`。
- CPU 数据处理，GPU/optimizer/seed/checkpoint 为 N/A；既有 split 保持不变。

### 可视化与导出计划

产物目录保存 progress、按文件候选/未匹配/历史版本、逐 observation coverage、字段冲突来源、
review.html、visualization manifest/index。媒体定位引用相对 dataset root 的 package manifest
与行号；原标注的历史路径只保留为来源证据。不开公开服务。

## 记录实验结果

两个 focused checks 通过，覆盖无 sidecar、旧 rejected 原文保留、按文件 resume 和 extra 帧号规范化。
输入版本通过 path/bytes/mtime 冻结；产物逐文件完成后落 receipt，支持串行断续处理。
服务器 `report.json` 已生成，`progress.json` 为 complete，后台进程已退出；11 份来源文件完成。

| 数据集 | 新 observation | 有 reasoning 候选 | 有 VQA 候选 | 至少一种标签 | 尚未关联标签 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SideWalk | 34,256 | 34,256 | 13,292 | 34,256 | 0 |
| Foxglove | 44,036 | 42,196 | 18,438 | 42,196 | 1,840 |
| EgoWalk | 281,190 | 251,751 | 69,611 | 259,052 | 22,138 |
| 合计 | 359,482 | 328,203 | 101,341 | 335,504 | 23,978 |

本表统计时间关联覆盖，不代表字段齐全或标签正确。未关联记录保留在 coverage；旧时间未能
从索引恢复的原回答保留在 unmatched_old，不能据此认定旧标注不存在。
候选差异清单共 1,873,308 个字段项，是答案内容不同的诊断，包含同时间不同任务条件；
它既不是错误率，也不直接触发重标。teacher_calls=0，formal_training_rows=0。

服务器原始目标索引已完成：359,482 个 observation。SideWalk main 已关联 32,654 条。
extra 初次出现全未关联，现场核实原因为双前缀与未补零的帧号（151 对应 000151）；
其新清单保留相同 frame_idx 和时间。规范化能定位另外 1,602 个目标，代码已修正并续跑。
规范化后 main 32,654 与 extra 1,602 合计覆盖全部 34,256 个 SideWalk observation。
SideWalk VQA 已关联 166,554 条答案；主/extra 历史 audit 分别关联 25,147/1,460 条。
首次 SideWalk 文件 01–04 的产物移至 `file-*-before-identity-normalization` 保留诊断后重新执行。
该修正统一身份表示，时间容差仍为 100ms。后台会话为 `history-time-only-20261010`。

reasoning 文件均已完成，按新 observation 去重后覆盖：SideWalk 34,256/34,256，
Foxglove 42,196/44,036，EgoWalk 251,751/281,190。EgoWalk 两个服务文件保留早期追加记录
2,888 与 3,330 条。EgoWalk VQA 输入 1,038,696 条，生成 2,999,793 条候选关联；
多目标关联造成记录数增大，覆盖统计按 observation 去重。

`portfolio_doctor.py` 报告既有 artifact existence drift（urbannav_viz_perf_current_20260826）
与 12 项 snapshot drift；保留现有跨仓 dirty state，本轮不修改其他工作流来消除这些告警。

### 可视化与产物

服务器产物根保存逐 observation 的 coverage 和汇总 review.html；候选保留原始回答与来源。
本地小型 receipt 位于 `.artifacts/history-strict-20261010/full-time-only-migration/`；
完整候选、历史版本、未匹配原文与差异清单保留在上述服务器目录的 `data/` 下。

## 分析实验结果，可能的原因

全量目标索引完成、现有来源迁移完成、每个 observation 标签齐全是不同结论。
历史 interesting VQA 和 EgoWalk 中断会形成真实标签缺口，不能通过放宽时间容差掩盖。

EgoWalk 实测同 clip、同时间有多个不同 source ID（例如 frame 000030 的三个后缀不同的
sample）。按时间关联保留全部并列目标，并写入 `ambiguous_time_match=true`；原始回答及其
来源保持独立。不同 instruction/任务条件是否允许复用由后续校准判定，当前关联不证明语义一致。
因此候选记录数可以超过 observation 数，最终覆盖按 observation 去重统计。

### 后续只读核查：自身任务覆盖与校准边界

时间关联覆盖与同 source_group_id 的自身标签覆盖分开统计。后续核查结果为：

| 数据集 | 缺自身 reasoning 记录 | 自身 reasoning/VQA 均缺失 |
| --- | ---: | ---: |
| SideWalk | 0 | 0 |
| Foxglove | 1,840 | 1,840 |
| EgoWalk | 29,651 | 22,732 |
| 合计 | 31,491 | 24,572 |

- Foxglove 的 1,840 个缺口全部不在旧 teacher 的 stamped_input 中，最近旧时刻均超过
  500ms；证据支持旧输入覆盖不同，而非 100ms 匹配失败。
- EgoWalk 时间关联完全缺失的 22,138 个样本均在旧输入中，集中于后段 16 个 clip；
  输出缺失与旧训练抢占记录一致。另有 594 个样本仅关联其他 instruction 的标签。
- EgoWalk 有 6,919 个样本保有自身 VQA、缺自身 reasoning。缺记录数量不等于缺字段数量，
  已有记录也不代表全部 section 完整或语义正确。
- 1,873,308 个字段差异全部属于 EgoWalk；96,125 组同时间多样本均包含不同 instruction。
  比较还分别计入原回答与结构化字段，因此差异数量不作为重标数量。
- 当前实现保留全部时间候选。自身候选与其他任务参考的显式分组、LVLM 字段级校准和必要补标
  属于后续工作；本提交未实现这些步骤，也未改变旧训练准入规则。

核查 receipt：本地小型产物目录中的 `calibration_scope_review.json`。先复用已有标签，
检查新输入下的适用性，再补缺失或修正受影响字段；不将基础补标池理解为全套标签重做。

## 用户讨论与决策记录

| 时间 | 用户讨论/关注点 | 最终决定 | 对实验的影响 | 未解决问题 |
| --- | --- | --- | --- | --- |
| 2026-10-10 | 按单 clip 原则完整迁移全部数据 | 原始 observation 全量时间关联 | 旧审核仅诊断，保留全部来源，不跑 teacher | 实际覆盖统计与缺失标注 |
| 2026-10-10 后续核查 | 查看缺口和并列候选，再确定校准范围 | 区分自身来源与同时间其他任务来源 | 保留全部候选；缺记录优先补标，已有字段优先复用 | 分组与 LVLM 校准待实现，teacher 保持关闭 |

## 接下来要做的实验

| 优先级 | 下一实验 | 假设 | 唯一主要变量 | 所需证据 | 通过/停止条件 | 依赖或授权 |
| --- | --- | --- | --- | --- | --- | --- |
| 已完成 | 串行全量迁移 | 原采样时刻支持关联 | 数据覆盖范围 | 11 份文件 receipt、coverage、report.json | 可用来源已遍历，缺口显式保留 | 已授权并执行 |
| 后续 | 讨论 LVLM 校准 | 旧标签可在新输入下复用 | 校准策略 | 原答案与校准对照 | 证据不足保持待定 | 本轮不执行 |
