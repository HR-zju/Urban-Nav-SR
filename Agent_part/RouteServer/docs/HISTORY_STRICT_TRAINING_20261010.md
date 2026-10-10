# 实验记录：严格复用首批训练数据导出

experiment_record_version: 1

## 实验的目的

执行用户确认的严格复用策略，将已证明输入等价的旧 VQA 答案与既有确定性方向标签导出为
实际 Stage I 训练行。输入发生变化或证据不足的答案保留在 pending manifest。
本轮没有 teacher 调用、GPU 训练或语义质量评测。

## 实验的 setting

- Urban-Nav-SR 基线 `890ed627880aa2ef9102ab790a1117c3e563701b`，使用现有未提交迁移代码。
- ms-swift 基线 `e3b6dec4cd0a938a3fbc789fd9c664c1dfa16f3c`，消费既有 Stage I manifest builder。
- 本地 Python `/home/ubuntu/anaconda3/envs/UrbanNav/bin/python`。
- 新 source 为 4s/2Hz、9 FPV，SideWalk/Foxglove 加 latest-causal BEV，原预算 1.5s。
- 输入来自 `/tmp/urbannav-canonical-migration-20261009.jjfY6B`；带 BEV 使用 `bev-contract/` 重建结果。
- 产物根 `/home/ubuntu/WorkSpace/project/SR/Code/.artifacts/history-strict-20261010`。
- 先执行已核实首个 SideWalk clip 的完整 VQA 计划；其余全量 source 先单独导出确定性 e0。
  e0 计划显式选择 task_ids，old_runs 为空，绝不据此声称旧 LVLM 已扫描或已复用。
- 全量旧 LVLM 复用依赖服务器旧记录与可信媒体身份，仍需执行。
- 训练行保留 split_group_id，尚未选择正式 train/validation/test 划分；禁止训练后将同 clip
  的其余 anchor 当作独立测试集。
- seed、optimizer、checkpoint 均 N/A：CPU 确定性元数据处理。

### 可视化与导出计划

本次为 CPU 数据迁移。使用已有 package review 定位原图，训练曲线和模型预测均 N/A。
完整计划、accepted annotation、pending question 和逐题 projection drop 保留为可定位 JSONL。
各数据集 export_report.json 记录 task/provenance 数量；最终 consumer manifest report 记录训练行数。
不启动 dashboard，不暴露媒体服务。

## 记录实验结果

### 2026-10-10 单 clip：100 ms 原采样时刻映射

用户批准同 clip、同稳定 anchor 下 `abs(t_old - t_new) <= 100_000_000 ns`，
随后要求先完成一个 clip 供人工检查，再决定批量迁移。
使用 portable manifest 保留的 `source_row.sidewalk_meta.timestamp` 恢复旧 prepared 时刻；
EgoWalk 对应 `source_row.timestamp`，单位为毫秒。已核实的物理帧时间优先，
000110 的真实帧偏移 48.151552 ms 在本次批准范围内。其余记录声明 prepared 时间对齐，
原始传感器同步与逐 JPEG 物理时刻仍属于未验证范围。

- 代码基线 `ec86136f97a9434226b16fab82afd4be67291ef4`，本轮追加 tolerance 和 anchor-map 实现。
- clip：`2025-07-13__C10003__9c6c72d9-fe02-4a37-816b-15c16effe13e__15`，26 个新 anchor。
- 旧输入：本地首 clip 快照，26 个 reasoning source、6 个 VQA source、85 个旧 VQA 答案。
- 产物：产物根下 `sidewalk-clip-100ms-review/`；复现入口 `run_clip_100ms.py`。
- 复用：26 条 Scene、6 条 Interaction，保存原文与 producer 至 `reused_sections.jsonl`。
- 训练：18 条旧 LVLM VQA（a1/a2/a3 各 6）+ 26 条既有 e0 方向标签，共 44 行。
- 待处理 VQA：346 条当前旧快照无答案、52 条题意或方向合同变化；这不是全量旧池缺失结论。
- Scene/Interaction 为独立片段，尚未生成完整 FullCoT/Decision；没有 teacher、训练或质量评测。
- `review.html` 展示每个 anchor 的 9 FPV、BEV、复用标签和详细原因，等待用户检查。
- 新训练行的媒体路径保持相对 dataset root；review 页面按本地目录关系引用媒体。

本轮新增时间边界和采样单位检查，连同相关迁移检查共 13 passed、51 deselected。
Urban-Nav-SR 初始迁移提交已 push origin；ms-swift fetch 后 HEAD 与 origin/main 一致，
既有 stage_ii_edge.toml 修改保留。31041/32011/32660 均 SSH banner 超时，服务器尚未同步。
全量标签迁移依照用户最新要求暂停，已启动的额外映射任务中止；部分映射产物保留，
不计为完整批量迁移。

### 2026-10-10 后续：按标签时刻优先复用

用户进一步明确：扩窗本身不改变当前状态标签，原 exact-input 筛选过严。
新增 `anchor_label_v1`，只对标签实际需要的 anchor、方向和时间区间判断，保留原答案/producer。
当前状态、Task 与 temporal VQA 分开处理；未启动 teacher，也没有重新生成方向训练集。
4 个 focused checks 通过（56 deselected）：覆盖 Scene 跨 prompt-policy 复用、Task、
时间冲突隔离、VQA 从新计划到离线导出的复用链路，以及既有迁移执行回归。

真实首 clip 新计划：Scene 1 个当前帧冲突、25 个 legacy 时间映射缺失；VQA 26 个 e0、
346 个旧来源无答案、52 个题意/方向合同变化、3 个当前帧冲突、15 个时间映射缺失。
新策略在真实样例中尚未增加旧 LVLM 训练行；这不表示全部旧标注不可复用。
旧 `000110` 图片的已记录 ordinal 109 与新 ordinal 110 相差 48.151552 ms，
来自本轮读取原 frame_times.npy；名义 anchor 相同而图片时刻有偏移，未自行批准近邻容差。
新计划和原因统计分别位于产物根 `sidewalk-anchor-reasoning-final/`、`sidewalk-anchor-vqa-final/`。
本轮 32011 仍发生 SSH banner exchange timeout，无法读取更多旧标注及服务器采样记录。

启动前状态：三个 SSH 端口首轮 hostname 成功；后续远端访问发生 banner exchange timeout。
本地 source 与一个真实 clip 的旧标注可用。本轮新增 offline-only 执行和 accepted-only 导出，
已有 focused fixture 通过，覆盖确切复用、确定性重建、待补标隔离和完整导出门禁。
首个 SideWalk clip 的完整计划为 442 题：26 个确定性 e0 成功导出为训练行，416 个问题
保留 pending，严格复用旧 LVLM 为 0。SideWalk 全量方向子集 30,428 行、Foxglove 40,042 行
均完成 source → annotation → view → ms-swift consumer，进程 exit 0，projection drop 为 0。
EgoWalk 168,583 行随后也完成，进程 exit 0。三个方向文件合计 239,053 行，
projection drop 为 0；以上方向专用计划的完成不代表全量语义迁移完成。

本次最小检查：既有 VQA migration fixture 1 passed（19 deselected），覆盖 offline-only
禁止 teacher、部分导出、在包含历史 teacher 答案的目录中仍保持离线筛选；git diff --check 通过。
已抽取三个数据集各一个真实 anchor，29 张图片均可读取。本次首条训练行与
2026-10-09 服务器 CPU processor/batch 已通过的行逐字段完全相同；旧 receipt 保留为
`prior_encoding_receipt_20261009.json`，没有伪称本轮重新运行 tokenizer。

服务器：首轮三个 hostname 成功，后续 31041/32011/32660 均发生 banner exchange timeout，
放宽本地网络沙箱后复核 32011/32660 仍超时。未同步新产物、未修改服务器工作区。
一次未设整体 timeout 的只读连接已中断；后续均采用短超时。
portfolio_doctor 仍报告既有 1 个缺失 artifact 与 12 项 snapshot drift；未为隐藏漂移覆盖快照。
ms-swift 的 stage_ii_edge.toml 与 CityWalker 的 pl_modules/citywalker_module.py 既有修改保持原样。

### 可视化与产物

各数据集目录保存 `train.jsonl`、`consumer_report.json` 与 `views/pending_questions.jsonl`。
保留原有 package review；本次没有新模型预测或 TensorBoard 曲线。
总结果在产物根 `generation_receipt.json`，其中 `subset_complete=true`，
`full_semantic_migration_complete=false`，`server_sync_complete=false`。
首个 clip 的训练行是 SideWalk 全量方向文件的子集，交付训练输入时仅使用三个数据集主文件。

## 分析实验结果，可能的原因

严格复用与确定性重建是两种 provenance，必须分别计数。方向训练行可用不代表完整 FullCoT
训练集完成。服务器握手不稳定尚未定位根因；不能解释成数据缺失。

## 用户讨论与决策记录

| 时间 | 用户讨论/关注点 | 最终决定 | 对实验的影响 | 未解决问题 |
| --- | --- | --- | --- | --- |
| 2026-10-10 | 严格复用并生成实际训练集 | 采用严格复用 | 只接纳已证明等价的旧答案；其余保持待补标，先导出可用子集 | 全量旧记录读取 |
| 2026-10-10 后续 | 扩展历史应优先复用同 anchor 标签 | 用 anchor_label_v1 替代新任务中的 exact-input 判据 | 当前标签保留原文；时间、方向或证据问题进入 pending | legacy 时间映射和已知帧偏移 |
| 2026-10-10 单 clip | 100ms 偏差可接受，先检查一个 clip | 同稳定 anchor 使用 inclusive 100ms 时间容差 | 先交付 26 anchor 的标签与 44 行训练子集 | 等待人工审核，服务器握手超时 |

## 接下来要做的实验

| 优先级 | 下一实验 | 假设 | 唯一主要变量 | 所需证据 | 通过/停止条件 | 依赖或授权 |
| --- | --- | --- | --- | --- | --- | --- |
| 当前 | 用户审核单 clip | 扩窗后当前时刻标签仍适用 | 100ms 时间对齐 | review.html、标签原文与 lineage | 用户指出矛盾时修正对应规则 | 等待用户审核；暂停扩大范围 |
| 后续 | 全量旧 VQA 迁移 | 相同标签目标可跨历史窗口复用 | 数据覆盖 | 逐题语义合同与来源记录 | 证据不足进入 pending | 单 clip 审核后确认；teacher 保持关闭 |
