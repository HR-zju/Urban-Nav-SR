# UrbanNav 数据集标注状态

更新时间：2026-08-01 23:03 CST。

本文维护服务器上 Reasoning V4.6 Compact FullCoT 与 VQA V2 的运行快照。它是运行证据索引，不替代
[`REASONING_V4_DATA_ANNOTATION_DESIGN.md`](REASONING_V4_DATA_ANNOTATION_DESIGN.md)
中的 schema、teacher、QC 和训练合同。每次刷新必须以服务器报告和 JSONL 行数为准，不能只
根据 tmux 是否退出判断完成。

## 1. 状态口径

| 状态 | 含义 |
| --- | --- |
| `未开始` | 尚未创建该数据集的正式 run root。 |
| `预处理中` | 正在做确定性方向投影、stamp、source build 或媒体检查，尚未调用 LVLM。 |
| `预处理失败` | 前置合同失败且任务已退出，尚未调用 LVLM。 |
| `标注中` | source coverage 已通过，LVLM worker 正在写 annotation。 |
| `标注行完成` | 每个 source 都有一条 annotation record，但不代表 section/QC/release 通过。 |
| `QC 待处理` | annotation 已写完，仍有 reject、conflict 或人工 reconciliation。 |
| `可发布` | completeness、artifact audit 和 formal-training gate 均通过，且已导出 formal view。 |

禁止把 `record_complete`、`all_sections_accepted`、`candidate_structurally_valid` 和
`formal_training_eligible` 当成同一个计数。训练只消费明确导出的 formal view；原始
annotation JSONL 不是默认训练 manifest。

## 2. 当前服务器与模型

| 项目 | 当前值 |
| --- | --- |
| 逻辑节点 | `leap-a800-2-2` |
| 实测主机 | `root@172.28.4.47:2222`，经 `wangjingke@8.162.3.74:12087` |
| 实测 hostname | `cce-ig-x78qqhwq-mzceh1rr-j22mvq8a-06` |
| 基线 commit | `fe39c954e61efa9edc6d481f00681ab15d9172c3`；EgoWalk 启动另含已测试 direction-source hotfix |
| 模型服务 0 | `http://127.0.0.1:18461/v1`，`qwen3.5-122b-a10b-behavior` |
| 模型服务 1 | `http://127.0.0.1:18462/v1`，`qwen3.5-122b-a10b-instruction` |
| 图片策略 | `reasoning_v4_section_images_routed_visual_v1` |
| 标注并发 | 当前 VQA 每服务 16，总客户端并发 32；EgoWalk FullCoT 恢复后为每服务 8、总并发 16 |
| 当前标注 tmux | `vla-annotation-vqa-interesting` |
| 最近服务器实测 | `2026-08-01 23:03:27 CST`；VQA queue、两个 worker 和两个模型服务均存活 |
| 最新连接检查 | `2026-08-01 23:03:27 CST`；经跳板机成功连接 `172.28.4.47:2222` |
| 当前活动阶段 | `vqa_label_foxglove`；EgoWalk FullCoT 已暂停，等待 VQA 队列完成后自动续跑 |

## 3. FullCoT 数据状态

| 数据集 | 输入行 | annotation 行 | 关键 QC | 当前状态 | 正式 run root |
| --- | ---: | ---: | --- | --- | --- |
| Foxglove | 47,249 | 47,249 | admission v2：`all_sections_accepted=47,249`；`candidate_structurally_valid=46,628`；`pending_human=13,087`；`formal_training_eligible=34,162` | `QC 待处理`；行级 completeness passed，但仍有 blocking action conflicts，尚未导出正式训练 view | `annotation_runs/reasoning_v4_6_qwen122b_foxglove_20way_routed_visual_full_v1` |
| SideWalk main | 45,397 | 45,397 | admission v2：`all_sections_accepted=35,038`；`candidate_structurally_valid=33,997`；`pending_human=10,012`；`formal_training_eligible=25,026` | `QC 待处理`；action reconciliation 仍 `passed=false`，另有 NavigationContext section gap | `annotation_runs/reasoning_v4_6_qwen122b_sidewalk_main_20way_routed_visual_full_v1` |
| SideWalk extra unique | 2,611 | 0 | 输入和 20-way profile 已检查 | `已排队`；当前 VQA 队列明确将该 FullCoT 阶段延后，尚无活动 worker | `annotation_runs/reasoning_v4_6_qwen122b_sidewalk_extra_20way_routed_visual_full_v1` |
| EgoWalk | 284,470 | 7,076 | `record_complete=7,076`、`record_error=0`、`all_sections_accepted=6,743`、`candidate_structurally_valid=6,733`、`pending_human=1,221`；rejected side records 为 343 | `已暂停`；service 0/1 分别停在 3,547/3,529 行，VQA 三数据集完成后自动断点续跑 | `annotation_runs/reasoning_v4_6_qwen122b_egowalk_12way_routed_visual_full_v1` |
| FrodoBots | 未纳入 | 0 | continuous trajectory/System1 不可信；当前正式队列未包含 | `未开始`；不得借用 FrodoBots trajectory 生成 strong action | 无 |

结论：当前不能表述为“已上传的数据全部完成 FullCoT”。已完成行级 annotation 的只有
Foxglove 和 SideWalk main。EgoWalk 完成 7,076 行后被当前 VQA 生产队列暂停；SideWalk extra
尚未开始。FrodoBots 不在本轮可信强动作 FullCoT 队列。

当前 VQA 队列证据：

```text
tmux: vla-annotation-vqa-interesting
launcher: Urban-Nav-SR/Agent_part/RouteServer/tools/model_processing/
          run_vqa_v2_qwen122b_interesting_queue.sh
queue: foxglove VQA -> sidewalk VQA -> egowalk VQA -> resume EgoWalk FullCoT
```

该队列设置了 `RESUME_EGOWALK_COT=1`。SideWalk extra FullCoT 在当前队列中明确为 deferred，
不能再把旧的独立轮询 tmux 表述为当前活动队列。

### 3.1 `QC 待处理` 的准确含义

`QC 待处理` 不表示 LVLM API 仍在运行，也不等同于整行标注失败。它表示每个 source 已有
annotation record，但从原始标注到正式训练 view 仍需经过以下逐级门禁：

```text
record_complete
  -> all_sections_accepted
  -> candidate_structurally_valid
  -> Action Reconciliation
  -> review_status in {not_required, resolved}
  -> formal_training_eligible
  -> formal view export + artifact audit
```

Action Reconciliation 是 annotation-only 的确定性审计，不是第七个 CoT section，也不计算 LM
loss。它比较 LVLM 的 visual-grounding-conditioned Decision proposal 与私有 strong action / direction
sidecar，主要阻断 stop/move 冲突、方向符号冲突、超出人工送审阈值的角度差和强动作约束冲突。
`unsupported`、`mixed`、`unknown` probe 当前只作审计，不能单独判定为人工必审。

当前已完成数据的发布差距为：

| 数据集 | 行级完成 | section 全接受 | candidate valid | 待人工 reconciliation | 当前 formal eligible |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foxglove | 47,249 | 47,249 | 46,628 | 13,087 | 34,162 |
| SideWalk main | 45,397 | 35,038 | 33,997 | 10,012 | 25,026 |

因此 Foxglove 的 47,249 行和 SideWalk main 的 45,397 行都不能直接整体作为正式训练集。
Foxglove 主要剩余 action reconciliation；SideWalk main 还存在 section/composition gap。只有
经过 formal exporter 明确写出的 view 才是默认可训练输入。

本次使用 `urbannav_action_reconciliation_review_admission_v2` 重算，没有修改 annotation 或重新
调用 LVLM。旧规则把任意 probe finding 都送人工；新规则把
`strong_action_probe_unsupported/mixed/unknown` 降为 Grounding QC advisory。结果是 SideWalk
减少 1,563 条待人工，Foxglove 减少 3,848 条待人工。新报告保留在：

```text
<run_root>/reports/action_reconciliation_report.admission_v2.json
<run_root>/review/action_reconciliation_queue.admission_v2.jsonl
<run_root>/annotations/action_reconciliation_audits.admission_v2.jsonl
```

角度策略有两个不同口径，不能混为一个阈值：

- direction profile compatibility tolerance 为 `10°`；`delta <= 10°` 记为
  `within_tolerance`，不会因角度本身进入人工队列；
- 当前独立 human-review angle threshold 为严格 `delta > 20°`；`10° < delta <= 20°`
  单独存在时也不送人工；
- `delta <= 10°` 仍可能因 stop/move、符号或强约束等**其他独立冲突**送审。本次重算后，
  SideWalk 这类样本为 801 条、Foxglove 为 413 条，全部是
  `strong_action_constraint_conflict`，不是角度误判。

### 3.2 SideWalk section/composition gap

SideWalk 的 45,397 行中，TaskGrounding、SceneGrounding、LocalTraversability 和
InteractionConstraintGrounding 都完成。缺口集中在 NavigationContextGrounding：

| 原因 | 行数 | 含义 |
| --- | ---: | --- |
| `instruction_task_direction_without_explicit_direction` | 8,367 | LVLM 把通用“选择局部方向”模板当成了具体方向 instruction，但文本没有 left/right/straight 等显式方向。 |
| `bev_route_task_direction_without_visible_authority` | 1,991 | LVLM 从 BEV 声称 route direction，但 source evidence 仍是 `legacy_bev_unknown`，没有声明 frame、ego-forward 或 route-direction authority。 |
| 图片解码 provider error | 1 | 一张输入图无法识别。 |

这 10,359 行的 NavigationContextGrounding 被 validator 拒绝，Decision 因依赖未闭合而跳过，
所以 candidate 状态为 `not_available`。另有 1,041 行已完成六段 section，但 composer 检出
`strong_action_constraint_conflict`，因此 candidate 为 `invalid`。这些问题不会被调整角度 QC
阈值修复：前两类需要修正 SideWalk BEV authority/source contract 并重跑失败 section，不能把
原有 annotation 直接当作完整 FullCoT。

2026-08-01 对 original input、stamped input、V4 source、rejected annotation 和媒体文件做了
逐行 provenance 审计，结论如下：

- 8,367 条 instruction 失败行在服务器本次 original input 和 V4 source 中都使用同一条通用
  “choose one local motion direction”模板，且没有显式 left/right/straight/forward/turn 指令。
  对应的本地 recovered upstream 45,397 行中，这 8,367 条统一为“follow the visible route”，
  同样有 0 条包含显式方向。说明 20-way 打包时发生过通用模板改写，但没有丢失任何原本存在的
  具体方向文本。旧 user prompt 中的候选项和旧 assistant CoT 不是当前 active instruction。
  该类是 LVLM 把任务格式误当成具体方向的 contract violation，不是文本字段丢失；validator
  拒绝正确。
- 1,991 条 BEV 失败行的 BEV 文件在本地和服务器均存在、均可解码，且逐文件 SHA-256 完全
  一致。原始行保留了嵌套的 `sidewalk_meta.bev_patch_meta_path` 和渲染后的 route overlay，
  但没有顶层 `bev_route_evidence`、`bev_route_overlay`、`bev_image_alignment`，因此 V4 source
  全部 fail-closed 为 `frame=unknown`、`ego_forward=unknown`、
  `route_direction_authority=none`、`producer_contract_id=legacy_bev_unknown`。这是 producer
  authority 元数据未迁移，不是 BEV 图片缺失。不能仅因图片中存在蓝色 route polyline 就批量
  默认成 `ego_aligned`；必须从权威 renderer/pose 元数据恢复 frame 和 ego-forward 后才能开放
  BEV direction supervision。后续回查原始 SideWalk producer index 已确认：这 1,991 条以及完整
  SideWalk main 45,397 条均可在对应 `chunk_*/<clip_id>/bev_index.json` 找到精确 frame entry，
  全部为 `schema_version=sidewalk_bev_index_v1`、`bev_alignment=ego_aligned`、
  `draw_trajectory=true`、`bev_route_scope=global_full` 且 route point count 为正。1,991 条归档 BEV
  与 raw producer BEV 的 SHA-256 也全部一致，因此可以从 producer index 做 deterministic contract
  recovery，不需要重新渲染这些图片。
- 唯一 provider decode error 已定位为
  `sidewalk:2025-09-15__C10343__c60cd4cb-4c35-463a-a542-e9e275fa5713__1757979510217546240:000000`。
  其五帧 padding FPV 均为有效 `1920x1080 JPEG`，但当前 BEV 在本地和服务器都是 0 字节，
  SHA-256 均为 `e3b0c442...b855`。进一步回查 chunk_29 raw producer 后，原始
  `bev_images/000000_bev.jpg` 仍是有效的 194,836 字节 JPEG，SHA-256 为
  `e9504012...1377b6`。因此这是多帧归档复制损坏，不是 raw 数据缺失；应从 raw producer 恢复
  本地归档文件并同步服务器，恢复后再重跑该行。

服务器审计产物保存在：

```text
<sidewalk_run_root>/reports/navigation_context_failure_provenance_audit.json
<sidewalk_run_root>/reports/navigation_context_failure_provenance_rows.jsonl
<sidewalk_run_root>/reports/navigation_context_failure_media_hashes.server.jsonl
```

## 4. EgoWalk 当前运行

正确链路为：

```text
egowalk_fpv_only_source.jsonl
  -> project_reasoning_v4_ego_directions.py
     profile=ego_degree10_12way_terminal_centroid_v2
     trusted source=egowalk_odometry
  -> stamp_reasoning_v4_directions.py
  -> build_reasoning_v4_sources.py
  -> 2 x Qwen3.5-122B worker, concurrency=8/service
  -> reconciliation/completeness reports
```

首次失败尝试保留在：

```text
annotation_runs/reasoning_v4_6_qwen122b_egowalk_20way_routed_visual_full_v1
```

该尝试错误地请求 FPV 20-way profile，在 direction stamp 阶段退出，没有调用 LVLM，不能作为
缓存或训练输入。当前正式投影与 run root 为：

```text
annotation_inputs/reasoning_v4_6/egowalk_ego_12way.jsonl
annotation_inputs/reasoning_v4_6/egowalk_ego_12way.projection_report.json
annotation_runs/reasoning_v4_6_qwen122b_egowalk_12way_routed_visual_full_v1
```

旧 `pixel_trajectory` proxy direction 仅保留在 `direction_projection.legacy_direction` 中审计；
正式 strong direction 从可信 metric `trajectory_ego_t` 确定性派生。LVLM 不读取该私有 GT 来
生成视觉事实。

本次全量投影的 12-way 分布为：

| label | 行数 | label | 行数 |
| --- | ---: | --- | ---: |
| `stop` | 4,445 | `straight` | 99,166 |
| `left_10` | 36,254 | `right_10` | 73,687 |
| `left_20` | 10,280 | `right_20` | 30,984 |
| `left_30` | 3,753 | `right_30` | 12,949 |
| `left_40` | 1,412 | `right_40` | 6,046 |
| `left_50plus` | 1,136 | `right_50plus` | 4,358 |

最近一次实时核验（2026-08-01 23:03:27 CST）：

| 检查项 | 实测结果 |
| --- | --- |
| tmux | 无活动 EgoWalk FullCoT worker；当前 GPU 由 `vla-annotation-vqa-interesting` 使用 |
| source coverage | 284,470 / 284,470，`passed=true` |
| stamped input | 284,470 行 |
| worker shards | service 0/1 各 142,235 行 |
| annotation | service 0 为 3,547 行；service 1 为 3,529 行；合计 7,076 行 |
| rejected | service 0 为 156 行；service 1 为 187 行；合计 343 行 |
| LVLM 请求 | 2026-08-01 17:12 CST 左右暂停；现有 report 的 `status=running` 是中断前快照，不代表 worker 仍存活 |
| GPU | GPU0–7 当前用于 VQA；每卡约 78.8 GiB，实测利用率 66%–100% |
| 暂停前吞吐 | 两路 report 合计约 4,198 rows/h |
| 恢复后 ETA | 剩余 277,394 行；若维持暂停前吞吐，约需 66 h，该 ETA 从实际恢复时刻重新起算 |

当前 7,076 行中 `record_error=0`，但只有 6,743 行达到 section 全接受、6,733 行达到 candidate
structurally valid，另有 1,221 行进入人工 reconciliation。reject 主要来自
`NavigationContextGrounding`，另有少量 candidate construction failure；暂停快照不能外推最终
接受率。

## 5. 输入模态覆盖

| 数据集 | FPV | temporal FPV | instruction | BEV | waypoint guidance | strong direction / System1 来源 |
| --- | --- | --- | --- | --- | --- | --- |
| Foxglove | 5 帧 | 有 | 有 | 有 | 无 | 可信 GT trajectory / FPV direction profile |
| SideWalk | 5 帧 | 有，部分边界为 repeated padding | 有 | 有 | 无 | 可信 pose trajectory / FPV direction profile |
| EgoWalk | 5 帧 | 有，序列边界可 padding | 有，来自 caption 条件化文本 | 无 | 无 | `egowalk_odometry` / ego 12-way profile |
| FrodoBots | 5 帧 | 有 | 有 | 有 | 无 | continuous trajectory 禁用；只保留获准的视觉/BEV监督 |

这里的“instruction-only”仅表示 EgoWalk 的**导航 guidance 模态**只有 instruction；模型输入
仍包含 FPV，训练 sidecar 仍可包含可信 odometry action。它不表示纯文本样本。

当前 staged manifest 的 `modality:waypoint` 均为 0。future odometry 是动作 target，不是独立
planner guidance，禁止把同一未来轨迹同时复制为模型输入 waypoint。

## 6. Behavior filter 与 VQA 候选池

服务器上的 Qwen3.5-122B behavior filter 已按**单个 current FPV**完成以下逐数据集判定；这里的
`interesting` 是视觉复杂度候选标签，不是 VQA 答案，也不是 FullCoT/QC 通过标志：

| 数据 | 输入 | interesting | boring | 备注 |
| --- | ---: | ---: | ---: | --- |
| SideWalk main | 45,397 | 17,535 | 27,862 | 完整 report |
| SideWalk extra 原始集 | 4,340 | 2,724 | 1,616 | 与 main 有重叠；不能直接和当前 2,611 unique extra 相加 |
| Foxglove | 47,249 | 20,680 | 26,569 | 完整 report |
| FrodoBots | 11,093 | 2,662 | 8,431 | 只可使用获准的视觉/BEV监督边界 |
| EgoWalk part A | 100,000 | 21,176 | 78,824 | 完整 report |
| EgoWalk part B | 184,470 | 45,080 | 139,390 | 完整 report |

EgoWalk 合计为 284,470 输入、66,256 interesting、218,214 boring。上表原始行合计为
392,549 输入和 109,857 interesting，但由于 SideWalk main/extra 重叠，它不是去重后的训练池
规模。服务器上的 `behavior_filter/combined/interesting_all.jsonl` 当前只有 9 行，是不完整的
旧合并产物，禁止作为 VQA 输入；应从各数据集完整 `interesting.jsonl` 按 stable sample ID
重新 join、去重并生成审计报告。

### 6.1 VQA V2 当前运行

当前生产队列已经完成 stable-ID selection、Reasoning source build 和 VQA source build，正在对
interesting 候选池做 VQA teacher 标注。这里的“运行中”只表示生产链路已启动，不等于 VQA
training view 已通过人工语义验收或 release gate。

| 数据集 | 已准备 source | 当前状态 |
| --- | ---: | --- |
| Foxglove | 20,680 | 标注中 |
| SideWalk | 19,248 | 等待 Foxglove |
| EgoWalk | 66,256 | 等待 SideWalk |
| 合计 | 106,184 | 顺序队列 |

Foxglove 最近快照（2026-08-01 23:03 CST）：

| 指标 | service 0 | service 1 | 合计 |
| --- | ---: | ---: | ---: |
| completed source | 7,800 | 7,900 | 15,700 |
| accepted QA annotation | 93,498 | 94,374 | 187,872 |
| task-level schema error | 151 | 148 | 299 |

因此 Foxglove source 进度为 `15,700 / 20,680 = 75.92%`。task-level error 约占已经尝试
QA 的 `299 / (187,872 + 299) = 0.16%`，主要是 `d4_applicable_constraint` 的
`affected_direction_labels` 超出 question candidates，另有少量 reason-code 格式错误。每个 source
会根据 eligibility 生成不同数量的 QA，不能用 annotation JSONL 行数除以 source 行数计算进度。

Foxglove 从 17:38 CST 开始标注，当前平均约 2,900 source rows/h；按该速度剩余约 1.7 h。
若后续数据集的 eligible task 组合和服务速度近似，三批 VQA 的总剩余时间约 31 h。该估计按
source 完成数计算；SideWalk/EgoWalk 的 task mix 不同，切换数据集后必须重新测量。

`interesting` 应作为 VQA 的**高价值优先候选池**，尤其服务于障碍、交互、规则、局部通行性
等视觉任务，但不能成为所有 VQA task 的唯一分布。instruction grounding、方向 E0 和正常可通行
判断仍需 boring/normal control，否则模型会学到“所有场景都复杂或受阻”的选择偏差。当前 run
只负责生产 interesting VQA annotation，结束后仍需：修复或重试 task-level schema error、按任务
抽样做语义 review、补充 boring/normal controls、导出 versioned training view，并运行 artifact
audit。未经这些门禁，当前 annotation JSONL 不能直接作为正式训练 manifest。

Canonical 语义和验收边界见
[`VQA_V2_DATA_ANNOTATION_DESIGN.md`](VQA_V2_DATA_ANNOTATION_DESIGN.md)。

## 7. 运行与核验路径

服务器工作区：

```text
/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation
```

数据集根目录：

```text
/lpai/volumes/mind-eb-bd-ga/suh/dataset/urbannav-sidewalk-agent-data-v1
```

EgoWalk source：

```text
manifests/source/egowalk_fpv_only_multiview_ms_swift/egowalk_fpv_only_source.jsonl
```

每次更新本表时至少核验：服务器时间、tmux 和实际进程、source/annotation 行数、两个 worker
report、completeness report、action reconciliation report，以及 `/v1/models` 返回的模型身份。
表中时间是最近一次服务器实测快照，不应被解释为后台自动刷新的监控面板。
