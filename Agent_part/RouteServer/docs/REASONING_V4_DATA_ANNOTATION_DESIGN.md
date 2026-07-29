# UrbanNav Reasoning V4.6 数据标注与训练合同

更新时间：2026-07-29。

本文是 Stage-A 单轮 E2E 导航 FullCoT 的 canonical 合同。V4.6 的目标不是表达一个
理论上完备的因果图，而是监督当前输入可回答、当前流水线能稳定构造、并且会影响短时
导航动作的最小语义集合。

代码权威：

- `src/mcp_server/contracts/reasoning_v4.py`：六段 target；
- `src/mcp_server/contracts/reasoning_dataset_v4.py`：source、annotation、review 与 view；
- `src/mcp_server/data/reasoning_v4_task_canonicalization.py`：任务规范化；
- `src/mcp_server/teacher/reasoning_v4_profiles.py`：独立 section teacher；
- `src/mcp_server/data/reasoning_v4_composer.py`：strong direction 与行为 tuple；
- `src/mcp_server/data/reasoning_v4_reconciliation.py`：不可变动作审计；
- `src/mcp_server/data/reasoning_v4_state.py`：review/training 派生状态；
- `src/mcp_server/data/reasoning_v4_views.py`：preview/formal view；
- `src/mcp_server/agent/reasoning_v4_text.py`：严格六 tag wire；
- `src/mcp_server/agent/reasoning_v4_validation.py`：语义 gate。

## 1. 版本与唯一 wire

```text
target:        urbannav_reasoning_trace_v4_6
source:        urbannav_reasoning_source_v4_5
annotation:    urbannav_reasoning_annotation_v4_6
audit:         urbannav_action_reconciliation_audit_v3
resolution:    urbannav_action_review_resolution_v2
training view: urbannav_reasoning_training_view_v4_6_compact_v1
```

source envelope 保持 `v4_5` 是有意的：它与 VQA V2 共用且本轮没有改变 prompt-visible
输入结构。target、annotation、audit、resolution 和 training view 已整体升级，不能凭
source 文件名把旧 V4.5 annotation/cache 误当作 V4.6。

assistant target 只有六段：

```text
<think>
<TaskGrounding>{...}</TaskGrounding>
<SceneGrounding>{...}</SceneGrounding>
<LocalTraversability>{...}</LocalTraversability>
<NavigationContextGrounding>{...}</NavigationContextGrounding>
<InteractionConstraintGrounding>{...}</InteractionConstraintGrounding>
<Decision>{...}</Decision>
</think>
```

V4.6 没有 `Finish`、readout、semantic ref graph、object ID、route progress 或时序标签。
动作只出现一次：`Decision.selected_action_label`，通过普通 LM loss 学习。`stop` 表示
本轮短时保持停止，不表示任务完成。

## 2. 第一性原理边界

六段共享读取同一份完整 prompt-visible 输入，各自独立完成职责。只有 Decision teacher
额外读取五段已验证 grounding；它仍看不到 private strong direction。

允许：

- instruction 的确定性任务规范化；
- FPV 可见场景、支撑面、障碍、净空和当前交互；
- 朝向明确的可见 BEV route overlay 对短时路线方向和 upcoming topology 的 weak 支持；
- 既有数据中已经冻结的 direction stamp；
- 既有 `trajectory_ego_t` 作为可选 System1 sidecar。

禁止构造或声称：

- GPS、pose、calibration、route matching、route theta；
- route progress、remaining distance、arrival truth；
- pixel-goal/waypoint 派生方向；
- mask、bbox、instance GT、tracking、ego-motion、TTC；
- metric clearance、calibrated confidence；
- history-frame temporal relation target。

历史 FPV 是模型输入上下文，但不是 tracking/TTC/时序真值。BEV 权限固定为：

> A prompt-visible BEV route overlay may support route direction and upcoming
> topology when its orientation is clear. The absence of local safety
> authority only prevents BEV from proving traversability, clearance, or
> collision safety.

## 3. 输入与 private sidecar

### 3.1 `NavObservationV4`

| 字段 | 定义 | 来源 | Student 可见 | Loss |
| --- | --- | --- | --- | --- |
| `source_group_id` | 同源样本稳定 ID | dataset | 是 | 无 |
| `dataset_source` | 数据集族 | dataset | 是 | 无 |
| `source_record_id` | 原始记录 ID | dataset | 是 | 无 |
| `split_group_id` | 防止同 clip 跨 split | dataset | 是 | 无 |
| `images[]` | 有序图像，恰有一个 current FPV | dataset | 是 | 输入 |
| `image_roles[]` | `fpv.current/history_*`、可选 BEV/goal | deterministic | 是 | 输入 |
| `instruction` | 样本真实指令，可为空 | dataset | 是 | 输入 |
| `travel_mode` | 当前步行辅助语境 | config | 是 | 输入 |
| `model_visible_context` | 额外可见非私有上下文 | dataset/adapter | 是 | 输入 |
| `evidence_catalog[]` | 可见输入地址簿 | deterministic | 是 | 输入 |
| `input_modalities` | 实际存在模态 | deterministic | 是 | 输入 |
| `direction_context` | action/probe profile ID 与版本 | registry/config | 是 | 输入 |
| `temporal_evidence_eligible` | VQA V2 兼容元数据 | adapter | 不被 FullCoT 使用 | 无 |

`evidence_catalog` 只用于组织输入，不再被六段输出引用。BEV 中保留的 legacy
route-progress presentation metadata 仅服务共享 VQA source 兼容，Reasoning V4.6 不消费。

### 3.2 `DirectionSupervisionV4`

这是 private deterministic strong label：

```text
profile_id
profile_version
label_index
label_name
choice
angle_deg
source
bin_mode
derivation_version
```

当前 allowlist：

| Profile | Geometry source | Stop rule |
| --- | --- | --- |
| `fpv_degree10_terminal_centroid_v2` | existing FPV pixel trajectory terminal centroid | displacement `<0.06` normalized FPV xy |
| `ego_degree10_12way_terminal_centroid_v2` | existing `trajectory_ego_t`, x-right/y-front terminal centroid | displacement `<0.25m` |

V4.6 不从 row trajectory、BEV、GPS、route、goal 或 LVLM 回答现场补方向。strong label
不进入 teacher/student 输入，只在 composer、audit 和训练 view 中作 identity 校验。

## 4. 六段字段合同

所有 `ReasonCode` 是受 lower-snake-case 约束的开放 code，不是人为闭集选择题。真正闭集
只用于 wire、状态机、action vocabulary、affordance/constraint effect 和行为 tuple。

### 4.1 `TaskGrounding`

科研问题：当前 instruction 的规范化任务是什么？

| 字段 | 定义 | 真值/标注来源 | Student 可推断 | Loss |
| --- | --- | --- | --- | --- |
| `objective` | 稳定任务 objective code | versioned registry；未知 fingerprint 只标一次并缓存 | 是 | LM |
| `requirements[]` | instruction 明示要求 | registry/cache | 是 | LM |
| `ambiguities[]` | 会改变动作的任务歧义 | registry/cache | 是 | LM |

规范化固定执行 NFKC、case-fold、标点/符号转空格、空白折叠和 SHA-256 fingerprint。
已知通用路线指令固定为：

```json
{
  "objective": "follow_route",
  "requirements": ["route_adherence"],
  "ambiguities": []
}
```

registry v2 将精确常用指令
`Navigate safely on the sidewalk and follow the visible route.` 规范化为上述结果。恢复
append-only cache 时，旧 annotation schema 或已进入 registry 的旧 LVLM task 结果均视为
stale，必须由当前 deterministic canonicalizer 重建。

相同 fingerprint 不能逐样本重复调用 LVLM。不存在 goal object、local objective、
runtime phase、goal-distance threshold 或 waypoint phase GT。

### 4.2 `SceneGrounding`

科研问题：对当前导航有用的可见环境摘要是什么？

| 字段 | 定义 | 真值/标注来源 | Student 可推断 | Loss |
| --- | --- | --- | --- | --- |
| `scene_type` | 开放粗场景 code | LVLM weak | 是 | LM |
| `walkable_support` | 当前可见步行支撑面 code | LVLM weak FPV | 是 | LM |
| `key_entities[]` | 影响局部导航的粗类别，不是实例 | LVLM weak FPV | 是 | LM |
| `traffic_context` | 当前粗交通控制语境 | LVLM weak FPV | 是 | LM |

本段没有 entity/control ID、位置、mask、bbox、metric depth、tracking 或 confidence。

### 4.3 `LocalTraversability`

科研问题：当前 FPV 对少量固定方向 probe 提供什么物理可执行性证据？

每个 `direction_affordance`：

| 字段 | 定义 | 真值/标注来源 | Student 可推断 | Loss |
| --- | --- | --- | --- | --- |
| `direction_label` | action profile 对应 7-way probe 中的 moving label | registry + LVLM weak | 是 | LM |
| `affordance` | direct/conditional/blocked/forbidden/unknown | LVLM weak FPV | 是 | LM |
| `clearance` | sufficient/narrow/insufficient/unknown | LVLM weak FPV | 是 | LM |
| `limiting_factors[]` | 仅物理原因闭集 | LVLM weak FPV | 是 | LM |

probe 不含 `stop`。`limiting_factors` 只允许 static obstacle、dynamic occupancy、narrow
clearance、non-walkable surface、curb、wall、vegetation、occlusion 或 unknown；交通规则
不属于物理 limiting factor。本段不使用 BEV 证明安全，也不声称 mIoU 或分割真值。

### 4.4 `NavigationContextGrounding`

科研问题：当前可见输入是否给出了可用的短时任务方向？

| 字段 | 定义 | 真值/标注来源 | Student 可推断 | Loss |
| --- | --- | --- | --- | --- |
| `task_direction.status` | aligned/conflict/not_visible/insufficient_evidence | LVLM weak | 是 | LM |
| `task_direction.direction` | aligned 时唯一 moving action label | 显式方向 instruction 或朝向明确的 visible BEV | 条件可见 | LM |
| `task_direction.source` | `bev_route`/`instruction`/`none`，记录方向证据来源 | LVLM weak + validator | 是 | LM |

通用 “follow route” 指令不能单独推出左右方向。aligned 时必须有 moving `direction`，
且 source 只能是 `bev_route` 或 `instruction`；其他状态必须为
`direction=null, source=none`。NavigationContext 是唯一使用 `image_detail=high` 的
section teacher，避免 prompt-visible BEV route overlay 在 provider 侧缩放后不可辨认。
这只保留 route direction/upcoming topology 证据，不授予 BEV traversability、clearance
或 collision-safety 权威。本段没有 goal alignment、source alignments、route progress、
arrival 或跨视角 route matching。

### 4.5 `InteractionConstraintGrounding`

科研问题：当前可见高层行为限制如何影响本轮候选方向？

每个 `constraint`：

| 字段 | 定义 | 真值/标注来源 | Student 可推断 | Loss |
| --- | --- | --- | --- | --- |
| `kind` | 开放粗约束类别 | LVLM weak | 是 | LM |
| `status` | applicable/uncertain | LVLM weak | 是 | LM |
| `effect` | slow/yield/stop/avoid | LVLM weak | 是 | LM |
| `affected_direction_labels[]` | 仅 avoid 可给非空方向子集 | LVLM weak + validator | 是 | LM |

`constraints=[]` 是合法负判断。slow/yield/stop 必须使用空 affected list；avoid 必须列
非空受影响子集。crosswalk 属于 walkable support，不是 entity；需要表达当前过街影响时
使用 `crossing_interaction`。这里不做 instance tracking、ego-motion、TTC 或 frozen
normative-policy GT。

### 4.6 `Decision`

科研问题：本轮唯一短时 action 与最小行为修饰是什么？

| 字段 | 定义 | 真值/标注来源 | Student 可推断 | Loss |
| --- | --- | --- | --- | --- |
| `selected_action_label` | action profile 中的唯一短时动作 | final candidate 由 strong direction 独占 | 是 | LM |
| `speed` | normal/slow/stop | applicable constraint 确定性映射 | 是 | LM |
| `response` | proceed/slow/yield/detour/stop | applicable constraint 确定性映射 | 是 | LM |

Decision teacher 先在看不到 strong label 时给出 `DecisionProposalV4`，只用于审计。composer
绝不让 proposal 覆盖 strong direction：

```text
no applicable effect -> normal / proceed
slow                 -> slow / slow
yield                -> slow / yield
avoid                -> slow / detour
strong stop          -> stop / stop
```

strong moving label 与 applicable global stop 冲突时 composition invalid 并进入 review；
不得把 moving label 静默重写为 stop。V4.6 没有 decision_type、motion、social_action、
fallback、support refs 或 critical uncertainties。

## 5. Action Reconciliation

Action Reconciliation 是独立 annotation-only audit，不是第七段，也不进入 LM loss。
`ActionReconciliationAuditV4` 只保存不能由 parent payload 直接重建后省略的审计事实：

| 字段 | 作用 |
| --- | --- |
| `audit_id` | source/proposal/groundings/checks 的 content address |
| `source_fingerprint` | parent source identity |
| `proposal_digest` | blind proposal digest，不复制 proposal |
| `groundings_digest` | 五段 grounding digest，不复制五段 |
| `comparison` | angular delta、stop/move mismatch、sign conflict |
| `probe_projections[]` | strong action 的全部等距离最近 probe 及其物理摘要 |
| `checks` | proposal relation、probe support、constraint consistency、tuple consistency |
| `review_trigger_codes[]` | 排序、去重的确定性触发原因 |

等距离最近 probe 不得任意选一个。例如 `right_20` 同时投影到 `right_10/right_30`。
聚合 support 只有 `all_supported/all_unsupported/mixed/unknown/not_applicable`；
`mixed/all_unsupported/unknown` 都触发 review。profile stamp、proposal、strong label 和
candidate 仍由 parent join 读取，不在 audit 中重复。review priority/status 是派生值，
不持久化在 audit。

人工处理使用 append-only `ActionReviewResolutionV4`：

```text
audit_id
review_status = resolved | rejected
resolved_action_label
resolution_code
reviewer
resolved_at
note
```

若 reviewer 认为 strong label 错误，不能在 resolution 中偷偷替换 source；必须修正上游
direction source、生成新 fingerprint、重新标注和重新审计。resolution 与 frozen strong
不一致时 training fail closed。

## 6. 独立状态轴

以下状态不得折叠成一个 `accepted`：

| 轴 | 值 | 语义 |
| --- | --- | --- |
| record | complete/error | bounded pipeline 是否运行完成 |
| section | accepted/rejected/error/skipped | 每段是否独立成功 |
| candidate structure | valid/invalid/not_available | strong-owned candidate 是否可构造 |
| review | not_required/pending_human/resolved/rejected | 审计是否完成 |
| formal eligibility | true/false | canonical record 是否满足 formal 条件 |
| artifact status | preview_only/training_ready | 当前导出物属于 review 还是训练 |
| artifact eligibility | true/false | 当前导出物是否允许进入 trainer |

正式 eligible 当且仅当：

```text
record complete
AND exact six stages accepted
AND candidate structure valid
AND action audit exists and reconstructs exactly
AND review is not_required or resolved
AND resolved action still equals frozen strong direction
AND candidate revalidates against source/profile
```

`review_preview` 永远为
`artifact_status=preview_only, artifact_training_eligible=false`，但可以显示独立计算出的
`formal_training_eligible` 供审阅；它不能因此进入训练。`formal_training` 必须为
`artifact_status=training_ready`，且 artifact/formal eligibility 都为 true。ms-swift 在
Pydantic default 前检查原始 envelope，拒绝 preview、缺失 digest、pending review 或
stale schema。

## 7. Training view 与 loss

V4.6 每个 eligible source 最多产生一个 `full_cot_sft` row。view 保存：

- 输入 identity、images/roles、instruction、modalities、direction context；
- exact six-section `assistant_text` 与 typed target；
- prompt profile/digest、renderer ID、canonical contract digest；
- strong direction stamp 与 action audit ID；
- 可选现有 System1 `trajectory_ego_t/trajectory_policy/action_mask`。

assistant message 使用 `reasoning_v4_6_sections` span strategy，六段均为 LM loss。动作只在
Decision 中训练，不增加独立 direction head，也不通过 Finish 重复监督。System1 sidecar
存在时继续走既有 trajectory loss；它与 FullCoT LM action 是不同训练通道。

带 System1 sidecar 的 view 固定：

```text
trajectory_ego_t = 16 steps
action_mask = 16 steps
prompt-side readout = [TRAJ_0] ... [TRAJ_15]
```

query block 只放在最后一个 user turn，assistant 中不得出现，LM label 为 `-100`。query
计数与 trajectory horizon 是两个独立冻结轴，不声称 token-step 一一对应；expanded encode
找不到完整 16-slot query 时 fail closed。

## 8. 审计、review 与发布 gate

固定展示六个指标：

```text
Sources
Candidate Structurally Valid
All Sections Accepted
Pending Human
Formal Training Eligible
Artifact Training Eligible
```

64-row diverse smoke 必须在正式批量标注前完成。runner 的 full gate 要求：

- exact 64 sources/record-complete/all-sections/candidate-structure-valid；
- `Pending Human = 0`；
- formal/artifact eligible 均为 64；
- task canonicalization drift、equal-nearest probe loss、preview training leak 均为 0；
- 64 条 formal views 通过 schema、prompt、renderer、direction 与 audit 校验；
- 人工打开 self-contained review HTML 后显式批准。

smoke/optimizer 测试只能证明流水线、格式和容量，不代表语义质量、导航收益或 benchmark
有效。真实实验仍需报告 proposal-vs-strong confusion、trigger 分布、标注延迟/成本、
方向准确率以及对 ADE/FDE、collision、route progress 的消融贡献。

## 9. 明确删除的历史设计

V4.6 不再生成或兼容以下 Reasoning target 字段：

```text
Finish / readout
goal / requirements objects with refs / ambiguities objects with refs
entities or controls with IDs and evidence refs
affordance_ref / support_type / evidence_status / evidence_refs
goal_alignment / source_alignments / route_progress
constraint_ref / source_type / source_state / relation / evidence_refs
decision_type / navigation_intent_ref / motion
social_action / fallback / critical_uncertainties
support_refs / evidence_refs
```

旧 V4.5 及更早 artifact 必须通过显式离线迁移或重新标注；不得在 V4.6 parser/model 中添加
别名，让旧数据静默进入新训练合同。VQA V2 是独立 answer-only 合同，本次升级不改变其
task schema、capability gate 或训练 adapter。
