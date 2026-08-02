# UrbanNav Reasoning V4.6 数据标注与训练合同

更新时间：2026-08-01。

本文是 Stage-A 单轮 E2E 导航 FullCoT 的 canonical 合同。V4.6 的目标不是表达一个
理论上完备的因果图，而是监督当前输入可回答、当前流水线能稳定构造、并且会影响短时
导航动作的最小语义集合。

代码权威：

- `src/mcp_server/contracts/reasoning_v4.py`：六段 target；
- `src/mcp_server/contracts/reasoning_dataset_v4.py`：source、annotation、review 与 view；
- `src/mcp_server/data/reasoning_v4_task_canonicalization.py`：任务规范化；
- `src/mcp_server/teacher/reasoning_v4_image_policy.py`：section 级图片路由；
- `src/mcp_server/teacher/reasoning_v4_profiles.py`：独立 section teacher；
- `src/mcp_server/data/reasoning_v4_composer.py`：strong direction 与行为 tuple；
- `src/mcp_server/data/reasoning_v4_reconciliation.py`：不可变动作审计；
- `src/mcp_server/teacher/reasoning_v4_review*.py`：两阶段 LVLM filtering；
- `src/mcp_server/data/reasoning_v4_review.py`：diagnosis/adjudication/repair lineage；
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
blind review:  urbannav_action_blind_review_v1
diagnosis:     urbannav_action_review_diagnosis_record_v1
adjudication:  urbannav_action_adjudication_v1
repair:        urbannav_reasoning_repair_request_v1
training view: urbannav_reasoning_training_view_v4_6_compact_v1
```

source envelope 保持 `v4_5` 是有意的：它与 VQA V2 共用且本轮没有改变 prompt-visible
输入结构。target、annotation、audit、review sidecars 和 training view 已整体升级，不能凭
source 文件名把旧 V4.5 annotation/cache 误当作 V4.6。

方向合同独立版本化，不能从 V4.6 wire 版本猜测：

```text
urbannav_direction_profile_registry_v1  # 已有正式标注：FPV-20 + Ego-12
urbannav_direction_profile_registry_v2  # 新实验：FPV/Ego × 20/12-way
```

registry-v1、对应 student prompt-v1 和 teacher profile digest 保持冻结，已有 V4.6 标注
继续按原 stamp 读取。registry-v2 使用独立的 student prompt-v2、teacher profile 和 cache
freshness stamp；不得用 v2 代码原地续写 v1 run root。

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

canonical source 和最终 student training view 始终保存完整 prompt-visible 输入。teacher
标注阶段则通过版本化 `SectionImagePolicy` 为不同 section 投影图片子集，非图像 instruction、
goal、waypoint 和可见结构化上下文仍按 source 合同提供。只有 Decision teacher 额外读取
五段已验证 grounding；它仍看不到 private strong direction。

当前注册两套不能混用缓存的 annotation policy：

| Policy | 用途 |
| --- | --- |
| `reasoning_v4_section_images_all_visible_v1` | 既有基线；每个 LVLM section 读取全部 source 图片 |
| `reasoning_v4_section_images_routed_visual_v1` | 候选优化；按 section 职责只传必要图片 |

`routed_visual_v1` 的图片合同如下：

| Section | teacher 图片输入 | 原因 |
| --- | --- | --- |
| `TaskGrounding` | current FPV + optional current BEV | 校验当前任务语境，不消费历史动态 |
| `SceneGrounding` | 5 FPV | 汇总当前场景并保留短时视觉上下文 |
| `LocalTraversability` | current FPV | 判断当前局部支撑面、净空与物理可执行性 |
| `NavigationContextGrounding` | current FPV + optional current BEV | 将显式 instruction/BEV route 映射到当前 FPV 方向 |
| `InteractionConstraintGrounding` | 5 FPV | 识别影响短动作的人车交互和动态占用 |
| `Decision` | current FPV + optional current BEV + 五段 validated groundings | 核验当前几何和路线，综合已有事实提出动作 |

图片投影会同步过滤 `image_roles/evidence_catalog/input_modalities`。当 section 不再接收历史
FPV 时，`history_fpv=false` 且 `temporal_evidence_eligible=false`；未接收 BEV 时
`bev=false`。因此不存在“图片未发送但 prompt 声称模态可用”的隐式输入。

policy ID、policy digest、section selector、profile ID 和 prompt digest 都进入 report、
stage telemetry、annotation metadata 与 cache freshness gate。更换 policy 必须新建 run
root；不能读取旧 policy annotation 作为命中缓存。

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

registry-v2 allowlist：

| Profile | Output | Geometry source | Frame/unit | Stop rule |
| --- | --- | --- | --- | --- |
| `fpv_degree10_terminal_centroid_v2` | 20-way | existing FPV pixel trajectory terminal centroid | FPV top-left, normalized xy | displacement `<0.06` |
| `fpv_degree10_12way_terminal_centroid_v2` | 12-way | same FPV producer | FPV top-left, normalized xy | displacement `<0.06` |
| `ego_degree10_terminal_centroid_v2` | 20-way | existing `trajectory_ego_t` terminal centroid | x-right/y-front, meter | displacement `<0.25m` |
| `ego_degree10_12way_terminal_centroid_v2` | 12-way | same ego producer | x-right/y-front, meter | displacement `<0.25m` |

`direction_profile_id` 必须逐 row 显式提供。dataset family 不再选择 profile；
SideWalk、Foxglove、EgoWalk 在框架层都可使用任一已注册 profile，但只有上游确实存在对应
frame/unit/producer 的 GT 时才能启用。FrodoBots 在没有获准 direction GT 前始终
fail-closed。

V4.6 source builder 不从 row trajectory、BEV、GPS、route、goal 或 LVLM 回答现场补方向。
FPV 12-way 到 FPV 20-way 只允许使用已保存的有限
`direction_raw_angle_deg`（或未来另行批准的原始 trajectory producer）；不能从
`left_50plus/right_50plus` 猜精确角度。同一 FPV 12-way profile 的 identity 投影不要求
raw angle。strong label 不进入 teacher/student 输入，只在 composer、audit 和 training
view 中作 identity 校验。

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
进入 validator 前执行确定性小词表归一化：car/sedan/parked vehicle 归为
`vehicle`，scooter/motorcycle 归为 `micromobility`，parked bicycle 归为
`bicycle`；`traffic_context` 只保留
`traffic_control_present/no_active_control_visible/unknown`。原始 provider 回答仍留在
stage record，归一化结果才进入 typed grounding 和训练目标。

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
非空受影响子集。该子集表示“选择这些粗方向时需要局部绕行”，不是禁止方向集合：
命中子集必须为 `slow/detour`，未命中可继续使用其他 applicable effect，`stop` 安全满足
avoid。constraint kind 至少把 pedestrian interaction/proximity 归为
`social_interaction`、static/sidewalk obstacle 归为 `local_obstacle`、traffic signal
归为 `traffic_control`。crosswalk 属于 walkable support，不是 entity；需要表达当前过街
影响时使用 `crossing_interaction`。这里不做 instance tracking、ego-motion、TTC 或
frozen normative-policy GT。

### 4.6 `Decision`

科研问题：本轮唯一短时 action 与最小行为修饰是什么？

| 字段 | 定义 | 真值/标注来源 | Student 可推断 | Loss |
| --- | --- | --- | --- | --- |
| `selected_action_label` | action profile 中的唯一短时动作 | final candidate 由 strong direction 独占 | 是 | LM |
| `speed` | normal/slow/stop | applicable constraint 确定性映射 | 是 | LM |
| `response` | proceed/slow/yield/detour/stop | applicable constraint 确定性映射 | 是 | LM |

Decision teacher 先在看不到 strong label 时，基于 current FPV、可选 current BEV 和五段
validated groundings 给出 visual grounding-conditioned `DecisionProposalV4`，只用于审计。
它可以核验当前几何和 route alignment，但不能新增五段中不存在的实体、通行性、路线方向
或交互约束。composer 绝不让 proposal 覆盖 strong direction：

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
| `proposal_digest` | visual grounding-conditioned proposal digest，不复制 proposal |
| `groundings_digest` | 五段 grounding digest，不复制五段 |
| `comparison` | angular delta、stop/move mismatch、sign conflict |
| `probe_projections[]` | strong action 的全部等距离最近 probe 及其物理摘要 |
| `checks` | proposal relation、probe support、constraint consistency、tuple consistency |
| `review_trigger_codes[]` | 排序、去重的确定性触发原因 |

等距离最近 probe 不得任意选一个。例如 `right_20` 同时投影到 `right_10/right_30`。
聚合 support 只有 `all_supported/all_unsupported/mixed/unknown/not_applicable`。audit 继续
保存 `strong_action_probe_unsupported/mixed/unknown`，但 admission-v2 将这三类定义为
Grounding QC advisory；它们单独出现时不再强制人工 Action Reconciliation。方向 profile 的
10° compatibility tolerance 用于记录 `proposal_strong_relation`，因此 `delta <= 10°` 不会因
角度本身送审；当前独立 human-review angle threshold 仍为严格 `delta > 20°`。sign conflict、
stop/move mismatch、constraint conflict 和 tuple conflict 是 blocking trigger，不受角度放宽
影响。profile stamp、proposal、strong label 和 candidate 仍由 parent join 读取，不在 audit
中重复。review admission policy、priority/status 都是派生值，不持久化在 audit。

### 5.1 LVLM filtering：只诊断，不改 target

deterministic audit 出现 blocking trigger 后，样本为 `pending_human`；仅含 Grounding QC
advisory 的样本保持 `not_required`。LVLM filtering 是 annotation-only
质量控制层，不是 FullCoT 第七段，不进入 student prompt、assistant target 或 LM loss。
它固定运行两步：

1. `blind_visual` 只读取完整 prompt-visible observation 与全部可见图片，不读取 proposal、
   strong direction、trajectory、audit、candidate 或人工结论；
2. `comparative_diagnosis` 再读取 blind assessment、五段 normalized groundings、
   Decision proposal、strong direction stamp、deterministic audit 与 stage provenance。

LVLM 顶层 verdict 严格只有四类：

```text
teacher_proposal_error
grounding_error
strong_label_error
unresolved
```

constraint 问题属于 `grounding_error/interaction_constraint_error`；证据不足属于
`unresolved` reason code；多组件问题写入 `affected_components[]`，不得新增第五类。
LVLM 输出不得携带 corrected/suggested/replacement/new value，也不能直接改变 release
状态。blind 与 comparative 的 profile、prompt digest、图片 policy 和 provider telemetry
均写入 `ActionReviewDiagnosisRecordV4`，与 source fingerprint 和 audit ID 绑定。

### 5.2 人工 adjudication 与 append-only lineage

self-contained review HTML 先展示全部可见图片和 blind assessment，再显式 reveal
proposal/strong/audit/comparative diagnosis。浏览器只下载未绑定 JSONL；导入 CLI 必须
重新 join 当前 source、annotation 和 diagnosis，验证所有 identity 后才能 append
`ActionAdjudicationV4`：

```text
source_group_id
source_fingerprint
audit_id
diagnosis_id
diagnosis_disposition = confirmed | overridden | deferred
final_verdict = exact four-way taxonomy
error_subtype
affected_components[]
reason_codes[]
repair_action
fields_to_inspect[]
reviewer
reviewed_at
note
```

只有 `repair_action=keep_without_change` 能把当前 audit 变为 `resolved`。其余动作都把
旧 lineage 标记为 `rejected`：

- `rerun_teacher_proposal`：只重跑 Decision，重新 compose/audit，append 新 annotation；
- `relabel_grounding`：只重跑人工选中的 LVLM grounding，再重跑 Decision 与
  compose/audit；deterministic Task 必须回其 registry 修复；
- `rebuild_strong_sidecar`：禁止原地修改，必须修正上游 direction source、生成新 source
  fingerprint，并做完整重标注；
- `quarantine`：保留审计证据但不进入 formal training。

repair request 是 adjudication 的确定性投影。旧 annotation、audit、diagnosis 和
adjudication 永不覆盖，始终保留完整父子 identity。

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
AND any resolving adjudication references the exact completed diagnosis
AND resolving adjudication is keep_without_change
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

导出的 ms-swift row 同时显式写入：

```text
reasoning_mode = reasoning_v4_6
response_contract = reasoning_v4_6_compact
required_loss_scale_strategy = reasoning_v4_6_sections
action_supervision_location = decision.selected_action_label
direction_profile_id = per-row profile
```

这组字段是一个不可交叉拼装的实验合同。Legacy FullCoT 使用另一组
`legacy_full_cot / legacy_think_finish / ignore_empty_think /
finish.direction`；它不是 V4.6 wire 的兼容别名。

带 System1 sidecar 的 view 固定：

```text
trajectory_ego_t = 16 steps
action_mask = 16 steps
prompt-side readout = [TRAJ_0] ... [TRAJ_15]
```

query block 只放在最后一个 user turn，assistant 中不得出现，LM label 为 `-100`。query
合同固定 `query_target_alignment=query_i_to_target_i`：expanded hidden state
`H[TRAJ_i]` 作为第 i 个连续轨迹 target 的 step-local condition。query count、trajectory
horizon、action-mask length 和有效 mask 和都必须为 16；expanded encode 找不到完整
16-slot query 或任一轴不一致时 fail closed。旧实验中 query 数与 horizon 不同的 row 只能
走显式 legacy pooled compatibility，不得伪装为 V4.6 formal row。

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

256–512 条自然随机 pilot 必须在正式批量标注前完成，不再使用 historical-interesting
富集。runner 的 pilot gate 要求：

- exact source/record/all-section/candidate counts 可闭合；
- 报告真实 pending-human 比例与三个主要 trigger 的自然分布；
- formal exporter 对 pending/rejected 的泄漏为 0；
- task canonicalization drift、equal-nearest probe loss、preview training leak 均为 0；
- formal views 通过 schema、prompt、renderer、direction 与 audit 校验；
- 报告 action tuple conflict、词表未归一化和七 probe 同质率；
- 人工打开 self-contained review HTML 后显式批准。
- 对 triggered audit 先运行 blind/comparative LVLM filtering，再导出人工 JSONL；
- browser 导出的 JSONL 必须经过 canonical importer，不能直接作为 release sidecar；
- repair/quarantine 后旧 audit 必须保持 rejected，只有新 source/annotation lineage 可重新入 gate。

QC 工具链固定为：

| 步骤 | CLI | 产物 |
| --- | --- | --- |
| deterministic projection | `export_reasoning_v4_reconciliation_review.py` | audit + pending queue + report |
| two-stage LVLM | `label_reasoning_v4_review_diagnoses.py` | append-only diagnosis JSONL |
| human workspace | `build_reasoning_v4_review_html.py` | self-contained HTML + browser-downloaded unbound JSONL |
| canonical import | `import_reasoning_v4_adjudications.py` | append-only bound adjudication JSONL |
| repair routing | `export_reasoning_v4_repair_requests.py` | immutable repair requests |
| partial repair | `repair_reasoning_v4_annotations.py` | appended annotation revisions；strong rebuild 只报告 upstream-blocked |
| release | `export_reasoning_v4_views.py` + `audit_reasoning_v4_artifacts.py` | formal views 与 gate report |

修改 section 图片输入属于 annotation policy 变更，必须额外使用同一组 source ID 做配对
A/B。至少报告：

- BEV route grounding 覆盖率和逐样本一致性；
- dynamic interaction constraint 非空率、effect 分布和逐样本一致性；
- all-section accepted、candidate structurally valid；
- Decision proposal 与 frozen GT action exact agreement；
- retry row、pending-human 和 review trigger 分布；
- provider calls、prompt/completion token、stage latency 与 rows/hour。
- strict raw-response parse、stable sample-ID 顺序与集合；
- direction accuracy、macro-F1、balanced accuracy、stop/non-stop、
  speed/response，以及按 dataset/profile/response-contract 分片；
- 20-way 到 12-way 的显式 common-space accuracy；包含 `50plus` 的 12-way 记录不报告为
  精确角误差。

只有 routed policy 的质量指标未越过预设退化阈值、人工复核变更样本通过，且吞吐收益达到
目标，才允许在新的全量 run root 中替代 all-visible policy。正在运行的旧 policy 全量任务
不得原地改 profile、digest 或缓存。

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
