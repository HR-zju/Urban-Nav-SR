# UrbanNav Reasoning V4.2 数据标注与训练合同

更新时间：2026-07-27。

本文是 Stage-A 单轮 E2E 导航 Full CoT 的 canonical 可编辑合同。V4.2 是 breaking
schema；不得与 V4.1 dual-write、静默转换或混入同一训练 manifest。实现权威位于
`src/mcp_server/contracts/reasoning_v4.py` 和
`src/mcp_server/contracts/reasoning_dataset_v4.py`，本文解释每个字段解决的科研问题、
标签来源、student 可见性和 loss。

## 1. 第一性原理边界

当前系统只监督已有输入可回答的问题：

- 当前 instruction/goal 明示了什么；
- FPV 直接看见了什么；
- FPV 对 public direction labels 提供什么局部可执行性证据；
- visible task source 与方向如何对齐；
- visible interaction/control 对当前动作或方向子集施加什么约束；
- 综合上述可见证据，当前应立即执行什么动作。

V4.2 不构造或声称 GPS/pose/calibration、route matching、route theta、metric geometry、
mask/bbox、instance tracking、ego-motion compensation、TTC、calibrated confidence、
arrival truth 或 agent-loop state。六段语义均是 LVLM weak label，不是 instance-level GT。

## 2. 版本与 wire

```text
urbannav_reasoning_trace_v4_2
urbannav_finish_v4_2
urbannav_reasoning_source_v4_2
urbannav_reasoning_annotation_v4_2
urbannav_reasoning_training_view_v4_2
```

固定顺序为：

```text
<think>
<TaskGrounding>{...}</TaskGrounding>
<SceneGrounding>{...}</SceneGrounding>
<LocalTraversability>{...}</LocalTraversability>
<NavigationContextGrounding>{...}</NavigationContextGrounding>
<InteractionConstraintGrounding>{...}</InteractionConstraintGrounding>
<Decision>{...}</Decision>
</think>
<finish>{...}</finish>
optional <readout>...</readout>
```

section 顺序、tag、schema version、closed action/state vocabulary 是 wire contract。
视觉类别、目标描述、原因、`support_type/source_type/source_state` 使用受约束开放
`ReasonCode` 或自然语言，不压成人工小枚举。

## 3. 输入权限与标签来源

### 3.1 Student-visible observation

`NavObservationV4` 至少包含唯一 `fpv.current`，按样本真实存在时才包含
`fpv.history_*`、instruction、BEV、goal 或 waypoint。`evidence_catalog` 是输出可引用
raw evidence 的闭包；`direction_output_space` 只公开 profile/version/allowed labels，
不公开选中 label、index、choice、angle、source 或未来轨迹。

system prompt 对 BEV 的权限原文是：

> A prompt-visible BEV route overlay may support route direction and upcoming
> topology when its orientation is clear. The absence of local safety authority
> only prevents BEV from proving traversability, clearance, or collision safety.

因此 BEV 可以在朝向清楚时提供 `route_intent` weak evidence，但不能提供
traversability、clearance 或 collision-safety truth。pixel goal 和 waypoint 当前没有
冻结 frame/unit/derivation contract，不能生成方向 label。

### 3.2 Private deterministic sidecars

`direction_supervision` 只能复制数据 source 已有、完整、版本化的 direction stamp：

| Profile | 数据集 | 强标签来源 | Stop |
| --- | --- | --- | --- |
| `fpv_degree10_terminal_centroid_v2` / `2.0` | SideWalk、Foxglove | existing FPV trajectory terminal centroid | `<0.06` normalized FPV xy |
| `ego_degree10_12way_terminal_centroid_v2` / `2.0` | EgoWalk | existing `trajectory_ego_t` terminal centroid | `<0.25m` |

`trajectory_ego_t` 只作为 System1 sidecar。两者都不进入五个 grounding teacher、student
输入或 evidence refs。Decision teacher 可读取仅含
`demonstrated_direction_label` 的 `TeacherActionConstraintV4`，但它只是 ordinary
navigation prior；visible stop/safety constraint 可覆盖它。

## 4. 引用空间

```text
raw evidence refs:
  instruction.* / fpv.* / bev.* / goal.* / waypoint.*

semantic refs:
  task.goal
  requirement.* / ambiguity.*
  entity.eNN / control.cNN
  alignment.aNN / nav.task_direction
  constraint.kNN / uncertainty.uNN
```

`evidence_refs` 只能使用 raw refs；`support_refs` 只能使用 semantic refs。V4.2 没有
region、route segment、mask、bbox、track 或 metric geometry ref。

## 5. 六段字段合同

下表中 “LVLM weak” 表示 teacher 和 student 都从 prompt-visible 输入推断，assistant
token 进入 LM loss；“deterministic” 表示 composer/profile 确定性投影。

### 5.1 TaskGrounding

科研问题：当前可见 task source 要求什么，目标是否明确，哪些显式要求会影响导航？

| 字段 | 定义与作用 | 来源 | Student | Loss |
| --- | --- | --- | --- | --- |
| `objective` | 当前任务的开放短描述；不读取 runtime phase 或 goal distance | LVLM weak | 可见输入可推断 | LM |
| `goal` | 唯一 `task.goal`，含 `goal_type/description/specification/source_refs` | LVLM weak + raw-ref closure | 是 | LM |
| `requirements[]` | `requirement_ref/code/polarity/target_description/evidence_refs` | LVLM weak | 是 | LM |
| `ambiguities[]` | 足以改变下一步动作的 `ambiguity_ref/code/source_refs` | LVLM weak | 是 | LM |

`goal_type` 和 `polarity` 是协议轴；开放 goal/requirement 内容不枚举。不存在
`local_objective`、runtime task phase、distance threshold 或 waypoint phase GT。

### 5.2 SceneGrounding

科研问题：FPV 直接观察到哪些与导航有关的实体和交通控制？

| 字段 | 定义与作用 | 来源 | Student | Loss |
| --- | --- | --- | --- | --- |
| `entities[]` | 中性 ref、开放 category/candidates、evidence status、粗位置、FPV refs | LVLM weak | 是 | LM |
| `traffic_controls[]` | 中性 ref、开放 category/state、evidence status、粗位置、FPV refs | LVLM weak | 是 | LM |

这些字段不声明 instance GT、mask/bbox、track identity、metric depth、速度或 calibrated
confidence。可见对象也不会自动被转换为 collision 或 rule truth。

### 5.3 LocalTraversability

科研问题：当前 FPV 对 public direction label 的近场可执行性提供什么证据？

| 字段 | 定义与作用 | 来源 | Student | Loss |
| --- | --- | --- | --- | --- |
| `direction_affordances[].direction_label` | public output space 中被评估方向 | LVLM weak | 是 | LM |
| `support_type` | 开放可见 walking-support 语义 | LVLM weak | 是 | LM |
| `affordance` | direct/conditional/blocked/forbidden/unknown | LVLM weak | 是 | LM |
| `clearance` | sufficient/narrow/insufficient/unknown 的视觉粗判断 | LVLM weak | 是 | LM |
| `limiting_factor_codes[]` | conditional/blocked/forbidden 的开放原因 | LVLM weak | 是 | LM |
| `evidence_status/evidence_refs` | 证据状态和 FPV raw closure | LVLM weak + validator | 是 | LM |

当前 SAM/SoM segmentation、mIoU、mask/bbox grounding 只可作为未来独立能力，不进入本
schema。BEV 不得证明本段任何局部安全字段。

### 5.4 NavigationContextGrounding

科研问题：visible task/route source 支持哪些 public directions？本段表达意图，不授予
局部安全许可。

| 字段 | 定义与作用 | 来源 | Student | Loss |
| --- | --- | --- | --- | --- |
| `goal_alignment` | 可选 goal description/candidates、status、supported labels、raw refs | LVLM weak | 是 | LM |
| `source_alignments[].alignment_ref` | 中性局部键 | deterministic shape | 是 | LM |
| `source_ref` | 被解释的 visible raw source，必须也出现在 evidence refs | raw-ref closure | 是 | LM |
| `alignment_role` | task_intent/route_intent/local_safety_bias | protocol | 是 | LM |
| `supported_direction_labels/status` | source 对 public labels 的支持、冲突或证据不足 | LVLM weak | 是 | LM |
| `task_direction` | 非 stop 的 task/route candidate labels 与解析状态 | LVLM weak | 是 | LM |

`local_safety_bias` 不得替 task/route source生成 task candidates。朝向清楚的 visible BEV
route overlay 可用 `route_intent`；pixel goal/waypoint 必须 fail closed 为 insufficient。
每个 `source_alignment` 返回前必须自检：`status=supports` 时
`supported_direction_labels` 不得为空；没有可支持的 label 时必须改为
`insufficient_evidence`。

### 5.5 InteractionConstraintGrounding

科研问题：哪个可见关系对当前动作全局生效，或只影响哪些方向子集？

| 字段 | 定义与作用 | 来源 | Student | Loss |
| --- | --- | --- | --- | --- |
| `constraint_ref` | 中性 `constraint.kNN` | deterministic shape | 是 | LM |
| `kind` | task/social/traffic/accessibility/local_safety | protocol | 是 | LM |
| `source_type/source_state` | 开放触发对象及其可见状态 | LVLM weak | 是 | LM |
| `affected_direction_labels[]` | 只填写受影响子集；全局约束为空 | LVLM weak | 是 | LM |
| `relation` | 当前路径控制、占用、交叉、接近、邻近、净空降低、阻挡、要求冲突或 unknown | LVLM weak | 是 | LM |
| `status` | applicable/uncertain | LVLM weak | 是 | LM |
| `effect` | slow/yield/stop/avoid | LVLM weak | 是 | LM |
| `evidence_refs[]` | visible raw closure | validator | 是 | LM |

红色 pedestrian signal 控制当前路径时，canonical 记录是
`source_type=pedestrian_signal`、`source_state=red_hand`、
`relation=controls_current_path`、`effect=stop`，且
`affected_direction_labels=[]`。禁止 `affected_direction_scope`，也禁止枚举全部
non-stop labels。只有“部分方向受影响且存在替代方向”时才填写子集，例如 cyclist 影响
`left_10/left_20`，配 `effect=avoid`。动态 crossing/approach 需要至少两个不同 FPV refs；
没有 tracker、TTC 或 normative-policy GT。

`constraints=[]` 是没有可见、当前适用动作限制时的正常默认输出。`effect` 表示限制，
不是许可或通用谨慎策略。通用的 “navigate safely / follow the route” instruction、
`active_navigation_instruction`、`clear_walkable_path` 或清晰 sidewalk 不能被转换成
`slow` constraint；只有能够引用具体可见触发证据，或显式改变当前动作的用户要求，才
进入本段。

### 5.6 Decision

科研问题：计划方向、当前立即动作和轨迹示范方向可能不相同；当前应执行哪一个？

| 字段 | 定义与作用 | 来源 | Student | Loss |
| --- | --- | --- | --- | --- |
| `decision_type` | Stage-A navigate/stop | LVLM weak | 是 | LM |
| `navigation_intent_direction_label` | task/route evidence 支持的非 stop 方向，可为空 | LVLM weak | 是 | LM |
| `selected_action_label` | 当前立即执行的 public label，可为 stop | LVLM weak | 是 | LM |
| `motion` | selected action 经 profile 映射的 coarse motion | deterministic composer | 可从输出复算 | LM |
| `speed/social_action/fallback` | 当前行为策略 | LVLM weak | 是 | LM |
| `critical_uncertainties[]` | 物质性不确定项及保守 resolution policy | LVLM weak | 是 | LM |
| `support_refs/evidence_refs` | semantic 支持和 raw evidence closure | teacher + validator | 是 | LM |

全局 stop constraint 覆盖 ordinary demonstrated-direction prior。此时必须：

```text
decision_type = stop
selected_action_label = stop
motion = stop
speed = stop
social_action = wait
fallback = stop
```

`navigation_intent_direction_label` 仍可为 straight/left/right，表示约束解除后的任务意图。

Decision 返回前必须逐条应用覆盖当前动作的 applicable constraint。空
`affected_direction_labels` 全局生效；非空列表仅在 selected label 落入子集时生效：

- `slow`：`speed=slow`，或完整 stop action；
- `yield`：`social_action=yield`，或完整 stop action；
- `stop`：完整 stop tuple；
- `avoid`：必须改选子集之外的 label。

改变动作的 `constraint_ref` 必须进入 `support_refs`。

## 6. Finish、强方向和 System1

`FinishV4` 精确复制 Decision 的
`decision_type/motion/speed/social_action/selected_action_label`，并额外携带：

```text
demonstrated_direction.label
```

该 label 只能来自 private direction stamp，并通过普通 LM loss 训练；没有独立 direction
head。它可以与当前 `selected_action_label` 不同，例如红灯样本：

```text
navigation_intent_direction_label = straight
selected_action_label = stop
demonstrated_direction.label = left_10
```

navigate 且存在合法 `trajectory_ego_t/trajectory_policy` 时，full view 可追加 masked
`[SYS1_i]` readout；readout LM 权重为 0，连续轨迹只位于顶层 sidecar。stop row 不携带
System1 trajectory/readout，但六段和 Finish 仍训练 LM。

## 7. Pipeline 与质量门禁

```text
existing dataset row
  -> ReasoningSourceV4
  -> five independent grounding teachers
  -> DecisionProposalV4
  -> profile-owned composer
  -> ReasoningAnnotationV4
  -> ReasoningTrainingViewV4
  -> ms-swift manifest
```

全量标注前必须通过：

- source/view/target Pydantic 校验与 evidence closure；
- RouteServer/ms-swift generated contract、JSON Schema、golden fixture byte parity；
- prompt policy digest parity；
- 小规模真实 LVLM smoke 和人工 review；
- red-signal global stop、subset avoid、BEV route-intent 权限的语义样例；
- 真实 tokenizer 的 LM/readout mask smoke。

parser-valid 只证明格式可读，不证明 weak label 正确；mini smoke 也不能表述为 benchmark
validation 或语义收敛。
