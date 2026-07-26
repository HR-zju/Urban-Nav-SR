# UrbanNav Reasoning V4 数据标注与训练合同

更新时间：2026-07-26。

本文是当前单轮 E2E 导航推理的 canonical 合同。V4 是新的 major version，不是对已删除
V3 代码的重命名。它只覆盖当前数据真实具备的输入和监督，不包含 agent loop、tool
response、ContextAsset、GPS、pose、calibration、route matching、route theta、arrival
state、tracker、TTC、mask、bbox 或 calibrated confidence。

本 Markdown 是可编辑的 source of truth；同目录
`UrbanNav Reasoning V4 Canonical Contract.pdf` 只是由本文生成的审阅快照，内容必须随
本文同步，不能独立定义另一套合同。

## 1. 当前实现结论

当前 V4 的职责边界是：

```text
NavObservationV4
  = student 和 LVLM teacher 共同可见的完整单轮输入

五个 grounding teacher
  = 分别直接读取同一份 NavObservationV4

Decision teacher
  = 读取同一份 NavObservationV4 + 五段已验证 grounding

composer
  = 不改写语义，只组装六段并投影 finish

private sidecars
  = 已有 direction stamp 和已有 trajectory_ego_t
```

五个 grounding section 不是严格串行链。跨段引用用于一致性、可执行性和审计；缺少非关键
跨段链接通常产生 warning，而不是把有原始证据的 section 删除。Decision 是唯一显式读取
五段结果的联合决策阶段。

## 2. 输入、弱标注和强监督

### 2.1 Student-visible `NavObservationV4`

必需：

```text
fpv.current
```

按样本真实存在时才提供：

```text
fpv.history_*
instruction.active
bev.*
goal.*
waypoint.*
```

`input_modalities` 只是输入 availability metadata，不由模型生成。V4 不为空样本合成
instruction、goal、waypoint 或 BEV。历史帧只有在至少两条不同图像路径存在时才允许支持
动态关系；重复 padding 不构成时序证据。

兼容已有 Phase1/ms-swift row 时，source builder 只允许从 user message 中显式的单行
`Instruction: ...` / `Navigation instruction: ...` 恢复 instruction；不会复制旧 CoT
模板、assistant answer，也不会用默认句子补缺失 instruction。

### 2.2 LVLM weak labels

六段语义字段来自 LVLM 对可见输入的结构化标注。它们是可训练的弱语言监督，不得称为：

```text
instance-level GT
mask/bbox grounding GT
tracking truth
TTC truth
metric geometry truth
calibrated confidence
```

### 2.3 Dataset/deterministic strong sidecars

V4 不现场从 trajectory、BEV、GPS、goal 或 LVLM 回答推方向。只有 source row 已携带完整、
合法、版本化的方向 stamp 时才复制：

| Profile | 数据集策略 | 类别 | 几何来源 | Stop |
| --- | --- | --- | --- | --- |
| `fpv_degree10_terminal_centroid_v2` / `2.0` | SideWalk、Foxglove | -90° 到 90° 每 10° 加 stop，共 20 类 | 已有 FPV pixel trajectory terminal centroid | `< 0.06` normalized FPV xy |
| `ego_degree10_12way_terminal_centroid_v2` / `2.0` | EgoWalk | stop、straight、左右 10/20/30/40/50+，共 12 类 | 已有 `trajectory_ego_t` terminal centroid | `< 0.25 m` |

FrodoBots 当前不接收方向和连续 System1 强监督。`route_theta` 在 profile、source 和
derivation 中均 fail closed。

`trajectory_ego_t` 只作为独立 System1 sidecar。它不进入 teacher prompt、六段 JSON 或
自然语言答案。

历史 source 中可能只有旧 choice/name/bin、没有 V4 profile/version/derivation 完整
stamp。此类字段不能升级为强监督：builder 保留可用 observation/System1，令
`direction_supervision=null` 并记录 `incomplete_legacy_direction_dropped`。若 row
显式声明 `direction_supervision`、`canonical_direction` 或 V4 flat stamp，则任何缺字段、
profile 不允许或 identity 不一致仍整行报错，避免损坏的 V4 标签被静默吞掉。

## 3. 引用合同

V4 严格分离两类引用：

```text
raw evidence refs:
  instruction.* / fpv.* / bev.* / goal.* / waypoint.*

semantic refs:
  task.goal
  requirement.* / ambiguity.*
  entity.* / control.*
  region.* / alignment.*
  nav.task_direction
  constraint.*
```

`evidence_refs` 永远只能引用当前 `evidence_catalog` 中的 raw ref。`support_refs`、
`related_entity_refs`、`candidate_region_refs` 等字段才承载 semantic ref。禁止把
`region.*` 或 `entity.*` 冒充底层视觉证据。

## 4. 六段字段、科研问题、来源和 loss

当前所有六段的 assistant token 都进入 LM loss。表中的 “LVLM” 表示 student-visible
weak label；“协议”表示固定 wire/state vocabulary；“raw ref” 表示确定性引用闭包。

### 4.1 `TaskGrounding`

科研问题：可见任务要求机器人完成什么，目标是否明确，有哪些显式要求和实质歧义？

| 字段 | 定义 | 来源 | Student 可推断 | Loss |
| --- | --- | --- | --- | --- |
| `objective` | 任务级目标阶段，不是局部动作 | LVLM + 版本内 coarse protocol | 是 | LM |
| `goal.goal_type` | 目标来自语言、视觉、pixel goal、waypoint、route goal 或组合 | LVLM + 输入 availability | 是 | LM |
| `goal.description` | 对可见目标的简短归一化描述 | LVLM | 是 | LM |
| `goal.specification` | specified / ambiguous / underspecified / absent | LVLM | 是 | LM |
| `goal.source_refs[]` | 直接定义目标的 raw 输入 | raw ref | 是 | LM |
| `requirements[]` | 显式 task-specific required/preferred/avoid/forbidden 要求 | LVLM，开放 `code` | 是 | LM |
| `ambiguities[]` | 会改变下一步动作的任务歧义 | LVLM，开放 `code` | 是 | LM |

`objective` 只能基于当前可见 task sources，不读取 `runtime_task_phase`、goal distance、
arrival、route state 或 future waypoint state。它是研究者定义的 coarse protocol，不是
数据集 GT；分析时必须报告各值覆盖和 teacher 一致性，不能把它当客观真值。
顶层 `destination` 若是自然语言可以作为可见目标；若是 list/object 坐标，则只有携带
显式 `goal_prompt_visible` 或允许的 `goal_source/destination_source` provenance 才能进入。
未经证明的坐标 destination 会被删除并记录 warning。预构造的 V4 row 也不能绕过这一
边界：`NavObservationV4` 会递归拒绝 GPS/pose/calibration/trajectory 等私有 key。

### 4.2 `SceneGrounding`

科研问题：FPV 中直接可见哪些导航相关实体和交通控制？

| 字段 | 定义 | 来源 | Student 可推断 | Loss |
| --- | --- | --- | --- | --- |
| `entities[].entity_ref` | 样本内语义引用键，不是 instance id | LVLM | 是 | LM |
| `entities[].category` | 开放 lower-snake-case 可见类别 | LVLM | 是 | LM |
| `lateral_position` | left/center/right/spanning/unknown 粗位置 | LVLM + 协议 | 是 | LM |
| `depth_position` | near/mid/far/unknown 粗深度 | LVLM + 协议 | 是 | LM |
| `traffic_controls[]` | 可见 control 的类别、开放状态和粗位置 | LVLM | 是 | LM |
| `evidence_refs[]` | 支持每个视觉事实的 FPV refs | raw ref | 是 | LM |

本段不输出 path blockage、traversability、tracking、motion、mask、bbox 或 metric depth。

### 4.3 `LocalTraversability`

科研问题：FPV 可见的局部区域有哪些，它们对当前短时运动是否可执行？

| 字段 | 定义 | 来源 | Student 可推断 | Loss |
| --- | --- | --- | --- | --- |
| `region_ref` | 样本内局部区域键 | LVLM | 是 | LM |
| `surface` | 开放表面/区域名称 | LVLM | 是 | LM |
| `lateral_position` | 粗左右位置 | LVLM + 协议 | 是 | LM |
| `state` | direct/conditional/blocked/forbidden/unknown | LVLM + 安全状态协议 | 是 | LM |
| `clearance` | sufficient/narrow/insufficient/unknown | LVLM + 协议 | 是 | LM |
| `related_entity_refs[]` | 可选跨段语义链接 | LVLM | 是 | LM |
| `limiting_factor_codes[]` | 非 direct 状态的开放原因 code | LVLM | 是 | LM |
| `evidence_refs[]` | 至少一个 FPV raw ref | raw ref | 是 | LM |

SAM/SoM、segmentation 和 mIoU 当前只保留未来接口，不是本段真值或当前指标。

### 4.4 `NavigationContextGrounding`

科研问题：实际提供的 task/goal/waypoint/BEV 条件如何与当前 FPV 局部场景对齐？

| 字段 | 定义 | 来源 | Student 可推断 | Loss |
| --- | --- | --- | --- | --- |
| `goal_alignment` | 目标的 resolved/ambiguous/not_visible/insufficient 状态 | LVLM | 是 | LM |
| `source_alignments[]` | 每个实际 source 的 local effect 和 aligned regions | LVLM + raw source identity | 是 | LM |
| `task_direction.desired_motion` | left/straight/right/stop/unknown 粗任务方向 | LVLM + 协议 | 是 | LM |
| `candidate_region_refs[]` | 与 task direction 相容的语义 regions | LVLM | 是 | LM |
| `status` | aligned/conflict/not_visible/insufficient_evidence | LVLM + 协议 | 是 | LM |
| `evidence_refs[]` | source ref 与必要 FPV ref | raw ref | 是 | LM |

本段不做 route matching、route theta、GPS/pose projection 或局部 free-space certification。
BEV 不能覆盖 FPV 中的可见危险。
`goal_type=none` 时 `goal_alignment` 必须为空；存在显式 goal 时必须提供
`goal_alignment`。`resolved` 只能有一个 `target_ref`，`ambiguous` 至少有两个
`candidate_target_refs`，`not_visible/insufficient_evidence` 不携带候选目标。

`candidate_region_refs` 表达“任务方向指向哪些区域”，不是安全许可。因此它允许包含
blocked/forbidden region，以保留“任务要求向前但前方被阻挡，Decision 应 stop/replan”
这类关键负例。只有 `Decision.selected_region_ref` 必须是可解析、位于 candidate 集合且
状态为 direct/conditional 的可执行区域。

### 4.5 `InteractionConstraintGrounding`

科研问题：哪些可见 task、traffic、social、accessibility 或 local-safety 关系会影响候选区域？

| 字段 | 定义 | 来源 | Student 可推断 | Loss |
| --- | --- | --- | --- | --- |
| `constraint_ref` | 样本内约束键 | LVLM | 是 | LM |
| `constraint_code` | 开放约束名称 | LVLM | 是 | LM |
| `kind` | social/traffic/accessibility/local_safety | LVLM + 协议 | 是 | LM |
| `source_refs[]` | raw 或 task/entity/control semantic sources | LVLM | 是 | LM |
| `applies_to_region_refs[]` | 被约束的局部区域 | LVLM | 是 | LM |
| `relation` | actor/control/requirement 与 candidate path 的粗关系 | LVLM + 协议 | 是 | LM |
| `temporal_basis` | single_frame/multi_frame/static_control/instruction | LVLM + 证据规则 | 是 | LM |
| `status` | applicable/uncertain | LVLM + 协议 | 是 | LM |
| `effect` | proceed/slow/yield/wait/stop/detour/request_clarification | LVLM + 协议 | 是 | LM |
| `evidence_refs[]` | 支持关系的 raw refs | raw ref | 是 | LM |

动态 `crosses_candidate_path` 或 `approaches_candidate_path` 必须引用至少两条不同的 FPV
图像证据。当前不实现 tracker、ego-motion compensation、TTC 或 frozen normative GT。

### 4.6 `Decision`

科研问题：基于完整输入和五段 grounding，下一次短时语义动作是什么？

| 字段 | 定义 | 来源 | Student 可推断 | Loss |
| --- | --- | --- | --- | --- |
| `decision_type` | Stage-A 只允许 navigate/stop；navigate 选择可执行局部区域，stop 保守保持当前位置且不宣称任务完成 | LVLM + Stage-A 协议 | 是 | LM |
| `motion` | go_straight/turn_left/turn_right/stop | LVLM + 协议 | 是 | LM |
| `speed` | normal/slow/stop | LVLM + 协议 | 是 | LM |
| `social_action` | none/yield/wait/follow/pass_left/pass_right/detour/unknown | LVLM + 协议 | 是 | LM |
| `selected_region_ref` | navigate 时唯一可执行候选 region | LVLM | 是 | LM |
| `fallback` | 证据恶化时的保守行为 | LVLM + 协议 | 是 | LM |
| `support_refs[]` | 真正影响决策的 semantic refs | LVLM | 是 | LM |
| `evidence_refs[]` | 决策的底层 raw evidence 闭包 | raw ref | 是 | LM |

Decision 不能携带 direction profile、label index、trajectory 或 confidence。若 strong
direction 与 Decision 的 coarse motion 不一致，整条 full view fail closed；composer
不会用 GT 回写或修复 LVLM Decision。

代码中的 `DecisionTypeV4` 仍保留 `finish/request_user/request_replan`，只用于未来
runtime/agentic 合同的类型兼容；它们不会出现在当前 teacher JSON Schema 或 student
prompt 的 enum 中，section validator、full target validator、training-view validator 和
ms-swift adapter 也都会拒绝它们。因此 Stage-A 数据不能借底层类型的兼容值扩张任务。

## 5. Wire、Finish 和 System1

完整输出固定为：

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
<readout>[SYS1_0] ... [SYS1_15]</readout>  # navigate 且有 trajectory sidecar
```

`finish` 确定性复制 Decision 的 `decision_type/motion/speed/social_action`。合法 direction
sidecar 存在时，`finish.direction` 再复制 `choice/label/angle_deg`。因此 direction 是
Finish 文本中的 LM target，不需要也不存在独立 direction head。

`[SYS1_i]` 是 action head 的 latent query anchor，不进入 LM loss。`trajectory_ego_t` 和
`trajectory_policy` 只位于 full view 的 sidecar；grounding/Decision section view 不携带
System1 target。若 source 虽有 trajectory，但 weak `Decision` 不是 `navigate`，full view
仍保留六段和 Finish 的 LM 监督，但省略 readout 和所有 System1 sidecar，并写入
`metadata.system1_omitted_reason=decision_not_navigate`。

## 6. Teacher、训练视图和 loss

Teacher 流水线：

```text
five independent calls:
  X -> Task
  X -> Scene
  X -> LocalTraversability
  X -> NavigationContext
  X -> InteractionConstraint

joint call:
  X + five validated groundings -> Decision

deterministic:
  six sections + private strong sidecars -> target / finish
```

每段独立重试并记录 `AnnotationStageRecordV4`。部分成功样本保留 accepted section，供质量
分析和 section-view export；只有五段、Decision 和全局 validator 都通过才生成 full view。
teacher backend 的函数参数是 `NavObservationV4`，不是 `ReasoningSourceV4`；因此
direction、trajectory、action mask 等 private sidecar 在类型边界之前已经移除，不能由
provider adapter 或自定义 backend 读取。
batch report 按 section 聚合 accepted/rejected/error/skipped、attempt、provider call、
warning/error code、round-trip latency 和 token usage。整体报告同时给出 annotation
acceptance fraction、总调用耗时与 token 数。成本不从模型名猜测：只有调用者显式提供
`--input-cost-per-million-tokens` 与 `--output-cost-per-million-tokens` 时才计算；
否则 `cost_estimate.status=unavailable_no_pricing_contract`。

当前 exporter 产生：

| View | 条件 | Assistant target | LM |
| --- | --- | --- | --- |
| `grounding_section_sft` | 对应 section stage accepted | 单个 section tag | 全开 |
| `decision_section_sft` | 五段完整且 Decision accepted | 单个 Decision tag | 全开 |
| `full_cot_sft` | annotation accepted | 六段 + finish + optional readout | 六段和 finish 开，readout 关 |

总损失可写为：

```text
L = L_full_lm
  + lambda_section * sum_s L_section_sft
  + lambda_system1 * L_system1
```

当前没有 `direction_finish_sft`：没有可靠 V4 annotation 时，不生成只有 Finish 的残缺 wire
样本。若以后需要该任务，必须定义独立 schema/profile，不能伪装成 `full_cot_sft`。

## 7. Hard error 与 warning

Hard error：

```text
schema/wire 错误
raw evidence ref 不在 catalog
视觉事实没有 FPV evidence
动态关系没有真实多帧证据
navigate selected region 不可解析、非 candidate 或不可执行
Decision support ref 不可解析
Decision 采用的 constraint 不作用于 selected region
Decision 与 applicable constraint 冲突
Stage-A 生成 navigate/stop 之外的 decision_type
strong direction stamp 不完整、profile 不允许或与 Decision 不兼容
private field 出现在 assistant text
```

Warning：

```text
独立 section 创建的 semantic ref 未被另一段复用
goal/alignment/constraint 的非关键跨段引用缺失
constraint 影响的 region 不在 task candidate 集合
历史 FPV 是重复 padding
缺 direction 或 System1 sidecar
```

## 8. 代码落点

RouteServer：

```text
contracts/reasoning_v4.py             六段、Finish、wire semantic contract
contracts/reasoning_dataset_v4.py     source/annotation/view/sidecar contract
data/reasoning_v4_source.py           数据集输入隔离与 fingerprint
teacher/reasoning_v4_profiles.py      六个 versioned teacher profiles
teacher/reasoning_v4_teacher.py       section prompt 与 parser
teacher/reasoning_v4_batch.py         五段独立 + Decision runner
data/reasoning_v4_composer.py         确定性组装
agent/reasoning_v4_validation.py      section/global validator
agent/reasoning_v4_text.py            strict renderer/parser
agent/reasoning_v4_prompt_policy.py   student system prompt profiles
data/reasoning_v4_views.py            section/full view export
data/reasoning_v4_audit.py            source/annotation/view provenance audit
tools/model_processing/
  build_reasoning_v4_sources.py       legacy dataset row -> frozen V4 source
  label_reasoning_v4.py               resumable six-stage teacher runner
  export_reasoning_v4_views.py        accepted section/full view exporter
  audit_reasoning_v4_source_coverage.py
                                      multi-dataset input/media/leakage gate
  audit_reasoning_v4_artifacts.py     cross-artifact hard gate
  build_reasoning_v4_review_html.py   self-contained human-review page
  export_reasoning_v4_contract_artifacts.py
                                      schema/golden/digest exporter and CI check
  sync_reasoning_v4_cross_repo.py     generated Python target/view/prompt mirror
schemas/reasoning_v4/                 canonical generated contract artifacts
```

ms-swift：

```text
sidewalk_nav/_generated_reasoning_v4_contract.py
                                      RouteServer-generated Pydantic contract
sidewalk_nav/_generated_reasoning_dataset_v4.py
                                      RouteServer-generated source/view contract
sidewalk_nav/_generated_reasoning_v4_prompt_policy.py
                                      RouteServer-generated student policy
sidewalk_nav/reasoning_v4.py          public facade
sidewalk_nav/reasoning_v4_codec.py    training-side strict text codec
sidewalk_nav/prompt_v4.py             public facade + local user builder
sidewalk_nav/data/reasoning_v4_manifest.py
loss_scale/reasoning_v4.py
```

`schemas/reasoning_v4/` 固定导出 target schema、training-view schema、valid/invalid golden
fixtures 和 contract/prompt digest；ms-swift 保存逐字节 artifact mirror。student prompt
中的完整嵌套字段骨架、required 标记、闭集 enum 及关键状态语义从 canonical Pydantic
schema/单源 prompt policy 自动生成，不再在两仓各写一份。
ms-swift manifest 入口先用生成的 `ReasoningTrainingViewV4` 完整校验 view，再构造消息；
不再依赖一套浅层手写 dict 检查来充当训练合同。
digest 显式包含 wire format、section definition、evidence policy、renderer/parser/
validator ID，并记录 canonical behavior-bearing source hashes。修改任何 enum、field、
validator、codec 或 prompt 文本时，必须重新生成 artifacts/Python mirrors、升级必要
policy ID，并同时通过 freshness、byte parity、generated-mirror check、valid/invalid
parser parity test。

## 9. Prompt 合同与优化原则

Prompt 是合同的可执行说明，不是字段真值的第二来源。当前 policy 版本为：

```text
section_definition = urbannav_reasoning_v4_section_definitions_v2
evidence_policy = urbannav_reasoning_v4_evidence_policy_v2
validator = validate_reasoning_target_v4_v2
student profiles = reasoning_v4_student_*_v2
teacher profiles = reasoning_v4_teacher_*_v2
```

共同 system policy 先说明任务本身：为盲人、低视力用户和行动不便用户提供安全、
可预测、无障碍且符合社会规范的短时步行导航。随后用正向规则规定模型应该如何形成决策：

1. 完整读取当前 prompt 中真实提供的 instruction、按序 FPV、可选 BEV/goal/pixel-goal/
   waypoint 和 evidence catalog。
2. FPV 决定即时局部安全；任务/地图类 source 提供方向，但不能证明不可见或已阻塞区域
   可通行。
3. `evidence_refs` 必须逐字复制 catalog 中的 raw ref；semantic ref 只组织推理对象，
   自身不能创造证据。
4. 五个 grounding section 各自读取完整输入并只回答自己的科研问题；它们不是一条
   “上一段答案成为下一段唯一输入”的串行链。
5. 单帧可以支持占用、邻近、静态阻挡和粗 clearance；crossing/approaching 等动态关系
   必须有不同历史 FPV，证据不足时改用静态关系或 `uncertain`。
6. Interaction 必须形成 `source -> candidate region -> relation -> effect`，不能因为
   看见实体或交通设施就机械生成规则。
7. Decision 联合原始输入与五段结果，只输出 `navigate` 或 `stop`；前者必须选择可解析且
   可执行的 region，后者不表示完成、重规划或发起人机交互。

禁止项压缩为一条边界：只使用 prompt-visible evidence，连续轨迹、未来 pose、方向标签
metadata、segmentation 等 private sidecar 不得进入 reasoning text。具体私有 key 的
fail-closed 列表仍由 source/view validator 维护，不在 prompt 中重复罗列，以免禁止项
挤占核心导航策略。

三类输出 profile 分工如下：

| Profile | 输入 | 唯一合法输出 |
| --- | --- | --- |
| full | 完整 observation | 六个固定 tag、`finish`、满足显式请求且 navigate/System1 eligible 时的唯一 readout |
| grounding | 完整 observation | 一个指定 grounding tag |
| Decision | 完整 observation + 五段已验证 grounding | 一个 `Decision` tag |

student prompt 的字段骨架、required 标记和 enum 由 canonical Pydantic schema 自动展开；
Stage-A 对 `DecisionV4.decision_type` 和 `FinishV4.decision_type` 显式投影为
`navigate | stop`。teacher 的 Decision response schema 使用同一投影，避免出现“文字说
不能生成、JSON Schema 却允许生成”的冲突。任何 prompt 或投影变化都进入 policy digest，
并通过生成镜像同步到 ms-swift。

## 10. Smoke 与大规模标注前门禁

先冻结 20–32 条 source coverage set，联合覆盖 SideWalk、Foxglove、FrodoBots、EgoWalk、
instruction/FPV/history/optional BEV、两个允许的 direction profile，并执行：

```bash
PYTHONPATH=src python tools/model_processing/audit_reasoning_v4_source_coverage.py \
  --source-jsonl <sidewalk-v4.jsonl> \
  --source-jsonl <foxglove-v4.jsonl> \
  --source-jsonl <frodobots-v4.jsonl> \
  --source-jsonl <egowalk-v4.jsonl> \
  --require-dataset sidewalk \
  --require-dataset foxglove \
  --require-dataset frodobots \
  --require-dataset egowalk \
  --min-source-rows 20 \
  --check-media \
  --strict \
  --report-json <source-coverage-report.json>
```

该 gate 只证明 source 可构造、媒体存在、private key 没有进入 visible context，并不证明
LVLM 标注语义正确。随后先跑小规模真实 teacher smoke，生成 append-only annotation、
training views、artifact audit report 和 self-contained HTML，人工检查六段职责、引用、
方向一致性与失败样本。只有 report 明确包含每段接受率、warning/error code、方向 coarse
mismatch、latency、token usage 和基于显式价格的成本，才允许估算或启动大规模标注。

最后在 ms-swift 使用目标 tokenizer 实测 section/finish 为正 LM token、readout token
全部为 `-100`，并完成两步训练与 generation smoke。离线 schema/unit test、source
coverage 或伪 backend 不能替代这三项真实门禁。
