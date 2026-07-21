# UrbanNav Reasoning V3.3 Full CoT 数据标注设计

更新时间：2026-07-21。

当前状态：V3.3 的 source、双阶段 LVLM teacher、GT 约束的 decision rationale、canonical composer、validator、训练 view、`ms-swift` consumer、section mask 与 provenance-aware sampler 已完成原理性实现。本版本只定义 Full CoT；VQA、typed `ToolResultUse` 和 agentic profile 仍是后续独立合同。

## 1. 核心结论

V3.3 不采用“规则直接写语义答案”，也不采用“把 GT 和图片一起交给 LVLM 自由解释”。职责固定为：

```text
GT/runtime：提供不可争议的结果约束
LVLM：生成模型可见证据支持的任务语义、视觉事实、跨模态关系和原因
规则：冻结输入、闭合结构、执行 trust policy、验证引用、组合结果和序列化 finish
```

设：

```text
x = student 可见输入：instruction/caption、FPV、optional BEV/goal/waypoint/route/tool
y = teacher-only 结果：trusted trajectory/action/direction、goal_reached、runtime phase
z_obs = 不看 y 的任务与视觉观察
z_dec = 在冻结 z_obs 后，对 y 的证据约束解释
```

则数据构造为：

```text
z_obs ~ p(z_obs | x)
y = trusted_rule(runtime, dataset policy, GT)
z_dec ~ p(z_dec | frozen(z_obs), y)
student 学习 p(z_obs, z_dec, y | x)
```

隐藏 GT 只能约束最终结果，不能伪装成视觉证据。若 `z_obs` 无法支持 `y`，则保留 GT 结果，但 `Decision.rationale_status=unsupported`，关闭 Decision 与 finish 的 reasoning loss，禁止事后编造障碍物或社会规则。

## 2. Canonical Full CoT

每条 accepted Full CoT 固定包含六段和一个确定性读出：

```text
TaskGrounding
SceneGrounding
LocalTraversability
NavigationContextGrounding
InteractionAndRuleGrounding
Decision
finish
optional System1 readout
```

严格文本顺序为：

```text
<think>
<TaskGrounding>{...}</TaskGrounding>
<SceneGrounding>{...}</SceneGrounding>
<LocalTraversability>{...}</LocalTraversability>
<NavigationContextGrounding>{...}</NavigationContextGrounding>
<InteractionAndRuleGrounding>{...}</InteractionAndRuleGrounding>
<Decision>{...}</Decision>
</think>
<finish>{...}</finish>
[<readout>[SYS1_0] ... [SYS1_15]</readout>]
```

段不适用时必须输出 typed `applicable=false`，不能删段。Parser 拒绝缺段、重复、乱序、非法 JSON、旧字段和未授权 readout。

## 3. 六段职责边界

| Section | 唯一职责 | 禁止承担的职责 |
| --- | --- | --- |
| `TaskGrounding` | 解析 instruction/caption 的语义类型、显式要求、极性、指代、歧义和权威任务阶段 | 不从 trajectory 猜用户要求，不判断障碍物 |
| `SceneGrounding` | 提取 FPV/时序 FPV 实体、控制设施、场景事实，并对每个 optional modality 记账 | 不选择动作，不宣告 route corridor 安全 |
| `LocalTraversability` | 独立分类 FPV 区域为 direct/conditional/blocked/forbidden，并给出视觉上首选的 direct corridor | 不依据 GT 或 route intent 改写区域状态 |
| `NavigationContextGrounding` | 将 instruction、BEV、goal、waypoint、route、tool 各自映射到 FPV corridor，再融合 | 不判断局部净空，不覆盖 FPV hazard |
| `InteractionAndRuleGrounding` | 绑定 actor/control、受影响 corridor、时序关系和 required behavior | 不因“看到人”就自动生成 yield/wait |
| `Decision` | 保存冻结 outcome，并仅引用前五段事实给出支持原因和 rejected alternatives | 不新增实体、区域、规则或任务要求 |
| `finish` | 从 Decision 确定性序列化紧凑读出 | 不独立调用 LVLM 或重复推理 |

## 4. 三层数据隔离

### 4.1 Model-visible input

```text
ordered FPV history/current
optional BEV
instruction/caption/dialogue
optional goal/waypoint/route/tool result
evidence catalog
```

这些内容可进入 annotation teacher 和 student prompt。

### 4.2 Teacher-only fixed facts

```text
dataset trust policy
trusted trajectory/odometry action and direction
projection-QC-passed FPV trajectory corridor
authoritative goal_reached/runtime phase/active waypoint
System1 eligibility and continuous sidecars
field-level provenance
```

它们不能进入第一阶段 observation prompt，也不能出现在 student user prompt 或 assistant 自然语言中。

### 4.3 LVLM-generated semantics

```text
TaskSemanticObservation
SceneGrounding observation
LocalTraversability observation
per-source NavigationContext observation
InteractionAndRuleGrounding observation
DecisionRationaleObservation
```

其中 Decision rationale 只能引用已冻结 observation，不能增加新事实。

## 5. 三阶段 teacher

逻辑上分为三阶段；为降低调用成本，Pass A/B 在一次 multimodal API call 中共同输出，Pass C 使用第二次 call。

### 5.1 Pass A：Task Semantic Teacher

输入：完整 instruction/caption、goal 和当前图像，不含 hidden GT。

输出：

```text
applicable
instruction_type
caption_role
inferred_objective
objective_evidence_refs
requirements[text_span, polarity, target_ref, evidence_refs]
referents[phrase, resolution, resolved_ref, evidence_refs]
ambiguities[type, text_span, evidence_refs]
confidence
```

`instruction_type` 是稳定分类标签；真正的 reasoning 来自原文 span、引用、关系、极性和歧义。禁止以下捷径：

```text
demonstration stop -> 用户要求 hold_position
存在 trajectory -> 用户要求 follow_local_path
关键词 ramp -> 必然是 accessibility instruction
存在 caption -> caption 必然是导航命令
```

权威 runtime phase、goal reached 或 active waypoint 可在 composer 中覆盖 `inferred_objective`，但必须记录 runtime provenance；这不等于用 trajectory 生成 instruction 语义。

### 5.2 Pass B：Visual Observation Teacher

与 Pass A 同一次调用，输入只有 model-visible evidence。输出：

```text
SceneGrounding
LocalTraversability
NavigationContextGrounding
InteractionAndRuleGrounding
```

该阶段禁止读取或推断 hidden trajectory、action、direction、arrival 和 dataset label，也禁止输出最终 Decision。

### 5.3 Pass C：Decision Rationale Teacher

输入：

```text
frozen Pass A/B typed observation
frozen symbolic outcome: motion/speed/social_action/selected corridor
minimal authoritative runtime facts
```

不提供 raw trajectory。输出：

```text
outcome_motion
status = supported | partial | unsupported
supporting_reasons[controlled reason + evidence refs]
rejected_alternatives[motion + reason + evidence refs]
confidence
```

Pass C 只能选择前两阶段已经存在的事实。若 GT stop 但图像中 corridor clear、没有 goal reached、interaction 或 visible blockage，则必须输出 `unsupported`，不得制造“行人挡路”。

## 6. TaskGrounding 的所有权

| 信息 | 所有者 |
| --- | --- |
| `instruction.active`、`goal.active`、输入源是否存在 | source builder |
| authoritative runtime phase、goal reached、active waypoint | runtime/fixed supervision |
| `instruction_type`、caption role、显式要求、否定、指代、歧义 | LVLM semantic teacher |
| 图像相关任务阶段校验 | LVLM observation |
| trajectory-derived motion/direction | teacher-only outcome label |
| `active_navigation_refs` | deterministic source closure |
| 最终 `TaskGrounding` | composer 合并权威状态和 validated LVLM semantics |

规则不再从 trajectory 或关键词直接写 `instruction_type/local_objective/requirements`。API 失败时也不生成可训练的关键词语义 fallback；失败样本进入 retry/reject，或关闭对应 section loss。

## 7. LocalTraversability

每个 FPV region 必须且只能属于一种状态：

| 状态 | 含义 | 必需信息 |
| --- | --- | --- |
| `direct` | 当前可直接通行 | region 与 FPV evidence |
| `conditional` | 满足尚未满足的条件后可走 | `required_condition` |
| `blocked` | 路径暂时被人、车、施工或障碍占用 | `blocker_refs` 或 reason |
| `forbidden` | 地形、交通规则或无障碍约束上不可进入 | reason |

`preferred_clear_corridor_ref` 是 LVLM 根据 FPV 给出的局部视觉偏好，只能引用 direct region。它与 `Decision.selected_corridor_ref` 分离：前者不受 GT/route 反向修改，后者才是任务与安全融合后的执行 corridor。

例如：FPV center clear、BEV route 指向 right，但 right blocked：

```text
LocalTraversability.preferred_clear_corridor_ref = corridor.center
NavigationContextGrounding.fused_target_corridor_ref = corridor.right
NavigationContextGrounding.fused_execution_status = locally_blocked
Decision.selected_corridor_ref = corridor.center 或 null
```

## 8. Scene 与 optional modality 记账

`SceneGrounding` 包含：

- `scene_facts`：受控但有 evidence 的事实；
- `entities`：行人、车辆、骑行者、静态障碍等；
- `traffic_controls`：信号灯、人行横道、停止线、标志、隔离设施；
- `context_observations`：每个非 FPV source 是 `used` 或 `ignored`；
- `cross_modal_relations`：source 与 FPV region 的关系；
- `visibility/confidence`：观察质量。

只要 row 提供 BEV/goal/waypoint/route/tool，就必须恰好记账一次。无法解析关系时使用 `context_only`、`not_visible` 或 `insufficient_evidence`，不能静默省略，也不能伪造 route-to-corridor。

## 9. NavigationContextGrounding

每个真实 source 独立输出：

```text
source_type
source_ref
directive
target_corridor_ref
status
reliability
ignored_reason
evidence_refs
```

source 类型包括：

```text
instruction / bev / goal / waypoint / route / tool
```

之后才融合为 `fused_direction`、`fused_target_corridor_ref`、`source_consistency` 和 `fused_execution_status`。

- `aligned`：成功映射到 FPV corridor；
- `locally_blocked`：任务目标保留，但当前不能直接执行；
- `conflict`：多个输入或跨模态证据冲突；
- `not_visible`：目标不在当前 FPV；
- `context_only`：有上下文价值，但不能决定 corridor；
- `insufficient_evidence`：真实关系存在但证据不足；
- `ignored`：不适用并给出具体原因。

BEV/waypoint/route 只提供方向和拓扑，不能证明局部净空。FPV 始终是 immediate traversability 的权威来源。

## 10. InteractionAndRuleGrounding

actor interaction 至少包含：

```text
actor_ref
relation
affected_corridor_ref
temporal_basis
required_behavior
rule
evidence_refs
```

traffic rule 至少包含：

```text
control_ref
rule
affected_corridor_ref
required_behavior
evidence_refs
```

约束：

- 看到行人不等于 yield；必须证明其占用、穿越或逼近 considered corridor；
- 动态关系需要 temporal FPV 或 tracked metadata，单帧不能伪造运动；
- 交通规则必须绑定可见 control 和受影响 corridor；
- `applicable=true` 必须至少存在一条真实 interaction/rule；
- 无真实关系时为 `applicable=false` 和空数组。

## 11. Decision 与反事实解释

`Decision` 保存：

```text
motion
speed
social_action
selected_corridor_ref
fallback
confidence
rationale_status
supporting_reasons
rejected_alternatives
```

受控 reason ontology 只限定类别，例如：

```text
direct_corridor_available
navigation_context_aligned
local_path_blocked
interaction_requires_wait
traffic_control_requires_action
authoritative_goal_reached
insufficient_visual_support
```

这不是规则排列答案。每个 reason 必须引用前五段存在的 evidence，validator 还检查 reason 与事实的语义对应。没有对应事实时，即使 JSON 合法也会 reject。

`supported/partial` 才允许 Decision reasoning loss。`unsupported` 保留 GT motion 供独立 action/System1 supervision 使用，但 Decision 与 finish token 权重为 0。

## 12. Few-shot 与 typed ontology

### 12.1 Ontology 的作用

硬编码枚举用于：

- 稳定标签空间；
- 防止同义词无限漂移；
- 让 parser/validator 可执行；
- 支持分布统计、采样和评测。

它不负责决定当前样本属于哪个标签。语义答案仍由 LVLM 基于当前图像、文本和 evidence catalog 生成。

### 12.2 Few-shot 的作用

Versioned few-shot registry 教模型：

- instruction 与 scene caption 的区别；
- exact span、极性、指代和歧义如何表达；
- BEV/waypoint/goal 如何映射到 FPV；
- GT outcome 如何解释而不伪造视觉事实；
- unsupported rationale 的正确用法。

运行时按 dataset、modalities 和 stage 选择 2 到 4 个相关示例。Few-shot 版本参与 cache key；升级示例后旧缓存自动失效。

## 13. Prompt 体系

原有可达性、模态权威、安全、社会导航和不确定性原则没有被丢弃，而是拆成共享组件：

```text
URBANNAV_ROLE_AND_ACCESSIBILITY
EVIDENCE_AND_MODALITY_ROLES
SAFETY_AND_SOCIAL_POLICY
UNCERTAINTY_POLICY
OUTPUT_DISCIPLINE
```

组合成三个 profile：

```text
semantic observation = shared policy + SEMANTIC_OBSERVATION_CONTRACT
decision rationale    = shared policy + DECISION_RATIONALE_CONTRACT
student Full CoT      = shared policy + FULL_V3_CHAIN_CONTRACT
```

`ms-swift/swift/sidewalk_nav/prompt_policy.py` 同时被 legacy V2 和 V3 prompt 使用；V3 不再维护孤立的安全政策文本。RouteServer 与 ms-swift 保存相同的 Full-CoT profile ID 和 SHA-256 digest，digest 不一致时 adapter 直接拒绝。

## 14. 数据集 trust boundary

| 数据集 | outcome GT | FPV projection | Navigation context | System1 |
| --- | --- | --- | --- | --- |
| SideWalk | trusted trajectory | 仅 projection QC 通过时可信 | instruction/BEV/goal/waypoint/route/tool | 开启 |
| Foxglove | repaired GT trajectory | 仅 projection QC 通过时可信 | instruction/BEV/goal/waypoint/route/tool | 开启 |
| EgoWalk | trusted odometry | 不自动作为强 FPV corridor | caption 经 LVLM 分类后 + goal/waypoint/route | 开启 |
| FrodoBots | trajectory 禁用 | 禁用 | 仅 model-visible instruction/BEV/goal/route | 关闭 |

对 ego-frame 终点 `(x_T,y_T)`：

```text
d_T = sqrt(x_T^2 + y_T^2)
theta_T = atan2(y_T, x_T) * 180 / pi
```

先用 `d_T` 判断 stop，再用 `theta_T` 判断 left/straight/right。该值是 private outcome，不是 instruction 语义，也不能直接映射到图像左中右。

## 15. Validator 与 correction retry

### 15.1 Semantic/observation validator

- exact instruction span 必须真实存在；
- task refs、region refs 和 context refs 必须闭合；
- 每个 optional modality 必须恰好记账一次；
- 每个 region 必须引用自身和 FPV；
- positive cross-modal relation 必须引用 source、target 和 FPV；
- 动态 actor relation 必须有 temporal evidence；
- interaction/rule 必须绑定 actor/control 和 corridor。

### 15.2 Rationale validator

- outcome motion 必须与 frozen outcome 相同；
- reason 只能引用 frozen observation 中存在的 refs；
- moving support 必须有 direct corridor 事实；
- stop support 必须有 stop/goal/blocked/interaction/control 事实；
- slow/yield/wait 必须存在对应 interaction 或 local risk；
- unsupported 不得携带 supporting reasons，也不能 high confidence。

### 15.3 Target validator

- 六段完整且 renderer/parser 无损 roundtrip；
- LocalTraversability 不被 GT/route 改写；
- Decision execution corridor 必须属于 direct region；
- supported clear context 必须与 Decision execution corridor 一致；
- interaction required behavior 与 supported Decision 一致；
- FrodoBots trajectory/System1 永远关闭；
- section mask 与 provenance 闭合。

Correction retry 只回传当前 stage 的 bounded errors。Pass C 与 GT 的冲突不能回流给 Pass A/B，否则会造成 hidden-GT observation leakage。

## 16. Provenance、loss mask 与采样

主要 provenance 类别：

```text
runtime_fact
gt_action
validated_lvlm_task_semantic
validated_lvlm_observation
validated_hindsight_rationale
```

Section mask 示例：

```json
{
  "task_grounding": true,
  "scene_grounding": true,
  "local_traversability": true,
  "navigation_context_grounding": true,
  "interaction_and_rule_grounding": false,
  "decision": false
}
```

`ms-swift` 的 `reasoning_v3_sections` 按 XML section 划分 token：

- section mask 为 true 才参与 LM loss；
- `finish` 跟随 Decision mask；
- readout 仅在合法 navigate System1 row 开启；
- unsupported rationale 和 insufficient evidence 分开统计、分层采样并限流；
- V2、V3.2、V3.3 或未知 schema 禁止静默混合。

默认 sampler 上限：

```text
max_insufficient_fraction = 0.10
max_unsupported_fraction = 0.05
```

## 17. V3.3 版本合同

```text
urbannav_reasoning_source_v3_3
urbannav_reasoning_observation_v3_3
urbannav_decision_rationale_observation_v3_3
urbannav_reasoning_annotation_v3_3
urbannav_reasoning_trace_v3_3
urbannav_finish_v3_3
urbannav_reasoning_training_view_v3_3
reasoning_v3_3_fixed_supervision_v1
urbannav_reasoning_v3_3_composer_v1
urbannav_reasoning_v3_3_semantic_observation_v1
urbannav_reasoning_v3_3_decision_rationale_v1
urbannav_reasoning_v3_3_full_cot_v1
urbannav_reasoning_v3_3_fewshot_v1
```

旧 V3/V3.1/V3.2 annotation、cache、prompt digest、review 和 sparse-QA 产物只能归档，不能作为 V3.3 cache 或训练输入。

## 18. 代码索引

| 职责 | 文件 |
| --- | --- |
| source/annotation/training schema | `src/mcp_server/contracts/reasoning_dataset_v3.py` |
| canonical target | `src/mcp_server/contracts/reasoning_v3.py` |
| semantic/rationale observation schema | `src/mcp_server/contracts/reasoning_v3_observation.py` |
| source normalization | `src/mcp_server/data/reasoning_v3_source.py` |
| fixed supervision | `src/mcp_server/data/reasoning_v3_labels.py` |
| navigation source fusion | `src/mcp_server/data/reasoning_v3_conditions.py` |
| shared/profile prompt policy | `src/mcp_server/agent/reasoning_v3_prompt_policy.py` |
| few-shot registry | `src/mcp_server/teacher/reasoning_v3_fewshot.py` |
| two teacher prompts/parsers | `src/mcp_server/teacher/reasoning_v3_teacher.py` |
| observation/rationale validator | `src/mcp_server/agent/reasoning_v3_observation_validation.py` |
| deterministic composer | `src/mcp_server/data/reasoning_v3_composer.py` |
| target validator | `src/mcp_server/agent/reasoning_v3_validation.py` |
| batch/cache/retry | `src/mcp_server/teacher/reasoning_v3_batch.py` |
| training view | `src/mcp_server/data/reasoning_v3_views.py` |
| ms-swift shared policy | `ms-swift/swift/sidewalk_nav/prompt_policy.py` |
| ms-swift mirror/adapter | `ms-swift/swift/sidewalk_nav/reasoning_v3.py`, `data/reasoning_v3_manifest.py` |
| section loss/sampler | `ms-swift/swift/loss_scale/reasoning_v3.py`, `data/sample_reasoning_v3_manifest.py` |

## 19. 验收条件

- accepted row 固定包含六段和 finish；
- TaskGrounding 的语义字段来自 validated LVLM 或 authoritative runtime fact，不来自 trajectory shortcut；
- first-stage prompt 与 student prompt 不包含 hidden GT；
- Pass C 不得创建新实体、区域、规则或任务要求；
- unsupported GT explanation 必须关闭 Decision/finish reasoning loss；
- FrodoBots trajectory/System1 row 为 0；
- SideWalk/Foxglove/EgoWalk 的 trusted outcome 保持强标签，但不反向修改视觉事实；
- 每个 BEV/goal/waypoint/route/tool source 都有 used/ignored 与独立 grounding；
- conditional/blocked/forbidden 分类携带条件、blocker 或 reason；
- actor/control rule 绑定 affected corridor 和 evidence；
- RouteServer 与 ms-swift schema、profile digest 和 section mask 一致；
- 四数据集真实 LVLM smoke、质量报告与 HTML 人工检阅通过后，才允许全量重标。

## 20. 当前非目标

本版本不把 sparse VQA 混入 Full CoT。后续 VQA 必须单独定义 question registry、typed rationale/answer、canonical-fact projector、task validator 和 rationale/answer loss。typed ToolResultUse 与 agentic runtime 也必须使用独立 profile，不能以裸字典塞进 V3.3 六段合同。
