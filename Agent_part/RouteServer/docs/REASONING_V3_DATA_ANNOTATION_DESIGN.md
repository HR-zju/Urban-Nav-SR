# UrbanNav Reasoning V3.4 Minimal Full-CoT 数据标注与训练合同

更新时间：2026-07-23。

本文是当前 `Reasoning V3.4 minimal` 的唯一设计说明，覆盖数据来源、字段语义、teacher
流水线、验证、loss 路由、数据集边界和 V3.3 迁移。其他文档中仍出现的 V3.3
`instruction_type`、`local_objective`、route matching、projected corridor、runtime phase、
confidence、tracking/TTC 或 mixed Full-CoT distillation 描述均为历史设计，不再定义
V3.4 行为。

本文区分“合同已实现”“自动化结构验收通过”和“人工语义/训练效果已验证”，不会把三者
混为一谈。2026-07-23 已使用豆包完成一次四数据集、每数据集 3 条的真实 LVLM smoke；
其 schema、provenance、availability、loss 路由和审阅包完整性通过自动检查，但人工语义
审阅与训练 smoke 仍未完成，因此本文不声称 grounding accuracy 或训练收益已经得到验证。

## 1. 结论：当前 Full-CoT 是否符合第一原理和最简原则

当前 V3.4 的结构边界符合以下最小原则：

1. 数据集已有并经过允许的标签，才可以成为强监督。
2. LVLM 只能描述 student 可见输入中的事实，不能把 GT、runtime state 或几何推导包装成
   “视觉推理真值”。
3. 六段结构用于固定通信协议、审计证据和研究分析，不等于六段都有训练 loss。
4. 当前六段语言、`finish` 文本和 readout 文本全部为 zero-loss。
5. 只有两个相互独立的强监督 channel 可以开启：
   `direction_choice` 和 `system1_action`。
6. 没有完整、允许的 canonical direction stamp 时，样本只能成为
   `observation_only`，不能拼出 Full-CoT target。
7. FrodoBots 固定为 observation-only，canonical direction 和连续 System1 均关闭。

因此必须准确理解当前产物：

```text
六段 Full-CoT = typed audit scaffold + LVLM weak observations + copied strong direction identity
direction sidecar = 独立离散动作强监督
System1 sidecar = 独立连续动作强监督
```

它不是“用 LVLM 生成六段伪真值并做语言蒸馏”。如果研究目标是训练模型生成完整 CoT
语言，当前 V3.4 **没有开启该实验**；这是为了避免在没有人工真值或确定性真值的情况下，
把 teacher 的主观描述当成 ground truth。

## 2. 第一原理的数据模型

定义：

```text
x       = student 可见输入
y_dir   = 已有轨迹标签经 approved direction adapter 得到的离散方向
y_sys1  = 数据集已有且 policy 允许的 trajectory_ego_t
z_obs   = LVLM 基于 x 生成的弱观察
z_rat   = LVLM 基于 x、冻结 z_obs 和冻结 y_dir 生成的 hindsight 弱解释
```

当前流水线为：

```text
z_obs ~ teacher_pass1(x)

if y_dir 不存在或 dataset policy 禁用:
    artifact = observation_only(x, z_obs)
else:
    z_rat  ~ teacher_pass2(x, frozen(z_obs), frozen(y_dir))
    target = deterministic_compose(z_obs, z_rat, copy(y_dir))

optional strong losses:
    L_direction = CE(direction_head(x), y_dir)
    L_system1   = action_loss(system1_head(x), y_sys1)

always:
    L_full_cot_lm = 0
    L_finish_lm   = 0
    L_readout_lm  = 0
    L_segmentation = 0
```

### 2.1 允许的来源类型

| 简写 | `source_kind` | 含义 | 当前可否进入 loss |
| --- | --- | --- | --- |
| `IR` | `input_reference` | 从 student 可见的 instruction、图像角色或 evidence catalog 确定性复制的引用 | 否 |
| `GT` | `dataset_gt` | 数据集已有、经 policy 允许并通过结构校验的标签 | 仅 `system1_action` |
| `DA` | `deterministic_adapter` | 对已有标签执行冻结、版本化、可复现 adapter 得到的标签 | 仅 `direction_choice` |
| `LW` | `lvlm_weak` | LVLM 对 student 可见输入的弱标注 | 当前全部否 |
| `P` | 非 label provenance | schema version、字段名、section 顺序等 wire protocol 常量 | 否 |

`student_inferable=true` 表示该目标被声明为可从 student 输入学习，不表示该字段已经被证明
具有高准确率。尤其对 direction 和 System1，它是训练资格声明，仍需要 held-out
泛化实验验证。

### 2.2 当前明确不存在的第五类来源

V3.4 没有 `runtime_fact`、`teacher_private_semantic` 或
`geometry_projection_gt` 这类 Full-CoT 来源。以下信息不会进入六段 reasoning：

```text
GPS / GNSS
camera pose / calibration / homography
route matching / route theta / route geometry
goal distance / runtime task phase / arrival state
active waypoint state
projected trajectory corridor
instance track / ego-motion compensation / TTC
mask / bbox / instance id
```

## 3. 冻结输入、private supervision 与输出分区

### 3.1 Student-visible input

Pass 1、Pass 2 和 student prompt 共享的 model-visible 部分只有：

```text
ordered FPV images
optional BEV or goal image
visible instruction/caption
visible text goal/route/waypoint/tool context
evidence_catalog
```

V3.4 source boundary filter 会删除数值 route/pose/GPS/projection/tracking 内容，并将
`goal_geometry`、`route_points_ego_t`、legacy QA context 和 V2 private dictionary 清空。
BEV、route text 或 tool text 只是可见上下文，不是局部净空或方向真值。

### 3.2 Teacher-only supervision

private partition 只保留：

```text
dataset supervision policy
complete canonical_direction record, or null
System1 eligibility
fixed supervision version
source-boundary audit
P0 shortcut flags
field-group provenance
```

`trajectory_ego_t` 是独立 System1 target，不进入六段文本，也不发送给 LVLM teacher。
canonical direction 只在 Pass 2 以冻结身份提供，Pass 1 看不到。

### 3.3 Annotation states

| `status` | 必需内容 | 禁止内容 | 用途 |
| --- | --- | --- | --- |
| `accepted` | observation、decision rationale、target、availability mask | 缺失 canonical direction | 可导出 audit view；可按 profile 导出强 sidecar |
| `observation_only` | Pass-1 observation | rationale、target、section mask | 视觉/语义审计，不生成训练 view |
| `rejected` | errors 和审计元数据 | target、section mask | schema/grounding 失败 |
| `error` | provider/运行错误和审计元数据 | target、section mask | retry 与故障分析 |

缺方向的合法样本不是坏数据；它保留为 `observation_only`，避免为了追求 Full-CoT 覆盖率
而伪造动作标签。

## 4. Canonical wire format

每条 accepted target 固定为六段、一个 finish 和可选 readout：

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
<readout>[SYS1_0] ... [SYS1_15]</readout>   # 仅有 System1 sidecar 时
```

六段不能缺失、重复或乱序。`<think>` 内不允许 section 外自由文本，整个 document 外也不
允许自由文本。每段内容必须是严格 JSON object。readout token 用空格分隔，只是 latent
query anchor，不是自然语言动作标签。

顶层字段：

| 字段 | 科研问题/作用 | 来源 | Student 可预测性 | Loss | 保留理由 |
| --- | --- | --- | --- | --- | --- |
| `reasoning.schema_version` | 当前 trace 由哪个不可混用的合同解释 | `P` | 不适用 | 无 | 防止 V3.3/V3.4 静默混合 |
| 六个 section | 是否形成完整、可解析、可审计的观察链 | 见后文 | 见后文 | 六段 LM 均为 0 | 固定研究输出接口 |
| `finish` | 是否无损投影同一 canonical direction | `DA` 的确定性复制 | 标签本身可学习；投影不单独学习 | finish LM 为 0 | 防止 Decision 与最终答案漂移 |

## 5. 六段逐字段合同

本节中的“无”表示当前 V3.4 不参与任何优化；不是“未来永远不能研究”。LW provenance
目前按 section group 记录，而不是为每个 leaf field 单独写一条 provenance。表中列出
leaf 的真实来源，不能把 group provenance 误解成字段级人工 GT。

### 5.1 `TaskGrounding`

研究问题：当前可见 instruction 明确说了什么、指向什么、哪里存在实质歧义？

| 字段 | 科研问题/语义 | 来源 | Student 可预测性 | Loss | 保留理由 |
| --- | --- | --- | --- | --- | --- |
| `applicable` | 可见输入中是否有可解析的活动 instruction | `LW + IR`；LVLM 判断且 composer 要求 `instruction_ref` 存在 | 是 | 无 | 区分空任务与有任务但无抽取项 |
| `instruction_ref` | 哪个 evidence ref 是活动 instruction | `IR`，由 composer 从 catalog 复制 | 非预测元数据 | 无 | 建立任务文本的审计锚点 |
| `goal_ref` | 是否存在 student 可见 goal source | `IR`，仅在 catalog 中存在时复制 | 非预测元数据 | 无 | 不把 hidden goal geometry 当作目标 |
| `active_navigation_refs[]` | 本样本实际提供了哪些可见 navigation sources | `IR`，从 catalog/source registry 去重复制 | 非预测元数据 | 无 | 做输入覆盖闭包，不做融合 |
| `requirements[].code` | instruction 中的开放要求是什么 | `LW`，`ReasonCode` 开放 lower-snake-case | 是 | 无 | 避免人为 `instruction_type` 选择题 |
| `requirements[].polarity` | 要求是 required/preferred/avoid/forbidden 中哪一轴 | `LW`，闭集协议状态 | 是 | 无 | 分离要求内容与约束极性 |
| `requirements[].text_span` | 哪段原文直接支持该要求 | `LW`，必须是 instruction 中真实 span | 是 | 无 | 防止关键词扩写或隐藏标签注入 |
| `requirements[].target_ref` | 要求显式指向哪个可见/local ref | `LW` 从合法 refs 中选择，可为空 | 是 | 无 | 把语言约束落到证据对象 |
| `requirements[].evidence_refs[]` | 哪些 refs 支持要求 | `LW` 选择已有 refs | 是 | 无 | 支持 evidence closure |
| `referents[].phrase` | instruction 中哪个短语需要 grounding | `LW`，必须是原文 span | 是 | 无 | 保留真实指代表达 |
| `referents[].resolution` | 指代是 resolved/unresolved/ambiguous | `LW`，闭集解析状态 | 是 | 无 | 不强迫模型伪造 resolved target |
| `referents[].resolved_ref` | resolved 时具体对应哪个合法 ref | `LW`；仅 `resolution=resolved` 可出现 | 是 | 无 | 可执行检查指代闭合 |
| `referents[].evidence_refs[]` | 哪些可见 refs 支持指代判断 | `LW` | 是 | 无 | 让 resolved/ambiguous 可审计 |
| `ambiguities[].type` | 实质歧义的开放原因 | `LW`，`ReasonCode` | 是 | 无 | 不预设封闭歧义 taxonomy |
| `ambiguities[].text_span` | 歧义来自哪段原文 | `LW`，可为空；非空必须在 instruction 中 | 是 | 无 | 区分文本歧义与视觉歧义 |
| `ambiguities[].evidence_refs[]` | 哪些可见 refs 支持歧义判断 | `LW` | 是 | 无 | 防止无证据的 `unknown` |

明确删除：

```text
instruction_type
caption_role
inferred_objective / local_objective
authoritative runtime phase
goal_distance / goal_reached
objective confidence
```

原因不是这些概念毫无研究价值，而是当前数据没有统一、student-visible、确定性的真值。
用研究者枚举强迫 teacher 选择会形成低熵 shortcut；用 runtime state 填充又会形成
teacher-only leakage。

### 5.2 `SceneGrounding`

研究问题：FPV 中能直接观察到哪些与导航有关的实体和交通控制？

| 字段 | 科研问题/语义 | 来源 | Student 可预测性 | Loss | 保留理由 |
| --- | --- | --- | --- | --- | --- |
| `applicable` | FPV 是否可用于场景观察 | `LW`；有 FPV 时 validator 要求为 true | 是 | 无 | 防止把图像存在但输出空段当合法省略 |
| `entities[].entity_ref` | 一个 sample-local 实体关系键 | `LW` 创建，格式 `entity.*` | 是 | 无 | 供同一样本内 interaction 交叉引用；不是 instance ID |
| `entities[].category` | 可见实体的开放名称/种类 | `LW`，`ReasonCode` | 是 | 无 | 保留 LVLM 可标注类别，不冻结数据集 ontology |
| `entities[].lateral_position` | 粗粒度 left/center/right/spanning/unknown | `LW`，闭集空间状态 | 是 | 无 | 只表达可见粗位置，不假装有像素 mask |
| `entities[].depth_position` | 粗粒度 near/mid/far/unknown | `LW`，闭集空间状态 | 是 | 无 | 提供导航相关相对深度，不声称米制距离 |
| `entities[].evidence_refs[]` | 哪些 FPV frame 支持该实体 | `LW` 选择 catalog refs | 是 | 无 | 每个实体必须有 FPV 证据 |
| `traffic_controls[].control_ref` | 一个 sample-local control 关系键 | `LW` 创建，格式 `control.*` | 是 | 无 | 供 traffic rule 交叉引用 |
| `traffic_controls[].category` | control 的开放类别 | `LW`，`ReasonCode` | 是 | 无 | 支持信号灯、标志、横道等长尾类型 |
| `traffic_controls[].state` | 可见 control 的开放状态，可为空 | `LW`，`ReasonCode` | 是 | 无 | 不把不同控制设施压进同一枚举 |
| `traffic_controls[].lateral_position` | control 的粗左右位置 | `LW` | 是 | 无 | 提供可见关系，不使用标定几何 |
| `traffic_controls[].evidence_refs[]` | 哪些 FPV frame 支持 control | `LW` | 是 | 无 | 强制视觉 grounding |

`entity_ref` 和 `control_ref` 只是一次样本内的引用键。当前没有 detector instance GT、
mask、bbox、track id 或 calibrated depth，因此这些键不能被解释为检测/分割真值。

### 5.3 `LocalTraversability`

研究问题：FPV 中可见的语义区域是什么，LVLM 对其当前可通行状态作何弱判断？

| 字段 | 科研问题/语义 | 来源 | Student 可预测性 | Loss | 保留理由 |
| --- | --- | --- | --- | --- | --- |
| `applicable` | 是否至少存在一个可见 region | `LW`；必须与 `regions` 是否为空一致 | 是 | 无 | 防止“applicable 但没有对象”的空标签 |
| `regions[].region_ref` | sample-local 可见区域键 | `LW` 创建并自引 | 是 | 无 | 支持语义区域关系；不是 pixel region/mask |
| `regions[].surface` | 区域表面/名称种类 | `LW`，开放 `ReasonCode` | 是 | 无 | 满足当前仅由 LVLM 标名称类别的能力边界 |
| `regions[].state` | direct/conditional/blocked/forbidden/unknown | `LW`，闭集安全状态 | 是 | 无 | 使通行状态逻辑可验证，同时允许 unknown |
| `regions[].blocker_refs[]` | 哪些已出现 entity 可能造成阻挡 | `LW`，必须解析到 Scene entity | 是 | 无 | 只表达可见关系，不计算 TTC |
| `regions[].reason` | conditional/blocked/forbidden 的开放原因 | `LW`，`ReasonCode`，可为空 | 是 | 无 | 支持长尾原因而不造固定 taxonomy |
| `regions[].evidence_refs[]` | 哪些 FPV frame 支持该区域与状态判断 | `LW`，只能选择 catalog refs | 是 | 无 | 至少包含一个 FPV evidence；不把 local `region_ref` 冒充 evidence |

一致性规则：

- `direct` 不能带 `blocker_refs`；
- `conditional/blocked/forbidden` 至少要有 blocker 或 reason；
- 有 FPV 时至少输出一个 region，证据不足时可以使用 `state=unknown`；
- `region_ref` 是 section 内的 local relation key，不要求也不允许为了“自引”而重复塞入
  `evidence_refs`；只有 evidence catalog 中真实存在的 ref 才是证据；
- 不输出 `visibility`、`risk`、`confidence`、`preferred_clear_corridor_ref`；
- 不使用 SAM/SoM mask、route projection、GPS 或 trajectory 反推当前区域状态。

因此当前 traversability 是 **LVLM semantic weak annotation**，不是 segmentation label，
也不能用 mIoU 评价。

### 5.4 `NavigationContextGrounding`

研究问题：每个真正提供给 student 的 navigation source 是否被使用，以及它与当前任务的
可见语义关系是什么？

| 字段 | 科研问题/语义 | 来源 | Student 可预测性 | Loss | 保留理由 |
| --- | --- | --- | --- | --- | --- |
| `applicable` | 是否存在至少一个可记账 navigation source | `LW + IR`；必须与 list 非空一致 | 是 | 无 | 让空上下文显式化 |
| `source_groundings[].source_type` | source 是 instruction/bev/goal/waypoint/route/tool 中哪一协议角色 | `IR` 身份 + `LW` 原样返回 | 非预测元数据 | 无 | 固定输入角色，不定义语义答案 |
| `source_groundings[].source_ref` | 对应 evidence catalog 的哪个真实 source | `IR`，必须逐个覆盖且唯一 | 非预测元数据 | 无 | 防止忽略 optional modality |
| `source_groundings[].usage` | 该 source 被 used 还是 ignored | `LW`，闭集记账状态 | 是 | 无 | 区分缺失、忽略和使用 |
| `source_groundings[].relation` | used source 提供什么开放语义关系 | `LW`，`ReasonCode`；used 时必需 | 是 | 无 | 不预设 route-to-FPV 融合类别 |
| `source_groundings[].ignored_reason` | ignored 的开放原因 | `LW`；ignored 时必需 | 是 | 无 | 避免静默丢弃 context |
| `source_groundings[].evidence_refs[]` | 哪些可见 refs 支持记账 | `LW`，必须包含 `source_ref` | 是 | 无 | 保持 source closure |

该 section **只做独立 source accounting**。它不做：

```text
route matching
route/BEV theta derivation
source reliability scoring
cross-modal direction fusion
target corridor projection
fused execution status
local free-space certification
```

`waypoint` 保留为 wire role，只在实际可见 evidence catalog 中存在该 source 时使用；V3.4
不会为了填满 schema 合成 waypoint。

### 5.5 `InteractionAndRuleGrounding`

研究问题：在冻结 navigation direction 后，哪些可见 actor/path 关系或 control/rule
可能与当前导航有关？

| 字段 | 科研问题/语义 | 来源 | Student 可预测性 | Loss | 保留理由 |
| --- | --- | --- | --- | --- | --- |
| `applicable` | 是否至少存在 interaction 或 traffic rule | `LW`；必须与两个 list 是否为空一致 | 是 | 无 | 无关系时明确输出 false |
| `interactions[].actor_ref` | 关系涉及哪个已观察 entity | `LW`，必须解析到 Scene `entity_ref` | 是 | 无 | 防止 Pass 2 创建新 actor |
| `interactions[].relation` | 可见 actor/path 的开放关系 | `LW`，`ReasonCode` | 是 | 无 | 不使用人工 interaction taxonomy |
| `interactions[].temporal_basis` | 关系依据 single_frame 还是 multi_frame | `LW`，闭集证据状态 | 是 | 无 | 区分静态观察与表观时序关系 |
| `interactions[].suggested_behavior` | 基于可见关系的开放、非执行性建议，可为空 | `LW`，`ReasonCode` | 是 | 无 | 保留可审计社会导航提示，不冒充 action GT |
| `interactions[].evidence_refs[]` | 哪些 FPV/local refs 支持关系 | `LW` | 是 | 无 | dynamic relation 必须至少两张 FPV |
| `traffic_rules[].control_ref` | 规则涉及哪个已观察 control | `LW`，必须解析到 Scene `control_ref` | 是 | 无 | 防止凭空创建交通设施 |
| `traffic_rules[].rule` | LVLM 对可见 control 的开放规则描述 | `LW`，`ReasonCode` | 是 | 无 | 当前没有 frozen normative policy GT |
| `traffic_rules[].suggested_behavior` | 开放、非执行性行为建议，可为空 | `LW` | 是 | 无 | 不与 runtime control action 混淆 |
| `traffic_rules[].evidence_refs[]` | 哪些 refs 支持规则 | `LW`，必须包含 `control_ref` 和 FPV | 是 | 无 | 绑定可见 control |

这里的 `suggested_behavior` 不是 `motion/speed/social_action/fallback`，不能进入动作 loss，
也不能覆盖 canonical direction。当前不实现：

```text
instance tracker
ego-motion compensation
TTC / overlap
control metadata GT
frozen normative policy labels
negative coverage gate based on tracker
```

多帧只支持“从两张以上可见 FPV 得到的表观关系”，不支持 persistent track 真值。

### 5.6 `Decision`

研究问题分成两条完全独立的轴：

1. 已有标签经过哪个 approved adapter 得到什么方向？
2. student-visible evidence 是否能支持解释这个冻结方向？

| 字段 | 科研问题/语义 | 来源 | Student 可预测性 | Loss | 保留理由 |
| --- | --- | --- | --- | --- | --- |
| `profile_id` | 用哪个冻结方向合同解释 label index | `DA` 确定性复制 | 非模型输出语义；训练配置元数据 | `direction_choice` 的合同组成 | 防止 20-way/12-way 混读 |
| `profile_version` | profile 的具体版本 | `DA` | 非预测元数据 | 同上 | profile 语义变化必须显式升级 |
| `label_index` | 该 profile 下的离散方向类别 | `DA`，strong | 是 | `direction_choice`，仅 strong profile | 真正的离散动作 target |
| `label_name` | label index 的稳定名称 | `DA` | 可由 index 确定，不单独训练 | 不单独计 loss | 人审和一致性检查 |
| `choice` | 可选显示 token | `DA` | 可由 index 确定，不单独训练 | 不单独计 loss | 兼容 prompt/评测显示 |
| `angle_deg` | profile 定义的类中心角；stop 可为空 | `DA` | 可由 index 确定，不作连续角 GT | 不单独计 loss | profile 审计，不允许重算 |
| `source` | 生成 stamp 的具体 adapter source code | `DA` | 非预测元数据 | 无独立 loss | 审核并禁止 route-theta fallback |
| `rationale_status` | 可见事实对冻结方向是 supported/partial/unsupported | `LW` | 是 | 无 | 允许诚实表达不可解释 GT |
| `supporting_reasons[].reason` | 支持方向的开放可见原因 | `LW`，`ReasonCode` | 是 | 无 | 不使用固定 reason 选择题 |
| `supporting_reasons[].evidence_refs[]` | 哪些冻结 observation refs 支持原因 | `LW` | 是 | 无 | Pass 2 不得创建新事实 |

`supported/partial` 必须至少有一条 support；`unsupported` 必须没有 support。无论
`rationale_status` 是什么，direction identity 都不能被 LVLM 修改。`unsupported` 只说明
当前可见证据不足以解释已有标签，不代表可以删除合法的独立 action target。

## 6. Approved direction profiles

V3.4 Full-CoT allowlist **只有**以下两个完整 stamp：

| `profile_id` | Version | Bin contract | Geometry source | Stop rule | Route theta |
| --- | --- | --- | --- | --- | --- |
| `fpv_degree10_terminal_centroid_v2` | `2.0` | `degree10_with_stop`，-90° 到 90° 每 10° 一类，加 stop，共 20 类，index 0–19 | existing FPV pixel trajectory terminal centroid | displacement `< 0.06`，单位 normalized FPV xy | 禁止 |
| `ego_degree10_12way_terminal_centroid_v2` | `2.0` | `degree10_12way_with_stop`，stop、straight、左右 10/20/30/40/50+，共 12 类，index 0–11 | existing `trajectory_ego_t`，`x_right/y_front` terminal centroid | displacement `< 0.25 m` | 禁止 |

一个可接受的 direction record 必须同时携带：

```text
profile_id
profile_version
label_index
name
bin_mode
derivation_version
source
optional choice
optional angle_deg
```

缺任一 core stamp、版本不匹配、index 越界、`label_name/choice/angle_deg` 与 profile
中的 index 不一致、profile 不在 allowlist、或 `source` 含 `route_theta` 时 fail closed。
V3.4 不从 row 内的 trajectory 现场推 direction，也不从 BEV、GPS、route、goal 或
LVLM 回答补 direction。

training view 使用以下显式 sidecar 名称，并要求与 `Decision` 七元 identity 完全一致：

```text
direction_profile_id
direction_profile_version
direction_label_index
direction_label_name
direction_choice
direction_angle_deg
direction_source
```

`ms-swift` direction registry 还保留其他实验/兼容 profiles，包括旧
`ego_degree10_12way_terminal_centroid_v1`。它们存在于通用 library 不等于被 V3.4
批准；V3.4 producer 和 consumer allowlist 均只接受上表两个 v2 profile。

profile 是逐样本显式选择的可插拔合同，不设一个隐式“全局默认方向定义”。数据准备阶段必须
根据真实 label 坐标系选择 profile。尤其 EgoWalk 的 FPV trajectory 是 proxy 时，只能从
可信 ego trajectory 构造 ego v2 stamp，不能把 proxy 当 FPV 强标签。

V3.4 producer 还执行以下 `dataset × profile` hard binding：

| 数据集 | 唯一允许的 direction profile | 原因 |
| --- | --- | --- |
| SideWalk | `fpv_degree10_terminal_centroid_v2` | 当前 canonical direction 来自已有 FPV trajectory 合同 |
| Foxglove | `fpv_degree10_terminal_centroid_v2` | 同上；不接受 route theta |
| EgoWalk | `ego_degree10_12way_terminal_centroid_v2` | 原始 legacy metric trajectory 为 `x_right/y_front`；FPV trajectory 标记为无相机投影 proxy，不能成为强标签 |
| FrodoBots | 无 | 固定 observation-only |

因此“profile 在全局 allowlist 中”并不足以通过；跨数据集错配会直接拒绝。EgoWalk 的
ego-v2 stamp 必须在 source builder 之前由 ms-swift adapter 从原始 legacy
`x_right/y_front` trajectory 生成；转换后的 FLU System1 trajectory 只服务独立连续动作
head，不能被重新解释成这个 legacy 12-way 几何合同。

## 7. `finish`：只做无损投影

V3.4 `FinishV3` 的每个字段如下：

| 字段 | 来源 | Student 可预测性 | Loss | 约束/保留理由 |
| --- | --- | --- | --- | --- |
| `schema_version` | `P` | 不适用 | 无 | 固定为 `urbannav_finish_v3_4` |
| `decision_type` | `P` | 不适用 | 无 | 当前仅 `navigate`；不把 agent runtime 决策混入该数据合同 |
| `profile_id` | 从 `Decision` 确定性复制 | 非预测元数据 | finish LM 0 | 必须逐值相等 |
| `profile_version` | 同上 | 非预测元数据 | finish LM 0 | 必须逐值相等 |
| `label_index` | 同上 | 由 direction sidecar 独立监督 | finish LM 0 | 不能由 renderer 重新推导 |
| `label_name` | 同上 | 可由 index 确定 | finish LM 0 | 人审 |
| `choice` | 同上 | 可由 index 确定 | finish LM 0 | 可为空 |
| `angle_deg` | 同上 | 可由 index 确定 | finish LM 0 | 可为空 |
| `source` | 同上 | 非预测元数据 | finish LM 0 | 防止来源漂移 |

`ReasoningTargetV3` validator 对七元 direction identity 做 exact equality。V3.4 不在
`finish` 中携带 trajectory、motion、speed、social action、fallback、ask/complete/abort
或 ToolResultUse。Agent runtime 的这些能力没有被删除，但属于另一个 typed contract。

## 8. System1 sidecar

System1 与六段 CoT、direction sidecar 都是独立监督：

| 字段 | 科研问题/作用 | 来源 | Student 可预测性 | Loss | 约束 |
| --- | --- | --- | --- | --- | --- |
| `trajectory_ego_t` | 从可见输入预测连续局部轨迹 | `GT`，数据集已有并通过 canonicalizer | 是，作为动作 target | `system1_action` | 非空有限 `[T,3]`；不进入 assistant text |
| `trajectory_policy` | 解释坐标、horizon、yaw 等 System1 合同 | `GT/P` 规范化元数据 | 非预测元数据 | 无独立 loss | 有 trajectory 时必须存在 |
| `action_mask` | 哪些时间步可监督 | `GT` 或缺省全 true | 非预测元数据 | 控制 `system1_action` | 长度必须等于 T |
| `[SYS1_i]` readout | 给 continuous head 提供 latent query anchors | `P` | hidden state 由模型产生 | readout token LM 0 | 仅 sidecar 存在时完整、有序、空格分隔 |

`loss_mask.trajectory` 永远为 false；V3.4 使用统一的 `system1_action` 逻辑 channel。
direction 存在不要求 System1 存在，System1 存在也不能替代缺失的 direction 来构造
accepted Full-CoT。

## 9. Provenance 与 mask 的精确定义

### 9.1 `LabelProvenanceEntryV3`

| 字段 | 含义 | 当前合法用法 |
| --- | --- | --- |
| `source` | 具体 label/teacher 来源 code | adapter、dataset trajectory 或 validated LVLM |
| `source_kind` | `input_reference/dataset_gt/deterministic_adapter/lvlm_weak` | 不允许 runtime/geometric pseudo-GT |
| `supervision` | `strong/weak/audit_only/disabled` | direction、System1 为 strong；六段语义为 weak |
| `student_inferable` | 是否有资格从 student-visible input 学习 | 两个强 channel 和 LW group 当前均为 true |
| `loss_channel` | `none/weak_lm/direction_choice/system1_action` | 当前仅两个强 channel 非 none |
| `generator_version` | 生成器/adapter 版本 | cache、复现和 drift 审计 |
| `details` | profile、field path 等非目标元数据 | 不得塞入隐藏 GT 语义 |

当前实际 provenance keys：

```text
decision.direction                 DA / strong / direction_choice
system1.trajectory                GT / strong / system1_action
task_grounding                    LW / weak / none
scene_grounding                   LW / weak / none
local_traversability              LW / weak / none
navigation_context_grounding      LW / weak / none
interaction_and_rule_grounding    LW / weak / none
decision.rationale                LW / weak / none
```

只有 applicable/存在的 weak group 才写对应 provenance。当前没有逐 leaf provenance；
若未来研究发现一个 section 内来源混杂，应该升级 schema，而不是在 `details` 里隐式表达。

`weak_lm` 虽保留在基础 type union 中，当前 V3.4 training-view exporter 会拒绝任何
`weak_lm` provenance，防止旧 mixed distillation 被静默恢复。

### 9.2 三种容易混淆的 mask

| 位置 | 含义 | 当前值 |
| --- | --- | --- |
| source `section_loss_mask` | 强 source 是否授权某个 reasoning section | 六项全 false |
| annotation `section_loss_mask` | composer 记录 weak section 是否存在 | 只是 availability；Decision false |
| training view `section_loss_mask` | 实际语言 section loss | 六项全 false |

training view 将 annotation mask 另存为
`metadata.weak_section_availability`，不能据此开启 LM loss。

实际训练 section mask 固定为：

```json
{
  "task_grounding": false,
  "scene_grounding": false,
  "local_traversability": false,
  "navigation_context_grounding": false,
  "interaction_and_rule_grounding": false,
  "decision": false
}
```

逻辑 loss profiles：

| Profile | `lm` | `direction_choice` | `system1_action` | `trajectory` | `segmentation` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `audit_only` | false | false | false | false | false |
| `strong_action_only` | false | 有完整 DA sidecar 时 true | 有合法 GT sidecar 时 true | false | false |

两个强 channel 独立开启；不存在“因为 Full-CoT accepted，所以所有 loss 都开”的规则。

## 10. 两阶段 teacher 与确定性 composer

### 10.1 Pass 1：semantic observation

输入：

```text
student-visible instruction
ordered visible images
filtered visible text context
evidence catalog
```

输出：

```text
TaskSemanticObservation
SceneGrounding
LocalTraversabilityObservation
NavigationContextObservation
```

Pass 1 看不到 canonical direction、trajectory、runtime state、GPS/pose、projection 或
dataset policy。任务语义与视觉事实不能为迎合 action label 而改写。

### 10.2 Observation-only gate

Pass 1 验证通过后立即检查：

```text
FrodoBots                         -> observation_only
canonical_direction missing      -> observation_only
otherwise                        -> Pass 2
```

observation-only 不执行 Pass 2，不生成 target，不生成 training view。

### 10.3 Pass 2：direction-conditioned rationale

输入：

```text
原始 student-visible input
frozen validated Pass-1 observation
frozen complete canonical direction identity
```

输出只包含：

```text
InteractionAndRuleGrounding
exact copied direction identity
rationale_status
supporting_reasons
```

Pass 2 不能修改方向、创建新的 Scene entity/control/region，不能输出 speed/social/fallback，
也不能查看 raw trajectory。方向缺乏可见支持时必须输出 `unsupported`。

### 10.4 Composer

Composer 不推理动作，只执行：

1. 从 private fixed labels 复制 complete direction。
2. 检查 Pass 2 direction identity exact match。
3. 将 source builder 的 instruction/goal/context refs 与 Pass-1 semantics 合并。
4. 原样加入已验证 Scene、Traversability、Navigation context。
5. 原样加入已验证 interaction/rule 和 rationale。
6. 从同一 direction identity 构造 `Decision` 与 `finish`。
7. 写 group provenance 和 weak availability。

当前 V3.4 few-shot registry 是空的：
`urbannav_reasoning_v3_4_minimal_fewshot_empty_v1`。旧 examples 含已删除 ontology、
projected corridor、heuristic action 和 self-reported confidence，因此没有迁移。未来加入
few-shot 必须逐条人工审核、版本化，并保证所有 claim 都有 student-visible evidence。

## 11. 验证与 fail-closed 路径

### 11.1 Source boundary

- 删除 GPS/pose/calibration/route geometry/projection/tracking 数值或引用；
- 清空 V2 private dictionary、legacy QA label validity 和几何 sidecars；
- 只保留可见文本 context；
- 对过滤后的完整 source 计算 SHA-256 fingerprint；
- annotation 必须匹配同一个 fingerprint。

### 11.2 Pass-1 validation

- requirement span、referent phrase、非空 ambiguity span 必须存在于 instruction；
- entity/control/traversability 必须引用 FPV；
- blocker 必须解析到本样本 Scene entity；
- 每个真实 navigation source 必须恰好出现一次；
- used source 必须有 relation，ignored source 必须有 ignored reason；
- 所有 evidence refs 必须在 catalog 或本样本 local refs 中闭合；
- 有 FPV 时不能省略 Scene 或 Traversability。

### 11.3 Pass-2 validation

- direction 七元 identity 必须与 frozen outcome 完全相同；
- interaction actor 必须已存在于 Scene；
- dynamic relation 必须是 multi-frame 且引用至少两张 FPV；
- traffic rule 必须引用已存在 control，并显式引用该 control；
- supporting reason 只能引用冻结 observation 中已有 refs；
- `unsupported` 不能携带 supporting reasons。

### 11.4 Target/text validation

- 六段和 finish 严格顺序、数量、JSON type、无自由文本；
- `extra="forbid"` 拒绝旧字段；
- Decision 必须逐值复制 canonical direction；
- finish 必须逐值复制 Decision；
- direction provenance 必须为
  `deterministic_adapter/strong/student_inferable/direction_choice`；
- weak groups 必须为 `lvlm_weak/weak/student_inferable`；
- Decision section language loss 永远禁止；
- FrodoBots 不能产生 canonical-direction Full-CoT；
- assistant text 禁止出现 trajectory、route theta、confidence、mask、bbox、track、TTC 等字段；
- renderer/parser 必须无损 roundtrip。

### 11.5 Producer/consumer parity

RouteServer 与 `ms-swift` 同时校验：

- semantic JSON Schema equality；
- schema digest equality；
- Full-CoT prompt profile ID 和 policy digest；
- approved direction profile/version/index range；
- direction sidecar 与 Decision exact identity；
- training profile 和 policy version；
- System1 readout、trajectory policy 和 action mask；
- all-language-zero-loss policy。

## 12. Shortcut gates

### 12.1 已实现的 P0 flags

当前 source audit 记录：

```text
target_derived_instruction_candidate
answer_projected_direction_choice_cot_candidate
```

第一项表示 instruction 可能由同一个 future motion target 通过 rule 生成；第二项表示旧
direction-choice manifest 的 `<think>` 可能直接由答案模板投影。它们可保留在 audit view，
但 `strong_action_only` exporter 会整行过滤，避免 instruction 或旧 CoT 直接泄漏 action。

### 12.2 必须继续统计的 shortcut 证据

真实 smoke 至少报告：

- 每数据集两个 P0 flag 的行数和比例；
- instruction source 与 direction label 的 mutual dependence；
- 同 clip/session split 是否泄漏；
- profile、label、城市和数据集分布；
- 去掉图像、只保留 instruction 的 direction baseline；
- 打乱 instruction 或图像后的性能变化；
- target-derived instruction 子集与原生 instruction 子集的独立结果。

这些是实验 gate，不应通过在 prompt 中写“不要 shortcut”来替代。

## 13. 数据集 truth matrix

当前代码中的 dataset policy：

| 数据集 | Model-visible observation | Canonical direction | System1 | V3.4 产物 |
| --- | --- | --- | --- | --- |
| SideWalk | 实际 row 中的 FPV、可选 BEV、instruction/可见文本 context | 仅完整 approved v2 stamp；不现场重算 | 有合法 `trajectory_ego_t` 时允许 | 有 direction 可 accepted；缺 direction observation-only |
| Foxglove | 实际 row 中的 FPV、可选 BEV、generic/已有 instruction | 仅完整 approved v2 stamp；不使用 route theta | 有合法 `trajectory_ego_t` 时允许 | 同上 |
| EgoWalk | FPV 和原生 caption/brief/instruction；不要求 BEV | 可信 ego trajectory 应使用新的 ego 12-way v2；FPV proxy 不作为强真值 | 有合法 `trajectory_ego_t` 时允许 | 有 direction 可 accepted；缺 direction observation-only |
| FrodoBots | FPV、实际可见 BEV/文本，可供 LVLM Pass 1 | policy 禁用，即使 row 有 compatibility 字段也忽略 | policy 禁用 | 固定 observation-only |

这个 matrix 表达的是信任边界，不保证当前所有 raw manifest 已经拥有 complete direction
stamp。旧字段“看起来像 direction/trajectory”不等于 V3.4 强标签。真实转换必须报告：

```text
input rows
source rows / rejected rows
direction available / unavailable
System1 available / unavailable
accepted / observation_only / rejected / error
profile distribution
P0 shortcut flags
```

本版本不要求也不允许新增以下 GT 构造：

```text
GPS stabilization
route matching
pose/calibration recovery
raw projection sidecar
D2/D3/D4 projectors
instance tracking
ego-motion compensation
TTC/overlap
normative policy engine
```

## 14. SegHead、SAM/SoM 与指标边界

Reasoning V3.4 当前没有 segmentation head 标签合同，也没有下列字段：

```text
semantic mask
instance mask
bbox
SAM/SoM region id
per-region IoU target
segmentation provenance
```

`loss_mask.segmentation` 仅作为未来兼容接口存在，producer 和 consumer 都要求为 false。
`SceneEntity.entity_ref`、`LocalTraversability.region_ref` 是 semantic relation keys，不是
mask IDs。

因此当前不能声称：

- SAM/SoM weak adapter 已接入 V3.4；
- traversability 有 mask-level ground truth；
- mIoU 可作为 V3.4 smoke 的通过指标；
- segmentation loss 已启用。

未来若接入 SegHead，应使用独立、版本化 sidecar 和 provenance，并在具备真实 mask/weak-mask
生成器、head 和评测代码后再启用 `segmentation`。现有 visual-prompt API 属于 runtime
核心能力，保留且不等于 Reasoning V3.4 segmentation supervision。

## 15. 从 V3.3 删除的字段与原因

| 旧字段/能力 | V3.4 处理 | 第一原理原因 |
| --- | --- | --- |
| `instruction_type` | 删除 | 没有统一客观真值；形成低熵选择题 |
| `local_objective/inferred_objective` | 删除 | 混合文本语义与 teacher-only runtime phase |
| `caption_role` | 删除 | 数据源差异被包装成主观 taxonomy |
| runtime phase、goal reached、goal distance、active waypoint | 从 reasoning/source visible boundary 删除 | student 输入不可见，属于 leakage |
| scene confidence/visibility/calibrated risk | 删除 | 没有校准标签；self-reported confidence 不可当真值 |
| mask、bbox、instance id、track id | 删除 | 当前无统一 GT/SegHead/tracker |
| preferred/selected/fused corridor | 删除 | 依赖未验证投影或人为 corridor ontology |
| route-to-FPV mapping、route matching、route theta | 删除并显式禁止 | GPS/route/pose 不稳定且未验证；route theta 不能替代 action GT |
| source reliability、fusion consistency、execution status | 删除 | 没有客观可计算真值，会变成 teacher 主观打分 |
| interaction affected corridor、required behavior enum | 删除 | 无 tracker/path-overlap/normative GT |
| TTC、overlap、ego-motion compensated track | 删除 | 当前不需要且没有可靠流水线 |
| `motion/speed/social_action/fallback` | 删除 | 当前只有 direction 和 trajectory 是允许的 action supervision |
| Decision confidence | 删除 | 没有 calibration target |
| rejected alternatives | 删除 | 容易由 outcome 反向编造，无独立反事实真值 |
| arbitrary reason enum | 改为受约束开放 `ReasonCode` | 保持可解析格式但不限制长尾事实 |
| `ToolResultUse` 裸扩展 | 禁用 | 未定义 typed 合同，不应塞入六段 |
| section LM / finish LM / weak-LM distillation | 全部关闭 | LVLM weak annotation 不是 ground truth |
| projection/geometry QC threshold | 从 V3.4 quality gate 删除 | 当前不生产也不验证该 label |
| subjective unknown/conflict/insufficient quota | 从 pass/fail gate 删除 | availability report 不能冒充 semantic accuracy |

保留的闭集只用于可执行协议轴：

```text
wire version and section order
requirement polarity
referent resolution
coarse relative position
traversability safety state
context source role and used/ignored
single-frame/multi-frame evidence basis
rationale support status
provenance/loss routing
approved direction profile label space
```

实体类别、surface、relation、rule、reason 和 blocker reason 使用开放
`ReasonCode`，不再由研究者预写完整答案集合。

## 16. V3.3 → V3.4 迁移

### 16.1 新版本标识

```text
urbannav_reasoning_source_v3_4
urbannav_reasoning_observation_v3_4
urbannav_decision_rationale_observation_v3_4
urbannav_reasoning_annotation_v3_4
urbannav_reasoning_trace_v3_4
urbannav_finish_v3_4
urbannav_reasoning_training_view_v3_4
reasoning_v3_4_fixed_supervision_v1
reasoning_v3_4_minimal_composer_v1
urbannav_reasoning_v3_4_minimal_observation_v3
urbannav_reasoning_v3_4_minimal_rationale_v2
urbannav_reasoning_v3_4_minimal_full_cot_v1
urbannav_reasoning_v3_4_minimal_fewshot_empty_v1
urbannav_reasoning_v3_4_minimal_training_policy_v1
```

### 16.2 迁移规则

1. V3.3 source、annotation、cache、review、training view 和 prompt digest 全部归档，不能
   改 version string 后复用。
2. 从 raw/current dataset row 重新执行 V3.4 source boundary filter 和 fingerprint。
3. 旧 `instruction_type/local_objective/corridor/confidence/runtime` 字段不迁移。
4. 旧 teacher rationale 不迁移；V3.4 Pass 1/Pass 2 schema 已改变。
5. 旧 ego 12-way v1 含 route-theta fallback，不可升级 stamp；必须由真实
   `trajectory_ego_t` 按 ego v2 adapter 重建。
6. 旧 direction 行只有 label/name 而没有完整 profile/version/bin/derivation/source stamp
   时，视为缺 direction，进入 observation-only。
7. legacy synthesized `<think>` 不作为 V3.4 target；命中 P0 flag 时强 action view 被过滤。
8. 旧 mixed teacher distillation profile、`require_complete_full_cot_supervision` 和
   `allow_flagged_legacy_sources` fail closed。
9. V2 Full-CoT、VQA、visual-prompt QA 和 agentic runtime 继续作为独立协议，不混入
   V3.4。

## 17. 代码索引

### 17.1 RouteServer producer

| 职责 | 文件 |
| --- | --- |
| 六段、finish、provenance、section mask | `src/mcp_server/contracts/reasoning_v3.py` |
| Pass-1/Pass-2 observation schema | `src/mcp_server/contracts/reasoning_v3_observation.py` |
| source/annotation/training-view wrapper | `src/mcp_server/contracts/reasoning_dataset_v3.py` |
| visible/private source boundary 和 shortcut flags | `src/mcp_server/data/reasoning_v3_source.py` |
| dataset policy、direction allowlist、System1 provenance | `src/mcp_server/data/reasoning_v3_labels.py` |
| visible navigation source registry | `src/mcp_server/data/reasoning_v3_conditions.py` |
| 两个 teacher prompts/parsers | `src/mcp_server/teacher/reasoning_v3_teacher.py` |
| prompt policy/profile/digest | `src/mcp_server/agent/reasoning_v3_prompt_policy.py` |
| 当前空 few-shot registry | `src/mcp_server/teacher/reasoning_v3_fewshot.py` |
| Pass-1/Pass-2 semantic validation | `src/mcp_server/agent/reasoning_v3_observation_validation.py` |
| deterministic composer | `src/mcp_server/data/reasoning_v3_composer.py` |
| target/evidence/provenance validation | `src/mcp_server/agent/reasoning_v3_validation.py` |
| strict renderer/parser | `src/mcp_server/agent/reasoning_v3_text.py` |
| batch、retry、cache 和 observation-only gate | `src/mcp_server/teacher/reasoning_v3_batch.py` |
| audit/strong-action views | `src/mcp_server/data/reasoning_v3_views.py` |
| availability-only quality report | `src/mcp_server/data/reasoning_v3_quality.py` |
| deterministic audit/recomposition | `src/mcp_server/data/reasoning_v3_audit.py` |

### 17.2 ms-swift consumer

| 职责 | 文件 |
| --- | --- |
| V3.4 mirror contract/schema digest | `ms-swift/swift/sidewalk_nav/reasoning_v3.py` |
| Full-CoT prompt runtime | `ms-swift/swift/sidewalk_nav/prompt_v3.py`, `prompt_v3_profiles.py` |
| producer view → training manifest | `ms-swift/swift/sidewalk_nav/data/reasoning_v3_manifest.py` |
| section span parser | `ms-swift/swift/loss_scale/reasoning_v3.py` |
| pluggable direction profiles | `ms-swift/swift/sidewalk_nav/direction_choice_profiles.py` |
| direction candidate/adapter implementation | `ms-swift/swift/sidewalk_nav/direction_choice.py` |

## 18. 真实 smoke 证据与训练前验收条件

### 18.1 2026-07-23 豆包四数据集 smoke

本次使用 `doubao-seed-2-0-lite-260428`，按冻结规则从 SideWalk、Foxglove、
FrodoBots、EgoWalk 各抽取 3 条，共 12 条。它只验证结构、来源、可用性、loss 路由和
审阅产物完整性；不验证语义准确率、置信度校准、几何投影、tracking、mask/bbox 或 TTC。

| 项目 | 实际结果 | 可以得出的结论 |
| --- | --- | --- |
| 输入覆盖 | 12 条；四个数据集各 3 条 | 四条数据适配路径都实际运行 |
| 最终状态 | `accepted=9`，`observation_only=3`，compact latest 中无 rejected/error | 缺方向的 3 条 FrodoBots 没有被伪造成 Full-CoT |
| Direction profile | FPV-20 v2 共 6 条；Ego-12 v2 共 3 条；FrodoBots 无 profile | dataset × profile hard binding 实际通过 |
| Provider 调用 | compact latest 聚合 24 次成功调用；一条 `invalid_json` 曾被严格拒绝，定向重试后通过 | 两阶段真实远程调用与 fail-closed retry 已完成；不能据此声称语义正确 |
| Token | prompt 158,028；completion 10,171；总计 168,199 | 只作为本次最终 latest smoke 的实测资源记录 |
| 延迟 | median 11,279.5 ms；p95 22,359.2 ms | 只适用于本次 endpoint 与小样本 |
| rationale 支持 | 9 条 accepted 全部为 `unsupported`，supporting reasons 为空 | teacher 没有为不可见方向强造视觉理由 |
| 弱观察计数 | observation 12；interaction item 0；traffic rule item 0 | 只报告产出数量，不声称 precision/recall |
| audit view | 9 条；所有 loss 均关闭 | 弱观察只用于审计 |
| strong-action view | 9 条；direction loss 9，System1 loss 9 | 强监督仅走两个独立 sidecar |
| 禁止 loss | reasoning LM 0；segmentation 0 | 未把弱 CoT 或预留 SegHead 接口当训练真值 |
| ms-swift 转换 | 9 条 | producer→consumer 实际转换成功 |
| 人工审阅包 | 12 个样本、69 张内嵌图、0 缺图 | 审阅文件可离线打开 |
| 地图链路 | 未调用 Amap | 符合本版禁止 GPS/route matching 的边界 |
| 凭据落盘 | 未发现 API key material | 运行产物不携带密钥 |
| 人工语义状态 | `pending` | **尚不能宣称语义验收通过** |

9/9 accepted 样本均为 `unsupported` 不是 parser 失败，而是当前 student-visible
instruction/FPV 不能充分解释冻结 action label。这个结果反而支持当前 zero-loss 决策：
不能把“为了迎合轨迹而写出的理由”当作 CoT 真值。如果后续研究目标确实需要可学习的解释，
应重新设计具有可见目标或可见决策证据的数据采样，再单独人工验证；不能用 route/GPS/runtime
私有信息把这些样本改写成 `supported`。

产物根目录：

```text
/home/ubuntu/WorkSpace/project/SR/Code/review_outputs/reasoning_v34_doubao_smoke_20260723_v2
```

关键文件：

```text
annotations/reasoning_v3_4_smoke.latest.jsonl
views/reasoning_v3_4_audit_only.jsonl
views/reasoning_v3_4_strong_action_only.jsonl
ms_swift/reasoning_v3_4_strong_action_only.jsonl
review/reasoning_v3_4_smoke_review_all.complete.html
reports/final.json
```

append-only annotation 与 rejected history 保留了首次 `invalid_json` 和定向重试记录；当前
结果判断必须读取 `reasoning_v3_4_smoke.latest.jsonl`，不能把早期失败尝试计入最终状态。
没有 `_v2` 后缀的早期 smoke 使用了错误的全局 FPV profile 提升 EgoWalk proxy，仅保留为
问题审计，不是 canonical 结果，也不能用于训练或论文统计。

### 18.2 静态合同 smoke

- RouteServer 与 ms-swift semantic JSON Schema 完全相等；
- schema digest 与 Full-CoT policy digest 相等；
- strict renderer/parser roundtrip；
- 六段缺失、重复、乱序、自由文本和旧字段均被拒绝；
- 两个 approved direction profiles 的 version/index/source gate 一致；
- ego v1、route theta、partial stamp、越界 index 全部 fail closed；
- all-language、finish、readout 和 segmentation loss 均为 false。

当前定向合同测试结果：RouteServer `63 passed`，ms-swift `58 passed`；覆盖 schema/prompt
parity、strict roundtrip、旧字段拒绝、dataset × profile binding、profile candidate
identity、source filtering、observation-only、provenance、view loss 和 versioned manifest
转换。该结果只说明代码合同通过，不替代真实训练回读。

### 18.3 每数据集小样本 smoke

对 SideWalk、Foxglove、EgoWalk、FrodoBots 分别抽取人工可读小批次，报告：

```text
source normalization success/error
filtered private-field paths
Pass-1 parse/validation/retry
accepted / observation_only / rejected / error
Pass-2 call count
direction profile and stamp completeness
System1 eligibility
P0 shortcut flags
evidence closure errors
latency and provider cost
```

硬条件：

- 无方向样本不调用 Pass 2；
- FrodoBots 全部 observation-only，direction/System1 view 为 0；
- Pass 1 prompt 无 canonical direction 和 trajectory；
- Pass 2 prompt 无 raw trajectory、GPS、route geometry、runtime state；
- EgoWalk FPV proxy 不产生 FPV strong direction stamp；
- 任意 route-theta-derived row 为 0；
- quality/audit 报告只声称 availability/provenance/schema/evidence，不声称 accuracy。

### 18.4 人工语义审查

人工至少检查：

- requirement/referent span 是否真实来自 instruction；
- entity/control/surface 名称是否能从图像观察；
- unknown 是否在证据不足时使用，而不是 forced fit；
- BEV/goal/route/tool 是否只作为 context；
- traversability 是否没有 trajectory/route 反推痕迹；
- interaction 是否没有单帧伪造动态关系；
- traffic rule 是否绑定真实可见 control；
- unsupported direction 是否诚实保留；
- rationale 是否只是解释冻结方向而没有修改它；
- 不存在 mask/bbox/track/TTC/confidence/runtime leakage。

### 18.5 强监督与训练 smoke

- `audit_only` row 的所有 logical loss 为 false；
- `strong_action_only` 只按 sidecar availability 独立开启 direction/System1；
- 命中 P0 shortcut flag 的 row 不产生 strong-action view；
- 无 direction 但有 System1 的 observation-only row不因 System1 被拼成 Full-CoT；
- ms-swift 实际 tokenizer 的 assistant span 权重全部为 0；
- direction head 只消费正确 profile 下的 label；
- System1 head 只消费合法 `trajectory_ego_t/trajectory_policy/action_mask`；
- 保存 checkpoint 后做 generation、direction 和 System1 回读；
- 分数据集、profile、城市/clip group 报告指标，不能只给 aggregate。

当前只完成了 18.1 的真实小样本运行和其中可自动验证的结构性检查。只有人工 review 与
训练回读继续落盘并通过后，才允许启动全量 V3.4 标注；本次 smoke 的 `passed=true`
不得被解释为人工语义或训练验收已经通过。

## 19. 当前非目标

V3.4 minimal 不实现：

```text
Full-CoT language distillation
VQA task registry
online Agent node
ToolResultUse
ask_user / complete_task / abort_task dataset decisions
speed/social/fallback action heads
GPS/route geometry training
route matching and projection
SegHead/SAM/SoM supervision
tracking/TTC
normative policy GT
confidence calibration
```

这些方向未来都可以研究，但必须各自拥有版本化输入、真实来源、student 可预测性说明、
provenance、loss channel、validator 和独立消融；不能继续向 V3.4 六段中堆字段。
