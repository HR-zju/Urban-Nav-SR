# UrbanNav Navigation VQA V2 数据标注与训练合同

> 状态：V2.0 contract、source builder、eligibility、question builder、producer、
> teacher、training view、audit、review 和 evaluator 已实现；已通过 synthetic/fake-backend
> focused tests，尚未完成真实 LVLM 人工 smoke、真实数据产量审计或训练收敛验证。
>
> Canonical owner：RouteServer。训练侧生成镜像与 LM row adapter 位于 `ms-swift`。
>
> VQA V1 已被 V2.0 取代，不保留 parser、manifest 或双写兼容入口。

## 1. 研究目标与第一性边界

Navigation VQA V2 与 Reasoning V4 Full CoT 是两套并列监督：

- V4 学习六段完整导航推理和 `<finish>`；
- VQA V2 学习一个 answer-independent 原子问题对应的最小 typed answer；
- 二者共享 `NavObservationV4` 和同一套 direction profile identity；
- VQA 不从 V4 section 反投影答案，也不把 teacher-only trajectory 暴露给 student。

一个 VQA task 只有同时满足以下条件才能进入训练：

1. 问题能够由 student 实际获得的输入回答；
2. 答案有明确 producer，来源是现有数据确定性派生、student-visible LVLM weak label，
   或未来通过 QC 的 external GT；
3. producer、输入、question、schema 和 capability 版本均可回溯；
4. question 不携带答案，answer 不复制已经给出的 conditioned direction；
5. training view 不包含 private trajectory、private source interface、Full CoT target、
   segmentation target 或 route/GPS sidecar。

因此当前不声称：

- pixel-level goal geometry、统一 waypoint frame 或独立 BEV route GT；
- BEV–FPV entity identity、future-direction truth 或 candidate trajectory preference GT；
- mask/bbox/instance tracking、TTC、calibrated confidence 或 metric clearance；
- GPS、route matching、route theta 或 private trajectory 可见性；
- LVLM weak label 等于人工真值。

## 2. 跨仓职责与完整数据流

```text
existing ReasoningSourceV4
  ├─ NavObservationV4                 student-visible
  ├─ DirectionSupervisionV4           private derived GT
  └─ trajectory/System1 sidecars      private, never copied
          |
          v
VQASourceV2
  ├─ observation                      visible partition
  ├─ direction_supervision            private label/question constructor input
  └─ interfaces                       private future-GT bindings
          |
          v
capability + per-sample eligibility
  ├─ unavailable -> audit counter only, no question/annotation/view
  └─ eligible -> answer-independent VQAQuestionV2
          |
          +------------------------------+
          |                              |
          v                              v
canonical E0 producer              LVLM visible teacher
derived direction GT               A/B/C1/D/E4 weak label
          |                              |
          +--------------+---------------+
                         v
                 accepted VQAAnnotationV2
                         |
              provenance/leakage validation
                         v
                 VQATrainingViewV2
                         |
       generated schema/profile parity boundary
                         v
ms-swift fresh messages row + ordinary full-target LM loss
```

RouteServer 负责 canonical Python contract、direction registry、source/annotation/view、
teacher、audit、review、evaluation 和 generated artifacts。`ms-swift` 只消费 RouteServer
生成的 schema/Python mirror，把 training view 转成新的 allowlisted messages row，并负责
训练和模型生成回读。在线 Agent runtime 不参与当前 VQA。

## 3. 版本与能力配置

V2.0 的任务全集是 27 个原子 task。任务是否可生产不写死在 schema enum 中，而由
[`vqa_v2_stage_a_capabilities.json`](../../config/vqa_v2_stage_a_capabilities.json) 冻结：

稳定的 task semantic registry `VQA_V2_TASK_SPECS` 另外为每个 task 冻结 answer model、
required modalities、dynamic eligibility requirements、allowed label sources、private
source-interface field、assistant projector 和 evaluation adapter。该 registry 不包含
active/dormant 状态；实验启用仍只能由 capability config 控制。

| 字段 | 定义 |
| --- | --- |
| `task_id` | 原子任务稳定 ID |
| `contract_ready` | typed question/answer contract 是否存在；V2.0 全部为 true |
| `source_interface_ready` | 强来源的输入接口是否已定义；不表示已有数据 |
| `label_producer_ready` | 是否已有实际 producer |
| `training_export_enabled` | 是否允许生成 training view |
| `evaluation_enabled` | 是否允许以该任务产生 reference/evaluation |
| `producer_ids[]` | 获准 producer allowlist |
| `depends_on[]` | 激活前必须已经激活的依赖任务 |
| `disabled_reason` | 关闭任务的稳定原因码 |

validator 强制：

- 没有 producer 不能打开 training/evaluation；
- 打开 producer 必须有 `producer_ids`；
- 关闭的 task 必须携带 `disabled_reason`；
- 打开的 task 不能依赖关闭 task；
- capability digest 必须在 RouteServer view 与 ms-swift mirror 中一致。

当前有 18 个可生产任务：17 个 visible LVLM weak tasks 和 1 个 derived-GT E0。具体样本
仍会因 instruction/goal、direction 或 temporal evidence 不存在而跳过。其余 9 个 task
只保留 private source interface，不产生空 annotation，也不进入 training schema。

## 4. 共享输入、问题、答案与 provenance 字段

### 4.1 `VQASourceV2`

| 字段 | 可见性 | 来源 | 作用 |
| --- | --- | --- | --- |
| `schema_version` | control | 固定 wire version | 拒绝 V1 或其他隐式兼容 |
| `observation` | student-visible | 原 `NavObservationV4` | 图像、角色、instruction、visible context、evidence catalog、modalities |
| `direction_supervision` | private | 现有数据集轨迹按批准 profile 确定性派生 | 构造 E0 label；为 direction-conditioned question 指定问题条件；不进入 view |
| `interfaces` | private | 未来 external GT pipeline | C2–C6、E1–E3 的强类型占位绑定；当前均为空且关闭 |
| `warnings[]` | audit only | source adapter | 记录非致命数据问题 |
| `source_fingerprint` | audit only | canonical JSON SHA-256 | 冻结整个 source，防止 annotation 与输入漂移 |

`build_vqa_source_v2` 只接受 `ReasoningSourceV4`。它复制 `observation` 和已验证 direction
stamp，不复制 `trajectory_ego_t`、`trajectory_policy` 或 `action_mask`。V1 schema
直接以 `legacy_vqa_v1_not_supported` 拒绝。

### 4.2 `VQAQuestionV2`

| 字段 | 定义与约束 |
| --- | --- |
| `task_id` | 27 个 task 的稳定 wire ID |
| `template_id` | answer-independent question template 版本 |
| `question_text` | 固定自然语言问题；禁止 region/region_r* 术语 |
| `direction_profile_id` | direction task 使用的 profile；不是答案 |
| `conditioned_direction_label` | B/D 单方向判断题明确要求检查的方向；answer 不得复制该值 |
| `candidate_direction_labels[]` | comparison/selection 题允许选择的有序候选集 |
| `target_refs[]` | prompt 中实际存在的 goal/waypoint/BEV target refs |
| `evidence_refs[]` | 本题可使用的 raw evidence 闭包 |
| `candidate_set_id` | E1–E3 冻结候选集合与 presentation order 的 ID |
| `visible_candidates[]` | `candidate_id` 到 student-visible representation ref 的显式映射 |
| `target_candidate_id` | E2/E3 明确指定要分类或判断的候选；必须属于 visible candidates |

所有 candidate selection 字段必须从 question 候选中选择；方向列表必须是候选子集。
候选顺序由 source fingerprint 确定性扰动，不能编码 preference。
student prompt 会逐项渲染本题 `evidence_refs` 对应的可见 evidence catalog，并单独列出
`target_refs`；teacher 与 student 因此读取同一个 evidence 闭包，不能只让 teacher
看到 target ID 或属性。

### 4.3 `VQAAnnotationV2`

| 字段 | 定义与作用 |
| --- | --- |
| `annotation_id` | `source + task` 的稳定 label identity |
| `source_group_id/source_fingerprint` | 与冻结 source 严格 join |
| `question/answer` | typed task-discriminated question 与 payload |
| `label_source` | `derived_gt / lvlm_weak / external_gt` |
| `source_kind` | 受约束开放 provenance reason code |
| `evidence_refs[]` | student-visible raw evidence；必须存在于 source catalog |
| `source_field_refs[]` | derived/external GT 的精确源字段；weak label 可为空 |
| `producer_id/version` | producer allowlist identity |
| `input_fingerprint` | producer 实际输入摘要 |
| `annotation_context_digest` | visible observation、question 与 producer policy 的联合 digest |
| `prompt_profile_id/prompt_policy_digest` | LVLM weak label 必填；其他 label 禁止 |
| `teacher_provider/model` | LVLM weak label 模型来源；不作为 student input |
| `metadata` | 有界审计信息；不进入 assistant target |

只保存 `accepted` annotation。unavailable、rejected 和 provider error 进入 batch report，
绝不使用空 payload 或默认 `none` 冒充负样本。
annotation validator 还会按 task spec 校验 `label_source`：E0 只接受 `derived_gt`，
当前 weak tasks 只接受 `lvlm_weak`，dormant interfaces 未来只接受 `external_gt`。

### 4.4 `VQATrainingViewV2`

training view 只保留 `images/image_roles/instruction/model_visible_context/evidence_catalog/
input_modalities`、question、最小 `assistant_text` 与审计 identity。以下字段明确不出现：

```text
interfaces
direction_supervision
trajectory_ego_t
pixel_trajectory
action_mask
reasoning_target
section_loss_mask
segmentation_target
```

`optional_visual_interfaces` 和 `optional_metrics` 只允许固定的 unavailable 状态：
SegHead、SAM/SoM weak adapter、mIoU 等均为 false/null。它们不是训练 target。

## 5. 27 个任务的科研问题、字段、来源与训练状态

表中的“可推断”指 student 能从实际 prompt 输入和 question 推断；不表示答案是客观世界
完备真值。“LM”表示整个最小 `<VQAAnswer>{payload}</VQAAnswer>` 使用普通 token
cross-entropy。

| Task | 科研问题 | 最小答案字段 | 当前标签来源 | Student 可推断 | V2.0 loss |
| --- | --- | --- | --- | --- | --- |
| A1 `goal_understanding` | 模型是否理解明确目标而非 runtime phase | `goal`, `status` | visible LVLM weak | 是，需 instruction/visible goal | LM |
| A2 `explicit_requirement` | 是否识别明确写出的约束/偏好 | `requirements[]`, `status` | visible LVLM weak | 是，需 task context | LM |
| A3 `task_ambiguity` | 是否区分语言歧义与场景未知 | `ambiguities[]`, `status` | visible LVLM weak | 是，需 task context | LM |
| B1 `direction_affordance` | 给定方向当前可否通行 | `affordance` | FPV visible LVLM weak | 是，非 metric | LM |
| B2 `direction_clearance` | 给定方向的粗语义净空 | `clearance` | FPV visible LVLM weak | 是，不声称距离 GT | LM |
| B3 `direction_surface_effect` | 可见支撑面怎样影响运动 | `surface`, `effect` | FPV visible LVLM weak | 是，surface 为开放 code | LM |
| B4 `conditional_affordance` | 能否立即使用及先决行为 | `usability`, `condition_code`, `effect` | FPV visible LVLM weak | 是 | LM |
| B5 `direction_comparison` | 两个可见局部方向哪一个更合适 | `preferred_direction_label`, `reason_codes[]` | visible LVLM weak | 是，只选 question candidates | LM |
| B6 `unavailable_directions` | 候选集中哪些当前不宜使用 | `direction_labels[]`, `reason_codes[]` | visible LVLM weak | 是；空集合法，不表示完备 free space | LM |
| C1 `goal_to_direction` | 可见 goal 对应哪个 local direction | `direction_label`, `candidate_direction_labels[]`, `status` | visible LVLM weak | 是，需 goal/goal image | LM |
| C2 pixel goal | pixel goal 到 direction 的确定性映射 | `direction_label`, `status` | 未来 calibrated external GT | 当前否：坐标/尺寸/stamp 未冻结 | 关闭 |
| C2 waypoint | waypoint 到 direction 的确定性映射 | `direction_label`, `status` | 未来 frame/unit external GT | 当前否：frame/unit 未统一 | 关闭 |
| C3 BEV route alignment | independent BEV route 在 FPV 的方向 | `direction_label`, `alignment_status` | 未来 independent route GT | 当前否；禁止 route matching/GPS/theta | 关闭 |
| C4 BEV object anchoring | BEV 元素与 FPV entity 的对应 | `bev_ref`, `fpv_entity_ref`, `direction_label`, `status` | 未来 cross-view binding GT | 当前否：无可靠 entity ID | 关闭 |
| C5 outside-view direction | 区分 current 与 future direction 信息 | `outside_view_information[]`, `current_direction_label`, `future_direction_label` | 未来 separated temporal GT | 当前否：未验证 current/future 分离 | 关闭 |
| C6 BEV–FPV executability | BEV 意图在当前 FPV 是否可执行 | `direction_label`, `status` | 未来 C3 + local executability GT | 当前否；依赖关闭的 C3 | 关闭 |
| D1 `agent_relevance` | 可见 agent 是否影响给定方向 | `relevance` | visible LVLM weak | 是，不生成 instance ID | LM |
| D2 `static_interaction_relation` | 单帧静态占用/邻近关系 | `relation` | visible LVLM weak | 是，仅静态 | LM |
| D3 `temporal_interaction_relation` | 多帧是否支持动态关系 | `relation` | ordered visible FPV LVLM weak | 仅 distinct history eligible 样本 | LM |
| D4 `applicable_constraint` | 当前哪类约束作用于给定方向 | `constraint_code`, `kind`, `effect`, `affected_direction_labels[]` | visible LVLM weak | 是，constraint code 开放 | LM |
| D5 `appropriate_response` | 采取给定方向前应如何响应 | `effect` | visible LVLM weak | 是 | LM |
| D6 `traffic_control_applicability` | 可见 control 是否实际控制该方向 | `applicability`, `effect`, `controlled_direction_labels[]` | visible LVLM weak | 是，不把“看见”当“适用” | LM |
| E0 `demonstrated_direction_selection` | 从输入预测数据示范的短时方向 | `direction_label` | approved profile 的 existing trajectory-derived GT | 是，label 私有、输入可见 | LM |
| E1 candidate preference | prompt-visible candidates 的 preference | `preferred_candidate`, `reason_codes[]` | 未来 candidate-set external GT | 当前否：没有显式候选集 | 关闭 |
| E2 candidate classification | 指定 visible candidate 的 direction 类别 | `direction_label` | 未来 candidate direction stamp | 当前否：private trajectory 不可见 | 关闭 |
| E3 trajectory compatibility | visible candidate 与任务/场景是否兼容 | `status`, `reason_codes[]` | 未来 compatibility external GT | 当前否：无 visible candidate + strong GT | 关闭 |
| E4 `alternative_direction` | 当前方向不合适时选择哪个 supplied alternative | `alternative_direction_label`, `reason_codes[]`, `status` | visible LVLM weak | 是，只选 question candidates | LM |

### 5.1 答案闭集与开放字段

闭集只用于状态或动作协议：

- A：resolved/absent/ambiguous/insufficient；
- B：direct/conditional/blocked/forbidden、semantic clearance、行为 effect；
- C：alignment/executability status；
- D：static/temporal relation、constraint kind、control applicability；
- E：compatibility/resolution status。

开放长尾内容使用 `ReasonCode`（lower-snake-case 受约束字符串）：
requirement、ambiguity、surface、condition、constraint、reason codes。自然语言 goal 仍是短开放
文本。没有 `instruction_type`、`local_objective`、region ontology 或 dataset-specific
封闭选择题。

## 6. Direction profile 是共享的实验合同

VQA 与 Full CoT 共同引用
[`direction_profile.py`](../../src/mcp_server/contracts/direction_profile.py)，不各自复制枚举。
V2.0 允许两个完整 profile stamp：

| Profile | Bin | 来源 | Stop rule |
| --- | --- | --- | --- |
| `fpv_degree10_terminal_centroid_v2` | -90° 到 90° 每 10° + stop，20 类 | existing normalized FPV pixel trajectory terminal centroid | displacement `< 0.06` |
| `ego_degree10_12way_terminal_centroid_v2` | stop、straight、左右 10/20/30/40/50+，12 类 | existing `trajectory_ego_t`, `x_right/y_front` | displacement `< 0.25 m` |

每条 label 必须携带：

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

profile/version/index/name/choice/angle 不一致时 fail closed；`source` 出现
`gps/gnss/route_match/route_theta` 时拒绝。E0 的 label 从 private stamp 生成，但
training view 和 ms-swift row 不携带该 stamp，只保留最小 direction answer。

## 7. 当前关闭任务的接口，不是空训练样本

[`vqa_dataset_v2.py`](../../src/mcp_server/contracts/vqa_dataset_v2.py) 为 C2–C6、E1–E3
定义了以下 private interface：

| Interface | 未来激活必须具备 |
| --- | --- |
| 所有 external interfaces | `source_group_id`、producer/version、input fingerprint、accepted QC、student-visible evidence refs、source field refs |
| `PixelGoalDirectionInterfaceV2` | pixel source ref、frame、width/height、coordinate convention、direction stamp、answer-stamp equality |
| `WaypointDirectionInterfaceV2` | waypoint source ref、frame、unit、transform version、direction stamp、answer-stamp equality |
| `BEVRouteAlignmentInterfaceV2` | independent source、BEV/FPV evidence refs、answer-stamp equality；禁止 trajectory/GPS/route matching/theta |
| `CrossViewEntityBindingInterfaceV2` | validated `bev.*` ref、`fpv.*` entity ref、direction stamp；两个 ID 都必须实际出现在 prompt-visible evidence catalog |
| `OutsideViewDirectionInterfaceV2` | 不同 current/future source、horizon、prompt-visible transition ref、`future_not_yet_executed=true` |
| `BEVFPVExecutabilityInterfaceV2` | validated BEV direction、BEV/FPV evidence refs、local executability producer、answer-stamp equality |
| `CandidateSetV2` | candidate-set ID、candidate producer/version、每个 candidate 的 visible representation ref、direction stamp 与固定 presentation order |
| E1/E2/E3 bindings | preference、direction 或 compatibility GT 与 exact candidate set 对齐 |

这些 interface 只能位于 private `VQASourceV2.interfaces`。当前 capability 关闭时：

- question builder 不生成问题；
- producer 返回 `unavailable` 且 annotation 为 `None`；
- training view/exporter 拒绝；
- evaluator 指标为 `N/A`；
- ms-swift 不接收这些 task；
- 不生成 null target、默认 negative 或 loss mask 占位 row。

未来真值流水线完成后，应发布 V2.1 capability/profile 和 provenance producer，而不是在
V2.0 中静默打开。

## 8. LVLM weak teacher 与人工 smoke

17 个 weak task 各有独立 prompt profile、payload JSON Schema 和 policy digest。teacher
只接收完整 student-visible observation、answer-independent question 和实际图像：

- current FPV 对即时可见安全权威；
- instruction/goal/waypoint/BEV 只在真实存在时作为条件；
- 单帧只能作静态判断；
- D3 必须有不同历史 FPV；
- 证据不足应返回 schema 中的 unknown/insufficient 状态；
- 输出必须恰好一个 `<VQAAnswer>`，无 rationale、task id、Markdown 或额外文本。

真实付费/外部标注前必须先跑小规模 smoke，生成自包含 HTML 供人工逐条检查。smoke
通过只表示 pipeline 与样例可审，不表示语义收敛或可以全量生产。

### 8.1 人工校阅与过滤合同

人工校阅不原地修改 canonical annotation，而是按 `annotation_id` 导出独立
`urbannav_navigation_vqa_review_decision_v2_0` sidecar。每条决定绑定
`source_group_id/source_fingerprint/task_id/label_source`，防止把同一 source 下其他 VQA
问题一起误删，也防止 decision 被应用到不同 source revision。

校阅页按 `dataset × task_id × label_source` 做确定性 SHA-256 分层抽样，支持：

- `keep`：标签可以继续进入训练；
- `reject`：标签错误，排除 annotation 和对应 training view；
- `needs_fix`：排除当前训练池并进入独立返修队列；
- `unreviewed`：仅为 UI 草稿状态，不写入 decision JSONL。

浏览器使用 `localStorage` 保存当前 review-set 草稿，并可导入、导出 reviewed-only JSONL。
抽样校阅不能静默决定未抽中样本，因此过滤工具要求显式指定
`--unreviewed-policy error|keep|drop`。常规抽样补丁使用 `keep`；全量人工门禁使用默认
`error`。过滤器流式同步 annotation 和 training view，并按 `annotation_id` 保持两者一致。

## 9. ms-swift loss 与训练语义

ms-swift 对每个 view 新建 messages row：

```text
system: canonical VQA student policy
user: actual images + complete visible context + one atomic question
assistant: <VQAAnswer>{minimal payload}</VQAAnswer>
```

最后一个 assistant message `loss=True`，不使用独立 direction head，也不使用
`loss_scale`。row 的显式 mask 为：

```json
{
  "lm": true,
  "vqa_answer": true,
  "system1_action": false,
  "trajectory": false,
  "segmentation": false
}
```

因此 E0 direction 同样由 LM token loss 学习。VQA row 不计算 trajectory、System1 或
segmentation loss；Phase-A dataset mixer 可以把 VQA row 与 Full CoT/trajectory row 放在
同一训练中，但各 row 的 loss route 必须独立。

## 10. 评测与审计

当前离线 evaluator 按 `annotation_id` join reference/prediction，输出：

- coverage 与 missing prediction；
- canonical parse fraction；
- exact payload match；
- per-field accuracy；
- direction accuracy；
- per-task metrics；
- derived/external GT 与 teacher-agreement scope 分开计数。

LVLM weak reference 的“准确率”只能称 teacher agreement，不能称人工/世界真值准确率。
关闭 task 和未实现 dense metric 显式输出 `N/A`。当前不输出 mIoU、tracking、TTC、
route projection 或 calibrated confidence。

artifact audit 另外检查：

- source/annotation/view schema 与 fingerprint；
- capability/producer/prompt digest；
- evidence refs 是否在 student-visible catalog；
- private field leakage；
- task、dataset、label source 分布；
- unavailable/rejected/error 原因；
- dormant/dense 指标是否保持 N/A。

## 11. 代码与命令入口

Canonical implementation：

```text
src/mcp_server/contracts/direction_profile.py
src/mcp_server/contracts/vqa_v2.py
src/mcp_server/contracts/vqa_dataset_v2.py
src/mcp_server/agent/vqa_v2_text.py
src/mcp_server/agent/vqa_v2_prompt_policy.py
src/mcp_server/data/vqa_v2_source.py
src/mcp_server/data/vqa_v2_eligibility.py
src/mcp_server/data/vqa_v2_questions.py
src/mcp_server/data/vqa_v2_producers.py
src/mcp_server/data/vqa_v2_review.py
src/mcp_server/data/vqa_v2_views.py
src/mcp_server/data/vqa_v2_audit.py
src/mcp_server/teacher/vqa_v2_profiles.py
src/mcp_server/teacher/vqa_v2_teacher.py
src/mcp_server/teacher/vqa_v2_batch.py
src/mcp_server/eval/vqa_v2.py
```

离线流水线：

```bash
PYTHONPATH=src python tools/model_processing/build_vqa_v2_sources.py \
  --input-jsonl reasoning_v4_sources.jsonl \
  --output-jsonl vqa_v2_sources.jsonl

# 仅在已授权且环境变量中提供 provider key 时调用真实 LVLM。
PYTHONPATH=src python tools/model_processing/label_vqa_v2.py \
  --source-jsonl vqa_v2_sources.jsonl \
  --annotation-jsonl vqa_v2_annotations.jsonl \
  --report-json vqa_v2_label_report.json

PYTHONPATH=src python tools/model_processing/export_vqa_v2_views.py \
  --source-jsonl vqa_v2_sources.jsonl \
  --annotation-jsonl vqa_v2_annotations.jsonl \
  --output-jsonl vqa_v2_views.jsonl

PYTHONPATH=src python tools/model_processing/build_vqa_v2_review_html.py \
  --source-jsonl vqa_v2_sources.jsonl \
  --annotation-jsonl vqa_v2_annotations.jsonl \
  --views-jsonl vqa_v2_views.jsonl \
  --output-html review_outputs/vqa_v2/smoke_review.html \
  --report-json review_outputs/vqa_v2/smoke_review.json

# 标准三数据集 run root 的小规模分层人工校阅页；默认嵌入去重缩略图。
PYTHONPATH=src python tools/model_processing/build_vqa_v2_review_html.py \
  --dataset-root /path/to/vqa_run_root \
  --datasets foxglove sidewalk egowalk \
  --samples-per-stratum 1 \
  --max-annotations 48 \
  --output-html review_outputs/vqa_v2/sample_review.html \
  --report-json review_outputs/vqa_v2/sample_review.report.json

# 将浏览器导出的 decision sidecar 同步应用到 annotation 与 training view。
PYTHONPATH=src python tools/model_processing/apply_vqa_v2_review_decisions.py \
  --annotation-jsonl vqa_v2_annotations.jsonl \
  --training-view-jsonl vqa_v2_views.jsonl \
  --decision-jsonl vqa_v2_review_decisions.jsonl \
  --output-annotation-jsonl vqa_v2_annotations.filtered.jsonl \
  --output-training-view-jsonl vqa_v2_views.filtered.jsonl \
  --rejected-annotation-jsonl vqa_v2_annotations.rejected.jsonl \
  --needs-fix-annotation-jsonl vqa_v2_annotations.needs_fix.jsonl \
  --unreviewed-policy keep \
  --report-json vqa_v2_review_filter.report.json

PYTHONPATH=src python tools/model_processing/audit_vqa_v2_artifacts.py \
  --source-jsonl vqa_v2_sources.jsonl \
  --annotation-jsonl vqa_v2_annotations.jsonl \
  --training-view-jsonl vqa_v2_views.jsonl \
  --output-json vqa_v2_audit.json
```

契约同步与 parity：

```bash
PYTHONPATH=src python tools/model_processing/export_vqa_v2_contract_artifacts.py \
  --output-dir schemas/vqa_v2 --check

PYTHONPATH=src python tools/model_processing/sync_vqa_v2_cross_repo.py \
  --ms-swift-root /home/ubuntu/WorkSpace/project/SR/Code/ms-swift \
  --check
```

## 12. 当前验收边界

已经可以验证：

- 27-task typed contract 和 18/9 capability gate；
- exact minimal wire parser/render；
- E0 deterministic producer；
- visible-only fake LVLM teacher path；
- source/annotation/view leakage gate；
- RouteServer–ms-swift schema、golden parser 与 Python mirror parity；
- ms-swift fresh LM-only row；
- offline evaluator 与 standalone review HTML。
- 27/27 answer valid/invalid golden fixtures；
- 9/9 dormant-interface valid/invalid provenance fixtures，以及 C3 forbidden-source fixture。

仍需真实 evidence 才能声称：

- 豆包或其他 teacher 的语义质量；
- 各数据集实际 eligible/accepted/rejected 分布；
- 人工 review 通过；
- 全量标注成本与延迟；
- VQA 对导航方向、ADE/FDE、collision 或 route progress 的贡献；
- C2–C6、E1–E3 任一任务已解锁。
