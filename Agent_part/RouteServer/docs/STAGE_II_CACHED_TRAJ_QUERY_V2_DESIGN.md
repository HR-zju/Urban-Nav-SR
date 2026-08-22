# UrbanNav Stage II Cached Trajectory-Query V2 设计

## 1. 文档定位

本文档是 Urban-Nav-SR 侧 `cached_traj_query_v2` 的集中设计和实现边界说明。
它补充 [STAGE_II_CFM_DATA_CONTRACT.md](./STAGE_II_CFM_DATA_CONTRACT.md) 中的
Stage II canonical data contract，并与 ms-swift 的双频 runtime、训练和 native
bundle 文档配套使用。

本文档只描述已经确定的合同、代码职责和验收门禁。代码合同闭合、单元测试通过、真实
checkpoint 学会条件输入以及实时部署验收是不同等级的证据，不能相互替代。

当前状态：

```text
schema / source-view / pair materializer：已实现
跨 split、时间和 target 隔离规则：已实现
ms-swift 的 V2 adapter / runtime 接口：已实现
正式训练后的 cached-query checkpoint：需要真实训练结果确认
native bundle、shuffle ablation、长序列双频 replay：未完成生产验收
ONNX/TensorRT/HIL/实物部署：未实现
```

## 2. 目标与非目标

### 2.1 目标

建立以下因果路径：

```text
低频 Stage I query observation
    -> query hidden states Q

高频 Stage II action observation
    -> 当前视觉 memory + 已缓存 query hidden states
    -> Flow Matching action head
    -> 未来轨迹 / action chunk
```

Stage I 和 Stage II 使用同一 parent 的输入合同，但不要求每个 action tick 等待一次
Stage I。query 必须来自不晚于 action observation 的历史输入，并由时间戳、模态 profile
和条件 digest 绑定。

### 2.2 非目标

本方案不引入以下未被当前训练合同支持的结构：

- 不新增 `[SEM_i]` semantic query；
- 不新增 semantic projector；
- 不把 `straight`、`left_20`、`stop` 等文本方向自动伪装成 action-head 条件；
- 不把 `traj_query_only` 描述成已经完成的显式 direction-conditioned action head；
- 不把 Pass-R、Pass-D 文本缓存作为 V2 的必要前置；
- 不把 future trajectory、GT direction 或归一化 target 写入模型可见 prompt；
- 不在 RouteServer 中保存 tokenizer、query position、pixel tensor 或 normalized tensor。

如果后续要让离散 direction 直接控制 Stage II，必须新增独立的 direction-conditioned
action-head contract、训练视图和 shuffle 验证，不能借用本文的 latent query 合同代替。

## 3. 两个主框架的责任边界

### 3.1 Urban-Nav-SR / RouteServer

RouteServer 负责不可变的数据和合同事实：

- Stage I parent、source group 和 split group 身份；
- FPV、BEV、instruction 等模型可见输入及其 modality profile；
- raw pose/odometry 到 current-ego FLU 的轨迹真值构造；
- 4 秒/32 点 target、mask、source registry 和 trajectory QC；
- query/action observation 的时间关系、模态关系和 condition digest；
- source view、paired training view、reject/drop audit 和 digest；
- 跨仓 schema artifact 的导出和 parity 检查。

RouteServer 不负责：

- tokenizer 或 chat template；
- `[TRAJ_i]` 的 token index；
- query hidden state 的具体 tensor；
- query 数量 `Q` 的模型实现；
- visual memory、fast visual adapter 或 CFM action head；
- normalized trajectory tensor 和 GPU 推理。

### 3.2 ms-swift

ms-swift 负责模型和运行时实现：

- V2 manifest adapter 和 prompt/input 编译；
- query hidden state 的提取与缓存；
- 当前视觉 memory 的编码；
- query 与 visual memory 的融合；
- CFM action head、loss、normalization 和 checkpoint sidecar；
- native bundle 加载、双频调度和推理指标。

两边通过 versioned schema、projection policy、source digest、modality profile 和
condition mode 连接，不通过隐式字段猜测格式。

## 4. Canonical condition contract

### 4.0 输入责任冻结

`instruction` 只作为 Stage I 的导航输入；当前 instruction profile 缺失时直接拒绝。`waypoint/goal`
采用相同的 Stage I 责任边界，只有独立来源、声明式 Stage I profile、goal encoder、训练产物和
Bundle capability 同时存在时才进入 Stage I。Stage II 的 `instruction`、`waypoint/goal` 和 route
revision 只是可选 request metadata，缺省时沿用已绑定的 cached query，携带时做一致性校验和
ledger 记录。当前 Stage II action head 不直接读取这些字段。

后续待做必须按此顺序执行：先完成 waypoint/goal Stage I capability，再验收 Stage II optional
presence/absence、binding consistency 和 shuffle，最后才做双频长序列及部署 gate。不得通过修改
YAML、增加 request 字段或复用现有 `cached_traj_query_v2` 名称，把未训练的 waypoint 能力标记为可用。

### 4.1 版本标识

RouteServer 当前冻结的 V2 标识为：

```text
training_contract:
    stage_ii_cfm_v2

condition_mode:
    cached_traj_query_v2

binding schema:
    urbannav_stage_ii_cached_traj_query_binding_v2

source-view schema:
    urbannav_stage_ii_cached_traj_query_source_view_v2

training-view schema:
    urbannav_stage_ii_cached_traj_query_training_view_v2

trajectory policy:
    urbannav_stage_ii_metric_flu_4s_32_speed_audit_v2_1
```

V1 `cached_traj_query_v1` 和普通 `traj_query_only` 继续兼容旧数据和旧 checkpoint，
不能与 V2 manifest、checkpoint 或 bundle 静默混用。

### 4.2 Observation

`StageIICachedTrajQueryObservationV1` 表示一次不含 future target 的模型可见观察。
它包含：

```text
sample/source/split identity
dataset_source
observation_timestamp_sec
modality_profile_id
images + image_roles
instruction / model-visible context
evidence catalog
projected_input_digest
```

它不包含：

```text
trajectory_target
action_mask
future pose
private direction supervision
query tensor
Pass-R / Pass-D text cache
```

因此 source view 可以作为 query observation 或 action observation 的共同输入来源，
但本身不是最终的训练 prompt row。

### 4.3 Source view

`StageIICachedTrajQuerySourceViewV2` 将一个可见 observation 与同一 physical source
对应的私有 metric trajectory target 绑定。它用于 materializer，不直接交给模型。

必须满足：

1. observation、target、source group 和 split group 身份一致；
2. target source provenance 与 source group 一致；
3. view digest 对所有字段可复现；
4. 模型可见字段中没有 target-like private key；
5. 轨迹 target 已通过 versioned source registry 和 trajectory QC。

### 4.4 Temporal binding

`StageIICachedTrajQueryBindingV2` 只表达 query observation 与 action observation 的
因果绑定：

```text
query_source_timestamp_sec
action_observation_timestamp_sec
query_age_sec
query_source_group_id
action_source_group_id
query_modality_profile_id
action_modality_profile_id
condition_digest
```

核心约束为：

\[
t_{query} \leq t_{action}
\]

\[
age=t_{action}-t_{query}\geq 0
\]

未来 query、时间戳不一致、source group 不一致、模态 profile 不一致或 digest 不一致
都必须拒绝，不能用最近一帧静默修复。

`condition_digest` 只应覆盖会改变 query 语义的稳定条件，例如 instruction、waypoint
或 route revision；传感器帧时间本身由 observation timestamp 表达，不能把每次视觉帧
变化错误地当成导航条件 revision。

### 4.5 Paired training view

`StageIICachedTrajQueryTrainingViewV2` 是训练入口，结构为：

```text
query_source:       target-free Stage I observation
action_observation: target-free current action observation
temporal_binding:   causal timing + modality + digest
trajectory_target:  private 4s/32 metric target
provenance:         source/target lineage
loss_mask:          CFM-only
view_digest:        immutable row digest
```

必须满足：

- query 和 action 来自同一 dataset/split group；
- action source group 与 target source group 一致；
- query timestamp 不晚于 action timestamp；
- query/action modality profile 与 binding 一致；
- sample id 使用 `source_group_id:cached_traj_query_v2`；
- loss mask 只能是 CFM-only mask；
- paired view 不包含 Pass-R、Pass-D 或自然语言 future trajectory。

## 5. 数据构造流水线

生产数据按以下顺序构造，任何一步 reject 都要写入结构化报告，不删除原始 source：

```text
Stage I formal visible views
    + 4s/32 raw-pose trajectory sidecars
    -> StageIICachedTrajQuerySourceViewV2
    -> query/action temporal pair
    -> StageIICachedTrajQueryTrainingViewV2
    -> ms-swift V2 manifest adapter
```

### 5.1 构造 source view

入口脚本：

```bash
PYTHONPATH=src python tools/model_processing/build_stage_ii_cached_traj_query_source_views.py \
  --stage-i-views-jsonl /path/to/stage_i_views.jsonl \
  --trajectory-sidecar-jsonl /path/to/sidecars.jsonl \
  --dataset-manifest sidewalk=/path/to/sidewalk.jsonl \
  --dataset-manifest foxglove=/path/to/foxglove.jsonl \
  --dataset-manifest egowalk=/path/to/egowalk.jsonl \
  --trajectory-source-registry-json /path/to/trajectory_registry.json \
  --output-source-views-jsonl /path/to/source_views.jsonl \
  --output-reject-jsonl /path/to/source_view_rejects.jsonl \
  --output-report-json /path/to/source_view_report.json
```

该步骤只绑定模型可见输入和已验收 trajectory target，不生成 query tensor，也不读取
Pass-R/Pass-D。

### 5.2 构造 temporal pairs

入口脚本：

```bash
PYTHONPATH=src python tools/model_processing/build_stage_ii_cached_traj_query_pairs.py \
  --condition-mode cached_traj_query_v2 \
  --source-views-jsonl /path/to/source_views.jsonl \
  --dataset-manifest sidewalk=/path/to/sidewalk.jsonl \
  --dataset-manifest foxglove=/path/to/foxglove.jsonl \
  --dataset-manifest egowalk=/path/to/egowalk.jsonl \
  --trajectory-source-registry-json /path/to/trajectory_registry.json \
  --output-pairs-jsonl /path/to/cached_query_pairs.jsonl \
  --output-reject-jsonl /path/to/cached_query_pair_rejects.jsonl \
  --output-report-json /path/to/cached_query_pair_report.json \
  --query-period-sec 0.5 \
  --max-query-age-sec 1.0
```

query period 和最大 age 是数据合同参数，不代表真实 runtime 已经达到对应频率。真实
runtime 必须另行记录 observed Stage I/Stage II rate 和 wall-clock latency。

## 6. 训练与推理语义

### 6.1 训练语义

V2 的目标是训练：

```text
query observation -> query hidden states
action observation -> current visual memory
query hidden states + current visual memory -> FlowActionHead
```

未来轨迹只作为 private CFM target 进入 loss，不能进入 query observation 或 action
observation。训练必须按 parent/split 隔离，不能用同一物理 parent 的 query 和 target
跨 split 配对。

当前合同不等价于显式 direction-conditioned action head。若训练后 action head 不读取
query condition，或 query condition 被置零，必须由 shuffle/ablation 暴露，而不能从 ledger
中“存在 query”推断模型使用了 query。

### 6.2 Runtime 语义

推荐的实时调度是：

```text
Stage I query worker：低频、latest-only、允许 stale
Stage II action worker：高频、读取最近有效 query、不得等待 Stage I
```

每个 action tick 至少记录：

```text
request timestamp
query source timestamp
query_age_sec
stale
condition_digest
Stage I latency
Stage II latency
queue drop
deadline miss
```

首次没有有效 query 时必须显式失败或进入定义好的 fallback；不能把随机/空 query 当作
有效条件。过期 query 是否允许使用由 bundle/runtime policy 明确声明，不能在 runner 内
静默改变。

`predict_cached_traj_query_many()` 是离线或多路微批入口。单路实时机器人不能为了凑 batch
无界等待；是否接入实时 batch 必须以真实 P50/P95 和 deadline replay 结果为依据。

## 7. Pass-R / Pass-D 的边界

Pass-R 和 Pass-D 属于旧的显式文本 reasoning route：

```text
Pass-R -> grounding/reasoning text cache
Pass-D -> standalone Decision text cache
Pass-T -> CFM trajectory
```

`cached_traj_query_v2` 不依赖这两个 cache。V2 query observation 直接从模型可见输入构造
query hidden state；不能把旧 R/D 产物重新包装成 V2 query 条件，也不能以旧 R/D 结果替代
V2 paired training view。

旧协议仍可用于 legacy experiment，但必须明确记录：

```text
condition_mode
training contract
prompt/parser identity
checkpoint identity
cache producer identity
```

禁止 V1、V2、legacy cache 在同一训练或推理 run 中静默混合。

## 8. 实现映射

### RouteServer

```text
src/mcp_server/contracts/stage_ii_cfm.py
    StageIICachedTrajQueryObservationV1
    StageIICachedTrajQueryBindingV2
    StageIICachedTrajQuerySourceViewV2
    StageIICachedTrajQueryTrainingViewV2
    StageIIProjectionDropV2

src/mcp_server/data/stage_ii/cached_pairs.py
    source-view materializer
    V1/V2 temporal pair builder

tools/model_processing/build_stage_ii_cached_traj_query_source_views.py
tools/model_processing/build_stage_ii_cached_traj_query_pairs.py
    轻量 CLI orchestration

schemas/stage_ii_cfm/
    cross-repository JSON schema artifacts
```

### ms-swift

```text
swift/sidewalk_nav/data/
    V2 manifest adapter 和 training view 映射

swift/sidewalk_nav/runtime/
    query cache、current visual memory、dual-rate scheduler

swift/sidewalk_nav/models/
    query/visual condition fusion、FlowActionHead

swift/sidewalk_nav/runtime/policy_bundle.py
    condition_mode、normalization、processor 和 checkpoint contract
```

RouteServer 不复制 ms-swift 的 tensor 实现；ms-swift 不根据 JSON 字段 shape 猜测
RouteServer 的 frame、unit、time 或 target 语义。

## 9. 验收等级与当前缺口

### 9.1 合同级验收

- [x] V2 schema、时间因果和 digest 校验存在；
- [x] source view 与 paired view 的 target/identity 隔离规则存在；
- [x] reject/drop 使用结构化 reason code；
- [x] V1 默认路径与 V2 opt-in 路径分开；
- [x] 跨仓 schema 导出和 parity 测试入口存在。

### 9.2 真实数据与模型验收

- [ ] 用正式 train/validation parent 构造完整 source view 和 pair manifest；
- [ ] 报告 source、accepted、rejected、split leakage 和 target QC 分布；
- [ ] 用真实 tokenizer/base model 完成 optimizer smoke；
- [ ] 完成 `cached_traj_query_v2` 正式训练并保存 checkpoint sidecar；
- [ ] native bundle 能校验并恢复 condition mode、query profile、normalization 和 processor；
- [ ] 运行 query/current visual/BEV/instruction shuffle，证明模型实际使用条件；
- [ ] 运行长序列 replay，报告 observed Stage I/Stage II Hz、P50/P95/P99、stale、queue drop
      和 deadline miss；
- [ ] 完成 native/legacy loader 和 fixed-seed parity；
- [ ] 完成 ONNX/TensorRT/HIL/实物部署前置评估。

### 9.3 不能由当前证据推出的结论

以下结论目前不能宣称：

```text
已有旧 checkpoint 已经学会 cached query 条件；
Stage I 文本 direction 已经控制 Stage II action；
配置的 2 Hz / 5 Hz 已经达到真实 wall-clock 频率；
V2 bundle 已经可以直接用于机器人实时部署；
Pass-R / Pass-D 产物可以作为 V2 训练完成依据。
```

## 10. 与 ms-swift 文档的对应关系

实现和训练时同时参考：

- `ms-swift/docs/Task/sidewalk_cached_traj_query_dual_rate_plan.md`：整体方案和阶段清单；
- `ms-swift/docs/Task/sidewalk_dual_rate_video_runtime.md`：实时双频 runtime 合同；
- `ms-swift/docs/Task/sidewalk_native_policy_checkpoint.md`：native bundle 合同；
- `ms-swift/docs/Task/urbannav_unified_policy_inference.md`：统一推理入口和 profile；
- `ms-swift/docs/Experiments/stage_ii_cached_traj_query_v2_4b_full_20260821.md`：训练实验记录。

本文件是 RouteServer 数据和合同边界的权威说明；ms-swift 文档是模型 tensor、训练、
bundle 和 runtime 的权威说明。两边发生版本变化时，必须同时更新 schema digest、实验
记录和 cross-repository parity 测试。

## 11. 输入责任与后续最小执行顺序

本轮配置冻结如下，后续 TODO 不再把 `instruction` 或 `waypoint/goal` 追加为
Stage II action-head 的直接输入：

1. `instruction` 是 Stage I 当前 profile 的导航输入；对当前声明的 instruction profile，缺失时
   直接拒绝。它不是 Stage II 的必需 action 字段。
2. `waypoint/goal` 与 `instruction` 采用同一责任边界：它首先属于 Stage I 的可选条件输入；只有
   独立来源、声明式 profile、goal encoder、训练产物和 Bundle capability 同时存在时，才允许进入
   Stage I。未声明时不能通过 YAML 或请求字段强行打开。
3. Stage II 的 `instruction`、`waypoint/goal` 和 route revision 都是可选 request metadata。缺省时
   复用已绑定的 cached query；携带时只做一致性校验和 ledger 记录，不能据此声称 action head 已
   直接消费这些字段。
4. 条件发生变化时，先刷新 Stage I condition，再让新的 cached query 进入 Stage II；旧 query 在
   刷新前不能被静默当作新条件。

| 项目 | Stage I | Stage II cached-query request | 当前状态 |
| --- | --- | --- | --- |
| `instruction` | 当前 profile 的必需导航输入，缺失拒绝 | 可选 metadata；缺省时复用已绑定 query | 已实现并有真实 smoke |
| `waypoint/goal` | V2 canonical projector、声明式 profile、独立 provenance 和 goal-encoder row adapter 已实现；训练产物和 Bundle capability 仍是前置条件 | 可选 metadata；缺省不破坏 cached-query action | 结构合同已完成，真实数据/训练/Bundle 未验收 |
| `BEV` | 按 profile 作为视觉输入校验 | 按 profile 作为当前视觉输入校验 | 当前 instruction+BEV profile 已有输入链路证据 |

这里的“Stage II 可选”只表示 request 可以省略或携带字段，不表示当前 action head 直接读取
`instruction` 或 `waypoint/goal`。默认 `cached_traj_query_v2` 只读取 Stage I 已编码的
cached query；字段发生变化时，必须先刷新 Stage I condition，旧 query 在刷新前不得静默使用。

后续按以下顺序执行：

1. **保持 Stage I instruction gate**：继续校验 instruction 进入 query observation、condition digest
   和 ledger；当前 profile 缺失 instruction 必须拒绝。该项不需要新 prompt 或新 action head。
2. **验收 Stage I waypoint/goal 能力**：Urban-Nav-SR V2 projector 已保留独立于 future trajectory
   target 的 waypoint context，ms-swift 已将其传入 prompt 和 goal-encoder row adapter；仍需用真实
   independent GPS/planner source 生成 view，训练对应 profile，并在 Bundle 中声明 capability。没有
   这些证据前，不得用配置文件打开 waypoint，也不得用 future odometry 伪造 waypoint input。
3. **验收 Stage II optional waypoint/goal**：在 waypoint capability 通过后，运行 presence、absence、
   binding consistency 和 condition shuffle。absence 只证明默认 cached-query 路径仍可用；presence
   只有在 Stage I 已消费并写入绑定 condition 后才可解释为有效条件输入。
4. **完成 V2 训练后验收**：保存正式 checkpoint、native bundle、normalization/processor contract，
   再执行 query/current visual/BEV/instruction/waypoint 对照、长序列双频 replay 和 native/legacy parity。
5. **进入部署前置**：只有输入条件和双频 replay gates 通过后，才评估 ONNX/TensorRT、HIL 和实物部署。

明确不在本轮 TODO 中：新增 semantic query、Pass-R/Pass-D 前置、Stage II direct-condition action
head，或通过改 YAML 把未训练的 waypoint profile 伪装成已支持能力。

跨仓实现状态：ms-swift 在线 resolver 已支持本文件定义的 `fpv_*_waypoint_v2` canonical profile，
并保留旧兼容 profile。该项是输入命名合同修复，不等于 waypoint checkpoint capability 已训练或
通过真实 GPU smoke。ms-swift 单请求和 micro-batch action path 也已统一校验 cached condition
binding；条件 revision 变化时必须先刷新 Stage I query。

### 11.1 本轮输入配置对应的待办

`instruction` 与 `waypoint/goal` 使用相同的责任边界：它们是 Stage I 的导航输入，
不是当前 Stage II action head 的直接输入。Stage II request 可以省略或携带这两类字段；
省略时沿用已绑定 cached query，携带时只做 binding consistency 校验和 ledger 记录。

后续只按以下顺序推进：

1. 保持 Stage I instruction gate，并复测真实 checkpoint 的输入证据和 2 Hz；
2. 训练并导出声明 waypoint/goal capability 的 Stage I Bundle，再运行 waypoint 正向 smoke；
3. 在 capability 通过后运行 Stage II optional waypoint/goal 的 presence、absence 和 shuffle；
4. 最后执行双频 replay、native/legacy parity、ONNX/TensorRT 和 HIL。

本轮不新增 Stage II direct-condition action head，也不通过配置字段伪造 waypoint 能力。
