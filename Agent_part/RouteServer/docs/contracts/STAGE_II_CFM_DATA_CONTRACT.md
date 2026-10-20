# UrbanNav Stage II CFM canonical data contract

本合同的 `cached_traj_query_v2` 集中设计见
[STAGE_II_CACHED_TRAJ_QUERY_V2_DESIGN.md](STAGE_II_CACHED_TRAJ_QUERY_V2_DESIGN.md)。
本文继续作为 Stage II canonical data contract；V2 双频数据构造、pair 绑定和验收状态
以该设计文档为索引入口。

## 1. 边界与结论

Stage II 不重新标注 FullCoT、VQA、Decision 或 direction。它复用 Stage I formal
FullCoT 投影视图的可见输入、action audit 和 strong direction sidecar，并从数据集中已有、
时间可验证的 trajectory/odometry 确定性构造连续轨迹真值。

RouteServer 只负责：

- metric、current-ego FLU 的 canonical trajectory truth；
- V1 的 2.0 秒/16 步，以及 V2 的 4.0 秒/32 步 temporal resampling 与 QC；
- Pass R/Pass D cache 的身份和严格解析合同；
- Stage II exchange view 与 fail-closed drop audit。

它不保存 token、pixel values、query index、normalized target，也不执行模型训练。
当前可执行训练/head 仍绑定 V1；V2 先作为独立数据资产落盘，不能据此宣称模型已经支持
32 点输出。

## 2. V1 冻结轨迹合同

| 字段 | 定义 | 来源 | student 可见 | 进入 loss |
| --- | --- | --- | --- | --- |
| `trajectory_ego_t` | `[16,3]` 累计相对位姿：forward 米、left 米、CCW yaw 弧度 | 已有 dataset trajectory/odom 的确定性重采样 | 否 | 是，ms-swift 内归一化后进入 CFM |
| `action_mask` | 至少一个有效 step 的 prefix mask；无效后缀对应全零 row | 确定性时域覆盖 | 否 | 是，只屏蔽 padding |
| `valid_step_count` | mask 中的有效 step 数 | 确定性计算 | 否 | 仅审计 |
| `policy` | frame、axis、unit、2.0s/16-step、source format、yaw source 和 resampler stamp | 版本化合同 | 否 | 校验 |
| `source_target_digest` | 原始 trajectory、mask 与 source 声明的摘要 | 确定性计算 | 否 | 审计 |

支持的首版 source format 只有：

```text
ego_xy_t_speed_right_front_v1  [x_right, y_front, time_sec, speed]
ego_pose_xyt_flu_timed_v1      [x_forward, y_left, yaw_ccw, time_sec]
```

旧 `x_right/y_front/dt/speed` 必须先校验时间并插值，再转换 FLU；不能从已经丢失 `dt`
的 V4.6 sidecar 反推。目标时间为 `0.125, ..., 2.0` 秒。数据只覆盖前缀时，剩余
step 用 `[0,0,0]` padding，并把 mask 置为 false；禁止重复最后一点伪造有效长度。

source allowlist 由 RouteServer 的版本化 registry 冻结，数据构建只能引用整份 registry，不能
从命令行临时拼接 allowlist。缺 mask 且不满足 dataset-specific deterministic policy、未知
frame/unit、非单调时间、GPS/GNSS、route matching、route theta、BEV route、LVLM
trajectory 或不在 registry 的 source 全部拒绝。

首版 registry 是：

```text
config/stage_ii_cfm/trajectory_source_registry_v1.json
registry_id      = urbannav_stage_ii_trajectory_sources_v1
registry_version = 1.1
registry_digest  = 8fb9891428f621daafdb273f7a433dd06f915ec93ed8e3cb37b8ef116e4aa402
split_policy     = stage_i_group_sha256_colon_v1, seed=20260731, 90/5/5
```

| dataset | 状态 | source | mask policy | current origin | yaw |
| --- | --- | --- | --- | --- | --- |
| SideWalk | enabled | `calibrated_pose_projection` / `stationary_pose_hold` | 只对已验证 timed rows 确定性构造 | 必须已有 `t=0` row | path tangent |
| Foxglove | enabled | `calibrated_pose_projection` / `stationary_pose_hold` | 只对已验证 timed rows 确定性构造 | 必须已有 `t=0` row | path tangent |
| EgoWalk | enabled | `egowalk_odometry` | 必须读取显式 `action_mask` | 确定性 prepend `[0,0,0,0]` | path tangent |
| FrodoBots | disabled | 无 | 无 | 无 | 无；原因是 approximate camera trajectory 未获准作为 metric GT |

V1.1 还冻结了能与 Stage I `source_group_id` 逐字符连接的 parent identity：SideWalk 使用
`sidewalk_meta.clip_id + sidewalk_meta.frame_name`（保留零填充），Foxglove 使用原始 `id`。
旧 V1.0 使用数值 `frame_idx`，会把 `000010` 变成 `10`，因此不能用于正式全量 join。

这里的 mask 不是全局 “missing -> all true” fallback。adapter 必须同时核对 dataset、source、
format、time unit、origin policy 和 mask policy；任一 stamp 漂移就拒绝。

### 2.1 Pose validity 与 legacy-direct admission

MCAP 预处理不能再把缺失 odometry 的帧仅编码为数值 `[0,0,0]`。当前 exporter/repairer 同时
写出 `frame_pose_valid.npy`；Phase1 只消费 mask 为 true 且 position/orientation finite 的帧。
对早于该 mask 的旧 prepared clip，Phase1 仍在投影和 resampling 之前对实际 future window
应用 `max_pose_step_m=3.0` 连续性门禁，拒绝 `pose_source_step_exceeded`。

兼容旧 FullCoT checkpoint 的 `legacy_full_cot_direct_latent_v1` target 仍保持原 V1 target
schema，但其批量 admission report 已升级为：

```text
policy_id      = urbannav_stage_ii_legacy_direct_source_qc_v2
policy_version = 2.0
max source step = 3.0 m
```

该门禁是 caller-stamped：generic V1 builder 未显式传入阈值时保持历史 digest 与行为；legacy
builder 显式传入阈值，并把阈值计入 source evidence digest。QC drop 表示该 parent 已完成审计但
不生成 target，不应把“存在合法 drop”误判为整批构建失败。

2026-08-03 对历史 legacy train/validation 的只读重建审计请求 55,268 条，接受 55,043 条，
因 `trajectory_source_step_exceeded` 拒绝 225 条；accepted train/validation 分别为
52,688/2,355。该审计没有读取 hidden test。

## 3. Predicted context cache

`StageIIContextCacheV1` 每条只保存一个独立 pass：

- `reasoning_grounding`：严格解析完整 V4.6 FullCoT，但 canonical text 只保留前五段；
- `decision`：严格解析 standalone `<Decision>`。

FullCoT 内的 Decision 不能替代独立 Pass D。cache identity 同时绑定：

```text
sample/source/split/dataset
exact modality_profile_id
parent_source_fingerprint
projected_input_digest
base model/checkpoint/adapter digest
prompt profile/digest
parser contract/digest
generation config/digest
context kind
```

`raw_generation` 只用于 annotation/inference audit；training view 通过 cache stamp 引用，
ms-swift physical prompt 只序列化 `canonical_text`。

同一个 training view 的 Pass R 与 Pass D 必须同时为 `ground_truth`（A0）或同时为
`model_prediction`（A1），并绑定同一 base model、checkpoint、adapter 和 parser contract。
两种来源混用，或两个 pass 来自不同 checkpoint，都会 fail closed。R/D 使用不同 prompt，
因此各自保留独立的 prompt/generation digest。

独立 Pass D 是 target-free inference task，不要求 Stage I 数据已有 Decision 训练标签。对每条
FullCoT parent，ms-swift 从同一份 prompt-visible input 确定性构造一条 Decision request；已有
Decision view 只作为 projected-input parity witness。这个操作不生成新的 GT、不复制 FullCoT
内嵌 Decision，也不改变 Stage I loss。

## 4. StageIICFMTrainingViewV1 字段

| 字段组 | 科研作用 | 真值/标签来源 | student 可推断 | loss |
| --- | --- | --- | --- | --- |
| identity 与 modality selection | 防 split、模态和 parent 混用 | Stage I formal view | 不作为答案 | 否 |
| images/instruction/evidence | 当前实际可见输入 | Stage I safety-bounded projection | 是 | 通过 query hidden 间接进入 CFM |
| `reasoning_context` | predicted 五段 grounding 条件 | A0 为 canonical GT；A1 为冻结 checkpoint prediction | prompt 中可见 | 不做 LM loss |
| `decision_context` | 独立 predicted Decision 条件 | A0 GT / A1 独立 prediction | prompt 中可见 | 不做 LM loss |
| `trajectory_target` | 连续行为真值 | 数据集已有 trajectory/odom 的确定性构造 | 否 | masked CFM |
| `direction_context` | 公共 action profile 和候选 ontology | canonical profile registry | 是 | 否 |
| `private_direction_supervision` | sampler/evaluator/audit | 既有 deterministic direction stamp | 否 | 否 |
| `parent_audit_id`/review | 排除 pending-human 或无审计样本 | Reasoning action reconciliation | 否 | gate |
| `loss_mask` | 保证该 row 只有 System1 action loss | 固定合同 | 否 | `lm=false, system1_action=true` |

exchange view 不存在 `assistant_text`、`decision_target`、normalized trajectory 或 query positions。
`model_visible_context` 中出现 target-like private key 会被递归拒绝。

## 5. 代码与命令

核心实现：

```text
src/mcp_server/contracts/stage_ii_cfm.py
src/mcp_server/data/stage_ii_sources.py
src/mcp_server/data/stage_ii_trajectory.py
src/mcp_server/data/stage_ii_context.py
src/mcp_server/data/stage_ii_sidecars.py
src/mcp_server/data/stage_ii_views.py
tools/model_processing/build_stage_ii_cfm_views.py
tools/model_processing/build_stage_ii_trajectory_sidecars.py
```

生成和同步合同：

```bash
PYTHONPATH=src python tools/model_processing/export_stage_ii_cfm_contract_artifacts.py \
  --output-dir schemas/stage_ii_cfm
PYTHONPATH=.:src python tools/model_processing/sync_stage_ii_cfm_cross_repo.py \
  --output-dir /path/to/UrbanNav-Contracts --check
```

批量构造 view：

```bash
PYTHONPATH=src python tools/model_processing/build_stage_ii_cfm_views.py \
  --stage-i-jsonl /path/to/formal_stage_i_full_cot.jsonl \
  --trajectory-jsonl /path/to/timed_trajectory_sidecars.jsonl \
  --reasoning-cache-jsonl /path/to/pass_r_cache.jsonl \
  --decision-cache-jsonl /path/to/pass_d_cache.jsonl \
  --trajectory-source-registry-json /path/to/trajectory_sources.json \
  --output-view-jsonl /path/to/stage_ii_views.jsonl \
  --output-qc-jsonl /path/to/stage_ii_trajectory_qc.jsonl \
  --output-drop-jsonl /path/to/stage_ii_projection_drops.jsonl \
  --report-json /path/to/stage_ii_build_report.json
```

`--trajectory-source-registry-json` 必须指向完整、digest 自校验的
`StageIITrajectorySourceRegistryV1`。工具不会按数据集名称猜 source，也不会从 V4.6 已经
丢失时间信息的 trajectory sidecar 反推。

全量 sidecar 构造先读取显式 lineage-safe Stage I train/validation manifest，不允许 registry
重新划分 split，也不读取 withheld test：

```bash
PYTHONPATH=src python tools/model_processing/build_stage_ii_trajectory_sidecars.py \
  --stage-i-jsonl /path/to/all_stage_i_views.jsonl \
  --split-manifest train=/path/to/train_views.jsonl \
  --split-manifest validation=/path/to/validation_views.jsonl \
  --dataset-manifest sidewalk=/path/to/sidewalk_main.jsonl \
  --dataset-manifest sidewalk=/path/to/sidewalk_extra.jsonl \
  --dataset-manifest foxglove=/path/to/foxglove.jsonl \
  --registry-json config/stage_ii_cfm/trajectory_source_registry_v1.json \
  --output-dir /path/to/stage_ii_sidecars
```

当前冻结 Stage I 全集有 59,188 个 FullCoT parent。它是 Stage II corpus denominator，不是可
全部用于梯度更新的 train 集：lineage-safe train 为 52,903，validation 为 2,365；withheld
test 与 unknown-lineage parent 保持封存。正式训练必须报告 target/context QC 后的实际 accepted
数，不能用 59,188 替代最终 train row 数。

真实 train-split target audit：

```bash
PYTHONPATH=src python tools/model_processing/audit_stage_ii_trajectory_targets.py \
  --registry-json config/stage_ii_cfm/trajectory_source_registry_v1.json \
  --dataset-manifest sidewalk=/path/to/sidewalk.jsonl \
  --dataset-manifest foxglove=/path/to/foxglove.jsonl \
  --dataset-manifest egowalk=/path/to/egowalk.jsonl \
  --parents-per-dataset 200 \
  --max-parents-per-group 8 \
  --scan-limit-per-dataset 25000 \
  --output-dir /path/to/audit
```

2026-08-01 的本地真实数据审计固定抽取 600 个 train parent（每个 enabled dataset 200）：

- accepted/drop = `600/0`；FrodoBots 按 registry 关闭且未扫描；
- target mask：377 条 16/16，191 条 8/16，其余 32 条分布在 5–15/16；
- target horizon：377 条 2.0s，191 条 1.0s，其余 32 条 0.625–1.875s；
- mask provenance：400 条 `validated_timed_rows_v1`，200 条 `explicit_field_v1`；
- yaw source：600 条全部为 `path_tangent`；
- selection digest：`16b3ff812d1b43f54d274336f149c3c1282bbfc82e5d87ef4ba970b43ddd7dec`。

该结果的 scope 是 `real_train_split_unique_parent_target_only`，只验证 source adapter、
resampling、mask 与 trajectory QC，不包含 Stage I formal admission、Pass R/D cache 或完整
`StageIICFMTrainingViewV1` 构建。

当前验证覆盖 synthetic contract/resampling/QC/cache/leakage/cross-repo parity，以及上述
600-parent target-only audit。真实 Qwen processor、完整 Pass-T view 和 GPU optimizer smoke
仍是独立后续 gate，不能由单元测试或 target audit 替代。

## 6. V2.1：4 秒、32 点 raw-pose 数据合同

V2 的唯一 policy ID 是：

```text
urbannav_stage_ii_metric_flu_4s_32_speed_audit_v2_1
```

它从 raw pose/odometry 重新生成标签，不延长 V1 的 2 秒轨迹，也不重复末点。目标时间网格为：

\[
t_k=k\Delta t,\qquad \Delta t=0.125\text{ s},\qquad k=1,\ldots,32,
\]

因此完整 horizon 为 \(4.0\) 秒。raw window 还必须包含当前原点
\([0,0,0,0]\)。对 map 位移和当前 base yaw 的 FLU 变换为：

\[
x_f=\Delta x\cos\psi_0+\Delta y\sin\psi_0,
\qquad
y_l=-\Delta x\sin\psi_0+\Delta y\cos\psi_0.
\]

XY 按时间线性插值；yaw 先 unwrap 到最短角路径，再插值并 wrap 回
\([-\pi,\pi)\)。V2 只接受完整的 `32 x 3` target，`action_mask` 必须全部为 true；raw
pose 不覆盖 4 秒时直接拒绝，禁止 extrapolation、terminal repeat 和 padding。

### 6.1 Raw adapter 与可信边界

| 数据 | raw truth | quaternion | odom-to-base yaw | 额外 gate |
| --- | --- | --- | ---: | --- |
| SideWalk main/extra | `frame_positions/orientations/times.npy` | metadata；旧资产缺 metadata 时为 `wxyz` | 0 deg | 拒绝 whole-clip static export |
| Foxglove | 同步后的 NPY pose | 必须由 metadata 显式给出 | +90 deg | 拒绝缺 quaternion-order 的 clip |
| EgoWalk | 原始 odometry parquet | `xyzw` | 0 deg | caption/instruction 必须覆盖未来完整 4 秒，并拒绝 backward discontinuity |
| FrodoBots | 无可信 metric trajectory | N/A | N/A | 不生成 V2 target |

单条 raw 文件损坏或局部身份缺失进入版本化 reject，不会终止整个 corpus；重复 clip identity
仍按全局数据冲突 fail closed。接受样本还需同时满足以下连续性硬门禁：

```text
max raw timestamp gap <= 0.5 s
max adjacent source step <= 3.0 m
max source yaw rate <= 4.0 rad/s
```

`max_source_speed_mps` 仍逐样本计算。超过 6.0 m/s 时追加
`trajectory_source_speed_audit_exceeded`，但不再拒绝；速度阈值不是部署速度策略，也不能单独
区分真实快速运动、时间采样抖动和位姿尺度异常。训练与评测必须保留该 audit slice，连续性异常
仍由 gap、step、yaw-rate、non-finite 和数据源专属 backward-discontinuity gate 拒绝。

#### 6.1.1 短 horizon 与参考实现

V2.1 继续要求 raw observation 覆盖完整 4 秒，不把不足 4 秒的序列补成监督。CityWalker 在
构造 LUT 时先扣除 `context_size + max(arrived_threshold*2, wp_length)`，因此靠近序列末尾且
拿不到完整 waypoint 的 anchor 不进入训练；其图像 frame clamp 只处理视觉输入边界，不是
trajectory padding。SocialNav 的公开 action head 固定消费 `[B,5,2]`，flow loss 对全部坐标做
无 mask MSE，公开训练 consumer 没有 partial-horizon action mask。其公开仓没有提供 SocNav
Dataset 上游 terminal-window builder，因此不能据此声称它对短轨迹做了安全 padding。

InternNav/RDP 采用了另一种明确的 episode-terminal 假设：当 future action chunk 越过 episode
末尾时，`fill_mode='constant'` 重复最后 GPS 与 yaw，再把绝对 waypoint 转为 delta，因此补齐段
成为零位移、零转角的 absorbing stop。该策略适用于“episode 已权威结束”的语义；其
`not_done_masks` 用于变长 rollout batch，不是 action chunk 内逐点有效性 mask，不能直接用于
普通 clip 截断或 instruction coverage 缺失。

UrbanNav 只在 raw pose 本身持续记录到 4 秒且真实呈现 stationary hold 时把后续静止点视为
有效监督。因 clip 结束、instruction 结束或文件缺失导致的 horizon 不足仍拒绝；禁止 zero
padding、末点重复和无 mask 的伪停止标签。若后续能提供权威 `episode_terminal/goal_reached`
证据，可在新 policy 中显式增加 `terminal_hold_steps` 与 provenance，再评估 InternNav 式
absorbing-stop；不能在 V2.1 中静默改变语义。

### 6.2 V2 产物与入口

核心实现：

```text
src/mcp_server/data/stage_ii_raw_trajectory_v2.py
src/mcp_server/data/stage_ii_trajectory_v2_corpus.py
tools/model_processing/build_stage_ii_trajectory_4s32.py
tools/model_processing/run_stage_ii_trajectory_4s32_v2_local.zsh
```

每个 run 输出：

```text
trajectory_sidecars_v2_{train,validation,test}.jsonl
trajectory_qc_v2.jsonl
trajectory_rejected_v2.jsonl
trajectory_build_report_v2.json
trajectory_normalization_v2.json
```

本地四个 prepared source manifest 的一键生成命令为：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR
Agent_part/RouteServer/tools/model_processing/run_stage_ii_trajectory_4s32_v2_local.zsh
```

可先运行每个 manifest 100 条的 smoke：

```bash
SMOKE_LIMIT_PER_MANIFEST=100 \
RUN_ROOT=/tmp/urbannav_stage_ii_metric_flu_4s_32_speed_audit_v2_1_smoke \
Agent_part/RouteServer/tools/model_processing/run_stage_ii_trajectory_4s32_v2_local.zsh
```

2026-08-02 的 400-row real-data smoke 接受 345、拒绝 55：SideWalk main 95/100、
SideWalk extra 89/100、Foxglove 94/100、EgoWalk 67/100。55 条拒绝全部来自
`trajectory_full_horizon_missing`（22）或
`trajectory_instruction_horizon_missing`（33），没有速度、step、gap 或 yaw-rate 异常。
该证据只验证数据生成和 QC，不验证后续 32 点模型训练。

### 6.3 2026-08-02 V2.0 历史全量结果

全量 prepared source 输入 383,526 行。按 stable parent 去重后为 381,797 条，其中 accepted
247,059、rejected 134,738；SideWalk main/extra 重叠产生 1,729 条等价重复，已确定性跳过。

| source | input | equivalent duplicate | accepted | rejected |
| --- | ---: | ---: | ---: | ---: |
| EgoWalk | 284,470 | 0 | 171,321 | 113,149 |
| Foxglove | 49,319 | 0 | 32,737 | 16,582 |
| SideWalk main | 45,397 | 0 | 40,922 | 4,475 |
| SideWalk extra | 4,340 | 1,729 | 2,079 | 532 |

accepted split 为 train 220,907、validation 10,502、test 15,650。主要 reject 是 EgoWalk
instruction 不覆盖完整未来 4 秒（105,904）、Foxglove/其他来源速度超过 6 m/s（12,912）、
full horizon 缺失（7,886）和 EgoWalk backward discontinuity（6,534）。这些是数据选择结果，
不是 padding 候选。

```text
run root:
/media/ubuntu/Extreme SSD/urbannav_stage_ii_metric_flu_4s_32_v2

train sidecar sha256:
567c7a0a565778fb640ae84f4d128d1923e74a6763c4a612efd9dda115b144a2

normalization profile digest:
8a723d910633465c04679df00bf4c91d2b8f03c23ed1b470d0c745620f62a6f2
```

独立流式复核验证了全部 247,059 个 accepted sidecar 和 381,797 个 QC row：Pydantic
schema 全部通过、accepted/rejected 交集为 0、8,115 个 split group 的跨 split 泄漏为 0，
normalization 绑定的 train SHA 与生成报告一致。

### 6.4 V2.1 speed-audit 策略

2026-08-03 将 6 m/s 从硬拒绝改为 audit-only，并提升 policy version 到 `2.1`；V2.0 目录、
sidecar SHA 和 normalization digest 保留为历史证据，不原地覆盖。新默认 run root 为：

```text
/media/ubuntu/Extreme SSD/urbannav_stage_ii_metric_flu_4s_32_speed_audit_v2_1
```

全量重建仍输入 383,526 行，按 stable parent 去重后为 381,797 条；其中 accepted
259,964、rejected 121,833，另有 1,729 条 SideWalk main/extra 等价重复。相对 V2.0 实际增加
12,905 条 accepted，原 12,912 个 speed reject 中其余 7 条因 yaw-rate gate 继续拒绝。

| source | input | equivalent duplicate | accepted | rejected | `>6 m/s` audit |
| --- | ---: | ---: | ---: | ---: | ---: |
| EgoWalk | 284,470 | 0 | 171,395 | 113,075 | 74 |
| Foxglove | 49,319 | 0 | 45,018 | 4,301 | 12,281 |
| SideWalk main | 45,397 | 0 | 41,451 | 3,946 | 529 |
| SideWalk extra | 4,340 | 1,729 | 2,100 | 511 | 21 |

accepted split 为 train 232,670、validation 11,036、test 16,258。121,833 条 reject 的主要原因
仍是 instruction horizon 不足（105,904）、full horizon 不足（7,886）和 EgoWalk backward
discontinuity（6,534）；其他连续性门禁拒绝 non-finite 537、step 548、gap 314、yaw-rate 110。

```text
train sidecar sha256:
2dd3d53d752aec86c2d529b9d9ef46c3d66d264afd4ed6fd8e9234e06cc7876a

normalization scales:
[13.747750070927463, 3.3851487938464113, 2.075140968969098]

normalization profile digest:
92048ce5df70566e5e60c7ac4266c2618d9270039824a848f2ed4782b2116daf
```

独立流式复核验证了全部 259,964 个 accepted sidecar 和 381,797 个 QC row：schema 全部通过、
accepted/rejected 交集为 0、8,123 个 split group 的跨 split 泄漏为 0，12,905 个 speed-audit
row 均存在于 accepted sidecar，normalization 绑定的 train SHA、parent 数和报告一致。

## 7. 计划中的 `stage_ii_cfm_v2`：Reasoning Dropout 双系统合同

本节是后续实现计划，不表示 V1 或 V2.1 数据已具备该能力。`stage_ii_cfm_v2` 是模型条件与
exchange view 的版本；`trajectory_target_v2_1` 是 raw-pose 轨迹数据版本，二者不能混为同一个
“V2”。V1 保持只读兼容，不能静默接收 V2 row。

### 7.1 目标与不变量

System 2 的 `[TRAJ_i]` token 是在 prompt 末尾读取 hidden state 的 latent query。它可注意到所有
此前的可见图像、instruction、BEV/goal/waypoint、Decision 与可选 grounding，但不生成语言轨迹：

\[
H_t^Q=F_\theta(O_t,C_t^D,C_t^R,Q),\qquad
Q=[TRAJ_0,\ldots,TRAJ_{Q-1}],
\]

\[
\hat W_t=\operatorname{CFM}_\phi(H_t^Q),\qquad
\hat W_t\in\mathbb{R}^{B\times32\times3}.
\]

`C_t^D` 始终是 accepted standalone Decision。`C_t^R` 只可以是 accepted FullCoT grounding，或由
训练路由显式声明为 absent；它不能是 parser failure、unknown、human reject、空字符串或缺失缓存。
trajectory truth、normalization、strong direction label 和 teacher-only GT 始终不得进入 prompt。

### 7.2 两条合法 context route

| `context_mode` | `reasoning_state` | 需要的 cache | Pass-T 可见内容 | 目的 |
| --- | --- | --- | --- | --- |
| `reasoning_plus_decision` | `present` | accepted R + accepted D；同一 producer identity | canonical grounding + Decision + `[TRAJ]` | 显式 reasoning 条件基线。 |
| `decision_only` | `absent_by_reasoning_dropout` | accepted D；无 R cache | explicit absence tag + Decision + `[TRAJ]` | 默认部署候选；训练 CFM 不依赖 FullCoT。 |

两条 route 的图片、`image_roles`、instruction、evidence catalog、可选 BEV/waypoint、direction
context、trajectory target、split 与 `[TRAJ]` query placement 必须一致。一个 parent 在一个 immutable
CFM epoch appearance 中只能选择其中一条 route；不能把同一轨迹作为 R+D 与 D-only 两次梯度样本。

### 7.3 V2 prompt 与 query 合同

V2 的 user prompt 计划收敛为：

```text
<image>...<image>
<TASK_TRAJECTORY_CFM>
<MODE profile_id="...">
{model-visible observation payload}
<REASONING_CONTEXT state="present|absent_by_reasoning_dropout">...</REASONING_CONTEXT>
<PREDICTED_DECISION>{accepted standalone Decision}</PREDICTED_DECISION>
[TRAJ_0] ... [TRAJ_Q-1]
```

`absent_by_reasoning_dropout` 只表达训练/部署选择，不表达不确定性或感知失败。`Q` 与 4 秒 target
长度 `T=32` 解耦；首个 profile 应把 `Q` 作为显式配置并比较 `Q in {4,8,16}`，而不是沿用 V1 的
“一个 query 对应一个 target step”假设。

### 7.4 实现职责与顺序

**RouteServer canonical owner**：新增 V2 cache union、`reasoning_state`、context-mode invariant、
V2 prompt-visible digest、drop/reject audit 和 cross-repository schema exporter。V2 builder 必须拒绝
R/D cache identity 不匹配，也必须拒绝将不合格 R cache 解释为 dropout absence。

**ms-swift training owner**：新增 V2 manifest adapter、query contract、profile-driven route sampler、
CFM condition-mode metric 和 checkpoint sidecar。CFM loss 在 valid trajectory point 上独立归一化；
它不与 Stage I FullCoT/Decision/VQA 的 LM loss 合并为一个虚假的单 forward 公式。

**先后顺序**：

1. 先用 `traj_query_only` 实现并验证 R+D/D-only paired CFM，冻结 4 秒/32 点轨迹与 query contract；
2. 在同一 held-out case 下报告 ADE/FDE/L2@1s/L2@2s、direction/stop/turn slices、coverage、latency 和
   `Delta_reason`；
3. 再把 System 1 的 `hybrid` memory 变体接入每个控制 tick 的新视觉特征，检验快速视觉纠偏；
4. 最后才进入实物 replay、独立 safety supervisor 与有限速度的 closed-loop 验证。

### 7.5 Cached trajectory-query 双频扩展

双频扩展不新增独立语义查询或语义 projector，而是复用已有 `[TRAJ_i]` hidden states。其
canonical mode 为 `cached_traj_query_v1`，由 `StageIICachedTrajQueryBindingV1` 绑定 query
观测时间、action 观测时间、query age 和 modality profile。训练 pair 另由
`StageIICachedTrajQueryObservationV1` 与 `StageIICachedTrajQueryTrainingViewV1` 表达：query
observation 只含模型可见输入，action view 才含轨迹 target；两者必须来自同一 source/split，且
query 时间不能晚于 action 时间。

`traj_query_only` 仍是旧 checkpoint 和旧 manifest 的默认路径。cached contract 只证明输入、
时间和条件接口闭合，不证明已有权重已经学会使用 fast visual memory；在真实 paired batch、
checkpoint 训练和 shuffle ablation 完成前，不得把它描述成已验收的 direction-conditioned
部署能力。

当前 V1 虽有 `traj_query_only` prompt 和 System1 condition 的基础设施，但 prompt builder 仍固定
要求 accepted R/D context，且 V1 使用 2 秒/16 步 target。它不能作为已完成的 no-CoT 4 秒双系统
实现，也不能直接加载 V2.1 sidecar。
