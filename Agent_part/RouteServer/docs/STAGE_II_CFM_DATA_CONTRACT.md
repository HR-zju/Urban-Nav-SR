# UrbanNav Stage II CFM canonical data contract

## 1. 边界与结论

Stage II 不重新标注 FullCoT、VQA、Decision 或 direction。它复用 Stage I formal
FullCoT 投影视图的可见输入、action audit 和 strong direction sidecar，并从数据集中已有、
时间可验证的 trajectory/odometry 确定性构造连续轨迹真值。

RouteServer 只负责：

- metric、current-ego FLU 的 canonical trajectory truth；
- 2.0 秒、16 步 temporal resampling 与 QC；
- Pass R/Pass D cache 的身份和严格解析合同；
- Stage II exchange view 与 fail-closed drop audit。

它不保存 token、pixel values、query index、normalized target，也不执行模型训练。

## 2. 冻结轨迹合同

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
PYTHONPATH=src python tools/model_processing/sync_stage_ii_cfm_cross_repo.py \
  --ms-swift-root /path/to/ms-swift --check
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
