# UrbanNav 数据质量问题与采集、筛选、标注需求

更新时间：2026-08-04。

状态：数据专家 handoff 草案。本文汇总现有 Stage I 与 Stage II 训练、离线评测和数据审计
暴露的问题，并定义下一版数据的采集、筛选、标注和发布要求。本文不替代具体 schema、训练合同
或当前运行状态文档。

## 0. 证据范围与结论

本文使用三类证据：

1. legacy Stage I direction-12 的 4B/9B LoRA/full 探索性评测；
2. 新 Reasoning V4.6 FullCoT + VQA V2 + Decision 的 Stage I 联合训练数据审计；
3. legacy Stage II CFM 2 秒/16 点训练、Pass-T 评测，以及 Stage II V2.1 4 秒/32 点数据重建。

需要保留以下边界：

- legacy direction-12 的 epoch 选择使用过 audit test，因此数值只能用于错误分析，不能作为正式
  benchmark；
- 新 Stage I 联合训练已证明 tokenizer、typed target 和 teacher-forcing 路径可运行，但最终
  generation/test gate 尚未闭合，低 eval loss 不能直接解释为 grounded navigation 已通过；
- legacy Stage II checkpoint 的 normalization 和训练包含 215 个后来被判定为坏轨迹的 train
  parent，因此该 checkpoint 不能作为通过验收的模型；
- V2.1 4 秒/32 点语料已经完成数据级构造与审计，但 legacy 2 秒/16 点 CFM 结果不能直接证明
  4 秒模型质量。

当前收敛出的优先级为：

```text
P0 物理轨迹 QC
  -> P1 stop/move 与方向 GT 一致性
  -> P2 方向长尾和 normal-control 分布补齐
  -> P3 FullCoT/VQA formal gate
  -> P4 独立 GPS/waypoint guidance
```

核心结论是：数据已经能够驱动模型学习，但尚未形成统一、干净、可发布的数据资产。少量物理异常
会主导 CFM 梯度；方向长尾、stop 定义冲突和 weak-label 分布偏置则限制 Stage I 的语义与动作质量。

## 1. 训练和评测暴露的问题

### 1.1 Stage I direction-12

legacy multiframe direction 池共有 96,986 条，原始分布为：

| 分组 | 数量 | 比例 |
| --- | ---: | ---: |
| `straight` | 47,912 | 49.40% |
| `stop` | 21,223 | 21.88% |
| `left/right_10` | 18,108 | 18.67% |
| `left/right_20` 及以上 | 9,743 | 10.05% |

其中 `right_50plus` 只有 642 条。sqrt-temperature 采样增加了长尾曝光，但重复少量 parent
不能替代新的路口、道路宽度、遮挡、动态交互和相机视角。

9B full 的探索性结果为：

| 指标 | 结果 |
| --- | ---: |
| strict 12-way accuracy | 63.27% |
| non-stop 10 度容忍率 | 87.20% |
| interval MAE | 5.30 度 |
| stop accuracy | 76.21% |

1,563 条 strict failure 中：

| 失败类型 | 数量 | 失败占比 |
| --- | ---: | ---: |
| straight/turn confusion | 889 | 56.88% |
| opposite turn | 160 | 10.24% |
| 10 度边界兼容 | 153 | 9.79% |
| missed stop | 152 | 9.72% |
| false stop | 130 | 8.32% |
| 10--20 度 warning | 54 | 3.45% |
| 同侧大幅度误差 | 25 | 1.60% |

这说明主要问题不是输出格式，而是：

- `straight` 与 `+/-10` 度的标签边界缺少足够校准样本；
- 中大角度预测明显向 0 度回缩；
- 右 30、40、50 度以上弱于对应左转；
- 282 条 stop/move 错误和 160 条 opposite-turn 不能通过放宽角度容忍阈值解决。

早期“模型全部预测 stop”主要来自评测 prefix/parser 不兼容。修复后 parse success 恢复到
99.98%--100%。这部分是评测链路问题，不应被当成数据删除依据。

### 1.2 新 Stage I 联合训练

当前联合 epoch 由三类独立物理 row 组成：

```text
FullCoT  = 17,455
VQA      = 21,818
Decision = 48,000
total    = 87,273
```

真实 tokenizer census 得到：

| task | active target tokens | token mass |
| --- | ---: | ---: |
| FullCoT | 7,034,175 | 81.736% |
| VQA | 508,188 | 5.905% |
| Decision | 1,063,605 | 12.359% |

因此 20% FullCoT row 实际贡献了超过 80% 的 token loss。该现象首先属于训练采样和 loss
accounting 问题，但数据交付必须保留 task、target length、source 和 provenance，使训练端可以
按 active-token mass 校准，不能只按 row 数声称任务均衡。

### 1.3 Stage II CFM 数值污染

legacy Stage II 请求 55,268 个 parent。事后启用 source-step QC 后：

```text
accepted          = 55,043
rejected          = 225
train rejected    = 215
validation reject = 10
Foxglove reject   = 224
SideWalk reject   = 1
```

坏样本比例只有约 0.41%，但三个异常 batch 占累计训练 loss 的 84.31%。每个 epoch 各出现一次：

| step | loss | grad norm | Foxglove loss |
| ---: | ---: | ---: | ---: |
| 739 | 832.2189 | 153 | 1,105.6908 |
| 1,556 | 860.1104 | 161 | 1,108.4819 |
| 2,203 | 828.8734 | 156 | 1,105.8314 |

最大异常样本在正常约 `(1610, -2511)` 的序列中包含零填充 pose，导致 0.05--0.06 秒内发生
约 2,983 米跳变，继而产生 1,888.9 米的两秒 target。normalization 后单坐标最大值达到
522.4；仅 25 个绝对值大于 100 的坐标就占 target 平方总量的 90.70%。

FPV overlay 因裁剪、投影和平滑仍可能看起来合理，所以视觉 review、behavior VLM 和
motion saliency 都不能代替 metric-pose QC。motion saliency 甚至可能把 pose jump 当作高价值
运动样本。

### 1.4 Stage II stop 失败

历史 Stage II train 的 52,903 条中只有 974 条显式 stop，占 1.84%。其中：

- 97 条在 0.06 米严格静止阈值下并不静止；
- 76 条在 0.25 米 metric direction 阈值下仍不静止；
- validation 只有 19 条显式 stop；
- 19 条 prediction 全部移动超过 0.06 米，平均 stop drift 为 2.7896 米；
- Euler step 从 8 增加到 32 仍无法修复。

当前单个 `stop` 标签混合了至少五种不同事实：用户要求停止、规则要求停车、轨迹物理静止、
减速后停止和到达目标。该语义冲突必须在数据层拆开，不能通过增加 loss 或调整积分步数掩盖。

## 2. 当前数据源问题

### 2.1 FullCoT 和 VQA 标注状态

截至当前状态文档快照：

| 数据集 | 行级 annotation | formal eligible | 主要问题 |
| --- | ---: | ---: | --- |
| Foxglove | 47,249 | 34,162，72.30% | 13,087 条 action reconciliation |
| SideWalk main | 45,397 | 25,026，55.13% | 10,359 条 NavigationContext repair |
| SideWalk extra unique | 0/2,611 | 0 | 尚未完成 FullCoT |
| EgoWalk | 标注中 | 未冻结 | 需完成、去 retry 重复、QC 和 formal export |
| FrodoBots | 未纳入强动作 | 0 | continuous trajectory/System1 不可信 |

SideWalk 的 10,359 条 NavigationContext failure 由以下问题组成：

| 原因 | 数量 | 根因 |
| --- | ---: | --- |
| 通用任务模板被误判为显式方向 | 8,367 | active instruction 与 task template 未正确区分 |
| BEV route 无显式 authority | 1,991 | producer alignment/frame 元数据未迁移 |
| 图片 decode error | 1 | 归档 BEV 为零字节，raw producer 文件可恢复 |

另外有 1,041 条虽然六段齐全，但存在 `strong_action_constraint_conflict`。这类冲突不能靠放宽
角度阈值修复。

### 2.2 VQA 分布偏置

VQA 当前有 106,184 个 source，展开为 1,519,896 个 view：

| 数据集 | unique source | VQA views |
| --- | ---: | ---: |
| Foxglove | 20,680 | 243,223 |
| SideWalk | 19,248 | 237,977 |
| EgoWalk | 66,256 | 1,038,696 |

EgoWalk 占 unique source 的 62.40%。当前 source 全部来自 behavior-filter 的 interesting pool。
它适合障碍、交互、规则和复杂通行性问题，但缺少 matched normal controls，会让模型错误学习
“场景总是复杂、阻塞或需要避让”的先验。

VQA 中只有 E0 是 trajectory-derived GT；其余已生产任务主要是 LVLM weak label。因此：

- weak-label evaluator 只能报告 teacher agreement，不能称为世界真值 accuracy；
- artifact/schema audit passed 不等于语义人工验收完成；
- VQA 数据必须按 task、dataset、modality、label source 和 normal/interesting 分层。

### 2.3 4 秒轨迹覆盖

Stage II V2.1 从 383,526 个输入 row 中得到 381,797 个去重 parent：

```text
accepted            = 259,964
rejected            = 121,833
equivalent duplicate = 1,729
train/val/test       = 232,670 / 11,036 / 16,258
```

| source | input | accepted | rejected | >6 m/s audit |
| --- | ---: | ---: | ---: | ---: |
| EgoWalk | 284,470 | 171,395 | 113,075 | 74 |
| Foxglove | 49,319 | 45,018 | 4,301 | 12,281 |
| SideWalk main | 45,397 | 41,451 | 3,946 | 529 |
| SideWalk extra | 4,340 | 2,100 | 511，另有 1,729 重复 | 21 |

主要 reject 为：

| 原因 | 数量 |
| --- | ---: |
| instruction horizon 不足 | 105,904 |
| raw full horizon 不足 | 7,886 |
| EgoWalk backward discontinuity | 6,534 |
| non-finite | 537 |
| source step | 548 |
| timestamp gap | 314 |
| yaw-rate | 110 |

超过 6 m/s 只作为 audit，不再单独拒绝。速度本身不能区分真实快速运动、timestamp 抖动、尺度错误
和 pose jump。

### 2.4 模态和可信边界

| 数据集 | FPV | BEV | metric trajectory | direction/System1 使用边界 |
| --- | --- | --- | --- | --- |
| SideWalk | 可信 | 可用 | pose/camera QC 后可信 | 可用于 strong direction 和 CFM |
| Foxglove | 可信 | 可用 | pose sync/validity QC 后可信 | 可用于 strong direction 和 CFM |
| EgoWalk | 可信 | 无 | odometry QC 后可信 | ego direction/CFM；无 FPV pixel GT |
| FrodoBots | 可信 | 受控使用 | 不可信 | 只用于 FPV-BEV/VQA 和获准弱方向 |

当前所有 staged manifest 的 `modality:waypoint` 都为 0。future odometry 是动作 target，不是
独立 planner guidance，禁止将同一未来轨迹复制为 waypoint 输入。

## 3. 统一采集合同

### 3.1 每条样本必须保存的字段

```text
stable_sample_id
clip_id / sequence_id / split_group_id
dataset_source / platform_id / sensor_rig_id
所有图像的绝对 timestamp、原始宽高、camera_id
intrinsics / distortion / extrinsics / calibration_digest
pose timestamp / position / quaternion / quaternion_order
world_frame / base_frame / camera_frame / metric_unit
frame_pose_valid / invalid_reason
instruction_raw / caption_raw / caption_role
instruction_start_time / instruction_end_time
episode_terminal / goal_reached / terminal_provenance
可选 BEV 的 frame / alignment / ego_forward / route_authority
可选 waypoint 的 CRS / planner_id / plan_timestamp / route_version
```

禁止行为：

- 禁止用 `[0,0,0]` 代替缺失 pose；
- 禁止缺省 quaternion order；
- 禁止在没有 producer metadata 时猜测 BEV 是 ego-aligned；
- 禁止把 task template 当成用户实际 instruction；
- 禁止把 future trajectory 同时作为模型可见 waypoint 和预测 target。

### 3.2 时间合同

历史视觉条件保持：

```text
5 个 FPV
5 Hz
offset = [-0.8, -0.6, -0.4, -0.2, 0.0] s
```

Stage II 目标使用当前时刻之后的 4 秒 metric FLU：

\[
t_k=k\Delta t,\qquad \Delta t=0.125\text{ s},\qquad k=1,\ldots,32.
\]

设当前 map pose 为 \((x_0,y_0,\psi_0)\)，未来位移为
\((\Delta x,\Delta y)\)，则：

\[
x_f=\Delta x\cos\psi_0+\Delta y\sin\psi_0,
\qquad
y_l=-\Delta x\sin\psi_0+\Delta y\cos\psi_0.
\]

XY 按时间线性插值；yaw 必须先 unwrap，再按最短角路径插值并 wrap 回
\([ -\pi,\pi )\)。accepted target 必须严格为 `32 x 3`，且 32 个 action mask 全部有效。

### 3.3 短 horizon

以下情况禁止 padding、extrapolation 或末点重复：

- clip 文件结束；
- instruction coverage 结束；
- pose 文件缺失；
- timestamp gap 导致未来不可观测。

只有同时存在权威 `episode_terminal=true` 或 `goal_reached=true`，并且 raw pose 真实记录了
stationary hold，才允许把终止段作为 stop target。若未来采用 terminal repeat，必须新建 policy，
显式保存 `terminal_hold_steps`、`action_mask` 和 provenance，不能静默改变 V2.1 语义。

## 4. 统一筛选和轨迹 QC

### 4.1 固定处理顺序

```text
媒体存在性和解码
  -> 时间戳单调性
  -> pose validity
  -> 坐标系、标定和单位
  -> metric continuity
  -> trajectory/action 派生
  -> stop/direction reconciliation
  -> behavior interestingness
  -> motion saliency
  -> FullCoT/VQA teacher
```

物理 QC 必须在 behavior 和 motion saliency 之前执行。`interesting` 不是 trajectory-valid 标签，
API failure/unknown 也不能自动变成 boring。

### 4.2 物理量

对相邻 pose：

\[
\Delta t_k=t_k-t_{k-1},
\qquad
s_k=\lVert p_k-p_{k-1}\rVert_2,
\]

\[
v_k=\frac{s_k}{\Delta t_k},
\qquad
\omega_k=\frac{|\operatorname{wrap}(\psi_k-\psi_{k-1})|}{\Delta t_k},
\]

\[
a_k=\frac{v_k-v_{k-1}}{\Delta t_k}.
\]

accepted row 至少满足：

```text
所有 pose、timestamp 和 target finite
timestamp 严格递增
max raw timestamp gap <= 0.5 s
max adjacent source step <= 3.0 m
max source yaw rate <= 4.0 rad/s
完整覆盖未来 4 秒
所有使用帧的 frame_pose_valid = true
```

同时必须保存 `max_speed/max_acceleration/max_yaw_rate/max_step/max_gap`，并按平台计算 p50、p90、
p99、p99.9。`speed > 6 m/s` 保留为 audit slice；是否拒绝应结合时间差、平台物理上限、step、
acceleration 和 yaw-rate，而不是只看一个速度阈值。

### 4.3 stop 重新定义

数据层至少保存：

```text
intent_stop: bool
physical_stationary: bool
decelerate_to_stop: bool
stop_reason: goal_reached / traffic_control / pedestrian / obstacle /
             instruction / no_valid_corridor / other
hold_duration_sec: float
terminal_displacement_2s_m: float
terminal_displacement_4s_m: float
max_speed_during_hold_mps: float
label_provenance
```

`intent_stop` 不得由终端 displacement 单独推导；`physical_stationary` 也不得由 instruction 单独
推导。两个标签都满足时才能形成高置信纯 stop CFM 样本。减速后停止必须保留完整 deceleration
轨迹，不能被改写成全零 target。

## 5. 分布和采样需求

### 5.1 方向数据

下一轮采集建议以 unique parent 为单位设置最低验收量：

| 类别 | 最低可信 unique parent |
| --- | ---: |
| `left/right_20` | 每类至少 2,000 |
| `left/right_30` | 每类至少 2,000 |
| `left/right_40` | 每类至少 2,000 |
| `left/right_50plus` | 每类至少 2,000 |

附加要求：

- 同角度左右数量比保持在 `[0.8, 1.25]`；
- 为 `straight`、`left/right_10` 单独构造连续角度在 5--15 度附近的边界样本；
- 同时保存连续 `theta_gt`、native profile label 和 common-12 label；
- 每个 parent 的重复曝光由训练 sampler 管理，数据交付不得复制 JSONL 伪造 unique support；
- 每类至少覆盖两个可信数据源、多种路口宽度、遮挡、昼夜、天气和动态交互条件。

### 5.2 stop 数据

- CFM train 的可信 `physical_stationary` unique parent 不低于 5%；
- `goal_reached`、`traffic_control`、`pedestrian`、`obstacle`、`instruction` 等主要原因每类至少
  500 个 unique parent；
- 必须包含 moving -> decelerate -> hold -> resume 的完整序列；
- stop validation/test 保持自然分布，但每个主要 reason 必须有可报告 support；
- stop/move 冲突、错误终端状态和无 provenance stop 一律进入 reject/review，不能自动重标。

### 5.3 interesting 与 normal control

对每个 interesting VQA source 构造 matched normal-control 候选，匹配：

```text
dataset_source
direction bucket
modality profile
时间/天气/照明条件
instruction/caption availability
```

评测集优先采用 1:1 matched pair。训练阶段 normal-control 实际质量由独立实验决定，但原始候选池
必须足以支持 30%--50% 的 normal exposure，不能只能从 interesting 数据中抽样。

### 5.4 数据源和 split

- split 在 VQA fan-out、FullCoT projection 和方向过采样之前按 `split_group_id` 冻结；
- 同一 clip、sequence、route 或语义等价 parent 不能跨 split；
- SideWalk main/extra 的 1,729 条等价重复必须继续确定性去重；
- train 允许 replacement，validation/test 必须保持 unique、自然分布、零过采样；
- 每个 release 输出 dataset x direction x modality x task 的支持矩阵和结构零。

## 6. FullCoT 与 VQA 标注合同

### 6.1 FullCoT

只发布 Reasoning V4.6 六段 canonical formal view：

```text
TaskGrounding
SceneGrounding
LocalTraversability
NavigationContextGrounding
InteractionConstraintGrounding
Decision
```

原始 LVLM annotation 不能直接作为训练输入。发布必须经过：

```text
record_complete
  -> all_sections_accepted
  -> candidate_structurally_valid
  -> Action Reconciliation
  -> review resolved/not_required
  -> formal_training_eligible
  -> formal view export
  -> artifact audit
```

标注时必须区分：

- active instruction 与任务模板；
- FPV physical traversability 与 BEV route direction；
- LVLM visual proposal 与 private strong action；
- 物理障碍、社会交互和交通规则；
- insufficient evidence、not applicable 和 provider failure。

### 6.2 VQA

- E0 direction 可使用获准 trajectory-derived GT；
- 其他 weak tasks 必须记录 teacher model、prompt digest、label source 和 confidence；
- 每个 dataset x task x modality 至少随机人工审阅 100 条；
- 所有 stop/move、opposite-turn、private-field leakage 和 evidence-ref closure conflict 全审；
- weak-label 指标只能称 teacher agreement；
- dormant 或无 producer 的任务不生成 placeholder row；
- interesting 与 normal-control 必须分别报告。

### 6.3 provenance

每个可训练字段必须属于以下来源之一：

```text
runtime_fact
gt_metric
gt_direction_derived
producer_metadata
lvlm_weak
human_verified
deterministic_projection
fallback_untrainable
```

`fallback_untrainable` 只能保留在审计产物中，对应 LM/trajectory/action loss 必须关闭。

## 7. 独立 GPS/waypoint guidance

新 waypoint 数据必须来自独立 planner 或导航任务，而不是复制未来 odometry target。每条样本至少
保存：

```text
current_gps / current_heading
goal_gps
planner_polyline_gps
planner_id / route_version / plan_timestamp
CRS 和 map datum
当前点投影到 route 的 along-track / cross-track
lookahead waypoint 的选择规则和距离
waypoint validity / reroute / stale-plan 状态
```

模型可见 waypoint 与 CFM GT 必须具有不同 provenance：

```text
waypoint input = independent planner guidance
CFM target     = executed future odometry
```

若二者来自同一原始轨迹，必须禁止该样本进入 waypoint-conditioned 训练，以避免 target leakage。

## 8. 分数据集处理要求

### 8.1 SideWalk

- 完成 NavigationContext contract repair 和 formal re-export；
- 从 producer index 恢复 BEV alignment、frame、ego-forward 和 route authority；
- 保持 main/extra stable-parent 去重；
- 对 repeated visual padding 单独打标，不允许它影响未来 trajectory mask；
- 继续保留可信 pose/camera trajectory，但必须经过 V2.1 continuity gate。

### 8.2 Foxglove

- raw exporter 必须写 `frame_pose_valid`，禁止缺失 pose 零填充；
- 全量重跑 non-finite、step、gap、yaw-rate 和 quaternion-order QC；
- `>6 m/s` 样本进入独立 audit slice，不批量删除；
- 针对其 Stage II ADE/FDE 高于 SideWalk 的现象，增加按 clip、速度、转向和 pose-quality 分层；
- normalization 只能在清理后的 train split 上重算。

### 8.3 EgoWalk

- caption 必须分类为 navigation instruction、scene description 或 mixed；
- instruction coverage 必须覆盖完整未来 4 秒，否则不得生成该 anchor 的 CFM target；
- backward discontinuity、risk、timestamp 和 odometry continuity 分别保留原因码；
- VQA 必须补 normal-control，防止 EgoWalk 62.40% source 质量主导；
- EgoWalk 没有 BEV，也没有可信 camera-ground FPV trajectory，不得制造这些监督。

### 8.4 FrodoBots

- 继续允许 FPV、受控 BEV、cross-view VQA 和获准弱方向；
- continuous trajectory、FPV pixel trajectory、System1/CFM loss 必须保持关闭；
- 若要升级为强轨迹数据，必须重新提供可信 odometry、相机内外参、时间同步和投影验证；
- 仅靠 pose outlier 删除或 anchor 平滑不能把近似标定升级为强 GT。

## 9. 数据专家交付物

每个数据版本必须输出：

```text
raw_index.jsonl
source_manifest.jsonl
trajectory_targets.jsonl
trajectory_qc_decisions.jsonl
accepted.jsonl
rejected.jsonl
duplicate_map.jsonl
split_report.json
distribution_report.json
normalization_profile.json
annotation_status.json
formal_training_view.jsonl
artifact_audit.json
review/index.html
dataset_status.json
```

报告至少包含：

- 输入、去重、accepted、rejected 和每个 reason code 的数量；
- dataset/direction/stop reason/modality/task/interesting-control 分布；
- `max_step/max_gap/max_speed/max_acceleration/max_yaw_rate` 的分位数；
- 每个媒体文件的 decode 结果和必要 SHA-256；
- split group overlap；
- annotation schema/prompt/profile/model digest；
- formal eligible 与 raw annotation 的差额；
- random、long-tail、stop、outlier、weak-label 和 normal-control review 页面。

## 10. 发布门禁

一个数据版本只有同时满足以下条件才可进入正式训练：

- accepted trajectory 的 non-finite、invalid pose、step/gap/yaw-rate hard violation 均为 0；
- trajectory shape、frame、单位、quaternion order 和 calibration provenance 全部闭合；
- train/validation/test 的 `split_group_id` overlap 为 0；
- FrodoBots trajectory/System1 train row 为 0；
- stop 标签同时具有 intent、physical state、reason 和 provenance；
- FullCoT 只来自 formal exporter，pending/rejected 泄漏为 0；
- VQA weak label 与 derived/external GT 分开统计，并具备 matched normal-control；
- stop/move 和 opposite-turn blocking conflict 为 0，或全部进入显式人工 adjudication；
- normalization 只读取当前 clean train split，并绑定准确 train SHA；
- HTML 人工 review 已覆盖各数据源、模态、方向长尾、stop 和物理异常切片；
- manifest、report、代码、schema、prompt、模型和数据 fingerprint 可复现。

## 11. 执行顺序

建议数据团队按以下批次交付：

1. 先重建 SideWalk/Foxglove clean metric trajectory，并生成全部 hard-QC reject；
2. 重新定义和审计 stop，重建 train-only normalization；
3. 补充方向长尾、straight/10 度边界和 stop 原因数据；
4. 为当前 VQA interesting pool 构造 matched normal controls；
5. 完成 SideWalk repair、EgoWalk FullCoT 和 SideWalk extra formal export；
6. 新采集带独立 planner provenance 的 GPS/waypoint guidance；
7. 冻结新数据版本后，重新运行 Stage I generation audit 和 Stage II CFM 训练/Pass-T。

## 12. 关联文档

- [当前标注状态](DATASET_ANNOTATION_STATUS.md)
- [Reasoning V4 数据标注设计](REASONING_V4_DATA_ANNOTATION_DESIGN.md)
- [VQA V2 数据标注设计](VQA_V2_DATA_ANNOTATION_DESIGN.md)
- [Stage II CFM 数据合同](STAGE_II_CFM_DATA_CONTRACT.md)
- [Stage I 联合训练设计](../../../../ms-swift/docs/Experiments/stage_i_fullcot_vqa_joint_training_design_20260803.md)
- [legacy direction-12 评测审计](../../../../ms-swift/docs/Experiments/legacy_full_cot_direction12_evaluation_audit_20260802.md)
- [Stage II CFM 实验记录](../../../../ms-swift/docs/Experiments/stage_ii_cfm_4b_full_59k_20260802.md)
- [ms-swift 数据处理说明](../../../../ms-swift/docs/dataset/sidewalk_data_processing.md)
