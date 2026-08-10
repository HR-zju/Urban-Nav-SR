# Language / Waypoint Benchmark V1

更新时间：2026-08-09。

## 1. 目的

本模块为 UrbanNav 自有数据构造可复现的开放环 benchmark。它借鉴 UrbanNav 的
"语言条件输入、未来 waypoint 监督" 思路，以及 CityWalker 的"冻结样本、局部坐标、
短时轨迹评分"原则，但不复制二者的随机抽样或未来信息尺度归一化。

第一版完成离线数据合同、预处理、冻结 package 与模型侧 target-free adapter 接口。语言目标
teacher 已完成三个数据集各 $64$ 条真实 LVLM smoke，并输出 portable HTML review；这不是全量
标注，也不是正式 benchmark 或训练结论。当前不修改模型结构或 loss。

## 2. 三类严格分离的数据

| 对象 | 模型可见 | 可信来源 | 用途 |
| --- | --- | --- | --- |
| `LanguageGoalAnnotationV1` | 是 | 通过 validator 的 LVLM 或人工标注 | 高层目标、无障碍和社会导航约束 |
| `WaypointGuidanceV1` | 是 | 独立地图、planner、GPS route 或人工 route | 局部路线 guidance |
| `StageIITrajectorySidecarV2` | 否 | 可信 future odometry | $32\times3$ CFM 监督和 benchmark reference |

禁止把 `trajectory_ego_t`、future pose 或 demonstration route 复制到
`WaypointGuidanceV1`。这一边界由 typed provenance 的
`independent_of_future_target=true` 与 `BenchmarkInputV1` 的私有键检查共同保证。

FrodoBots 的连续 trajectory 不可信，因此没有合法的 Stage II reference 时不能进入
该 trajectory benchmark；它仍可单独用于 FPV-BEV 理解和离散方向任务。

## 3. 语言目标合同

`LanguageGoalCandidateV1` 可以从 V2 sidecar 的未来点生成，但仅供 teacher 标注阶段
选择目标候选，永远不可进入学生 prompt。真正可见的文本必须是
`LanguageGoalAnnotationV1`，并且在 `accepted` 状态时强制包含：

- `instruction_text` 和 `instruction_type`；
- 当前 anchor 可见的 `target_ref`；
- `visibility_at_anchor=true`；
- 至少一个当前样本证据引用 `evidence_refs`；
- teacher/human provenance、validator 状态和可选人工复核状态。

因此合法文本可以是“向右侧可见的高楼方向移动”，但不能由 `right_20` 这类动作标签
直接模板化伪造。旧的 motion-derived instruction 仍只能用于 audit 或消融。

## 4. Waypoint 几何与归一化

规划器输入使用 WGS84。局部切平面近似下，当前经纬度为
$(\phi_0,\lambda_0)$，route 点为 $(\phi,\lambda)$：

$$
n=R(\phi-\phi_0),\qquad
e=R\cos(\phi_0)(\lambda-\lambda_0).
$$

其中角度先转换为弧度，$R=6\,378\,137\ \mathrm{m}$。以从北顺时针为正的当前
heading $\psi$ 转为当前 ego FLU：

$$
x_{\mathrm{forward}}=n\cos\psi+e\sin\psi,
$$

$$
y_{\mathrm{left}}=n\sin\psi-e\cos\psi.
$$

路线按弧长重采样。默认参数为每 $2\ \mathrm{m}$ 一个点、最多 $8$ 点、公开 horizon
$D=30\ \mathrm{m}$、局部 goal 为 $10\ \mathrm{m}$。学生只接收固定尺度特征：

$$
g_i=\left[
\frac{x_i}{D},
\frac{y_i}{D},
\sin\Delta\psi_i,
\cos\Delta\psi_i,
\frac{s_i}{D},
\mathbb{1}_{\mathrm{terminal}}
\right].
$$

$D$ 是公开 planner policy，绝不能通过 future trajectory 的统计量拟合。只有 planner
真实终点落在公开 horizon 内时，最后一项为 $1$。

`route_step_m`、`route_horizon_m`、`route_max_points` 和 `local_goal_distance_m` 会写入
guidance artifact；package builder 会把它们与 CLI/config 中的公开 policy 逐项比对。任何
不一致 row 以 `waypoint_sampling_policy_mismatch` 拒绝，避免不同归一化分布静默混合。

## 5. 配对 benchmark arm

同一 `parent_sample_id` 和同一 hidden reference 可产生四个配对输入：

```text
fpv_only
fpv_language
fpv_waypoint
fpv_language_waypoint
```

BEV 是独立控制变量：`--include-bev` 会为所有 arm 同时要求并加入 `bev.current`；它不能
在某一个 arm 中偷偷可见。默认要求恰好 5 帧 FPV，数据集差异通过
`LanguageWaypointBenchmarkBuildConfigV1` 参数显式配置，而非隐式按字段猜测。

每个 package 均写：

```text
inputs.jsonl       # 仅模型可见 messages/images/normalized public route
references.jsonl   # evaluator-only [32,3] trajectory、timestamp、mask
rejects.jsonl      # 逐 parent fail-closed 原因
package.json       # digest、split、profile 和 source fingerprint
report.json
```

已接受 language annotation 不足时，默认连 `fpv_only` 也拒绝该 parent。这样四个 arm
共用同一物理样本集合，才能计算配对增益；若只需要非配对 coverage，明确传
`--no-require-language-annotation-for-controls`。

## 6. 命令行

先从可信 V4 source 与 V2 sidecar 写 teacher-only 候选：

```bash
cd Agent_part/RouteServer
conda run -n UrbanNav env PYTHONPATH=.:src python \
  tools/evaluation/build_language_waypoint_benchmark.py candidates \
  --source-jsonl SOURCE_V4.jsonl \
  --trajectory-sidecar-jsonl TRAJECTORY_SIDECAR_V2.jsonl \
  --output-jsonl RUN/language_goal_candidates.jsonl \
  --report-json RUN/reports/candidates.json
```

独立 planner adapter 先输出严格 `PlannerRouteRecordWgs84V1` JSONL，再做统一 conversion：

```bash
conda run -n UrbanNav env PYTHONPATH=.:src python \
  tools/evaluation/build_language_waypoint_benchmark.py guidance \
  --planner-route-jsonl PLANNER_ROUTE_WGS84.jsonl \
  --output-jsonl RUN/waypoint_guidance.jsonl \
  --report-json RUN/reports/guidance.json
```

Foxglove 原始 `frame_gps.npy` 只能生成当前定位锚点。该操作仅使用当前帧及历史 GPS
计算 heading；没有 planner log 时所有可用锚点仍会以 `planner_log_missing` 拒绝，绝不会
输出 waypoint guidance：

```bash
conda run -n UrbanNav env PYTHONPATH=.:src python \
  tools/evaluation/build_native_planner_records.py foxglove-anchors \
  --source-jsonl FOXGLOVE_SOURCE_V4.jsonl \
  --foxglove-raw-root FOXGLOVE_RAW_ROOT \
  --output-jsonl RUN/foxglove_native_anchors.jsonl \
  --report-json RUN/reports/foxglove_native_anchors.json
```

只有上游提供具有 request id、时间、当前 pose、goal、polyline、provider/version/datum 的
独立 planner ledger 后，才能进行 fail-closed join：

```bash
conda run -n UrbanNav env PYTHONPATH=.:src python \
  tools/evaluation/build_native_planner_records.py planner-records \
  --anchor-jsonl RUN/foxglove_native_anchors.jsonl \
  --planner-log-jsonl INDEPENDENT_PLANNER_LOG.jsonl \
  --output-jsonl RUN/planner_route_records.jsonl \
  --decisions-jsonl RUN/planner_route_decisions.jsonl \
  --report-json RUN/reports/planner_route_join.json
```

join 仅接受与 observation 同一 native record、在 observation 之前生成、未超过公开时效，且
GPS origin 与 heading 均匹配的 route。`NativePlannerRouteLogV1` 在 schema 中强制
`independent_of_future_target=true`；不满足任一条件的 parent 只写 reject decision。

最后构造一个测试 arm：

```bash
conda run -n UrbanNav env PYTHONPATH=.:src python \
  tools/evaluation/build_language_waypoint_benchmark.py package \
  --source-jsonl SOURCE_V4.jsonl \
  --trajectory-sidecar-jsonl TRAJECTORY_SIDECAR_V2.jsonl \
  --language-annotation-jsonl ACCEPTED_LANGUAGE.jsonl \
  --waypoint-guidance-jsonl RUN/waypoint_guidance.jsonl \
  --output-root RUN/packages/fpv_language_waypoint \
  --split test --package-id benchmark_test_lang_wp_v1 \
  --arm fpv_language_waypoint
```

## 7. 2026-08-09 真实 smoke 与原始数据审计

语言目标 teacher 使用单帧 `fpv.current` 及模型可见文本；候选 future point 只在 teacher
内部挑选目标，未写入 student prompt 或 review 页面。已完成的结果如下：

| 数据集 | 输入 | accepted | rejected | API/schema error | 吞吐 | 本地 review |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SideWalk | 64 | 64 | 0 | 0 | 5,153 rows/h | `review_outputs/language_waypoint_smoke_20260809/sidewalk_diverse_64_language_goal_review.html` |
| Foxglove | 64 | 62 | 2 | 0 | 4,895 rows/h | `review_outputs/language_waypoint_smoke_20260809/foxglove_diverse_64/review/foxglove_diverse_64_language_goal_review.html` |
| EgoWalk | 64 | 64 | 0 | 0 | 5,653 rows/h | `review_outputs/language_waypoint_smoke_20260809/egowalk_diverse_64/review/egowalk_diverse_64_language_goal_review.html` |

这三份页面均为 self-contained artifact，嵌入当前 FPV thumbnail、语言标注、可见 evidence
和可选人工裁决；静态审计确认其中不含 `candidate_xy_flu_m`、future route 距离或时间。人工
review 尚未完成，accepted 仅表示 teacher + schema validator 接受。

review 页会下载 `language_goal_human_adjudications.json`。在任意全量语言标注前，必须先运行
离线 gate；默认要求所有 teacher-accepted 卡片都有裁决、accept rate 不低于 $0.90$、uncertain
rate 不高于 $0.05$，并对每个出现的 instruction type 至少审阅一条。阈值是显式 CLI 参数，不能
只因输出量较大而绕过：

```bash
conda run -n UrbanNav env PYTHONPATH=.:src python \
  tools/evaluation/evaluate_language_waypoint_human_review.py \
  --annotation-jsonl LANGUAGE_GOAL_ANNOTATIONS.jsonl \
  --adjudications-json language_goal_human_adjudications.json \
  --report-json RUN/reports/language_goal_human_review_gate.json
```

原始数据与 staged V4 source 的 provenance 审计结论：

| 数据集 | 当前可用于 anchor 的原始信息 | 独立 route/goal/planner log | waypoint arm 结论 |
| --- | --- | --- | --- |
| SideWalk | 当前 staged source 无 WGS84 GPS | 无 | 不可构造 |
| Foxglove | 原始每帧 `frame_gps.npy`、`frame_times.npy`，可用过去 GPS 求当前 heading | 无；BEV route overlay 与 `route_theta_deg` 只是演示/地图上下文 | 仅可产 anchor；不可构造 guidance |
| EgoWalk | parquet odometry 是局部运动，不是 WGS84 route | 无 | 不可构造 |

录制 GPS history 也属于 demonstration，不是用户请求的 planner route。只有补齐独立 provider
route、人工 route 或可审计 mission log 后，Foxglove 才可能首先进入真实 waypoint arm。

对 Foxglove $64$ 条真实 smoke source 的 native adapter 实跑进一步确认该边界：只有 $1/64$
能够在不看 future GPS 的情况下获得 past-bearing heading，另外 $63/64$ 因
`native_past_heading_unavailable` 停留在 anchor unavailable。对全部 $64$ 条输入空的独立
planner ledger 后，输出 `PlannerRouteRecordWgs84V1` 为 $0$；决策分布为 $63$ 条
`anchor_native_past_heading_unavailable` 和 $1$ 条 `planner_log_missing`。该结果只量化这批
smoke 的可用性，不能外推为全量 Foxglove 覆盖率，但足以证明当前现有路径没有可用 waypoint
guidance。

### 7.1 已批准的全量语言目标 queue

全量语言 teacher 不改变上面的 waypoint guidance 边界：它只生成 `LanguageGoalAnnotationV1`，而不
把 demonstration trajectory 改造成学生可见的 `WaypointGuidanceV1`。候选 future target 仍只来自完整的
`StageIITrajectorySidecarV2` `$4\mathrm{s}/32$` ledger，并且只在 teacher 内部决定语义目标关系。

`run_language_waypoint_full_dual_122b.sh` 先将 train/validation/test sidecar 去重为 dataset-local
ledger，再以

$$
\operatorname{lane}(p)=\operatorname{SHA256}(p)\bmod 2
$$

对 stable `parent_sample_id` 进行双服务分片。每条 parent 因此只由一个服务写入；合并器按 canonical
candidate 顺序输出 annotation 与 provider/validator decision，并单独报告 annotation coverage、decision
coverage 和 missing decision。服务中断后的重跑只复用同一 parent 的 accepted append-only annotation，
不会覆盖历史 audit。

完整 `$4\mathrm{s}/32$` V2 sidecar 必须先在运行服务器可读；旧 `$2\mathrm{s}/16$` sidecar 与仅 $64$ 条
smoke sidecar 都不能用于全量 language-goal candidate。全量语言结果也不能被描述为 waypoint guidance
已构造：当前 SideWalk、Foxglove 和 EgoWalk 仍缺独立 planner route，因此仅能构造 `fpv_language`，不能
构造带真实 route guidance 的 waypoint arm。

## 8. 模型侧边界与下一步

`ms-swift` 的 generic adapter 只转发 package 的 public fields，并已声明四个 arm 及其
BEV 变体为可接收 profile。当前 Stage II CFM head 尚未训练 waypoint-conditioned 模型；
因此它可以用于 input contract / zero-shot 路径 smoke，但不能把结果解释为 waypoint
guidance 已被模型利用。

在真实全量运行前，后续工作必须是：

1. 人工完成三个 $64$ 条 review，并回写 `accepted/rejected/needs_rewrite` 裁决；
2. 对每个数据集盘点独立 goal、planner request 和 route response；SideWalk/EgoWalk 当前仍无
   可实现的 native route adapter；
3. 使用 `NativePlannerAnchorV1` / `NativePlannerRouteLogV1` 对补齐的 Foxglove planner log
   做真实 join smoke，并审计 stale、origin mismatch 和 heading mismatch；
4. 仅在 guidance 实际 accepted 后生成冻结四 arm paired package 与 HTML review；
5. 冻结 parent/split registry，检查 media、路线 provenance 与目标泄漏；
6. 最后才决定是否增加 waypoint encoder、condition-aware Stage I/II 训练和正式 benchmark。
