# Language / Waypoint Benchmark V1

更新时间：2026-08-09。

## 1. 目的

本模块为 UrbanNav 自有数据构造可复现的开放环 benchmark。它借鉴 UrbanNav 的
"语言条件输入、未来 waypoint 监督" 思路，以及 CityWalker 的"冻结样本、局部坐标、
短时轨迹评分"原则，但不复制二者的随机抽样或未来信息尺度归一化。

第一版只完成离线数据合同、预处理、冻结 package 与模型侧 target-free adapter 接口；
不调用 LVLM、不修改模型结构或 loss、不生成全量数据和 HTML review。

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

## 7. 模型侧边界与下一步

`ms-swift` 的 generic adapter 只转发 package 的 public fields，并已声明四个 arm 及其
BEV 变体为可接收 profile。当前 Stage II CFM head 尚未训练 waypoint-conditioned 模型；
因此它可以用于 input contract / zero-shot 路径 smoke，但不能把结果解释为 waypoint
guidance 已被模型利用。

在真实全量运行前，后续工作必须是：

1. 为每个原始数据集实现显式 native planner adapter，而不是字段名启发式；
2. 完成 $50$--$100$ 条 language annotation smoke，检查 referent 和约束真实可见；
3. 生成四 arm 的 paired HTML review；
4. 冻结 parent/split registry，检查 media、路线 provenance 与目标泄漏；
5. 再决定是否增加 waypoint encoder 和 condition-aware Stage I/II 训练。
