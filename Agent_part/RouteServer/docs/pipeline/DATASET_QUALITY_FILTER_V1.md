# UrbanNav 可通行几何与训练可用性过滤 V1

## 1. 目的和边界

本模块为现有 `ReasoningSourceV4` 新增三个**独立、只读 sidecar**：

1. `LocalTraversabilityGeometryV1`：当前 FPV 中的局部可通行、条件通行、阻塞和禁止区域；
2. `NavigationTrainabilityFilterV1`：场景是否属于室外步行导航语料，以及记录运动与时序 FPV 是否一致。
3. `SceneEventTagsV1`：从前两类 sidecar 和 teacher-only direction audit 投影出的 CityWalker-compatible
   多标签场景切片。

它们不会修改原始 source、媒体文件、V4.6 FullCoT、VQA 或旧 language-goal manifest。任何训练使用都只能读取
本模块导出的新 view，不能把 sidecar 字段原地写回 V4 source。

V1 **不使用** `fpv_segmentation`、SoM 或检测框作为区域输入。几何 teacher 直接从 `fpv.current` 预测透视四边形；
后续若接入候选区域 adapter，必须发布新 profile、独立 digest 和新 run root，不能与本版本缓存混用。

## 2. 合同

### 2.1 `LocalTraversabilityGeometryV1`

代码：[local_traversability_geometry.py](../../src/mcp_server/contracts/local_traversability_geometry.py)。

每个 region 使用归一化图像坐标的四边形：

```text
near_left -> near_right -> far_right -> far_left
```

几何 validator 要求：

- 四点有限且在 $[0,1]^2$；
- 四边形凸、无自交、非退化；
- 近边在图像中低于远边；
- 面积既不趋近于零，也不能接近整幅图像；
- `direct_traversable` / `conditional_traversable` 不得标成 grass、stairs、vehicle、pedestrian cluster、wall/fence 或 construction；
- 所有 region 必须引用 `fpv.current`；
- `accepted` 必须选择一个 direct 或 conditional corridor。

该合同表达的是**局部图像平面支持区域**，不表达米制深度、BEV 对齐、轨迹、waypoint 或最终动作。
旧 language-goal 的 `target_bbox_anchor_xyxy` 是可见语言目标锚点，语义不同，保持原样。

### 2.2 `NavigationTrainabilityFilterV1`

代码：[navigation_trainability.py](../../src/mcp_server/contracts/navigation_trainability.py)。

过滤由三部分组成：

```text
deterministic MotionEpisodeSummaryV1
  + SceneSuitabilityAssessmentV1
  + MotionSceneAssessmentV1
  -> NavigationTrainabilityFilterV1
```

其中 motion summary 是 teacher-only QC 信息。对累计 ego 轨迹 $p_i=(x_i,y_i,\psi_i)$ 和时间间隔
$\Delta t$，计算：

$$
v_i=\frac{\left\|p_i^{xy}-p_{i-1}^{xy}\right\|_2}{\Delta t},\qquad
a_i=\frac{\left|v_i-v_{i-1}\right|}{\Delta t}.
$$

极端速度、加速度、步长或 yaw rate 为确定性拒绝条件；LVLM 只判断可见场景与时序画面是否支持该记录运动，
不能自行放行 Stage I/II。最终状态为：

| 状态 | 含义 | 可导出视图 |
| --- | --- | --- |
| `accepted` | scene 与 motion 证据完整且一致 | Stage I、Stage II、VQA |
| `quarantine` | 视觉/运动证据不足或存在可审计风险 | 视具体 eligibility，仅可能 VQA 或 Stage I |
| `rejected` | 室内/地铁/非步行场景或硬运动异常 | 无 |
| `pending_retry` | provider、schema 或 source revision 未闭合 | 无 |

`ReasoningSourceV4` 为避免把 GT 暴露给 FullCoT teacher，通常不携带完整 CFM target。因此 motion
QC 优先使用可选的 `StageIITrajectorySidecarV2`，以 `sample_id/source_group_id + dataset_source + split_group_id`
严格关联。sidecar 的 `sidecar_digest` 会写入 `MotionEpisodeSummaryV1`，并参与 filter cache identity；它不会
写回 source，也不会出现在 student prompt。未提供可信 V2 sidecar 时，模块只完成 geometry/current-scene 筛选，
并将该行明确标为 Stage II 不可用，而不是臆造时序运动结论。

### 2.3 `SceneEventTagsV1`

代码：[scene_event_tags.py](../../src/mcp_server/contracts/scene_event_tags.py) 与
[scene_events.py](../../src/mcp_server/data/quality/scene_events.py)。该层不调用 LVLM：它只在 geometry 与
suitability 完成后，按 source fingerprint 对齐当前 sidecar，确定性写出六个**可重叠**标签的三态结果：

| 标签 | `present` 的最小证据 | 不可推断的边界 |
| --- | --- | --- |
| `crowd` | near-path `moderate/dense`，或 geometry 中有 `pedestrian_cluster` | 不从轨迹速度推断人群 |
| `person_close_by` | shared/blocked corridor 且有可见人群，或 `pedestrian_cluster` | 不把普通 sparse crowd 一律判为近距离交互 |
| `turn` | teacher-only 强方向为左/右，或 recorded motion 为 `turning` | 仅作 audit/slice，绝不写回 student 输入 |
| `crossing` | `red_light/crossing_wait`，或 geometry 中有 `crosswalk` | 不从 stop 标签单独推断路口 |
| `other` | 明确的非 `outdoor_pedestrian_supported` scene class | 不把 `unknown` 伪装成负例 |
| `action_target_mismatch` | 需要未来独立 action/target audit | 当前 V1 固定为 `unavailable`，不从 trajectory 或 direction 猜造 |

每个标签均为 `present`、`absent` 或 `unavailable`：`absent` 仅表示对应 sidecar 已提供足够证据但未触发；
`unavailable` 表示缺少独立证据，训练和分析不得把它当作负标签。标签名与 CityWalker 公共类别顺序一致，方便
slice 对齐；但 UrbanNav 投影不是 CityWalker 的人工标注，不可与其 reference label 混合训练或直接比较比例。

可单独执行投影，不会改变任何输入 ledger：

```bash
python tools/model_processing/project_citywalker_scene_event_tags.py \
  --source-jsonl "$RUN_ROOT/source/smoke.jsonl" \
  --filter-jsonl "$RUN_ROOT/suitability/filter.jsonl" \
  --geometry-jsonl "$RUN_ROOT/geometry/geometry.jsonl" \
  --output-jsonl "$RUN_ROOT/events/scene_event_tags.jsonl" \
  --report-json "$RUN_ROOT/events/report.json"
```

## 3. 两个独立 LVLM profile

| profile | 图像输入 | private motion | 作用 |
| --- | --- | --- | --- |
| `urbannav_local_traversability_quad_direct_v1` | 仅 `fpv.current` | 不可见 | 预测四边形与局部支持状态 |
| `urbannav_scene_suitability_visible_fpv_v1` | 仅 `fpv.current` | 不可见 | 判定室外步行数据池适用性 |
| `urbannav_motion_scene_consistency_temporal_fpv_v1` | 最多 5 帧 history/current FPV | 仅 teacher 可见 | 判断画面能否解释记录运动 |

每个 profile 的 `profile_id + prompt/profile schema digest + source_fingerprint` 共同组成缓存边界。修改 prompt、schema、
source 或 motion summary 后会自动 cache miss；不会静默复用旧结果。所有 provider 原始文本、retry 和 parser 错误写入
独立 `decisions.jsonl`，密钥与图片 bytes 不会写入 ledger。

## 4. 分层 smoke

脚本：[build_navigation_quality_smoke.py](../../tools/model_processing/build_navigation_quality_smoke.py)。

每个数据集先构造 $100$--$200$ 条 source-only smoke。默认 6 个 strata 各请求 $24$ 条，最多 $160$ 条：

```text
indoor_or_transit, stop, high_speed, dense_crowd, ordinary_sidewalk, turn
```

`stop` 与 `turn` 使用已有强方向 stamp，`high_speed` 使用 teacher-only trajectory summary。室内/地铁和密集人群
只使用 source 的可见 instruction/context/warning 作**选样 hint**，并在报告中写明短缺；这不是 LVLM 或训练标签。
不足的 stratum 不复制样本，而是记录 `shortfalls`，再由人工调整数据源或采样配置。

## 5. 推荐 run root 和命令

不要复用旧 V3/V4.6 annotation run root。每个数据集独立创建：

```text
<run_root>/
  source/smoke.jsonl
  geometry/geometry.jsonl
  geometry/decisions.jsonl
  geometry/report.json
  suitability/filter.jsonl
  suitability/decisions.jsonl
  suitability/report.json
  events/scene_event_tags.jsonl
  events/report.json
  views/{accepted,quarantine,rejected,pending_retry}.jsonl
  views/{stage_i_eligible,stage_ii_eligible,vqa_eligible}.jsonl
  reports/*.json
  review/geometry_review.html
  review/suitability_review.html
```

以下示例只说明本地入口。`LLM_PROVIDER`、`LLM_MODEL`、`LLM_API_BASE_URL` 和 API key 按服务器环境显式设置；
真正调用图片前仍需独立获得外发批准。

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer
export PYTHONPATH="$PWD/src"

python tools/model_processing/build_navigation_quality_smoke.py \
  --source-jsonl "$SOURCE_JSONL" \
  --dataset-source sidewalk \
  --output-jsonl "$RUN_ROOT/source/smoke.jsonl" \
  --report-json "$RUN_ROOT/reports/smoke_selection.json" \
  --target-per-stratum 24 \
  --max-rows 160

python tools/model_processing/label_traversability_geometry.py \
  --source-jsonl "$RUN_ROOT/source/smoke.jsonl" \
  --geometry-jsonl "$RUN_ROOT/geometry/geometry.jsonl" \
  --decisions-jsonl "$RUN_ROOT/geometry/decisions.jsonl" \
  --report-json "$RUN_ROOT/geometry/report.json" \
  --concurrency 4 --max-retries 1 --image-detail high

python tools/model_processing/filter_navigation_trainability.py \
  --source-jsonl "$RUN_ROOT/source/smoke.jsonl" \
  --trajectory-sidecar-jsonl "$TRAJECTORY_SIDECAR_V2_JSONL" \
  --filter-jsonl "$RUN_ROOT/suitability/filter.jsonl" \
  --decisions-jsonl "$RUN_ROOT/suitability/decisions.jsonl" \
  --report-json "$RUN_ROOT/suitability/report.json" \
  --views-root "$RUN_ROOT/views" \
  --concurrency 4 --max-retries 1 --max-temporal-fpv-frames 5

python tools/evaluation/build_traversability_geometry_review_html.py \
  --source-jsonl "$RUN_ROOT/source/smoke.jsonl" \
  --geometry-jsonl "$RUN_ROOT/geometry/geometry.jsonl" \
  --geometry-decisions-jsonl "$RUN_ROOT/geometry/decisions.jsonl" \
  --output-html "$RUN_ROOT/review/geometry_review.html"

python tools/evaluation/build_navigation_trainability_review_html.py \
  --source-jsonl "$RUN_ROOT/source/smoke.jsonl" \
  --filter-jsonl "$RUN_ROOT/suitability/filter.jsonl" \
  --output-html "$RUN_ROOT/review/suitability_review.html"
```

完整自包含页会嵌入每条样本的全部 FPV；例如 $150$ 条、每条 $5$ 帧的 suitability page 可能达到数十 MB。
需要经 SSH 转发远程浏览时，应使用确定性分页，而不是误把浏览器仍在下载的空白区域判为媒体丢失：

```bash
python tools/evaluation/build_navigation_trainability_review_html.py \
  --source-jsonl "$RUN_ROOT/source/smoke.jsonl" \
  --filter-jsonl "$RUN_ROOT/suitability/filter.jsonl" \
  --output-html "$RUN_ROOT/review_pages/suitability/page_000.html" \
  --sample-offset 0 --max-samples 10 \
  --thumbnail-max-width-px 512 --jpeg-quality 72
```

将 `--sample-offset` 依次设置为 `10, 20, ...` 即可得到相同排序下的后续页。geometry renderer 使用相同的
`--sample-offset`，并同时保留 parser failure/retry diagnostics。分页只改变 browser artifact 的载入量，
不会改变 source、sidecar、review 顺序或 validator 结果。

`--retry-nonaccepted-on-resume` 只重跑已解析但非 accepted 的几何 sidecar；`--retry-pending-on-resume`
只重跑 filter 的 `pending_retry`。二者不会重写既有 JSONL，只追加当前 profile/source revision 的新 row。
若某个数据集尚未准备可信 V2 sidecar，可省略 `--trajectory-sidecar-jsonl`，但该 run 只能验收 geometry 和
scene suitability，不能据此启用 Stage II。

若 prompt/schema producer 已升级，先以旧 decision ledger 构造失败 parent 的 source-only repair view，
避免重跑已 accepted 样本：

```bash
python tools/model_processing/build_traversability_geometry_retry_manifest.py \
  --source-jsonl "$RUN_ROOT/source/smoke.jsonl" \
  --geometry-decisions-jsonl "$RUN_ROOT/geometry/decisions.jsonl" \
  --output-jsonl "$RUN_ROOT/geometry_repair/source.jsonl" \
  --report-json "$RUN_ROOT/geometry_repair/selection_report.json"
```

repair view 必须写入新的 geometry/decision ledger。不同 prompt digest 的结果可以并存用于审计，
但在人工 gate 通过前不得合并为训练 supervision。

完成一个多数据集 smoke 后，使用只读汇总器生成单一 QC receipt：

```bash
python tools/evaluation/summarize_navigation_quality_run.py \
  --run-root "$RUN_ROOT" \
  --output-json "$RUN_ROOT/navigation_quality_run_report.json"
```

该 receipt 分别统计 geometry parser error、区域 role/surface、scene suitability、motion QC、
Stage eligibility、按 disposition 划分的方向分布，以及各 scene-event 标签的 `present/absent/unavailable`
分布。它不把 `quarantine` 解释为 reject，也不改变任何 source 或 sidecar。

## 6. 人工 gate 与全量运行

必须先检查两份 HTML：

1. **geometry review**：走廊四边形是否沿真实可步行表面收敛，是否把草地、机动车道、车辆、楼梯、密集人群或
   图片底部大矩形误标为 direct support；
   同时应筛选 `error` / `rejected` 状态，检查原始 JSON、validator error 和 retry 是否属于 prompt 格式问题，
   不能把未能解析的 teacher 输出误解为场景不通行；
2. **suitability review**：室内/地铁、密集人群、停止、异常高速与普通 sidewalk 的 accepted/quarantine/rejected
   是否符合数据策略；特别核对 `likely_unexplained_stop` 的误杀率。

在确认最小面积、最大面积、hard motion threshold、quarantine policy 和误杀率之前，禁止对全量 source 运行，
也禁止将 `views/stage_i_eligible.jsonl` 或 `views/stage_ii_eligible.jsonl` 接入 compiler。

人工确认后，全量运行仍写入新的 run root，且必须输出：

```text
geometry_report.json
suitability_report.json
navigation_trainability_views_report.json
geometry_review.html
suitability_review.html
```

最后单独比较过滤前后 physical parent 数、12-way direction、stop、trajectory QC 和各 Stage eligibility；
该比较完成前，过滤结果不构成 Stage I/II 训练集变更。

## 7. 2026-08-09 真实 Qwen 122B smoke 记录

本节记录一次真实 provider smoke，不等同于全量筛选或训练集验收。运行目录位于私有服务器：

```text
/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation/runs/
navigation_quality_v1_qwen122b_smoke_20260809/stratified_4x150_r2
```

四个数据集各抽取 $150$ 条，共 $600$ 条；source、原始媒体和既有 FullCoT/VQA 均未改写。每个数据集的
selection receipt 都保留 strata shortfall，因此下表不表示所有 indoor、high-speed 或 dense-crowd 条件都被充分覆盖。

### 7.1 第一轮 geometry

| 数据集 | selected | accepted | error | rejected |
| --- | ---: | ---: | ---: | ---: |
| SideWalk main | 150 | 137 | 13 | 0 |
| SideWalk extra | 150 | 143 | 7 | 0 |
| Foxglove | 150 | 122 | 26 | 2 |
| EgoWalk | 150 | 140 | 9 | 1 |
| 合计 | 600 | 542 | 55 | 3 |

第一轮接受率为 $542 / 600 = 90.33\%$。主要失败不是场景语义拒绝，而是严格四边形合同拒绝：
凸性/自交 $87$ 次、面积过小 $19$ 次、近边未低于远边 $10$ 次。`ontology_literal_error` 共 $6$ 次，主要来自
模型输出 `road` 而非受控 surface type。

### 7.2 V1.2 受控 repair

V1.2 将 `road` 解析为受控的 `road_edge`，仅对第一轮的 $58$ 条非 accepted row 构造 source-only repair view，
不重跑既有 accepted row，也不对不合格四边形做坐标修补。repair 结果如下：

| 数据集 | repair 输入 | repair accepted | 残余 error | 残余 rejected |
| --- | ---: | ---: | ---: | ---: |
| SideWalk main | 13 | 8 | 5 | 0 |
| SideWalk extra | 7 | 4 | 3 | 0 |
| Foxglove | 28 | 17 | 9 | 2 |
| EgoWalk | 10 | 5 | 5 | 0 |
| 合计 | 58 | 34 | 22 | 2 |

该 repair 消除了 `road` literal 问题，但剩余 $24$ 条仍属于真实几何或 surface-role 合同不满足，不能自动放宽
validator 或转换为训练正样本。若将 repair accepted 仅作为候选 sidecar 计入审计，累计可解析候选为
$542 + 34 = 576$ 条；在人工 gate 前，这不是正式 accepted view。

### 7.3 suitability 与 Stage eligibility

本 smoke 的 canonical Reasoning source 没有绑定同一 parent 的 `StageIITrajectorySidecarV2`，因此每条都正确带有
`trajectory_missing_or_dt_unavailable`，并且 `stage_ii_eligible=0`。这表示当前 run **没有 Stage II 运动证据**，
不是 LVLM 对 $600$ 条画面的统一否定。

| 数据集 | Stage I / VQA eligible | scene rejected | Stage II eligible |
| --- | ---: | ---: | ---: |
| SideWalk main | 146 | 4 | 0 |
| SideWalk extra | 149 | 1 | 0 |
| Foxglove | 146 | 4 | 0 |
| EgoWalk | 81 | 69 | 0 |
| 合计 | 522 | 78 | 0 |

EgoWalk 的 $69$ 条 scene rejection 主要是 `indoor_other`，其次是 indoor transit、narrow corridor 或画面不适用；
其余三个数据集的 scene rejection 合计 $9$ 条。`quarantine` 的 $522$ 条仍保留 Stage I decision/VQA eligibility，
仅因为缺少 V2 sidecar 而禁止 Stage II；不得将该 disposition 写入原始 dataset quality 标签。

### 7.4 当前人工 gate

以下是原始 smoke gate。2026-08-09 用户已人工确认当前 geometry/suitability 质量足以进入**全量
标注和过滤**；该批准不等价于把结果接入 Stage I/II compiler。全量运行仍必须保留下列 audit 约束：

1. 对四个 geometry review 分别抽查 direct、conditional、blocked、forbidden，以及 repair 后的 accepted polygon；
   重点检查是否把车道、草地、车辆、楼梯、密集行人或图片底部大矩形误作 direct corridor。
2. 对 geometry 的 $24$ 个 repair 残余样本确认：继续保留 error/rejected 是否合理；不能通过降低凸性、面积或 near/far
   几何阈值来批量放行。
3. 检查 $78$ 条 scene rejection，尤其 EgoWalk indoor 类是否符合目标数据域；确认误杀率后才冻结
   `scene_not_in_outdoor_pedestrian_core` 策略。
4. 为每个可用于 Stage II 的 source 绑定精确的 `StageIITrajectorySidecarV2`，重新运行 motion/QC gate；当前
   `stage_ii_eligible=0` 的 smoke 不能用于 Stage II 样本量或分布结论。
5. 完成各数据集的 source-visible strata 覆盖审计。当前 Foxglove/SideWalk 的 indoor、high-speed、dense-crowd
   metadata shortfall 已被记录，不能用本次 $600$ 条声称覆盖这些场景。

完整自包含 review 与 repair review 保留在上述 run root 的各 `review/*.html` 和
`geometry_repair_v1_2/*/geometry_repair_review.html`。运行汇总为
`navigation_quality_run_report.json`；它明确记录 `source_manifests_unchanged=true` 和
`legacy_language_goal_manifests_unchanged=true`。

### 7.5 已批准的全量队列

全量 queue 固定为两个阶段：先由双 122B 服务生成可见语言目标，再启动双服务质量过滤。它使用独立
run root、append-only sidecar 和只读 derived view：

```text
language_waypoint/
  inputs/sidecars_by_dataset/       # 从完整 V2 train/validation/test ledger 去重得到
  datasets/<dataset>/lanes/         # stable parent hash 分配的双服务 append-only shard
  datasets/<dataset>/merged/        # 保持 candidate 顺序的合并 annotation/decision ledger
navigation_quality/
  <dataset>/geometry/               # LocalTraversabilityGeometryV1 sidecar
  <dataset>/suitability/            # NavigationTrainabilityFilterV1 sidecar
  <dataset>/views/                  # source-only accepted/quarantine/rejected/pending views
```

入口脚本是：

```bash
tools/model_processing/run_language_waypoint_then_navigation_quality_full_122b.sh
```

它强制要求冒号分隔的 `TRAJECTORY_SIDECAR_JSONLS` 指向完整、typed
`StageIITrajectorySidecarV2` `$4\mathrm{s}/32$` train/validation/test ledger。旧 `$2\mathrm{s}/16$`
trajectory 文件会在 materialization parser 阶段失败，不能成为 language target 或 Stage II motion QC。
默认 language 为每服务 $8$ 个 in-flight request；quality 阶段为每服务 $16$ 个 in-flight request
（四数据集各 $4$）。服务端可通过环境变量下调，不应靠改动 source/sidecar 来恢复。
