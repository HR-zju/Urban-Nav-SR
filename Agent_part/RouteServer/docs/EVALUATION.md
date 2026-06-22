# Destination / Route Evaluation

本文档定义 UrbanNav 当前核心离线评估口径。我们不再把 CitySeeker 的 `from/to/complete_route` 离散图像节点格式作为主格式；CitySeeker 和 SpotAgent 只作为设计参考。

更完整的分层 benchmark 设计见 [TASK_INSTRUCTION_ROUTE_EVAL_DESIGN.md](/home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer/docs/TASK_INSTRUCTION_ROUTE_EVAL_DESIGN.md)。

本评估框架参考 MobilityBench 的可复现 sandbox 思想：地图 API 查询、POI 候选和路线响应必须在
GT 构建阶段冻结，评估阶段只读 `ground-truth`、`predictions` 和可选 replay cache，不访问
live map API。这样同一批 case 在不同模型、不同时间运行时仍可比较。

## 1. 评估入口

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

python tools/evaluation/eval_destination_route.py \
  --ground-truth /path/to/gt.jsonl \
  --predictions /path/to/pred.jsonl \
  --destination-radius-m 30 \
  --output-json /tmp/urbannav_eval_summary.json \
  --output-jsonl /tmp/urbannav_eval_rows.jsonl
```

评估脚本只比较已经保存好的预测和真值，不调用 LLM，不调用地图 API。
`--destination-radius-m` 是坐标兜底匹配容差，不是主指标语义。优先使用
`provider_poi_id/place_id` 判断 POI 身份；没有稳定 ID 时才使用名称、类型和坐标容差。

GT 和 prediction 通过 `id / case_id / sample_id / qid / question_id` 对齐；正式数据集必须统一使用
`id`。无法对齐的预测不进入主指标，并应写入 per-case error。

## 2. Ground Truth 格式

主格式是 UrbanNav JSONL，每行一个任务：

```json
{
  "id": "case-001",
  "category": "implicit_need_navigation",
  "subcategory": "thirsty",
  "instruction": "我有点渴了，帮我找附近合适的地方。",
  "origin": {"lat": 30.2645, "lng": 120.1180},
  "gt_destination": {
    "name": "便利店 A",
    "type": "convenience_store",
    "lat": 30.2651,
    "lng": 120.1190,
    "provider_poi_id": "amap_xxx",
    "reference_route_distance_m": 135.0
  },
  "acceptable_destinations": [
    {
      "name": "便利店 A",
      "type": "convenience_store",
      "lat": 30.2651,
      "lng": 120.1190,
      "provider_poi_id": "amap_xxx",
      "reference_route_distance_m": 135.0,
      "reference_route": [[120.1180, 30.2645], [120.1190, 30.2651]],
      "rank": 1
    }
  ],
  "gt_route": [[120.1180, 30.2645], [120.1190, 30.2651]],
  "gt_distance_m": 135.0,
  "ranked_destinations": [
    {
      "rank": 1,
      "provider_poi_id": "amap_xxx",
      "route_distance_m": 135.0,
      "score_breakdown": {"relevance": 1.0, "route_distance_m": 135.0}
    }
  ],
  "optimality_scope": {
    "poi_provider": "amap",
    "search_radius_m": 1500,
    "candidate_limit": 30,
    "route_provider": "amap",
    "route_mode": "foot",
    "route_strategy": "default",
    "cost": "route_distance_m",
    "builder_version": "urbannav_gt_v1"
  },
  "cache_keys": {
    "poi_search": ["poi:case-001:thirsty"],
    "routes": ["route:case-001:amap_xxx"]
  }
}
```

说明：

- `gt_destination` 是 rank-1 真值目的地，即固定候选召回范围内路线代价最小的 satisfying POI。
- `acceptable_destinations` 用于隐式需求和语义偏好，因为这类任务可能有多个合理答案；每个 acceptable destination 应保存自身参考路线距离和参考路线。
- `gt_route` 是从 `origin` 到 `gt_destination` 的参考路线 polyline，坐标顺序为 `[lng, lat]`。
- `gt_distance_m` 是参考路线长度；没有路线时可以为空，但路线指标会跳过该样本。
- `ranked_destinations` 是所有成功召回且成功算路候选的排序证据；`gt_destination` 必须等于 rank-1。
- `optimality_scope` 记录“最优”成立的 provider、搜索半径、候选上限、路线模式、路线策略、代价函数和 builder 版本。
- `cache_keys` 指向冻结的 POI/route replay cache，用于审计，不参与评分。

## 3. Prediction 格式

RouteServer 输出保存后可作为 prediction：

```json
{
  "id": "case-001",
  "chosen_destination": {
    "id": "poi-123",
    "name": "便利店 A",
    "latitude": 30.2651,
    "longitude": 120.1190,
    "location": "120.1190,30.2651"
  },
  "candidate_destinations": [
    {"name": "便利店 A", "latitude": 30.2651, "longitude": 120.1190}
  ],
  "route_polyline": [[120.1180, 30.2645], [120.1190, 30.2651]],
  "route_summary": {
    "distance_m": 142.0
  },
  "trace": {
    "tool_calls": [],
    "latency_ms": 0,
    "token_usage": {}
  }
}
```

`trace` 可用于工具调用、延迟、token 等诊断；主目的地和路线指标不要求工具调用序列与 GT 完全一致。

## 4. 核心指标

### 4.1 Destination Success

目的地是否推荐正确：

\[
DS = \mathbb{1}[\hat d \text{ matches any } d^* \in G]
\]

其中：

- `G` 是 `acceptable_destinations`，如果为空则使用 `gt_destination`。
- 优先用 `provider_poi_id/place_id` 匹配 POI 身份。
- 没有稳定 ID 时，使用 `name + type + coordinate_tolerance_m` 兜底。
- 坐标容差只处理 provider 坐标偏移、入口点和路网吸附差异，不代表“半径内任意目标都成功”。

主报告指标：

```text
destination_success
```

### 4.2 Destination Ranking

如果系统返回候选列表，评估 GT 是否出现在 Top-K：

```text
hit@1
hit@5
```

`hit@1` 等价于首选目的地命中；`hit@5` 衡量检索/候选生成是否把正确目标找出来。

### 4.3 Route Success

路线是否到达正确目的地：

\[
RS = DS \land route\_polyline \neq \emptyset \land endpoint(\hat P) \text{ reaches matched destination}
\]

主报告指标：

```text
route_success
```

### 4.4 Route Efficiency

路线效率用预测路线长度与参考路线长度的比例：

\[
route\_ratio = \frac{L(\hat P)}{L(P^*_{\mathrm{matched}})}
\]

如果预测命中的是非 rank-1 但仍可接受的目的地，分母使用该 matched destination 的冻结参考路线，
而不是 rank-1 `gt_destination` 的路线。只在 `route_success = true` 的样本上汇总。推荐报告：

```text
median_route_ratio
```

相比平均值，中位数对极端绕路样本更鲁棒。

### 4.5 SPL

保留一个综合指标：

\[
SPL = RS \cdot \frac{L(P^*_{\mathrm{matched}})}{\max(L(P^*_{\mathrm{matched}}), L(\hat P))}
\]

它同时惩罚不到达和绕路。

## 5. 最小报告格式

第一版 benchmark 只需要固定报告这 5 个指标：

| 指标 | 作用 |
| --- | --- |
| `destination_success` | 是否选中任一合理目的地 |
| `hit@5` | 候选检索是否召回正确目标 |
| `route_success` | 是否成功规划到 matched 合理目的地 |
| `median_route_ratio` | 成功路线是否绕路 |
| `spl` | 到达率和效率综合分 |

按以下维度切分即可：

```text
overall
by_category
by_subcategory
by_city_or_scene
by_difficulty
```

## 6. 诊断指标

以下指标不作为主排行榜，只在 debug 时使用：

| 指标 | 用途 |
| --- | --- |
| `mean_destination_error_m` | 看错误目的地偏离多远 |
| `mrr` | 分析候选排序质量 |
| `curve_ade_m / curve_mde_m` | 比较路线形状是否贴近参考路线 |
| `step_ratio` | 有 step 级导航时分析转向/步数效率 |

如果主指标已经能解释系统差异，不需要在论文主表里堆这些指标。

## 7. Replay cache 和验证产物

正式 benchmark 目录建议包含：

```text
data/benchmark/<version>/
  task_gt.jsonl
  predictions/<model>.jsonl
  poi_cache.jsonl
  route_cache.jsonl
  task_rejects.jsonl
  validation_report.json
```

`validation_report.json` 至少检查：

```text
gt_destination 等于 ranked_destinations[0]
ranked_destinations 按 cost 升序或 score 降序
acceptable_destinations 都有 reference_route_distance_m 和 reference_route
gt_route 起点接近 origin
gt_route 终点接近 gt_destination
cache_keys 能在 replay cache 中找到
reject_reason 覆盖所有丢弃样本
```

评估脚本默认只需要 `task_gt.jsonl` 和 prediction JSONL；replay cache 用于复查和调试，不应在评分时动态改写结果。

## 8. Runner smoke

当前 runner 入口：

```bash
PYTHONPATH=src python tools/evaluation/run_urbannav_predictions.py \
  --cases data/benchmark/<version>/task_gt.jsonl \
  --runner routeserver-replay \
  --poi-cache data/benchmark/<version>/poi_cache.jsonl \
  --route-cache data/benchmark/<version>/route_cache.jsonl \
  --output-jsonl data/benchmark/<version>/predictions/routeserver_replay.jsonl
```

`routeserver-replay` 使用冻结 cache 跑 RouteServer-style 流程：

\[
q \rightarrow \hat I \rightarrow \hat C \rightarrow \{\hat R_i\} \rightarrow \hat d
\]

因此它适合 smoke benchmark 和回归测试。真实 provider 预测使用：

```bash
PYTHONPATH=src python tools/evaluation/run_urbannav_predictions.py \
  --cases data/benchmark/<version>/task_gt.jsonl \
  --runner routeserver-live \
  --map-provider amap \
  --output-jsonl data/benchmark/<version>/predictions/routeserver_live.jsonl
```

多可接受目的地样例仍由 `acceptable_destinations` 评分；route-update 样例作为
`category=route_update` 的普通 OD 任务评分，更新事件保存在 `metadata.route_update`。

## Multi-Stop Review

`acceptable_destinations` 和 `multi_stop_route` 不表示同一件事：

- `acceptable_destinations` 是单阶段任务里的多选一合理答案，例如“咖啡店、书店、图书馆都可以”。
- `multi_stop_route` 是一个指令中要求按顺序经过多个途经点，例如“先吃饭，再取钱，最后休息”。

Multi-stop GT 构造采用逐段最优，而不是一次性只选最终点。对于 ordered goals
\(g_1,\ldots,g_K\)，每一段从当前点 \(p_{i-1}\) 搜索并规划：

\[
d_i^\* = \arg\min_{d \in D_i^{routeable}} c(p_{i-1}, d)
\]

完整路线是各段路线拼接：

\[
R = R(p_0,d_1^\*) \oplus R(d_1^\*,d_2^\*) \oplus \cdots \oplus R(d_{K-1}^\*,d_K^\*)
\]

真实 API review 构建入口：

```bash
PYTHONPATH=src python tools/dataset_processing/build_urbannav_multistop_review.py \
  --output-root /tmp/urbannav_yuquan_multistop_review \
  --origin-json tests/fixtures/urbannav_yuquan_origin.jsonl \
  --tasks-json tests/fixtures/urbannav_yuquan_multistop_tasks.jsonl \
  --map-provider amap \
  --num-cases 5
```

该脚本输出 `gt_destination_sequence` 和 `gt_route_legs`。HTML review 会在地图上标出
`S, 1, 2, ..., D`，并列出每一段的候选、rank-1 选择、near-tie alternatives 和
route distance。当前 multi-stop 先作为人工 review/分析集，不混入单目标
`destination_success / route_success / spl` 主表。

Multi-stop prediction 已接入 `RouteServerUrbanNavRunner`。runner 会先调用
`GoalDecomposer` 生成 ordered goals，再对每个 goal 从当前点搜索和规划：

```bash
PYTHONPATH=src python tools/evaluation/run_urbannav_predictions.py \
  --cases /tmp/urbannav_yuquan_multistop_review/task_gt.jsonl \
  --runner routeserver-replay \
  --poi-cache /tmp/urbannav_yuquan_multistop_review/poi_cache.jsonl \
  --route-cache /tmp/urbannav_yuquan_multistop_review/route_cache.jsonl \
  --output-jsonl /tmp/urbannav_yuquan_multistop_review/routeserver_replay_predictions.jsonl \
  --mode foot
```

输出 prediction 包含：

- `pred_destination_sequence`
- `pred_route_legs`
- 拼接后的 `route_polyline`

HTML review 传入 prediction 后会生成 GT/Pred 双图：

```bash
PYTHONPATH=src python tools/evaluation/build_urbannav_review_html.py \
  --ground-truth /tmp/urbannav_yuquan_multistop_review/task_gt.jsonl \
  --predictions /tmp/urbannav_yuquan_multistop_review/routeserver_replay_predictions.jsonl \
  --output-dir /tmp/urbannav_yuquan_multistop_review/html_with_pred \
  --download-staticmaps
```

## Step 10 扩展任务指标

Step 10 的扩展任务不替换单目标五个主指标，而是在 per-case row 的
`extended_metrics` 和 summary 中额外报告。

Multi-stop 指标：

\[
stop\_recall = \frac{1}{K}\sum_{i=1}^{K}
\mathbb{1}[\operatorname{match}(\hat d_i, G_i)]
\]

\[
sequence\_success =
\mathbb{1}[\forall i,\operatorname{match}(\hat d_i,G_i)]
\]

\[
multi\_stop\_SPL =
sequence\_success \cdot leg\_route\_success \cdot
\frac{L(R^\*)}{\max(L(R^\*), L(\hat R))}
\]

报告字段：

- `multi_stop_sequence_success`
- `multi_stop_stop_recall`
- `multi_stop_leg_route_success`
- `multi_stop_leg_route_success_rate`
- `multi_stop_spl`

Route-update 指标：

\[
route\_update\_success =
updated\_route\_success \land current\_origin\_used \land previous\_destination\_avoided
\]

报告字段：

- `route_update_success`
- `route_update_destination_success`
- `route_update_route_success`
- `route_update_current_origin_used`
- `route_update_previous_destination_avoided`

这些字段由现有评估入口自动输出：

```bash
PYTHONPATH=src python tools/evaluation/eval_urbannav_destination_route.py \
  --ground-truth /path/to/task_gt.jsonl \
  --predictions /path/to/predictions.jsonl \
  --output-json /tmp/urbannav_eval_summary.json \
  --output-jsonl /tmp/urbannav_eval_rows.jsonl
```
