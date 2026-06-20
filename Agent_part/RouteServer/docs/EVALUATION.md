# Destination / Route Evaluation

本文档定义 UrbanNav 当前核心离线评估口径。我们不再把 CitySeeker 的 `from/to/complete_route` 离散图像节点格式作为主格式；CitySeeker 和 SpotAgent 只作为设计参考。

更完整的分层 benchmark 设计见 [TASK_INSTRUCTION_ROUTE_EVAL_DESIGN.md](/home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer/docs/TASK_INSTRUCTION_ROUTE_EVAL_DESIGN.md)。

## 1. 评估入口

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

python tools/evaluation/eval_destination_route.py \
  --ground-truth /path/to/gt.jsonl \
  --predictions /path/to/pred.jsonl \
  --destination-radius-m 50 \
  --output-json /tmp/urbannav_eval_summary.json \
  --output-jsonl /tmp/urbannav_eval_rows.jsonl
```

评估脚本只比较已经保存好的预测和真值，不调用 LLM，不调用地图 API。

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
    "provider_poi_id": "amap_xxx"
  },
  "acceptable_destinations": [
    {"lat": 30.2651, "lng": 120.1190, "provider_poi_id": "amap_xxx"}
  ],
  "gt_route": [[120.1180, 30.2645], [120.1190, 30.2651]],
  "gt_distance_m": 135.0
}
```

说明：

- `gt_destination` 是主真值目的地。
- `acceptable_destinations` 用于隐式需求和语义偏好，因为这类任务可能有多个合理答案。
- `gt_route` 是从 `origin` 到 `gt_destination` 的参考路线 polyline，坐标顺序为 `[lng, lat]`。
- `gt_distance_m` 是参考路线长度；没有路线时可以为空，但路线指标会跳过该样本。

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
  }
}
```

case 通过 `id / case_id / sample_id / qid / question_id` 匹配。新数据集应统一使用 `id`。

## 4. 核心指标

### 4.1 Destination Success

目的地是否推荐正确：

\[
DS@r = \mathbb{1}[\min_{d^* \in G} dist(\hat d, d^*) \le r]
\]

其中：

- `G` 是 `acceptable_destinations`，如果为空则使用 `gt_destination`。
- 默认 `r = 50m`。
- 如果 `provider_poi_id` 完全一致，可直接视为命中。

主报告指标：

```text
destination_success@50m
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
RS@r = DS@r \land route\_polyline \neq \emptyset
\]

主报告指标：

```text
route_success@50m
```

### 4.4 Route Efficiency

路线效率用预测路线长度与参考路线长度的比例：

\[
route\_ratio = \frac{L(\hat P)}{L(P^*)}
\]

只在 `route_success@50m = true` 的样本上汇总。推荐报告：

```text
median_route_ratio
```

相比平均值，中位数对极端绕路样本更鲁棒。

### 4.5 SPL

保留一个综合指标：

\[
SPL = RS \cdot \frac{L(P^*)}{\max(L(P^*), L(\hat P))}
\]

它同时惩罚不到达和绕路。

## 5. 最小报告格式

第一版 benchmark 只需要固定报告这 5 个指标：

| 指标 | 作用 |
| --- | --- |
| `destination_success@50m` | 目的地推荐是否正确 |
| `hit@5` | 候选检索是否召回正确目标 |
| `route_success@50m` | 是否规划到正确目的地 |
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
