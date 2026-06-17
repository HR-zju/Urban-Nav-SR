# Destination / Route Evaluation

本文档定义当前 RouteServer 侧目的地推荐和路线规划的离线评估口径。设计参考 CitySeeker：

- CitySeeker ground truth 记录 `from/to/complete_route/total_weight/total_steps`。
- CitySeeker simulation 记录 `flag/round_success/total_weight/total_steps`。
- 本仓库把这个思想拆成两层：目的地是否推荐对、路线是否可用且代价是否接近 GT。

## 1. 评估入口

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

python tools/evaluation/eval_destination_route.py \
  --ground-truth /path/to/gt.json \
  --predictions /path/to/pred.jsonl \
  --destination-radius-m 50 \
  --output-json /tmp/urbannav_eval_summary.json \
  --output-jsonl /tmp/urbannav_eval_rows.jsonl
```

该脚本不调用地图 API、不调用 LLM，只比较已经保存好的预测结果。

## 2. Ground Truth 格式

兼容 CitySeeker 风格 JSON：

```json
{
  "question_idx": 0,
  "idx": 2,
  "question": "Please find the nearest convenience store.",
  "from": "00001864.jpg",
  "to": "00002415.jpg",
  "total_weight": 131.16,
  "total_steps": 10,
  "complete_route": [
    {"filename": "00001864.jpg", "latitude": 39.91933865, "longitude": 116.3535388},
    {"filename": "00002415.jpg", "latitude": 39.91843821, "longitude": 116.3535685}
  ]
}
```

也支持 UrbanNav 风格 JSONL：

```json
{
  "id": "case-001",
  "question": "我想找最近的便利店",
  "origin": {"lat": 30.2645, "lng": 120.1180},
  "gt_destination": {"lat": 30.2651, "lng": 120.1190, "name": "便利店"},
  "gt_route": [[120.1180, 30.2645], [120.1190, 30.2651]],
  "total_weight": 135.0,
  "total_steps": 8
}
```

## 3. Prediction 格式

RouteServer 输出保存后可直接作为 prediction：

```json
{
  "id": "case-001",
  "chosen_destination": {
    "id": "poi-123",
    "name": "便利店",
    "latitude": 30.2651,
    "longitude": 120.1190,
    "location": "120.1190,30.2651"
  },
  "candidate_destinations": [
    {"name": "便利店", "latitude": 30.2651, "longitude": 120.1190}
  ],
  "route_polyline": [[120.1180, 30.2645], [120.1190, 30.2651]],
  "route_summary": {
    "distance_m": 142.0,
    "step_count": 8
  }
}
```

case id 匹配优先级：

```text
id / case_id / sample_id / qid / question_id
question_idx + idx + round
question_idx + idx
question + from
```

## 4. 目的地推荐指标

目的地命中：

\[
success_{dest} =
\mathbb{1}[d(\hat d, d^*) \le r]
\]

默认：

```text
r = 50m
```

同时支持 `id` 或 `name` 完全一致直接命中。

输出指标：

| 指标 | 含义 |
| --- | --- |
| `destination.success_rate` | 预测目的地在半径内的比例 |
| `destination.mean_error_m` | 预测目的地到 GT 目的地的平均球面距离 |
| `destination.median_error_m` | 目的地距离误差中位数 |
| `destination.hit_at_k` | GT 目的地是否出现在 Top-K 候选中 |
| `destination.mrr` | 候选排序的 Mean Reciprocal Rank |

## 5. 路线指标

路线成功：

\[
success_{route} =
success_{dest} \land |\hat P| > 0
\]

路线长度比：

\[
ratio_{dist} =
\frac{L(\hat P)}{L(P^*)}
\]

步数比：

\[
ratio_{steps} =
\frac{\hat N}{N^*}
\]

SPL 风格效率：

\[
SPL =
success_{route}
\cdot
\frac{L(P^*)}{\max(L(P^*), L(\hat P))}
\]

曲线误差会把 GT route 和 prediction route 按弦长重采样到同样点数：

\[
ADE =
\frac{1}{K}\sum_i d(\hat p_i,p_i^*)
\]

\[
FDE = d(\hat p_K,p_K^*)
\]

\[
MDE = \max_i d(\hat p_i,p_i^*)
\]

输出指标：

| 指标 | 含义 |
| --- | --- |
| `route.success_rate` | 目的地命中且有 route polyline 的比例 |
| `route.mean_endpoint_error_m` | 预测路线终点到 GT 目的地的平均距离 |
| `route.mean_distance_ratio` | 预测路线长度 / GT 路线长度 |
| `route.mean_step_ratio` | 预测步数 / GT 步数 |
| `route.spl` | 成功率和路径效率合成指标 |
| `route.curve_ade_m` | 路线曲线平均误差 |
| `route.curve_fde_m` | 路线终点误差 |
| `route.curve_mde_m` | 路线最大偏差 |

## 6. 和 CitySeeker 的对应关系

| CitySeeker 字段/指标 | 本仓库对应 |
| --- | --- |
| `complete_route[-1]` | `gt_destination` |
| `total_weight` | `gt_distance_m` |
| `total_steps` | `gt_steps` |
| `flag` / `round_success` | `route_success` / `destination_success` |
| `sum(distance_container)` | `pred_distance_m` |
| `shortestPath steps` | 可选 `gt_steps` / `pred_steps` |
| topological distance progress | 后续可接 RouteSession runtime logs |

当前脚本先覆盖离线推荐/路线结果；如果后续要评估在线逐步导航，可以在每步日志里加入：

```text
current_distance_to_destination_m
current_route_progress_m
topological_distance_to_goal
action_heading_error_deg
```

再复用 CitySeeker 的 progress/backtracking 指标。

