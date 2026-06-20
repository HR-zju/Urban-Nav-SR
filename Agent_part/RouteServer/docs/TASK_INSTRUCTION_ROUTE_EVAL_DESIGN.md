# Task Instruction and Route Evaluation Design

本文档定义 UrbanNav benchmark 的核心评估设计。目标是指标少、边界清楚、可复现。

我们不评估 CitySeeker/SpotAgent 那种“两个离散图像节点之间选路”的任务。CitySeeker 只用于参考需求类型和数据构造思想；SpotAgent 只用于参考“结构化输出 + 分层评分”的 benchmark 设计。

## 1. 评估拆分

UrbanNav 评估分成三层：

```text
Instruction Parsing -> Destination Recommendation -> Route Planning
```

另有一个扩展层：

```text
Route Update / Add Via Point
```

每层只回答一个问题：

| 层 | 问题 | 输入 | 输出 |
| --- | --- | --- | --- |
| 指令解析 | 是否理解用户需求 | instruction | category / subcategory / slots |
| 目的地推荐 | 是否推荐了合理目的地 | instruction + origin | chosen destination + candidates |
| 路线规划 | 是否到达且不绕路 | origin + destination | route polyline |
| 中途加点 | 是否正确修改已有路线 | route session + update instruction | added via + updated route |

## 2. 任务类型

当前主任务类型：

| Category | 中文 | 是否进入主 benchmark |
| --- | --- | --- |
| `poi_navigation` | POI/品牌导航 | 是 |
| `implicit_need_navigation` | 隐式需求导航 | 是 |
| `multi_stop_route` | 多目标导航 | 是，单独报告 |
| `route_update_add_via` | 中途增加途经点 | 是，单独报告 |
| `semantic_preference` | 语义偏好导航 | 暂作为分析集 |

`semantic_preference` 依赖 Yelp/评论/人工标签，真值噪声比普通 POI 高，不建议第一版放进主排行榜。

## 3. 指令解析 Benchmark

### 3.1 输出格式

解析器或模型输出统一 JSON：

```json
{
  "category": "implicit_need_navigation",
  "subcategory": "phone_battery_or_recharge",
  "target_types": ["cafe", "library", "coworking_space"],
  "brand_aliases": [],
  "semantic_tags": ["charging", "seating"],
  "ordered_goals": [],
  "route_update": null
}
```

多目标：

```json
{
  "category": "multi_stop_route",
  "subcategory": "coffee_then_work",
  "ordered_goals": [
    {"target_types": ["cafe", "coffee_shop"]},
    {"target_types": ["library", "coworking_space", "cafe"], "semantic_tags": ["wifi", "quiet"]}
  ]
}
```

中途加点：

```json
{
  "category": "route_update_add_via",
  "subcategory": "add_atm_stop",
  "target_types": ["atm", "bank"],
  "route_update": {"operation": "add_via"}
}
```

### 3.2 核心指标

只保留 4 个核心指标：

| 指标 | 定义 |
| --- | --- |
| `parse_valid_rate` | 输出能否解析为合法 JSON |
| `category_acc` | 大类是否正确 |
| `subcategory_acc` | 子类是否正确 |
| `slot_f1` | `target_types / brand_aliases / semantic_tags / ordered_goals / route_update` 的集合或序列 F1 |

主分数：

\[
instruction\_score = 0.4 \cdot category\_acc + 0.3 \cdot subcategory\_acc + 0.3 \cdot slot\_f1
\]

说明：

- `parse_valid_rate` 单独报告，不并入主分数；格式错误的样本其余得分为 0。
- `poi_navigation` 重点看 `target_types` 和 `brand_aliases`。
- `implicit_need_navigation` 重点看 `target_types`。
- `multi_stop_route` 重点看 `ordered_goals` 顺序。
- `route_update_add_via` 重点看 `route_update.operation` 和新增 `target_types`。

## 4. 目的地推荐 Benchmark

### 4.1 真值格式

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
  ]
}
```

`acceptable_destinations` 是必要字段：隐式需求通常有多个合理答案，不能只用单点硬判。

### 4.2 核心指标

只保留 2 个核心指标：

| 指标 | 定义 |
| --- | --- |
| `destination_success@50m` | Top-1 目的地距离任一可接受 GT <= 50m，或 provider id 命中 |
| `hit@5` | Top-5 候选中是否包含任一可接受 GT |

`destination_success@50m` 评估最终推荐质量；`hit@5` 区分候选检索失败还是 rerank/选择失败。

## 5. 路线规划 Benchmark

### 5.1 真值格式

```json
{
  "id": "case-001",
  "origin": {"lat": 30.2645, "lng": 120.1180},
  "gt_destination": {"lat": 30.2651, "lng": 120.1190},
  "gt_route": [[120.1180, 30.2645], [120.1190, 30.2651]],
  "gt_distance_m": 135.0
}
```

### 5.2 核心指标

只保留 3 个核心指标：

| 指标 | 定义 |
| --- | --- |
| `route_success@50m` | 目的地命中且返回非空路线 |
| `median_route_ratio` | 成功样本中 `pred_route_distance / gt_route_distance` 的中位数 |
| `spl` | `route_success * gt_distance / max(gt_distance, pred_distance)` |

这三个指标足够回答：

```text
有没有到达？
到达路线是否绕路？
综合表现如何？
```

`curve_ade / curve_mde / step_ratio / turn_edit_distance` 只作为 debug 指标，不进入主表。

## 6. 多目标导航 Benchmark

多目标不要混进单目标 route success，单独报告。

真值：

```json
{
  "id": "multi-001",
  "instruction": "先找个地方吃饭，然后去买点日用品。",
  "origin": {"lat": 30.2645, "lng": 120.1180},
  "gt_destination_sequence": [
    {"type": "restaurant", "lat": 30.2650, "lng": 120.1188},
    {"type": "convenience_store", "lat": 30.2660, "lng": 120.1195}
  ],
  "gt_route": [[120.1180, 30.2645], [120.1188, 30.2650], [120.1195, 30.2660]],
  "gt_distance_m": 230.0
}
```

核心指标：

| 指标 | 定义 |
| --- | --- |
| `stop_sequence_success@50m` | 所有目标按顺序命中 |
| `stop_recall` | 命中的目标数 / GT 目标数 |
| `multi_stop_spl` | 多段路线的 SPL |

## 7. 中途增加途经点 Benchmark

该任务评估 route session 的增量修改，不是从零规划。

真值：

```json
{
  "id": "via-001",
  "old_route": [[120.0, 30.0], [120.3, 30.3]],
  "current_position": {"lat": 30.05, "lng": 120.05},
  "final_destination": {"lat": 30.3, "lng": 120.3},
  "instruction": "中途加一个 ATM。",
  "gt_added_via": {"type": "atm", "lat": 30.1, "lng": 120.1},
  "gt_updated_route": [[120.05, 30.05], [120.1, 30.1], [120.3, 30.3]],
  "gt_updated_distance_m": 260.0
}
```

核心指标：

| 指标 | 定义 |
| --- | --- |
| `via_success@50m` | 新增途经点是否命中 |
| `final_destination_preserved` | 原终点是否仍保留 |
| `updated_route_spl` | 更新路线是否到达且不绕路 |

## 8. 最终主表

主 benchmark 表只放这些指标：

| 层 | 主指标 |
| --- | --- |
| 指令解析 | `parse_valid_rate`, `category_acc`, `subcategory_acc`, `slot_f1`, `instruction_score` |
| 目的地推荐 | `destination_success@50m`, `hit@5` |
| 单目标路线 | `route_success@50m`, `median_route_ratio`, `spl` |
| 多目标路线 | `stop_sequence_success@50m`, `stop_recall`, `multi_stop_spl` |
| 中途加点 | `via_success@50m`, `final_destination_preserved`, `updated_route_spl` |

报告维度只保留：

```text
overall
by_category
by_subcategory
by_city_or_scene
```

## 9. 真值制作原则

所有真值必须预先固化，评估时不动态生成。

目的地真值：

```text
origin + task type + POI candidates
-> filter satisfying POIs
-> choose nearest by route distance
-> save gt_destination + acceptable_destinations
```

路线真值：

```text
origin + gt_destination
-> route provider
-> save gt_route + gt_distance_m
```

多目标真值：

```text
origin + ordered goals
-> choose each stop
-> route through stops
-> save gt_destination_sequence + gt_route + gt_distance_m
```

中途加点真值：

```text
old session + current position + added via + original final destination
-> updated route
-> save gt_added_via + gt_updated_route + gt_updated_distance_m
```

如果找不到满足条件的 POI 或路线，样本应丢弃或进入 reject，不应强行制造 GT。

## 10. 实现优先级

1. 保持现有 `eval_destination_route.py`，但把文档主格式切到 UrbanNav JSONL。
2. 新增 `eval_instruction_parse.py`，只实现 4 个解析指标。
3. 扩展 destination evaluator 支持 `acceptable_destinations`。
4. 新增 `eval_multi_stop_route.py` 和 `eval_route_update.py`，只实现上面的核心指标。
