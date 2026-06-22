# Task Instruction and Route Evaluation Design

本文档定义 UrbanNav benchmark 的核心评估设计。目标是指标少、边界清楚、可复现。

我们不评估 CitySeeker/SpotAgent 那种“两个离散图像节点之间选路”的任务。CitySeeker 只用于参考需求类型和数据构造思想；SpotAgent 只用于参考“结构化输出 + 分层评分”的 benchmark 设计。

UrbanNav 第一版是基于地图 API 的 OD benchmark，不要求构建 CitySeeker 式
city graph。也就是说，我们不需要为主评估维护街景节点、可走边、方位角和逐步动作图；
真值路线由固定 POI provider + route provider 离线生成并冻结。只有后续评估逐步视觉动作
或 embodied graph navigation 时，才需要引入 CitySeeker 式拓扑图。

MobilityBench 对本设计的主要启发是工程闭环，而不是任务语义：它把地图 API
结果冻结成 sandbox/replay 数据，把 agent 运行轨迹保存成 case 级输出，再由独立
metrics 按 `case_id` 对齐评估。UrbanNav 应采用同样的可复现框架，但主评估对象仍是
“需求指令 -> 合理目的地 -> 到该目的地的导航路径”，不能退化成通用地图问答或工具调用匹配。

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
  ]
}
```

`acceptable_destinations` 是必要字段：隐式需求通常有多个合理答案，不能只用单点硬判。
它必须是离线生成并排序后的可接受目标集合，而不是评估时动态查询。`gt_destination`
固定为满足约束且路线代价最小的 rank-1 目标；其它可接受目标需要保存各自的参考路线
距离和参考路线，供路线指标按 matched destination 计算。

### 4.2 核心指标

只保留 2 个核心指标：

| 指标 | 定义 |
| --- | --- |
| `destination_success` | Top-1 目的地命中任一可接受目标 |
| `hit@5` | Top-5 候选中是否包含任一可接受目标 |

目的地命中优先按 `provider_poi_id/place_id` 判断；没有稳定 ID 时再用
`name + type + coordinate_tolerance_m` 兜底。坐标容差只是不同 provider、POI 中心点、
入口点或路网吸附点之间的匹配容差，不是主指标语义，也不应写进指标名。
`hit@5` 区分候选检索失败还是 rerank/选择失败。

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
| `route_success` | 目的地命中、返回非空路线，且路线终点到达 matched acceptable destination |
| `median_route_ratio` | 成功样本中 `pred_route_distance / reference_route_distance_to_matched_destination` 的中位数 |
| `spl` | `route_success * reference_distance / max(reference_distance, pred_distance)` |

这三个指标足够回答：

```text
有没有到达？
到达路线是否绕路？
综合表现如何？
```

如果模型命中的是非 rank-1 但仍可接受的目标，路线指标使用该 matched destination 的
冻结参考路线作为分母，而不是强行和 rank-1 `gt_destination` 的路线比较。
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
| `stop_sequence_success` | 所有目标按顺序命中可接受 stop |
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
| `via_success` | 新增途经点是否命中可接受 via |
| `final_destination_preserved` | 原终点是否仍保留 |
| `updated_route_spl` | 更新路线是否到达且不绕路 |

## 8. 最终主表

主 benchmark 表只放这些指标：

| 层 | 主指标 |
| --- | --- |
| 指令解析 | `parse_valid_rate`, `category_acc`, `subcategory_acc`, `slot_f1`, `instruction_score` |
| 目的地推荐 | `destination_success`, `hit@5` |
| 单目标路线 | `route_success`, `median_route_ratio`, `spl` |
| 多目标路线 | `stop_sequence_success`, `stop_recall`, `multi_stop_spl` |
| 中途加点 | `via_success`, `final_destination_preserved`, `updated_route_spl` |

报告维度只保留：

```text
overall
by_category
by_subcategory
by_city_or_scene
```

## 9. Benchmark 工程框架

第一版 UrbanNav benchmark 采用离线构建、冻结回放、离线评估的结构：

```text
taxonomy + regions + provider config
-> GT builder
-> frozen cases + POI cache + route cache + audit
-> RouteServer / baseline agent run
-> predictions + trace
-> offline evaluator
-> summary + per-case rows
```

模块职责：

| 模块 | 作用 | 输入 | 输出 |
| --- | --- | --- | --- |
| Taxonomy | 定义任务类别、子类、模板、目标类型和约束 | `task_taxonomy.json` | category/subcategory/slots |
| Region Sampler | 提供城市区域和 origin 候选 | region bbox/grid/quota | origin candidates |
| GT Builder | 检索候选目的地、全候选算路、排序和过滤 | taxonomy + origins + map providers | `task_gt.jsonl`, `task_rejects.jsonl` |
| Replay Cache | 冻结 POI/route API 响应 | provider request/response | `poi_cache.jsonl`, `route_cache.jsonl` |
| Validator | 检查 GT 最优性和格式合法性 | GT + cache + rejects | `validation_report.json` |
| Runner | 运行 RouteServer 或 baseline agent | cases | predictions + trace |
| Evaluator | 对齐 GT 和预测，计算主指标 | GT + predictions | summary + per-case rows |

### 9.1 Case schema

每个 case 必须能独立复现目的地和路线真值：

```json
{
  "id": "hangzhou_yuquan_000001",
  "city": "hangzhou",
  "scene_id": "yuquan",
  "category": "implicit_need_navigation",
  "subcategory": "thirsty",
  "instruction": "我有点渴了，帮我找附近可以买饮料的地方。",
  "origin": {"lat": 30.2645, "lng": 120.1180},
  "gt_destination": {"provider_poi_id": "poi_a", "name": "便利店 A", "rank": 1},
  "acceptable_destinations": [
    {
      "provider_poi_id": "poi_a",
      "name": "便利店 A",
      "type": "convenience_store",
      "reference_route_distance_m": 135.0,
      "reference_route": [[120.1180, 30.2645], [120.1190, 30.2651]],
      "rank": 1
    }
  ],
  "ranked_destinations": [
    {
      "rank": 1,
      "provider_poi_id": "poi_a",
      "route_distance_m": 135.0,
      "score_breakdown": {"relevance": 1.0, "route_distance_m": 135.0}
    }
  ],
  "gt_route": [[120.1180, 30.2645], [120.1190, 30.2651]],
  "gt_distance_m": 135.0,
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
    "poi_search": ["poi:hangzhou_yuquan_000001:thirsty"],
    "routes": ["route:hangzhou_yuquan_000001:poi_a"]
  }
}
```

`cache_keys` 不参与评分，但用于审计：给定 case 可以追溯到当时冻结的 POI 候选和路线响应。

### 9.2 数学模型

单目标任务的候选集合由固定 provider 和 scope 产生：

\[
C(o, q; \Omega)=\operatorname{POISearch}(o, q, \Omega)
\]

其中 \(o\) 是起点，\(q\) 是需求指令，\(\Omega\) 是 `optimality_scope`。

候选先通过硬约束：

\[
H(d, q) \in \{0, 1\}
\]

再对可达候选算路并排序：

\[
R(o,d)=\operatorname{RouteProvider}(o,d;\Omega)
\]

\[
d^* = \arg\min_{d \in C(o,q;\Omega),\ H(d,q)=1,\ R(o,d)\neq \varnothing}
\operatorname{cost}(R(o,d))
\]

默认 cost 是路线距离：

\[
\operatorname{cost}(R(o,d)) = \operatorname{route\_distance\_m}(R(o,d))
\]

对语义偏好或约束型任务，可使用固定加权函数排序，但必须保存分项得分：

\[
\begin{aligned}
\operatorname{score}(d \mid q,o)={}&
w_{\mathrm{rel}}\operatorname{rel}(d,q)
-w_{\mathrm{dist}}\operatorname{dist}_{\mathrm{route}}(o,d) \\
&-w_{\mathrm{time}}\operatorname{time}_{\mathrm{route}}(o,d)
+w_{\mathrm{pref}}\operatorname{pref}(d,q)
-w_{\mathrm{viol}}\operatorname{viol}(d,q)
\end{aligned}
\]

可接受目的地集合不是指标，而是 GT 制作阶段处理多合理答案的集合：

\[
D_{\mathrm{acc}}=
\{d_i \mid H(d_i,q)=1 \land
\operatorname{score}(d_i) \ge \operatorname{score}(d^*)-\epsilon\}
\]

对于“最近的 X”类任务，可使用路线距离相对阈值：

\[
\operatorname{dist}_{\mathrm{route}}(o,d_i)
\le (1+\delta)\operatorname{dist}_{\mathrm{route}}(o,d^*)
\]

### 9.3 Replay sandbox

评估阶段禁止调用 live map API。所有可变外部信息必须在构建阶段冻结：

```text
data/benchmark/<version>/
  task_gt.jsonl
  task_rejects.jsonl
  poi_cache.jsonl
  route_cache.jsonl
  validation_report.json
```

RouteServer 可以在 benchmark 模式下通过 replay backend 读取缓存：

```text
search_destinations(query, origin, scope) -> frozen POI candidates
plan_route(origin, destination, mode, strategy) -> frozen route response
```

这与 MobilityBench 的 sandbox 工具思想一致：同一 case 和同一 tool args 必须得到同一
response，保证不同模型、不同时间运行结果可比。

### 9.4 Runner and evaluator contract

Runner 输出统一 prediction JSONL：

```json
{
  "id": "hangzhou_yuquan_000001",
  "chosen_destination": {"provider_poi_id": "poi_a", "name": "便利店 A"},
  "candidate_destinations": [{"provider_poi_id": "poi_a", "name": "便利店 A"}],
  "route_polyline": [[120.1180, 30.2645], [120.1190, 30.2651]],
  "route_summary": {"distance_m": 142.0},
  "trace": {
    "tool_calls": [],
    "latency_ms": 0,
    "token_usage": {}
  }
}
```

`trace` 只做诊断和效率分析，不作为目的地/路线成功的必要条件。工具调用序列可以参考
MobilityBench 做 schema/compliance 诊断，但主指标不要求工具调用与 GT 完全一致。

当前实现提供三类 runner：

| Runner | 用途 | 是否访问 live API | 主要命令参数 |
| --- | --- | --- | --- |
| `oracle` | evaluator contract test | 否 | `--runner oracle` |
| `routeserver-replay` | 离线 RouteServer smoke / baseline | 否 | `--runner routeserver-replay --poi-cache ... --route-cache ...` |
| `routeserver-live` | 真实 provider 预测 | 是 | `--runner routeserver-live --map-provider amap` |

`routeserver-replay` 不是 oracle。它仍执行：

```text
instruction -> IntentParser -> frozen POI candidates -> route all candidates -> select destination -> prediction
```

区别是 POI 和路线来自 replay cache，而不是 live map provider。因此它用于验证 RouteServer
runner 的结构化输出、候选排序和 route-update 语义，不用于证明真实地图 API 可用。

`routeserver-live` 使用同一 runner 逻辑，但 backend 切到 `LiveMapBackend`。如果环境中缺少
`aiohttp`，AMap 路径会退化到 `AmapUrbanNavProvider` 的 `urllib` fallback；仍要求
`AMAP_KEY`。

Runner 的核心模型是：

\[
\hat I = \operatorname{IntentParser}(q)
\]

\[
\hat C = \operatorname{POISearch}(o, \hat I.\mathrm{search\_keywords}, \hat I.\mathrm{radius})
\]

\[
\hat R_i = \operatorname{Route}(o, \hat d_i), \quad \hat d_i \in \hat C
\]

默认选择最短可规划路线：

\[
\hat d =
\arg\min_{\hat d_i \in \hat C,\ \hat R_i \neq \varnothing}
\operatorname{route\_distance\_m}(\hat R_i)
\]

如果启用 `--use-llm-reranker --selection-strategy semantic_then_route`，则先用
`CandidateReranker` 对候选按语义相关性排序，再选择第一个可规划候选：

\[
\hat d = \operatorname{first}_{\hat d_i \in \operatorname{Rerank}(\hat C, q)}
[\hat R_i \neq \varnothing]
\]

这两个策略都输出同一个 prediction schema，因此 evaluator 不需要新增主指标。

Evaluator 必须先匹配目的地，再评估路线：

\[
\operatorname{match}(\hat d, D_{\mathrm{acc}}) \rightarrow d_j \ \text{or}\ \varnothing
\]

如果匹配到 \(d_j\)，路线指标使用 \(d_j\) 的冻结参考路线：

\[
\operatorname{route\_ratio}=
\frac{L(\hat P)}{L(P^*_{d_j})}
\]

\[
SPL = RS \cdot \frac{L(P^*_{d_j})}{\max(L(P^*_{d_j}), L(\hat P))}
\]

这能避免模型选择另一个合理目的地时，被错误地拿 rank-1 路线作为分母。

## 10. 真值制作原则

所有真值必须预先固化，评估时不动态生成。

目的地真值：

```text
origin + task type + POI candidates
-> filter satisfying POIs
-> route each recalled candidate
-> choose rank-1 by route distance as gt_destination
-> save ranked_destinations + acceptable_destinations + audit
```

`origin` 不应由 Frodobots/SideWalk 等基础视觉采样分布决定。第一版 benchmark 以地图
API 规划为核心，origin 是任务构造变量，应由城市区域、目标类型、路线距离、路线形态、
候选歧义度共同分层采样。可以先采样城市区域内候选起点，也可以先采样目标 POI 后反向找
可产生多样路线的起点；最终必须回查该 origin 下的 satisfying POIs，并确认 rank-1
确实是固定候选召回范围内路线代价最小的目标。

每条样本的“最优”只在固定 provider、固定搜索半径、固定候选召回策略、固定路线模式下成立。
GT builder 必须保存 `optimality_scope`、`ranked_destinations` 和 reject reason，用于审计：

```json
{
  "optimality_scope": {
    "poi_provider": "amap",
    "search_radius_m": 1500,
    "candidate_limit": 30,
    "route_provider": "amap",
    "route_mode": "foot",
    "cost": "route_distance_m"
  },
  "ranked_destinations": [
    {"rank": 1, "provider_poi_id": "poi_a", "route_distance_m": 135.0},
    {"rank": 2, "provider_poi_id": "poi_b", "route_distance_m": 168.0}
  ]
}
```

Validator 必须检查 `gt_destination` 等于 `ranked_destinations[0]`，候选按路线代价升序，
rank-1 满足 task 约束，所有未进入候选或路线失败的目标有 `reject_reason`。

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

第一版实现先支持“多合理目的地”而不是完整多站序列。样例通过
`acceptable_destinations` 表达同一需求下多个正确目标：

\[
D_{\mathrm{acc}} = \{d_1, d_2, \ldots, d_k\}
\]

预测只要命中任意 \(d_i \in D_{\mathrm{acc}}\)，就按该 matched destination 的
`reference_route` 计算路线指标。这样可以覆盖“便利店或咖啡店都可以买水”这类开放目标，
同时不增加主排行榜指标。

中途加点真值：

```text
old session + current position + added via + original final destination
-> updated route
-> save gt_added_via + gt_updated_route + gt_updated_distance_m
```

第一版 route update 采用“更新后任务等价为新的 OD”的评估方式。case 的主字段保存更新后
的 `origin / instruction / gt_destination / gt_route`，原始会话和更新事件保存在
`metadata.route_update`：

```json
{
  "category": "route_update",
  "instruction": "我临时想改去可以买水的地方。",
  "origin": {"lat": 30.0005, "lng": 120.0005},
  "metadata": {
    "route_update": {
      "update_type": "replace_destination",
      "previous_instruction": "先去咖啡店。",
      "updated_instruction": "我临时想改去可以买水的地方。",
      "current_origin": {"lat": 30.0005, "lng": 120.0005},
      "previous_destination": {"name": "咖啡店 Previous"}
    }
  }
}
```

Runner 在发现 `metadata.route_update` 时使用：

\[
q_{\mathrm{eff}} = q_{\mathrm{updated}}, \quad
o_{\mathrm{eff}} = o_{\mathrm{current}}
\]

然后仍然走普通 `instruction -> destination -> route` 流程。该设计的好处是主指标不变；
route-update 能力通过 category/subcategory slice 单独报告。

如果找不到满足条件的 POI 或路线，样本应丢弃或进入 reject，不应强行制造 GT。

## 11. 实现优先级

1. 固定 `task_gt.jsonl`、prediction JSONL、`poi_cache.jsonl`、`route_cache.jsonl` schema。
2. 实现 GT builder smoke 版本：单城市/小区域/100-300 cases。
3. 实现 replay map backend，确保 benchmark 运行不访问 live API。
4. 保持现有 `eval_destination_route.py`，但主格式切到 UrbanNav JSONL。
5. 扩展 destination evaluator 支持 `acceptable_destinations`、ID 优先匹配和 matched destination 参考路线。
6. 新增 `eval_instruction_parse.py`，只实现 4 个解析指标。
7. 新增 `eval_multi_stop_route.py` 和 `eval_route_update.py`，只实现上面的核心指标。
8. 输出 `validation_report.json` 和数据集卡，说明 optimality scope、采样分布、reject 统计和人工审核比例。

## 12. 当前 Step 8-10 实现状态

已实现文件：

```text
src/mcp_server/eval/urbannav_runners.py
tools/evaluation/run_urbannav_predictions.py
tools/dataset_processing/build_urbannav_mock_benchmark.py
tools/dataset_processing/build_urbannav_benchmark.py
```

离线 smoke 命令：

```bash
PYTHONPATH=src python tools/dataset_processing/build_urbannav_benchmark.py \
  --provider mock \
  --output-root /tmp/urbannav_step8_10_smoke \
  --num-cases 3 \
  --include-multi-target \
  --include-route-updates

PYTHONPATH=src python tools/evaluation/run_urbannav_predictions.py \
  --cases /tmp/urbannav_step8_10_smoke/task_gt.jsonl \
  --runner routeserver-replay \
  --poi-cache /tmp/urbannav_step8_10_smoke/poi_cache.jsonl \
  --route-cache /tmp/urbannav_step8_10_smoke/route_cache.jsonl \
  --output-jsonl /tmp/urbannav_step8_10_smoke/routeserver_replay_predictions.jsonl

PYTHONPATH=src python tools/evaluation/eval_urbannav_destination_route.py \
  --ground-truth /tmp/urbannav_step8_10_smoke/task_gt.jsonl \
  --predictions /tmp/urbannav_step8_10_smoke/routeserver_replay_predictions.jsonl \
  --output-json /tmp/urbannav_step8_10_smoke/routeserver_replay_summary.json \
  --output-jsonl /tmp/urbannav_step8_10_smoke/routeserver_replay_rows.jsonl
```

真实 provider runner 命令：

```bash
export MAP_PROVIDER=amap
export AMAP_KEY=...

PYTHONPATH=src python tools/evaluation/run_urbannav_predictions.py \
  --cases data/benchmark/<version>/task_gt.jsonl \
  --runner routeserver-live \
  --map-provider amap \
  --candidate-limit 10 \
  --search-radius-m 1500 \
  --output-jsonl data/benchmark/<version>/predictions/routeserver_live.jsonl
```

如需启用 LLM 意图解析或候选 rerank：

```bash
PYTHONPATH=src python tools/evaluation/run_urbannav_predictions.py \
  --cases data/benchmark/<version>/task_gt.jsonl \
  --runner routeserver-live \
  --map-provider amap \
  --use-llm-intent \
  --use-llm-reranker \
  --selection-strategy semantic_then_route \
  --output-jsonl data/benchmark/<version>/predictions/routeserver_live_llm.jsonl
```

注意：真实 live runner 只用于生成系统预测；正式评估仍只读取冻结 GT 和 prediction JSONL，
不会在 evaluator 内调用 live map API。
