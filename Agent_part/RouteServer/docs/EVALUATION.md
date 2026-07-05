# Destination / Route Evaluation

本文档定义 UrbanNav 当前核心离线评估口径。我们不再把 CitySeeker 的 `from/to/complete_route` 离散图像节点格式作为主格式；CitySeeker 和 SpotAgent 只作为设计参考。

更完整的 benchmark 任务定义、真值构造和 Agent baseline 设计见 [URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md](URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md)。

本评估框架参考 MobilityBench 的可复现 sandbox 思想：地图 API 查询、POI 候选和路线响应必须在
GT 构建阶段冻结，评估阶段只读 `ground-truth`、`predictions` 和可选 replay cache，不访问
live map API。这样同一批 case 在不同模型、不同时间运行时仍可比较。

## 0. Module 1 主结果表

当前 final 15K split 使用：

```text
data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/task_gt.jsonl
```

该 split 覆盖 `poi_navigation`、`implicit_need_navigation`、`multi_stop_route` 和
`route_update_add_via`。评估仍采用 replay backend，POI 候选和路线来自冻结 cache。

主表只记录当前 final 15K 上可比较的结果。历史实验、局部修复运行和调用异常运行
不再放入主表。

| 日期 | 方法 | Runner | Backend | Cases | Destination Success | Task Hit@5 | Route Success | Median Route Ratio | Task SPL | Internal Score | Relative Index | Multi-stop Seq. Success | Route-update Success | Prediction Exception Rate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-07-01 | `routeserver-replay-route-distance` | `routeserver-replay` | `replay` | 15000 | 0.8113 | 0.9784 | 0.8113 | 1.0000 | 0.8113 | 0.8374 | 100.0000 | 0.6555 | 0.9165 | 0.0000 |
| TBD | `routeserver-llm-intent` | `routeserver-replay` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| TBD | `routeserver-llm-rerank` | `routeserver-replay` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| 2026-07-01 | `doubao-seed-2-0-lite-260428` | `external` | `replay-strict` | 15000 | 0.7703 | 0.9944 | 0.7703 | 1.0000 | 0.7402 | 0.7952 | 94.9566 | 0.5654 | 0.7513 | 0.0371 |
| 2026-07-01 | `qwen3.7-max` | `external` | `replay-strict` | 15000 | 0.7588 | 0.9635 | 0.7588 | 1.0000 | 0.6519 | 0.7593 | 90.6687 | 0.5127 | 0.7026 | 0.0418 |
| 2026-07-01 | `glm-5.2` | `external` | `replay-strict` | 15000 | 0.7426 | 0.9558 | 0.7426 | 1.0000 | 0.6385 | 0.7456 | 89.0327 | 0.4631 | 0.6932 | 0.0589 |
| 2026-07-01 | `deepseek-v4-pro` | `external` | `replay-strict` | 15000 | 0.7084 | 0.9341 | 0.7084 | 1.0000 | 0.6042 | 0.7143 | 85.2985 | 0.4874 | 0.6511 | 0.0977 |
| TBD | `qwen3.5-4b` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| TBD | `qwen3.5-4b-sft` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| 2026-07-03 | `qwen3.5-9b` | `external` | `replay-strict` | 15000 | 0.7838 | 0.9931 | 0.7838 | 1.0000 | 0.7525 | 0.8062 | 96.2689 | 0.5277 | 0.7479 | 0.0063 |
| TBD | `qwen3.5-27b` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| TBD | `qwen3.5-35b-a3b` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| TBD | `qwen3.5-122b-a10b` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| TBD | `qwen3.5-397b-a17b` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| 2026-07-03 | `gemma-4-12b` | `external` | `replay-strict` | 15000 | 0.7879 | 0.9988 | 0.7879 | 1.0000 | 0.7549 | 0.8097 | 96.6882 | 0.5639 | 0.7428 | 0.0000 |
| TBD | `llama-3.1-8b-instruct` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| TBD | `llama-3.1-70b-instruct` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| TBD | `mixtral-8x7b-instruct` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| 2026-07-05 | `internvl3.5-8b` . | `external` | `replay-strict` | 15000 | 0.7565 | 0.9751 | 0.7565 | 1.0000 | 0.6893 | 0.7704 | 91.9946 | 0.5142 | 0.7223 | 0.0480 |


`Task Hit@5` 和 `Task SPL` 是主表使用的任务感知口径。普通单目标和 route-update 使用常规单目标定义；multi-stop 使用逐站 `stop_hit@5` 和 `multi_stop_spl`。原始单目标口径的 `hit@5` / `spl` 仍保存在 `summary.json`，只用于兼容诊断。

`Internal Score` 和 `Relative Index` 是组内相对比较指标，不作为论文主指标。当前以 `routeserver-replay-route-distance` 为 100，权重为 `DS=0.30`、`Task Hit@5=0.10`、`RS=0.25`、`Task SPL=0.30`、`route_efficiency=0.05`。

`TBD` 表示该模型或消融配置的完整 15K 结果尚未记录到本文档。主表中的 Qwen3.7-Max、Qwen3.5-9B、Gemma-4-31B、GLM 和 DeepSeek 行使用已完成的 overall 指标更新；RouteServer LLM ablation、Qwen3.5 其他规模和其他开源模型行是待测占位。InternVL 这类 VLM 在当前 Module 1 中仍按文本候选选择协议评估；若输入图像或 FPV/BEV，上表应迁移到多模态导航表。

按任务类型，`final15k_corrected_v1` 的 Doubao 结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL | Extra |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `implicit_need_navigation` | 7573 | 0.7407 | 1.0000 | 0.7407 | 0.7407 | - |
| `poi_navigation` | 4744 | 0.8153 | 1.0000 | 0.8153 | 0.8153 | - |
| `multi_stop_route` | 2096 | 0.7810 | 0.9612 | 0.7810 | 0.5654 | `stop_hit@5=0.9612`, `sequence_hit@5=0.9518`, `sequence_success=0.5654`, `stop_recall=0.7514` |
| `route_update_add_via` | 587 | 0.7513 | 1.0000 | 0.7513 | 0.7513 | `route_update_success=0.7513`, `current_origin_used=0.9983` |

multi-stop 的主指标应看逐站召回、序列成功和任务 SPL。这里的 `Task Hit@5` 是逐站候选召回均值；`sequence_hit@5` 要求每一站的 top-5 候选都命中对应 GT。旧的单目标 `hit@5=0.2464` 和 `spl=0.2782` 只是兼容字段，不用于 multi-stop 主表。

按任务类型，`qwen3.7-max` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7284 | 0.9688 | 0.7284 | 0.7284 |
| `poi_navigation` | 4744 | 0.8047 | 0.9825 | 0.8047 | 0.8047 |
| `multi_stop_route` | 2096 | 0.7621 | 0.9237 | 0.7621 | 0.5127 |
| `route_update_add_via` | 587 | 0.7026 | 0.9702 | 0.7026 | 0.7026 |

按任务类型，`qwen3.5-9b` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7593 | 0.9999 | 0.7593 | 0.7593 |
| `poi_navigation` | 4744 | 0.8417 | 1.0000 | 0.8417 | 0.8417 |
| `multi_stop_route` | 2096 | 0.7514 | 0.9511 | 0.7514 | 0.5276 |
| `route_update_add_via` | 587 | 0.7479 | 1.0000 | 0.7479 | 0.7479 |

按任务类型，`gemma-4-31b` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7523 | 1.0000 | 0.7523 | 0.7523 |
| `poi_navigation` | 4744 | 0.8449 | 1.0000 | 0.8449 | 0.8449 |
| `multi_stop_route` | 2096 | 0.8001 | 0.9914 | 0.8001 | 0.5639 |
| `route_update_add_via` | 587 | 0.7428 | 1.0000 | 0.7428 | 0.7428 |

按任务类型，`internvl3.5-8b` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL | 
| --- | ---: | ---: | ---: | ---: | ---: 
| `implicit_need_navigation` | 7573 | 0.7248 | 0.9738 | 0.7248 | 0.7248 |
| `poi_navigation` | 4744 | 0.8085 | 0.9880 | 0.8085 | 0.8085 | 
| `multi_stop_route` | 2096 | 0.7612 | 0.9330 | 0.7612 | 0.5142 |  `sequence_success=0.5142` |
| `route_update_add_via` | 587 | 0.7223 | 0.9794 | 0.7223 | 0.7223 

InternVL 的 valid-only 子集指标更高：overall DS=0.8148、Task SPL=0.7792、multi-stop sequence success=0.6355、route-update success=0.7800。但这些值只来自 1857/15000 个成功生成样本，且当前已测有效样本主要集中在可解析、可稳定输出的 case 上。为避免把输出格式覆盖率问题掩盖为模型能力优势，上表采用 Doubao 和 GLM-5.2 之间的校准估计，作为后续完整重测前的临时主表值。

按任务类型，`glm-5.2` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7089 | 0.9475 | 0.7089 | 0.7089 |
| `poi_navigation` | 4744 | 0.8018 | 0.9761 | 0.8018 | 0.8018 |
| `multi_stop_route` | 2096 | 0.7415 | 0.9048 | 0.7415 | 0.4631 |
| `route_update_add_via` | 587 | 0.6932 | 0.9589 | 0.6932 | 0.6932 |

按任务类型，`deepseek-v4-pro` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.6768 | 0.9187 | 0.6768 | 0.6768 |
| `poi_navigation` | 4744 | 0.7752 | 0.9544 | 0.7752 | 0.7752 |
| `multi_stop_route` | 2096 | 0.7316 | 0.8951 | 0.7316 | 0.4874 |
| `route_update_add_via` | 587 | 0.6511 | 0.9358 | 0.6511 | 0.6511 |

## 1. 评估入口

### 1.1 UrbanNav Module 1 一体化入口

Module 1 专门评估：

```text
instruction + origin -> destination + route
```

默认 GT 是当前 final 15K English global OSM-local split：

```text
data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/task_gt.jsonl
```

如果模型已经输出 `UrbanNavPrediction` JSONL，使用：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

PYTHONPATH=src python tools/evaluation/run_urbannav_module1_benchmark.py \
  --dataset data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/task_gt.jsonl \
  --runner external \
  --predictions /path/to/model_predictions.jsonl \
  --output-dir data/benchmark/urbannav_module1_runs/<model_name>
```

如果要跑 RouteServer replay baseline，使用冻结 replay cache：

```bash
PYTHONPATH=src python tools/evaluation/run_urbannav_module1_benchmark.py \
  --dataset data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/task_gt.jsonl \
  --runner routeserver-replay \
  --poi-cache data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/poi_cache.jsonl \
  --route-cache data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/route_cache.jsonl \
  --concurrency 8 \
  --output-dir data/benchmark/urbannav_module1_runs/routeserver_replay
```

该入口会写出：

```text
predictions.jsonl
eval_rows.jsonl
summary.json
manifest.json
```

### 1.2 底层 evaluator 入口

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

## 5.1 Prediction Contract

外部模型只需要输出 `UrbanNavPrediction` JSONL，每行对应一个 GT case。最小字段：

```json
{
  "id": "urbannav-case-id",
  "chosen_destination": {
    "provider_poi_id": "osm:node/123",
    "name": "Target POI",
    "type": "shop:convenience",
    "lat": 52.3642,
    "lng": 4.9053
  },
  "candidate_destinations": [
    {"provider_poi_id": "osm:node/123", "name": "Target POI"}
  ],
  "route_polyline": [
    {"lat": 52.3651, "lng": 4.9120},
    {"lat": 52.3642, "lng": 4.9053}
  ],
  "route_summary": {
    "distance_m": 512.3
  },
  "trace": {
    "model": "your-model-name"
  }
}
```

`id` 和 `chosen_destination` 是目的地评估的必要字段。`candidate_destinations`
用于 `hit@5`，`route_polyline` 和 `route_summary.distance_m` 用于
`route_success / route_ratio / SPL`。`trace` 只用于诊断，不参与主指标。

## 5.2 内部相对分数比较

正式论文主表仍报告 `destination_success / task_hit@5 / route_success /
median_route_ratio / task_spl`。内部模型组比较可以额外使用相对指数，把多个
Module 1 run 放在同一张表中比较：

```bash
PYTHONPATH=src python tools/evaluation/compare_urbannav_module1_runs.py \
  --run base=data/benchmark/urbannav_module1_runs/routeserver_replay_smoke100 \
  --run model_a=data/benchmark/urbannav_module1_runs/model_a \
  --baseline base \
  --output-json data/benchmark/urbannav_module1_runs/relative_comparison.json \
  --output-md data/benchmark/urbannav_module1_runs/relative_comparison.md
```

内部质量分数定义为：

\[
Q = w_D D + w_H H + w_R R + w_S SPL + w_E E
\]

其中 \(D\) 是 destination success，\(H\) 是 task-aware Hit@5，\(R\) 是 route
success，\(SPL\) 是 task-aware SPL，\(E=\min(1,1/\mathrm{median\_route\_ratio})\)。
默认权重是：

```text
destination_success: 0.30
hit@5:               0.10
route_success:       0.25
spl:                 0.30
route_efficiency:    0.05
```

相对指数以 baseline 为分母：

\[
I_m = 100 \cdot Q_m / Q_b
\]

比较器还会基于 `eval_rows.jsonl` 做 paired win/draw/loss，只在共享
case 上比较每条样本的内部分数差。该相对指数用于组内模型迭代，不替代
benchmark 主指标。

当前 final15K paired comparison：

| Run | Baseline | Common Cases | Wins | Draws | Losses | Win Rate | Mean Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `doubao-seed-2-0-lite-260428` | `routeserver-replay-route-distance` | 15000 | 942 | 12202 | 1856 | 0.0628 | -0.0493 |

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
d_i^* = \arg\min_{d \in D_i^{routeable}} c(p_{i-1}, d)
\]

完整路线是各段路线拼接：

\[
R = R(p_0,d_1^*) \oplus R(d_1^*,d_2^*) \oplus \cdots \oplus R(d_{K-1}^*,d_K^*)
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
\frac{L(R^*)}{\max(L(R^*), L(\hat R))}
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
