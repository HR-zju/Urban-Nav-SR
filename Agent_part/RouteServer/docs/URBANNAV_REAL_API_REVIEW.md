# UrbanNav Real API Review

本文档说明如何用真实 AMap API 跑一小批需求指令，并生成可人工审阅的 HTML。

## 1. 运行真实 GT builder

```bash
cd Agent_part/RouteServer

export MAP_PROVIDER=amap
export AMAP_KEY=...

PYTHONPATH=src python tools/dataset_processing/build_urbannav_benchmark.py \
  --provider live \
  --map-provider amap \
  --output-root /tmp/urbannav_real_api_review \
  --num-cases 8 \
  --candidate-limit 8 \
  --search-radius-m 1200 \
  --include-multi-target
```

输出：

```text
/tmp/urbannav_real_api_review/
  task_gt.jsonl
  poi_cache.jsonl
  route_cache.jsonl
  task_rejects.jsonl
  validation_report.json
```

GT builder 的核心原则不变：

\[
d^* = \arg\min_{d \in D_{routeable}} \operatorname{route\_distance\_m}(o,d)
\]

`acceptable_destinations` 保存同一阶段需求下的 near-tie 或多合理目标。这是“多选一”
语义，不是“多个途经点”语义。

## 2. 跑 RouteServer-style prediction

离线 replay smoke：

```bash
PYTHONPATH=src python tools/evaluation/run_urbannav_predictions.py \
  --cases /tmp/urbannav_real_api_review/task_gt.jsonl \
  --runner routeserver-replay \
  --poi-cache /tmp/urbannav_real_api_review/poi_cache.jsonl \
  --route-cache /tmp/urbannav_real_api_review/route_cache.jsonl \
  --output-jsonl /tmp/urbannav_real_api_review/routeserver_replay_predictions.jsonl
```

真实 live runner：

```bash
PYTHONPATH=src python tools/evaluation/run_urbannav_predictions.py \
  --cases /tmp/urbannav_real_api_review/task_gt.jsonl \
  --runner routeserver-live \
  --map-provider amap \
  --candidate-limit 8 \
  --search-radius-m 1200 \
  --output-jsonl /tmp/urbannav_real_api_review/routeserver_live_predictions.jsonl
```

如需启用 LLM 意图解析和候选 rerank：

```bash
export LLM_PROVIDER=ark
export ARK_API_KEY=...
export LLM_MODEL=...
export LLM_API_BASE_URL=...

PYTHONPATH=src python tools/evaluation/run_urbannav_predictions.py \
  --cases /tmp/urbannav_real_api_review/task_gt.jsonl \
  --runner routeserver-live \
  --map-provider amap \
  --candidate-limit 8 \
  --search-radius-m 1200 \
  --use-llm-intent \
  --use-llm-reranker \
  --selection-strategy semantic_then_route \
  --output-jsonl /tmp/urbannav_real_api_review/routeserver_live_llm_predictions.jsonl
```

## 3. 评估

```bash
PYTHONPATH=src python tools/evaluation/eval_urbannav_destination_route.py \
  --ground-truth /tmp/urbannav_real_api_review/task_gt.jsonl \
  --predictions /tmp/urbannav_real_api_review/routeserver_live_predictions.jsonl \
  --output-json /tmp/urbannav_real_api_review/routeserver_live_summary.json \
  --output-jsonl /tmp/urbannav_real_api_review/routeserver_live_rows.jsonl
```

## 4. 多途经点 review

多途经点指令使用 `multi_stop_route`，每个 case 保存：

- `gt_destination_sequence`：按顺序必须经过的 stop。
- `gt_route_legs`：每一段的搜索词、候选排序、rank-1 stop、可接受 near-tie 和路线。
- `gt_route`：各段路线拼接后的完整路线。

生成玉泉校区固定起点的多途经点 review：

```bash
PYTHONPATH=src python tools/dataset_processing/build_urbannav_multistop_review.py \
  --output-root /tmp/urbannav_yuquan_multistop_review \
  --origin-json tests/fixtures/urbannav_yuquan_origin.jsonl \
  --tasks-json tests/fixtures/urbannav_yuquan_multistop_tasks.jsonl \
  --map-provider amap \
  --num-cases 5 \
  --search-radius-m 1800 \
  --candidate-limit 8
```

再生成 HTML：

```bash
PYTHONPATH=src python tools/evaluation/build_urbannav_review_html.py \
  --ground-truth /tmp/urbannav_yuquan_multistop_review/task_gt.jsonl \
  --output-dir /tmp/urbannav_yuquan_multistop_review/html \
  --title "UrbanNav Yuquan Multi-Stop Review" \
  --download-staticmaps \
  --max-cases 5
```

静态图中会标出 `S, 1, 2, ..., D`，其中数字表示中间途经点。multi-stop 当前作为
人工 review/分析集，不混入单目标 `destination_success / route_success / spl` 主表。

## 5. 生成 HTML review

```bash
PYTHONPATH=src python tools/evaluation/build_urbannav_review_html.py \
  --ground-truth /tmp/urbannav_real_api_review/task_gt.jsonl \
  --predictions /tmp/urbannav_real_api_review/routeserver_live_predictions.jsonl \
  --eval-rows /tmp/urbannav_real_api_review/routeserver_live_rows.jsonl \
  --output-dir /tmp/urbannav_real_api_review/html \
  --title "UrbanNav Real API Review" \
  --download-staticmaps \
  --max-cases 20
```

打开：

```text
/tmp/urbannav_real_api_review/html/index.html
```

HTML 中每条 case 包含：

- 需求指令、类别、场景、origin。
- GT destination 和 predicted destination。
- `acceptable_destinations`，用于人工判断多目标是否合理。
- evaluator row，包含 `destination_success / hit@5 / route_success / route_ratio / spl`。
- AMap staticmap BEV 图：绿色路线、S 起点、D 终点。图片已下载到本地
  `html/images/`，HTML 不写入 API key。

## 6. 人工 review 重点

1. 指令语义是否与候选 POI 类型一致。
2. rank-1 GT 是否确实是当前 `optimality_scope` 下路线最短的合理目标。
3. `acceptable_destinations` 是否过宽或过窄。
4. 路线是否从 origin 到达目标，是否存在明显绕路或不可行路径。
5. `acceptable_destinations` 是否被误当成多途经点；多途经点应检查
   `gt_destination_sequence` 的顺序是否符合指令。
6. route-update 或 multi-stop 样例是否应进入主 benchmark，还是只作为分析集。
