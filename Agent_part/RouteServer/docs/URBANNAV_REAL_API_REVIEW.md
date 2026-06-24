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

## 7. 大规模 AMap GT 采集

中国区域数据使用 AMap live provider 构建。采集阶段会真实调用 POI 检索和步行路线 API；没有 POI、没有可规划路线、路线距离超出范围或类别约束不满足的样例会进入 `task_rejects.jsonl`，不会写入 `task_gt.jsonl`。

```bash
cd Agent_part/RouteServer

export MAP_PROVIDER=amap
export AMAP_KEY="<your-amap-key>"

PYTHONPATH=src python tools/dataset_processing/collect_urbannav_live_gt.py \
  --output-root data/benchmark/urbannav_live_v1 \
  --config config/urbannav_live_gt_collection.json \
  --target-cases 10000 \
  --map-provider amap \
  --origins-per-region 45 \
  --candidate-limit 10 \
  --search-radius-m 1500 \
  --acceptable-route-ratio 1.15 \
  --throttle-s 0.05 \
  --resume
```

dry-run 只检查采样计划，不访问 API：

```bash
PYTHONPATH=src python tools/dataset_processing/collect_urbannav_live_gt.py \
  --output-root /tmp/urbannav_live_gt_dryrun \
  --target-cases 10000 \
  --dry-run \
  --overwrite
```

## 8. Global OSM/OSRM Pilot

多国家公开 split 使用 OSM/Overpass 做 POI 检索，并使用 OSRM/Valhalla 类 routing backend 生成步行路线。公共 Overpass 不适合高并发，pilot 阶段建议保守设置并开启 `--resume`。

100-case smoke：

```bash
cd Agent_part/RouteServer

PYTHONPATH=src python tools/dataset_processing/collect_urbannav_global_gt.py \
  --output-root /tmp/urbannav_global_osm_smoke_100 \
  --target-cases 100 \
  --origins-per-region 12 \
  --candidate-limit 3 \
  --search-radius-m 1200 \
  --throttle-s 0.5 \
  --retry 1 \
  --overwrite
```

1K pilot + Leaflet review：

```bash
cd Agent_part/RouteServer

PYTHONPATH=src python tools/dataset_processing/run_urbannav_global_pilot_review.py \
  --output-root data/benchmark/urbannav_global_osm_v1 \
  --target-cases 1000 \
  --origins-per-region 45 \
  --candidate-limit 3 \
  --search-radius-m 1200 \
  --throttle-s 0.8 \
  --retry 2 \
  --case-concurrency 2 \
  --route-concurrency 3 \
  --resume \
  --review-cases 160 \
  --review-dir data/benchmark/urbannav_global_osm_v1/review_leaflet_160
```

## 8.1 Local OSM Extract Pilot

公开主集更推荐使用本地 OSM extract，而不是公共 Overpass。流程是先从 PBF 或预抽 JSONL 构建本地 POI index，再用本地 OSRM/Valhalla 提供步行路线。

先生成本地资产计划、router config 和构建脚本：

```bash
cd Agent_part/RouteServer

PYTHONPATH=src python tools/dataset_processing/prepare_urbannav_osm_local_assets.py \
  --global-config config/urbannav_global_gt_collection.json \
  --source-config config/urbannav_osm_extract_sources.json \
  --output-root data/osm \
  --router-config-output config/urbannav_osm_local_sources.json \
  --write-osrm-scripts \
  --write-valhalla-scripts \
  --write-router-config \
  --router-type osrm \
  --overwrite
```

下载、裁剪、构建 POI index 可以分阶段执行：

```bash
PYTHONPATH=src python tools/dataset_processing/prepare_urbannav_osm_local_assets.py \
  --output-root data/osm \
  --download \
  --overwrite

PYTHONPATH=src python tools/dataset_processing/prepare_urbannav_osm_local_assets.py \
  --output-root data/osm \
  --extract \
  --overwrite

PYTHONPATH=src python tools/dataset_processing/prepare_urbannav_osm_local_assets.py \
  --output-root data/osm \
  --build-poi-index \
  --overwrite
```

这些步骤依赖本机工具：

```text
osmium          # 裁剪 Geofabrik/BBBike PBF
pyosmium        # 从 PBF 抽取 POI JSONL
osrm-extract    # 构建 OSRM 步行图
valhalla_*      # 构建 Valhalla pedestrian tiles，可选
```

构建并启动 OSRM：

```bash
bash data/osm/osrm/build_osrm_all.sh
bash data/osm/osrm/run_osrm_all.sh
```

如果使用 Valhalla，生成 config 时把 `--router-type valhalla` 打开，然后构建并启动：

```bash
PYTHONPATH=src python tools/dataset_processing/prepare_urbannav_osm_local_assets.py \
  --output-root data/osm \
  --router-config-output config/urbannav_osm_local_sources.json \
  --write-valhalla-scripts \
  --write-router-config \
  --router-type valhalla \
  --overwrite

bash data/osm/osrm/build_valhalla_all.sh
bash data/osm/osrm/run_valhalla_all.sh
```

构建本地 POI JSONL：

```bash
cd Agent_part/RouteServer

PYTHONPATH=src python tools/dataset_processing/build_osm_poi_index.py \
  --input-pbf data/osm/cities/tokyo_shinjuku.osm.pbf \
  --output-jsonl data/osm/index/tokyo_shinjuku.jsonl \
  --country japan \
  --city tokyo \
  --region-id tokyo_shinjuku \
  --overwrite
```

如果已经用外部工具预抽了 POI，也可以直接从 JSONL 构建：

```bash
PYTHONPATH=src python tools/dataset_processing/build_osm_poi_index.py \
  --input-jsonl /path/to/preextracted_tokyo_pois.jsonl \
  --output-jsonl data/osm/index/tokyo_shinjuku.jsonl \
  --country japan \
  --city tokyo \
  --region-id tokyo_shinjuku \
  --overwrite
```

本地 provider 的索引和 router 配置见：

```text
config/urbannav_osm_local_sources.json
```

采集时使用 `osm_local`：

```bash
PYTHONPATH=src python tools/dataset_processing/collect_urbannav_global_gt.py \
  --output-root data/benchmark/urbannav_global_osm_local_v1 \
  --config config/urbannav_global_gt_collection.json \
  --provider osm_local \
  --poi-index-root data/osm/index \
  --local-router-config config/urbannav_osm_local_sources.json \
  --target-cases 1000 \
  --origins-per-region 45 \
  --candidate-limit 3 \
  --search-radius-m 1200 \
  --case-concurrency 2 \
  --route-concurrency 3 \
  --resume
```

该路径不访问公共 Overpass；是否访问网络取决于 `local-router-config` 里配置的 OSRM/Valhalla endpoint。论文级公开 split 应使用本机或内网固定 OSRM/Valhalla 服务，并记录 PBF 版本和 SHA。

## 9. Multi-Stop 和 Route-Update Review

多途经点 review 用于检查有序 stop 的构造质量。HTML 地图会标注 `S, 1, 2, ..., D`。

```bash
cd Agent_part/RouteServer

export MAP_PROVIDER=amap
export AMAP_KEY="<your-amap-key>"

PYTHONPATH=src python tools/dataset_processing/build_urbannav_multistop_review.py \
  --output-root data/benchmark/urbannav_yuquan_multistop_live_v1 \
  --origin-json tests/fixtures/urbannav_yuquan_origin.jsonl \
  --tasks-json tests/fixtures/urbannav_yuquan_multistop_tasks.jsonl \
  --map-provider amap \
  --num-cases 5 \
  --search-radius-m 1500 \
  --candidate-limit 8 \
  --acceptable-route-ratio 1.15
```

Route-update review 用于检查已有路线上的增量修改。HTML 地图用灰色虚线表示旧路线，红色实线表示更新后路线，并标注原起点、更新位置、旧目标、新增或替换目标和最终目标。

```bash
cd Agent_part/RouteServer

export MAP_PROVIDER=amap
export AMAP_KEY="<your-amap-key>"

PYTHONPATH=src python tools/dataset_processing/build_urbannav_route_update_review.py \
  --output-root data/benchmark/urbannav_yuquan_route_update_live_v1 \
  --origin-json tests/fixtures/urbannav_yuquan_origin.jsonl \
  --tasks-json config/urbannav_route_update_review_tasks.jsonl \
  --map-provider amap \
  --num-cases 4 \
  --search-radius-m 1500 \
  --candidate-limit 3 \
  --acceptable-route-ratio 1.15
```

## 10. Commercial Provider Review

商业 provider 对照集用于分析开放地图真值质量和 provider 偏差，不改变 UrbanNav schema 与 evaluator。TomTom smoke 需要提供 `TOMTOM_API_KEY`。

```bash
cd Agent_part/RouteServer

export TOMTOM_API_KEY="<your-tomtom-key>"

PYTHONPATH=src python tools/dataset_processing/run_urbannav_commercial_smoke_review.py \
  --output-root data/benchmark/urbannav_global_tomtom_smoke_v1 \
  --review-dir data/benchmark/urbannav_global_tomtom_smoke_v1/review \
  --target-cases 50 \
  --origins-per-region 8 \
  --candidate-limit 5 \
  --search-radius-m 1500 \
  --case-concurrency 2 \
  --route-concurrency 4 \
  --resume
```

## 11. Combined Overview HTML

综合 review 页面用于把 AMap、OSM、multi-stop、route-update 和 commercial review split 放在同一个 HTML 中比较。页面包含样本地图、provider/country/task 分布、10K 目标分布饼图和核心指标公式。

```bash
cd Agent_part/RouteServer

PYTHONPATH=src python tools/evaluation/build_urbannav_combined_overview_html.py \
  --output-dir data/benchmark/urbannav_combined_overview_v1 \
  --multi-stop-tasks data/benchmark/urbannav_yuquan_multistop_live_v1/task_gt.jsonl \
  --max-samples 190 \
  --title "UrbanNav Combined Provider and 10K Plan Review"
```
