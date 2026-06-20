# Yelp Instruction Data Design

本文档说明如何使用 Yelp Open Dataset 为 UrbanNav 制作需求指令数据。Yelp 不直接提供导航任务真值；它提供的是大众真实消费需求、商户类别、属性、评论证据和城市分布。

## 1. 当前数据

本地路径：

```text
/media/ubuntu/data2/SR-Nav Dataset/archive
```

主要文件：

| 文件 | 作用 |
| --- | --- |
| `yelp_dataset/yelp_academic_dataset_business.json` | 商户、坐标、类别、评分、属性、营业时间 |
| `yelp_dataset/yelp_academic_dataset_review.json` | 长评论，可挖掘真实需求表达和语义属性 |
| `yelp_dataset/yelp_academic_dataset_tip.json` | 短文本提示，更接近日常需求表达 |
| `yelp_dataset/yelp_academic_dataset_checkin.json` | 到访时间分布，可辅助热门/开放时段分析 |
| `yelp_photos/photos.json` | 图片 metadata，含 `food/inside/outside/drink/menu` 标签 |
| `yelp_photos/photos/` | 商户图片，可作为视觉证据或品牌/场景扩展，不是第一阶段主输入 |

快速统计：

| 项 | 数量 |
| --- | ---: |
| business | 150,346 |
| reviews | 约 6.99M |
| tips | 908,915 |
| photos metadata | 200,100 |
| food/restaurant related businesses | 67,533 |

城市分布 Top：

```text
Philadelphia 14569
Tucson 9250
Tampa 9050
Indianapolis 7540
Nashville 6971
New Orleans 6209
Reno 5935
Edmonton 5054
Saint Louis 4827
Santa Barbara 3829
```

类别分布 Top：

```text
Restaurants 52268
Food 27781
Shopping 24395
Home Services 14356
Beauty & Spas 14292
Nightlife 12281
Health & Medical 11890
Local Services 11198
Bars 11065
Automotive 10773
Coffee & Tea 6703
Fast Food 6472
Breakfast & Brunch 6239
```

属性覆盖较好的字段：

```text
BusinessParking 91085
WiFi 56914
GoodForKids 53375
OutdoorSeating 48802
Ambience 44279
RestaurantsGoodForGroups 44170
NoiseLevel 37993
WheelchairAccessible 28953
DogsAllowed 18284
HappyHour 15171
```

图片标签分布：

```text
food 108152
inside 56031
outside 18569
drink 15670
menu 1678
```

## 2. Yelp 适合解决什么

Yelp 适合做三件事：

1. 给需求指令提供真实分布。
2. 给语义偏好提供属性/评论证据。
3. 给目的地候选提供 POI 类型和坐标。

Yelp 不适合直接做：

1. 路线真值。路线仍需 RouteServer / AMap / OSM / OSRM 生成。
2. 交通枢纽导航。Yelp 商业活动为主，公交/地铁覆盖弱。
3. 强视觉街景导航。photos 是商户图片，不是连续街景轨迹。

## 3. 与 UrbanNav 类型的对应关系

### 3.1 POI/品牌导航

Yelp 可支撑：

| Subcategory | Yelp 来源 | 强度 |
| --- | --- | --- |
| `restaurant` | `Restaurants` | 强 |
| `coffee_shop` / `cafe` | `Coffee & Tea`, `Cafes` | 强 |
| `fast_food` | `Fast Food`, brand/name | 强 |
| `shopping_mall` / shopping | `Shopping`, `Shopping Centers` | 中 |
| `hotel` | `Hotels & Travel`, `Hotels` | 中 |
| `healthcare_facility` | `Health & Medical`, `Doctors`, `Pharmacy` | 中 |
| brand-specific | `name` 字段 + categories | 中 |

不建议主做：

```text
bank / ATM / public transit
```

这些不是 Yelp 主覆盖范围，应该从 OSM/AMap/Baidu 获取。

### 3.2 隐式需求导航

Yelp 很适合把评论需求映射到目标类型：

| 需求 | 可映射目标 | Yelp 证据 |
| --- | --- | --- |
| 饿了 | restaurant / fast_food / food_court | Restaurants, Food, review |
| 渴了 | cafe / coffee / juice / convenience-like food | Coffee & Tea, Juice Bars, drinks |
| 想喝咖啡 | coffee_shop / cafe | Coffee & Tea |
| 想吃甜点 | bakery / dessert / ice cream | Desserts, Bakeries |
| 想找地方办公 | cafe / coworking-like / library-like | WiFi, quiet, outlet, laptop |
| 带孩子吃饭 | family-friendly restaurant | GoodForKids, family/kids reviews |
| 想省钱 | affordable restaurant | price range + cheap/budget review |
| 想停车方便 | restaurant/place with parking | BusinessParking + parking review |

这些可以进入主数据集，但要注意：Yelp 地点多在北美城市，与我们实际地图 provider 城市未必重合。若做中国城市 GT，Yelp 只适合作为“需求模板来源”，不适合作为 POI 真值来源。

### 3.3 语义偏好导航

Yelp 最有价值的是语义偏好：

| Subcategory | Yelp 证据 |
| --- | --- |
| `romantic_restaurant` | review: romantic/date night/anniversary；Ambience |
| `family_friendly_restaurant` | GoodForKids, RestaurantsGoodForGroups, family/kids review |
| `outdoor_seating_restaurant` | OutdoorSeating, patio/outdoor review |
| `quiet_work_place` | WiFi, NoiseLevel=quiet, laptop/work/outlet review |
| `place_with_parking` | BusinessParking, parking review |
| `healthy_vegan_place` | Vegan/Vegetarian/Healthy categories + review |
| `late_night_place` | hours + late/24h review |
| `pet_friendly_place` | DogsAllowed + dog/pet review |

建议：第一版语义偏好作为分析集或小规模人工审核集，后续再进入主 benchmark。

### 3.4 多目标导航

Yelp 可提供多目标组合的真实需求来源，但路线和 stop 选择仍需我们生成：

```text
coffee_then_work
eat_then_dessert
dinner_then_drinks
breakfast_then_shopping
parking_then_restaurant
restaurant_then_bar
```

多目标指令来自 Yelp 评论中的常见活动链，而不是 Yelp 直接给出路线。

### 3.5 中途增加途经点

Yelp 可提供“中途加点”的目标类型：

```text
add_coffee_stop
add_dessert_stop
add_takeout_stop
add_restaurant_stop
add_bar_stop
add_pet_friendly_stop
```

但 `route_update_add_via` 的 GT 必须由已有 route session + RouteServer 生成。

## 4. 评论/Tip 需求分布

对 review 抽样 500k、tip 全量 908,915 的规则统计：

| Demand group | review hits | tip hits |
| --- | ---: | ---: |
| quick_takeout | 82,505 | 32,205 |
| coffee_drink | 54,515 | 29,726 |
| dessert_snack | 51,299 | 17,514 |
| parking | 44,871 | 13,660 |
| group_friendly | 43,947 | 7,818 |
| healthy_vegan | 42,781 | 19,877 |
| quiet_work_wifi | 39,167 | 11,814 |
| family_kids | 37,401 | 12,280 |
| breakfast_brunch | 35,326 | 21,130 |
| cheap_budget | 19,488 | 8,655 |
| romantic_date | 15,313 | 4,369 |
| outdoor_seating | 13,239 | 6,453 |
| open_late_24h | 4,583 | 3,530 |
| pet_dog_friendly | 998 | 1,226 |

结论：

1. 高频需求明显集中在吃饭、外带、饮品、甜点、停车、家庭/多人、健康饮食、办公 WiFi。
2. Yelp 是商业活动需求数据，不是“所有城市导航需求”的均衡分布。
3. 如果按真实频率采样，数据会被餐饮压倒；benchmark 应做均衡采样。

## 5. 生成指令的推荐流程

不要直接把 review 文本当 instruction。评论是事后评价，不是用户导航请求。推荐流程：

```text
Yelp business/review/tip
-> 规则挖掘 demand group
-> 聚合到 business_id
-> 生成 semantic evidence
-> 映射到 UrbanNav category/subcategory
-> 生成 instruction templates
-> LLM paraphrase
-> 人工抽检
```

输出 instruction seed：

```json
{
  "id": "yelp_seed_quiet_work_wifi_000001",
  "source": "yelp_mined",
  "category": "implicit_need_navigation",
  "subcategory": "work_with_wifi",
  "instruction": "我想找个安静、有 Wi-Fi、适合坐下办公的地方。",
  "target_types": ["cafe", "coffee_shop", "library", "coworking_space"],
  "semantic_tags": ["quiet", "wifi", "work_friendly", "power_outlet"],
  "evidence": {
    "demand_group": "quiet_work_wifi",
    "signals": ["WiFi", "NoiseLevel", "review_terms"],
    "source_dataset": "yelp"
  },
  "gt_status": "needs_rebuild_with_route_provider"
}
```

## 6. 指令模板设计

### 6.1 POI/品牌导航

Yelp 用于提供真实类别/品牌名字：

```text
找一家附近的 {category}。
我想去附近评分不错的 {category}。
附近有没有 {brand_or_name}？
帮我找一家还在营业的 {category}。
```

建议只保留通用 category 和高频 brand/name，避免长尾商户名导致 benchmark 太依赖记忆。

### 6.2 隐式需求导航

```text
我有点饿了，找个附近能快速吃点东西的地方。
我想喝咖啡，附近哪里合适？
我想找个地方坐下办公，最好有 Wi-Fi。
带孩子出来吃饭，找个适合家庭的地方。
我想找个停车方便的地方吃饭。
我想吃点甜的，附近有什么合适的地方？
```

### 6.3 语义偏好导航

```text
我想找一家适合约会的餐厅。
找个安静一点、适合聊天的咖啡馆。
找一家有户外座位的餐厅。
找一家适合多人聚餐的地方。
找个价格不贵、评价还可以的地方吃饭。
```

### 6.4 多目标导航

```text
先买杯咖啡，然后找个地方办公。
先吃晚饭，再找个地方喝一杯。
先找个方便停车的地方，再去附近吃饭。
先吃早餐，然后去附近逛逛。
先买甜点，再找个咖啡店坐一会儿。
```

### 6.5 中途加点

```text
去目的地之前，中途加一个咖啡店。
当前路线中帮我加一个能外带吃的地方。
先绕去买点甜品，然后继续去原来的目的地。
中途加一个停车方便的餐厅。
```

## 7. 采样策略

不要按 Yelp 原始频率直接采样。推荐 benchmark 均衡采样：

| 类型 | 建议数量 |
| --- | ---: |
| `poi_navigation` | 800-1200 |
| `implicit_need_navigation` | 1000-1500 |
| `semantic_preference` 分析集 | 500-1000 |
| `multi_stop_route` | 500-800 |
| `route_update_add_via` | 300-500 |

第一阶段 smoke：

```text
每类 50-100 条
总计 300-500 条
```

第一版正式 dev/test：

```text
dev 500-1000
test 2000-3000
```

每个 subcategory 至少 50 条，低于 50 条不单独报告。

## 8. 关键质量控制

### 8.1 去重

需要按规范化 instruction 去重：

```text
lowercase
去标点
同义模板合并
限制同一 business_id 贡献数量
```

### 8.2 证据强度

每条语义需求应保存证据强度：

```json
{
  "evidence_strength": {
    "business_attribute": true,
    "review_match_count": 12,
    "tip_match_count": 2,
    "rating": 4.3,
    "review_count": 180
  }
}
```

推荐过滤：

```text
review_count >= 20
is_open = 1
semantic review/tip hits >= 2
```

### 8.3 语言多样性

Yelp 是英文数据，但 UrbanNav 需要中英文指令。做法：

```text
先生成英文 canonical instruction
再用 LLM 生成中文 paraphrase
保留同一 semantic label
```

每条 seed 可以生成 2-4 个 paraphrase，但同一 GT 不应无限复制，避免 benchmark 过拟合模板。

### 8.4 城市迁移

如果评估城市不在 Yelp 覆盖范围，Yelp 只能作为需求模板/分布来源：

```text
Yelp -> instruction seed / semantic tags
本地 POI provider -> gt_destination
RouteServer -> gt_route
```

如果评估城市就在 Yelp 覆盖范围，可使用 Yelp business 坐标作为 POI 候选，但仍需用 RouteServer 生成路线。

## 9. 与 GT 构建的衔接

Yelp 阶段输出不直接是最终 benchmark。它是 instruction seed：

```text
instruction seed
-> sample origin
-> search/map POI candidates
-> filter satisfying candidates
-> route all candidates
-> choose nearest satisfying candidate
-> write gt_destination / acceptable_destinations / gt_route
```

也就是说：

```text
Yelp 负责“人们通常会提出什么需求”
Map/Route provider 负责“这个需求在当前城市应该去哪”
```

## 10. 实现优先级

1. 扩展 `analyze_yelp_for_instruction_seeds.py`，输出候选 instruction seed JSONL，而不仅是 summary。
2. 增加 `--business-limit / --review-limit / --tip-limit / --min-evidence-count`。
3. 先生成 300-500 条 smoke seed，人工 review taxonomy 和文本质量。
4. 接入 LLM paraphrase，生成中英文指令。
5. 接入 RouteServer GT builder，重建 `gt_destination` 和 `gt_route`。

## 11. 8-9K Instruction Seed 采样方案

8-9K 条数据适合作为 `instruction seed` 规模，而不是一次性全部作为带 `gt_destination/gt_route` 的最终 benchmark。推荐先生成 8,500 条 Yelp-guided instruction seeds，再从中筛选和重建 2,000-3,000 条高质量 GT benchmark。

原因：

1. Yelp 能稳定提供需求和语义证据，但路线 GT 仍需地图 provider 重建。
2. 语义偏好和中途加点存在弱真值，需要 review。
3. 过多 paraphrase 如果共享同一个需求模板和目标，会降低 benchmark 有效多样性。

### 11.1 推荐总量

```text
target_total = 8500
range = 8000-9000
```

按任务类型均衡，而不是按 Yelp 原始频率采样。

| Category | Subcategories | 建议数量 | 说明 |
| --- | ---: | ---: | --- |
| `implicit_need_navigation` | 8 | 3200 | 可进入主 GT 构建，需求稳定 |
| `semantic_preference` | 7 | 2450 | Yelp 最强项，但需证据和 review |
| `multi_stop_route` | 4 | 1600 | 组合需求，需合成路线 GT |
| `route_update_add_via` | 4 | 1250 | 交互式扩展，主要由模板+Yelp 需求证据生成 |
| total | 23 | 8500 | 每个 subcategory 平均约 370 |

### 11.2 子类分配

`implicit_need_navigation`：

| Subcategory | 数量 |
| --- | ---: |
| `hungry` | 450 |
| `thirsty` | 400 |
| `coffee_shop` | 400 |
| `dessert_snack` | 400 |
| `work_with_wifi` | 450 |
| `family_kids` | 400 |
| `parking_lot` | 350 |
| `cheap_budget` | 350 |

`semantic_preference`：

| Subcategory | 数量 |
| --- | ---: |
| `romantic_restaurant` | 350 |
| `outdoor_seating_restaurant` | 350 |
| `family_friendly_restaurant` | 350 |
| `quiet_work_place` | 400 |
| `pet_friendly_place` | 250 |
| `healthy_vegan_place` | 400 |
| `late_night_place` | 350 |

`multi_stop_route`：

| Subcategory | 数量 |
| --- | ---: |
| `coffee_then_work` | 450 |
| `eat_then_drink` | 400 |
| `parking_then_restaurant` | 350 |
| `dessert_then_coffee` | 400 |

`route_update_add_via`：

| Subcategory | 数量 |
| --- | ---: |
| `add_coffee_stop` | 350 |
| `add_takeout_stop` | 350 |
| `add_dessert_stop` | 300 |
| `add_parking_friendly_restaurant` | 250 |

### 11.3 输出格式

沿用 CitySeeker conversion 的 instruction-only 格式，但改成 Yelp 来源字段：

```json
{
  "id": "yelp_instruction_implicit_need_navigation_work_with_wifi_000001",
  "dataset": "yelp_instruction_seed",
  "conversion_status": "accepted",
  "conversion_reason": "yelp_evidence_and_template",
  "conversion_confidence": 0.86,
  "category": "implicit_need_navigation",
  "category_zh": "隐式需求导航",
  "subcategory": "work_with_wifi",
  "instruction": "我想找个安静、有 Wi-Fi、适合坐下办公的地方。",
  "instruction_templates": [
    "I want to find a quiet place with Wi-Fi where I can work.",
    "我想找个安静、有 Wi-Fi、适合坐下办公的地方。"
  ],
  "target_types": ["cafe", "coffee_shop", "library", "coworking_space"],
  "ordered_target_types": null,
  "brand_aliases": null,
  "semantic_tags": ["quiet", "wifi", "work_friendly", "power_outlet"],
  "source": {
    "name": "yelp_open_dataset",
    "format": "Yelp",
    "demand_group": "quiet_work_wifi",
    "business_id": "optional",
    "city": "optional",
    "evidence_policy": "attribute_or_review_signal",
    "taxonomy": "cityseeker_refactored_v2",
    "gt_policy": "rebuild_with_poi_provider_and_routeserver"
  },
  "yelp_evidence": {
    "business_attribute": ["WiFi", "NoiseLevel"],
    "review_signal_count": 8,
    "tip_signal_count": 1,
    "rating": 4.2,
    "review_count": 130
  },
  "gt_build_plan": {
    "origin": "sample_from_target_urban_nav_region",
    "gt_destination": "rebuild_with_poi_provider",
    "gt_route": "rebuild_with_routeserver_or_graph_shortest_path",
    "ranking_policy": "nearest_satisfying_poi_by_route_distance"
  }
}
```

### 11.4 Accepted / Review / Reject

沿用 CitySeeker 转换输出的三路文件：

```text
task_instructions.accepted.jsonl
task_instructions.review.jsonl
task_instructions.reject.jsonl
summary.json
```

建议规则：

| Split | 条件 |
| --- | --- |
| accepted | template 与 taxonomy 明确匹配，且 Yelp 属性或评论证据足够 |
| review | 语义偏好、低频需求、证据弱、LLM paraphrase 语义可能漂移 |
| reject | 无法映射到目标类型、过度主观、路线/POI 不可重建 |

第一版 8,500 seed 的预期比例：

```text
accepted: 6000-7000
review: 1200-2000
reject: 300-800
```

### 11.5 Paraphrase 控制

每个 evidence-backed canonical seed 生成 2-4 条 paraphrase，不建议更多。

推荐语言比例：

```text
Chinese 60%
English 40%
```

原因是 Yelp 原始文本为英文，而目标应用需要中文需求表达。中文不能只是直译，应覆盖口语表达：

```text
我有点饿了，附近哪里能快点吃上饭？
找个能坐下办公、最好有 Wi-Fi 的地方。
去目的地前帮我加一家能外带的店。
```

### 11.6 去重和污染控制

必须限制以下重复：

1. 同一 business_id 贡献过多样本。
2. 同一 review/tip 生成过多 paraphrase。
3. 同一模板只替换少量词导致近重复。
4. 同一 subcategory 中某个城市或商户名称占比过高。

建议限制：

```text
max_seeds_per_business = 3
max_paraphrases_per_canonical = 4
max_same_template_ratio_per_subcategory = 0.15
min_instruction_normalized_edit_distance = 0.25
```

### 11.7 从 8.5K seed 到最终 benchmark

推荐不要把 8.5K 全部直接作为最终测试集。更合理的下游划分：

| Dataset | 数量 | 用途 |
| --- | ---: | --- |
| instruction seed pool | 8,000-9,000 | 需求覆盖池 |
| parser dev/test | 2,000-3,000 | 只评估指令解析 |
| GT rebuild candidates | 4,000-5,000 | 尝试构建目的地和路线真值 |
| final route benchmark | 2,000-3,000 | 通过 POI/route GT 校验后的最终评测 |
| hard/review set | 500-1,000 | 语义偏好、多目标、中途加点人工检查 |

这样可以保证最终 benchmark 不是“生成了多少条”，而是“多少条能稳定重建 GT 并可复现”。

### 11.8 当前生成脚本实现

脚本位置：

```bash
Agent_part/RouteServer/tools/dataset_processing/generate_yelp_instruction_seeds.py
```

核心实现：

| 目标 | 当前实现 |
| --- | --- |
| 每个子类 10-20 个基础模板 | `EXTRA_TEMPLATES`、`MULTI_STOP_EXTRA`、`ROUTE_UPDATE_EXTRA` 扩展模板池；smoke 中每个 demand group 为 12-20 个模板 |
| 接 LLM paraphrase | 支持两种方式：`--llm-paraphrase-jsonl` 接收外部结果；`--llm-paraphrase-mode generate` 直接调用 RouteServer 的 OpenAI-compatible LLM client |
| 控制模板占比 | `--max-template-ratio` 默认 0.15；脚本按 subcategory 限制同一 `base_template` 的使用次数 |
| 可审计 | 每条样本写入 `base_template`、`template_source`；`summary.json` 写入 `templates_per_demand_group`、`max_template_share_by_subcategory` |
| 多目标和中途加点自然表达 | 单独维护 `MULTI_STOP_EXTRA` 和 `ROUTE_UPDATE_EXTRA`，避免只有 “先 A 再 B” / “中途加 A” |

外部 paraphrase JSONL 格式：

```json
{"demand_group": "quiet_work_wifi", "subcategory": "work_with_wifi", "instruction": "找个能安静开电脑处理事情的地方。"}
{"demand_group": "coffee_then_work", "instruction": "我想先补一杯咖啡，然后去一个能坐下来工作的地方。"}
```

在线 LLM 生成命令：

```bash
LLM_PROVIDER=ark \
LLM_MODEL="doubao-seed-2-0-pro-260215" \
LLM_API_BASE_URL="https://ark.cn-beijing.volces.com/api/v3" \
LLM_API_STYLE=chat_completions \
python Agent_part/RouteServer/tools/dataset_processing/generate_yelp_instruction_seeds.py \
  --yelp-root "/media/ubuntu/data2/SR-Nav Dataset/archive/yelp_dataset" \
  --output-dir data/yelp_instruction_seed_llm_smoke \
  --target-total 400 \
  --review-limit 100000 \
  --tip-limit 100000 \
  --min-review-count 20 \
  --seed 13 \
  --llm-paraphrase-mode generate \
  --api-keys-file key.txt \
  --llm-concurrency 4 \
  --llm-limit 3 \
  --llm-paraphrases-per-subcategory 16 \
  --max-template-ratio 0.15
```

其中 `--llm-limit 3` 是 smoke，只让 3 个子类调用 LLM。正式生成时去掉 `--llm-limit`，脚本会为所有子类生成 Yelp-grounded paraphrase，并保存：

```text
llm_paraphrases.generated.jsonl
task_instructions.accepted.jsonl
task_instructions.review.jsonl
task_instructions.reject.jsonl
summary.json
```

smoke 命令：

```bash
python Agent_part/RouteServer/tools/dataset_processing/generate_yelp_instruction_seeds.py \
  --yelp-root "/media/ubuntu/data2/SR-Nav Dataset/archive/yelp_dataset" \
  --output-dir data/yelp_instruction_seed_smoke400_v2 \
  --target-total 400 \
  --review-limit 100000 \
  --tip-limit 100000 \
  --min-review-count 20 \
  --seed 13 \
  --max-template-ratio 0.15
```

当前 smoke 结果：

```text
generated_total: 400
accepted: 286
review: 114
rejected: 0
exact_unique: 400
normalized_unique: 400
template_source: manual_expanded 310, base 90
max_template_share_by_subcategory: all <= 0.143
```
