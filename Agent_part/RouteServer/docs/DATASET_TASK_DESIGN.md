# Dataset Task Design for Instruction-to-Destination/Route

本文档参考 CitySeeker 的数据集设置，说明本仓库如何构建“需求指令 -> 目标真值 -> 路线真值”的评测/训练数据。

## 1. CitySeeker 的做法

CitySeeker 的核心样本不是单纯的问答，而是：

```text
需求指令 question
+ 起点 from
+ 目标 to
+ 参考路线 complete_route
+ 路线代价 total_weight / total_steps
+ 类型标签 category / folder / difficulty / city
```

典型 JSON：

```json
{
  "folder": "convenience store",
  "question_idx": 0,
  "question": "Please find the nearest convenience store.",
  "idx": 2,
  "from": "00001864.jpg",
  "to": "00002415.jpg",
  "total_weight": 131.16464774846568,
  "total_steps": 10,
  "complete_route": [
    {"filename": "00001864.jpg", "latitude": 39.91933865, "longitude": 116.3535388},
    {"filename": "00002415.jpg", "latitude": 39.91843821, "longitude": 116.3535685}
  ],
  "difficulty": "easy",
  "category": "question1",
  "city": "beijing1"
}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `question` | 用户需求指令 |
| `folder` | 需求子类或语义目标，例如 `convenience store`、`I am thirsty` |
| `category` | 需求大类，例如基础 POI、品牌、隐式需求 |
| `from` | 起点图节点 id / 图片名 |
| `to` | 目标图节点 id / 图片名 |
| `complete_route` | 从起点到目标的参考路线节点序列 |
| `total_weight` | 参考路线距离或图边权重 |
| `total_steps` | 参考路线步数 |
| `difficulty` | 按路线长度/步数划分的难度 |
| `city` | 城市场景 |

CitySeeker 的 `example/final_all_QA_trajectory.json` 中实际使用了这些类别：

| Category | 语义 | 示例 |
| --- | --- | --- |
| `question1` | 基础 POI | `Please find the nearest restaurant.` |
| `question2` | 顺序多目标 | `Please find the nearest restaurant and bank in order.` |
| `question4` | 品牌/特定商户 | `Please find the nearest Starbucks.` |
| `question5` | 交通枢纽 | `Please find the nearest bus station.` |
| `question6` | 潜在/间接 POI | `Please find the nearest restroom.` |
| `question7` | 隐式人类需求 | `I'm feeling thirsty ... find a nearby place.` |
| `question8` | 无障碍/包容性设施，第一版删除 | `Please find the nearest bank with an accessible entrance.` |
| `question9` | 语义偏好 | `Please find the nearest romantic restaurant.` |

注意：README 中写的是 7 类描述，实际数据文件里编号是 `question1/2/4/5/6/7/8/9`。这是命名遗留，不影响我们复用其设计思想。

## 2. 本仓库建议的数据单元

本仓库的任务应该拆成两级：

```text
Task A: Destination Recommendation
  user_instruction + origin -> gt_destination

Task B: Route Planning
  origin + gt_destination / predicted destination -> gt_route
```

每条样本保存为 JSONL：

```json
{
  "id": "hangzhou_yuquan_000001",
  "city": "hangzhou",
  "scene_id": "yuquan",
  "category": "poi_navigation",
  "subcategory": "convenience_store",
  "difficulty": "easy",
  "instruction": "请帮我找最近的便利店。",
  "instruction_templates": ["请帮我找最近的{target}。"],
  "origin": {
    "lat": 30.2645,
    "lng": 120.1180,
    "heading_deg": 35.0,
    "node_id": "frame_000120"
  },
  "gt_destination": {
    "name": "便利店",
    "type": "convenience_store",
    "lat": 30.2651,
    "lng": 120.1190,
    "provider_poi_id": "amap_xxx",
    "node_id": "frame_000245"
  },
  "candidate_destinations": [
    {"name": "便利店 A", "lat": 30.2651, "lng": 120.1190, "distance_m": 135.0}
  ],
  "gt_route": [
    {"lat": 30.2645, "lng": 120.1180, "node_id": "frame_000120"},
    {"lat": 30.2651, "lng": 120.1190, "node_id": "frame_000245"}
  ],
  "gt_waypoints": [
    {
      "coordinate": {"latitude": 30.2648, "longitude": 120.1184},
      "action": "go_straight",
      "instruction": "继续直行。",
      "distance_m": 40.0
    }
  ],
  "total_weight": 135.0,
  "total_steps": 8,
  "source": {
    "map_provider": "amap",
    "image_provider": "sidewalk_or_mcap",
    "route_provider": "amap",
    "generated_at": "YYYY-MM-DD"
  }
}
```

这个 schema 可直接被 `tools/evaluation/eval_destination_route.py` 消费。

## 3. 公开数据来源边界

当前任务不是采集街景图像，而是构建：

```text
human_need_instruction + origin
-> gt_destination
-> gt_route / gt_waypoints
```

公开数据通常只能覆盖其中一部分，不能直接得到完整三元组。

| 数据来源 | 能直接提供 | 不能直接提供 | 建议用途 |
| --- | --- | --- | --- |
| CitySeeker | 人类需求/POI 类指令、目标节点、参考路线、难度 | 不一定覆盖我们的地图 provider、城市和轨迹格式 | 参考 taxonomy、样本 schema、难度和评估口径 |
| Touchdown / Talk2Nav / Map2Seq 等 Outdoor VLN | 自然语言路线指令和演示轨迹 | 通常不是“需求推荐目的地”，而是“按路线走” | 可参考语言风格和轨迹格式，不建议直接当目的地推荐 GT |
| OpenStreetMap / Overture Places / Foursquare Places | POI 名称、类别、坐标、部分属性 | 人类需求指令、唯一正确目的地、路线轨迹 | 生成 `candidate_destinations` 和 `gt_destination` |
| OSRM / GraphHopper / AMap / Baidu / Google 路线 API | origin 到 destination 的路线、距离、时间、steps | 需求语义和目标选择 | 生成 `gt_route`、`gt_waypoints`、`total_weight` |
| LLM / 人工模板 | 多样化需求指令 | 地理真值 | 生成 `instruction`，再由 POI/路线系统绑定真值 |

因此可行路线是“半自动合成”：

```text
公开 POI/地图数据
+ 人类需求 taxonomy / templates / LLM paraphrase
+ 路线规划器
=> 指令-目的地-路线真值数据集
```

不要把公开 VLN 数据集直接等同于本任务。VLN 数据集多数回答的是“给定路线描述，沿轨迹走到终点”；我们要评估的是“从需求中推断应该去哪，再规划到那里”。

## 4. 需求类型 Taxonomy

建议先保留 CitySeeker 的大类，但换成本仓库更明确的命名：

| 我们的 category | CitySeeker 对应 | 任务定义 | 目标真值来源 |
| --- | --- | --- | --- |
| `poi_navigation` | `question1+4` | 基础 POI 或品牌商户导航 | POI type / brand alias |
| `implicit_need_navigation` | `question6+7` | Latent POI 和抽象人类需求导航 | need -> candidate type mapping |
| `multi_stop_route` | `question2` + 本仓库扩展 | 一次性提出多个目的地/活动并规划多段路线 | ordered POI list + `plan_route_with_via_points` |
| `semantic_preference` | `question9` | 主观偏好或语义描述 | POI + tag / review / LLM filter |
| `route_update_add_via` | 本仓库扩展 | 已有路线中途追加途经点 | route session + inserted via point |

### 4.1 CitySeeker 类型迁移建议

| CitySeeker 类型 | 建议 | 原因 |
| --- | --- | --- |
| `question1` 基础 POI | 合并进 `poi_navigation` | 和品牌商户同属“找某类/某个商户”的目的地推荐 |
| `question4` 品牌商户 | 合并进 `poi_navigation` | 品牌只是 POI 检索多了 alias/name 约束 |
| `question5` 交通枢纽 | 删除 | 当前任务不重点评估公交/地铁接驳，和一般 POI 检索差异不大 |
| `question6` 潜在/间接 POI | 合并进 `implicit_need_navigation` | 需要从需求推断宿主设施或候选类型，例如洗手间可落在商场/快餐店/公共设施 |
| `question7` 抽象需求 | 合并进 `implicit_need_navigation` | 这是“需求 -> 目的地类型”的核心能力 |
| `question2` 顺序多目标 | 合并进 `multi_stop_route` | CitySeeker 固定两类 POI 顺序和本仓库自然语言多目标路线，本质都是有序多站点路线 |
| `question8` 无障碍设施 | 删除 | 依赖 `wheelchair/accessible entrance` 等属性，公开 POI 覆盖不稳定，暂不纳入第一版任务 |
| `question9` 语义偏好 | 保留为 `semantic_preference`，但优先小规模审核 | `romantic/upscale/comfortable` 依赖评论、标签或人工判断，不是稳定地图真值 |

推荐第一阶段合并后的主任务类型：

```text
poi_navigation
implicit_need_navigation
multi_stop_route
route_update_add_via
```

第二阶段再加入：

```text
semantic_preference
```

因为语义偏好需要评论标签或人工审核，否则真值噪声会明显高于路线规划误差。无障碍类先删除，后续只有在目标城市具备可靠 accessibility 标签时再单独恢复。

### 4.2 绝对位置或外部属性相关类型

严格说，CitySeeker 的 `question1/2/4/5/6/7` 不依赖某个固定城市的绝对位置，可以迁移到任意城市，只要当地有对应 POI 和路线图。

迁移风险主要来自三类：

| 类型 | 风险 | 处理方式 |
| --- | --- | --- |
| 品牌子类 `poi_navigation` | 品牌具有地域性，例如某城市没有 `FamilyMart` 或 `Bank of China` | 按城市配置品牌白名单；无候选时写入 rejects |
| 语义偏好 `semantic_preference` | `romantic/upscale/comfortable` 不是地图基础字段 | 需要评论/评分/标签/人工或 LLM judge；第一版不要作为强 GT |

因此“无法直接迁移”的不是经纬度本身，而是依赖本地商业生态、属性覆盖和主观语义标签的类别。

### 4.3 本仓库新增的交互式任务类型

`multi_stop_route`：

```text
输入：一次性多目标指令 + origin
例子：先吃饭，再买日用品。
GT：gt_destination_sequence + gt_route_legs + merged gt_route
工具：decompose_itinerary_goals / recommend_itinerary_and_route / plan_route_with_via_points
```

`route_update_add_via`：

```text
输入：已有 route session + 当前路线 + 新增途经点指令
例子：在去目的地之前，中途加一个咖啡店。
GT：old_route_plan + inserted_via_point + updated_via_points + updated_route_plan
工具：create_route_session / add_via_point_to_session / add_via_point_to_route_plan
```

这两类应单独划分，因为评估对象不同：

- `multi_stop_route` 评估从零生成多目标序列和整条多段路线。
- `route_update_add_via` 评估在已有路线状态上做增量修改，重点是 session state、插入位置和重规划结果。

本仓库已提供一个可执行的配置草案：

```text
config/task_taxonomy_cityseeker.json
```

后续数据生成脚本应该从这个配置读取：

```text
    category
    category_zh
subcategory
cityseeker_category
templates
target_types / ordered_target_types
    brand_aliases
semantic_tags
ranking_policy
```

配置中的单个 item 形态如下：

```json
{
  "category": "implicit_need_navigation",
  "items": [
    {
      "subcategory": "thirsty",
      "instructions": [
        "我有点渴，帮我找附近可以买饮料的地方。",
        "I'm thirsty and would like something to drink. Help me find a nearby place."
      ],
      "target_types": ["convenience_store", "coffee_shop", "beverage_shop"],
      "ranking_policy": "nearest_satisfying_poi"
    }
  ]
}
```

## 5. 真值生成流程

整体流程按 CitySeeker 的思想执行，但把 `from/to/complete_route` 换成我们自己的 `origin/gt_destination/gt_route`：

```text
task_taxonomy_cityseeker.json
  -> 需求类型和指令模板

origin samples
  -> 从 SideWalk/MCAP trajectory 或地图采样点获得起点

POI provider
  -> 按 target_types / brand_aliases / attributes 检索候选目的地

route provider
  -> 对候选目的地计算真实路线距离

nearest satisfying target
  -> gt_destination

route provider / graph shortest path
  -> gt_route / gt_waypoints / total_weight / total_steps

templates + optional LLM paraphrase
  -> instruction

task_gt.jsonl
  -> 训练/评测真值
```

### 5.0.1 Stage 1：迁移 CitySeeker 需求指令

第一阶段先不调用地图 API，也不重新规划路线。CitySeeker 只作为需求指令和任务类型来源；不要直接复用它已经冻结的 `to` 和 `complete_route`，因为那是真值依赖它自己的城市、图节点和路径生成策略。我们只迁移 instruction、category/subcategory 和目标约束，后续再用本仓库的 POI provider 和 RouteServer 重新构建 `gt_destination` / `gt_route`。

分类不应只依赖硬编码规则。推荐流程是：

```text
CitySeeker row
  -> 规则生成 weak prior
  -> LVLM/LLM 按 taxonomy 输出 accepted / needs_review / rejected
  -> 结构化校验 category/subcategory
  -> 写入 instruction accepted/review/reject
```

运行：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

export LLM_PROVIDER=ark
export ARK_API_KEY="<your-key>"
export LLM_MODEL="doubao-seed-2-0-pro-260215"

python tools/dataset_processing/convert_cityseeker_to_urbannav.py \
  --classification-mode hybrid \
  --instruction-only \
  --input /home/ubuntu/WorkSpace/project/SR/Code/CitySeeker/example/final_all_QA_trajectory.json \
  --output-jsonl /tmp/urbannav_cityseeker/task_instructions.accepted.jsonl \
  --review-jsonl /tmp/urbannav_cityseeker/task_instructions.review.jsonl \
  --reject-jsonl /tmp/urbannav_cityseeker/task_instructions.reject.jsonl \
  --summary-json /tmp/urbannav_cityseeker/summary.json \
  --llm-review-jsonl /tmp/urbannav_cityseeker/llm_review_prompts.jsonl
```

`--classification-mode` 的含义：

```text
hybrid          # 推荐：规则接受明确样本，LVLM/LLM 审核不确定/拒绝样本
llm             # 每条样本都由 LVLM/LLM 分类，规则只作为 weak prior
llm_review_only # 只把规则 needs_review 的样本交给 LVLM/LLM
rule            # 完全离线规则 baseline，用于无 API key 的复现
```

规则先验：

```text
question1 + question4 -> poi_navigation
question6 + question7 -> implicit_need_navigation
question2             -> multi_stop_route，进入 review
question9             -> semantic_preference，进入 review
question5 + question8 -> reject
```

这张表不是最终判定器，只是给模型的弱先验。`question2` 进入 review 的原因是 CitySeeker JSON 只保存最终 `to` 和完整 `complete_route`，没有显式标出中间 POI stop，因此不能直接构造 `gt_destination_sequence` 和 `gt_route_legs`。`question9` 进入 review 的原因是语义偏好需要额外属性、评论、人工标签或多模态证据确认。

产物：

```text
task_instructions.accepted.jsonl  # 可进入本仓库 GT 重建流程的需求指令
task_instructions.review.jsonl    # 需要 LLM 或人工确认后再进入 GT 重建
task_instructions.reject.jsonl    # 当前 taxonomy 不覆盖或质量不合格
llm_review_prompts.jsonl     # 给外部 LLM 批量分类/修正的 prompt 输入
summary.json                 # 转换数量、类别分布、reject/review 原因
```

### 5.0 GT 制作算法

CitySeeker 的 `to` 和 `complete_route` 是离线预生成并冻结的。我们也必须离线制作真值，评估时只读取，不再动态查地图。

输入：

```text
task_taxonomy_cityseeker.json  # 需求类型、指令模板、目标 POI 类型
origin_regions.jsonl           # 可采样城市区域/网格/边界，不是固定 origin 真值
POI provider                   # OSM/Overture/AMap/Baidu 等
Route provider                 # OSRM/GraphHopper/AMap/Baidu 等
```

输出：

```text
task_gt.jsonl                  # 最终冻结真值，每行一个 case
task_rejects.jsonl             # 无候选、无路线、质量不合格样本
poi_cache.jsonl                # 每个 search request 的冻结 POI response
route_cache.jsonl              # origin -> candidate 的冻结路线 response
validation_report.json         # GT 最优性、格式和 cache 可追溯性检查
```

UrbanNav 第一版依赖 map API 生成路线，不需要构建 CitySeeker 式 city graph。CitySeeker
的 `from/to/complete_route` 思想可以参考，但本任务不要求维护街景节点、可走边、方位角和逐步动作。
origin 是 benchmark 构造变量，应服务于任务、目的地和路线多样性，而不是来自 Frodobots/SideWalk
等基础采样数据。

与 MobilityBench 类似，数据构建阶段必须把外部地图服务响应冻结成 replay cache。后续
RouteServer/baseline 运行和评估都应读取冻结数据，不能在评估阶段静默重查 live API。

单目标样本算法：

```text
for item in taxonomy.items:
  instruction = sample(item.templates)

  for region in origin_regions:
    origin_candidates = sample_points(region)
    candidates = poi_search(
      center=region.center or origin_candidate.latlng,
      target_types=item.target_types,
      brand_aliases=item.brand_aliases,
      required_attributes=item.required_attributes,
      semantic_tags=item.semantic_tags,
      radius=R
    )

    candidates = merge_dedupe_candidates_from_multi_query(candidates)

    for origin in origin_candidates:
      for candidate in top_k_by_direct_distance(candidates, origin):
        route = route_provider.route(origin, candidate)
        save route_cache

      valid = candidates with successful route and task constraints
      ranked_destinations = sort(valid, by=route.distance_m)
      gt_destination = ranked_destinations[0]
      acceptable_destinations = destinations within acceptance policy
      route_features = distance / steps / turn_count / directness / ambiguity

      keep only if route_features fill benchmark quotas
      write task_gt.jsonl row
```

多目标顺序样本算法：

```text
for ordered item in taxonomy.multi_stop_route:
  for region in origin_regions:
    origin_candidates = sample_points(region)
    for origin in origin_candidates:
      leg_1_candidates = search first target type near origin
      for d1 in top_k(leg_1_candidates):
        leg_2_candidates = search second target type near d1
        for d2 in top_k(leg_2_candidates):
          route_1 = route(origin, d1)
          route_2 = route(d1, d2)
          total_distance = route_1.distance_m + route_2.distance_m

      gt_destination_sequence = argmin(sequence, total_distance)
      gt_route = concat(route_1, route_2)
      keep only if multi-leg route features fill benchmark quotas
      write task_gt.jsonl row
```

关键约束：

- `gt_destination` 必须来自 POI provider 的真实候选，不能由 LLM 生成。
- `gt_route` 必须来自 route provider 或本地图最短路，不能用直线段代替正式真值。
- `nearest` 必须按路线距离算；直线距离只能用于 Top-K 预筛。
- `gt_destination` 必须等于 `ranked_destinations[0]`；这是固定 provider、搜索半径、候选上限和路线模式下的最优。
- `acceptable_destinations` 必须保存每个可接受目标自己的参考路线和参考路线距离。
- 所有候选、路线、ranked destinations、reject reason 都要落盘，保证后续可以复查。
- 评估集一旦生成，`gt_destination` 和 `gt_route` 必须冻结；地图数据更新后不能静默重算。
- 每个样本应保存 `cache_keys`，能从 `poi_cache.jsonl` 和 `route_cache.jsonl` 反查构建时使用的原始 response。
- 工具调用 trace 可用于诊断，但不能作为目的地/路线成功的必要条件。

### 5.1 准备采样区域和 origin 候选

第一版数据生成不要求 city graph。需要准备的是可采样的城市区域、网格或边界：

```text
region_id
city
scene_id / district / grid_id
bbox or center + radius
sampling_quota
target task coverage hints
```

来源可以是：

- 城市行政区、商圈、校园、街区 bbox。
- 地图 provider 返回的道路附近可导航点。
- 人工定义的 benchmark 区域。
- 后续如需视觉评测，可再把已生成 OD 样本对齐到街景/FPV/BEV 资源。

输出区域表：

```json
{
  "region_id": "hangzhou_yuquan_grid_001",
  "city": "hangzhou",
  "scene_id": "yuquan",
  "bbox": [120.1170, 30.2635, 120.1210, 30.2675],
  "sampling_quota": 50,
  "source": "manual_region_v1"
}
```

origin 最终在 GT builder 内部从区域中采样，并按任务可达性、路线距离、路线形态和候选歧义度筛选。
因此 `origin_candidates` 是中间产物，不是必须人工预先固定的 benchmark 输入。

### 5.2 生成需求指令

对 `task_taxonomy_cityseeker.json` 中每个 item，先使用模板生成基础指令：

```text
poi_navigation/restaurant:
  Please find the nearest restaurant.
  请帮我找最近的餐厅。

implicit_need_navigation/thirsty:
  I'm feeling thirsty and would like something to drink. Could you help me find a nearby place?
  我有点渴，帮我找附近可以买饮料的地方。
```

可选再用 LLM 改写，但必须保留结构化标签：

```json
{
  "instruction": "我想喝点东西，附近哪里可以买饮料？",
  "category": "implicit_need_navigation",
  "subcategory": "thirsty",
  "target_types": ["convenience_store", "cafe", "beverage_shop"]
}
```

注意：LLM 只负责语言多样化，不能决定真值目的地。

### 5.3 检索候选目标并构造 origin-task pairs

对每个区域、每种需求类型 \(c\)，先召回满足任务约束的 POI，再和区域内采样出的
origin candidates 组合。对于隐式需求，要用多 query 召回，避免只搜一个关键词造成漏召回：

```text
thirsty:
  convenience_store
  cafe
  beverage_shop
  drink
  coffee
```

过滤规则：

- 必须在可达半径内。
- 必须能规划路线。
- 对 `poi_navigation` 中的品牌子类，名称或 alias 必须匹配。
- 对 `semantic_preference`，可先用规则/标签粗筛，再用 LLM judge 或人工审核。
- 对 `multi_stop_route`，每一段都必须存在至少一个可达候选。

候选目标标准格式：

```json
{
  "name": "便利店 A",
  "type": "convenience_store",
  "lat": 30.2651,
  "lng": 120.1190,
  "provider_poi_id": "osm_node_123",
  "matched_rule": {
    "category": "implicit_need_navigation",
    "subcategory": "thirsty",
    "target_type": "convenience_store"
  }
}
```

### 5.4 选择 GT 目标

默认策略是“nearest satisfying target by route distance”：

\[
d^* = \arg\min_{d \in C(o_i,c)} \mathrm{RouteDistance}(o_i,d)
\]

不要只用直线距离，优先用路线距离：

```text
route_distance_m > direct_distance_m
```

如果 route provider 太慢，可先用直线距离取 Top-K，再对 Top-K 计算真实路线。
为了保证 rank-1 最优可审计，每条样本必须保存：

```text
optimality_scope
ranked_destinations
candidate_audit
reject_reason
```

示例：

```json
{
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
  "candidate_audit": {
    "searched_candidate_count": 18,
    "routed_candidate_count": 8,
    "rejected_candidate_count": 10
  },
  "ranked_destinations": [
    {
      "rank": 1,
      "provider_poi_id": "poi_a",
      "route_distance_m": 168.2,
      "score_breakdown": {"relevance": 1.0, "route_distance_m": 168.2}
    },
    {
      "rank": 2,
      "provider_poi_id": "poi_b",
      "route_distance_m": 210.5,
      "score_breakdown": {"relevance": 1.0, "route_distance_m": 210.5}
    }
  ],
  "cache_keys": {
    "poi_search": ["poi:hangzhou_yuquan_restaurant_000120"],
    "routes": ["route:hangzhou_yuquan_restaurant_000120:poi_a"]
  }
}
```

如果两个目的地路线代价差异很小，可同时进入 `acceptable_destinations`，但
`gt_destination` 仍为 rank-1。可接受集合由固定策略生成，例如：

```text
route_distance_m <= min(best_distance * 1.2, best_distance + 100m)
```

对 `multi_stop_route`，目标不是单个 \(d^*\)，而是有序目标序列：

```text
d_1^*, d_2^*, ... = argmin total_route_distance(origin -> d_1 -> d_2 -> ...)
```

### 5.5 生成 GT route / waypoints

对选中的 \(d^*\)：

```text
gt_route = MapProvider.get_route(origin, gt_destination, mode)
gt_waypoints = route steps / coarse waypoints
total_weight = route_summary.distance_m
total_steps = len(gt_waypoints) or graph shortest path edge count
```

如果使用已有图节点/轨迹图：

```text
complete_route = shortestPath(origin_node, target_node)
total_weight = sum(edge.distance)
total_steps = number_of_edges
```

如果使用地图 provider：

```text
complete_route = route_polyline sampled points
total_weight = route_summary.distance_m
total_steps = coarse_waypoints length
```

### 5.6 写入最终 GT 样本

单目标任务写成：

```json
{
  "id": "hangzhou_yuquan_poi_navigation_restaurant_000120",
  "split": "train",
  "city": "hangzhou",
  "scene_id": "yuquan",
  "category": "poi_navigation",
  "subcategory": "restaurant",
  "instruction": "请帮我找最近的餐厅。",
  "origin": {"lat": 30.2645, "lng": 120.1180, "heading_deg": 35.0, "node_id": "frame_000120"},
  "gt_destination": {"name": "餐厅 A", "type": "restaurant", "lat": 30.2658, "lng": 120.1193, "provider_poi_id": "osm_node_123"},
  "acceptable_destinations": [
    {
      "name": "餐厅 A",
      "type": "restaurant",
      "lat": 30.2658,
      "lng": 120.1193,
      "provider_poi_id": "osm_node_123",
      "reference_route_distance_m": 168.2,
      "reference_route": [{"lat": 30.2645, "lng": 120.1180}, {"lat": 30.2658, "lng": 120.1193}],
      "rank": 1
    }
  ],
  "ranked_destinations": [{"rank": 1, "name": "餐厅 A", "lat": 30.2658, "lng": 120.1193, "route_distance_m": 168.2}],
  "gt_route": [{"lat": 30.2645, "lng": 120.1180}, {"lat": 30.2658, "lng": 120.1193}],
  "gt_waypoints": [{"coordinate": {"latitude": 30.2649, "longitude": 120.1184}, "action": "go_straight", "distance_m": 42.0}],
  "total_weight": 168.2,
  "total_steps": 6,
  "difficulty": "medium",
  "optimality_scope": {
    "poi_provider": "osm",
    "search_radius_m": 1500,
    "candidate_limit": 30,
    "route_provider": "osrm",
    "route_mode": "foot",
    "cost": "route_distance_m",
    "builder_version": "urbannav_gt_v1"
  },
  "cache_keys": {
    "poi_search": ["poi:hangzhou_yuquan_poi_navigation_restaurant_000120"],
    "routes": ["route:hangzhou_yuquan_poi_navigation_restaurant_000120:osm_node_123"]
  },
  "source": {"taxonomy": "cityseeker_style_v1", "poi_provider": "osm", "route_provider": "osrm"}
}
```

多目标顺序任务增加 `gt_destination_sequence` 和 `gt_route_legs`：

```json
{
  "category": "multi_stop_route",
  "subcategory": "restaurant_then_bank",
  "instruction": "请先带我去最近的餐厅，然后去最近的银行。",
  "gt_destination_sequence": [
    {"type": "restaurant", "provider_poi_id": "osm_node_123"},
    {"type": "bank", "provider_poi_id": "osm_node_456"}
  ],
  "gt_route_legs": [
    {"from": "origin", "to": "osm_node_123", "distance_m": 168.2},
    {"from": "osm_node_123", "to": "osm_node_456", "distance_m": 220.4}
  ],
  "total_weight": 388.6
}
```

## 6. 难度划分

CitySeeker 使用 `easy/medium/hard`。我们建议按路线距离和步数联合划分：

```text
easy:
  total_weight <= 150m 或 total_steps <= 10

medium:
  150m < total_weight <= 400m 或 10 < total_steps <= 25

hard:
  total_weight > 400m 或 total_steps > 25
```

如果场景更大，可以按每个 city/scene 的分位数动态划分：

```text
easy   = bottom 40%
medium = 40%-80%
hard   = top 20%
```

推荐最终写入：

```json
{
  "difficulty": "medium",
  "difficulty_reason": {
    "total_weight": 238.4,
    "total_steps": 18,
    "policy": "distance_step_threshold_v1"
  }
}
```

## 7. 数据划分

不要随机按样本打散，否则同一个地点/目标可能同时出现在 train/test。建议：

### 7.1 按场景划分

```text
train: scenes A/B/C
val:   scene D
test:  scene E
```

适合检验跨区域泛化。

### 7.2 按起点节点划分

```text
同一个 origin node 只能出现在一个 split。
```

适合防止模型记忆某个起点附近的 POI。

### 7.3 按目标 POI 划分

```text
同一个 gt_destination.provider_poi_id 只能出现在一个 split。
```

适合评估目的地推荐泛化。

实际推荐：

```text
优先 scene split；
同一 scene 内再保证 origin 和 destination 不跨 split 泄露。
```

## 8. 质量控制

每条样本生成后必须检查：

```text
instruction 非空
origin 坐标合法
gt_destination 坐标合法
origin != gt_destination
gt_route 至少两个点
total_weight > 0
total_steps > 0
gt_route 起点接近 origin
gt_route 终点接近 gt_destination
目标类型与 category/subcategory 一致
gt_destination 等于 ranked_destinations[0]
ranked_destinations 按 route_distance_m 升序
acceptable_destinations 都有 reference_route_distance_m
acceptable_destinations 都有 reference_route
cache_keys 指向存在的 poi_cache / route_cache 记录
optimality_scope 完整记录 provider、半径、候选上限、route mode、cost、builder_version
```

建议自动写 `reject_reason`：

```json
{
  "id": "bad-case",
  "status": "rejected",
  "reject_reason": "route_endpoint_far_from_destination"
}
```

## 9. 评估对接

生成 GT 后，预测文件保存 RouteServer 输出：

```json
{
  "id": "hangzhou_yuquan_000001",
  "chosen_destination": {...},
  "candidate_destinations": [...],
  "route_polyline": [[120.1180, 30.2645], [120.1190, 30.2651]],
  "route_summary": {"distance_m": 142.0, "step_count": 8}
}
```

运行：

```bash
python tools/evaluation/eval_destination_route.py \
  --ground-truth /path/to/task_gt.jsonl \
  --predictions /path/to/predictions.jsonl \
  --destination-radius-m 30 \
  --output-json /tmp/urbannav_eval_summary.json \
  --output-jsonl /tmp/urbannav_eval_rows.jsonl
```

核心指标：

```text
destination_success
hit@5
route_success
median_route_ratio
spl
```

`--destination-radius-m` 只是坐标兜底匹配容差；主匹配逻辑应优先使用
`provider_poi_id/place_id`。当预测命中非 rank-1 但仍可接受目标时，路线成功和 SPL
使用该 matched destination 的 `reference_route_distance_m`。

`destination.mrr`、`mean_destination_error_m`、`curve_ade_m` 等只作为诊断指标，不进入主表。

## 10. 推荐落地顺序

1. 定义 `task_taxonomy.json`，固定 category/subcategory/instruction templates/target_types。
2. 定义 benchmark region 文件，包含 city/scene/bbox/quota。
3. 选定 POI 数据源和路线生成器，例如 OSM/Overture/AMap/Baidu + OSRM/GraphHopper。
4. 实现 GT builder smoke 版本，从一个小区域生成 100-300 个候选 OD case。
5. 对每个 task 做多 query POI 召回，并把原始 response 写入 `poi_cache.jsonl`。
6. 对 Top-K 候选计算 route distance，并把原始 route response 写入 `route_cache.jsonl`。
7. 选最近可达目标为 rank-1 GT，生成 `ranked_destinations`、`acceptable_destinations`、`cache_keys` 和 `candidate_audit`。
8. 运行 validator，输出 `validation_report.json`，不通过的样本写入 `task_rejects.jsonl`。
9. 用模板或 LLM paraphrase 生成多条同义需求指令，但不能改变冻结 GT。
10. 用 RouteServer 在 replay backend 上生成 predictions。
11. 跑 `eval_destination_route.py`，主表只报告 `destination_success / hit@5 / route_success / median_route_ratio / spl`。
12. 抽样人工检查 hard、implicit 和 semantic_preference 类，并在数据集卡中记录审核比例和 reject 统计。
