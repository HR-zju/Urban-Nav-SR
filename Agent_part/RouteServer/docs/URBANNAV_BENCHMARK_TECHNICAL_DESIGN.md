# UrbanNav Benchmark 技术设计

## 摘要

UrbanNav 评估智能体能否把开放式人类出行需求转化为合理目的地，并进一步生成可执行的步行导航路径。它不是单纯的问答任务，也不是给定路线描述后的视觉轨迹跟随任务。每个样例由用户指令、地理起点、冻结的可接受目的地集合，以及固定地图服务生成的一条或多条参考路线组成。评估对象包括三个相互关联的能力：需求理解、目的地选择和路线规划。

该 benchmark 遵循三个原则。第一，真值必须在离线阶段由可审计的地图和 POI 证据生成。第二，评估过程必须确定、可复现，不能依赖 live map API 的即时返回。第三，当一个指令存在多个合理答案时，评估应基于可接受目的地集合，而不是单一坐标点。

## 1. 任务定义

给定起点 \(o\)、自然语言指令 \(q\) 和固定地图环境 \(\Omega\)，智能体需要输出目的地 \(\hat d\)、可选的候选目的地排序列表 \(\hat C\)，以及路线 polyline \(\hat P\)：

\[
f_\theta(q,o;\Omega) \rightarrow (\hat d,\hat C,\hat P)
\]

benchmark 的真值样例表示为：

\[
x = (q,o,D_{\mathrm{acc}},d^*,P^*,\mathcal A,\Omega)
\]

其中，\(D_{\mathrm{acc}}\) 是可接受目的地集合，\(d^*\) 是 rank-1 参考目的地，\(P^*\) 是 rank-1 参考路线，\(\mathcal A\) 是构建审计信息，包括候选排序、路线距离、拒绝原因和 replay cache key。

UrbanNav 与 Touchdown、Talk2Nav 等街景 VLN 数据集不同。后者主要评估“给定路线描述后，沿视觉图或图像节点到达终点”的能力；UrbanNav 从需求或目标描述出发，要求系统先判断应该去哪，再规划如何到达。UrbanNav 也不同于 CitySeeker 的 embodied city graph 任务：CitySeeker 在构建好的城市图上给出图像节点路线，而 UrbanNav 使用地图服务进行 POI 检索和步行路线规划。

## 2. 与相关 benchmark 的关系

CitySeeker 对 UrbanNav 的主要启发是需求类型设计，包括显式 POI 导航、品牌商户搜索、有序多目标导航、潜在 POI 需求、抽象人类需求和语义偏好查询。UrbanNav 保留这种“需求驱动”的任务结构，但不沿用 CitySeeker 的 `from/to/complete_route` 图节点格式，而是使用地图服务生成的 `origin/gt_destination/gt_route`。

MobilityBench 对 UrbanNav 的主要启发是评估工程结构。地图服务会随时间变化，路线规划智能体也经常依赖工具调用。UrbanNav 因此在 benchmark 构建阶段冻结 POI 和路线响应，形成 replay cache；评估阶段只读取真值和预测文件。工具调用 trace 保留为诊断信息，但最终成功与否由目的地和路线结果决定。

Touchdown、Talk2Nav、StreetNav 等户外导航工作说明，城市导航不能只在单一区域或单一路线形态上评估。UrbanNav 因此采用多城市、分层采样和路线形态覆盖，但任务目标仍然是“需求指令 -> 目的地推荐 -> 路线规划”，而不是路线描述跟随。

## 3. 任务类型

benchmark 使用五类任务。

| Category | 定义 | 真值对象 |
| --- | --- | --- |
| `poi_navigation` | 显式 POI 或品牌请求，例如银行、餐厅、星巴克、中国银行。 | 单阶段目的地集合 |
| `implicit_need_navigation` | 由人类需求隐含 POI 类型，例如口渴、取现金、手机充电、看书。 | 可接受目的地集合 |
| `multi_stop_route` | 有序行程请求，例如先吃饭，再买日用品。 | 有序 stop 集合和多段路线 |
| `route_update_add_via` | 对已有 route session 做增量修改。 | 新增途经点和更新后路线 |
| `semantic_preference` | 带主观偏好或语义约束的请求，例如安静、适合办公、适合家庭聚餐。 | 经属性或人工审核的目的地集合 |

主 benchmark 报告 `poi_navigation`、`implicit_need_navigation`、`multi_stop_route` 和 `route_update_add_via`。`semantic_preference` 作为分析集，只有在数据源能提供可靠偏好属性或人工审核证据时，才进入正式主表。

## 4. 真值构建

### 4.1 候选检索

对于样例 \((q,o)\)，构建器首先将指令映射为结构化约束：

\[
z = \phi(q) = (\mathrm{category},\mathrm{subcategory},T,B,S,G)
\]

其中，\(T\) 是目标类型集合，\(B\) 是品牌 alias 集合，\(S\) 是语义标签集合，\(G\) 是多途经点任务中的有序目标列表。

POI provider 在固定 scope \(\Omega\) 下返回候选：

\[
C(o,z;\Omega)=\operatorname{POISearch}(o,z,\Omega)
\]

每个候选目的地都需要通过任务约束谓词：

\[
H(d,z)\in\{0,1\}
\]

对于显式 POI 任务，\(H\) 检查类型和可选品牌 alias；对于隐式需求任务，\(H\) 检查该 POI 类型是否能满足需求；对于语义偏好任务，\(H\) 必须依赖冻结属性、标签或人工审核证据。

### 4.2 路线感知排序

对每个满足约束的候选，构建器调用 route provider：

\[
R(o,d;\Omega)=\operatorname{RouteProvider}(o,d,\Omega)
\]

只保留可规划路线的候选：

\[
C_{\mathrm{valid}}=\{d\in C(o,z;\Omega)\mid H(d,z)=1 \land R(o,d;\Omega)\neq\varnothing\}
\]

rank-1 目的地由路线代价决定：

\[
d^*=\arg\min_{d\in C_{\mathrm{valid}}} c(R(o,d;\Omega))
\]

默认代价是步行路线距离：

\[
c(R)=L(R)=\sum_{i=1}^{n-1}\operatorname{hav}(p_i,p_{i+1})
\]

其中，\(\operatorname{hav}\) 表示 haversine 距离，\(P=(p_1,\ldots,p_n)\) 是路线 polyline。

对于带偏好约束的任务，可以使用包含相关性和偏好项的固定打分函数，但必须保存分项得分：

\[
s(d\mid q,o)=w_{\mathrm{rel}}\operatorname{rel}(d,q)
-w_{\mathrm{dist}}L(R(o,d))
+w_{\mathrm{pref}}\operatorname{pref}(d,q)
-w_{\mathrm{viol}}\operatorname{viol}(d,q)
\]

如果使用该形式，则 \(d^*\) 是 \(s(d\mid q,o)\) 的最大值对应目的地，`score_breakdown` 必须记录每个得分项。

### 4.3 可接受目的地集合

部分指令不存在唯一正确答案。例如“我有点渴”可以由便利店、饮品店或咖啡店满足。UrbanNav 用可接受目的地集合处理这类多合理答案：

\[
D_{\mathrm{acc}}=\{d_i\mid H(d_i,z)=1 \land A(d_i,d^*,q,o)=1\}
\]

对于“最近的 X”这类指令，\(A\) 通常较严格，默认只保留 rank-1，除非明确允许近似并列目标：

\[
L(R(o,d_i)) \le (1+\delta)L(R(o,d^*))
\]

对于开放式需求，\(A\) 可以纳入在固定路线距离和相关性策略下满足需求的候选。每个可接受目的地都必须保存自己的参考路线：

\[
\forall d_i\in D_{\mathrm{acc}},\quad P^*_{d_i}=R(o,d_i;\Omega)
\]

这点是评估设计的关键：如果预测选择了非 rank-1 但仍然合理的目的地，路线指标应使用该 matched destination 自己的参考路线，而不是强行使用 rank-1 路线。

### 4.4 多途经点真值

对于有序目标序列 \(G=(g_1,\ldots,g_K)\)，构建器从当前点逐段选择 stop：

\[
d_i^*=\arg\min_{d\in C_{\mathrm{valid}}(p_{i-1},g_i;\Omega)}L(R(p_{i-1},d))
\]

其中：

\[
p_0=o,\quad p_i=d_i^*
\]

完整参考路线由每段路线拼接得到：

\[
P^*=R(p_0,d_1^*)\oplus R(d_1^*,d_2^*)\oplus\cdots\oplus R(d_{K-1}^*,d_K^*)
\]

样例保存 `gt_destination_sequence`、`gt_route_legs` 和合并后的 `gt_route`。

### 4.5 Route Update 真值

route-update 任务评估已有路线状态上的增量修改。一个样例包含旧路线、当前位置 \(o_t\)、最终目的地 \(d_f\)、更新指令 \(q_t\)，以及新增途经点 \(v^*\)：

\[
(q_t,o_t,d_f,P_{\mathrm{old}})\rightarrow(v^*,P_{\mathrm{updated}}^*)
\]

更新后的路线必须从当前位置出发，经过新增途经点，并保留原最终目的地：

\[
P_{\mathrm{updated}}^*=R(o_t,v^*)\oplus R(v^*,d_f)
\]

## 5. 数据格式

单目标样例以 JSONL 存储，每行一个 case：

```json
{
  "id": "hangzhou_yuquan_000001",
  "city": "hangzhou",
  "scene_id": "yuquan",
  "category": "implicit_need_navigation",
  "subcategory": "thirsty",
  "instruction": "我有点渴，帮我找附近可以买饮料的地方。",
  "origin": {"lat": 30.2639, "lng": 120.1236},
  "gt_destination": {
    "name": "便利店 A",
    "type": "convenience_store",
    "lat": 30.2648,
    "lng": 120.1245,
    "provider_poi_id": "amap_xxx",
    "rank": 1,
    "reference_route_distance_m": 128.0
  },
  "acceptable_destinations": [
    {
      "name": "便利店 A",
      "type": "convenience_store",
      "lat": 30.2648,
      "lng": 120.1245,
      "provider_poi_id": "amap_xxx",
      "rank": 1,
      "reference_route_distance_m": 128.0,
      "reference_route": [[120.1236, 30.2639], [120.1245, 30.2648]]
    }
  ],
  "gt_route": [[120.1236, 30.2639], [120.1245, 30.2648]],
  "gt_distance_m": 128.0,
  "ranked_destinations": [
    {
      "rank": 1,
      "provider_poi_id": "amap_xxx",
      "route_distance_m": 128.0,
      "score_breakdown": {"route_distance_m": 128.0}
    }
  ],
  "optimality_scope": {
    "poi_provider": "amap",
    "route_provider": "amap",
    "search_radius_m": 1500,
    "candidate_limit": 30,
    "route_mode": "foot",
    "cost": "route_distance_m",
    "builder_version": "urbannav_gt_v1"
  },
  "cache_keys": {
    "poi_search": ["poi:hangzhou_yuquan_000001:thirsty"],
    "routes": ["route:hangzhou_yuquan_000001:amap_xxx"]
  }
}
```

`acceptable_destinations` 和 `ranked_destinations` 必须分开理解。前者定义正确答案集合，后者是构建和审计证据。

## 6. 采样策略

UrbanNav 不应从无关视觉数据的采样分布中直接继承 origin。origin 是 benchmark 构造变量，应该服务于目的地歧义度、路线长度、路线形态和城市覆盖。

采样按以下维度分层：

| 维度 | 目的 |
| --- | --- |
| 城市 | 避免过拟合单一路网或 POI 分布 |
| 区域类型 | 覆盖校园、商圈、居民区、景区、交通设施周边等场景 |
| 任务类型 | 平衡显式 POI、隐式需求、多途经点和 route update |
| 路线长度 | 覆盖短、中、长步行路线 |
| 路线形态 | 覆盖直行、转弯、绕行、过街、多段路线 |
| 候选歧义度 | 同时包含明确最近目标和近似并列候选 |
| provider 证据质量 | 剔除 POI ID 缺失、类别不稳定或路线失败样例，除非专门用于 fallback 测试 |

难度可由路线长度、转弯数量和候选歧义度组成：

\[
\operatorname{difficulty}(x)=
\alpha\operatorname{bin}(L(P^*))
+\beta\operatorname{bin}(\operatorname{turns}(P^*))
+\gamma\operatorname{bin}(|D_{\mathrm{acc}}|)
+\eta\operatorname{bin}(\operatorname{nearTie}(x))
\]

这些系数不参与最终评分，只用于平衡采样和分组报告。

## 7. 预测格式

runner 输出与真值分开保存：

```json
{
  "id": "hangzhou_yuquan_000001",
  "chosen_destination": {
    "provider_poi_id": "amap_xxx",
    "name": "便利店 A",
    "type": "convenience_store",
    "lat": 30.2648,
    "lng": 120.1245
  },
  "candidate_destinations": [
    {"provider_poi_id": "amap_xxx", "name": "便利店 A"}
  ],
  "route_polyline": [[120.1236, 30.2639], [120.1245, 30.2648]],
  "route_summary": {"distance_m": 130.0},
  "trace": {
    "tool_calls": [],
    "latency_ms": 0,
    "token_usage": {}
  }
}
```

评估器使用 `id` 对齐预测和真值。工具调用过程不要求与 GT 构建过程一致，除非后续版本单独定义工具调用质量指标。

## 8. 评估方法

### 8.1 目的地匹配

如果预测目的地与真值目的地具有相同稳定 provider ID，则判定为同一目的地：

\[
\operatorname{match}(\hat d,d)=
\mathbb{1}[\operatorname{id}(\hat d)=\operatorname{id}(d)]
\]

当稳定 ID 缺失时，fallback 匹配要求名称兼容、类型兼容，且坐标距离小于配置容差 \(\tau\)：

\[
\operatorname{match}_{\mathrm{fallback}}(\hat d,d)=
\mathbb{1}[
\operatorname{name}(\hat d)\sim\operatorname{name}(d)
\land
\operatorname{type}(\hat d)\sim\operatorname{type}(d)
\land
\operatorname{dist}(\hat d,d)\le\tau
]
\]

该容差只用于处理 provider 坐标偏移、入口点差异或路网吸附差异，不是成功语义本身。

### 8.2 单目标指标

目的地成功率：

\[
DS=\mathbb{1}[\exists d_j\in D_{\mathrm{acc}},\operatorname{match}(\hat d,d_j)=1]
\]

Top-5 召回：

\[
Hit@5=\mathbb{1}[\exists \hat d_i\in \hat C_{1:5},\exists d_j\in D_{\mathrm{acc}},
\operatorname{match}(\hat d_i,d_j)=1]
\]

路线成功率：

\[
RS=DS\land \hat P\neq\varnothing \land
\operatorname{dist}(\operatorname{end}(\hat P),d_{\mathrm{matched}})\le\tau_{\mathrm{end}}
\]

路线效率：

\[
route\_ratio=\frac{L(\hat P)}{L(P^*_{d_{\mathrm{matched}}})}
\]

SPL：

\[
SPL=RS\cdot
\frac{L(P^*_{d_{\mathrm{matched}}})}
{\max(L(P^*_{d_{\mathrm{matched}}}),L(\hat P))}
\]

单目标主表包含以下指标：

| 指标 | 含义 |
| --- | --- |
| `destination_success` | Top-1 目的地是否命中任一可接受目的地 |
| `hit@5` | 候选列表前五项是否召回任一可接受目的地 |
| `route_success` | 路线是否到达 matched acceptable destination |
| `median_route_ratio` | 成功路线中的路线效率中位数 |
| `spl` | 到达成功率和路线效率的综合指标 |

### 8.3 多途经点指标

设 \(G_i\) 是第 \(i\) 段的可接受 stop 集合，\(\hat d_i\) 是预测 stop：

\[
stop\_recall=\frac{1}{K}\sum_{i=1}^{K}
\mathbb{1}[\exists d\in G_i,\operatorname{match}(\hat d_i,d)=1]
\]

\[
sequence\_success=
\mathbb{1}[\forall i,\exists d\in G_i,\operatorname{match}(\hat d_i,d)=1]
\]

\[
multi\_stop\_SPL=
sequence\_success\cdot leg\_route\_success\cdot
\frac{L(P^*)}{\max(L(P^*),L(\hat P))}
\]

多途经点报告包含 `multi_stop_sequence_success`、`multi_stop_stop_recall`、`multi_stop_leg_route_success_rate` 和 `multi_stop_spl`。

### 8.4 Route Update 指标

route-update 成功需要同时满足三个条件：

\[
route\_update\_success =
via\_success \land final\_destination\_preserved \land updated\_route\_success
\]

其中：

\[
via\_success=\mathbb{1}[\operatorname{match}(\hat v,v^*)=1]
\]

\[
final\_destination\_preserved=
\mathbb{1}[\operatorname{match}(\hat d_f,d_f)=1]
\]

`updated_route_success` 使用更新后的路线和 matched destination 计算。

## 9. 可复现性和验证

正式发布的 benchmark split 目录包含：

```text
data/benchmark/<version>/
  task_gt.jsonl
  task_rejects.jsonl
  poi_cache.jsonl
  route_cache.jsonl
  validation_report.json
  predictions/
```

评估器只读取 `task_gt.jsonl` 和 prediction 文件。replay cache 用于 runner 和审计，不允许在评估时动态改写真值。

validator 检查以下内容：

| 检查项 | 要求 |
| --- | --- |
| Schema 合法性 | 必填字段可以解析为 UrbanNav case 对象 |
| 最优性 | `gt_destination` 等于 `optimality_scope` 下的 `ranked_destinations[0]` |
| 路线证据 | 每个 acceptable destination 都有 reference route 和 reference distance |
| 路线几何 | `gt_route` 起点接近 `origin`，终点接近 `gt_destination` |
| Cache 可追溯性 | `cache_keys` 能在 POI 和 route replay cache 中找到 |
| Reject 覆盖 | 被丢弃候选或失败样例包含 reject reason |

## 10. 报告方式

主表报告总体指标和切分指标：

```text
overall
by_category
by_subcategory
by_city
by_difficulty
```

主表应保持紧凑。`mean_destination_error`、`mrr`、`curve_ade/mde`、`turn_edit_distance`、工具调用合规性等诊断量可用于分析，但不应混入主 benchmark 分数，除非后续版本明确将它们提升为主指标。

## 参考文献

1. CitySeeker: How Do VLMs Explore Embodied Urban Navigation With Implicit Human Needs? https://arxiv.org/abs/2512.16755
2. MobilityBench: A Benchmark for Evaluating Route-Planning Agents in Real-World Mobility Scenarios. https://arxiv.org/abs/2602.22638
3. Touchdown: Natural Language Navigation and Spatial Reasoning in Visual Street Environments. https://arxiv.org/abs/1811.12354
4. Talk2Nav: Long-Range Vision-and-Language Navigation with Dual Attention and Spatial Memory. https://arxiv.org/abs/1910.02029
5. Learning To Follow Directions in Street View. https://arxiv.org/abs/1903.00401
