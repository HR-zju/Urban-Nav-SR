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

## 4. 指令集合生成

正式 benchmark 不直接使用采集脚本中的手写任务模板。所有用户指令先由任务 taxonomy、城市配额、任务族配额和 provider ontology 编译为一个冻结的 instruction set，然后 GT builder 只消费该 instruction set 中的任务定义。这样可以把“benchmark 任务设计”和“地图采集执行”解耦，避免因为调试脚本或 smoke config 造成数据分布漂移。

设任务族配额为 \(\{N_c\}_{c\in C}\)，某任务族 \(c\) 下的 taxonomy 子类集合为 \(S_c\)，每个子类权重为 \(w_s\)。子类计划样本数为：

\[
N_s=\operatorname{round}\left(N_c\frac{w_s}{\sum_{j\in S_c}w_j}\right)
\]

城市配额 \(\{M_u\}_{u\in U}\) 独立展开为 city slots，并与任务 slots 交错分配。最终指令集合为：

\[
\mathcal I=\{(q_i,u_i,c_i,s_i,T_i,G_i)\}_{i=1}^{N}
\]

其中 \(q_i\) 来自 taxonomy 模板，\(u_i\) 是计划城市，\(T_i\) 是单目标任务的目标类型集合，\(G_i\) 是多途经点任务的有序目标类型序列。provider ontology 只负责把 \(T_i\) 或 \(G_i\) 映射到 OSM/TomTom/AMap 等 provider 的检索标签，不允许在代码里重新定义用户指令。

当前实现入口为：

```bash
PYTHONPATH=src python tools/dataset_processing/build_urbannav_instruction_set.py \
  --taxonomy config/yelp_demand_taxonomy.json \
  --distribution config/urbannav_10k_distribution_plan.json \
  --provider-map config/urbannav_target_type_provider_map.json \
  --output-root data/benchmark/urbannav_instruction_set_full_en_v1 \
  --language en \
  --overwrite
```

该入口写出：

```text
instruction_set.jsonl                  # 逐样本指令计划，覆盖完整 10K 分布
task_catalog.json                      # taxonomy 子类级任务目录
collection_config_full_single_stop.json # 单目标 GT builder 可消费的采集配置
manifest.json                          # 城市、任务族、子类和模板分布审计
```

`multi_stop_route` 和 `route_update_add_via` 不应被单目标 collector 当作普通 POI 任务处理。它们在 `instruction_set.jsonl` 中保留完整计划，由对应的 multi-stop builder 和 route-update builder 构造真值。

## 5. 真值构建

### 5.1 候选检索

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

### 5.2 路线感知排序

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

### 5.3 可接受目的地集合

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

### 5.4 多途经点真值

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

### 5.5 Route Update 真值

route-update 任务评估已有路线状态上的增量修改。一个样例包含旧路线、当前位置 \(o_t\)、最终目的地 \(d_f\)、更新指令 \(q_t\)，以及新增途经点 \(v^*\)：

\[
(q_t,o_t,d_f,P_{\mathrm{old}})\rightarrow(v^*,P_{\mathrm{updated}}^*)
\]

更新后的路线必须从当前位置出发，经过新增途经点，并保留原最终目的地：

\[
P_{\mathrm{updated}}^*=R(o_t,v^*)\oplus R(v^*,d_f)
\]

## 6. 数据格式

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

## 7. 采样策略

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

## 8. 预测格式

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

## 9. 评估方法

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

## 10. 可复现性和验证

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

## 11. Agent 测评框架

UrbanNav 的评估对象不是 GT builder，而是在冻结 benchmark 上运行的导航智能体。二者的输入相同，职责不同：

| 阶段 | 输入 | 可访问资源 | 输出 | 作用 |
| --- | --- | --- | --- | --- |
| GT 构建 | \(q,o,\Omega\) | live provider、候选 POI、路线服务、采样配置 | \(D_{\mathrm{acc}},d^*,P^*,\mathcal A\) | 生成可审计真值 |
| Agent 推理 | \(q,o\) | replay backend 或指定 live tools | \(\hat d,\hat C,\hat P,\hat T\) | 生成系统预测 |
| Evaluator | GT 和 prediction | 冻结 JSONL | 指标行和汇总表 | 比较结果，不重查地图 |

其中 \(\hat T\) 是工具调用、LLM 输出、延迟和 token 等诊断 trace。它用于定位错误来源，但不改变主指标定义。

当前 Agent 框架按以下模块组织：

```text
instruction
  -> intent / goal parser
  -> POI candidate retriever
  -> candidate ranker
  -> route planner
  -> route-update controller
  -> structured prediction
```

单目标任务中，Agent 需要从指令中得到目标约束，检索候选目的地，选择首选目的地并规划路线。多途经点任务中，Agent 需要先把指令分解为有序目标序列，再逐段从当前点规划下一站。route-update 任务中，Agent 还需要读取已有路线状态，区分“替换目的地”和“追加途经点”两类更新。

为了避免把地图服务波动误计为模型能力变化，正式对比优先使用 replay runner。replay runner 只读取 GT 构建阶段冻结的 POI 和 route cache，但仍要求 Agent 自己完成意图解析、候选选择和路线组合。live runner 用于系统联调和真实 API smoke，不作为跨模型主表的默认入口。

Agent 的错误可以分解为三类：

| 错误类型 | 表现 | 对应指标 |
| --- | --- | --- |
| 需求理解错误 | 选中不满足 \(H(d,z)\) 的 POI | `destination_success` 下降 |
| 候选召回错误 | 正确目的地不在候选列表 | `hit@5` 下降 |
| 路线规划错误 | 目的地正确但路线不到达或绕路 | `route_success`、`median_route_ratio`、`spl` 下降 |

这种分解也是 UrbanNav 与普通 POI 推荐或静态路线规划 benchmark 的区别：系统必须同时完成语义约束理解、空间候选选择和可执行路线生成。

## 12. RouteServer Agent Baseline

除 benchmark 本身外，我们实现了一个与 UrbanNav 任务定义对齐的 RouteServer Agent baseline。该 baseline 不把 GT 构建器作为预测器使用，而是模拟真实导航系统的决策链路：从自然语言需求出发，解析意图，检索候选目的地，基于语义与路线代价选择目标，并输出可执行路线。它的设计目标不是追求特定地图 provider 上的启发式最优，而是提供一个结构清晰、可复现、可替换模块的 agentic baseline。

### 11.1 系统结构

RouteServer Agent 由五个核心模块组成：

| 模块 | 作用 | 主要输出 |
| --- | --- | --- |
| Intent Parser | 将开放式需求转化为可检索的目标约束 | POI 类型、关键词、半径、候选上限 |
| Goal Decomposer | 将多途经点指令分解为有序子目标 | \((g_1,\ldots,g_K)\) |
| Candidate Retriever | 调用 replay 或 live provider 获取候选 POI | \(\hat C\) |
| Candidate Ranker | 结合语义相关性与路线可达性重排候选 | \(\hat C_{\mathrm{ranked}}\) |
| Route Planner | 对候选目标或途经点逐段规划路线 | \(\hat P\) |

单目标任务中，Agent 首先从指令 \(q\) 得到结构化意图 \(\hat z\)，然后在起点 \(o\) 附近检索候选集合 \(\hat C\)。候选选择有两种策略：`route_distance` 直接选择可规划路线中距离最短的候选；`semantic_then_route` 先用语义重排约束候选顺序，再选择第一个可规划目标。前者更接近传统地图工具，后者更适合处理“适合办公”“可以买饮料”这类语义需求。

多途经点任务采用递推式规划。Agent 先将指令分解为有序目标序列，再从当前位置逐段执行“检索、排序、规划”。每完成一段，当前位置更新为该段选择的目的地。这样可以避免只对最终终点规划，而忽略中间停靠点的错误。

Route-update 任务把当前路线状态作为输入的一部分。Agent 需要从更新指令中判断是替换目的地，还是在保留最终目的地的情况下追加途经点。对于追加途经点，预测路线必须从当前位置出发，先经过新增目标，再到达原最终目的地；对于替换目的地，预测路线只需要到达新的目标。该设计使 route update 不退化为普通单点导航。

### 11.2 Baseline 模型

RouteServer Agent baseline 是一个工具增强的神经符号模型。语言模型负责把开放式指令压缩为结构化导航意图或候选偏好，地图工具负责提供外部世界状态，最终决策由可解释的路线感知选择函数完成。给定指令 \(q\) 和起点 \(o\)，baseline 的预测过程为：

\[
\hat z=\phi_\theta(q),\quad
\hat C=\operatorname{Retrieve}(o,\hat z),\quad
\hat P_i=\operatorname{Route}(o,\hat d_i),\quad
\hat d=\pi(\hat C,\{\hat P_i\},q)
\]

其中 \(\phi_\theta\) 可以是规则解析器，也可以是 LLM intent parser；\(\pi\) 是目标选择策略。当前实现包含三类可报告 baseline：

| Baseline | 语言模型作用 | 目标选择 |
| --- | --- | --- |
| RouteServer-Rule | 不调用 LLM，使用关键词和类别规则解析意图 | 在可规划候选中选择路线最短目标 |
| RouteServer-LLM-Intent | LLM 输出结构化 intent，包括关键词、类别和搜索半径 | 在可规划候选中选择路线最短目标 |
| RouteServer-LLM-Rerank | LLM 同时解析 intent，并对候选做语义重排 | 按语义顺序选择第一个可规划目标 |

RouteServer-Rule 是最小可复现 baseline，用于衡量地图检索和路线规划本身能达到的下限。RouteServer-LLM-Intent 评估语言模型是否能把隐式需求转换为正确的 POI 查询。RouteServer-LLM-Rerank 进一步评估模型是否能在多个可达候选中识别更符合指令语义的目标。

在 `route_distance` 策略下，Agent 使用路线代价选择目标：

\[
\hat d=\arg\min_{\hat d_i\in \hat C_{\mathrm{routeable}}} L(\hat P_i)
\]

在 `semantic_then_route` 策略下，LLM 先给出候选顺序 \(\sigma\)，Agent 选择该顺序中第一个可规划候选：

\[
\hat d=\hat d_{\sigma(j)},\quad
j=\min\{k\mid \hat P_{\sigma(k)}\neq\varnothing\}
\]

这两个策略对应两种不同能力边界：前者强调路线效率，后者强调语义适配。所有 baseline 都输出同一 prediction schema，因此可以用完全相同的 evaluator 比较。

### 11.3 Replay 与 Live 两种执行模式

RouteServer Agent 支持 replay 和 live 两种 backend。二者共享同一套意图解析、候选排序和路线组合逻辑，差别只在工具响应来源。

| 模式 | 工具响应来源 | 用途 |
| --- | --- | --- |
| replay | GT 构建阶段冻结的 POI 和 route cache | 主实验、回归测试、跨模型公平比较 |
| live | 当前地图 provider 实时返回 | 系统联调、真实 API smoke、部署前检查 |

replay 模式并不等价于 oracle。它只提供冻结工具返回，Agent 仍然需要自己决定查询意图、候选顺序、目标选择和路线组合。Oracle runner 只在 evaluator contract test 中使用，会直接复制 rank-1 GT，不作为 baseline 报告。

这种设计把“地图数据是否变化”和“Agent 是否做出正确决策”分离开来。正式实验中，不同模型面对相同的候选证据和路线证据；live 模式只用于检查系统在真实 provider 上是否能稳定运行。

### 11.4 与 GT 构建器的边界

GT 构建器和 RouteServer Agent 使用相似的地图工具，但优化目标不同。GT 构建器离线枚举候选、验证约束、计算路线，并保存审计证据；Agent 在线产生一个预测结果，并可能因为意图解析、候选召回、排序或路线组合错误而失败。

| 项目 | GT 构建器 | RouteServer Agent |
| --- | --- | --- |
| 目标 | 构造可审计真值 | 生成系统预测 |
| 候选处理 | 枚举并过滤可接受候选 | 根据解析意图检索候选 |
| 最优性 | 在 `optimality_scope` 内定义 rank-1 | 不知道 rank-1，只能基于工具证据选择 |
| 多合理答案 | 写入 \(D_{\mathrm{acc}}\) | 输出一个首选目标和候选列表 |
| 输出用途 | benchmark 数据 | model prediction |

因此，RouteServer Agent 的高分并不来自复用 GT，而来自在相同任务约束下重新完成需求理解、候选选择和路线规划。这个边界是 UrbanNav 评估成立的前提。

## 13. 报告方式

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
