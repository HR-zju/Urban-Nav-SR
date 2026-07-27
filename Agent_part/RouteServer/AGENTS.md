# RouteServer 代码说明

这些说明适用于 `Agent_part/RouteServer` 下的代码和文档。

本文件细化 `../../AGENTS.md` 以及 `../../../ResearchWorkflow/policies/` 下的共享组合层
规范。以下本地规则继续作为 RouteServer benchmark 和 evaluator 行为的权威。

## 代码文档

新增或修改 Python 代码时：

- 每个 public class 都必须有 docstring。
- 每个 public function 或 method 都必须有 docstring。
- Docstring 应包含：
  - 简短的用途说明。
  - 参数的 `Args:`。
  - 返回值的 `Returns:`。
  - function 可能抛出 meaningful exception 时的 `Raises:`。
- Dataclass docstring 应在 `Args:` 下说明 constructor fields。
- private helper 的行为不直观，或实现 benchmark 语义时，应提供简洁 docstring。
- 核心 algorithm、metric、ranking rule 和 geometry calculation 在适用时必须在
  docstring 中包含数学定义。
- 数学公式使用 LaTeX-style，例如：

```text
\[
SPL = RS \cdot \frac{L(P^*)}{\max(L(P^*), L(\hat P))}
\]
```

## 基准与评测代码

对于 dataset、benchmark 和 evaluator 代码：

- 分离 schema、IO、metrics、runner、exporter 和 CLI 职责。
- Evaluator 必须 offline 且 deterministic。
- Evaluation code 禁止调用 live map API、LLM 或可变外部服务。
- 使用 frozen cache 或 replay data 保证可复现。
- 优先通过 stable ID 匹配 prediction 和 ground truth；坐标 fallback 只能作为显式
  tolerance rule。
- tool-call trace 默认作为 diagnostic，除非 benchmark 明确定义 tool-call quality metric。
- 主 benchmark metric 集合保持精简；其他数值先作为 debug 或 slice analysis，除非明确提升。

## 架构与迭代管理

实现核心代码时：

- 显式管理 iteration；优先扩展稳定 interface，避免复制近似的 versioned implementation。
- 从最通用且最小的 base abstraction 开始，通过 inheritance、adapter 或小型
  composition layer 实现专用变体。
- 解耦 provider、backend、runner、evaluator、builder 和 visualization 职责。
- 不要把多个 feature version、provider-specific branch 和 shared utility 混入一个大脚本。
- 两个或以上脚本需要相同行为时，将 reusable utility 提取到 library module。
- CLI script 应是轻量 orchestration layer：解析参数、调用 library code、写入 artifacts。
- 新版本通过 class name、config field 或 strategy object 明确表达行为差异，不能把隐藏条件
  散落在无关代码中。

## UrbanNav 基准规则

对于 `instruction -> destination -> route` benchmark：

- GT case 在数据提供时必须保留 `acceptable_destinations`、`ranked_destinations`、
  `optimality_scope` 和 `cache_keys`。
- prediction 匹配非 rank-1 acceptable destination 时，route metrics 必须使用该
  matched destination 的 reference route 和 reference distance。
- predicted destination 是另一个 acceptable target 时，禁止把 route 与 rank-1 GT 比较。
- `destination_success`、`hit@5`、`route_success`、`median_route_ratio` 和 `spl`
  是主要 single-target metrics。
- Coordinate radius 参数是 fallback matching tolerance，不是 metric 名称。

## 测试

对于新增 benchmark/evaluator 行为：

- 为 schema parsing、IO roundtrip、destination matching、route success、SPL 和
  edge case 增加 focused unit tests。
- route metrics 依赖 matched destination 时，必须包含非 rank-1 acceptable
  destination matching 测试。
- 实现后运行 targeted tests。

## 修改纪律

- 修改范围保持在当前任务内。
- 不修改无关文件。
- 优先遵循已有项目模式，避免引入新 abstraction。
- 除非任务确有需要，否则避免 broad refactor。
- 脚本已经包含多个 version 或无关工具时，应先拆分再添加功能。
