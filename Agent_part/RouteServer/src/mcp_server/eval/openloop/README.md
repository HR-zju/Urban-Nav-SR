# Open-loop 评测

这里维护 Stage I、Stage II CFM 和 VQA 的 open-loop package、registry、reporting
及公共评测协议。它与 agent-tools 在线评测和 UrbanNav benchmark 分开。

`benchmark_plugin_*` 与 `citywalker_benchmark_metrics.py` 是新增的模型无关 plugin
层：冻结数据集插件、target-free 模型适配器和指标插件三者的交换边界。它不替换现有
Stage I/II V2 contract；详细协议见
[`PLUGGABLE_OPENLOOP_BENCHMARK_V1.md`](../../../../docs/evaluation/PLUGGABLE_OPENLOOP_BENCHMARK_V1.md)。

`benchmark_plugin_review_base.py` 是 review-only 的可复用基类/工具层：它统一 score
horizon 的轨迹切分，并严格解析 public-only `native_adapter_trace` sidecar。通用 HTML
renderer 以此渲染二栏（普通模型）或三栏（native adapter audit）页面；trace/funnel 不属于
模型输入、hidden reference、prediction 或 canonical scorecard。

旧的顶层 `mcp_server.eval.openloop_*` 路径由兼容 shim 保留。
