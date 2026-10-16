# 实验记录：预热后开放服务及在线刷新条件

experiment_record_version: 1

## 实验元数据

| 字段 | 记录 |
| --- | --- |
| 实验 ID | r10_service_ready_20261016 |
| 状态 | passed（启动与受控在线刷新）；导航质量未验收 |
| 代码版本 | Agent_part 独立子仓当前改动，source_identity.json 记录 SHA256；模型沿用 r10_cache_service_continuous_20261015 冻结源码 |
| 数据版本 | r10_dual_rate_shadow_20261015 前五个已准备观测，保留源时间和媒体；本轮固定输入另存 requests.jsonl |
| 模型版本 | 沿用双频运行的模型 digest 89a5d147c5455332e7b754280eef6af3bd0d5101e12d30804260220a34f2a9d0 及 engine digest 824da2bc012467096aa1f155b52bb4d6bb9ab7caeb2315d6d2c94bf549b01888 |
| 环境 | A100 GPU6，维护环境 urbannav_ms_swift_v31 |
| 运行入口 | python -m urbannav_deploy.serve_warm --config model_config.json --warmup-request warmup.json --lifecycle lifecycle.jsonl |
| 产物根目录 | /lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation/runs/r10_service_ready_20261016 |
| 验证范围 | 五个记录观测的在线刷新工程用例，含人工设置的版本标记和指令变化；不评判导航成功率 |

## 实验的目的

修复加载后提前报告 ready 的启动边界。只在一次明确记录的预热成功后开放 HTTP 端口，清除预热 query 调度与读出缓存。
核验 initial、cache_reused 以及 instruction/route_revision/scene_revision 的在线刷新；只覆盖尚未验收的组合。

## 实验的 setting

旧 PID10872 已不存在，旧日志最后一条为成功健康读取。当前容器 PID1 启动时间为 2026-10-16 09:56:37，晚于旧运行；
容器重启解释了旧服务进程消失。容器重启的触发原因未知，不归因为模型异常。
GPU6 在恢复前重新检查空闲。模型、engine、预算、精度及缓存参数保持原设置；服务输出使用新目录，旧回执保留。
唯一新增启动行为是一次预热后才开放端口。服务计数包含预热，验收明确区分预热和业务响应。
业务输入五条：初始、保持条件、改变 route_revision、改变 scene_revision、改变 instruction；后续条目保持此前版本标记。
版本标记由测试明确构造，不声称来自真实地图检测；指令改变通过原模型 adapter 重建 prompt/digest，保持合同一致。
停止条件：任何原生错误立即报告，禁止重试请求；本次不是性能基准，不重复完整 clip 测试。

### 可视化与导出计划

RUN/visualization/visualization_manifest.json 启动前生成；lifecycle.jsonl 记录 loading/warming/warmed/ready/stop/closed。
RUN/scorecard.json 汇总刷新原因、query 年龄、响应形状与延迟；visualization/review.html 保存五条输出及轨迹供审阅。
推理服务只绑定 loopback，访问通过受信 SSH。无训练，TensorBoard 不适用。

## 记录实验结果

新增启动逻辑 CPU 两个用例通过；补充清除 query 读出缓存后仅复查受影响的成功路径。
原生服务已恢复，PID1601、GPU6、loopback 18716，模型与 engine 各加载一次。
一次启动预热后，五条新增用例全部通过，累计请求 6，零重试。每条身份、刷新原因、复用标记、年龄、有限 [32,3] 均符合预期。

| 用例 | 原生刷新原因 | 客户端耗时 | query 年龄 |
| --- | --- | --- | --- |
| 初始业务输入 | initial | 0.134732 秒 | 0 |
| 保持条件 | cache_reused | 0.115054 秒 | 0.195712 秒 |
| route_revision 改变 | condition_change | 0.158499 秒 | 0 |
| scene_revision 改变 | condition_change | 0.156370 秒 | 0 |
| instruction 改变 | condition_change | 0.176671 秒 | 0 |

### 效果好的结果

预热后才开放端口；首个业务输入 0.135 秒，没有再次承担此前 3.95 秒的首次执行成本。
该数据只是五个固定形状的新增用例，不扩大为持续延迟保证或真实传感器性能。
本次启动预热本身耗时 44.202 秒，其中 query forward 为 43.477 秒；模型加载约 21.665 秒。
从 lifecycle 的 loading 到 ready 为 66.225 秒（不含此前 Python 导入）。这部分成本被移到服务就绪之前，并未消失。
首次 forward 的具体算子初始化/编译归因尚未剖析，不据此推测热请求也需要 44 秒。

### 效果不好的结果

旧服务缺少退出记录；容器启动时间补充了重启证据。新入口及父进程监督记录未来退出状态；容器整体停止时仍可能缺少尾记录。

### 可视化与产物

启动计划见上述路径；所有推理 request/response 由底层服务保存在 RUN/service/requests。

## 分析实验结果，可能的原因

旧入口在模型加载后直接开放 /readyz，而第一次完整执行发生于第一个业务请求；这解释了就绪边界不含首次执行成本。
先完成一次完整执行可将该成本移到就绪之前。后续真实输入的首次尺寸变化是否仍有额外开销，需要按实际输入另行判断。
没有充分证据支持非零 query 年龄的导航语义质量；本次只验证刷新和输出合同。

## 用户讨论与决策记录

| 时间 | 用户讨论/关注点 | 最终决定 | 对实验的影响 | 未解决问题 |
| --- | --- | --- | --- | --- |
| 2026-10-16 | 完成真实采集及实物部署之前的流程 | 恢复服务、收尾启动和刷新，整理独立子仓 | 只在 Agent_part 修改，冻结模型复用 | 非零年龄轨迹质量需审阅；实物采集另行接入 |

## 接下来要做的实验

| 优先级 | 下一实验 | 假设 | 唯一主要变量 | 所需证据 | 通过/停止条件 | 依赖或授权 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 已保存轨迹离线对比 | 可以定位非零年龄引入的差异 | query 来源年龄 | 相同当前观测的 fresh/aged 输出和输入身份 | 无匹配对则明确证据缺口；不重复前向 | 已授权只读审阅 |
