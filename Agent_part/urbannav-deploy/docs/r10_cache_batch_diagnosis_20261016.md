# 实验记录：同一 fresh 输入的视觉缓存批次定位

experiment_record_version: 1

## 实验元数据

| 字段 | 记录 |
| --- | --- |
| 实验 ID | r10_cache_batch_diagnosis_20261016 |
| 状态 | 逐图修复三分支原生一致性通过；导航质量未验收 |
| 代码版本 | 独立 Agent_part 子仓 ca126f6 加本次诊断工具；源码 SHA256 固定在运行包 |
| 数据版本 | dual-rate-0046；与 r10-cache-continuous-20261015-0018 的完整 row 和 row_digest 相同 |
| 模型版本 | 冻结 r10 模型、engine 和维护环境，配置从双频运行继承 |
| 环境 | A100 GPU4（启动前须空闲）；GPU6 常驻服务保持 |
| 运行入口 | tools/diagnose_cache_batch.py --root RUN --baseline ORIGINAL_DUAL_RUN |
| 产物根目录 | /lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation/runs/r10_cache_batch_diagnosis_20261016 |
| 验证范围 | 一个输入、四次有区分目的的完整前向、一次 TRT-only 回放；不重跑完整 clip |

## 实验的目的

解释完整输入、模型/engine 身份和 age=0 相同却出现最大单点 0.398905 m 差异的现象。
暂定一致性门禁为同一计算条件的逐值一致；这用于定位变化位置，不虚构实车容差。

## 实验的 setting

排除项：两条历史记录 row_digest、1473 输入 token、engine digest、noise_scale=0、flow_steps=8 相同，query 均实际重算。
已知差异：历史视觉缓存命中组合和请求顺序不同；历史条件张量未保存。
最高优先假设：BF16 视觉计算批次变化影响特征，再传导至语言和动作。备选：语言或 TRT 自身的重复执行变化。
仅控制视觉缓存出生批次：A 六图冷计算，B 六图全部命中，C 先编码前三图再计算目标的剩余三图，D 复用 C 的全部特征。
每次清空 query 读出缓存以确保 query 年龄为零且重新计算；图像、时间、指令、归一化、模型、engine 和 GPU 不变。
额外将 A 的已保存条件送入 TRT 一次，区分后端重复执行变化。
预测：若 A=B、C=D，但 A 与 C 的视觉和条件不同，则支持视觉批次机制；若相同条件的 TRT 结果不同，则转查后端。
资源或原生运行失败立即保留日志并停止；本轮不自动修复、重试或扩展样本。

### 可视化与导出计划

启动前创建 visualization/visualization_manifest.json；保存实际 visual_input.pt、每臂 condition.pt 和 normalized_trajectory.pt。
scorecard.json、visualization/review.html 报告第一处差异、每图特征差、条件差与轨迹差；只通过私有文件和 SSH 访问。
无训练，不使用 TensorBoard。全部原始失败和负结果保留。

## 记录实验结果

第一阶段已完成：4 次完整前向、1 次 TRT-only 回放、0 次重试。
六张图的视觉特征逐值一致，五个条件张量（query_hidden、visual_tokens、visual_mask、role_ids、time_offsets_sec）
在四个分支全部逐值一致；四条轨迹相同；TRT-only 回放也相同。没有修改缓存实现。

### 效果好的结果

已有身份匹配排除了当前观测和指令不同这一混杂因素。
在此样本上，三图/六图批次变化未产生差异，固定条件的 TRT 重复执行也未产生差异。

### 效果不好的结果

历史两条输出缺少条件张量，不能只靠末端轨迹确定根因。
新进程 cold 输出与旧 fresh、旧 dual 输出最大点位差分别为 0.284988 m、0.189072 m，尚未复现任一历史状态。

### 可视化与产物

路径见 setting；完成后记录实际计数与结果。

## 分析实验结果，可能的原因

第一阶段结论为 inconclusive；第二阶段对实际历史批次取得正向证据。
不能由两次独立运行的差异直接推出缓存错误，更不能推断保留 query 的质量提升或下降。
第一阶段结果降低“三图/六图批次”的解释优先级。重建历史请求发现实际出生批次不等于这个简化划分：
fresh 的目标帧分别出生于三图（含旧地图）和四图批次；dual 则出生于单图或两图批次。
历史 cache entries 只有 50/69 个，低于容量，无需通过重跑历史模型请求重建这些出生批次。
下一阶段直接重放这些视觉出生批次；只有视觉特征确实改变时，才执行对应完整前向观察传导。
第二阶段产物根目录为同级 `r10_cache_birth_batches_20261016`，入口为 `tools/diagnose_historical_batches.py`。
先执行一条同进程 cold 作为条件参考，再做历史 fresh/dual 的视觉出生批次回放；只有特征不同的分支才追加完整前向，最多三条。
同时比较两个独立进程的 cold 条件，检查跨进程差异，并保存实际媒体 SHA256。GPU6 常驻服务继续保持不变。

第二阶段已完成：2 次完整前向、0 重试。fresh 历史批次的视觉特征与 cold 相同，因此未追加其完整前向。
dual 历史中的两图批次使三张图特征改变，最大绝对差 0.0625；query_hidden 最大差 0.1875，
最终最大 XY 点位差 0.2412295 m。两个新进程 cold 的五个条件张量完全一致。
因此“视觉缓存出生批次可影响结果”为 supported，但尚未完整复现旧 0.398905 m 差异。

第三阶段计划：在独立部署 adapter 中固定每次调用视觉塔只处理一张图，保持原生模型和 engine。
产物目录 `r10_per_image_validation_20261016`；使用同一输入、同一实际 fresh/dual 出生批次，
分别做 cold/fresh/dual 共三次完整前向，禁止重试和整段回放。输入、源码 SHA256 在启动包固定，
visualization/visualization_manifest.json 在 started 前写入，scorecard 和 review.html 保存结果。
通过条件为三种历史的每图特征、全部条件张量及最终轨迹逐值一致；记录与原批量 cold 的条件差。
这是单样本机制回归，不是全数据集性能或导航质量验收。
首次启动因 GPU4 已被占用，在模型加载前退出，0 前向。保留原 run.log 和 resources_before.json。
核验 GPU2 空闲后改用新目录 `r10_per_image_validation_gpu2_20261016`；不覆盖失败目录。
GPU2 首次包缺少 __init__.py，旧包优先导致 adapter 导入失败，已退出，0 前向。修正打包并将 adapter 导入提前到模型加载前；新目录 `r10_per_image_validation_gpu2_packagefix_20261016` 保留此次独立执行。
三个修复分支均使用同一 GPU2；与旧 GPU4 cold 的比较同时包含设备差异，不据此单独归因编码策略。

第三阶段结果：3 次完整前向，fresh/dual 历史与新 cold 的逐图特征、五个条件张量、最终轨迹全部逐值一致。
`history_independence_passed=true`，两组最大 XY 差均为 0。新 cold 与旧 GPU4 cold 的五个条件张量也逐值一致。
三次服务内部 latency 分别 3.828903 s、0.119948 s、0.108840 s；后两次视觉缓存已由诊断主动填充，
不包含填充成本，不能据此宣称逐图编码持续吞吐或 2 Hz 验收通过。前两次启动失败均发生在前向之前，单独保留，不计为通过。
下一步服务更新使用 `r10_per_image_service_20261016`：仅停止健康、空闲且身份匹配的旧 PID1601，
在 GPU6 使用相同模型、engine、输入合同、端口18716，以新视觉策略预热一次后开放服务。
未重跑五类刷新用例或完整 clip。服务更新已完成，旧 PID1601 正常退出 returncode=0。
新 PID3928 在 GPU6、loopback18716 报告 ready=true、busy=false，completed_requests=1（仅启动预热）。
lifecycle 确认 vision_policy 的 batch_unit=one_image；加载 10.400859 s、预热 2.693895 s，
loading 到 ready 13.518428 s。启动状态不同，不能将相较上次的缩短全部归因逐图策略。
模型和 engine 各加载一次且 digest 未变；业务热推理持续性能未重新基准测试。

## 用户讨论与决策记录

| 时间 | 用户讨论/关注点 | 最终决定 | 对实验的影响 | 未解决问题 |
| --- | --- | --- | --- | --- |
| 2026-10-16 | 完成采集前模型收尾；只做必要验证 | 定位新发现的最坏 fresh 样本 | 四个受控分支，不重跑旧整段验收 | 条件差异根因与语义质量 |

## 接下来要做的实验

| 优先级 | 下一实验 | 假设 | 唯一主要变量 | 所需证据 | 通过/停止条件 | 依赖或授权 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 固定视觉策略后的非零 query 年龄质量核对 | 分离视觉数值混杂后可判断 query 保留影响 | query 年龄 | 已有可信校准真值及同观测输出；缺失时先查 provenance | 先确认真值身份、坐标和有效范围，不能以 fresh 预测当真值 | 模型侧收尾；不接 Orin 或实车 |
