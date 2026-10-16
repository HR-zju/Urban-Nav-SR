# 实验记录：同一当前观测的已保存轨迹对比

experiment_record_version: 1

## 实验元数据

| 字段 | 记录 |
| --- | --- |
| 实验 ID | r10_saved_output_review_20261016 |
| 状态 | completed；导航质量验收仍未通过 |
| 代码版本 | Agent_part/urbannav-deploy/tools/review_saved_predictions.py；本轮独立子仓改动 |
| 数据版本 | continuous_cache/replay 的 117 条 fresh 输出，对比 dual_rate_shadow 的 274 条输出；文件指纹在 paired_differences.json |
| 模型版本 | 相同 checkpoint、normalization、engine 的已保存输出；逐对检查模型和坐标/时间合同 |
| 环境 | 本地 UrbanNav Python 3.12；CPU 离线审阅；新增模型请求 0 |
| 运行入口 | python tools/review_saved_predictions.py --artifacts ARTIFACTS --output REVIEW |
| 产物根目录 | /home/ubuntu/.codex/visualizations/2026/10/14/01a13a44-ecc8-7490-acab-ba7d3714f8db/r10-full-clip-20261015/predeployment_review_20261016 |
| 验证范围 | 一个 clip 上 55 对匹配输出；不是冻结 benchmark，也没有轨迹真值误差 |

## 实验的目的

只读已有产物，判断 query 保留后输出改变多少，寻找需要下一步定位的差异。
本次不把 fresh 模型输出当作真值，不从有限数值或延迟通过推导导航正确。

## 实验的 setting

匹配完整 action_observation（只忽略 metadata sample_id），还要求当前 messages 完全一致。
每对输出 checkpoint、normalization、coordinate_frame、trajectory_format、trajectory_timestamps_sec 和 valid_mask 一致。
比较平均点位距离、末点距离、最大点位距离以及绕回后的 yaw 差：
`d_xy(i)=hypot(x_f(i)-x_r(i), y_f(i)-y_r(i))`；
`d_yaw(i)=abs(atan2(sin(yaw_f(i)-yaw_r(i)), cos(yaw_f(i)-yaw_r(i))))`。
这些量是输出之间的差异，不命名为 ADE/FDE 真值误差。
18 对 query age=0 作为重复 fresh 输出参考，37 对 age>0 单独报告。既有运行的缓存状态与请求序列不同。

### 可视化与导出计划

paired_differences.json 保存完整配对及 fingerprint；differences.png 展示年龄与点位差，
matched_cases.png 展示三个最大平均差异样本及一个 fresh 差异样本，旁列其真实当前 FPV 和历史 BEV。
review.html 可离线打开。坐标明确为 x forward、y left，图中两条线都属于模型预测；无训练、无 TensorBoard。

## 记录实验结果

| 组别 | 匹配数 | 平均点位差均值 | 末点差均值 | 最大单点差 | 平均 yaw 差均值 |
| --- | --- | --- | --- | --- | --- |
| fresh 对 fresh | 18 | 0.047386 m | 0.020748 m | 0.398905 m | 0.000577 rad |
| fresh 对保留 query | 37 | 0.657857 m | 1.173897 m | 6.038192 m | 0.014413 rad |

### 效果好的结果

匹配了相同当前图像、时间、指令与 prompt 的 55 对结果，全部按同一坐标与时间网格比较；没有重新前向。
可视化以实际图片和请求 ID 定位输出，修正了原回放 review 将 y_left 标为 y_right 的展示问题。

### 效果不好的结果

最大差异位于 dual-rate-0291（query age 0.459354 秒），平均点位差 3.330316 m，末点差 6.038192 m。
另两个较大差异是 dual-rate-0021 与 dual-rate-0201，平均点位差分别为 2.038333 m 与 1.559182 m。
fresh 对 fresh 的最大单点差来自 dual-rate-0046：完整输入 row 相同，仍出现 0.398905 m 的单点差；其末点差仅 0.011328 m。
因此当前证据不支持“任意缓存历史下所有 fresh 结果逐值一致”，此前窄样本原生精确一致结论保留原验证范围。

### 可视化与产物

所有产物位于上表 REVIEW 目录。已经查看 matched_cases.png；差异主要表现为预测沿前向的进度/长度变化，
但仅凭图片与两条预测线不能判断哪条更正确，也不能替代碰撞或任务成功率验收。
原 clip 的原始姿态与位置文件存在，但尚未绑定本次时间窗的可信校准真值。提取元数据还记录
`overall_pose_quality_valid=false`，不能直接把这些数组当成已验收真值；该旧元数据是否仍适用于后续修复资产也未确认。

## 分析实验结果，可能的原因

非零年龄组输出变化显著大于重复 fresh 组，但两次运行请求序列、视觉缓存命中组合不同。
fresh 对 fresh 的差异说明仍有其他变量，不能把全部变化归因于 query 年龄。
视觉批次形状或内核数值路径是待验证假设，不是已确认根因。现有输出未保存各请求条件张量，不能仅凭最终轨迹定位。
训练记录全部为 age=0，保留 query 的语义质量仍是部署门禁。工程双频通过不覆盖这个门禁。

## 用户讨论与决策记录

| 时间 | 用户讨论/关注点 | 最终决定 | 对实验的影响 | 未解决问题 |
| --- | --- | --- | --- | --- |
| 2026-10-16 | 先完成采集和实物之前的流程，禁止冗余测试 | 优先审阅既有 fresh/aged 输出 | 零新增推理，完整记录正负结果 | 非零年龄导航质量；重复 fresh 输出差异来源 |

## 接下来要做的实验

| 优先级 | 下一实验 | 假设 | 唯一主要变量 | 所需证据 | 通过/停止条件 | 依赖或授权 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 定位重复 fresh 的条件差异 | 缓存命中组合影响数值路径 | 同一输入下缓存状态 | 最坏样本的条件张量及最终轨迹 | 找到最早差异位置；禁止重跑完整 clip | 新发现的问题，属于本轮模型侧收尾 |
| P1 | 校准真值来源核对 | 可复用已有可信标签 | 真值证据绑定 | 源时间、校准 provenance、有效 mask | 证明身份和坐标一致后才计算误差；证据缺失则保留未验收 | 只读已有资产 |
