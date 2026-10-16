# 实验记录：已保存双频输出的里程计参考审阅

experiment_record_version: 1

## 实验元数据

| 字段 | 记录 |
| --- | --- |
| 实验 ID | r10_odometry_review_20261016 |
| 状态 | completed；单clip参考误差明显，导航质量未通过 |
| 代码版本 | 独立子仓 25f909e 加本次离线工具；只读 canonical raw_trajectory_v2.py 的实际 SHA256 写入产物 |
| 数据版本 | 当前单 clip 的 55 对已保存预测；91 条来源声明及 908 帧原始位姿 |
| 模型版本 | 既有 fresh/dual 输出；未使用新修复重新推理 |
| 环境 | 本地 UrbanNav Python，CPU；无 GPU、训练或新模型调用 |
| 运行入口 | tools/review_odometry.py |
| 产物根目录 | /home/ubuntu/.codex/visualizations/2026/10/14/01a13a44-ecc8-7490-acab-ba7d3714f8db/r10-full-clip-20261015/odometry_review_20261016 |
| 验证范围 | 单 clip 历史输出诊断，不是冻结 benchmark 或实车放行 |

## 实验的目的

在不重新推理的前提下，比较已有预测与时间对应的原始里程计，辨别输出差异是否伴随参考误差增大。
旧预测的视觉批次数值混杂仍在，因此不能单独把变化归因 query 年龄。不能将 fresh 输出当真值。

## 实验的 setting

来源审计已保存于同级 pose_provenance_20261016/audit.json、selected_rows.jsonl。
原始多帧清单中本 clip 91 条记录：81 calibrated_pose_projection，10 stationary_pose_hold。
所有来源时间戳与 frame_times.npy 精确相等，该文件 SHA256 与已执行回放的 source_identity 相同。
当前四元数均为单位范数、位置非全静态。canonical reader 接受85个完整四秒窗口，另外6个拒绝原因均为尾部覆盖不足。
旧 mcap_extract_meta 仍记录 missing_attitude_samples；新的同步报告和非零姿态数组不能证明传感器物理精度。
因此本轮称为“原始里程计参考”，不宣称经过外部测量认证的真值。
旧四列标签是 right/front/time/speed，包含可见投影和空间重采样；相较原始位姿的最大点位差中位数1.01456 m、最高1.26710 m。
不得把旧标签补长成四秒或将其时间列当 yaw。

使用只读 canonical loader 和相对位姿投影：x_forward、y_left、相对 yaw，Foxglove offset=90度。
逐请求核对当前 FPV ordinal 与源时间，仅接受有完整四秒参考和完整预测掩码的配对。
在预测的32个时间点线性插值 XY；yaw 展开后插值再绕回。保存平均点位误差、末点误差和平均绝对 yaw 误差。
所有55对均须有 accepted/rejected 记录；不挑选好样本。没有预先定义导航容差，不从均值自行判定放行。

### 可视化与导出计划

启动前创建 visualization_manifest.json；scorecard.json 保存分母、排除原因、逐对结果和来源哈希。
review.png/review.html 展示三个输出差异最大且有完整参考的样本，包含参考、两条预测及当前FPV。
本地私有路径访问，不提供公网服务。TensorBoard与训练曲线 N/A：仅离线审阅。

## 记录实验结果

### 效果好的结果

已完成来源审计与回放时间身份核对，新增模型调用0。
55 对配对中51对有完整四秒参考，4对尾部不足四秒，已明确排除，未补齐或混入完整指标。

| 分组 | 配对数 | fresh 平均点位误差 | dual 平均点位误差 | fresh 平均末点误差 | dual 平均末点误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| age=0 | 17 | 4.611615 m | 4.608208 m | 8.279331 m | 8.283345 m |
| age>0 | 34 | 4.474685 m | 4.719614 m | 8.003676 m | 8.405394 m |

非零年龄组平均绝对 yaw 误差 fresh=0.066788 rad，dual=0.065602 rad。
以上为记录里程计参考误差，不能与仅两条预测之间的差异混用。

### 效果不好的结果

校准报告不等于物理测量精度认证；旧缓存混杂及单clip范围限制因果解释。
fresh 基线本身也有较大误差，不能将全部问题归因 query 保留。已查看 review.png，部分预测前向距离
明显超过记录运动；例如卡车场景 dual-rate-0201 的参考只前进约1m，而预测约6–9m。
没有证明模型可可靠预测停止或实际速度；服务吞吐与数值一致性均不能覆盖该质量缺口。

### 可视化与产物

来源审计在上述路径；本轮 scorecard.json、visualization_manifest.json、review.png、review.html 已生成。
已查看三例可视化；末尾最大输出差异案例 dual-rate-0291 因参考不足四秒被排除，未伪造完整末点指标。
源码与请求/事件/配对报告哈希在 scorecard.json；旧 raw reader 为共享仓当前只读工作树，其哈希已冻结。

## 分析实验结果，可能的原因

输入与原始时间身份一致为 supported；非零年龄导航质量是否可接受仍 inconclusive。
旧标签经过空间处理，不能直接当作严格时间对齐的里程计参考；本次分开记录。
“fresh 基线也偏离记录运动”为 supported；“保留 query 单独导致全部质量问题”为 contradicted。
尚未排除训练标签时间/坐标合同、归一化/反归一化、数据分布和模型未学会速度变化等机制。
下一步只读训练配置与已保存动作产物，先确认四秒目标与米制输出的实际绑定，禁止直接重训或重跑clip。

### 后续只读合同与历史基线核对

产物位于同级 `contract_audit_20261016`：training_clip_evidence.json、contract_checks.json、
historical_control.json、historical_summary.json，均为已存在数据的审计，没有新增模型前向。

- 原 r10 数据划分中此 clip 的 train=0、validation=0、test=42。它是原 test clip，不能把此次诊断变成训练调参依据后仍声称原测试独立。
- 42 条原 test 目标为 current_ego_flu、累计相对位姿、4秒32点、0.125秒间隔、米/米/弧度。
  与本轮 raw reader 对应参考的最大 XY 差 4.44e-16 m、yaw差 2.36e-16 rad，支持参考提取合同一致。
- 训练 normalization_v2.json、恢复模型 contracts 与 action_head 的 normalization_profile 完全相同；
  digest=091758430e8897c1f59e6c516b603815504056d472247879e31f66400162d821，
  scales=[16.41384396625424,3.268804861815276,0.8134308943799216]。
- 三条已保存修复验证 normalized_trajectory.pt 乘上述尺度后，与服务 response 的米制轨迹逐值相同。
  因而“时间网格/坐标误读”和“当前解码尺度错误”在本次检查范围内均为 contradicted。
- 2026-08-23 owner-scoped matrix 的 control 原始结果仍存在；同一clip42条历史预测
  ADE=4.790700 m、FDE=8.574910 m、yaw MAE=0.067577 rad。42条中 is_stop=0，不能用它证明停车能力。
  action_noise_scale=0、flow_steps=8，与当前部署相同。
- 历史窗口与新回放窗口不同，不能直接用聚合均值证明部署无回归；但大偏差在服务化之前已存在为 supported。

训练合同声明 standard_normal_v1，而旧 cached-query 评测与部署均用零初始动作。
这是一项已有采样选择，尚不能当作错误根因。若继续质量定位，优先复用保存的条件张量做
动作端初始分布的最小对照，不重新运行视觉/语言编码，也不据单例结果改动正式采样策略。
当前仍不接受导航质量；基础服务完成不等于模型已具备实车能力。

### 最小动作采样诊断计划

仅使用 `r10_per_image_validation_gpu2_packagefix_20261016/service/requests/cold_same_process/condition.pt`，
即已定位的frame152。零初始动作基线直接读取已有response，不重算；新分支固定CPU生成器seed=0和1，
x0~N(0,I)，输入backend时转换为原engine dtype。各执行一次8步Euler，只加载动作engine，无LVLM或视觉前向。
参考为相同frame152、32个0.125秒间隔的原始里程计；哈希与输入身份在started.json保存。
新run `r10_initial_noise_diagnosis_20261016`，scorecard.json含三条轨迹及误差，review.html与可视化manifest在run内。
资源占用、异常或非有限数值立即停止，不重试；单条件两个固定seed仅用于检验机制，不选最好seed修改生产配置。
若两seed未一致改善，则降低“零初始化是主要原因”的优先级；即使一致改善，也需另有验证集证据才能改采样策略。

### 最小采样诊断结果

`r10_initial_noise_diagnosis_20261016` 已完成，2次动作积分、0次完整模型前向、0次重试；零基线直接复用。

| 初始动作 | 平均XY误差 | 末点XY误差 | 平均绝对yaw误差 |
| --- | ---: | ---: | ---: |
| 保存的零初值 | 4.339577 m | 8.027780 m | 0.047666 rad |
| 标准正态seed0 | 1.728958 m | 2.107690 m | 0.155157 rad |
| 标准正态seed1 | 0.989971 m | 1.044791 m | 0.181601 rad |

两seed一致改善XY但增加yaw误差，支持采样初始化会影响本样本的质量取舍；不支持已找到可直接上线修复。
标准正态只检查两个预先固定seed，没有挑选最佳seed；生产服务仍保持原零初始化。
查看本地 noise_comparison.png 后进一步确认：seed0/1 的首点x分别为-0.735096m、-1.666393m，
前向坐标为负的点分别有3/6个，而参考为前进。因此不能仅凭平均XY改善替换策略。
本地 offline_review.html 保存三轴时间曲线，展示位置改善与朝向/早期反向问题。
下一步需在独立验证条件上判断整体取舍，并确认质量验收阈值；不把此test clip用于选择上线超参数。

## 用户讨论与决策记录

| 时间 | 用户讨论/关注点 | 最终决定 | 对实验的影响 | 未解决问题 |
| --- | --- | --- | --- | --- |
| 2026-10-16 | 完成采集前流程，禁止冗余测试 | 复用保存输出，先核对参考来源 | 零新增推理，缺四秒窗口明确排除 | query保留质量与可接受误差 |

## 接下来要做的实验

| 优先级 | 下一实验 | 假设 | 唯一主要变量 | 所需证据 | 通过/停止条件 | 依赖或授权 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 动作采样初始分布最小定位 | 零初始动作可能影响CFM输出质量 | 固定条件下的初始动作 | 复用已保存条件和参考；标准正态与原零初始化对照 | 只判断采样机制，不凭单例放行；无证据不改正式策略 | 模型侧诊断，先固定样本与seed；禁止重训或整clip重跑 |
