# 实验记录：新窗口单 clip 缺失 reasoning 补标

experiment_record_version: 1

## 实验元数据

- 实验 ID：`vla-gap-clip-20261011`；状态：running；负责人：UrbanNav annotation。
- 代码：Urban-Nav-SR `081bd24e1c491796c058e0f452d85611ca04daf0`，部署脚本本地 dirty（10 图、32k 上下文）；Contracts `33a4a976ace0ecbeb53f8781dc3ce7224e0dbee5`，AgentTools `5214c32b0d261dc756c54324dac1b97646b2834e`。无关变更保留。
- 服务器根目录：`/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation/annotation_runs/vla_gap_clip_20261011`。
- 本地执行文件：`.artifacts/vla-gap-clip-20261011/{prepare.py,label.py,run.sh}`；服务器保存隔离代码副本，不覆盖部署仓。
- 范围：review-subset，35 个 observation，一个 clip；本轮不扩大到全量。

## 实验的目的

优先补充缺自身 reasoning 的新样本，沿用正式分 section Prompt；保留已有 reasoning/VQA 和历史诊断。验证 35 条缺标输入能否获得可检查的正式结构化回答，不以 accepted 状态证明语义正确，不直接导出正式训练集。

## 实验的 setting

- 数据根目录：`/lpai/volumes/mind-eb-bd-ga/suh/dataset/urbannav_history_4s_2hz_v1`。
- Clip：Foxglove `20260510_014849_rec_0eRbtAGrveNcMxj7`；迁移索引中选择不存在自身 reasoning ID 的 35 条。旧 annotation 大小和 mtime 与冻结 inventory 一致，选择过程不使用 accepted/rejected。
- 输入：9 帧 FPV、latest-causal recorded BEV、自身 instruction。媒体路径相对 dataset root，provider 读取时绑定根目录；GT 方向保留在私有监督侧，不把旧 assistant 文本送入新 teacher。
- 方向：`ego_degree10_12way_terminal_centroid_v2`，registry v2；probe `ego_12way_affordance7_v1`。沿用新数据已经计算的标签，未重算 GT。
- Prompt：当前正式 `reasoning_v4_profiles.py` 与 `reasoning_v4_teacher.py`；Reasoning V4.6、`reasoning_v4_section_images_routed_visual_v1`。Scene/Interaction 看历史 FPV，LocalTraversability 看当前 FPV，Task/NavigationContext/Decision 看当前 FPV 与可用 BEV。既有确定性 Task 规则、parser/composer 保持原样。
- 模型：Qwen3.5-122B-A10B 固定挂载 `26-02-24-1554`；原环境 `qwen35_vllm_cuda128`，Python 3.11.13、torch 2.11.0+cu128、vLLM 0.20.2、transformers 5.13.0。
- 硬件：31041，0–3 号 A100 80GB，TP=4；4–7 号已有任务保持原样。32k 上下文，image limit 10；单 worker、temperature=0、max output=4096，原 section retry=2。
- 命令：服务器隔离副本 `bash code/.artifacts/vla-gap-clip-20261011/run.sh`；tmux `vla-gap-clip-20261011`，模型 `vla-eval-1`。退出释放本次模型四卡。
- 基线：此前十图真实请求成功，19568 输入 tokens、322 输出 tokens、11.24 秒；该简短提示仅为可用性证据，不用于本次正式标注。

### 可视化与导出计划

服务器 run root 保存 `sources.jsonl`、`selection.json`、`prompt_profiles.json`、`annotations.jsonl`、`rejected.jsonl`、`report.json`、`review.html` 和 `visualization_index.json`。review 按 source ID 展示全部原始回答和结构化结果。`controller.log` 保存启动/标注/退出日志，`service/instruction/logs/vllm_18462.log` 保存模型日志。TensorBoard N/A：无训练；benchmark 图表 N/A：单 clip 补标，不做性能评测。模型只监听 loopback，review 为私有本地文件。

## 记录实验结果

35 条新 canonical source 已构建，方向合同与新窗口角色核实完成。逐条确认十张媒体存在、ego 12-way 合同及正式 batch 导入；section 图片数量为 Scene=9、Local=1、NavigationContext=2、Interaction=9、Decision=2。

2026-10-11 17:22:57 +08:00 已启动 `vla-gap-clip-20261011`，17:23:04 创建 `vla-eval-1`；API PID 9204。启动日志确认四卡 0–3、TP=4、32k、10 图，处于模型初始化阶段，尚无 annotation 回答。作业会自动执行全部 35 条并退出释放本实例；不会继续下一个 clip。

Shell 语法、diff check、实验记录结构及上述真实输入检查通过。portfolio doctor 仍报告 1 个历史产物缺失与 12 个快照漂移；CityWalker 和其他仓既有 dirty changes 保留。未运行全仓测试，未 commit/push。

### 可视化与产物

当前已有 `selection.json`、`sources.jsonl`、`prompt_profiles.json`；annotation/review 尚未产生。新旧覆盖缺口不是语义错误率。

## 分析实验结果，可能的原因

当前为 inconclusive。输入选择证明这些 ID 在已冻结的当前 Foxglove reasoning 文件中没有自身回答；正式 Prompt 返回和语义质量仍待本次运行与检查。保留所有 section 诊断，避免把 QC 不通过转换为缺标再重复调用。

## 用户讨论与决策记录

| 时间 | 用户讨论/关注点 | 最终决定 | 对实验的影响 | 未解决问题 |
| --- | --- | --- | --- | --- |
| 2026-10-11 | 优先缺失标注，使用最新正式流程和 Prompt | 先补一个 clip，再讨论扩大 | 不启动已有标签校准或全量 VQA；最多四卡，tmux 使用 vla 前缀 | 正式 section 输出待核对 |

## 接下来要做的实验

| 优先级 | 下一实验 | 假设 | 唯一主要变量 | 所需证据 | 通过/停止条件 | 依赖或授权 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 单 clip 正式缺标补充 | 新窗口适配现有 teacher | 使用正式 section Prompt | 35 条回答及每 section 状态、review | 执行后检查完整性和明显内容问题；错误保留 | 已授权 |
| P1 | 扩展剩余缺标样本 | 单 clip 流程可复用 | 数据范围 | 单 clip 检查结果 | 检查并确认后启动 | 尚未批准全量 |
