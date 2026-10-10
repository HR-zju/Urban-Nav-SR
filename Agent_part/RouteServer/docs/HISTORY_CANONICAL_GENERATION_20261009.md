# 实验记录：真实 4s/2Hz canonical source 全量生成

experiment_record_version: 1

## 实验元数据

| 字段 | 记录 |
| --- | --- |
| 实验 ID | history-canonical-generation-20261009 |
| 状态 | passed：仅全量 canonical source 构建与输出核对 |
| 负责人 | UrbanNav annotation/data |
| 时间 | 2026-10-09，Asia/Shanghai；21:21:07 完成最终产物核对 |
| 代码版本 | Urban-Nav-SR `890ed627880aa2ef9102ab790a1117c3e563701b`，使用工作区未提交的 history migration 实现 |
| 数据版本 | 本地 `urbannav_history_4s_2hz_modelscope_v1`；`metadata/history_contract.json` 中 `history_4s_2hz_9f_v1` |
| 模型版本 | N/A：确定性元数据转换，无 teacher 或模型加载 |
| 环境 | 本地 Ubuntu，`/home/ubuntu/anaconda3/envs/UrbanNav/bin/python`，CPU |
| 产物根目录 | `/tmp/urbannav-canonical-migration-20261009.jjfY6B`；临时产物，正式发布前需迁入数据管理目录 |
| 验证范围 | 三个数据集全量 source 转换；标签迁移和训练验证不在本次结果范围 |

## 实验的目的

验证新历史数据可经现有 builder 生成完整 canonical source，保留 anchor、方向合同及数据根相对路径。
预期输入行与输出行一一对应。解析、身份或媒体引用失败时直接报错，保留已写产物供定位。
本次不判断旧标签能否语义复用，不补标、不训练、不修改原媒体。

## 实验的 setting

输入根目录：`/media/ubuntu/data2/urbannav_history_4s_2hz_modelscope_v1`。
每个数据集读取 `<dataset>/manifests/<dataset>_observations_direction12.jsonl`。
合同为过去 4 秒、2 Hz、包含当前帧的 9 个 FPV 槽；BEV 使用已有 latest-causal 选择。
数据原有 split 和锚点保持不变，方向合同为 `ego_degree10_12way_terminal_centroid_v2`。
seed、batch、学习率、checkpoint 选择均 N/A：本次为确定性转换。

运行入口为 RouteServer 的 `tools/model_processing/plan_history_annotation_migration.py sources`：

```bash
python tools/model_processing/plan_history_annotation_migration.py sources \
  --dataset-root /media/ubuntu/data2/urbannav_history_4s_2hz_modelscope_v1 \
  --dataset sidewalk \
  --output /tmp/urbannav-canonical-migration-20261009.jjfY6B/sidewalk.sources.jsonl
```

SideWalk 成功后串行执行 egowalk、foxglove，替换 `--dataset` 和输出文件名前缀。
EgoWalk 失败会停止该串行命令；完成状态以进程退出及返回的行数为准，文件存在不代表完成。
使用工作区 PYTHONPATH：RouteServer 根目录、RouteServer/src、UrbanNav-Contracts/src、UrbanNav-AgentTools/src。

### 可视化与导出计划

本次无 GPU 训练、模型生成或正式评测，TensorBoard、预测图表、checkpoint 均 N/A。
沿用数据包已有各 clip 的 `review/index.html`；本次不生成新图像或预测。
运行中通过进程与输出行数观察进度，结束后记录 CLI 返回计数。
本记录为本轮执行后的事实补记，没有伪造启动 receipt。

## 记录实验结果

| 数据集 | 输入预期行数 | 已完成输出行数 | 状态 |
| --- | ---: | ---: | --- |
| SideWalk | 30,428 | 30,428 | 正常退出，约 203 MiB |
| EgoWalk | 168,583 | 168,583 | 正常退出，约 892 MiB |
| Foxglove | 40,042 | 40,042 | 正常退出，约 259 MiB |

合计 239,053 条。每个数据集均无重复 anchor，每条有 9 个 FPV 引用；全部使用
`ego_degree10_12way_terminal_centroid_v2`，绝对图像路径数为零。

计数口径更正：本轮最初把媒体包 `summary.samples` 的 EgoWalk 281,190 / Foxglove 44,036
当成 source 输入预期；实际运行入口读取方向投影后的 manifest。后者的 168,583 / 40,042
与 `metadata/portable_manifest_receipt.json`、`generation_receipt.json` 及 CLI 退出结果一致。
这是既有方向投影阶段的样本筛选，本轮 source builder 没有再减少样本；媒体包总数不能当作
formal Stage I 有效标注数。例如 EgoWalk clip 000000 的方向报告记录 236 条缺少 linked sidecar。

### 可视化与产物

产物为根目录内的 `sidewalk.sources.jsonl`、`egowalk.sources.jsonl`、`foxglove.sources.jsonl`。
三个数据集均返回 `media_path_base=dataset_root`；生成器逐行验证媒体存在并输出相对路径。
实际字节数：SideWalk 212,277,302；EgoWalk 934,391,326；Foxglove 270,857,271。
EgoWalk/Foxglove 串行进程已退出，exit code 0；最终全量字段扫描也已退出，exit code 0。
未生成 TensorBoard、checkpoint 或新的 review：本轮仅生成文本 source，媒体沿用已有数据包。

### 后续真实首个 clip 规划

全量 source 之后，从 32011 的旧主线标注 run 读取第一个 SideWalk clip 对应的 26 条 source
和 26 条 annotation。读取范围由新 canonical 的实际 anchor 集合限定，没有修改远端原记录。
输入保存在产物根目录的 `sidewalk-first-clip-inputs/`；config 使用相对文件路径，目标 registry
为 `urbannav_direction_profile_registry_v2`，旧 run 为既有 20-way routed-visual baseline。

当前使用 `sidewalk-first-clip-plan-reason-fix/` 的计划与报告：

| Section | 数量 | 规划结果 |
| --- | ---: | --- |
| TaskGrounding | 26 | task registry 确定性重建，无需 teacher |
| SceneGrounding | 26 | producer contract changed，补标候选 |
| LocalTraversability | 26 | producer contract changed，补标候选 |
| NavigationContextGrounding | 18 | producer contract changed，补标候选 |
| NavigationContextGrounding | 8 | 旧 section 未被接受 |
| InteractionConstraintGrounding | 26 | producer contract changed，补标候选 |

media identity 文件保持为空，表示尚未冻结图片身份；没有把未证明的路径填写为相同 identity。
当前视觉项在 producer 合同检查处即停止，以上不是完整图片依赖比较，也不是最终全量补标预算。
未调用 annotate、teacher、compose 或训练；这 26 条不能代表其余 clip。

首次实际计划暴露原因描述错误：存在旧 annotation 但 section 没有 accepted 候选时，代码写成
`old_annotation_missing`。已修为 `accepted_section_missing`，重新执行实际计划并逐项核对旧
记录；最初 `sidewalk-first-clip-plan/` 保留作诊断，以 `*-reason-fix/` 为当前报告。

### 同 clip 的真实 VQA 规划

从已有 `vqa_v2_0_qwen122b_interesting_full_v1/datasets/sidewalk` 按相同 26 个 anchor 读取，
实际找到 6 个旧 VQA source、85 条旧答案。输入保存于 `sidewalk-first-clip-inputs/old_vqa_*.jsonl`。
这与 reasoning 覆盖不同，不能用 26 条 reasoning annotation 推断 VQA 也全部覆盖。

首次 VQA 计划为 416 题，26 个 temporal interaction 问题错误地被标记不可用。原因是
Reasoning V4.6 有意把兼容字段 `temporal_evidence_eligible` 设为 false；VQA source adapter
虽校验了 9 帧时间合同，却没有启用自己的时序资格。已在完成已有时间、帧序、图片与角色校验后
将该字段设为 true，Reasoning source 本身保持不变。这里是 prepared 数据层面的时序资格，
不扩大为原始传感器同步证明。

修复后当前计划位于 `sidewalk-first-clip-vqa-plan-temporal-fix/`：26 个 source、442 题，
其中 26 题 e0 交给原有确定性方向 producer，新增恢复 26 题 d3 temporal interaction。
其余 416 题当前是补标候选，尚未执行；media identity 文件仍为空，不能作为最终补标预算。
原先假设指令类 VQA 可像 Task 一样直接复用，实际旧 a1/a2/a3 question 仍引用图像，
现有逐题判定正确要求媒体身份。这次没有将 Task 的 instruction-only 复用推广到这些题。
最初 `sidewalk-first-clip-vqa-plan/` 保留作失败资格诊断，使用修复后的计划继续后续工作。

### 首个 clip 的实际媒体内容索引

已从 32011 对旧 reasoning/VQA source 引用的 156 个去重图片读取内容摘要，并对本地新 source
的 172 个引用读取同样摘要。结果保存为 `sidewalk-first-clip-inputs/media_content_identity.jsonl`；
每条记录含原引用、`jpeg-sha256:` identity、`proof=exact_file_bytes` 和字节数。
该索引只证明完全相同的文件内容，不能把不同 JPEG 编码自动归为不同原视频帧。

旧新共享的 5 个 identity 全部是 BEV，frame 名为 000360、000480、000750、000870、000960。
它们的旧 `_bev.jpg` 与新 `.jpg` 路径虽然不同，内容完全相同。FPV 没有字节相同项；已核实的
编号偏移样例和 JPEG 编码差异仍须分开处理，不能把本次字节比较当成全量帧对齐结论。

使用 `vqa_content_migration.json` 重建的当前计划是 `sidewalk-first-clip-vqa-plan-content/`：

| 逐题原因 | 数量 | 含义 |
| --- | ---: | --- |
| question_input_changed | 70 | 找到对应旧答案，但实际输入比较不同 |
| old_answer_missing | 346 | 当前 anchor/task/condition 的问题没有匹配旧答案 |
| use_existing_label_producer | 26 | 原有确定性方向 producer 可重建 |

共 442 题，未解析媒体引用数降为零。这里的 70 题包括可见文本、上下文与图片内容的整体比较，
不表示每题都是单纯图片变化；也不证明旧答案语义错误。当前精确复用数为零，近似弱标签迁移
尚未启用，重新编码同帧的等价性尚未证明，因此该数值仍不能作为全量重标预算。
本次没有启动 teacher 或全量图片摘要扫描；只核对这一 clip 的计划引用。

### 真实 Stage I 消费者检查与 BEV 合同修复

使用一个真实新 source 的 e0 方向真值，经现有 deterministic producer、Stage I VQA projector
进入 ms-swift `build_stage_i_manifest_row`。首次失败于 `history_input_fields`：旧包的 BEV 有
timestamp 和 age，却缺少 `bev_tolerance_ns`。这说明上述 239,053 条 source 的结构校验
不能代替消费者验收；其中带 BEV 的旧输出尚不可作为本轮训练发布产物。

已读取原生成计划 `.../clips/*/summary.json`：

- `/tmp/urbannav-recorded-sidewalk-corpus-plan-20260920`：7,121 份均为 `max_bev_age_ns=1500000000`。
- `/tmp/urbannav-recorded-foxglove-corpus-plan-mapped-20260920`：763 份均为相同预算。

迁移 source 入口增加显式 `--bev-tolerance-ns`，只为缺少字段的旧包补回原始预算；核对
timestamp、age 与预算，保持图片和 GT。原消费者校验完整保留，ms-swift 本轮没有代码改动。
修复后真实一条 e0 已通过 source → VQA annotation → Stage I view → ms-swift training row：
10 张图片（9 FPV + 1 BEV），BEV age 为 448,479,232 ns，预算为 1,500,000,000 ns。
产物位于 `real-e0-consumer-bev-contract/{source,annotation,view,train}.jsonl`，没有调用 teacher。

该阶段尚未进行 tokenizer/processor 编码与 batch collator，后续结果见下一节。
新字段改变 source fingerprint，相应临时迁移计划需重建；原始数据和旧 annotation 保持不变。
EgoWalk 无 BEV，不受此字段影响。

### BEV 合同全量重建与真实 processor/batch 结果

SideWalk 30,428 条和 Foxglove 40,042 条已使用 `--bev-tolerance-ns 1500000000` 重新生成，
串行进程 exit code 0；70,470 条逐条通过 timestamp/age/预算核对。当前 source 位于
`bev-contract/{sidewalk,foxglove}.sources.jsonl`；EgoWalk 继续使用根目录的原文件。
根目录中旧 SideWalk/Foxglove source 保留诊断，不作为当前训练发布输入。
首 clip 的新输入为 `sidewalk-first-clip-inputs/new_sources_bev_contract.jsonl`，当前 reasoning
与 VQA 计划分别为 `sidewalk-first-clip-plan-bev-contract/`、
`sidewalk-first-clip-vqa-plan-bev-contract/`，其 report 固定新配置及 source 指纹。

在 32011（`lsm-a100-copy-copy-0`）使用现有 qwen3 环境及服务器 ms-swift 代码，以真实
`merged_stage_i_checkpoint_256parent_v1_rebuilt_t58` 的 Qwen3.5 processor 编码上述 e0 行。
仅 CPU，`get_processor(..., download_model=False)` 不加载模型权重，Hub offline；
Stage I profile 的预算为 128 image/video tokens、max_length 8192，loss scale 为
`stage_i_sft_v3_1`，template 为 train mode、`add_non_thinking_prefix=False`。
运行时将工作目录绑定到服务器数据根，JSONL 中的图片引用继续保持相对路径。

结果：3,363 个输入 token、16 个 supervised token，10 张图片均被处理。
`input_ids/labels/loss_scale` 均为 `[1,3363]`，`image_grid_thw=[10,3]`，
`pixel_values=[4804,1536]`，`position_ids=[3,1,3363]`，processor 编码和 data_collator 均正常退出。
凭据为 `real-e0-consumer-bev-contract/encoding_receipt.json`，transformers 版本 `5.16.0.dev0`。
缺少 decord 的提示未阻断此次静态多图编码，没有为此改环境。

这完成了一个真实 e0 batch 的消费者接口验证；没有模型 forward、optimizer step 或 teacher 调用，
不证明 FullCoT、全部 VQA 标签已迁移，也不证明新窗口的模型质量。服务器代码与 checkpoint
均原样使用，已有 dirty changes 保留；未以本地 commit 冒充服务器消费者版本。

### 三个来源的真实 baseline 覆盖

在 32011 流式扫描 EgoWalk/Foxglove 的实际旧 source 和 annotation，按新方向 manifest 的
`source_parent_id` 连接旧 `source_group_id`，并要求旧 annotation 指纹匹配该 run 的旧 source。
SideWalk 使用此前已核实主线结果。汇总见产物根目录的 `baseline_coverage.json`：

| 数据集 | 新 anchor | 旧 source 覆盖 | 有匹配版本旧 annotation | 本次 baseline 未覆盖 |
| --- | ---: | ---: | ---: | ---: |
| SideWalk | 30,428 | 29,212 | 29,212 | 1,216 |
| Foxglove | 40,042 | 38,348 | 38,348 | 1,694 |
| EgoWalk | 168,583 | 168,583 | 150,748 | 17,835 |
| 合计 | 239,053 | 236,143 | 218,308 | 20,745 |

Foxglove 使用 `reasoning_v4_6_qwen122b_foxglove_20way_routed_visual_full_v1` 的 merged
`reasoning_v4_6_annotations.jsonl`，38,348 个 anchor 的六个 section 均存在 accepted 记录。
EgoWalk 的 20-way run 未发现顶层 source/annotation JSONL；实际可用的是
`reasoning_v4_6_qwen122b_egowalk_12way_routed_visual_full_v1`，扫描 `service_0.jsonl` 与
`service_1.jsonl`。154,514 条匹配新 anchor 的记录对应 150,748 个去重 anchor；Task、Scene、
LocalTraversability、Interaction 各有 150,748 个 accepted，Navigation/Decision 各有 145,897 个。
这里统计“存在 accepted section”，不声称每个 anchor 的最新重试都成功。

这些是旧结果存在性与 lineage 计数，不是新窗口复用计数。未覆盖也不等于原始数据丢失，
例如 EgoWalk 的 17,835 个新 anchor 都有旧 source，只是本次选定的两个当前 annotation 分片
未覆盖；本轮没有把历史 cache-mismatch/rejected 文件混入正式结果，也没有启动补标。

## 分析实验结果，可能的原因

- supported：三个数据集全量 source 构建可运行，输出计数与实际输入 manifest 行数一致。
- inconclusive：全量 labels、formal training views 和训练 batch 仍未验证。
- 已排除：SideWalk source 构建无需 teacher、图片复制或修改服务器 dirty worktree。
- 尚未验证：新输入能复用哪些旧标签；已有图片编号偏移证据记录在 `HISTORY_ANNOTATION_MIGRATION.md`。
- 首个真实 clip 的计划验证了旧 source/annotation lineage 和执行入口；视觉复用仍受合同变化及
  尚未冻结的媒体身份限制。已修复缺失原因的误报，保持旧记录存在性与 section 有效性分离。

## 用户讨论与决策记录

| 时间 | 用户讨论/关注点 | 最终决定 | 对实验的影响 | 未解决问题 |
| --- | --- | --- | --- | --- |
| 2026-10-09 | 继续迁移、使用相对路径、减少无关测试 | 采用已有 source builder 处理真实数据 | 不复制媒体，不调用 teacher | Scene 近似弱标签迁移尚待确认 |
| 2026-10-09 | 全量生成时纠正计数口径 | 使用实际方向 manifest 行数，保留媒体总数作另一层统计 | 三个来源共 239,053 条 source 均完成 | 旧标签复用数量尚未确定 |

## 接下来要做的实验

| 优先级 | 下一实验 | 假设 | 唯一主要变量 | 所需证据 | 通过/停止条件 | 依赖或授权 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 真实复用/补标清单 | 已核实媒体可支持精确复用 | 旧图实际身份 | 新旧 source、旧答案、媒体证据 | 未知项明确保留，同名图片不推断相等 | Scene 近似复用尚待确认 |
| P1 | 后续 FullCoT 正式导出 | 补齐标签后可完成原 composer 与正式资格门禁 | 实际迁移标签 | formal export 与拒绝原因 | 保持原门禁，不以已通过的 e0 batch 替代 | 依赖复用政策与补标决策，全量付费补标尚未启动 |
