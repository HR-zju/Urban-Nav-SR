# 实验记录：新窗口全量缺失 reasoning 补标

experiment_record_version: 1

## 实验元数据

- 实验 ID：`vla-gap-full-20261011`；状态：running；负责人：UrbanNav annotation。
- 代码：Urban-Nav-SR `081bd24e1c491796c058e0f452d85611ca04daf0`，部署脚本 dirty（10 图、32k）；Contracts `33a4a976ace0ecbeb53f8781dc3ce7224e0dbee5`；AgentTools `5214c32b0d261dc756c54324dac1b97646b2834e`。服务器使用隔离源码副本，保留原部署仓。
- 产物根目录：`/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation/annotation_runs/vla_gap_full_20261011`。
- 执行文件：工作区 `.artifacts/vla-gap-full-20261011/{prepare.py,label.py,run.sh}`；tmux `vla-gap-full-20261011`，模型 tmux `vla-eval-1`。
- 验证范围：full-split 的缺自身 reasoning 子集；不是完整数据集语义验收。

## 实验的目的

为新窗口缺少自身 reasoning 的样本补充正式 section 回答。保留既有 reasoning/VQA 和刚完成的 35 条 pilot；已有标签校准另行处理。本次完成条件是全量缺标输入具有可追溯的输出或明确错误记录，语义质量及最终 action 监督另行确认。

## 实验的 setting

- 数据根目录：`/lpai/volumes/mind-eb-bd-ga/suh/dataset/urbannav_history_4s_2hz_v1`；9 帧 FPV、可用的当前 causal BEV、自身 instruction，媒体路径相对该根目录。
- 冻结依据：`annotation_migrations/history_time_only_20261010_r1/data/index.sqlite3` 与 `input_inventory.json`；验证旧 reasoning 文件大小和 mtime；`selection.json` 保存来源。按自身 identity 缺失选择，历史 QC 不参与过滤。原始 31,491 条减去 pilot 35 条，剩余 Foxglove 1,805、EgoWalk 29,651，共 31,456 条；沿用源 split。
- Prompt：当前 Reasoning V4.6，`reasoning_v4_section_images_routed_visual_v1`，Task canonicalization v3。方向 `ego_degree10_12way_terminal_centroid_v2`，registry v2。Scene/Interaction 使用历史；其他 section 沿用正式 routed image policy。GT 保持私有监督侧。
- 模型：Qwen3.5-122B-A10B，挂载版本 `26-02-24-1554`，沿用 maintained `qwen35_vllm_cuda128` 环境：Python 3.11.13、torch 2.11.0+cu128、vLLM 0.20.2、transformers 5.13.0。
- 配置：TP=4，GPU 0–3；32k 上下文，最多 10 图，MAX_NUM_SEQS=4，四个 annotation workers（19:14 从两个调高），temperature=0，输出上限 4096，section retry=2。采样 seed 沿用原 provider 默认，未声称 bitwise deterministic。
- 顺序：先构建两个数据集 source，再启动单模型，Foxglove 完成后接着 EgoWalk。已有输出沿用 batch 的 resume；错误保留，退出释放本次模型。
- 对照：pilot 单 worker 35/35 完成；本轮只扩大数据范围及 worker 数量，Prompt/parser/composer 保持一致。没有新增 VQA、训练、checkpoint 选择或 benchmark。

### 可视化与导出计划

根目录保存 `selection.json`、`prompt_profiles.json`、`controller.log`、`visualization_manifest.json`；各数据集目录保存 `sources.jsonl`、`source_report.json`、`annotations.jsonl`、`rejected.jsonl`、`report.json`、前 100 条 `review.html` 和 `visualization_index.json`。source ID 可回溯 sources 中相对媒体路径；所有原始回答保存于全量 JSONL。运行每 10 条输出进度，模型日志为 `service/instruction/logs/vllm_18462.log`。TensorBoard/训练曲线 N/A：本轮仅标注。文件通过受信 SSH 读取；模型仅监听 loopback。

## 记录实验结果

启动前已确认缺口数量；GPU 0–3 空闲，4–7 有训练任务，保持原样。19:03:27 恢复 `vla-gap-full-20261011`；19:04:34 source 准备已完成 Foxglove 1,659 条，EgoWalk 已构建超过 17,500 条。

19:04:43 两个数据集 source 准备完成并通过四卡空闲 preflight，创建模型会话 `vla-eval-1`。19:05:14 两个 tmux 均在，API PID 12983，模型初始化中，尚未生成本次回答；4–7 号训练保持运行。模型就绪后自动依次处理 1,659 和 17,835 条，正常结束释放本实例。

Shell/Python 语法、实验记录结构与 diff check 通过。portfolio doctor 仍为 1 个历史 artifact 缺失、12 项 snapshot drift；保留 CityWalker 及其他仓既有 dirty changes。无全仓测试、commit 或 push。

19:00 首次准备因 Foxglove 方向清单只覆盖 1,659/1,805 条而退出，GPU 未分配。逐条回查原 observation，另外 146 条均缺 linked trajectory sidecar。正式 teacher 需要方向 GT：这些 ID 单独保存到 `pending_missing_direction.jsonl`，保留为待处理；其余样本继续。此限制来自现有输入依赖，不使用历史 QC 删除候选。EgoWalk 同样核对来源，意外漏选继续报错。

EgoWalk 回查完成：29,651 条缺标中有 11,816 条缺 linked trajectory sidecar，方向清单覆盖 17,835 条。可进入当前 direction-projected source 入口的总数为 **19,494**；另 **11,962** 条保存为缺方向依赖的待处理项。本次执行完成也不能宣称原 31,456 条缺口全部补齐。该依赖属于 source 构建及 GT composer，不代表 LVLM 无法描述这些图像；batch 本身已有无方向监督时 composer `not_available` 分支，后续可适配无 GT 的公开方向合同入口，不能为此伪造方向真值。

正向证据：pilot 35 条均完成，抽查场景与障碍描述有视觉依据。限制：34/35 条 teacher action tuple 与确定性 composer 不同，属于来源差异，不能直接当错误率；个别强动作存在待确认矛盾。全量原回答和 composer 分开保存，暂不导出正式训练集。

### 可视化与产物

启动前已建立执行脚本与 visual manifest。全量 annotations、report 和 review 将随本次运行产生；完成前不宣称已生成。

## 分析实验结果，可能的原因

缺口属于旧 reasoning 未覆盖，不因历史 QC 拒绝而重做。pilot 支持链路可执行及部分场景描述合理；全量内容质量仍为 inconclusive。composer 依据 GT 方向及约束构造动作，和 teacher proposal 属于两个来源，accepted 仅为程序状态。

## 用户讨论与决策记录

| 时间 | 用户讨论/关注点 | 最终决定 | 对实验的影响 | 未解决问题 |
| --- | --- | --- | --- | --- |
| 2026-10-11 | 单 clip 完成后启动全量标注 | 授权全量缺自身 reasoning 补标，最多四卡，tmux 使用 vla 前缀 | 替代单 clip 的执行范围限制；保留已有标签 | 最终 action 监督语义尚待确认 |
| 2026-10-11 19:14 | 增加 worker 并行，提高 GPU 利用率 | annotation concurrency 从 2 增至 4，匹配服务 MAX_NUM_SEQS=4 | 模型仍在加载，仅更新尚未执行的 label.py；保持同一四卡模型及 Prompt | 实际吞吐需标注开始后测量，不能按 worker 数线性预测 |
| 2026-10-12 10:32 | 使用当前空闲卡按原设置继续标注 | 切换至 GPU 4–7，TP=4、concurrency=4、32k、10 图及 Prompt 保持原样 | 断点恢复；新模型日志位于 service-resume-20261012，原产物保留；控制会话 vla-gap-resume-20261012 | 上次中断原因尚未确认 |

## 接下来要做的实验

| 优先级 | 下一实验 | 假设 | 唯一主要变量 | 所需证据 | 通过/停止条件 | 依赖或授权 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 全量缺标补充 | 现有 teacher 可覆盖剩余输入 | 数据范围 | 全量输出、错误记录、分数据集 report | source 构建或服务故障保留 traceback；输出覆盖逐项汇总 | 已授权 |
| P1 | 最终 action 差异核对 | 部分差异来自 composer 来源选择 | teacher 与 composer 的来源 | 定位明显矛盾样本 | 先解释差异再决定训练监督 | 本轮不改标签规则 |
