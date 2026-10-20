# 实验记录：Qwen122B 新窗口单样本推理

experiment_record_version: 1

## 实验元数据

- 实验 ID：`qwen122b-history-smoke-20261011`；状态：running。
- 负责人：UrbanNav annotation；日期：2026-10-11。
- 代码：Urban-Nav-SR `081bd24e1c491796c058e0f452d85611ca04daf0`，本轮 dirty 仅部署脚本图片上限及本记录；原有六个未跟踪文件保留。
- 验证范围：mini-sample；单实例、单样本，无批量 teacher。
- 本地产物：`/home/ubuntu/WorkSpace/project/SR/Code/.artifacts/qwen122b-history-smoke-20261011`。
- 服务器产物：`/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation/model_services/qwen122b_history_smoke_20261011`。

## 实验的目的

验证原 Qwen3.5-122B-A10B 环境在 TP=4 下能够加载并接受真实 9 帧 FPV + 当前 BEV 请求，返回非空回答。单样本通路成功只证明可推理，不证明标签质量或批量吞吐。

## 实验的 setting

- 数据：`/lpai/volumes/mind-eb-bd-ga/suh/dataset/urbannav_history_4s_2hz_v1`，选择一个包含 BEV 的 SideWalk observation；使用新 observation 的相对媒体路径，禁止使用保留的旧 source_row.images。
- 模型：`/lpai/inputs/models/Qwen__Qwen3_5-122B-A10B/26-02-24-1554`，39 个权重分片；使用该固定挂载，无 checkpoint 选择。
- 环境：原 `qwen35_vllm_cuda128`，Python 3.11.13、torch 2.11.0+cu128、vLLM 0.20.2、transformers 5.13.0；GPU CUDA 初始化已通过。
- 硬件：31041，4–7 号 A100 80GB，单 TP=4；保持 0–3 号卡现有任务。
- 基线：原 `serve_qwen35_122b_dual.sh`，图片上限从 6 调整到 10，其余模型后端保持原配置；验证只启动 instruction 角色，端口 18462，loopback。
- 单请求使用 temperature=0，最多输出 512 tokens；提示只检查场景与历史变化，不导出正式标签，不更改方向合同。
- 启动入口：同步后的 `serve_qwen35_122b_dual.sh start instruction`，显式传入独立 SERVICE_ROOT 和 MAX_IMAGES_PER_PROMPT=10。
- 停止条件：服务加载失败或真实请求失败时保存原始异常，停止扩大执行；验证结束关闭本次实例，释放四卡。

### 可视化与导出计划

本地产物根目录的 `visualization_manifest.json`；导出 request metadata、response.json 和 review.html，并绑定 visualization_index.json。训练曲线和分层评测 N/A：只有一次通路验证。

## 记录实验结果

续轮实测：首轮已经完成四卡权重加载，但唯一请求 HTTP 400，原始错误为 `Input length (19568) exceeds model's maximum context length (16384)`；自动清理已完成，旧 tmux 会话全部退出。原始失败日志保留，不视作显存不足。

第二轮仅调整上下文至 32768，单样本服务并发容量设为 4，保持同一模型及全部十张原图。重新确认 0–3 号卡空闲后改用该四卡；4–7 号卡现有占用保持原样。产物单独写入 `/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation/model_services/vla_qwen122b_history_32k_20261011`。本地入口 `.artifacts/qwen122b-history-smoke-20261011/run_32k.sh`；本轮会话命名为 `vla-history-smoke-32k` 和 `vla-eval-1`。第二轮结果尚未产生。下文保留首轮运行中的历史记录。

第二轮于 16:33:57 +08:00 启动，API PID 5680；日志确认 `tensor_parallel_size=4`、`max_model_len=32768`、`image=10`、`max_num_seqs=4`。16:35:09 开始权重读取，最后复核为 1/39 分片、首分片 58.91 秒。两个 vla 会话已实测存在；加载完成后自动执行原单样本，任何退出都会清理本次服务，尚未启动批量标注。

启动前：31041 的 4–7 号 GPU 空闲；18462 无服务；原模型挂载存在，CUDA init 成功。

15:13:04 +08:00 已启动 instruction 单实例，API PID 1827，tmux `vla-eval-1`；服务日志确认 TP=4、image=10 生效。15:14:19 开始加载权重；当前观察到 3/39 分片，权重总量 232.99 GiB，FUSE 首分片耗时 80 秒。加载仍在进行，真实请求结果尚未产生。

单次验证已托管到 tmux `q122b-history-smoke-20261011`，作业入口为服务器产物根目录下 `code/.artifacts/qwen122b-history-smoke-20261011/run_once.sh`。该作业最多等待服务 5400 秒；成功就绪后执行一次 smoke.py，退出时关闭本轮 instruction 实例。日志位于服务器产物根目录 `single_sample.log`。服务启动 CLI 自身的健康等待仍为原默认 1800 秒，该等待超时不等于模型已停止。

选定样本为 `sidewalk:2025-07-13__C10003__9c6c72d9-fe02-4a37-816b-15c16effe13e__15:000110`，package 为 `sidewalk/packages/clips/000000`。请求不使用旧 assistant 答案，也不使用旧 source_row.images。

### 可视化与产物

实际响应及 review 尚未生成；启动前计划和 started.json 已落盘并同步到服务器 code 目录。部署日志为服务器产物根目录 `instruction/logs/vllm_18462.log`。单样本响应完成后才可生成 finished receipt。

## 分析实验结果，可能的原因

当前结论 inconclusive。环境可导入及 CUDA 初始化不能替代权重加载与十图真实请求；本轮以实际 HTTP 响应和服务日志判断。已观察到的延迟发生在 FUSE 权重加载阶段，尚未出现 GPU 初始化或权重加载异常；不能由此推断最终推理耗时。

本地 bash 语法、Python AST、diff whitespace 和实验记录结构检查通过。portfolio doctor 报告 1 个历史 artifact 缺失及 12 个快照漂移；包括 CityWalker 已有 dirty 文件，均保持原样。

## 用户讨论与决策记录

| 时间 | 用户讨论/关注点 | 最终决定 | 对实验的影响 | 未解决问题 |
| --- | --- | --- | --- | --- |
| 2026-10-11 | 图片上限及真实单样本验证 | 授权原环境 TP=4 单样本 | 批量校准、补标及环境升级留待后续 | 真实十图推理待验证 |
| 2026-10-11 续轮 | 最多四张显卡，tmux 使用 vla 变式 | 一个 TP=4 实例，所有新会话 vla 前缀 | 根据真实 token 超限提高上下文至 32k，再验证原十图样本 | 32k 请求结果待验证 |

## 接下来要做的实验

| 优先级 | 下一实验 | 假设 | 唯一主要变量 | 所需证据 | 通过/停止条件 | 依赖或授权 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 新窗口单样本 | 原环境支持十图输入 | 图片窗口 | 请求身份、回答和日志 | 返回非空内容；异常则停止 | 用户已授权 |

通过后再讨论旧标签校准提示与单 clip 校准，后者尚未授权。
