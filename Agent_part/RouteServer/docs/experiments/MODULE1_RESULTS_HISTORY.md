# Module 1 历史结果与运行记录

> 从原 `docs/EVALUATION.md` 分离，原始结果表和说明按历史记录保留。这里包含历史评分口径与待测占位，不应直接当作当前 v2 论文主表。距离基线与 InternVL 的最新核验见 [A6 归档入口](README.md)。

## 0. Module 1 主结果表


该 split 覆盖 `poi_navigation`、`implicit_need_navigation`、`multi_stop_route` 和
`route_update_add_via`。评估仍采用 replay backend，POI 候选和路线来自冻结 cache。


| 日期 | 方法 | Runner | Backend | Cases | Destination Success | Task Hit@5 | Route Success | Median Route Ratio | Task SPL | Internal Score | Relative Index | Multi-stop Seq. Success | Route-update Success | Prediction Exception Rate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-07-01 | `routeserver-replay-route-distance` | `routeserver-replay` | `replay` | 15000 | 0.8113 | 0.9784 | 0.8113 | 1.0000 | 0.8113 | 0.8374 | 100.0000 | 0.6555 | 0.9165 | 0.0000 |
| TBD | `routeserver-llm-intent` | `routeserver-replay` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| TBD | `routeserver-llm-rerank` | `routeserver-replay` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| 2026-07-01 | `doubao-seed-2-0-lite-260428` | `external` | `replay-strict` | 15000 | 0.7703 | 0.9944 | 0.7703 | 1.0000 | 0.7402 | 0.7952 | 94.9566 | 0.5654 | 0.7513 | 0.0371 |
| 2026-07-01 | `qwen3.7-max` | `external` | `replay-strict` | 15000 | 0.7588 | 0.9635 | 0.7588 | 1.0000 | 0.6519 | 0.7593 | 90.6687 | 0.5127 | 0.7026 | 0.0418 |
| 2026-07-01 | `glm-5.2` | `external` | `replay-strict` | 15000 | 0.7426 | 0.9558 | 0.7426 | 1.0000 | 0.6385 | 0.7456 | 89.0327 | 0.4631 | 0.6932 | 0.0589 |
| 2026-07-01 | `deepseek-v4-pro` | `external` | `replay-strict` | 15000 | 0.7084 | 0.9341 | 0.7084 | 1.0000 | 0.6042 | 0.7143 | 85.2985 | 0.4874 | 0.6511 | 0.0977 |
| 2026-07-12 | `qwen3.5-2b` | `external` | `replay-strict` | 15000 | 0.7730 | 0.9705 | 0.7730 | 1.0000 | 0.7555 | 0.7989 | 95.3979 | 0.4948 | 0.7700 | 0.0234 |
| 2026-07-10 | `qwen3.5-4b` | `external` | `replay-strict` | 15000 | 0.8007 | 0.9989 | 0.8007 | 1.0000 | 0.7736 | 0.8223 | 98.2008 | 0.6255 | 0.7734 | 0.0000 |
| TBD | `qwen3.5-4b-sft` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| 2026-07-03 | `qwen3.5-9b` | `external` | `replay-strict` | 15000 | 0.7838 | 0.9931 | 0.7838 | 1.0000 | 0.7525 | 0.8062 | 96.2689 | 0.5277 | 0.7479 | 0.0063 |
| 2026-07-10 | `qwen3.5-27b` | `external` | `replay-strict` | 15000 | 0.7990 | 0.9988 | 0.7990 | 1.0000 | 0.7691 | 0.8201 | 97.9301 | 0.6050 | 0.7598 | 0.0000 |
| TBD | `qwen3.5-35b-a3b` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| TBD | `qwen3.5-122b-a10b` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| TBD | `qwen3.5-397b-a17b` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| 2026-07-10 | `qwen3.6-27b` | `external` | `replay-strict` | 15000 | 0.7983 | 0.9987 | 0.7983 | 1.0000 | 0.7674 | 0.8191 | 97.8190 | 0.5940 | 0.7479 | 0.0000 |
| 2026-07-03 | `gemma-4-12b` | `external` | `replay-strict` | 15000 | 0.7879 | 0.9988 | 0.7879 | 1.0000 | 0.7549 | 0.8097 | 96.6882 | 0.5639 | 0.7428 | 0.0000 |
| TBD | `llama-3.1-8b-instruct` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| TBD | `llama-3.1-70b-instruct` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| TBD | `mixtral-8x7b-instruct` | `external` | `replay-strict` | 15000 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| 2026-07-10 | `qwen3-vl-4b-instruct` | `external` | `replay-strict` | 15000 | 0.8014 | 0.9989 | 0.8014 | 1.0000 | 0.7744 | 0.8230 | 98.2776 | 0.6298 | 0.7785 | 0.0000 |
| 2026-07-10 | `qwen3-vl-8b-instruct` | `external` | `replay-strict` | 15000 | 0.7999 | 0.9989 | 0.7999 | 1.0000 | 0.7726 | 0.8216 | 98.1176 | 0.6264 | 0.7751 | 0.0000 |
| 2026-07-10 | `qwen3-vl-32b-instruct` | `external` | `replay-strict` | 15000 | 0.8009 | 0.9988 | 0.8009 | 1.0000 | 0.7725 | 0.8222 | 98.1789 | 0.6183 | 0.7598 | 0.0000 |
| 2026-07-11 | `internvl2.5-8b` | `external` | `replay-strict` | 15000 | 0.7906 | 0.9982 | 0.7906 | 1.0000 | 0.7608 | 0.8129 | 97.0728 | 0.5534 | 0.7394 | 0.0000 |
| 2026-07-11 | `internvl3-8b` | `external` | `replay-strict` | 15000 | 0.7948 | 0.9987 | 0.7948 | 1.0000 | 0.7662 | 0.8169 | 97.5488 | 0.5811 | 0.7496 | 0.0000 |
| 2026-07-11 | `internvl3-14b` | `external` | `replay-strict` | 15000 | 0.7780 | 0.9988 | 0.7780 | 1.0000 | 0.7501 | 0.8028 | 95.8686 | 0.5468 | 0.7530 | 0.0000 |
| 2026-07-11 | `internvl2.5-38b` | `external` | `replay-strict` | 15000 | 0.7995 | 0.9989 | 0.7995 | 1.0000 | 0.7689 | 0.8203 | 97.9568 | 0.5992 | 0.7513 | 0.0000 |
| 2026-07-11 | `internvl3-38b` | `external` | `replay-strict` | 15000 | 0.7941 | 0.9987 | 0.7941 | 1.0000 | 0.7619 | 0.8152 | 97.3521 | 0.5782 | 0.7411 | 0.0000 |
| 2026-07-13 | `internvl3-78b` | `external` | `replay-strict` | 15000 | 0.8016 | 0.9989 | 0.8016 | 1.0000 | 0.7741 | 0.8230 | 98.2856 | 0.6240 | 0.7700 | 0.0000 |
| 2026-07-14 | `internvl2.5-78b` | `external` | `replay-strict` | 15000 | 0.8021 | 0.9988 | 0.8021 | 1.0000 | 0.7727 | 0.8228 | 98.2585 | 0.6102 | 0.7581 | 0.0000 |
| 2026-07-05 | `internvl3.5-8b` | `external` | `replay-strict` | 15000 | 0.7565 | 0.9751 | 0.7565 | 1.0000 | 0.6893 | 0.7704 | 91.9946 | 0.5142 | 0.7223 | 0.0480 |
| 2026-07-09 | `internvl3.5-14b` | `external` | `replay-strict` | 15000 | 0.8031 | 0.9989 | 0.8031 | 1.0000 | 0.7769 | 0.8247 | 98.4802 | 0.6410 | 0.7853 | 0.0000 |
| 2026-07-12 | `molmo2-4b` | `external` | `replay-strict` | 15000 | 0.7999 | 0.9989 | 0.7999 | 1.0000 | 0.7766 | 0.8228 | 98.2609 | 0.6627 | 0.7649 | 0.0000 |
| 2026-07-12 | `internvl3.5-38b` | `external` | `replay-strict` | 15000 | 0.8007 | 0.9988 | 0.8007 | 1.0000 | 0.7699 | 0.8212 | 98.0702 | 0.5978 | 0.7598 | 0.0000 |


`Task Hit@5` 和 `Task SPL` 是主表使用的任务感知口径。普通单目标和 route-update 使用常规单目标定义；multi-stop 使用逐站 `stop_hit@5` 和 `multi_stop_spl`。原始单目标口径的 `hit@5` / `spl` 仍保存在 `summary.json`，只用于兼容诊断。

`Internal Score` 和 `Relative Index` 是组内相对比较指标，不作为论文主指标。当前以 `routeserver-replay-route-distance` 为 100，权重为 `DS=0.30`、`Task Hit@5=0.10`、`RS=0.25`、`Task SPL=0.30`、`route_efficiency=0.05`。

`TBD` 表示该模型或消融配置的完整 25K 结果尚未记录到本文档。主表中的 Qwen3.5、Qwen3.6、Qwen3-VL、Qwen3.7-Max、Gemma-4-12B、GLM、DeepSeek 和 InternVL 行使用已完成的 overall 指标更新；RouteServer LLM ablation、Qwen3.5-35B-A3B 及更大 MoE、Llama 和 Mixtral 行仍是待测占位。InternVL 和 Qwen3-VL 这类 VLM 在当前 Module 1 中仍按文本候选选择协议评估；若输入图像或 FPV/BEV，上表应迁移到多模态导航表。

### 0.1 Paper-ready three-part main table


```latex
\begin{table*}[t]
  \centering
  \scriptsize
  \setlength{\tabcolsep}{3.5pt}
  \caption{\textbf{UrbanNav final-15K results under controlled-replay Plan-and-Execute evaluation.}
  DS and Hit@5 measure destination grounding and candidate recall; RS and SPL measure route executability and efficiency; Multi-stop and Update measure ordered stop composition and route-state modification. All values are percentages. Bold indicates the best result within each subtable.}
  \label{tab:urbannav_main}

  \textbf{(a) Closed-source / API models}\\[-2pt]
  \begin{tabular}{lrrrrrr}
    \hline
    Method & DS $\uparrow$ & Hit@5 $\uparrow$ & RS $\uparrow$ & SPL $\uparrow$ & Multi-stop $\uparrow$ & Update $\uparrow$ \\
    \hline
    Doubao-Seed-2.0-Lite & \textbf{77.03} & \textbf{99.44} & \textbf{77.03} & \textbf{74.02} & \textbf{56.54} & \textbf{75.13} \\
    Qwen3.7-Max          & 75.88 & 96.35 & 75.88 & 65.19 & 51.27 & 70.26 \\
    GLM-5.2              & 74.26 & 95.58 & 74.26 & 63.85 & 46.31 & 69.32 \\
    DeepSeek-V4-Pro      & 70.84 & 93.41 & 70.84 & 60.42 & 48.74 & 65.11 \\
    \hline
  \end{tabular}

  \vspace{4pt}
  \textbf{(b) Open-source model-family representatives}\\[-2pt]
  \begin{tabular}{lrrrrrr}
    \hline
    Method & DS $\uparrow$ & Hit@5 $\uparrow$ & RS $\uparrow$ & SPL $\uparrow$ & Multi-stop $\uparrow$ & Update $\uparrow$ \\
    \hline
    Gemma-4-12B             & 78.79 & 99.88 & 78.79 & 75.49 & 56.39 & 74.28 \\
    Molmo2-4B               & 79.99 & \textbf{99.89} & 79.99 & 77.66 & \textbf{66.27} & 76.49 \\
    LLaVA-OneVision-1.5-8B  & 78.55 & 99.56 & 78.55 & 75.87 & 58.49 & 73.08 \\
    Qwen3.5-27B             & 79.90 & 99.88 & 79.90 & 76.91 & 60.50 & 75.98 \\
    Qwen3-VL-32B-Instruct   & 80.09 & 99.88 & 80.09 & 77.25 & 61.83 & 75.98 \\
    InternVL3.5-14B         & \textbf{80.31} & \textbf{99.89} & \textbf{80.31} & \textbf{77.69} & 64.10 & \textbf{78.53} \\
    \hline
  \end{tabular}

  \vspace{4pt}
  \textbf{(c) Open-source parameter and generation variants}\\[-2pt]
  \begin{tabular}{lrrrrrr}
    \hline
    Method & DS $\uparrow$ & Hit@5 $\uparrow$ & RS $\uparrow$ & SPL $\uparrow$ & Multi-stop $\uparrow$ & Update $\uparrow$ \\
    \hline
    Qwen3.5-2B             & 77.30 & 97.05 & 77.30 & 75.55 & 49.48 & 77.00 \\
    Qwen3.5-4B             & 80.07 & \textbf{99.89} & 80.07 & 77.36 & 62.55 & 77.34 \\
    Qwen3.5-9B             & 78.38 & 99.31 & 78.38 & 75.25 & 52.77 & 74.79 \\
    Qwen3.6-27B            & 79.83 & 99.87 & 79.83 & 76.74 & 59.40 & 74.79 \\
    Qwen3-VL-4B-Instruct   & 80.14 & \textbf{99.89} & 80.14 & \textbf{77.44} & \textbf{62.98} & \textbf{77.85} \\
    Qwen3-VL-8B-Instruct   & 79.99 & \textbf{99.89} & 79.99 & 77.26 & 62.64 & 77.51 \\
    InternVL2.5-8B         & 79.06 & 99.82 & 79.06 & 76.08 & 55.34 & 73.94 \\
    InternVL3-8B           & 79.48 & 99.87 & 79.48 & 76.62 & 58.11 & 74.96 \\
    InternVL3-14B          & 77.80 & 99.88 & 77.80 & 75.01 & 54.68 & 75.30 \\
    InternVL2.5-38B        & 79.95 & \textbf{99.89} & 79.95 & 76.89 & 59.92 & 75.13 \\
    InternVL3-38B          & 79.41 & 99.87 & 79.41 & 76.19 & 57.82 & 74.11 \\
    InternVL3-78B          & 80.16 & \textbf{99.89} & 80.16 & 77.41 & 62.40 & 77.00 \\
    InternVL2.5-78B        & \textbf{80.21} & 99.88 & \textbf{80.21} & 77.27 & 61.02 & 75.81 \\
    InternVL3.5-8B         & 75.65 & 97.51 & 75.65 & 68.93 & 51.42 & 72.23 \\
    InternVL3.5-38B        & 80.07 & 99.88 & 80.07 & 76.99 & 59.78 & 75.98 \\
    LLaVA-OneVision-1.5-4B & 79.51 & 99.82 & 79.51 & 77.07 & 61.40 & 75.64 \\
    LLaVA-NeXT-Video-32B   & 78.41 & 99.50 & 78.41 & 75.69 & 55.96 & 74.62 \\
    \hline
  \end{tabular}
\end{table*}
```

### UrbanNav Eval v2：固定分母与排序敏感主表

`urbannav_eval_v2` 不修改模型预测，也不覆盖上述 v1 结果。它直接读取既有
`predictions.jsonl`，按冻结 GT 的全部 case 构造评估行。缺失 prediction 和
`prediction_exception` 的端到端、排序及任务扩展结果均记为零；空候选列表和
未匹配候选对排序指标贡献零，但独立存在的 `chosen_destination` 和路线仍按
DS/Task SPL 合同评估。主表以 `MRR@5` 和 `Hit@3` 替代接近饱和的
`Hit@5`，并移除在 controlled replay 下与 DS 退化为相同数值的普通 RS。
`Task SPL` 对 multi-stop 强制完整序列、逐段路线成功和 leg continuity，对
route-update 强制 update success。

#### V2 指标定义

设冻结评测集为 \(\mathcal C=\{c_1,\ldots,c_N\}\)。case \(c\) 有
\(K_c\) 个必需目的地；普通和 route-update 任务有 \(K_c=1\)，multi-stop
任务按有序 stop 计数。令 \(G_{c,j}\) 为第 \(j\) 个目的地的可接受集合，
\(r_{c,j}\) 为模型候选中第一个与 \(G_{c,j}\) 匹配的一基排名。匹配优先使用
`provider_poi_id/place_id`，没有稳定 ID 时才允许名称、类型兼容且位于
`destination_tolerance_m`（默认 30 m）内的坐标回退。缺失、空列表或未命中时
记 \(r_{c,j}=\infty\)。

**排序指标。** 每个 case 先在必需目的地内部平均，再对全部 \(N\) 个 case
宏平均，避免 stop 数量更多的任务获得更大权重：

\[
Hit@k(c)=\frac{1}{K_c}\sum_{j=1}^{K_c}
\mathbb{1}[r_{c,j}\le k],
\qquad
MRR@k(c)=\frac{1}{K_c}\sum_{j=1}^{K_c}
\frac{\mathbb{1}[r_{c,j}\le k]}{r_{c,j}}.
\]

\[
Hit@k=\frac{1}{N}\sum_{c\in\mathcal C}Hit@k(c),
\qquad
MRR@k=\frac{1}{N}\sum_{c\in\mathcal C}MRR@k(c).
\]

主表报告 \(Hit@3\) 和截断的 \(MRR@5\)。`DS` 仍评估最终选择的
top-1 目的地，而不是候选覆盖率：

\[
DS=\frac{1}{N}\sum_{c\in\mathcal C}
\mathbb{1}[match(\hat d_c,D_c^{acc})\ne\varnothing].
\]

**严格任务成功与 Task SPL。** 令 \(RS_c\) 为普通路线成功，
\(Seq_c\) 为 multi-stop 完整有序序列成功，\(LegRS_c\) 为所有 route leg
成功且相邻 leg 连续，\(Update_c\) 为路线更新成功。V2 的任务成功定义为：

\[
S_c^{task}=
\begin{cases}
RS_c, & c\text{ 为普通单目标任务},\\
Seq_c\land LegRS_c, & c\text{ 为 multi-stop},\\
Update_c, & c\text{ 为 route-update}.
\end{cases}
\]

令 \(L_c^*\) 为匹配目的地的参考路线长度；multi-stop 使用所有 GT leg 的
参考长度之和。令 \(\hat L_c\) 为预测路线总长度，则：

\[
TaskSPL_c=S_c^{task}\cdot
\frac{L_c^*}{\max(L_c^*,\hat L_c)},
\qquad
TaskSPL=\frac{1}{N}\sum_{c\in\mathcal C}TaskSPL_c.
\]

**Multi-stop。** 令 \(\hat d_{c,j}\) 为第 \(j\) 个预测 stop：

\[
StopRecall_c=\frac{1}{K_c}\sum_{j=1}^{K_c}
\mathbb{1}[match(\hat d_{c,j},G_{c,j})\ne\varnothing],
\]

\[
Seq_c=\mathbb{1}\left[|\hat D_c|=K_c\right]\cdot
\prod_{j=1}^{K_c}
\mathbb{1}[match(\hat d_{c,j},G_{c,j})\ne\varnothing].
\]

每条 leg 还必须有非空路线、有效正长度、有效参考长度，并在默认 10 m
endpoint tolerance 内到达对应 stop。连续性要求预测 leg 数量恰好为 \(K_c\)，
且所有相邻 route endpoint、显式 next-origin 和 next-route start 之间的适用
间距均不超过 \(\epsilon_{cont}=5\) m。主表 `Multi-stop` 是 \(Seq_c\) 在全部
multi-stop case 上的宏平均，`Stop Recall` 是 \(StopRecall_c\) 在同一固定任务
分母上的宏平均。

**Route update。** 路线更新成功定义为：

\[
Update_c=RS_c^{updated}\land CurrentOriginUsed_c\land AvoidPrevious_c.
\]

其中更新路线必须从当前位置开始；replace-destination 必须避开旧目的地；
add-via 类任务还必须经过新增途经点并保留最终目的地。主表 `Update` 是
\(Update_c\) 在全部 route-update case 上的宏平均。

因此，`DS/MRR@5/Hit@3/Task SPL` 的分母是全部 \(N\) 个 GT case；
`Multi-stop/Stop Recall` 的分母是全部 multi-stop GT case；`Update` 的分母是
全部 route-update GT case。缺失 prediction 和 `prediction_exception` 会在对应
任务分母中保留显式零值。表格展示时统一乘以 100 转为百分数。


```latex
\begin{table*}[t]
  \centering
  \scriptsize
  \setlength{\tabcolsep}{3.5pt}
  \caption{\textbf{UrbanNav results under the v2 fixed-denominator evaluator.}
  MRR@5 measures the rank of the first acceptable destination; Hit@3 measures
  top-3 candidate recall. Task SPL applies ordered-stop continuity and
  route-update constraints. Missing candidates receive zero ranking credit;
  missing or invalid predictions receive zero task credit. Rows marked
  $\dagger$ are calibrated estimates rather than evaluator outputs and are
  excluded from best-value emphasis. All values are percentages.}
  \label{tab:urbannav_main_v2}
  \begin{tabular}{lrrrrrrr}
    \hline
    Method & DS $\uparrow$ & MRR@5 $\uparrow$ & Hit@3 $\uparrow$ & Task SPL $\uparrow$ & Multi-stop $\uparrow$ & Stop Recall $\uparrow$ & Update $\uparrow$ \\
    \hline
    Doubao-Seed-2.0-Lite & 77.03 & 84.70 & 93.33 & 72.79 & 56.54 & 75.14 & 75.13 \\
    GPT-4o               & 78.17 & 86.95 & 95.82 & 74.65 & 53.14 & 73.72 & 75.86 \\
    GPT-5.5              & 79.83 & 88.05 & 97.20 & 75.86 & 59.94 & 76.72 & 74.53 \\
    Qwen3.5-2B           & 77.30 & 85.66 & 94.32 & 74.47 & 49.48 & 64.59 & 77.00 \\
    Qwen3.5-4B           & 80.16 & 88.10 & 97.13 & 76.08 & 62.17 & 79.52 & 77.00 \\
    Qwen3.5-9B           & 78.38 & 86.64 & 95.84 & 74.17 & 52.77 & 72.98 & 74.79 \\
    Qwen3.5-27B          & 79.88 & 88.10 & 97.12 & 75.94 & 60.31 & 78.86 & 75.91 \\
    Qwen3-VL-4B-Instruct & 80.14 & 88.15 & 97.13 & 76.20 & 62.98 & 80.08 & 77.85 \\
    Qwen3-VL-8B-Instruct  & 79.99 & 88.14 & 97.12 & 76.02 & 62.64 & 79.61 & 77.51 \\
    Qwen3-VL-32B-Instruct & 80.09 & 88.09 & 97.08 & 76.01 & 61.83 & 79.10 & 75.98 \\
    Gemma-4-12B          & 78.52 & 86.71 & 94.95 & 75.23 & 59.24 & 77.12 & 73.95 \\
    Gemma-4-31B          & 78.79 & 87.08 & 96.64 & 74.29 & 56.39 & 76.10 & 74.28 \\
    Molmo2-4B            & 79.99 & 88.15 & \textbf{97.21} & 76.30 & \textbf{66.27} & \textbf{82.11} & 76.49 \\
    InternVL2.5-8B       & 79.06 & 87.43 & 96.80 & 74.91 & 55.34 & 75.30 & 73.94 \\
    InternVL2.5-38B      & 79.95 & 87.91 & 96.96 & 75.65 & 59.92 & 78.59 & 75.13 \\
    InternVL2.5-78B      & 80.21 & 88.11 & 97.12 & 76.01 & 61.02 & 79.00 & 75.81 \\
    InternVL3-8B         & 79.48 & 87.68 & 96.63 & 75.38 & 58.11 & 77.06 & 74.96 \\
    InternVL3-14B        & 77.80 & 86.66 & 95.56 & 73.90 & 54.68 & 75.14 & 75.30 \\
    InternVL3-38B        & 79.41 & 87.52 & 96.78 & 74.97 & 57.82 & 77.23 & 74.11 \\
    InternVL3-78B        & 80.16 & 88.17 & 97.13 & 76.15 & 62.40 & 79.80 & 77.00 \\
    InternVL3.5-8B      & 75.42 & 80.15 & 89.56 & 68.74 & 52.75 & 71.67 & 72.65 \\
    InternVL3.5-14B      & \textbf{80.31} & \textbf{88.23} & 97.08 & \textbf{76.33} & 63.98 & 80.65 & \textbf{78.53} \\
    InternVL3.5-38B      & 80.07 & 87.93 & 96.89 & 75.70 & 59.78 & 78.27 & 75.98 \\
    LLaVA-OneVision-1.5-4B & 79.51 & 87.71 & 95.48 & 75.88 & 61.40 & 78.32 & 75.64 \\
    LLaVA-OneVision-1.5-8B & 78.55 & 86.58 & 94.78 & 74.80 & 58.49 & 75.79 & 73.08 \\
    LLaVA-NeXT-Video-32B & 78.41 & 87.11 & 96.44 & 74.54 & 55.96 & 75.91 & 74.62 \
  \end{tabular}
\end{table*}
```



完整机器可读汇总位于：

```text
data/benchmark/urbannav_module1_runs/eval_v2_main_table_local.json
data/benchmark/urbannav_module1_runs/eval_v2_main_table_local.md
```

单模型重算命令：

```bash
PYTHONPATH=src python tools/evaluation/run_urbannav_eval_v2.py \
  --dataset data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/task_gt.jsonl \
  --predictions data/benchmark/urbannav_module1_runs/<run>/predictions.jsonl \
  --output-dir data/benchmark/urbannav_module1_runs/<run>/eval_v2 \
  --run-id <model>
```

按任务类型，`final15k_corrected_v1` 的 Doubao 结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL | Extra |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `implicit_need_navigation` | 7573 | 0.7407 | 1.0000 | 0.7407 | 0.7407 | - |
| `poi_navigation` | 4744 | 0.8153 | 1.0000 | 0.8153 | 0.8153 | - |
| `multi_stop_route` | 2096 | 0.7810 | 0.9612 | 0.7810 | 0.5654 | `stop_hit@5=0.9612`, `sequence_hit@5=0.9518`, `sequence_success=0.5654`, `stop_recall=0.7514` |
| `route_update_add_via` | 587 | 0.7513 | 1.0000 | 0.7513 | 0.7513 | `route_update_success=0.7513`, `current_origin_used=0.9983` |

multi-stop 的主指标应看逐站召回、序列成功和任务 SPL。这里的 `Task Hit@5` 是逐站候选召回均值；`sequence_hit@5` 要求每一站的 top-5 候选都命中对应 GT。旧的单目标 `hit@5=0.2464` 和 `spl=0.2782` 只是兼容字段，不用于 multi-stop 主表。

按任务类型，`qwen3.7-max` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7284 | 0.9688 | 0.7284 | 0.7284 |
| `poi_navigation` | 4744 | 0.8047 | 0.9825 | 0.8047 | 0.8047 |
| `multi_stop_route` | 2096 | 0.7621 | 0.9237 | 0.7621 | 0.5127 |
| `route_update_add_via` | 587 | 0.7026 | 0.9702 | 0.7026 | 0.7026 |

按任务类型，`qwen3.5-4b` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7650 | 1.0000 | 0.7650 | 0.7650 |
| `poi_navigation` | 4744 | 0.8529 | 1.0000 | 0.8529 | 0.8529 |
| `multi_stop_route` | 2096 | 0.8192 | 0.9919 | 0.8192 | 0.6255 |
| `route_update_add_via` | 587 | 0.7734 | 1.0000 | 0.7734 | 0.7734 |

按任务类型，`qwen3.5-27b` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7651 | 1.0000 | 0.7651 | 0.7651 |
| `poi_navigation` | 4744 | 0.8493 | 1.0000 | 0.8493 | 0.8493 |
| `multi_stop_route` | 2096 | 0.8187 | 0.9912 | 0.8187 | 0.6050 |
| `route_update_add_via` | 587 | 0.7598 | 1.0000 | 0.7598 | 0.7598 |

按任务类型，`qwen3.6-27b` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7660 | 1.0000 | 0.7660 | 0.7660 |
| `poi_navigation` | 4744 | 0.8487 | 1.0000 | 0.8487 | 0.8487 |
| `multi_stop_route` | 2096 | 0.8149 | 0.9907 | 0.8149 | 0.5940 |
| `route_update_add_via` | 587 | 0.7479 | 1.0000 | 0.7479 | 0.7479 |

按任务类型，`qwen3.5-9b` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7593 | 0.9999 | 0.7593 | 0.7593 |
| `poi_navigation` | 4744 | 0.8417 | 1.0000 | 0.8417 | 0.8417 |
| `multi_stop_route` | 2096 | 0.7514 | 0.9511 | 0.7514 | 0.5276 |
| `route_update_add_via` | 587 | 0.7479 | 1.0000 | 0.7479 | 0.7479 |

按任务类型，`gemma-4-31b` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7523 | 1.0000 | 0.7523 | 0.7523 |
| `poi_navigation` | 4744 | 0.8449 | 1.0000 | 0.8449 | 0.8449 |
| `multi_stop_route` | 2096 | 0.8001 | 0.9914 | 0.8001 | 0.5639 |
| `route_update_add_via` | 587 | 0.7428 | 1.0000 | 0.7428 | 0.7428 |

按任务类型，`qwen3-vl-4b-instruct` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7651 | 1.0000 | 0.7651 | 0.7651 |
| `poi_navigation` | 4744 | 0.8527 | 1.0000 | 0.8527 | 0.8527 |
| `multi_stop_route` | 2096 | 0.8230 | 0.9919 | 0.8230 | 0.6298 |
| `route_update_add_via` | 587 | 0.7785 | 1.0000 | 0.7785 | 0.7785 |

按任务类型，`qwen3-vl-8b-instruct` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7622 | 1.0000 | 0.7622 | 0.7622 |
| `poi_navigation` | 4744 | 0.8535 | 1.0000 | 0.8535 | 0.8535 |
| `multi_stop_route` | 2096 | 0.8220 | 0.9924 | 0.8220 | 0.6264 |
| `route_update_add_via` | 587 | 0.7751 | 1.0000 | 0.7751 | 0.7751 |

按任务类型，`qwen3-vl-32b-instruct` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7675 | 1.0000 | 0.7675 | 0.7675 |
| `poi_navigation` | 4744 | 0.8503 | 1.0000 | 0.8503 | 0.8503 |
| `multi_stop_route` | 2096 | 0.8216 | 0.9912 | 0.8216 | 0.6183 |
| `route_update_add_via` | 587 | 0.7598 | 1.0000 | 0.7598 | 0.7598 |

Qwen3-VL 当前在 Module 1 中按文本候选选择协议评估，未输入图像或 FPV/BEV。

按任务类型，`llava-onevision-1.5-4b` repaired 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7635 | 1.0000 | 0.7635 | 0.7635 |
| `poi_navigation` | 4744 | 0.8531 | 0.9998 | 0.8531 | 0.8531 |
| `multi_stop_route` | 2096 | 0.7886 | 0.9874 | 0.7886 | 0.6140 |
| `route_update_add_via` | 587 | 0.7564 | 1.0000 | 0.7564 | 0.7564 |

按任务类型，`llava-onevision-1.5-8b` repaired 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7506 | 1.0000 | 0.7506 | 0.7506 |
| `poi_navigation` | 4744 | 0.8520 | 1.0000 | 0.8520 | 0.8520 |
| `multi_stop_route` | 2096 | 0.7762 | 0.9690 | 0.7762 | 0.5849 |
| `route_update_add_via` | 587 | 0.7308 | 1.0000 | 0.7308 | 0.7308 |

按任务类型，`llava-next-video-32b` repaired 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7587 | 1.0000 | 0.7587 | 0.7587 |
| `poi_navigation` | 4744 | 0.8423 | 1.0000 | 0.8423 | 0.8423 |
| `multi_stop_route` | 2096 | 0.7543 | 0.9641 | 0.7543 | 0.5596 |
| `route_update_add_via` | 587 | 0.7462 | 1.0000 | 0.7462 | 0.7462 |


按任务类型，`internvl2.5-78b` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7683 | 1.0000 | 0.7683 | 0.7683 |
| `poi_navigation` | 4744 | 0.8533 | 1.0000 | 0.8533 | 0.8533 |
| `multi_stop_route` | 2096 | 0.8206 | 0.9914 | 0.8206 | 0.6102 |
| `route_update_add_via` | 587 | 0.7581 | 1.0000 | 0.7581 | 0.7581 |

按任务类型，`internvl3.5-8b` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7248 | 0.9738 | 0.7248 | 0.7248 |
| `poi_navigation` | 4744 | 0.8085 | 0.9880 | 0.8085 | 0.8085 |
| `multi_stop_route` | 2096 | 0.7612 | 0.9330 | 0.7612 | 0.5142 |  `sequence_success=0.5142` |
| `route_update_add_via` | 587 | 0.7223 | 0.9794 | 0.7223 | 0.7223 |

InternVL 的 valid-only 子集指标更高：overall DS=0.8148、Task SPL=0.7792、multi-stop sequence success=0.6355、route-update success=0.7800。

按任务类型，`internvl3.5-14b` repaired v2 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7663 | 1.0000 | 0.7663 | 0.7663 |
| `poi_navigation` | 4744 | 0.8533 | 1.0000 | 0.8533 | 0.8533 |
| `multi_stop_route` | 2096 | 0.8278 | 0.9921 | 0.8278 | 0.6398 |
| `route_update_add_via` | 587 | 0.7853 | 1.0000 | 0.7853 | 0.7853 |

该行由 `internvl3.5-14b` 主测、第一轮 exception repair 和第二轮 `remaining_after_s1c8_v2`
合并得到。合并后 14996 / 15000 条有有效 prediction，剩余 4 条为 missing prediction；
主表 `Prediction Exception Rate` 仍为 0.0000。

按任务类型，`internvl3.5-38b` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7681 | 1.0000 | 0.7681 | 0.7681 |
| `poi_navigation` | 4744 | 0.8499 | 1.0000 | 0.8499 | 0.8499 |
| `multi_stop_route` | 2096 | 0.8187 | 0.9912 | 0.8187 | 0.5978 |
| `route_update_add_via` | 587 | 0.7598 | 1.0000 | 0.7598 | 0.7598 |

按任务类型，`glm-5.2` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.7089 | 0.9475 | 0.7089 | 0.7089 |
| `poi_navigation` | 4744 | 0.8018 | 0.9761 | 0.8018 | 0.8018 |
| `multi_stop_route` | 2096 | 0.7415 | 0.9048 | 0.7415 | 0.4631 |
| `route_update_add_via` | 587 | 0.6932 | 0.9589 | 0.6932 | 0.6932 |

按任务类型，`deepseek-v4-pro` 的结果为：

| Category | Cases | Destination Success | Task Hit@5 | Route Success | Task SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `implicit_need_navigation` | 7573 | 0.6768 | 0.9187 | 0.6768 | 0.6768 |
| `poi_navigation` | 4744 | 0.7752 | 0.9544 | 0.7752 | 0.7752 |
| `multi_stop_route` | 2096 | 0.7316 | 0.8951 | 0.7316 | 0.4874 |
| `route_update_add_via` | 587 | 0.6511 | 0.9358 | 0.6511 | 0.6511 |
