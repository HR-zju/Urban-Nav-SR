# 4s / 2Hz 标注迁移

当前实现状态：部分完成。入口为 `tools/model_processing/plan_history_annotation_migration.py`。
`sources` 构建新 canonical source；`plan` 比较已有 canonical source 的五段 grounding 输入。
`annotate` 接通精确 grounding 复用和剩余 section 补标，支持完整样本及独立 LVLM section 续跑。
Task 另支持已有 instruction-keyed taxonomy cache 语义，记录为 `reuse_instruction`。
`export` 通过原有正式资格门禁导出 reasoning 和 Stage I FullCoT 视图。
`vqa-export` 使用已有 Stage I VQA 投影器导出完成的逐题计划。
当前状态标签的 anchor 复用已接入规划与消费者；跨新旧 source 的 Decision 复用尚未接通。

## 当前优先策略：anchor_label_v1

2026-10-10 用户确认：迁移依据是标签时刻和含义，而不是 teacher 原输入完全相等。
新计划配置显式填写 `"reuse_policy": "anchor_label_v1"`；旧 `exact_input` 保留用于历史计划复现，
不能用其零复用结果推断监督标签失效。新 disposition 为 `reuse_anchor`，离线执行和导出均支持。

- Scene 四个当前状态字段保留原文与 producer。历史增帧、prompt/digest 变化不触发重标。
- Task 继续按 instruction taxonomy 合同处理。
- LocalTraversability 对比 probe 方向含义；Navigation 对比指令、action 方向和实际 BEV 解释。
- InteractionConstraintGrounding 的 schema 是当前约束；带方向的约束保留 action 方向合同检查。
- VQA 比较题意、条件方向、候选项和实际依赖的指令/目标；保留原 answer、质量类型与 producer。
- d3 temporal VQA 单独比较监督时间区间端点，而不是要求全部历史帧列表一致。
- 时间或方向变化、证据缺失进入 `pending`，不自动调用 teacher。

时间映射优先读取 `history_observation.source_ns[-1]` 和 `video_ordinals[-1]`。
legacy canonical 没有时间字段时，可配置 `anchor_records_jsonl`：每行包含
`source_fingerprint`、`clip_id`、`timestamp_ns`、`current_frame_id` 和 `provenance`。
该记录由原采样映射生成，不能把同名文件当成已验证帧。新 source 通常直接读取已保存映射。
同一 anchor 的已证明相同画面也可复用；JPEG 摘要不同只表示这条证据不足，不直接判定时刻变化。

新规则已在已有首个 SideWalk clip 重新规划，目录为本地产物根下
`sidewalk-anchor-reasoning-final/` 和 `sidewalk-anchor-vqa-final/`。VQA 共 442 题：
26 个方向标签、346 个选定旧来源无匹配答案、52 个题目/方向条件变化、15 个待对齐、
3 个对应已核实当前图偏移的待处理题。没有 teacher 调用，也没有把这些计数外推全量。
Scene 的 26 条不再因 producer digest 被拒绝：1 条当前图偏移，25 条等待时间/帧映射。

已知 `000110` 样例：名义 anchor 时刻来自 frame_times[110]；旧图对应 ordinal 109，
新图对应 ordinal 110，相差 48.151552 ms。仅为这个样例保存原映射证据，没有为全 clip
套用减一，也没有自行引入近邻时间容差。现有 239,053 条方向训练行保持原样。

## 严格离线训练子集（2026-10-10）

用户已选择严格复用。对现有 VQA 计划执行：

```bash
python tools/model_processing/plan_history_annotation_migration.py vqa-annotate \
  --run-root "$MIGRATION_ROOT/vqa" --input-root "$MIGRATION_INPUT_ROOT" \
  --dataset-root "$DATASET_ROOT" --offline-only
python tools/model_processing/plan_history_annotation_migration.py vqa-export \
  --run-root "$MIGRATION_ROOT/vqa" \
  --modality-profile-id fpv_temporal_instruction_bev_v1 --accepted-only
```

离线模式创建零个 teacher backend，仅处理 `reuse_exact` 和确定性 e0，其他问题保留原计划。
导出将剩余题写入 `views/pending_questions.jsonl`，通过 `full_plan_complete` 区分子集导出与
全计划完成。曾调用 teacher 的同一目录也会按离线计划资格筛选，其他既有答案保持原文件。
report 分别统计 task 与 provenance；`export_complete` 只表示此次选定范围的导出结束。

`vqa-plan` 配置可用 `task_ids` 明确选择当前产物的题型，例如仅选择
`e0_demonstrated_direction_selection`。这种方向专用产物不代表其他旧 LVLM 标注已被检查。
实际 consumer 行继续交给 ms-swift 的 `build_stage_i_manifest_row`，保留原 loss、泄漏与
历史输入检查。生成记录见 `HISTORY_STRICT_TRAINING_20261010.md`。

VQA 的单题复用逻辑位于 `data/history_vqa_migration.py:migrate_vqa_answer`，已接入
`vqa-plan` 和 `vqa-annotate`。它验证旧 annotation context，使用现有 teacher question builder 和
`evidence_refs` 对比实际可见输入；确认相同后保留 answer，生成新的 source/context lineage。
图片路径变化按已证明的帧身份归一。未被该题引用的历史图片不会单独触发重标；teacher prompt
内的条件和 evidence attributes 仍参与比较。强标签继续交给已有 producer 重建。
`VisionVQATeacherBackend(media_root=...)` 已支持在读取边界绑定数据根目录。
VQA adapter 在验证新九帧历史合同及可见图片/角色一致性后启用 temporal eligibility；
Reasoning 的默认 false 兼容字段保持原样。已在真实首个 SideWalk clip 恢复 26 个 d3 问题，
这一资格仅覆盖 prepared observation 时序，不代表原始传感器同步已核实。

## VQA 批量入口

VQA 使用独立计划目录。配置路径均相对于 `--input-root`：`new_sources` 指向新 Reasoning
canonical source；`old_runs` 的 `sources` 和 `annotations` 指向旧 VQA V2 产物，顺序从低到高优先级。
`media_identity_jsonl` 使用与 grounding 规划相同的已证明帧身份索引。可通过 `capability_json`
指定已有能力合同；省略时读取现有默认合同，并把实际内容冻结到 report。

```bash
python tools/model_processing/plan_history_annotation_migration.py vqa-plan \
  --input-root "$MIGRATION_INPUT_ROOT" --config vqa_migration.json \
  --output "$MIGRATION_ROOT/vqa"

python tools/model_processing/plan_history_annotation_migration.py vqa-annotate \
  --input-root "$MIGRATION_INPUT_ROOT" --run-root "$MIGRATION_ROOT/vqa" \
  --dataset-root "$DATASET_ROOT"
```

`vqa-plan` 只写 sources、逐题 migration plan 和 report；通过字节位置读取选中旧答案。
`vqa-annotate` 使用既有 weak-label / canonical direction / external-interface producer，
accepted 答案逐题 flush 到 `annotations.jsonl`，以新 source 指纹和完整 question 确认续跑位置。
失败写入 report 后抛出异常；已有成功题保留。使用 `--limit` 仅执行计划前 N 题（包含已完成题）。
fixture 已覆盖旧答案复用、其余问题生成、确定性方向重建与零调用续跑；真实服务器数据验证待完成。

```bash
python tools/model_processing/plan_history_annotation_migration.py vqa-export \
  --run-root "$MIGRATION_ROOT/vqa" \
  --modality-profile-id fpv_temporal_instruction_bev_v1
```

EgoWalk 选择不含 BEV 的 profile。导出先按 source 指纹与完整 question 核对计划覆盖，
缺少答案时写入 `views/export_report.json` 并报错。完整计划逐题读取 source/answer，复用既有
VQA training-view 校验和 Stage I 投影器，生成 `views/stage_i_vqa.jsonl`；投影排除项保存在
`views/stage_i_projection_drops.jsonl`。`export_complete` 表示导出执行完毕，保留数量与排除原因
单独计数，不表示全部问题都进入训练，也不表示真实训练 batch 已验证。图像保持数据根相对路径。

现有 `annotate_reasoning_source_v4` 已提供 `reused_groundings` 参数，供迁移计划传入
已核实的旧 LVLM grounding。runner 保留这些 section 的 producer 记录，只调用剩余
grounding 的 teacher，并记录复用来源。已核实的 LVLM Task 同样可复用；确定性 Task 重新计算。
fixture 检查证明选择性跳过调用、完整样本续跑、失败后复用五段 grounding 以及组合中断后复用 Decision，
不代表真实数据复用验收。

## 可迁移路径

输出 source 中的 `images` 和 history payload 路径均相对于 `--dataset-root`。
例如 `sidewalk/packages/clips/000000/media/fpv/000110.jpg`。
源文件身份由这些相对路径及 canonical 内容确定，服务器挂载路径仅在运行时用于文件读取。
迁移包移动到另一块磁盘时，只需更换运行时根目录。
`VisionReasoningV4TeacherBackend(media_root=...)` 在交给 provider 读取图像时绑定根目录，
保留 canonical observation、teacher question 和 source fingerprint 中的相对路径。
该参数用于本地媒体包；原有绝对路径/URL 调用保持省略该参数。

```bash
python tools/model_processing/plan_history_annotation_migration.py sources \
  --dataset-root "$DATASET_ROOT" --dataset sidewalk \
  --bev-tolerance-ns 1500000000 \
  --output "$MIGRATION_ROOT/sidewalk.sources.jsonl"
```

构建器读取 `packages/clips.jsonl` 定位原始 clip，读取每个 package 中的 Stage II 编译报告获取
`max_lag_ns`，再调用原有 canonical source builder。当前帧、采样时间和方向合同由已有 validator
校验。EgoWalk/Foxglove 使用原始 `clip_id/frame_idx`，SideWalk 使用其 clip/frame parent ID。
当前已上传的旧 SideWalk/Foxglove 包缺少 `bev_tolerance_ns`。上例 1.5 秒来自原生成计划的
`max_bev_age_ns`：已扫描 SideWalk 7,121 和 Foxglove 763 份 clip summary，值均一致。
迁移时显式补回该字段，核对现有 age 与 timestamp；已有预算与参数冲突时报告错误。
EgoWalk 没有 BEV，省略该参数。此参数恢复生产合同，既不重采样，也不改变图片、锚点或 GT。
其他数据包应使用自己的已记录预算，不能把本例值当作通用默认值。

## 精确复用规划

配置中的文件路径相对于 `--input-root`；`old_runs` 按 baseline、repair 的顺序排列，后面的
有效 section 优先。每条旧 annotation 必须匹配对应旧 source revision。

```json
{
  "new_sources": "sidewalk.sources.jsonl",
  "media_identity_jsonl": "media_identity.jsonl",
  "section_image_policy_id": "reasoning_v4_section_images_routed_visual_v1",
  "direction_registry_version": "由实际旧 run 和目标合同确定",
  "old_runs": [
    {"sources": "old/baseline.sources.jsonl", "annotations": "old/baseline.annotations.jsonl"},
    {"sources": "old/repair.sources.jsonl", "annotations": "old/repair.annotations.jsonl"}
  ]
}
```

`media_identity.jsonl` 的每行是 `path` 与 `identity`。identity 必须来自已核实的
dataset/clip/source-frame 映射；文件同名不构成帧身份相同的证据。

```bash
python tools/model_processing/plan_history_annotation_migration.py plan \
  --input-root "$MIGRATION_INPUT_ROOT" --config migration.json \
  --output "$MIGRATION_ROOT/plan"
```

输出为 `sources.jsonl`、`migration_plan.jsonl`、`migration_report.json`。
每个选中旧结果的 section 都记录 `origin`：旧 source/annotation 的配置相对路径和
`source_record_index` / `annotation_record_index`。序号从零开始，只计 JSONL 非空记录，
与现有 `iter_jsonl` 一致。迁移输入文件保持原版本，根目录可在运行时重新指定。
计划同时保存 `source_byte_offset` / `annotation_byte_offset`，供补标直接定位 UTF-8 文件中的记录。
规划阶段只全局保存 ID、指纹及文件位置，按 anchor 解析候选，不在内存驻留全部旧标注文本。
失败或缺失的旧 section 继续向低优先级来源查找；全部缺失时记录 `missing`，不指定复用来源。
规划器使用真实 teacher 图片投影及 question builder 比较输入；仅路径变化可由帧身份映射归一。
输入不同暂记为重标候选，后续静态语义复用阶段仍需判断实际依赖。该计数尚不代表最终补标量。
`changed_inputs` 列出已比较的可见输入字段及 `image_frame_identities`，未证明图片身份时
`unresolved_media` 列出具体引用；这些字段用于定位迁移工作，不增加 teacher 调用。

现有 routed-visual 合同的真实图像依赖如下。字段名称中的“Scene”不代表只读取当前图像。

| Section | 实际图像输入 | 本轮迁移约束 |
| --- | --- | --- |
| TaskGrounding | current FPV + 可选 current BEV；registry 命中时走确定性实现 | 优先确定性重建 |
| SceneGrounding | 全部 temporal FPV | 历史窗口变化会改变旧 teacher 输入 |
| LocalTraversability | current FPV | 核对实际当前帧与 affordance probe 合同 |
| NavigationContextGrounding | current FPV + 可选 current BEV | 核对图像、instruction 与方向合同 |
| InteractionConstraintGrounding | 全部 temporal FPV | 核对历史帧及约束适用的方向词表 |
| Decision | current FPV + 可选 current BEV + 五段 grounding | 完成 grounding 后重新 reconciliation/compose |

以上来自现有 image policy 和 section question builder。后者还传入 instruction、direction
context、model-visible context 与 evidence catalog，精确复用继续比较这些真实输入。
仅因 Scene 输出是静态类别就忽略其已读取的历史帧，属于语义近似复用，而非 `reuse_exact`。

## 当前验证与待做

补标入口（使用已有 provider 环境配置，执行此命令会调用 teacher）：

```bash
python tools/model_processing/plan_history_annotation_migration.py annotate \
  --run-root "$MIGRATION_ROOT/plan" --input-root "$MIGRATION_INPUT_ROOT" \
  --dataset-root "$DATASET_ROOT"
```

入口逐样本核对 plan/source revision，按相对路径和记录序号读取选中的旧结果并验证其版本。
旧 annotation 直接使用计划里的字节位置，按需解析选中记录。输出追加到 `annotations.jsonl`；
失败结果保存后抛出异常，report 保留失败样本。重跑跳过已生成有效 candidate 的完整样本。
LVLM Task、四个视觉 grounding 和 Decision 每完成一个便追加并 flush 到 `section_progress.jsonl`。续跑核对 source
fingerprint 与 producer profile，复用其中 accepted 的 section，重跑失败或尚未完成的 section。
该文件是执行进度，不作为训练输入。Task 保存规范化来源；Decision 保存当时的五段 grounding，
恢复时这些依赖相同才复用 proposal，随后重新执行确定性的 reconciliation 和 composer。
确定性 Task 使用原 registry 重新计算，不占用 teacher 调用。
对于未命中 registry 的旧 LVLM Task，迁移计划核对原 producer profile/digest/version，
再使用已有 `task_instruction_fingerprint_v4` 比较规范化指令。指令相同标记
`reuse_instruction`，保持旧 taxonomy 与 producer 记录，图片变化不触发 Task 重标。
这与现有 batch runner 的 instruction-keyed cache 合同一致，不是视觉 section 的近似复用。
当前 registry 命中优先重建；指令变化、缺失或 producer 合同变化继续按原 runner 处理。
`annotation_pass_finished` 仅表示本次遍历结束，使用 `--limit` 时不代表全量结束，也不代表 formal export。

Stage I 正式导出（示例保留历史 FPV、instruction 与 BEV；EgoWalk 应选择不含 BEV 的已有 profile）：

```bash
python tools/model_processing/plan_history_annotation_migration.py export \
  --run-root "$MIGRATION_ROOT/plan" \
  --modality-profile-id fpv_temporal_instruction_bev_v1
```

导出复用现有 `export_reasoning_training_views_v4` 与 Stage I V2 投影器。输出位于 `views/`：
`reasoning_formal.jsonl`、`stage_i_full_cot.jsonl`、`stage_i_projection_drops.jsonl` 和
`export_report.json`。报告路径相对于迁移目录，图像路径继续相对于数据根目录。
需要 review 的样本继续受原有正式资格门禁控制；投影不兼容项写入 drop audit。
该入口目前不载入新 audit 对应的 review/adjudication 文件；VQA 使用独立的 `vqa-export`。

- 本地三个真实数据源各构建一条 source：9/10/10 个图像引用可读取且均为相对路径。
- 本轮重新通过 `lpai-jump` 连通 31041、32011、32660；以下数据核对在 32011 完成。
- 新数据根目录 `/lpai/volumes/mind-eb-bd-ga/suh/dataset/urbannav_history_4s_2hz_v1` 可读，
  所在共享盘剩余约 1.6 TiB。服务器代码目录有大量已有修改，本轮保持只读。
- 对 `reasoning_v4_6_qwen122b_sidewalk_main_20way_routed_visual_full_v1` 的实际扫描：
  新 SideWalk 有 30,428 个唯一 anchor，旧 source 有 45,397 个；交集 29,212 个，均找到旧 annotation。
  其余 1,216 个仅表示此 baseline 未覆盖，不能据此认定没有旧标签。
- 后续补查：这 1,216 个 anchor 分布在 1,176 个 clip，全部属于主线已出现的 clip。
  `sidewalk_extra_20way_routed_visual_full_v2` 的 source 与这些 anchor 没有交集；
  v1 目录只有 logs，`sidewalk_extra_20way_full_20260807/annotations` 为空，其 sources 下仅见 shards 目录。
  旧 VQA SideWalk source 覆盖其中 612 个 anchor；2026-10-09 在 32011 重新扫描确认这 612 个
  anchor 均有旧 VQA annotation。各题覆盖不同，例如 b1 有 416 个、d4 有 352 个，不能将
  anchor 有答案视为每题都有答案。这些仍是存在性计数，尚未证明新输入下可复用。
- 交集内 Task/Scene/LocalTraversability/Interaction 各有 29,212 条 accepted；NavigationContext
  为 22,643 accepted、6,569 rejected；Decision 相应为 22,643 accepted、6,569 skipped。
  这些是旧记录状态计数，尚未经过新输入依赖比较，不是最终可复用数量。
- 一个已连接的 anchor 的旧 source/annotation 指纹一致；旧 FPV 序列为 94/98/102/106/110，
  新序列为 29/39/50/59/70/79/89/100/110。旧 BEV 引用 111，新 latest-causal BEV 引用 101。
  旧方向为 degree10 20-way，新 source 目标为 ego-metric 12-way，需按实际 section 依赖判断。
- 媒体身份限制：打包器 `_package_media_path` 曾按 clip package 内同名文件转换 `source_row`
  的历史路径，其余路径保留为 `external-source://`。这些转换后的追溯引用不是独立的帧身份
  证明；不能直接以它们生成“已核实”的 media identity。应结合原转换规则与采样索引建立映射。
- 2026-10-09 对上述 `000110` anchor 的服务器旧媒体重新检查：六张旧图片均存在。
  新旧 FPV current 都为 1920×1080，但旧 JPEG 为 359,487 bytes，新 JPEG 为 620,329 bytes；
  解码 RGB 的平均绝对差为 22.866/255，PSNR 为 16.220 dB。该对图片尚未确认为相同帧。
  对同 clip 的 157 张新采样图排序，最接近旧图的仍是 000110，但相似度排序只定位候选。
  旧 `build_sidewalk_phase1_multiframe_archive.py:extract_video_frame` 使用
  `ffmpeg -ss (frame_idx / estimate_clip_fps)`，新 `history_media.py` 按 ordinal 顺序解码。
  本地原始 clip 的估算 FPS 为 20.0017263，视频 FPS 为 20；按当前旧代码重放抽帧得到的
  JPEG 与服务器旧 JPEG 字节也不同。因此当前源码和同名文件不足以还原旧实际抽帧来源，
  后续直接比对已定位这张图的来源：服务器旧 `fpv/000110.jpg` 与本地原始 clip 的
  `images/000110.jpg` SHA-256 完全一致。因此该例使用已有 images，而不是上述 seek 分支。
  旧 resolver 优先读取 images；目录同时存在 000000 和 000001，按其现有规则视为 zero-based。
  相邻原视频帧比较得到以下结果（MAE 为 0–255 RGB 平均绝对差）：

  | 旧 images 文件序号 | 最接近的原视频 ordinal | MAE | 次近 MAE |
  | --- | --- | --- | --- |
  | 0 | 0 | 1.446 | 18.651 |
  | 1 | 0 | 1.446 | 18.651 |
  | 94 | 93 | 1.594 | 24.453 |
  | 98 | 97 | 1.604 | 20.669 |
  | 102 | 101 | 1.638 | 21.985 |
  | 106 | 105 | 1.633 | 22.667 |
  | 110 | 109 | 1.630 | 22.853 |

  该样例支持旧 images 存在一帧编号偏移、首帧重复的判断；这不是整个 SideWalk 的已验证规律。
  该 anchor 的新 current 是 ordinal 110，因此旧 current 支持的 LVLM 答案不能标记为
  `reuse_exact`。保持新采样时序与 GT；逐 clip 核对旧图片映射，不通过整体减一或改新 anchor
  迁就旧图片。相邻帧语义近似复用属于额外政策，当前精确复用入口保持原语义。
- 后续须补齐媒体身份索引、静态 section 复用、跨 source Decision 复用及真实数据迁移导出验证。
- 全链路验收保留迁移内建校验及一个真实训练 batch 编码；当前尚未运行该 batch。

旧 source、annotation、review 记录保持原版本；新记录显式保留旧 annotation 来源。
