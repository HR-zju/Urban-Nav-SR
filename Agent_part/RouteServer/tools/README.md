# RouteServer Tools

`tools/` 按职责分区，避免数据处理、模型处理、数学工具和部署脚本混在同一层。新增脚本优先放到对应子目录；只有跨模块共享的稳定入口才应考虑放到更高层。

## 目录结构

| 目录 | 用途 |
|------|------|
| `dataset_processing/` | MCAP/ROS bag 转换、sidewalk 数据生成、chunk 批处理、数据扫描 |
| `model_processing/` | SAM3、SoM、Visual Prompt、Phase2/训练样本生成 |
| `bev/` | BEV 地图渲染、BEV 数据集构建、轨迹到 BEV 的独立工具 |
| `math_tools/` | 位姿、航向、EKF 和轨迹诊断等数学工具 |
| `route_debug/` | Route context 捕获、调试和指标统计 |
| `docker/` | Dockerfile、镜像构建、部署和服务器批处理脚本 |
| `archive/legacy_tests/` | 旧的手动测试脚本，仅作参考，不作为主流程入口 |

## SideWalk 主流程命令速查

以下命令按实际开发顺序组织。默认原始数据根目录：

```bash
SIDEWALK_DATASET="/media/ubuntu/data1/SR-Nav Dataset/sidewalk"
CLIP="${SIDEWALK_DATASET}/chunk_0/<clip_id>"
```

### 1. 扫描 chunk/clip 状态

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

python tools/dataset_processing/scan_chunk.py \
  --sidewalk-root "${SIDEWALK_DATASET}/chunk_0" \
  --out-csv "${SIDEWALK_DATASET}/chunk_0/scan_summary.csv"
```

用途：检查每个 clip 是否已有 `front.mp4`、pose、GPS、BEV、Phase1 产物等，适合批处理前确认缺口。

### 2. 生成单个 clip 的最简 Phase1 BEV

```bash
python tools/dataset_processing/process_sidewalk_dataset.py \
  --sidewalk-root "${CLIP}" \
  --bev-sample-fps 2.0 \
  --map-provider osm \
  --use-pose-ekf
```

默认写回原始 clip 目录：

```text
trajectory.json
samples.json
bev_images/
bev_index.json
```

说明：

- `bev_images/` 是 2fps ego-aligned BEV，默认画完整 route/start/end。
- `bev_index.json` 记录 BEV 帧与原始 FPV timestamp 的映射。
- `--use-pose-ekf` 现在只用于 heading 平滑；BEV center 和 route overlay 仍使用 GPS-smoothed 位置，避免无平移里程计的 EKF 把路线距离压缩。
- `--use-ekf-position` 仅用于诊断/复现旧行为，默认不要用于数据集生成。
- 默认不生成 `bev_video*.mp4`、north-up BEV、重复 FPV 抽帧；这些只在 debug 参数显式打开。

### 3. 批量生成 sidewalk 基础输入和 BEV

```bash
python tools/dataset_processing/process_chunks_26_29.py \
  --dataset-root "/media/ubuntu/data1/SR-Nav Dataset" \
  --sidewalk-root "${SIDEWALK_DATASET}" \
  --chunks 0 \
  --sample-fps 20.0 \
  --generate-bev \
  --use-pose-ekf \
  --num-workers 4
```

用途：批量从 chunk 生成 `front.mp4`、pose/GPS 中间文件和最简 BEV。若使用 `--skip-existing`，注意旧逻辑可能按 `bev_video.mp4` 判断已有产物；需要重生 Phase1 BEV 时，建议先清理对应 clip 下的 `bev_images/`、`bev_index.json` 或去掉 `--skip-existing`。

### 4. 生成 Phase1 source JSONL 和几何真值

```bash
SIDEWALK_ROOT="${CLIP}" \
OUTPUT_ROOT="${CLIP}/phase1" \
SPLIT_NAME=clip_test \
FRAME_STEP=10 \
WRITE_OVERLAYS=1 \
./tools/model_processing/run_sidewalk_phase1_dataset.sh
```

主要产物：

```text
phase1/manifests/phase1_source_clip_test.jsonl
phase1/clips/<clip_id>/trajectory_labels/*.json
phase1/clips/<clip_id>/bev_patch_meta/*.json
phase1/clips/<clip_id>/qa/*_qa.json
phase1/clips/<clip_id>/overlays/*.jpg
```

这一步是“真值生成”主入口：

- `trajectory_labels/*.json`：FPV normalized trajectory GT、intent、route theta。
- `bev_patch_meta/*.json`：BEV grid、ego/target cell、route cells、intent。
- `qa/*_qa.json`：离线 QA bundle。
- FPV 高频样本会通过 `bev_index.json` 匹配最近的 2fps BEV，默认最大时间差 `300ms`。

如需旧 GNSS/`trajectory.json` smoke fallback，必须显式允许：

```bash
ALLOW_HEURISTIC_TRAJECTORY=1 ./tools/model_processing/run_sidewalk_phase1_dataset.sh
```

### 5. 生成 CoT teacher 标注

```bash
LLM_PROVIDER=ark \
ARK_API_KEY="<your-key>" \
LLM_MODEL="doubao-seed-2-0-pro-260215" \
OUTPUT_ROOT="${CLIP}/phase1" \
SPLIT_NAME=clip_test \
NUM_POINTS=16 \
FALLBACK_POLICY=drop \
./tools/model_processing/run_sidewalk_phase1_cot_teacher.sh
```

主要产物：

```text
phase1/manifests/phase1_cot_clip_test.jsonl
phase1/manifests/phase1_cot_report_clip_test.json
```

输出格式固定为 Phase1 CoT v4：

```xml
<think>
<Scene>{...}</Scene>
<FPVScene>{...}</FPVScene>
<BEVContext>{...}</BEVContext>
<CrossView>{...}</CrossView>
<Plan>{...}</Plan>
</think>
<finish>{"trajectory":[[x,y],...],"intent":"...","route_theta_deg":...,"confidence_level":"..."}</finish>
```

### 6. 归档为 ms-swift 直接训练数据

```bash
OUTPUT_ROOT="${CLIP}/phase1" \
ARCHIVE_ROOT="${CLIP}/phase1_archive" \
SPLIT_NAME=clip_test \
MEDIA_MODE=hardlink \
PATH_MODE=absolute \
WRITE_REVIEW_HTML=1 \
REVIEW_LIMIT=200 \
./tools/model_processing/run_sidewalk_phase1_archive.sh
```

这一步先判定 CoT 标注是否合理，再把通过校验的样本写成 ms-swift 直接可训练 JSONL。失败样本不会进入训练集，而是写到 `errors.jsonl`。

主要产物：

```text
phase1_archive/manifests/clip_test.jsonl          # ms-swift 直接训练输入
phase1_archive/manifests/clip_test.errors.jsonl   # 被拒绝样本和原因
phase1_archive/manifests/clip_test.review.csv     # 人工审计索引
phase1_archive/manifests/clip_test.stats.json     # 接受率、intent、teacher 来源等统计
phase1_archive/media/clips/<clip_id>/fpv/*.jpg    # hardlink/copy/symlink/reference
phase1_archive/media/clips/<clip_id>/bev/*.jpg
phase1_archive/sidecars/clips/<clip_id>/*         # label/meta/qa/cot_sections
phase1_archive/review/index.html                  # 可选人工检查页
```

本地训练推荐：

```text
MEDIA_MODE=hardlink
PATH_MODE=absolute
```

发布/上传数据集推荐：

```text
MEDIA_MODE=copy
PATH_MODE=relative
```

Archive 校验内容包括：

- FPV/BEV 路径存在。
- user prompt 中 `<image>` 数量与 `images` 数量一致。
- `pixel_trajectory` shape、范围、点数正确。
- `trajectory_fpv_t / trajectory_fpv_tv / trajectory_ego_t` shape 正确，`dt` 单调。
- assistant 输出包含 `<think>`、`<Scene>`、`<FPVScene>`、`<BEVContext>`、`<CrossView>`、`<Plan>`、`<finish>`。
- CoT section JSON 可解析。
- `<BEVContext>` 和 `<finish>` 中的 intent、route theta、trajectory 与几何真值一致。

### 7. 审计 Phase1 几何 source 产物

```bash
python tools/model_processing/eval_sidewalk_phase1_artifacts.py \
  --source-jsonl "${CLIP}/phase1/manifests/phase1_source_clip_test.jsonl" \
  --qa-root "${CLIP}/phase1/clips" \
  --summary-json "${CLIP}/phase1/manifests/phase1_eval_clip_test.json" \
  --output-jsonl "${CLIP}/phase1/manifests/phase1_eval_rows_clip_test.jsonl" \
  --expected-points 16
```

用途：检查几何 source JSONL、trajectory shape、归一化范围、QA schema；后续可接预测 JSON 计算 ADE/FDE。CoT 是否进入训练集由 archive 阶段最终决定。

### 8. 跑 Agent CoT baseline 并可视化预测轨迹

推荐使用 exporter BEV，与训练输入完全一致：

```bash
LLM_PROVIDER=ark \
ARK_API_KEY="<your-key>" \
LLM_MODEL="doubao-seed-2-0-pro-260215" \
python tools/model_processing/run_sidewalk_phase1_cot_baseline.py \
  --input-mode phase1_source \
  --source-jsonl "${CLIP}/phase1/manifests/phase1_source_clip_test.jsonl" \
  --output-dir "${CLIP}/agent_cot_baseline_clip_test" \
  --frame-start 130 \
  --frame-end 150 \
  --frame-step 10
```

输出：

```text
agent_cot_baseline_clip_test/json/*.json
agent_cot_baseline_clip_test/overlay/*.svg
agent_cot_baseline_clip_test/manifest.json
```

说明：baseline 输出也必须是 CoT v4；脚本从 `<finish>.trajectory` 解析模型预测，转换为 FPV pixel `path_uv` 只用于 overlay。不能用规则 fallback 补一条路径。每帧 JSON 会保留：

- `system_prompt` 和 `prompt`：审计模型实际收到的系统约束和 user prompt。
- `response`：模型原始 CoT。
- `prediction.trajectory_norm`：模型 `<finish>.trajectory` 的归一化预测。
- `prediction.rendered_prompt.path_uv`：仅用于 overlay 的 source-pixel 轨迹。
- `trajectory_warnings`：例如轨迹延伸到过高图像区域、终点明显高于几何 GT 等，不做自动修正。

### 9. 使用 ms-swift 训练

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/ms-swift

swift sft \
  --model Qwen/Qwen3.5-4B \
  --tuner_type lora \
  --dataset "${CLIP}/phase1_archive/manifests/clip_test.jsonl" \
  --remove_unused_columns false \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --max_length 2048 \
  --output_dir output/sidewalk_phase1_cot_smoke
```

`phase1_archive/manifests/*.jsonl` 已经是标准 ms-swift 多模态格式，不再需要额外运行 `swift.sidewalk_nav.dataset_builder`。如果训练 trajectory head，仍需保留 `--remove_unused_columns false`，让 `pixel_trajectory` 等额外字段进入 trainer。

### 10. Phase2 SAM/SoM 产物

Phase1 不依赖 SAM/SoM。需要 Phase2 dense grounding 时再运行：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer/tools/model_processing
./run_phase_chunk0.sh
```

或按 chunk 全量运行：

```bash
./run_phase_all.sh
```

## 推荐入口

### 数据集处理

| 文件 | 说明 |
|------|------|
| `dataset_processing/process_chunks_26_29.py` | 批量处理 chunk，支持 `--chunks 0 26 27 ...`、并行、BEV、Phase2 |
| `dataset_processing/mcap_to_sidewalk_inputs.py` | 单个 MCAP 抽帧并生成中间轨迹 |
| `dataset_processing/process_sidewalk_dataset.py` | 从 sidewalk 格式生成 Phase1 最简 2fps ego-aligned BEV、`bev_index.json` 和轨迹索引；debug 时可显式写 FPV JPG、north-up BEV 和 BEV mp4 |
| `dataset_processing/process_realword_mcap.py` | realword MCAP 转 sidewalk/chunk 格式 |
| `dataset_processing/scan_chunk.py` | 扫描 chunk 产物状态 |
| `dataset_processing/srnav_processor.py` | 给其他项目调用的 Python API 封装 |

常用命令：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

python tools/dataset_processing/process_chunks_26_29.py \
  --dataset-root "/media/ubuntu/data1/SR-Nav Dataset" \
  --sidewalk-root "/media/ubuntu/data1/SR-Nav Dataset/sidewalk" \
  --chunks 0 \
  --sample-fps 20.0 \
  --generate-bev \
  --use-pose-ekf \
  --num-workers 4 \
  --skip-existing
```

注意：`--skip-existing` 会跳过已有 `frame_gps.npy`、`front.mp4` 且已有 `bev_video.mp4` 的 clip。Phase1 最简数据不再默认依赖 `bev_video.mp4`；若要重生 BEV，先删除对应 clip 下的 `bev_images/`、`bev_index.json`，或去掉 `--skip-existing`。

### 模型处理

| 文件 | 说明 |
|------|------|
| `model_processing/run_sam3_som_phase.py` | 主入口，生成 fpv segmentation、SoM、route support 等 Phase2 artifacts |
| `model_processing/run_phase_chunk0.sh` | 一键处理 chunk_0 |
| `model_processing/run_phase_all.sh` | 一键处理 sidewalk 下所有数据 |
| `model_processing/run_sidewalk_images_visual_prompt_pipeline.py` | sidewalk Agent expert trajectory 推理与可视化流水线；默认输出 `rendered_prompt.path_uv`，不依赖 SAM/SoM corridor |
| `model_processing/run_sidewalk_phase1_cot_baseline.py` | 使用 Phase1 CoT v4 输入/输出格式运行 Agent baseline；支持 exporter BEV 和在线 BEV 两种输入 |
| `model_processing/sidewalk_phase1_dataset.py` | 生成 Phase1 ms-swift source JSONL、trajectory label、BEV topology meta 和 QA bundle；默认使用 InternNav pose/camera 标定投影 |
| `model_processing/sidewalk_phase1_cot_teacher.py` | 调用 Agent VLM/LLM，把 Phase1 source JSONL 升级为带 `<think>/<Scene>/<FPVScene>/<BEVContext>/<CrossView>/<Plan>/<finish>` 的 CoT SFT 数据 |
| `model_processing/sidewalk_phase1_archive.py` | 先校验 CoT/几何一致性，再把通过样本归档成 ms-swift 可直接训练的 `messages/images` JSONL |
| `model_processing/eval_sidewalk_phase1_artifacts.py` | 审计 Phase1 数据产物，并可选评估预测 ADE/FDE |
| `model_processing/build_sam_semantic_dataset.py` | 汇总/构建 SAM semantic 数据集 |
| `model_processing/README_PHASE.md` | Phase2 详细说明 |

```bash
cd Agent_part/RouteServer/tools/model_processing
./run_phase_chunk0.sh
```

Phase1 不依赖 SAM/SoM，可先构建 FPV+BEV+trajectory 数据闭环：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

SIDEWALK_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0" \
OUTPUT_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1" \
LIMIT=200 \
FRAME_STEP=10 \
WRITE_OVERLAYS=1 \
./tools/model_processing/run_sidewalk_phase1_dataset.sh
```

`FRAME_STEP=10` 是默认值，在 20Hz FPV clip 上约等于 2fps 训练采样；低频 BEV 由 `bev_index.json` 按 timestamp 最近邻匹配，默认允许的最大时间差为 `MAX_BEV_TIME_DELTA_MS=300.0`。若需要每帧样本，可显式设置 `FRAME_STEP=1`。

只生成排序后的前三个 clip，可使用 `MAX_CLIPS=3`，不要用 `LIMIT` 代替；`LIMIT` 表示最多写入多少条样本。

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

SIDEWALK_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0" \
OUTPUT_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1_3clips" \
SPLIT_NAME=train_3clips \
MAX_CLIPS=3 \
WRITE_OVERLAYS=1 \
./tools/model_processing/run_sidewalk_phase1_dataset.sh
```

生成 BEV 时推荐使用最简 Phase1 设置：只保留 2fps ego-aligned BEV 和 `bev_index.json`，不复制全量 FPV JPG，不写 north-up BEV，不写 mp4。BEV 图默认包含完整 route/start/end，这是 System2 的训练输入条件。图例固定为蓝色路线、绿色圆形起点、红色方形终点、橙色圆形当前 ego。

```bash
python tools/dataset_processing/process_sidewalk_dataset.py \
  --sidewalk-root "/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0/<clip_id>" \
  --bev-sample-fps 2.0 \
  --map-provider osm \
  --use-pose-ekf
```

默认会把 `trajectory.json`、`samples.json`、`bev_images/`、`bev_index.json` 写回原始 clip 目录。`--output-root` 仅保留兼容旧流程；只有显式加 `--write-to-output-root` 才会写到外部输出目录。

调试视频和额外视角必须显式开启：

```bash
python tools/dataset_processing/process_sidewalk_dataset.py \
  --sidewalk-root "<clip>" \
  --bev-sample-fps 2.0 \
  --write-bev-videos \
  --write-north-up-bev \
  --draw-bev-trajectory \
  --extract-fpv-images
```

正式 Phase1 训练数据默认在 BEV 图上绘制完整路线。路线点和 OSM raster 共享同一套 Web-Mercator -> PIL rotate -> crop -> resize 坐标变换，因此起点、终点和路线应相对 OSM 地图保持锚定。若要做消融，可加 `--bev-route-scope local_future` 只画局部未来路线，或加 `--no-bev-route-overlay` 完全关闭路线。若希望 OSM 不随车头方向旋转，可以加 `--bev-alignment north_up`；但只要 agent 固定在中心，连续播放时背景仍会随位移平移，这是 centered BEV 的坐标系定义。

正式 Phase1 trajectory label 默认来自 `frame_positions.npy + frame_orientations.npy + camera_params.json` 的标定投影。投影后会固定起点为 `(0.5, 0.9)`，并平滑过近 future pose 造成的贴底回折，使轨迹从 bottom-center 单调向前。旧 `trajectory.json`/GNSS smoke 数据必须显式打开：

Phase1 label 现在同时写出时间条件轨迹：

- `pixel_trajectory`: 兼容旧 SFT 的 `[x,y]` FPV 轨迹。
- `trajectory_fpv_t`: 主时间监督 `[x,y,dt_s]`。
- `trajectory_fpv_tv`: 增强监督 `[x,y,dt_s,speed_mps]`。
- `trajectory_ego_t`: ego/local meter 监督 `[x_right_m,y_front_m,dt_s,speed_mps]`。
- `trajectory_policy.dynamic_obstacle_audit`: 记录动态障碍审计状态。

如果没有动态 mask/tracking，未来 odom 只表示“实际执行路径”，不表示当前帧静态 free-space。此时不会自动改写避障轨迹，`dynamic_obstacle_audit.status=not_evaluated_no_dynamic_masks`。缺少 `frame_times.npy` 时可用 `FRAME_RATE_HZ=20.0` 控制 `dt` 回退。

```bash
ALLOW_HEURISTIC_TRAJECTORY=1 ./tools/model_processing/run_sidewalk_phase1_dataset.sh
```

如果要训练 `<Scene>/<FPVScene>/<BEVContext>/<CrossView>/<Plan>` CoT，必须在 source JSONL 后运行 LLM Teacher。该阶段把几何真值作为离线 evidence 交给 teacher，但最终 ms-swift user prompt 不包含 `intent_gt`、`route_theta_gt` 或 trajectory label，避免 GT 泄露。

```bash
LLM_PROVIDER=ark \
ARK_API_KEY="<your-key>" \
LLM_MODEL="doubao-seed-2-0-pro-260215" \
OUTPUT_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1" \
LIMIT=200 \
./tools/model_processing/run_sidewalk_phase1_cot_teacher.sh
```

`FALLBACK_POLICY=drop` 是正式构建推荐值；`template` 只用于 smoke test，因为它不能提供高质量自然语言 CoT。

### Agent expert trajectory 推理与可视化

数据标注链路里的 `trajectory_labels/*.json` 是几何真值；Agent baseline 链路需要让 VLM/LLM 基于同样的 FPV+BEV QA 语境真实推理出一条 `<finish>.trajectory`，再把该归一化轨迹转换成 `rendered_prompt.path_uv` 做可视化和指标评估。这里不使用手写 fallback 路径，也不要求模型输出 `path_corridor`、左右边界、coverage polygon、mask 或 SAM/SoM region id。

推荐模式是直接读取 `phase1_source_*.jsonl`，这样 FPV、BEV、prompt、输出格式都与 ms-swift CoT 标注链路一致：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

CLIP="/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0/<clip_id>"

LLM_PROVIDER=ark \
ARK_API_KEY="<your-key>" \
LLM_MODEL="doubao-seed-2-0-pro-260215" \
python tools/model_processing/run_sidewalk_phase1_cot_baseline.py \
  --input-mode phase1_source \
  --source-jsonl "${CLIP}/phase1/manifests/phase1_source_clip_test.jsonl" \
  --output-dir "${CLIP}/agent_cot_baseline_clip_test" \
  --frame-start 130 \
  --frame-end 150 \
  --frame-step 10
```

兼容模式可以在线生成 BEV，但只允许 BEV 来源不同，VLM 的输入 prompt 和输出仍然是 Phase1 CoT v4：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

CLIP="/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0/<clip_id>"

LLM_PROVIDER=ark \
ARK_API_KEY="<your-key>" \
LLM_MODEL="doubao-seed-2-0-pro-260215" \
MAP_PROVIDER=osm \
python tools/model_processing/run_sidewalk_phase1_cot_baseline.py \
  --input-mode online_bev \
  --images-dir "${CLIP}/images" \
  --trajectory-json "${CLIP}/agent_baseline_frames_130_150_trajectory.json" \
  --output-dir "${CLIP}/agent_cot_baseline_online_bev_130_150" \
  --frame-start 130 \
  --frame-end 150 \
  --frame-step 10
```

输出检查重点：

- `json/{frame}.json`：保留原始 CoT `response`、解析后的 `<finish>`、`prediction.trajectory_norm`、用于 overlay 的 `prediction.rendered_prompt.path_uv`，以及与 GT 的 normalized ADE/FDE。
- `overlay/{frame}.svg`：只显示预测轨迹中心线，不显示 corridor 左右边界。
- `manifest.json`：统计每帧是否成功解析 CoT 与轨迹。
- 如果 `<finish>.trajectory` 为空，应优先检查 prompt/schema 和模型输出解析，不应通过规则 fallback 补一条路径。

### BEV 和数学工具

| 文件 | 说明 |
|------|------|
| `bev/frodobot_bev_enhanced.py` | BEV 渲染核心模块 |
| `bev/generate_bev_from_trajectory.py` | 从轨迹 JSON 独立生成 BEV |
| `math_tools/pose_ekf.py` | 2D pose EKF |
| `math_tools/diag_traj_vs_ekf.py` | 轨迹航向和 EKF 对比诊断 |

当前 BEV 航向优先级：

- 默认不加 `--use-pose-ekf`：使用 GPS 轨迹点计算 compass heading，并跳过距离过近的点，避免 20 FPS 相邻帧 GPS 重复或抖动导致 0°/乱跳。
- 加 `--use-pose-ekf`：只使用 EKF 平滑 `heading_deg`；BEV 的 `lat/lng` 仍来自 GPS-smoothed 轨迹。若四元数 yaw 近似常数或无效，EKF 会改用鲁棒 GPS bearing 作为 heading 观测。
- 不要默认使用 `--use-ekf-position`：当前 EKF 没有平移速度/里程计输入，直接使用其 x/y 会让长 clip 的 BEV 位姿滞后或距离被压缩。
- 产物 `trajectory.json` 的 `heading_deg` 是实际送入 BEV 渲染的朝向，`position_source` 是实际送入 BEV 渲染的位置来源；`bev_images/*.bevmeta.json` 里的 `used_heading_deg`、`position_source` 是每张 BEV 图实际使用的位姿元数据。
- Phase1 最简 BEV 默认只生成 `bev_images/` ego-aligned 图像和 `bev_index.json`，并绘制完整 route/start/end 作为任务输入条件。`bev_index.json` 记录每张 BEV 对应的原始 `frame_idx` 和 timestamp，ms-swift Phase1 exporter 会用它把每个高频 FPV 帧映射到最近的低频 BEV 帧，默认最大时间差 `300ms`。
- `bev_images_north_up/`、`bev_video.mp4`、`bev_video_north_up.mp4`、`bev_video_ego_aligned.mp4` 仅在显式 debug 参数打开时生成。

## Docker

Docker 文件已归档到 `docker/`：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer
tools/docker/build-urbannav-conda-image.sh
```

容器内默认工作目录是 `/app/tools`，脚本路径按新目录调用：

```bash
docker run -it --rm \
  -v "/media/ubuntu/data1/SR-Nav Dataset:/data/dataset:ro" \
  -v "/media/ubuntu/data1/SR-Nav Dataset/sidewalk:/data/output" \
  -v "/tmp/chunk_mcap_temp:/data/temp" \
  urbannav-processor:latest \
  dataset_processing/process_chunks_26_29.py \
  --dataset-root /data/dataset \
  --sidewalk-root /data/output \
  --temp-dir /data/temp \
  --chunks 26 27 28 29 \
  --sample-fps 20.0 \
  --generate-bev \
  --use-pose-ekf
```

部署服务器镜像：

```bash
tools/docker/deploy-to-server.sh
```

更多 Docker 说明见 `docker/README-DOCKER.md`。

## 文件管理规则

- 数据格式转换、chunk 批处理、dataset 扫描放 `dataset_processing/`。
- 模型推理、SAM3、SoM、Visual Prompt、训练样本构建放 `model_processing/`。
- 地图渲染和 BEV 数据集构建放 `bev/`。
- EKF、坐标/角度/轨迹诊断放 `math_tools/`。
- Docker、部署、服务器运行脚本放 `docker/`。
- 不再维护但可能有参考价值的脚本移入 `archive/`；确认无价值的缓存和生成物直接删除，不提交。
