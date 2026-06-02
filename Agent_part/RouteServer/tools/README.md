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
| `model_processing/run_sidewalk_images_visual_prompt_pipeline.py` | sidewalk 图像视觉提示流水线 |
| `model_processing/sidewalk_phase1_dataset.py` | 生成 Phase1 ms-swift source JSONL、trajectory label、BEV topology meta 和 QA bundle；默认使用 InternNav pose/camera 标定投影 |
| `model_processing/sidewalk_phase1_cot_teacher.py` | 调用 Agent VLM/LLM，把 Phase1 source JSONL 升级为带 `<think>/<Scene>/<FPV>/<BEV>/<CrossView>/<Plan>/<finish>` 的 CoT SFT 数据 |
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
FRAME_STEP=1 \
WRITE_OVERLAYS=1 \
./tools/model_processing/run_sidewalk_phase1_dataset.sh
```

`FRAME_STEP=1` 是默认值，表示对每一个 FPV 帧生成训练样本；低频 BEV 由 `bev_index.json` 按 timestamp 最近邻匹配，默认允许的最大时间差为 `MAX_BEV_TIME_DELTA_MS=300.0`。

生成 BEV 时推荐使用最简 Phase1 设置：只保留 2fps ego-aligned BEV 和 `bev_index.json`，不复制全量 FPV JPG，不写 north-up BEV，不写 mp4。BEV 图默认包含完整 route/start/end，这是 System2 的训练输入条件。

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

```bash
ALLOW_HEURISTIC_TRAJECTORY=1 ./tools/model_processing/run_sidewalk_phase1_dataset.sh
```

如果要训练 `<Scene>/<FPV>/<BEV>/<CrossView>/<Plan>` CoT，必须在 source JSONL 后运行 LLM Teacher。该阶段把几何真值作为离线 evidence 交给 teacher，但最终 ms-swift user prompt 不包含 `intent_gt`、`route_theta_gt` 或 trajectory label，避免 GT 泄露。

```bash
LLM_PROVIDER=ark \
ARK_API_KEY="<your-key>" \
LLM_MODEL="doubao-seed-2-0-pro-260215" \
OUTPUT_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1" \
LIMIT=200 \
./tools/model_processing/run_sidewalk_phase1_cot_teacher.sh
```

`FALLBACK_POLICY=drop` 是正式构建推荐值；`template` 只用于 smoke test，因为它不能提供高质量自然语言 CoT。

### BEV 和数学工具

| 文件 | 说明 |
|------|------|
| `bev/frodobot_bev_enhanced.py` | BEV 渲染核心模块 |
| `bev/generate_bev_from_trajectory.py` | 从轨迹 JSON 独立生成 BEV |
| `math_tools/pose_ekf.py` | 2D pose EKF |
| `math_tools/diag_traj_vs_ekf.py` | 轨迹航向和 EKF 对比诊断 |

当前 BEV 航向优先级：

- 默认不加 `--use-pose-ekf`：使用 GPS 轨迹点计算 compass heading，并跳过距离过近的点，避免 20 FPS 相邻帧 GPS 重复或抖动导致 0°/乱跳。
- 加 `--use-pose-ekf`：优先检查四元数 yaw 是否可靠；若 yaw 近似常数或四元数无效，EKF 会改用鲁棒 GPS bearing 作为 heading 观测，而不是只靠 IMU 角速度积分。
- 产物 `trajectory.json` 的 `heading_deg` 是实际送入 BEV 渲染的朝向；`bev_images/*.bevmeta.json` 里的 `used_heading_deg` 是每张 BEV 图实际使用的角度。
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
