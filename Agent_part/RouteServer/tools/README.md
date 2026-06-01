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
| `dataset_processing/process_sidewalk_dataset.py` | 从 sidewalk 格式生成 BEV 图像和 `bev_video.mp4` |
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

注意：`--skip-existing` 会跳过已有 `frame_gps.npy`、`front.mp4` 且已有 `bev_video.mp4` 的 clip。若要重生 BEV，先删除对应 clip 下的 `bev_video.mp4`/`bev_images/`，或去掉 `--skip-existing`。

### 模型处理

| 文件 | 说明 |
|------|------|
| `model_processing/run_sam3_som_phase.py` | 主入口，生成 fpv segmentation、SoM、route support 等 Phase2 artifacts |
| `model_processing/run_phase_chunk0.sh` | 一键处理 chunk_0 |
| `model_processing/run_phase_all.sh` | 一键处理 sidewalk 下所有数据 |
| `model_processing/run_sidewalk_images_visual_prompt_pipeline.py` | sidewalk 图像视觉提示流水线 |
| `model_processing/build_sam_semantic_dataset.py` | 汇总/构建 SAM semantic 数据集 |
| `model_processing/README_PHASE.md` | Phase2 详细说明 |

```bash
cd Agent_part/RouteServer/tools/model_processing
./run_phase_chunk0.sh
```

### BEV 和数学工具

| 文件 | 说明 |
|------|------|
| `bev/frodobot_bev_enhanced.py` | BEV 渲染核心模块 |
| `bev/generate_bev_from_trajectory.py` | 从轨迹 JSON 独立生成 BEV |
| `math_tools/pose_ekf.py` | 2D pose EKF |
| `math_tools/diag_traj_vs_ekf.py` | 轨迹航向和 EKF 对比诊断 |

当前 BEV 航向优先级：

- 默认不加 `--use-pose-ekf`：使用相邻 GPS 点计算 compass heading 并平滑。
- 加 `--use-pose-ekf`：使用四元数 yaw 作为 EKF 观测；若 yaw 几乎不变，会自动禁用 yaw 更新并回退到 GPS 航向。

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
