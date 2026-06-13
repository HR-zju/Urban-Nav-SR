# SideWalk Phase1 数据生成与标注流水线

本文档说明 Agent 仓库中 Phase1 System2 数据集流水线的职责、产物格式和与 ms-swift 的交界。Phase1 只训练 FPV+BEV 到轨迹的 System2，不依赖 SAM/SoM segmentation，也不启用 ms-swift 的 seg head。

## 1. 设计边界

Agent 仓库负责离线生成可审计数据：

```text
raw/prepared SideWalk clip
  -> frame_positions.npy + frame_orientations.npy + camera_params.json + FPV/BEV images
  -> full-FPV trajectory.json + 2fps ego-aligned bev_images/bev_index.json
  -> trajectory label + BEV topology meta + QA bundle
  -> phase1_source_*.jsonl
  -> LLM Teacher structured CoT + validator
  -> phase1_cot_*.jsonl
  -> Archive validator + ms-swift-ready train.jsonl
  -> SidewalkSeq2SeqTrainer
```

ms-swift 仓库只消费稳定 JSONL，不直接读取 Agent runtime、地图服务或 SAM 服务。

Phase1 分成四段，不能混用职责：

| 阶段 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| Geometry Exporter | raw clip、FPV、BEV、pose、camera params | `phase1_source_*.jsonl`、trajectory label、BEV patch meta | 只生成可审计几何真值。 |
| LLM Teacher | source JSONL、FPV/BEV 图、geometry evidence | `<think>{Scene/FPVScene/BEVContext/CrossView/Plan}</think><finish>` | CoT SFT 必需；LLM 只生成解释文本，不允许改写几何真值。 |
| CoT Validator | teacher answer、structured target | `phase1_cot_*.jsonl` | 校验 tags、JSON、intent、theta、trajectory，保留 teacher 中间产物。 |
| Archive Validator | `phase1_cot_*.jsonl`、FPV/BEV、sidecar | `archive/manifests/train.jsonl` | 再次判定样本合理性，通过样本才写入 ms-swift 直接训练 manifest。 |

## 2. 入口脚本

| 脚本 | 职责 |
| --- | --- |
| `tools/dataset_processing/process_sidewalk_dataset.py` | 从 `front.mp4`、pose npy 生成全量 FPV `trajectory.json`，同时生成最简 2fps ego-aligned `bev_images/`、`bev_index.json`、`bev_sample_trajectory.json`；FPV JPG、north-up BEV、BEV mp4 均为显式 debug 选项。 |
| `tools/model_processing/sidewalk_phase1_dataset.py` | 从 InternNav clip 或显式 fallback clip 生成 Phase1 source JSONL、trajectory label、BEV patch meta、QA bundle。 |
| `tools/model_processing/sidewalk_phase1_cot_teacher.py` | 调用 Agent VLM/LLM，为 Phase1 source JSONL 生成强制校验的 structured CoT 监督。 |
| `tools/model_processing/sidewalk_phase1_archive.py` | 校验 CoT/几何一致性，归档 FPV/BEV/sidecar，并生成 ms-swift 直接训练 JSONL。 |
| `tools/model_processing/eval_sidewalk_phase1_artifacts.py` | 检查 source JSONL、sidecar 文件、trajectory shape、越界率、QA bundle schema；可选评估预测 ADE/FDE。 |
| `tools/model_processing/run_sidewalk_phase1_dataset.sh` | 一键运行 Phase1 数据构建和数据评估。 |
| `tools/model_processing/run_sidewalk_phase1_cot_teacher.sh` | 一键运行 Phase1 CoT teacher 和 validator。 |
| `tools/model_processing/run_sidewalk_phase1_archive.sh` | 一键运行 CoT 合理性判定和训练数据归档。 |
| `tools/dataset_processing/repair_sidewalk_odom_from_mcap.py` | 只从原始 MCAP `/odom` 修复已处理 clip 的 odom pose/velocity，不重抽 FPV/BEV 媒体。 |

## 3. 输入要求

正式训练默认读取 InternNav clip 标定格式。每个 clip 至少需要：

```text
clip/
  frame_positions.npy
  frame_orientations.npy
  images/000100.jpg 或 front_images/000100.jpg 或 front.mp4
  bev_images/000100_bev.jpg
  camera_params.json 或 ../camera_params.json
```

关键文件：

| 文件 | 作用 |
| --- | --- |
| `frame_positions.npy` | `[N, 2 or 3]` odom/map 位置，作为未来轨迹来源。 |
| `frame_orientations.npy` | `[N, 4]` 四元数 `[w,x,y,z]`，用于当前机器人坐标系变换。 |
| `camera_params.json` | `K`、`image_width`、`image_height`、`camera_height_meters`，用于 FPV 投影。 |
| `frame_times.npy` | 可选 `[N]` 时间戳，优先用于生成 trajectory waypoint 的 `dt`。常见单位为 ns。 |
| `frame_speeds.npy` | 可选 `[N]` 速度，优先用于生成 `trajectory_fpv_tv` 的速度监督；缺失时由 local path 差分估计。 |
| `images/` / `front_images/` / `front.mp4` | FPV 输入。若只有 `front.mp4`，构建器会把访问到的帧抽到输出目录。 |
| `bev_images/` | BEV 输入。 |

如果原始 MCAP 中存在 `/odom`，推荐先用 `repair_sidewalk_odom_from_mcap.py` 原地修复
prepared clip 的 odom 产物：

```bash
python tools/dataset_processing/repair_sidewalk_odom_from_mcap.py \
  --dataset-root "/media/ubuntu/Extreme SSD/data" \
  --sidewalk-root "/media/ubuntu/data1/SR-Nav Dataset/sidewalk" \
  --chunks 26 27 28 29 \
  --num-workers 4 \
  --only-static-positions
```

该脚本优先使用全量 `frame_times.npy` / `frame_gps.npy` 重建 FPV timeline，再同步 `/odom`；旧版 2fps `trajectory.json` 不再决定输出数组长度。它只更新 `trajectory.json`、`frames_manifest.json`、`mcap_extract_meta.json` 和
`frame_positions/orientations/velocities/speeds.npy`，不会重抽 `images/`、`front.mp4` 或
`bev_images/`。若个别 FPV 帧缺 odom，同步结果会用最近有效 odom 填充并在 report 记录 `filled_missing_odom_frames`，避免写入 `[0,0]` 跳变。`frame_orientations.npy` 的磁盘约定为 ROS `xyzw`，并通过
`mcap_extract_meta.json.frame_orientations_order = "xyzw"` 告诉 Phase1 exporter 在内部转换成
`wxyz`。

例外情况：部分 prepared chunk 的 `frame_positions.npy` 可能全零或近似静止，但
`trajectory.json` / `frame_gps.npy` 中仍有真实 GNSS 轨迹。Phase1 exporter 会检测这类静止
pose；如果同时显式传入 `--allow-heuristic-trajectory`，则回退到 `trajectory.json` /
GNSS heading 生成标签，并在 metadata 中写入：

```json
{
  "trajectory_source": "heuristic_gnss_projection",
  "pose_source": "trajectory_json_gnss",
  "pose_fallback_reason": "static_frame_positions"
}
```

该 fallback 仅用于修复无效 pose 的 prepared 数据，不等同于标定投影。生成后必须检查
`local_path_m`、`intent_gt`、`route_theta_deg` 和 `trajectory_norm`，确认不再出现全零路径或
全量 `stop_hold`。

标定投影链路：

```text
frame_positions + frame_orientations
  -> robot frame future path, x=forward, y=left
  -> camera frame, z=forward, x=right, y=down
  -> FPV pixel trajectory
```

公式与 InternNav `SidewalkTrajectoryGenerator` 对齐：

```text
x_robot = dx * cos(yaw) + dy * sin(yaw)
y_robot = -dx * sin(yaw) + dy * cos(yaw)

z_cam = x_robot
x_cam = -y_robot
y_cam = -camera_height

u = fx * x_cam / z_cam + cx
v = cy - fy * y_cam / z_cam
```

`route_theta_gt` 和 `intent_gt` 来自同一条 robot-frame future path，不再由另一个独立几何流程生成：

```text
x_right = -y_robot
y_front = x_robot
route_theta = atan2(x_right, y_front)
```

### 3.1 时间条件轨迹真值

动态障碍物会让“未来 odom 轨迹”和“当前帧可通行区域”发生分叉：机器人未来可能在行人离开后通过该位置，但当前 FPV 上该位置仍被行人占据。因此 Phase1 exporter 不再只把 future pose 投影解释成静态 free-space GT，而是同时写入时间条件标签：

```json
{
  "pixel_trajectory": [[0.50, 0.90], [0.49, 0.86]],
  "trajectory_fpv_t": [[0.50, 0.90, 0.00], [0.49, 0.86, 0.10]],
  "trajectory_fpv_tv": [[0.50, 0.90, 0.00, 0.40], [0.49, 0.86, 0.10, 0.50]],
  "trajectory_ego_t": [[0.0, 0.0, 0.00, 0.40], [-0.1, 0.8, 0.10, 0.50]],
  "trajectory_policy": {
    "primary_temporal_target": "trajectory_fpv_t",
    "static_compatibility_target": "pixel_trajectory",
    "trajectory_role": "executed_future_path_projected_to_current_fpv",
    "dynamic_obstacle_audit": {
      "status": "not_evaluated_no_dynamic_masks",
      "has_dynamic_masks": false,
      "current_frame_conflict": "unknown"
    }
  }
}
```

字段含义：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `pixel_trajectory` | `[N,2]` | 兼容旧 Phase0/Phase1 SFT 的归一化 FPV 轨迹。 |
| `trajectory_fpv_t` | `[N,3]` | 主时间监督 `[x_norm, y_norm, dt_s]`。 |
| `trajectory_fpv_tv` | `[N,4]` | 增强监督 `[x_norm, y_norm, dt_s, speed_mps]`。 |
| `trajectory_ego_t` | `[N,4]` | ego/local meter 轨迹 `[x_right_m, y_front_m, dt_s, speed_mps]`。 |
| `trajectory_policy` | object | 说明该轨迹是 future executed path，动态障碍审计是否可用，以及推荐的速度策略。 |

生成规则：

- 优先用 `frame_times.npy` 计算 `dt_s`；缺失时使用 `frame_idx / --frame-rate-hz`，默认 `20Hz`。
- 优先用 `frame_speeds.npy` 填 `speed_mps`；缺失时用 `trajectory_ego_t` 相邻点距离除以 `dt` 估计。
- 第 0 个点仍是当前机器人 FPV anchor `(0.5, 0.9, 0.0)`。
- 没有动态 mask/tracking 时，`dynamic_obstacle_audit.status` 固定为 `not_evaluated_no_dynamic_masks`，不得自动把 future pose 改写成避障轨迹。

### 3.2 Legacy fallback

旧版 prepared clip 只有 `trajectory.json` 时，可以用于 smoke test，但必须显式允许：

```bash
ALLOW_HEURISTIC_TRAJECTORY=1 ./tools/model_processing/run_sidewalk_phase1_dataset.sh
```

`trajectory.json` 每行需要：

```json
{
  "timestamp": 1752796902908683776,
  "lat": 30.123,
  "lng": 120.456,
  "heading_deg": 12.3,
  "image_path": "/abs/clip/images/000100.jpg",
  "roll_deg": 0.0,
  "pitch_deg": 0.0
}
```

如果缺 `bev_images/`，先运行 `process_sidewalk_dataset.py` 或已有 BEV 批处理脚本补齐。Phase1 最简模式只要求 2fps ego-aligned `bev_images/` 和 `bev_index.json`；FPV 可以继续引用 `front.mp4`，由 Phase1 exporter 对被采样的训练帧按需抽图。调试时可给 `sidewalk_phase1_dataset.py --allow-missing-bev`，但正式训练不建议这么做。

## 4. 输出结构

```text
sidewalk_phase1/
  clips/
    {clip_id}/
      trajectory_labels/{frame}.json
      bev_patch_meta/{frame}.json
      qa/{frame}_qa.json
      overlays/{frame}_debug.jpg
  manifests/
    phase1_source_train.jsonl
    phase1_cot_train.jsonl
    phase1_qa_index_train.jsonl
    phase1_dataset_report_train.json
    phase1_cot_report_train.json
    phase1_eval_train.json

sidewalk_phase1_archive/
  media/
    clips/{clip_id}/
      fpv/{frame}.jpg
      bev/{frame}_bev.jpg
  sidecars/
    clips/{clip_id}/
      trajectory_labels/{frame}.json
      bev_patch_meta/{frame}.json
      qa/{frame}_qa.json
      cot_sections/{frame}.json
  manifests/
    train.jsonl
    train.errors.jsonl
    train.review.csv
    train.stats.json
  review/
    index.html
```

### 4.1 source JSONL

`phase1_source_train.jsonl` 是 ms-swift 的 Phase1 主输入：

```json
{
  "id": "clip_a_000100",
  "clip_id": "clip_a",
  "frame_idx": 100,
  "fpv_image_path": "/abs/clip/images/000100.jpg",
  "bev_image_path": "/abs/clip/bev_images/000100_bev.jpg",
  "pixel_trajectory": [[0.5, 0.9], [0.51, 0.72]],
  "trajectory_fpv_t": [[0.5, 0.9, 0.0], [0.51, 0.72, 0.1]],
  "trajectory_fpv_tv": [[0.5, 0.9, 0.0, 0.4], [0.51, 0.72, 0.1, 0.5]],
  "trajectory_ego_t": [[0.0, 0.0, 0.0, 0.4], [0.1, 1.2, 0.1, 0.5]],
  "trajectory_policy": {
    "primary_temporal_target": "trajectory_fpv_t",
    "dynamic_obstacle_audit": {
      "status": "not_evaluated_no_dynamic_masks"
    }
  },
  "image_width": 1920,
  "image_height": 1080,
  "instruction": "Navigate safely on the sidewalk and follow the visible route.",
  "trajectory_label_path": "/abs/trajectory_labels/000100.json",
  "bev_patch_meta_path": "/abs/bev_patch_meta/000100.json",
  "qa_bundle_path": "/abs/qa/000100_qa.json",
  "sidewalk_meta": {
    "trajectory_source": "calibrated_pose_projection"
  }
}
```

这里的 `pixel_trajectory` 是归一化 FPV 坐标，范围 `[0, 1]`，直接被 ms-swift Phase1 trajectory head 作为回归目标。

`phase1_source_train.jsonl` 只适合：

- 文本轨迹 baseline。
- trajectory head 回归监督。
- 作为 LLM Teacher 的输入。

它没有高质量自然语言 CoT，不能直接当作 `<Scene>/<FPVScene>/<CrossView>/<Plan>` CoT SFT 的最终数据。

当 FPV 原始帧率高于 BEV 保留帧率时，`bev_image_path` 不要求和 `frame_idx` 同名。Phase1 exporter 会优先读取 `clip/bev_index.json`，按 timestamp 选择最近的 BEV 帧，并在 `sidewalk_meta` 里写入：

```json
{
  "bev_frame_idx": 120,
  "bev_timestamp": 1752796902908683776,
  "bev_time_delta_ms": 23.5,
  "bev_mapping_source": "nearest_timestamp"
}
```

这样可以只保存 2fps BEV，同时让 `trajectory.json` 保持 20fps FPV/pose 全量帧；Phase1 exporter 可按 `FRAME_STEP` 自行采样 FPV 样本，并为每个样本匹配 timestamp 最近、且默认时间差不超过 `300ms` 的 BEV。若输入来自旧版稀疏 `trajectory.json`，需要先重新运行 `process_sidewalk_dataset.py`，或临时把 `FRAME_STEP` 设为 `1` 避免二次降采样。

FPV trajectory label 会在标定投影后执行一次近场修正：第一个点固定为 bottom-center anchor `(0.5, 0.9)`，后续点必须沿图像 y 方向单调向前移动。过近 future pose 投影到图像底部时，会被插值平滑到下一个有效前向点，避免训练目标出现“先往下再往前”的回折。

`trajectory_source` 取值：

| 值 | 含义 | 是否正式训练推荐 |
| --- | --- | --- |
| `calibrated_pose_projection` | InternNav pose + camera 标定投影。 | 是 |
| `heuristic_pose_projection` | 有 pose 但缺 camera params，用 HFOV 启发式生成。 | 否 |
| `heuristic_gnss_projection` | `trajectory.json` GNSS/heading fallback；用于旧格式或静止 `frame_positions.npy` 修复。 | 条件使用 |

### 4.2 BEV patch meta

`bev_patch_meta/{frame}.json` 是 Stage1 拓扑 QA 和审计目标。这里不再输出
`ego_cell_gt`、`target_cell_gt` 或 `grid_size`：这些字段来自 exporter 的局部窗口配置，
不是外部标注真值，不适合作为 CoT/SFT 监督目标。Phase1 保留连续局部米制路线和语义导航状态。

```json
{
  "schema_version": "sidewalk_bev_patch_meta_v1",
  "coordinate_frame": "ego_aligned_local_meters",
  "route_polyline_local_m": [[0.0, 0.0], [0.2, 2.0], [0.5, 4.0]],
  "route_target_local_m": [0.5, 4.0],
  "route_theta_deg": 0.0,
  "intent_gt": "go_straight",
  "target_direction": "front",
  "confidence_level_gt": "normal"
}
```

`route_polyline_local_m` 和 `route_target_local_m` 只用于离线 teacher/validator 约束
`route_theta_deg`、`intent_gt`、`target_direction` 和最终 FPV trajectory，不会作为学生 prompt 中的输入泄露。

### 4.3 QA bundle

`qa/{frame}_qa.json` 使用 ms-swift SideWalk QA schema。Phase1 QA 已和主线
CoT v5 对齐：teacher answer 输出 `<BEVContext>/<FPVScene>/<CrossView>/<Plan>/<finish>`
这些 CoT section tag，而不是旧的 `<BEVIntent>/<PointPrompt>/<TrajectorySupport>` 语法。

```json
{
  "schema_version": "sidewalk_bev_fpv_qa_v1",
  "clip_name": "clip_a",
  "fpv_frame_idx": 100,
  "fpv_image_path": "/abs/images/000100.jpg",
  "bev_image_path": "/abs/bev_images/000100_bev.jpg",
  "gt_pixel_trajectory": [[0.5, 0.9], [0.51, 0.72]],
  "samples": [
    {
      "task_family": "A_bev",
      "task_type": "bev_route_theta",
      "teacher_answer": "<BEVContext>{...}</BEVContext>",
      "structured_target": {
        "route_theta_deg": 0.0,
        "route_theta_bin": "[-5,5)",
        "route_direction_bin": "front",
        "intent": "go_straight",
        "target_direction": "front"
      }
    }
  ]
}
```

Phase1 默认生成：

- `bev_route_theta`
- `bev_navigation_stage`
- `crossview_topology_grounding`
- `point_prompt`
- `trajectory_intent_check`

其中 `bev_route_theta`、`bev_navigation_stage`、`crossview_topology_grounding`、`point_prompt` 与在线 RouteServer QA 核心 task 对齐；`trajectory_intent_check` 是 Phase1 离线几何轨迹一致性审计项。没有 segmentation 时，不生成强 `fpv_region_affordance` 和 `mask_route_sequence` 监督。

## 5. 使用方式

### 5.1 一键构建小样本

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

python tools/dataset_processing/process_sidewalk_dataset.py \
  --sidewalk-root "/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0/<clip_id>" \
  --bev-sample-fps 2.0 \
  --map-provider osm \
  --use-pose-ekf

SIDEWALK_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0" \
OUTPUT_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1" \
LIMIT=200 \
FRAME_STEP=1 \
WRITE_OVERLAYS=1 \
./tools/model_processing/run_sidewalk_phase1_dataset.sh
```

`process_sidewalk_dataset.py` 的 Phase1 最简默认行为：

- 只写 `bev_images/` ego-aligned BEV。
- `--bev-sample-fps` 默认 2fps。
- 写 `trajectory.json`，包含每个原始 FPV frame 的 pose/GNSS/heading；它不再按 BEV 频率降采样。
- 写 `bev_index.json` 和 `bev_sample_trajectory.json`，用于 FPV 高帧率到 BEV 低帧率的 timestamp 最近邻匹配。
- 默认把 `trajectory.json`、`bev_sample_trajectory.json`、`samples.json`、`bev_images/`、`bev_index.json` 全部写入原始 clip 目录，便于后续直接扫 `/media/ubuntu/data1/SR-Nav Dataset/sidewalk` 制作数据集。
- `--use-pose-ekf` 只平滑 BEV heading，不再覆盖 BEV center/route 的 GPS 位置；如需诊断旧行为才显式加 `--use-ekf-position`。
- 默认在 BEV 上绘制完整 route/start/end。route 是训练输入条件，不是 debug overlay；图例固定为蓝色路线、绿色圆形起点、红色方形终点、橙色圆形当前 ego。模型需要学会从这些标记判断自己处于路线的哪个阶段。其 GPS 点与 OSM raster 使用同一套 Web-Mercator -> PIL rotate -> crop -> resize 变换，保证起点、终点和路线相对地图不漂移。
- 如需消融，可用 `--bev-route-scope local_future` 只画当前帧之后的局部路线，或用 `--no-bev-route-overlay` 关闭路线。
- 不写 `bev_video*.mp4`，除非显式加 `--write-bev-videos`。
- 不写 `bev_images_north_up/`，除非显式加 `--write-north-up-bev`。
- 不重复抽取 FPV JPG，除非显式加 `--extract-fpv-images`。

如果只是不希望 OSM 背景随车头方向旋转，可以生成 centered north-up BEV：

```bash
python tools/dataset_processing/process_sidewalk_dataset.py \
  --sidewalk-root "/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0/<clip_id>" \
  --bev-sample-fps 2.0 \
  --bev-alignment north_up \
  --map-provider osm \
  --use-pose-ekf
```

注意：只要 agent 固定在中心，连续播放时 OSM 背景就会随 agent 位移发生平移；这是 ego-centered/local-centered 坐标系的必然结果。固定 OSM 背景、agent 在地图上移动属于 global debug map，不应作为 Phase1 默认训练输入。若 FPV 已经过路口但 BEV 仍明显滞后，优先检查 `trajectory.json` 是否为全量 FPV 帧、`bev_index.json` 是否按 2fps timestamp 对齐，以及 raw GPS/`frame_positions.npy` 位移；不要用无平移里程计的 EKF x/y 作为 BEV 位置。

### 5.2 只运行构建器

```bash
python tools/model_processing/sidewalk_phase1_dataset.py \
  --sidewalk-root "/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0" \
  --output-root "/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1" \
  --split-name train \
  --frame-step 5 \
  --future-frames 50 \
  --num-points 16 \
  --frame-rate-hz 20.0
```

### 5.3 数据审计

```bash
python tools/model_processing/eval_sidewalk_phase1_artifacts.py \
  --source-jsonl "/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1/manifests/phase1_source_train.jsonl" \
  --qa-root "/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1/clips" \
  --summary-json "/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1/manifests/phase1_eval_train.json"
```

### 5.4 LLM Teacher 生成 CoT SFT 数据

训练真正的 `<Scene>/<FPVScene>/<BEVContext>/<CrossView>/<Plan>` CoT 时，必须先运行 teacher 阶段。这个阶段使用 FPV 作为主图、BEV 作为额外图，并把 `bev_patch_meta`、`trajectory_label`、`route_context` 作为离线 evidence 提供给 teacher。注意：这些 GT evidence 只进入 teacher prompt，不进入最终 ms-swift user prompt。

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

LLM_PROVIDER=ark \
ARK_API_KEY="<your-key>" \
LLM_MODEL="doubao-seed-2-0-pro-260215" \
LLM_API_BASE_URL="https://ark.cn-beijing.volces.com/api/v3" \
OUTPUT_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1" \
LIMIT=200 \
FALLBACK_POLICY=drop \
./tools/model_processing/run_sidewalk_phase1_cot_teacher.sh
```

Teacher 输出格式固定为单轮 assistant target。`<think>` 内只放结构化观察和规划 section，不再额外生成叙述性 reasoning，避免和 JSON section 重复：

```text
<think>
<Scene>{...}</Scene>
<FPVScene>{...}</FPVScene>
<BEVContext>{...}</BEVContext>
<CrossView>{...}</CrossView>
<Plan>{...}</Plan>
</think>
<finish>{"trajectory":[[x,y],...],"intent":"...","confidence_level":"...","trajectory_type":"...","trajectory_validity":"valid|invalid"}</finish>
```

Validator 会检查：

- 所有 tag 是否存在，且没有 tag 外说明文字。
- `<think>` 内是否只包含 `Scene/FPVScene/BEVContext/CrossView/Plan` 五个结构化 JSON section，不允许额外叙述性文本。
- `Scene/FPVScene/BEVContext/CrossView/Plan/finish` 是否为可解析 JSON。
- `<BEVContext>.intent` 和 `<finish>.intent` 是否等于 `intent_gt`。
- `route_theta_deg` 是否在容差内等于几何真值。
- `<BEVContext>.route_direction_bin`、`intent`、`target_direction` 是否与几何真值派生标签一致。
- `<FPVScene>.risk_objects` 是否为 `{name, side, risk}` 结构化对象。
- `<CrossView>.fpv_route_region`、`route_to_fpv_direction`、`safe_corridor` 是否与轨迹区域派生标签一致。
- `<Plan>.motion_primitive/lateral_preference/avoidance_side/speed_policy` 是否存在。
- `<finish>.trajectory_type` 和 `trajectory_validity` 是否存在，且正式样本必须标记为 `valid`。
- `<finish>.trajectory` 点数、范围、与 `trajectory_label` 的偏差是否合格。
- 没有 SAM/SoM segmentation context 时，禁止声称 `region_id`、mask overlap、mIoU、support ratio 等 Phase2 才有的证据。

`FALLBACK_POLICY` 建议：

| 值 | 行为 | 用途 |
| --- | --- | --- |
| `drop` | teacher 多次失败后丢弃该样本。 | 正式构建推荐。 |
| `error` | teacher 失败立即报错。 | 调试 provider 或 prompt。 |
| `template` | teacher 失败后写入显式模板降级答案。 | 小样本 smoke test；不能代表高质量 CoT。 |

产物：

```text
manifests/phase1_cot_train.jsonl
manifests/phase1_cot_report_train.json
```

### 5.5 合理性判定与训练数据归档

CoT teacher 产物不是最终训练集。正式训练前必须运行 archive 阶段：它会再次校验 CoT/几何一致性、媒体路径和 trajectory 结构，只把通过校验的样本写入 ms-swift 可直接训练 JSONL。

```bash
OUTPUT_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1" \
ARCHIVE_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1_archive" \
SPLIT_NAME=train \
MEDIA_MODE=hardlink \
PATH_MODE=absolute \
WRITE_REVIEW_HTML=1 \
./tools/model_processing/run_sidewalk_phase1_archive.sh
```

归档阶段主要检查：

- FPV/BEV 图片路径存在。
- user prompt 的 `<image>` 数量与 `images` 数量一致。
- `pixel_trajectory` 点数、shape、范围正确。
- `trajectory_fpv_t / trajectory_fpv_tv / trajectory_ego_t` shape 正确，`dt` 单调。
- assistant 输出包含 `<think>/<Scene>/<FPVScene>/<BEVContext>/<CrossView>/<Plan>/<finish>`。
- 每个 CoT section 是合法 JSON。
- `<BEVContext>` 和 `<finish>` 的 intent、route theta、trajectory 与几何真值一致。

产物：

```text
sidewalk_phase1_archive/manifests/train.jsonl          # ms-swift 直接训练输入
sidewalk_phase1_archive/manifests/train.errors.jsonl   # 被拒绝样本和原因
sidewalk_phase1_archive/manifests/train.review.csv     # 人工审计索引
sidewalk_phase1_archive/manifests/train.stats.json     # 接受率和分布统计
```

本地训练推荐 `MEDIA_MODE=hardlink PATH_MODE=absolute`；发布/上传推荐 `MEDIA_MODE=copy PATH_MODE=relative`。

ms-swift 训练时直接使用：

```bash
swift sft \
  --dataset "/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1_archive/manifests/train.jsonl" \
  --remove_unused_columns false \
  ...
```

Archive manifest 已经是标准 ms-swift 多模态 JSONL，不需要再运行 `swift.sidewalk_nav.dataset_builder`。

若只做 QA-only manifest，仍可在 ms-swift 仓库中执行：

QA-only manifest：

```bash
python -m swift.sidewalk_nav.dataset_builder \
  --input-format qa \
  --qa-root "/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1/clips" \
  --output-jsonl "/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1/manifests/ms_swift_phase1_qa_train.jsonl"
```

### 5.6 Agent baseline CoT 推理

Agent baseline 不应再使用旧的 visual-prompt JSON schema。为了与数据标注、ms-swift SFT 和后续评估统一，baseline 的输入输出必须采用同一套 Phase1 CoT v5 格式：

```text
FPV + BEV
  -> <think>
       <Scene>{...}</Scene>
       <FPVScene>{...}</FPVScene>
       <BEVContext>{...}</BEVContext>
       <CrossView>{...}</CrossView>
       <Plan>{...}</Plan>
     </think>
     <finish>{"trajectory":[[x,y],...],"intent":"...","confidence_level":"...","trajectory_type":"...","trajectory_validity":"valid|invalid"}</finish>
```

推荐使用 exporter BEV：

```bash
python tools/model_processing/run_sidewalk_phase1_cot_baseline.py \
  --input-mode phase1_source \
  --source-jsonl "/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1/manifests/phase1_source_train.jsonl" \
  --output-dir "/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1/agent_cot_baseline_train" \
  --limit 200
```

兼容模式可使用在线 OSM BEV，但仅 BEV 图来源不同，prompt 和输出仍保持 CoT v5：

```bash
python tools/model_processing/run_sidewalk_phase1_cot_baseline.py \
  --input-mode online_bev \
  --images-dir "<clip>/images" \
  --trajectory-json "<clip>/trajectory.json" \
  --output-dir "<clip>/agent_cot_baseline_online_bev" \
  --frame-step 10
```

输出中的 `prediction.trajectory_norm` 直接来自模型 `<finish>.trajectory`，`prediction.rendered_prompt.path_uv` 只用于 FPV overlay。若 `<finish>.trajectory` 缺失，样本应记为解析失败，不允许用规则 fallback 补轨迹。

## 6. 评估指标

数据构建阶段：

- `missing_path_rate`
- `target_shape_ok`
- `target_in_bounds`
- `intent_counts`
- `qa_bundle_summary.task_counts`

模型预测阶段：

- `prediction_parse_success`
- `ADE`
- `FDE`

Phase1 不计算 mIoU、harmful recall、AP50；这些属于 Phase2 segmentation/dense grounding。

## 7. 后续扩展

1. 将 `bev_patch_meta` 从 local metric grid 扩展到真实 BEV raster pixel grid。
2. 接入 route provider 或 OSM semantic layers，补 `crossing/entrance/sidewalk` 等拓扑实体。
3. SAM/SoM 服务器产物稳定后，增加 `fpv_region_affordance`、`mask_route_sequence`、`trajectory_safety_check` 的强监督。
4. 将 prediction eval 输出与 ms-swift `swift.sidewalk_nav.eval` 的 ADE/FDE 对齐，形成统一 dashboard。
