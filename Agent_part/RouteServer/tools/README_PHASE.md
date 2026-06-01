# SAM3 + SoM 标注工具 - Phase1/Phase2 训练格式

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `run_sam3_som_phase.py` | 主脚本 - 生成Phase1/Phase2格式 |
| `run_phase_chunk0.sh` | 一键处理chunk_0 |
| `run_phase_all.sh` | 一键批量处理所有数据集 |

## 输出目录结构

```
sidewalk/
├── chunk_0/
│   ├── images/                    # 原始图像
│   ├── trajectory.json            # 轨迹文件
│   ├── fpv_segmentation/          # SAM分割结果
│   │   ├── 000000_summary.json
│   │   ├── 000000_mask.npy
│   │   ├── 000000_overlay.jpg
│   │   ├── 000001_summary.json
│   │   ├── ...
│   ├── som/                       # SoM标注
│   │   ├── 000000_som.json
│   │   ├── 000000_overlay.jpg
│   │   ├── ...
│   ├── debug_video/               # 调试视频（可选）
│   │   ├── segmentation_overlay.mp4
│   │   └── som_overlay.mp4
│   ├── sam3_intermediate/         # SAM3中间输出
│   ├── sam3_work/                 # SAM3工作目录
│   └── processing_manifest.json   # 处理清单
├── realword/
│   └── (同上结构)
└── batch_processing_summary.json  # 批量处理总结
```

## fpv_segmentation JSON 格式

`000100_summary.json`:

```json
{
  "frame_idx": 100,
  "image_width": 1920,
  "image_height": 1080,
  "navigation_mode": "sidewalk_robot",
  "mask_path": "000100_mask.npy",
  "overlay_path": "000100_overlay.jpg",
  "classes": [
    {
      "class_id": 1,
      "class_name": "sidewalk",
      "affordance": "preferred_navigable",
      "area": 420000,
      "bbox_xyxy": [120, 520, 1720, 1079],
      "bottom_connected": true
    },
    {
      "class_id": 2,
      "class_name": "road",
      "affordance": "forbidden",
      "area": 250000,
      "bbox_xyxy": [0, 620, 600, 1079],
      "bottom_connected": false
    }
  ],
  "preferred_navigable_class_names": ["sidewalk", "crosswalk", "plaza"],
  "conditional_navigable_class_names": ["curb_cut", "driveway"],
  "forbidden_class_names": ["road", "vehicle", "pedestrian", "pole", "wall"]
}
```

## SoM JSON 格式

`000100_som.json`:

```json
{
  "frame_idx": 100,
  "image_width": 1920,
  "image_height": 1080,
  "regions": [
    {
      "region_id": "som_1",
      "source_mask_id": 1,
      "label": "sidewalk",
      "affordance": "preferred_navigable",
      "bbox_xyxy": [120, 520, 1720, 1079],
      "centroid_xy": [930, 810],
      "area": 420000,
      "bottom_connected": true,
      "touches_trajectory": true
    },
    {
      "region_id": "obs_1",
      "source_mask_id": 7,
      "label": "pedestrian",
      "affordance": "forbidden",
      "bbox_xyxy": [1100, 420, 1230, 760],
      "centroid_xy": [1165, 590],
      "area": 18000,
      "bottom_connected": false,
      "touches_trajectory": false
    }
  ],
  "route_trace": ["som_1"],
  "preferred": ["som_1"],
  "conditional": [],
  "forbidden": ["obs_1"]
}
```

## Mask 格式

`000100_mask.npy`:
- shape: [H, W]
- dtype: int32
- value: class_id

## 使用方法

### 方法1：一键处理 chunk_0

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer/tools

# 修改 CHECKPOINT_PATH
nano run_phase_chunk0.sh

# 运行
./run_phase_chunk0.sh
```

### 方法2：一键批量处理所有

```bash
# 修改 CHECKPOINT_PATH
nano run_phase_all.sh

# 运行
./run_phase_all.sh
```

### 方法3：直接运行 Python 脚本

处理单个数据集：

```bash
python run_sam3_som_phase.py \
  --dataset-dir "/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0" \
  --run-sam3 \
  --sam3-env-name "sam3" \
  --checkpoint-path "/path/to/checkpoint_35_merged.pt" \
  --gpus 0 \
  --min-score 0.3 \
  --generate-video
```

批量处理：

```bash
python run_sam3_som_phase.py \
  --sidewalk-dir "/media/ubuntu/data1/SR-Nav Dataset/sidewalk" \
  --run-sam3 \
  --sam3-env-name "sam3" \
  --checkpoint-path "/path/to/checkpoint_35_merged.pt" \
  --gpus 0 \
  --min-score 0.3 \
  --generate-video \
  --skip-existing
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `--dataset-dir` | 单个数据集目录 |
| `--sidewalk-dir` | Sidewalk根目录（批量处理） |
| `--run-sam3` | 运行SAM3推理 |
| `--sam3-env-name` | SAM3 conda环境名（默认sam3） |
| `--checkpoint-path` | SAM3 checkpoint路径 |
| `--gpus` | GPU IDs（默认0） |
| `--min-score` | SAM3最小置信度（默认0.3） |
| `--frame-start/--frame-end` | 处理帧范围 |
| `--frame-step` | 帧采样步长 |
| `--generate-video` | 生成debug_video mp4 |
| `--video-fps` | 视频帧率（默认10） |
| `--skip-existing` | 跳过已有结果的数据集 |

## 类别配置

默认类别affordance配置：

```python
{
    "preferred_navigable": ["sidewalk", "crosswalk", "plaza", "pedestrian_area"],
    "conditional_navigable": ["curb_cut", "driveway", "road_shoulder"],
    "forbidden": ["road", "vehicle", "pedestrian", "pole", "wall", "building", "tree", "grass", "fence"],
}
```

可在代码中修改 `DEFAULT_CLASS_CONFIG` 来自定义。

## Phase1 训练使用

```bash
# Phase1 annotation teacher
--segmentation-root clip_dir/fpv_segmentation
--som-root clip_dir/som
```

## Phase2 训练使用

- `mask.npy` -> affordance / obstacle / route_support loss
- `som.json` -> region grounding / teacher route trace / evaluation
- `overlay.mp4` -> 人工检查用

