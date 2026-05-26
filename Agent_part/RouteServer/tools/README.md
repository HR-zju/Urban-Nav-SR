# RouteServer Tools

## 主要工具

### SAM3 + SoM 标注 (Phase1/Phase2 格式)

| 文件 | 说明 |
|------|------|
| `run_sam3_som_phase.py` | 主脚本 - 生成Phase1/Phase2训练格式 |
| `run_phase_chunk0.sh` | 一键处理 chunk_0 数据集 |
| `run_phase_all.sh` | 一键批量处理 sidewalk 下所有数据集 |
| `README_PHASE.md` | 详细使用文档 |

### BEV 生成

| 文件 | 说明 |
|------|------|
| `frodobot_bev_dataset_generator.py` | BEV 数据集生成 |
| `frodobot_bev_enhanced.py` | 增强版 BEV 生成 |

### MCAP 处理

| 文件 | 说明 |
|------|------|
| `mcap_to_sidewalk_inputs.py` | MCAP 转 sidewalk 输入格式 |
| `process_realword_mcap.py` | 处理 realword MCAP |
| `process_sidewalk_dataset.py` | 处理 sidewalk 数据集 |

### 视觉提示

| 文件 | 说明 |
|------|------|
| `run_sidewalk_images_visual_prompt_pipeline.py` | 完整视觉提示管道 |
| `fodorobot_sam3_bridge.py` | SAM3 桥接 |
| `make_visual_prompt_seg_video.py` | 生成分割视频 |

### 其他工具

| 文件 | 说明 |
|------|------|
| `ros2_bag_to_trajectory.py` | ROS2 bag 转轨迹 |
| `route_context_capture.py` | 路由上下文捕获 |
| `route_context_debugger.py` | 路由上下文调试 |

## 快速开始

### 生成 SAM3 + SoM 标注 (推荐)

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer/tools

# 1. 修改 checkpoint 路径
nano run_phase_all.sh

# 2. 批量处理所有数据集
./run_phase_all.sh

# 或只处理 chunk_0
./run_phase_chunk0.sh
```

详细文档: [README_PHASE.md](./README_PHASE.md)

### 输出格式

```
chunk_0/
├── fpv_segmentation/          # SAM分割结果
│   ├── 000000_summary.json    # 类别和affordance
│   ├── 000000_mask.npy        # 分割掩码
│   └── 000000_overlay.jpg     # 可视化
├── som/                       # SoM标注
│   ├── 000000_som.json        # 区域标注
│   └── 000000_overlay.jpg     # 可视化
└── debug_video/               # 调试视频(可选)
    ├── segmentation_overlay.mp4
    └── som_overlay.mp4
```

## Archive

旧的测试/临时脚本已归档到 `archive/` 目录。

