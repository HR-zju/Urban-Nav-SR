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
| `process_chunks_26_29.py` | 批量处理 chunk_26-29（支持并行） |
| `process_realword_mcap.py` | 处理 realword MCAP |
| `process_sidewalk_dataset.py` | 处理 sidewalk 数据集 |
| `srnav_processor.py` | **封装模块** - 提供简洁 API 供其他项目调用 |
| `srnav_processor_examples.py` | 封装模块使用示例 |

#### Chunk 26-29 快速处理

```bash
# 基础转换（4 worker 并行）
python process_chunks_26_29.py \
    --chunks 26 27 28 29 \
    --skip-existing \
    --num-workers 4

# 完整处理（含 BEV + Phase2）
python process_chunks_26_29.py \
    --chunks 26 27 28 29 \
    --generate-bev \
    --generate-phase2 \
    --num-workers 4
```

**关键修复（2026-05-28）：**
- NavSatFix 解析：修复 GPS 坐标乱码问题（改用手动偏移量）
- 并行处理：支持 `--num-workers` 多进程加速

### 在其他项目中调用（推荐）

使用 `srnav_processor.py` 封装模块，提供简洁 API：

```python
# 在你的项目中添加路径
import sys
sys.path.insert(0, "/home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer/tools")

# 方式 1: 快捷函数
from srnav_processor import process_chunk, batch_process_chunks
result = process_chunk("chunk_26", num_workers=4)
results = batch_process_chunks(["chunk_26", "chunk_27"], num_workers=4)

# 方式 2: 使用类
from srnav_processor import SRNavProcessor, ProcessingConfig
config = ProcessingConfig(num_workers=4, sample_fps=20.0)
processor = SRNavProcessor(config)
result = processor.process_chunk("chunk_26")
```

详细示例见 `srnav_processor_examples.py`

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

