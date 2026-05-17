# FrodoBot System2 数据集生成指南

本指南介绍如何基于 FrodoBot 数据集生成 System2 训练所需的 BEV 视图数据。

## 目录
1. [架构概述](#架构概述)
2. [数据集格式](#数据集格式)
3. [快速开始](#快速开始)
4. [详细说明](#详细说明)
5. [配置选项](#配置选项)

---

## 架构概述

### 整体数据流程
```
FrodoBot 原始数据
    │
    ├── GPS 轨迹 (gps_data_*.csv)
    ├── 视频时间戳 (front_camera_timestamps_*.csv)
    └── H.264 视频 (recordings/*.ts)
    │
    ▼
时间同步与 GPS 插值
    │
    ├─ 为每个视频帧查找/插值对应的 GPS 坐标
    ├─ 从 GPS 轨迹估计航向
    └─ 提取未来轨迹用于视觉提示
    │
    ▼
渲染 BEV 地图
    │
    ├─ 地图瓦片获取 (OSM/高德/等)
    ├─ 坐标变换 (WGS84 → Web Mercator → 局部 BEV)
    ├─ 按航向旋转
    └─ 叠加自我标记和路径
    │
    ▼
生成视觉提示
    │
    └─ 将未来 GPS 轨迹投影到 FPV 图像
    │
    ▼
保存训练数据集
```

### 关键模块

| 模块 | 文件 | 功能 |
|------|------|------|
| **GPS 解析** | `frodobot_bev_dataset_generator.py` | 读取 GPS CSV，时间戳对齐 |
| **BEV 渲染** | `test_bev_renderer.py` / `visual_prompt_service.py` | 地图瓦片拼接、旋转、裁剪 |
| **轨迹投影** | `frodobot_bev_dataset_generator.py` | 未来轨迹 → FPV 投影 |
| **数据集输出** | `frodobot_bev_dataset_generator.py` | 保存为 JSONL + 图像 |

---

## 数据集格式

### FrodoBot 原始数据格式
```
output_rides_0/
└── ride_16446_20240115041752/
    ├── gps_data_16446.csv
    │   └─ 列: latitude, longitude, timestamp (ms)
    ├── front_camera_timestamps_16446.csv
    │   └─ 列: frame_id, timestamp (sec)
    ├── rear_camera_timestamps_16446.csv
    ├── imu_data_16446.csv
    ├── control_data_16446.csv
    └── recordings/
        ├── *.ts (HLS 视频分片)
        └── *.m3u8 (播放列表)
```

### 生成的 System2 数据集格式
```
data/frodobot_system2/
├── train.jsonl      # 训练集标注
├── val.jsonl        # 验证集标注
└── images/
    ├── ride_xxx_frame_000001_fpv.jpg
    ├── ride_xxx_frame_000001_bev.jpg
    ├── ride_xxx_frame_000001_prompt.jpg
    └── ...
```

### JSONL 样本格式
```json
{
  "fpv_image": "images/ride_xxx_frame_000001_fpv.jpg",
  "bev_image": "images/ride_xxx_frame_000001_bev.jpg",
  "instruction": "Follow the path to the destination.",
  "visual_prompt_image": "images/ride_xxx_frame_000001_prompt.jpg",
  "visual_prompt_text": "Follow the path indicated by the orange line.",
  "metadata": {
    "ride_id": "ride_xxx",
    "frame_id": 1,
    "current_gps": {
      "lat": 35.7167,
      "lng": 139.8543
    },
    "heading_deg": 90.0,
    "timestamp_sec": 1705292281.0
  }
}
```

---

## 快速开始

### 1. 环境准备

```bash
cd Agent_part/RouteServer

# 安装依赖（如果需要）
pip install aiohttp pillow opencv-python numpy
```

### 2. 测试 BEV 渲染（单张）

```bash
# 使用默认坐标测试
python tools/test_bev_renderer.py \
    --lat 35.7167 \
    --lng 139.8543 \
    --heading 90.0 \
    --output bev_test.jpg

# 自定义瓦片服务器（可选）
python tools/test_bev_renderer.py \
    --lat 35.7167 \
    --lng 139.8543 \
    --tile-url "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
```

### 3. 测试 BEV 渲染（批量，从 GPS CSV）

```bash
python tools/test_bev_renderer.py \
    --gps-csv /home/ubuntu/WorkSpace/dataset/output_rides_0/ride_16446_20240115041752/gps_data_16446.csv \
    --output-dir bev_outputs \
    --sample-rate 10
```

### 4. 生成完整数据集

```bash
python tools/frodobot_bev_dataset_generator.py \
    --dataset-dir /home/ubuntu/WorkSpace/dataset/output_rides_0 \
    --output-dir data/frodobot_system2 \
    --sample-rate 10 \
    --train-split 0.8

# 只处理特定的 rides
python tools/frodobot_bev_dataset_generator.py \
    --dataset-dir /home/ubuntu/WorkSpace/dataset/output_rides_0 \
    --ride-ids ride_16446_20240115041752 ride_16448_20240115042257
```

---

## 详细说明

### 坐标变换详解

BEV 渲染涉及多次坐标变换：

```
WGS84 (GPS 坐标)
    │
    │ lat/lng -> 瓦片坐标
    ▼
Web Mercator (瓦片坐标)
    │
    │ 拼接瓦片 -> 裁剪 -> 旋转
    ▼
Ego-centric BEV (像素坐标)
```

#### 1. WGS84 → Web Mercator 瓦片坐标
```python
def latlon_to_tile_xy(lat, lng, zoom):
    n = 2.0 ** zoom
    x = (lng + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y
```

#### 2. 航向旋转
- 0° = 北，90° = 东
- 旋转后，图像向上 = 车辆前进方向

```python
# 旋转镶嵌图
rotated_mosaic = mosaic.rotate(
    heading_deg,
    center=(local_center_px, local_center_py),
    fillcolor=(242, 239, 233),
)
```

### 视觉提示投影

将未来 GPS 轨迹投影到 FPV 图像使用简化的针孔相机模型：

```
未来 GPS 点
    │
    │ WGS84 → 局部东北天
    ▼
局部坐标系 (x=右, y=前, z=下)
    │
    │ 针孔投影 + 向下倾斜
    ▼
FPV 像素坐标
```

关键参数：
- 水平视场角 (HFOV): 90°
- 垂直视场角 (VFOV): 60°
- 相机高度: 1.2 m
- 向下倾斜: 10°

---

## 配置选项

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BEV_MAP_TILE_URL_TEMPLATE` | `https://tile.openstreetmap.org/{z}/{x}/{y}.png` | 地图瓦片 URL |
| `BEV_MAP_TILE_CACHE_DIR` | `/tmp/urban_nav_osm_tiles` | 瓦片缓存目录 |
| `BEV_MAP_TILE_TIMEOUT_S` | `8` | 瓦片请求超时（秒） |
| `BEV_MAP_TILE_USER_AGENT` | `UrbanNav/1.0` | HTTP User-Agent |
| `BEV_MAP_HEADING_CONVENTION` | `compass` | 航向约定 (`compass` 或其他) |
| `BEV_MAP_HEADING_OFFSET_DEG` | `0` | 航向偏移角度 |

### 可选的瓦片服务器

```bash
# OpenStreetMap (默认)
export BEV_MAP_TILE_URL_TEMPLATE="https://tile.openstreetmap.org/{z}/{x}/{y}.png"

# 高德地图 (需要 API key)
# 注意：需要按照高德 API 格式调整代码

# Carto (风格化地图)
export BEV_MAP_TILE_URL_TEMPLATE="https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
```

### 命令行选项

#### frodobot_bev_dataset_generator.py

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-dir` | `output_rides_0` | FrodoBot 数据集根目录 |
| `--output-dir` | `data/frodobot_system2` | 输出目录 |
| `--sample-rate` | `10` | 每 N 帧采样一次 |
| `--max-frames-per-ride` | `None` | 每个 ride 最多帧数 |
| `--train-split` | `0.8` | 训练集比例 |
| `--bev-tile-url` | `None` | BEV 瓦片 URL 模板 |
| `--ride-ids` | `None` | 只处理指定 rides |

#### test_bev_renderer.py

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--lat` | `35.7167` | 纬度 |
| `--lng` | `139.8543` | 经度 |
| `--heading` | `90.0` | 航向 |
| `--output` | `bev_output.jpg` | 输出路径 |
| `--zoom` | `19` | 缩放级别 |
| `--size` | `640` | 输出图像尺寸 |
| `--gps-csv` | `None` | GPS CSV 批量模式 |
| `--output-dir` | `bev_outputs` | 批量输出目录 |

---

## 训练集成

生成的数据集可直接用于训练，参考 `FRODOBOT_SYSTEM2_PLAN.md` 中的说明：

```bash
# 使用现有的 InternNav 训练器
python internnav/trainer/internvla_n1_som_trainer.py \
    --model_name_or_path checkpoints/InternVLA-N1-wo-dagger \
    --dataset_mode som \
    --data_root ./data/frodobot_system2 \
    --output_dir ./output/frodobot_system2
```

---

## 故障排除

### 问题：瓦片下载失败
- 检查网络连接
- 尝试更换瓦片服务器
- 查看缓存目录是否有写入权限

### 问题：视频无法解码
- 确认 OpenCV 已正确安装
- 尝试使用 `ffmpeg` 预提取视频帧

### 问题：航向估计不准确
- GPS 信号差时可能出现
- 可以考虑使用 IMU 数据（扩展功能）

---

## 扩展开发

### 添加自定义地图提供商

参考 `map_providers.py` 中的实现，主要需要：
1. 瓦片 URL 模板
2. 坐标转换（如果需要）

### 添加更精确的相机标定

修改 `project_trajectory_to_fpv()` 函数，使用真实的相机内参：
```python
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # 内参矩阵
dist_coeffs = np.array([k1, k2, p1, p2, k3])  # 畸变系数
```

---

## 参考文献

- RouteServer 原有 BEV 实现: `services/visual_prompt_service.py`
- 坐标变换: `route_context/build_bev.py`
- 地图客户端: `map_providers.py`
