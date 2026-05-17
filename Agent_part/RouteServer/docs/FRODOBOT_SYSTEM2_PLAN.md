# FrodoBot System2 实现计划

## 目标
在 FrodoBot 数据集上训练 System2 (VLM brain)，替代 RouteServer 的 Agent 部分。输入：FPV + 实时 BEV 地图 → 输出：FPV 上的视觉提示（Visual Prompt）。

数据集格式：`/home/ubuntu/WorkSpace/dataset/output_rides_0`

---

## 一、整体架构

```
RouteServer 项目中制作数据集
    ├── FrodoBot 数据解析 (GPS + 视频 + 时间戳)
    ├── AMap BEV 地图渲染
    ├── 伪指令生成 (起点→终点)
    └── 伪 Visual Prompt 生成 (未来轨迹→FPV投影)

InternNav 训练 (现有 SoM 训练器)
    └── FPV + BEV + instruction → Visual Prompt
```

---

## 二、FrodoBot 数据集格式

```
output_rides_0/
└── ride_16446_20240115041752/
    ├── gps_data_16446.csv            # GPS 轨迹
    │   ├── latitude, longitude, timestamp (ms)
    ├── front_camera_timestamps_16446.csv  # 视频帧时间戳
    │   ├── frame_id, timestamp (sec)
    ├── rear_camera_timestamps_16446.csv
    ├── imu_data_16446.csv
    ├── control_data_16446.csv
    ├── speaker_audio_timestamps_16446.csv
    └── recordings/
        ├── *.ts  # HLS 视频分片
        └── *.m3u8
```

---

## 三、模块详细设计

### 模块 1：FrodoBot 数据解析 (RouteServer 中实现)
参考 `/home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer`

功能：
1. 读取 gps_data_*.csv：GPS 轨迹
2. 读取 front_camera_timestamps_*.csv：视频帧对应时间戳
3. GPS 插值：为每个视频帧精确计算对应 GPS
4. 航向估计：从相邻 GPS 计算
5. 未来轨迹提取：从当前帧往后 N 步
6. 视频帧提取：从 recordings/ 中解码

### 模块 2：AMap BEV 生成 (复用 RouteServer 现有代码)
RouteServer 已有完整实现：
- `map_providers.py`: AMap 客户端
- `route_context/build_bev.py`: 坐标变换
- `route_context/bev_map_provider.py`: BEV 渲染

坐标变换流程：
```
GPS (WGS84)
    ↓ (局部平面投影)
Easting/Northing (米)
    ↓ (按航向旋转)
BEV 坐标 (x=右, y=前)
```

### 模块 3：伪指令生成 (RouteServer 中实现)
简单的模板化导航指令：
- "Go to the destination"
- "Follow the path to the end"
- "Navigate to the target location"

后续可以扩展成基于航点和转向的更丰富指令。

### 模块 4：伪 Visual Prompt 生成 (RouteServer 中实现)
核心：将未来 GPS 轨迹投影到 FPV 图像上

投影步骤：
1. GPS → Ego-local 坐标（复用 RouteServer 的 `_gps_to_bev`）
2. Ego-local → FPV 像素坐标（简化的针孔相机模型）
3. 在 FPV 上渲染路径覆盖层 + 方向箭头
4. 返回图像 + 结构化描述（可选：点坐标序列）

---

## 四、数据集存储格式

制作完成后存储为类似 SoM 数据集的格式，方便 InternNav 训练：

```
data/frodobot_system2/
├── train.jsonl
├── val.jsonl
└── images/
    ├── ride_xxx_frame_xxx_fpv.jpg
    ├── ride_xxx_frame_xxx_bev.jpg
    └── ride_xxx_frame_xxx_prompt.jpg
```

每个样本：
```json
{
  "fpv_image": "path/to/fpv.jpg",
  "bev_image": "path/to/bev.jpg",
  "instruction": "Go to the destination",
  "visual_prompt_image": "path/to/prompt.jpg",
  "visual_prompt_text": "Path: [(x1,y1), (x2,y2), ...]",
  "metadata": {
    "ride_id": "ride_xxx",
    "frame_id": 123,
    "current_gps": [lat, lng],
    "heading_deg": 90.0
  }
}
```

---

## 五、训练流程

数据集制作完成后，直接复用现有的 `internvla_n1_som_trainer.py`：

```bash
python internnav/trainer/internvla_n1_som_trainer.py \
    --model_name_or_path checkpoints/InternVLA-N1-wo-dagger \
    --dataset_mode som \  # 或新增 frodobot 模式
    --data_root ./data/frodobot_system2 \
    --output_dir ./output/frodobot_system2
```

---

## 六、关键注意事项

1. **AMap API 调用限制**：注意缓存 BEV 图片，避免重复请求
2. **时间戳同步**：GPS、视频、控制数据的精确对齐
3. **视频解码效率**：预提取关键帧并缓存
4. **航向估计**：GPS 可能有噪声，需要平滑处理
5. **投影参数**：相机内参需要标定或用合理的启发式值
