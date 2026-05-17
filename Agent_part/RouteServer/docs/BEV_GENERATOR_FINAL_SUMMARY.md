# FrodoBot BEV 地图生成器 - 最终总结

## 已完成的功能

### 1. 多地图提供商支持

现已支持以下地图瓦片提供商：

| 提供商 | 说明 | 样式 |
|--------|------|------|
| `osm` | OpenStreetMap 标准 | 默认、通用 |
| `carto` | Carto Voyager | 美观、现代风格 |
| `carto-dark` | Carto Dark Matter | 深色模式 |
| `osm-toner` | Stamen Toner | 高对比度黑白 |
| `osm-terrain` | OpenTopoMap | 地形数据 |
| `custom` | 自定义 URL | 用户提供 |

### 2. 批量处理功能

- 自动发现数据集中的所有 ride
- 进度条显示
- 支持断点续传 (`--resume`)
- 可配置采样率和最大帧数
- 瓦片缓存提高效率

### 3. 可用的工具

#### (1) `test_frodobot_bev.py` - 轻量级测试
使用项目现有代码进行快速测试，支持单帧和 GPS CSV 批量模式。

#### (2) `test_bev_renderer.py` - 独立渲染器
完全独立的 BEV 渲染器，带 asyncio 并行瓦片下载。

#### (3) `frodobot_bev_enhanced.py` - **推荐使用**
增强版批量生成器，支持多地图提供商。

#### (4) `frodobot_bev_dataset_generator.py` - 完整版（已有）
完整的数据集生成器，整合 FPV + BEV + 视觉提示。

---

## 使用示例

### 快速测试 - 单帧

```bash
cd Agent_part/RouteServer

# 使用默认 OSM
python tools/test_frodobot_bev.py \
    --lat 35.716705 --lng 139.854324 --heading 90.0 \
    --output /tmp/test_bev.jpg
```

### 批量处理 - 单个 Ride

```bash
# 使用 Carto 美观样式
python tools/frodobot_bev_enhanced.py \
    --dataset-dir /home/ubuntu/WorkSpace/dataset/output_rides_0 \
    --ride-ids ride_16446_20240115041752 \
    --map-provider carto \
    --output-dir /tmp/my_bev_output \
    --sample-rate 10 \
    --max-frames 50
```

### 批量处理 - 全部 Rides

```bash
# 使用深色模式
python tools/frodobot_bev_enhanced.py \
    --dataset-dir /home/ubuntu/WorkSpace/dataset/output_rides_0 \
    --map-provider carto-dark \
    --output-dir ./data/frodobot_bev_dark \
    --sample-rate 20
```

### 测试多种地图样式对比

```bash
cd tools
python test_multiple_providers.py
# 输出到: /tmp/frodobot_multiple_providers/
```

### 断点续传

如果处理被中断，可以使用 `--resume`：

```bash
python tools/frodobot_bev_enhanced.py \
    --dataset-dir /home/ubuntu/WorkSpace/dataset/output_rides_0 \
    --output-dir ./data/frodobot_bev \
    --resume
```

---

## 输出结果示例

已生成的测试输出：

| 目录 | 说明 |
|------|------|
| `/tmp/frodobot_bev_test/` | 早期 OSM 测试 |
| `/tmp/frodobot_bev_enhanced_test/` | 增强版 OSM 测试（带轨迹） |
| `/tmp/frodobot_bev_carto_test/` | Carto 样式测试 |
| `/tmp/frodobot_multiple_providers/` | 5种地图样式对比 |

---

## 完整数据集生成（下一步）

如需生成完整的 System2 训练数据集（FPV + BEV + 视觉提示），
使用 `frodobot_bev_dataset_generator.py` 并参考 `FRODOBOT_BEV_DATASET_GUIDE.md`。

---

## 主要特性总结

✅ BEV 地图渲染（使用现有项目代码）  
✅ 多地图提供商支持（OSM/Carto/Toner/等）  
✅ GPS 轨迹叠加  
✅ 航向估计与旋转  
✅ 批量处理 + 进度条  
✅ 瓦片缓存  
✅ 断点续传  
✅ 输出 JSON 样本清单  

地图样式对比文件已生成在 `/tmp/frodobot_multiple_providers/` 目录。
