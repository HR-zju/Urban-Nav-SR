# ROS2 MCAP 快速审核工具（Python 3.8 / 无 ROS 环境）

该版本专门适配：

- Python 3.8.15；
- Synology / NAS；
- 机器上没有 ROS、rclpy、rosbag2_py、cv_bridge；
- 数据为 rosbag2_storage_mcap 生成的 MCAP，或独立 `.mcap` 文件。

读取后生成一个同步审核视频：左侧为前视相机，右侧为 GPS 与 VIO 当前轨迹位置，随后通过 Streamlit 标记 Good / Bad。

## 1. 安装

建议删除旧虚拟环境，避免旧依赖残留：

```bash
cd /volume1/Datasets/mcap_review/ros2_mcap_quick_review_py38
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip<25.1" setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
python -m pip check
```

`pip` 的 cache 权限警告不影响安装；这里使用 `--no-cache-dir` 可避免该警告对应的缓存目录。

## 2. 配置

复制并修改：

```bash
cp config.example.yaml config.yaml
nano config.yaml
```

至少设置：

```yaml
paths:
  data_root: /volume1/Datasets/your_mcap_directory
  work_dir: /volume1/Datasets/mcap_review/work
```

并根据实际 Topic 修改：

```yaml
topics:
  camera:
    names: [/camera_x001/image_raw/compressed]
  gps:
    names: [/baton/gnss/spp_fix]
  vio:
    names: [/baton/stereo3/odometry]
    # Stereo3 的前进方向为机体系 +Y；标准 ROS base_link 通常为 +X。
    heading_yaw_offset_deg: 90.0
```

`heading_yaw_offset_deg` 只旋转审核视频中的 VIO 朝向箭头，不修改 MCAP
原始姿态。标准 `base_link` 数据通常保持 `0.0`；Stereo3 使用 `90.0`。

## 3. 查看 Topic

独立文件：

```bash
python inspect_bag.py /path/to/example.mcap --config config.yaml
```

rosbag2 MCAP 目录：

```bash
python inspect_bag.py /path/to/bag_directory --config config.yaml
```

## 4. 扫描与预处理

```bash
python scan_files.py --config config.yaml
python preprocess.py --config config.yaml --limit 5 --workers 1
```

先检查 5 条结果无误，再全量执行：

```bash
python preprocess.py --config config.yaml --workers 1
```

NAS 上建议先使用 1 个 worker，确认 CPU、内存和磁盘吞吐后再增至 2。

修改 Topic、朝向偏移或渲染参数后，已有缓存不会自动重画，需要执行：

```bash
python preprocess.py --config config.yaml --workers 1 --force
```

## 增量添加 MCAP

支持按新文件路径增量添加，不会重做已经为 `ready` 的旧文件，也不会清除
`annotations.csv` 中已有的 Good / Bad 标签。将新文件放入 `data_root` 后执行：

```bash
python scan_files.py --config config.yaml
python preprocess.py --config config.yaml --workers 1
```

正在运行的审核页面刷新后即可看到新增条目。为避免扫描到尚未复制完整的文件，
建议先使用 `.download` 临时后缀，复制完成后再原子重命名为 `.mcap`。

注意：当前 `bag_id` 由路径生成。如果覆盖同一路径下的旧 MCAP，或者向一个已处理的
rosbag2 目录追加分片，扫描会把它视为原条目；此时请使用 `preprocess.py --force`
重建审核视频。

## 5. 启动审核页面

```bash
export REVIEW_CONFIG=$PWD/config.yaml
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

局域网访问：

```text
http://NAS_IP:8501
```

## 6. 导出标签

```bash
python export_labels.py --config config.yaml
```

输出位于 `work_dir`：

- `annotations.csv`
- `good_files.txt`
- `bad_files.txt`

## 支持范围

支持：

- 独立 `.mcap`；
- 含一个或多个 `.mcap` 的 rosbag2 目录；
- `sensor_msgs/msg/Image`；
- `sensor_msgs/msg/CompressedImage`；
- `sensor_msgs/msg/NavSatFix`；
- `nav_msgs/msg/Odometry`；
- `geometry_msgs/msg/PoseStamped`；
- MCAP 内嵌 `ros2msg` schema 的自定义消息。

当前 Python 3.8 版本不读取 SQLite3 `.db3`。若数据确实包含 `.db3`，建议单独升级到 Python 3.10+ 后使用新版 `rosbags`，不要在此环境里把 `rosbags` 强行降级。

## 常见问题

### `No matching distribution found for rosbags==0.11.3`

原因是 `rosbags 0.11.3` 要求 Python 3.10+。本版本已经移除 `rosbags`，改用 `mcap` 和 `mcap-ros2-support`。

### `Failed to decode ... schema`

MCAP 需要内嵌 `ros2msg` schema。正常由 `rosbag2_storage_mcap` 录制的标准 ROS2 MCAP 通常满足这一条件。若使用特殊录制器只写入 CDR 而未写 schema，则无 ROS 环境下无法自动反序列化。

### OpenCV wheel 无法安装

该版本固定使用 `opencv-python-headless==4.8.1.78`，避免新版 wheel 对较新 glibc 的要求。如果 NAS 仍不兼容，请执行：

```bash
ldd --version | head -1
uname -m
```

并提供输出。
