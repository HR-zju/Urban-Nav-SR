<h1 align="center">Viobot 控制与 GNSS 集成</h1>

<div align="center">

[![ROS2 CI](https://github.com/HR-zju/Urban-Nav-SR/actions/workflows/ros2_ci.yml/badge.svg)](https://github.com/HR-zju/Urban-Nav-SR/actions/workflows/ros2_ci.yml)

Urban-Nav-SR 项目中的 ROS 2 工作区，用于 Viobot 算法控制、GNSS 单点定位（SPP）、Stereo3/GNSS 融合、相机发布和同步录包。

[English README](README.md) | [中文说明](README_zh.md) | [录制数据手册](DATA_RECORDING_MANUAL_zh.md) | [技术实现](TECHNICAL_IMPLEMENTATION_README_zh.md)

</div>

## 目录
- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [核心参数与话题](#核心参数与话题)
- [快速开始](#快速开始)
  - [依赖环境](#依赖环境)
  - [安装构建](#安装构建)
- [路线图与 TODO](#路线图与-todo)

## 项目简介

本工作区包含 Urban-Nav-SR 当前使用的 **Viobot 控制包**（`viobot_ctrl`）、**GNSS SPP 节点**（`gnss_comm_impl`）、**MJPEG 相机发布节点**（`cam_pub`）以及 ROS 2 接口包。

当前实现提供：

- `algo_ctrl_node`：交互式终端控制工具，向 `/baton/stereo3_ctrl` 发布 Stereo3 控制命令，并显示 `/baton/algo_status`。
- `stereo3_auto_start_node`：等待首个原始 GNSS SPP fix 后，自动发送一次 Stereo3 启用命令。
- `gnss_spp_node`：SPP 与 Stereo3/GNSS 融合节点，发布原始 GNSS fix、融合 fix、GNSS 融合状态和航向角。
- `cam_pub`：面向 `/camera_x001/image_raw/compressed` 的 V4L2 MJPEG 图像发布节点，使用 best-effort QoS，并支持帧率限制。
- `start_bag.sh` 与 systemd 服务：用于部署环境中的相机启动和 MCAP 录包。

## 功能特性

- **Viobot 算法控制**：通过 `system_ctrl/msg/AlgoCtrl` 发送启用、关闭、重启和重置命令。
- **GNSS 单点定位（SPP）**：基于 `/baton/gnss/ephem` 和 `/baton/gnss/meas` 计算基础 GNSS 位置。
- **RAIM 粗差剔除**：在 `psr_pos` 中根据后验残差执行卫星剔除。
- **高度角、观测噪声与 URA 加权**：在加权最小二乘模型中使用高度角、伪距标准差和广播星历精度。
- **GNSS 数据校验**：在解算前过滤异常星历和观测消息。
- **Hatch 滤波**：在观测有效时使用载波相位平滑 L1 伪距。
- **Stereo3/GNSS 融合**：建立稳定 GNSS 原点，动态对齐 Stereo3 里程计到 GNSS ENU 坐标，并发布融合后的全局位置。
- **航向估计**：动态 GNSS 对齐后，根据 Stereo3 姿态发布航向角。
- **GNSS 状态导出**：在 `/baton/gnss/status` 发布 `gnss_idle`、`gnss_waiting_origin`、`gnss_aligning`、`gnss_fusing`。
- **KML 轨迹导出**：可选导出原始和融合轨迹，便于地图可视化。
- **MJPEG 相机发布**：从 V4L2 设备采集压缩图像，使用 best-effort QoS，支持帧率限制，连续超时后退出以便 systemd 重启。
- **MCAP 录包支持**：使用 Zstd 压缩录制 GNSS、里程计、状态、航向和压缩图像话题。

## 核心参数与话题

### 关键话题

- **Viobot 控制**
  - 发布 `/baton/stereo3_ctrl`（`system_ctrl/msg/AlgoCtrl`）
  - 订阅 `/baton/algo_status`（`system_ctrl/msg/AlgoStatus`）

- **GNSS SPP 与融合订阅**
  - `/baton/gnss/ephem`（`gnss_comm/msg/GnssEphemMsg`）：卫星星历数据。
  - `/baton/gnss/meas`（`gnss_comm/msg/GnssMeasMsg`）：GNSS 观测历元。
  - `/baton/stereo3/odometry`（`nav_msgs/msg/Odometry`）：Stereo3 相对里程计。
  - `/baton/algo_status`（`system_ctrl/msg/AlgoStatus`）：算法运行状态。

- **GNSS SPP 与融合发布**
  - `/baton/gnss/status`（`system_ctrl/msg/AlgoStatus`）：GNSS 融合状态机。
  - `/baton/gnss/spp_fix_raw`（`sensor_msgs/msg/NavSatFix`）：经过 GNSS 过滤和标定后的原始 SPP fix。
  - `/baton/gnss/spp_fix`（`sensor_msgs/msg/NavSatFix`）：Stereo3/GNSS 融合后的全局 fix。
  - `/baton/gnss/heading`（`std_msgs/msg/Float64`）：融合后的航向角，单位为度。

- **相机**
  - 默认发布 `/camera_x001/image_raw/compressed`（`sensor_msgs/msg/CompressedImage`）。

`start_bag.sh` 会使用 MCAP/Zstd 压缩录制上面的 GNSS、里程计、算法状态、相机和航向话题。

### 参数

- **`stereo3_auto_start_node`**
  - `gnss_fix_topic`（string，默认：`/baton/gnss/spp_fix_raw`）
  - `stereo3_ctrl_topic`（string，默认：`/baton/stereo3_ctrl`）

- **`gnss_spp_node`**
  - `enable_kml_export`（bool，默认：`true`）
  - `kml_directory`（string，默认：`/home/symc/data/kml`）
  - `kml_raw_directory`（string，默认：`/home/symc/data/kml_raw`）
  - `gnss.initial_calibration_enu`（vector3，配置值：`[-3.0, -4.5, 0.0]`）
  - `dynamic_aligner.window_size`（int，配置值：`100`）
  - `dynamic_aligner.min_dist`（double，配置值：`1.0`）
  - `dynamic_aligner.ransac_iters`（int，配置值：`100`）
  - `dynamic_aligner.inlier_thresh`（double，配置值：`2.0`）
  - `dynamic_aligner.filter_gain`（double，配置值：`0.05`）
  - `dynamic_aligner.anchor_window_size`（int，配置值：`100`）
  - `dynamic_aligner.anchor_min_dist`（double，配置值：`10.0`）
  - `dynamic_aligner.max_consecutive_jumps`（int，配置值：`5`）

- **`cam_pub`**
  - `topic_name`（string，默认：`/camera_x001/image_raw/compressed`）
  - `frame_id`（string，默认：`camera_link`）
  - `video_device`（string，默认：`/dev/video40`）
  - `image_width`（int，默认：`1920`）
  - `image_height`（int，默认：`1080`）
  - `framerate`（double，默认：`20.0`；设为 `0.0` 可关闭帧率控制）
  - `buffer_count`（int，默认：`8`）
  - `qos_depth`（int，默认：`10`，best-effort 发布端历史深度）
  - `dqbuf_timeout_ms`（int，默认：`2000`）
  - `max_consecutive_timeouts`（int，默认：`5`）

## 快速开始

### 依赖环境

- **ROS 2 Humble** 或兼容的 ROS 2 环境
- **C++17** 编译器，用于 `gnss_comm_impl`
- 软件包依赖：`rclcpp`、`std_msgs`、`sensor_msgs`、`nav_msgs`、`launch`、`launch_ros`、`Eigen3`、`glog`、`OpenCV`、`cv_bridge`
- 部署脚本可选依赖：`zstd`、`ros2bag`、`rosbag2_storage_mcap`

### 安装构建

0. 安装依赖：
   ```bash
   sudo apt-get update
   rosdep update
   rosdep install --from-paths Urban-Nav-SR/Locomotion_part/viobot_ctrl/src --ignore-src -r -y
   ```

1. 进入工作区：
   ```bash
   cd Urban-Nav-SR/Locomotion_part/viobot_ctrl
   ```

2. 编译工作区：
   ```bash
   colcon build --symlink-install
   ```

3. 加载环境：
   ```bash
   source install/setup.bash
   ```

加载环境后可使用这些常用命令：

```bash
ros2 launch viobot_ctrl start_gnss_then_stereo3.launch.py
ros2 run viobot_ctrl algo_ctrl_node
ros2 run cam_pub cam_pub
```

## 路线图与 TODO

*原项目近期开发进度与后续目标记录如下：*

- [x] 添加 RAIM 粗差剔除
- [x] 引入高度角、观测噪声和 URA 加权
- [x] 添加 GNSS 星历和观测数据校验
- [x] 实现星历定期清理
- [x] 保存原始和融合轨迹用于 KML 可视化
- [x] 重构并拆分 ROS 2 节点代码
- [x] 融合 Stereo3 里程计和 GNSS SPP 输出
- [x] 添加 GNSS 状态心跳
- [x] 添加 raw GNSS fix 后自动启动 Stereo3
- [x] 使用 Stereo3 与 GNSS SPP 估计航向角
- [x] 添加 V4L2 MJPEG 相机发布节点
- [x] 添加 MCAP 录包与修复脚本
- [ ] 采集更多实测数据并评估鲁棒性
- [x] 迁移到 Viobot2 平台
