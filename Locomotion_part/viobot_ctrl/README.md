<h1 align="center">Viobot Control & GNSS Integration</h1>

<div align="center">

[![ROS2 CI](https://github.com/HR-zju/Urban-Nav-SR/actions/workflows/ros2_ci.yml/badge.svg)](https://github.com/HR-zju/Urban-Nav-SR/actions/workflows/ros2_ci.yml)

A ROS 2 workspace for Viobot algorithm control, GNSS Single Point Positioning (SPP), Stereo3/GNSS fusion, camera publishing, and synchronized bag recording.

[English README](README.md) | [中文说明](README_zh.md) | [录制数据手册](DATA_RECORDING_MANUAL_zh.md)

</div>

## Table of Contents
- [About The Project](#about-the-project)
- [Features](#features)
- [Core Parameters & Topics](#core-parameters--topics)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Roadmap & TODO](#roadmap--todo)

## About The Project

This workspace contains the current **Viobot Control** (`viobot_ctrl`), **GNSS SPP Node** (`gnss_comm_impl`), **MJPEG camera publisher** (`cam_pub`), and ROS 2 interface packages used by Urban-Nav-SR.

The current implementation provides:

- `algo_ctrl_node`: an interactive terminal tool that publishes Stereo3 control commands to `/baton/stereo3_ctrl` and displays `/baton/algo_status`.
- `stereo3_auto_start_node`: a helper that waits for the first raw GNSS SPP fix and then sends one Stereo3 enable command.
- `gnss_spp_node`: an SPP and Stereo3/GNSS fusion node that publishes raw GNSS fixes, fused fixes, GNSS fusion status, and heading.
- `cam_pub`: a V4L2 MJPEG publisher for `/camera_x001/image_raw/compressed` using best-effort QoS and configurable framerate limiting.
- `start_bag.sh` and systemd units for deployment-time camera startup and MCAP recording.

## Features

- **Viobot Algorithm Control**: Sends enable, disable, reboot, and reset commands through `system_ctrl/msg/AlgoCtrl`.
- **GNSS Single Point Positioning (SPP)**: Calculates baseline GNSS positions from `/baton/gnss/ephem` and `/baton/gnss/meas`.
- **RAIM Gross Error Rejection**: Performs residual-based satellite exclusion in `psr_pos`.
- **Elevation, Observation Noise, and URA Weighting**: Uses elevation angle, pseudorange standard deviation, and broadcast ephemeris accuracy in the weighted least-squares model.
- **GNSS Data Validation**: Filters invalid ephemeris and observation messages before solving.
- **Hatch Filtering**: Smooths L1 pseudorange with carrier phase when valid observations are available.
- **Stereo3/GNSS Fusion**: Builds a stable GNSS origin, dynamically aligns Stereo3 odometry to GNSS ENU coordinates, and publishes fused global fixes.
- **Heading Estimation**: Publishes heading derived from Stereo3 orientation after dynamic GNSS alignment.
- **GNSS Status Export**: Publishes `/baton/gnss/status` as `gnss_idle`, `gnss_waiting_origin`, `gnss_aligning`, or `gnss_fusing`.
- **KML Trajectory Export**: Optionally writes raw and fused trajectories for map visualization.
- **MJPEG Camera Publishing**: Captures compressed frames from a V4L2 device with best-effort QoS, configurable framerate limiting, and timeout-based restart behavior.
- **MCAP Recording Support**: Records GNSS, odometry, status, heading, and compressed image topics with Zstd compression.

## Core Parameters & Topics

### Key Topics

- **Viobot control**
  - Publishes `/baton/stereo3_ctrl` (`system_ctrl/msg/AlgoCtrl`)
  - Subscribes `/baton/algo_status` (`system_ctrl/msg/AlgoStatus`)

- **GNSS SPP and fusion subscriptions**
  - `/baton/gnss/ephem` (`gnss_comm/msg/GnssEphemMsg`): satellite ephemeris data.
  - `/baton/gnss/meas` (`gnss_comm/msg/GnssMeasMsg`): GNSS observation epochs.
  - `/baton/stereo3/odometry` (`nav_msgs/msg/Odometry`): Stereo3 relative odometry.
  - `/baton/algo_status` (`system_ctrl/msg/AlgoStatus`): algorithm running state.

- **GNSS SPP and fusion publications**
  - `/baton/gnss/status` (`system_ctrl/msg/AlgoStatus`): GNSS fusion state machine.
  - `/baton/gnss/spp_fix_raw` (`sensor_msgs/msg/NavSatFix`): raw SPP fix after GNSS filtering and calibration.
  - `/baton/gnss/spp_fix` (`sensor_msgs/msg/NavSatFix`): Stereo3/GNSS fused global fix.
  - `/baton/gnss/heading` (`std_msgs/msg/Float64`): fused heading in degrees.

- **Camera**
  - Publishes `/camera_x001/image_raw/compressed` (`sensor_msgs/msg/CompressedImage`) by default.

`start_bag.sh` records the GNSS, odometry, algorithm status, camera, and heading topics above with MCAP/Zstd compression.

### Parameters

- **`stereo3_auto_start_node`**
  - `gnss_fix_topic` (string, default: `/baton/gnss/spp_fix_raw`)
  - `stereo3_ctrl_topic` (string, default: `/baton/stereo3_ctrl`)

- **`gnss_spp_node`**
  - `enable_kml_export` (bool, default: `true`)
  - `kml_directory` (string, default: `/home/symc/data/kml`)
  - `kml_raw_directory` (string, default: `/home/symc/data/kml_raw`)
  - `gnss.initial_calibration_enu` (vector3, config value: `[-3.0, -4.5, 0.0]`)
  - `dynamic_aligner.window_size` (int, config value: `100`)
  - `dynamic_aligner.min_dist` (double, config value: `1.0`)
  - `dynamic_aligner.ransac_iters` (int, config value: `100`)
  - `dynamic_aligner.inlier_thresh` (double, config value: `2.0`)
  - `dynamic_aligner.filter_gain` (double, config value: `0.05`)
  - `dynamic_aligner.anchor_window_size` (int, config value: `100`)
  - `dynamic_aligner.anchor_min_dist` (double, config value: `10.0`)
  - `dynamic_aligner.max_consecutive_jumps` (int, config value: `5`)

- **`cam_pub`**
  - `topic_name` (string, default: `/camera_x001/image_raw/compressed`)
  - `frame_id` (string, default: `camera_link`)
  - `video_device` (string, default: `/dev/video40`)
  - `image_width` (int, default: `1920`)
  - `image_height` (int, default: `1080`)
  - `framerate` (double, default: `20.0`; set `0.0` to disable framerate control)
  - `buffer_count` (int, default: `8`)
  - `qos_depth` (int, default: `10`, best-effort publisher history depth)
  - `dqbuf_timeout_ms` (int, default: `2000`)
  - `max_consecutive_timeouts` (int, default: `5`)

## Getting Started

### Prerequisites

- **ROS 2 Humble** or a compatible ROS 2 environment
- **C++17** compiler for `gnss_comm_impl`
- Package dependencies: `rclcpp`, `std_msgs`, `sensor_msgs`, `nav_msgs`, `launch`, `launch_ros`, `Eigen3`, `glog`, `OpenCV`, `cv_bridge`
- Optional tools for deployment scripts: `zstd`, `ros2bag`, `rosbag2_storage_mcap`

### Installation

0. Install dependencies:
   ```bash
   sudo apt-get update
   rosdep update
   rosdep install --from-paths Urban-Nav-SR/Locomotion_part/viobot_ctrl/src --ignore-src -r -y
   ```

1. Navigate to the workspace:
   ```bash
   cd Urban-Nav-SR/Locomotion_part/viobot_ctrl
   ```

2. Build the workspace:
   ```bash
   colcon build --symlink-install
   ```

3. Source the environment setup:
   ```bash
   source install/setup.bash
   ```

Common runtime commands after sourcing:

```bash
ros2 launch viobot_ctrl start_gnss_then_stereo3.launch.py
ros2 run viobot_ctrl algo_ctrl_node
ros2 run cam_pub cam_pub
```

## Roadmap & TODO

*The recent development progress and goals of the original project are recorded below:*

- [x] Add RAIM gross error rejection
- [x] Introduce elevation angle, observation noise, and URA weighting
- [x] Implement GNSS ephemeris and observation validation
- [x] Implement periodic ephemeris cleanup
- [x] Save raw and fused trajectories for KML visualization
- [x] Refactor and separate ROS 2 node code
- [x] Fuse Stereo3 odometry and GNSS SPP outputs
- [x] Add GNSS status heartbeat
- [x] Add Stereo3 auto-start after raw GNSS fix
- [x] Add heading estimation using Stereo3 and GNSS SPP
- [x] Add V4L2 MJPEG camera publisher
- [x] Add MCAP bag recording and repair scripts
- [ ] Collect more field data and evaluate robustness
- [x] Migrate to Viobot2 platform
