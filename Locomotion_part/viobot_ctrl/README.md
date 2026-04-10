<h1 align="center">Viobot Control & GNSS Integration</h1>

<div align="center">

[![ROS2 CI](https://github.com/HR-zju/Urban-Nav-SR/actions/workflows/ros2_ci.yml/badge.svg)](https://github.com/HR-zju/Urban-Nav-SR/actions/workflows/ros2_ci.yml)

A ROS 2 module for Viobot control and  GNSS Single Point Positioning (SPP) integration, as part of the Urban-Nav-SR project.

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

This project contains the **Viobot Control** (`viobot_ctrl`) and **GNSS SPP Node** (`gnss_comm_impl`) implementations. Developed within the ROS 2 framework, it is specifically designed for navigation scenarios in urban environments. The primary goal is to provide stable and reliable global positioning by integrating GNSS SPP calculations with Stereo Odometry data.

## Features

- **GNSS Single Point Positioning (SPP)**: Baseline positioning calculations.
- **RAIM Gross Error Rejection**: Implements Receiver Autonomous Integrity Monitoring to effectively detect and reject anomalous observations.
- **Elevation Angle and SNR Weighting**: Enhances positioning accuracy by establishing a weighted observation model.
- **Multi-sensor Fusion**: Fuses high-frequency relative displacement from Stereo3 odometry with GNSS SPP outputs.
- **GNSS Status Export**: Publishes GNSS fusion state on `/baton/gnss/status` for external consumers (`gnss_idle`, `gnss_waiting_origin`, `gnss_fusing`).
- **Ephemeris Management**: Periodically cleans up satellite ephemeris data to prevent the accumulation of outdated or incorrect data.
- **Trajectory Export**: Saves latitude and longitude parameters during operation to generate paths compatible with Google Earth visualization.

## Core Parameters & Topics

### Key Topics
- **Subscriptions:**
  - `gnss_comm::msg::GnssEphemMsg` : Satellite ephemeris data.
  - `gnss_comm::msg::GnssMeasMsg` : GNSS observation and measurement epochs.
- **Publications:**
  - `/baton/gnss/status` (`system_ctrl::msg::AlgoStatus`) : Publishes the state machine of the GNSS fusion (`gnss_idle`, `gnss_waiting_origin`, `gnss_fusing`).
  - `/baton/gnss/spp_fix` (`sensor_msgs::msg::NavSatFix`) : Publishes the SPP position fix.
  - `/baton/gnss/spp_fix_raw` (`sensor_msgs::msg::NavSatFix`) : Publishes the raw SPP position fix without stereo3 fusion.
  - `/baton/gnss/heading` (`std_msgs::msg::Float64`) : Publishes the heading estimation derived from stereo3 and GNSS SPP fusion.

### Parameters
- `gnss_initial_calibration_enu_` (vector3): Initial offset applied to the GNSS calculation natively in the ENU frame (default: 0.0, 0.0, 0.0).
- more parameters are defined in the config files.

## Getting Started

### Prerequisites

- **ROS 2** base environment
- **C++14/17** compiler
- Package dependencies: `rclcpp`, `std_msgs`, `sensor_msgs`, `nav_msgs`, `Eigen3`, `glog`

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
2. Build the target packages:
   ```bash
   colcon build
   ```
3. Source the environment setup:
   ```bash
   source install/setup.bash
   ```

## Roadmap & TODO

*The recent development progress and goals of the original project are recorded below:*

- [x] Add RAIM gross error rejection
- [x] Introduce elevation angle and SNR weighting
- [x] Implement periodic ephemeris cleanup
- [x] Save coordinates (latitude/longitude) for Google Earth visualization
- [x] Refactor and separate ROS 2 node code
- [x] Fuse stereo3 odometry and GNSS SPP outputs
- [ ] Collect data and perform field testing
- [x] Test Euler angle output of stereo3
- [x] Heading estimation using stereo3 and GNSS SPP
- [ ] Migrate to Viobot2 platform