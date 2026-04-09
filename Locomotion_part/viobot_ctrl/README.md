<h1 align="center">Viobot Control & GNSS Integration</h1>

<div align="center">

[![ROS2 CI](https://github.com/HR-zju/Urban-Nav-SR/actions/workflows/ros2_ci.yml/badge.svg)](https://github.com/HR-zju/Urban-Nav-SR/actions/workflows/ros2_ci.yml)

A ROS 2 module for Viobot control and  GNSS Single Point Positioning (SPP) integration, as part of the Urban-Nav-SR project.

</div>

## Table of Contents
- [About The Project](#about-the-project)
- [Features](#features)
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
- **Ephemeris Management**: Periodically cleans up satellite ephemeris data to prevent the accumulation of outdated or incorrect data.
- **Trajectory Export**: Saves latitude and longitude parameters during operation to generate paths compatible with Google Earth visualization.

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
- [ ] Test Euler angle output of stereo3
