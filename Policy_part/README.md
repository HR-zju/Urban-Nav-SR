# Environment
mamba env [foundation_nav python=3.10 ] 

# Install
``` 
# 隔离本地环境 或者不用
# export PYTHONNOUSERSITE=1  
 # install cuda和cudnn
mamba install cudatoolkit=11.8 cudnn=8.9.7 -c nvidia 
# onnx runtime 依赖
pip install coloredlogs flatbuffers numpy packaging protobuf sympy 
# 安装onnxruntime的gpu版本
pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/ 
# 安装torch和torchvision
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# 安装torchcodec
pip install torchcodec==0.2.1 --index-url https://download.pytorch.org/whl/cu118
# other packages
pip install "pytorch-lightning>=2.5.0" six pyyaml tqdm scipy matplotlib
# install unitree sdk
https://github.com/unitreerobotics/unitree_sdk2_python
```

## env version conclusion：
```bash
> pip list | grep orch
pytorch-lightning                        2.6.1
torch                                    2.6.0+cu118
torchaudio                               2.6.0+cu118
TorchCodec                               0.2.0+cu118
torchmetrics                             1.9.0
torchvision                              0.21.0+cu118

numpy                                    1.26.4
```

## environment validation
```bash
import torch
import torchvision

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"当前使用的 CUDA 版本: {torch.version.cuda}")
print(f"显卡型号: {torch.cuda.get_device_name(0)}")

import onnxruntime as ort
# 检查是否真的加载了 CUDA 11 的支持
pr = ort.get_available_providers()
print(pr)
if 'CUDAExecutionProvider' in pr:
    print("已激活")
```

# Deployment
## docker ros2 + pytorch conda env
```
# 确定容器工具安装完成
nvidia-container-toolkit --version
# 拉取ros2镜像
docker pull osrf/ros:humble-desktop
# 构建容器
docker run -it \
  --name sr_ros2 \
  --gpus all \
  --network host \
  --privileged \
  --env CONN_TYPE="cyclonedds" \
  --env WEBRTC_SERVER_HOST=0.0.0.0 \
  --env WEBRTC_SERVER_PORT=${WEBRTC_SERVER_PORT:-9991} \
  --env DISPLAY=${DISPLAY:-} \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --volume ${HOME}/.Xauthority:/root/.Xauthority:rw \
  --volume ${HOME}:${HOME} \
  --device /dev/input:/dev/input \
  osrf/ros:humble-desktop \
  /bin/bash
  
# 激活环境
/home/nuvo/miniforge3/bin/mamba shell init --shell bash
# 唤醒mamba环境
source ~/.bashrc
```
# Usage

## Realtime NavFlow runner with overlay topic

`run_navflow_realsense_vis.py` 和 `run_navflow_realsense_vis_goal.py` 都会订阅 RealSense 压缩图像和 odom，实时推理 NavFlow，并把预测轨迹画回图像。

当前默认输入话题：

| Script | Image topic | Odom topic |
| --- | --- | --- |
| `run_navflow_realsense_vis.py` | `/camera/camera/color/image_raw/compressed` | `/globalEstimator/global_odometry` |
| `run_navflow_realsense_vis_goal.py` | `/camera/camera/color/image_raw/compressed` | `/globalEstimator/global_odometry` |

默认行为：
- 开启 OpenCV 预览窗口，除非加 `--no_show_window`
- 发布轨迹 overlay 图像话题，默认是 compressed JPEG
- overlay 话题默认名：`/navflow/overlay/image_raw/compressed`
- overlay 话题类型：`sensor_msgs/msg/CompressedImage`

示例：

```bash
python run_navflow_realsense_vis.py \
  --onnx onnx_ckpts/vint_point_goal.onnx \
  --targets ./topomaps/odom_poses.txt
```

无窗口运行，但继续发布 compressed overlay：

```bash
python run_navflow_realsense_vis.py \
  --onnx onnx_ckpts/vint_point_goal.onnx \
  --targets ./topomaps/odom_poses.txt \
  --no_show_window
```

关闭 overlay 图像话题发布：

```bash
python run_navflow_realsense_vis.py \
  --onnx onnx_ckpts/vint_point_goal.onnx \
  --targets ./topomaps/odom_poses.txt \
  --no_publish_overlay
```

修改 overlay 话题名或 JPEG 质量：

```bash
python run_navflow_realsense_vis.py \
  --onnx onnx_ckpts/vint_point_goal.onnx \
  --targets ./topomaps/odom_poses.txt \
  --overlay_topic /debug/navflow_overlay/compressed \
  --overlay_jpeg_quality 90
```

如果需要发布 raw `sensor_msgs/msg/Image` 而不是 compressed：

```bash
python run_navflow_realsense_vis.py \
  --onnx onnx_ckpts/vint_point_goal.onnx \
  --targets ./topomaps/odom_poses.txt \
  --overlay_transport raw
```

raw 默认话题为 `/navflow/overlay/image_raw`，compressed 默认话题为 `/navflow/overlay/image_raw/compressed`。

如果轨迹在图像中偏高、不贴地，可以调可视化专用 pitch。这个参数只影响画图投影，不改变 ONNX 推理输出和发布到 `waypoint` 的控制轨迹：

```bash
python run_navflow_realsense_vis.py \
  --onnx onnx_ckpts/vint_point_goal.onnx \
  --targets ./topomaps/odom_poses.txt \
  --overlay_camera_pitch_deg 8.0
```

`--overlay_camera_pitch_deg` 默认是 `8.0`。正值表示虚拟相机抬头，会让投影到地面的轨迹在图像中更靠下；如果仍然偏高，可以试 `10.0` 或 `12.0`。如果压得太低，则减小到 `6.0`、`3.0` 或 `0.0`。

## Offline GT odom trajectory overlay

`tools/vis_odom_gt_video.py` 用于直接检查 `data_format` 数据集里的 GT odom。它会读取每个 clip 目录中的：

- `frame_data.npy`
- `front_pinhole_crop_resize.mp4`

脚本按目录名去掉 `__clip_XXXX` 后分组，把同一个原视频的多个 clip 按编号合成为一个输出视频，并把未来一段 odom 轨迹投影到前视画面上。

下面命令默认在仓库根目录执行。如果还没有 `data_format` 入口，先创建软链接：

```bash
ln -s /media/zjy/Elements\ SE/vint_fine_tune_data_format ./data_format
```

默认输入路径：

```bash
./data_format
```

默认输出路径：

```bash
Policy_part/vis/gt_odom_overlay/
```

查看分组但不渲染：

```bash
python Policy_part/tools/vis_odom_gt_video.py --dry-run
```

渲染全部视频组：

```bash
python Policy_part/tools/vis_odom_gt_video.py
```

只渲染某一个视频组：

```bash
python Policy_part/tools/vis_odom_gt_video.py --include 20260510_081025_2
```

常用可调参数：

```bash
python Policy_part/tools/vis_odom_gt_video.py \
  --future-frames 120 \
  --stride 5 \
  --max-forward-m 20 \
  --heading-smooth-window 8
```

其中 `future-frames` 控制每帧向后看的 odom horizon，`stride` 控制 waypoint 抽样间隔，`max-forward-m` 控制投影到画面上的最大前向距离。

## Offline ONNX mp4 runner

```
# test runner scripts
python run_onnx_from_mp4.py --mp4 front/front.mp4 --onnx onnx_ckpts/s1_ours_coco_sft_stage2_goal_less.onnx --saveOverlay --sample_fps 5
```


# troubleshooting

``` 
# 降级numpy numpy在2.0以后发生了ABI的重大修改，pytorch版本不适配
pip install "numpy<2.0"
# if install pytorch by mamba install may need degrade mkl
conda install mkl=2024.0
# 如果报错pkg_resources不存在
pip install "setuptools<70.0.0"
```
