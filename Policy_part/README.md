# Environment
mamba env [foundation_nav python=3.10 ] 

# Install locally
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
## (for nuvo)docker ros2 + pytorch conda env
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
# (for Orin26) aarch64 deployment

Orin26 uses a separate aarch64/Jetson deployment path from the local x86/RTX
3090 environment. For the current ROS2 Humble launch flow, benchmarked
ONNX/engine files, FPS numbers, TensorRT notes, and Conv fallback pitfall, see:

```text
Policy_part/ORIN26_DEPLOYMENT.md
```

# usage
## no goal
```
python run_navflow_realsense_vis.py --onnx onnx_ckpts/vint_goal_less.onnx --no_show_window
or
python run_navflow_realsense_vis_orin.py --onnx onnx_ckpts/test_mask90_goal_orin_static224_goal_free.onnx  --image_topic /image_raw --image_transport raw --overlay_transport compressed --no_show_window 
```
## point goal
preparing goal point before inference from /home/zjy/code/myk/Urban-Nav-SR/Policy_part/tools/create_topomap_ros2.py
```
python tools/create_topomap_ros2.py -distance-threshold 2.0  # track goal point from odom
# after odom recorded
python run_navflow_realsense_vis_goal_orin.py --onnx onnx_ckpts/test_mixed_mask80_goal_orin_static224_goal_conditioned.onnx --targets ./topomaps/odom_poses.txt --image_topic /image_raw --image_transport raw --overlay_transport raw --no_show_window  --save_vis
```

## LLM json file(got route_polylines from upstream tasks)
```
# transform json gps points to ENU local route polylines before inference
python tools/ros2_cpp_logic_fusion_node.py
# inference with goal point input from LLM_ENU_odom_poses
python run_navflow_realsense_vis_goal_orin.py --onnx onnx_ckpts/test_mixed_mask80_goal_orin_static224_goal_conditioned.onnx --targets ./topomaps/LLM_Policy/LLM_UTM_odom_poses.txt --image_topic /image_raw --image_transport raw --overlay_transport raw --no_show_window 
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
