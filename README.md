虚拟环境 foundation_nav python=3.10 

其中需要注意的pip环境

[onnxruntime-gpu 适配cuda 11](https://onnxruntime.ai/docs/install/#cuda-11x)

[torch torchvision和cuda版本对应](https://pytorch.org/get-started/previous-versions/)

[tochcodec的版本和torch还有torchvision的版本对应](https://pypi.org/project/torchcodec/)

[mkl版本降级适配mamba安装的torch](https://blog.csdn.net/mr_hore/article/details/138961434)

> 降级numpy numpy在2.0以后发生了ABI的重大修改，pytorch版本不适配
pip install "numpy<2.0"

``` 
# 隔离本地环境 或者不用
# export PYTHONNOUSERSITE=1  
 # 安装cuda和cudnn
mamba install cudatoolkit=11.8 cudnn=8.9.7 -c nvidia 
# onnx runtime 依赖
pip install coloredlogs flatbuffers numpy packaging protobuf sympy 
# 安装onnxruntime的gpu版本
pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/ 
# 安装torch和torchvision
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# 安装torchcodec
pip install torchcodec==0.2.1 --index-url https://download.pytorch.org/whl/cu118
# 需要使用的其他包
pip install "pytorch-lightning>=2.5.0" six pyyaml tqdm scipy matplotlib
# 安装unitree sdk
https://github.com/unitreerobotics/unitree_sdk2_python
```
```
# 如果使用mamba安装的pytorch需要mkl降级
conda install mkl=2024.0
# 如果报错pkg_resources不存在
pip install "setuptools<70.0.0"
```


版本总结：

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

验证

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