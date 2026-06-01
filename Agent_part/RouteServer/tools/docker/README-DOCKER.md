# UrbanNav Docker 部署指南

本指南说明如何使用 Docker 来运行 UrbanNav 数据处理脚本，避免复杂的环境配置。

## 前置要求

- Docker 已安装（服务端和本地都需要）
- 数据集已准备好

---

## 快速开始

### 1. 构建 Docker 镜像

在 `Agent_part/RouteServer` 目录下运行：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer
docker build -f tools/docker/Dockerfile -t urbannav-processor:latest .
```

注意：构建上下文必须是 `Agent_part/RouteServer`，因为 `--generate-bev` 会导入 `src/mcp_server` 中的 BEV/地图辅助函数。

如果要完整复现本机 conda 环境，推荐使用封装脚本：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer/tools
./docker/build-urbannav-conda-image.sh
```

该脚本会把 `/home/ubuntu/anaconda3/envs/UrbanNav` 打包为 `tools/docker/UrbanNav-env.tar.gz`，再构建镜像。镜像内默认 Python 是 `/home/ubuntu/anaconda3/envs/UrbanNav/bin/python`。

### 2. 准备数据目录

在服务器上创建数据目录：

```bash
mkdir -p /path/to/dataset
mkdir -p /path/to/output
mkdir -p /path/to/temp
```

将你的 MCAP 数据集放到 `/path/to/dataset`。

### 3. 运行容器处理数据

```bash
docker run -it --rm \
  -v /path/to/your/dataset:/data/dataset:ro \
  -v /path/to/output:/data/output \
  -v /path/to/temp:/data/temp \
  urbannav-processor:latest \
  dataset_processing/process_chunks_26_29.py \
  --dataset-root /data/dataset \
  --sidewalk-root /data/output \
  --temp-dir /data/temp \
  --chunks 26 27 28 29 \
  --sample-fps 20.0 \
  --generate-bev \
  --use-pose-ekf
```

---

## 使用 Docker Compose（推荐）

### 1. 修改 docker-compose.yml

根据你的实际路径修改 `docker-compose.yml` 中的卷挂载配置：

```yaml
volumes:
  - /your/actual/dataset/path:/data/dataset:ro
  - ./output:/data/output
  - ./temp:/data/temp
```

### 2. 构建并运行

```bash
# 构建镜像
docker-compose build

# 运行处理（修改 command 部分）
docker-compose run --rm urbannav-processor \
  dataset_processing/process_chunks_26_29.py \
  --dataset-root /data/dataset \
  --sidewalk-root /data/output \
  --temp-dir /data/temp \
  --chunks 26 27 28 29 \
  --sample-fps 20.0 \
  --generate-bev \
  --use-pose-ekf
```

---

## 部署到服务器

### 方法一：直接在服务器构建

1. 将以下文件复制到服务器：
   - `tools/docker/Dockerfile`
   - `tools/docker/requirements.txt`
   - `tools/docker/docker-compose.yml`
   - `tools/` 下各分类目录中的脚本

2. 在服务器上构建并运行：

```bash
# SSH 登录服务器
ssh -p 10522 haosu@10.12.120.164

# 在本地一键上传源码并在服务器构建镜像
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer/tools
./docker/deploy-to-server.sh

# 在服务器上构建镜像
cd ~/urbannav-processor
docker build -f tools/docker/Dockerfile -t urbannav-processor:latest .

# 运行处理
docker run -it --rm \
  -v /path/on/server/dataset:/data/dataset:ro \
  -v /path/on/server/output:/data/output \
  -v /path/on/server/temp:/data/temp \
  urbannav-processor:latest \
  dataset_processing/process_chunks_26_29.py \
  --dataset-root /data/dataset \
  --sidewalk-root /data/output \
  --temp-dir /data/temp \
  --chunks 26 27 28 29 \
  --sample-fps 20.0 \
  --generate-bev \
  --use-pose-ekf
```

### 方法二：保存并加载镜像（如果服务器网络不好）

```bash
# 在本地保存镜像
docker save -o urbannav-processor.tar urbannav-processor:latest

# 上传到服务器
scp -P 10522 urbannav-processor.tar haosu@10.12.120.164:~/

# 在服务器加载镜像
ssh -p 10522 haosu@10.12.120.164
docker load -i ~/urbannav-processor.tar
```

---

## 常用命令参考

### 查看帮助

```bash
docker run --rm urbannav-processor:latest dataset_processing/process_chunks_26_29.py --help
```

### 运行单个脚本

```bash
# 运行 mcap_to_sidewalk_inputs.py
docker run -it --rm \
  -v /your/data:/data \
  urbannav-processor:latest \
  dataset_processing/mcap_to_sidewalk_inputs.py --help

# 运行 process_sidewalk_dataset.py
docker run -it --rm \
  -v /your/data:/data \
  urbannav-processor:latest \
  dataset_processing/process_sidewalk_dataset.py --help
```

### 进入交互式 shell

```bash
docker run -it --rm \
  -v /your/data:/data \
  --entrypoint bash \
  urbannav-processor:latest
```

---

## 目录结构说明

容器内的目录结构：

```
/app/                    # 脚本目录
  └── tools/
      ├── dataset_processing/
      ├── model_processing/
      ├── bev/
      ├── math_tools/
      └── docker/

/data/
  ├── dataset/           # 数据集（只读挂载）
  │     └── chunk_26/
  │         └── mcap/
  ├── output/            # 输出目录
  └── temp/              # 临时文件目录
```

---

## 故障排查

### 权限问题

如果遇到权限问题，可以在运行时指定用户 ID：

```bash
docker run -it --rm \
  -u $(id -u):$(id -g) \
  -v /your/data:/data \
  urbannav-processor:latest \
  ...
```

### 内存不足

如果处理大数据集时内存不足，可以：

1. 减少 `--num-workers`
2. 增加 Docker 内存限制
3. 每次只处理一个 chunk

```bash
# 限制内存使用
docker run -it --rm --memory=16g \
  -v /your/data:/data \
  urbannav-processor:latest \
  dataset_processing/process_chunks_26_29.py \
  --chunks 26 \
  --num-workers 2
```

### ffmpeg 相关问题

镜像中已包含 ffmpeg，如果有问题可以进入容器检查：

```bash
docker run -it --rm --entrypoint bash urbannav-processor:latest
ffmpeg -version
```

---

## 完整示例：部署到 haosu@10.12.120.164

### 步骤 1: 本地准备、上传并构建

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer/tools
./deploy-to-server.sh
```

脚本只使用服务器上的独立目录 `~/urbannav-processor`，上传本地 UrbanNav conda 环境包，并构建 Docker 镜像 `urbannav-processor:latest`；不会修改服务器已有 conda、Python 环境或其他代码目录。

### 步骤 2: 在服务器上运行

```bash
# 登录服务器
ssh -p 10522 haosu@10.12.120.164

# 创建工作目录
cd ~/urbannav-processor
```

如果数据集就在服务器 `/media/ubuntu/data1/SR-Nav Dataset`：

```bash
docker run -it --rm \
  -v "/media/ubuntu/data1/SR-Nav Dataset:/data/dataset:ro" \
  -v "/media/ubuntu/data1/SR-Nav Dataset/sidewalk:/data/output" \
  -v "/tmp/chunk_mcap_temp:/data/temp" \
  urbannav-processor:latest \
  dataset_processing/process_chunks_26_29.py \
  --dataset-root /data/dataset \
  --sidewalk-root /data/output \
  --temp-dir /data/temp \
  --chunks 26 27 28 29 \
  --sample-fps 20.0 \
  --generate-bev \
  --use-pose-ekf
```
