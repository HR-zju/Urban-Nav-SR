# Agent Part 复现说明

`Agent_part` 是 Urban-Nav-SR 的 Agent 侧代码，主要包含在线导航后端、SideWalk 数据构建、Visual Prompt 推理、CoT teacher 标注和 ms-swift 训练数据归档。

这份 README 按“第一次接触项目的人可以逐行复现”的方式组织。除非特别说明，所有命令都在 Ubuntu/Linux 环境执行。

## 目录说明

```text
Agent_part/
├── RouteServer/     # 当前主要代码：MCP/HTTP 后端、地图/LLM provider、SideWalk 数据处理、Visual Prompt、Phase1/Phase2 工具
└── SideWalkNav/     # SideWalk System2 数据、评估口径和轨迹指标说明
```

重点入口：

```text
Agent_part/RouteServer/README.md        # RouteServer 能力和 examples
Agent_part/RouteServer/tools/README.md  # SideWalk 数据处理与模型处理命令速查
Agent_part/RouteServer/docs/README.md   # RouteServer 文档索引和维护状态
Agent_part/RouteServer/docs/FRAMEWORK_MATH_MODEL.md  # 当前框架数学模型和数据/地图接口边界
Agent_part/RouteServer/docs/URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md  # UrbanNav benchmark 技术设计
Agent_part/RouteServer/docs/EVALUATION.md  # 目的地推荐和路线规划评估指标
Agent_part/SideWalkNav/README.md        # Phase1 数据、CoT、baseline 和指标口径
```

## 一、准备代码和基础环境

### 1. 进入仓库

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR
```

如果你的仓库路径不同，后面所有 `/home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR` 都替换成你的实际路径。

### 2. 创建 Python 环境

仓库推荐 Python 3.13，最低要求 Python 3.12。

使用 conda：

```bash
conda create -n UrbanNav python=3.13 -y
conda activate UrbanNav
python --version
```

如果本机没有 Python 3.13，也可以用 3.12：

```bash
conda create -n UrbanNav python=3.12 -y
conda activate UrbanNav
python --version
```

### 3. 安装 RouteServer

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer
python -m pip install --upgrade pip
python -m pip install -e .
```

数据处理和可视化脚本还需要以下常用依赖：

```bash
python -m pip install numpy Pillow tqdm opencv-python
```

安装后验证命令是否可用：

```bash
python -m py_compile src/mcp_server/server.py
osm-mcp-server --help
```

`osm-mcp-server --help` 能打印帮助或 MCP 启动信息即可。历史包名仍叫 `osm-mcp-server`，但当前已经支持 OSM、AMap、Baidu、Google 等 provider。

## 二、配置环境变量

### 1. 地图 provider

默认可用 OSM：

```bash
export MAP_PROVIDER=osm
```

中国大陆在线路线和 POI 推荐使用 AMap：

```bash
export MAP_PROVIDER=amap
export AMAP_KEY="<your-amap-key>"
```

其他 provider：

```bash
export MAP_PROVIDER=baidu
export BAIDU_AK="<your-baidu-ak>"
```

```bash
export MAP_PROVIDER=google
export GOOGLE_MAPS_API_KEY="<your-google-maps-api-key>"
```

### 2. LLM/VLM provider

使用 Volcengine ARK / Doubao：

```bash
export LLM_PROVIDER=ark
export ARK_API_KEY="<your-ark-api-key>"
export LLM_MODEL="doubao-seed-2-0-pro-260215"
export LLM_API_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export LLM_API_STYLE="chat_completions"
```

使用 OpenAI-compatible endpoint：

```bash
export LLM_API_KEY="<your-api-key>"
export LLM_MODEL="gpt-4o-mini"
export LLM_API_BASE_URL="https://api.openai.com/v1"
```

### 3. 可选超时参数

网络较慢时建议加：

```bash
export MAP_HTTP_TOTAL_TIMEOUT_S=60
export MAP_HTTP_CONNECT_TIMEOUT_S=12
export MAP_HTTP_SOCK_READ_TIMEOUT_S=30
export MAP_HTTP_TRUST_ENV=1
export LLM_TIMEOUT_S=30
export LLM_VISION_TIMEOUT_S=60
export LLM_ITINERARY_TIMEOUT_S=60
```

## 三、快速验证在线 RouteServer

### 1. 启动 MCP server

打开一个终端：

```bash
conda activate UrbanNav
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer
export MAP_PROVIDER=osm
osm-mcp-server
```

如果使用 AMap：

```bash
conda activate UrbanNav
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer
export MAP_PROVIDER=amap
export AMAP_KEY="<your-amap-key>"
osm-mcp-server
```

### 2. 运行 examples

另开一个终端：

```bash
conda activate UrbanNav
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer
```

路线和 RouteContext 调试：

```bash
python examples/route_context_debug_client.py \
  --prompt "我想去西湖" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --heading-deg 35 \
  --lookahead-m 20
```

动态途经点：

```bash
python examples/route_session_client.py \
  --origin 120.1180,30.2645 \
  --destination 120.116555,30.270429 \
  --mode foot \
  --add-via 120.118376,30.257515 \
  --add-via 120.12601,30.26017
```

多目标 itinerary：

```bash
python examples/itinerary_client.py \
  --prompt "我今天累了，想先休息，吃顿晚餐，再看个电影" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --render-map
```

FPV 图像理解需要配置 VLM：

```bash
python examples/image_understanding_client.py \
  --image-path /home/ubuntu/WorkSpace/dataset/urban.jpeg \
  --question "Describe the navigable directions and obstacles in this FPV image."
```

Zero-shot Visual Prompt：

```bash
python examples/zero_shot_visual_prompt_agent_client.py \
  --prompt "我想去西湖" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --heading-deg 35 \
  --lookahead-m 20 \
  --image-path /home/ubuntu/WorkSpace/dataset/urban.jpeg \
  --save-overlay /tmp/visual_prompt.svg
```

## 四、SideWalk 数据目录要求

下面流程假设原始数据已经整理为 SideWalk clip 结构。

推荐变量：

```bash
export REPO_ROOT="/home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR"
export ROUTE_SERVER="${REPO_ROOT}/Agent_part/RouteServer"
export DATASET_ROOT="/media/ubuntu/data1/SR-Nav Dataset"
export SIDEWALK_ROOT="${DATASET_ROOT}/sidewalk"
export CHUNK_ROOT="${SIDEWALK_ROOT}/chunk_0"
export CLIP="${CHUNK_ROOT}/<clip_id>"
```

把 `<clip_id>` 替换成真实 clip 目录名。

Phase1 最少需要每个 clip 下有：

```text
front.mp4 或 images/
frame_positions.npy
frame_orientations.npy
camera_params.json
bev_images/
bev_index.json
```

如果还没有 `bev_images/` 和 `bev_index.json`，按下一节生成。

## 五、生成基础输入和 BEV

进入 RouteServer：

```bash
conda activate UrbanNav
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer
```

### 1. 扫描 chunk 状态

```bash
python tools/dataset_processing/scan_chunk.py \
  --sidewalk-root "${CHUNK_ROOT}" \
  --out-csv "${CHUNK_ROOT}/scan_summary.csv"
```

输出：

```text
${CHUNK_ROOT}/scan_summary.csv
```

用于检查每个 clip 是否已有 `front.mp4`、pose、GPS、BEV、Phase1 产物等。

### 2. 单个 clip 生成 Phase1 BEV

```bash
python tools/dataset_processing/process_sidewalk_dataset.py \
  --sidewalk-root "${CLIP}" \
  --bev-sample-fps 2.0 \
  --map-provider osm \
  --use-pose-ekf
```

主要输出会写回 `${CLIP}`：

```text
trajectory.json
bev_sample_trajectory.json
samples.json
bev_images/
bev_index.json
```

说明：

- `trajectory.json` 是全量 FPV 每帧轨迹。
- `bev_images/` 是默认 2fps ego-aligned BEV。
- `bev_index.json` 记录 BEV 帧和原始 FPV timestamp 的映射。
- `--use-pose-ekf` 只用于 heading 平滑；默认不要加 `--use-ekf-position`。

### 3. 批量处理 chunk

```bash
python tools/dataset_processing/process_chunks_26_29.py \
  --dataset-root "${DATASET_ROOT}" \
  --sidewalk-root "${SIDEWALK_ROOT}" \
  --chunks 0 \
  --sample-fps 20.0 \
  --generate-bev \
  --use-pose-ekf \
  --num-workers 4
```

如果用 `--skip-existing`，旧产物可能被跳过。需要重生 BEV 时，先删除对应 clip 下的 `bev_images/` 和 `bev_index.json`，或不要加 `--skip-existing`。

## 六、生成 Phase1 source 数据

Phase1 不依赖 SAM/SoM。它会把 FPV、BEV、pose/camera 标定投影和 QA bundle 转成训练前的 source JSONL。

单个 clip 推荐先小规模验证：

```bash
conda activate UrbanNav
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

SIDEWALK_ROOT="${CLIP}" \
OUTPUT_ROOT="${CLIP}/phase1" \
SPLIT_NAME=clip_test \
FRAME_STEP=10 \
WRITE_OVERLAYS=1 \
./tools/model_processing/run_sidewalk_phase1_dataset.sh
```

主要输出：

```text
${CLIP}/phase1/manifests/phase1_source_clip_test.jsonl
${CLIP}/phase1/manifests/phase1_eval_clip_test.json
${CLIP}/phase1/clips/<clip_id>/trajectory_labels/*.json
${CLIP}/phase1/clips/<clip_id>/bev_patch_meta/*.json
${CLIP}/phase1/clips/<clip_id>/qa/*_qa.json
${CLIP}/phase1/clips/<clip_id>/overlays/*.jpg
```

全 chunk 示例：

```bash
SIDEWALK_ROOT="${CHUNK_ROOT}" \
OUTPUT_ROOT="${DATASET_ROOT}/sidewalk_phase1" \
SPLIT_NAME=train \
FRAME_STEP=10 \
WRITE_OVERLAYS=0 \
./tools/model_processing/run_sidewalk_phase1_dataset.sh
```

只处理前三个 clip：

```bash
SIDEWALK_ROOT="${CHUNK_ROOT}" \
OUTPUT_ROOT="${DATASET_ROOT}/sidewalk_phase1_3clips" \
SPLIT_NAME=train_3clips \
MAX_CLIPS=3 \
FRAME_STEP=10 \
WRITE_OVERLAYS=1 \
./tools/model_processing/run_sidewalk_phase1_dataset.sh
```

常用参数说明：

```text
FRAME_STEP=10                 # 20Hz 输入约等于 2fps 采样；每帧样本用 FRAME_STEP=1
MAX_CLIPS=3                   # 最多处理几个 clip
LIMIT=200                     # 最多写多少条样本，不等于 clip 数
WRITE_OVERLAYS=1              # 写 GT 轨迹可视化，正式大规模构建可设为 0
MAX_BEV_TIME_DELTA_MS=300.0   # FPV 帧匹配最近 BEV 的最大时间差
ALLOW_HEURISTIC_TRAJECTORY=1  # 只用于旧 GNSS/trajectory smoke fallback
```

## 七、生成 CoT teacher 标注并归档为 ms-swift 数据

这一步需要可用的 LLM/VLM provider。正式构建推荐 `FALLBACK_POLICY=drop`，失败样本不进入训练集。

单个 clip：

```bash
conda activate UrbanNav
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

LLM_PROVIDER=ark \
ARK_API_KEY="<your-ark-api-key>" \
LLM_MODEL="doubao-seed-2-0-pro-260215" \
OUTPUT_ROOT="${CLIP}/phase1" \
SPLIT_NAME=clip_test \
NUM_POINTS=16 \
FALLBACK_POLICY=drop \
ARCHIVE_AFTER_COT=1 \
MEDIA_MODE=hardlink \
PATH_MODE=absolute \
WRITE_REVIEW_HTML=1 \
./tools/model_processing/run_sidewalk_phase1_cot_teacher.sh
```

主要输出：

```text
${CLIP}/phase1/manifests/phase1_cot_clip_test.jsonl
${CLIP}/phase1/manifests/phase1_cot_report_clip_test.json
${CLIP}/phase1_archive/manifests/clip_test.jsonl
${CLIP}/phase1_archive/manifests/clip_test.errors.jsonl
${CLIP}/phase1_archive/manifests/clip_test.review.csv
${CLIP}/phase1_archive/manifests/clip_test.stats.json
${CLIP}/phase1_archive/review/index.html
```

如果已经有 CoT JSONL，只想重新 archive：

```bash
OUTPUT_ROOT="${CLIP}/phase1" \
ARCHIVE_ROOT="${CLIP}/phase1_archive" \
SPLIT_NAME=clip_test \
MEDIA_MODE=hardlink \
PATH_MODE=absolute \
WRITE_REVIEW_HTML=1 \
./tools/model_processing/run_sidewalk_phase1_archive.sh
```

全 chunk 示例：

```bash
LLM_PROVIDER=ark \
ARK_API_KEY="<your-ark-api-key>" \
LLM_MODEL="doubao-seed-2-0-pro-260215" \
OUTPUT_ROOT="${DATASET_ROOT}/sidewalk_phase1" \
SPLIT_NAME=train \
NUM_POINTS=16 \
FALLBACK_POLICY=drop \
ARCHIVE_AFTER_COT=1 \
ARCHIVE_ROOT="${DATASET_ROOT}/sidewalk_phase1_archive" \
MEDIA_MODE=hardlink \
PATH_MODE=absolute \
WRITE_REVIEW_HTML=1 \
./tools/model_processing/run_sidewalk_phase1_cot_teacher.sh
```

归档后的 ms-swift 训练入口是：

```text
${ARCHIVE_ROOT}/manifests/${SPLIT_NAME}.jsonl
```

## 八、使用 ms-swift 训练

ms-swift 不在本目录内，假设已经放在：

```text
/home/ubuntu/WorkSpace/project/SR/Code/ms-swift
```

进入 ms-swift：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/ms-swift
```

单 clip smoke train：

```bash
swift sft \
  --model Qwen/Qwen3.5-4B \
  --tuner_type lora \
  --dataset "${CLIP}/phase1_archive/manifests/clip_test.jsonl" \
  --remove_unused_columns false \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --max_length 2048 \
  --output_dir output/sidewalk_phase1_cot_smoke
```

全量训练时把 `--dataset` 改成：

```text
${DATASET_ROOT}/sidewalk_phase1_archive/manifests/train.jsonl
```

`--remove_unused_columns false` 必须保留，否则 `pixel_trajectory` 等额外字段可能被 trainer 丢掉。

## 九、运行 Agent CoT baseline

baseline 用同样的 Phase1 CoT v5 输入格式，让模型真实预测 `<finish>.trajectory`，再转成 overlay 和指标。不要用规则 fallback 补轨迹。

推荐读取 Phase1 source：

```bash
conda activate UrbanNav
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

LLM_PROVIDER=ark \
ARK_API_KEY="<your-ark-api-key>" \
LLM_MODEL="doubao-seed-2-0-pro-260215" \
python tools/model_processing/run_sidewalk_phase1_cot_baseline.py \
  --input-mode phase1_source \
  --source-jsonl "${CLIP}/phase1/manifests/phase1_source_clip_test.jsonl" \
  --output-dir "${CLIP}/agent_cot_baseline_clip_test" \
  --frame-start 130 \
  --frame-end 150 \
  --frame-step 10
```

输出：

```text
${CLIP}/agent_cot_baseline_clip_test/json/*.json
${CLIP}/agent_cot_baseline_clip_test/overlay/*.svg
${CLIP}/agent_cot_baseline_clip_test/manifest.json
```

每帧 JSON 里重点看：

```text
response
prediction.trajectory_norm
prediction.rendered_prompt.path_uv
metrics.normalized_ade
metrics.normalized_fde
trajectory_warnings
```

## 十、运行 SideWalk Visual Prompt pipeline

如果你已经有 `images/`、`trajectory.json`，并希望逐帧输出 visual prompt overlay/json：

```bash
conda activate UrbanNav
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

WORK_DIR="${CLIP}"
OUT_DIR="${CLIP}/visual_prompt_output"
SAM3_DIR="${CLIP}/sam3"

python tools/model_processing/run_sidewalk_images_visual_prompt_pipeline.py \
  --images-dir "${WORK_DIR}/images" \
  --trajectory-json "${WORK_DIR}/trajectory.json" \
  --output-dir "${OUT_DIR}" \
  --sequence-name "$(basename "${CLIP}")" \
  --camera-view camera_front \
  --sam3-output-base-dir "${SAM3_DIR}" \
  --frame-start 50 \
  --frame-end 52 \
  --frame-step 1 \
  --enable-segmentation-overlay-input \
  --mode foot
```

输出：

```text
${OUT_DIR}/overlay/{frame}.svg
${OUT_DIR}/json/{frame}.json
${OUT_DIR}/logs/{frame}.log
${OUT_DIR}/manifest.json
```

如果需要同时运行 SAM3，先确保有单独的 `sam3` conda 环境、SAM3 代码和 checkpoint，然后加：

```bash
  --run-sam3 \
  --sam3-env-name sam3 \
  --sam3-inference-script /home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/batch_sam3_demo_260124.py \
  --checkpoint-path /media/ubuntu/data1/checkpoints/checkpoint_35_merged.pt \
  --gpus 0 \
  --min-score 0.3 \
  --enforce-alignment
```

完整命令：

```bash
python tools/model_processing/run_sidewalk_images_visual_prompt_pipeline.py \
  --images-dir "${WORK_DIR}/images" \
  --trajectory-json "${WORK_DIR}/trajectory.json" \
  --output-dir "${OUT_DIR}" \
  --sequence-name "$(basename "${CLIP}")" \
  --camera-view camera_front \
  --sam3-output-base-dir "${SAM3_DIR}" \
  --run-sam3 \
  --sam3-env-name sam3 \
  --sam3-inference-script /home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/batch_sam3_demo_260124.py \
  --checkpoint-path /media/ubuntu/data1/checkpoints/checkpoint_35_merged.pt \
  --gpus 0 \
  --min-score 0.3 \
  --frame-start 50 \
  --frame-end 52 \
  --frame-step 1 \
  --enable-segmentation-overlay-input \
  --enforce-alignment \
  --mode foot
```

## 十一、可选 Phase2 SAM/SoM

Phase1 数据、CoT teacher、archive、baseline 都不强制依赖 SAM/SoM。只有需要 dense grounding、segmentation、SoM 或 Phase2 artifact 时再跑。

单个 chunk：

```bash
conda activate UrbanNav
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer/tools/model_processing
./run_phase_chunk0.sh
```

全量：

```bash
conda activate UrbanNav
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer/tools/model_processing
./run_phase_all.sh
```

运行前必须检查脚本里的：

```text
CHECKPOINT_PATH
SIDEWALK_ROOT 或 SIDEWALK_DIR
```

确保路径指向真实数据和真实 checkpoint。

## 十二、Docker 可选路径

Docker 文件在：

```text
Agent_part/RouteServer/tools/docker/
```

构建镜像：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer
tools/docker/build-urbannav-conda-image.sh
```

容器内批量处理示例：

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

更多说明见：

```text
Agent_part/RouteServer/tools/docker/README-DOCKER.md
```

## 十三、复现检查清单

按顺序确认：

```bash
conda activate UrbanNav
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer
python --version
python -m py_compile src/mcp_server/server.py examples/*.py
python tools/dataset_processing/scan_chunk.py --sidewalk-root "${CHUNK_ROOT}" --out-csv /tmp/scan_summary.csv
```

单 clip Phase1 成功后应存在：

```text
${CLIP}/phase1/manifests/phase1_source_clip_test.jsonl
${CLIP}/phase1/manifests/phase1_eval_clip_test.json
```

CoT teacher 成功后应存在：

```text
${CLIP}/phase1/manifests/phase1_cot_clip_test.jsonl
${CLIP}/phase1/manifests/phase1_cot_report_clip_test.json
```

archive 成功后应存在：

```text
${CLIP}/phase1_archive/manifests/clip_test.jsonl
${CLIP}/phase1_archive/manifests/clip_test.stats.json
```

baseline 成功后应存在：

```text
${CLIP}/agent_cot_baseline_clip_test/manifest.json
${CLIP}/agent_cot_baseline_clip_test/json/
${CLIP}/agent_cot_baseline_clip_test/overlay/
```

## 常见问题

### 1. `ModuleNotFoundError: No module named mcp_server`

先安装本地包：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer
python -m pip install -e .
```

### 2. `ModuleNotFoundError: numpy/PIL/cv2/tqdm`

补装数据处理依赖：

```bash
python -m pip install numpy Pillow tqdm opencv-python
```

### 3. Phase1 找不到 BEV

先给 clip 生成 BEV：

```bash
python tools/dataset_processing/process_sidewalk_dataset.py \
  --sidewalk-root "${CLIP}" \
  --bev-sample-fps 2.0 \
  --map-provider osm \
  --use-pose-ekf
```

### 4. `FRAME_STEP` 怎么选

如果输入是 20Hz 全量 `trajectory.json`，`FRAME_STEP=10` 约等于 2fps。调试用小范围和 `WRITE_OVERLAYS=1`；正式构建可用 `FRAME_STEP=10` 或按数据规模调整。若 `trajectory.json` 已经是低频，避免二次降采样，设 `FRAME_STEP=1`。

### 5. CoT teacher 输出为空或大量失败

检查：

```text
LLM_PROVIDER
ARK_API_KEY 或 LLM_API_KEY
LLM_MODEL
SOURCE_JSONL 是否存在
FPV/BEV 图片路径是否存在
```

正式构建保持 `FALLBACK_POLICY=drop`。`FALLBACK_POLICY=template` 只适合 smoke test，不适合高质量 CoT 训练数据。

### 6. SAM3 不是 Phase1 必需项

Phase1 source、CoT teacher、archive、baseline 都可以不跑 SAM3。只有 Visual Prompt segmentation overlay 或 Phase2 dense grounding 需要 SAM3 环境和 checkpoint。
