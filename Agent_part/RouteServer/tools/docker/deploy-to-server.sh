#!/usr/bin/env bash
# Build an isolated UrbanNav processor Docker image on the target server.

set -euo pipefail

SERVER_HOST="${SERVER_HOST:-10.12.120.164}"
SERVER_PORT="${SERVER_PORT:-10522}"
SERVER_USER="${SERVER_USER:-haosu}"
SERVER_PASS="${SERVER_PASS:- }"
SERVER_DIR="${SERVER_DIR:-~/urbannav-processor}"
IMAGE_NAME="${IMAGE_NAME:-urbannav-processor}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENV_PREFIX="${ENV_PREFIX:-/home/ubuntu/anaconda3/envs/UrbanNav}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUTE_SERVER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUNDLE_PATH="/tmp/urbannav-route-server-src.tar.gz"
ENV_TARBALL="$SCRIPT_DIR/UrbanNav-env.tar.gz"

if command -v sshpass >/dev/null 2>&1; then
    SSH_BASE=(
        sshpass -p "$SERVER_PASS"
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
        -p "$SERVER_PORT"
        "$SERVER_USER@$SERVER_HOST"
    )
    SCP_BASE=(
        sshpass -p "$SERVER_PASS"
        scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
        -P "$SERVER_PORT"
    )
else
    echo -e "${YELLOW}sshpass not found; falling back to interactive SSH/SCP password prompts.${NC}"
    SSH_BASE=(
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
        -p "$SERVER_PORT"
        "$SERVER_USER@$SERVER_HOST"
    )
    SCP_BASE=(
        scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
        -P "$SERVER_PORT"
    )
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  UrbanNav Docker deploy${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Target: $SERVER_USER@$SERVER_HOST:$SERVER_PORT"
echo "Remote dir: $SERVER_DIR"
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "Conda env: $ENV_PREFIX"

echo -e "\n${GREEN}[1/5] Checking server access...${NC}"
"${SSH_BASE[@]}" "whoami && docker --version"

echo -e "\n${GREEN}[2/5] Creating source bundle...${NC}"
tar -czf "$BUNDLE_PATH" \
    -C "$ROUTE_SERVER_DIR" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='tools/output' \
    --exclude='tools/temp' \
    --exclude='tools/**/*.tar' \
    --exclude='tools/**/*.tar.gz' \
    --exclude='tools/**/urbannav-processor.tar' \
    pyproject.toml README.md .dockerignore src tools
echo "Bundle: $BUNDLE_PATH"

echo -e "\n${GREEN}[3/5] Packing local UrbanNav conda env...${NC}"
if [[ ! -d "$ENV_PREFIX" ]]; then
    echo -e "${RED}Conda env not found: $ENV_PREFIX${NC}"
    exit 1
fi
tar -czf "$ENV_TARBALL" \
    -C "$ENV_PREFIX" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='conda-meta/history' \
    .
echo "Env bundle: $ENV_TARBALL"

echo -e "\n${GREEN}[4/5] Uploading bundles...${NC}"
"${SSH_BASE[@]}" "mkdir -p $SERVER_DIR"
"${SCP_BASE[@]}" "$BUNDLE_PATH" "$SERVER_USER@$SERVER_HOST:$SERVER_DIR/source.tar.gz"
"${SCP_BASE[@]}" "$ENV_TARBALL" "$SERVER_USER@$SERVER_HOST:$SERVER_DIR/UrbanNav-env.tar.gz"

echo -e "\n${GREEN}[5/5] Extracting and building image on server...${NC}"
"${SSH_BASE[@]}" "cd $SERVER_DIR && tar -xzf source.tar.gz && mkdir -p tools/docker && mv -f UrbanNav-env.tar.gz tools/docker/UrbanNav-env.tar.gz && if docker image inspect $IMAGE_NAME:$IMAGE_TAG >/dev/null 2>&1; then docker tag $IMAGE_NAME:$IMAGE_TAG $IMAGE_NAME:previous-base && docker build --build-arg BASE_IMAGE=$IMAGE_NAME:previous-base --build-arg INSTALL_SYSTEM_DEPS=0 -f tools/docker/Dockerfile -t $IMAGE_NAME:$IMAGE_TAG .; else docker build -f tools/docker/Dockerfile -t $IMAGE_NAME:$IMAGE_TAG .; fi"

echo -e "\n${GREEN}Verifying image...${NC}"
"${SSH_BASE[@]}" "docker run --rm $IMAGE_NAME:$IMAGE_TAG dataset_processing/process_chunks_26_29.py --help | head -40"
"${SSH_BASE[@]}" "docker run --rm $IMAGE_NAME:$IMAGE_TAG -c 'import sys, numpy, cv2, lz4, aiohttp, openai; print(sys.executable); print(\"UrbanNav image OK\")'"

cat <<EOF

${GREEN}Deploy complete.${NC}

Run on the server:

  cd $SERVER_DIR
  docker run -it --rm \\
    -v "/media/ubuntu/data1/SR-Nav Dataset:/data/dataset:ro" \\
    -v "/media/ubuntu/data1/SR-Nav Dataset/sidewalk:/data/output" \\
    -v "/tmp/chunk_mcap_temp:/data/temp" \\
    $IMAGE_NAME:$IMAGE_TAG \\
    dataset_processing/process_chunks_26_29.py \\
    --dataset-root /data/dataset \\
    --sidewalk-root /data/output \\
    --temp-dir /data/temp \\
    --chunks 26 27 28 29 \\
    --sample-fps 20.0 \\
    --generate-bev \\
    --use-pose-ekf
EOF
