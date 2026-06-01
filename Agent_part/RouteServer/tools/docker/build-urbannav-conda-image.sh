#!/usr/bin/env bash
# Pack the local UrbanNav conda environment and build the processor image.

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/home/ubuntu/anaconda3/envs/UrbanNav}"
IMAGE_NAME="${IMAGE_NAME:-urbannav-processor}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUTE_SERVER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_TARBALL="$SCRIPT_DIR/UrbanNav-env.tar.gz"

if [[ ! -d "$ENV_PREFIX" ]]; then
    echo "Conda env not found: $ENV_PREFIX" >&2
    exit 1
fi

echo "Packing conda env:"
echo "  source: $ENV_PREFIX"
echo "  output: $ENV_TARBALL"
tar -czf "$ENV_TARBALL" \
    -C "$ENV_PREFIX" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='conda-meta/history' \
    .

echo "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
docker build -f "$SCRIPT_DIR/Dockerfile" -t "$IMAGE_NAME:$IMAGE_TAG" "$ROUTE_SERVER_DIR"

echo "Verifying image..."
docker run --rm "$IMAGE_NAME:$IMAGE_TAG" dataset_processing/process_chunks_26_29.py --help >/dev/null
docker run --rm "$IMAGE_NAME:$IMAGE_TAG" -c "import sys, numpy, cv2, lz4, aiohttp, openai; print(sys.executable); print('UrbanNav image OK')"
