ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

LABEL maintainer="UrbanNav Team"
LABEL description="Docker image with packed local UrbanNav conda environment"

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

ARG INSTALL_SYSTEM_DEPS=1
RUN if [ "$INSTALL_SYSTEM_DEPS" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            ffmpeg \
            git \
            curl \
            ca-certificates \
            build-essential \
            pkg-config \
            libgl1 \
            libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*; \
    else \
        echo "Skipping system dependency install; using base image contents."; \
    fi

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CONDA_PREFIX=/home/ubuntu/anaconda3/envs/UrbanNav \
    PATH=/home/ubuntu/anaconda3/envs/UrbanNav/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN mkdir -p /home/ubuntu/anaconda3/envs/UrbanNav

# Build context is Agent_part/RouteServer. This tarball is produced from the
# local UrbanNav conda env by tools/docker/build-urbannav-conda-image.sh.
COPY tools/docker/UrbanNav-env.tar.gz /tmp/UrbanNav-env.tar.gz
RUN tar -xzf /tmp/UrbanNav-env.tar.gz -C /home/ubuntu/anaconda3/envs/UrbanNav \
    && rm /tmp/UrbanNav-env.tar.gz

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY tools/docker/requirements.txt /app/tools/docker/requirements.txt
COPY tools /app/tools

RUN python -m pip install --no-deps -e /app

RUN find /app/tools -type f \( -name "*.py" -o -name "*.sh" \) -exec chmod +x {} \; \
    && mkdir -p /data/dataset /data/output /data/temp

WORKDIR /app/tools

ENTRYPOINT ["python"]
CMD ["--help"]
