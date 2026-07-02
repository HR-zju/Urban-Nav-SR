#!/usr/bin/env zsh
set -euo pipefail

ROOT="${URBANNAV_ROUTESERVER_ROOT:-/home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer}"
cd "$ROOT"

LOG_DIR="${URBANNAV_OSRM_LOG_DIR:-data/osm/osrm/logs}"
mkdir -p "$LOG_DIR"

echo "[osrm-local] root=$ROOT"
echo "[osrm-local] log_dir=$LOG_DIR"
date

echo "[osrm-local] stopping existing containers"
bash data/osm/osrm/stop_osrm_all_docker.sh 2>&1 | tee "$LOG_DIR/00_stop.log"

if [[ -x data/osm/osrm/build_osmium_tool_image_docker.sh ]]; then
  echo "[osrm-local] building osmium-tool image"
  bash data/osm/osrm/build_osmium_tool_image_docker.sh 2>&1 | tee "$LOG_DIR/00_build_osmium_image.log"
fi

echo "[osrm-local] extracting city PBF files"
if [[ -x data/osm/osrm/extract_city_pbf_all_osmium_docker.sh ]]; then
  bash data/osm/osrm/extract_city_pbf_all_osmium_docker.sh 2>&1 | tee "$LOG_DIR/01_extract.log"
else
  bash data/osm/osrm/extract_city_pbf_all_docker.sh 2>&1 | tee "$LOG_DIR/01_extract.log"
fi

echo "[osrm-local] building OSRM graphs"
bash data/osm/osrm/build_osrm_all_docker.sh 2>&1 | tee "$LOG_DIR/02_build.log"

echo "[osrm-local] starting OSRM services"
bash data/osm/osrm/run_osrm_all_docker.sh 2>&1 | tee "$LOG_DIR/03_run.log"

echo "[osrm-local] completed"
date
