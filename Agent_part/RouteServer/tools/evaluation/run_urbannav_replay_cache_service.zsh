#!/usr/bin/env zsh
set -euo pipefail

# Manage the shared UrbanNav replay cache service.
#
# Usage:
#   tools/evaluation/run_urbannav_replay_cache_service.zsh start
#   tools/evaluation/run_urbannav_replay_cache_service.zsh status
#   tools/evaluation/run_urbannav_replay_cache_service.zsh stop
#   tools/evaluation/run_urbannav_replay_cache_service.zsh tail

SCRIPT_PATH="${0:A}"
ROUTE_ROOT="${ROUTE_ROOT:-${SCRIPT_PATH:h:h:h}}"
ACTION="${1:-status}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/UrbanNav/bin/python}"
CACHE_HOST="${CACHE_HOST:-127.0.0.1}"
PORT="${PORT:-18180}"
SESSION="${SESSION:-urbannav_replay_cache_${PORT}}"
POI_CACHE="${POI_CACHE:-${ROUTE_ROOT}/data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/poi_cache.jsonl}"
ROUTE_CACHE="${ROUTE_CACHE:-${ROUTE_ROOT}/data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/route_cache.jsonl}"
LOG="${LOG:-${ROUTE_ROOT}/data/benchmark/urbannav_module1_runs/replay_cache_${PORT}.log}"

case "${ACTION}" in
  start)
    mkdir -p "${LOG:h}"
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      print "already running: ${SESSION}"
      exit 0
    fi
    tmux new-session -d -s "${SESSION}" \
      "cd '${ROUTE_ROOT}' && PYTHONUNBUFFERED=1 PYTHONPATH=src '${PYTHON_BIN}' tools/evaluation/serve_urbannav_replay_cache.py --poi-cache '${POI_CACHE}' --route-cache '${ROUTE_CACHE}' --host '${CACHE_HOST}' --port '${PORT}' 2>&1 | tee '${LOG}'"
    print "started ${SESSION}"
    print "url=http://${CACHE_HOST}:${PORT}"
    print "log=${LOG}"
    ;;
  status)
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      print "tmux: running ${SESSION}"
    else
      print "tmux: not running ${SESSION}"
    fi
    curl --noproxy 127.0.0.1,localhost -sS "http://${CACHE_HOST}:${PORT}/healthz" || true
    print
    ;;
  stop)
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      tmux kill-session -t "${SESSION}"
      print "stopped ${SESSION}"
    else
      print "not running: ${SESSION}"
    fi
    ;;
  tail)
    tail -f "${LOG}"
    ;;
  *)
    print -u2 "unknown action: ${ACTION}"
    exit 2
    ;;
esac
