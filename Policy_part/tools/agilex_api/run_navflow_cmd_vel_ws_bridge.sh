#!/usr/bin/env bash
set -euo pipefail

cd /home/orin26/code/Urban-Nav-SR/Policy_part

python3 tools/agilex_api/ros2_cmd_vel_to_agilex_ws.py "$@"
