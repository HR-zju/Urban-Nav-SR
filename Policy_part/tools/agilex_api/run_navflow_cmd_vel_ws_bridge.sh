#!/usr/bin/env bash
set -euo pipefail

cd /home/orin26/code/Urban-Nav-SR/Policy_part

# Default is dry-run: subscribe /cmd_vel and print observed commands only.
# Add --publish to actually publish WebSocket /navflow_cmd_vel to AgileX.
python3 tools/ros2_cmd_vel_to_agilex_ws.py \
  --ros2-topic /cmd_vel \
  --ws-topic /navflow_cmd_vel \
  "$@"
