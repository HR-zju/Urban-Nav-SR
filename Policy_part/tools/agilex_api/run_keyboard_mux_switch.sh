#!/usr/bin/env bash
set -euo pipefail

cd /home/orin26/code/Urban-Nav-SR/Policy_part
python3 tools/agilex_api/keyboard_mux_switch.py "$@"
