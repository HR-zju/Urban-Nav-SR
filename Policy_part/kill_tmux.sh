#!/bin/zsh
tmux kill-session -t foundation_policy_record
tmux kill-session -t run_topomap
echo "Session 'foundation_policy_record' has been terminated."
echo "Session 'run_topomap' has been terminated."
# tmux kill-session -t coco_bridge
# tmux kill-session -t rosbridge