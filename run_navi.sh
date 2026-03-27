# !/bin/bash

Check if an argument was provided
if [ $# -eq 0 ]; then
   echo "Error: Please provide a model name (e.g., citywalker)"
   exit 1
fi

MODEL=$1
# Second argument: bridge mode (local or global), default to local
BRIDGE_MODE=${2:-local}

# 环境变量
source /home/mayukai/miniconda3/etc/profile.d/conda.sh
source /opt/ros/humble/setup.bash
bash kill_tmux.sh
bash run_ros_pub.sh go2 $BRIDGE_MODE
bash run_ros_pub.sh coco $BRIDGE_MODE
ulimit -c 0



# 根据模型名称选择不同的启动命令
if [ "$MODEL" == "citywalker1" ]; then
    echo "Starting CityWalker model..."
    cd CityWalker/
    conda activate citywalker
    python test_ros2.py --config config/finetune.yaml --checkpoint ckpt/finetune.ckpt
elif [ "$MODEL" == "nomad1" ] || [ "$MODEL" == "vint1" ] || [ "$MODEL" == "gnm1" ]; then
    echo "Starting {$MODEL} model..."
    cd visualnav_ros/deployment/src
    conda activate vint_deployment
    python navigate_ros2.py --model "$MODEL" 

elif [ "$MODEL" == "corl" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate bc
    python test_ros2.py --config e2enav/configs/real_sim_all_data_efficientnet.yaml 
elif [ "$MODEL" == "corl2" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate bc
    python test_ros2_finetune.py --config e2enav/configs/real_sim_all_data_efficientnet.yaml 
elif [ "$MODEL" == "pg" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_pg.py --config e2enav/configs/real_sim_all_data_efficientnet.yaml 
elif [ "$MODEL" == "cousin" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_cousin.py --config e2enav/configs/real_sim_all_data_efficientnet.yaml
elif [ "$MODEL" == "cousin_customized" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_cousin_customized.py --config e2enav/configs/real_sim_all_data_efficientnet.yaml
elif [ "$MODEL" == "cousin_full" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_cousin_full.py --config e2enav/configs/real_sim_all_data_efficientnet.yaml
elif [ "$MODEL" == "s2e_cousin" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_s2e_cousin.py
elif [ "$MODEL" == "s2e_v2" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_s2e_v2_data_aug.py
elif [ "$MODEL" == "s2e_v20" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_s2e_v2_noaug.py
elif [ "$MODEL" == "s2e_v22" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_s2e_v2_data_aug2.py
elif [ "$MODEL" == "s2e_v3" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_s2e_icra.py
elif [ "$MODEL" == "s2e_v4" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_s2e_iclr.py
elif [ "$MODEL" == "s2e_v5" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_s2e_iclr_2.py
elif [ "$MODEL" == "s2e_v2c" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_s2e_citywalker.py
elif [ "$MODEL" == "s2e_350_v1" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_350.py
elif [ "$MODEL" == "s2e_350_v2" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_350_v2.py
elif [ "$MODEL" == "s2e_350_myk" ]; then
    echo "Starting {$MODEL} model..."
    cd aura/
    conda activate cousin
    python test_ros2_350_myk.py

elif [ "$MODEL" == "nav_flow_il" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_nav_flow_il_go2.py
elif [ "$MODEL" == "nav_flow_il_goal" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_nav_flow_il_goal.py

elif [ "$MODEL" == "nav_flow_pref" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_nav_flow_pref.py
elif [ "$MODEL" == "nav_flow_pref_go2" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_nav_flow_pref_go2.py
elif [ "$MODEL" == "nav_flow_pref_goal" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_nav_flow_pref_goal.py

elif [ "$MODEL" == "gnm" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_gnm.py
elif [ "$MODEL" == "gnm_goal" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_gnm_goal.py

elif [ "$MODEL" == "vint" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_vint.py
elif [ "$MODEL" == "vint_goal" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_vint_goal.py

elif [ "$MODEL" == "nomad" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_nomad.py
elif [ "$MODEL" == "nomad_goal" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_nomad_goal.py

elif [ "$MODEL" == "citywalker" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_citywalker.py
elif [ "$MODEL" == "citywalker_goal" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_citywalker_goal.py

elif [ "$MODEL" == "s2e_350_myk2" ]; then
    echo "Starting {$MODEL} model..."
    cd aura/
    conda activate cousin
    python test_ros2_350_myk2.py

elif [ "$MODEL" == "brad" ]; then
    echo "Starting {$MODEL} model..."
    cd Ours_backbone/
    conda activate cousin
    python test_ros2_brad.py
elif [ "$MODEL" == "aura" ]; then
    echo "Starting {$MODEL} model..."
    cd aura/
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    export PYTHONPATH=/home/mayukai/code/aura/InternVL/internvl_chat:$PYTHONPATH
    conda activate bc
    python test_ros2_aura.py 
elif [ "$MODEL" == "aura_5s" ]; then
    echo "Starting {$MODEL} model..."
    cd aura/
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    export PYTHONPATH=/home/mayukai/code/aura/InternVL/internvl_chat:$PYTHONPATH
    conda activate bc
    python test_ros2_aura_5s.py 
else
    echo "Error: Unknown model '$MODEL'"
    exit 1
fi
