#!/bin/bash
# EXP-120 greedy launcher for lab-lry GPU 1
# 9 runs (3 variants x 3 seeds) launch to GPU 1 dynamically by memory
cd /home/lry/code/federated-learning/FDSE_CVPR25
PY=/home/lry/conda/envs/pfllib/bin/python
EXP=../experiments/ablation/EXP-120_uc1_decomp
LOG=$EXP/logs
mkdir -p $LOG $EXP/results

TASKS=(
    "only_s2|feddsa_sgpa_only_r200.yml|2"
    "only_s15|feddsa_sgpa_only_r200.yml|15"
    "only_s333|feddsa_sgpa_only_r200.yml|333"
    "w_s2|feddsa_sgpa_w_r200.yml|2"
    "w_s15|feddsa_sgpa_w_r200.yml|15"
    "w_s333|feddsa_sgpa_w_r200.yml|333"
    "c_s2|feddsa_sgpa_c_r200.yml|2"
    "c_s15|feddsa_sgpa_c_r200.yml|15"
    "c_s333|feddsa_sgpa_c_r200.yml|333"
)

MIN_FREE_MB=2800
echo "[$(date +'%F %H:%M:%S')] EXP-120 greedy launcher start, MIN_FREE=${MIN_FREE_MB}MB"

for task in "${TASKS[@]}"; do
    IFS="|" read -r label config seed <<< "$task"
    while true; do
        free_mb=$(/usr/bin/nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
        if [ "$free_mb" -ge $MIN_FREE_MB ]; then
            echo "[$(date +'%H:%M:%S')] LAUNCH $label config=$config seed=$seed (free=${free_mb}MB)"
            CUDA_VISIBLE_DEVICES=1 $PY run_single.py \
                --task domainnet_c6 --algorithm feddsa_sgpa --gpu 0 \
                --config ./config/domainnet/$config --logger PerRunLogger --seed $seed \
                > $LOG/${label}.log 2>&1 &
            sleep 20
            break
        fi
        sleep 45
    done
done
echo "[$(date +'%F %H:%M:%S')] all 9 runs launched, waiting..."
wait
echo "[$(date +'%F %H:%M:%S')] ALL DONE"
