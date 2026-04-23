#!/bin/bash
# EXP-120 Office-Caltech10 分解消融: 3 variants × 3 seeds = 9 runs
# Office E=1 显存小 ~1.5-2GB/run，greedy 等剩余显存
cd /home/lry/code/federated-learning/FDSE_CVPR25
PY=/home/lry/conda/envs/pfllib/bin/python
EXP=../experiments/ablation/EXP-120_uc1_decomp
LOG=$EXP/logs_office
mkdir -p $LOG

TASKS=(
    "office_only_s2|feddsa_sgpa_only_office_r200.yml|2"
    "office_only_s15|feddsa_sgpa_only_office_r200.yml|15"
    "office_only_s333|feddsa_sgpa_only_office_r200.yml|333"
    "office_w_s2|feddsa_sgpa_w_office_r200.yml|2"
    "office_w_s15|feddsa_sgpa_w_office_r200.yml|15"
    "office_w_s333|feddsa_sgpa_w_office_r200.yml|333"
    "office_c_s2|feddsa_sgpa_c_office_r200.yml|2"
    "office_c_s15|feddsa_sgpa_c_office_r200.yml|15"
    "office_c_s333|feddsa_sgpa_c_office_r200.yml|333"
)

MIN_FREE_MB=2200
echo "[$(date +'%F %H:%M:%S')] EXP-120 Office launcher start, MIN_FREE=${MIN_FREE_MB}MB"

for task in "${TASKS[@]}"; do
    IFS="|" read -r label config seed <<< "$task"
    while true; do
        free_mb=$(/usr/bin/nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
        if [ "$free_mb" -ge $MIN_FREE_MB ]; then
            echo "[$(date +'%H:%M:%S')] LAUNCH $label config=$config seed=$seed (free=${free_mb}MB)"
            CUDA_VISIBLE_DEVICES=1 $PY run_single.py \
                --task office_caltech10_c4 --algorithm feddsa_sgpa --gpu 0 \
                --config ./config/office/$config --logger PerRunLogger --seed $seed \
                > $LOG/${label}.log 2>&1 &
            sleep 20
            break
        fi
        sleep 45
    done
done
echo "[$(date +'%F %H:%M:%S')] 9 Office runs launched, waiting..."
wait
echo "[$(date +'%F %H:%M:%S')] OFFICE DONE"
