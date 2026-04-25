#!/bin/bash
# EXP-127 BiProto 完整 R200 pipeline (绕过 S0 gate kill verdict, B 选项)
# 6 runs: Office × 3 seeds + PACS × 3 seeds (全程 from-scratch)
# lab-lry GPU 1, greedy memory dispatch (CLAUDE.md 17.8 规范)
set -e

PY=/home/lry/conda/envs/pfllib/bin/python
EXP_DIR=/home/lry/code/federated-learning/experiments/ablation/EXP-127_biproto_full_r200
mkdir -p $EXP_DIR/logs $EXP_DIR/results
cd /home/lry/code/federated-learning/FDSE_CVPR25

MIN_FREE_MB=4500
GPU=1

declare -a TASKS=(
    "office_s2|office_caltech10_c4|./config/office/feddsa_biproto_office_r200.yml|2"
    "office_s15|office_caltech10_c4|./config/office/feddsa_biproto_office_r200.yml|15"
    "office_s333|office_caltech10_c4|./config/office/feddsa_biproto_office_r200.yml|333"
    "pacs_s2|PACS_c4|./config/pacs/feddsa_biproto_pacs_r200.yml|2"
    "pacs_s15|PACS_c4|./config/pacs/feddsa_biproto_pacs_r200.yml|15"
    "pacs_s333|PACS_c4|./config/pacs/feddsa_biproto_pacs_r200.yml|333"
)

for task in "${TASKS[@]}"; do
    IFS="|" read -r label dataset config seed <<< "$task"
    while true; do
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk -v g=$GPU 'NR==g+1{print $1}')
        if [ "$free_mb" -ge $MIN_FREE_MB ]; then
            echo "[$(date +%H:%M:%S)] LAUNCH $label (GPU $GPU free=${free_mb}MB) seed=$seed"
            CUDA_VISIBLE_DEVICES=$GPU nohup $PY run_single.py \
                --task $dataset --algorithm feddsa_biproto --gpu 0 \
                --config $config --logger PerRunLogger --seed $seed \
                > $EXP_DIR/logs/${label}.log 2>&1 &
            echo "  PID=$!"
            sleep 25
            break
        fi
        echo "[$(date +%H:%M:%S)] WAIT $label (GPU $GPU free=${free_mb}MB < $MIN_FREE_MB)"
        sleep 45
    done
done

echo "All 6 launched. Waiting for completion..."
wait
echo "[$(date +%H:%M:%S)] All 6 runs done."
