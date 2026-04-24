#!/bin/bash
# EXP-126 S0 Matched-Intervention Gate launcher (lab-lry GPU 1)
# 启动 4 runs: Office A/B + PACS A/B (R30, seed=2)
# 用 greedy GPU memory check (CLAUDE.md 17.8 规范)
set -e

PY=/home/lry/conda/envs/pfllib/bin/python
EXP_DIR=/home/lry/code/federated-learning/experiments/ablation/EXP-126_biproto_S0_office_gate
mkdir -p $EXP_DIR/logs $EXP_DIR/results
cd /home/lry/code/federated-learning/FDSE_CVPR25

MIN_FREE_MB=4500
GPU=1

declare -a TASKS=(
    "office_A|office_caltech10_c4|./config/office/feddsa_biproto_s0_gate.yml"
    "office_B|office_caltech10_c4|./config/office/feddsa_biproto_s0_gate_baseline.yml"
    "pacs_A|PACS_c4|./config/pacs/feddsa_biproto_s0_gate.yml"
    "pacs_B|PACS_c4|./config/pacs/feddsa_biproto_s0_gate_baseline.yml"
)

for task in "${TASKS[@]}"; do
    IFS="|" read -r label dataset config <<< "$task"
    while true; do
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk -v g=$GPU 'NR==g+1{print $1}')
        if [ "$free_mb" -ge $MIN_FREE_MB ]; then
            echo "[$(date +%H:%M:%S)] LAUNCH $label (GPU $GPU free=${free_mb}MB) on $dataset"
            CUDA_VISIBLE_DEVICES=$GPU nohup $PY run_single.py \
                --task $dataset --algorithm feddsa_biproto --gpu 0 \
                --config $config --logger PerRunLogger --seed 2 \
                > $EXP_DIR/logs/${label}.log 2>&1 &
            echo "  PID=$!"
            sleep 20
            break
        fi
        echo "[$(date +%H:%M:%S)] WAIT $label (GPU $GPU free=${free_mb}MB < $MIN_FREE_MB)"
        sleep 30
    done
done

echo "All 4 launched. Run 'wait' or 'ps -p <PID>' to track."
wait
echo "[$(date +%H:%M:%S)] All 4 runs done."
