#!/bin/bash
# EXP-123 Stage B — Launch FDSE × 3 seeds on lab-lry GPU 1 (~13GB free)
# Each fdse run ~3.5GB on PACS, 3 parallel = ~10.5GB fits in 13GB free.

cd /home/lry/code/federated-learning/FDSE_CVPR25
PY=/home/lry/conda/envs/pfllib/bin/python
EXP_DIR=../experiments/ablation/EXP-123_art_diagnostic/stageB_full
LOG_DIR=$EXP_DIR/logs_lablry
mkdir -p $LOG_DIR

echo "[$(date +%H:%M:%S)] Launching fdse × 3 seeds on GPU 1 (lab-lry)"
for seed in 2 15 333; do
    CUDA_VISIBLE_DEVICES=1 $PY run_single.py \
        --task PACS_c4 --algorithm fdse --gpu 0 \
        --config ./config/pacs/fdse_r200.yml \
        --logger PerRunDiagLogger --seed $seed \
        > $LOG_DIR/fdse_s${seed}.log 2>&1 &
    echo "[$(date +%H:%M:%S)] fdse_s${seed} PID=$!"
    sleep 20  # ramp up VRAM
done
echo "[$(date +%H:%M:%S)] All 3 fdse runs launched on lab-lry GPU 1"
