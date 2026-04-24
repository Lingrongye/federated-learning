#!/bin/bash
# EXP-125 Save-Model Batch 1 — 占 lab-lry GPU 1, 跑 3 个 runs 带 --save_model
# seetacloud2 已经在跑 orth_only s=2 with save_model, 这里补:
#   - fedbn_s2 (save_model)
#   - fdse_s2 (save_model)
#   - orth_s15 (save_model)
# 后续 batch 2: fedbn_s15 + fdse_s15 + orth_s333
# 再 batch 3: fedbn_s333 + fdse_s333

cd /home/lry/code/federated-learning/FDSE_CVPR25
PY=/home/lry/conda/envs/pfllib/bin/python
EXP_DIR=../experiments/EXP-125_ocsd_verify/savemodel_runs
mkdir -p $EXP_DIR/logs

echo "[$(date +%H:%M:%S)] Launch batch 1 on lab-lry GPU 1"

# fedbn s=2
CUDA_VISIBLE_DEVICES=1 nohup $PY run_single.py \
    --task PACS_c4 --algorithm fedbn --gpu 0 \
    --config ./config/pacs/fedbn_r200.yml \
    --logger PerRunDiagLogger --seed 2 --save_model \
    > $EXP_DIR/logs/fedbn_s2.log 2>&1 &
echo "fedbn_s2 PID=$!"
sleep 20

# fdse s=2
CUDA_VISIBLE_DEVICES=1 nohup $PY run_single.py \
    --task PACS_c4 --algorithm fdse --gpu 0 \
    --config ./config/pacs/fdse_r200.yml \
    --logger PerRunDiagLogger --seed 2 --save_model \
    > $EXP_DIR/logs/fdse_s2.log 2>&1 &
echo "fdse_s2 PID=$!"
sleep 20

# orth s=15 (feddsa_scheduled sm=0)
CUDA_VISIBLE_DEVICES=1 nohup $PY run_single.py \
    --task PACS_c4 --algorithm feddsa_scheduled --gpu 0 \
    --config ./config/pacs/feddsa_orth_lr05.yml \
    --logger PerRunDiagLogger --seed 15 --save_model \
    > $EXP_DIR/logs/orth_s15.log 2>&1 &
echo "orth_s15 PID=$!"
sleep 20

echo "[$(date +%H:%M:%S)] 3 runs launched"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tail -1
