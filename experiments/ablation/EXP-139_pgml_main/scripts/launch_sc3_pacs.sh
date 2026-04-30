#!/bin/bash
# EXP-139 sc3 launch: PG-DFC-ML PACS s15 + s333 R100 并跑
# 前置: 必须 sc3 EXP-137 PACS s333 已 R100 完成 (检查 ps + GPU free)
# 用法: ssh sc3 'bash -s' < launch_sc3_pacs.sh

set -e

PY=/root/miniconda3/bin/python
PROJ=/root/autodl-tmp/federated-learning
EXP_DIR=$PROJ/experiments/ablation/EXP-139_pgml_main

cd $PROJ/F2DC

# Pre-check: GPU free 至少 11GB (两个 PACS run 各 ~5GB)
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free_mb" -lt 11000 ]; then
    echo "ERROR: GPU free=${free_mb}MB < 11000MB, 不能启动两个 R100 run"
    echo "当前 GPU 占用进程:"
    ps -eo pid,etime,cmd | grep -E 'main_run|f2dc' | grep -v grep
    exit 1
fi

mkdir -p $EXP_DIR/logs

echo "=== Launching PACS s15 (R100, ml_aux_alpha=0.1) ==="
setsid $PY -u main_run.py \
  --model f2dc_pg_ml --dataset fl_pacs --seed 15 \
  --communication_epoch 100 \
  --ml_aux_alpha 0.1 \
  --pg_proto_weight 0.3 --pg_warmup_rounds 30 --pg_ramp_rounds 20 \
  --use_daa False --device_id 0 \
  > $EXP_DIR/logs/pgml_pacs_s15_R100.log 2>&1 < /dev/null & disown

sleep 30   # 让 s15 ramp up 显存再启动 s333

echo "=== Launching PACS s333 (R100, ml_aux_alpha=0.1) ==="
setsid $PY -u main_run.py \
  --model f2dc_pg_ml --dataset fl_pacs --seed 333 \
  --communication_epoch 100 \
  --ml_aux_alpha 0.1 \
  --pg_proto_weight 0.3 --pg_warmup_rounds 30 --pg_ramp_rounds 20 \
  --use_daa False --device_id 0 \
  > $EXP_DIR/logs/pgml_pacs_s333_R100.log 2>&1 < /dev/null & disown

sleep 5
echo "=== 启动后状态 ==="
ps -eo pid,etime,cmd | grep f2dc_pg_ml | grep -v grep
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
