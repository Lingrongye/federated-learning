#!/bin/bash
# EXP-139 v100 launch: PG-DFC-ML Office s15 + s333 R100 并跑
# 前置: v100 EXP-137 PACS s15 已 R100 完成
# 用法: ssh v100 'bash -s' < launch_v100_office.sh

set -e

PROJ=/workspace/federated-learning
EXP_DIR=$PROJ/experiments/ablation/EXP-139_pgml_main

cd $PROJ/F2DC

# Pre-check: GPU free 至少 9GB (两个 Office run 各 ~4GB)
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free_mb" -lt 9000 ]; then
    echo "ERROR: GPU free=${free_mb}MB < 9000MB, 不能启动两个 R100 run"
    ps -eo pid,etime,cmd | grep -E 'main_run|f2dc' | grep -v grep
    exit 1
fi

mkdir -p $EXP_DIR/logs

echo "=== Launching Office s15 (R100, ml_aux_alpha=0.1) ==="
setsid python -u main_run.py \
  --model f2dc_pg_ml --dataset fl_officecaltech --seed 15 \
  --communication_epoch 100 \
  --ml_aux_alpha 0.1 \
  --pg_proto_weight 0.3 --pg_warmup_rounds 30 --pg_ramp_rounds 20 \
  --use_daa False --device_id 0 \
  > $EXP_DIR/logs/pgml_office_s15_R100.log 2>&1 < /dev/null & disown

sleep 20

echo "=== Launching Office s333 (R100, ml_aux_alpha=0.1) ==="
setsid python -u main_run.py \
  --model f2dc_pg_ml --dataset fl_officecaltech --seed 333 \
  --communication_epoch 100 \
  --ml_aux_alpha 0.1 \
  --pg_proto_weight 0.3 --pg_warmup_rounds 30 --pg_ramp_rounds 20 \
  --use_daa False --device_id 0 \
  > $EXP_DIR/logs/pgml_office_s333_R100.log 2>&1 < /dev/null & disown

sleep 5
echo "=== 启动后状态 ==="
ps -eo pid,etime,cmd | grep f2dc_pg_ml | grep -v grep
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
