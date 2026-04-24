#!/bin/bash
# EXP-123 Stage B Full — 9 runs R=200 with diagnostic hook
# Algos: fedbn / feddsa_scheduled / fdse  × Seeds: {2, 15, 333}
# Greedy scheduler on seetacloud2 (RTX 4090 24GB).

cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python
EXP_DIR=../experiments/ablation/EXP-123_art_diagnostic/stageB_full
LOG_DIR=$EXP_DIR/logs
mkdir -p $LOG_DIR

TASKS=(
  "fedbn_s2|fedbn|fedbn_r200.yml|2"
  "fedbn_s15|fedbn|fedbn_r200.yml|15"
  "fedbn_s333|fedbn|fedbn_r200.yml|333"
  "orth_s2|feddsa_scheduled|feddsa_orth_lr05.yml|2"
  "orth_s15|feddsa_scheduled|feddsa_orth_lr05.yml|15"
  "orth_s333|feddsa_scheduled|feddsa_orth_lr05.yml|333"
  "fdse_s2|fdse|fdse_r200.yml|2"
  "fdse_s15|fdse|fdse_r200.yml|15"
  "fdse_s333|fdse|fdse_r200.yml|333"
)

MIN_FREE_MB=4500  # PACS R=200 feddsa 约 3-4GB, 留余量

echo "[$(date +%H:%M:%S)] Launching 9 runs with greedy scheduler (MIN_FREE_MB=$MIN_FREE_MB)"

for task in "${TASKS[@]}"; do
    IFS="|" read -r label algo config seed <<< "$task"
    while true; do
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if [ "$free_mb" -ge $MIN_FREE_MB ]; then
            echo "[$(date +%H:%M:%S)] LAUNCH $label (free=${free_mb}MB)"
            CUDA_VISIBLE_DEVICES=0 $PY run_single.py \
                --task PACS_c4 --algorithm $algo --gpu 0 \
                --config ./config/pacs/$config \
                --logger PerRunDiagLogger --seed $seed \
                > $LOG_DIR/${label}.log 2>&1 &
            sleep 20  # ramp up before next slot check
            break
        fi
        sleep 45
    done
done

echo "[$(date +%H:%M:%S)] All 9 runs launched. Waiting for completion..."
wait
echo "[$(date +%H:%M:%S)] DONE"
