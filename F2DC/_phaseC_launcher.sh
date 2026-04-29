#!/bin/bash
# EXP-130 Phase C: 27 runs greedy launcher
# 3 algo (f2dc/fedavg/moon) × 3 dataset (pacs/office/digits) × 3 seed (2/15/333)
# Greedy: 按 GPU 0 free memory 动态 launch, 不串行 wait wave
# 单 run 显存估: F2DC ~5GB, FedAvg ~3GB, MOON ~6GB (3 份模型). 取 max 6GB + 余量 → MIN_FREE_MB=6500

cd /root/autodl-tmp/federated-learning/F2DC
mkdir -p _phaseC_logs

PY=/root/miniconda3/bin/python
declare -A PNUM=( [fl_pacs]=10 [fl_officecaltech]=10 [fl_digits]=20 )

ALGOS=(f2dc fedavg moon)
DSETS=(fl_pacs fl_officecaltech fl_digits)
SEEDS=(2 15 333)
MIN_FREE_MB=6500

# build task list
TASKS=()
for algo in "${ALGOS[@]}"; do
    for dset in "${DSETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            TASKS+=("${algo}|${dset}|${seed}")
        done
    done
done

echo "[$(date +%H:%M:%S)] Phase C launcher: ${#TASKS[@]} runs queued, MIN_FREE_MB=$MIN_FREE_MB"
echo

# greedy launch
for task in "${TASKS[@]}"; do
    IFS="|" read -r algo dset seed <<< "$task"
    pn=${PNUM[$dset]}
    label="${algo}_${dset}_s${seed}"
    log="_phaseC_logs/${label}.log"

    # wait until GPU 0 has enough free memory
    while true; do
        free_mb=$(/usr/bin/nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
        if [ "$free_mb" -ge $MIN_FREE_MB ]; then
            echo "[$(date +%H:%M:%S)] LAUNCH $label (free=${free_mb}MB)"
            $PY -u main_run.py \
                --device_id 0 --communication_epoch 100 --local_epoch 10 \
                --parti_num $pn --seed $seed --model $algo --dataset $dset \
                > $log 2>&1 &
            sleep 25  # ramp up before next check
            break
        fi
        sleep 60
    done
done

echo "[$(date +%H:%M:%S)] All ${#TASKS[@]} tasks dispatched, waiting for completion..."
wait
echo "[$(date +%H:%M:%S)] ALL DONE"
