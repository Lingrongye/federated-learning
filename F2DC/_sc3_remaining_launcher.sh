#!/bin/bash
# sc3 剩余 F2DC tasks (6 个: Office + Digits × 3 seed)
# 注意: F2DC × PACS × {2,15,333} 已经在跑, dispatcher 不要再 launch 它们.

cd /root/autodl-tmp/federated-learning/F2DC
mkdir -p _phaseC_logs

PY=/root/miniconda3/bin/python
declare -A PNUM=( [fl_pacs]=10 [fl_officecaltech]=10 [fl_digits]=20 )
MIN_FREE_MB=4500   # F2DC × Office/Digits 32×32 单 run ~3-4GB, 阈值 4500 比较保险

# 6 个剩余 F2DC tasks (Office + Digits × 3 seed)
TASKS=(
    "f2dc|fl_officecaltech|2"
    "f2dc|fl_officecaltech|15"
    "f2dc|fl_officecaltech|333"
    "f2dc|fl_digits|2"
    "f2dc|fl_digits|15"
    "f2dc|fl_digits|333"
)

echo "[$(date +%H:%M:%S)] sc3 剩余 launcher: ${#TASKS[@]} F2DC Office/Digits tasks, MIN_FREE_MB=$MIN_FREE_MB"
echo

for task in "${TASKS[@]}"; do
    IFS="|" read -r algo dset seed <<< "$task"
    pn=${PNUM[$dset]}
    label="${algo}_${dset}_s${seed}"
    log="_phaseC_logs/${label}.log"
    while true; do
        free_mb=$(/usr/bin/nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
        if [ "$free_mb" -ge $MIN_FREE_MB ]; then
            echo "[$(date +%H:%M:%S)] LAUNCH $label (free=${free_mb}MB)"
            $PY -u main_run.py \
                --device_id 0 --communication_epoch 100 --local_epoch 10 \
                --parti_num $pn --seed $seed --model $algo --dataset $dset \
                > $log 2>&1 &
            sleep 25
            break
        fi
        sleep 60
    done
done

echo "[$(date +%H:%M:%S)] All ${#TASKS[@]} tasks dispatched, waiting..."
wait
echo "[$(date +%H:%M:%S)] sc3 ALL DONE"
