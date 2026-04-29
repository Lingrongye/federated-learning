#!/bin/bash
# EXP-130 v2: sc3 跑 9 个 F2DC tasks (2 seed × 3 dataset)
# fixed allocation 已生效 (commit 648eaa9)

cd /root/autodl-tmp/federated-learning/F2DC
mkdir -p _phaseC_v2_logs
PY=/root/miniconda3/bin/python
declare -A PNUM=( [fl_pacs]=10 [fl_officecaltech]=10 [fl_digits]=20 )

get_thr() {
    case "f2dc_$1" in
        f2dc_fl_pacs)         echo 8000 ;;
        f2dc_fl_officecaltech) echo 4000 ;;
        f2dc_fl_digits)       echo 4000 ;;
    esac
}

TASKS=(
    "f2dc|fl_pacs|15"
    "f2dc|fl_pacs|333"
    "f2dc|fl_officecaltech|15"
    "f2dc|fl_officecaltech|333"
    "f2dc|fl_digits|15"
    "f2dc|fl_digits|333"
)

echo "[$(date +%H:%M:%S)] sc3 v2 launcher: ${#TASKS[@]} F2DC tasks (seeds 15 + 333)"

for task in "${TASKS[@]}"; do
    IFS="|" read -r algo dset seed <<< "$task"
    pn=${PNUM[$dset]}
    label="${algo}_${dset}_s${seed}"
    log="_phaseC_v2_logs/${label}.log"
    threshold=$(get_thr $dset)
    while true; do
        free_mb=$(/usr/bin/nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
        if [ "$free_mb" -ge $threshold ]; then
            echo "[$(date +%H:%M:%S)] LAUNCH $label (free=${free_mb}MB, need=${threshold}MB)"
            $PY -u main_run.py \
                --device_id 0 --communication_epoch 100 --local_epoch 10 \
                --parti_num $pn --seed $seed --model $algo --dataset $dset \
                > $log 2>&1 &
            sleep 60   # 长 sleep 等真正 ramp up (避免 v1 30s 不够导致 OOM)
            break
        fi
        sleep 60
    done
done

echo "[$(date +%H:%M:%S)] All ${#TASKS[@]} dispatched, waiting..."
wait
echo "[$(date +%H:%M:%S)] sc3 v2 ALL DONE"
