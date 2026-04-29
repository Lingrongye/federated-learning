#!/bin/bash
# sc3 moon × pacs 收尾 launcher
# 单 moon pacs 实测占 12.8 GB, sc3 24 GB 只能塞 1 个并行
# 等 GPU free >= 13500 MB (留 700 MB 余量) 就 launch, 串行单跑

cd /root/autodl-tmp/federated-learning/F2DC
mkdir -p _phaseC_v2_logs
PY=/root/miniconda3/bin/python
MIN_FREE_MB=13500

SEEDS=(15 333)

echo "[$(date +%H:%M:%S)] sc3 moon_pacs launcher: 2 tasks, MIN_FREE_MB=$MIN_FREE_MB"
echo "  实测 moon pacs 单 run ~12.8 GB, 必须串行 (24 GB GPU 只塞 1 个)"

for seed in "${SEEDS[@]}"; do
    label="moon_fl_pacs_s${seed}"
    log="_phaseC_v2_logs/${label}.log"
    while true; do
        free_mb=$(/usr/bin/nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
        if [ "$free_mb" -ge $MIN_FREE_MB ]; then
            echo "[$(date +%H:%M:%S)] LAUNCH $label (free=${free_mb}MB)"
            $PY -u main_run.py \
                --device_id 0 --communication_epoch 100 --local_epoch 10 \
                --parti_num 10 --seed $seed --model moon --dataset fl_pacs \
                > $log 2>&1 &
            sleep 90  # 等 ramp up 稳定再下一轮 check
            break
        fi
        sleep 60
    done
done

echo "[$(date +%H:%M:%S)] All 2 dispatched, waiting..."
wait
echo "[$(date +%H:%M:%S)] moon_pacs ALL DONE"
