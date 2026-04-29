#!/bin/bash
# 重跑 3 个 moon × pacs (在 sc4 launcher v2 因连续 launch sleep 30s 不够 ramp up 而 OOM 后)
# moon pacs 单 run 实测 ~11 GB, 24 GB GPU 顶多塞 2 个并行
# 策略: sleep 长一点 (180s 让真正 ramp up 完成), 阈值高一点 (12000 留 1GB 余量)

cd /root/autodl-tmp/federated-learning/F2DC
mkdir -p _phaseC_logs

PY=/root/miniconda3/bin/python
MIN_FREE_MB=12000  # 留余量
SEEDS=(2 15 333)

echo "[$(date +%H:%M:%S)] sc3 moon_pacs rerun (3 tasks, MIN_FREE_MB=$MIN_FREE_MB, sleep 180s after launch)"

for seed in "${SEEDS[@]}"; do
    label="moon_fl_pacs_s${seed}"
    log="_phaseC_logs/${label}.log"
    while true; do
        free_mb=$(/usr/bin/nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
        if [ "$free_mb" -ge $MIN_FREE_MB ]; then
            echo "[$(date +%H:%M:%S)] LAUNCH $label (free=${free_mb}MB)"
            $PY -u main_run.py \
                --device_id 0 --communication_epoch 100 --local_epoch 10 \
                --parti_num 10 --seed $seed --model moon --dataset fl_pacs \
                > $log 2>&1 &
            sleep 180  # 真正等 ramp up 完成 (vs 之前 30s 太短)
            break
        fi
        sleep 60
    done
done

echo "[$(date +%H:%M:%S)] All 3 dispatched, waiting..."
wait
echo "[$(date +%H:%M:%S)] All moon_pacs DONE"
