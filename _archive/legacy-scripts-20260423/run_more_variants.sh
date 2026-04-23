#!/bin/bash
DIR=/root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python
cd $DIR

# EXP-016: V4 seed=15
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 --config ./config/pacs/feddsa_v4.yml --logger PerRunLogger --seed 15 > /tmp/exp016.out 2>&1 &

# EXP-017: V4 no HSIC
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 --config ./config/pacs/feddsa_v4_no_hsic.yml --logger PerRunLogger --seed 2 > /tmp/exp017.out 2>&1 &

# EXP-018: FedDSA+ late stages
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_plus --gpu 0 --config ./config/pacs/feddsa_plus_late.yml --logger PerRunLogger --seed 2 > /tmp/exp018.out 2>&1 &

sleep 8
echo "All processes running:"
ps aux | grep run_single | grep python | grep -v grep | wc -l
echo "GPU:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Restart watcher
pkill -f watch_experiments 2>/dev/null
sleep 1
nohup bash /tmp/watch_experiments.sh > /tmp/watch.out 2>&1 &
echo "Watcher restarted with all experiments"
