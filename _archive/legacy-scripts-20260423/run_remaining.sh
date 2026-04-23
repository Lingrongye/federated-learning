#!/bin/bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python

nohup $PY run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 --config ./config/pacs/feddsa_v4_tau05.yml --logger PerRunLogger --seed 2 > /tmp/exp020.out 2>&1 &
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 --config ./config/pacs/feddsa_v4_strong_orth.yml --logger PerRunLogger --seed 2 > /tmp/exp021.out 2>&1 &

sleep 5
ps aux | grep run_single | grep python | grep -v grep | wc -l
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

pkill -f watch_experiments 2>/dev/null
sleep 1
nohup bash /tmp/watch_experiments.sh > /tmp/watch.out 2>&1 &
echo "Watcher restarted"
