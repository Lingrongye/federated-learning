#!/bin/bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python

# EXP-022: HSIC=0 + lr=0.05 + fast decay
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 --config ./config/pacs/feddsa_exp022.yml --logger PerRunLogger --seed 2 > /tmp/exp022.out 2>&1 &

# EXP-023: HSIC=0 + EMA=0.9 (new algorithm)
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_stable --gpu 0 --config ./config/pacs/feddsa_exp023.yml --logger PerRunLogger --seed 2 > /tmp/exp023.out 2>&1 &

# EXP-024: HSIC=0 + soft aug Beta(0.5, 0.5) (new algorithm)
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_stable --gpu 0 --config ./config/pacs/feddsa_exp024.yml --logger PerRunLogger --seed 2 > /tmp/exp024.out 2>&1 &

sleep 8
echo "Running processes:"
ps aux | grep run_single | grep python | grep -v grep | wc -l
echo "GPU:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Restart watcher
pkill -f watch_experiments 2>/dev/null
sleep 1
nohup bash /tmp/watch_experiments.sh > /tmp/watch.out 2>&1 &
echo "Watcher restarted"
