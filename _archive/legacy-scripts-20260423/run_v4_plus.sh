#!/bin/bash
DIR=/root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python
cd $DIR

nohup $PY run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 --config ./config/pacs/feddsa_v4.yml --logger PerRunLogger --seed 2 > /tmp/exp014.out 2>&1 &
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_plus --gpu 0 --config ./config/pacs/feddsa_plus.yml --logger PerRunLogger --seed 2 > /tmp/exp015.out 2>&1 &

sleep 8
echo "Running:"
ps aux | grep run_single | grep python | grep -v grep | wc -l
echo "GPU:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Restart watcher to track new experiments
pkill -f watch_experiments 2>/dev/null
sleep 1
nohup bash /tmp/watch_experiments.sh > /tmp/watch.out 2>&1 &
echo "Watcher restarted"
