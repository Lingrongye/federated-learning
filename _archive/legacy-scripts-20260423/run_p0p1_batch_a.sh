#!/bin/bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python

# EXP-032: Triple combo (HSIC=0 + orth=2.0 + PCGrad)
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_pcgrad --gpu 0 --config ./config/pacs/feddsa_exp032.yml --logger PerRunLogger --seed 2 > /tmp/exp032.out 2>&1 &

# EXP-028b: Fixed Uncertainty Weighting
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_auto --gpu 0 --config ./config/pacs/feddsa_exp028.yml --logger PerRunLogger --seed 2 > /tmp/exp028b.out 2>&1 &

sleep 8
echo "Total processes:"
ps aux | grep run_single | grep python | grep -v grep | wc -l
echo "GPU:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

pkill -f watch_experiments 2>/dev/null
sleep 1
nohup bash /tmp/watch_experiments.sh > /tmp/watch.out 2>&1 &
echo "Watcher restarted"
