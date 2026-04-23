#!/bin/bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python

# EXP-025: No InfoNCE
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 --config ./config/pacs/feddsa_exp025.yml --logger PerRunLogger --seed 2 > /tmp/exp025.out 2>&1 &

# EXP-028: Uncertainty Weighting
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_auto --gpu 0 --config ./config/pacs/feddsa_exp028.yml --logger PerRunLogger --seed 2 > /tmp/exp028.out 2>&1 &

# EXP-029: PCGrad
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_pcgrad --gpu 0 --config ./config/pacs/feddsa_exp029.yml --logger PerRunLogger --seed 2 > /tmp/exp029.out 2>&1 &

# EXP-030: Triplet Loss
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_triplet --gpu 0 --config ./config/pacs/feddsa_exp030.yml --logger PerRunLogger --seed 2 > /tmp/exp030.out 2>&1 &

# EXP-031: CKA Loss
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_cka --gpu 0 --config ./config/pacs/feddsa_exp031.yml --logger PerRunLogger --seed 2 > /tmp/exp031.out 2>&1 &

sleep 10
echo "Total processes:"
ps aux | grep run_single | grep python | grep -v grep | wc -l
echo ""
echo "GPU:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Restart watcher
pkill -f watch_experiments 2>/dev/null
sleep 1
nohup bash /tmp/watch_experiments.sh > /tmp/watch.out 2>&1 &
echo "Watcher restarted"
