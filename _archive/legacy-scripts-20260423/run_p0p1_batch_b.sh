#!/bin/bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python

# EXP-032: Triple combo (PCGrad + orth=2.0 + HSIC=0)
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_pcgrad --gpu 0 --config ./config/pacs/feddsa_exp032.yml --logger PerRunLogger --seed 2 > /tmp/exp032.out 2>&1 &

# EXP-033: PCGrad + warmup=80 + orth=2.0
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_pcgrad --gpu 0 --config ./config/pacs/feddsa_exp033.yml --logger PerRunLogger --seed 2 > /tmp/exp033.out 2>&1 &

# EXP-035: Multi-seed (seed=15)
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 --config ./config/pacs/feddsa_v4_no_hsic.yml --logger PerRunLogger --seed 15 > /tmp/exp035.out 2>&1 &

# EXP-036: Multi-seed (seed=333)
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 --config ./config/pacs/feddsa_v4_no_hsic.yml --logger PerRunLogger --seed 333 > /tmp/exp036.out 2>&1 &

sleep 10
echo "Total processes:"
ps aux | grep run_single | grep python | grep -v grep | wc -l
echo "GPU:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

pkill -f watch_experiments 2>/dev/null
sleep 1
nohup bash /tmp/watch_experiments.sh > /tmp/watch.out 2>&1 &
