#!/bin/bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python

nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_vae --gpu 0 --config ./config/pacs/feddsa_exp041.yml --logger PerRunLogger --seed 2 > /tmp/exp041.out 2>&1 &
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_asym --gpu 0 --config ./config/pacs/feddsa_exp042.yml --logger PerRunLogger --seed 2 > /tmp/exp042.out 2>&1 &
sleep 8
ps aux | grep run_single | grep python | grep -v grep | wc -l
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
