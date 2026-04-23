#!/bin/bash
DIR=/root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python
cd $DIR

nohup $PY run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 --config ./config/pacs/feddsa_v1.yml --logger PerRunLogger --seed 2 > /tmp/exp011.out 2>&1 &
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 --config ./config/pacs/feddsa_v2.yml --logger PerRunLogger --seed 2 > /tmp/exp012.out 2>&1 &
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 --config ./config/pacs/feddsa_v3.yml --logger PerRunLogger --seed 2 > /tmp/exp013.out 2>&1 &

sleep 8
echo "Running:"
ps aux | grep run_single | grep python | grep -v grep | wc -l
echo "processes"
echo "GPU:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
