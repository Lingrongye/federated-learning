#!/bin/bash
DIR=/root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python
cd $DIR

nohup $PY run_single.py --task PACS_c4 --algorithm fedavg --gpu 0 --config ./config/pacs/fedavg.yml --logger PerRunLogger --seed 2 > /tmp/exp007.out 2>&1 &
nohup $PY run_single.py --task PACS_c4 --algorithm fedbn --gpu 0 --config ./config/pacs/fedbn.yml --logger PerRunLogger --seed 2 > /tmp/exp008.out 2>&1 &
nohup $PY run_single.py --task PACS_c4 --algorithm fedproto --gpu 0 --config ./config/pacs/fedavg.yml --logger PerRunLogger --seed 2 > /tmp/exp009.out 2>&1 &
nohup $PY run_single.py --task PACS_c4 --algorithm fdse --gpu 0 --config ./config/pacs/fdse.yml --logger PerRunLogger --seed 2 > /tmp/exp010.out 2>&1 &

sleep 5
echo "Running:"
ps aux | grep run_single | grep python | grep -v grep | sed 's/.*--algorithm /  /' | sed 's/ --gpu.*//'
echo "GPU:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
