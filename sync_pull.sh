#!/bin/bash
# 从服务器拉文件到本地，排除项与 .rr.yaml / .gitignore 一致
REMOTE="lab-lry:/home/lry/code/federated-learning/"
LOCAL="$(dirname "$(realpath "$0")")/"

rsync -avz --progress \
  --exclude='.git' \
  --exclude='无关文件' \
  --exclude='Qwen3-VL-4B-mlc' \
  --exclude='mlc-qwen3-vl' \
  --exclude='mlc-qwen3-vl.zip' \
  --exclude='papers' \
  --exclude='PFLlib/dataset/MNIST' \
  --exclude='PFLlib/dataset/utils/LEAF' \
  --exclude='PFLlib/dataset/*/rawdata' \
  --exclude='PFLlib/dataset/*/train' \
  --exclude='PFLlib/dataset/*/test' \
  --exclude='RethinkFL/data' \
  --exclude='PARDON-FedDG/style_stats' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.so' \
  --exclude='*.bin' \
  --exclude='wandb' \
  --exclude='.venv' \
  --exclude='.ipynb_checkpoints' \
  --exclude='.idea' \
  --exclude='.vscode' \
  --exclude='sync_pull.sh' \
  "$@" "$REMOTE" "$LOCAL"
