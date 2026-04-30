#!/bin/bash
#SBATCH --job-name=pgml_smoke_s2
#SBATCH --account=a_csxlzhang
#SBATCH --partition=gpuA800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# === SCUT HPC PACS R10 smoke test ===
# 目的: 验证 SCUT HPC A800 上能跑 f2dc_pg_ml R10, 看诊断输出
# 单 run, PACS s2, ml_aux_alpha=0.1, R10
# 跟 EXP-138 v100 smoke (s2, R10, 同 config) 横向对比验证一致

set -e

cd ~/projects/federated-learning/F2DC

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate f2dc

echo "=== HPC env check ==="
which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo

echo "=== Smoke run: PACS s2, R10, ml_aux_alpha=0.1 ==="
python -u main_run.py \
  --model f2dc_pg_ml \
  --dataset fl_pacs \
  --seed 2 \
  --communication_epoch 10 \
  --ml_aux_alpha 0.1 \
  --pg_proto_weight 0.3 \
  --pg_warmup_rounds 5 \
  --pg_ramp_rounds 3 \
  --use_daa False \
  --device_id 0
