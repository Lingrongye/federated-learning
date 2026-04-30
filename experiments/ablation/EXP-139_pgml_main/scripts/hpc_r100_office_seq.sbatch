#!/bin/bash
#SBATCH --job-name=pgml_office_seq
#SBATCH --account=a_csxlzhang
#SBATCH --partition=gpuA800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# === Office s2/s15/s333 R100 sequential (single A800) ===
# 单 A800 顺序跑 3 seed, 节省并发 (vs 3 卡同时), wall ~1.5-2h

set -e
cd ~/projects/federated-learning/F2DC
source ~/miniconda3/etc/profile.d/conda.sh
conda activate f2dc

EXP_DIR=~/projects/federated-learning/experiments/ablation/EXP-139_pgml_main

echo "=== HPC env ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

for SEED in 2 15 333; do
  echo
  echo "================================================================"
  echo "=== R100 Office s${SEED} ml_aux_alpha=0.1 (start $(date +%H:%M:%S)) ==="
  echo "================================================================"
  python -u main_run.py \
    --model f2dc_pg_ml --dataset fl_officecaltech --seed $SEED \
    --communication_epoch 100 \
    --num_classes 10 \
    --ml_aux_alpha 0.1 \
    --pg_proto_weight 0.3 --pg_warmup_rounds 30 --pg_ramp_rounds 20 \
    --use_daa False --device_id 0 \
    > $EXP_DIR/logs/pgml_office_s${SEED}_R100.log 2>&1
  echo "=== Office s${SEED} done $(date +%H:%M:%S) ==="
done

echo
echo "=== ALL 3 Office seeds done ==="
