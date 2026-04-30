#!/bin/bash
#SBATCH --job-name=digits_pgdfc_daa_s333
#SBATCH --account=a_csxlzhang
#SBATCH --partition=gpuA800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# === Digits R100 pgdfc_daa s=333 only (前次 sbatch TIMEOUT 3h 没跑完 s333) ===
set -e
cd ~/projects/federated-learning/F2DC
source ~/miniconda3/etc/profile.d/conda.sh
conda activate f2dc

EXP_DIR=~/projects/federated-learning/experiments/ablation/EXP-139_pgml_main
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

echo "=== Digits pgdfc_daa s333 R100 $(date +%H:%M:%S) ==="
python -u main_run.py \
  --model f2dc_pg --dataset fl_digits --seed 333 \
  --communication_epoch 100 --parti_num 20 \
  --num_classes 10 \
   \
  --pg_proto_weight 0.3 --pg_warmup_rounds 30 --pg_ramp_rounds 20 \
  --use_daa True --agg_a 1.0 --agg_b 0.4 \
  --device_id 0 \
  > $EXP_DIR/logs/digits_pgdfc_daa_s333_R100.log 2>&1
echo "=== done $(date +%H:%M:%S) ==="
