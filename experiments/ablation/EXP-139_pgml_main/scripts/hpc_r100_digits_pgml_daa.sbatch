#!/bin/bash
#SBATCH --job-name=pgml_digits_pgml_daa
#SBATCH --account=a_csxlzhang
#SBATCH --partition=gpuA800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# === Digits R100 pgml_daa (2 seed seq: s15 + s333) ===

set -e
cd ~/projects/federated-learning/F2DC
source ~/miniconda3/etc/profile.d/conda.sh
conda activate f2dc

EXP_DIR=~/projects/federated-learning/experiments/ablation/EXP-139_pgml_main

echo "=== HPC env ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

for SEED in 15 333; do
  echo
  echo "================================================================"
  echo "=== Digits pgml_daa s${SEED} R100 $(date +%H:%M:%S) ==="
  echo "================================================================"
  python -u main_run.py \
    --model f2dc_pg_ml --dataset fl_digits --seed $SEED \
    --communication_epoch 100 --parti_num 20 \
    --num_classes 10 \
    --ml_aux_alpha 0.1 \
    --pg_proto_weight 0.3 --pg_warmup_rounds 30 --pg_ramp_rounds 20 \
    --use_daa True --agg_a 1.0 --agg_b 0.4 \
    --device_id 0 \
    > $EXP_DIR/logs/digits_pgml_daa_s${SEED}_R100.log 2>&1
  echo "=== Digits pgml_daa s${SEED} done $(date +%H:%M:%S) ==="
done

echo "=== ALL Digits pgml_daa 2 seeds done ==="
