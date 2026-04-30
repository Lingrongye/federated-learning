#!/bin/bash
#SBATCH --job-name=pgml_office_daa
#SBATCH --account=a_csxlzhang
#SBATCH --partition=gpuA800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# === Office +DaA s2/s15/s333 R100 sequential ===
# DaA 在 Office 上是 +3pp 提升 (vs 在 PACS 上拖性能, 不对称)
# 必须加才能跟 PG-DFC+DaA Office 64.69 同跑道比

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
  echo "=== R100 Office +DaA s${SEED} ml_aux_alpha=0.1 (start $(date +%H:%M:%S)) ==="
  echo "================================================================"
  python -u main_run.py \
    --model f2dc_pg_ml --dataset fl_officecaltech --seed $SEED \
    --communication_epoch 100 \
    --num_classes 10 \
    --ml_aux_alpha 0.1 \
    --pg_proto_weight 0.3 --pg_warmup_rounds 30 --pg_ramp_rounds 20 \
    --use_daa True --agg_a 1.0 --agg_b 0.4 \
    --device_id 0 \
    > $EXP_DIR/logs/pgml_daa_office_s${SEED}_R100.log 2>&1
  echo "=== Office +DaA s${SEED} done $(date +%H:%M:%S) ==="
done

echo
echo "=== ALL 3 Office +DaA seeds done ==="
