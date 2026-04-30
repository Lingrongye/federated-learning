#!/bin/bash
#SBATCH --job-name=pgml_smoke_digits
#SBATCH --account=a_csxlzhang
#SBATCH --partition=gpuA800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# === Digits R10 smoke 4 variants (s2, ml_aux_alpha=0.1) ===
# 顺序跑 4 个 variant 验证全部能跑通 + ML diag 健康
# Digits 32×32 input, A800 上估单 R10 ~3-5 min, 4 个总 ~15-20 min

set -e
cd ~/projects/federated-learning/F2DC
source ~/miniconda3/etc/profile.d/conda.sh
conda activate f2dc

EXP_DIR=~/projects/federated-learning/experiments/ablation/EXP-139_pgml_main

echo "=== HPC env ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

# 通用参数 (Digits 是 10 类, parti_num=20, fixed allocation: mnist:3/usps:6/svhn:6/syn:5)
COMMON="--dataset fl_digits --seed 2 --communication_epoch 10 \
  --num_classes 10 --pg_proto_weight 0.3 --pg_warmup_rounds 5 --pg_ramp_rounds 3 \
  --device_id 0"

echo
echo "================================================================"
echo "=== Variant 1: PG-DFC vanilla (no DaA, no ML) $(date +%H:%M:%S) ==="
echo "================================================================"
python -u main_run.py --model f2dc_pg --use_daa False $COMMON \
  > $EXP_DIR/logs/smoke_digits_pgdfc_vanilla_s2_R10.log 2>&1
echo "done"

echo
echo "================================================================"
echo "=== Variant 2: PG-DFC + DaA (no ML) $(date +%H:%M:%S) ==="
echo "================================================================"
python -u main_run.py --model f2dc_pg --use_daa True --agg_a 1.0 --agg_b 0.4 $COMMON \
  > $EXP_DIR/logs/smoke_digits_pgdfc_daa_s2_R10.log 2>&1
echo "done"

echo
echo "================================================================"
echo "=== Variant 3: PG-DFC-ML vanilla (no DaA) $(date +%H:%M:%S) ==="
echo "================================================================"
python -u main_run.py --model f2dc_pg_ml --use_daa False --ml_aux_alpha 0.1 $COMMON \
  > $EXP_DIR/logs/smoke_digits_pgml_vanilla_s2_R10.log 2>&1
echo "done"

echo
echo "================================================================"
echo "=== Variant 4: PG-DFC-ML + DaA $(date +%H:%M:%S) ==="
echo "================================================================"
python -u main_run.py --model f2dc_pg_ml --use_daa True --agg_a 1.0 --agg_b 0.4 --ml_aux_alpha 0.1 $COMMON \
  > $EXP_DIR/logs/smoke_digits_pgml_daa_s2_R10.log 2>&1
echo "done"

echo
echo "=== ALL 4 Digits smoke variants done $(date +%H:%M:%S) ==="
for v in pgdfc_vanilla pgdfc_daa pgml_vanilla pgml_daa; do
  log=$EXP_DIR/logs/smoke_digits_${v}_s2_R10.log
  acc_list=$(grep "Accuracy List" $log | head -1)
  echo "[$v] $acc_list"
done
