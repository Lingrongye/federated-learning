#!/bin/bash
#SBATCH --job-name=pgml_office_s333
#SBATCH --account=a_csxlzhang
#SBATCH --partition=gpuA800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

set -e
cd ~/projects/federated-learning/F2DC
source ~/miniconda3/etc/profile.d/conda.sh
conda activate f2dc

echo "=== R100 Office s333 ml_aux_alpha=0.1 ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python -u main_run.py \
  --model f2dc_pg_ml --dataset fl_officecaltech --seed 333 \
  --communication_epoch 100 \
  --num_classes 10 \
  --ml_aux_alpha 0.1 \
  --pg_proto_weight 0.3 --pg_warmup_rounds 30 --pg_ramp_rounds 20 \
  --use_daa False --device_id 0
