#!/bin/bash
# Run FedDSA-Adaptive experiments (EXP-072 series)
# Usage: bash run_adaptive.sh [phase]
#   phase=A: Fixed-alpha baselines (072a/b/c) — 9 runs
#   phase=B: M1 adaptive + M3-only + sanity (072/072d/072e) — 9 runs
#   phase=C: M1+M3 full (073) — 3 runs
# Default: phase A

set -e

cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python
EXP_DIR=../../experiments/adaptive/EXP-072_adaptive_baselines
EXP_DIR_073=../../experiments/adaptive/EXP-073_m1_m3_full

mkdir -p $EXP_DIR/results $EXP_DIR/logs
mkdir -p $EXP_DIR_073/results $EXP_DIR_073/logs

PHASE=${1:-A}
SEEDS=(2 333 42)

echo "=== FedDSA-Adaptive Phase $PHASE ==="
echo "Seeds: ${SEEDS[@]}"
date

if [ "$PHASE" = "A" ]; then
    echo "--- Phase A: Fixed-alpha baselines ---"
    for SEED in "${SEEDS[@]}"; do
        echo "Starting 072a (fixed alpha=0.2) seed=$SEED"
        nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_adaptive --gpu 0 \
            --config ./config/pacs/feddsa_072a.yml --seed $SEED \
            > $EXP_DIR/logs/072a_s${SEED}.log 2>&1 &
        sleep 2

        echo "Starting 072b (fixed alpha=0.5) seed=$SEED"
        nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_adaptive --gpu 0 \
            --config ./config/pacs/feddsa_072b.yml --seed $SEED \
            > $EXP_DIR/logs/072b_s${SEED}.log 2>&1 &
        sleep 2

        echo "Starting 072c (fixed alpha=0.8) seed=$SEED"
        nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_adaptive --gpu 0 \
            --config ./config/pacs/feddsa_072c.yml --seed $SEED \
            > $EXP_DIR/logs/072c_s${SEED}.log 2>&1 &
        sleep 2
    done
    echo "Phase A: 9 jobs launched"

elif [ "$PHASE" = "B" ]; then
    echo "--- Phase B: M1 adaptive + M3-only + sanity ---"
    for SEED in "${SEEDS[@]}"; do
        echo "Starting 072 (M1 adaptive) seed=$SEED"
        nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_adaptive --gpu 0 \
            --config ./config/pacs/feddsa_072.yml --seed $SEED \
            > $EXP_DIR/logs/072_s${SEED}.log 2>&1 &
        sleep 2

        echo "Starting 072d (M3-only) seed=$SEED"
        nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_adaptive --gpu 0 \
            --config ./config/pacs/feddsa_072d.yml --seed $SEED \
            > $EXP_DIR/logs/072d_s${SEED}.log 2>&1 &
        sleep 2

        echo "Starting 072e (disable low-gap aug) seed=$SEED"
        nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_adaptive --gpu 0 \
            --config ./config/pacs/feddsa_072e.yml --seed $SEED \
            > $EXP_DIR/logs/072e_s${SEED}.log 2>&1 &
        sleep 2
    done
    echo "Phase B: 9 jobs launched"

elif [ "$PHASE" = "C" ]; then
    echo "--- Phase C: M1+M3 full ---"
    for SEED in "${SEEDS[@]}"; do
        echo "Starting 073 (M1+M3 full) seed=$SEED"
        nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_adaptive --gpu 0 \
            --config ./config/pacs/feddsa_073.yml --seed $SEED \
            > $EXP_DIR_073/logs/073_s${SEED}.log 2>&1 &
        sleep 2
    done
    echo "Phase C: 3 jobs launched"

else
    echo "Unknown phase: $PHASE. Use A, B, or C."
    exit 1
fi

echo "All jobs launched. Check with: ps -eo pid,etime,cmd | grep run_single | grep -v grep"
