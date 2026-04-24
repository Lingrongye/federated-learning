#!/bin/bash
# Collect EXP-126 final results
cd /home/lry/code/federated-learning/experiments/ablation/EXP-126_biproto_S0_office_gate
for f in office_A office_B pacs_A pacs_B; do
    echo "=== $f ==="
    ls -la logs/$f.log
    echo "--- tail 10 ---"
    tail -10 logs/$f.log
    echo
done

echo "=== Record JSONs ==="
ls /home/lry/code/federated-learning/FDSE_CVPR25/task/office_caltech10_c4/record/ 2>&1 | grep biproto | grep s0_gate | head -5
ls /home/lry/code/federated-learning/FDSE_CVPR25/task/PACS_c4/record/ 2>&1 | grep biproto | grep s0_gate | head -5
