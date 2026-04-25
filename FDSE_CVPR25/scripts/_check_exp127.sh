#!/bin/bash
# Check EXP-127 progress
echo "=== Running processes ==="
ps -eo pid,etime,cmd | grep run_single | grep biproto | grep -v grep
echo
echo "=== GPU 1 ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | awk 'NR==2'
echo
echo "=== Per-run progress (last test_acc) ==="
EXP_DIR=/home/lry/code/federated-learning/experiments/ablation/EXP-127_biproto_full_r200
PY=/home/lry/conda/envs/pfllib/bin/python
for label in office_s2 office_s15 office_s333 pacs_s2 pacs_s15 pacs_s333; do
    log=$EXP_DIR/logs/$label.log
    if [ -f "$log" ]; then
        # Count complete rounds via "Local Training" 出现次数 (每 round 一次, 但 \r 覆盖, grep -o 计数 不准)
        last_size=$(stat -c%s $log)
        # Extract last record JSON if any
        echo "[$label] log_size=${last_size}B"
    fi
done
echo
echo "=== Record JSONs ==="
for ds in office_caltech10_c4 PACS_c4; do
    echo "--- $ds ---"
    ls -la /home/lry/code/federated-learning/FDSE_CVPR25/task/$ds/record/ 2>&1 | grep biproto | grep R200 | grep fz0 | tail -3
done
echo
echo "=== Best ckpt collected ==="
ls -la /home/lry/fl_checkpoints/ | grep R200 | grep -E "$(date +%Y_%m_%d)|$(date -d 'yesterday' +%Y_%m_%d)" 2>&1 | tail -10 || ls -la /home/lry/fl_checkpoints/ 2>&1 | tail -10
echo
echo "=== Try extract progress from JSON ==="
$PY -c "
import json, glob, os
for ds, key in [('Office', 'office_caltech10_c4'), ('PACS', 'PACS_c4')]:
    files = sorted(glob.glob(f'/home/lry/code/federated-learning/FDSE_CVPR25/task/{key}/record/feddsa_biproto*R200*fz0*.json'))
    print(f'--- {ds} ({len(files)} runs) ---')
    for f in files:
        try:
            with open(f) as fp: d = json.load(fp)
            a = d.get('mean_local_test_accuracy', [])
            if not a: continue
            a = [x*100 if x<2 else x for x in a]
            seed = 'unknown'
            for s in ['S2', 'S15', 'S333']:
                if f'_{s}_' in f: seed = s
            best = max(a)
            best_r = a.index(best)+1
            print(f'  {seed}: {len(a)}/200 rounds, best={best:.2f}@R{best_r}, last={a[-1]:.2f}')
        except Exception as e:
            print(f'  {os.path.basename(f)[:40]}: read error {e}')
"
