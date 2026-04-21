#!/bin/bash
# EXP-113 Office probe batch: 12 checkpoints (A VIB / B VSC / C SupCon / orth_uc1) × 3 seeds
# Requires: all 12 Office runs finished + se=1 saved checkpoints
# Output:   experiments/ablation/EXP-113_vib_vsc_supcon/office/probes/*.json
set -e
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python
OUTDIR=/root/autodl-tmp/federated-learning/experiments/ablation/EXP-113_vib_vsc_supcon/office/probes
mkdir -p $OUTDIR

echo "[$(date)] EXP-113 Office probe batch start"

# Use Python to resolve (method, seed) → checkpoint path
$PY <<'PYEOF' > /tmp/exp113_ckpt_map.txt
import os, json, glob
ckpts = sorted(glob.glob('/root/fl_checkpoints/sgpa_office_caltech10_c4_*_R200_*'))
for ck in ckpts:
    meta_p = os.path.join(ck, 'meta.json')
    if not os.path.exists(meta_p):
        continue
    try:
        meta = json.load(open(meta_p))
    except Exception:
        continue
    cfg = meta.get('config', meta.get('cfg', ''))
    # extract seed from dir name
    ts = int(ck.split('_')[-1])
    # skip pre-EXP-113 ckpts (ts < 1776770000 = before 2026-04-21 ~19:00)
    if ts < 1776770000:
        continue
    seed = ck.split('_s')[1].split('_')[0]
    # parse config basename
    cfg_name = os.path.basename(cfg) if cfg else ''
    # Try to infer method from config name
    if 'vib_office' in cfg_name:
        m = 'A_vib'
        probe_cfg = './config/office/feddsa_vib_office_r200.yml'
    elif 'vsc_office' in cfg_name:
        m = 'B_vsc'
        probe_cfg = './config/office/feddsa_vsc_office_r200.yml'
    elif 'supcon_office' in cfg_name:
        m = 'C_supcon'
        probe_cfg = './config/office/feddsa_supcon_office_r200.yml'
    elif 'orth_uc1' in cfg_name:
        m = 'orth_uc1'
        probe_cfg = './config/office/feddsa_orth_uc1_office_r200.yml'
    else:
        # fallback: read algo_para from checkpoint / meta
        algo_para = meta.get('algo_para', [])
        if len(algo_para) >= 15:
            vib = int(algo_para[13])
            us = int(algo_para[14])
            if vib == 1 and us == 0: m = 'A_vib'; probe_cfg = './config/office/feddsa_vib_office_r200.yml'
            elif vib == 1 and us == 1: m = 'B_vsc'; probe_cfg = './config/office/feddsa_vsc_office_r200.yml'
            elif vib == 0 and us == 1: m = 'C_supcon'; probe_cfg = './config/office/feddsa_supcon_office_r200.yml'
            else: m = 'orth_uc1'; probe_cfg = './config/office/feddsa_orth_uc1_office_r200.yml'
        elif len(algo_para) >= 13:
            m = 'orth_uc1'; probe_cfg = './config/office/feddsa_orth_uc1_office_r200.yml'
        else:
            continue
    print(f'{m}\ts{seed}\t{ck}\t{probe_cfg}')
PYEOF

echo "=== Resolved checkpoint map ==="
cat /tmp/exp113_ckpt_map.txt
echo "==============================="

# Run probes
while IFS=$'\t' read -r method seed ckpt cfg; do
    if [ -z "$method" ]; then continue; fi
    out="$OUTDIR/${method}_${seed}.json"
    if [ -f "$out" ]; then
        echo "[$(date)] SKIP $method $seed (already probed)"
        continue
    fi
    echo "[$(date)] PROBE $method $seed ckpt=$(basename $ckpt)"
    $PY scripts/run_capacity_probes.py \
        --ckpt "$ckpt" \
        --task office_caltech10_c4 \
        --config "$cfg" \
        --output "$out" \
        --gpu 0 2>&1 | tail -15
done < /tmp/exp113_ckpt_map.txt

echo "[$(date)] ALL EXP-113 Office probes DONE"
echo "Results in $OUTDIR"
ls -la $OUTDIR
