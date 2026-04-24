"""Extract EXP-126 S0 gate results from flgo record JSONs."""
import json
import os
from pathlib import Path

RECORD_DIRS = {
    'Office': '/home/lry/code/federated-learning/FDSE_CVPR25/task/office_caltech10_c4/record',
    'PACS': '/home/lry/code/federated-learning/FDSE_CVPR25/task/PACS_c4/record',
}

def find_biproto_records(record_dir):
    out = {}
    for f in Path(record_dir).iterdir():
        if 'biproto' not in f.name or 'R30' not in f.name:
            continue
        # 解析 lp / fz 来区分 A (biproto-lite, lp=0.5) vs B (baseline, lp=0.0)
        if 'lp0.5' in f.name and 'fz1' in f.name:
            label = 'A_biproto'
        elif 'lp0.0' in f.name and 'fz1' in f.name:
            label = 'B_baseline'
        else:
            label = f'unknown_{f.name[:30]}'
        out[label] = f
    return out

for ds, d in RECORD_DIRS.items():
    print(f'\n=== {ds} ===')
    if not os.path.isdir(d):
        print(f'  no record dir')
        continue
    files = find_biproto_records(d)
    for label, fpath in files.items():
        with open(fpath) as fp:
            data = json.load(fp)
        avg_history = data.get('mean_local_test_accuracy', [])
        if avg_history:
            avg_arr = [a*100 if a < 2 else a for a in avg_history]
            best = max(avg_arr)
            best_round = avg_arr.index(best) + 1
            last = avg_arr[-1]
            print(f'  [{label}] file={fpath.name[:60]}...')
            print(f'    AVG Best @R{best_round} = {best:.2f}%')
            print(f'    AVG Last (R{len(avg_arr)}) = {last:.2f}%')
            print(f'    Trace (last 10 rounds): {[round(x, 2) for x in avg_arr[-10:]]}')
        else:
            print(f'  [{label}] no mean_local_test_accuracy in record')

print('\n=== Δ (BiProto-lite − Head-only baseline) ===')
for ds, d in RECORD_DIRS.items():
    files = find_biproto_records(d)
    if 'A_biproto' in files and 'B_baseline' in files:
        with open(files['A_biproto']) as fp:
            a = json.load(fp).get('mean_local_test_accuracy', [])
        with open(files['B_baseline']) as fp:
            b = json.load(fp).get('mean_local_test_accuracy', [])
        if a and b:
            a_pct = [x*100 if x < 2 else x for x in a]
            b_pct = [x*100 if x < 2 else x for x in b]
            ab = max(a_pct)
            bb = max(b_pct)
            print(f'  {ds}: A.best = {ab:.2f}, B.best = {bb:.2f}, Δ = {ab-bb:+.2f}pp')
