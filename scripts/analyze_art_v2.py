# -*- coding: utf-8 -*-
"""EXP-123 Stage A v2 — 用 paper-standard metric 算 per-domain.

Metric 定义 (严格对齐 FDSE paper):
  AVG Best = max over rounds of (mean over 4 clients per round)   <- 一个 "global best round"
  per-domain at best = pc[best_round, c] for c in [Art, Cartoon, Photo, Sketch]
  3-seed mean ± std of all above
"""
import os, json, glob, sys, io
import numpy as np

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

RECORD_DIR = r"D:\桌面文件\联邦学习\FDSE_CVPR25\task\PACS_c4\record"
DOMAINS = ['Art', 'Cartoon', 'Photo', 'Sketch']
SEEDS = [2, 15, 333]

PATTERNS = {
    'orth_only': {
        'include': 'feddsa_scheduled_lo1.0_lh0.0_ls1.0_tau0.2_sdn5_pd128_sm0_bp60_bw30_cr80_gli10_lm1.0_al0.25_Mfeddsa_scheduled',
        'exclude': ['_se1_', '_sas', '_scpr', '_sas_tau', '_scpr_tau'],
        'lr': '5.00e-02',
    },
    'fdse': {
        'include': 'fdse_lmbd0.5_tau0.5_beta0.001_Mfdse',
        'exclude': [],
        'lr': '5.00e-02',
    },
}


def seed_of(fn):
    for s in SEEDS:
        if f'_S{s}_' in fn:
            return s
    return None


def match(fn, pat):
    if pat['include'] not in fn: return False
    if f"_LR{pat['lr']}_" not in fn: return False
    for bad in pat['exclude']:
        if bad in fn: return False
    return True


def find():
    result = {}
    for method, pat in PATTERNS.items():
        result[method] = {}
        for fn in sorted(os.listdir(RECORD_DIR)):
            if not fn.endswith('.json'): continue
            s = seed_of(fn)
            if s is None: continue
            if not match(fn, pat): continue
            if s not in result[method] or len(fn) < len(result[method][s]):
                result[method][s] = fn
    return result


def load(fp):
    with open(fp) as f: return json.load(f)


def analyze(d):
    """Paper-standard: AVG Best + per-domain at best-global-round."""
    avg_curve = np.array(d.get('mean_local_test_accuracy', [])) * 100    # (R+1,)
    pc_dist = d.get('local_test_accuracy_dist', [])
    pc = np.array(pc_dist) * 100                                          # (R+1, 4)
    best_round = int(avg_curve.argmax())
    avg_best = float(avg_curve[best_round])
    avg_last = float(avg_curve[-1])
    per_domain_at_best = pc[best_round]                                   # (4,)
    per_domain_last = pc[-1]
    return {
        'best_round': best_round,
        'avg_best': avg_best,
        'avg_last': avg_last,
        'per_domain_at_best': per_domain_at_best,
        'per_domain_last': per_domain_last,
    }


# ---------------------------------------------------
print('=' * 70)
print('EXP-123 Stage A v2 — paper-standard per-domain metric')
print('=' * 70)

records = find()
for method, sm in records.items():
    print(f'{method}: {sorted(sm.keys())}')

print('\n--- Per-seed detail (paper-standard) ---')
all_stats = {}
for method, sm in records.items():
    all_stats[method] = []
    for s in SEEDS:
        if s not in sm: continue
        d = load(os.path.join(RECORD_DIR, sm[s]))
        stat = analyze(d)
        all_stats[method].append(stat)
        print(f'  {method}_s{s}:  R_best={stat["best_round"]:3d}  '
              f'AVG Best={stat["avg_best"]:.2f}  Last={stat["avg_last"]:.2f}')
        pd_str = '  '.join(f'{DOMAINS[i]}={stat["per_domain_at_best"][i]:.2f}'
                           for i in range(4))
        print(f'    per-domain @ best:  {pd_str}')

print('\n--- 3-seed mean ± std (paper-standard) ---')
print(f'{"":14s} {"AVG Best":>10s} {"AVG Last":>10s} '
      f'{"Art":>10s} {"Cartoon":>10s} {"Photo":>10s} {"Sketch":>10s}')
print('-' * 80)
summary = {}
for method, stats in all_stats.items():
    if len(stats) < 3: continue
    avg_best = np.array([s['avg_best'] for s in stats])
    avg_last = np.array([s['avg_last'] for s in stats])
    pd_best = np.stack([s['per_domain_at_best'] for s in stats])    # (3, 4)
    pd_last = np.stack([s['per_domain_last'] for s in stats])

    line_mean = f'  {method}_mean  {avg_best.mean():>10.2f} {avg_last.mean():>10.2f}  '
    line_mean += '  '.join(f'{pd_best[:, i].mean():>8.2f}' for i in range(4))
    line_std = f'  {method}_±std   {avg_best.std():>10.2f} {avg_last.std():>10.2f}  '
    line_std += '  '.join(f'{pd_best[:, i].std():>8.2f}' for i in range(4))
    print(line_mean)
    print(line_std)
    summary[method] = {
        'avg_best_mean': avg_best.mean(),
        'avg_best_std': avg_best.std(),
        'per_domain_best_mean': pd_best.mean(axis=0),
        'per_domain_best_std': pd_best.std(axis=0),
        'per_domain_last_mean': pd_last.mean(axis=0),
        'per_domain_last_std': pd_last.std(axis=0),
    }

# --- Δ (orth_only − FDSE) per domain ---
if 'orth_only' in summary and 'fdse' in summary:
    print('\n--- Δ (orth_only − FDSE) per-domain at AVG Best round ---')
    oo = summary['orth_only']['per_domain_best_mean']
    fd = summary['fdse']['per_domain_best_mean']
    oo_std = summary['orth_only']['per_domain_best_std']
    fd_std = summary['fdse']['per_domain_best_std']
    print(f'{"Domain":10s} {"orth_only":>18s} {"FDSE":>18s} {"Δ":>10s}')
    for i, dom in enumerate(DOMAINS):
        d = oo[i] - fd[i]
        sign = '✅' if d > 0 else ('❌' if d < 0 else '=')
        print(f'{dom:10s} {oo[i]:>8.2f} ± {oo_std[i]:<5.2f}   '
              f'{fd[i]:>8.2f} ± {fd_std[i]:<5.2f}   {d:>+7.2f} {sign}')
    print(f'\n  AVG Best: orth_only {summary["orth_only"]["avg_best_mean"]:.2f} '
          f'vs FDSE {summary["fdse"]["avg_best_mean"]:.2f}  '
          f'Δ = {summary["orth_only"]["avg_best_mean"] - summary["fdse"]["avg_best_mean"]:+.2f}')
