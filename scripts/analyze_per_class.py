# -*- coding: utf-8 -*-
"""EXP-123 Stage B per-class 7×4 matrix 分析.

输入: fedbn/feddsa_scheduled/fdse × 3 seeds R=200 record JSON (含 diag hook 数据).
输出: 每 method 的 4 domain × 7 class matrix (3-seed mean), 找 hard cells.

Metric: per_class_test_acc_dist[best_round][client_id][class_id]
best_round = mean_local_test_accuracy.argmax()  (paper standard)
"""
import os, json, sys, io, glob
import numpy as np

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DOMAINS = ['Art', 'Cartoon', 'Photo', 'Sketch']
CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
SEEDS = [2, 15, 333]

# Local record dirs (pulled or mirrored)
SC2_REC = r"D:\桌面文件\联邦学习\FDSE_CVPR25\task\PACS_c4\record"
# Remote fdse records need ssh or assume mirrored. For now, analyze both combined.


def find_record(method: str, seed: int, rec_dir: str):
    """Find the R=200 DiagLogger JSON for this method+seed."""
    candidates = []
    for fn in sorted(os.listdir(rec_dir)):
        if not fn.endswith('.json'): continue
        if 'DiagLogger' not in fn: continue
        if f'_S{seed}_' not in fn: continue
        if '_R200_' not in fn: continue

        if method == 'fedbn':
            if fn.startswith('fedbn_Mdefault_model_'):
                candidates.append(fn)
        elif method == 'orth_only':
            # feddsa_scheduled with lo1.0 sm=0 (orth_only mode)
            if 'feddsa_scheduled_lo1.0' in fn and '_sm0_' in fn:
                candidates.append(fn)
        elif method == 'fdse':
            if fn.startswith('fdse_lmbd0.5_tau0.5_'):
                candidates.append(fn)
    if not candidates: return None
    # Take shortest filename (no _bak etc)
    return sorted(candidates, key=len)[0]


def analyze_run(fp: str):
    """Extract best_round + per-class matrix at that round + confidence stats."""
    d = json.load(open(fp))
    avg = np.array(d.get('mean_local_test_accuracy', [])) * 100
    if len(avg) == 0: return None

    best_round = int(avg.argmax())
    # per_class_test_acc_dist: list[round][client][class]
    pcl = d.get('per_class_test_acc_dist', [])
    if len(pcl) <= best_round:
        return None
    pc_at_best = np.array(pcl[best_round])  # shape (4, 7)

    # per-client overall at best
    pc_overall = np.array(d.get('local_test_accuracy_dist', [])) * 100
    pc_overall_at_best = pc_overall[best_round] if len(pc_overall) > best_round else None

    # confidence stats at best
    cs = d.get('confidence_stats_dist', [])
    cs_at_best = cs[best_round] if len(cs) > best_round else None

    return {
        'avg_best': float(avg.max()),
        'best_round': best_round,
        'per_class_matrix': pc_at_best * 100 if pc_at_best.max() <= 1.0 else pc_at_best,  # ensure %
        'per_domain': pc_overall_at_best,  # shape (4,)
        'conf_stats': cs_at_best,
    }


def main():
    # Check if sc2 records are accessible locally (they're on server, need to pull)
    if not os.path.exists(SC2_REC):
        print(f"ERROR: local record dir missing: {SC2_REC}")
        return
    available = os.listdir(SC2_REC)
    print(f"Found {len(available)} files in local record dir (sc2 pulled copy)")

    all_results = {}  # {method: {seed: result}}
    for method in ['fedbn', 'orth_only', 'fdse']:
        all_results[method] = {}
        for seed in SEEDS:
            fn = find_record(method, seed, SC2_REC)
            if fn is None:
                print(f"  {method}_s{seed}: NOT FOUND (pull from remote?)")
                continue
            res = analyze_run(os.path.join(SC2_REC, fn))
            if res is None:
                print(f"  {method}_s{seed}: EMPTY record")
                continue
            all_results[method][seed] = res
            print(f"  {method}_s{seed}: R_best={res['best_round']:3d}  "
                  f"AVG={res['avg_best']:.2f}  "
                  f"Art={res['per_domain'][0]:.2f}  "
                  f"Cart={res['per_domain'][1]:.2f}  "
                  f"Photo={res['per_domain'][2]:.2f}  "
                  f"Sketch={res['per_domain'][3]:.2f}")

    # ------ Per-class 7x4 matrix per method (3-seed mean) ------
    print('\n' + '=' * 80)
    print('PER-CLASS 7x4 MATRIX (3-seed mean, at best_round)')
    print('=' * 80)

    for method, seed_dict in all_results.items():
        if len(seed_dict) == 0: continue
        mats = np.stack([r['per_class_matrix'] for r in seed_dict.values()])  # (n_seeds, 4, 7)
        mean_mat = mats.mean(axis=0)
        std_mat = mats.std(axis=0)

        print(f"\n--- {method} ({len(seed_dict)} seeds) ---")
        header = f"{'':10s} " + ' '.join(f'{c:>8s}' for c in CLASSES) + '  | per-dom'
        print(header)
        print('-' * len(header))
        for d_idx, domain in enumerate(DOMAINS):
            row = f"{domain:10s} "
            for c_idx in range(7):
                row += f'{mean_mat[d_idx, c_idx]:>8.2f}'
            dom_mean = mean_mat[d_idx].mean()
            row += f'  | {dom_mean:.2f}'
            print(row)
        # per-class mean over domains
        cls_mean = mean_mat.mean(axis=0)
        print(f"{'per-cls':10s} " + ' '.join(f'{v:>8.2f}' for v in cls_mean))

    # ------ 对比: 找 hard cells ------
    if 'fdse' in all_results and len(all_results['fdse']) > 0 and \
       'fedbn' in all_results and len(all_results['fedbn']) > 0:
        print('\n' + '=' * 80)
        print('DELTA: FDSE - FedBN (per-cell, 3-seed mean)')
        print('=' * 80)
        fdse_mean = np.stack([r['per_class_matrix'] for r in all_results['fdse'].values()]).mean(axis=0)
        fedbn_mean = np.stack([r['per_class_matrix'] for r in all_results['fedbn'].values()]).mean(axis=0)
        delta = fdse_mean - fedbn_mean
        header = f"{'':10s} " + ' '.join(f'{c:>8s}' for c in CLASSES)
        print(header)
        print('-' * len(header))
        for d_idx, domain in enumerate(DOMAINS):
            row = f"{domain:10s} "
            for c_idx in range(7):
                v = delta[d_idx, c_idx]
                mark = '+' if v > 0 else ''
                row += f'{mark}{v:>7.2f}'
            print(row)

        print('\nTop 10 hard cells (low FedBN acc):')
        flat_idx = np.argsort(fedbn_mean.flatten())[:10]
        for idx in flat_idx:
            d_idx, c_idx = idx // 7, idx % 7
            print(f'  ({DOMAINS[d_idx]}, {CLASSES[c_idx]}): FedBN={fedbn_mean[d_idx,c_idx]:.1f} '
                  f'FDSE={fdse_mean[d_idx,c_idx]:.1f} Δ={delta[d_idx,c_idx]:+.1f}')

    # ------ Confidence stats comparison ------
    print('\n' + '=' * 80)
    print('CONFIDENCE STATS (3-seed mean, Client 0 = Art domain)')
    print('=' * 80)
    for method, seed_dict in all_results.items():
        if len(seed_dict) == 0: continue
        # cs_at_best is list of 4 dicts (per client)
        clients_stats = [[r['conf_stats'][c_idx] for r in seed_dict.values() if r['conf_stats'] and len(r['conf_stats']) > c_idx] for c_idx in range(4)]
        print(f"\n--- {method} ---")
        for c_idx, dom in enumerate(DOMAINS):
            if not clients_stats[c_idx]: continue
            keys = clients_stats[c_idx][0].keys()
            avg_stats = {k: np.mean([cs[k] for cs in clients_stats[c_idx]]) for k in keys}
            print(f"  {dom}: ECE={avg_stats['ece']:.3f} "
                  f"over_conf_err={avg_stats['over_conf_err_ratio']:.3f} "
                  f"wrong_conf={avg_stats['wrong_conf_mean']:.3f} "
                  f"mean_conf={avg_stats['mean']:.3f}")


if __name__ == '__main__':
    main()
