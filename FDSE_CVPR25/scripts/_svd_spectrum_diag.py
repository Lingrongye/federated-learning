"""F2DC-inspired diagnostics: feature covariance SVD spectrum + effective rank.

对 BiProto vs orth_only Office s=333 (best round ckpts) 做:
  D-1: z_sem 协方差 SVD spectrum 图 (类似 F2DC Fig1)
  D-2: effective rank = exp(entropy(eigvals_normalized))
  D-3: z_sty 协方差 SVD spectrum

如果 BiProto 的 z_sem 比 orth_only 更 collapsed → 解释 accuracy 下降.
如果 z_sty 完全 collapse 到 4 个点 → 解释 mode collapse trivial.
"""
import sys, json, os, copy
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FDSE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FDSE_ROOT))

import flgo

# 选所有 3 个 seed 的 BiProto + orth_only Office best ckpts (这些都是从 fl_checkpoints 找)
TARGETS = {
    'BiProto_s2':   '/home/lry/fl_checkpoints/feddsa_s2_R200_best64_BIPROTO',  # 占位, 下面 detect
    'BiProto_s15':  '/home/lry/fl_checkpoints/feddsa_s15_R200_bestXX_BIPROTO',
    'BiProto_s333': '/home/lry/fl_checkpoints/feddsa_s333_R200_best137_1777085380',
    'orth_only_s333': '/home/lry/fl_checkpoints/feddsa_s333_R200_best155_1776428179',
}
TASK = './task/office_caltech10_c4'
GPU = 0


def auto_find_ckpts():
    """从 fl_checkpoints 自动 locate BiProto / orth_only Office s=2/15/333 best ckpts.
    BiProto: 含 encoder_sty.net.0.weight; orth_only: 不含."""
    base = '/home/lry/fl_checkpoints'
    found = {}
    for d in sorted(os.listdir(base)):
        full = os.path.join(base, d, 'global_model.pt')
        meta_f = os.path.join(base, d, 'meta.json')
        if not os.path.isfile(full) or not os.path.isfile(meta_f):
            continue
        try:
            m = json.load(open(meta_f))
            if m.get('num_clients') != 4:
                continue
            seed = m.get('seed')
            if seed not in (2, 15, 333):
                continue
            sd = torch.load(full, map_location='cpu')
            if 'head.weight' not in sd:
                continue
            if sd['head.weight'].shape[0] != 10:  # Office only
                continue
            is_bp = 'encoder_sty.net.0.weight' in sd
            algo = 'BiProto' if is_bp else 'orth_only'
            label = f'{algo}_s{seed}'
            # 选 best_avg_acc 高的覆盖
            cur = found.get(label, (-1, None))
            if m.get('best_avg_acc', 0) > cur[0]:
                found[label] = (m.get('best_avg_acc', 0), os.path.join(base, d))
        except Exception as e:
            continue
    return {k: v[1] for k, v in found.items() if v[1]}


def load_runner(is_biproto):
    if is_biproto:
        import algorithm.feddsa_biproto as algo_mod
        ap = [1.0, 0.0, 1.0, 0.2, 5, 128, 0, 60, 30, 80, 0, 1.0,
              0.5, 0.3, 0.1, 0.5, 50, 80, 150, 0.9, 0, 0]
    else:
        import algorithm.feddsa_scheduled as algo_mod
        ap = None
    option = {'num_rounds': 0, 'proportion': 1.0, 'seed': 2,
              'gpu': [GPU], 'load_mode': '', 'num_parallels': 1}
    if ap:
        option['algo_para'] = ap
    return flgo.init(task=TASK, algorithm=algo_mod, option=option)


def load_ckpt(server, ckpt_dir, is_biproto):
    device = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
    ck = Path(ckpt_dir)
    gs = torch.load(ck / 'global_model.pt', map_location=device)
    server.model.load_state_dict(gs, strict=False)
    for c in server.clients:
        if c.model is None:
            c.model = copy.deepcopy(server.model)
    for cid, c in enumerate(server.clients):
        cf = ck / f'client_{cid}.pt'
        if cf.is_file():
            c.model.load_state_dict(torch.load(cf, map_location=device), strict=False)
        c.model = c.model.to(device)


def collect_features(client, is_biproto):
    """从 client.test_data + train_data 共同收集 features (max samples for SVD stability)."""
    device = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
    pools = []
    for attr in ['train_data', 'val_data', 'test_data']:
        ds = getattr(client, attr, None)
        if ds is not None and len(ds) > 0:
            pools.append(ds)
    z_sem_list, z_sty_list = [], []
    client.model.eval()
    with torch.no_grad():
        for ds in pools:
            loader = client.calculator.get_dataloader(ds, batch_size=client.batch_size)
            for batch in loader:
                batch = client.calculator.to_device(batch)
                x = batch[0]
                if is_biproto:
                    pooled, taps = client.model.encode_with_taps(x)
                    z_sem = client.model.semantic_head(pooled)
                    z_sty = client.model.encoder_sty(taps)
                else:
                    h = client.model.encode(x)
                    z_sem = client.model.get_semantic(h)
                    z_sty = client.model.get_style(h)
                z_sem_list.append(z_sem.cpu())
                z_sty_list.append(z_sty.cpu())
    return torch.cat(z_sem_list).numpy(), torch.cat(z_sty_list).numpy()


def svd_diagnostics(Z, name):
    """Returns: singular values (sorted desc), effective rank.
    effective rank = exp(entropy(p)), p = svals_squared / sum(svals_squared) (Roy & Vetterli 2007).
    """
    Z_centered = Z - Z.mean(axis=0, keepdims=True)
    # 协方差矩阵 = Z_centered.T @ Z_centered / (n-1)
    n = Z_centered.shape[0]
    cov = Z_centered.T @ Z_centered / max(n - 1, 1)
    svals = np.linalg.svd(cov, compute_uv=False)
    # effective rank
    eigs = svals  # cov 是 PSD, svd = eig
    p = eigs / max(eigs.sum(), 1e-12)
    p = p[p > 1e-12]
    H = -np.sum(p * np.log(p))
    eff_rank = np.exp(H)
    print(f'  [{name}] dim={Z.shape[1]}, n_samples={n}, '
          f'svals top5={[f"{v:.3f}" for v in svals[:5]]}, '
          f'tail (last 10): {[f"{v:.4f}" for v in svals[-10:]]}, '
          f'effective_rank={eff_rank:.2f} / {Z.shape[1]} ({eff_rank/Z.shape[1]*100:.1f}%)')
    return svals, eff_rank


def main():
    OUT = '/home/lry/code/federated-learning/experiments/ablation/EXP-127_biproto_full_r200/figs'
    os.makedirs(OUT, exist_ok=True)
    targets = auto_find_ckpts()
    print('=== Located ckpts ===')
    for k, v in sorted(targets.items()):
        print(f'  {k}: {v}')

    results = {}
    for label, ckpt in sorted(targets.items()):
        is_bp = label.startswith('BiProto')
        print(f'\n=== {label} ===')
        runner = load_runner(is_bp)
        load_ckpt(runner, ckpt, is_bp)
        Z_sem_all, Z_sty_all = [], []
        for cid, c in enumerate(runner.clients):
            zs, zt = collect_features(c, is_bp)
            Z_sem_all.append(zs); Z_sty_all.append(zt)
        Z_sem = np.vstack(Z_sem_all); Z_sty = np.vstack(Z_sty_all)
        sv_sem, er_sem = svd_diagnostics(Z_sem, f'{label} z_sem')
        sv_sty, er_sty = svd_diagnostics(Z_sty, f'{label} z_sty')
        results[label] = {
            'sv_sem': sv_sem.tolist(), 'er_sem': float(er_sem),
            'sv_sty': sv_sty.tolist(), 'er_sty': float(er_sty),
        }

    # 画 spectrum 对比图 (类似 F2DC Fig1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # 排序 BiProto 在前 (用蓝), orth_only 用红
    colors = {'BiProto': 'tab:blue', 'orth_only': 'tab:red'}
    styles = {'s2': '-', 's15': '--', 's333': ':'}
    for label, r in sorted(results.items()):
        algo, seed = ('BiProto', label.split('_s')[1]) if 'BiProto' in label else ('orth_only', label.split('_s')[1])
        c = colors.get(algo, 'gray'); s = styles.get(f's{seed}', '-')
        sv_sem = np.array(r['sv_sem']); sv_sty = np.array(r['sv_sty'])
        # log-scale rank-ordered singular values
        axes[0].plot(range(1, len(sv_sem)+1), sv_sem, color=c, ls=s,
                     label=f'{label} ER={r["er_sem"]:.1f}', alpha=0.8)
        axes[1].plot(range(1, len(sv_sty)+1), sv_sty, color=c, ls=s,
                     label=f'{label} ER={r["er_sty"]:.1f}', alpha=0.8)
    for ax, title in [(axes[0], 'z_sem covariance singular values (lower tail = collapse)'),
                      (axes[1], 'z_sty covariance singular values')]:
        ax.set_yscale('log')
        ax.set_xlabel('rank index (1=largest)')
        ax.set_ylabel('singular value (log)')
        ax.set_title(title)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(OUT, 'svd_spectrum_biproto_vs_orth_office.png')
    plt.savefig(fig_path, dpi=110, bbox_inches='tight')
    print(f'\n[svd_diag] saved {fig_path}')

    # 写 json
    json_path = os.path.join(OUT, 'svd_spectrum_results.json')
    with open(json_path, 'w') as f:
        json.dump({k: {kk: vv if not isinstance(vv, list) else vv[:30] + ['...'] + vv[-10:]
                       for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f'[svd_diag] saved {json_path}')


if __name__ == '__main__':
    main()
