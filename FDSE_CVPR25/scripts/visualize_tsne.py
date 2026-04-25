"""T-SNE visualization of z_sem and z_sty, colored by class AND by domain.

This is the REAL information-level decoupling diagnosis.
- If z_sty clusters by class    → class info still in z_sty (decoupling failed)
- If z_sty clusters by domain   → style info in z_sty (expected)
- If z_sem clusters by class    → class info in z_sem (expected)
- If z_sem clusters by domain   → domain leakage in z_sem (bad)

Usage (on server):
    python visualize_tsne.py \
        --ckpt /root/fl_checkpoints/sgpa_PACS_c4_s2_R200_1776795770/ \
        --task PACS_c4 \
        --out /tmp/tsne_figs --label PACS_s2_vib
"""
import argparse, os, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

FDSE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FDSE_ROOT))

import flgo


def detect_algo(ckpt_dir: Path):
    """Auto-detect algorithm by inspecting ckpt keys.

    Returns (algo_name, vib_flag, is_biproto):
        - 'vib'      : feddsa_sgpa_vib (semantic_head.mu_head.0.weight)
        - 'biproto'  : feddsa_biproto (encoder_sty.net.0.weight + Pc + Pd buffers)
        - 'sgpa'    : feddsa_sgpa (default)
    """
    gs = torch.load(ckpt_dir / 'global_model.pt', map_location='cpu')
    if 'semantic_head.mu_head.0.weight' in gs:
        return 'vib', 1, False
    if 'encoder_sty.net.0.weight' in gs and 'Pd' in gs and 'Pc' in gs:
        return 'biproto', 0, True
    return 'sgpa', 0, False


def collect_features(client, device, is_biproto=False):
    data = client.train_data
    loader = client.calculator.get_dataloader(data, batch_size=client.batch_size)
    z_sem_list, z_sty_list, y_list = [], [], []
    client.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = client.calculator.to_device(batch)
            x, y = batch[0], batch[-1]
            if is_biproto:
                # BiProto: encoder_sty 吃 conv1-3 taps, 不是 pooled feature
                pooled, taps = client.model.encode_with_taps(x)
                z_sem = client.model.semantic_head(pooled)
                z_sty = client.model.encoder_sty(taps)
            else:
                h = client.model.encode(x)
                z_sem = client.model.get_semantic(h)
                z_sty = client.model.get_style(h)
            z_sem_list.append(z_sem.cpu())
            z_sty_list.append(z_sty.cpu())
            y_list.append(y.cpu())
    return (torch.cat(z_sem_list).numpy(),
            torch.cat(z_sty_list).numpy(),
            torch.cat(y_list).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max_samples', type=int, default=3000,
                    help='subsample for speed')
    ap.add_argument('--perplexity', type=float, default=30)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    ckpt_dir = Path(args.ckpt)

    # 1. Detect algo + import
    algo_name, vib_flag, is_biproto = detect_algo(ckpt_dir)
    print(f'[tsne] detected algo={algo_name}, vib={vib_flag}, biproto={is_biproto}')
    if algo_name == 'vib':
        import algorithm.feddsa_sgpa_vib as algo_mod
    elif algo_name == 'biproto':
        import algorithm.feddsa_biproto as algo_mod
    else:
        import algorithm.feddsa_sgpa as algo_mod

    # 2. flgo.init with num_rounds=0
    task_path = f'./task/{args.task}'
    option = {'num_rounds': 0, 'proportion': 1.0, 'seed': 2,
              'gpu': [args.gpu], 'load_mode': '', 'num_parallels': 1}
    if vib_flag:
        option['algo_para'] = [1, 0]  # vib=1 us=0 (default EXP-113 A)
    elif is_biproto:
        # BiProto Server.initialize 会调 init_algo_para with 22 项 dict, 给默认值即可
        option['algo_para'] = [
            1.0, 0.0, 1.0, 0.2, 5, 128, 0, 60, 30, 80, 0, 1.0,  # base 12
            0.5, 0.3, 0.1, 0.5, 50, 80, 150, 0.9, 0, 0,           # BiProto 10
        ]
    runner = flgo.init(task=task_path, algorithm=algo_mod, option=option)
    server = runner
    clients = server.clients

    # 3. Load weights
    global_state = torch.load(ckpt_dir / 'global_model.pt', map_location=device)
    server.model.load_state_dict(global_state, strict=False)
    # client_models.pt 可能不存在 (e.g. S0 gate ckpt), 跳过即可
    client_ckpt = ckpt_dir / 'client_models.pt'
    if client_ckpt.is_file():
        client_states = torch.load(client_ckpt, map_location=device)
        for cid, c in enumerate(clients):
            if isinstance(client_states, list) and cid < len(client_states):
                c.model.load_state_dict(client_states[cid], strict=False)
            c.model = c.model.to(device)
    else:
        # 兜底: 个别 ckpt 用 client_<i>.pt 单文件存
        for cid, c in enumerate(clients):
            cf = ckpt_dir / f'client_{cid}.pt'
            if cf.is_file():
                c.model.load_state_dict(torch.load(cf, map_location=device), strict=False)
            c.model = c.model.to(device)
    print(f'[tsne] loaded {len(clients)} client models')

    # 4. Collect features per client
    Z_sem, Z_sty, Y, D = [], [], [], []
    for cid, c in enumerate(clients):
        zs, zt, y = collect_features(c, device, is_biproto=is_biproto)
        Z_sem.append(zs); Z_sty.append(zt); Y.append(y)
        D.append(np.full(len(y), cid))
        print(f'[tsne] client {cid}: {len(y)} samples')
    Z_sem = np.vstack(Z_sem); Z_sty = np.vstack(Z_sty)
    Y = np.concatenate(Y); D = np.concatenate(D)

    # 5. Subsample for t-SNE speed
    N = len(Y)
    if N > args.max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, size=args.max_samples, replace=False)
        Z_sem = Z_sem[idx]; Z_sty = Z_sty[idx]
        Y = Y[idx]; D = D[idx]
    print(f'[tsne] running t-SNE on {len(Y)} samples...')

    # 6. t-SNE
    tsne_kwargs = dict(n_components=2, perplexity=args.perplexity,
                       random_state=42, init='pca', learning_rate='auto')
    E_sem = TSNE(**tsne_kwargs).fit_transform(Z_sem)
    print('[tsne] sem done')
    E_sty = TSNE(**tsne_kwargs).fit_transform(Z_sty)
    print('[tsne] sty done')

    # 7. 4-panel plot (z_sem by class, z_sem by domain, z_sty by class, z_sty by domain)
    n_classes = int(Y.max() + 1)
    n_domains = int(D.max() + 1)
    class_cmap = plt.cm.tab10  # up to 10 classes
    domain_cmap = plt.cm.Set1

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, E, C, n_c, cmap, title in [
        (axes[0, 0], E_sem, Y, n_classes, class_cmap,
         f'[{args.label}] z_sem by class'),
        (axes[0, 1], E_sem, D, n_domains, domain_cmap,
         f'[{args.label}] z_sem by domain'),
        (axes[1, 0], E_sty, Y, n_classes, class_cmap,
         f'[{args.label}] z_sty by class  ⚠️ 看这张'),
        (axes[1, 1], E_sty, D, n_domains, domain_cmap,
         f'[{args.label}] z_sty by domain'),
    ]:
        for c in range(n_c):
            mask = C == c
            ax.scatter(E[mask, 0], E[mask, 1], c=[cmap(c)], s=6, alpha=0.7,
                       label=f'{c}', edgecolors='none')
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='best', fontsize=7, markerscale=2, framealpha=0.5)

    plt.tight_layout()
    png = os.path.join(args.out, f'tsne_{args.label}.png')
    plt.savefig(png, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'[tsne] saved {png}')

    # 8. Save raw embeddings for later analysis
    np.savez(os.path.join(args.out, f'tsne_{args.label}.npz'),
             E_sem=E_sem, E_sty=E_sty, Y=Y, D=D,
             Z_sem_orig=Z_sem, Z_sty_orig=Z_sty)


if __name__ == '__main__':
    main()
