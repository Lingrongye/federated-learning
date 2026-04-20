"""Frozen post-hoc linear probes for FedDSA-CDANN.

After R200 training, load checkpoint, freeze encoder/heads/whitening.
Train 3 linear probes on frozen TRAIN features, report test accuracy on held-out TEST features.

Three probes (all on post-whitening features):
- probe_sem_domain: z_sem → domain (expected ≈ random 1/N, GRL 成功)
- probe_sty_domain: z_sty → domain (expected ≈ 1.0, 正向监督生效)
- probe_sty_class:  z_sty → class  (KEY: PACS CDANN ≥ 40% vs baseline ≈ 15%)

See FINAL_PROPOSAL.md for claim alignment.

Usage:
    python scripts/run_frozen_probes.py \\
        --ckpt /root/fl_checkpoints/<sgpa_xxx>/ \\
        --task PACS_c4 \\
        --config ./config/pacs/feddsa_cdann_pacs_r200.yml \\
        --output <exp_dir>/probe_result_<seed>.json \\
        --gpu 0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FDSE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FDSE_ROOT))

import flgo
from algorithm.feddsa_sgpa import FedDSASGPAModel, _resolve_num_classes, _resolve_num_clients


def collect_features(client, is_train: bool, device: str):
    """Collect post-whitening z_sem, z_sty from client's data loader.

    Note: flgo.init(num_rounds=0) 可能未触发 test_data split, 所以我们只用
    client.train_data 收集全部 features, 在 sklearn 那边 train_test_split 80/20.
    """
    data = client.train_data
    if data is None:
        raise RuntimeError(f"client {getattr(client, 'id', '?')} train_data is None")
    loader = client.calculator.get_dataloader(
        data, batch_size=client.batch_size
    )
    z_sem_list, z_sty_list, y_list = [], [], []
    client.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = client.calculator.to_device(batch)
            x, y = batch[0], batch[-1]
            h = client.model.encode(x)
            z_sem = client.model.get_semantic(h)
            z_sty = client.model.get_style(h)
            # 注意: 训练时 dom_head 看到的是 pre-whitening z_sem/z_sty
            # (Client.train 直接 dom_head(z_sem), 无 whitening). 所以 probe 也用
            # pre-whitening 保持和训练时特征空间一致.
            z_sem_list.append(z_sem.cpu())
            z_sty_list.append(z_sty.cpu())
            y_list.append(y.cpu())
    Z_sem = torch.cat(z_sem_list, dim=0).numpy()
    Z_sty = torch.cat(z_sty_list, dim=0).numpy()
    Y = torch.cat(y_list, dim=0).numpy()
    return Z_sem, Z_sty, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint dir from se=1 training')
    ap.add_argument('--task', required=True, help='office_caltech10_c4 / PACS_c4')
    ap.add_argument('--config', required=True, help='config yml path')
    ap.add_argument('--output', required=True, help='output JSON path')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    task_path = f'./task/{args.task}'
    print(f'[probe] task={args.task}, ckpt={args.ckpt}')

    # Init flgo task
    import algorithm.feddsa_sgpa as algo
    runner = flgo.init(
        task=task_path,
        algorithm=algo,
        option={'num_rounds': 0, 'proportion': 1.0, 'seed': 2, 'gpu': [args.gpu],
                'load_mode': '', 'num_parallels': 1},
    )
    server = runner
    clients = server.clients

    # Load checkpoint
    ckpt_dir = Path(args.ckpt)
    global_state = torch.load(ckpt_dir / 'global_model.pt', map_location=device)
    server.model.load_state_dict(global_state, strict=False)
    client_states = torch.load(ckpt_dir / 'client_models.pt', map_location=device)
    whitening = torch.load(ckpt_dir / 'whitening.pt', map_location=device)
    # client_models.pt 是 list (按 client index 存), 不是 dict!
    for cid, c in enumerate(clients):
        if isinstance(client_states, list):
            if cid < len(client_states):
                missing, unexpected = c.model.load_state_dict(client_states[cid], strict=False)
                if missing or unexpected:
                    print(f"[probe] client {cid} load warning: missing={len(missing)} unexpected={len(unexpected)}")
        elif isinstance(client_states, dict):
            if str(cid) in client_states:
                c.model.load_state_dict(client_states[str(cid)], strict=False)
        # 把 model 移到 device (load_state_dict 恢复后 weights 可能在 CPU)
        c.model = c.model.to(device)
        # Debug: verify load worked
        if cid == 0:
            import itertools
            p_name, p_val = next(itertools.dropwhile(lambda kv: kv[0] == 'M', c.model.state_dict().items()))
            print(f"[probe] client 0 {p_name[:40]} sum = {p_val.sum().item():.4f}")
        c.mu_global = whitening.get('mu_global', None)
        c.sigma_inv_sqrt = whitening.get('sigma_inv_sqrt', None)
        c.source_mu_k = whitening.get('source_mu_k', None)

    # Collect features per client (only train_data, split 80/20 via sklearn)
    Z_sem_all, Z_sty_all, Y_all, D_all = [], [], [], []
    for cid, c in enumerate(clients):
        print(f'[probe] client {cid}: collecting features...')
        z_sem, z_sty, y = collect_features(c, is_train=True, device=device)
        Z_sem_all.append(z_sem); Z_sty_all.append(z_sty); Y_all.append(y)
        D_all.append(np.full(len(y), cid))

    Z_sem_all = np.vstack(Z_sem_all)
    Z_sty_all = np.vstack(Z_sty_all)
    Y_all = np.concatenate(Y_all)
    D_all = np.concatenate(D_all)

    # Stratified 80/20 split (seed固定确保复现)
    idx_train, idx_test = train_test_split(
        np.arange(len(Y_all)), test_size=0.2, random_state=42, stratify=D_all)
    Z_sem_train = Z_sem_all[idx_train]; Z_sem_test = Z_sem_all[idx_test]
    Z_sty_train = Z_sty_all[idx_train]; Z_sty_test = Z_sty_all[idx_test]
    Y_train = Y_all[idx_train];         Y_test = Y_all[idx_test]
    D_train = D_all[idx_train];         D_test = D_all[idx_test]

    print(f'[probe] total: {len(Y_all)}, train: {len(Y_train)}, test: {len(Y_test)}')

    # Fit 3 probes on train features, evaluate on test
    results = {}
    for name, X_tr, y_tr, X_te, y_te in [
        ('probe_sem_domain', Z_sem_train, D_train, Z_sem_test, D_test),
        ('probe_sty_domain', Z_sty_train, D_train, Z_sty_test, D_test),
        ('probe_sty_class',  Z_sty_train, Y_train, Z_sty_test, Y_test),
    ]:
        clf = LogisticRegression(max_iter=500, multi_class='multinomial')
        clf.fit(X_tr, y_tr)
        train_acc = clf.score(X_tr, y_tr)
        test_acc = clf.score(X_te, y_te)
        print(f'[probe] {name}: train={train_acc:.4f}, test={test_acc:.4f}')
        results[name] = {'train_acc': float(train_acc), 'test_acc': float(test_acc)}

    # Random baselines (for context)
    results['random_baseline_domain'] = 1.0 / len(clients)
    results['random_baseline_class'] = 1.0 / server.model.num_classes
    results['task'] = args.task
    results['ckpt'] = str(ckpt_dir)
    results['n_train'] = int(len(Y_train))
    results['n_test'] = int(len(Y_test))

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'[probe] saved to {args.output}')

    # Pretty-print verdict
    print('\n=== Frozen Probe Verdict ===')
    print(f'probe_sem_domain test: {results["probe_sem_domain"]["test_acc"]:.3f} '
          f'(expect ≈ {results["random_baseline_domain"]:.3f} random)')
    print(f'probe_sty_domain test: {results["probe_sty_domain"]["test_acc"]:.3f} '
          f'(expect → 1.0)')
    print(f'probe_sty_class  test: {results["probe_sty_class"]["test_acc"]:.3f} '
          f'(KEY: PACS ≥ 0.40, Office ~ 0.20-0.30, baseline ≈ {results["random_baseline_class"]:.3f})')


if __name__ == '__main__':
    main()
