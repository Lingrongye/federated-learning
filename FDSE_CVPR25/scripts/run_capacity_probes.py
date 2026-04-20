"""Capacity probe sweep + per-domain decomposition.

扩展 run_frozen_probes.py 的 MLP probe, 加 4 种容量 (hidden=16/64/128/256)
验证"probe 高是因为信息真泄漏, 不是 probe 容量上限".

同时对 probe_sty_class 做 per-domain 分解, 看泄漏是否集中在某几个 domain.

Usage: 同 run_frozen_probes.py.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

FDSE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FDSE_ROOT))

import flgo
from algorithm.feddsa_sgpa import FedDSASGPAModel, _resolve_num_classes, _resolve_num_clients


def collect(client, device):
    loader = client.calculator.get_dataloader(client.train_data, batch_size=client.batch_size)
    zs, zt, ys = [], [], []
    client.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = client.calculator.to_device(batch)
            x, y = batch[0], batch[-1]
            h = client.model.encode(x)
            zs.append(client.model.get_semantic(h).cpu())
            zt.append(client.model.get_style(h).cpu())
            ys.append(y.cpu())
    return (torch.cat(zs).numpy(), torch.cat(zt).numpy(), torch.cat(ys).numpy())


def fit_score(clf, X_tr, y_tr, X_te, y_te):
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_tr, y_tr)), float(clf.score(X_te, y_te))


def probe_sweep(X_tr, y_tr, X_te, y_te):
    """Linear + MLP hidden ∈ {16,64,128,256}, 全部 test accuracy."""
    out = {}
    tr, te = fit_score(LogisticRegression(max_iter=500), X_tr, y_tr, X_te, y_te)
    out['linear'] = {'train': tr, 'test': te}
    for h in [16, 64, 128, 256]:
        mlp = MLPClassifier(hidden_layer_sizes=(h,), max_iter=500,
                            random_state=42, early_stopping=True, validation_fraction=0.1)
        tr, te = fit_score(mlp, X_tr, y_tr, X_te, y_te)
        out[f'mlp_{h}'] = {'train': tr, 'test': te}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'[cap] task={args.task}, ckpt={args.ckpt}')

    import algorithm.feddsa_sgpa as algo
    runner = flgo.init(
        task=f'./task/{args.task}', algorithm=algo,
        option={'num_rounds': 0, 'proportion': 1.0, 'seed': 2, 'gpu': [args.gpu],
                'load_mode': '', 'num_parallels': 1},
    )
    server = runner
    clients = server.clients

    ckpt = Path(args.ckpt)
    global_state = torch.load(ckpt / 'global_model.pt', map_location=device)
    server.model.load_state_dict(global_state, strict=False)
    client_states = torch.load(ckpt / 'client_models.pt', map_location=device)
    whitening = torch.load(ckpt / 'whitening.pt', map_location=device)
    for cid, c in enumerate(clients):
        if isinstance(client_states, list) and cid < len(client_states):
            c.model.load_state_dict(client_states[cid], strict=False)
        c.model = c.model.to(device)
        c.mu_global = whitening.get('mu_global', None)
        c.sigma_inv_sqrt = whitening.get('sigma_inv_sqrt', None)
        c.source_mu_k = whitening.get('source_mu_k', None)

    Z_sem, Z_sty, Y, D = [], [], [], []
    for cid, c in enumerate(clients):
        zs, zt, y = collect(c, device)
        Z_sem.append(zs); Z_sty.append(zt); Y.append(y)
        D.append(np.full(len(y), cid))
    Z_sem = np.vstack(Z_sem); Z_sty = np.vstack(Z_sty)
    Y = np.concatenate(Y); D = np.concatenate(D)

    idx_tr, idx_te = train_test_split(np.arange(len(Y)), test_size=0.2,
                                      random_state=42, stratify=D)
    sem_tr, sem_te = Z_sem[idx_tr], Z_sem[idx_te]
    sty_tr, sty_te = Z_sty[idx_tr], Z_sty[idx_te]
    y_tr, y_te = Y[idx_tr], Y[idx_te]
    d_tr, d_te = D[idx_tr], D[idx_te]
    print(f'[cap] total={len(Y)} train={len(y_tr)} test={len(y_te)}')

    results = {}
    # 4 targets × capacity sweep (pooled across domains)
    for name, Xtr, ytr, Xte, yte in [
        ('probe_sem_domain', sem_tr, d_tr, sem_te, d_te),
        ('probe_sty_domain', sty_tr, d_tr, sty_te, d_te),
        ('probe_sem_class',  sem_tr, y_tr, sem_te, y_te),
        ('probe_sty_class',  sty_tr, y_tr, sty_te, y_te),
    ]:
        sweep = probe_sweep(Xtr, ytr, Xte, yte)
        print(f'[cap] {name}: ' + ' '.join(
            f'{k}={v["test"]:.3f}' for k, v in sweep.items()))
        results[name] = sweep

    # Per-domain probe_sty_class: train on all domains except 1, test on held-out domain
    # 目的: 识别"哪个 domain 对类别泄漏贡献最大"
    per_domain = {}
    n_d = len(clients)
    for held in range(n_d):
        tr_mask = (D != held); te_mask = (D == held)
        if te_mask.sum() < 10:
            continue
        # 只用 sty_train pooled 学, 测 held 上的 class acc
        mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                            random_state=42, early_stopping=True, validation_fraction=0.1)
        mlp.fit(Z_sty[tr_mask], Y[tr_mask])
        acc = float(mlp.score(Z_sty[te_mask], Y[te_mask]))
        per_domain[f'holdout_d{held}'] = {'test_acc': acc, 'n_test': int(te_mask.sum())}
        print(f'[cap] per-domain holdout d{held}: sty→class mlp={acc:.3f} (n={te_mask.sum()})')
    results['probe_sty_class_per_domain'] = per_domain

    results['task'] = args.task; results['ckpt'] = str(ckpt)
    results['n_train'] = int(len(y_tr)); results['n_test'] = int(len(y_te))
    results['random_class'] = 1.0 / server.model.num_classes
    results['random_domain'] = 1.0 / n_d

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'[cap] saved to {args.output}')


if __name__ == '__main__':
    main()
