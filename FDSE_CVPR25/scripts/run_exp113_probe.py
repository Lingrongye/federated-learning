"""EXP-113 capacity probe — handles both VIB and non-VIB checkpoints.

Usage:
    python scripts/run_exp113_probe.py --ckpt <dir> --task PACS_c4 --output <file>

Auto-detects vib=0/1 from the ckpt's global_model.pt state_dict by checking for
`log_var_head` keys. Constructs the right FedDSAVIBModel variant.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

FDSE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FDSE_ROOT))

import flgo
from algorithm.feddsa_sgpa_vib import FedDSAVIBModel


def detect_vib(ckpt_dir):
    state = torch.load(os.path.join(ckpt_dir, 'global_model.pt'), map_location='cpu')
    return int(any('log_var_head' in k for k in state.keys()))


def collect(client, device):
    loader = client.calculator.get_dataloader(client.train_data, batch_size=client.batch_size)
    zs, zt, ys = [], [], []
    client.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = client.calculator.to_device(batch)
            x, y = batch[0], batch[-1]
            h = client.model.encode(x)
            # For VIB model get_semantic returns mu (eval mode), for non-VIB returns sequential output
            z_sem = client.model.get_semantic(h)
            z_sty = client.model.get_style(h)
            zs.append(z_sem.cpu()); zt.append(z_sty.cpu()); ys.append(y.cpu())
    return (torch.cat(zs).numpy(), torch.cat(zt).numpy(), torch.cat(ys).numpy())


def fit_sweep(X_tr, y_tr, X_te, y_te):
    out = {}
    clf = LogisticRegression(max_iter=500, multi_class='multinomial')
    clf.fit(X_tr, y_tr)
    out['linear'] = {'train': float(clf.score(X_tr, y_tr)),
                     'test': float(clf.score(X_te, y_te))}
    for h in [16, 64, 128, 256]:
        mlp = MLPClassifier(hidden_layer_sizes=(h,), max_iter=500,
                            random_state=42, early_stopping=True, validation_fraction=0.1)
        mlp.fit(X_tr, y_tr)
        out[f'mlp_{h}'] = {'train': float(mlp.score(X_tr, y_tr)),
                           'test': float(mlp.score(X_te, y_te))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--task', required=True, help='PACS_c4 or office_caltech10_c4')
    ap.add_argument('--output', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    vib_flag = detect_vib(args.ckpt)
    print(f'[probe] ckpt={args.ckpt}  auto_detected_vib={vib_flag}')

    # Pick a config aligned with the ckpt (matters for task/model construction)
    if 'office' in args.task.lower():
        cfg_path = str(FDSE_ROOT / 'config' / 'office' /
                       ('feddsa_vib_office_r200.yml' if vib_flag
                        else 'feddsa_orth_uc1_office_r200.yml'))
    else:
        cfg_path = str(FDSE_ROOT / 'config' / 'pacs' /
                       ('feddsa_vib_pacs_r200.yml' if vib_flag
                        else 'feddsa_baseline_pacs_saveckpt_r200.yml'))

    # Use feddsa_sgpa_vib algorithm module for both VIB and non-VIB
    import algorithm.feddsa_sgpa_vib as algo

    runner = flgo.init(
        task=f'./task/{args.task}', algorithm=algo,
        option={'num_rounds': 0, 'proportion': 1.0, 'seed': 2, 'gpu': [args.gpu],
                'load_mode': '', 'num_parallels': 1},
    )
    server = runner
    clients = server.clients

    ckpt_dir = Path(args.ckpt)
    global_state = torch.load(ckpt_dir / 'global_model.pt', map_location=device)
    # load with strict=False to tolerate schema differences
    server.model.load_state_dict(global_state, strict=False)
    client_states = torch.load(ckpt_dir / 'client_models.pt', map_location=device)
    whitening = torch.load(ckpt_dir / 'whitening.pt', map_location=device)
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
    print(f'[probe] N_train={len(y_tr)} N_test={len(y_te)} vib={vib_flag}')

    results = {'vib': vib_flag, 'task': args.task, 'ckpt': str(ckpt_dir)}
    for name, Xtr, ytr, Xte, yte in [
        ('probe_sem_domain', sem_tr, d_tr, sem_te, d_te),
        ('probe_sty_domain', sty_tr, d_tr, sty_te, d_te),
        ('probe_sem_class',  sem_tr, y_tr, sem_te, y_te),
        ('probe_sty_class',  sty_tr, y_tr, sty_te, y_te),
    ]:
        sweep = fit_sweep(Xtr, ytr, Xte, yte)
        print(f'[probe] {name}: ' + ' '.join(
            f'{k}={v["test"]:.3f}' for k, v in sweep.items()))
        results[name] = sweep

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'[probe] saved to {args.output}')


if __name__ == '__main__':
    main()
