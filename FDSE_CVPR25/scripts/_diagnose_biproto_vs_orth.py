"""为什么 BiProto disentanglement 更好但 accuracy 没涨? 诊断脚本.

输出:
  1. per-domain accuracy (BiProto Office s=333 vs orth_only Office s=333)
  2. probe ladder 4 方向 × 3 容量 (linear / MLP-64 / MLP-256)
"""
import sys, json, os, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

FDSE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FDSE_ROOT))

import flgo

CKPTS = {
    'BiProto':   '/home/lry/fl_checkpoints/feddsa_s333_R200_best137_1777085380',
    'orth_only': '/home/lry/fl_checkpoints/feddsa_s333_R200_best155_1776428179',
}
TASK = './task/office_caltech10_c4'
DOMAIN_NAMES = ['Amazon', 'Caltech', 'DSLR', 'Webcam']  # client_id 0..3 (按 task gen 顺序)
GPU = 0


def load_runner(algo_name, is_biproto):
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
    # client_<i>.pt
    for cid, c in enumerate(server.clients):
        cf = ck / f'client_{cid}.pt'
        if cf.is_file():
            c.model.load_state_dict(torch.load(cf, map_location=device), strict=False)
        c.model = c.model.to(device)


def collect_features(client, is_biproto):
    """Returns z_sem, z_sty, y, plus eval-time logits accuracy on this client.

    严格用 test_data (验证 acc), 如果不存在则报错 (避免 fallback to train 出现 train acc 误导).
    """
    device = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
    test_data = getattr(client, 'test_data', None)
    if test_data is None or len(test_data) == 0:
        # 兜底: 使用 client.val_data (split=test_holdout) 这是 client 端的 holdout test
        test_data = getattr(client, 'val_data', None)
    if test_data is None or len(test_data) == 0:
        raise RuntimeError(f"client {client.id} has no test_data/val_data; cannot evaluate test acc")
    print(f"    [client {client.id}] using test_data of size {len(test_data)}", flush=True)
    loader = client.calculator.get_dataloader(test_data, batch_size=client.batch_size)
    z_sem_list, z_sty_list, y_list, correct = [], [], [], 0
    total = 0
    client.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = client.calculator.to_device(batch)
            x, y = batch[0], batch[-1]
            if is_biproto:
                pooled, taps = client.model.encode_with_taps(x)
                z_sem = client.model.semantic_head(pooled)
                z_sty = client.model.encoder_sty(taps)
            else:
                h = client.model.encode(x)
                z_sem = client.model.get_semantic(h)
                z_sty = client.model.get_style(h)
            logits = client.model.head(z_sem)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            z_sem_list.append(z_sem.cpu())
            z_sty_list.append(z_sty.cpu())
            y_list.append(y.cpu())
    return (torch.cat(z_sem_list).numpy(),
            torch.cat(z_sty_list).numpy(),
            torch.cat(y_list).numpy(),
            correct / max(total, 1) * 100)


def probe_ladder(X, y, label):
    """Linear + MLP-64 + MLP-256 train/test acc on 80/20 split."""
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None)
    out = {}
    for name, clf in [
        ('linear', LogisticRegression(max_iter=500)),
        ('mlp_64', MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)),
        ('mlp_256', MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)),
    ]:
        try:
            clf.fit(X_tr, y_tr)
            te = clf.score(X_te, y_te)
        except Exception as e:
            te = float('nan')
        out[name] = te
    print(f"  {label}: linear={out['linear']:.3f}, mlp_64={out['mlp_64']:.3f}, mlp_256={out['mlp_256']:.3f}")
    return out


def run_diag(name, ckpt_dir, is_biproto):
    print(f"\n=== {name} ===")
    print(f"ckpt: {ckpt_dir}")
    runner = load_runner(name, is_biproto)
    load_ckpt(runner, ckpt_dir, is_biproto)
    Z_sem_all, Z_sty_all, Y_all, D_all = [], [], [], []
    print("--- per-domain accuracy ---")
    for cid, c in enumerate(runner.clients):
        zs, zt, y, acc = collect_features(c, is_biproto)
        dname = DOMAIN_NAMES[cid] if cid < len(DOMAIN_NAMES) else f'C{cid}'
        n = len(y)
        print(f"  Client{cid} ({dname}, n={n}): test_acc = {acc:.2f}%")
        Z_sem_all.append(zs); Z_sty_all.append(zt); Y_all.append(y)
        D_all.append(np.full(n, cid))
    Z_sem = np.vstack(Z_sem_all); Z_sty = np.vstack(Z_sty_all)
    Y = np.concatenate(Y_all); D = np.concatenate(D_all)
    print(f"\n--- probe ladder ({len(Y)} samples) ---")
    print("  z_sem -> class:")
    sem_class = probe_ladder(Z_sem, Y, '    ')
    print("  z_sem -> domain:")
    sem_dom = probe_ladder(Z_sem, D, '    ')
    print("  z_sty -> class:")
    sty_class = probe_ladder(Z_sty, Y, '    ')
    print("  z_sty -> domain:")
    sty_dom = probe_ladder(Z_sty, D, '    ')
    return {
        'sem_class': sem_class, 'sem_dom': sem_dom,
        'sty_class': sty_class, 'sty_dom': sty_dom,
    }


if __name__ == '__main__':
    results = {}
    for name, ckpt in CKPTS.items():
        is_bp = (name == 'BiProto')
        results[name] = run_diag(name, ckpt, is_bp)
    # 写结果到 json
    with open('/home/lry/code/federated-learning/experiments/ablation/EXP-127_biproto_full_r200/probe_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved probe_results.json')
