"""DualEnc decoupling probes — 4-probe + diagnostic metrics.

Offline: 读取 ckpt, 跑 4 个 MLP probe + 算 SVD ER + 风格多样性.

P1: z_sty -> class      (期望 < 25%, 风险 > 50% 类别泄漏)
P2: z_sty -> domain     (期望 > 80%, 风险 < 50% 风格码没学到域)
P3: z_sem -> domain     (期望 < 50%, 风险 > 80% 语义码偷风格)
P4: z_sem -> class      (期望 > 60%, 健康检查)

Usage:
    python scripts/dualenc_probes.py \\
        --ckpt task/Office_c4/.../checkpoint.pt \\
        --task Office_c4 \\
        --out_json experiments/ablation/EXP-128_.../probes_R200.json
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 让 import algorithm 可用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect_features(model, loader, device, max_samples=None):
    """Forward 一遍, 收集 z_sem / z_sty (mu) / labels / domain_id.

    domain_id 假设 dataloader 输出 (x, y, domain_id) 三元组.
    若只输出 (x, y), 则使用单一 domain_id=0 (单 client probe 时调用方控制).
    """
    model.eval()
    z_sems, z_stys, labels, domains = [], [], [], []
    n = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, d = batch
            else:
                x, y = batch[0], batch[-1]
                d = torch.zeros_like(y)
            x = x.to(device)
            h = model.encode(x)
            z_sem = model.get_semantic(h).cpu()
            mu, _ = model.get_style(h)
            z_sty = mu.cpu()
            z_sems.append(z_sem)
            z_stys.append(z_sty)
            labels.append(y.cpu() if torch.is_tensor(y) else torch.tensor(y))
            domains.append(d.cpu() if torch.is_tensor(d) else torch.tensor(d))
            n += x.size(0)
            if max_samples is not None and n >= max_samples:
                break
    return (
        torch.cat(z_sems),
        torch.cat(z_stys),
        torch.cat(labels),
        torch.cat(domains),
    )


def train_probe_mlp(features, targets, num_classes, epochs=10, hidden=256, lr=1e-3,
                    val_split=0.2, seed=0):
    """Train a 2-layer MLP probe and return (val_acc, train_acc).
    冻住 features, 不参与梯度.
    """
    g = torch.Generator().manual_seed(seed)
    N = features.size(0)
    perm = torch.randperm(N, generator=g)
    n_val = int(N * val_split)
    if n_val < 1 or N - n_val < 1:
        # 数据太少, 直接训练全集 + 报训练 acc
        n_val = 0

    if n_val > 0:
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
    else:
        val_idx = perm
        tr_idx = perm

    X_tr, y_tr = features[tr_idx], targets[tr_idx]
    X_val, y_val = features[val_idx], targets[val_idx]

    in_dim = features.size(1)
    probe = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, num_classes),
    )
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    bs = min(64, X_tr.size(0))

    for ep in range(epochs):
        idx = torch.randperm(X_tr.size(0))
        for i in range(0, X_tr.size(0), bs):
            sel = idx[i: i + bs]
            xb, yb = X_tr[sel], y_tr[sel]
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        val_pred = probe(X_val).argmax(dim=1)
        val_acc = (val_pred == y_val).float().mean().item()
        tr_pred = probe(X_tr).argmax(dim=1)
        tr_acc = (tr_pred == y_tr).float().mean().item()
    return val_acc, tr_acc


def effective_rank(features, eps=1e-9):
    """SVD effective rank = exp(entropy of normalized singular values).
    等价 'participation ratio of singular values'.
    """
    if features.size(0) < 2:
        return 0.0
    # Center
    F_centered = features - features.mean(dim=0, keepdim=True)
    U, S, Vt = torch.linalg.svd(F_centered, full_matrices=False)
    s = S.cpu().numpy()
    s = s / (s.sum() + eps)
    # Shannon entropy
    H = -(s * np.log(s + eps)).sum()
    return float(np.exp(H))


def domain_distance_stats(z_sty, domains):
    """Within-domain vs across-domain mean L2 distance."""
    z_sty = F.normalize(z_sty, dim=1)
    unique_domains = torch.unique(domains)
    intra = []
    inter = []
    for d in unique_domains:
        mask_in = (domains == d)
        mask_out = ~mask_in
        z_in = z_sty[mask_in]
        z_out = z_sty[mask_out]
        if z_in.size(0) >= 2:
            # intra: mean pairwise distance
            d_in = torch.cdist(z_in, z_in)
            n = z_in.size(0)
            tri_idx = torch.triu_indices(n, n, offset=1)
            intra.append(d_in[tri_idx[0], tri_idx[1]].mean().item())
        if z_in.size(0) >= 1 and z_out.size(0) >= 1:
            d_io = torch.cdist(z_in, z_out)
            inter.append(d_io.mean().item())
    return {
        'intra_domain_mean': float(np.mean(intra)) if intra else 0.0,
        'inter_domain_mean': float(np.mean(inter)) if inter else 0.0,
        'inter_intra_ratio': float(np.mean(inter) / (np.mean(intra) + 1e-6)) if intra and inter else 0.0,
    }


def run_4_probes(model, loader, device, num_classes, num_domains,
                 max_samples=2000, probe_epochs=10, seed=0):
    """Top-level entry: collect features and run all 4 probes + diagnostics."""
    z_sem, z_sty, labels, domains = collect_features(
        model, loader, device, max_samples=max_samples,
    )

    results = {
        'n_samples': z_sem.size(0),
        'sem_dim': z_sem.size(1),
        'sty_dim': z_sty.size(1),
    }

    # P1: z_sty -> class
    p1_val, p1_tr = train_probe_mlp(z_sty, labels, num_classes, epochs=probe_epochs, seed=seed)
    results['P1_z_sty_class_val'] = p1_val
    results['P1_z_sty_class_train'] = p1_tr

    # P2: z_sty -> domain
    if num_domains > 1:
        p2_val, p2_tr = train_probe_mlp(z_sty, domains, num_domains, epochs=probe_epochs, seed=seed)
        results['P2_z_sty_domain_val'] = p2_val
        results['P2_z_sty_domain_train'] = p2_tr
    else:
        results['P2_z_sty_domain_val'] = None
        results['P2_z_sty_domain_train'] = None

    # P3: z_sem -> domain
    if num_domains > 1:
        p3_val, p3_tr = train_probe_mlp(z_sem, domains, num_domains, epochs=probe_epochs, seed=seed)
        results['P3_z_sem_domain_val'] = p3_val
        results['P3_z_sem_domain_train'] = p3_tr
    else:
        results['P3_z_sem_domain_val'] = None
        results['P3_z_sem_domain_train'] = None

    # P4: z_sem -> class
    p4_val, p4_tr = train_probe_mlp(z_sem, labels, num_classes, epochs=probe_epochs, seed=seed)
    results['P4_z_sem_class_val'] = p4_val
    results['P4_z_sem_class_train'] = p4_tr

    # Diagnostics
    results['z_sem_eff_rank'] = effective_rank(z_sem)
    results['z_sty_eff_rank'] = effective_rank(z_sty)
    results['style_dist'] = domain_distance_stats(z_sty, domains)

    # 期望阈值检查 (只标记是否达标, 不强制 fail)
    results['health_check'] = {
        'P1_z_sty_class_pass': p1_val < 0.25,
        'P2_z_sty_domain_pass': (results['P2_z_sty_domain_val'] or 0) > 0.80 if num_domains > 1 else None,
        'P3_z_sem_domain_pass': (results['P3_z_sem_domain_val'] or 1) < 0.50 if num_domains > 1 else None,
        'P4_z_sem_class_pass': p4_val > 0.60,
        'z_sty_eff_rank_pass': results['z_sty_eff_rank'] > 10,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to .pt checkpoint')
    parser.add_argument('--task', required=True, help='e.g. PACS_c4 or office_caltech10_c4')
    parser.add_argument('--out_json', required=True)
    parser.add_argument('--max_samples', type=int, default=2000)
    parser.add_argument('--probe_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # 简化: 这里只示范 API. 实际 ckpt loading + dataloader 需要按 flgo 训练存档格式适配,
    # 通常做法是在 logger.PerRunLogger 里 hook 调用 collect_features+run_4_probes,
    # 而不是离线读 ckpt (flgo 不存独立 model state_dict).
    raise NotImplementedError(
        '此脚本预留为入口. 实际 probe 调用应在 PerRunLogger 中每 N round hook,\n'
        '直接拿 server.model + client val_loader 调 run_4_probes(). 示例集成代码见 NOTE.md.'
    )


if __name__ == '__main__':
    main()
