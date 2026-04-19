"""
EXP-099: SGPA 推理独立 script — 测 Layer 3 完整 13 指标 + proto_vs_etf_gain

目的:
    flgo 默认 test 走 model.forward() = 直接 ETF argmax, 绕过 test_with_sgpa.
    本 script 加载训练好的 checkpoint (由 feddsa_sgpa se=1 保存), 跑完整 SGPA 推理
    (双 gate + top-m proto + ETF fallback), 报 proto_vs_etf_gain 回答 C3 主 claim.

用法:
    python run_sgpa_inference.py --ckpt ~/fl_checkpoints/sgpa_office_c4_s2_R200_xxx
                                  --task office_caltech10_c4
                                  [--warmup 5] [--tau_h_q 0.5] [--tau_s_q 0.3]
                                  [--m_top 50] [--output results.json]

Prerequisite:
    1. 训练时 config 加 `se: 1` → checkpoint 自动保存到 ~/fl_checkpoints/
    2. 要推理的 checkpoint 必须是 feddsa_sgpa 框架 (含 M buffer 和 style bank)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# 让 script 能直接 import algorithm
FDSE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FDSE_ROOT))

from algorithm.feddsa_sgpa import FedDSASGPAModel, _resolve_num_classes
from diagnostics.sgpa_diagnostic_logger import SGPADiagnosticLogger as DL


# ============================================================
# 加载 checkpoint
# ============================================================


def load_checkpoint(ckpt_dir: str, device: str = 'cpu'):
    """Load SGPA checkpoint saved by feddsa_sgpa Server._save_sgpa_checkpoint.

    Returns:
        dict with keys: global_state, client_states (list or None),
                        source_style_bank (dict or None),
                        whitening (dict or None), meta (dict)
    """
    d = Path(ckpt_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")

    meta = json.load(open(d / 'meta.json'))
    global_state = torch.load(d / 'global_model.pt', map_location=device)
    client_states = None
    if (d / 'client_models.pt').exists():
        client_states = torch.load(d / 'client_models.pt', map_location=device)
    source_style_bank = None
    if (d / 'source_style_bank.pt').exists():
        source_style_bank = torch.load(d / 'source_style_bank.pt', map_location=device)
    whitening = None
    if (d / 'whitening.pt').exists():
        whitening = torch.load(d / 'whitening.pt', map_location=device)

    return {
        'global_state': global_state,
        'client_states': client_states,
        'source_style_bank': source_style_bank,
        'whitening': whitening,
        'meta': meta,
    }


def build_model_from_meta(meta: dict, device: str = 'cpu') -> FedDSASGPAModel:
    """根据 checkpoint meta 重建 model 结构."""
    num_classes = _resolve_num_classes(meta['task'])
    model = FedDSASGPAModel(
        num_classes=num_classes,
        feat_dim=1024,
        proj_dim=128,
        tau_etf=meta.get('tau_etf', 0.1),
        use_etf=bool(meta.get('use_etf', 1)),
    ).to(device)
    return model


# ============================================================
# 重建 whitening (if checkpoint 没存)
# ============================================================


def compute_whitening_from_bank(source_style_bank: dict, eps: float = 1e-3):
    """从 source_style_bank 重建 pooled whitening (与 Server._compute_pooled_whitening 一致)."""
    cids = sorted(source_style_bank.keys())
    if len(cids) < 2:
        raise ValueError(f"Need ≥2 clients, got {len(cids)}")
    mus = [source_style_bank[c][0] for c in cids]
    sigmas = [source_style_bank[c][1] for c in cids]
    d = mus[0].shape[0]

    mu_stack = torch.stack(mus, dim=0)
    mu_global = mu_stack.mean(dim=0)
    sigma_within = torch.stack(sigmas, dim=0).mean(dim=0)
    diffs = mu_stack - mu_global.unsqueeze(0)
    sigma_between = diffs.t() @ diffs / len(cids)
    sigma_global = sigma_within + sigma_between + eps * torch.eye(d)
    sigma_global = 0.5 * (sigma_global + sigma_global.t())
    L, Q = torch.linalg.eigh(sigma_global.double())
    L = L.clamp(min=eps)
    sigma_inv_sqrt = (Q @ torch.diag(L.pow(-0.5)) @ Q.t()).float()

    return {
        'source_mu_k': {c: source_style_bank[c][0].clone() for c in cids},
        'mu_global': mu_global.clone(),
        'sigma_inv_sqrt': sigma_inv_sqrt,
    }


# ============================================================
# SGPA 推理核心 (inline 实现, 避免依赖 flgo Client 实例)
# ============================================================


@torch.no_grad()
def run_sgpa_on_client(model: FedDSASGPAModel,
                        client_state_dict: dict,
                        client_id: int,
                        test_loader,
                        whitening: dict,
                        device: str = 'cpu',
                        warmup_batches: int = 5,
                        tau_H_quantile: float = 0.5,
                        tau_S_quantile: float = 0.3,
                        ema_decay: float = 0.95,
                        m_top: int = 50):
    """完整 SGPA 推理 + Layer 3 诊断. 返回 metrics dict."""
    # Load client-specific BN (if provided)
    if client_state_dict is not None:
        # Merge client state (with private BN) into global model
        merged = dict(model.state_dict())
        for k, v in client_state_dict.items():
            if k in merged:
                merged[k] = v
        model.load_state_dict(merged)

    model.eval()
    model.to(device)
    K = model.num_classes
    d = model.proj_dim
    M_cpu = model.M.detach().cpu()

    # Pre-compute whitened source bank
    mu_global = whitening['mu_global'].to(device)
    sigma_inv_sqrt = whitening['sigma_inv_sqrt'].to(device)
    cids = sorted(whitening['source_mu_k'].keys())
    mu_stack = torch.stack([whitening['source_mu_k'][c].to(device) for c in cids], dim=0)
    mu_k_white = (mu_stack - mu_global.unsqueeze(0)) @ sigma_inv_sqrt

    # State
    supports = {c: [] for c in range(K)}
    proto = F.normalize(M_cpu.t(), dim=-1).clone().to(device)  # cold-start = ETF vertices
    proto_activated = torch.zeros(K, dtype=torch.bool, device=device)
    tau_H = None
    tau_S = None
    warmup_H, warmup_D = [], []

    all_labels = []
    all_pred_etf = []
    all_pred_sgpa = []
    all_H = []
    all_dist_min = []
    all_z_sty = []
    reliable_count = 0
    total_count = 0

    for batch_idx, batch in enumerate(test_loader):
        if isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[-1]
        else:
            x = batch; y = None
        x = x.to(device)
        if y is not None:
            y = y.to(device)
        B = x.size(0)
        total_count += B

        h = model.encode(x)
        z_sem = model.get_semantic(h)
        z_sty = model.get_style(h)
        logits_etf = model.classify(z_sem)
        pred_etf = logits_etf.argmax(dim=-1)

        p = F.softmax(logits_etf, dim=-1)
        log_p = F.log_softmax(logits_etf, dim=-1)
        H = -(p * log_p).sum(dim=-1)

        z_sty_white = (z_sty - mu_global.unsqueeze(0)) @ sigma_inv_sqrt
        diffs = z_sty_white.unsqueeze(1) - mu_k_white.unsqueeze(0)
        dist_min = (diffs ** 2).sum(dim=-1).min(dim=-1).values

        all_labels.append(y.detach().cpu() if y is not None else torch.zeros(B))
        all_pred_etf.append(pred_etf.detach().cpu())
        all_H.append(H.detach().cpu())
        all_dist_min.append(dist_min.detach().cpu())
        all_z_sty.append(z_sty.detach().cpu())

        # Warmup: 输出 ETF, 收集统计
        if batch_idx < warmup_batches:
            warmup_H.extend(H.detach().cpu().tolist())
            warmup_D.extend(dist_min.detach().cpu().tolist())
            all_pred_sgpa.append(pred_etf.detach().cpu())
            continue
        elif batch_idx == warmup_batches and warmup_H:
            tau_H = float(np.quantile(warmup_H, tau_H_quantile))
            tau_S = float(np.quantile(warmup_D, tau_S_quantile))

        # EMA 更新
        if tau_H is not None:
            cur_H_q = float(np.quantile(H.detach().cpu().numpy(), tau_H_quantile))
            cur_S_q = float(np.quantile(dist_min.detach().cpu().numpy(), tau_S_quantile))
            tau_H = ema_decay * tau_H + (1 - ema_decay) * cur_H_q
            tau_S = ema_decay * tau_S + (1 - ema_decay) * cur_S_q

            reliable = (H < tau_H) & (dist_min < tau_S)
            reliable_count += reliable.sum().item()

            for c in range(K):
                mask_c = reliable & (pred_etf == c)
                if mask_c.any():
                    idxs = mask_c.nonzero(as_tuple=False).flatten().tolist()
                    for i in idxs:
                        supports[c].append((H[i].item(), z_sem[i].detach().cpu()))
                    supports[c] = sorted(supports[c], key=lambda t: t[0])[:m_top]
                    sup = torch.stack([s[1] for s in supports[c]]).mean(dim=0)
                    proto[c] = F.normalize(sup.to(device), dim=-1)
                    proto_activated[c] = True

            # 分类
            z_sem_n = F.normalize(z_sem, dim=-1)
            proto_logits = z_sem_n @ proto.t()
            pred_proto = proto_logits.argmax(dim=-1)
            activated_of_pred = proto_activated[pred_proto]
            pred_sgpa = torch.where(activated_of_pred, pred_proto, pred_etf)
        else:
            pred_sgpa = pred_etf
        all_pred_sgpa.append(pred_sgpa.detach().cpu())

    # Aggregate
    labels = torch.cat(all_labels)
    pred_etf_all = torch.cat(all_pred_etf)
    pred_sgpa_all = torch.cat(all_pred_sgpa)
    H_all = torch.cat(all_H)
    dist_min_all = torch.cat(all_dist_min)
    z_sty_all = torch.cat(all_z_sty)

    # Layer 3 metrics
    metrics = {}
    # Gate rates
    if tau_H is not None:
        metrics.update(DL.gate_rates(H_all, dist_min_all, tau_H, tau_S))
        metrics['tau_H_final'] = tau_H
        metrics['tau_S_final'] = tau_S
    metrics.update(DL.dist_distribution(dist_min_all))
    # Sigma cond
    try:
        sigma_g = whitening['sigma_inv_sqrt']
        metrics['sigma_inv_sqrt_cond'] = DL.sigma_condition_number(sigma_g.float())
    except Exception:
        metrics['sigma_inv_sqrt_cond'] = float('nan')
    # Proto stats
    _, mean_fill = DL.proto_fill(supports, K)
    metrics['proto_fill_mean'] = mean_fill
    per_class_fill, _ = DL.proto_fill(supports, K)
    metrics['proto_zero_fill_count'] = sum(1 for v in per_class_fill.values() if v == 0)
    proto_list = [proto[c].detach().cpu() for c in range(K)]
    mean_off, _, n_valid = DL.proto_etf_offset(proto_list, M_cpu, K)
    metrics['proto_etf_offset_mean'] = mean_off
    metrics['proto_n_valid'] = n_valid
    # Fallback + prediction
    activated_batches = proto_activated.any().item()
    # Per-sample activated (proto for pred_sgpa != ETF)
    fallback_mask = (pred_sgpa_all == pred_etf_all)
    metrics['fallback_rate'] = fallback_mask.float().mean().item()
    if labels.numel() > 0 and labels.any():
        pred_dict = DL.prediction_accuracy(pred_sgpa_all, pred_etf_all, labels)
        metrics.update(pred_dict)
    # 即: metrics['proto_acc'] / 'etf_acc' / 'proto_vs_etf_gain' / 'pred_agree'
    metrics['reliable_rate_total'] = reliable_count / max(total_count, 1)
    metrics['n_samples'] = total_count
    metrics.update(DL.feature_norm_stats(z_sty_all, name='z_sty'))
    metrics['_client_id'] = client_id

    return metrics


# ============================================================
# main
# ============================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True, help='checkpoint dir from feddsa_sgpa se=1')
    ap.add_argument('--task', type=str, default=None, help='task name override (default: from meta)')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--tau_h_q', type=float, default=0.5)
    ap.add_argument('--tau_s_q', type=float, default=0.3)
    ap.add_argument('--m_top', type=int, default=50)
    ap.add_argument('--ema_decay', type=float, default=0.95)
    ap.add_argument('--output', type=str, default=None, help='output json path')
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = ap.parse_args()

    ckpt = load_checkpoint(args.ckpt, device=args.device)
    meta = ckpt['meta']
    task = args.task or meta['task']
    print(f"Loaded checkpoint: task={task}, seed={meta['seed']}, R={meta['num_rounds']}, "
          f"num_clients={meta['num_clients']}, use_etf={meta['use_etf']}")

    # Build model
    model = build_model_from_meta(meta, device=args.device)
    model.load_state_dict(ckpt['global_state'])

    # Reconstruct whitening
    if ckpt['whitening'] is not None:
        whitening = ckpt['whitening']
        print("Using saved whitening payload")
    elif ckpt['source_style_bank'] is not None:
        whitening = compute_whitening_from_bank(ckpt['source_style_bank'])
        print("Reconstructed whitening from source_style_bank")
    else:
        raise RuntimeError("Checkpoint has no whitening or source_style_bank; cannot run SGPA inference")

    # Load flgo task (need flgo infrastructure for test data)
    import flgo
    from flgo.benchmark.toolkits.cv.classification import BasicTaskCalculator
    task_dir = os.path.join(FDSE_ROOT, 'task', task)
    # 简化: 用 flgo.read_task 或自定义 loader
    # 实际用法需要 flgo.gen_task 或配合 feddsa_sgpa.Server 实例化
    # 这里留一个 TODO - 实际跑时参考 run_single.py 的 task 加载

    # Placeholder - 完整实现需要 flgo 读 task data loaders
    print("TODO: Load test data loaders via flgo task reader")
    print("To run properly, use run_single.py in --infer mode or add flgo task loader here")

    results = {'meta': meta, 'whitening_cond': DL.sigma_condition_number(whitening['sigma_inv_sqrt'])}

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved results to {args.output}")
    else:
        print(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
