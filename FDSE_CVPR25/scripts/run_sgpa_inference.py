"""
EXP-099: SGPA 推理独立 script — 测 Layer 3 完整指标 + proto_vs_etf_gain

设计思路:
    1. 复用 flgo.init() 机制创建 Server + Clients (不跑训练, num_rounds=0)
    2. 加载 checkpoint 到 server.model (+ client models for FedBN BN)
    3. 把 whitening payload (μ_global, Σ_inv_sqrt, source_μ_k) 灌到 server
    4. 对每个 client: server.pack() → client.unpack() → client.test_with_sgpa()
    5. 收集 Layer 3 metrics, 输出 json/jsonl

用法:
    python run_sgpa_inference.py \\
        --ckpt ~/fl_checkpoints/sgpa_office_caltech10_c4_s2_R200_xxx \\
        --task office_caltech10_c4 \\
        --output ../experiments/ablation/EXP-099_sgpa_inference/results.json \\
        [--m_top 50] [--warmup 5] [--tau_h_q 0.5] [--tau_s_q 0.3]

前置条件:
    - Checkpoint 由 feddsa_sgpa se=1 保存
    - Algorithm module: algorithm.feddsa_sgpa (use_etf 不影响, ckpt 自带)
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml

FDSE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FDSE_ROOT))


def load_checkpoint(ckpt_dir: str, device: str = 'cpu'):
    """Load SGPA checkpoint saved by Server._save_sgpa_checkpoint."""
    d = Path(ckpt_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")
    meta = json.load(open(d / 'meta.json'))
    global_state = torch.load(d / 'global_model.pt', map_location=device)
    client_states = None
    if (d / 'client_models.pt').exists():
        client_states = torch.load(d / 'client_models.pt', map_location=device)
    whitening = None
    if (d / 'whitening.pt').exists():
        whitening = torch.load(d / 'whitening.pt', map_location=device)
    return {'global_state': global_state, 'client_states': client_states,
            'whitening': whitening, 'meta': meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint dir')
    ap.add_argument('--task', default=None, help='task name (default from meta)')
    ap.add_argument('--algorithm', default='feddsa_sgpa')
    ap.add_argument('--config', default=None,
                    help='config yaml (default: derive from meta)')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--tau_h_q', type=float, default=0.5)
    ap.add_argument('--tau_s_q', type=float, default=0.3)
    ap.add_argument('--m_top', type=int, default=50)
    ap.add_argument('--output', default=None)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    # Load ckpt
    device_str = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    ckpt = load_checkpoint(args.ckpt, device=device_str)
    meta = ckpt['meta']
    task = args.task or meta['task']
    print(f"[load] task={task}, seed={meta['seed']}, R={meta['num_rounds']}, "
          f"use_etf={meta['use_etf']}, use_whitening={meta.get('use_whitening',1)}")

    # Import flgo + algorithm
    import flgo
    import flgo.simulator
    sys.path.insert(0, str(FDSE_ROOT))
    # flgo.init expects task path relative to cwd
    os.chdir(FDSE_ROOT)
    algo_mod = importlib.import_module(f'algorithm.{args.algorithm}')

    # Build minimal option (num_rounds=0: skip training but flgo still needs it ≥1)
    # 实际 workaround: 用 num_rounds=1 + proportion=0.0 → 不 sample client, 或者就跑 1 轮
    # 更简洁: 用 config 原样 + override num_rounds=1 让 flgo 正常 init
    config_path = args.config
    if config_path is None:
        # guess: same dir as check yaml with similar name
        print("[warn] no --config given, constructing minimal option")
        option = {
            'num_rounds': 1, 'num_epochs': 0,  # 不跑 train
            'batch_size': 50, 'learning_rate': 0.05,
            'proportion': 1.0, 'train_holdout': 0.2,
            'local_test': True, 'no_log_console': True, 'log_file': False,
            'algo_para': [
                1.0, 0.1, 128, 10, 1e-3, 2, 0,
                int(meta['use_etf']),
                int(meta.get('use_whitening', 1)),
                int(meta.get('use_centers', 1)),
                0,  # se
            ],
        }
    else:
        with open(config_path) as f:
            option = yaml.safe_load(f)
        option['num_rounds'] = 0  # skip training loop
    option['seed'] = int(meta['seed'])
    option['dataseed'] = int(meta['seed'])
    option['gpu'] = args.gpu

    # init logger (用 SimpleLogger 避免 PerRunLogger 复杂性)
    from flgo.experiment.logger import BasicLogger
    try:
        from logger import PerRunLogger
    except Exception:
        PerRunLogger = BasicLogger

    # flgo.init
    runner = flgo.init(
        os.path.join('task', task), algo_mod, option,
        Logger=PerRunLogger, Simulator=flgo.simulator.DefaultSimulator,
    )
    server = runner
    print(f"[flgo] server initialized, num_clients={len(server.clients)}")

    # Load model into server
    server.model.load_state_dict(ckpt['global_state'])
    print("[ckpt] global_state loaded to server.model")

    # Load whitening
    if ckpt['whitening'] is not None:
        server.source_mu_k = ckpt['whitening']['source_mu_k']
        server.mu_global = ckpt['whitening']['mu_global']
        server.sigma_inv_sqrt = ckpt['whitening']['sigma_inv_sqrt']
        print(f"[ckpt] whitening loaded: N_clients={len(server.source_mu_k)}, d={server.mu_global.shape[0]}")
    else:
        print("[warn] no whitening in ckpt; test_with_sgpa will fallback to ETF argmax")

    # Load client states (for FedBN BN)
    if ckpt['client_states'] is not None:
        for cid, cstate in enumerate(ckpt['client_states']):
            if cstate is not None and cid < len(server.clients):
                try:
                    server.clients[cid].model = copy.deepcopy(server.model)
                    server.clients[cid].model.load_state_dict(cstate)
                except Exception as e:
                    print(f"[warn] client {cid} state load failed: {e}")
        print("[ckpt] client BN states loaded")

    # Run SGPA inference on each client
    all_results = []
    for c in server.clients:
        # Server pack → client unpack
        svr_pkg = server.pack(c.id) if hasattr(server, 'pack') else {
            'model': server.model, 'current_round': 0,
            'source_mu_k': server.source_mu_k, 'mu_global': server.mu_global,
            'sigma_inv_sqrt': server.sigma_inv_sqrt,
        }
        if hasattr(c, 'unpack'):
            c.unpack(svr_pkg)

        # Run SGPA inference
        try:
            result = c.test_with_sgpa(
                m_top=args.m_top,
                warmup_batches=args.warmup,
                tau_H_quantile=args.tau_h_q,
                tau_S_quantile=args.tau_s_q,
                ema_decay=0.95,
            )
        except Exception as e:
            print(f"[client {c.id}] ERROR: {e}")
            result = {'error': str(e)}
        result['client_id'] = c.id
        all_results.append(result)
        print(f"[client {c.id}] {result}")

    # Aggregate
    import numpy as np
    aggregated = {}
    for key in ['sgpa_acc', 'etf_acc', 'proto_vs_etf_gain',
                'reliable_rate', 'fallback_rate']:
        vals = [r[key] for r in all_results if key in r]
        if vals:
            aggregated[f'{key}_mean'] = float(np.mean(vals))
            aggregated[f'{key}_std'] = float(np.std(vals))
    aggregated['per_client'] = all_results
    aggregated['meta'] = meta

    print("\n=== SGPA Inference Summary ===")
    for k in ['sgpa_acc_mean', 'etf_acc_mean', 'proto_vs_etf_gain_mean',
              'reliable_rate_mean', 'fallback_rate_mean']:
        if k in aggregated:
            print(f"  {k}: {aggregated[k]:.4f}")

    out_path = args.output or f"sgpa_inference_{meta['task']}_s{meta['seed']}.json"
    with open(out_path, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    print(f"[save] {out_path}")


if __name__ == '__main__':
    main()
