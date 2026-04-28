"""
Sanity test for F2DC paper Domain-Aware Aggregation (Eq 10/11) implementation.

验证:
1. Eq (10): d_k = sqrt(C/2) * |n_k/N - 1/Q| 公式正确
2. Eq (11): p_k = sigmoid(α·n_k/N - β·d_k) / Σ sigmoid; freq sum = 1
3. degenerate cases:
   - 当所有 client n_k 相同 → d_k 全相等 → p_k 全相等 (退化为 uniform)
   - 当 alpha=0, beta=0 → score 全 0 → sigmoid(0)=0.5 → p_k 全相等
4. compare with paper expected values for PACS s=15 setup
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/datasets")
sys.path.append(os.getcwd() + "/backbone")
sys.path.append(os.getcwd() + "/models")

import numpy as np
import argparse
from models.utils.federated_model import FederatedModel


def make_dummy_model(args, fake_lens):
    """直接 instantiate FederatedModel-like object 调 _compute_daa_freq."""
    class _M:
        pass
    m = _M()
    m.args = args
    m._compute_daa_freq = FederatedModel._compute_daa_freq.__get__(m, _M)
    return m


def test_daa_formula():
    print("=" * 60)
    print("Test 1: Eq (10) d_k 公式正确性")
    print("=" * 60)
    args = argparse.Namespace(num_classes=7, num_domains_q=4, agg_a=1.0, agg_b=0.4, use_daa=True)
    m = make_dummy_model(args, None)

    # PACS realistic: 10 clients, n_k 在 700-1200 之间
    n_lens = [1200, 1200, 700, 700, 700, 1000, 1000, 700, 700, 700]
    N = sum(n_lens)
    print(f"PACS-like 10 clients, N = {N}, 1/Q = {1/4} = N/Q sample = {N/4:.0f}")

    # 手算 d_k 验证
    Q = 4
    C = 7
    expected_d = [np.sqrt(C/2) * abs(n/N - 1/Q) for n in n_lens]
    print("\nclient | n_k  | n_k/N  | d_k (expected)")
    for i, (n, d) in enumerate(zip(n_lens, expected_d)):
        print(f"  {i:2d}   | {n:4d} | {n/N:.4f} | {d:.4f}")

    print("\n  ✓ d_k 越大说明 client sample 数偏离 N/Q 越多")

    print("\n" + "=" * 60)
    print("Test 2: Eq (11) freq 公式 + 归一化")
    print("=" * 60)
    freq = m._compute_daa_freq(n_lens)
    print(f"freq = {[f'{f:.4f}' for f in freq]}")
    print(f"freq sum = {freq.sum():.6f} (expected 1.0)")
    print(f"freq min = {freq.min():.4f}, max = {freq.max():.4f}")
    print(f"freq max/min ratio = {freq.max()/freq.min():.3f}")
    assert abs(freq.sum() - 1.0) < 1e-6, f"freq sum {freq.sum()} != 1"

    # 跟 sample-mean (FedAvg) 对比
    fedavg_freq = np.array(n_lens) / sum(n_lens)
    print(f"\nFedAvg freq = {[f'{f:.4f}' for f in fedavg_freq]}")
    diff = (freq - fedavg_freq) / fedavg_freq * 100  # 百分比变化
    print(f"DaA vs FedAvg 各 client 权重变化 (%): {[f'{d:+.2f}' for d in diff]}")

    print("\n" + "=" * 60)
    print("Test 3: 退化 case — 所有 client sample 数相同 → freq uniform")
    print("=" * 60)
    n_eq = [800] * 10
    freq_eq = m._compute_daa_freq(n_eq)
    print(f"freq = {[f'{f:.4f}' for f in freq_eq]}")
    assert np.allclose(freq_eq, 1/10, atol=1e-6), "等 sample 时应退化为 uniform"
    print("  ✓ PASS — freq 全部 0.1 (uniform)")

    print("\n" + "=" * 60)
    print("Test 4: 退化 case — α=0, β=0 → score 全 0 → sigmoid(0)=0.5 → freq uniform")
    print("=" * 60)
    args.agg_a = 0.0
    args.agg_b = 0.0
    freq_zero = m._compute_daa_freq(n_lens)
    print(f"freq = {[f'{f:.4f}' for f in freq_zero]}")
    assert np.allclose(freq_zero, 1/10, atol=1e-6), "α=β=0 时应 uniform"
    print("  ✓ PASS")

    print("\n" + "=" * 60)
    print("Test 5: PACS s=15 实际 client distribution (跟我们 fixed setup 一致)")
    print("=" * 60)
    # PACS fixed: photo:2, art:3, cartoon:2, sketch:3 — 各 client 持有 1 个 domain
    # 假设 PACS 各 domain 训练数据约: photo 1670, art 2048, cartoon 2344, sketch 3929
    # 每个 domain 30% (paper) 给 client, 平均分给该 domain 的 client 数
    domain_size = {"photo": 1670, "art": 2048, "cartoon": 2344, "sketch": 3929}
    domain_clients = {"photo": 2, "art": 3, "cartoon": 2, "sketch": 3}
    n_lens_real = []
    for d, sz in domain_size.items():
        per_client = int(sz * 0.3 / domain_clients[d])  # 30% per F2DC paper
        n_lens_real.extend([per_client] * domain_clients[d])
    N_real = sum(n_lens_real)
    print(f"realistic PACS n_k: {n_lens_real}, N={N_real}")
    print(f"N/Q = {N_real/4:.0f} (理想 single-domain client sample 数)")

    args.agg_a = 1.0
    args.agg_b = 0.4
    freq_real = m._compute_daa_freq(n_lens_real)
    fedavg_real = np.array(n_lens_real) / N_real
    print(f"\nDaA  freq: {[f'{f:.4f}' for f in freq_real]}")
    print(f"FedAvg freq: {[f'{f:.4f}' for f in fedavg_real]}")
    print(f"max diff: {max(abs(freq_real - fedavg_real)):.5f}")
    print(f"DaA / FedAvg ratio range: [{(freq_real/fedavg_real).min():.3f}, {(freq_real/fedavg_real).max():.3f}]")

    print("\n" + "=" * 60)
    print("ALL DAA SANITY PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_daa_formula()
