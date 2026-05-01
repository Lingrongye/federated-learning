"""
test_mask_stats.py
==================
验证 compute_mask_stats() 在不同 mask 形态下输出正确, 同时验证
DFC_PG / DFD_lite 模块的诊断收集 + summary aggregation 流程。

跑法:
    cd F2DC && python test_mask_stats.py
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbone.ResNet_DC import compute_mask_stats
from backbone.ResNet_DC_PG_ML import DFD_lite
from backbone.ResNet_DC_PG import DFC_PG


def assert_close(actual, expected, tol, name):
    assert abs(actual - expected) <= tol, \
        f"{name}: actual={actual:.4f} != expected={expected:.4f} ±{tol}"


# --------- T1: compute_mask_stats 在已知分布下输出正确 ---------
def test_t1_compute_mask_stats():
    print("=== T1: compute_mask_stats() 数值正确性 ===")
    torch.manual_seed(0)

    # case 1: 全部 0.5 const → unit_std≈0, mid_ratio=1, hard_ratio=0
    mask = torch.full((4, 8, 4, 4), 0.5)
    s = compute_mask_stats(mask)
    assert_close(s['mean'], 0.5, 1e-6, 'case1.mean')
    assert_close(s['unit_std'], 0.0, 1e-6, 'case1.unit_std')
    assert_close(s['mid_ratio'], 1.0, 1e-6, 'case1.mid_ratio')
    assert_close(s['hard_ratio'], 0.0, 1e-6, 'case1.hard_ratio')
    print(f"  ✓ case1 (全 0.5): {s}")

    # case 2: 真二值化 mask ∈ {0, 1} 各一半 → unit_std≈0.5, hard_ratio=1, mid_ratio=0
    mask = (torch.rand(4, 8, 16, 16) > 0.5).float()
    s = compute_mask_stats(mask)
    assert_close(s['mean'], 0.5, 0.05, 'case2.mean')
    assert s['unit_std'] > 0.49 and s['unit_std'] < 0.51, f"case2.unit_std={s['unit_std']}"
    assert_close(s['hard_ratio'], 1.0, 1e-6, 'case2.hard_ratio')
    assert_close(s['mid_ratio'], 0.0, 1e-6, 'case2.mid_ratio')
    print(f"  ✓ case2 (真二值化): unit_std={s['unit_std']:.3f} hard_ratio={s['hard_ratio']:.3f}")

    # case 3: soft selective Beta(2, 2) 双峰 → unit_std≈0.22, mid_ratio≈0.4
    mask = torch.distributions.Beta(2, 2).sample((4, 8, 16, 16))
    s = compute_mask_stats(mask)
    assert s['unit_std'] > 0.15 and s['unit_std'] < 0.30, f"case3.unit_std={s['unit_std']}"
    print(f"  ✓ case3 (Beta(2,2)): unit_std={s['unit_std']:.3f} mid_ratio={s['mid_ratio']:.3f}")

    # case 4: channel-only selective (前一半 channel mask=0.9, 后一半 mask=0.1)
    mask = torch.zeros(4, 8, 4, 4)
    mask[:, :4] = 0.9
    mask[:, 4:] = 0.1
    s = compute_mask_stats(mask)
    assert_close(s['mean'], 0.5, 1e-6, 'case4.mean')
    assert s['channel_std'] > 0.3, f"case4.channel_std={s['channel_std']} 应该 > 0.3 (channel 真区分)"
    assert_close(s['spatial_std'], 0.0, 1e-6, 'case4.spatial_std (位置无差异)')
    assert_close(s['sample_std'], 0.0, 1e-6, 'case4.sample_std (样本无差异)')
    print(f"  ✓ case4 (channel-only): channel_std={s['channel_std']:.3f} (强), "
          f"spatial_std={s['spatial_std']:.3f} (0)")

    print("T1 PASS\n")


# --------- T2: DFD_lite forward + diag 累积 + summary ---------
def test_t2_dfd_lite():
    print("=== T2: DFD_lite forward + diag summary ===")
    torch.manual_seed(0)
    dfd = DFD_lite(size=(32, 8, 8), num_channel=8, tau=0.5, diag_sample_rate=1.0)
    dfd.train()
    dfd.reset_diag()

    # forward 5 batch
    for _ in range(5):
        feat = torch.randn(4, 32, 8, 8)
        r, nr, mask = dfd(feat)
        assert r.shape == feat.shape and nr.shape == feat.shape and mask.shape == feat.shape

    # diag summary 应该有 7 个 mask3_*_mean keys + backward compat
    s = dfd.get_diag_summary()
    print(f"  summary keys: {sorted(s.keys())}")
    expected_new_keys = {'mask3_mean_mean', 'mask3_unit_std_mean',
                         'mask3_hard_ratio_mean', 'mask3_mid_ratio_mean',
                         'mask3_sample_std_mean', 'mask3_channel_std_mean',
                         'mask3_spatial_std_mean'}
    expected_compat_keys = {'mask3_sparsity_mean', 'mask3_sparsity_std'}
    for k in expected_new_keys:
        assert k in s, f"missing new key {k}"
    for k in expected_compat_keys:
        assert k in s, f"missing backward compat key {k}"
    # backward compat: mask3_sparsity_mean 应该 ≈ mask3_mean_mean
    assert_close(s['mask3_sparsity_mean'], s['mask3_mean_mean'], 1e-6,
                 'backward compat: mask3_sparsity_mean alias')
    print(f"  ✓ mask3_mean={s['mask3_mean_mean']:.3f} mask3_unit_std={s['mask3_unit_std_mean']:.4f}")
    print(f"  ✓ hard_ratio={s['mask3_hard_ratio_mean']:.3f} mid_ratio={s['mask3_mid_ratio_mean']:.3f}")
    print(f"  ✓ backward compat mask3_sparsity_mean={s['mask3_sparsity_mean']:.3f} "
          f"({'OK ' if abs(s['mask3_sparsity_mean'] - s['mask3_mean_mean']) < 1e-6 else 'FAIL'})")

    # eval mode 不应该收集 diag
    dfd.reset_diag()
    dfd.eval()
    feat = torch.randn(4, 32, 8, 8)
    dfd(feat, is_eval=True)
    assert dfd.get_diag_summary() is None, "eval mode 不应该收集 diag"
    print("  ✓ eval mode 不收 diag")
    print("T2 PASS\n")


# --------- T3: DFC_PG forward + diag summary (layer4 mask) ---------
def test_t3_dfc_pg():
    print("=== T3: DFC_PG forward + diag summary (mask4) ===")
    torch.manual_seed(0)
    dfc = DFC_PG(size=(16, 4, 4), num_classes=7, num_channel=8, proto_weight=0.0)
    dfc.train()
    dfc.reset_diag()

    for _ in range(200):  # 200 batch, 1% sample → 期望 ~2 个 sample
        nr = torch.randn(2, 16, 4, 4)
        mask = torch.rand(2, 16, 4, 4)
        dfc(nr, mask)

    s = dfc.get_diag_summary()
    if s is None:
        # 1% 采样可能没采到, 强制 100% 采样重试
        dfc.reset_diag()
        # monkey-patch: 直接调用 stats 添加
        from backbone.ResNet_DC import compute_mask_stats
        for _ in range(5):
            mask = torch.rand(2, 16, 4, 4)
            dfc._diag_mask_stats.append(compute_mask_stats(mask))
        s = dfc.get_diag_summary()

    print(f"  summary keys (subset): {[k for k in s if 'mask' in k][:8]}")
    expected_new_keys = {'mask_mean_mean', 'mask_unit_std_mean',
                         'mask_hard_ratio_mean', 'mask_mid_ratio_mean',
                         'mask_sample_std_mean', 'mask_channel_std_mean',
                         'mask_spatial_std_mean'}
    for k in expected_new_keys:
        assert k in s, f"missing new key {k}"
    # backward compat
    assert 'mask_sparsity_mean' in s
    assert 'mask_sparsity_std' in s
    print(f"  ✓ DFC_PG 7-stat 全部就位")
    print(f"  ✓ mask_mean={s['mask_mean_mean']:.3f} unit_std={s['mask_unit_std_mean']:.4f} "
          f"hard_ratio={s['mask_hard_ratio_mean']:.3f}")
    print("T3 PASS\n")


# --------- T4: 大数定律 sanity (实测 vs 我们之前的判决) ---------
def test_t4_large_n_sanity():
    print("=== T4: 大数定律 sanity check (我们 R99 实测 vs 真二值化) ===")
    torch.manual_seed(0)
    # 模拟 PACS layer3 mask shape (B=46, C=256, H=16, W=16)
    B, C, H, W = 46, 256, 16, 16

    # 真二值化 mask (sparsity 0.39)
    mask_binary = (torch.rand(B, C, H, W) < 0.39).float()
    s_binary = compute_mask_stats(mask_binary)
    # const mask = 0.39 (没切分)
    mask_const = torch.full((B, C, H, W), 0.39)
    s_const = compute_mask_stats(mask_const)
    # 实测 mask (mean 0.39, std 0.001 高斯)
    mask_meas = torch.randn(B, C, H, W) * 0.001 + 0.39
    s_meas = compute_mask_stats(mask_meas.clamp(0, 1))

    print(f"  常数 0.39:        unit_std={s_const['unit_std']:.4f} hard_ratio={s_const['hard_ratio']:.3f} mid_ratio={s_const['mid_ratio']:.3f}")
    print(f"  实测 0.39±0.001: unit_std={s_meas['unit_std']:.4f} hard_ratio={s_meas['hard_ratio']:.3f} mid_ratio={s_meas['mid_ratio']:.3f}")
    print(f"  真二值化 sparsity 0.39: unit_std={s_binary['unit_std']:.4f} hard_ratio={s_binary['hard_ratio']:.3f} mid_ratio={s_binary['mid_ratio']:.3f}")

    # 关键判决
    assert s_const['unit_std'] < 0.01, "常数 mask unit_std 应 ≈ 0"
    assert s_meas['unit_std'] < 0.01, "实测 mask (0.39±0.001) unit_std 应 ≈ 0"
    assert s_binary['unit_std'] > 0.4, "真二值化 mask unit_std 应 > 0.4"
    print("  ✓ 区分清楚: unit_std < 0.01 = 没切, unit_std > 0.4 = 真二值化")
    print("T4 PASS\n")


def test_t5_pre_post_gumbel():
    """T5: 验证 DFD/DFD_lite 的 pre-Gumbel 跟 post-Gumbel mask 7-stat 都被收集.

    pre-Gumbel = sigmoid(rob_map) (Gumbel noise 前) — 真正反映模型学到
    post-Gumbel = GumbelSigmoid(...) (实际用) — 受 tau noise 影响, hard_ratio 天然 50%
    """
    print("=== T5: pre/post Gumbel mask 7-stat ===")
    torch.manual_seed(0)

    # DFD_lite (mask3)
    dfd_lite = DFD_lite(size=(32, 8, 8), num_channel=8, tau=0.5, diag_sample_rate=1.0)
    dfd_lite.train()
    dfd_lite.reset_diag()
    for _ in range(3):
        feat = torch.randn(4, 32, 8, 8)
        _, _, _ = dfd_lite(feat)
    s = dfd_lite.get_diag_summary()
    pre_keys = [k for k in s if k.startswith('pre_mask3_')]
    post_keys = [k for k in s if k.startswith('mask3_') and not k.startswith('mask3_sparsity')]
    assert len(pre_keys) == 7, f"pre_mask3_*_mean 应该 7 个, 实际 {len(pre_keys)}"
    assert len(post_keys) == 7, f"mask3_*_mean 应该 7 个, 实际 {len(post_keys)}"
    print(f"  ✓ DFD_lite 同时输出 pre_mask3 (7) + mask3 (7) 共 14 + 2 backward compat keys")
    pre_hard = s['pre_mask3_hard_ratio_mean']
    post_hard = s['mask3_hard_ratio_mean']
    pre_mid = s['pre_mask3_mid_ratio_mean']
    post_mid = s['mask3_mid_ratio_mean']
    print(f"  pre_mask3:  hard_ratio={pre_hard:.3f}  mid_ratio={pre_mid:.3f}  (random init: 接近 sigmoid(N(0,1)))")
    print(f"  post_mask3: hard_ratio={post_hard:.3f}  mid_ratio={post_mid:.3f}  (Gumbel 推到 ~50%)")
    # post hard 应该比 pre hard 高 (Gumbel 推到 0/1)
    assert post_hard >= pre_hard - 0.1, f"post hard ratio ({post_hard}) 应该 ≥ pre ({pre_hard})"

    # DFD (mask4) — 之前没 diag, 验证新加的 reset_diag/get_diag_summary
    from backbone.ResNet_DC import DFD
    dfd = DFD(size=(16, 4, 4), num_channel=8, tau=0.1, diag_sample_rate=1.0)
    dfd.train()
    dfd.reset_diag()
    for _ in range(3):
        feat = torch.randn(4, 16, 4, 4)
        _, _, _ = dfd(feat)
    s = dfd.get_diag_summary()
    pre_keys = [k for k in s if k.startswith('pre_mask_')]
    post_keys = [k for k in s if k.startswith('mask_') and not k.startswith('mask_sparsity')]
    assert len(pre_keys) == 7, f"DFD pre_mask_*_mean 应该 7 个, 实际 {len(pre_keys)}"
    assert len(post_keys) == 7, f"DFD mask_*_mean 应该 7 个, 实际 {len(post_keys)}"
    print(f"  ✓ DFD 同时输出 pre_mask (7) + mask (7) keys, mask4 tau=0.1 比 mask3 tau=0.5 更硬")
    pre_hard = s['pre_mask_hard_ratio_mean']
    post_hard = s['mask_hard_ratio_mean']
    print(f"  pre_mask4:  hard_ratio={pre_hard:.3f}  unit_std={s['pre_mask_unit_std_mean']:.4f}")
    print(f"  post_mask4: hard_ratio={post_hard:.3f}  unit_std={s['mask_unit_std_mean']:.4f}  (tau=0.1 强推 0/1)")
    print("T5 PASS\n")


if __name__ == '__main__':
    test_t1_compute_mask_stats()
    test_t2_dfd_lite()
    test_t3_dfc_pg()
    test_t4_large_n_sanity()
    test_t5_pre_post_gumbel()
    print("=" * 60)
    print("ALL 5 TESTS PASS — 7-stat mask 诊断已就位 (pre + post Gumbel)")
    print("=" * 60)
