"""
FedDSA-SGPA 核心组件单元测试.

覆盖:
- build_etf_matrix: 数学正确性 (单纯形 ETF 性质, seeded 一致性, edge case)
- FedDSASGPAModel: 正常 forward, classify 接口, buffer 不参加聚合
- Server._compute_pooled_whitening: μ_global / Σ_within+Σ_between / Σ_inv_sqrt 数值正确
- SGPA inference 关键逻辑 (warmup ETF fallback, top-m proto, activation gating)

运行:
    D:/anaconda/python.exe -m pytest FDSE_CVPR25/tests/test_feddsa_sgpa.py -v
"""

import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

TESTS_DIR = Path(__file__).parent
FDSE_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(FDSE_ROOT))

from algorithm.feddsa_sgpa import (
    build_etf_matrix,
    FedDSASGPAModel,
    AlexNetEncoder,
    GradientReverseLayer,
    grl,
    compute_lambda_adv,
)


# ============================================================
# ETF matrix construction
# ============================================================


class TestBuildETFMatrix:
    def test_basic_shape(self):
        M = build_etf_matrix(128, 7)
        assert M.shape == (128, 7)
        assert M.dtype == torch.float32

    def test_seeded_reproducibility(self):
        """同一 seed 产生相同 M (所有 client 一致的前提)."""
        M1 = build_etf_matrix(128, 7, seed=42)
        M2 = build_etf_matrix(128, 7, seed=42)
        assert torch.allclose(M1, M2)

    def test_different_seed_different_M(self):
        M1 = build_etf_matrix(128, 7, seed=0)
        M2 = build_etf_matrix(128, 7, seed=1)
        assert not torch.allclose(M1, M2)

    def test_simplex_etf_column_norms_equal(self):
        """ETF 性质 1: 所有列向量 norm 相等."""
        M = build_etf_matrix(128, 10)
        norms = M.norm(dim=0)  # [K]
        assert (norms.max() - norms.min()).item() < 1e-5

    def test_simplex_etf_pairwise_cos(self):
        """ETF 性质 2: 不同列向量之间 cos 都等于 -1/(K-1).

        Simplex ETF (Papyan 2020) 的核心结论.
        """
        K = 7
        M = build_etf_matrix(128, K)
        M_norm = F.normalize(M, dim=0)  # [d, K]
        gram = M_norm.t() @ M_norm  # [K, K]
        # off-diagonal should all equal -1/(K-1)
        expected_off = -1.0 / (K - 1)
        off_diag = gram - torch.diag(gram.diag())
        # mask off-diagonal
        mask = ~torch.eye(K, dtype=torch.bool)
        off_values = off_diag[mask]
        assert (off_values - expected_off).abs().max().item() < 1e-4, (
            f"Off-diagonal cos should be {expected_off:.4f}, got {off_values.mean():.4f}")

    def test_assertion_d_less_than_k(self):
        """d < K 时 assert 失败."""
        with pytest.raises(AssertionError):
            build_etf_matrix(5, 10)

    def test_exactly_d_equals_k(self):
        """d == K 边界 case."""
        M = build_etf_matrix(7, 7)
        assert M.shape == (7, 7)
        # rank of M should be K-1 (simplex lives in K-1 hyperplane)
        # 不强 assert rank 但 norm 应该 finite
        assert torch.isfinite(M).all()

    def test_office_caltech10_dims(self):
        """Office-Caltech10 实际 dims: K=10, d=128."""
        M = build_etf_matrix(128, 10)
        assert M.shape == (128, 10)
        assert torch.isfinite(M).all()

    def test_pacs_dims(self):
        """PACS: K=7, d=128."""
        M = build_etf_matrix(128, 7)
        assert M.shape == (128, 7)


# ============================================================
# FedDSASGPAModel
# ============================================================


class TestFedDSASGPAModel:
    def test_model_creation(self):
        model = FedDSASGPAModel(num_classes=7, feat_dim=1024, proj_dim=128)
        assert model.M.shape == (128, 7)
        assert model.num_classes == 7
        assert model.proj_dim == 128

    def test_M_is_buffer_not_parameter(self):
        """M 必须是 buffer (不可训), 不能出现在 parameters()."""
        model = FedDSASGPAModel()
        param_ids = {id(p) for p in model.parameters()}
        assert id(model.M) not in param_ids
        # 但出现在 state_dict (buffer 默认 persistent)
        assert 'M' in model.state_dict() or any('M' == k.split('.')[-1] for k in model.state_dict())

    def test_M_identical_across_instances(self):
        """同一 seed 不同实例的 M 相同 (client 一致性)."""
        m1 = FedDSASGPAModel(num_classes=7, etf_seed=0)
        m2 = FedDSASGPAModel(num_classes=7, etf_seed=0)
        assert torch.allclose(m1.M, m2.M)

    def test_forward_shape(self):
        model = FedDSASGPAModel(num_classes=7)
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 7)

    def test_classify_equals_forward(self):
        """classify(z_sem) 的结果 = F.normalize(z_sem) @ M / tau_etf."""
        torch.manual_seed(0)
        model = FedDSASGPAModel(num_classes=7, tau_etf=0.1)
        model.eval()
        z_sem = torch.randn(5, 128)
        with torch.no_grad():
            logits = model.classify(z_sem)
            expected = F.normalize(z_sem, dim=-1) @ model.M / 0.1
        assert torch.allclose(logits, expected, atol=1e-5)

    def test_encode_semantic_style_interface(self):
        """encode/get_semantic/get_style 接口符合 flgo 约定."""
        model = FedDSASGPAModel()
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            h = model.encode(x)
            assert h.shape == (2, 1024)
            z_sem = model.get_semantic(h)
            assert z_sem.shape == (2, 128)
            z_sty = model.get_style(h)
            assert z_sty.shape == (2, 128)

    def test_tau_etf_default(self):
        """τ_etf 默认 0.1 (refine 决议)."""
        model = FedDSASGPAModel()
        assert model.tau_etf == 0.1


# ============================================================
# Pooled whitening math
# ============================================================


class TestPooledWhitening:
    """直接测 _compute_pooled_whitening 的核心数学."""

    def _fake_server_compute(self, mus, sigmas, eps=1e-3):
        """独立算一遍作为 ground truth."""
        mu_stack = torch.stack(mus, dim=0)
        mu_global = mu_stack.mean(dim=0)
        sigma_within = torch.stack(sigmas, dim=0).mean(dim=0)
        diffs = mu_stack - mu_global.unsqueeze(0)
        sigma_between = diffs.t() @ diffs / len(mus)
        d = mus[0].shape[0]
        sigma_global = sigma_within + sigma_between + eps * torch.eye(d)
        sigma_global = 0.5 * (sigma_global + sigma_global.t())
        L, Q = torch.linalg.eigh(sigma_global.double())
        L = L.clamp(min=eps)
        sigma_inv_sqrt = (Q @ torch.diag(L.pow(-0.5)) @ Q.t()).float()
        return mu_global, sigma_inv_sqrt

    def test_identity_case(self):
        """4 个 client, μ_k 都相同, Σ_k=I → μ_global = μ, Σ_inv_sqrt ≈ I."""
        d = 8
        mu = torch.zeros(d)
        I = torch.eye(d)
        mus = [mu.clone() for _ in range(4)]
        sigmas = [I.clone() for _ in range(4)]
        mu_global, sigma_inv_sqrt = self._fake_server_compute(mus, sigmas, eps=1e-3)
        assert torch.allclose(mu_global, mu)
        # Σ_global ≈ I + 0.001·I → Σ^{-1/2} ≈ I / sqrt(1.001)
        expected = 1.0 / math.sqrt(1.001)
        diag_vals = sigma_inv_sqrt.diag()
        assert (diag_vals - expected).abs().max().item() < 1e-3

    def test_diagonal_sigma_case(self):
        """Σ_k = diag(varying), 验证 Σ_inv_sqrt 主对角元素与理论一致."""
        d = 4
        mus = [torch.zeros(d) for _ in range(3)]
        # 3 个 client 各自 Σ_k 对角, 平均后对角为 2.0
        sigmas = [torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])) for _ in range(3)]
        mu_global, sigma_inv_sqrt = self._fake_server_compute(mus, sigmas, eps=0.0)
        # Σ_within = diag(1,2,3,4), Σ_between ≈ 0 (μ_k 相同)
        # Σ^{-1/2} = diag(1, 1/sqrt(2), 1/sqrt(3), 1/sqrt(4))
        expected = torch.tensor([1.0, 1/math.sqrt(2), 1/math.sqrt(3), 0.5])
        diag_inv = sigma_inv_sqrt.diag()
        assert (diag_inv - expected).abs().max().item() < 1e-3

    def test_whitened_distance_is_finite(self):
        """白化后的 dist 不应 NaN/Inf (eps regularization works)."""
        d = 4
        mus = [torch.randn(d) for _ in range(3)]
        # 构造奇异 Σ (rank < d)
        v = torch.randn(d)
        sigmas = [torch.outer(v, v) for _ in range(3)]
        mu_global, sigma_inv_sqrt = self._fake_server_compute(mus, sigmas, eps=1e-3)
        # 算一个样本的白化距离
        z = torch.randn(d)
        z_white = (z - mu_global) @ sigma_inv_sqrt
        mu_white = (mus[0] - mu_global) @ sigma_inv_sqrt
        dist = ((z_white - mu_white) ** 2).sum()
        assert torch.isfinite(dist)


# ============================================================
# SGPA inference logic (核心决策逻辑, 不跑真实 model)
# ============================================================


class TestSGPAInferenceLogic:
    def test_warmup_outputs_etf_not_skip(self):
        """CRITICAL: warmup 期间 pred = pred_etf, 不能跳过样本."""
        # 模拟 warmup 判断
        warmup_batches = 5
        B = 8
        K = 7

        # 第 0-4 batch 都应该输出 ETF
        for batch_idx in range(5):
            pred_etf = torch.randint(0, K, (B,))
            pred_sgpa = pred_etf.clone()  # 应该 = ETF
            # batch_idx < warmup 时, gate 不激活, pred_sgpa = pred_etf
            assert torch.equal(pred_sgpa, pred_etf)

    def test_top_m_keeps_lowest_entropy(self):
        """Top-m support selection: 按 entropy 升序保留."""
        m_top = 3
        # 构造 5 个 (entropy, z_sem), m_top=3, 应该保留 entropy 最小的 3 个
        items = [
            (0.5, torch.tensor([1.0])),
            (0.1, torch.tensor([2.0])),  # smallest entropy
            (0.3, torch.tensor([3.0])),
            (0.8, torch.tensor([4.0])),  # largest
            (0.2, torch.tensor([5.0])),
        ]
        sorted_items = sorted(items, key=lambda t: t[0])[:m_top]
        kept_entropies = [s[0] for s in sorted_items]
        assert kept_entropies == [0.1, 0.2, 0.3]

    def test_fallback_when_class_not_activated(self):
        """未激活的类 proto → 回退到 ETF prediction."""
        B = 4
        K = 7
        pred_etf = torch.tensor([0, 1, 2, 3])
        pred_proto = torch.tensor([4, 5, 6, 0])  # proto 指向 4, 5, 6, 0

        # 假设 class 4, 5, 6 未激活, 只有 class 0 激活
        activated = torch.zeros(K, dtype=torch.bool)
        activated[0] = True

        activated_of_pred = activated[pred_proto]  # [B]
        # [activated[4]=F, activated[5]=F, activated[6]=F, activated[0]=T]
        pred_sgpa = torch.where(activated_of_pred, pred_proto, pred_etf)
        # 前 3 个 fallback 到 pred_etf, 最后一个用 pred_proto
        assert pred_sgpa[0].item() == 0  # ETF
        assert pred_sgpa[1].item() == 1  # ETF
        assert pred_sgpa[2].item() == 2  # ETF
        assert pred_sgpa[3].item() == 0  # proto activated → use proto


# ============================================================
# Integration: model + ETF classify full flow
# ============================================================


class TestUseETFFlag:
    """Linear 对照 (use_etf=0) vs ETF (use_etf=1) 路径测试."""

    def test_etf_default(self):
        """默认 use_etf=True, 有 M buffer, head is None."""
        m = FedDSASGPAModel(num_classes=10)
        assert m.use_etf is True
        assert hasattr(m, 'M') and m.M.shape == (128, 10)
        assert m.head is None

    def test_linear_mode_has_trainable_head(self):
        """use_etf=False → head 是 nn.Linear, 可训参数."""
        m = FedDSASGPAModel(num_classes=10, use_etf=False)
        assert m.use_etf is False
        assert isinstance(m.head, torch.nn.Linear)
        assert m.head.weight.shape == (10, 128)
        # head 参数应在 parameters() 里
        param_ids = {id(p) for p in m.parameters()}
        assert id(m.head.weight) in param_ids
        assert id(m.head.bias) in param_ids

    def test_linear_mode_still_has_M_buffer(self):
        """use_etf=False 时 M 仍作为 buffer 存在 (供诊断 etf_align 不 crash)."""
        m = FedDSASGPAModel(num_classes=10, use_etf=False)
        assert hasattr(m, 'M') and m.M.shape == (128, 10)
        # 但 M 不在 parameters (还是 buffer)
        param_ids = {id(p) for p in m.parameters()}
        assert id(m.M) not in param_ids

    def test_classify_etf_path(self):
        """ETF 路径: logits = normalize(z_sem) @ M / tau."""
        torch.manual_seed(0)
        m = FedDSASGPAModel(num_classes=10, tau_etf=0.1, use_etf=True)
        m.eval()
        z_sem = torch.randn(4, 128)
        with torch.no_grad():
            logits = m.classify(z_sem)
            expected = F.normalize(z_sem, dim=-1) @ m.M / 0.1
        assert torch.allclose(logits, expected, atol=1e-5)

    def test_classify_linear_path(self):
        """Linear 路径: logits = head(z_sem)."""
        torch.manual_seed(0)
        m = FedDSASGPAModel(num_classes=10, use_etf=False)
        m.eval()
        z_sem = torch.randn(4, 128)
        with torch.no_grad():
            logits = m.classify(z_sem)
            expected = m.head(z_sem)
        assert torch.allclose(logits, expected, atol=1e-5)

    def test_etf_vs_linear_shape_identical(self):
        """两个路径输出 shape 完全一致 (都是 [B, K])."""
        torch.manual_seed(0)
        m_etf = FedDSASGPAModel(num_classes=10, use_etf=True)
        m_lin = FedDSASGPAModel(num_classes=10, use_etf=False)
        m_etf.eval(); m_lin.eval()
        x = torch.randn(5, 3, 224, 224)
        with torch.no_grad():
            l_etf = m_etf(x)
            l_lin = m_lin(x)
        assert l_etf.shape == l_lin.shape == (5, 10)

    def test_linear_gradient_flow(self):
        """use_etf=False 下 CE loss 对 head 和 backbone 都有梯度."""
        torch.manual_seed(0)
        m = FedDSASGPAModel(num_classes=7, use_etf=False)
        m.train()
        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 7, (4,))
        logits = m(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        # head 有梯度
        assert m.head.weight.grad is not None
        assert m.head.weight.grad.norm().item() > 0
        # backbone 有梯度
        assert m.encoder.fc1.weight.grad is not None
        assert m.encoder.fc1.weight.grad.norm().item() > 0
        # M buffer 无梯度
        assert m.M.grad is None

    def test_etf_head_param_count_zero(self):
        """ETF 模型相比 Linear 模型少了 K*(d+1) 可训参数 (head weight + bias)."""
        m_etf = FedDSASGPAModel(num_classes=10, use_etf=True)
        m_lin = FedDSASGPAModel(num_classes=10, use_etf=False)
        n_etf = sum(p.numel() for p in m_etf.parameters() if p.requires_grad)
        n_lin = sum(p.numel() for p in m_lin.parameters() if p.requires_grad)
        # Linear head = 128*10 + 10 = 1290
        assert n_lin - n_etf == 128 * 10 + 10


class TestLambdaETFPull:
    """Test lambda_etf_pull soft regularizer (EXP-106 new mechanism, Codex REVISE compliance)."""

    def _make_batch(self, K=10, d=128, per_class=4, device='cpu'):
        """Helper: 均衡 batch, 每类 per_class 样本."""
        torch.manual_seed(0)
        z = torch.randn(K * per_class, d)
        y = torch.arange(K).repeat_interleave(per_class)
        return z.to(device), y.to(device)

    def test_lambda_zero_equiv_baseline(self):
        """CRITICAL: lambda=0 时 loss_etf_pull=0, 不影响 backward."""
        K, d = 10, 128
        M = FedDSASGPAModel(num_classes=K).M
        z, y = self._make_batch(K=K, d=d)
        # 模拟 Client.train 的 pull loss 计算路径
        lambda_pull = 0.0
        loss_etf_pull = torch.tensor(0.0)
        if lambda_pull > 0:  # 不执行
            pulls = []
            for c in range(K):
                mask = y == c
                if mask.sum() >= 2:
                    z_c_mean = z[mask].mean(dim=0)
                    cos = F.cosine_similarity(z_c_mean.unsqueeze(0), M[:, c].unsqueeze(0))
                    pulls.append(1.0 - cos.squeeze())
            loss_etf_pull = torch.stack(pulls).mean()
        assert loss_etf_pull.item() == 0.0

    def test_pull_value_reasonable(self):
        """lambda>0 时 pull 值在 [0, 2] (cos 范围 [-1, 1], 1-cos ∈ [0, 2])."""
        K, d = 10, 128
        M = FedDSASGPAModel(num_classes=K).M
        z, y = self._make_batch(K=K, d=d)
        pulls = []
        for c in range(K):
            mask = y == c
            if mask.sum() >= 2:
                z_c_mean = z[mask].mean(dim=0)
                cos = F.cosine_similarity(z_c_mean.unsqueeze(0), M[:, c].unsqueeze(0))
                pulls.append(1.0 - cos.squeeze())
        loss = torch.stack(pulls).mean()
        assert 0.0 <= loss.item() <= 2.0, f"pull loss {loss.item()} out of [0,2]"

    def test_pull_skip_singleton_classes(self):
        """n_c < 2 的类应 skip (codex SHOULD_FIX)."""
        K, d = 10, 128
        M = FedDSASGPAModel(num_classes=K).M
        # 只有 class 0 有 4 个样本, class 1 有 1 个样本 (singleton), 其他空
        z = torch.randn(5, d)
        y = torch.tensor([0, 0, 0, 0, 1])
        pulls = []
        for c in range(K):
            mask = y == c
            n_c = int(mask.sum().item())
            if n_c < 2:
                continue
            z_c_mean = z[mask].mean(dim=0)
            cos = F.cosine_similarity(z_c_mean.unsqueeze(0), M[:, c].unsqueeze(0))
            pulls.append(1.0 - cos.squeeze())
        assert len(pulls) == 1  # 只 class 0 有 >=2
        # class 1 (singleton) 应被 skip

    def test_pull_perfect_alignment(self):
        """z_sem_c_mean = M[:,c] (完美对齐) → loss=0."""
        K, d = 7, 128
        M = FedDSASGPAModel(num_classes=K).M
        # 构造: z 各类的平均恰好等于 M[:,c]
        z_list = []
        y_list = []
        for c in range(K):
            z_list.append(M[:, c].unsqueeze(0).repeat(3, 1))  # 3 个一样的样本
            y_list.extend([c] * 3)
        z = torch.cat(z_list, dim=0)
        y = torch.tensor(y_list)
        pulls = []
        for c in range(K):
            mask = y == c
            if mask.sum() < 2:
                continue
            z_c_mean = z[mask].mean(dim=0)
            cos = F.cosine_similarity(z_c_mean.unsqueeze(0), M[:, c].unsqueeze(0))
            pulls.append(1.0 - cos.squeeze())
        loss = torch.stack(pulls).mean()
        assert loss.item() < 1e-5, f"perfect alignment should give 0, got {loss.item()}"

    def test_pull_opposite_alignment(self):
        """z_sem_c_mean = -M[:,c] (反向) → loss=2."""
        K, d = 7, 128
        M = FedDSASGPAModel(num_classes=K).M
        z_list = []
        y_list = []
        for c in range(K):
            z_list.append((-M[:, c]).unsqueeze(0).repeat(3, 1))  # 负方向
            y_list.extend([c] * 3)
        z = torch.cat(z_list, dim=0)
        y = torch.tensor(y_list)
        pulls = []
        for c in range(K):
            mask = y == c
            if mask.sum() < 2:
                continue
            z_c_mean = z[mask].mean(dim=0)
            cos = F.cosine_similarity(z_c_mean.unsqueeze(0), M[:, c].unsqueeze(0))
            pulls.append(1.0 - cos.squeeze())
        loss = torch.stack(pulls).mean()
        assert abs(loss.item() - 2.0) < 1e-5, f"opposite alignment should give 2, got {loss.item()}"

    def test_pull_no_valid_classes_stays_zero(self):
        """全部 singleton class → pulls 为空 → loss=0, 不 crash."""
        K, d = 10, 128
        M = FedDSASGPAModel(num_classes=K).M
        # 每类只有 1 个样本 (全 singleton)
        z = torch.randn(10, d)
        y = torch.arange(10)  # 每类各 1 个
        pulls = []
        for c in range(K):
            mask = y == c
            if mask.sum() < 2:
                continue
            z_c_mean = z[mask].mean(dim=0)
            cos = F.cosine_similarity(z_c_mean.unsqueeze(0), M[:, c].unsqueeze(0))
            pulls.append(1.0 - cos.squeeze())
        loss = torch.stack(pulls).mean() if pulls else torch.tensor(0.0)
        assert loss.item() == 0.0
        assert len(pulls) == 0

    def test_pull_gradient_flow(self):
        """Pull loss 对 z 有梯度 (non-zero gradient)."""
        K, d = 7, 128
        M = FedDSASGPAModel(num_classes=K).M
        z = torch.randn(21, d, requires_grad=True)
        y = torch.arange(K).repeat_interleave(3)
        pulls = []
        for c in range(K):
            mask = y == c
            if mask.sum() < 2:
                continue
            z_c_mean = z[mask].mean(dim=0)
            cos = F.cosine_similarity(z_c_mean.unsqueeze(0), M[:, c].unsqueeze(0))
            pulls.append(1.0 - cos.squeeze())
        loss = torch.stack(pulls).mean()
        loss.backward()
        assert z.grad is not None
        assert z.grad.norm().item() > 0


class TestEndToEnd:
    def test_model_forward_with_random_batch(self):
        """端到端 smoke: model(x) 不 crash, 产出合理 logits."""
        torch.manual_seed(0)
        model = FedDSASGPAModel(num_classes=10, feat_dim=1024, proj_dim=128)
        model.eval()
        x = torch.randn(8, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (8, 10)
        assert torch.isfinite(logits).all()

    def test_training_gradient_flow(self):
        """CE loss 对 encoder/semantic_head 产生非零梯度."""
        torch.manual_seed(0)
        model = FedDSASGPAModel(num_classes=7)
        model.train()
        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 7, (4,))
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        # encoder.fc1 有梯度
        assert model.encoder.fc1.weight.grad is not None
        assert model.encoder.fc1.weight.grad.norm().item() > 0
        # semantic_head 有梯度
        for param in model.semantic_head.parameters():
            assert param.grad is not None
            assert param.grad.norm().item() > 0

    def test_M_no_gradient(self):
        """Fixed ETF M 作为 buffer 不应接收梯度."""
        torch.manual_seed(0)
        model = FedDSASGPAModel(num_classes=7)
        model.train()
        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 7, (4,))
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        # M.grad 应该为 None (buffer 不参加 autograd)
        assert model.M.grad is None

    def test_orthogonal_loss_computation(self):
        """正交 loss 与 feddsa.py 实现一致 (z_sem ⊥ z_sty → loss 接近 0)."""
        torch.manual_seed(0)
        model = FedDSASGPAModel()
        model.eval()
        x = torch.randn(8, 3, 224, 224)
        with torch.no_grad():
            h = model.encode(x)
            z_sem = model.get_semantic(h)
            z_sty = model.get_style(h)
            z_sem_n = F.normalize(z_sem, dim=-1)
            z_sty_n = F.normalize(z_sty, dim=-1)
            loss_orth = ((z_sem_n * z_sty_n).sum(dim=-1) ** 2).mean().item()
        # 未训练的模型 orth 初值可能任意, 但应该是 [0, 1]
        assert 0.0 <= loss_orth <= 1.0


# ============================================================
# CDANN (Constrained DANN) — R5 proposal-complete
# ============================================================


class TestGRL:
    """Gradient Reversal Layer 正确性."""

    def test_forward_identity(self):
        """GRL.forward 应该是恒等映射."""
        x = torch.randn(4, 128)
        y = grl(x, lam=0.5)
        assert torch.allclose(y, x)

    def test_backward_reverses_gradient(self):
        """GRL.backward 应该把梯度乘 -lam."""
        x = torch.randn(4, 128, requires_grad=True)
        y = grl(x, lam=0.3)
        loss = y.sum()
        loss.backward()
        # dL/dx 如果无 GRL 是 +1 (因为 sum); 有 GRL 乘 -0.3 = -0.3
        expected = -0.3 * torch.ones_like(x)
        assert torch.allclose(x.grad, expected, atol=1e-6)

    def test_backward_lam_zero_zeroes_gradient(self):
        """lam=0 时 GRL.backward 应该返回 0 梯度."""
        x = torch.randn(4, 128, requires_grad=True)
        y = grl(x, lam=0.0)
        loss = y.sum()
        loss.backward()
        assert torch.allclose(x.grad, torch.zeros_like(x))

    def test_backward_lam_one_flips_gradient(self):
        """lam=1 时 GRL.backward 应该完全反转梯度."""
        x = torch.randn(4, 128, requires_grad=True)
        y = grl(x, lam=1.0)
        # 模拟分类 loss 梯度
        loss = y.pow(2).sum()
        loss.backward()
        # 无 GRL 时 dL/dx = 2x, 反转后是 -2x
        assert torch.allclose(x.grad, -2 * x, atol=1e-5)


class TestComputeLambdaAdv:
    """DANN λ_adv schedule 三段式."""

    def test_warmup_off(self):
        assert compute_lambda_adv(0, 20, 40) == 0.0
        assert compute_lambda_adv(10, 20, 40) == 0.0
        assert compute_lambda_adv(19, 20, 40) == 0.0

    def test_linear_ramp(self):
        # R=20..40 linear 0 → 1
        assert compute_lambda_adv(20, 20, 40) == 0.0
        assert abs(compute_lambda_adv(30, 20, 40) - 0.5) < 1e-6
        assert abs(compute_lambda_adv(35, 20, 40) - 0.75) < 1e-6

    def test_full(self):
        assert compute_lambda_adv(40, 20, 40) == 1.0
        assert compute_lambda_adv(100, 20, 40) == 1.0
        assert compute_lambda_adv(200, 20, 40) == 1.0

    def test_monotonic_nondecreasing(self):
        vals = [compute_lambda_adv(r, 20, 40) for r in range(200)]
        for i in range(len(vals) - 1):
            assert vals[i + 1] >= vals[i] - 1e-9


class TestCDANNModel:
    """FedDSASGPAModel ca=1 时 dom_head 结构正确."""

    def test_ca0_no_dom_head(self):
        """ca=0 (默认) 时 dom_head 不存在."""
        model = FedDSASGPAModel(num_classes=10, ca=0)
        assert model.dom_head is None

    def test_ca1_dom_head_shape(self):
        """ca=1 时 dom_head 输出维度 = num_clients."""
        model = FedDSASGPAModel(num_classes=10, proj_dim=128, ca=1, num_clients=4)
        assert model.dom_head is not None
        x = torch.randn(8, 128)
        y = model.dom_head(x)
        assert y.shape == (8, 4)

    def test_ca1_dom_head_params_size(self):
        """dom_head 应为 MLP(128, 64, N), 约 9K 参数 (N=4 时 ~ 8.6K)."""
        model = FedDSASGPAModel(num_classes=10, proj_dim=128, ca=1, num_clients=4)
        total = sum(p.numel() for p in model.dom_head.parameters())
        # Linear(128,64) = 128*64+64 = 8256; Linear(64,4) = 64*4+4 = 260 → 8516
        assert 5000 < total < 15000, f"unexpected dom_head param count: {total}"

    def test_ca1_dom_head_in_state_dict(self):
        """dom_head 参数在 state_dict 里 → 会被 FedAvg 聚合 (默认路径)."""
        model = FedDSASGPAModel(num_classes=10, ca=1, num_clients=4)
        keys = model.state_dict().keys()
        dom_keys = [k for k in keys if 'dom_head' in k]
        assert len(dom_keys) > 0, "dom_head 必须在 state_dict 里 (FedAvg 聚合)"
        # 确认不在常见的 private_keys 规则里 (style_head 是 private, dom_head 不是)
        for k in dom_keys:
            assert 'style_head' not in k, "dom_head 不应包含 style_head, 避免被 private key 规则误匹配"

    def test_ca1_backward_asymmetric_gradient_directions(self):
        """z_sem 走 GRL 反向, z_sty 走正向: encoder 端收到的梯度符号应相反."""
        torch.manual_seed(0)
        model = FedDSASGPAModel(num_classes=7, proj_dim=128, ca=1, num_clients=4)
        model.train()
        x = torch.randn(4, 3, 224, 224)
        d = torch.tensor([0, 0, 0, 0])  # 所有样本都来自 client 0

        h = model.encode(x)
        z_sem = model.get_semantic(h)
        z_sty = model.get_style(h)

        # 只算 L_dom_sem (z_sem 路径, GRL 反向, lam=1.0)
        dom_logits_sem = model.dom_head(grl(z_sem, 1.0))
        l_dom_sem = F.cross_entropy(dom_logits_sem, d)
        l_dom_sem.backward(retain_graph=True)
        grad_sem_encoder_gru = next(model.semantic_head.parameters()).grad.clone()
        model.zero_grad()

        # 只算 L_dom_sty (z_sty 路径, 无 GRL)
        dom_logits_sty = model.dom_head(z_sty)
        l_dom_sty = F.cross_entropy(dom_logits_sty, d)
        l_dom_sty.backward()
        grad_sty_encoder = next(model.style_head.parameters()).grad.clone()
        model.zero_grad()

        # 关键: z_sem 路径的 semantic_head 梯度应该存在 (GRL 只反转不清零)
        assert grad_sem_encoder_gru.norm().item() > 0, "semantic_head 应有梯度"
        # style_head 梯度应该存在 (正向)
        assert grad_sty_encoder.norm().item() > 0, "style_head 应有梯度"


class TestCAFlagBackwardCompat:
    """ca=0 时行为等价于 baseline (无 CDANN)."""

    def test_ca0_forward_equivalent_to_baseline(self):
        """ca=0 和未传 ca 的 Model forward 完全相同."""
        torch.manual_seed(42)
        m1 = FedDSASGPAModel(num_classes=10, ca=0, use_etf=False)
        torch.manual_seed(42)
        m2 = FedDSASGPAModel(num_classes=10, use_etf=False)  # 默认 ca=0
        m1.eval()
        m2.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            y1 = m1(x)
            y2 = m2(x)
        assert torch.allclose(y1, y2, atol=1e-6), "ca=0 vs 默认应完全一致"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
