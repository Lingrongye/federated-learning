"""
FedDSA-SGPA Diagnostic Logger 单元测试.

覆盖:
- Layer 1 训练端 6 个 metric 的正常/edge case
- Layer 2 聚合端 2 个 metric
- Layer 3 推理端 13 个 metric
- 接口 record/dump (含 edge case: 空 buffer / 多 client / jsonl 格式)
- 类测试 (端到端 workflow)

运行:
    D:/anaconda/python.exe -m pytest FDSE_CVPR25/tests/test_sgpa_diagnostic_logger.py -v
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# 添加 FDSE_CVPR25 到 path 以 import diagnostics
TESTS_DIR = Path(__file__).parent
FDSE_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(FDSE_ROOT))

from diagnostics.sgpa_diagnostic_logger import SGPADiagnosticLogger as DL


# =========================================================================
# Layer 1 训练端
# =========================================================================


class TestOrthogonality:
    def test_identical_vectors(self):
        """两向量完全相同 → cos² = 1."""
        z = torch.randn(10, 128)
        assert DL.orthogonality(z, z) > 0.99

    def test_orthogonal_vectors(self):
        """完全正交向量 → cos² ≈ 0."""
        z_sem = torch.zeros(4, 128)
        z_sem[:, 0] = 1.0
        z_sty = torch.zeros(4, 128)
        z_sty[:, 1] = 1.0
        assert DL.orthogonality(z_sem, z_sty) < 1e-6

    def test_antiparallel(self):
        """反向平行 → cos² = 1."""
        z = torch.randn(10, 128)
        assert DL.orthogonality(z, -z) > 0.99

    def test_batch_size_1(self):
        """edge case: B=1, normalize 不崩."""
        z_sem = torch.randn(1, 128)
        z_sty = torch.randn(1, 128)
        val = DL.orthogonality(z_sem, z_sty)
        assert 0.0 <= val <= 1.0


class TestETFAlignment:
    def test_perfect_alignment(self):
        """z_sem_c_mean = M[:, c] → align = 1."""
        K, d = 7, 128
        torch.manual_seed(0)
        U, _ = torch.linalg.qr(torch.randn(d, K))
        M = U @ (torch.eye(K) - torch.ones(K, K) / K) * math.sqrt(K / (K - 1))

        z_sem = torch.zeros(K * 5, d)
        labels = torch.zeros(K * 5, dtype=torch.long)
        for c in range(K):
            z_sem[c * 5:(c + 1) * 5] = M[:, c].unsqueeze(0).expand(5, -1)
            labels[c * 5:(c + 1) * 5] = c

        mean_align, per_class = DL.etf_alignment(z_sem, labels, M, K)
        assert mean_align > 0.99
        assert len(per_class) == K
        assert all(a > 0.99 for a in per_class)

    def test_random_vs_etf(self):
        """随机特征 → align 接近 0."""
        K, d = 7, 128
        torch.manual_seed(42)
        U, _ = torch.linalg.qr(torch.randn(d, K))
        M = U @ (torch.eye(K) - torch.ones(K, K) / K) * math.sqrt(K / (K - 1))

        z_sem = torch.randn(K * 20, d)
        labels = torch.arange(K).repeat_interleave(20)
        mean_align, _ = DL.etf_alignment(z_sem, labels, M, K)
        assert abs(mean_align) < 0.3  # 随机应接近 0

    def test_missing_classes(self):
        """部分类没样本 → 只对有的类算."""
        K, d = 7, 128
        M = torch.randn(d, K)
        z_sem = torch.randn(10, d)
        labels = torch.zeros(10, dtype=torch.long)  # 全是类 0
        mean_align, per_class = DL.etf_alignment(z_sem, labels, M, K)
        assert len(per_class) == 1  # 只有类 0 有样本

    def test_empty_batch(self):
        """空 batch → 返回 0, []."""
        K, d = 7, 128
        M = torch.randn(d, K)
        z_sem = torch.zeros(0, d)
        labels = torch.zeros(0, dtype=torch.long)
        mean_align, per_class = DL.etf_alignment(z_sem, labels, M, K)
        assert mean_align == 0.0
        assert per_class == []


class TestIntraClassSimilarity:
    def test_identical_samples(self):
        """同类样本全相同 → cos = 1."""
        z = torch.randn(1, 128)
        z_expanded = z.expand(5, -1).contiguous()
        labels = torch.zeros(5, dtype=torch.long)
        val = DL.intra_class_similarity(z_expanded, labels, num_classes=1)
        assert val > 0.99

    def test_single_sample_class(self):
        """某类只有 1 个样本 → 跳过."""
        z = torch.randn(3, 128)
        labels = torch.tensor([0, 1, 2])  # 每类 1 个
        val = DL.intra_class_similarity(z, labels, num_classes=3)
        assert val == 0.0

    def test_mixed(self):
        """同类相似 + 跨类低 → 接近 1."""
        z = torch.zeros(6, 128)
        z[0:3] = torch.tensor([1.0] + [0.0] * 127)  # 类 0 全 [1,0,...]
        z[3:6] = torch.tensor([0.0, 1.0] + [0.0] * 126)  # 类 1 全 [0,1,...]
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        val = DL.intra_class_similarity(z, labels, num_classes=2)
        assert val > 0.99


class TestInterClassSimilarity:
    def test_orthogonal_centers(self):
        """不同类 centers 正交 → 0."""
        K, d = 4, 128
        z = torch.zeros(K * 3, d)
        labels = torch.zeros(K * 3, dtype=torch.long)
        for c in range(K):
            z[c * 3:(c + 1) * 3, c] = 1.0
            labels[c * 3:(c + 1) * 3] = c
        val = DL.inter_class_similarity(z, labels, num_classes=K)
        assert abs(val) < 1e-5

    def test_identical_centers(self):
        """所有类 center 相同 → 1."""
        d = 128
        z = torch.ones(9, d)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        val = DL.inter_class_similarity(z, labels, num_classes=3)
        assert val > 0.99

    def test_single_class(self):
        """只有 1 类 → 返回 0."""
        z = torch.randn(5, 128)
        labels = torch.zeros(5, dtype=torch.long)
        val = DL.inter_class_similarity(z, labels, num_classes=1)
        assert val == 0.0


class TestGradientCosine:
    def test_parallel(self):
        g = torch.randn(1000)
        assert DL.gradient_cosine(g, g) > 0.999

    def test_antiparallel(self):
        g = torch.randn(1000)
        assert DL.gradient_cosine(g, -g) < -0.999

    def test_orthogonal(self):
        g1 = torch.zeros(1000); g1[0] = 1.0
        g2 = torch.zeros(1000); g2[1] = 1.0
        assert abs(DL.gradient_cosine(g1, g2)) < 1e-6


class TestGradientNormRatio:
    def test_equal(self):
        g = torch.randn(100)
        ratio = DL.gradient_norm_ratio(g, g)
        assert abs(ratio - 1.0) < 1e-5

    def test_triple_ratio(self):
        g1 = torch.ones(10) * 3.0
        g2 = torch.ones(10)
        ratio = DL.gradient_norm_ratio(g1, g2)
        assert abs(ratio - 3.0) < 1e-4

    def test_zero_orth(self):
        """∇L_orth = 0 → ratio 是大数但不 inf."""
        g_ce = torch.ones(10)
        g_orth = torch.zeros(10)
        ratio = DL.gradient_norm_ratio(g_ce, g_orth)
        assert ratio > 1e6  # 大数
        assert not math.isinf(ratio)


# =========================================================================
# Layer 2 聚合端
# =========================================================================


class TestClientCenterVariance:
    def test_identical(self):
        """所有 client 相同 → 方差 0."""
        c = torch.randn(7, 128)
        centers = [c.clone() for _ in range(4)]
        val = DL.client_center_variance(centers)
        assert val < 1e-10

    def test_nonzero(self):
        """随机不同 client → 方差 > 0."""
        torch.manual_seed(0)
        centers = [torch.randn(7, 128) for _ in range(4)]
        val = DL.client_center_variance(centers)
        assert val > 0.1  # 标准正态 ≈ 1

    def test_two_clients(self):
        c1 = torch.zeros(3, 4)
        c2 = torch.ones(3, 4)
        # 每个位置 (0, 1), var = 0.25 (unbiased=False)
        val = DL.client_center_variance([c1, c2])
        assert abs(val - 0.25) < 1e-5


class TestParamDrift:
    def test_zero_drift(self):
        """client = global → drift = 0."""
        p = torch.randn(100)
        clients = [p.clone() for _ in range(3)]
        val = DL.param_drift(clients, p)
        assert val < 1e-6

    def test_unit_drift(self):
        """client = global + unit → drift = sqrt(len)."""
        p = torch.zeros(100)
        clients = [torch.ones(100) for _ in range(3)]
        val = DL.param_drift(clients, p)
        assert abs(val - 10.0) < 1e-3  # ‖ones(100)‖ = 10


# =========================================================================
# Layer 3 推理/SGPA
# =========================================================================


class TestGateRates:
    def test_all_reliable(self):
        H = torch.zeros(10)
        dist = torch.zeros(10)
        r = DL.gate_rates(H, dist, tau_H=1.0, tau_S=1.0)
        assert r['reliable_rate'] == 1.0
        assert r['entropy_rate'] == 1.0
        assert r['dist_rate'] == 1.0

    def test_none_reliable(self):
        H = torch.ones(10) * 100
        dist = torch.ones(10) * 100
        r = DL.gate_rates(H, dist, tau_H=0.01, tau_S=0.01)
        assert r['reliable_rate'] == 0.0

    def test_half_each(self):
        """entropy gate 一半通过, dist gate 全通过, 交集一半."""
        H = torch.tensor([0.0, 0.0, 100.0, 100.0])
        dist = torch.tensor([0.0, 0.0, 0.0, 0.0])
        r = DL.gate_rates(H, dist, tau_H=1.0, tau_S=1.0)
        assert r['reliable_rate'] == 0.5
        assert r['entropy_rate'] == 0.5
        assert r['dist_rate'] == 1.0


class TestDistDistribution:
    def test_basic(self):
        arr = torch.arange(11, dtype=torch.float)  # 0..10
        d = DL.dist_distribution(arr)
        assert d['dist_min_min'] == 0.0
        assert d['dist_min_max'] == 10.0
        assert d['dist_min_p50'] == 5.0


class TestWhiteningReduction:
    def test_reduces(self):
        """白化后 scatter 小 → ratio < 1."""
        torch.manual_seed(0)
        B, N, d = 10, 4, 8
        z_raw = torch.randn(B, d) * 10
        mu_raw = torch.randn(N, d) * 10
        z_white = z_raw / 10
        mu_white = mu_raw / 10
        r = DL.whitening_reduction(z_raw, z_white, mu_raw, mu_white)
        assert r['reduction_ratio'] < 0.1
        assert r['white_scatter'] < r['raw_scatter']

    def test_no_reduction(self):
        """raw 和 white 一样 → ratio = 1."""
        z = torch.randn(5, 4)
        mu = torch.randn(3, 4)
        r = DL.whitening_reduction(z, z, mu, mu)
        assert abs(r['reduction_ratio'] - 1.0) < 1e-5


class TestSigmaConditionNumber:
    def test_identity(self):
        I = torch.eye(8)
        cond = DL.sigma_condition_number(I)
        assert abs(cond - 1.0) < 1e-3

    def test_degenerate(self):
        """近奇异矩阵 → cond 大."""
        M = torch.eye(4)
        M[0, 0] = 1e-6
        cond = DL.sigma_condition_number(M)
        assert cond > 1e5


class TestProtoFill:
    def test_mixed(self):
        supports = {0: [1, 2, 3], 1: [1], 2: []}
        fill, avg = DL.proto_fill(supports, num_classes=3)
        assert fill == {0: 3, 1: 1, 2: 0}
        assert abs(avg - 4.0 / 3.0) < 1e-6

    def test_missing_class(self):
        """某类不在 dict 里 → 填 0."""
        supports = {0: [1, 2]}
        fill, _ = DL.proto_fill(supports, num_classes=3)
        assert fill == {0: 2, 1: 0, 2: 0}


class TestProtoETFOffset:
    def test_aligned(self):
        """proto = M[:, c] → offset = 0."""
        K, d = 4, 8
        M = torch.randn(d, K)
        proto = [M[:, c].clone() for c in range(K)]
        mean_off, _ = DL.proto_etf_offset(proto, M, K)
        assert abs(mean_off) < 1e-5

    def test_opposite(self):
        """proto = -M[:, c] → offset = 2."""
        K, d = 4, 8
        M = torch.randn(d, K)
        proto = [-M[:, c].clone() for c in range(K)]
        mean_off, offsets = DL.proto_etf_offset(proto, M, K)
        assert all(abs(o - 2.0) < 1e-5 for o in offsets)

    def test_none_proto(self):
        """proto[c] = None → offset = 0 (跳过)."""
        K, d = 4, 8
        M = torch.randn(d, K)
        proto = [None] * K
        mean_off, offsets = DL.proto_etf_offset(proto, M, K)
        assert mean_off == 0.0
        assert all(o == 0.0 for o in offsets)

    def test_zero_proto(self):
        """proto[c] 是零向量 → offset = 0."""
        K, d = 4, 8
        M = torch.randn(d, K)
        proto = [torch.zeros(d) for _ in range(K)]
        mean_off, _ = DL.proto_etf_offset(proto, M, K)
        assert mean_off == 0.0


class TestFallbackRate:
    def test_basic(self):
        activated = torch.tensor([True, True, False, False, True])
        r = DL.fallback_rate(activated)
        assert abs(r - 0.4) < 1e-6

    def test_all_activated(self):
        activated = torch.ones(5, dtype=torch.bool)
        assert DL.fallback_rate(activated) == 0.0

    def test_none_activated(self):
        activated = torch.zeros(5, dtype=torch.bool)
        assert DL.fallback_rate(activated) == 1.0


class TestPredictionAccuracy:
    def test_with_labels(self):
        labels = torch.tensor([0, 1, 2, 3])
        pred_proto = torch.tensor([0, 1, 2, 0])
        pred_etf = torch.tensor([0, 1, 0, 3])
        r = DL.prediction_accuracy(pred_proto, pred_etf, labels)
        assert abs(r['proto_acc'] - 0.75) < 1e-6
        assert abs(r['etf_acc'] - 0.75) < 1e-6
        assert abs(r['pred_agree'] - 0.5) < 1e-6
        assert abs(r['proto_vs_etf_gain']) < 1e-6

    def test_without_labels(self):
        pred_proto = torch.tensor([0, 1, 2])
        pred_etf = torch.tensor([0, 1, 3])
        r = DL.prediction_accuracy(pred_proto, pred_etf, labels=None)
        assert 'proto_acc' not in r
        assert 'pred_agree' in r
        assert abs(r['pred_agree'] - 2.0 / 3.0) < 1e-6

    def test_proto_better(self):
        """proto 比 etf 更准."""
        labels = torch.tensor([0, 1, 2, 3])
        pred_proto = torch.tensor([0, 1, 2, 3])  # 100%
        pred_etf = torch.tensor([0, 0, 0, 0])  # 25%
        r = DL.prediction_accuracy(pred_proto, pred_etf, labels)
        assert r['proto_vs_etf_gain'] == 0.75


# =========================================================================
# 接口测试: record + dump
# =========================================================================


class TestRecordAndDump:
    def test_basic_dump(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = DL(client_id=0, stage='train', log_dir=tmpdir, dump_every_n=2)
            dl.record(round_id=1, metrics_dict={'orth': 0.05, 'etf_align': 0.3})
            dl.record(round_id=2, metrics_dict={'orth': 0.04, 'etf_align': 0.5})
            # 触发 dump
            path = Path(tmpdir) / 'diag_train_client0.jsonl'
            assert path.exists()
            lines = path.read_text(encoding='utf-8').strip().split('\n')
            assert len(lines) == 2
            d1 = json.loads(lines[0])
            assert d1['_round'] == 1
            assert d1['_client'] == 0
            assert d1['_stage'] == 'train'
            assert d1['orth'] == 0.05

    def test_manual_dump_then_continue(self):
        """手动 dump 后继续 record 应该 append."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = DL(client_id=5, stage='test', log_dir=tmpdir, dump_every_n=100)
            dl.record(round_id=1, metrics_dict={'reliable_rate': 0.5})
            dl.dump()  # 手动 flush
            dl.record(round_id=2, metrics_dict={'reliable_rate': 0.6})
            dl.dump()
            path = Path(tmpdir) / 'diag_test_client5.jsonl'
            lines = path.read_text(encoding='utf-8').strip().split('\n')
            assert len(lines) == 2

    def test_invalid_stage(self):
        with pytest.raises(AssertionError):
            DL(client_id=0, stage='invalid', log_dir='/tmp')

    def test_dump_empty_no_file(self):
        """空 buffer dump 不创建文件."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = DL(client_id=0, stage='train', log_dir=tmpdir)
            dl.dump()
            path = Path(tmpdir) / 'diag_train_client0.jsonl'
            assert not path.exists()

    def test_tensor_serialization(self):
        """tensor 自动转 float/list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = DL(client_id=0, stage='train', log_dir=tmpdir, dump_every_n=1)
            dl.record(round_id=1, metrics_dict={
                'scalar_tensor': torch.tensor(3.14),
                'array_tensor': torch.tensor([1.0, 2.0]),
                'numpy_val': np.float32(2.71),
            })
            path = Path(tmpdir) / 'diag_train_client0.jsonl'
            d = json.loads(path.read_text(encoding='utf-8').strip())
            assert abs(d['scalar_tensor'] - 3.14) < 1e-3
            assert d['array_tensor'] == [1.0, 2.0]
            assert abs(d['numpy_val'] - 2.71) < 1e-3

    def test_multi_client_separate_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl0 = DL(client_id=0, stage='train', log_dir=tmpdir, dump_every_n=1)
            dl1 = DL(client_id=1, stage='train', log_dir=tmpdir, dump_every_n=1)
            dl0.record(1, {'metric': 0.1})
            dl1.record(1, {'metric': 0.9})
            assert (Path(tmpdir) / 'diag_train_client0.jsonl').exists()
            assert (Path(tmpdir) / 'diag_train_client1.jsonl').exists()


# =========================================================================
# End-to-end workflow
# =========================================================================


class TestEndToEnd:
    def test_full_training_round(self):
        """模拟一轮训练: 计算所有 Layer 1 指标并 record."""
        torch.manual_seed(0)
        K, d, B = 7, 128, 64
        z_sem = torch.randn(B, d)
        z_sty = torch.randn(B, d)
        labels = torch.randint(0, K, (B,))
        U, _ = torch.linalg.qr(torch.randn(d, K))
        M = U @ (torch.eye(K) - torch.ones(K, K) / K) * math.sqrt(K / (K - 1))
        g_ce = torch.randn(10000)
        g_orth = torch.randn(10000) * 0.1  # 小 L_orth 梯度

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = DL(client_id=0, stage='train', log_dir=tmpdir, dump_every_n=1)
            metrics = {
                'orth': DL.orthogonality(z_sem, z_sty),
                'etf_align_mean': DL.etf_alignment(z_sem, labels, M, K)[0],
                'intra_cls': DL.intra_class_similarity(z_sem, labels, K),
                'inter_cls': DL.inter_class_similarity(z_sem, labels, K),
                'cos_grad': DL.gradient_cosine(g_ce, g_orth),
                'grad_norm_ratio': DL.gradient_norm_ratio(g_ce, g_orth),
            }
            dl.record(round_id=0, metrics_dict=metrics)

            path = Path(tmpdir) / 'diag_train_client0.jsonl'
            d = json.loads(path.read_text(encoding='utf-8').strip())
            # 所有 metric 都应该被记录
            assert 'orth' in d
            assert 'etf_align_mean' in d
            assert 'intra_cls' in d
            assert 'inter_cls' in d
            assert 'cos_grad' in d
            assert 'grad_norm_ratio' in d
            # grad_norm_ratio 应该 > 1 (g_ce 比 g_orth 大)
            assert d['grad_norm_ratio'] > 5

    def test_full_inference_batch(self):
        """模拟一次推理 batch: 所有 Layer 3 指标."""
        torch.manual_seed(0)
        K, d, d_sty, B, N = 7, 128, 128, 32, 4

        # 伪造 test 数据
        entropy = torch.rand(B) * 2.0
        dist_min = torch.rand(B)
        z_sty_raw = torch.randn(B, d_sty) * 5
        mu_raw = torch.randn(N, d_sty) * 5
        z_sty_white = z_sty_raw / 5
        mu_white = mu_raw / 5
        Sigma = torch.eye(d_sty) + 0.1 * torch.randn(d_sty, d_sty)
        Sigma = Sigma @ Sigma.T  # 正定
        supports = {c: list(range(c * 3)) for c in range(K)}
        U, _ = torch.linalg.qr(torch.randn(d, K))
        M = U @ (torch.eye(K) - torch.ones(K, K) / K) * math.sqrt(K / (K - 1))
        proto = [torch.randn(d) for _ in range(K)]
        activated = torch.rand(B) > 0.3
        labels = torch.randint(0, K, (B,))
        pred_proto = torch.randint(0, K, (B,))
        pred_etf = torch.randint(0, K, (B,))

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = DL(client_id=1, stage='test', log_dir=tmpdir, dump_every_n=1)
            metrics = {}
            metrics.update(DL.gate_rates(entropy, dist_min, tau_H=1.0, tau_S=0.5))
            metrics.update(DL.dist_distribution(dist_min))
            metrics.update(DL.whitening_reduction(z_sty_raw, z_sty_white, mu_raw, mu_white))
            metrics['sigma_cond'] = DL.sigma_condition_number(Sigma)
            _, metrics['proto_fill_mean'] = DL.proto_fill(supports, K)
            metrics['proto_etf_offset_mean'], _ = DL.proto_etf_offset(proto, M, K)
            metrics['fallback_rate'] = DL.fallback_rate(activated)
            metrics.update(DL.prediction_accuracy(pred_proto, pred_etf, labels))

            dl.record(round_id=100, metrics_dict=metrics)
            path = Path(tmpdir) / 'diag_test_client1.jsonl'
            d = json.loads(path.read_text(encoding='utf-8').strip())
            # 全部 Layer 3 关键指标都应存在
            for key in ['reliable_rate', 'entropy_rate', 'dist_rate',
                        'dist_min_p50', 'reduction_ratio', 'sigma_cond',
                        'proto_fill_mean', 'proto_etf_offset_mean', 'fallback_rate',
                        'proto_acc', 'etf_acc', 'pred_agree', 'proto_vs_etf_gain']:
                assert key in d, f"missing {key}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
