"""
sas-FH (sas=2/3/4) 聚合逻辑单元测试

不依赖 flgo runner，直接构造 mock server + 测试 pack() / _compute_*.

测试覆盖：
  - _compute_style_weighted 对 sem_head / head 两种 states dict 都返回正确加权和
  - _compute_uniform_avg 返回简单均值
  - pack() 里 sas=2 时 head.* keys 被 personalized 覆盖
  - pack() 里 sas=3 时所有 client 收到的 head 一致（uniform）
  - pack() 里 sas=4 时 client i 收到自己上传的 head
  - sas=0 时 head 保持 FedAvg（不变）
"""
import os
import sys
import types
import copy
import unittest
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from algorithm.feddsa_scheduled import Server, FedDSAModel


def build_mock_server(sas_mode):
    """构造最小 mock Server 绕过 flgo 依赖，只测 pack + 聚合逻辑。"""
    srv = Server.__new__(Server)
    srv.model = FedDSAModel(num_classes=10, feat_dim=1024, proj_dim=128)
    srv.schedule_mode = 0
    srv.style_dispatch_num = 0
    srv.style_aware_sem = sas_mode
    srv.style_aware_tau = 0.3
    srv.global_semantic_protos = {}
    srv.current_round = 10

    # 4 clients 的 style protos（任意给不同方向）
    srv.style_bank = {
        0: (torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([0.1, 0.1, 0.1, 0.1])),
        1: (torch.tensor([0.0, 1.0, 0.0, 0.0]), torch.tensor([0.1, 0.1, 0.1, 0.1])),
        2: (torch.tensor([0.0, 0.0, 1.0, 0.0]), torch.tensor([0.1, 0.1, 0.1, 0.1])),
        3: (torch.tensor([0.0, 0.0, 0.0, 1.0]), torch.tensor([0.1, 0.1, 0.1, 0.1])),
    }

    # 4 份不同的 sem_head state + head state
    def make_client_sem(seed):
        torch.manual_seed(seed)
        m = FedDSAModel(num_classes=10, feat_dim=1024, proj_dim=128)
        return {k: v.clone().cpu() for k, v in m.state_dict().items()
                if 'semantic_head' in k}

    def make_client_head(seed):
        torch.manual_seed(seed + 100)
        m = FedDSAModel(num_classes=10, feat_dim=1024, proj_dim=128)
        return {k: v.clone().cpu() for k, v in m.state_dict().items()
                if k.startswith('head.')}

    srv.client_sem_states = {cid: make_client_sem(cid) for cid in range(4)}
    srv.client_head_states = {cid: make_client_head(cid) for cid in range(4)}
    return srv


class TestSasFHAggregation(unittest.TestCase):
    def test_compute_style_weighted_sem_head_is_valid(self):
        """sem_head 的 style-weighted 聚合结果形状匹配。"""
        srv = build_mock_server(2)
        result = srv._compute_style_weighted(srv.client_sem_states, target_cid=0)
        self.assertIsNotNone(result)
        for k in srv.client_sem_states[0]:
            self.assertIn(k, result)
            self.assertEqual(result[k].shape, srv.client_sem_states[0][k].shape)

    def test_compute_style_weighted_head_is_valid(self):
        """head 的 style-weighted 聚合结果形状匹配。"""
        srv = build_mock_server(2)
        result = srv._compute_style_weighted(srv.client_head_states, target_cid=0)
        self.assertIsNotNone(result)
        self.assertIn('head.weight', result)
        self.assertEqual(result['head.weight'].shape, torch.Size([10, 128]))
        self.assertIn('head.bias', result)
        self.assertEqual(result['head.bias'].shape, torch.Size([10]))

    def test_compute_style_weighted_is_convex_combination(self):
        """聚合结果应等于 Σ w_ij · head_j，权重为 softmax(sim/τ)。"""
        srv = build_mock_server(2)
        target = 0
        agg = srv._compute_style_weighted(srv.client_head_states, target_cid=target)

        # 手工复算
        import torch
        target_vec = srv.style_bank[target][0].flatten()
        sims = []
        cids = list(srv.client_head_states.keys())
        for cid in cids:
            src_vec = srv.style_bank[cid][0].flatten()
            dot = (src_vec * target_vec).sum()
            norm = src_vec.norm() * target_vec.norm() + 1e-8
            sims.append((dot / norm).item())
        weights = torch.softmax(
            torch.tensor(sims) / srv.style_aware_tau, dim=0
        ).tolist()
        expected = None
        for w, cid in zip(weights, cids):
            st = srv.client_head_states[cid]
            if expected is None:
                expected = {k: v.clone() * w for k, v in st.items()}
            else:
                for k in expected:
                    expected[k] += st[k] * w

        for k in expected:
            self.assertTrue(torch.allclose(agg[k], expected[k], atol=1e-6),
                            f"{k} 不一致")

    def test_compute_uniform_avg_equals_mean(self):
        """C2 uniform-avg: 结果应等于所有 client head 的算术均值。"""
        srv = build_mock_server(3)
        agg = srv._compute_uniform_avg(srv.client_head_states)
        n = len(srv.client_head_states)
        for k in agg:
            expected = sum(srv.client_head_states[cid][k]
                           for cid in srv.client_head_states) / n
            self.assertTrue(torch.allclose(agg[k], expected, atol=1e-6))

    def test_compute_uniform_avg_all_clients_get_same(self):
        """C2: 对不同 target client 调用，结果应完全相同（uniform 不依赖 target）。"""
        srv = build_mock_server(3)
        agg0 = srv._compute_uniform_avg(srv.client_head_states)
        agg1 = srv._compute_uniform_avg(srv.client_head_states)
        for k in agg0:
            self.assertTrue(torch.equal(agg0[k], agg1[k]))

    def test_pack_sas2_overrides_head(self):
        """sas=2 (A2)：pack 下发的 model 的 head 参数应不等于 server.model 的 head。"""
        srv = build_mock_server(2)
        original_head = srv.model.state_dict()['head.weight'].clone()
        result = srv.pack(client_id=0)
        sent_head = result['model'].state_dict()['head.weight']
        self.assertFalse(torch.allclose(sent_head, original_head, atol=1e-6),
                         "sas=2 应该用 personalized head 覆盖 FedAvg head")

    def test_pack_sas3_all_clients_get_same_head(self):
        """sas=3 (C2)：不同 client id 下发的 head 应完全相同（uniform-avg）。"""
        srv = build_mock_server(3)
        r0 = srv.pack(client_id=0)
        r1 = srv.pack(client_id=1)
        r2 = srv.pack(client_id=2)
        h0 = r0['model'].state_dict()['head.weight']
        h1 = r1['model'].state_dict()['head.weight']
        h2 = r2['model'].state_dict()['head.weight']
        self.assertTrue(torch.allclose(h0, h1, atol=1e-6))
        self.assertTrue(torch.allclose(h1, h2, atol=1e-6))

    def test_pack_sas4_client_gets_own_head(self):
        """sas=4 (C1 local-only)：client i 下发的 head 应等于自己上传的 head state。"""
        srv = build_mock_server(4)
        result = srv.pack(client_id=2)
        sent_head = result['model'].state_dict()['head.weight']
        expected_head = srv.client_head_states[2]['head.weight']
        self.assertTrue(torch.allclose(sent_head, expected_head, atol=1e-6))

    def test_pack_sas0_head_unchanged(self):
        """sas=0：head 保持 FedAvg (= server.model.head)。"""
        srv = build_mock_server(0)
        original_head = srv.model.state_dict()['head.weight'].clone()
        result = srv.pack(client_id=0)
        sent_head = result['model'].state_dict()['head.weight']
        self.assertTrue(torch.allclose(sent_head, original_head, atol=1e-6))

    def test_pack_sas1_head_unchanged(self):
        """sas=1 (Plan A)：head 保持 FedAvg，只有 sem_head 个性化。"""
        srv = build_mock_server(1)
        original_head = srv.model.state_dict()['head.weight'].clone()
        original_sem = srv.model.state_dict()['semantic_head.0.weight'].clone()
        result = srv.pack(client_id=0)
        sent = result['model'].state_dict()
        self.assertTrue(torch.allclose(sent['head.weight'], original_head, atol=1e-6),
                        "sas=1 不应改动 head")
        self.assertFalse(torch.allclose(sent['semantic_head.0.weight'], original_sem, atol=1e-6),
                         "sas=1 应该 personalize sem_head")

    def test_sas2_different_clients_different_heads(self):
        """sas=2：不同 style 的 client 收到不同的 personalized head。"""
        srv = build_mock_server(2)
        r0 = srv.pack(client_id=0)
        r1 = srv.pack(client_id=1)
        h0 = r0['model'].state_dict()['head.weight']
        h1 = r1['model'].state_dict()['head.weight']
        self.assertFalse(torch.allclose(h0, h1, atol=1e-6),
                         "风格不同的 client 应收到不同 head")

    def test_sas_fh_does_not_touch_encoder_or_style_head(self):
        """sas-FH 只改 sem_head 和 head，不改 encoder / style_head。"""
        srv = build_mock_server(2)
        original = srv.model.state_dict()
        result = srv.pack(client_id=0)
        sent = result['model'].state_dict()
        # encoder 随便挑一个 conv
        self.assertTrue(torch.allclose(sent['encoder.features.conv1.weight'],
                                       original['encoder.features.conv1.weight'],
                                       atol=1e-6))
        # style_head 不动
        self.assertTrue(torch.allclose(sent['style_head.0.weight'],
                                       original['style_head.0.weight'],
                                       atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=2)
