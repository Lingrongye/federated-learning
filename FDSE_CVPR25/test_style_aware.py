"""测试方案 A: style-aware semantic head aggregation"""
import os, sys, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ast

with open(os.path.join(os.path.dirname(__file__), 'algorithm/feddsa_scheduled.py'), encoding='utf-8') as f:
    ast.parse(f.read())
print('[PASS] Syntax OK')

import importlib.util
spec = importlib.util.spec_from_file_location(
    'feddsa_scheduled',
    os.path.join(os.path.dirname(__file__), 'algorithm/feddsa_scheduled.py')
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('[PASS] Import OK')

import inspect
src = inspect.getsource(mod.Server.initialize)
assert "'sas': 0" in src and "'sas_tau': 0.3" in src
assert 'self.style_aware_sem' in src and 'self.style_aware_tau' in src
assert 'self.client_sem_states' in src
print('[PASS] sas/sas_tau flag + client_sem_states 初始化正确')

# 方法存在
assert hasattr(mod.Server, '_compute_style_weighted_sem')
print('[PASS] _compute_style_weighted_sem 方法存在')

# 测试 _compute_style_weighted_sem 逻辑
import torch, torch.nn as nn

class MockServer:
    def __init__(self, tau=0.3):
        self.style_aware_tau = tau
        # 3 个 client，每个有 style 和 semantic state
        # client 0 style 和 client 2 最近；和 client 1 远
        self.style_bank = {
            0: (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])),
            1: (torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 1.0])),
            2: (torch.tensor([0.9, 0.1]), torch.tensor([1.0, 1.0])),
        }
        # semantic state 差别明显
        self.client_sem_states = {
            0: {'semantic_head.weight': torch.tensor([[1.0, 0.0]])},
            1: {'semantic_head.weight': torch.tensor([[0.0, 1.0]])},
            2: {'semantic_head.weight': torch.tensor([[0.9, 0.1]])},
        }

MockServer._compute_style_weighted_sem = mod.Server._compute_style_weighted_sem

srv = MockServer(tau=0.3)
# target = client 0：应该给 client 2 高权重（相似），client 1 低权重
result = srv._compute_style_weighted_sem(0)
assert result is not None, 'weighted sem shouldnt be None'
weight_tensor = result['semantic_head.weight']
# 期望接近 client 0 和 client 2 的混合（[1.0, 0.0] 和 [0.9, 0.1]）
# 而不是 client 1 的 [0.0, 1.0]
# dim 0 应该 > dim 1
assert weight_tensor[0, 0] > weight_tensor[0, 1], f'target=0 result should prefer client 0/2 (sim=1.0,0.995), got {weight_tensor}'
print(f'[PASS] target=0 → weighted sem = {weight_tensor.tolist()} (偏向 client 0/2)')

# target = client 1
result = srv._compute_style_weighted_sem(1)
weight_tensor = result['semantic_head.weight']
# client 1 应该更像自己
assert weight_tensor[0, 1] > weight_tensor[0, 0], f'target=1 should prefer client 1, got {weight_tensor}'
print(f'[PASS] target=1 → weighted sem = {weight_tensor.tolist()} (偏向 client 1)')

# 测试 tau 影响：tau 低 → 更 peaky（更像 self）；tau 高 → 更平均
srv_low = MockServer(tau=0.05)
r_low = srv_low._compute_style_weighted_sem(0)['semantic_head.weight']
srv_high = MockServer(tau=5.0)
r_high = srv_high._compute_style_weighted_sem(0)['semantic_head.weight']
# tau=0.05 → 结果几乎等于 client 0 (target) 或 client 2 (最近邻)
# tau=5.0 → 结果接近 3 client 平均 ≈ [0.63, 0.37]
low_diff = (r_low[0] - torch.tensor([1.0, 0.0])).abs().sum().item()
high_diff = (r_high[0] - torch.tensor([0.63, 0.37])).abs().sum().item()
assert low_diff < high_diff, f'low tau should be more peaky, got low_diff={low_diff} high_diff={high_diff}'
print(f'[PASS] tau 影响正确: low_tau diff={low_diff:.3f} < high_tau diff={high_diff:.3f}')

# 测试 sas=0 时 pack 行为不变（已通过语法检查 + flag 分支控制）
# 测试 style_bank 为空时不崩
srv_empty = MockServer()
srv_empty.style_bank = {}
srv_empty.client_sem_states = {}
r = srv_empty._compute_style_weighted_sem(0)
assert r is None
print('[PASS] style_bank 空时返回 None（优雅降级）')

print('\n=== 方案 A 所有测试通过 ===')
