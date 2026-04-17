"""测试 save_errors：追踪 best + 最终保存"""
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

# 检查 Server 类新方法
assert hasattr(mod.Server, '_track_best'), '_track_best 缺失'
assert hasattr(mod.Server, '_save_best_checkpoints'), '_save_best_checkpoints 缺失'
print('[PASS] Server 有 _track_best + _save_best_checkpoints')

# 检查 init_algo_para 的 'se'
import inspect
src = inspect.getsource(mod.Server.initialize)
assert "'se': 0" in src, "'se' flag 缺失"
assert 'self._best_avg_acc' in src, '_best_avg_acc 初始化缺失'
print('[PASS] se flag + best tracking state 初始化正确')

# 检查 iterate 有追踪 + 最后一轮保存逻辑
src = inspect.getsource(mod.Server.iterate)
assert '_track_best' in src
assert '_save_best_checkpoints' in src
print('[PASS] iterate 调用 _track_best 和 _save_best_checkpoints')

# 模拟 best 追踪
import torch, torch.nn as nn, copy

class MockLogger:
    output = {'mean_local_test_accuracy': []}
    def info(self, msg): print(f'  [log] {msg}')
    def warning(self, msg): print(f'  [warn] {msg}')

class MockClient:
    def __init__(self):
        self.model = nn.Linear(10, 2)

class MockGV:
    logger = MockLogger()

class MockServer:
    def __init__(self):
        self.model = nn.Linear(10, 2)
        self.clients = [MockClient() for _ in range(3)]
        self.current_round = 0
        self.num_rounds = 5
        self.schedule_mode = 0
        self.lambda_orth = 1.0
        self.tau = 0.2
        self.option = {'seed': 42}
        self.gv = MockGV()
        self._best_avg_acc = -1.0
        self._best_round = 0
        self._best_global_state = None
        self._best_client_states = None

MockServer._track_best = mod.Server._track_best
MockServer._save_best_checkpoints = mod.Server._save_best_checkpoints

# 模拟 5 轮：test acc = [0.7, 0.75, 0.80, 0.78, 0.72]，best 应该是 R3 (0.80)
srv = MockServer()
acc_history = [0.70, 0.75, 0.80, 0.78, 0.72]
for r, acc in enumerate(acc_history, start=1):
    srv.current_round = r
    # 模拟 flgo 在 iterate 前把 test 结果写进 logger.output
    srv.gv.logger.output['mean_local_test_accuracy'].append(acc)
    # 每轮改变 model 权重 (模拟训练)
    with torch.no_grad():
        for p in srv.model.parameters():
            p.add_(0.01 * r)
    srv._track_best()

assert abs(srv._best_avg_acc - 80.0) < 0.01, f'best acc 错误: {srv._best_avg_acc}'
assert srv._best_round == 3, f'best round 错误: {srv._best_round}'
print(f'[PASS] best tracked correctly: R{srv._best_round} avg={srv._best_avg_acc:.2f}')

# 保存到 tmp 目录
original_expanduser = os.path.expanduser
tmp_home = tempfile.mkdtemp()
def mock_expanduser(path):
    if path.startswith('~'):
        return os.path.join(tmp_home, path[2:])
    return original_expanduser(path)
os.path.expanduser = mock_expanduser

try:
    srv._save_best_checkpoints()
    ckpt_base = os.path.join(tmp_home, 'fl_checkpoints')
    assert os.path.exists(ckpt_base)
    subdirs = os.listdir(ckpt_base)
    assert len(subdirs) == 1
    saved = os.listdir(os.path.join(ckpt_base, subdirs[0]))
    expected = {'global_model.pt', 'client_0.pt', 'client_1.pt', 'client_2.pt', 'meta.json'}
    assert expected.issubset(set(saved)), f'缺少: {expected - set(saved)}'
    # 检查 meta
    import json
    with open(os.path.join(ckpt_base, subdirs[0], 'meta.json')) as f:
        meta = json.load(f)
    assert meta['best_round'] == 3
    assert abs(meta['best_avg_acc'] - 80.0) < 0.01
    assert meta['is_last_round_fallback'] == False
    print(f'[PASS] Checkpoint 保存: {subdirs[0]} (best={meta["best_round"]} acc={meta["best_avg_acc"]:.2f})')
finally:
    os.path.expanduser = original_expanduser
    shutil.rmtree(tmp_home, ignore_errors=True)

# 额外测试：fallback (test 从未跑过) 的行为
srv2 = MockServer()
tmp2 = tempfile.mkdtemp()
os.path.expanduser = lambda p: os.path.join(tmp2, p[2:]) if p.startswith('~') else p
try:
    # 不调用 _track_best，直接 save — 应该走 fallback
    srv2.current_round = 5
    srv2._save_best_checkpoints()
    ckpt = os.listdir(os.path.join(tmp2, 'fl_checkpoints'))[0]
    import json
    with open(os.path.join(tmp2, 'fl_checkpoints', ckpt, 'meta.json')) as f:
        meta = json.load(f)
    assert meta['is_last_round_fallback'] == True
    print(f'[PASS] Fallback 正确（test 未跑时存 last round）')
finally:
    os.path.expanduser = original_expanduser
    shutil.rmtree(tmp2, ignore_errors=True)

print('\n=== ALL TESTS PASSED ===')
