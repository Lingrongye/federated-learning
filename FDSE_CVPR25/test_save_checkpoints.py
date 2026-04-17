"""快速测试：save_errors flag 的语法 + 逻辑正确性"""
import os, sys, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1. 语法 check
import ast
with open(os.path.join(os.path.dirname(__file__), 'algorithm/feddsa_scheduled.py'), encoding='utf-8') as f:
    ast.parse(f.read())
print('[PASS] Syntax OK')

# 2. 导入 check — 不触发 flgo 依赖
import importlib.util
spec = importlib.util.spec_from_file_location(
    'feddsa_scheduled',
    os.path.join(os.path.dirname(__file__), 'algorithm/feddsa_scheduled.py')
)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print('[PASS] Import OK')
except Exception as e:
    print(f'[FAIL] Import failed: {e}')
    sys.exit(1)

# 3. 检查 Server 类有 _save_checkpoints 方法
assert hasattr(mod.Server, '_save_checkpoints'), '_save_checkpoints method missing'
print('[PASS] _save_checkpoints method exists')

# 4. 检查 init_algo_para 包含 'se'
import inspect
src = inspect.getsource(mod.Server.initialize)
assert "'se': 0" in src, "save_errors 'se' flag missing from init_algo_para"
print('[PASS] se flag in init_algo_para')

# 5. 检查 iterate 方法有 save_errors 分支
src = inspect.getsource(mod.Server.iterate)
assert 'save_errors' in src, 'save_errors check missing from iterate()'
assert '_save_checkpoints' in src, '_save_checkpoints call missing from iterate()'
print('[PASS] iterate calls _save_checkpoints on last round')

# 6. 模拟调用 _save_checkpoints（mock server）
import torch, torch.nn as nn

class MockLogger:
    def info(self, msg): print(f'  [mock log] {msg}')

class MockClient:
    def __init__(self):
        self.model = nn.Linear(10, 2)

class MockGV:
    logger = MockLogger()

class MockServer:
    def __init__(self):
        self.model = nn.Linear(10, 2)
        self.clients = [MockClient() for _ in range(3)]
        self.current_round = 200
        self.num_rounds = 200
        self.schedule_mode = 0
        self.lambda_orth = 1.0
        self.tau = 0.2
        self.option = {'seed': 42}
        self.gv = MockGV()

# Bind _save_checkpoints to MockServer
MockServer._save_checkpoints = mod.Server._save_checkpoints

# 在 tmp 目录里跑（重定向 ~ 到 tmp）
original_expanduser = os.path.expanduser
tmp_home = tempfile.mkdtemp()
def mock_expanduser(path):
    if path.startswith('~'):
        return os.path.join(tmp_home, path[2:])
    return original_expanduser(path)
os.path.expanduser = mock_expanduser

try:
    srv = MockServer()
    srv._save_checkpoints()
    # 检查 checkpoint 文件
    ckpt_base = os.path.join(tmp_home, 'fl_checkpoints')
    assert os.path.exists(ckpt_base), f'checkpoint dir not created'
    subdirs = os.listdir(ckpt_base)
    assert len(subdirs) == 1, f'expected 1 subdir, got {subdirs}'
    saved = os.listdir(os.path.join(ckpt_base, subdirs[0]))
    expected = {'global_model.pt', 'client_0.pt', 'client_1.pt', 'client_2.pt', 'meta.json'}
    assert expected.issubset(set(saved)), f'missing files: {expected - set(saved)}'
    print(f'[PASS] Checkpoint save creates {saved}')
finally:
    os.path.expanduser = original_expanduser
    shutil.rmtree(tmp_home, ignore_errors=True)

print('\n=== ALL TESTS PASSED ===')
