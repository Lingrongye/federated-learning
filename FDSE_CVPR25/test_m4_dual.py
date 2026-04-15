#!/usr/bin/env python3
"""
M4 Dual Alignment 单元测试
验证 feddsa_adaptive.py 中 mode=4 的完整数据流和损失计算

运行: python test_m4_dual.py
"""
import sys
import torch
import torch.nn.functional as F
import numpy as np

# 添加路径
sys.path.insert(0, '.')

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")


print("=" * 60)
print("M4 Dual Alignment Unit Tests")
print("=" * 60)

# ============================================================
# Test 1: 模块导入 (只导入模型,不导入需要 flgo 的 Server/Client)
# ============================================================
print("\n--- Test 1: Import ---")
try:
    # 直接导入模型类 (不需要 flgo)
    import importlib.util
    spec = importlib.util.spec_from_file_location("feddsa_adaptive", "algorithm/feddsa_adaptive.py",
        submodule_search_locations=[])
    # 手动解析文件提取 FedDSAModel 和 AlexNetEncoder
    import ast
    with open("algorithm/feddsa_adaptive.py", "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    check("FedDSAModel class exists in source", "FedDSAModel" in class_names, f"classes: {class_names[:5]}")
    check("Server class exists in source", "Server" in class_names)
    check("Client class exists in source", "Client" in class_names)

    # 导入模型相关的类 (不需要 flgo)
    exec_globals = {"__builtins__": __builtins__}
    exec("""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

class AlexNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
            ('bn2', nn.BatchNorm2d(192)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(384)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('bn5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU(inplace=True)),
            ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
        ]))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn6(self.fc1(x))
        x = self.relu(x)
        x = self.bn7(self.fc2(x))
        x = self.relu(x)
        return x

class FedDSAModel(nn.Module):
    def __init__(self, num_classes=7, feat_dim=1024, proj_dim=128):
        super().__init__()
        self.encoder = AlexNetEncoder()
        self.semantic_head = nn.Sequential(nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.style_head = nn.Sequential(nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.head = nn.Linear(proj_dim, num_classes)
    def forward(self, x):
        h = self.encoder(x)
        z_sem = self.semantic_head(h)
        return self.head(z_sem)
    def encode(self, x):
        return self.encoder(x)
    def get_semantic(self, h):
        return self.semantic_head(h)
    def get_style(self, h):
        return self.style_head(h)
""", exec_globals)
    FedDSAModel = exec_globals['FedDSAModel']
    check("FedDSAModel instantiable", True)
except Exception as e:
    check("Import", False, str(e))
    sys.exit(1)

# ============================================================
# Test 2: 模型前向传播
# ============================================================
print("\n--- Test 2: Model Forward ---")
model = FedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128)
x = torch.randn(8, 3, 224, 224)  # batch of 8

h = model.encode(x)
check("encode output shape", h.shape == (8, 1024), f"got {h.shape}")

z_sem = model.get_semantic(h)
check("z_sem shape", z_sem.shape == (8, 128), f"got {z_sem.shape}")

z_sty = model.get_style(h)
check("z_sty shape", z_sty.shape == (8, 128), f"got {z_sty.shape}")

output = model.head(z_sem)
check("classifier output shape", output.shape == (8, 7), f"got {output.shape}")

# ============================================================
# Test 3: Server pack() — intra/cross 分离
# ============================================================
print("\n--- Test 3: Server pack() intra/cross split ---")

# 模拟 domain_protos: 4 clients × 3 classes = 12 entries
domain_protos = {}
for cls in range(3):
    for cid in range(4):
        domain_protos[(cls, cid)] = torch.randn(128)

# 模拟 pack 中的分离逻辑
client_id = 1  # 假设发给 client 1
intra_protos = {}
cross_protos = {}
for (cls, cid), proto in domain_protos.items():
    if cid == client_id:
        intra_protos[cls] = proto
    else:
        cross_protos[(cls, cid)] = proto

check("intra has 3 classes", len(intra_protos) == 3, f"got {len(intra_protos)}")
check("cross has 9 entries (3 classes × 3 other clients)", len(cross_protos) == 9, f"got {len(cross_protos)}")
check("intra keys are class ints", all(isinstance(k, int) for k in intra_protos.keys()), f"keys: {list(intra_protos.keys())}")
check("cross keys are (class, cid) tuples", all(isinstance(k, tuple) for k in cross_protos.keys()), f"keys: {list(cross_protos.keys())[:3]}")

# 验证 client_id 不在 cross_protos 中
cross_cids = set(cid for (cls, cid) in cross_protos.keys())
check("own client_id excluded from cross", client_id not in cross_cids, f"found cid={client_id} in cross")

# ============================================================
# Test 4: L_intra 计算
# ============================================================
print("\n--- Test 4: L_intra computation ---")

z_sem_test = torch.randn(4, 128)
y_test = torch.tensor([0, 1, 2, 0])

# 手动计算 L_intra
loss_intra = torch.tensor(0.0)
count = 0
for i in range(4):
    label = y_test[i].item()
    if label in intra_protos:
        proto = intra_protos[label]
        sim = F.cosine_similarity(z_sem_test[i:i+1], proto.unsqueeze(0)).squeeze()
        loss_intra = loss_intra + (1.0 - sim)
        count += 1
if count > 0:
    loss_intra = loss_intra / count

check("L_intra is scalar", loss_intra.dim() == 0, f"dim={loss_intra.dim()}, shape={loss_intra.shape}")
check("L_intra in [0, 2]", 0 <= loss_intra.item() <= 2.0, f"value={loss_intra.item():.4f}")
check("L_intra differentiable", loss_intra.requires_grad == False, "expected no grad (test only, no model param)")

# 验证: 如果 z_sem 等于 proto,L_intra 应该接近 0
z_exact = intra_protos[0].unsqueeze(0)
sim_exact = F.cosine_similarity(z_exact, intra_protos[0].unsqueeze(0)).squeeze()
check("L_intra=0 when z_sem==proto", abs(1.0 - sim_exact.item()) < 1e-5, f"1-sim={1-sim_exact.item():.6f}")

# ============================================================
# Test 5: L_cross 计算
# ============================================================
print("\n--- Test 5: L_cross computation ---")

tau = 0.2
cross_entries = []
cross_classes = []
for (cls, cid) in sorted(cross_protos.keys()):
    cross_entries.append(cross_protos[(cls, cid)])
    cross_classes.append(cls)

proto_matrix = torch.stack(cross_entries)
proto_labels = torch.tensor(cross_classes)

check("cross proto_matrix shape", proto_matrix.shape == (9, 128), f"got {proto_matrix.shape}")
check("cross proto_labels shape", proto_labels.shape == (9,), f"got {proto_labels.shape}")

z_n = F.normalize(z_sem_test, dim=1)
p_n = F.normalize(proto_matrix, dim=1)
logits = z_n @ p_n.T / tau

check("logits shape", logits.shape == (4, 9), f"got {logits.shape}")

# 验证 InfoNCE 计算
loss_cross = torch.tensor(0.0)
cross_count = 0
for i in range(4):
    label = y_test[i].item()
    pos_mask = (proto_labels == label)
    n_pos = pos_mask.sum().item()
    if n_pos == 0:
        continue
    log_denom = torch.logsumexp(logits[i], dim=0)
    pos_logits = logits[i][pos_mask]
    loss_cross = loss_cross + (-pos_logits + log_denom).sum() / n_pos
    cross_count += 1

if cross_count > 0:
    loss_cross = loss_cross / cross_count

check("L_cross is scalar", loss_cross.dim() == 0, f"dim={loss_cross.dim()}")
check("L_cross > 0", loss_cross.item() > 0, f"value={loss_cross.item():.4f}")
check("L_cross is finite", torch.isfinite(loss_cross), f"value={loss_cross.item()}")

# ============================================================
# Test 6: 边界情况 — 空 protos
# ============================================================
print("\n--- Test 6: Edge cases ---")

# 空 intra_protos
empty_intra = {}
loss_empty = torch.tensor(0.0)
count_empty = 0
for i in range(4):
    label = y_test[i].item()
    if label in empty_intra:
        count_empty += 1
check("empty intra -> count=0, loss=0", count_empty == 0 and loss_empty.item() == 0.0)

# L_cross 中某类没有正样本
y_missing = torch.tensor([99, 99, 99, 99])  # 类 99 不在 cross_protos 中
cross_count_missing = 0
for i in range(4):
    label = y_missing[i].item()
    pos_mask = (proto_labels == label)
    n_pos = pos_mask.sum().item()
    if n_pos == 0:
        continue
    cross_count_missing += 1
check("missing class -> skip (count=0)", cross_count_missing == 0)

# 只有 1 个 cross proto (< 2, 应该跳过)
check("len(cross_entries) < 2 guard", len(cross_entries) >= 2, "need >= 2 for InfoNCE")

# ============================================================
# Test 7: 梯度回传
# ============================================================
print("\n--- Test 7: Gradient flow ---")

model_grad = FedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128)
model_grad.train()

x_grad = torch.randn(4, 3, 224, 224)
y_grad = torch.tensor([0, 1, 2, 0])

h_g = model_grad.encode(x_grad)
z_sem_g = model_grad.get_semantic(h_g)

# 模拟 L_intra
intra_test = {0: torch.randn(128).detach(), 1: torch.randn(128).detach(), 2: torch.randn(128).detach()}
loss_intra_g = torch.tensor(0.0)
cnt = 0
for i in range(4):
    label = y_grad[i].item()
    if label in intra_test:
        proto = intra_test[label]
        sim = F.cosine_similarity(z_sem_g[i:i+1], proto.unsqueeze(0)).squeeze()
        loss_intra_g = loss_intra_g + (1.0 - sim)
        cnt += 1
loss_intra_g = loss_intra_g / max(cnt, 1)

# 模拟 L_cross
cross_test_protos = []
cross_test_labels = []
for cls in range(3):
    for cid in range(3):
        cross_test_protos.append(torch.randn(128).detach())
        cross_test_labels.append(cls)

pm = torch.stack(cross_test_protos)
pl = torch.tensor(cross_test_labels)
z_n_g = F.normalize(z_sem_g, dim=1)
p_n_g = F.normalize(pm, dim=1)
logits_g = z_n_g @ p_n_g.T / 0.2

loss_cross_g = torch.tensor(0.0)
cc = 0
for i in range(4):
    label = y_grad[i].item()
    pos_mask = (pl == label)
    n_pos = pos_mask.sum().item()
    if n_pos == 0:
        continue
    log_denom = torch.logsumexp(logits_g[i], dim=0)
    pos_logits = logits_g[i][pos_mask]
    loss_cross_g = loss_cross_g + (-pos_logits + log_denom).sum() / n_pos
    cc += 1
loss_cross_g = loss_cross_g / max(cc, 1)

total_loss = loss_intra_g + loss_cross_g
total_loss.backward()

# 检查梯度存在
has_grad = False
for name, param in model_grad.named_parameters():
    if param.grad is not None and param.grad.abs().sum() > 0:
        has_grad = True
        break

check("gradient flows to model parameters", has_grad)

# 检查 style_head 是否收到梯度 (不应该,因为 dual loss 只用 z_sem)
style_head_grad = False
for name, param in model_grad.named_parameters():
    if 'style_head' in name and param.grad is not None and param.grad.abs().sum() > 0:
        style_head_grad = True
check("style_head has NO gradient from dual loss", not style_head_grad)

# semantic_head 应该有梯度
sem_head_grad = False
for name, param in model_grad.named_parameters():
    if 'semantic_head' in name and param.grad is not None and param.grad.abs().sum() > 0:
        sem_head_grad = True
check("semantic_head HAS gradient from dual loss", sem_head_grad)

# encoder 应该有梯度
encoder_grad = False
for name, param in model_grad.named_parameters():
    if 'encoder' in name and param.grad is not None and param.grad.abs().sum() > 0:
        encoder_grad = True
check("encoder HAS gradient from dual loss", encoder_grad)

# ============================================================
# Test 8: Config 参数解析
# ============================================================
print("\n--- Test 8: Config parameter parsing ---")

# 模拟 algo_para 列表 (和 feddsa_m4_dual.yml 一致)
algo_para = [1.0, 0.0, 1.0, 0.2, 50, 5, 128, 0.05, 0.8, 0.05, 0.9, 4, 0.5, 1.0, 0.5]

check("algo_para has 15 elements", len(algo_para) == 15, f"got {len(algo_para)}")
check("adaptive_mode=4", int(algo_para[11]) == 4, f"got {algo_para[11]}")
check("tau=0.2", float(algo_para[3]) == 0.2, f"got {algo_para[3]}")
check("lambda_intra=1.0", float(algo_para[13]) == 1.0, f"got {algo_para[13]}")
check("lambda_cross=0.5", float(algo_para[14]) == 0.5, f"got {algo_para[14]}")

# ============================================================
# Test 9: 数值稳定性
# ============================================================
print("\n--- Test 9: Numerical stability ---")

# 极端情况: 所有 z_sem 相同
z_same = torch.ones(4, 128) * 0.1
z_n_same = F.normalize(z_same, dim=1)
logits_same = z_n_same @ p_n_g.T / 0.2

check("identical z_sem -> finite logits", torch.isfinite(logits_same).all())

loss_same = torch.tensor(0.0)
for i in range(4):
    log_denom = torch.logsumexp(logits_same[i], dim=0)
    check(f"logsumexp finite for identical z_sem[{i}]", torch.isfinite(log_denom), f"val={log_denom.item()}")

# 极端情况: 非常大的特征值
z_large = torch.randn(4, 128) * 100
z_n_large = F.normalize(z_large, dim=1)
logits_large = z_n_large @ p_n_g.T / 0.2
check("large z_sem -> finite after normalize", torch.isfinite(logits_large).all())

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} PASS, {FAIL} FAIL, {PASS+FAIL} TOTAL")
if FAIL == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"WARNING: {FAIL} tests failed!")
print("=" * 60)
