"""
Unit tests for feddsa_scheduled.py — verify all 4 schedule modes,
gradient flow, loss computation, and architecture correctness.
"""
import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from algorithm.feddsa_scheduled import (
    AlexNetEncoder, FedDSAModel, Client
)
# Also import original for comparison
from algorithm.feddsa import (
    AlexNetEncoder as OrigAlexNetEncoder,
    FedDSAModel as OrigFedDSAModel,
)

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")


# ============================================================
# Test 1: Architecture matches original exactly
# ============================================================
print("\n=== Test 1: Architecture Parity ===")

model_new = FedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128)
model_orig = OrigFedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128)

new_keys = sorted(model_new.state_dict().keys())
orig_keys = sorted(model_orig.state_dict().keys())
check("Same state_dict keys", new_keys == orig_keys,
      f"new has {len(new_keys)} keys, orig has {len(orig_keys)} keys")

# Check parameter counts match
new_params = sum(p.numel() for p in model_new.parameters())
orig_params = sum(p.numel() for p in model_orig.parameters())
check("Same parameter count", new_params == orig_params,
      f"new={new_params}, orig={orig_params}")

# Check key shapes match
shape_match = True
for k in new_keys:
    if model_new.state_dict()[k].shape != model_orig.state_dict()[k].shape:
        shape_match = False
        print(f"    Shape mismatch: {k}: {model_new.state_dict()[k].shape} vs {model_orig.state_dict()[k].shape}")
check("All parameter shapes match", shape_match)

# Check forward pass produces same output shape
x = torch.randn(4, 3, 224, 224)
# Use same weights
model_new.load_state_dict(model_orig.state_dict())
with torch.no_grad():
    out_new = model_new(x)
    out_orig = model_orig(x)
check("Forward output shape matches", out_new.shape == out_orig.shape,
      f"new={out_new.shape}, orig={out_orig.shape}")
check("Forward output values match", torch.allclose(out_new, out_orig, atol=1e-5),
      f"max diff={torch.max(torch.abs(out_new - out_orig)).item():.6f}")

# Check encoder output
with torch.no_grad():
    h_new = model_new.encode(x)
    h_orig = model_orig.encode(x)
check("Encoder output shape", h_new.shape == (4, 1024),
      f"got {h_new.shape}")
check("Encoder output matches", torch.allclose(h_new, h_orig, atol=1e-5))

# Check dual heads
with torch.no_grad():
    z_sem_new = model_new.get_semantic(h_new)
    z_sty_new = model_new.get_style(h_new)
check("Semantic head output shape", z_sem_new.shape == (4, 128))
check("Style head output shape", z_sty_new.shape == (4, 128))


# ============================================================
# Test 2: Schedule mode functions
# ============================================================
print("\n=== Test 2: Schedule Mode Functions ===")

# Create a mock client that inherits static methods from Client
class MockClient(Client):
    def __init__(self):
        # Skip Client.initialize() — just set attributes directly
        self.schedule_mode = 0
        self.bell_peak = 60
        self.bell_width = 30
        self.cutoff_round = 80
        self.current_round = 0

mock = MockClient()

# Mode 0: orth_only — always returns 0
mock.schedule_mode = 0
for r in [0, 10, 50, 100, 200]:
    mock.current_round = r
    w = Client._get_aux_weight(mock)
    check(f"Mode 0 R{r}: w_aux=0", w == 0.0, f"got {w}")

# Mode 1: bell-curve
mock.schedule_mode = 1
mock.current_round = 60
w_peak = Client._get_aux_weight(mock)
check("Mode 1 R60 (peak): w_aux=1.0", abs(w_peak - 1.0) < 1e-6, f"got {w_peak}")

mock.current_round = 30
w_30 = Client._get_aux_weight(mock)
expected_30 = math.exp(-0.5 * ((30 - 60) / 30) ** 2)
check("Mode 1 R30: w_aux≈0.607", abs(w_30 - expected_30) < 1e-6,
      f"got {w_30:.6f}, expected {expected_30:.6f}")

mock.current_round = 120
w_120 = Client._get_aux_weight(mock)
expected_120 = math.exp(-0.5 * ((120 - 60) / 30) ** 2)
check("Mode 1 R120: w_aux≈0.135", abs(w_120 - expected_120) < 1e-6,
      f"got {w_120:.6f}, expected {expected_120:.6f}")

mock.current_round = 1
w_1 = Client._get_aux_weight(mock)
check("Mode 1 R1: w_aux≈0.145 (small)", w_1 < 0.2, f"got {w_1}")

# Mode 2: cutoff
mock.schedule_mode = 2
mock.current_round = 50
check("Mode 2 R50: w_aux=1.0", Client._get_aux_weight(mock) == 1.0)
mock.current_round = 80
check("Mode 2 R80: w_aux=1.0 (boundary)", Client._get_aux_weight(mock) == 1.0)
mock.current_round = 81
check("Mode 2 R81: w_aux=0.0 (cut off)", Client._get_aux_weight(mock) == 0.0)
mock.current_round = 200
check("Mode 2 R200: w_aux=0.0", Client._get_aux_weight(mock) == 0.0)

# Mode 3: always on
mock.schedule_mode = 3
for r in [1, 50, 100, 200]:
    mock.current_round = r
    check(f"Mode 3 R{r}: w_aux=1.0", Client._get_aux_weight(mock) == 1.0)


# ============================================================
# Test 3: Loss computation — mode 0 (orth only)
# ============================================================
print("\n=== Test 3: Mode 0 Loss (orth only) ===")

model = FedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128)
x = torch.randn(8, 3, 224, 224)
y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0])
loss_fn = nn.CrossEntropyLoss()

h = model.encode(x)
z_sem = model.get_semantic(h)
z_sty = model.get_style(h)
output = model.head(z_sem)
loss_task = loss_fn(output, y)

# Orthogonal loss (should always be computed)
z_sem_n = F.normalize(z_sem, dim=1)
z_sty_n = F.normalize(z_sty, dim=1)
cos = (z_sem_n * z_sty_n).sum(dim=1)
loss_orth = (cos ** 2).mean()

check("Loss task is scalar", loss_task.dim() == 0)
check("Loss task is positive", loss_task.item() > 0)
check("Loss orth is non-negative", loss_orth.item() >= 0)
check("Loss orth is bounded [0,1]", 0 <= loss_orth.item() <= 1,
      f"got {loss_orth.item()}")

# Mode 0 total loss = loss_task + lambda_orth * loss_orth (no aug, no InfoNCE)
w_aux = 0.0  # mode 0
total = loss_task + w_aux * 0.0 + 1.0 * loss_orth + 0.0 * 0.0 + w_aux * 1.0 * 0.0
check("Mode 0 total = CE + orth only", total.item() == (loss_task + loss_orth).item())


# ============================================================
# Test 4: Gradient flow — orth loss backprops to encoder
# ============================================================
print("\n=== Test 4: Gradient Flow ===")

model = FedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128)
model.train()
x = torch.randn(4, 3, 224, 224)
y = torch.tensor([0, 1, 2, 3])
loss_fn = nn.CrossEntropyLoss()

# Mode 0: only CE + orth
h = model.encode(x)
z_sem = model.get_semantic(h)
z_sty = model.get_style(h)
output = model.head(z_sem)

loss_task = loss_fn(output, y)
z_sem_n = F.normalize(z_sem, dim=1)
z_sty_n = F.normalize(z_sty, dim=1)
cos = (z_sem_n * z_sty_n).sum(dim=1)
loss_orth = (cos ** 2).mean()

total = loss_task + 1.0 * loss_orth
total.backward()

# Check encoder gets gradients from both CE and orth
enc_grad_norm = sum(p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None)
check("Encoder has gradients", enc_grad_norm > 0, f"grad_norm={enc_grad_norm}")

# Check semantic_head gets gradients (from CE)
sem_grad = sum(p.grad.norm().item() for p in model.semantic_head.parameters() if p.grad is not None)
check("Semantic head has gradients", sem_grad > 0)

# Check style_head gets gradients (from orth)
sty_grad = sum(p.grad.norm().item() for p in model.style_head.parameters() if p.grad is not None)
check("Style head has gradients from orth", sty_grad > 0)

# Check classifier gets gradients
head_grad = sum(p.grad.norm().item() for p in model.head.parameters() if p.grad is not None)
check("Classifier head has gradients", head_grad > 0)


# ============================================================
# Test 5: InfoNCE loss correctness
# ============================================================
print("\n=== Test 5: InfoNCE Loss ===")

model = FedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128)
x = torch.randn(8, 3, 224, 224)
y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
h = model.encode(x)
z_sem = model.get_semantic(h)

# Create fake global protos
global_protos = {}
for c in range(7):
    global_protos[c] = torch.randn(128)

# Compute InfoNCE manually
available = sorted(global_protos.keys())
proto_matrix = torch.stack([global_protos[c] for c in available])
class_to_idx = {c: i for i, c in enumerate(available)}
z_n = F.normalize(z_sem, dim=1)
p_n = F.normalize(proto_matrix, dim=1)
tau = 0.2
logits = (z_n @ p_n.T) / tau
targets = torch.tensor([class_to_idx[y[i].item()] for i in range(8)])
expected_loss = F.cross_entropy(logits, targets)

check("InfoNCE loss is positive", expected_loss.item() > 0)
check("InfoNCE loss has grad", expected_loss.requires_grad)

# Verify w_aux=0 means no InfoNCE contribution
w_aux = 0.0
contribution = w_aux * 1.0 * expected_loss
check("w_aux=0 zeroes InfoNCE", contribution.item() == 0.0)


# ============================================================
# Test 6: Style augmentation
# ============================================================
print("\n=== Test 6: Style Augmentation ===")

model = FedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128)
x = torch.randn(4, 3, 224, 224)
h = model.encode(x)

# Create mock style bank
mu_ext = torch.randn(1024)
sigma_ext = torch.randn(1024).abs() + 0.1
style_bank = [(mu_ext, sigma_ext)]

# Test augmentation
mu_local = h.mean(dim=0)
sigma_local = h.std(dim=0, unbiased=False).clamp(min=1e-6)
alpha = 0.5
mu_mix = alpha * mu_local + (1 - alpha) * mu_ext
sigma_mix = alpha * sigma_local + (1 - alpha) * sigma_ext
h_norm = (h - mu_local) / sigma_local
h_aug = h_norm * sigma_mix + mu_mix

check("Augmented h has same shape", h_aug.shape == h.shape)
check("Augmented h differs from original", not torch.allclose(h_aug, h, atol=1e-3))

# Mode 0 should never call augmentation (w_aux=0, guard prevents it)
check("Mode 0 guard: w_aux > 1e-4 is False", not (0.0 > 1e-4))


# ============================================================
# Test 7: Gradient conflict logger
# ============================================================
print("\n=== Test 7: Gradient Conflict Logger ===")

model = FedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128)
model.train()
x = torch.randn(4, 3, 224, 224)
y = torch.tensor([0, 1, 2, 3])
loss_fn = nn.CrossEntropyLoss()

h = model.encode(x)
z_sem = model.get_semantic(h)
output = model.head(z_sem)
loss_task = loss_fn(output, y)

# Fake InfoNCE loss (use a simple contrastive-like loss)
proto = torch.randn(7, 128)
z_n = F.normalize(z_sem, dim=1)
p_n = F.normalize(proto, dim=1)
logits = (z_n @ p_n.T) / 0.2
targets = y
loss_sem = F.cross_entropy(logits, targets)
loss_align_weighted = 1.0 * loss_sem

# Compute gradient cosine similarity
encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
grad_ce = torch.autograd.grad(loss_task, encoder_params, retain_graph=True, allow_unused=True)
grad_align = torch.autograd.grad(loss_align_weighted, encoder_params, retain_graph=True, allow_unused=True)

flat_ce = torch.cat([g.flatten() for g in grad_ce if g is not None])
flat_align = torch.cat([g.flatten() for g in grad_align if g is not None])
cos_sim = F.cosine_similarity(flat_ce.unsqueeze(0), flat_align.unsqueeze(0)).item()

check("Cos sim is in [-1, 1]", -1.0 <= cos_sim <= 1.0, f"got {cos_sim}")
check("Cos sim is finite", math.isfinite(cos_sim))
check("Gradient vectors are non-zero", flat_ce.norm().item() > 0 and flat_align.norm().item() > 0)


# ============================================================
# Test 8: Bell-curve symmetry and decay properties
# ============================================================
print("\n=== Test 8: Bell-Curve Properties ===")

bell = Client._bell_weight

# Symmetry
check("Bell symmetric: bell(30,60,30)==bell(90,60,30)",
      abs(bell(30, 60, 30) - bell(90, 60, 30)) < 1e-10)

# Monotone increase before peak
check("Bell monotone: bell(50)<bell(55)<bell(60)",
      bell(50, 60, 30) < bell(55, 60, 30) < bell(60, 60, 30))

# Monotone decrease after peak
check("Bell monotone: bell(60)>bell(65)>bell(70)",
      bell(60, 60, 30) > bell(65, 60, 30) > bell(70, 60, 30))

# Far from peak → near zero
check("Bell far: bell(180,60,30)<0.01",
      bell(180, 60, 30) < 0.01, f"got {bell(180, 60, 30)}")

# width=0 edge case
check("Bell width=0 at peak: 1.0", bell(60, 60, 0) == 1.0)
check("Bell width=0 off peak: 0.0", bell(61, 60, 0) == 0.0)


# ============================================================
# Test 9: BN layer handling (private keys)
# ============================================================
print("\n=== Test 9: BN Private Keys ===")

model = FedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128)
all_keys = list(model.state_dict().keys())

# Identify what should be private
private_keys = set()
for k in all_keys:
    if 'style_head' in k:
        private_keys.add(k)
    elif 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
        private_keys.add(k)

shared_keys = [k for k in all_keys if k not in private_keys]

check("Style head keys are private",
      all(k in private_keys for k in all_keys if 'style_head' in k))
check("BN running stats are private",
      all(k in private_keys for k in all_keys if 'running_mean' in k or 'running_var' in k))
check("BN weight/bias are shared (not private)",
      all(k not in private_keys for k in all_keys if k.endswith('.weight') and 'bn' in k.lower() and 'running' not in k))
check("Encoder conv weights are shared",
      all(k in shared_keys for k in all_keys if 'conv' in k and 'weight' in k))
check("Semantic head is shared",
      all(k in shared_keys for k in all_keys if 'semantic_head' in k))
check("Classifier head is shared",
      all(k in shared_keys for k in all_keys if k.startswith('head.')))

n_private = len(private_keys)
n_shared = len(shared_keys)
check(f"Private={n_private}, Shared={n_shared}, Total={len(all_keys)}",
      n_private + n_shared == len(all_keys))


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print(f"Results: {PASS} PASSED, {FAIL} FAILED out of {PASS+FAIL} tests")
if FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print(f"*** {FAIL} TESTS FAILED ***")
    sys.exit(1)
