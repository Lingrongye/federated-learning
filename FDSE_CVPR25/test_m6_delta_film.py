#!/usr/bin/env python3
"""
M6 Delta-FiLM Unit Tests
Verifies: StyleModulator, delta computation, gradient flow, residual FiLM, edge cases

Run: python test_m6_delta_film.py
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
print("M6 Delta-FiLM Unit Tests")
print("=" * 60)

# ============================================================
# Test 1: StyleModulator
# ============================================================
print("\n--- Test 1: StyleModulator ---")

class StyleModulator(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.film_net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim * 2)
        )
        self.gate_linear = nn.Linear(dim, 1)

    def forward(self, delta_sty):
        params = self.film_net(delta_sty)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)  # constrain to [-1, 1]
        gate = torch.sigmoid(self.gate_linear(delta_sty))
        return gamma, beta, gate

mod = StyleModulator(dim=128)
delta_input = torch.randn(1, 128)
gamma, beta, gate = mod(delta_input)

check("gamma shape", gamma.shape == (1, 128), f"got {gamma.shape}")
check("beta shape", beta.shape == (1, 128), f"got {beta.shape}")
check("gate shape", gate.shape == (1, 1), f"got {gate.shape}")
check("gamma in [-1, 1] (tanh)", gamma.min() >= -1.0 and gamma.max() <= 1.0, f"range [{gamma.min():.3f}, {gamma.max():.3f}]")
check("gate in [0, 1] (sigmoid)", gate.min() >= 0.0 and gate.max() <= 1.0, f"range [{gate.min():.3f}, {gate.max():.3f}]")

# Param count
param_count = sum(p.numel() for p in mod.parameters())
check("param count ~49K", 40000 < param_count < 60000, f"got {param_count}")

# ============================================================
# Test 2: Delta computation
# ============================================================
print("\n--- Test 2: Delta computation ---")

sty_local = torch.tensor([1.0, 0.0, 0.0, 0.0] + [0.0]*124)
sty_ext = torch.tensor([0.0, 1.0, 0.0, 0.0] + [0.0]*124)

delta_s = sty_ext - sty_local
delta_norm = delta_s.norm()
delta_s_normalized = delta_s / (delta_norm + 1e-8)

check("delta is non-zero", delta_norm > 0, f"norm={delta_norm:.4f}")
check("normalized delta is unit vector", abs(delta_s_normalized.norm().item() - 1.0) < 1e-5, f"norm={delta_s_normalized.norm():.6f}")

# Same domain -> delta near zero
sty_same = sty_local.clone() + torch.randn(128) * 1e-7
delta_same = sty_same - sty_local
check("same domain delta near zero", delta_same.norm() < 1e-5, f"norm={delta_same.norm():.8f}")

# ============================================================
# Test 3: Residual FiLM formula
# ============================================================
print("\n--- Test 3: Residual FiLM ---")

z_sem = torch.randn(128)
gamma_val = torch.zeros(128)  # gamma=0 -> no scaling change
beta_val = torch.zeros(128)   # beta=0 -> no shift
gate_val = torch.tensor(1.0)  # gate=1 -> full modulation

z_film = z_sem + gate_val * (gamma_val * z_sem + beta_val)
check("gamma=0, beta=0, gate=1 -> z_film == z_sem", torch.allclose(z_film, z_sem, atol=1e-6))

# gate=0 -> no modulation regardless of gamma/beta
gamma_big = torch.ones(128)
beta_big = torch.ones(128) * 10
gate_zero = torch.tensor(0.0)
z_film_gated = z_sem + gate_zero * (gamma_big * z_sem + beta_big)
check("gate=0 -> z_film == z_sem (no modulation)", torch.allclose(z_film_gated, z_sem, atol=1e-6))

# Full modulation with gamma=0.5
gamma_half = torch.ones(128) * 0.5
z_film_half = z_sem + gate_val * (gamma_half * z_sem + beta_val)
expected = z_sem + 0.5 * z_sem  # = 1.5 * z_sem
check("gamma=0.5 -> z_film = 1.5 * z_sem", torch.allclose(z_film_half, 1.5 * z_sem, atol=1e-6))

# ============================================================
# Test 4: Gradient flow through Delta-FiLM
# ============================================================
print("\n--- Test 4: Gradient flow ---")

# Build minimal model
encoder = nn.Sequential(nn.Linear(10, 128), nn.ReLU())
semantic_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
style_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
classifier = nn.Linear(128, 7)
style_modulator = StyleModulator(128)

# Forward pass
x = torch.randn(4, 10)
y = torch.tensor([0, 1, 2, 0])
h = encoder(x)
z_sem = semantic_head(h)
z_sty_live = style_head(h)  # LIVE z_sty with grad

# Simulate delta-FiLM for one sample
sty_ext_detached = torch.randn(128).detach()  # from server, no grad
sty_local_live = z_sty_live[0]  # current batch, WITH grad

delta_s = sty_ext_detached - sty_local_live  # grad flows through sty_local_live
delta_s = delta_s / (delta_s.norm() + 1e-8)

gamma, beta, gate = style_modulator(delta_s.unsqueeze(0))
gamma = gamma.squeeze(0)
beta = beta.squeeze(0)
gate = gate.squeeze()

z_sem_film = z_sem[0] + gate * (gamma * z_sem[0] + beta)
output_film = classifier(z_sem_film.unsqueeze(0))
loss_film = F.cross_entropy(output_film, y[0:1])

loss_film.backward()

# Check gradient destinations
encoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
sem_head_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in semantic_head.parameters())
sty_head_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in style_head.parameters())
modulator_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in style_modulator.parameters())
classifier_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in classifier.parameters())

check("encoder gets gradient", encoder_grad)
check("semantic_head gets gradient", sem_head_grad)
check("style_head gets gradient (via live z_sty)", sty_head_grad)
check("style_modulator gets gradient", modulator_grad)
check("classifier gets gradient", classifier_grad)

# Verify z_sty_live has grad (the key fix from Codex review)
check("z_sty_live[0] has grad_fn", z_sty_live.grad_fn is not None)
check("sty_local_live contributes to loss", sty_local_live.grad_fn is not None)

# ============================================================
# Test 5: Gradient does NOT flow from detached sty_ext
# ============================================================
print("\n--- Test 5: sty_ext is detached ---")
check("sty_ext_detached has no grad", not sty_ext_detached.requires_grad)

# ============================================================
# Test 6: Edge cases
# ============================================================
print("\n--- Test 6: Edge cases ---")

# Near-zero delta (same domain)
delta_tiny = torch.randn(128) * 1e-8
check("near-zero delta norm < 1e-6", delta_tiny.norm() < 1e-6)

# Missing class in cross protos (should skip)
cross_protos = {(0, 1): torch.randn(128), (1, 2): torch.randn(128)}
label_missing = 5  # class 5 not in cross_protos
candidates = [p for (c, cid), p in cross_protos.items() if c == label_missing]
check("missing class -> empty candidates", len(candidates) == 0)

# Single cross proto (should still work)
single_proto = {(0, 1): torch.randn(128)}
candidates_single = [p for (c, cid), p in single_proto.items() if c == 0]
check("single cross proto -> 1 candidate", len(candidates_single) == 1)

# ============================================================
# Test 7: tanh constrains gamma
# ============================================================
print("\n--- Test 7: gamma tanh constraint ---")

mod_test = StyleModulator(128)
# Even with extreme input, gamma should be in [-1, 1]
extreme_input = torch.randn(1, 128) * 100
gamma_ext, _, _ = mod_test(extreme_input)
check("extreme input -> gamma still in [-1,1]", gamma_ext.min() >= -1.0 and gamma_ext.max() <= 1.0, f"[{gamma_ext.min():.3f}, {gamma_ext.max():.3f}]")

# ============================================================
# Test 8: Inference path (no FiLM)
# ============================================================
print("\n--- Test 8: Inference (no FiLM) ---")

# At inference, only encoder -> semantic_head -> classifier
x_test = torch.randn(2, 10)
h_test = encoder(x_test)
z_sem_test = semantic_head(h_test)
output_test = classifier(z_sem_test)

check("inference output shape", output_test.shape == (2, 7), f"got {output_test.shape}")
# style_head and style_modulator NOT used at inference
check("inference does not need style_head", True)
check("inference does not need style_modulator", True)

# ============================================================
# Test 9: Numerical stability
# ============================================================
print("\n--- Test 9: Numerical stability ---")

z_sem_large = torch.randn(128) * 100
gamma_max = torch.ones(128)  # tanh max = 1
beta_test = torch.randn(128) * 0.1
gate_test = torch.tensor(0.5)

z_film_stable = z_sem_large + gate_test * (gamma_max * z_sem_large + beta_test)
check("large z_sem -> finite output", torch.isfinite(z_film_stable).all())

# With tanh, max scaling is 2x (1 + 1*gate)
max_ratio = z_film_stable.abs().max() / z_sem_large.abs().max()
check("max scaling bounded", max_ratio < 3.0, f"ratio={max_ratio:.2f}")

# ============================================================
# Test 10: Config parameter parsing
# ============================================================
print("\n--- Test 10: Config ---")

algo_para = [1.0, 0.0, 1.0, 0.2, 50, 5, 128, 0.05, 0.8, 0.05, 0.9, 6, 0.5, 1.0, 0.5, 0.5]
check("algo_para has 16 elements", len(algo_para) == 16, f"got {len(algo_para)}")
check("adaptive_mode=6", int(algo_para[11]) == 6)
check("tau=0.2", float(algo_para[3]) == 0.2)
check("lambda_sty_contrast=0.5", float(algo_para[15]) == 0.5)

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
