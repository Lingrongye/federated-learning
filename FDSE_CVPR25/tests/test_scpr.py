"""Unit tests for SCPR (Self-Masked Style-Weighted Multi-Positive InfoNCE).

Covers:
  - Self-mask correctness (w_{k->k} = 0)
  - scpr_mode=1 (uniform) style-free — fix for codex IMPORTANT #1
  - scpr_mode=2 requires style bank
  - tau -> infinity approaches uniform
  - tau -> 0 approaches one-hot (nearest-style)
  - weights sum to 1 (per-sample renormalize)
  - K=1 edge case (no other client)
  - _scpr_loss basic run + gradient detach on protos
  - has_pos=0 sample excluded cleanly
  - scpr_payload=None returns zero loss

Run:
    cd FDSE_CVPR25
    python -m pytest tests/test_scpr.py -v
"""
import os
import sys
import math

import pytest
import torch
import torch.nn.functional as F

# Add FDSE_CVPR25 to path so we can import algorithm.feddsa_scheduled
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algorithm.feddsa_scheduled import Server, Client  # noqa: E402


# ============================================================
# Minimal stand-ins that only expose the SCPR methods under test
# (we don't want to boot a full flgo server for these unit tests).
# ============================================================


class MockServer:
    """Bare-bones Server-like object that can call Server._compute_scpr_payload."""
    # Python 3.8 returns the underlying function directly for @staticmethod attr access.
    _extract_style_vec = staticmethod(Server._extract_style_vec)
    _compute_scpr_payload = Server._compute_scpr_payload

    def __init__(self, scpr_mode, scpr_tau_val=0.3):
        self.scpr_mode = scpr_mode
        self.scpr_tau_val = scpr_tau_val
        self.style_bank = {}          # SAS: 1024d pool5
        self.scpr_style_bank = {}     # SCPR: 128d z_sty (NEW 2026-04-19)
        self.client_class_protos = {}


class MockClient:
    """Bare-bones Client-like object that can call Client._scpr_loss."""
    _scpr_loss = Client._scpr_loss

    def __init__(self, tau=0.3):
        self.tau = tau
        self.scpr_payload = None


# ============================================================
# Helpers
# ============================================================


def make_style_bank(K, dim=1024, seed=42):
    torch.manual_seed(seed)
    bank = {}
    for k in range(K):
        mu = torch.randn(dim)
        sigma = torch.ones(dim)
        bank[k] = (mu, sigma)  # (mu, sigma) tuple, matching style_bank format
    return bank


def make_class_proto_bank(K, C, dim=128, seed=42):
    torch.manual_seed(seed)
    bank = {}
    for k in range(K):
        protos = {}
        for c in range(C):
            protos[c] = torch.randn(dim)
        bank[k] = protos
    return bank


# ============================================================
# Test 1: self-mask — w_{k->k} must not be in weights
# ============================================================


def test_self_mask_excludes_self():
    srv = MockServer(scpr_mode=2)
    srv.style_bank = make_style_bank(4)
    srv.client_class_protos = make_class_proto_bank(4, 7)

    payload = srv._compute_scpr_payload(target_cid=0)
    assert payload is not None
    assert 0 not in payload['weights'], "Self-mask broken: target cid is in weights"
    assert 0 not in payload['protos'], "Self-mask broken: target cid is in protos"
    assert set(payload['weights'].keys()) == {1, 2, 3}


def test_self_mask_scpr_mode_1():
    """Self-mask must work in uniform mode too."""
    srv = MockServer(scpr_mode=1)
    srv.style_bank = {}  # scpr=1 does not need style
    srv.client_class_protos = make_class_proto_bank(4, 7)

    payload = srv._compute_scpr_payload(target_cid=2)
    assert payload is not None
    assert 2 not in payload['weights']
    assert set(payload['weights'].keys()) == {0, 1, 3}


# ============================================================
# Test 2: scpr=1 uniform must be STYLE-FREE (codex IMPORTANT #1 regression)
# ============================================================


def test_scpr_uniform_is_style_free():
    """scpr_mode=1 (M3 lower bound) should work with empty style_bank."""
    srv = MockServer(scpr_mode=1)
    srv.style_bank = {}  # deliberately empty
    srv.client_class_protos = make_class_proto_bank(4, 7)

    payload = srv._compute_scpr_payload(target_cid=0)
    assert payload is not None, "codex-fix #1 regression: scpr=1 wrongly requires style"
    # Uniform 1/(K-1) = 1/3
    for cid, w in payload['weights'].items():
        assert math.isclose(w, 1.0 / 3, abs_tol=1e-6), f"cid={cid}: got {w}, expected 1/3"


def test_scpr_mode2_requires_style():
    srv = MockServer(scpr_mode=2)
    srv.style_bank = {}  # empty
    srv.client_class_protos = make_class_proto_bank(4, 7)

    payload = srv._compute_scpr_payload(target_cid=0)
    assert payload is None, "scpr=2 should return None when target style missing"


def test_scpr_mode2_prefers_scpr_style_bank_over_sas():
    """NEW 2026-04-19: SCPR should use its own 128d z_sty bank, not SAS's 1024d h bank.

    When both banks are populated, SCPR must consult scpr_style_bank (128d),
    NOT the legacy 1024d style_bank. This encodes the style-key fix.
    """
    srv = MockServer(scpr_mode=2, scpr_tau_val=0.001)  # tiny tau → one-hot to nearest

    # SAS bank: all 4 clients equal (pretend pool5 averages close, noisy)
    srv.style_bank = {k: (torch.ones(1024), torch.ones(1024)) for k in range(4)}

    # SCPR bank: 128d z_sty; client 0 is close to client 3 but far from 1 and 2
    # so if SCPR uses scpr_style_bank it should nearest-match client 3
    srv.scpr_style_bank = {
        0: (torch.tensor([1.0] * 128), torch.ones(128)),
        1: (torch.tensor([-1.0] * 128), torch.ones(128)),  # opposite
        2: (torch.tensor([-1.0] * 128), torch.ones(128)),  # opposite
        3: (torch.tensor([0.95] * 128), torch.ones(128)),  # near client 0
    }
    srv.client_class_protos = make_class_proto_bank(4, 7)

    payload = srv._compute_scpr_payload(target_cid=0)
    assert payload is not None
    ws = payload['weights']
    # With tiny tau SCPR should one-hot to client 3 (nearest in z_sty space),
    # which proves SCPR consulted scpr_style_bank, not the all-equal style_bank.
    assert 3 in ws and ws[3] > 0.95, \
        "SCPR should use scpr_style_bank: expected ws[3]≈1 (nearest in 128d), got %s" % ws


# ============================================================
# Test 3: tau -> infinity approaches uniform
# ============================================================


def test_weights_approach_uniform_large_tau():
    srv = MockServer(scpr_mode=2, scpr_tau_val=1e4)  # huge tau -> uniform
    srv.style_bank = make_style_bank(4)
    srv.client_class_protos = make_class_proto_bank(4, 7)

    payload = srv._compute_scpr_payload(target_cid=0)
    assert payload is not None
    uniform = 1.0 / 3
    for cid, w in payload['weights'].items():
        assert abs(w - uniform) < 1e-3, f"cid={cid}: got {w}, expected near-uniform {uniform}"


# ============================================================
# Test 4: tau -> 0 approaches one-hot (nearest style)
# ============================================================


def test_weights_approach_one_hot_small_tau():
    srv = MockServer(scpr_mode=2, scpr_tau_val=1e-3)  # tiny tau -> one-hot
    srv.style_bank = make_style_bank(4)
    srv.client_class_protos = make_class_proto_bank(4, 7)

    payload = srv._compute_scpr_payload(target_cid=0)
    assert payload is not None
    ws = sorted(payload['weights'].values(), reverse=True)
    # Top weight should be ~1, others ~0
    assert ws[0] > 0.99, f"Small tau: top weight should be ~1, got {ws[0]}"
    for w in ws[1:]:
        assert w < 0.01, f"Small tau: non-top weight should be ~0, got {w}"


# ============================================================
# Test 5: weights sum to 1 (both modes)
# ============================================================


def test_weights_sum_to_one_mode1():
    srv = MockServer(scpr_mode=1)
    srv.style_bank = {}
    srv.client_class_protos = make_class_proto_bank(4, 7)
    payload = srv._compute_scpr_payload(target_cid=0)
    assert payload is not None
    total = sum(payload['weights'].values())
    assert math.isclose(total, 1.0, abs_tol=1e-6), f"mode=1 weights sum={total}"


def test_weights_sum_to_one_mode2():
    srv = MockServer(scpr_mode=2, scpr_tau_val=0.3)
    srv.style_bank = make_style_bank(4)
    srv.client_class_protos = make_class_proto_bank(4, 7)
    payload = srv._compute_scpr_payload(target_cid=0)
    assert payload is not None
    total = sum(payload['weights'].values())
    assert math.isclose(total, 1.0, abs_tol=1e-6), f"mode=2 weights sum={total}"


# ============================================================
# Test 6: K=1 returns None (nothing to attend after self-mask)
# ============================================================


def test_k1_returns_none():
    srv = MockServer(scpr_mode=2, scpr_tau_val=0.3)
    srv.style_bank = make_style_bank(1)
    srv.client_class_protos = make_class_proto_bank(1, 7)
    assert srv._compute_scpr_payload(target_cid=0) is None


# ============================================================
# Test 7: _scpr_loss basic run + gradient detach on protos
# ============================================================


def test_scpr_loss_finite_and_positive():
    client = MockClient(tau=0.3)
    B, D = 8, 128
    K, C = 4, 7

    z_sem = torch.randn(B, D, requires_grad=True)
    y = torch.randint(0, C, (B,))

    protos_bank = make_class_proto_bank(K, C, dim=D)
    client.scpr_payload = {
        'weights': {1: 1 / 3, 2: 1 / 3, 3: 1 / 3},
        'protos': {1: protos_bank[1], 2: protos_bank[2], 3: protos_bank[3]},
    }
    loss = client._scpr_loss(z_sem, y)
    assert torch.isfinite(loss), "Loss must be finite"
    assert loss.item() > 0, "Loss must be positive"


def test_scpr_loss_grad_reaches_z_but_not_protos():
    """Gradient should flow to z_sem (encoder pathway) but NOT to proto bank."""
    client = MockClient(tau=0.3)
    B, D = 4, 64
    C = 3

    z_sem = torch.randn(B, D, requires_grad=True)
    y = torch.tensor([0, 1, 2, 0])

    # Make protos require grad to verify they DON'T receive grad
    protos = {
        1: {c: torch.randn(D, requires_grad=True) for c in range(C)},
        2: {c: torch.randn(D, requires_grad=True) for c in range(C)},
    }
    client.scpr_payload = {
        'weights': {1: 0.5, 2: 0.5},
        'protos': protos,
    }

    loss = client._scpr_loss(z_sem, y)
    loss.backward()

    assert z_sem.grad is not None, "z_sem must receive gradient"
    assert torch.isfinite(z_sem.grad).all(), "z_sem grad finite"
    # Protos must NOT accumulate grad (detached inside _scpr_loss)
    for cid in (1, 2):
        for c, p in protos[cid].items():
            assert p.grad is None, f"Proto {cid}/{c} received gradient (detach broken)"


# ============================================================
# Test 8: has_pos=0 (class c absent from all sources) -> loss = 0
# ============================================================


def test_no_positive_returns_zero_loss():
    client = MockClient(tau=0.3)
    D = 64
    # Source sources cover only class 5 and 6
    z_sem = torch.randn(2, D, requires_grad=True)
    y = torch.tensor([0, 0])

    client.scpr_payload = {
        'weights': {1: 0.5, 2: 0.5},
        'protos': {
            1: {5: torch.randn(D), 6: torch.randn(D)},
            2: {5: torch.randn(D), 6: torch.randn(D)},
        },
    }
    loss = client._scpr_loss(z_sem, y)
    # All samples lack positives -> weighted_sum=0 / denom=1 -> 0
    assert loss.item() == 0.0, f"no-positive samples should give 0 loss, got {loss.item()}"


def test_no_payload_returns_zero_loss():
    client = MockClient(tau=0.3)
    client.scpr_payload = None
    z_sem = torch.randn(4, 128)
    y = torch.randint(0, 7, (4,))
    loss = client._scpr_loss(z_sem, y)
    assert loss.item() == 0.0


# ============================================================
# Test 9: Per-class renormalize — some sources lack class c
# ============================================================


def test_per_class_renormalize():
    """Sample's positive weights renormalize over the sources that have class c."""
    client = MockClient(tau=0.3)
    D = 64
    z_sem = torch.randn(1, D, requires_grad=True)
    y = torch.tensor([0])  # class 0

    # Source 1 has class 0, source 2 does NOT. Both equally weighted a priori.
    p_target = torch.randn(D)
    client.scpr_payload = {
        'weights': {1: 0.5, 2: 0.5},
        'protos': {
            1: {0: p_target, 1: torch.randn(D)},  # has class 0
            2: {1: torch.randn(D), 2: torch.randn(D)},  # does NOT have class 0
        },
    }
    loss = client._scpr_loss(z_sem, y)
    # Sample y=0: only source 1 provides positive -> pos_w_norm = {1: 1.0}
    # i.e. full weight goes to source 1's class-0 proto; loss finite positive.
    assert torch.isfinite(loss), "Loss must be finite under per-class renorm"
    assert loss.item() > 0, "Loss must be positive when one source covers class c"


# ============================================================
# Test 10: scpr=1 pure-uniform has predictable structure for gradient
# ============================================================


def test_scpr_mode1_uniform_deterministic():
    """With uniform weights and no self-mask needed (K=2), check loss is deterministic."""
    client = MockClient(tau=1.0)  # high tau for stable logits
    D = 8
    torch.manual_seed(0)

    # Simple setup: 1 sample class 0, 2 source clients each with class {0, 1}
    z_sem = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    y = torch.tensor([0])

    p01 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # class 0 close to z
    p11 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # class 1 orthogonal
    p02 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # class 0 (same as p01)
    p12 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    client.scpr_payload = {
        'weights': {1: 0.5, 2: 0.5},
        'protos': {
            1: {0: p01, 1: p11},
            2: {0: p02, 1: p12},
        },
    }
    loss = client._scpr_loss(z_sem, y)
    # Both source clients have class 0 -> pos_w_norm = {1: 0.5, 2: 0.5}
    # z aligns with class-0 protos -> loss should be relatively small but > 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


if __name__ == '__main__':
    # Run directly (also usable with pytest)
    pytest.main([__file__, '-v', '-s'])
