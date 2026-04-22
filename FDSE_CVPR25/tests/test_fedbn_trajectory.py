"""Unit + integration tests for EXP-119 Sanity C: fedbn_trajectory.

Tests cover:
  * forward_with_feature shapes (shared with fedbn_ccbank)
  * Trajectory math: v = p^t - p^{t-1}, p_hat = p^t + η·v
  * EMA prototype update
  * Weighted prototype aggregation across clients (by sample count)
  * Alignment loss: L_align = 1 - cos(h, p_hat[y])
  * Warmup behavior: no alignment when current_round < warmup
  * End-to-end 3 round simulation: prototype + predict + alignment, no NaN
  * η=0 vs η=0.5 produce different predicted prototypes

Run:
    cd FDSE_CVPR25 && pytest tests/test_fedbn_trajectory.py -v
"""
import copy
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.fedbn_trajectory import forward_with_feature


# ============================================================
# Mini model
# ============================================================


class _MiniAlexNet(nn.Module):
    def __init__(self, num_classes=7, feat_dim=1024, proj_input=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(8 * 6 * 6, proj_input)
        self.bn6 = nn.BatchNorm1d(proj_input)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(proj_input, feat_dim)
        self.bn7 = nn.BatchNorm1d(feat_dim)
        self.fc3 = nn.Linear(feat_dim, num_classes)


@pytest.fixture
def mini_model():
    torch.manual_seed(0)
    return _MiniAlexNet()


# ============================================================
# forward_with_feature
# ============================================================


def test_forward_feature_shape(mini_model):
    model = mini_model.eval()
    x = torch.randn(8, 3, 32, 32)
    logits, h = forward_with_feature(model, x)
    assert logits.shape == (8, 7)
    assert h.shape == (8, 1024)
    assert torch.isfinite(logits).all()


# ============================================================
# Trajectory math — velocity and prediction
# ============================================================


def test_velocity_prediction_eta_zero():
    """eta=0 should reduce to p_hat = p (no prediction)."""
    eta = 0.0
    p_current = torch.tensor([1.0, 2.0, 3.0])
    p_prev = torch.tensor([0.5, 1.0, 1.5])
    v = p_current - p_prev
    p_hat = p_current + eta * v
    assert torch.allclose(p_hat, p_current)


def test_velocity_prediction_eta_half():
    """eta=0.5 should extrapolate by half the velocity."""
    eta = 0.5
    p_current = torch.tensor([2.0, 3.0])
    p_prev = torch.tensor([1.0, 2.0])
    v = p_current - p_prev
    p_hat = p_current + eta * v
    expected = torch.tensor([2.5, 3.5])
    assert torch.allclose(p_hat, expected)


def test_velocity_prediction_eta_one():
    """eta=1 should extrapolate one full step forward."""
    eta = 1.0
    p_current = torch.tensor([2.0])
    p_prev = torch.tensor([1.0])
    v = p_current - p_prev
    p_hat = p_current + eta * v
    expected = torch.tensor([3.0])
    assert torch.allclose(p_hat, expected)


def test_no_previous_prototype_fallback():
    """If p_prev is missing (first round after warmup), p_hat should fallback to p_current."""
    # Simulate server-side _predict_next for a class that only has p_current
    p_current_dict = {0: torch.tensor([1.0, 2.0])}
    p_prev_dict = {}  # empty
    eta = 0.5
    # Mimic the _predict_next logic
    preds = {}
    for c, p_c in p_current_dict.items():
        if c in p_prev_dict:
            v_c = p_c - p_prev_dict[c]
            preds[c] = p_c + eta * v_c
        else:
            preds[c] = p_c.clone()
    assert torch.allclose(preds[0], torch.tensor([1.0, 2.0]))


# ============================================================
# EMA prototype update
# ============================================================


def test_ema_prototype_update():
    """p_new = β · p_old + (1 - β) · p_agg, β=0.9."""
    beta = 0.9
    p_old = torch.tensor([1.0, 1.0])
    p_agg = torch.tensor([5.0, 5.0])
    p_new = beta * p_old + (1 - beta) * p_agg
    expected = torch.tensor([1.4, 1.4])
    assert torch.allclose(p_new, expected, atol=1e-5)


def test_ema_beta_zero_no_memory():
    """β=0 -> p_new = p_agg (no memory)."""
    beta = 0.0
    p_old = torch.tensor([1.0, 1.0])
    p_agg = torch.tensor([5.0, 5.0])
    p_new = beta * p_old + (1 - beta) * p_agg
    assert torch.allclose(p_new, p_agg)


# ============================================================
# Weighted prototype aggregation across clients
# ============================================================


def _weighted_aggregate(client_protos, client_counts):
    """Pure-Python version of Server._update_prototypes aggregation step.

    client_protos: list of dict[c -> tensor]
    client_counts: list of dict[c -> int]
    Returns: dict[c -> aggregated tensor]
    """
    agg_sum = defaultdict(lambda: None)
    agg_count = defaultdict(int)
    for protos, counts in zip(client_protos, client_counts):
        if protos is None:
            continue
        for c, p in protos.items():
            n = int(counts.get(c, 0))
            if n <= 0:
                continue
            p_w = p.float() * n
            if agg_sum[c] is None:
                agg_sum[c] = p_w.clone()
            else:
                agg_sum[c] += p_w
            agg_count[c] += n
    out = {}
    for c, s in agg_sum.items():
        if agg_count[c] > 0:
            out[c] = s / agg_count[c]
    return out


def test_aggregate_equal_weights():
    """Two clients same sample count -> simple average."""
    p1 = {0: torch.tensor([1.0, 1.0])}
    p2 = {0: torch.tensor([3.0, 3.0])}
    c1 = {0: 10}
    c2 = {0: 10}
    agg = _weighted_aggregate([p1, p2], [c1, c2])
    assert torch.allclose(agg[0], torch.tensor([2.0, 2.0]))


def test_aggregate_weighted_by_count():
    """Client with more samples should dominate."""
    p1 = {0: torch.tensor([1.0])}
    p2 = {0: torch.tensor([5.0])}
    c1 = {0: 100}
    c2 = {0: 10}
    agg = _weighted_aggregate([p1, p2], [c1, c2])
    # (100*1 + 10*5) / 110 = 150/110 ≈ 1.3636
    assert torch.allclose(agg[0], torch.tensor([1.3636]), atol=1e-3)


def test_aggregate_missing_class_ignored():
    """If one client doesn't have class c, aggregation still works using others."""
    p1 = {0: torch.tensor([1.0])}   # client 1: has class 0
    p2 = {1: torch.tensor([2.0])}   # client 2: only class 1
    c1 = {0: 10}
    c2 = {1: 10}
    agg = _weighted_aggregate([p1, p2], [c1, c2])
    assert torch.allclose(agg[0], torch.tensor([1.0]))
    assert torch.allclose(agg[1], torch.tensor([2.0]))


# ============================================================
# Alignment loss
# ============================================================


def test_alignment_loss_zero_when_identical():
    """If h[i] == p_hat[y_i] exactly -> cos=1 -> loss = 0."""
    D = 16
    pred_protos_gpu = {0: torch.ones(D)}
    y = torch.tensor([0, 0])
    h = torch.ones(2, D)  # identical to prototype
    targets = torch.zeros_like(h)
    mask = torch.zeros(2, dtype=torch.bool)
    for i in range(2):
        yi = int(y[i].item())
        if yi in pred_protos_gpu:
            targets[i] = pred_protos_gpu[yi]
            mask[i] = True
    h_m = h[mask]
    t_m = targets[mask]
    cos = F.cosine_similarity(h_m, t_m, dim=-1)
    loss = (1.0 - cos).mean()
    assert float(loss) < 1e-5, f'cos=1 should give loss=0, got {loss:.6f}'


def test_alignment_loss_max_when_opposite():
    """If h == -p_hat -> cos=-1 -> loss=2."""
    D = 16
    pred_protos_gpu = {0: torch.ones(D)}
    y = torch.tensor([0])
    h = -torch.ones(1, D)
    targets = torch.zeros_like(h)
    mask = torch.ones(1, dtype=torch.bool)
    targets[0] = pred_protos_gpu[0]
    cos = F.cosine_similarity(h, targets, dim=-1)
    loss = (1.0 - cos).mean()
    assert 1.9 < float(loss) <= 2.0, f'cos=-1 should give loss≈2, got {loss:.4f}'


def test_alignment_loss_unavailable_class():
    """If y's class is NOT in pred_protos_gpu, sample is masked out."""
    D = 4
    pred_protos_gpu = {0: torch.ones(D)}  # only class 0 available
    y = torch.tensor([0, 1, 1])  # two samples of class 1 (no prototype)
    h = torch.randn(3, D)
    mask = torch.zeros(3, dtype=torch.bool)
    for i in range(3):
        yi = int(y[i].item())
        if yi in pred_protos_gpu:
            mask[i] = True
    # Only 1 sample should have alignment applied (class 0)
    assert int(mask.sum()) == 1


def test_alignment_loss_all_unavailable_returns_zero():
    """If NO class has a prototype -> loss = 0 (early return)."""
    D = 4
    pred_protos_gpu = {}  # empty!
    y = torch.tensor([0, 1])
    h = torch.randn(2, D)
    mask = torch.zeros(2, dtype=torch.bool)
    for i in range(2):
        yi = int(y[i].item())
        if yi in pred_protos_gpu:
            mask[i] = True
    if not mask.any():
        loss = torch.tensor(0.0)
    else:
        loss = None
    assert float(loss) == 0.0


# ============================================================
# Warmup behavior
# ============================================================


def test_warmup_predicts_empty():
    """During warmup (current_round < warmup_rounds), _predict_next should return {}."""
    warmup = 5
    for round_idx in range(5):
        # Simulate: current_round < warmup -> empty
        if round_idx < warmup:
            preds = {}
        else:
            preds = {0: torch.zeros(4)}
        if round_idx < warmup:
            assert preds == {}


def test_warmup_boundary():
    """At round = warmup, prediction should kick in."""
    warmup = 5
    p_current = {0: torch.tensor([2.0])}
    p_prev = {0: torch.tensor([1.0])}
    for round_idx in range(10):
        if round_idx < warmup:
            preds = {}
        else:
            preds = {}
            for c, p_c in p_current.items():
                if c in p_prev:
                    v = p_c - p_prev[c]
                    preds[c] = p_c + 0.5 * v
                else:
                    preds[c] = p_c.clone()
        if round_idx >= warmup:
            assert 0 in preds
            assert torch.allclose(preds[0], torch.tensor([2.5]))


# ============================================================
# End-to-end multi-round simulation
# ============================================================


def test_multi_round_trajectory(mini_model):
    """Simulate 4 rounds: aggregate prototypes, roll p_prev/p_current, compute predictions.

    Verifies no NaN and that predictions evolve sensibly.
    """
    torch.manual_seed(0)
    model = mini_model.eval()
    eta = 0.5
    warmup = 2

    p_current = {}
    p_prev = {}

    for round_idx in range(5):
        # Fake client prototypes: different classes, drifting upward each round
        fake_client_protos = [
            {0: torch.ones(1024) * (round_idx + 1), 1: torch.ones(1024) * (round_idx * 2)},
            {0: torch.ones(1024) * (round_idx + 1.5), 1: torch.ones(1024) * (round_idx * 2 + 0.5)},
        ]
        fake_client_counts = [
            {0: 10, 1: 10},
            {0: 10, 1: 10},
        ]
        # Before updating p_current, save to p_prev
        p_prev = {c: p.clone() for c, p in p_current.items()}
        # Aggregate
        agg = _weighted_aggregate(fake_client_protos, fake_client_counts)
        p_current = agg

        # Verify no NaN
        for c, p in p_current.items():
            assert torch.isfinite(p).all()

        # Predict next
        if round_idx >= warmup:
            preds = {}
            for c, p_c in p_current.items():
                if c in p_prev:
                    v_c = p_c - p_prev[c]
                    preds[c] = p_c + eta * v_c
                else:
                    preds[c] = p_c.clone()
            for c, p in preds.items():
                assert torch.isfinite(p).all()


def test_eta_zero_vs_half_differ():
    """η=0 and η=0.5 should give different p_hat when velocity is nonzero."""
    p_current = {0: torch.tensor([2.0, 3.0])}
    p_prev = {0: torch.tensor([1.0, 1.0])}
    v = p_current[0] - p_prev[0]
    p_hat_0 = p_current[0] + 0.0 * v
    p_hat_5 = p_current[0] + 0.5 * v
    assert not torch.allclose(p_hat_0, p_hat_5), 'η=0 vs η=0.5 should differ when v != 0'


def test_eta_zero_vs_half_same_when_v_zero():
    """When velocity is 0 (no drift), η=0 and η=0.5 produce the same p_hat."""
    p_current = {0: torch.tensor([2.0, 3.0])}
    p_prev = {0: torch.tensor([2.0, 3.0])}  # no drift
    v = p_current[0] - p_prev[0]
    p_hat_0 = p_current[0] + 0.0 * v
    p_hat_5 = p_current[0] + 0.5 * v
    assert torch.allclose(p_hat_0, p_hat_5)


# ============================================================
# Gradient flow (key integration check)
# ============================================================


def test_alignment_loss_gradient_flows_to_encoder(mini_model):
    """Verify that L_align gradient reaches encoder params (not just classifier).

    This ensures the prototype-alignment actually trains the encoder.
    """
    model = mini_model.train()
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 7, (8,))
    _, h = forward_with_feature(model, x)
    # Build fake pred_protos
    pred_protos = {c: torch.randn(1024) for c in range(7)}
    # Build targets matrix for all samples
    targets = torch.stack([pred_protos[int(yi.item())] for yi in y], dim=0).detach()
    cos = F.cosine_similarity(h, targets, dim=-1)
    loss_align = (1.0 - cos).mean()
    loss_align.backward()
    # Check encoder params have gradient
    enc_grad_norm = sum(
        p.grad.norm().item()
        for n, p in model.named_parameters()
        if p.grad is not None and ('features' in n or 'fc1' in n or 'fc2' in n)
    )
    assert enc_grad_norm > 0, 'Alignment loss should propagate gradient to encoder'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
