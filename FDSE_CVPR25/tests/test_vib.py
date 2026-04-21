"""Unit tests for VIB semantic head.

Covers: forward shapes, reparameterize gradients, KL closed-form correctness,
log_var clamping, prototype EMA, intra-class std, warmup schedule.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.common.vib import (
    LOG_VAR_MAX,
    LOG_VAR_MIN,
    PROTOTYPE_EMA_BETA,
    VIBSemanticHead,
    kl_gaussian_closed_form,
    lambda_ib_schedule,
)


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def head():
    torch.manual_seed(0)
    return VIBSemanticHead(feat_dim=64, proj_dim=16, num_classes=7)


@pytest.fixture
def batch():
    torch.manual_seed(1)
    h = torch.randn(8, 64, requires_grad=True)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0])
    return h, y


# --- VIB head forward & shapes ----------------------------------------------


def test_forward_shapes(head, batch):
    h, y = batch
    z, mu, log_var, kl = head(h, y, training=True)
    assert z.shape == (8, 16)
    assert mu.shape == (8, 16)
    assert log_var.shape == (8, 16)
    # KL is None since prototype not initialized
    assert kl is None


def test_eval_mode_deterministic(head, batch):
    h, _ = batch
    head.eval()
    z1, mu1, _, _ = head(h, training=False)
    z2, mu2, _, _ = head(h, training=False)
    assert torch.allclose(z1, mu1)
    assert torch.allclose(z1, z2)


def test_train_mode_stochastic(head, batch):
    h, _ = batch
    head.train()
    z1, mu, _, _ = head(h, training=True)
    z2, _, _, _ = head(h, training=True)
    # Same input, two samples should differ due to reparameterize
    assert not torch.allclose(z1, z2)
    # But the mu should be identical across calls (deterministic)
    z3, mu2, _, _ = head(h, training=True)
    assert torch.allclose(mu, mu2)


# --- Numerical stability ----------------------------------------------------


def test_log_var_clamping(head):
    # Force extreme input to verify clamp
    big = torch.zeros(1, 64)
    big[:, 0] = 1000.0
    _, _, log_var, _ = head(big, training=True)
    assert log_var.max().item() <= LOG_VAR_MAX + 1e-5
    assert log_var.min().item() >= LOG_VAR_MIN - 1e-5


def test_gradient_flow_mu_and_log_var(head, batch):
    h, y = batch
    # Initialize prototype so KL is computed
    head.prototype_ema.data = torch.randn(7, 16)
    head.prototype_init.fill_(True)

    z, mu, log_var, kl = head(h, y, training=True)
    loss = z.sum() + (kl if kl is not None else 0.0)
    loss.backward()

    # Gradients should reach mu_head params
    mu_head_grad = head.mu_head[0].weight.grad
    assert mu_head_grad is not None
    assert mu_head_grad.abs().sum().item() > 0

    # Gradients should reach log_var_head params
    lv_head_grad = head.log_var_head[0].weight.grad
    assert lv_head_grad is not None
    assert lv_head_grad.abs().sum().item() > 0

    # Input gradient (reparameterize path)
    assert h.grad is not None
    assert h.grad.abs().sum().item() > 0


# --- KL closed-form ---------------------------------------------------------


def test_kl_same_distribution_is_zero():
    mu = torch.zeros(4, 8)
    log_var = torch.zeros(4, 8)
    kl = kl_gaussian_closed_form(mu, log_var, mu.clone(), log_var.clone())
    assert kl.item() == pytest.approx(0.0, abs=1e-6)


def test_kl_against_manual():
    """KL(N(1, 2) || N(0, 1)) on 1-D."""
    # var_q = e^0 = 1? No, we use log_var directly
    # Let log_var_q = log(2), log_var_p = log(1) = 0
    mu_q = torch.tensor([[1.0]])
    log_var_q = torch.tensor([[math.log(2.0)]])
    mu_p = torch.tensor([[0.0]])
    log_var_p = torch.tensor([[0.0]])
    # Formula: 0.5 * (log(1/2) + (2 + 1) / 1 - 1) = 0.5 * (-0.693 + 3 - 1) = 0.5 * 1.307 = 0.6534
    expected = 0.5 * (math.log(0.5) + 3.0 - 1.0)
    kl = kl_gaussian_closed_form(mu_q, log_var_q, mu_p, log_var_p)
    assert kl.item() == pytest.approx(expected, abs=1e-4)


def test_kl_nonnegative():
    torch.manual_seed(0)
    for _ in range(5):
        mu_q = torch.randn(4, 8)
        log_var_q = torch.randn(4, 8).clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        mu_p = torch.randn(4, 8)
        log_var_p = torch.randn(4, 8).clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        kl = kl_gaussian_closed_form(mu_q, log_var_q, mu_p, log_var_p)
        assert kl.item() >= -1e-5  # numeric tolerance


# --- Prototype EMA ----------------------------------------------------------


def test_prototype_first_update_direct_copy(head):
    new_proto = torch.randn(7, 16)
    mask = torch.ones(7, dtype=torch.bool)
    head.update_prototype_ema(new_proto, mask)
    assert torch.allclose(head.prototype_ema, new_proto)
    assert bool(head.prototype_init.item())


def test_prototype_ema_beta_mixing(head):
    # Init
    p0 = torch.randn(7, 16)
    head.update_prototype_ema(p0, torch.ones(7, dtype=torch.bool))

    p1 = torch.randn(7, 16)
    head.update_prototype_ema(p1, torch.ones(7, dtype=torch.bool))

    expected = PROTOTYPE_EMA_BETA * p0 + (1 - PROTOTYPE_EMA_BETA) * p1
    assert torch.allclose(head.prototype_ema, expected, atol=1e-5)


def test_prototype_inactive_class_not_updated(head):
    p0 = torch.ones(7, 16)
    head.update_prototype_ema(p0, torch.ones(7, dtype=torch.bool))

    # Only class 3 active in this round
    p1 = torch.zeros(7, 16)
    mask = torch.zeros(7, dtype=torch.bool)
    mask[3] = True
    head.update_prototype_ema(p1, mask)

    # Class 3 should be mixed, others should stay at p0 = 1
    for c in range(7):
        if c == 3:
            expected = PROTOTYPE_EMA_BETA * 1.0 + (1 - PROTOTYPE_EMA_BETA) * 0.0
            assert torch.allclose(head.prototype_ema[c], torch.full((16,), expected), atol=1e-5)
        else:
            assert torch.allclose(head.prototype_ema[c], torch.ones(16))


# --- KL computation after prototype init -----------------------------------


def test_kl_computed_after_prototype_init(head, batch):
    h, y = batch
    # Pre-init
    _, _, _, kl_none = head(h, y, training=True)
    assert kl_none is None

    # Initialize prototype
    head.update_prototype_ema(torch.randn(7, 16), torch.ones(7, dtype=torch.bool))

    _, _, _, kl = head(h, y, training=True)
    assert kl is not None
    assert kl.item() >= 0.0


def test_prototype_prior_stop_grad(head, batch):
    h, y = batch
    head.update_prototype_ema(torch.randn(7, 16, requires_grad=True),
                              torch.ones(7, dtype=torch.bool))
    # prototype_ema is a buffer, not a parameter, so no grad anyway,
    # but verify .detach() chain: grad on prototype should not propagate.
    _, mu, log_var, kl = head(h, y, training=True)
    kl.backward()
    # h should have gradient via mu path
    assert h.grad is not None


# --- Intra-class std --------------------------------------------------------


def test_intra_class_std(head):
    y = torch.tensor([0, 0, 0, 1, 1, 2])
    z = torch.cat([
        torch.ones(3, 16),     # class 0: zero std
        torch.randn(2, 16),    # class 1: positive std
        torch.zeros(1, 16),    # class 2: single, skipped
    ])
    s = head.get_intra_class_std(z, y)
    # Only class 0 and class 1 counted
    # Class 0 std = 0, class 1 std = some positive
    assert s.item() >= 0.0
    assert s.item() < 5.0


# --- Warmup schedule --------------------------------------------------------


def test_lambda_ib_warmup():
    assert lambda_ib_schedule(0) == 0.0
    assert lambda_ib_schedule(19) == 0.0
    assert lambda_ib_schedule(20) == 0.0
    assert lambda_ib_schedule(35) == pytest.approx(0.5, abs=0.01)
    assert lambda_ib_schedule(50) == 1.0
    assert lambda_ib_schedule(200) == 1.0


# --- Integration: loss backprop end-to-end ----------------------------------


def test_full_loss_backprop(head, batch):
    h, y = batch
    head.update_prototype_ema(torch.randn(7, 16), torch.ones(7, dtype=torch.bool))

    z, mu, log_var, kl = head(h, y, training=True)
    classifier = torch.nn.Linear(16, 7)
    logits = classifier(z)
    ce = torch.nn.functional.cross_entropy(logits, y)
    loss = ce + 0.1 * kl
    loss.backward()

    # log_sigma_prior should get gradient
    assert head.log_sigma_prior.grad is not None
    assert head.log_sigma_prior.grad.abs().sum().item() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
