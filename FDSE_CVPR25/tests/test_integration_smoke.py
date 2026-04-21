"""Integration smoke test for FedDSA-VIB/VSC/SupCon.

Simulates one training step end-to-end (encoder -> heads -> losses -> backward)
without flgo framework dependency. Validates:
- All loss components compute without NaN
- Gradients flow through mu_head, log_var_head, style_head, sem_classifier
- Prototype EMA updates work
- SupCon + VIB combined doesn't explode
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.common.vib import VIBSemanticHead, lambda_ib_schedule
from algorithm.common.supcon import supcon_loss, supcon_diagnostics


# --- Mini model surrogate (bypass flgo) --------------------------------------


class MiniEncoder(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, feat_dim)
        )

    def forward(self, x):
        return self.net(x)


class MiniFedDSAModel(nn.Module):
    """Minimal analog of FedDSAVIBModel for smoke testing."""

    def __init__(self, num_classes=7, feat_dim=128, proj_dim=32, vib=True):
        super().__init__()
        self.num_classes = num_classes
        self.vib = vib
        self.encoder = MiniEncoder(feat_dim)
        self.style_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        if vib:
            self.semantic_head = VIBSemanticHead(feat_dim, proj_dim, num_classes)
        else:
            self.semantic_head = nn.Sequential(
                nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
            )
        self.sem_classifier = nn.Linear(proj_dim, num_classes)


# --- Smoke tests -------------------------------------------------------------


def _gen_batch(B=16, C=7):
    torch.manual_seed(0)
    x = torch.randn(B, 32)
    y = torch.arange(C).repeat(B // C + 1)[:B]  # every class appears
    y = y[torch.randperm(B)]
    return x, y


def test_vib_supcon_full_forward_backward():
    """Variant B: VIB + SupCon full loss backward, no NaN, grads flow."""
    model = MiniFedDSAModel(num_classes=7, vib=True)
    x, y = _gen_batch(B=14, C=7)

    # Initialize prototype so KL is computed
    model.semantic_head.update_prototype_ema(
        torch.randn(7, 32),
        torch.ones(7, dtype=torch.bool),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    # Forward
    h = model.encoder(x)
    z_sem, mu_sem, log_var_sem, kl = model.semantic_head(h, y=y, training=True)
    z_sty = model.style_head(h)
    logits = model.sem_classifier(z_sem)

    # Losses
    loss_ce = F.cross_entropy(logits, y)
    mu_n = F.normalize(mu_sem, dim=-1)
    sty_n = F.normalize(z_sty, dim=-1)
    loss_orth = ((mu_n * sty_n).sum(dim=-1) ** 2).mean()
    loss_vib = kl
    loss_sup = supcon_loss(mu_sem, y, temperature=0.07)

    total = loss_ce + 1.0 * loss_orth + 1.0 * loss_vib + 1.0 * loss_sup
    assert not torch.isnan(total), f"Total loss NaN: CE={loss_ce} orth={loss_orth} vib={loss_vib} sup={loss_sup}"
    total.backward()

    # Every learnable parameter should receive a gradient
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


def test_vib_alone_no_supcon():
    """Variant A: VIB only, no SupCon term."""
    model = MiniFedDSAModel(num_classes=7, vib=True)
    x, y = _gen_batch(B=14, C=7)
    model.semantic_head.update_prototype_ema(
        torch.randn(7, 32), torch.ones(7, dtype=torch.bool),
    )

    h = model.encoder(x)
    z_sem, mu_sem, log_var_sem, kl = model.semantic_head(h, y=y, training=True)
    loss = F.cross_entropy(model.sem_classifier(z_sem), y) + 0.5 * kl
    loss.backward()
    assert not torch.isnan(loss)
    assert model.semantic_head.log_sigma_prior.grad is not None


def test_supcon_alone_no_vib():
    """Variant M6: SupCon on plain head, no VIB."""
    model = MiniFedDSAModel(num_classes=7, vib=False)
    x, y = _gen_batch(B=14, C=7)

    h = model.encoder(x)
    z_sem = model.semantic_head(h)
    z_sty = model.style_head(h)
    logits = model.sem_classifier(z_sem)

    loss = (F.cross_entropy(logits, y)
            + 1.0 * supcon_loss(z_sem, y, temperature=0.07))
    loss.backward()
    assert not torch.isnan(loss)
    # Ensure sem_classifier receives gradient (main task)
    assert model.sem_classifier.weight.grad is not None


def test_prototype_ema_drives_kl_down():
    """Prototype ema should make KL reasonable (not explode)."""
    model = MiniFedDSAModel(num_classes=7, vib=True)
    x, y = _gen_batch(B=14, C=7)

    # Before init: kl is None
    h = model.encoder(x)
    _, _, _, kl_none = model.semantic_head(h, y=y, training=True)
    assert kl_none is None

    # After init with reasonable prior
    mu_prior = torch.randn(7, 32) * 0.1  # small-norm prior
    model.semantic_head.update_prototype_ema(mu_prior, torch.ones(7, dtype=torch.bool))

    _, _, _, kl = model.semantic_head(h, y=y, training=True)
    assert kl is not None
    # Sanity: KL is non-negative finite
    assert kl.item() >= -1e-5
    assert kl.item() < 1e5  # not exploded


def test_warmup_schedule_applied():
    """Verify lambda_ib_schedule integration."""
    assert lambda_ib_schedule(0) == 0.0
    assert lambda_ib_schedule(100) == 1.0
    # Intermediate
    r = 35
    lam = lambda_ib_schedule(r)
    assert 0.0 < lam < 1.0


def test_log_sigma_prior_learnable():
    """log_sigma_prior must accumulate gradient under KL loss."""
    model = MiniFedDSAModel(num_classes=7, vib=True)
    x, y = _gen_batch(B=14, C=7)
    model.semantic_head.update_prototype_ema(
        torch.randn(7, 32), torch.ones(7, dtype=torch.bool),
    )

    initial_sigma = model.semantic_head.log_sigma_prior.detach().clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    for _ in range(3):
        optimizer.zero_grad()
        h = model.encoder(x)
        _, _, _, kl = model.semantic_head(h, y=y, training=True)
        kl.backward()
        optimizer.step()

    # log_sigma_prior should have moved
    delta = (model.semantic_head.log_sigma_prior - initial_sigma).abs().max()
    assert delta.item() > 1e-4, "log_sigma_prior unchanged after 3 steps — gradient not flowing"


def test_supcon_with_sparse_positives():
    """Edge case: batch size small so some classes have 0 positives."""
    model = MiniFedDSAModel(num_classes=7, vib=False)
    # Only 3 unique classes in batch of 8 → many classes have 0 positives
    x = torch.randn(8, 32)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4])

    h = model.encoder(x)
    z_sem = model.semantic_head(h)
    loss = supcon_loss(z_sem, y, temperature=0.07)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0


def test_integration_end_to_end_multiple_steps():
    """Run 5 training steps, verify loss descends (or at least stays finite)."""
    torch.manual_seed(42)
    model = MiniFedDSAModel(num_classes=7, vib=True)
    x, y = _gen_batch(B=14, C=7)
    model.semantic_head.update_prototype_ema(
        torch.randn(7, 32), torch.ones(7, dtype=torch.bool),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        h = model.encoder(x)
        z_sem, mu_sem, log_var_sem, kl = model.semantic_head(h, y=y, training=True)
        z_sty = model.style_head(h)
        logits = model.sem_classifier(z_sem)

        L_ce = F.cross_entropy(logits, y)
        mu_n = F.normalize(mu_sem, dim=-1)
        sty_n = F.normalize(z_sty, dim=-1)
        L_orth = ((mu_n * sty_n).sum(dim=-1) ** 2).mean()
        L_sup = supcon_loss(mu_sem, y, temperature=0.07)
        total = L_ce + 0.5 * L_orth + 0.5 * kl + 1.0 * L_sup
        assert not torch.isnan(total)
        total.backward()
        optimizer.step()
        losses.append(total.item())

    # After 5 steps, total loss should not have exploded
    assert losses[-1] < 1e4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
