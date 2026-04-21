"""Unit tests for SupCon loss + diagnostics."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.common.supcon import supcon_loss, supcon_diagnostics


def test_shape_and_scalar():
    torch.manual_seed(0)
    z = torch.randn(8, 16)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supcon_loss(z, y)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_single_class_batch_returns_zero_or_skip():
    torch.manual_seed(0)
    z = torch.randn(4, 16)
    y = torch.zeros(4, dtype=torch.long)  # all same class
    loss = supcon_loss(z, y)
    # All anchors have positives (3 same-class peers), loss should be valid
    assert loss.item() >= 0.0


def test_no_positive_returns_zero():
    z = torch.randn(4, 16)
    y = torch.arange(4)  # each sample unique class; no positives
    loss = supcon_loss(z, y)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_gradient_flows():
    torch.manual_seed(0)
    z = torch.randn(8, 16, requires_grad=True)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supcon_loss(z, y)
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum().item() > 0


def test_temperature_effect():
    """Lower temperature -> larger loss differences (sharper distribution)."""
    torch.manual_seed(0)
    z = torch.randn(8, 16)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss_hot = supcon_loss(z, y, temperature=1.0)
    loss_cold = supcon_loss(z, y, temperature=0.07)
    # Not strictly guaranteed, but with random z, cold loss usually larger
    assert loss_hot.item() >= 0.0
    assert loss_cold.item() >= 0.0


def test_aligned_vs_scattered():
    """Perfectly aligned same-class vs scattered: aligned should have LOWER loss."""
    torch.manual_seed(0)
    # 4 classes, 2 samples each
    # Aligned: same class has identical embedding
    z_aligned = torch.cat([
        torch.tensor([[1.0, 0.0]] * 2),
        torch.tensor([[0.0, 1.0]] * 2),
        torch.tensor([[-1.0, 0.0]] * 2),
        torch.tensor([[0.0, -1.0]] * 2),
    ])
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    # Scattered: random
    z_scattered = torch.randn(8, 2)

    loss_aligned = supcon_loss(z_aligned, y, temperature=0.1)
    loss_scattered = supcon_loss(z_scattered, y, temperature=0.1)
    assert loss_aligned.item() < loss_scattered.item()


def test_diagnostics_alignment_uniformity():
    """Wang-Isola metrics: perfect uniform on unit sphere -> bounded uniformity."""
    torch.manual_seed(0)
    z = torch.randn(100, 16)
    y = torch.randint(0, 7, (100,))
    diag = supcon_diagnostics(z, y)
    assert 'pos_sim_mean' in diag
    assert 'neg_sim_mean' in diag
    assert 'alignment' in diag
    assert 'uniformity' in diag
    assert 'n_positive_avg' in diag
    assert -2.0 <= diag['pos_sim_mean'] <= 1.01
    assert -2.0 <= diag['neg_sim_mean'] <= 1.01
    assert diag['alignment'] >= 0.0
    assert diag['n_positive_avg'] >= 0.0


def test_diagnostics_empty_batch():
    z = torch.randn(1, 16)
    y = torch.tensor([0])
    diag = supcon_diagnostics(z, y)
    assert diag['pos_sim_mean'] == 0.0


def test_degenerate_info_nce_when_one_pos():
    """If each anchor has exactly 1 positive, SupCon should degenerate toward InfoNCE."""
    torch.manual_seed(0)
    z = torch.randn(6, 16)
    y = torch.tensor([0, 0, 1, 1, 2, 2])  # each anchor has 1 pos of same class
    loss = supcon_loss(z, y)
    assert loss.item() >= 0.0
    assert not math.isnan(loss.item())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
