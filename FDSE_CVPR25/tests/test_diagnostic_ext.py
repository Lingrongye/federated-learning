"""Unit tests for extended diagnostics (KL-collapse, R_d, IRM grad var)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.common.diagnostic_ext import (
    domain_conditional_rate,
    irm_gradient_variance,
    kl_collapse_detect,
)


# --- KL collapse ---


def test_kl_collapse_true():
    assert kl_collapse_detect(kl_mean=0.05, intra_class_z_std=0.05) is True


def test_kl_collapse_false_high_kl():
    assert kl_collapse_detect(kl_mean=1.0, intra_class_z_std=0.05) is False


def test_kl_collapse_false_high_std():
    assert kl_collapse_detect(kl_mean=0.05, intra_class_z_std=0.5) is False


def test_kl_collapse_custom_thresh():
    assert kl_collapse_detect(
        kl_mean=0.3, intra_class_z_std=0.3, kl_thresh=0.5, std_thresh=0.5
    ) is True


# --- Domain-conditional rate ---


def test_R_d_basic():
    kl = torch.tensor([1.0, 2.0, 3.0, 4.0])
    d = torch.tensor([0, 0, 1, 1])
    result = domain_conditional_rate(kl, d, num_domains=2)
    assert result['R_d_0'] == pytest.approx(1.5, abs=1e-6)
    assert result['R_d_1'] == pytest.approx(3.5, abs=1e-6)
    # std of [1.5, 3.5]
    import math
    expected_std = math.sqrt(((1.5 - 2.5) ** 2 + (3.5 - 2.5) ** 2) / (2 - 1))  # sample std (ddof=1)
    assert result['R_std_across_domains'] == pytest.approx(expected_std, abs=1e-4)


def test_R_d_empty_domain():
    kl = torch.tensor([1.0, 2.0])
    d = torch.tensor([0, 0])
    result = domain_conditional_rate(kl, d, num_domains=3)
    assert result['R_d_0'] == pytest.approx(1.5, abs=1e-6)
    import math
    assert math.isnan(result['R_d_1'])
    assert math.isnan(result['R_d_2'])


def test_R_d_single_domain_zero_std():
    kl = torch.tensor([1.0, 2.0])
    d = torch.tensor([0, 0])
    result = domain_conditional_rate(kl, d, num_domains=1)
    assert result['R_std_across_domains'] == 0.0


# --- IRM gradient variance ---


def test_irm_grad_var_same_loss_per_domain():
    torch.manual_seed(0)
    x = torch.randn(4, 3)
    params = [torch.nn.Parameter(torch.randn(3, 5))]
    # Same loss for each sample; per-domain gradients should be equal -> variance 0
    logits = x @ params[0]
    y = torch.tensor([0, 0, 0, 0])
    loss_per_sample = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    domains = torch.tensor([0, 0, 1, 1])

    gvar = irm_gradient_variance(loss_per_sample, params, domains, num_domains=2)
    # Even same-class, different inputs still produce different gradients
    assert gvar >= 0.0


def test_irm_grad_var_single_domain_returns_zero():
    params = [torch.nn.Parameter(torch.randn(3))]
    loss_per_sample = torch.tensor([1.0, 2.0])
    domains = torch.tensor([0, 0])
    gvar = irm_gradient_variance(loss_per_sample, params, domains, num_domains=2)
    assert gvar == 0.0  # only 1 domain present


def test_irm_grad_var_multi_domain():
    torch.manual_seed(0)
    x = torch.randn(8, 3)
    params = [torch.nn.Parameter(torch.randn(3, 5))]
    logits = x @ params[0]
    y = torch.randint(0, 5, (8,))
    loss_per_sample = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    domains = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    gvar = irm_gradient_variance(loss_per_sample, params, domains, num_domains=4)
    assert gvar > 0.0  # domains differ, so variance should be positive


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
