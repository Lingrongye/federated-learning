"""VIB (Variational Information Bottleneck) semantic head for FedDSA-VIB.

Design decisions (from refine-logs/2026-04-22_FedDSA-Swap/round-3-revised-proposal.md):
- Stochastic encoder q(z_sem | x) = N(mu, sigma^2) via reparameterization.
- Class-conditional Gaussian prior N(prototype_ema[y], sigma_prior_y^2).
- EMA-lagged prototype with stop-gradient (fix: chicken-and-egg).
- Per-class learnable log_sigma_prior (fix: degenerate lookup solution).
- log_var clamped to [-5, 2] for numerical stability.
- Closed-form KL (no MC estimation).

Server MUST add 'log_var_head' and 'log_sigma_prior' to private_keys
to preserve per-domain uncertainty (fix: FedAvg sigma pollution).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


LOG_VAR_MIN = -5.0
LOG_VAR_MAX = 2.0
PROTOTYPE_EMA_BETA = 0.99


class VIBSemanticHead(nn.Module):
    """Stochastic semantic head with EMA prototype prior."""

    def __init__(self, feat_dim: int, proj_dim: int, num_classes: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.proj_dim = proj_dim
        self.num_classes = num_classes

        # Learnable mean head (participates in FedAvg)
        self.mu_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        # Learnable log-variance head (LOCAL: must NOT participate in FedAvg)
        self.log_var_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        # Learnable per-class log sigma prior (LOCAL: must NOT participate in FedAvg)
        self.log_sigma_prior = nn.Parameter(torch.zeros(num_classes))

        # EMA-lagged prototype buffer (no gradient, server-side updated via aggregation)
        self.register_buffer(
            'prototype_ema',
            torch.zeros(num_classes, proj_dim),
        )
        self.register_buffer(
            'prototype_init',
            torch.zeros(1, dtype=torch.bool),
        )

    def forward(
        self,
        h: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return (z_sem, mu, log_var, kl_loss).

        z_sem: sample from q (train) or mu (eval)
        mu: deterministic mean (used for orth/HSIC losses)
        log_var: clamped log variance
        kl_loss: scalar KL divergence (None if y is None or not training or prototype not initialized)
        """
        mu = self.mu_head(h)
        log_var = torch.clamp(self.log_var_head(h), LOG_VAR_MIN, LOG_VAR_MAX)

        if training:
            eps = torch.randn_like(mu)
            z_sem = mu + torch.exp(0.5 * log_var) * eps
        else:
            z_sem = mu

        kl_loss = None
        if training and y is not None and bool(self.prototype_init.item()):
            prior_mu = self.prototype_ema[y].detach()  # stop-grad
            prior_log_var = (
                2.0 * self.log_sigma_prior[y].unsqueeze(-1).expand_as(mu)
            )
            kl_loss = kl_gaussian_closed_form(mu, log_var, prior_mu, prior_log_var)

        return z_sem, mu, log_var, kl_loss

    @torch.no_grad()
    def update_prototype_ema(
        self,
        new_prototypes: torch.Tensor,
        class_active_mask: torch.Tensor,
    ) -> None:
        """Update prototype_ema with beta=0.99.

        new_prototypes: [num_classes, proj_dim] fresh aggregated prototypes
        class_active_mask: [num_classes] bool, True if class had >=1 sample this round
        """
        assert new_prototypes.shape == (self.num_classes, self.proj_dim)
        assert class_active_mask.shape == (self.num_classes,)

        device = self.prototype_ema.device
        new_prototypes = new_prototypes.to(device)
        class_active_mask = class_active_mask.to(device)

        if not bool(self.prototype_init.item()):
            # First valid update: use new values directly, not mixed with zeros
            self.prototype_ema.copy_(new_prototypes)
            self.prototype_init.fill_(True)
            return

        # EMA: for classes with data, mix; for missing classes, keep old
        mask_f = class_active_mask.float().unsqueeze(-1)
        mixed = (
            PROTOTYPE_EMA_BETA * self.prototype_ema
            + (1.0 - PROTOTYPE_EMA_BETA) * new_prototypes
        )
        # Only update active classes
        self.prototype_ema.copy_(mask_f * mixed + (1.0 - mask_f) * self.prototype_ema)

    def get_intra_class_std(
        self,
        z_sem: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """KL-collapse diagnostic: return mean per-class std of z_sem.

        If this is very small (< 0.1), prototype lookup degeneracy likely.
        """
        with torch.no_grad():
            stds = []
            for c in range(self.num_classes):
                mask = y == c
                if int(mask.sum().item()) < 2:
                    continue
                cls_std = z_sem[mask].std(dim=0).mean()
                stds.append(cls_std)
            if not stds:
                return torch.tensor(0.0, device=z_sem.device)
            return torch.stack(stds).mean()


def kl_gaussian_closed_form(
    mu_q: torch.Tensor,
    log_var_q: torch.Tensor,
    mu_p: torch.Tensor,
    log_var_p: torch.Tensor,
) -> torch.Tensor:
    """Closed-form KL(N(mu_q, var_q) || N(mu_p, var_p)).

    All tensors are [..., D]. Returns scalar (mean over batch, sum over dim).

    KL = 0.5 * sum_d [ log(var_p/var_q) + (var_q + (mu_q - mu_p)^2) / var_p - 1 ]
    """
    assert mu_q.shape == log_var_q.shape == mu_p.shape == log_var_p.shape
    var_q = torch.exp(log_var_q)
    var_p = torch.exp(log_var_p)
    kl = 0.5 * (
        (log_var_p - log_var_q)
        + (var_q + (mu_q - mu_p).pow(2)) / var_p
        - 1.0
    )
    return kl.sum(dim=-1).mean()


def lambda_ib_schedule(
    current_round: int,
    warmup_start: int = 20,
    warmup_end: int = 50,
) -> float:
    """Warmup schedule for lambda_IB: 0 before start, linear ramp, 1.0 after end."""
    if current_round < warmup_start:
        return 0.0
    if current_round < warmup_end:
        return float(current_round - warmup_start) / max(warmup_end - warmup_start, 1)
    return 1.0
