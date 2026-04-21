"""Supervised Contrastive Loss (Khosla et al. NeurIPS 2020).

Multi-positive contrastive: in a batch, all same-class samples are positives.
Fallback: if a class has only 1 sample in the batch, that sample is skipped.
"""

import torch
import torch.nn.functional as F


def supcon_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute SupCon loss.

    z: [B, D] embedding (NOT required L2-normalized; we normalize internally)
    y: [B] class labels (long tensor)
    temperature: scaling temperature
    Returns: scalar loss. If no valid sample has positives, returns 0.

    Reference: Khosla et al. "Supervised Contrastive Learning", NeurIPS 2020.
    """
    assert z.dim() == 2
    assert y.dim() == 1
    assert z.size(0) == y.size(0)

    B = z.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z.device, requires_grad=z.requires_grad)

    # L2 normalize
    z = F.normalize(z, dim=-1)

    # Similarity matrix [B, B]
    sim = z @ z.T / temperature

    # Numerical stability: subtract per-row max (no affect due to log-sum-exp shift-invariance)
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # Build positive mask [B, B]: same class, exclude self-pair
    y = y.view(-1, 1)
    pos_mask = (y == y.T).float()
    # Remove diagonal
    diag_mask = torch.eye(B, device=z.device)
    pos_mask = pos_mask - diag_mask
    pos_mask = pos_mask.clamp(min=0.0)

    # Denominator mask: all except self
    denom_mask = 1.0 - diag_mask

    # log prob
    exp_sim = torch.exp(sim) * denom_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + eps)

    # Mean over positives per anchor, then mean over anchors with at least 1 positive
    n_pos_per_anchor = pos_mask.sum(dim=1)  # [B]
    # Avoid div-by-zero: compute mean_log_prob = sum(pos * log_prob) / n_pos
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / n_pos_per_anchor.clamp(min=1)

    # Skip anchors with zero positives
    valid_mask = (n_pos_per_anchor > 0).float()
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=z.requires_grad)

    loss = -(valid_mask * mean_log_prob_pos).sum() / valid_mask.sum().clamp(min=1)
    return loss


def supcon_diagnostics(z: torch.Tensor, y: torch.Tensor) -> dict:
    """Return diagnostic metrics for SupCon training.

    pos_sim_mean: average cosine of same-class pairs
    neg_sim_mean: average cosine of different-class pairs
    alignment: E[|| z_i - z_j ||^2] for same class (Wang-Isola 2020)
    uniformity: log E[exp(-2 * || z_i - z_j ||^2)] for all pairs (Wang-Isola 2020)
    n_positive_avg: mean number of positives per anchor
    """
    with torch.no_grad():
        B = z.size(0)
        if B < 2:
            return {
                'pos_sim_mean': 0.0, 'neg_sim_mean': 0.0,
                'alignment': 0.0, 'uniformity': 0.0,
                'n_positive_avg': 0.0,
            }

        z_n = F.normalize(z, dim=-1)
        cos = z_n @ z_n.T
        dist2 = 2.0 - 2.0 * cos.clamp(min=-1.0, max=1.0)  # ||z_i - z_j||^2 for unit vec

        y = y.view(-1, 1)
        pos_mask = (y == y.T).float()
        diag = torch.eye(B, device=z.device)
        pos_mask = (pos_mask - diag).clamp(min=0.0)
        neg_mask = 1.0 - pos_mask - diag

        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        pos_sim_mean = (cos * pos_mask).sum() / n_pos.clamp(min=1)
        neg_sim_mean = (cos * neg_mask).sum() / n_neg.clamp(min=1)

        # Alignment: same-class expected distance
        alignment = (dist2 * pos_mask).sum() / n_pos.clamp(min=1)

        # Uniformity: log E[exp(-2 ||z_i - z_j||^2)] on all non-diagonal pairs
        off_diag = 1.0 - diag
        exp_term = torch.exp(-2.0 * dist2) * off_diag
        uniformity_raw = exp_term.sum() / off_diag.sum().clamp(min=1)
        uniformity = torch.log(uniformity_raw.clamp(min=1e-10))

        n_positive_avg = n_pos / B

        return {
            'pos_sim_mean': pos_sim_mean.item(),
            'neg_sim_mean': neg_sim_mean.item(),
            'alignment': alignment.item(),
            'uniformity': uniformity.item(),
            'n_positive_avg': n_positive_avg.item(),
        }
