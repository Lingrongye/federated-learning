"""Extended diagnostics for FedDSA-VIB: KL collapse, domain-conditional rate, IRM-style grad var.

Plugs into SGPADiagnosticLogger (existing in FDSE_CVPR25/diagnostics.py).
These functions are pure / callable from client.train() after backward.
"""

from typing import Dict, List

import torch


def kl_collapse_detect(
    kl_mean: float,
    intra_class_z_std: float,
    kl_thresh: float = 0.1,
    std_thresh: float = 0.1,
) -> bool:
    """Return True if VIB likely collapsed to prototype lookup.

    Collapse signal: simultaneously
    - KL very low (q ~= prior, so mu ~= prototype)
    - intra-class z_sem std very low (all samples of class c collapse to one point)
    """
    return kl_mean < kl_thresh and intra_class_z_std < std_thresh


def domain_conditional_rate(
    kl_per_sample: torch.Tensor,
    domain_labels: torch.Tensor,
    num_domains: int,
) -> Dict[str, float]:
    """Compute per-domain mean KL and std across domains.

    kl_per_sample: [B] KL contribution per sample (already summed over dim)
    domain_labels: [B] int, [0, num_domains)
    Returns: {'R_d_0': ..., 'R_d_1': ..., ..., 'R_std_across_domains': ...}
    """
    result = {}
    with torch.no_grad():
        per_domain_mean = []
        for d in range(num_domains):
            mask = domain_labels == d
            if int(mask.sum().item()) == 0:
                result[f'R_d_{d}'] = float('nan')
                continue
            r_d = kl_per_sample[mask].mean().item()
            result[f'R_d_{d}'] = r_d
            per_domain_mean.append(r_d)
        if len(per_domain_mean) >= 2:
            result['R_std_across_domains'] = float(
                torch.tensor(per_domain_mean).std().item()
            )
        else:
            result['R_std_across_domains'] = 0.0
    return result


def irm_gradient_variance(
    loss_per_sample: torch.Tensor,
    params: List[torch.nn.Parameter],
    domain_labels: torch.Tensor,
    num_domains: int,
) -> float:
    """Compute IRM-style cross-domain gradient variance.

    For each domain, compute gradient of mean(L_d) w.r.t. params.
    Then return variance of flattened gradient vector across domains.

    Lower variance => more domain-invariant representation.

    loss_per_sample: [B] per-sample task loss (CE not reduced)
    params: list of parameters to compute gradient on
    domain_labels: [B] int
    Returns: scalar gradient variance (float)

    Note: uses create_graph=False, retain_graph=True so caller's backward still works.
    """
    # Early exit: if fewer than 2 domains have data, no variance to compute
    with torch.no_grad():
        unique_d = torch.unique(domain_labels)
    if int(unique_d.numel()) < 2:
        return 0.0

    grads_per_domain = []
    for d in range(num_domains):
        mask = domain_labels == d
        if int(mask.sum().item()) == 0:
            continue
        loss_d = loss_per_sample[mask].mean()
        grad_list = torch.autograd.grad(
            loss_d, params, retain_graph=True, create_graph=False, allow_unused=True
        )
        flat = torch.cat(
            [g.flatten() if g is not None else torch.zeros_like(p).flatten()
             for g, p in zip(grad_list, params)]
        )
        grads_per_domain.append(flat.detach())
    if len(grads_per_domain) < 2:
        return 0.0
    stacked = torch.stack(grads_per_domain)
    return stacked.var(dim=0).mean().item()
