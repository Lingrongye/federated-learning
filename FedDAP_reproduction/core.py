"""FedDAP equations 3--10, isolated from the broken upstream import registry.

Paper: arXiv:2604.06795v1; upstream: 988211eb826d81251bd8b3b8ae53fbdcc7f3dc20.
No LAB, FedBN, privacy noise, projection head, or personalized evaluation.
"""
from collections import defaultdict

import torch
from torch.nn import functional as F


def alignment_losses(features, labels, domain, prototypes, temperature):
    """Batch-mean DPA (upstream convention) and Eq. 8 CPCL.

    Missing anchors contribute zero, with the full batch as denominator.
    All prototype references are detached; only current features receive grads.
    Eq. 6 does not unambiguously specify batch normalization; see PROTOCOL.md.
    """
    if temperature <= 0:
        raise ValueError("CPCL temperature must be positive")
    zero = features.sum() * 0
    dpa, cpcl = zero, zero
    valid_dpa = valid_cpcl = 0
    for feature, label in zip(features, labels):
        cls = int(label)
        anchor = prototypes.get((cls, domain))
        if anchor is not None:
            dpa = dpa + 1 - F.cosine_similarity(
                feature[None], anchor.detach().to(feature)[None], dim=1
            ).squeeze(0)
            valid_dpa += 1
        cross = [(key, value) for key, value in sorted(prototypes.items())
                 if key[1] != domain]
        positive = [key[0] == cls for key, _ in cross]
        if not any(positive) or all(positive):
            continue
        anchors = torch.stack([value.detach().to(feature) for _, value in cross])
        logits = F.cosine_similarity(feature[None], anchors, dim=1) / temperature
        mask = torch.tensor(positive, dtype=torch.bool, device=feature.device)
        # Exact log of ratio of sums in Eq. 8, without exp overflow.
        cpcl = cpcl + torch.logsumexp(logits, 0) - torch.logsumexp(logits[mask], 0)
        valid_cpcl += 1
    return dpa / len(labels), cpcl / len(labels), valid_dpa, valid_cpcl


@torch.no_grad()
def attention_aggregate(local_prototypes, temperature):
    """Eq. 4 excludes self-similarity; Eq. 5 fuses same (class, domain) only."""
    if temperature <= 0:
        raise ValueError("Aggregation temperature must be positive")
    groups = defaultdict(list)
    for client, prototypes in enumerate(local_prototypes):
        for key, value in prototypes.items():
            groups[key].append((client, value.detach()))
    result, diagnostics = {}, {}
    for key, entries in sorted(groups.items()):
        stacked = torch.stack([value for _, value in entries])
        similarities = F.cosine_similarity(
            stacked[:, None, :], stacked[None, :, :], dim=-1
        )
        similarities.fill_diagonal_(0)
        weights = torch.softmax(similarities.sum(1) / temperature, 0)
        result[key] = (weights[:, None] * stacked).sum(0).detach()
        diagnostics[key] = {
            "clients": [client for client, _ in entries],
            "weights": weights.cpu().tolist(),
            "scores": similarities.sum(1).cpu().tolist(),
        }
    return result, diagnostics


@torch.no_grad()
def extract_prototypes(net, loader, domain, device):
    """Eq. 3 recomputed on ALL assigned samples using the FINAL local encoder.

    Deterministic eval transform and eval BN are explicit implementation choices
    where the paper is silent. Restore previous training mode on exit.
    """
    previous = net.training
    net.eval()
    sums, counts = {}, defaultdict(int)
    try:
        for images, labels in loader:
            features = net.features(images.to(device))
            for cls in labels.unique().tolist():
                mask = labels.to(device) == cls
                value = features[mask].sum(0).detach().cpu()
                key = (int(cls), domain)
                sums[key] = sums.get(key, torch.zeros_like(value)) + value
                counts[key] += int(mask.sum())
    finally:
        net.train(previous)
    return {key: value / counts[key] for key, value in sums.items()}, dict(counts)


@torch.no_grad()
def aggregate_states(states, sample_counts):
    """Sample-weighted FedAvg, including floating BN buffers (not FedBN).

    Integer buffers are counters, not parameters; take max for num_batches_tracked
    rather than silently truncating a floating weighted average.
    """
    if not states or len(states) != len(sample_counts) or min(sample_counts) <= 0:
        raise ValueError("Require one nonempty sample count per client")
    weights = torch.tensor(sample_counts, dtype=torch.float64)
    weights /= weights.sum()
    result = {}
    for key, first in states[0].items():
        if first.is_floating_point() or first.is_complex():
            result[key] = sum(state[key] * float(weight)
                              for state, weight in zip(states, weights))
        else:
            result[key] = torch.stack([state[key] for state in states]).amax(0)
    return result, weights.tolist()
