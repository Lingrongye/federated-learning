# -*- coding: utf-8 -*-
"""Unit tests for feddsa_pch weighted CE logic."""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from algorithm.feddsa_pch import HARD_CELLS


def compute_weighted_ce(logits, labels, client_id, hw):
    """Replicate the weighted_ce logic (without real Client instance)."""
    hard = HARD_CELLS.get(client_id, None)
    if not hard:
        return F.cross_entropy(logits, labels)
    w = torch.ones(labels.size(0), dtype=logits.dtype, device=logits.device)
    for c in hard:
        w[labels == c] = hw
    loss_unred = F.cross_entropy(logits, labels, reduction='none')
    return (loss_unred * w).mean()


def test_no_hard_cell_client_matches_plain_ce():
    """Client 1 (Cartoon) and 3 (Sketch) have no hard cells."""
    logits = torch.randn(8, 7, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0])
    plain = F.cross_entropy(logits, labels)
    for cid in [1, 3, -1]:
        weighted = compute_weighted_ce(logits, labels, cid, 2.0)
        assert torch.allclose(plain, weighted), f"cid={cid}: {plain.item()} vs {weighted.item()}"
    print('PASS: no-hard-cell clients match plain CE')


def test_art_hard_cells_apply_weight():
    """Client 0 (Art): hard = {guitar=3, horse=4, person=6}.

    If all labels are hard (all 3), weighted loss = hw * plain_loss.
    """
    logits = torch.randn(6, 7)
    labels = torch.tensor([3, 3, 4, 4, 6, 6])  # all hard
    hw = 2.0
    plain = F.cross_entropy(logits, labels)
    weighted = compute_weighted_ce(logits, labels, 0, hw)
    assert torch.allclose(weighted, hw * plain, atol=1e-5), \
        f"all-hard: plain={plain.item()}, weighted={weighted.item()}, expected={hw * plain.item()}"
    print('PASS: Art all-hard labels give hw * plain')


def test_mixed_hard_easy():
    """Client 0 (Art): half hard, half easy.

    Manual: loss = (sum of all plain_per_sample * weight) / n
         = (n_easy * plain_easy_mean + n_hard * hw * plain_hard_mean) / n
    """
    logits = torch.randn(8, 7)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0])
    # Hard for client 0: 3, 4, 6. So 3 hard (idx 3,4,6), 5 easy (idx 0,1,2,5,7)
    per_sample = F.cross_entropy(logits, labels, reduction='none')
    hw = 3.0
    hard_mask = torch.tensor([False, False, False, True, True, False, True, False])
    expected = (per_sample[~hard_mask].sum() + hw * per_sample[hard_mask].sum()) / 8

    weighted = compute_weighted_ce(logits, labels, 0, hw)
    assert torch.allclose(weighted, expected, atol=1e-5), \
        f"mixed: got {weighted.item()}, expected {expected.item()}"
    print('PASS: mixed hard/easy weighted correctly')


def test_photo_only_horse_hard():
    """Client 2 (Photo): only horse=4 is hard."""
    logits = torch.randn(4, 7)
    labels = torch.tensor([3, 4, 4, 6])  # guitar (easy for Photo), 2 horse (hard), person (easy for Photo)
    hw = 2.0
    per_sample = F.cross_entropy(logits, labels, reduction='none')
    # Only label=4 entries get weight
    mask = labels == 4
    expected = (per_sample[~mask].sum() + hw * per_sample[mask].sum()) / 4

    weighted = compute_weighted_ce(logits, labels, 2, hw)
    assert torch.allclose(weighted, expected, atol=1e-5)
    print('PASS: Photo horse-only weight')


def test_hw_1_is_identity():
    """hw=1.0 should match plain CE for any client."""
    logits = torch.randn(5, 7)
    labels = torch.tensor([3, 4, 6, 4, 3])
    plain = F.cross_entropy(logits, labels)
    for cid in [0, 2]:
        weighted = compute_weighted_ce(logits, labels, cid, 1.0)
        assert torch.allclose(weighted, plain, atol=1e-5)
    print('PASS: hw=1.0 is identity')


def test_hard_cells_constants():
    """Verify HARD_CELLS constants match design."""
    assert HARD_CELLS[0] == {3, 4, 6}, "Art hard cells must be {guitar, horse, person}"
    assert HARD_CELLS[2] == {4}, "Photo hard cells must be {horse}"
    assert 1 not in HARD_CELLS, "Cartoon should have no hard cells"
    assert 3 not in HARD_CELLS, "Sketch should have no hard cells"
    print('PASS: HARD_CELLS constants correct')


def test_gradient_flows():
    """Weighted CE must be differentiable."""
    logits = torch.randn(6, 7, requires_grad=True)
    labels = torch.tensor([3, 4, 6, 0, 1, 2])
    loss = compute_weighted_ce(logits, labels, 0, 2.0)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0, 'no gradient on logits'
    print('PASS: gradient flows through weighted CE')


if __name__ == '__main__':
    test_no_hard_cell_client_matches_plain_ce()
    test_art_hard_cells_apply_weight()
    test_mixed_hard_easy()
    test_photo_only_horse_hard()
    test_hw_1_is_identity()
    test_hard_cells_constants()
    test_gradient_flows()
    print('\nALL 7 tests PASSED')
