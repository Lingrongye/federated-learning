# -*- coding: utf-8 -*-
"""Unit tests for diagnostics.per_class_eval.run_diagnostic."""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from diagnostics.per_class_eval import run_diagnostic, _ece_10bin


class ToyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class IdentityModel(torch.nn.Module):
    """Model whose output is a learnable-bias-free linear map.

    We set weights so the output exactly equals an input-provided logit
    vector (see the test synthesis below)."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # x is already [B, C] logits
        return x


def test_perfect_predictions_no_calibration_gap():
    """When model confidence matches accuracy exactly, ECE should be ~0."""
    # 4 classes, 8 samples all correct with confidence 0.9
    # Build logits so softmax max = 0.9 for the correct class
    C = 4
    B = 8
    # logits[i, y] = log(0.9 * (C-1) / 0.1) so softmax == 0.9 on y_true
    logits = torch.full((B, C), -10.0)  # suppress others
    labels = torch.arange(B) % C
    for i in range(B):
        logits[i, labels[i]] = 0.0  # other logits are -10, softmax(0) ≈ 1.0 on true class
    # Now softmax for a vector [0, -10, -10, -10] is [~1.0, ~0, ~0, ~0]
    # That gives confidence ~1.0 everywhere and correct=1 everywhere → ECE = |1-1| = 0
    ds = ToyDataset(logits, labels)
    model = IdentityModel(num_classes=C)
    res = run_diagnostic(model, ds, device='cpu', num_classes=C, batch_size=4)

    assert res['overall_acc'] == 1.0, f"expected overall_acc=1 got {res['overall_acc']}"
    assert res['ece'] < 0.01, f"expected tiny ECE, got {res['ece']}"
    # every class should have acc 1.0
    for c in range(C):
        assert res['per_class_acc'][c] == 1.0


def test_per_class_acc_correctness():
    """Class 0: all correct. Class 1: half correct.

    7 classes, 10 samples:
      labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
      preds  = [0, 0, 0, 0, 0, 1, 1, 0, 1, 1]  # class 1: 4/6 correct
    """
    C = 7
    B = 10
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    preds  = torch.tensor([0, 0, 0, 0, 0, 1, 1, 0, 1, 1])

    logits = torch.full((B, C), -10.0)
    for i in range(B):
        logits[i, preds[i]] = 0.0

    ds = ToyDataset(logits, labels)
    model = IdentityModel(num_classes=C)
    res = run_diagnostic(model, ds, device='cpu', num_classes=C, batch_size=5)

    assert res['per_class_acc'][0] == 1.0
    assert abs(res['per_class_acc'][1] - 4/6) < 1e-6, f"got {res['per_class_acc'][1]}"
    # classes 2..6 should be NaN because support=0
    for c in range(2, 7):
        import math
        assert math.isnan(res['per_class_acc'][c])
        assert res['per_class_support'][c] == 0


def test_over_confident_wrong_ratio():
    """3 wrong preds with confidence 0.95, 5 correct with 0.95.

    over_conf_err_ratio = 3/8 = 0.375 (all wrong are > 0.8)
    """
    C = 3
    B = 8
    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
    preds  = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])  # last 3 wrong

    # Make confidence ~0.95 for all
    logits = torch.full((B, C), 0.0)
    for i in range(B):
        logits[i, preds[i]] = 3.0  # softmax max ~0.9

    ds = ToyDataset(logits, labels)
    model = IdentityModel(num_classes=C)
    res = run_diagnostic(model, ds, device='cpu', num_classes=C, batch_size=4)

    # Overall acc = 5/8
    assert abs(res['overall_acc'] - 5/8) < 1e-6
    # All 3 wrong preds have conf > 0.8 (softmax of [3,0,0] ≈ 0.88 — check)
    conf_val = torch.softmax(torch.tensor([3.0, 0.0, 0.0]), dim=0)[0].item()
    if conf_val > 0.8:
        expected = 3 / 8
    else:
        expected = 0 / 8
    assert abs(res['over_conf_err_ratio'] - expected) < 1e-6, \
        f"got {res['over_conf_err_ratio']}, expected {expected} (softmax max={conf_val:.4f})"


def test_histogram_included():
    """When include_histogram=True, hist_correct/hist_wrong present."""
    C = 3
    B = 6
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    preds  = torch.tensor([0, 1, 2, 1, 0, 1])

    logits = torch.full((B, C), 0.0)
    for i in range(B):
        logits[i, preds[i]] = 3.0

    ds = ToyDataset(logits, labels)
    model = IdentityModel(num_classes=C)
    res = run_diagnostic(model, ds, device='cpu', num_classes=C, batch_size=3,
                        include_histogram=True, hist_bins=10)

    assert 'hist_correct' in res and 'hist_wrong' in res
    assert len(res['hist_correct']) == 10
    assert len(res['hist_wrong']) == 10
    assert sum(res['hist_correct']) + sum(res['hist_wrong']) == B


def test_empty_class_handling():
    """All samples from class 0 — class 1..C-1 should have NaN acc."""
    import math
    C = 4
    B = 5
    labels = torch.zeros(B, dtype=torch.long)
    logits = torch.zeros(B, C)
    logits[:, 0] = 3.0  # all predict class 0

    ds = ToyDataset(logits, labels)
    model = IdentityModel(num_classes=C)
    res = run_diagnostic(model, ds, device='cpu', num_classes=C, batch_size=2)

    assert res['per_class_acc'][0] == 1.0
    for c in range(1, C):
        assert math.isnan(res['per_class_acc'][c])
        assert res['per_class_support'][c] == 0


def test_ece_function():
    """Direct ECE test with known bin behavior."""
    # Half with conf 0.9, half wrong | gap 0.9 - 0 = 0.9 for first half
    # Half with conf 0.9 all correct | gap 0 for second half
    conf = np.array([0.9] * 10)
    correct = np.array([0] * 5 + [1] * 5)  # acc in that bin is 0.5
    # All 10 samples go into bin [0.8, 0.9) or [0.9, 1.0)
    # Since conf exactly 0.9, they go into [0.9, 1.0) bin
    # bin conf mean = 0.9, bin acc = 0.5 → ECE = |0.9 - 0.5| = 0.4
    ece = _ece_10bin(conf, correct)
    assert abs(ece - 0.4) < 1e-6, f"got {ece}"


def test_eval_mode_restored():
    """Model should return to training mode if it started training."""
    C = 3
    B = 4
    labels = torch.zeros(B, dtype=torch.long)
    logits = torch.zeros(B, C)
    ds = ToyDataset(logits, labels)
    model = IdentityModel(num_classes=C)
    model.train()
    run_diagnostic(model, ds, device='cpu', num_classes=C)
    assert model.training


if __name__ == '__main__':
    test_perfect_predictions_no_calibration_gap()
    print('PASS: perfect predictions')
    test_per_class_acc_correctness()
    print('PASS: per-class acc')
    test_over_confident_wrong_ratio()
    print('PASS: over-confident wrong ratio')
    test_histogram_included()
    print('PASS: histogram')
    test_empty_class_handling()
    print('PASS: empty class')
    test_ece_function()
    print('PASS: ECE function')
    test_eval_mode_restored()
    print('PASS: eval mode restored')
    print('\nALL 7 tests PASSED')
