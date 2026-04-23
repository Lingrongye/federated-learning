# -*- coding: utf-8 -*-
"""Per-class + confidence diagnostic eval (algorithm-agnostic).

Plugs into any flgo classification algo (FedBN/FDSE/FedDSA/etc.). Given a
model and a test Dataset, runs ONE forward pass and extracts:

  - per-class accuracy  (num_classes floats)
  - confidence stats    (mean/std/p10/p50/p90/ECE_10bin/over_conf_err_ratio)
  - per-class mean confidence (num_classes floats)
  - confidence histogram (correct vs wrong, 20 bins)

Assumptions:
  - model(x) returns logits of shape [B, num_classes]
  - dataset yields (x, y) tuples with integer labels in [0, num_classes)
  - ECE uses 10 equal-width bins over [0, 1]

Caller is expected to run under torch.no_grad() scope; this function also
forces eval() mode (and restores training state on exit).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _ece_10bin(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with n equal-width bins over [0, 1]."""
    if confidences.size == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = confidences.size
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


@torch.no_grad()
def run_diagnostic(
    model: torch.nn.Module,
    dataset,
    device,
    num_classes: int,
    batch_size: int = 64,
    num_workers: int = 0,
    collate_fn=None,
    include_histogram: bool = False,
    hist_bins: int = 20,
) -> Dict[str, object]:
    """Run one eval pass and return diagnostic dict.

    Returned keys:
      per_class_acc: List[float] len=num_classes (NaN if class unseen)
      per_class_conf: List[float] len=num_classes (mean max-softmax per true class)
      per_class_support: List[int] len=num_classes
      conf_mean, conf_std, conf_p10, conf_p50, conf_p90: float
      ece: float
      over_conf_err_ratio: float (fraction of WRONG preds with conf > 0.8, over total samples)
      wrong_conf_mean: float (mean conf of wrong preds; NaN if no wrong)
      overall_acc: float
      (if include_histogram) hist_correct, hist_wrong: List[int] len=hist_bins
    """
    was_training = model.training
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
    )

    all_confidences: List[np.ndarray] = []
    all_correct: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    for batch in loader:
        # Expect (x, y) — aligned with config.data_to_device
        x, y = batch[0], batch[-1]
        x = x.to(device)
        y = y.to(device)
        if x.dtype == torch.uint8 or x.dtype == torch.int8:
            x = x / 255.0
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        correct = pred.eq(y).long()

        all_confidences.append(conf.detach().cpu().numpy())
        all_correct.append(correct.detach().cpu().numpy().astype(np.int64))
        all_labels.append(y.detach().cpu().numpy().astype(np.int64))
        all_preds.append(pred.detach().cpu().numpy().astype(np.int64))

    if was_training:
        model.train()

    if len(all_confidences) == 0:
        return {
            'per_class_acc': [float('nan')] * num_classes,
            'per_class_conf': [float('nan')] * num_classes,
            'per_class_support': [0] * num_classes,
            'conf_mean': float('nan'),
            'conf_std': float('nan'),
            'conf_p10': float('nan'),
            'conf_p50': float('nan'),
            'conf_p90': float('nan'),
            'ece': float('nan'),
            'over_conf_err_ratio': float('nan'),
            'wrong_conf_mean': float('nan'),
            'overall_acc': float('nan'),
        }

    conf = np.concatenate(all_confidences)
    correct = np.concatenate(all_correct)
    labels = np.concatenate(all_labels)

    # per-class
    per_class_acc: List[float] = []
    per_class_conf: List[float] = []
    per_class_support: List[int] = []
    for c in range(num_classes):
        m = labels == c
        support = int(m.sum())
        per_class_support.append(support)
        if support == 0:
            per_class_acc.append(float('nan'))
            per_class_conf.append(float('nan'))
        else:
            per_class_acc.append(float(correct[m].mean()))
            per_class_conf.append(float(conf[m].mean()))

    # confidence stats
    conf_mean = float(conf.mean())
    conf_std = float(conf.std())
    conf_p10 = float(np.percentile(conf, 10))
    conf_p50 = float(np.percentile(conf, 50))
    conf_p90 = float(np.percentile(conf, 90))

    ece = _ece_10bin(conf, correct.astype(np.float64))

    wrong_mask = correct == 0
    n_total = conf.size
    over_conf_err = int(((conf > 0.8) & wrong_mask).sum())
    over_conf_err_ratio = float(over_conf_err / n_total) if n_total > 0 else 0.0
    wrong_conf_mean = float(conf[wrong_mask].mean()) if wrong_mask.sum() > 0 else float('nan')
    overall_acc = float(correct.mean())

    out: Dict[str, object] = {
        'per_class_acc': per_class_acc,
        'per_class_conf': per_class_conf,
        'per_class_support': per_class_support,
        'conf_mean': conf_mean,
        'conf_std': conf_std,
        'conf_p10': conf_p10,
        'conf_p50': conf_p50,
        'conf_p90': conf_p90,
        'ece': ece,
        'over_conf_err_ratio': over_conf_err_ratio,
        'wrong_conf_mean': wrong_conf_mean,
        'overall_acc': overall_acc,
    }

    if include_histogram:
        bin_edges = np.linspace(0.0, 1.0, hist_bins + 1)
        hist_correct, _ = np.histogram(conf[correct == 1], bins=bin_edges)
        hist_wrong, _ = np.histogram(conf[correct == 0], bins=bin_edges)
        out['hist_correct'] = hist_correct.astype(int).tolist()
        out['hist_wrong'] = hist_wrong.astype(int).tolist()
        out['hist_bins'] = hist_bins

    return out
