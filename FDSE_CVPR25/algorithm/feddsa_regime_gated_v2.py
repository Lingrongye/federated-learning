"""
FedDSA-RegimeGated v2: Fix regime signal source.

Motivation (from EXP-067 v1 findings, 2026-04-12):
    In v1, the style bank stores stats of `h = encoder(x)` (backbone features).
    Empirical measurement shows r is NOT discriminative between PACS and Office:
        PACS r ≈ 2.0-2.4
        Office r ≈ 2.1-2.2
    Reason: the AlexNet encoder has multiple BN layers; after the final bn7,
    cross-client activation statistics are roughly normalized to similar ranges,
    erasing the "raw style" difference we wanted to detect.

Fix A (signal source):
    Add a SECOND style bank `style_bank_proj` that stores stats of the
    style_head OUTPUT `z_sty = style_head(h)` instead of `h`. The style head
    is trained with orthogonal+HSIC constraints to capture what the backbone
    "discards" — i.e. style-specific residual — which should be more
    discriminative across regimes.

    - `style_bank` (1024-d, h stats) → kept as-is for AdaIN augmentation
    - `style_bank_proj` (128-d, z_sty stats) → new, used for regime signal
      and style graph dispatch distance

Fix B (dispatch direction, from EXP-067 v1 failure analysis):
    v1 used KNN (nearest neighbor) dispatch for ALL clients → HURT PACS by
    -1.28 vs Consensus because high-gap clients (sketch, art) need DIVERSE
    augmentation from FAR styles, not similar ones.

    v2 uses FARTHEST-K dispatch: always pick the K most different styles.
    - PACS: farthest neighbors are the most different domains → maximizes
      augmentation diversity (reverses v1's -1.28 penalty)
    - Office: all clients are similar → farthest ≈ nearest → harmless

    Augmentation continues to use the existing `style_bank` (unchanged), so
    AdaIN behavior is preserved. Only the dispatch selection and regime
    decision now use the projection bank.

Clean inheritance structure:
    Client: override train() to call super() then do ONE extra eval pass over
    training data to compute style_head projection statistics. Override pack()
    to include the new field.

    Server: override initialize() to add style_bank_proj. Override iterate()
    to collect it. Override _compute_regime_score() and _dispatch_knn_styles()
    to use the new bank.

Keeps FedDSA identity:
    - Dual-head decouple + orth loss ✓
    - Global style bank + AdaIN augmentation ✓ (unchanged)
    - InfoNCE prototype alignment ✓
    - style_head private ✓
    - New: regime signal and KNN dispatch use projection-space statistics
"""
import os
import sys
import copy
import numpy as np
import torch

from algorithm.feddsa_regime_gated import Server as RGServer, Client as RGClient


class Server(RGServer):
    def initialize(self):
        super().initialize()
        # Second bank for projection-space (style_head output) stats
        self.style_bank_proj = {}

    def iterate(self):
        """Extend parent to also collect the projection stats."""
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)

        models = res['model']
        protos_list = res['protos']
        proto_counts_list = res['proto_counts']
        style_stats_list = res['style_stats']
        # New: may not be present if client is v1 — default to all None
        style_proj_stats_list = res.get('style_proj_stats',
                                         [None] * len(models))

        # 1. Aggregate shared parameters via consensus QP + regime-gated SAM
        self._aggregate_shared_consensus(models)

        # 2. Collect augmentation style bank (unchanged)
        for cid, style in zip(self.received_clients, style_stats_list):
            if style is not None:
                self.style_bank[cid] = style

        # 3. NEW: Collect projection style bank
        for cid, proj in zip(self.received_clients, style_proj_stats_list):
            if proj is not None:
                self.style_bank_proj[cid] = proj

        # 4. Aggregate prototypes (unchanged)
        self._aggregate_protos(protos_list, proto_counts_list)

    # ------------------------------------------------------------------
    # Override regime score + KNN to use projection bank
    # ------------------------------------------------------------------

    def _select_signal_bank(self):
        """Prefer style_bank_proj when available; fall back to style_bank."""
        if len(self.style_bank_proj) >= 2:
            return self.style_bank_proj
        return self.style_bank

    def _compute_regime_score(self):
        """Mean pairwise style distance from the SIGNAL bank.

        Uses style_bank_proj (style_head output stats) if available,
        otherwise falls back to the legacy backbone bank.
        """
        bank = self._select_signal_bank()
        if len(bank) < 2:
            return None

        cids = list(bank.keys())
        distances = []
        for i in range(len(cids)):
            mu_i, sigma_i = bank[cids[i]]
            for j in range(i + 1, len(cids)):
                mu_j, sigma_j = bank[cids[j]]
                mu_d = (mu_i - mu_j).norm().item()
                log_s_i = torch.log(sigma_i.clamp(min=1e-6))
                log_s_j = torch.log(sigma_j.clamp(min=1e-6))
                sig_d = (log_s_i - log_s_j).norm().item()
                distances.append(mu_d + sig_d)

        if len(distances) == 0:
            return None
        return float(np.mean(distances))

    def _dispatch_knn_styles(self, client_id, available, k):
        """Farthest-K style dispatch using the SIGNAL bank for distance,
        returning augmentation styles from the AUG bank.

        Key insight (from EXP-067 v1 failure analysis):
            - KNN (nearest) HURTS PACS by -1.28 vs Consensus because
              high-gap clients need DIVERSE augmentation from FAR styles.
            - KNN (nearest) is neutral on Office (all styles similar anyway).
            - Therefore: farthest-K is the correct default for style dispatch.
              It maximizes augmentation diversity on PACS while being harmless
              on Office (where all distances are small, so farthest ≈ nearest).

        Distance is computed on the projection space (style_head output),
        but dispatched items come from style_bank (1024-d backbone) for
        AdaIN compatibility.
        """
        signal_bank = self._select_signal_bank()
        my_signal = signal_bank.get(client_id)

        # Fall back to random if we can't compute distance
        if my_signal is None:
            keys = list(available.keys())
            chosen = np.random.choice(keys, k, replace=False)
            return [available[c] for c in chosen]

        mu_i, sigma_i = my_signal
        log_s_i = torch.log(sigma_i.clamp(min=1e-6))

        scored = []
        for cid in available.keys():
            if cid not in signal_bank:
                continue
            mu_j, sigma_j = signal_bank[cid]
            mu_d = (mu_i - mu_j).norm().item()
            log_s_j = torch.log(sigma_j.clamp(min=1e-6))
            sig_d = (log_s_i - log_s_j).norm().item()
            scored.append((mu_d + sig_d, cid))

        if len(scored) == 0:
            keys = list(available.keys())
            chosen = np.random.choice(keys, k, replace=False)
            return [available[c] for c in chosen]

        # FARTHEST K: descending sort by distance
        scored.sort(key=lambda t: -t[0])
        chosen_cids = [cid for _, cid in scored[:k]]
        return [available[c] for c in chosen_cids]


class Client(RGClient):
    """Override train() to add one extra eval pass computing projection stats,
    and override pack() to include them."""

    def train(self, model, *args, **kwargs):
        # Parent runs the full training loop and sets _local_style_stats (h)
        super().train(model, *args, **kwargs)

        # Extra eval pass: compute style_head output stats over training data
        self._local_style_proj_stats = self._compute_proj_style_stats(model)

    def _compute_proj_style_stats(self, model, num_batches=10):
        """Compute (mu, sigma) of style_head(encoder(x)) over a few batches."""
        model.eval()
        style_sum = None
        style_sq_sum = None
        n = 0

        try:
            with torch.no_grad():
                for _ in range(num_batches):
                    try:
                        batch_data = self.get_batch_data()
                    except Exception:
                        break
                    batch_data = self.calculator.to_device(batch_data)
                    x = batch_data[0]
                    h = model.encode(x)
                    z_sty = model.get_style(h)  # [B, proj_dim]
                    b = z_sty.size(0)
                    batch_mu = z_sty.mean(dim=0).cpu()
                    batch_sq = (z_sty ** 2).mean(dim=0).cpu()
                    if style_sum is None:
                        style_sum = batch_mu * b
                        style_sq_sum = batch_sq * b
                        n = b
                    else:
                        style_sum += batch_mu * b
                        style_sq_sum += batch_sq * b
                        n += b
        finally:
            model.train()

        if n > 1 and style_sum is not None:
            mu = style_sum / n
            var = style_sq_sum / n - mu ** 2
            return (mu, var.clamp(min=1e-6).sqrt())
        return None

    def pack(self):
        pkg = super().pack()
        pkg['style_proj_stats'] = getattr(
            self, '_local_style_proj_stats', None)
        return pkg


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    from algorithm.feddsa import init_global_module as base_init
    base_init(object)
