# -*- coding: utf-8 -*-
"""FedDSA-PCH (Per-Cell Hardness)

基于 EXP-123 Stage B 诊断 (2026-04-24 ANALYSIS.md) 发现:
  FDSE 的 +2.31pp 优势里 ~90% 来自 3 个 hard cells:
    (Art, guitar) +24.33, (Art, horse) +15.52, (Photo, horse) +17.50

本方法: 在 orth_only (feddsa_scheduled sm=0) 基础上, 对 **hard (client, class) cells**
的样本做 CE loss weight multiplier (w = hw), 其他保持 w=1.

hard cells 根据 FedBN 诊断 hardcode (非 adaptive), 作为 "perfect knowledge" pilot —
如果 hardcoded hw>1 都无法涨 accuracy, 说明 per-cell hardness 方向本身不成立.

Hard cells (from diagnostic):
  Client 0 (Art):   {guitar=3, horse=4, person=6}
  Client 2 (Photo): {horse=4}

Hyperparams (via config yml top-level keys, NOT algo_para):
  hw: float (default 2.0) — hard weight multiplier

All other algo_para inherit from feddsa_scheduled (sm=0 for orth_only base).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.feddsa_scheduled import (
    Server as BaseServer,
    Client as BaseClient,
    FedDSAModel,
    AlexNetEncoder,
    init_global_module,
    init_local_module,
    init_dataset,
    model_map,
)


# -----------------------------------------------------------------------------
# Hardcoded hard cells (EXP-123 Stage B FedBN diagnostic, 2026-04-24)
# -----------------------------------------------------------------------------
# PACS_c4 domains: (art_painting, cartoon, photo, sketch) -> client id 0..3
# PACS classes:   (dog=0, elephant=1, giraffe=2, guitar=3, horse=4, house=5, person=6)
HARD_CELLS = {
    0: frozenset({3, 4, 6}),  # Art: guitar, horse, person
    2: frozenset({4}),         # Photo: horse
}


class Server(BaseServer):
    def initialize(self):
        super().initialize()
        # Read hw from config yml top-level (not in algo_para list for compatibility)
        hw = float(self.option.get('hw', 2.0))
        self.hw = hw
        for c in self.clients:
            c.hw = hw


class Client(BaseClient):
    def initialize(self):
        super().initialize()
        # hw set by Server after clients created; set a safe default in case
        if not hasattr(self, 'hw'):
            self.hw = 2.0
        # Wrap self.loss_fn with per-cell hard weight
        self._base_loss_fn = self.loss_fn  # keep original nn.CrossEntropyLoss
        self.loss_fn = self._make_weighted_ce()

    def _make_weighted_ce(self):
        """Return a callable that computes per-sample weighted CE.

        Captures self via closure. client_id resolves at call time (lazy)
        so works with flgo's late set_id() mechanism.
        """
        client_ref = self

        def weighted_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            cid_raw = client_ref.id
            # flgo may set id as int or leave None
            try:
                cid = int(cid_raw) if cid_raw is not None else -1
            except (TypeError, ValueError):
                cid = -1
            hard = HARD_CELLS.get(cid, None)
            if not hard:
                return client_ref._base_loss_fn(logits, labels)
            # Per-sample weight
            w = torch.ones(labels.size(0), dtype=logits.dtype, device=logits.device)
            hw_val = float(getattr(client_ref, 'hw', 2.0))
            for c in hard:
                w[labels == c] = hw_val
            loss_unred = F.cross_entropy(logits, labels, reduction='none')
            return (loss_unred * w).mean()

        return weighted_ce
