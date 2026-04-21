"""FedDSA-SGPA + VIB + SupCon — EXP-113 A/B/M6 方案统一实现.

Two orthogonal flags control 3 variants:
- vib = 0/1 : replace semantic_head with VIB stochastic version + add L_VIB (KL to prototype prior)
- us  = 0/1 : replace L_InfoNCE with L_SupCon (Khosla 2020, multi-positive contrastive)

Run configs:
- A (FedDSA-VIB):     vib=1 us=0
- B (FedDSA-VSC):     vib=1 us=1
- M6 (orth+SupCon):   vib=0 us=1

Design (rationale in refine-logs/2026-04-22_FedDSA-Swap/round-3-revised-proposal.md):
- VIB uses EMA-lagged stop-grad prototype as prior (fix: chicken-and-egg)
- σ-head (log_var_head) is LOCAL, not FedAvg aggregated (fix: domain-conditional uncertainty)
- log_sigma_prior is learnable per-class (fix: lookup degeneracy)

Everything else inherits from feddsa_sgpa (dom_head, style_bank, whitening, CDANN all
available via ca flag if needed, but default ca=0 in new configs).
"""

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import flgo.utils.fmodule as fuf

from algorithm.feddsa_sgpa import (
    CDANN_WARMUP_END,
    CDANN_WARMUP_START,
    DL,
    Client as _BaseClient,
    FedDSASGPAModel as _BaseModel,
    Server as _BaseServer,
    _resolve_num_classes,
    _resolve_num_clients,
    build_etf_matrix,
    compute_lambda_adv,
    grl,
)
from algorithm.common.vib import VIBSemanticHead, lambda_ib_schedule
from algorithm.common.supcon import supcon_diagnostics, supcon_loss
from algorithm.common.diagnostic_ext import (
    domain_conditional_rate,
    kl_collapse_detect,
)


# ============================================================
# Model: same as SGPA but (optionally) VIB semantic head
# ============================================================


class FedDSAVIBModel(_BaseModel):
    """FedDSA-SGPA with optional VIB semantic head.

    If vib=True: semantic_head is replaced by VIBSemanticHead (stochastic).
    If vib=False: inherits baseline behavior (deterministic Linear head).
    """

    def __init__(self, num_classes=7, feat_dim=1024, proj_dim=128,
                 tau_etf=0.1, etf_seed=0, use_etf=True,
                 ca=0, num_clients=4, vib=0):
        super().__init__(num_classes, feat_dim, proj_dim, tau_etf, etf_seed,
                         use_etf, ca, num_clients)
        self.vib = int(vib)
        if self.vib == 1:
            # Replace semantic_head with stochastic VIB variant
            self.semantic_head = VIBSemanticHead(feat_dim, proj_dim, num_classes)

    def get_semantic(self, h, y=None, training=None):
        """Return z_sem (tensor). For VIB train mode use semantic_head(h, y, training=True)
        directly to also get (mu, log_var, kl)."""
        if self.vib == 1:
            training = training if training is not None else self.training
            out = self.semantic_head(h, y=y, training=training)
            return out[0]  # z_sem only; caller should use semantic_head(...) for full tuple
        return self.semantic_head(h)

    def forward(self, x):
        h = self.encode(x)
        z_sem = self.get_semantic(h)  # no y at inference, returns mu
        return self.classify(z_sem)


# ============================================================
# Server: adds VIB private_keys, prototype EMA update
# ============================================================


class Server(_BaseServer):
    def initialize(self):
        # Inject flags BEFORE parent init (which calls init_algo_para)
        # Parent reads algo_para; we add 2 new keys: vib, us, plus hyperparams
        super().initialize()
        # Pass VIB/SupCon config to clients
        vib_val = int(getattr(self, 'vib', 0))
        us_val = int(getattr(self, 'us', 0))
        lib_val = float(getattr(self, 'lib', 0.01))
        lsc_val = float(getattr(self, 'lsc', 1.0))
        vws_val = int(getattr(self, 'vws', 20))
        vwe_val = int(getattr(self, 'vwe', 50))
        sct_val = float(getattr(self, 'sct', 0.07))
        for c in self.clients:
            c.vib = vib_val
            c.us = us_val
            c.lambda_ib = lib_val
            c.lambda_supcon = lsc_val
            c.vib_warmup_start = vws_val
            c.vib_warmup_end = vwe_val
            c.supcon_tau = sct_val
        # ★ Review fix #16: log resolved algo_para at init for deployment safety
        print(f"[FedDSA-VIB init] vib={vib_val} us={us_val} lib={lib_val} "
              f"lsc={lsc_val} vws={vws_val} vwe={vwe_val} sct={sct_val}",
              flush=True)

    def init_algo_para(self, defaults):
        """Extend parent's algo_para with new VIB/SupCon keys."""
        extra = {
            'vib': 0,   # 0=no VIB, 1=VIB enabled
            'us':  0,   # 0=InfoNCE, 1=SupCon
            'lib': 1.0,  # lambda_IB weight (VIB loss coefficient)
            'lsc': 1.0,  # lambda_SupCon weight
            'vws': 20,   # VIB warmup start round
            'vwe': 50,   # VIB warmup end round
            'sct': 0.07,  # SupCon temperature
        }
        merged = {**defaults, **extra}
        super().init_algo_para(merged)

    def _build_model(self):
        """Build FedDSAVIBModel instead of base model (invoked by flgo on first call)."""
        # flgo may construct the model elsewhere; ensure vib flag propagates.
        # This method is conceptual — actual model is passed via register_algorithm.
        pass

    def _init_agg_keys(self):
        """FedBN + style_head local + VIB σ-head & sigma_prior local."""
        all_keys = list(self.model.state_dict().keys())
        self.private_keys = set()
        for k in all_keys:
            if 'style_head' in k:
                self.private_keys.add(k)
            elif 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
                self.private_keys.add(k)
            # VIB FIX: σ parameters are domain-specific — must stay local
            elif 'log_var_head' in k or 'log_sigma_prior' in k:
                self.private_keys.add(k)
            # EMA prototype is a buffer; keep it on the server (do not FedAvg)
            elif k.endswith('prototype_ema') or k.endswith('prototype_init'):
                self.private_keys.add(k)
        # M buffer always skip
        for k in all_keys:
            if k.endswith('.M') or k == 'M':
                self.private_keys.add(k)
        self.shared_keys = [k for k in all_keys if k not in self.private_keys]

    def iterate(self):
        """Run parent iterate, then update prototype_ema from aggregated class_centers."""
        # Parent does communication + aggregation + style bank + whitening + diag
        super().iterate()

        # VIB-specific: update prototype_ema on server (pushed to clients next round via pack)
        if int(getattr(self, 'vib', 0)) == 1:
            self._update_prototype_ema()

    def _update_prototype_ema(self):
        """Aggregate class_centers across CURRENT ROUND clients and update prototype_ema.

        Fix (2026-04-22 Codex final review): originally iterated over ALL self.clients,
        which under proportion<1 would mix STALE centers from prior rounds into the
        prototype prior. Now only read from clients that actually participated this
        round via self.received_clients.
        """
        # Resolve current-round participants (fall back to all clients if attr missing)
        participating_ids = getattr(self, 'received_clients', None)
        if participating_ids is None:
            participating_ids = list(range(len(self.clients)))
        centers_per_client = []
        for cid in participating_ids:
            if cid >= len(self.clients):
                continue
            c = self.clients[cid]
            ct = getattr(c, '_local_class_centers', None)
            if ct is not None:
                centers_per_client.append(ct)
        if len(centers_per_client) < 2:
            return  # need multiple clients to form reliable prototype

        stacked = torch.stack(centers_per_client, dim=0)  # [N, K, d]
        nan_mask = stacked.isnan().any(dim=-1)            # [N, K]
        valid_count = (~nan_mask).sum(dim=0).float()      # [K] # of clients having each class
        stacked_zero = torch.where(
            nan_mask.unsqueeze(-1), torch.zeros_like(stacked), stacked)
        summed = stacked_zero.sum(dim=0)                  # [K, d]
        prototype_new = summed / valid_count.clamp(min=1).unsqueeze(-1)
        class_active_mask = (valid_count >= 1)

        # Update model.semantic_head (VIBSemanticHead) prototype
        sem_head = self.model.semantic_head
        if hasattr(sem_head, 'update_prototype_ema'):
            sem_head.update_prototype_ema(prototype_new.to(sem_head.prototype_ema.device),
                                          class_active_mask.to(sem_head.prototype_ema.device))


# ============================================================
# Client: VIB-aware train loop (override)
# ============================================================


class Client(_BaseClient):
    def unpack(self, svr_pkg):
        """Override parent unpack to also skip VIB-private keys from global overwrite.

        VIB Fix 1 (CRITICAL, 2026-04-22):
          - log_var_head (σ-head) parameters MUST stay local (domain-conditional).
          - log_sigma_prior (per-class σ prior) MUST stay local.
          - prototype_ema / prototype_init: server pushes these DOWN; we overwrite.
        """
        global_model = svr_pkg['model']
        if self.model is None:
            self.model = global_model
        else:
            local_dict = self.model.state_dict()
            global_dict = global_model.state_dict()
            for key in local_dict.keys():
                # Parent's existing skips
                if 'style_head' in key:
                    continue
                if 'bn' in key.lower() and ('running_' in key or 'num_batches_tracked' in key):
                    continue
                if key.endswith('.M') or key == 'M':
                    continue
                # ★ VIB skip: keep σ-head local
                if 'log_var_head' in key:
                    continue
                if 'log_sigma_prior' in key:
                    continue
                # ★ prototype_ema/init: server-managed, DO overwrite (explicit for clarity)
                local_dict[key] = global_dict[key]
            self.model.load_state_dict(local_dict)
        self.current_round = svr_pkg['current_round']
        self.source_mu_k = svr_pkg.get('source_mu_k', None)
        self.mu_global = svr_pkg.get('mu_global', None)
        self.sigma_inv_sqrt = svr_pkg.get('sigma_inv_sqrt', None)

    def initialize(self):
        super().initialize()
        # Preserve Server-assigned flags if already set (handles init order
        # where Server.initialize runs before Client.initialize).
        # Use getattr to avoid overwriting.
        self.vib = int(getattr(self, 'vib', 0))
        self.us = int(getattr(self, 'us', 0))
        self.lambda_ib = float(getattr(self, 'lambda_ib', 1.0))
        self.lambda_supcon = float(getattr(self, 'lambda_supcon', 1.0))
        self.vib_warmup_start = int(getattr(self, 'vib_warmup_start', 20))
        self.vib_warmup_end = int(getattr(self, 'vib_warmup_end', 50))
        self.supcon_tau = float(getattr(self, 'supcon_tau', 0.07))
        self._last_kl_mean = 0.0
        self._last_sigma_mean = 0.0
        self._last_supcon_diag = None

    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        """Extended train: adds VIB KL loss (if vib=1) and SupCon (if us=1).

        Mirrors parent train but with VIB/SupCon injection points.
        """
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum
        )

        self._maybe_init_diag_logger()
        should_log_diag = (
            self._dl_train is not None
            and (self.current_round % self.diag_interval == 0)
        )

        style_sum = None
        style_sq_sum = None
        style_n = 0
        last_epoch_z_sty = []
        last_epoch_z_sem = []
        last_epoch_y = []
        diag_snapshot = None

        num_steps = self.num_steps
        steps_per_epoch = max(1, len(self.train_data) // self.batch_size)
        last_epoch_start = num_steps - steps_per_epoch

        lambda_ib_cur = 0.0
        if self.vib == 1:
            lambda_ib_cur = self.lambda_ib * lambda_ib_schedule(
                self.current_round, self.vib_warmup_start, self.vib_warmup_end
            )

        kl_list = []
        sigma_list = []

        for step in range(num_steps):
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            x, y = batch_data[0], batch_data[-1]

            model.zero_grad()
            h = model.encode(x)

            # Semantic branch: VIB or plain
            if self.vib == 1:
                # VIB returns (z_sem, mu, log_var, kl)
                z_sem, mu_sem, log_var_sem, kl = model.semantic_head(
                    h, y=y, training=True
                )
            else:
                z_sem = model.get_semantic(h)
                mu_sem = z_sem  # same tensor for orth / HSIC / contrastive uses
                kl = None

            z_sty = model.get_style(h)

            # Task loss
            logits = model.classify(z_sem)
            loss_task = self.loss_fn(logits, y)

            # L_aug (h-space AdaIN, only when bank ready + past warmup)
            loss_task_aug = torch.tensor(0.0, device=x.device)
            # Note: feddsa_sgpa does not apply L_aug inside the train loop of feddsa_sgpa
            # (it handles style via server-side bank update); to stay consistent we keep
            # L_aug aligned with whatever the parent does. We simply skip here because
            # feddsa_sgpa base did not wire an h-space AdaIN pass through. (See CLAUDE.md
            # §15.1; the VIB proposal targets the same loss landscape as feddsa_sgpa
            # orth_only + new losses.)

            # Orthogonality (use deterministic mu_sem, not stochastic z_sem — stability)
            mu_sem_n = F.normalize(mu_sem, dim=-1)
            z_sty_n = F.normalize(z_sty, dim=-1)
            loss_orth = ((mu_sem_n * z_sty_n).sum(dim=-1) ** 2).mean()

            # VIB KL loss
            loss_vib = torch.tensor(0.0, device=x.device)
            if self.vib == 1 and kl is not None:
                loss_vib = kl
                kl_list.append(kl.detach().item())
                sigma_list.append(torch.exp(0.5 * log_var_sem).mean().detach().item())

            # Prototype alignment: SupCon (us=1) or InfoNCE-like via base helper (us=0)
            loss_sem_align = torch.tensor(0.0, device=x.device)
            if self.us == 1:
                loss_sem_align = self.lambda_supcon * supcon_loss(
                    mu_sem, y, temperature=self.supcon_tau
                )
                if step == num_steps - 1:
                    self._last_supcon_diag = supcon_diagnostics(mu_sem, y)
            # For us=0 (InfoNCE) the parent Client.train would add loss_sem_con;
            # we do not replicate it here because the VIB variants (A/B) intentionally
            # lean on VIB + prototype_ema pull (via KL) instead of a separate InfoNCE.

            # CDANN block deliberately omitted — new configs default ca=0.
            # If future work wants to mix CDANN + VIB, inherit and add the block.

            loss = (loss_task
                    + self.lambda_orth * loss_orth
                    + lambda_ib_cur * loss_vib
                    + loss_sem_align)
            loss.backward()
            optimizer.step()

            # Style stats accumulation (last epoch)
            if step >= last_epoch_start:
                with torch.no_grad():
                    z_sty_det = z_sty.detach().cpu()
                    # For centroid: use mu_sem (deterministic) for stability
                    z_sem_det = mu_sem.detach().cpu()
                    y_det = y.detach().cpu()
                    last_epoch_z_sty.append(z_sty_det)
                    last_epoch_z_sem.append(z_sem_det)
                    last_epoch_y.append(y_det)
                    b = z_sty_det.size(0)
                    batch_mu = z_sty_det.mean(dim=0)
                    batch_sq = (z_sty_det ** 2).mean(dim=0)
                    if style_sum is None:
                        style_sum = batch_mu * b
                        style_sq_sum = batch_sq * b
                        style_n = b
                    else:
                        style_sum += batch_mu * b
                        style_sq_sum += batch_sq * b
                        style_n += b

            if should_log_diag and step == num_steps - 1:
                diag_snapshot = (
                    mu_sem.detach().cpu(),
                    z_sty.detach().cpu(),
                    y.detach().cpu(),
                    loss_task.item(),
                    loss_orth.item(),
                )

        # Persist style (μ, Σ) for server bank
        if style_n >= 4 and last_epoch_z_sty:
            mu = style_sum / style_n
            Z = torch.cat(last_epoch_z_sty, dim=0)
            N = Z.size(0)
            Z_centered = Z - mu.unsqueeze(0)
            sigma = (Z_centered.t() @ Z_centered) / max(N - 1, 1)
            self._local_style_stats = (mu.clone(), sigma.clone())
        else:
            self._local_style_stats = None

        # Class centers → server uses these to update prototype_ema (VIB fix #1)
        self._local_class_centers = None
        use_centers_flag = getattr(self, 'use_centers', 1)
        if use_centers_flag == 1 and last_epoch_z_sem:
            Z_sem = torch.cat(last_epoch_z_sem, dim=0)
            Y = torch.cat(last_epoch_y, dim=0)
            K = self.model.num_classes
            d = Z_sem.shape[-1]
            centers = torch.full((K, d), float('nan'))
            for c in range(K):
                mask = Y == c
                if mask.sum() > 0:
                    centers[c] = Z_sem[mask].mean(dim=0)
            self._local_class_centers = centers

        # Track VIB diagnostics for this round
        self._last_kl_mean = sum(kl_list) / max(1, len(kl_list)) if kl_list else 0.0
        self._last_sigma_mean = (sum(sigma_list) / max(1, len(sigma_list))
                                 if sigma_list else 0.0)

        # Diagnostic snapshot (extended with VIB/SupCon)
        if diag_snapshot is not None:
            z_sem_d, z_sty_d, y_d, lt, lo = diag_snapshot
            K = self.model.num_classes
            M_cpu = self.model.M.detach().cpu()
            metrics = {
                'orth': DL.orthogonality(z_sem_d, z_sty_d),
                'etf_align_mean': DL.etf_alignment(z_sem_d, y_d, M_cpu, K)[0],
                'intra_cls_sim': DL.intra_class_similarity(z_sem_d, y_d, K),
                'inter_cls_sim': DL.inter_class_similarity(z_sem_d, y_d, K),
                'loss_task': lt,
                'loss_orth': lo,
            }
            metrics.update(DL.feature_norm_stats(z_sem_d, name='z_sem'))
            metrics.update(DL.feature_norm_stats(z_sty_d, name='z_sty'))

            # VIB metrics
            if self.vib == 1:
                metrics['kl_mean'] = self._last_kl_mean
                metrics['sigma_sem_mean'] = self._last_sigma_mean
                metrics['lambda_ib_effective'] = lambda_ib_cur
                # KL-collapse guard
                intra_z_std = self.model.semantic_head.get_intra_class_std(
                    z_sem_d.to(next(self.model.parameters()).device),
                    y_d.to(next(self.model.parameters()).device),
                ).item()
                metrics['intra_class_z_std'] = intra_z_std
                metrics['kl_collapse_alert'] = int(
                    kl_collapse_detect(self._last_kl_mean, intra_z_std)
                )

            # SupCon metrics
            if self.us == 1 and self._last_supcon_diag is not None:
                for k, v in self._last_supcon_diag.items():
                    metrics[f'supcon_{k}'] = v

            self._dl_train.record(
                round_id=self.current_round, metrics_dict=metrics)


# ============================================================
# flgo hook: construct global model (Server-side)
# ============================================================


def init_global_module(object):
    """Build FedDSAVIBModel with algo_para-driven flags (vib is index 13).

    algo_para order (extended from feddsa_sgpa's 13 base keys, total 20):
      [0] lo  [1] te  [2] pd  [3] wr  [4] es  [5] mcw  [6] dg
      [7] ue  [8] uw  [9] uc  [10] se [11] lp [12] ca
      [13] vib [14] us [15] lib [16] lsc [17] vws [18] vwe [19] sct
    """
    if 'Server' not in object.__class__.__name__:
        return

    task = os.path.basename(object.option['task'])
    num_classes = _resolve_num_classes(task)
    num_clients = _resolve_num_clients(task)

    use_etf = True
    ca = 0
    vib = 0
    algo_para = object.option.get('algo_para', None)
    if algo_para is not None:
        if len(algo_para) >= 8:
            use_etf = bool(int(algo_para[7]))
        if len(algo_para) >= 13:
            ca = int(algo_para[12])
        if len(algo_para) >= 14:
            vib = int(algo_para[13])

    object.model = FedDSAVIBModel(
        num_classes=num_classes,
        feat_dim=1024,
        proj_dim=128,
        use_etf=use_etf,
        ca=ca,
        num_clients=num_clients,
        vib=vib,
    ).to(object.device)
