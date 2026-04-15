"""
FedDSA-Gradual: Stabilized Decoupled Prototype Learning with Gradual Ramp-Up

Fixes FedDSA's training instability with three targeted changes:
  1. Decoupled Sigmoid Ramp-Up — replaces the hard warmup gate with two independent
     sigmoid schedules for augmentation and alignment weights, eliminating the
     abrupt stability break at round 50.
  2. Shallow Feature Augmentation — applies AdaIN at conv3 output (384-dim spatial
     feature maps) instead of the encoder output (1024-dim), perturbing low-level
     style statistics early in the network where it is less harmful.
  3. Gradient Conflict Logger — periodically computes cosine similarity between
     task-CE and alignment gradients on the encoder for diagnostics.

algo_para keys (short names for filename compatibility):
  lo=lambda_orth, lh=lambda_hsic, ls=lambda_sem, tau=InfoNCE temperature,
  sdn=style_dispatch_num, pd=proj_dim,
  tma=t_mid_aug, twa=tau_w_aug, tml=t_mid_align, twl=tau_w_align,
  al=aug_level (0=deep h-space, 1=shallow conv3), gli=grad_log_interval
"""
import os
import copy
import math
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flgo.algorithm.fedbase
import flgo.utils.fmodule as fuf
from collections import OrderedDict


# ============================================================
# Model: AlexNet backbone with two-stage forward + dual-head
# ============================================================

class AlexNetEncoder(nn.Module):
    """Same AlexNet as config.py / feddsa.py but with split-forward support
    for shallow augmentation at the conv3 boundary."""

    # Layer names for the two stages (conv3 boundary)
    _STAGE1_NAMES = [
        'conv1', 'bn1', 'relu1', 'maxpool1',
        'conv2', 'bn2', 'relu2', 'maxpool2',
        'conv3', 'bn3', 'relu3',
    ]
    _STAGE2_NAMES = [
        'conv4', 'bn4', 'relu4',
        'conv5', 'bn5', 'relu5',
        'maxpool5',
    ]

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
            ('bn2', nn.BatchNorm2d(192)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(384)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('bn5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU(inplace=True)),
            ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
        ]))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.bn7 = nn.BatchNorm1d(1024)

    def forward(self, x):
        """Standard full forward. Returns h [B, 1024]."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn6(self.fc1(x))
        x = self.relu(x)
        x = self.bn7(self.fc2(x))
        x = self.relu(x)
        return x  # [B, 1024]

    def forward_shallow(self, x):
        """Two-stage forward returning both h and f_shallow.

        Returns:
            h: [B, 1024] — full encoder output
            f_shallow: [B, 384, H, W] — feature map after conv3+bn3+relu3
        """
        # Stage 1: conv1 → conv3+bn3+relu3
        for name in self._STAGE1_NAMES:
            x = self.features._modules[name](x)
        f_shallow = x  # [B, 384, H, W]

        # Stage 2: conv4 → maxpool5 + FC layers
        x = self._forward_stage2(x)
        return x, f_shallow

    def forward_from_shallow(self, f_shallow):
        """Run stage2 only: conv4 → fc2. Used after augmenting f_shallow.

        Args:
            f_shallow: [B, 384, H, W] — possibly augmented conv3 output
        Returns:
            h: [B, 1024]
        """
        return self._forward_stage2(f_shallow)

    def _forward_stage2(self, x):
        """Shared stage2 logic: conv4-conv5 + avgpool + FC1-FC2."""
        for name in self._STAGE2_NAMES:
            x = self.features._modules[name](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn6(self.fc1(x))
        x = self.relu(x)
        x = self.bn7(self.fc2(x))
        x = self.relu(x)
        return x  # [B, 1024]


class FedDSAGradualModel(fuf.FModule):
    """Dual-head model identical in parameter names to FedDSAModel for
    compatibility with existing configs, checkpoints, and aggregation logic."""

    def __init__(self, num_classes=7, feat_dim=1024, proj_dim=128):
        super().__init__()
        self.encoder = AlexNetEncoder()
        # Semantic head
        self.semantic_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        # Style head (private, not aggregated)
        self.style_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        # Classifier (named "head" for framework compatibility)
        self.head = nn.Linear(proj_dim, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        z_sem = self.semantic_head(h)
        return self.head(z_sem)

    def encode(self, x):
        """Return backbone features for style computation."""
        return self.encoder(x)

    def get_semantic(self, h):
        return self.semantic_head(h)

    def get_style(self, h):
        return self.style_head(h)


# ============================================================
# Server
# ============================================================

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({
            'lo': 1.0,    # lambda_orth
            'lh': 0.0,    # lambda_hsic (disabled by default)
            'ls': 1.0,    # lambda_sem (InfoNCE weight)
            'tau': 0.2,   # InfoNCE temperature
            'sdn': 5,     # style_dispatch_num
            'pd': 128,    # proj_dim
            'tma': 15,    # t_mid_aug (sigmoid center for augmentation ramp)
            'twa': 8,     # tau_w_aug (sigmoid width for augmentation ramp)
            'tml': 40,    # t_mid_align (sigmoid center for alignment ramp)
            'twl': 10,    # tau_w_align (sigmoid width for alignment ramp)
            'al': 1,      # aug_level: 0=deep(h-space), 1=shallow(conv3)
            'gli': 10,    # grad_log_interval (0 to disable)
        })
        # Readable aliases
        self.lambda_orth = float(self.lo)
        self.lambda_hsic = float(self.lh)
        self.lambda_sem = float(self.ls)
        self.style_dispatch_num = int(self.sdn)
        self.proj_dim = int(self.pd)
        self.t_mid_aug = float(self.tma)
        self.tau_w_aug = float(self.twa)
        self.t_mid_align = float(self.tml)
        self.tau_w_align = float(self.twl)
        self.aug_level = int(self.al)
        self.grad_log_interval = int(self.gli)

        self.sample_option = 'full'

        # Style bank: stores (mu, sigma) per client
        #   aug_level=0: 1024-dim h-space stats
        #   aug_level=1: 384-dim shallow conv3 channel stats
        self.style_bank = {}  # client_id -> (mu, sigma)

        # Domain-aware protos: (class, client_id) -> z_sem proto
        self.domain_protos = {}
        # Global protos (weighted mean fallback): class -> z_sem proto
        self.global_semantic_protos = {}

        # Gradient conflict log: round -> {client_id: cosine_sim}
        self.grad_conflict_log = {}

        self._init_agg_keys()

        # Pass hyperparameters to all clients
        for c in self.clients:
            c.lambda_orth = self.lambda_orth
            c.lambda_hsic = self.lambda_hsic
            c.lambda_sem = self.lambda_sem
            c.tau = self.tau
            c.proj_dim = self.proj_dim
            c.t_mid_aug = self.t_mid_aug
            c.tau_w_aug = self.tau_w_aug
            c.t_mid_align = self.t_mid_align
            c.tau_w_align = self.tau_w_align
            c.aug_level = self.aug_level
            c.grad_log_interval = self.grad_log_interval

    def _init_agg_keys(self):
        """Split model parameters into shared (FedAvg) and private sets."""
        all_keys = list(self.model.state_dict().keys())
        self.private_keys = set()
        for k in all_keys:
            if 'style_head' in k:
                self.private_keys.add(k)
            elif 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
                self.private_keys.add(k)
        self.shared_keys = [k for k in all_keys if k not in self.private_keys]

    def pack(self, client_id, mtype=0):
        """Send global model + domain-aware protos + global protos + style bank."""
        # Dispatch styles (exclude client's own)
        dispatched_styles = None
        if len(self.style_bank) > 0:
            available = {cid: s for cid, s in self.style_bank.items() if cid != client_id}
            if len(available) == 0:
                available = self.style_bank
            n = min(self.style_dispatch_num, len(available))
            keys = list(available.keys())
            chosen = np.random.choice(keys, n, replace=False)
            dispatched_styles = [available[k] for k in chosen]

        return {
            'model': copy.deepcopy(self.model),
            'global_protos': copy.deepcopy(self.global_semantic_protos),
            'domain_protos': copy.deepcopy(self.domain_protos),
            'style_bank': dispatched_styles,
            'current_round': self.current_round,
        }

    def iterate(self):
        # 1. Sample all clients
        self.selected_clients = self.sample()
        # 2. Communicate
        res = self.communicate(self.selected_clients)

        models = res['model']
        protos_list = res['protos']
        proto_counts_list = res['proto_counts']
        style_stats_list = res['style_stats']
        grad_conflict_list = res['grad_conflict']

        # 3. Aggregate shared parameters (FedAvg)
        self._aggregate_shared(models)

        # 4. Collect style bank
        for cid, style in zip(self.received_clients, style_stats_list):
            if style is not None:
                self.style_bank[cid] = style

        # 5. Store domain-aware protos: (class, client_id) -> proto
        self._store_domain_protos(protos_list)

        # 6. Aggregate global protos (weighted mean fallback)
        self._aggregate_protos(protos_list, proto_counts_list)

        # 7. Log gradient conflict data
        for cid, gc in zip(self.received_clients, grad_conflict_list):
            if gc is not None:
                if self.current_round not in self.grad_conflict_log:
                    self.grad_conflict_log[self.current_round] = {}
                self.grad_conflict_log[self.current_round][cid] = gc
        if self.current_round in self.grad_conflict_log:
            entries = self.grad_conflict_log[self.current_round]
            vals = list(entries.values())
            mean_cos = sum(vals) / len(vals)
            self.gv.logger.info(
                f"[GradConflict] round={self.current_round} "
                f"mean_cos={mean_cos:.4f} per_client={entries}"
            )

    def _aggregate_shared(self, models):
        """FedAvg on shared keys only (excludes style_head + BN running stats)."""
        if len(models) == 0:
            return
        weights = np.array(
            [len(self.clients[cid].train_data) for cid in self.received_clients],
            dtype=float,
        )
        weights /= weights.sum()

        global_dict = self.model.state_dict()
        model_dicts = [m.state_dict() for m in models]

        for k in self.shared_keys:
            if 'num_batches_tracked' in k:
                continue
            global_dict[k] = sum(w * md[k] for w, md in zip(weights, model_dicts))

        self.model.load_state_dict(global_dict, strict=False)

    def _store_domain_protos(self, protos_list):
        """Store per-(class, client_id) prototypes for SupCon InfoNCE."""
        for cid, protos in zip(self.received_clients, protos_list):
            if protos is None:
                continue
            for c, proto in protos.items():
                self.domain_protos[(c, cid)] = proto

    def _aggregate_protos(self, protos_list, counts_list):
        """Weighted average of per-class semantic prototypes (global fallback)."""
        agg = {}
        for protos, counts in zip(protos_list, counts_list):
            if protos is None:
                continue
            for c, proto in protos.items():
                cnt = counts.get(c, 1)
                if c not in agg:
                    agg[c] = (proto * cnt, cnt)
                else:
                    prev_sum, prev_cnt = agg[c]
                    agg[c] = (prev_sum + proto * cnt, prev_cnt + cnt)

        self.global_semantic_protos = {}
        for c, (s, n) in agg.items():
            self.global_semantic_protos[c] = s / n


# ============================================================
# Client
# ============================================================

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = copy.deepcopy(self.server.model).to('cpu')
        self.loss_fn = nn.CrossEntropyLoss()
        self.current_round = 0
        # Received from server
        self.local_style_bank = None    # list of (mu, sigma) tuples
        self.global_protos = None       # class -> z_sem proto (fallback)
        self.domain_protos = None       # (class, client_id) -> z_sem proto
        # Gradient conflict diagnostic
        self._grad_conflict_log = None

    def reply(self, svr_pkg):
        model, global_protos, domain_protos, style_bank, current_round = self.unpack(svr_pkg)
        self.current_round = current_round
        self.global_protos = global_protos
        self.domain_protos = domain_protos
        self.local_style_bank = style_bank
        self.train(model)
        return self.pack()

    def unpack(self, svr_pkg):
        """Receive global model; keep style_head + BN running stats local."""
        global_model = svr_pkg['model']
        if self.model is None:
            self.model = global_model
        else:
            new_dict = self.model.state_dict()
            global_dict = global_model.state_dict()
            for key in new_dict.keys():
                if 'style_head' in key:
                    continue
                if 'bn' in key.lower() and ('running_' in key or 'num_batches_tracked' in key):
                    continue
                new_dict[key] = global_dict[key]
            self.model.load_state_dict(new_dict)
        return (
            self.model,
            svr_pkg['global_protos'],
            svr_pkg.get('domain_protos', {}),
            svr_pkg['style_bank'],
            svr_pkg['current_round'],
        )

    def pack(self):
        return {
            'model': copy.deepcopy(self.model.to('cpu')),
            'protos': self._local_protos,
            'proto_counts': self._local_proto_counts,
            'style_stats': self._local_style_stats,
            'grad_conflict': self._grad_conflict_log,
        }

    # ----------------------------------------------------------------
    # Sigmoid ramp-up schedule
    # ----------------------------------------------------------------

    @staticmethod
    def _sigmoid_ramp(t, t_mid, tau_w):
        """Smooth sigmoid ramp from ~0 to ~1.

        Args:
            t: current round
            t_mid: round at which weight = 0.5
            tau_w: width parameter (larger = slower transition)
        Returns:
            float in (0, 1)
        """
        return 1.0 / (1.0 + math.exp(-(t - t_mid) / max(tau_w, 1e-6)))

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------

    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum
        )

        # Sigmoid ramp weights (change 1)
        w_aug = self._sigmoid_ramp(self.current_round, self.t_mid_aug, self.tau_w_aug)
        w_align = self._sigmoid_ramp(self.current_round, self.t_mid_align, self.tau_w_align)

        # Online accumulators for prototypes and style stats
        proto_sum = {}
        proto_count = {}
        style_sum = None
        style_sq_sum = None
        style_n = 0

        # Determine shallow feature dim for style stats
        shallow_dim = 384   # conv3 output channels
        deep_dim = 1024     # h output dim

        num_steps = self.num_steps
        # Estimate steps per epoch for last-epoch collection
        steps_per_epoch = max(1, len(self.train_data) // self.batch_size)

        # Gradient conflict logging state
        should_log_grad = (
            self.grad_log_interval > 0
            and self.current_round % self.grad_log_interval == 0
        )
        self._grad_conflict_log = None

        for step in range(num_steps):
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            x, y = batch_data[0], batch_data[-1]

            model.zero_grad()

            # ---- Forward pass ----
            if self.aug_level == 1:
                # Shallow mode: split forward to get f_shallow
                h, f_shallow = model.encoder.forward_shallow(x)
            else:
                # Deep mode: standard forward
                h = model.encode(x)
                f_shallow = None

            z_sem = model.get_semantic(h)   # [B, proj_dim]
            z_sty = model.get_style(h)      # [B, proj_dim]

            # ---- Loss 1: Task CE on semantic path (always active) ----
            output = model.head(z_sem)
            loss_task = self.loss_fn(output, y)

            # ---- Loss 2: Style-augmented CE (with sigmoid ramp w_aug) ----
            loss_aug = torch.tensor(0.0, device=x.device)
            if self.local_style_bank is not None and w_aug > 1e-4:
                if self.aug_level == 1 and f_shallow is not None:
                    # Shallow augmentation (change 2)
                    f_aug = self._style_augment_shallow(f_shallow)
                    h_aug = model.encoder.forward_from_shallow(f_aug)
                else:
                    # Deep augmentation (original h-space AdaIN)
                    h_aug = self._style_augment_deep(h)
                z_sem_aug = model.get_semantic(h_aug)
                output_aug = model.head(z_sem_aug)
                loss_aug = self.loss_fn(output_aug, y)

            # ---- Loss 3: Decoupling (L_orth always full weight from round 0) ----
            loss_orth, loss_hsic = self._decouple_loss(z_sem, z_sty)

            # ---- Loss 4: Domain-aware SupCon InfoNCE (with sigmoid ramp w_align) ----
            loss_sem = torch.tensor(0.0, device=x.device)
            if self.domain_protos and len(self.domain_protos) >= 2:
                loss_sem = self._infonce_domain_aware(z_sem, y)
            elif self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_global(z_sem, y)

            # ---- Total loss ----
            # L_orth: always full weight (already stable)
            # L_aug: weighted by w_aug
            # L_align: weighted by w_align
            loss = (
                loss_task
                + w_aug * loss_aug
                + self.lambda_orth * loss_orth
                + self.lambda_hsic * loss_hsic
                + w_align * self.lambda_sem * loss_sem
            )

            # ---- Gradient conflict logging (change 3) ----
            # On the LAST batch of the LAST epoch, if this is a logging round
            is_last_batch = (step == num_steps - 1)
            if should_log_grad and is_last_batch and loss_sem.item() > 0:
                self._log_grad_conflict(
                    model, loss_task, w_align * self.lambda_sem * loss_sem
                )

            loss.backward()
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
            optimizer.step()

            # ---- Accumulate prototypes + style stats (last epoch only) ----
            if step >= num_steps - steps_per_epoch:
                with torch.no_grad():
                    z_det = z_sem.detach().cpu()

                    # Per-class semantic prototype accumulation
                    for i, label in enumerate(y):
                        c = label.item()
                        if c not in proto_sum:
                            proto_sum[c] = z_det[i].clone()
                            proto_count[c] = 1
                        else:
                            proto_sum[c] += z_det[i]
                            proto_count[c] += 1

                    # Style stats accumulation (Welford online)
                    if self.aug_level == 1 and f_shallow is not None:
                        # Shallow: channel-wise stats on conv3 output [B, 384, H, W]
                        f_det = f_shallow.detach()
                        b = f_det.size(0)
                        # Channel-wise mean: [384]
                        batch_mu = f_det.mean(dim=[0, 2, 3]).cpu()
                        # Channel-wise mean of squares: [384]
                        batch_sq = (f_det ** 2).mean(dim=[0, 2, 3]).cpu()
                    else:
                        # Deep: feature-wise stats on h [B, 1024]
                        h_det = h.detach()
                        b = h_det.size(0)
                        batch_mu = h_det.mean(dim=0).cpu()
                        batch_sq = (h_det ** 2).mean(dim=0).cpu()

                    if style_sum is None:
                        style_sum = batch_mu * b
                        style_sq_sum = batch_sq * b
                        style_n = b
                    else:
                        style_sum += batch_mu * b
                        style_sq_sum += batch_sq * b
                        style_n += b

        # ---- Store results for pack() ----
        self._local_protos = {c: proto_sum[c] / proto_count[c] for c in proto_sum}
        self._local_proto_counts = proto_count

        if style_n > 1:
            mu = style_sum / style_n
            var = style_sq_sum / style_n - mu ** 2
            self._local_style_stats = (mu, var.clamp(min=1e-6).sqrt())
        else:
            self._local_style_stats = None

    # ----------------------------------------------------------------
    # Style Augmentation — Shallow (Change 2)
    # ----------------------------------------------------------------

    def _style_augment_shallow(self, f_shallow):
        """AdaIN-style augmentation on conv3 spatial feature maps [B, 384, H, W].

        Channel-wise normalization then re-styling with mixed external stats.
        """
        B, C, H, W = f_shallow.shape

        # Local channel-wise statistics
        mu_local = f_shallow.mean(dim=[2, 3], keepdim=True)      # [B, C, 1, 1]
        sigma_local = f_shallow.std(dim=[2, 3], keepdim=True).clamp(min=1e-6)  # [B, C, 1, 1]

        # Sample external style
        idx = np.random.randint(0, len(self.local_style_bank))
        mu_ext, sigma_ext = self.local_style_bank[idx]
        mu_ext = mu_ext.to(f_shallow.device).view(1, C, 1, 1)
        sigma_ext = sigma_ext.to(f_shallow.device).view(1, C, 1, 1)

        # Normalize then re-style with Beta-mixed statistics
        f_norm = (f_shallow - mu_local) / sigma_local
        alpha = np.random.beta(0.1, 0.1)
        mu_mix = alpha * mu_local + (1 - alpha) * mu_ext
        sigma_mix = alpha * sigma_local + (1 - alpha) * sigma_ext
        return f_norm * sigma_mix + mu_mix

    # ----------------------------------------------------------------
    # Style Augmentation — Deep (original h-space)
    # ----------------------------------------------------------------

    def _style_augment_deep(self, h):
        """AdaIN-style augmentation on encoder output h [B, 1024].

        Same as original feddsa.py _style_augment.
        """
        idx = np.random.randint(0, len(self.local_style_bank))
        mu_ext, sigma_ext = self.local_style_bank[idx]
        mu_ext = mu_ext.to(h.device)
        sigma_ext = sigma_ext.to(h.device)

        mu_local = h.mean(dim=0)
        sigma_local = h.std(dim=0, unbiased=False).clamp(min=1e-6)

        h_norm = (h - mu_local) / sigma_local
        alpha = np.random.beta(0.1, 0.1)
        mu_mix = alpha * mu_local + (1 - alpha) * mu_ext
        sigma_mix = alpha * sigma_local + (1 - alpha) * sigma_ext
        return h_norm * sigma_mix + mu_mix

    # ----------------------------------------------------------------
    # Gradient Conflict Logger (Change 3)
    # ----------------------------------------------------------------

    def _log_grad_conflict(self, model, loss_task, loss_align_weighted):
        """Compute cosine similarity between task-CE and alignment gradients
        on encoder parameters. Diagnostic only — does NOT modify training."""
        # Collect encoder parameters that require grad
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        if not encoder_params:
            return

        try:
            grad_ce = torch.autograd.grad(
                loss_task, encoder_params,
                retain_graph=True, allow_unused=True, create_graph=False
            )
            grad_align = torch.autograd.grad(
                loss_align_weighted, encoder_params,
                retain_graph=True, allow_unused=True, create_graph=False
            )
        except RuntimeError:
            # Gradient computation may fail if graph has been freed
            return

        # Flatten and compute cosine similarity
        flat_ce = []
        flat_align = []
        for g_ce, g_al in zip(grad_ce, grad_align):
            if g_ce is not None and g_al is not None:
                flat_ce.append(g_ce.detach().flatten())
                flat_align.append(g_al.detach().flatten())

        if not flat_ce:
            return

        flat_ce = torch.cat(flat_ce)
        flat_align = torch.cat(flat_align)

        cos_sim = F.cosine_similarity(flat_ce.unsqueeze(0), flat_align.unsqueeze(0)).item()
        self._grad_conflict_log = cos_sim

    # ----------------------------------------------------------------
    # Decoupling losses (orthogonal + HSIC)
    # ----------------------------------------------------------------

    def _decouple_loss(self, z_sem, z_sty):
        """Orthogonal + HSIC dual constraint."""
        # Orthogonal: cos^2 penalty
        z_sem_n = F.normalize(z_sem, dim=1)
        z_sty_n = F.normalize(z_sty, dim=1)
        cos = (z_sem_n * z_sty_n).sum(dim=1)
        loss_orth = (cos ** 2).mean()

        # HSIC
        loss_hsic = self._hsic(z_sem, z_sty)
        return loss_orth, loss_hsic

    def _hsic(self, x, y):
        n = x.size(0)
        if n < 4:
            return torch.tensor(0.0, device=x.device)
        Kx = self._gaussian_kernel(x)
        Ky = self._gaussian_kernel(y)
        H = torch.eye(n, device=x.device) - torch.ones(n, n, device=x.device) / n
        return torch.trace(Kx @ H @ Ky @ H) / (n * n)

    def _gaussian_kernel(self, x):
        n = x.size(0)
        dist = torch.cdist(x, x, p=2) ** 2
        nonzero = dist[dist > 0]
        if nonzero.numel() == 0:
            return torch.ones(n, n, device=x.device)
        median = torch.median(nonzero)
        bw = median / (2.0 * np.log(n + 1) + 1e-6)
        return torch.exp(-dist / (2.0 * bw.clamp(min=1e-6)))

    # ----------------------------------------------------------------
    # Domain-Aware SupCon InfoNCE (from feddsa_adaptive.py M3)
    # ----------------------------------------------------------------

    def _infonce_domain_aware(self, z_sem, y):
        """Multi-positive SupCon InfoNCE with per-(class, client) prototypes.

        All same-class cross-domain prototypes serve as positives; different-class
        prototypes (from any domain) serve as negatives.
        """
        entries = []
        entry_classes = []
        for key in sorted(self.domain_protos.keys()):
            cls = key[0]
            proto = self.domain_protos[key]
            entries.append(proto)
            entry_classes.append(cls)

        if len(entries) < 2:
            return self._infonce_global(z_sem, y)

        proto_matrix = torch.stack([p.to(z_sem.device) for p in entries])
        proto_labels = torch.tensor(entry_classes, device=z_sem.device)

        z_n = F.normalize(z_sem, dim=1)
        p_n = F.normalize(proto_matrix, dim=1)
        logits = z_n @ p_n.T / self.tau

        loss = torch.tensor(0.0, device=z_sem.device)
        count = 0

        for i in range(y.size(0)):
            label = y[i].item()
            pos_mask = (proto_labels == label)
            n_pos = pos_mask.sum().item()
            if n_pos == 0:
                continue
            log_denom = torch.logsumexp(logits[i], dim=0)
            pos_logits = logits[i][pos_mask]
            loss += (-pos_logits + log_denom).sum() / n_pos
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=z_sem.device)
        return loss / count

    # ----------------------------------------------------------------
    # Global InfoNCE (fallback when domain protos unavailable)
    # ----------------------------------------------------------------

    def _infonce_global(self, z_sem, y):
        """InfoNCE: pull toward same-class global proto, push away others."""
        available = sorted([c for c, p in self.global_protos.items() if p is not None])
        if len(available) < 2:
            return torch.tensor(0.0, device=z_sem.device)

        proto_matrix = torch.stack([self.global_protos[c].to(z_sem.device) for c in available])
        class_to_idx = {c: i for i, c in enumerate(available)}

        z_n = F.normalize(z_sem, dim=1)
        p_n = F.normalize(proto_matrix, dim=1)
        logits = (z_n @ p_n.T) / self.tau

        targets = []
        valid = []
        for i in range(y.size(0)):
            label = y[i].item()
            if label in class_to_idx:
                targets.append(class_to_idx[label])
                valid.append(i)

        if len(valid) == 0:
            return torch.tensor(0.0, device=z_sem.device)

        valid_t = torch.tensor(valid, device=z_sem.device)
        targets_t = torch.tensor(targets, device=z_sem.device, dtype=torch.long)
        return F.cross_entropy(logits[valid_t], targets_t)


# ============================================================
# Model initialization (called by flgo framework)
# ============================================================

model_map = {
    'PACS': lambda: FedDSAGradualModel(num_classes=7, feat_dim=1024, proj_dim=128),
    'office': lambda: FedDSAGradualModel(num_classes=10, feat_dim=1024, proj_dim=128),
    'domainnet': lambda: FedDSAGradualModel(num_classes=10, feat_dim=1024, proj_dim=128),
}


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    dataset = os.path.split(object.option['task'])[-1].split('_')[0]
    if 'Server' in object.__class__.__name__:
        object.model = model_map.get(dataset, lambda: FedDSAGradualModel())().to(object.device)
