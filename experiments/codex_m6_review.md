OpenAI Codex v0.117.0 (research preview)
--------
workdir: D:\桌面文件\联邦学习
model: gpt-5.4
provider: fox
approval: never
sandbox: read-only
reasoning effort: high
reasoning summaries: none
session id: 019d91fe-8bf7-72e1-a9be-09e79a76a895
--------
user
Review M6 Delta-FiLM implementation in feddsa_adaptive.py (adaptive_mode=6). Focus on:
1. StyleModulator class: gamma/beta/gate generation, param count
2. _delta_film_augment(): delta normalization, class-matched pairing, residual FiLM, edge cases
3. Data flow: server collects z_sty protos -> splits intra/cross -> client does delta-FiLM
4. Gradient: does gradient flow z_sem_film -> StyleModulator -> delta_s -> z_sty -> style_head -> encoder?
5. mode=6 inherits M4 dual alignment + M5 style contrastive correctly?
6. style_head shared in mode=6 (same as mode=5)?
7. Training-only augmentation (inference disabled)?
8. Any bugs or numerical issues?
List bugs concisely.

=== KEY CODE ===
72:class StyleModulator(nn.Module):
100:        self.style_modulator = StyleModulator(proj_dim)  # M6: shared via FedAvg
201:            if 'style_head' in k and self.adaptive_mode not in (5, 6):
224:        use_domain = self.adaptive_mode in (2, 3, 4, 5, 6)
229:        if self.adaptive_mode in (4, 5, 6) and self.domain_protos:
239:        if self.adaptive_mode in (5, 6) and self.style_domain_protos:
286:        if self.adaptive_mode in (2, 3, 4, 5, 6):
290:        if self.adaptive_mode in (5, 6):
416:                if 'style_head' in key and self.adaptive_mode not in (5, 6):
444:        if self.adaptive_mode in (5, 6):
506:            if self.adaptive_mode in (4, 5, 6) and self.intra_protos and self.cross_protos:
509:            elif self.adaptive_mode in (4, 5, 6) and self.global_protos and len(self.global_protos) >= 2:
519:            if self.adaptive_mode in (5, 6) and self.sty_intra_protos and self.sty_cross_protos:
523:            loss_film = torch.tensor(0.0, device=x.device)
524:            if self.adaptive_mode == 6 and self.sty_intra_protos and self.sty_cross_protos \
529:                    loss_film = self.loss_fn(output_film, y)
531:            if self.adaptive_mode in (4, 5, 6):
539:                       aux_w * loss_film
569:                    if self.adaptive_mode in (5, 6):
611:        if self.adaptive_mode in (5, 6) and sty_proto_sum:
775:    def _delta_film_augment(self, model, z_sem, y):
815:            gamma, beta, gate = model.style_modulator(delta_s.unsqueeze(0))
=== FULL FILE ===
"""
FedDSA-Adaptive: Adaptive Augmentation + Domain-Aware Prototypes
Merges feddsa.py (base) + feddsa_domain_aware.py (M3) + new adaptive aug (M1).

Key changes vs base FedDSA:
  M1: Adaptive augmentation strength — gap from z_sty stats → per-client alpha
  M3: Domain-aware prototype alignment — per-(class, client) protos + SupCon InfoNCE
  M4: Dual alignment — L_intra (own-domain cosine) + L_cross (cross-domain InfoNCE)
  M5: M4 + style utilization — L_sty_contrast (domain-contrastive on z_sty)

Modes controlled by algo_para:
  adaptive_mode: 0=fixed_alpha, 1=adaptive (M1), 2=M3-only, 3=M1+M3 full, 4=M4 dual, 5=M4+style
  fixed_alpha_value: used when adaptive_mode=0
"""
import os
import copy
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flgo.algorithm.fedbase
import flgo.utils.fmodule as fuf
from collections import OrderedDict


# ============================================================
# Model: AlexNet backbone + dual-head (same as base FedDSA)
# ============================================================

class AlexNetEncoder(nn.Module):
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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn6(self.fc1(x))
        x = self.relu(x)
        x = self.bn7(self.fc2(x))
        x = self.relu(x)
        return x  # [B, 1024]


class StyleModulator(nn.Module):
    """M6: Delta-FiLM — maps style difference to modulation parameters."""
    def __init__(self, dim=128):
        super().__init__()
        self.film_net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim * 2)
        )
        self.gate_linear = nn.Linear(dim, 1)

    def forward(self, delta_sty):
        """delta_sty: [B, dim] or [1, dim] normalized style difference."""
        params = self.film_net(delta_sty)
        gamma, beta = params.chunk(2, dim=-1)
        gate = torch.sigmoid(self.gate_linear(delta_sty))  # [B, 1]
        return gamma, beta, gate


class FedDSAModel(fuf.FModule):
    def __init__(self, num_classes=7, feat_dim=1024, proj_dim=128):
        super().__init__()
        self.encoder = AlexNetEncoder()
        self.semantic_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.style_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.head = nn.Linear(proj_dim, num_classes)
        self.style_modulator = StyleModulator(proj_dim)  # M6: shared via FedAvg

    def forward(self, x):
        h = self.encoder(x)
        z_sem = self.semantic_head(h)
        return self.head(z_sem)

    def encode(self, x):
        return self.encoder(x)

    def get_semantic(self, h):
        return self.semantic_head(h)

    def get_style(self, h):
        return self.style_head(h)


# ============================================================
# Server — Adaptive Gap Computation + Domain-Aware Protos
# ============================================================

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        # Short keys used to keep record filename under 255 chars (Linux limit).
        # lo=lambda_orth, lh=lambda_hsic, ls=lambda_sem, wr=warmup_rounds,
        # sdn=style_dispatch_num, pd=proj_dim, am=aug_min, ax=aug_max,
        # ns=noise_std, ed=ema_decay, md=adaptive_mode, fa=fixed_alpha_value
        self.init_algo_para({
            'lo': 1.0,   # lambda_orth
            'lh': 0.0,   # lambda_hsic
            'ls': 1.0,   # lambda_sem
            'tau': 0.1,
            'wr': 50,    # warmup_rounds
            'sdn': 5,    # style_dispatch_num
            'pd': 128,   # proj_dim
            'am': 0.05,  # aug_min
            'ax': 0.8,   # aug_max
            'ns': 0.05,  # noise_std
            'ed': 0.9,   # ema_decay
            'md': 1,     # adaptive_mode: 0=fixed_alpha,1=M1,2=M3,3=M1+M3,4=M4-dual
            'fa': 0.5,   # fixed_alpha_value
            'li': 1.0,   # lambda_intra (M4 dual alignment)
            'lc': 0.5,   # lambda_cross (M4 dual alignment)
            'lsc': 0.5,  # lambda_sty_contrast (M5 style contrastive)
        })
        # Readable aliases so all downstream code is unchanged
        self.lambda_orth = float(self.lo)
        self.lambda_hsic = float(self.lh)
        self.lambda_sem = float(self.ls)
        self.warmup_rounds = int(self.wr)
        self.style_dispatch_num = int(self.sdn)
        self.proj_dim = int(self.pd)
        self.aug_min = float(self.am)
        self.aug_max = float(self.ax)
        self.noise_std = float(self.ns)
        self.ema_decay = float(self.ed)
        self.adaptive_mode = int(self.md)
        self.fixed_alpha_value = float(self.fa)
        self.lambda_intra = float(self.li)
        self.lambda_cross = float(self.lc)
        self.lambda_sty_contrast = float(self.lsc)
        self.sample_option = 'full'

        # Style bank: h-space stats for AdaIN augmentation
        self.style_bank = {}  # client_id -> (mu_h, sigma_h)
        # Style gap bank: z_sty-space stats for gap measurement (dual bank, R2 review)
        self.style_gap_bank = {}  # client_id -> (mu_zsty, sigma_zsty)

        # Prototypes
        self.global_semantic_protos = {}  # class -> avg proto
        self.domain_protos = {}  # (class, client_id) -> z_sem proto (M3/M4)
        self.style_domain_protos = {}  # (class, client_id) -> z_sty proto (M5)

        # Gap metrics (EMA smoothed)
        self.client_gaps = {}  # client_id -> normalized gap [0,1]
        self._ema_gap_mean = None
        self._ema_gap_std = None

        self._init_agg_keys()

        # Pass config to clients
        for c in self.clients:
            c.lambda_orth = self.lambda_orth
            c.lambda_hsic = self.lambda_hsic
            c.lambda_sem = self.lambda_sem
            c.tau = self.tau
            c.warmup_rounds = self.warmup_rounds
            c.proj_dim = self.proj_dim
            c.aug_min = self.aug_min
            c.aug_max = self.aug_max
            c.noise_std = self.noise_std
            c.adaptive_mode = int(self.adaptive_mode)
            c.fixed_alpha_value = float(self.fixed_alpha_value)
            c.lambda_intra = float(self.lambda_intra)
            c.lambda_cross = float(self.lambda_cross)
            c.lambda_sty_contrast = float(self.lambda_sty_contrast)

    def _init_agg_keys(self):
        all_keys = list(self.model.state_dict().keys())
        self.private_keys = set()
        for k in all_keys:
            if 'style_head' in k and self.adaptive_mode not in (5, 6):
                # M5/M6: style_head shared so z_sty is in a common space across clients
                self.private_keys.add(k)
            elif 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
                self.private_keys.add(k)
        self.shared_keys = [k for k in all_keys if k not in self.private_keys]

    def pack(self, client_id, mtype=0):
        # Dispatch style bank (h-space, for AdaIN augmentation)
        dispatched_styles = None
        if len(self.style_bank) > 0 and self.current_round >= self.warmup_rounds:
            available = {cid: s for cid, s in self.style_bank.items() if cid != client_id}
            if len(available) == 0:
                available = self.style_bank
            n = min(self.style_dispatch_num, len(available))
            keys = list(available.keys())
            chosen = np.random.choice(keys, n, replace=False)
            dispatched_styles = [available[k] for k in chosen]

        # Gap for this client (M1)
        gap_normalized = self.client_gaps.get(client_id, 0.5)

        # Choose which protos to send based on mode
        use_domain = self.adaptive_mode in (2, 3, 4, 5, 6)

        # M4/M5: split domain_protos into intra (own-domain) + cross (other-domain)
        intra_protos = {}
        cross_protos = {}
        if self.adaptive_mode in (4, 5, 6) and self.domain_protos:
            for (cls, cid), proto in self.domain_protos.items():
                if cid == client_id:
                    intra_protos[cls] = proto
                else:
                    cross_protos[(cls, cid)] = proto

        # M5: split style_domain_protos the same way
        sty_intra_protos = {}
        sty_cross_protos = {}
        if self.adaptive_mode in (5, 6) and self.style_domain_protos:
            for (cls, cid), proto in self.style_domain_protos.items():
                if cid == client_id:
                    sty_intra_protos[cls] = proto
                else:
                    sty_cross_protos[(cls, cid)] = proto

        return {
            'model': copy.deepcopy(self.model),
            'global_protos': copy.deepcopy(self.global_semantic_protos),
            'domain_protos': copy.deepcopy(self.domain_protos) if use_domain else {},
            'intra_protos': copy.deepcopy(intra_protos),
            'cross_protos': copy.deepcopy(cross_protos),
            'sty_intra_protos': copy.deepcopy(sty_intra_protos),
            'sty_cross_protos': copy.deepcopy(sty_cross_protos),
            'style_bank': dispatched_styles,
            'current_round': self.current_round,
            'gap_normalized': gap_normalized,
        }

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)

        models = res['model']
        protos_list = res['protos']
        proto_counts_list = res['proto_counts']
        style_stats_list = res['style_stats']
        style_gap_stats_list = res['style_gap_stats']

        # 1. Aggregate shared parameters (FedAvg)
        self._aggregate_shared(models)

        # 2. Collect dual style banks
        for cid, h_style, zsty_style in zip(
            self.received_clients, style_stats_list, style_gap_stats_list
        ):
            if h_style is not None:
                self.style_bank[cid] = h_style
            if zsty_style is not None:
                self.style_gap_bank[cid] = zsty_style

        # 3. Compute gap metrics from z_sty bank (M1)
        if self.adaptive_mode in (1, 3):
            self._compute_gap_metrics()

        # 4. Store domain protos (M3 / M4 / M5 / M6)
        if self.adaptive_mode in (2, 3, 4, 5, 6):
            self._store_domain_protos(protos_list)

        # 5. Store style domain protos (M5 / M6)
        if self.adaptive_mode in (5, 6):
            sty_protos_list = res.get('sty_protos', [None] * len(self.received_clients))
            for cid, sty_protos in zip(self.received_clients, sty_protos_list):
                if sty_protos is None:
                    continue
                for c, proto in sty_protos.items():
                    self.style_domain_protos[(c, cid)] = proto

        # 5. Aggregate global protos (always, as fallback)
        self._aggregate_protos(protos_list, proto_counts_list)

    def _compute_gap_metrics(self):
        """Compute per-client domain gap using z_sty statistics (pure domain signal)."""
        if len(self.style_gap_bank) < 2:
            return

        all_mu = torch.stack([s[0] for s in self.style_gap_bank.values()])
        all_sigma = torch.stack([s[1] for s in self.style_gap_bank.values()])

        mu_center = all_mu.mean(dim=0)
        sigma_center = all_sigma.mean(dim=0)

        raw_gaps = {}
        for cid, (mu, sigma) in self.style_gap_bank.items():
            gap = ((mu - mu_center) ** 2).sum() + ((sigma - sigma_center) ** 2).sum()
            raw_gaps[cid] = gap.item()

        # EMA z-score normalization
        gap_vals = list(raw_gaps.values())
        batch_mean = np.mean(gap_vals)
        batch_std = np.std(gap_vals) + 1e-8

        if self._ema_gap_mean is None:
            self._ema_gap_mean = batch_mean
            self._ema_gap_std = batch_std
        else:
            self._ema_gap_mean = self.ema_decay * self._ema_gap_mean + (1 - self.ema_decay) * batch_mean
            self._ema_gap_std = self.ema_decay * self._ema_gap_std + (1 - self.ema_decay) * batch_std

        for cid, g in raw_gaps.items():
            z = (g - self._ema_gap_mean) / (self._ema_gap_std + 1e-8)
            self.client_gaps[cid] = float(np.clip(z * 0.5 + 0.5, 0.0, 1.0))

    def _aggregate_shared(self, models):
        if len(models) == 0:
            return
        weights = np.array(
            [len(self.clients[cid].train_data) for cid in self.received_clients], dtype=float
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
        for cid, protos in zip(self.received_clients, protos_list):
            if protos is None:
                continue
            for c, proto in protos.items():
                self.domain_protos[(c, cid)] = proto

    def _aggregate_protos(self, protos_list, counts_list):
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
# Client — Adaptive Augmentation + Domain-Aware Alignment
# ============================================================

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = copy.deepcopy(self.server.model).to('cpu')
        self.loss_fn = nn.CrossEntropyLoss()
        self.current_round = 0
        self.local_style_bank = None
        self.global_protos = None
        self.domain_protos = None
        self.intra_protos = {}
        self.cross_protos = {}
        self.sty_intra_protos = {}
        self.sty_cross_protos = {}
        self.gap_normalized = 0.5

    def reply(self, svr_pkg):
        model, global_protos, domain_protos, style_bank, current_round, gap_normalized, intra_protos, cross_protos, sty_intra, sty_cross = self.unpack(svr_pkg)
        self.current_round = current_round
        self.global_protos = global_protos
        self.domain_protos = domain_protos
        self.local_style_bank = style_bank
        self.gap_normalized = gap_normalized
        self.intra_protos = intra_protos
        self.cross_protos = cross_protos
        self.sty_intra_protos = sty_intra
        self.sty_cross_protos = sty_cross
        self.train(model)
        return self.pack()

    def unpack(self, svr_pkg):
        global_model = svr_pkg['model']
        if self.model is None:
            self.model = global_model
        else:
            new_dict = self.model.state_dict()
            global_dict = global_model.state_dict()
            for key in new_dict.keys():
                if 'style_head' in key and self.adaptive_mode not in (5, 6):
                    # M5/M6: style_head is shared, load global weights
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
            svr_pkg.get('gap_normalized', 0.5),
            svr_pkg.get('intra_protos', {}),
            svr_pkg.get('cross_protos', {}),
            svr_pkg.get('sty_intra_protos', {}),
            svr_pkg.get('sty_cross_protos', {}),
        )

    def pack(self):
        result = {
            'model': copy.deepcopy(self.model.to('cpu')),
            'protos': self._local_protos,
            'proto_counts': self._local_proto_counts,
            'style_stats': self._local_style_stats,        # h-space (for AdaIN bank)
            'style_gap_stats': self._local_style_gap_stats, # z_sty-space (for gap metric)
        }
        if self.adaptive_mode in (5, 6):
            result['sty_protos'] = getattr(self, '_local_sty_protos', None)
        return result

    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum
        )

        aux_w = min(1.0, self.current_round / max(self.warmup_rounds, 1))
        use_adaptive_aug = self.adaptive_mode in (1, 3)
        use_domain_protos = self.adaptive_mode in (2, 3)

        # Online accumulators
        proto_sum = {}
        proto_count = {}
        sty_proto_sum = {}   # M5: z_sty class prototypes
        sty_proto_count = {}
        # h-space style stats (for style bank / AdaIN)
        h_style_sum = None
        h_style_sq_sum = None
        h_style_n = 0
        # z_sty-space style stats (for gap metric, dual bank)
        zsty_sum = None
        zsty_sq_sum = None
        zsty_n = 0

        num_steps = self.num_steps
        for step in range(num_steps):
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            x, y = batch_data[0], batch_data[-1]

            model.zero_grad()

            h = model.encode(x)            # [B, 1024]
            z_sem = model.get_semantic(h)   # [B, proj_dim]
            z_sty = model.get_style(h)     # [B, proj_dim]

            # Loss 1: Task CE on semantic path
            output = model.head(z_sem)
            loss_task = self.loss_fn(output, y)

            # Loss 2: Augmented CE (adaptive or fixed alpha)
            loss_aug = torch.tensor(0.0, device=x.device)
            if self.local_style_bank is not None and self.current_round >= self.warmup_rounds:
                h_aug = self._style_augment(h, use_adaptive_aug)
                z_sem_aug = model.get_semantic(h_aug)
                output_aug = model.head(z_sem_aug)
                loss_aug = self.loss_fn(output_aug, y)

            # Loss 3: Decoupling (orthogonal + HSIC)
            loss_orth, loss_hsic = self._decouple_loss(z_sem, z_sty)

            # Loss 4: Semantic alignment
            loss_sem = torch.tensor(0.0, device=x.device)
            loss_dual_intra = torch.tensor(0.0, device=x.device)
            loss_dual_cross = torch.tensor(0.0, device=x.device)

            if self.adaptive_mode in (4, 5, 6) and self.intra_protos and self.cross_protos:
                # M4/M5/M6: Dual alignment (intra-domain + cross-domain)
                loss_dual_intra, loss_dual_cross = self._dual_alignment_loss(z_sem, y)
            elif self.adaptive_mode in (4, 5, 6) and self.global_protos and len(self.global_protos) >= 2:
                # M4/M5/M6 fallback: use global protos until domain-level protos available
                loss_sem = self._infonce_global(z_sem, y)
            elif use_domain_protos and self.domain_protos and len(self.domain_protos) >= 2:
                loss_sem = self._infonce_domain_aware(z_sem, y)
            elif self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_global(z_sem, y)

            # M5/M6: style domain contrastive loss
            loss_sty_con = torch.tensor(0.0, device=x.device)
            if self.adaptive_mode in (5, 6) and self.sty_intra_protos and self.sty_cross_protos:
                loss_sty_con = self._style_domain_contrastive(z_sty, y)

            # M6: Delta-FiLM augmentation (training only)
            loss_film = torch.tensor(0.0, device=x.device)
            if self.adaptive_mode == 6 and self.sty_intra_protos and self.sty_cross_protos \
               and self.current_round >= self.warmup_rounds:
                z_sem_film = self._delta_film_augment(model, z_sem, y)
                if z_sem_film is not None:
                    output_film = model.head(z_sem_film)
                    loss_film = self.loss_fn(output_film, y)

            if self.adaptive_mode in (4, 5, 6):
                loss = loss_task + loss_aug + \
                       aux_w * self.lambda_orth * loss_orth + \
                       aux_w * self.lambda_hsic * loss_hsic + \
                       aux_w * self.lambda_intra * loss_dual_intra + \
                       aux_w * self.lambda_cross * loss_dual_cross + \
                       aux_w * self.lambda_sty_contrast * loss_sty_con + \
                       aux_w * self.lambda_sem * loss_sem + \
                       aux_w * loss_film
            else:
                loss = loss_task + loss_aug + \
                       aux_w * self.lambda_orth * loss_orth + \
                       aux_w * self.lambda_hsic * loss_hsic + \
                       aux_w * self.lambda_sem * loss_sem

            loss.backward()
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
            optimizer.step()

            # Accumulate stats (last epoch)
            if step >= num_steps - len(self.train_data) // self.batch_size - 1:
                with torch.no_grad():
                    z_det = z_sem.detach().cpu()
                    h_det = h.detach()
                    z_sty_det = z_sty.detach()

                    # Prototype accumulation (z_sem)
                    for i, label in enumerate(y):
                        c = label.item()
                        if c not in proto_sum:
                            proto_sum[c] = z_det[i].clone()
                            proto_count[c] = 1
                        else:
                            proto_sum[c] += z_det[i]
                            proto_count[c] += 1

                    # M5/M6: z_sty prototype accumulation
                    if self.adaptive_mode in (5, 6):
                        z_sty_cpu = z_sty_det.cpu()
                        for i, label in enumerate(y):
                            c = label.item()
                            if c not in sty_proto_sum:
                                sty_proto_sum[c] = z_sty_cpu[i].clone()
                                sty_proto_count[c] = 1
                            else:
                                sty_proto_sum[c] += z_sty_cpu[i]
                                sty_proto_count[c] += 1

                    b = h_det.size(0)

                    # h-space style stats (for AdaIN bank)
                    batch_mu_h = h_det.mean(dim=0).cpu()
                    batch_sq_h = (h_det ** 2).mean(dim=0).cpu()
                    if h_style_sum is None:
                        h_style_sum = batch_mu_h * b
                        h_style_sq_sum = batch_sq_h * b
                        h_style_n = b
                    else:
                        h_style_sum += batch_mu_h * b
                        h_style_sq_sum += batch_sq_h * b
                        h_style_n += b

                    # z_sty-space stats (for gap metric)
                    batch_mu_sty = z_sty_det.mean(dim=0).cpu()
                    batch_sq_sty = (z_sty_det ** 2).mean(dim=0).cpu()
                    if zsty_sum is None:
                        zsty_sum = batch_mu_sty * b
                        zsty_sq_sum = batch_sq_sty * b
                        zsty_n = b
                    else:
                        zsty_sum += batch_mu_sty * b
                        zsty_sq_sum += batch_sq_sty * b
                        zsty_n += b

        # Store for pack()
        self._local_protos = {c: proto_sum[c] / proto_count[c] for c in proto_sum}
        self._local_proto_counts = proto_count

        # M5/M6: z_sty prototypes
        if self.adaptive_mode in (5, 6) and sty_proto_sum:
            self._local_sty_protos = {c: sty_proto_sum[c] / sty_proto_count[c] for c in sty_proto_sum}
        else:
            self._local_sty_protos = None

        # h-space style stats
        if h_style_n > 1:
            mu_h = h_style_sum / h_style_n
            var_h = h_style_sq_sum / h_style_n - mu_h ** 2
            self._local_style_stats = (mu_h, var_h.clamp(min=1e-6).sqrt())
        else:
            self._local_style_stats = None

        # z_sty-space style stats (for gap)
        if zsty_n > 1:
            mu_sty = zsty_sum / zsty_n
            var_sty = zsty_sq_sum / zsty_n - mu_sty ** 2
            self._local_style_gap_stats = (mu_sty, var_sty.clamp(min=1e-6).sqrt())
        else:
            self._local_style_gap_stats = None

    # ---- Augmentation ----

    def _style_augment(self, h, use_adaptive):
        """AdaIN augmentation with adaptive or fixed alpha."""
        idx = np.random.randint(0, len(self.local_style_bank))
        mu_ext, sigma_ext = self.local_style_bank[idx]
        mu_ext = mu_ext.to(h.device)
        sigma_ext = sigma_ext.to(h.device)

        mu_local = h.mean(dim=0)
        sigma_local = h.std(dim=0, unbiased=False).clamp(min=1e-6)

        # Full AdaIN
        h_norm = (h - mu_local) / sigma_local
        h_adain = h_norm * sigma_ext + mu_ext

        if use_adaptive:
            # M1: gap → adaptive alpha + stochastic jitter
            alpha_mean = self.aug_min + (self.aug_max - self.aug_min) * self.gap_normalized
            alpha = float(np.clip(
                alpha_mean + np.random.normal(0, self.noise_std),
                self.aug_min, self.aug_max
            ))
        elif self.adaptive_mode == 0:
            # Fixed alpha mode (for baselines EXP-072a/b/c)
            alpha = self.fixed_alpha_value
        else:
            # M3-only mode: use original Beta mixing
            alpha = np.random.beta(0.1, 0.1)

        h_aug = alpha * h_adain + (1 - alpha) * h
        return h_aug

    # ---- M4: Dual Alignment (intra-domain + cross-domain) ----

    def _dual_alignment_loss(self, z_sem, y):
        """M4: Decomposed alignment — L_intra (own-domain cosine) + L_cross (cross-domain InfoNCE)."""
        device = z_sem.device
        B = z_sem.size(0)

        # --- L_intra: pull toward own-domain prototype ---
        loss_intra = torch.tensor(0.0, device=device)
        intra_count = 0
        for i in range(B):
            label = y[i].item()
            if label in self.intra_protos:
                proto = self.intra_protos[label].to(device)
                sim = F.cosine_similarity(z_sem[i:i+1], proto.unsqueeze(0)).squeeze()
                loss_intra = loss_intra + (1.0 - sim)
                intra_count += 1
        if intra_count > 0:
            loss_intra = loss_intra / intra_count

        # --- L_cross: InfoNCE with cross-domain protos (own domain excluded) ---
        loss_cross = torch.tensor(0.0, device=device)

        # Build cross-domain prototype matrix
        cross_entries = []
        cross_classes = []
        for (cls, cid) in sorted(self.cross_protos.keys()):
            cross_entries.append(self.cross_protos[(cls, cid)])
            cross_classes.append(cls)

        if len(cross_entries) >= 2:
            proto_matrix = torch.stack([p.to(device) for p in cross_entries])
            proto_labels = torch.tensor(cross_classes, device=device)

            z_n = F.normalize(z_sem, dim=1)
            p_n = F.normalize(proto_matrix, dim=1)
            logits = z_n @ p_n.T / self.tau  # tau=0.2

            cross_count = 0
            for i in range(B):
                label = y[i].item()
                pos_mask = (proto_labels == label)
                n_pos = pos_mask.sum().item()
                if n_pos == 0:
                    continue
                log_denom = torch.logsumexp(logits[i], dim=0)
                pos_logits = logits[i][pos_mask]
                loss_cross = loss_cross + (-pos_logits + log_denom).sum() / n_pos
                cross_count += 1

            if cross_count > 0:
                loss_cross = loss_cross / cross_count

        return loss_intra, loss_cross

    # ---- M5: Style Domain Contrastive ----

    def _style_domain_contrastive(self, z_sty, y):
        """M5: Domain-contrastive loss on z_sty.
        Same-domain same-class z_sty protos = positive (pull together).
        Cross-domain z_sty protos = negative (push apart).
        Effect: reinforces style head to capture domain-specific information.
        """
        device = z_sty.device

        # Build: own-domain z_sty protos are positive, cross-domain are negative
        # For each sample z_sty_i with label y_i:
        #   positive = sty_intra_protos[y_i] (same domain, same class)
        #   negatives = all sty_cross_protos (any class, other domains)

        # Gather all cross-domain style protos as negatives
        neg_entries = []
        for key in sorted(self.sty_cross_protos.keys()):
            neg_entries.append(self.sty_cross_protos[key])

        if len(neg_entries) == 0:
            return torch.tensor(0.0, device=device)

        neg_matrix = torch.stack([p.to(device) for p in neg_entries])  # [N_neg, 128]

        B = z_sty.size(0)
        loss = torch.tensor(0.0, device=device)
        count = 0

        for i in range(B):
            label = y[i].item()
            if label not in self.sty_intra_protos:
                continue

            pos_proto = self.sty_intra_protos[label].to(device)  # [128]

            # Cosine similarity
            z_n = F.normalize(z_sty[i:i+1], dim=1)    # [1, 128]
            pos_n = F.normalize(pos_proto.unsqueeze(0), dim=1)  # [1, 128]
            neg_n = F.normalize(neg_matrix, dim=1)      # [N_neg, 128]

            pos_sim = (z_n @ pos_n.T).view(-1)[0] / self.tau  # scalar
            neg_sims = (z_n @ neg_n.T).view(-1) / self.tau  # [N_neg], safe for any N_neg

            # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
            all_logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])  # [1 + N_neg]
            loss = loss + (-pos_sim + torch.logsumexp(all_logits, dim=0))
            count += 1

        if count > 0:
            loss = loss / count
        return loss

    # ---- M6: Delta-FiLM Augmentation ----

    def _delta_film_augment(self, model, z_sem, y):
        """M6: Cross-domain counterfactual style conditioning via Delta-FiLM.
        Uses normalized style DIFFERENCE (ext - local) as FiLM condition.
        Returns augmented z_sem or None if not enough protos.
        """
        device = z_sem.device
        B = z_sem.size(0)

        # Collect cross-domain z_sty protos with class info
        cross_list = [(cls, cid, p) for (cls, cid), p in self.sty_cross_protos.items()]
        if not cross_list or not self.sty_intra_protos:
            return None

        z_sem_film = z_sem.clone()
        augmented = False

        for i in range(B):
            label = y[i].item()

            # Need local z_sty proto for this class
            sty_local = self.sty_intra_protos.get(label)
            if sty_local is None:
                continue

            # Find same-class cross-domain z_sty protos
            candidates = [p for c, cid, p in cross_list if c == label]
            if not candidates:
                continue

            # Random pick one external style proto
            sty_ext = candidates[np.random.randint(len(candidates))]

            # Delta: direction of style shift from local to external domain
            delta_s = sty_ext.to(device) - sty_local.to(device)
            delta_norm = delta_s.norm()
            if delta_norm < 1e-6:
                continue  # domains too similar, skip
            delta_s = delta_s / delta_norm  # normalize to unit direction

            # FiLM modulation via StyleModulator
            gamma, beta, gate = model.style_modulator(delta_s.unsqueeze(0))
            gamma = gamma.squeeze(0)   # [proj_dim]
            beta = beta.squeeze(0)     # [proj_dim]
            gate = gate.squeeze()      # scalar

            # Residual FiLM: z_sem + gate * (gamma * z_sem + beta)
            z_sem_film[i] = z_sem[i] + gate * (gamma * z_sem[i] + beta)
            augmented = True

        return z_sem_film if augmented else None

    # ---- Domain-Aware InfoNCE (M3) ----

    def _infonce_domain_aware(self, z_sem, y):
        """Multi-positive SupCon InfoNCE with per-domain prototypes."""
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

    # ---- Global InfoNCE (fallback) ----

    def _infonce_global(self, z_sem, y):
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

    # ---- Decoupling losses ----

    def _decouple_loss(self, z_sem, z_sty):
        z_sem_n = F.normalize(z_sem, dim=1)
        z_sty_n = F.normalize(z_sty, dim=1)
        cos = (z_sem_n * z_sty_n).sum(dim=1)
        loss_orth = (cos ** 2).mean()

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


# ============================================================
# Model initialization (called by flgo framework)
# ============================================================

model_map = {
    'PACS': lambda: FedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128),
    'office': lambda: FedDSAModel(num_classes=10, feat_dim=1024, proj_dim=128),
    'domainnet': lambda: FedDSAModel(num_classes=10, feat_dim=1024, proj_dim=128),
}


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    dataset = os.path.split(object.option['task'])[-1].split('_')[0]
    if 'Server' in object.__class__.__name__:
        object.model = model_map.get(dataset, lambda: FedDSAModel())().to(object.device)

codex
Reviewing `feddsa_adaptive.py` with a code-review focus. I’m pulling the actual file with line numbers first, then I’ll trace the M6 path end to end against your eight checkpoints and return only concrete findings.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg --files -g 'feddsa_adaptive.py'" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content 'C:/Users/31279/.agents/skills/code-review-excellence/SKILL.md'" in D:\桌面文件\联邦学习
 succeeded in 872ms:
FDSE_CVPR25\algorithm\feddsa_adaptive.py
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1057ms:
---
name: code-review-excellence
description: Master effective code review practices to provide constructive feedback, catch bugs early, and foster knowledge sharing while maintaining team morale. Use when reviewing pull requests, establishing review standards, or mentoring developers.
---

# Code Review Excellence

Transform code reviews from gatekeeping to knowledge sharing through constructive feedback, systematic analysis, and collaborative improvement.

## When to Use This Skill

- Reviewing pull requests and code changes
- Establishing code review standards for teams
- Mentoring junior developers through reviews
- Conducting architecture reviews
- Creating review checklists and guidelines
- Improving team collaboration
- Reducing code review cycle time
- Maintaining code quality standards

## Core Principles

### 1. The Review Mindset

**Goals of Code Review:**

- Catch bugs and edge cases
- Ensure code maintainability
- Share knowledge across team
- Enforce coding standards
- Improve design and architecture
- Build team culture

**Not the Goals:**

- Show off knowledge
- Nitpick formatting (use linters)
- Block progress unnecessarily
- Rewrite to your preference

### 2. Effective Feedback

**Good Feedback is:**

- Specific and actionable
- Educational, not judgmental
- Focused on the code, not the person
- Balanced (praise good work too)
- Prioritized (critical vs nice-to-have)

```markdown
? Bad: "This is wrong."
? Good: "This could cause a race condition when multiple users
access simultaneously. Consider using a mutex here."

? Bad: "Why didn't you use X pattern?"
? Good: "Have you considered the Repository pattern? It would
make this easier to test. Here's an example: [link]"

? Bad: "Rename this variable."
? Good: "[nit] Consider `userCount` instead of `uc` for
clarity. Not blocking if you prefer to keep it."
```

### 3. Review Scope

**What to Review:**

- Logic correctness and edge cases
- Security vulnerabilities
- Performance implications
- Test coverage and quality
- Error handling
- Documentation and comments
- API design and naming
- Architectural fit

**What Not to Review Manually:**

- Code formatting (use Prettier, Black, etc.)
- Import organization
- Linting violations
- Simple typos

## Review Process

### Phase 1: Context Gathering (2-3 minutes)

```markdown
Before diving into code, understand:

1. Read PR description and linked issue
2. Check PR size (>400 lines? Ask to split)
3. Review CI/CD status (tests passing?)
4. Understand the business requirement
5. Note any relevant architectural decisions
```

### Phase 2: High-Level Review (5-10 minutes)

```markdown
1. **Architecture & Design**
   - Does the solution fit the problem?
   - Are there simpler approaches?
   - Is it consistent with existing patterns?
   - Will it scale?

2. **File Organization**
   - Are new files in the right places?
   - Is code grouped logically?
   - Are there duplicate files?

3. **Testing Strategy**
   - Are there tests?
   - Do tests cover edge cases?
   - Are tests readable?
```

### Phase 3: Line-by-Line Review (10-20 minutes)

```markdown
For each file:

1. **Logic & Correctness**
   - Edge cases handled?
   - Off-by-one errors?
   - Null/undefined checks?
   - Race conditions?

2. **Security**
   - Input validation?
   - SQL injection risks?
   - XSS vulnerabilities?
   - Sensitive data exposure?

3. **Performance**
   - N+1 queries?
   - Unnecessary loops?
   - Memory leaks?
   - Blocking operations?

4. **Maintainability**
   - Clear variable names?
   - Functions doing one thing?
   - Complex code commented?
   - Magic numbers extracted?
```

### Phase 4: Summary & Decision (2-3 minutes)

```markdown
1. Summarize key concerns
2. Highlight what you liked
3. Make clear decision:
   - ? Approve
   - ?? Comment (minor suggestions)
   - ?? Request Changes (must address)
4. Offer to pair if complex
```

## Review Techniques

### Technique 1: The Checklist Method

```markdown
## Security Checklist

- [ ] User input validated and sanitized
- [ ] SQL queries use parameterization
- [ ] Authentication/authorization checked
- [ ] Secrets not hardcoded
- [ ] Error messages don't leak info

## Performance Checklist

- [ ] No N+1 queries
- [ ] Database queries indexed
- [ ] Large lists paginated
- [ ] Expensive operations cached
- [ ] No blocking I/O in hot paths

## Testing Checklist

- [ ] Happy path tested
- [ ] Edge cases covered
- [ ] Error cases tested
- [ ] Test names are descriptive
- [ ] Tests are deterministic
```

### Technique 2: The Question Approach

Instead of stating problems, ask questions to encourage thinking:

```markdown
? "This will fail if the list is empty."
? "What happens if `items` is an empty array?"

? "You need error handling here."
? "How should this behave if the API call fails?"

? "This is inefficient."
? "I see this loops through all users. Have we considered
the performance impact with 100k users?"
```

### Technique 3: Suggest, Don't Command

````markdown
## Use Collaborative Language

? "You must change this to use async/await"
? "Suggestion: async/await might make this more readable:
`typescript
    async function fetchUser(id: string) {
        const user = await db.query('SELECT * FROM users WHERE id = ?', id);
        return user;
    }
    `
What do you think?"

? "Extract this into a function"
? "This logic appears in 3 places. Would it make sense to
extract it into a shared utility function?"
````

### Technique 4: Differentiate Severity

```markdown
Use labels to indicate priority:

?? [blocking] - Must fix before merge
?? [important] - Should fix, discuss if disagree
?? [nit] - Nice to have, not blocking
?? [suggestion] - Alternative approach to consider
?? [learning] - Educational comment, no action needed
?? [praise] - Good work, keep it up!

Example:
"?? [blocking] This SQL query is vulnerable to injection.
Please use parameterized queries."

"?? [nit] Consider renaming `data` to `userData` for clarity."

"?? [praise] Excellent test coverage! This will catch edge cases."
```

## Language-Specific Patterns

### Python Code Review

```python
# Check for Python-specific issues

# ? Mutable default arguments
def add_item(item, items=[]):  # Bug! Shared across calls
    items.append(item)
    return items

# ? Use None as default
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

# ? Catching too broad
try:
    result = risky_operation()
except:  # Catches everything, even KeyboardInterrupt!
    pass

# ? Catch specific exceptions
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise

# ? Using mutable class attributes
class User:
    permissions = []  # Shared across all instances!

# ? Initialize in __init__
class User:
    def __init__(self):
        self.permissions = []
```

### TypeScript/JavaScript Code Review

```typescript
// Check for TypeScript-specific issues

// ? Using any defeats type safety
function processData(data: any) {  // Avoid any
    return data.value;
}

// ? Use proper types
interface DataPayload {
    value: string;
}
function processData(data: DataPayload) {
    return data.value;
}

// ? Not handling async errors
async function fetchUser(id: string) {
    const response = await fetch(`/api/users/${id}`);
    return response.json();  // What if network fails?
}

// ? Handle errors properly
async function fetchUser(id: string): Promise<User> {
    try {
        const response = await fetch(`/api/users/${id}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Failed to fetch user:', error);
        throw error;
    }
}

// ? Mutation of props
function UserProfile({ user }: Props) {
    user.lastViewed = new Date();  // Mutating prop!
    return <div>{user.name}</div>;
}

// ? Don't mutate props
function UserProfile({ user, onView }: Props) {
    useEffect(() => {
        onView(user.id);  // Notify parent to update
    }, [user.id]);
    return <div>{user.name}</div>;
}
```

## Advanced Review Patterns

### Pattern 1: Architectural Review

```markdown
When reviewing significant changes:

1. **Design Document First**
   - For large features, request design doc before code
   - Review design with team before implementation
   - Agree on approach to avoid rework

2. **Review in Stages**
   - First PR: Core abstractions and interfaces
   - Second PR: Implementation
   - Third PR: Integration and tests
   - Easier to review, faster to iterate

3. **Consider Alternatives**
   - "Have we considered using [pattern/library]?"
   - "What's the tradeoff vs. the simpler approach?"
   - "How will this evolve as requirements change?"
```

### Pattern 2: Test Quality Review

```typescript
// ? Poor test: Implementation detail testing
test('increments counter variable', () => {
    const component = render(<Counter />);
    const button = component.getByRole('button');
    fireEvent.click(button);
    expect(component.state.counter).toBe(1);  // Testing internal state
});

// ? Good test: Behavior testing
test('displays incremented count when clicked', () => {
    render(<Counter />);
    const button = screen.getByRole('button', { name: /increment/i });
    fireEvent.click(button);
    expect(screen.getByText('Count: 1')).toBeInTheDocument();
});

// Review questions for tests:
// - Do tests describe behavior, not implementation?
// - Are test names clear and descriptive?
// - Do tests cover edge cases?
// - Are tests independent (no shared state)?
// - Can tests run in any order?
```

### Pattern 3: Security Review

```markdown
## Security Review Checklist

### Authentication & Authorization

- [ ] Is authentication required where needed?
- [ ] Are authorization checks before every action?
- [ ] Is JWT validation proper (signature, expiry)?
- [ ] Are API keys/secrets properly secured?

### Input Validation

- [ ] All user inputs validated?
- [ ] File uploads restricted (size, type)?
- [ ] SQL queries parameterized?
- [ ] XSS protection (escape output)?

### Data Protection

- [ ] Passwords hashed (bcrypt/argon2)?
- [ ] Sensitive data encrypted at rest?
- [ ] HTTPS enforced for sensitive data?
- [ ] PII handled according to regulations?

### Common Vulnerabilities

- [ ] No eval() or similar dynamic execution?
- [ ] No hardcoded secrets?
- [ ] CSRF protection for state-changing operations?
- [ ] Rate limiting on public endpoints?
```

## Giving Difficult Feedback

### Pattern: The Sandwich Method (Modified)

```markdown
Traditional: Praise + Criticism + Praise (feels fake)

Better: Context + Specific Issue + Helpful Solution

Example:
"I noticed the payment processing logic is inline in the
controller. This makes it harder to test and reuse.

[Specific Issue]
The calculateTotal() function mixes tax calculation,
discount logic, and database queries, making it difficult
to unit test and reason about.

[Helpful Solution]
Could we extract this into a PaymentService class? That
would make it testable and reusable. I can pair with you
on this if helpful."
```

### Handling Disagreements

```markdown
When author disagrees with your feedback:

1. **Seek to Understand**
   "Help me understand your approach. What led you to
   choose this pattern?"

2. **Acknowledge Valid Points**
   "That's a good point about X. I hadn't considered that."

3. **Provide Data**
   "I'm concerned about performance. Can we add a benchmark
   to validate the approach?"

4. **Escalate if Needed**
   "Let's get [architect/senior dev] to weigh in on this."

5. **Know When to Let Go**
   If it's working and not a critical issue, approve it.
   Perfection is the enemy of progress.
```

## Best Practices

1. **Review Promptly**: Within 24 hours, ideally same day
2. **Limit PR Size**: 200-400 lines max for effective review
3. **Review in Time Blocks**: 60 minutes max, take breaks
4. **Use Review Tools**: GitHub, GitLab, or dedicated tools
5. **Automate What You Can**: Linters, formatters, security scans
6. **Build Rapport**: Emoji, praise, and empathy matter
7. **Be Available**: Offer to pair on complex issues
8. **Learn from Others**: Review others' review comments

## Common Pitfalls

- **Perfectionism**: Blocking PRs for minor style preferences
- **Scope Creep**: "While you're at it, can you also..."
- **Inconsistency**: Different standards for different people
- **Delayed Reviews**: Letting PRs sit for days
- **Ghosting**: Requesting changes then disappearing
- **Rubber Stamping**: Approving without actually reviewing
- **Bike Shedding**: Debating trivial details extensively

## Templates

### PR Review Comment Template

```markdown
## Summary

[Brief overview of what was reviewed]

## Strengths

- [What was done well]
- [Good patterns or approaches]

## Required Changes

?? [Blocking issue 1]
?? [Blocking issue 2]

## Suggestions

?? [Improvement 1]
?? [Improvement 2]

## Questions

? [Clarification needed on X]
? [Alternative approach consideration]

## Verdict

? Approve after addressing required changes
```
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I’ve located the file and started tracing the M6 path in the concrete implementation rather than the summary markers. Next I’m reading the relevant sections with exact line numbers so the review can point to real failure modes, not intent.
2026-04-15T16:35:06.030041Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'; "'$lines=Get-Content $p; for($i=261;$i -le [Math]::Min(620,$lines.Length);$i++){ '"'{0,4}:{1}' -f "'$i,$lines[$i-1] }'` rejected: blocked by policy
2026-04-15T16:35:06.030530Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'; "'$lines=Get-Content $p; for($i=621;$i -le $lines.Length;$i++){ '"'{0,4}:{1}' -f "'$i,$lines[$i-1] }'` rejected: blocked by policy
2026-04-15T16:35:06.110930Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'; "'$lines=Get-Content $p; for($i=1;$i -le [Math]::Min(260,$lines.Length);$i++){ '"'{0,4}:{1}' -f "'$i,$lines[$i-1] }'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'; "'$lines=Get-Content $p; for($i=1;$i -le [Math]::Min(260,$lines.Length);$i++){ '"'{0,4}:{1}' -f "'$i,$lines[$i-1] }' in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'; "'$lines=Get-Content $p; for($i=261;$i -le [Math]::Min(620,$lines.Length);$i++){ '"'{0,4}:{1}' -f "'$i,$lines[$i-1] }' in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'; "'$lines=Get-Content $p; for($i=621;$i -le $lines.Length;$i++){ '"'{0,4}:{1}' -f "'$i,$lines[$i-1] }' in D:\桌面文件\联邦学习
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'; "'$lines=Get-Content $p; for($i=261;$i -le [Math]::Min(620,$lines.Length);$i++){ '"'{0,4}:{1}' -f "'$i,$lines[$i-1] }'` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'; "'$lines=Get-Content $p; for($i=621;$i -le $lines.Length;$i++){ '"'{0,4}:{1}' -f "'$i,$lines[$i-1] }'` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'; "'$lines=Get-Content $p; for($i=1;$i -le [Math]::Min(260,$lines.Length);$i++){ '"'{0,4}:{1}' -f "'$i,$lines[$i-1] }'` rejected: blocked by policy
codex
The shell is filtering some broader PowerShell snippets, so I’m switching to `rg -n -C` on the specific symbols. That’s enough to reconstruct the M6 control flow with line references and keeps the review precise.
2026-04-15T16:35:21.353704Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 6 \"def unpack\\(|def pack\\(|def iterate\\(|def _aggregate_shared\\(|def _compute_gap_metrics\\(|self.shared_keys|self.private_keys|current_round >= self.warmup_rounds\" \"D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py\""` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 6 \"def unpack\\(|def pack\\(|def iterate\\(|def _aggregate_shared\\(|def _compute_gap_metrics\\(|self.shared_keys|self.private_keys|current_round >= self.warmup_rounds\" \"D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py\"" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 6 \"class StyleModulator|style_modulator = StyleModulator|_init_agg_keys|if 'style_head' in k|use_domain =|sty_intra_protos|sty_cross_protos|style_gap_bank|_store_domain_protos|style_domain_protos\" \"D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py\"" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 8 \"def train\\(|loss_film|_delta_film_augment|output_film|lambda_sty_contrast|_style_domain_contrastive|_dual_alignment_loss|_infonce_global|_infonce_domain_aware\" \"D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py\"" in D:\桌面文件\联邦学习
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 6 \"class StyleModulator|style_modulator = StyleModulator|_init_agg_keys|if 'style_head' in k|use_domain =|sty_intra_protos|sty_cross_protos|style_gap_bank|_store_domain_protos|style_domain_protos\" \"D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py\""` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 6 \"def unpack\\(|def pack\\(|def iterate\\(|def _aggregate_shared\\(|def _compute_gap_metrics\\(|self.shared_keys|self.private_keys|current_round >= self.warmup_rounds\" \"D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py\""` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 8 \"def train\\(|loss_film|_delta_film_augment|output_film|lambda_sty_contrast|_style_domain_contrastive|_dual_alignment_loss|_infonce_global|_infonce_domain_aware\" \"D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py\""` rejected: blocked by policy
2026-04-15T16:35:21.382385Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 6 \"class StyleModulator|style_modulator = StyleModulator|_init_agg_keys|if 'style_head' in k|use_domain =|sty_intra_protos|sty_cross_protos|style_gap_bank|_store_domain_protos|style_domain_protos\" \"D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py\""` rejected: blocked by policy
2026-04-15T16:35:21.469974Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 8 \"def train\\(|loss_film|_delta_film_augment|output_film|lambda_sty_contrast|_style_domain_contrastive|_dual_alignment_loss|_infonce_global|_infonce_domain_aware\" \"D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py\""` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n "StyleModulator" FDSE_CVPR25/algorithm/feddsa_adaptive.py' in D:\桌面文件\联邦学习
 succeeded in 805ms:
72:class StyleModulator(nn.Module):
100:        self.style_modulator = StyleModulator(proj_dim)  # M6: shared via FedAvg
814:            # FiLM modulation via StyleModulator
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n -C 8 "def _delta_film_augment|delta_s =|delta_norm|gamma, beta, gate|Residual FiLM|same-class cross-domain" FDSE_CVPR25/algorithm/feddsa_adaptive.py' in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 8 \"def train\\(|loss_film|output_film|style_domain_contrastive|dual alignment|self.adaptive_mode in \\(5, 6\\)|self.adaptive_mode == 6\" FDSE_CVPR25/algorithm/feddsa_adaptive.py" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n -C 8 "StyleModulator|style_modulator = StyleModulator" FDSE_CVPR25/algorithm/feddsa_adaptive.py' in D:\桌面文件\联邦学习
 succeeded in 840ms:
78-        )
79-        self.gate_linear = nn.Linear(dim, 1)
80-
81-    def forward(self, delta_sty):
82-        """delta_sty: [B, dim] or [1, dim] normalized style difference."""
83-        params = self.film_net(delta_sty)
84-        gamma, beta = params.chunk(2, dim=-1)
85-        gate = torch.sigmoid(self.gate_linear(delta_sty))  # [B, 1]
86:        return gamma, beta, gate
87-
88-
89-class FedDSAModel(fuf.FModule):
90-    def __init__(self, num_classes=7, feat_dim=1024, proj_dim=128):
91-        super().__init__()
92-        self.encoder = AlexNetEncoder()
93-        self.semantic_head = nn.Sequential(
94-            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
--
767-            count += 1
768-
769-        if count > 0:
770-            loss = loss / count
771-        return loss
772-
773-    # ---- M6: Delta-FiLM Augmentation ----
774-
775:    def _delta_film_augment(self, model, z_sem, y):
776-        """M6: Cross-domain counterfactual style conditioning via Delta-FiLM.
777-        Uses normalized style DIFFERENCE (ext - local) as FiLM condition.
778-        Returns augmented z_sem or None if not enough protos.
779-        """
780-        device = z_sem.device
781-        B = z_sem.size(0)
782-
783-        # Collect cross-domain z_sty protos with class info
--
791-        for i in range(B):
792-            label = y[i].item()
793-
794-            # Need local z_sty proto for this class
795-            sty_local = self.sty_intra_protos.get(label)
796-            if sty_local is None:
797-                continue
798-
799:            # Find same-class cross-domain z_sty protos
800-            candidates = [p for c, cid, p in cross_list if c == label]
801-            if not candidates:
802-                continue
803-
804-            # Random pick one external style proto
805-            sty_ext = candidates[np.random.randint(len(candidates))]
806-
807-            # Delta: direction of style shift from local to external domain
808:            delta_s = sty_ext.to(device) - sty_local.to(device)
809:            delta_norm = delta_s.norm()
810:            if delta_norm < 1e-6:
811-                continue  # domains too similar, skip
812:            delta_s = delta_s / delta_norm  # normalize to unit direction
813-
814-            # FiLM modulation via StyleModulator
815:            gamma, beta, gate = model.style_modulator(delta_s.unsqueeze(0))
816-            gamma = gamma.squeeze(0)   # [proj_dim]
817-            beta = beta.squeeze(0)     # [proj_dim]
818-            gate = gate.squeeze()      # scalar
819-
820:            # Residual FiLM: z_sem + gate * (gamma * z_sem + beta)
821-            z_sem_film[i] = z_sem[i] + gate * (gamma * z_sem[i] + beta)
822-            augmented = True
823-
824-        return z_sem_film if augmented else None
825-
826-    # ---- Domain-Aware InfoNCE (M3) ----
827-
828-    def _infonce_domain_aware(self, z_sem, y):
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 815ms:
133-            'sdn': 5,    # style_dispatch_num
134-            'pd': 128,   # proj_dim
135-            'am': 0.05,  # aug_min
136-            'ax': 0.8,   # aug_max
137-            'ns': 0.05,  # noise_std
138-            'ed': 0.9,   # ema_decay
139-            'md': 1,     # adaptive_mode: 0=fixed_alpha,1=M1,2=M3,3=M1+M3,4=M4-dual
140-            'fa': 0.5,   # fixed_alpha_value
141:            'li': 1.0,   # lambda_intra (M4 dual alignment)
142:            'lc': 0.5,   # lambda_cross (M4 dual alignment)
143-            'lsc': 0.5,  # lambda_sty_contrast (M5 style contrastive)
144-        })
145-        # Readable aliases so all downstream code is unchanged
146-        self.lambda_orth = float(self.lo)
147-        self.lambda_hsic = float(self.lh)
148-        self.lambda_sem = float(self.ls)
149-        self.warmup_rounds = int(self.wr)
150-        self.style_dispatch_num = int(self.sdn)
--
231-                if cid == client_id:
232-                    intra_protos[cls] = proto
233-                else:
234-                    cross_protos[(cls, cid)] = proto
235-
236-        # M5: split style_domain_protos the same way
237-        sty_intra_protos = {}
238-        sty_cross_protos = {}
239:        if self.adaptive_mode in (5, 6) and self.style_domain_protos:
240-            for (cls, cid), proto in self.style_domain_protos.items():
241-                if cid == client_id:
242-                    sty_intra_protos[cls] = proto
243-                else:
244-                    sty_cross_protos[(cls, cid)] = proto
245-
246-        return {
247-            'model': copy.deepcopy(self.model),
--
282-        if self.adaptive_mode in (1, 3):
283-            self._compute_gap_metrics()
284-
285-        # 4. Store domain protos (M3 / M4 / M5 / M6)
286-        if self.adaptive_mode in (2, 3, 4, 5, 6):
287-            self._store_domain_protos(protos_list)
288-
289-        # 5. Store style domain protos (M5 / M6)
290:        if self.adaptive_mode in (5, 6):
291-            sty_protos_list = res.get('sty_protos', [None] * len(self.received_clients))
292-            for cid, sty_protos in zip(self.received_clients, sty_protos_list):
293-                if sty_protos is None:
294-                    continue
295-                for c, proto in sty_protos.items():
296-                    self.style_domain_protos[(c, cid)] = proto
297-
298-        # 5. Aggregate global protos (always, as fallback)
--
436-    def pack(self):
437-        result = {
438-            'model': copy.deepcopy(self.model.to('cpu')),
439-            'protos': self._local_protos,
440-            'proto_counts': self._local_proto_counts,
441-            'style_stats': self._local_style_stats,        # h-space (for AdaIN bank)
442-            'style_gap_stats': self._local_style_gap_stats, # z_sty-space (for gap metric)
443-        }
444:        if self.adaptive_mode in (5, 6):
445-            result['sty_protos'] = getattr(self, '_local_sty_protos', None)
446-        return result
447-
448-    @fuf.with_multi_gpus
449:    def train(self, model, *args, **kwargs):
450-        model.train()
451-        optimizer = self.calculator.get_optimizer(
452-            model, lr=self.learning_rate,
453-            weight_decay=self.weight_decay, momentum=self.momentum
454-        )
455-
456-        aux_w = min(1.0, self.current_round / max(self.warmup_rounds, 1))
457-        use_adaptive_aug = self.adaptive_mode in (1, 3)
--
511-                loss_sem = self._infonce_global(z_sem, y)
512-            elif use_domain_protos and self.domain_protos and len(self.domain_protos) >= 2:
513-                loss_sem = self._infonce_domain_aware(z_sem, y)
514-            elif self.global_protos and len(self.global_protos) >= 2:
515-                loss_sem = self._infonce_global(z_sem, y)
516-
517-            # M5/M6: style domain contrastive loss
518-            loss_sty_con = torch.tensor(0.0, device=x.device)
519:            if self.adaptive_mode in (5, 6) and self.sty_intra_protos and self.sty_cross_protos:
520:                loss_sty_con = self._style_domain_contrastive(z_sty, y)
521-
522-            # M6: Delta-FiLM augmentation (training only)
523:            loss_film = torch.tensor(0.0, device=x.device)
524:            if self.adaptive_mode == 6 and self.sty_intra_protos and self.sty_cross_protos \
525-               and self.current_round >= self.warmup_rounds:
526-                z_sem_film = self._delta_film_augment(model, z_sem, y)
527-                if z_sem_film is not None:
528:                    output_film = model.head(z_sem_film)
529:                    loss_film = self.loss_fn(output_film, y)
530-
531-            if self.adaptive_mode in (4, 5, 6):
532-                loss = loss_task + loss_aug + \
533-                       aux_w * self.lambda_orth * loss_orth + \
534-                       aux_w * self.lambda_hsic * loss_hsic + \
535-                       aux_w * self.lambda_intra * loss_dual_intra + \
536-                       aux_w * self.lambda_cross * loss_dual_cross + \
537-                       aux_w * self.lambda_sty_contrast * loss_sty_con + \
538-                       aux_w * self.lambda_sem * loss_sem + \
539:                       aux_w * loss_film
540-            else:
541-                loss = loss_task + loss_aug + \
542-                       aux_w * self.lambda_orth * loss_orth + \
543-                       aux_w * self.lambda_hsic * loss_hsic + \
544-                       aux_w * self.lambda_sem * loss_sem
545-
546-            loss.backward()
547-            if self.clip_grad > 0:
--
561-                        if c not in proto_sum:
562-                            proto_sum[c] = z_det[i].clone()
563-                            proto_count[c] = 1
564-                        else:
565-                            proto_sum[c] += z_det[i]
566-                            proto_count[c] += 1
567-
568-                    # M5/M6: z_sty prototype accumulation
569:                    if self.adaptive_mode in (5, 6):
570-                        z_sty_cpu = z_sty_det.cpu()
571-                        for i, label in enumerate(y):
572-                            c = label.item()
573-                            if c not in sty_proto_sum:
574-                                sty_proto_sum[c] = z_sty_cpu[i].clone()
575-                                sty_proto_count[c] = 1
576-                            else:
577-                                sty_proto_sum[c] += z_sty_cpu[i]
--
603-                        zsty_sq_sum += batch_sq_sty * b
604-                        zsty_n += b
605-
606-        # Store for pack()
607-        self._local_protos = {c: proto_sum[c] / proto_count[c] for c in proto_sum}
608-        self._local_proto_counts = proto_count
609-
610-        # M5/M6: z_sty prototypes
611:        if self.adaptive_mode in (5, 6) and sty_proto_sum:
612-            self._local_sty_protos = {c: sty_proto_sum[c] / sty_proto_count[c] for c in sty_proto_sum}
613-        else:
614-            self._local_sty_protos = None
615-
616-        # h-space style stats
617-        if h_style_n > 1:
618-            mu_h = h_style_sum / h_style_n
619-            var_h = h_style_sq_sum / h_style_n - mu_h ** 2
--
714-
715-            if cross_count > 0:
716-                loss_cross = loss_cross / cross_count
717-
718-        return loss_intra, loss_cross
719-
720-    # ---- M5: Style Domain Contrastive ----
721-
722:    def _style_domain_contrastive(self, z_sty, y):
723-        """M5: Domain-contrastive loss on z_sty.
724-        Same-domain same-class z_sty protos = positive (pull together).
725-        Cross-domain z_sty protos = negative (push apart).
726-        Effect: reinforces style head to capture domain-specific information.
727-        """
728-        device = z_sty.device
729-
730-        # Build: own-domain z_sty protos are positive, cross-domain are negative
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 829ms:
64-        x = torch.flatten(x, 1)
65-        x = self.bn6(self.fc1(x))
66-        x = self.relu(x)
67-        x = self.bn7(self.fc2(x))
68-        x = self.relu(x)
69-        return x  # [B, 1024]
70-
71-
72:class StyleModulator(nn.Module):
73-    """M6: Delta-FiLM — maps style difference to modulation parameters."""
74-    def __init__(self, dim=128):
75-        super().__init__()
76-        self.film_net = nn.Sequential(
77-            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim * 2)
78-        )
79-        self.gate_linear = nn.Linear(dim, 1)
80-
--
92-        self.encoder = AlexNetEncoder()
93-        self.semantic_head = nn.Sequential(
94-            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
95-        )
96-        self.style_head = nn.Sequential(
97-            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
98-        )
99-        self.head = nn.Linear(proj_dim, num_classes)
100:        self.style_modulator = StyleModulator(proj_dim)  # M6: shared via FedAvg
101-
102-    def forward(self, x):
103-        h = self.encoder(x)
104-        z_sem = self.semantic_head(h)
105-        return self.head(z_sem)
106-
107-    def encode(self, x):
108-        return self.encoder(x)
--
806-
807-            # Delta: direction of style shift from local to external domain
808-            delta_s = sty_ext.to(device) - sty_local.to(device)
809-            delta_norm = delta_s.norm()
810-            if delta_norm < 1e-6:
811-                continue  # domains too similar, skip
812-            delta_s = delta_s / delta_norm  # normalize to unit direction
813-
814:            # FiLM modulation via StyleModulator
815-            gamma, beta, gate = model.style_modulator(delta_s.unsqueeze(0))
816-            gamma = gamma.squeeze(0)   # [proj_dim]
817-            beta = beta.squeeze(0)     # [proj_dim]
818-            gate = gate.squeeze()      # scalar
819-
820-            # Residual FiLM: z_sem + gate * (gamma * z_sem + beta)
821-            z_sem_film[i] = z_sem[i] + gate * (gamma * z_sem[i] + beta)
822-            augmented = True
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 8 \"def unpack\\(|def pack\\(|if 'style_head' in key and self.adaptive_mode not in \\(5, 6\\)|load global weights\" FDSE_CVPR25/algorithm/feddsa_adaptive.py" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 8 \"def iterate\\(|style_gap_bank|_compute_gap_metrics|gap_normalized|style_bank|dispatched_styles\" FDSE_CVPR25/algorithm/feddsa_adaptive.py" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n -C 8 "def _init_agg_keys|private_keys|shared_keys|style_head shared" FDSE_CVPR25/algorithm/feddsa_adaptive.py' in D:\桌面文件\联邦学习
 succeeded in 841ms:
189-            c.aug_max = self.aug_max
190-            c.noise_std = self.noise_std
191-            c.adaptive_mode = int(self.adaptive_mode)
192-            c.fixed_alpha_value = float(self.fixed_alpha_value)
193-            c.lambda_intra = float(self.lambda_intra)
194-            c.lambda_cross = float(self.lambda_cross)
195-            c.lambda_sty_contrast = float(self.lambda_sty_contrast)
196-
197:    def _init_agg_keys(self):
198-        all_keys = list(self.model.state_dict().keys())
199:        self.private_keys = set()
200-        for k in all_keys:
201-            if 'style_head' in k and self.adaptive_mode not in (5, 6):
202:                # M5/M6: style_head shared so z_sty is in a common space across clients
203:                self.private_keys.add(k)
204-            elif 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
205:                self.private_keys.add(k)
206:        self.shared_keys = [k for k in all_keys if k not in self.private_keys]
207-
208-    def pack(self, client_id, mtype=0):
209-        # Dispatch style bank (h-space, for AdaIN augmentation)
210-        dispatched_styles = None
211-        if len(self.style_bank) > 0 and self.current_round >= self.warmup_rounds:
212-            available = {cid: s for cid, s in self.style_bank.items() if cid != client_id}
213-            if len(available) == 0:
214-                available = self.style_bank
--
336-        weights = np.array(
337-            [len(self.clients[cid].train_data) for cid in self.received_clients], dtype=float
338-        )
339-        weights /= weights.sum()
340-
341-        global_dict = self.model.state_dict()
342-        model_dicts = [m.state_dict() for m in models]
343-
344:        for k in self.shared_keys:
345-            if 'num_batches_tracked' in k:
346-                continue
347-            global_dict[k] = sum(w * md[k] for w, md in zip(weights, model_dicts))
348-
349-        self.model.load_state_dict(global_dict, strict=False)
350-
351-    def _store_domain_protos(self, protos_list):
352-        for cid, protos in zip(self.received_clients, protos_list):
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 910ms:
156-        self.adaptive_mode = int(self.md)
157-        self.fixed_alpha_value = float(self.fa)
158-        self.lambda_intra = float(self.li)
159-        self.lambda_cross = float(self.lc)
160-        self.lambda_sty_contrast = float(self.lsc)
161-        self.sample_option = 'full'
162-
163-        # Style bank: h-space stats for AdaIN augmentation
164:        self.style_bank = {}  # client_id -> (mu_h, sigma_h)
165-        # Style gap bank: z_sty-space stats for gap measurement (dual bank, R2 review)
166:        self.style_gap_bank = {}  # client_id -> (mu_zsty, sigma_zsty)
167-
168-        # Prototypes
169-        self.global_semantic_protos = {}  # class -> avg proto
170-        self.domain_protos = {}  # (class, client_id) -> z_sem proto (M3/M4)
171-        self.style_domain_protos = {}  # (class, client_id) -> z_sty proto (M5)
172-
173-        # Gap metrics (EMA smoothed)
174-        self.client_gaps = {}  # client_id -> normalized gap [0,1]
--
202-                # M5/M6: style_head shared so z_sty is in a common space across clients
203-                self.private_keys.add(k)
204-            elif 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
205-                self.private_keys.add(k)
206-        self.shared_keys = [k for k in all_keys if k not in self.private_keys]
207-
208-    def pack(self, client_id, mtype=0):
209-        # Dispatch style bank (h-space, for AdaIN augmentation)
210:        dispatched_styles = None
211:        if len(self.style_bank) > 0 and self.current_round >= self.warmup_rounds:
212:            available = {cid: s for cid, s in self.style_bank.items() if cid != client_id}
213-            if len(available) == 0:
214:                available = self.style_bank
215-            n = min(self.style_dispatch_num, len(available))
216-            keys = list(available.keys())
217-            chosen = np.random.choice(keys, n, replace=False)
218:            dispatched_styles = [available[k] for k in chosen]
219-
220-        # Gap for this client (M1)
221:        gap_normalized = self.client_gaps.get(client_id, 0.5)
222-
223-        # Choose which protos to send based on mode
224-        use_domain = self.adaptive_mode in (2, 3, 4, 5, 6)
225-
226-        # M4/M5: split domain_protos into intra (own-domain) + cross (other-domain)
227-        intra_protos = {}
228-        cross_protos = {}
229-        if self.adaptive_mode in (4, 5, 6) and self.domain_protos:
--
246-        return {
247-            'model': copy.deepcopy(self.model),
248-            'global_protos': copy.deepcopy(self.global_semantic_protos),
249-            'domain_protos': copy.deepcopy(self.domain_protos) if use_domain else {},
250-            'intra_protos': copy.deepcopy(intra_protos),
251-            'cross_protos': copy.deepcopy(cross_protos),
252-            'sty_intra_protos': copy.deepcopy(sty_intra_protos),
253-            'sty_cross_protos': copy.deepcopy(sty_cross_protos),
254:            'style_bank': dispatched_styles,
255-            'current_round': self.current_round,
256:            'gap_normalized': gap_normalized,
257-        }
258-
259:    def iterate(self):
260-        self.selected_clients = self.sample()
261-        res = self.communicate(self.selected_clients)
262-
263-        models = res['model']
264-        protos_list = res['protos']
265-        proto_counts_list = res['proto_counts']
266-        style_stats_list = res['style_stats']
267-        style_gap_stats_list = res['style_gap_stats']
--
269-        # 1. Aggregate shared parameters (FedAvg)
270-        self._aggregate_shared(models)
271-
272-        # 2. Collect dual style banks
273-        for cid, h_style, zsty_style in zip(
274-            self.received_clients, style_stats_list, style_gap_stats_list
275-        ):
276-            if h_style is not None:
277:                self.style_bank[cid] = h_style
278-            if zsty_style is not None:
279:                self.style_gap_bank[cid] = zsty_style
280-
281-        # 3. Compute gap metrics from z_sty bank (M1)
282-        if self.adaptive_mode in (1, 3):
283:            self._compute_gap_metrics()
284-
285-        # 4. Store domain protos (M3 / M4 / M5 / M6)
286-        if self.adaptive_mode in (2, 3, 4, 5, 6):
287-            self._store_domain_protos(protos_list)
288-
289-        # 5. Store style domain protos (M5 / M6)
290-        if self.adaptive_mode in (5, 6):
291-            sty_protos_list = res.get('sty_protos', [None] * len(self.received_clients))
--
293-                if sty_protos is None:
294-                    continue
295-                for c, proto in sty_protos.items():
296-                    self.style_domain_protos[(c, cid)] = proto
297-
298-        # 5. Aggregate global protos (always, as fallback)
299-        self._aggregate_protos(protos_list, proto_counts_list)
300-
301:    def _compute_gap_metrics(self):
302-        """Compute per-client domain gap using z_sty statistics (pure domain signal)."""
303:        if len(self.style_gap_bank) < 2:
304-            return
305-
306:        all_mu = torch.stack([s[0] for s in self.style_gap_bank.values()])
307:        all_sigma = torch.stack([s[1] for s in self.style_gap_bank.values()])
308-
309-        mu_center = all_mu.mean(dim=0)
310-        sigma_center = all_sigma.mean(dim=0)
311-
312-        raw_gaps = {}
313:        for cid, (mu, sigma) in self.style_gap_bank.items():
314-            gap = ((mu - mu_center) ** 2).sum() + ((sigma - sigma_center) ** 2).sum()
315-            raw_gaps[cid] = gap.item()
316-
317-        # EMA z-score normalization
318-        gap_vals = list(raw_gaps.values())
319-        batch_mean = np.mean(gap_vals)
320-        batch_std = np.std(gap_vals) + 1e-8
321-
--
377-# Client — Adaptive Augmentation + Domain-Aware Alignment
378-# ============================================================
379-
380-class Client(flgo.algorithm.fedbase.BasicClient):
381-    def initialize(self):
382-        self.model = copy.deepcopy(self.server.model).to('cpu')
383-        self.loss_fn = nn.CrossEntropyLoss()
384-        self.current_round = 0
385:        self.local_style_bank = None
386-        self.global_protos = None
387-        self.domain_protos = None
388-        self.intra_protos = {}
389-        self.cross_protos = {}
390-        self.sty_intra_protos = {}
391-        self.sty_cross_protos = {}
392:        self.gap_normalized = 0.5
393-
394-    def reply(self, svr_pkg):
395:        model, global_protos, domain_protos, style_bank, current_round, gap_normalized, intra_protos, cross_protos, sty_intra, sty_cross = self.unpack(svr_pkg)
396-        self.current_round = current_round
397-        self.global_protos = global_protos
398-        self.domain_protos = domain_protos
399:        self.local_style_bank = style_bank
400:        self.gap_normalized = gap_normalized
401-        self.intra_protos = intra_protos
402-        self.cross_protos = cross_protos
403-        self.sty_intra_protos = sty_intra
404-        self.sty_cross_protos = sty_cross
405-        self.train(model)
406-        return self.pack()
407-
408-    def unpack(self, svr_pkg):
--
419-                if 'bn' in key.lower() and ('running_' in key or 'num_batches_tracked' in key):
420-                    continue
421-                new_dict[key] = global_dict[key]
422-            self.model.load_state_dict(new_dict)
423-        return (
424-            self.model,
425-            svr_pkg['global_protos'],
426-            svr_pkg.get('domain_protos', {}),
427:            svr_pkg['style_bank'],
428-            svr_pkg['current_round'],
429:            svr_pkg.get('gap_normalized', 0.5),
430-            svr_pkg.get('intra_protos', {}),
431-            svr_pkg.get('cross_protos', {}),
432-            svr_pkg.get('sty_intra_protos', {}),
433-            svr_pkg.get('sty_cross_protos', {}),
434-        )
435-
436-    def pack(self):
437-        result = {
--
484-            z_sty = model.get_style(h)     # [B, proj_dim]
485-
486-            # Loss 1: Task CE on semantic path
487-            output = model.head(z_sem)
488-            loss_task = self.loss_fn(output, y)
489-
490-            # Loss 2: Augmented CE (adaptive or fixed alpha)
491-            loss_aug = torch.tensor(0.0, device=x.device)
492:            if self.local_style_bank is not None and self.current_round >= self.warmup_rounds:
493-                h_aug = self._style_augment(h, use_adaptive_aug)
494-                z_sem_aug = model.get_semantic(h_aug)
495-                output_aug = model.head(z_sem_aug)
496-                loss_aug = self.loss_fn(output_aug, y)
497-
498-            # Loss 3: Decoupling (orthogonal + HSIC)
499-            loss_orth, loss_hsic = self._decouple_loss(z_sem, z_sty)
500-
--
628-            self._local_style_gap_stats = (mu_sty, var_sty.clamp(min=1e-6).sqrt())
629-        else:
630-            self._local_style_gap_stats = None
631-
632-    # ---- Augmentation ----
633-
634-    def _style_augment(self, h, use_adaptive):
635-        """AdaIN augmentation with adaptive or fixed alpha."""
636:        idx = np.random.randint(0, len(self.local_style_bank))
637:        mu_ext, sigma_ext = self.local_style_bank[idx]
638-        mu_ext = mu_ext.to(h.device)
639-        sigma_ext = sigma_ext.to(h.device)
640-
641-        mu_local = h.mean(dim=0)
642-        sigma_local = h.std(dim=0, unbiased=False).clamp(min=1e-6)
643-
644-        # Full AdaIN
645-        h_norm = (h - mu_local) / sigma_local
646-        h_adain = h_norm * sigma_ext + mu_ext
647-
648-        if use_adaptive:
649-            # M1: gap → adaptive alpha + stochastic jitter
650:            alpha_mean = self.aug_min + (self.aug_max - self.aug_min) * self.gap_normalized
651-            alpha = float(np.clip(
652-                alpha_mean + np.random.normal(0, self.noise_std),
653-                self.aug_min, self.aug_max
654-            ))
655-        elif self.adaptive_mode == 0:
656-            # Fixed alpha mode (for baselines EXP-072a/b/c)
657-            alpha = self.fixed_alpha_value
658-        else:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 894ms:
200-        for k in all_keys:
201-            if 'style_head' in k and self.adaptive_mode not in (5, 6):
202-                # M5/M6: style_head shared so z_sty is in a common space across clients
203-                self.private_keys.add(k)
204-            elif 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
205-                self.private_keys.add(k)
206-        self.shared_keys = [k for k in all_keys if k not in self.private_keys]
207-
208:    def pack(self, client_id, mtype=0):
209-        # Dispatch style bank (h-space, for AdaIN augmentation)
210-        dispatched_styles = None
211-        if len(self.style_bank) > 0 and self.current_round >= self.warmup_rounds:
212-            available = {cid: s for cid, s in self.style_bank.items() if cid != client_id}
213-            if len(available) == 0:
214-                available = self.style_bank
215-            n = min(self.style_dispatch_num, len(available))
216-            keys = list(available.keys())
--
400-        self.gap_normalized = gap_normalized
401-        self.intra_protos = intra_protos
402-        self.cross_protos = cross_protos
403-        self.sty_intra_protos = sty_intra
404-        self.sty_cross_protos = sty_cross
405-        self.train(model)
406-        return self.pack()
407-
408:    def unpack(self, svr_pkg):
409-        global_model = svr_pkg['model']
410-        if self.model is None:
411-            self.model = global_model
412-        else:
413-            new_dict = self.model.state_dict()
414-            global_dict = global_model.state_dict()
415-            for key in new_dict.keys():
416:                if 'style_head' in key and self.adaptive_mode not in (5, 6):
417:                    # M5/M6: style_head is shared, load global weights
418-                    continue
419-                if 'bn' in key.lower() and ('running_' in key or 'num_batches_tracked' in key):
420-                    continue
421-                new_dict[key] = global_dict[key]
422-            self.model.load_state_dict(new_dict)
423-        return (
424-            self.model,
425-            svr_pkg['global_protos'],
--
428-            svr_pkg['current_round'],
429-            svr_pkg.get('gap_normalized', 0.5),
430-            svr_pkg.get('intra_protos', {}),
431-            svr_pkg.get('cross_protos', {}),
432-            svr_pkg.get('sty_intra_protos', {}),
433-            svr_pkg.get('sty_cross_protos', {}),
434-        )
435-
436:    def pack(self):
437-        result = {
438-            'model': copy.deepcopy(self.model.to('cpu')),
439-            'protos': self._local_protos,
440-            'proto_counts': self._local_proto_counts,
441-            'style_stats': self._local_style_stats,        # h-space (for AdaIN bank)
442-            'style_gap_stats': self._local_style_gap_stats, # z_sty-space (for gap metric)
443-        }
444-        if self.adaptive_mode in (5, 6):
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I’ve now pinned the M6 branch: server stores detached `z_sty` class prototypes from the previous round, splits them into intra/cross sets, and the client uses those fixed tensors to condition `style_modulator`. I’m checking the remaining gradient and inheritance details against the actual detach points so the final review distinguishes intended behavior from what can really receive gradients.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 8 \"def _style_domain_contrastive|neg_entries|pos_proto|F.normalize\\(z_sty|all_logits\" FDSE_CVPR25/algorithm/feddsa_adaptive.py" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n -C 8 \"with torch.no_grad\\(|z_sty_det = z_sty.detach\\(|z_det = z_sem.detach\\(|_local_sty_protos|style_gap_stats|batch_mu_sty\" FDSE_CVPR25/algorithm/feddsa_adaptive.py" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg -n -C 8 "def _dual_alignment_loss|loss_intra|loss_cross|proto_matrix|proto_labels" FDSE_CVPR25/algorithm/feddsa_adaptive.py' in D:\桌面文件\联邦学习
 succeeded in 863ms:
714-
715-            if cross_count > 0:
716-                loss_cross = loss_cross / cross_count
717-
718-        return loss_intra, loss_cross
719-
720-    # ---- M5: Style Domain Contrastive ----
721-
722:    def _style_domain_contrastive(self, z_sty, y):
723-        """M5: Domain-contrastive loss on z_sty.
724-        Same-domain same-class z_sty protos = positive (pull together).
725-        Cross-domain z_sty protos = negative (push apart).
726-        Effect: reinforces style head to capture domain-specific information.
727-        """
728-        device = z_sty.device
729-
730-        # Build: own-domain z_sty protos are positive, cross-domain are negative
731-        # For each sample z_sty_i with label y_i:
732-        #   positive = sty_intra_protos[y_i] (same domain, same class)
733-        #   negatives = all sty_cross_protos (any class, other domains)
734-
735-        # Gather all cross-domain style protos as negatives
736:        neg_entries = []
737-        for key in sorted(self.sty_cross_protos.keys()):
738:            neg_entries.append(self.sty_cross_protos[key])
739-
740:        if len(neg_entries) == 0:
741-            return torch.tensor(0.0, device=device)
742-
743:        neg_matrix = torch.stack([p.to(device) for p in neg_entries])  # [N_neg, 128]
744-
745-        B = z_sty.size(0)
746-        loss = torch.tensor(0.0, device=device)
747-        count = 0
748-
749-        for i in range(B):
750-            label = y[i].item()
751-            if label not in self.sty_intra_protos:
752-                continue
753-
754:            pos_proto = self.sty_intra_protos[label].to(device)  # [128]
755-
756-            # Cosine similarity
757:            z_n = F.normalize(z_sty[i:i+1], dim=1)    # [1, 128]
758:            pos_n = F.normalize(pos_proto.unsqueeze(0), dim=1)  # [1, 128]
759-            neg_n = F.normalize(neg_matrix, dim=1)      # [N_neg, 128]
760-
761-            pos_sim = (z_n @ pos_n.T).view(-1)[0] / self.tau  # scalar
762-            neg_sims = (z_n @ neg_n.T).view(-1) / self.tau  # [N_neg], safe for any N_neg
763-
764-            # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
765:            all_logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])  # [1 + N_neg]
766:            loss = loss + (-pos_sim + torch.logsumexp(all_logits, dim=0))
767-            count += 1
768-
769-        if count > 0:
770-            loss = loss / count
771-        return loss
772-
773-    # ---- M6: Delta-FiLM Augmentation ----
774-
--
892-        valid_t = torch.tensor(valid, device=z_sem.device)
893-        targets_t = torch.tensor(targets, device=z_sem.device, dtype=torch.long)
894-        return F.cross_entropy(logits[valid_t], targets_t)
895-
896-    # ---- Decoupling losses ----
897-
898-    def _decouple_loss(self, z_sem, z_sty):
899-        z_sem_n = F.normalize(z_sem, dim=1)
900:        z_sty_n = F.normalize(z_sty, dim=1)
901-        cos = (z_sem_n * z_sty_n).sum(dim=1)
902-        loss_orth = (cos ** 2).mean()
903-
904-        loss_hsic = self._hsic(z_sem, z_sty)
905-        return loss_orth, loss_hsic
906-
907-    def _hsic(self, x, y):
908-        n = x.size(0)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 823ms:
259-    def iterate(self):
260-        self.selected_clients = self.sample()
261-        res = self.communicate(self.selected_clients)
262-
263-        models = res['model']
264-        protos_list = res['protos']
265-        proto_counts_list = res['proto_counts']
266-        style_stats_list = res['style_stats']
267:        style_gap_stats_list = res['style_gap_stats']
268-
269-        # 1. Aggregate shared parameters (FedAvg)
270-        self._aggregate_shared(models)
271-
272-        # 2. Collect dual style banks
273-        for cid, h_style, zsty_style in zip(
274:            self.received_clients, style_stats_list, style_gap_stats_list
275-        ):
276-            if h_style is not None:
277-                self.style_bank[cid] = h_style
278-            if zsty_style is not None:
279-                self.style_gap_bank[cid] = zsty_style
280-
281-        # 3. Compute gap metrics from z_sty bank (M1)
282-        if self.adaptive_mode in (1, 3):
--
434-        )
435-
436-    def pack(self):
437-        result = {
438-            'model': copy.deepcopy(self.model.to('cpu')),
439-            'protos': self._local_protos,
440-            'proto_counts': self._local_proto_counts,
441-            'style_stats': self._local_style_stats,        # h-space (for AdaIN bank)
442:            'style_gap_stats': self._local_style_gap_stats, # z_sty-space (for gap metric)
443-        }
444-        if self.adaptive_mode in (5, 6):
445:            result['sty_protos'] = getattr(self, '_local_sty_protos', None)
446-        return result
447-
448-    @fuf.with_multi_gpus
449-    def train(self, model, *args, **kwargs):
450-        model.train()
451-        optimizer = self.calculator.get_optimizer(
452-            model, lr=self.learning_rate,
453-            weight_decay=self.weight_decay, momentum=self.momentum
--
545-
546-            loss.backward()
547-            if self.clip_grad > 0:
548-                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
549-            optimizer.step()
550-
551-            # Accumulate stats (last epoch)
552-            if step >= num_steps - len(self.train_data) // self.batch_size - 1:
553:                with torch.no_grad():
554:                    z_det = z_sem.detach().cpu()
555-                    h_det = h.detach()
556:                    z_sty_det = z_sty.detach()
557-
558-                    # Prototype accumulation (z_sem)
559-                    for i, label in enumerate(y):
560-                        c = label.item()
561-                        if c not in proto_sum:
562-                            proto_sum[c] = z_det[i].clone()
563-                            proto_count[c] = 1
564-                        else:
--
587-                        h_style_sq_sum = batch_sq_h * b
588-                        h_style_n = b
589-                    else:
590-                        h_style_sum += batch_mu_h * b
591-                        h_style_sq_sum += batch_sq_h * b
592-                        h_style_n += b
593-
594-                    # z_sty-space stats (for gap metric)
595:                    batch_mu_sty = z_sty_det.mean(dim=0).cpu()
596-                    batch_sq_sty = (z_sty_det ** 2).mean(dim=0).cpu()
597-                    if zsty_sum is None:
598:                        zsty_sum = batch_mu_sty * b
599-                        zsty_sq_sum = batch_sq_sty * b
600-                        zsty_n = b
601-                    else:
602:                        zsty_sum += batch_mu_sty * b
603-                        zsty_sq_sum += batch_sq_sty * b
604-                        zsty_n += b
605-
606-        # Store for pack()
607-        self._local_protos = {c: proto_sum[c] / proto_count[c] for c in proto_sum}
608-        self._local_proto_counts = proto_count
609-
610-        # M5/M6: z_sty prototypes
611-        if self.adaptive_mode in (5, 6) and sty_proto_sum:
612:            self._local_sty_protos = {c: sty_proto_sum[c] / sty_proto_count[c] for c in sty_proto_sum}
613-        else:
614:            self._local_sty_protos = None
615-
616-        # h-space style stats
617-        if h_style_n > 1:
618-            mu_h = h_style_sum / h_style_n
619-            var_h = h_style_sq_sum / h_style_n - mu_h ** 2
620-            self._local_style_stats = (mu_h, var_h.clamp(min=1e-6).sqrt())
621-        else:
622-            self._local_style_stats = None
623-
624-        # z_sty-space style stats (for gap)
625-        if zsty_n > 1:
626-            mu_sty = zsty_sum / zsty_n
627-            var_sty = zsty_sq_sum / zsty_n - mu_sty ** 2
628:            self._local_style_gap_stats = (mu_sty, var_sty.clamp(min=1e-6).sqrt())
629-        else:
630:            self._local_style_gap_stats = None
631-
632-    # ---- Augmentation ----
633-
634-    def _style_augment(self, h, use_adaptive):
635-        """AdaIN augmentation with adaptive or fixed alpha."""
636-        idx = np.random.randint(0, len(self.local_style_bank))
637-        mu_ext, sigma_ext = self.local_style_bank[idx]
638-        mu_ext = mu_ext.to(h.device)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 794ms:
659-            # M3-only mode: use original Beta mixing
660-            alpha = np.random.beta(0.1, 0.1)
661-
662-        h_aug = alpha * h_adain + (1 - alpha) * h
663-        return h_aug
664-
665-    # ---- M4: Dual Alignment (intra-domain + cross-domain) ----
666-
667:    def _dual_alignment_loss(self, z_sem, y):
668-        """M4: Decomposed alignment — L_intra (own-domain cosine) + L_cross (cross-domain InfoNCE)."""
669-        device = z_sem.device
670-        B = z_sem.size(0)
671-
672-        # --- L_intra: pull toward own-domain prototype ---
673:        loss_intra = torch.tensor(0.0, device=device)
674-        intra_count = 0
675-        for i in range(B):
676-            label = y[i].item()
677-            if label in self.intra_protos:
678-                proto = self.intra_protos[label].to(device)
679-                sim = F.cosine_similarity(z_sem[i:i+1], proto.unsqueeze(0)).squeeze()
680:                loss_intra = loss_intra + (1.0 - sim)
681-                intra_count += 1
682-        if intra_count > 0:
683:            loss_intra = loss_intra / intra_count
684-
685-        # --- L_cross: InfoNCE with cross-domain protos (own domain excluded) ---
686:        loss_cross = torch.tensor(0.0, device=device)
687-
688-        # Build cross-domain prototype matrix
689-        cross_entries = []
690-        cross_classes = []
691-        for (cls, cid) in sorted(self.cross_protos.keys()):
692-            cross_entries.append(self.cross_protos[(cls, cid)])
693-            cross_classes.append(cls)
694-
695-        if len(cross_entries) >= 2:
696:            proto_matrix = torch.stack([p.to(device) for p in cross_entries])
697:            proto_labels = torch.tensor(cross_classes, device=device)
698-
699-            z_n = F.normalize(z_sem, dim=1)
700:            p_n = F.normalize(proto_matrix, dim=1)
701-            logits = z_n @ p_n.T / self.tau  # tau=0.2
702-
703-            cross_count = 0
704-            for i in range(B):
705-                label = y[i].item()
706:                pos_mask = (proto_labels == label)
707-                n_pos = pos_mask.sum().item()
708-                if n_pos == 0:
709-                    continue
710-                log_denom = torch.logsumexp(logits[i], dim=0)
711-                pos_logits = logits[i][pos_mask]
712:                loss_cross = loss_cross + (-pos_logits + log_denom).sum() / n_pos
713-                cross_count += 1
714-
715-            if cross_count > 0:
716:                loss_cross = loss_cross / cross_count
717-
718:        return loss_intra, loss_cross
719-
720-    # ---- M5: Style Domain Contrastive ----
721-
722-    def _style_domain_contrastive(self, z_sty, y):
723-        """M5: Domain-contrastive loss on z_sty.
724-        Same-domain same-class z_sty protos = positive (pull together).
725-        Cross-domain z_sty protos = negative (push apart).
726-        Effect: reinforces style head to capture domain-specific information.
--
833-            cls = key[0]
834-            proto = self.domain_protos[key]
835-            entries.append(proto)
836-            entry_classes.append(cls)
837-
838-        if len(entries) < 2:
839-            return self._infonce_global(z_sem, y)
840-
841:        proto_matrix = torch.stack([p.to(z_sem.device) for p in entries])
842:        proto_labels = torch.tensor(entry_classes, device=z_sem.device)
843-
844-        z_n = F.normalize(z_sem, dim=1)
845:        p_n = F.normalize(proto_matrix, dim=1)
846-        logits = z_n @ p_n.T / self.tau
847-
848-        loss = torch.tensor(0.0, device=z_sem.device)
849-        count = 0
850-
851-        for i in range(y.size(0)):
852-            label = y[i].item()
853:            pos_mask = (proto_labels == label)
854-            n_pos = pos_mask.sum().item()
855-            if n_pos == 0:
856-                continue
857-
858-            log_denom = torch.logsumexp(logits[i], dim=0)
859-            pos_logits = logits[i][pos_mask]
860-            loss += (-pos_logits + log_denom).sum() / n_pos
861-            count += 1
--
866-
867-    # ---- Global InfoNCE (fallback) ----
868-
869-    def _infonce_global(self, z_sem, y):
870-        available = sorted([c for c, p in self.global_protos.items() if p is not None])
871-        if len(available) < 2:
872-            return torch.tensor(0.0, device=z_sem.device)
873-
874:        proto_matrix = torch.stack([self.global_protos[c].to(z_sem.device) for c in available])
875-        class_to_idx = {c: i for i, c in enumerate(available)}
876-
877-        z_n = F.normalize(z_sem, dim=1)
878:        p_n = F.normalize(proto_matrix, dim=1)
879-        logits = (z_n @ p_n.T) / self.tau
880-
881-        targets = []
882-        valid = []
883-        for i in range(y.size(0)):
884-            label = y[i].item()
885-            if label in class_to_idx:
886-                targets.append(class_to_idx[label])
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
**Findings**
- [important] `loss_film` does not backprop into the live style branch. `z_sty` is detached under `torch.no_grad()`, reduced to class prototypes, sent through the server, and reused as fixed tensors inside `_delta_film_augment()`. That means the M6 path is `loss_film -> head -> z_sem_film -> z_sem/semantic_head/encoder` and `-> style_modulator`, but not `-> delta_s -> current z_sty -> style_head -> encoder`. See [feddsa_adaptive.py:553](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L553), [feddsa_adaptive.py:611](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L611), [feddsa_adaptive.py:290](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L290), [feddsa_adaptive.py:775](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L775).
- [important] The FiLM branch is numerically unconstrained. `gamma` and `beta` are raw linear outputs and are applied as `z + gate * (gamma * z + beta)` with no clamp/tanh/scale normalization, while `loss_film` is added as a full extra CE term with no dedicated weight. Large modulator outputs can destabilize training or overpower the base task loss. See [feddsa_adaptive.py:83](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L83), [feddsa_adaptive.py:815](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L815), [feddsa_adaptive.py:531](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L531).
- [nit] The comments on `style_head` sharing are backwards. Runtime behavior is correct, but both comments say “M5/M6 shared” on the branch guarded by `not in (5, 6)`, which is maintenance-dangerous. See [feddsa_adaptive.py:201](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L201), [feddsa_adaptive.py:416](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L416).

**Checks**
- `StyleModulator`: `gamma/beta` come from `Linear(d,d) -> ReLU -> Linear(d,2d)` and `gate` comes from separate `Linear(d,1) + sigmoid`. Param count at `d=128` is `49,665`.
- `_delta_film_augment()`: class-matched cross-domain pairing is correct; delta is `normalize(sty_ext - sty_local)`; residual FiLM is `z_sem + gate * (gamma * z_sem + beta)`; edge cases are handled by skipping missing local proto, missing same-class cross proto, and near-zero delta, returning `None` if nothing was augmented.
- Data flow is implemented as: client collects detached `z_sty` class protos -> server stores `style_domain_protos` -> server splits into intra/cross in `pack()` -> client performs Delta-FiLM from those fixed protos next round.
- M6 correctly inherits M4 dual alignment and M5 style contrastive in training. `style_head` is shared in mode 6, same as mode 5. Inference is cleanly disabled for M6 because `forward()` only uses `encoder -> semantic_head -> head`.

Open question: if M6 was intended to be conditioned on current-sample `z_sty`, this implementation is not that; it is prototype-conditioned and effectively one round stale.
tokens used
44,672
**Findings**
- [important] `loss_film` does not backprop into the live style branch. `z_sty` is detached under `torch.no_grad()`, reduced to class prototypes, sent through the server, and reused as fixed tensors inside `_delta_film_augment()`. That means the M6 path is `loss_film -> head -> z_sem_film -> z_sem/semantic_head/encoder` and `-> style_modulator`, but not `-> delta_s -> current z_sty -> style_head -> encoder`. See [feddsa_adaptive.py:553](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L553), [feddsa_adaptive.py:611](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L611), [feddsa_adaptive.py:290](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L290), [feddsa_adaptive.py:775](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L775).
- [important] The FiLM branch is numerically unconstrained. `gamma` and `beta` are raw linear outputs and are applied as `z + gate * (gamma * z + beta)` with no clamp/tanh/scale normalization, while `loss_film` is added as a full extra CE term with no dedicated weight. Large modulator outputs can destabilize training or overpower the base task loss. See [feddsa_adaptive.py:83](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L83), [feddsa_adaptive.py:815](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L815), [feddsa_adaptive.py:531](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L531).
- [nit] The comments on `style_head` sharing are backwards. Runtime behavior is correct, but both comments say “M5/M6 shared” on the branch guarded by `not in (5, 6)`, which is maintenance-dangerous. See [feddsa_adaptive.py:201](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L201), [feddsa_adaptive.py:416](/D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L416).

**Checks**
- `StyleModulator`: `gamma/beta` come from `Linear(d,d) -> ReLU -> Linear(d,2d)` and `gate` comes from separate `Linear(d,1) + sigmoid`. Param count at `d=128` is `49,665`.
- `_delta_film_augment()`: class-matched cross-domain pairing is correct; delta is `normalize(sty_ext - sty_local)`; residual FiLM is `z_sem + gate * (gamma * z_sem + beta)`; edge cases are handled by skipping missing local proto, missing same-class cross proto, and near-zero delta, returning `None` if nothing was augmented.
- Data flow is implemented as: client collects detached `z_sty` class protos -> server stores `style_domain_protos` -> server splits into intra/cross in `pack()` -> client performs Delta-FiLM from those fixed protos next round.
- M6 correctly inherits M4 dual alignment and M5 style contrastive in training. `style_head` is shared in mode 6, same as mode 5. Inference is cleanly disabled for M6 because `forward()` only uses `encoder -> semantic_head -> head`.

Open question: if M6 was intended to be conditioned on current-sample `z_sty`, this implementation is not that; it is prototype-conditioned and effectively one round stale.
