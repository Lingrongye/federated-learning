OpenAI Codex v0.117.0 (research preview)
--------
workdir: D:\桌面文件\联邦学习
model: gpt-5.4
provider: fox
approval: never
sandbox: read-only
reasoning effort: high
reasoning summaries: none
session id: 019d91b9-e065-77d3-bf70-a1e276cc77ce
--------
user
Review M5 (style domain contrastive) code in feddsa_adaptive.py. M5=M4+style contrastive on z_sty. Check: data flow, _style_domain_contrastive() correctness, edge cases, gradient flow, mode=5 inherits M4. List bugs concisely.

=== CODE ===
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
            if 'style_head' in k:
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
        use_domain = self.adaptive_mode in (2, 3, 4, 5)

        # M4/M5: split domain_protos into intra (own-domain) + cross (other-domain)
        intra_protos = {}
        cross_protos = {}
        if self.adaptive_mode in (4, 5) and self.domain_protos:
            for (cls, cid), proto in self.domain_protos.items():
                if cid == client_id:
                    intra_protos[cls] = proto
                else:
                    cross_protos[(cls, cid)] = proto

        # M5: split style_domain_protos the same way
        sty_intra_protos = {}
        sty_cross_protos = {}
        if self.adaptive_mode == 5 and self.style_domain_protos:
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

        # 4. Store domain protos (M3 / M4 / M5)
        if self.adaptive_mode in (2, 3, 4, 5):
            self._store_domain_protos(protos_list)

        # 5. Store style domain protos (M5)
        if self.adaptive_mode == 5:
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
        if self.adaptive_mode == 5:
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

            if self.adaptive_mode in (4, 5) and self.intra_protos and self.cross_protos:
                # M4/M5: Dual alignment (intra-domain + cross-domain)
                loss_dual_intra, loss_dual_cross = self._dual_alignment_loss(z_sem, y)
            elif self.adaptive_mode in (4, 5) and self.global_protos and len(self.global_protos) >= 2:
                # M4/M5 fallback: use global protos until domain-level protos available
                loss_sem = self._infonce_global(z_sem, y)
            elif use_domain_protos and self.domain_protos and len(self.domain_protos) >= 2:
                loss_sem = self._infonce_domain_aware(z_sem, y)
            elif self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_global(z_sem, y)

            # M5: style domain contrastive loss
            loss_sty_con = torch.tensor(0.0, device=x.device)
            if self.adaptive_mode == 5 and self.sty_intra_protos and self.sty_cross_protos:
                loss_sty_con = self._style_domain_contrastive(z_sty, y)

            if self.adaptive_mode in (4, 5) and (loss_dual_intra.item() > 0 or loss_dual_cross.item() > 0):
                loss = loss_task + loss_aug + \
                       aux_w * self.lambda_orth * loss_orth + \
                       aux_w * self.lambda_hsic * loss_hsic + \
                       aux_w * self.lambda_intra * loss_dual_intra + \
                       aux_w * self.lambda_cross * loss_dual_cross + \
                       aux_w * self.lambda_sty_contrast * loss_sty_con
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

                    # M5: z_sty prototype accumulation
                    if self.adaptive_mode == 5:
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

        # M5: z_sty prototypes
        if self.adaptive_mode == 5 and sty_proto_sum:
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

            pos_sim = (z_n @ pos_n.T).squeeze() / self.tau  # scalar
            neg_sims = (z_n @ neg_n.T).squeeze() / self.tau  # [N_neg]

            # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
            all_logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])  # [1 + N_neg]
            loss = loss + (-pos_sim + torch.logsumexp(all_logits, dim=0))
            count += 1

        if count > 0:
            loss = loss / count
        return loss

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
=== END ===

codex
Reviewing `feddsa_adaptive.py` for M5 specifically: I’ll locate the file, read the relevant sections, and then check data flow, `_style_domain_contrastive()`, edge cases, gradient behavior, and whether mode `5` cleanly inherits M4.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg --files -g 'feddsa_adaptive.py'" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content 'C:/Users/31279/.agents/skills/code-review-excellence/SKILL.md' -TotalCount 220" in D:\桌面文件\联邦学习
 succeeded in 896ms:
FDSE_CVPR25\algorithm\feddsa_adaptive.py
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1027ms:
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
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I found the file. Next I’m reading the actual implementation with line numbers so I can cite concrete findings instead of reviewing the pasted snippet abstractly.
2026-04-15T15:20:01.755531Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_style_domain_contrastive|adaptive_mode in \\(4, 5\\)|sty_intra_protos|sty_cross_protos|use_domain_protos =|loss_sty_con|_dual_alignment_loss|style_gap_bank|style_domain_protos\" 'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'"` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_style_domain_contrastive|adaptive_mode in \\(4, 5\\)|sty_intra_protos|sty_cross_protos|use_domain_protos =|loss_sty_con|_dual_alignment_loss|style_gap_bank|style_domain_protos\" 'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'; "'$i=1; Get-Content $p | ForEach-Object { '"'{0,4}: {1}' -f "'$i, $_; $i++ }' in D:\桌面文件\联邦学习
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_style_domain_contrastive|adaptive_mode in \\(4, 5\\)|sty_intra_protos|sty_cross_protos|use_domain_protos =|loss_sty_con|_dual_alignment_loss|style_gap_bank|style_domain_protos\" 'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'"` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'; "'$i=1; Get-Content $p | ForEach-Object { '"'{0,4}: {1}' -f "'$i, $_; $i++ }'` rejected: blocked by policy
2026-04-15T15:20:01.765261Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$p='"'D:\\桌面文件\\联邦学习\\FDSE_CVPR25\\algorithm\\feddsa_adaptive.py'; "'$i=1; Get-Content $p | ForEach-Object { '"'{0,4}: {1}' -f "'$i, $_; $i++ }'` rejected: blocked by policy
codex
The command policy is stricter than usual in this workspace, so I’m switching to simpler read-only commands against the repo-relative path to get the exact lines I need.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content 'FDSE_CVPR25\\algorithm\\feddsa_adaptive.py' | Select-Object -Skip 260 -First 320" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content 'FDSE_CVPR25\\algorithm\\feddsa_adaptive.py' | Select-Object -First 260" in D:\桌面文件\联邦学习
 succeeded in 1047ms:
"""
FedDSA-Adaptive: Adaptive Augmentation + Domain-Aware Prototypes
Merges feddsa.py (base) + feddsa_domain_aware.py (M3) + new adaptive aug (M1).

Key changes vs base FedDSA:
  M1: Adaptive augmentation strength бк gap from z_sty stats б· per-client alpha
  M3: Domain-aware prototype alignment бк per-(class, client) protos + SupCon InfoNCE
  M4: Dual alignment бк L_intra (own-domain cosine) + L_cross (cross-domain InfoNCE)
  M5: M4 + style utilization бк L_sty_contrast (domain-contrastive on z_sty)

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
# Server бк Adaptive Gap Computation + Domain-Aware Protos
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
            if 'style_head' in k:
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
        use_domain = self.adaptive_mode in (2, 3, 4, 5)

        # M4/M5: split domain_protos into intra (own-domain) + cross (other-domain)
        intra_protos = {}
        cross_protos = {}
        if self.adaptive_mode in (4, 5) and self.domain_protos:
            for (cls, cid), proto in self.domain_protos.items():
                if cid == client_id:
                    intra_protos[cls] = proto
                else:
                    cross_protos[(cls, cid)] = proto

        # M5: split style_domain_protos the same way
        sty_intra_protos = {}
        sty_cross_protos = {}
        if self.adaptive_mode == 5 and self.style_domain_protos:
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
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1038ms:

        # 3. Compute gap metrics from z_sty bank (M1)
        if self.adaptive_mode in (1, 3):
            self._compute_gap_metrics()

        # 4. Store domain protos (M3 / M4 / M5)
        if self.adaptive_mode in (2, 3, 4, 5):
            self._store_domain_protos(protos_list)

        # 5. Store style domain protos (M5)
        if self.adaptive_mode == 5:
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
# Client бк Adaptive Augmentation + Domain-Aware Alignment
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
        if self.adaptive_mode == 5:
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

            if self.adaptive_mode in (4, 5) and self.intra_protos and self.cross_protos:
                # M4/M5: Dual alignment (intra-domain + cross-domain)
                loss_dual_intra, loss_dual_cross = self._dual_alignment_loss(z_sem, y)
            elif self.adaptive_mode in (4, 5) and self.global_protos and len(self.global_protos) >= 2:
                # M4/M5 fallback: use global protos until domain-level protos available
                loss_sem = self._infonce_global(z_sem, y)
            elif use_domain_protos and self.domain_protos and len(self.domain_protos) >= 2:
                loss_sem = self._infonce_domain_aware(z_sem, y)
            elif self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_global(z_sem, y)

            # M5: style domain contrastive loss
            loss_sty_con = torch.tensor(0.0, device=x.device)
            if self.adaptive_mode == 5 and self.sty_intra_protos and self.sty_cross_protos:
                loss_sty_con = self._style_domain_contrastive(z_sty, y)

            if self.adaptive_mode in (4, 5) and (loss_dual_intra.item() > 0 or loss_dual_cross.item() > 0):
                loss = loss_task + loss_aug + \
                       aux_w * self.lambda_orth * loss_orth + \
                       aux_w * self.lambda_hsic * loss_hsic + \
                       aux_w * self.lambda_intra * loss_dual_intra + \
                       aux_w * self.lambda_cross * loss_dual_cross + \
                       aux_w * self.lambda_sty_contrast * loss_sty_con
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

                    # M5: z_sty prototype accumulation
                    if self.adaptive_mode == 5:
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

        # M5: z_sty prototypes
        if self.adaptive_mode == 5 and sty_proto_sum:
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content 'FDSE_CVPR25\\algorithm\\feddsa_adaptive.py' | Select-Object -Skip 520 -First 220" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Get-Content 'FDSE_CVPR25\\algorithm\\feddsa_adaptive.py' | Select-Object -Skip 420 -First 120" in D:\桌面文件\联邦学习
 succeeded in 1071ms:
            'style_stats': self._local_style_stats,        # h-space (for AdaIN bank)
            'style_gap_stats': self._local_style_gap_stats, # z_sty-space (for gap metric)
        }
        if self.adaptive_mode == 5:
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

            if self.adaptive_mode in (4, 5) and self.intra_protos and self.cross_protos:
                # M4/M5: Dual alignment (intra-domain + cross-domain)
                loss_dual_intra, loss_dual_cross = self._dual_alignment_loss(z_sem, y)
            elif self.adaptive_mode in (4, 5) and self.global_protos and len(self.global_protos) >= 2:
                # M4/M5 fallback: use global protos until domain-level protos available
                loss_sem = self._infonce_global(z_sem, y)
            elif use_domain_protos and self.domain_protos and len(self.domain_protos) >= 2:
                loss_sem = self._infonce_domain_aware(z_sem, y)
            elif self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_global(z_sem, y)

            # M5: style domain contrastive loss
            loss_sty_con = torch.tensor(0.0, device=x.device)
            if self.adaptive_mode == 5 and self.sty_intra_protos and self.sty_cross_protos:
                loss_sty_con = self._style_domain_contrastive(z_sty, y)

            if self.adaptive_mode in (4, 5) and (loss_dual_intra.item() > 0 or loss_dual_cross.item() > 0):
                loss = loss_task + loss_aug + \
                       aux_w * self.lambda_orth * loss_orth + \
                       aux_w * self.lambda_hsic * loss_hsic + \
                       aux_w * self.lambda_intra * loss_dual_intra + \
                       aux_w * self.lambda_cross * loss_dual_cross + \
                       aux_w * self.lambda_sty_contrast * loss_sty_con
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

                    # M5: z_sty prototype accumulation
                    if self.adaptive_mode == 5:
                        z_sty_cpu = z_sty_det.cpu()
                        for i, label in enumerate(y):
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 1114ms:
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

                    # M5: z_sty prototype accumulation
                    if self.adaptive_mode == 5:
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

        # M5: z_sty prototypes
        if self.adaptive_mode == 5 and sty_proto_sum:
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
            # M1: gap б· adaptive alpha + stochastic jitter
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
        """M4: Decomposed alignment бк L_intra (own-domain cosine) + L_cross (cross-domain InfoNCE)."""
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

            pos_sim = (z_n @ pos_n.T).squeeze() / self.tau  # scalar
            neg_sims = (z_n @ neg_n.T).squeeze() / self.tau  # [N_neg]

            # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
            all_logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])  # [1 + N_neg]
            loss = loss + (-pos_sim + torch.logsumexp(all_logits, dim=0))
            count += 1

        if count > 0:
            loss = loss / count
        return loss
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I have the implementation now. I’m pulling exact line numbers for the M5 branches so the findings are precise and limited to actual defects.
2026-04-15T15:20:34.624315Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path 'FDSE_CVPR25\\algorithm\\feddsa_adaptive.py' -Pattern 'result\\[''sty_protos''\\]|sty_proto_sum|_local_sty_protos|self\\._store_domain_protos\\(protos_list\\)|sty_protos_list = res\\.get|self\\.style_gap_bank|self\\.client_gaps|self\\.adaptive_mode == 5 and self\\.sty_intra_protos and self\\.sty_cross_protos' | ForEach-Object { \"{0}:{1}\" -f "'$_.LineNumber, $_.Line.Trim() }'` rejected: blocked by policy
2026-04-15T15:20:34.669909Z ERROR codex_core::tools::router: error=`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path 'FDSE_CVPR25\\algorithm\\feddsa_adaptive.py' -Pattern 'if self.adaptive_mode in \\(4, 5\\) and self.intra_protos and self.cross_protos|loss_sty_con = self\\._style_domain_contrastive|if self\\.adaptive_mode in \\(4, 5\\) and \\(loss_dual_intra\\.item\\(\\) > 0 or loss_dual_cross\\.item\\(\\) > 0\\)|def _style_domain_contrastive|def _dual_alignment_loss|style_domain_protos|sty_intra_protos|sty_cross_protos|use_domain_protos = self\\.adaptive_mode in \\(2, 3\\)|adaptive_mode in \\(1, 3\\)' | ForEach-Object { \"{0}:{1}\" -f "'$_.LineNumber, $_.Line.Trim() }'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path 'FDSE_CVPR25\\algorithm\\feddsa_adaptive.py' -Pattern 'if self.adaptive_mode in \\(4, 5\\) and self.intra_protos and self.cross_protos|loss_sty_con = self\\._style_domain_contrastive|if self\\.adaptive_mode in \\(4, 5\\) and \\(loss_dual_intra\\.item\\(\\) > 0 or loss_dual_cross\\.item\\(\\) > 0\\)|def _style_domain_contrastive|def _dual_alignment_loss|style_domain_protos|sty_intra_protos|sty_cross_protos|use_domain_protos = self\\.adaptive_mode in \\(2, 3\\)|adaptive_mode in \\(1, 3\\)' | ForEach-Object { \"{0}:{1}\" -f "'$_.LineNumber, $_.Line.Trim() }' in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path 'FDSE_CVPR25\\algorithm\\feddsa_adaptive.py' -Pattern 'result\\[''sty_protos''\\]|sty_proto_sum|_local_sty_protos|self\\._store_domain_protos\\(protos_list\\)|sty_protos_list = res\\.get|self\\.style_gap_bank|self\\.client_gaps|self\\.adaptive_mode == 5 and self\\.sty_intra_protos and self\\.sty_cross_protos' | ForEach-Object { \"{0}:{1}\" -f "'$_.LineNumber, $_.Line.Trim() }' in D:\桌面文件\联邦学习
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path 'FDSE_CVPR25\\algorithm\\feddsa_adaptive.py' -Pattern 'result\\[''sty_protos''\\]|sty_proto_sum|_local_sty_protos|self\\._store_domain_protos\\(protos_list\\)|sty_protos_list = res\\.get|self\\.style_gap_bank|self\\.client_gaps|self\\.adaptive_mode == 5 and self\\.sty_intra_protos and self\\.sty_cross_protos' | ForEach-Object { \"{0}:{1}\" -f "'$_.LineNumber, $_.Line.Trim() }'` rejected: blocked by policy
 declined in 0ms:
`"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "Select-String -Path 'FDSE_CVPR25\\algorithm\\feddsa_adaptive.py' -Pattern 'if self.adaptive_mode in \\(4, 5\\) and self.intra_protos and self.cross_protos|loss_sty_con = self\\._style_domain_contrastive|if self\\.adaptive_mode in \\(4, 5\\) and \\(loss_dual_intra\\.item\\(\\) > 0 or loss_dual_cross\\.item\\(\\) > 0\\)|def _style_domain_contrastive|def _dual_alignment_loss|style_domain_protos|sty_intra_protos|sty_cross_protos|use_domain_protos = self\\.adaptive_mode in \\(2, 3\\)|adaptive_mode in \\(1, 3\\)' | ForEach-Object { \"{0}:{1}\" -f "'$_.LineNumber, $_.Line.Trim() }'` rejected: blocked by policy
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"def _style_domain_contrastive|def _dual_alignment_loss|style_domain_protos|sty_intra_protos|sty_cross_protos|use_domain_protos = self.adaptive_mode in \\(2, 3\\)|self.adaptive_mode == 5 and self.sty_intra_protos and self.sty_cross_protos|loss_sty_con = self._style_domain_contrastive|loss_dual_intra.item\\(\\) > 0 or loss_dual_cross.item\\(\\) > 0\" FDSE_CVPR25/algorithm/feddsa_adaptive.py" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"adaptive_mode in \\(1, 3\\)|self._store_domain_protos\\(protos_list\\)|sty_protos_list = res.get|result\\['sty_protos'\\]|sty_proto_sum|_local_sty_protos|self.style_gap_bank|self.client_gaps\" FDSE_CVPR25/algorithm/feddsa_adaptive.py" in D:\桌面文件\联邦学习
 succeeded in 867ms:
148:        self.style_gap_bank = {}  # client_id -> (mu_zsty, sigma_zsty)
156:        self.client_gaps = {}  # client_id -> normalized gap [0,1]
202:        gap_normalized = self.client_gaps.get(client_id, 0.5)
260:                self.style_gap_bank[cid] = zsty_style
263:        if self.adaptive_mode in (1, 3):
268:            self._store_domain_protos(protos_list)
272:            sty_protos_list = res.get('sty_protos', [None] * len(self.received_clients))
284:        if len(self.style_gap_bank) < 2:
287:        all_mu = torch.stack([s[0] for s in self.style_gap_bank.values()])
288:        all_sigma = torch.stack([s[1] for s in self.style_gap_bank.values()])
294:        for cid, (mu, sigma) in self.style_gap_bank.items():
312:            self.client_gaps[cid] = float(np.clip(z * 0.5 + 0.5, 0.0, 1.0))
425:            result['sty_protos'] = getattr(self, '_local_sty_protos', None)
437:        use_adaptive_aug = self.adaptive_mode in (1, 3)
443:        sty_proto_sum = {}   # M5: z_sty class prototypes
542:                            if c not in sty_proto_sum:
543:                                sty_proto_sum[c] = z_sty_cpu[i].clone()
546:                                sty_proto_sum[c] += z_sty_cpu[i]
580:        if self.adaptive_mode == 5 and sty_proto_sum:
581:            self._local_sty_protos = {c: sty_proto_sum[c] / sty_proto_count[c] for c in sty_proto_sum}
583:            self._local_sty_protos = None
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 894ms:
153:        self.style_domain_protos = {}  # (class, client_id) -> z_sty proto (M5)
217:        # M5: split style_domain_protos the same way
218:        sty_intra_protos = {}
219:        sty_cross_protos = {}
220:        if self.adaptive_mode == 5 and self.style_domain_protos:
221:            for (cls, cid), proto in self.style_domain_protos.items():
223:                    sty_intra_protos[cls] = proto
225:                    sty_cross_protos[(cls, cid)] = proto
233:            'sty_intra_protos': copy.deepcopy(sty_intra_protos),
234:            'sty_cross_protos': copy.deepcopy(sty_cross_protos),
277:                    self.style_domain_protos[(c, cid)] = proto
371:        self.sty_intra_protos = {}
372:        self.sty_cross_protos = {}
384:        self.sty_intra_protos = sty_intra
385:        self.sty_cross_protos = sty_cross
412:            svr_pkg.get('sty_intra_protos', {}),
413:            svr_pkg.get('sty_cross_protos', {}),
438:        use_domain_protos = self.adaptive_mode in (2, 3)
499:            if self.adaptive_mode == 5 and self.sty_intra_protos and self.sty_cross_protos:
500:                loss_sty_con = self._style_domain_contrastive(z_sty, y)
502:            if self.adaptive_mode in (4, 5) and (loss_dual_intra.item() > 0 or loss_dual_cross.item() > 0):
636:    def _dual_alignment_loss(self, z_sem, y):
691:    def _style_domain_contrastive(self, z_sty, y):
701:        #   positive = sty_intra_protos[y_i] (same domain, same class)
702:        #   negatives = all sty_cross_protos (any class, other domains)
706:        for key in sorted(self.sty_cross_protos.keys()):
707:            neg_entries.append(self.sty_cross_protos[key])
720:            if label not in self.sty_intra_protos:
723:            pos_proto = self.sty_intra_protos[label].to(device)  # [128]
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"torch.cat\\(\\[pos_sim.unsqueeze\\(0\\), neg_sims\\]\\)|neg_sims = \\(z_n @ neg_n.T\\).squeeze\\(\\) / self.tau|if self.adaptive_mode in \\(4, 5\\) and self.intra_protos and self.cross_protos|elif self.adaptive_mode in \\(4, 5\\) and self.global_protos and len\\(self.global_protos\\) >= 2\" FDSE_CVPR25/algorithm/feddsa_adaptive.py" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"loss = loss_task \\+ loss_aug|aux_w \\* self.lambda_sty_contrast \\* loss_sty_con|aux_w \\* self.lambda_sem \\* loss_sem\" FDSE_CVPR25/algorithm/feddsa_adaptive.py" in D:\桌面文件\联邦学习
 succeeded in 837ms:
503:                loss = loss_task + loss_aug + \
508:                       aux_w * self.lambda_sty_contrast * loss_sty_con
510:                loss = loss_task + loss_aug + \
513:                       aux_w * self.lambda_sem * loss_sem
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 879ms:
486:            if self.adaptive_mode in (4, 5) and self.intra_protos and self.cross_protos:
489:            elif self.adaptive_mode in (4, 5) and self.global_protos and len(self.global_protos) >= 2:
731:            neg_sims = (z_n @ neg_n.T).squeeze() / self.tau  # [N_neg]
734:            all_logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])  # [1 + N_neg]
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
I found a more important data-flow issue than the local loss math: M5 is comparing `z_sty` across clients even though `style_head` is explicitly private per client. I’m extracting those lines now because that breaks the cross-client prototype assumption itself.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"style_domain_protos\\[\\(c, cid\\)\\] = proto|sty_cross_protos\\[\\(cls, cid\\)\\] = proto|sty_intra_protos\\[cls\\] = proto|result\\['sty_protos'\\]\" FDSE_CVPR25/algorithm/feddsa_adaptive.py" in D:\桌面文件\联邦学习
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"if 'style_head' in k|if 'style_head' in key|self.private_keys|self.shared_keys|new_dict\\[key\\] = global_dict\\[key\\]\" FDSE_CVPR25/algorithm/feddsa_adaptive.py" in D:\桌面文件\联邦学习
 succeeded in 913ms:
181:        self.private_keys = set()
183:            if 'style_head' in k:
184:                self.private_keys.add(k)
186:                self.private_keys.add(k)
187:        self.shared_keys = [k for k in all_keys if k not in self.private_keys]
325:        for k in self.shared_keys:
397:                if 'style_head' in key:
401:                new_dict[key] = global_dict[key]
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

 succeeded in 885ms:
223:                    sty_intra_protos[cls] = proto
225:                    sty_cross_protos[(cls, cid)] = proto
277:                    self.style_domain_protos[(c, cid)] = proto
425:            result['sty_protos'] = getattr(self, '_local_sty_protos', None)
[31;1mInvalidOperation: [31;1mCannot set property. Property setting is supported only on core types in this language mode.[0m

codex
**Findings**

- High: M5 compares `z_sty` across clients even though `style_head` is explicitly private per client. The server excludes `style_head` from aggregation at [feddsa_adaptive.py:183](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L183), clients also refuse to load global `style_head` weights at [feddsa_adaptive.py:397](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L397), but M5 stores/broadcasts other clients’ `z_sty` prototypes at [feddsa_adaptive.py:225](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L225) and [feddsa_adaptive.py:277](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L277), then contrasts against them in [feddsa_adaptive.py:691](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L691). Those embeddings are not in a shared space, so cross-client negatives are not semantically valid.

- High: Mode `5` does not reliably behave as “M4 + style contrastive”. Dual alignment is only entered if both `intra_protos` and `cross_protos` are non-empty at [feddsa_adaptive.py:486](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L486), even though `_dual_alignment_loss()` can compute either term independently. Then the final loss includes `loss_sty_con` only inside the dual branch at [feddsa_adaptive.py:503](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L503) and [feddsa_adaptive.py:508](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L508); the fallback branch at [feddsa_adaptive.py:510](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L510) drops it completely. So early rounds, partial-prototype cases, or valid zero-valued dual losses silently remove the M5 gradient.

- Medium: `_style_domain_contrastive()` breaks when there is exactly one cross-domain negative. `neg_sims = ...squeeze()` at [feddsa_adaptive.py:731](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L731) becomes a 0-D tensor for `N_neg == 1`, and `torch.cat([pos_sim.unsqueeze(0), neg_sims])` at [feddsa_adaptive.py:734](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L734) will fail.

Gradient-wise, `_style_domain_contrastive()` itself does backprop to current `z_sty`; the real gradient bugs are that the loss is often omitted entirely, and when it is used, its negatives come from incompatible private style spaces.
tokens used
41,166
**Findings**

- High: M5 compares `z_sty` across clients even though `style_head` is explicitly private per client. The server excludes `style_head` from aggregation at [feddsa_adaptive.py:183](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L183), clients also refuse to load global `style_head` weights at [feddsa_adaptive.py:397](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L397), but M5 stores/broadcasts other clients’ `z_sty` prototypes at [feddsa_adaptive.py:225](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L225) and [feddsa_adaptive.py:277](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L277), then contrasts against them in [feddsa_adaptive.py:691](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L691). Those embeddings are not in a shared space, so cross-client negatives are not semantically valid.

- High: Mode `5` does not reliably behave as “M4 + style contrastive”. Dual alignment is only entered if both `intra_protos` and `cross_protos` are non-empty at [feddsa_adaptive.py:486](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L486), even though `_dual_alignment_loss()` can compute either term independently. Then the final loss includes `loss_sty_con` only inside the dual branch at [feddsa_adaptive.py:503](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L503) and [feddsa_adaptive.py:508](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L508); the fallback branch at [feddsa_adaptive.py:510](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L510) drops it completely. So early rounds, partial-prototype cases, or valid zero-valued dual losses silently remove the M5 gradient.

- Medium: `_style_domain_contrastive()` breaks when there is exactly one cross-domain negative. `neg_sims = ...squeeze()` at [feddsa_adaptive.py:731](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L731) becomes a 0-D tensor for `N_neg == 1`, and `torch.cat([pos_sim.unsqueeze(0), neg_sims])` at [feddsa_adaptive.py:734](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_adaptive.py#L734) will fail.

Gradient-wise, `_style_domain_contrastive()` itself does backprop to current `z_sty`; the real gradient bugs are that the loss is often omitted entirely, and when it is used, its negatives come from incompatible private style spaces.
