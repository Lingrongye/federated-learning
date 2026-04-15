"""
FedDSA-Adaptive: Adaptive Augmentation + Domain-Aware Prototypes
Merges feddsa.py (base) + feddsa_domain_aware.py (M3) + new adaptive aug (M1).

Key changes vs base FedDSA:
  M1: Adaptive augmentation strength — gap from z_sty stats → per-client α
  M3: Domain-aware prototype alignment — per-(class, client) protos + SupCon InfoNCE

Modes controlled by algo_para:
  adaptive_mode: 0=fixed_alpha, 1=adaptive (M1), 2=M3-only, 3=M1+M3 full
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
            'md': 1,     # adaptive_mode: 0=fixed_alpha,1=M1,2=M3,3=M1+M3
            'fa': 0.5,   # fixed_alpha_value
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
        self.sample_option = 'full'

        # Style bank: h-space stats for AdaIN augmentation
        self.style_bank = {}  # client_id -> (mu_h, sigma_h)
        # Style gap bank: z_sty-space stats for gap measurement (dual bank, R2 review)
        self.style_gap_bank = {}  # client_id -> (mu_zsty, sigma_zsty)

        # Prototypes
        self.global_semantic_protos = {}  # class -> avg proto
        self.domain_protos = {}  # (class, client_id) -> proto (M3)

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
        use_domain = self.adaptive_mode in (2, 3)

        return {
            'model': copy.deepcopy(self.model),
            'global_protos': copy.deepcopy(self.global_semantic_protos),
            'domain_protos': copy.deepcopy(self.domain_protos) if use_domain else {},
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

        # 4. Store domain protos (M3)
        if self.adaptive_mode in (2, 3):
            self._store_domain_protos(protos_list)

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
        self.gap_normalized = 0.5

    def reply(self, svr_pkg):
        model, global_protos, domain_protos, style_bank, current_round, gap_normalized = self.unpack(svr_pkg)
        self.current_round = current_round
        self.global_protos = global_protos
        self.domain_protos = domain_protos
        self.local_style_bank = style_bank
        self.gap_normalized = gap_normalized
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
        )

    def pack(self):
        return {
            'model': copy.deepcopy(self.model.to('cpu')),
            'protos': self._local_protos,
            'proto_counts': self._local_proto_counts,
            'style_stats': self._local_style_stats,        # h-space (for AdaIN bank)
            'style_gap_stats': self._local_style_gap_stats, # z_sty-space (for gap metric)
        }

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

            # Loss 4: Semantic alignment (domain-aware or global)
            loss_sem = torch.tensor(0.0, device=x.device)
            if use_domain_protos and self.domain_protos and len(self.domain_protos) >= 2:
                loss_sem = self._infonce_domain_aware(z_sem, y)
            elif self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_global(z_sem, y)

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

                    # Prototype accumulation
                    for i, label in enumerate(y):
                        c = label.item()
                        if c not in proto_sum:
                            proto_sum[c] = z_det[i].clone()
                            proto_count[c] = 1
                        else:
                            proto_sum[c] += z_det[i]
                            proto_count[c] += 1

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
