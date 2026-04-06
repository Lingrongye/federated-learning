"""
FedDSA: Decouple-Share-Align — Decoupled Prototype Learning with Style Asset Sharing
Ported to flgo/FDSE framework for fair comparison with published baselines.

Architecture: Same AlexNet backbone as baselines + dual-head (semantic/style) + sem_classifier
Aggregation: FedAvg on encoder+semantic_head+head, style_head private, BN private
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
# Model: AlexNet backbone (same as config.py) + dual-head
# ============================================================

class AlexNetEncoder(nn.Module):
    """Same AlexNet as config.py but without the final classifier fc3."""
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
        self.head = nn.Linear(proj_dim, num_classes)  # named "head" for framework compat

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
            'lambda_orth': 1.0,
            'lambda_hsic': 0.1,
            'lambda_sem': 1.0,
            'tau': 0.1,
            'warmup_rounds': 10,
            'style_dispatch_num': 5,
            'proj_dim': 128,
        })
        self.sample_option = 'full'

        # Style bank: one slot per client
        self.style_bank = {}  # client_id -> (mu, sigma)
        # Global semantic prototypes: class -> (weighted_sum, total_count)
        self.global_semantic_protos = {}

        # Identify aggregation groups
        self._init_agg_keys()

        # Pass config to clients
        for c in self.clients:
            c.lambda_orth = self.lambda_orth
            c.lambda_hsic = self.lambda_hsic
            c.lambda_sem = self.lambda_sem
            c.tau = self.tau
            c.warmup_rounds = self.warmup_rounds
            c.proj_dim = self.proj_dim

    def _init_agg_keys(self):
        """Classify model keys into: shared (FedAvg), private (style_head+BN)."""
        all_keys = list(self.model.state_dict().keys())
        self.private_keys = set()
        for k in all_keys:
            if 'style_head' in k:
                self.private_keys.add(k)
            elif 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
                self.private_keys.add(k)
        self.shared_keys = [k for k in all_keys if k not in self.private_keys]

    def pack(self, client_id, mtype=0):
        """Send global model (shared keys only) + protos + styles."""
        # Dispatch style bank (exclude client's own)
        dispatched_styles = None
        if len(self.style_bank) > 0 and self.current_round >= self.warmup_rounds:
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
            'style_bank': dispatched_styles,
            'current_round': self.current_round,
        }

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)

        models = res['model']
        protos_list = res['protos']
        proto_counts_list = res['proto_counts']
        style_stats_list = res['style_stats']

        # 1. Aggregate shared parameters (FedAvg, excluding style_head and BN running stats)
        self._aggregate_shared(models)

        # 2. Collect style bank (one slot per client)
        for cid, style in zip(self.received_clients, style_stats_list):
            if style is not None:
                self.style_bank[cid] = style

        # 3. Aggregate semantic prototypes (weighted by sample count)
        self._aggregate_protos(protos_list, proto_counts_list)

    def _aggregate_shared(self, models):
        """FedAvg on shared keys only."""
        if len(models) == 0:
            return
        n = len(models)
        weights = np.array([len(self.clients[cid].train_data) for cid in self.received_clients], dtype=float)
        weights /= weights.sum()

        global_dict = self.model.state_dict()
        model_dicts = [m.state_dict() for m in models]

        for k in self.shared_keys:
            if 'num_batches_tracked' in k:
                continue
            global_dict[k] = sum(w * md[k] for w, md in zip(weights, model_dicts))

        self.model.load_state_dict(global_dict, strict=False)

    def _aggregate_protos(self, protos_list, counts_list):
        """Weighted average of per-class semantic prototypes."""
        agg = {}  # class -> (sum, count)
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
        # Will be set by server in Server.initialize()
        self.current_round = 0
        self.local_style_bank = None
        self.global_protos = None

    def reply(self, svr_pkg):
        model, global_protos, style_bank, current_round = self.unpack(svr_pkg)
        self.current_round = current_round
        self.global_protos = global_protos
        self.local_style_bank = style_bank
        self.train(model)
        return self.pack()

    def unpack(self, svr_pkg):
        """Receive global model (apply shared keys only, keep BN+style_head local)."""
        global_model = svr_pkg['model']
        if self.model is None:
            self.model = global_model
        else:
            new_dict = self.model.state_dict()
            global_dict = global_model.state_dict()
            for key in new_dict.keys():
                # Keep style_head and BN stats local
                if 'style_head' in key:
                    continue
                if 'bn' in key.lower() and ('running_' in key or 'num_batches_tracked' in key):
                    continue
                new_dict[key] = global_dict[key]
            self.model.load_state_dict(new_dict)
        return (
            self.model,
            svr_pkg['global_protos'],
            svr_pkg['style_bank'],
            svr_pkg['current_round'],
        )

    def pack(self):
        return {
            'model': copy.deepcopy(self.model.to('cpu')),
            'protos': self._local_protos,
            'proto_counts': self._local_proto_counts,
            'style_stats': self._local_style_stats,
        }

    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum
        )

        # Warmup factor for auxiliary losses
        aux_w = min(1.0, self.current_round / max(self.warmup_rounds, 1))

        # Online accumulators
        proto_sum = {}
        proto_count = {}
        style_sum = None
        style_sq_sum = None
        style_n = 0

        num_steps = self.num_steps
        for step in range(num_steps):
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            x, y = batch_data[0], batch_data[-1]

            model.zero_grad()

            # Forward
            h = model.encode(x)            # [B, 1024]
            z_sem = model.get_semantic(h)   # [B, proj_dim]
            z_sty = model.get_style(h)     # [B, proj_dim]

            # Loss 1: Task CE on semantic path
            output = model.head(z_sem)
            loss_task = self.loss_fn(output, y)

            # Loss 2: Task CE on augmented features
            loss_aug = torch.tensor(0.0, device=x.device)
            if self.local_style_bank is not None and self.current_round >= self.warmup_rounds:
                h_aug = self._style_augment(h)
                z_sem_aug = model.get_semantic(h_aug)
                output_aug = model.head(z_sem_aug)
                loss_aug = self.loss_fn(output_aug, y)

            # Loss 3: Decoupling (orthogonal + HSIC)
            loss_orth, loss_hsic = self._decouple_loss(z_sem, z_sty)

            # Loss 4: Semantic contrastive alignment
            loss_sem = torch.tensor(0.0, device=x.device)
            if self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_loss(z_sem, y)

            # Total loss
            loss = loss_task + loss_aug + \
                   aux_w * self.lambda_orth * loss_orth + \
                   aux_w * self.lambda_hsic * loss_hsic + \
                   aux_w * self.lambda_sem * loss_sem

            loss.backward()
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
            optimizer.step()

            # Accumulate prototypes (last epoch only)
            if step >= num_steps - len(self.train_data) // self.batch_size - 1:
                with torch.no_grad():
                    z_det = z_sem.detach().cpu()
                    h_det = h.detach()
                    for i, label in enumerate(y):
                        c = label.item()
                        if c not in proto_sum:
                            proto_sum[c] = z_det[i].clone()
                            proto_count[c] = 1
                        else:
                            proto_sum[c] += z_det[i]
                            proto_count[c] += 1

                    # Style stats (Welford online)
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

        # Store for pack()
        self._local_protos = {c: proto_sum[c] / proto_count[c] for c in proto_sum}
        self._local_proto_counts = proto_count

        if style_n > 1:
            mu = style_sum / style_n
            var = style_sq_sum / style_n - mu ** 2
            self._local_style_stats = (mu, var.clamp(min=1e-6).sqrt())
        else:
            self._local_style_stats = None

    # ---- Helper functions ----

    def _style_augment(self, h):
        """AdaIN-style augmentation using external style."""
        idx = np.random.randint(0, len(self.local_style_bank))
        mu_ext, sigma_ext = self.local_style_bank[idx]
        mu_ext = mu_ext.to(h.device)
        sigma_ext = sigma_ext.to(h.device)

        mu_local = h.mean(dim=0)
        sigma_local = h.std(dim=0, unbiased=False).clamp(min=1e-6)

        alpha = np.random.beta(0.1, 0.1)
        mu_mix = alpha * mu_local + (1 - alpha) * mu_ext
        sigma_mix = alpha * sigma_local + (1 - alpha) * sigma_ext

        h_norm = (h - mu_local) / sigma_local
        return h_norm * sigma_mix + mu_mix

    def _decouple_loss(self, z_sem, z_sty):
        """Orthogonal + HSIC dual constraint."""
        # Orthogonal
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

    def _infonce_loss(self, z_sem, y):
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
    'PACS': lambda: FedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128),
    'office': lambda: FedDSAModel(num_classes=10, feat_dim=1024, proj_dim=128),
}


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    dataset = os.path.split(object.option['task'])[-1].split('_')[0]
    if 'Server' in object.__class__.__name__:
        object.model = model_map.get(dataset, lambda: FedDSAModel())().to(object.device)
