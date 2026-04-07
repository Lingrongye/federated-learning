"""
FedDSA+ : Improved version of FedDSA with three algorithmic enhancements:

1. **Sigmoid adaptive loss weights** — smooth transition (vs hard linear warmup)
   weight(round) = lambda * sigmoid((round - warmup) / tau_w)

2. **Three-stage explicit training**:
   Stage 1 (round < W1): Pure CE only — let backbone learn stable features
   Stage 2 (W1 <= round < W2): + decoupling losses (orth + HSIC)
   Stage 3 (round >= W2): + style augmentation + InfoNCE

3. **Per-loss gradient clipping** — limit auxiliary losses gradient norms separately
   to prevent any single loss from dominating

Reference: Original feddsa.py in same directory.
"""
import os
import copy
import collections
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import flgo.algorithm.fedbase
import flgo.utils.fmodule as fuf
from collections import OrderedDict

# Reuse model from feddsa.py
from algorithm.feddsa import FedDSAModel, AlexNetEncoder


# ============================================================
# Server (mostly identical to feddsa.py, but passes more config)
# ============================================================

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({
            'lambda_orth': 1.0,
            'lambda_hsic': 0.1,
            'lambda_sem': 1.0,
            'tau': 0.1,
            'stage1_end': 50,    # round < 50: pure CE
            'stage2_end': 100,   # 50 <= round < 100: +decouple; >= 100: +style+infonce
            'transition_width': 10,  # sigmoid transition smoothness
            'aux_grad_clip': 1.0,    # gradient clip for aux losses
            'style_dispatch_num': 5,
            'proj_dim': 128,
        })
        self.sample_option = 'full'

        self.style_bank = {}
        self.global_semantic_protos = {}

        self._init_agg_keys()

        # Pass config to clients
        for c in self.clients:
            c.lambda_orth = self.lambda_orth
            c.lambda_hsic = self.lambda_hsic
            c.lambda_sem = self.lambda_sem
            c.tau = self.tau
            c.stage1_end = self.stage1_end
            c.stage2_end = self.stage2_end
            c.transition_width = self.transition_width
            c.aux_grad_clip = self.aux_grad_clip
            c.proj_dim = self.proj_dim

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
        dispatched_styles = None
        if len(self.style_bank) > 0 and self.current_round >= self.stage2_end:
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

        self._aggregate_shared(models)

        for cid, style in zip(self.received_clients, style_stats_list):
            if style is not None:
                self.style_bank[cid] = style

        self._aggregate_protos(protos_list, proto_counts_list)

    def _aggregate_shared(self, models):
        if len(models) == 0:
            return
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
# Client with three-stage training and adaptive loss weights
# ============================================================

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = copy.deepcopy(self.server.model).to('cpu')
        self.loss_fn = nn.CrossEntropyLoss()
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

    def _sigmoid_weight(self, round_num, threshold):
        """Smooth sigmoid transition: 0 -> 1 around 'threshold'."""
        x = (round_num - threshold) / max(self.transition_width, 1)
        return 1.0 / (1.0 + math.exp(-x))

    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum
        )

        # === Three-stage adaptive weights ===
        # Stage 1: round < stage1_end → pure CE (decouple_w=0, style_w=0)
        # Stage 2: stage1_end <= round < stage2_end → +decouple (decouple_w ramps up, style_w=0)
        # Stage 3: round >= stage2_end → +style aug + infonce
        stage = 1
        if self.current_round >= self.stage2_end:
            stage = 3
        elif self.current_round >= self.stage1_end:
            stage = 2

        # Smooth weights via sigmoid centered at stage boundaries
        decouple_w = self._sigmoid_weight(self.current_round, self.stage1_end)
        style_w = self._sigmoid_weight(self.current_round, self.stage2_end)

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

            h = model.encode(x)
            z_sem = model.get_semantic(h)
            z_sty = model.get_style(h)

            # Loss 1: Task CE on semantic path
            output = model.head(z_sem)
            loss_task = self.loss_fn(output, y)

            # Loss 2: Task CE on augmented features (Stage 3 only)
            loss_aug = torch.tensor(0.0, device=x.device)
            if stage >= 3 and self.local_style_bank is not None:
                h_aug = self._style_augment(h)
                z_sem_aug = model.get_semantic(h_aug)
                output_aug = model.head(z_sem_aug)
                loss_aug = self.loss_fn(output_aug, y)

            # Loss 3: Decoupling (Stage 2+)
            loss_orth = torch.tensor(0.0, device=x.device)
            loss_hsic = torch.tensor(0.0, device=x.device)
            if stage >= 2 and decouple_w > 0.01:
                loss_orth, loss_hsic = self._decouple_loss(z_sem, z_sty)

            # Loss 4: Semantic InfoNCE (Stage 3 only)
            loss_sem = torch.tensor(0.0, device=x.device)
            if stage >= 3 and style_w > 0.01 and self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_loss(z_sem, y)

            # Total loss with sigmoid-smoothed weights
            loss = loss_task + style_w * loss_aug + \
                   decouple_w * self.lambda_orth * loss_orth + \
                   decouple_w * self.lambda_hsic * loss_hsic + \
                   style_w * self.lambda_sem * loss_sem

            loss.backward()

            # Per-loss gradient clipping (auxiliary losses limited separately)
            # Standard model clip
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)

            # Additional aux loss clip on semantic_head and style_head if active
            if stage >= 2 and self.aux_grad_clip > 0:
                aux_params = list(model.semantic_head.parameters()) + list(model.style_head.parameters())
                torch.nn.utils.clip_grad_norm_(aux_params, max_norm=self.aux_grad_clip)

            optimizer.step()

            # Online accumulation (last few batches)
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

        self._local_protos = {c: proto_sum[c] / proto_count[c] for c in proto_sum}
        self._local_proto_counts = proto_count

        if style_n > 1:
            mu = style_sum / style_n
            var = style_sq_sum / style_n - mu ** 2
            self._local_style_stats = (mu, var.clamp(min=1e-6).sqrt())
        else:
            self._local_style_stats = None

    def _style_augment(self, h):
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

    def _infonce_loss(self, z_sem, y):
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
# Model initialization (reuse FedDSAModel from feddsa.py)
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
