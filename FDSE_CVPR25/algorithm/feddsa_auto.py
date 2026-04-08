"""
FedDSA-Auto: Automatic loss weighting via Kendall et al. Uncertainty Weighting (CVPR 2018).

Instead of hand-tuning lambda_orth, lambda_hsic, lambda_sem, we learn a log-variance
parameter for each loss and weight them automatically.

Total loss = sum_i [ exp(-log_sigma_i) * loss_i + log_sigma_i ]

References:
- Kendall, Gal, Cipolla "Multi-Task Learning Using Uncertainty to Weigh Losses..."
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flgo.algorithm.fedbase
import flgo.utils.fmodule as fuf

from algorithm.feddsa import FedDSAModel, AlexNetEncoder


class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({
            'tau': 0.1,
            'warmup_rounds': 50,
            'style_dispatch_num': 5,
            'proj_dim': 128,
        })
        self.sample_option = 'full'

        self.style_bank = {}
        self.global_semantic_protos = {}

        self._init_agg_keys()

        for c in self.clients:
            c.tau = self.tau
            c.warmup_rounds = self.warmup_rounds
            c.proj_dim = self.proj_dim
            # Store initial values (will be converted to device-specific Parameters in train)
            c._log_sigma_init = 0.0

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

    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        device = next(model.parameters()).device
        # Create learnable log_sigma as leaf tensors on device
        # Load from previous round if exists
        init_task_aug = getattr(self, '_log_sigma_task_aug_val', 0.0)
        init_orth = getattr(self, '_log_sigma_orth_val', 0.0)
        init_sem = getattr(self, '_log_sigma_sem_val', 0.0)
        self.log_sigma_task_aug = torch.tensor([init_task_aug], device=device, requires_grad=True)
        self.log_sigma_orth = torch.tensor([init_orth], device=device, requires_grad=True)
        self.log_sigma_sem = torch.tensor([init_sem], device=device, requires_grad=True)

        # Include log_sigma parameters in optimizer
        params = list(model.parameters()) + [
            self.log_sigma_task_aug, self.log_sigma_orth, self.log_sigma_sem
        ]
        optimizer = torch.optim.SGD(
            params, lr=self.learning_rate,
            momentum=self.momentum, weight_decay=self.weight_decay
        )

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
            optimizer.zero_grad()

            h = model.encode(x)
            z_sem = model.get_semantic(h)
            z_sty = model.get_style(h)

            # L1: Task (no uncertainty weight, this is the "anchor" loss)
            output = model.head(z_sem)
            loss_task = self.loss_fn(output, y)

            # L2: Task aug (with uncertainty weight)
            loss_aug = torch.tensor(0.0, device=device)
            if self.local_style_bank is not None and self.current_round >= self.warmup_rounds:
                h_aug = self._style_augment(h)
                z_sem_aug = model.get_semantic(h_aug)
                output_aug = model.head(z_sem_aug)
                loss_aug = self.loss_fn(output_aug, y)

            # L3: Orth
            loss_orth = self._orth_loss(z_sem, z_sty)

            # L4: InfoNCE
            loss_sem = torch.tensor(0.0, device=device)
            if self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_loss(z_sem, y)

            # Uncertainty weighting:
            # total = loss_task + sum_i[ exp(-log_sigma_i) * loss_i + log_sigma_i ]
            precision_aug = torch.exp(-self.log_sigma_task_aug)
            precision_orth = torch.exp(-self.log_sigma_orth)
            precision_sem = torch.exp(-self.log_sigma_sem)

            total_loss = loss_task + \
                precision_aug.squeeze() * loss_aug + self.log_sigma_task_aug.squeeze() + \
                precision_orth.squeeze() * loss_orth + self.log_sigma_orth.squeeze() + \
                precision_sem.squeeze() * loss_sem + self.log_sigma_sem.squeeze()

            total_loss.backward()
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
            optimizer.step()

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

        # Save log_sigma values for next round (with clamping to prevent collapse)
        # Clamp to [-2, 2]: prevents weight from going too extreme (exp(-2)~0.13 to exp(2)~7.4)
        self._log_sigma_task_aug_val = float(self.log_sigma_task_aug.detach().cpu().clamp(-2, 2).item())
        self._log_sigma_orth_val = float(self.log_sigma_orth.detach().cpu().clamp(-2, 2).item())
        self._log_sigma_sem_val = float(self.log_sigma_sem.detach().cpu().clamp(-2, 2).item())

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

    def _orth_loss(self, z_sem, z_sty):
        z_sem_n = F.normalize(z_sem, dim=1)
        z_sty_n = F.normalize(z_sty, dim=1)
        cos = (z_sem_n * z_sty_n).sum(dim=1)
        return (cos ** 2).mean()

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
