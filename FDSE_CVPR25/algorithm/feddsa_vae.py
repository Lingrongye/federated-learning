"""
FedDSA-VAE: Probabilistic style head with KL regularization.

Instead of deterministic z_sty = style_head(h), we output (mu, log_var) and
sample z_sty = mu + exp(0.5*log_var) * eps. A KL loss pushes the distribution
toward N(0, I), providing principled regularization and disentanglement.
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flgo.algorithm.fedbase
import flgo.utils.fmodule as fuf

from algorithm.feddsa import AlexNetEncoder, Client as BaseClient


class FedDSAVAEModel(fuf.FModule):
    def __init__(self, num_classes=7, feat_dim=1024, proj_dim=128):
        super().__init__()
        self.encoder = AlexNetEncoder()
        self.semantic_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        # VAE style head: outputs mu and log_var
        self.style_mu = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.style_logvar = nn.Sequential(
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
        mu = self.style_mu(h)
        logvar = self.style_logvar(h)
        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, logvar


class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({
            'lambda_orth': 1.0,
            'lambda_hsic': 0.0,
            'lambda_sem': 1.0,
            'lambda_kl': 0.01,   # NEW: KL regularization weight
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
            c.lambda_orth = self.lambda_orth
            c.lambda_hsic = self.lambda_hsic
            c.lambda_sem = self.lambda_sem
            c.lambda_kl = self.lambda_kl
            c.tau = self.tau
            c.warmup_rounds = self.warmup_rounds
            c.proj_dim = self.proj_dim

    def _init_agg_keys(self):
        all_keys = list(self.model.state_dict().keys())
        self.private_keys = set()
        for k in all_keys:
            if 'style_mu' in k or 'style_logvar' in k:
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
        if len(models) == 0: return
        weights = np.array([len(self.clients[cid].train_data) for cid in self.received_clients], dtype=float)
        weights /= weights.sum()
        global_dict = self.model.state_dict()
        model_dicts = [m.state_dict() for m in models]
        for k in self.shared_keys:
            if 'num_batches_tracked' in k: continue
            global_dict[k] = sum(w * md[k] for w, md in zip(weights, model_dicts))
        self.model.load_state_dict(global_dict, strict=False)

    def _aggregate_protos(self, protos_list, counts_list):
        agg = {}
        for protos, counts in zip(protos_list, counts_list):
            if protos is None: continue
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


class Client(BaseClient):
    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum
        )
        aux_w = min(1.0, self.current_round / max(self.warmup_rounds, 1))

        proto_sum = {}; proto_count = {}
        style_sum = None; style_sq_sum = None; style_n = 0

        for step in range(self.num_steps):
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            x, y = batch_data[0], batch_data[-1]

            model.zero_grad()
            h = model.encode(x)
            z_sem = model.get_semantic(h)
            z_sty, sty_mu, sty_logvar = model.get_style(h)

            output = model.head(z_sem)
            loss_task = self.loss_fn(output, y)

            loss_aug = torch.tensor(0.0, device=x.device)
            if self.local_style_bank is not None and self.current_round >= self.warmup_rounds:
                h_aug = self._style_augment(h)
                z_sem_aug = model.get_semantic(h_aug)
                output_aug = model.head(z_sem_aug)
                loss_aug = self.loss_fn(output_aug, y)

            # Decouple loss using sampled z_sty
            loss_orth, loss_hsic = self._decouple_loss(z_sem, z_sty)

            # VAE KL loss: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
            loss_kl = -0.5 * torch.mean(1 + sty_logvar - sty_mu.pow(2) - sty_logvar.exp())

            loss_sem = torch.tensor(0.0, device=x.device)
            if self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_loss(z_sem, y)

            loss = loss_task + loss_aug + \
                aux_w * self.lambda_orth * loss_orth + \
                aux_w * self.lambda_hsic * loss_hsic + \
                aux_w * self.lambda_sem * loss_sem + \
                aux_w * self.lambda_kl * loss_kl

            loss.backward()
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
            optimizer.step()

            if step >= self.num_steps - len(self.train_data) // self.batch_size - 1:
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
                    bm = h_det.mean(dim=0).cpu()
                    bsq = (h_det ** 2).mean(dim=0).cpu()
                    if style_sum is None:
                        style_sum = bm * b; style_sq_sum = bsq * b; style_n = b
                    else:
                        style_sum += bm * b; style_sq_sum += bsq * b; style_n += b

        self._local_protos = {c: proto_sum[c] / proto_count[c] for c in proto_sum}
        self._local_proto_counts = proto_count
        if style_n > 1:
            mu = style_sum / style_n
            var = style_sq_sum / style_n - mu ** 2
            self._local_style_stats = (mu, var.clamp(min=1e-6).sqrt())
        else:
            self._local_style_stats = None

    def _decouple_loss(self, z_sem, z_sty):
        z_sem_n = F.normalize(z_sem, dim=1)
        z_sty_n = F.normalize(z_sty, dim=1)
        cos = (z_sem_n * z_sty_n).sum(dim=1)
        return (cos ** 2).mean(), torch.tensor(0.0, device=z_sem.device)


model_map = {
    'PACS': lambda: FedDSAVAEModel(num_classes=7, feat_dim=1024, proj_dim=128),
    'office': lambda: FedDSAVAEModel(num_classes=10, feat_dim=1024, proj_dim=128),
}


def init_dataset(object): pass
def init_local_module(object): pass
def init_global_module(object):
    dataset = os.path.split(object.option['task'])[-1].split('_')[0]
    if 'Server' in object.__class__.__name__:
        object.model = model_map.get(dataset, lambda: FedDSAVAEModel())().to(object.device)
