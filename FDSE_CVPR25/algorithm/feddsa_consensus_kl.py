"""
FedDSA-Consensus+KL: Combine consensus-max aggregation with FDSE's consistency regularization.

Motivation (per-domain analysis of Office):
    Caltech delta: FDSE +13.39, Consensus +5.36 → still 8% gap
    FDSE paper Table 3: +C module adds +1.20 on Office AVG

    Hypothesis: Consensus + Consistency Reg additively closes more of the Caltech gap.

FDSE's consistency regularization (Eq. near L256 of fdse.py):
    For each BN layer in the encoder, maintain running BN stats during local training
    using EMA update with global BN running stats as target. The loss:
    L_reg = sum_l w_l * [KL((mu_l^f, sigma_l^f), (mu_l^g, sigma_l^g))]
    where (mu^g, sigma^g) are global BN running stats, (mu^f, sigma^f) are
    local-during-training EMA stats, computed via forward hooks.

    Weights w_l = softmax(beta * layer_index) — later layers weighted more.

Simplification:
    Implement a KL-like L2 penalty between local BN running stats and global BN running stats
    at the start of each communication round, applied during local training.
"""
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxopt

from algorithm.feddsa import Server as BaseServer, Client as BaseClient
from algorithm.feddsa_consensus import Server as ConsensusServer
import flgo.utils.fmodule as fuf


class Server(ConsensusServer):
    def initialize(self):
        self.init_algo_para({
            'lambda_orth': 1.0,
            'lambda_hsic': 0.0,
            'lambda_sem': 1.0,
            'tau': 0.1,
            'warmup_rounds': 50,
            'style_dispatch_num': 5,
            'proj_dim': 128,
            'lambda_kl': 0.01,  # consistency reg strength (FDSE uses 0.01-0.05)
            'kl_beta': 0.1,     # layer weighting
        })
        self.sample_option = 'full'
        self.style_bank = {}
        self.global_semantic_protos = {}
        self._init_agg_keys()
        for c in self.clients:
            c.lambda_orth = self.lambda_orth
            c.lambda_hsic = self.lambda_hsic
            c.lambda_sem = self.lambda_sem
            c.tau = self.tau
            c.warmup_rounds = self.warmup_rounds
            c.proj_dim = self.proj_dim
            c.lambda_kl = self.lambda_kl
            c.kl_beta = self.kl_beta


class Client(BaseClient):
    def _collect_bn_layers(self, model):
        """Find all BN layers in the encoder (for consistency reg)."""
        layers = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) and 'encoder' in name:
                layers.append(m)
        return layers

    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum
        )

        aux_w = min(1.0, self.current_round / max(self.warmup_rounds, 1))

        # Collect BN layers + snapshot global running stats
        bn_layers = self._collect_bn_layers(model)
        global_means = [l.running_mean.clone().detach() for l in bn_layers]
        global_vars = [l.running_var.clone().detach() for l in bn_layers]
        num_bn = len(bn_layers)
        # Layer weighting: exp(beta * i) then softmax
        if num_bn > 0:
            w_layers = np.exp(np.array([self.kl_beta * i for i in range(num_bn)]))
            w_layers = w_layers / w_layers.sum()
        else:
            w_layers = []

        # Hook BN layers to capture per-step input statistics
        captured_stats = []
        hooks = []
        def make_hook():
            def hook(module, input, output):
                x = input[0]
                if x.dim() == 4:  # BatchNorm2d
                    mu = x.mean(dim=(0, 2, 3)).detach()
                    var = x.var(dim=(0, 2, 3), unbiased=False).detach()
                else:  # BatchNorm1d
                    mu = x.mean(dim=0).detach()
                    var = x.var(dim=0, unbiased=False).detach()
                captured_stats.append((mu, var))
            return hook

        for l in bn_layers:
            h = l.register_forward_hook(make_hook())
            hooks.append(h)

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
            captured_stats.clear()

            h = model.encode(x)
            z_sem = model.get_semantic(h)
            z_sty = model.get_style(h)

            output = model.head(z_sem)
            loss_task = self.loss_fn(output, y)

            loss_aug = torch.tensor(0.0, device=x.device)
            if self.local_style_bank is not None and self.current_round >= self.warmup_rounds:
                h_aug = self._style_augment(h)
                z_sem_aug = model.get_semantic(h_aug)
                output_aug = model.head(z_sem_aug)
                loss_aug = self.loss_fn(output_aug, y)

            loss_orth, loss_hsic = self._decouple_loss(z_sem, z_sty)

            loss_sem = torch.tensor(0.0, device=x.device)
            if self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_loss(z_sem, y)

            # Consistency regularization: KL of local batch BN stats vs global running stats
            loss_reg = torch.tensor(0.0, device=x.device)
            if self.current_round > 1 and self.lambda_kl > 0 and num_bn > 0:
                # Use the first num_bn captured stats (from the encoder forward pass only)
                # Note: get_semantic/get_style also use BN (bn6/bn7 are in encoder),
                # but captured_stats includes all encoder BN runs
                cap = captured_stats[:num_bn]
                for i, ((mu_f, var_f), g_mu, g_var) in enumerate(zip(cap, global_means, global_vars)):
                    g_mu = g_mu.to(x.device)
                    g_var = g_var.to(x.device) + 1e-6
                    var_f = var_f + 1e-6
                    # Symmetric KL-ish: L2 on mu + L2 on log var
                    mu_diff = ((mu_f - g_mu) ** 2).mean()
                    var_diff = ((torch.log(var_f) - torch.log(g_var)) ** 2).mean()
                    loss_reg = loss_reg + w_layers[i] * (mu_diff + 0.5 * var_diff)

            loss = loss_task + loss_aug + \
                   aux_w * self.lambda_orth * loss_orth + \
                   aux_w * self.lambda_hsic * loss_hsic + \
                   aux_w * self.lambda_sem * loss_sem + \
                   aux_w * self.lambda_kl * loss_reg

            loss.backward()
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

        for h in hooks:
            h.remove()

        self._local_protos = {c: proto_sum[c] / proto_count[c] for c in proto_sum}
        self._local_proto_counts = proto_count

        if style_n > 1:
            mu = style_sum / style_n
            var = style_sq_sum / style_n - mu ** 2
            self._local_style_stats = (mu, var.clamp(min=1e-6).sqrt())
        else:
            self._local_style_stats = None


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    from algorithm.feddsa import init_global_module as base_init
    base_init(object)
