"""
FedDSA-AugDown: Fix for the Best-Last gap bug.

Root cause (code evidence in feddsa.py L320-323):
    loss = loss_task + loss_aug + aux_w * (lambda_orth*loss_orth + ...)
                       ^^^^^^^^
    loss_aug has NO aux_w decay -> stays at weight 1.0 forever after warmup.
    This doubles the CE gradient with style-noisy features, causing late-stage oscillation.

Fix:
    Add a ramp-down schedule to loss_aug weight:
        aug_w = 1.0                       if round <= aug_peak_round (e.g. 100)
        aug_w = max(aug_min, 1.0 - (round - aug_peak_round) / aug_decay_span)
                                          if round > aug_peak_round

    Defaults: aug_peak_round=100, aug_decay_span=80, aug_min=0.1
    -> round=100: aug_w=1.0
    -> round=180: aug_w=0.1 (decay span of 80 rounds)
    -> round>=180: aug_w=0.1 (floor)
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.feddsa import Server as BaseServer, Client as BaseClient
import flgo.utils.fmodule as fuf


class Server(BaseServer):
    def initialize(self):
        self.init_algo_para({
            'lambda_orth': 1.0,
            'lambda_hsic': 0.0,
            'lambda_sem': 1.0,
            'tau': 0.1,
            'warmup_rounds': 50,
            'style_dispatch_num': 5,
            'proj_dim': 128,
            'aug_peak_round': 100,
            'aug_decay_span': 80,
            'aug_min': 0.1,
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
            c.aug_peak_round = self.aug_peak_round
            c.aug_decay_span = self.aug_decay_span
            c.aug_min = self.aug_min


class Client(BaseClient):
    def _aug_weight(self):
        """Ramp-down schedule for loss_aug weight."""
        r = self.current_round
        if r < self.warmup_rounds:
            return 0.0
        if r <= self.aug_peak_round:
            return 1.0
        progress = (r - self.aug_peak_round) / max(self.aug_decay_span, 1)
        return max(self.aug_min, 1.0 - progress)

    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum
        )

        aux_w = min(1.0, self.current_round / max(self.warmup_rounds, 1))
        aug_w = self._aug_weight()

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

            output = model.head(z_sem)
            loss_task = self.loss_fn(output, y)

            loss_aug = torch.tensor(0.0, device=x.device)
            if (self.local_style_bank is not None
                    and self.current_round >= self.warmup_rounds
                    and aug_w > 0.0):
                h_aug = self._style_augment(h)
                z_sem_aug = model.get_semantic(h_aug)
                output_aug = model.head(z_sem_aug)
                loss_aug = self.loss_fn(output_aug, y)

            loss_orth, loss_hsic = self._decouple_loss(z_sem, z_sty)

            loss_sem = torch.tensor(0.0, device=x.device)
            if self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_loss(z_sem, y)

            # --- THE FIX: aug_w * loss_aug instead of plain loss_aug ---
            loss = loss_task + aug_w * loss_aug + \
                   aux_w * self.lambda_orth * loss_orth + \
                   aux_w * self.lambda_hsic * loss_hsic + \
                   aux_w * self.lambda_sem * loss_sem

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
