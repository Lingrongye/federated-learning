"""
FedDSA-PCGrad: Project Conflicting Gradients (NeurIPS 2020) to resolve multi-task conflicts.

For each pair of task gradients (g_i, g_j), if they conflict (cos(g_i, g_j) < 0),
project g_i onto the orthogonal complement of g_j.

Core idea:
  if g_i . g_j < 0:
      g_i = g_i - (g_i.g_j / ||g_j||^2) * g_j

We have 3 effective loss tasks: task (+aug), orth, sem.
"""
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.feddsa import FedDSAModel, AlexNetEncoder, Server as BaseServer, Client as BaseClient
import flgo.utils.fmodule as fuf


def pcgrad_project(grads_list):
    """
    Project conflicting gradients.
    grads_list: list of gradient dicts, one per loss
    Returns a single merged gradient dict with conflicts removed.
    """
    # Flatten each gradient dict to a single vector for cosine computation
    param_names = list(grads_list[0].keys())
    flat_grads = []
    for g_dict in grads_list:
        flat = torch.cat([g_dict[k].view(-1) for k in param_names])
        flat_grads.append(flat)

    n = len(flat_grads)
    projected = [g.clone() for g in flat_grads]

    # Random order for each task (PCGrad paper uses random ordering)
    for i in range(n):
        order = list(range(n))
        random.shuffle(order)
        for j in order:
            if i == j:
                continue
            g_j = flat_grads[j]
            dot = (projected[i] * g_j).sum()
            if dot < 0:
                projected[i] = projected[i] - (dot / (g_j.norm() ** 2 + 1e-12)) * g_j

    # Sum projected gradients
    final_flat = torch.stack(projected).sum(dim=0)

    # Unflatten back to dict
    result = {}
    offset = 0
    for k in param_names:
        shape = grads_list[0][k].shape
        numel = grads_list[0][k].numel()
        result[k] = final_flat[offset:offset+numel].view(shape)
        offset += numel
    return result


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


class Client(BaseClient):
    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum
        )

        aux_w = min(1.0, self.current_round / max(self.warmup_rounds, 1))

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

            # Compute each loss separately
            output = model.head(z_sem)
            loss_task = self.loss_fn(output, y)

            loss_aug = torch.tensor(0.0, device=x.device)
            if self.local_style_bank is not None and self.current_round >= self.warmup_rounds:
                h_aug = self._style_augment(h)
                z_sem_aug = model.get_semantic(h_aug)
                output_aug = model.head(z_sem_aug)
                loss_aug = self.loss_fn(output_aug, y)

            loss_orth, _ = self._decouple_loss(z_sem, z_sty)

            loss_sem = torch.tensor(0.0, device=x.device)
            if self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_loss(z_sem, y)

            # Merge task and aug into one "classification" loss
            loss_cls = loss_task + loss_aug
            loss_orth_w = aux_w * self.lambda_orth * loss_orth
            loss_sem_w = aux_w * self.lambda_sem * loss_sem

            # PCGrad: compute gradient for each loss separately
            losses = [loss_cls, loss_orth_w, loss_sem_w]
            # Skip zero losses
            active_losses = [l for l in losses if l.requires_grad and l.item() > 0]

            if len(active_losses) <= 1:
                # Only one active loss, no projection needed
                if active_losses:
                    active_losses[0].backward()
            else:
                # Compute gradients separately
                grads_list = []
                for i, loss_i in enumerate(active_losses):
                    optimizer.zero_grad()
                    retain = (i < len(active_losses) - 1)
                    loss_i.backward(retain_graph=retain)
                    grads = {}
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            grads[name] = p.grad.clone()
                        else:
                            grads[name] = torch.zeros_like(p)
                    grads_list.append(grads)

                # Project conflicting gradients
                final_grads = pcgrad_project(grads_list)

                # Apply projected gradient
                optimizer.zero_grad()
                for name, p in model.named_parameters():
                    if name in final_grads:
                        p.grad = final_grads[name]

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
