"""
FedDSA-Triplet: Replace InfoNCE with Triplet Margin Loss.

InfoNCE with tau=0.1 is very sharp and unstable. Triplet loss is a smoother alternative
used in FISC/PARDON paper for similar purposes.

loss = max(0, d(z, anchor_same) - d(z, anchor_diff) + margin)

This only changes _infonce_loss to _triplet_loss. Everything else identical to feddsa.py.
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse everything from feddsa.py
from algorithm.feddsa import FedDSAModel, AlexNetEncoder, Server as BaseServer, Client as BaseClient
import flgo.utils.fmodule as fuf


class Server(BaseServer):
    def initialize(self):
        self.init_algo_para({
            'lambda_orth': 1.0,
            'lambda_hsic': 0.0,
            'lambda_sem': 1.0,
            'margin': 0.3,          # NEW: triplet margin
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
            c.margin = self.margin
            c.warmup_rounds = self.warmup_rounds
            c.proj_dim = self.proj_dim
            # For Client's unused tau attribute
            c.tau = 0.1


class Client(BaseClient):
    def _infonce_loss(self, z_sem, y):
        """Replaced: Triplet margin loss instead of InfoNCE."""
        available = sorted([c for c, p in self.global_protos.items() if p is not None])
        if len(available) < 2:
            return torch.tensor(0.0, device=z_sem.device)

        proto_matrix = torch.stack([self.global_protos[c].to(z_sem.device) for c in available])
        class_to_idx = {c: i for i, c in enumerate(available)}

        # Normalize for cosine distance
        z_n = F.normalize(z_sem, dim=1)
        p_n = F.normalize(proto_matrix, dim=1)

        losses = []
        for i in range(y.size(0)):
            label = y[i].item()
            if label not in class_to_idx:
                continue
            anchor_idx = class_to_idx[label]
            # Distance to positive (same class)
            d_pos = 1.0 - (z_n[i] * p_n[anchor_idx]).sum()
            # Minimum distance to all negatives (other classes)
            neg_mask = torch.ones(len(available), device=z_sem.device)
            neg_mask[anchor_idx] = 0
            dists_to_all = 1.0 - (p_n @ z_n[i])  # [C]
            dists_to_neg = dists_to_all + (1 - neg_mask) * 1e9  # mask positive
            d_neg = dists_to_neg.min()
            losses.append(F.relu(d_pos - d_neg + self.margin))

        if len(losses) == 0:
            return torch.tensor(0.0, device=z_sem.device)
        return torch.stack(losses).mean()


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
