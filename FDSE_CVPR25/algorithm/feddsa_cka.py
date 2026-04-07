"""
FedDSA-CKA: Replace cos²(z_sem, z_sty) orthogonality with 1 - CKA similarity.

CKA (Centered Kernel Alignment) is a smoother measure of representation similarity.
It has more stable gradients than squared cosine.

linear CKA(X, Y) = ||Y'X||_F^2 / (||X'X||_F * ||Y'Y||_F)

loss_decouple = 1 - linear_CKA(z_sem, z_sty)
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.feddsa import FedDSAModel, AlexNetEncoder, Server as BaseServer, Client as BaseClient
import flgo.utils.fmodule as fuf


class Server(BaseServer):
    def initialize(self):
        self.init_algo_para({
            'lambda_orth': 1.0,  # now weighs CKA loss
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
    def _decouple_loss(self, z_sem, z_sty):
        """Replace cos² with (1 - linear CKA). Smoother gradient."""
        # Center the features
        X = z_sem - z_sem.mean(dim=0, keepdim=True)
        Y = z_sty - z_sty.mean(dim=0, keepdim=True)

        # Linear CKA
        # CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
        yx = torch.matmul(Y.t(), X)
        xx = torch.matmul(X.t(), X)
        yy = torch.matmul(Y.t(), Y)

        num = (yx ** 2).sum()
        den = torch.sqrt(((xx ** 2).sum() * (yy ** 2).sum()).clamp(min=1e-8))

        cka = num / (den + 1e-8)
        loss_cka = cka  # minimizing CKA = pushing decorrelation

        # HSIC disabled (we found it harmful)
        loss_hsic = torch.tensor(0.0, device=z_sem.device)
        return loss_cka, loss_hsic


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
