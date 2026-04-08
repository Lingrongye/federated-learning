"""
FedDSA-Asym: Asymmetric dual-head with residual connection in semantic head.

Architecture changes vs FedDSA baseline:
1. Semantic head: deeper (3 layers) with residual connection to prevent info loss
2. Style head: shallower (1 layer) - style is just statistics
3. L2-normalized style output for stability
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


class FedDSAAsymModel(fuf.FModule):
    def __init__(self, num_classes=7, feat_dim=1024, proj_dim=128):
        super().__init__()
        self.encoder = AlexNetEncoder()

        # Deeper semantic head (3 layers) with residual
        self.sem_layer1 = nn.Linear(feat_dim, 256)
        self.sem_layer2 = nn.Linear(256, 256)
        self.sem_layer3 = nn.Linear(256, proj_dim)
        self.sem_bn1 = nn.BatchNorm1d(256)
        self.sem_bn2 = nn.BatchNorm1d(256)
        self.sem_residual = nn.Linear(feat_dim, proj_dim)  # residual from input

        # Shallow style head (1 layer, L2 normalized)
        self.style_head = nn.Linear(feat_dim, proj_dim)

        self.head = nn.Linear(proj_dim, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.encoder(x)
        z_sem = self.get_semantic(h)
        return self.head(z_sem)

    def encode(self, x):
        return self.encoder(x)

    def get_semantic(self, h):
        # 3-layer path
        x = self.relu(self.sem_bn1(self.sem_layer1(h)))
        x = self.relu(self.sem_bn2(self.sem_layer2(x)))
        x = self.sem_layer3(x)
        # Residual from input (bypass deep path)
        res = self.sem_residual(h)
        return x + 0.1 * res  # small residual weight

    def get_style(self, h):
        # Shallow + L2 normalized
        z = self.style_head(h)
        return F.normalize(z, dim=1)


class Server(flgo.algorithm.fedbase.BasicServer):
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
    pass  # Inherits everything from feddsa.py


model_map = {
    'PACS': lambda: FedDSAAsymModel(num_classes=7, feat_dim=1024, proj_dim=128),
    'office': lambda: FedDSAAsymModel(num_classes=10, feat_dim=1024, proj_dim=128),
}


def init_dataset(object): pass
def init_local_module(object): pass
def init_global_module(object):
    dataset = os.path.split(object.option['task'])[-1].split('_')[0]
    if 'Server' in object.__class__.__name__:
        object.model = model_map.get(dataset, lambda: FedDSAAsymModel())().to(object.device)
