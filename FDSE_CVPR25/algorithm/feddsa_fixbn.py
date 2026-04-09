"""
FedDSA-FixBN: Fix the BN running stats aggregation bug.

Bug (verified in feddsa.py L128-137):
    Server marks BN running_mean/running_var as `private_keys` and skips them
    during aggregation. But the server NEVER trains (it only aggregates), so its
    BN running stats stay at init (running_mean=0, running_var=1) forever.
    All evaluations use server model -> BN effectively becomes identity +
    learned affine at test time. This causes:
      - train/test distribution mismatch
      - huge Best-Last gap (~5% vs FDSE's ~2%)
      - seed-dependent fluctuation

Fix (Option A - matches FDSE's approach):
    Only mark `style_head` as private on the server. Aggregate ALL BN keys
    (weight, bias, running_mean, running_var) via FedAvg. This ensures the
    server's test-time BN reflects the averaged client statistics.

    Client-side unpack() still skips BN running stats (FedBN local behavior
    for training), so clients keep their own domain-specific stats.
    This hybrid matches FDSE:
      - Server: aggregated stats (for testing)
      - Client: local stats (for training)
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.feddsa import Server as BaseServer, Client as BaseClient
import flgo.utils.fmodule as fuf


class Server(BaseServer):
    def _init_agg_keys(self):
        """Only style_head is private. BN stats aggregated like FDSE."""
        all_keys = list(self.model.state_dict().keys())
        self.private_keys = set()
        for k in all_keys:
            if 'style_head' in k:
                self.private_keys.add(k)
        self.shared_keys = [k for k in all_keys if k not in self.private_keys]


class Client(BaseClient):
    """Reuse base client behavior; local BN stats still skipped in unpack()."""
    pass


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    from algorithm.feddsa import init_global_module as base_init
    base_init(object)
