"""
FedDSA-SoftBeta: Use Beta(1, 1) uniform mixing instead of Beta(0.1, 0.1) extreme.

Hypothesis test:
    Original Beta(0.1, 0.1) is heavily U-shaped — α is almost always near 0 or 1.
    α≈1: no mixing (no value); α≈0: full external style (too aggressive).

    Beta(1, 1) is uniform — α ~ U[0, 1] — produces moderate mixing.
    This is a SOFTER version of the style sharing mechanism.

    If this improves Office without hurting PACS → Beta(0.1, 0.1) was indeed
    too aggressive and the fix is just softer mixing.
"""
import numpy as np
import torch
from algorithm.feddsa import Server as BaseServer, Client as BaseClient
import flgo.utils.fmodule as fuf


class Server(BaseServer):
    pass


class Client(BaseClient):
    def _style_augment(self, h):
        """AdaIN with Beta(1,1) uniform mixing (softer than Beta(0.1,0.1))."""
        idx = np.random.randint(0, len(self.local_style_bank))
        mu_ext, sigma_ext = self.local_style_bank[idx]
        mu_ext = mu_ext.to(h.device)
        sigma_ext = sigma_ext.to(h.device)

        mu_local = h.mean(dim=0)
        sigma_local = h.std(dim=0, unbiased=False).clamp(min=1e-6)

        # Beta(1, 1) = Uniform(0, 1) — softer than Beta(0.1, 0.1)
        alpha = np.random.beta(1.0, 1.0)
        mu_mix = alpha * mu_local + (1 - alpha) * mu_ext
        sigma_mix = alpha * sigma_local + (1 - alpha) * sigma_ext

        h_norm = (h - mu_local) / sigma_local
        return h_norm * sigma_mix + mu_mix


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    from algorithm.feddsa import init_global_module as base_init
    base_init(object)
