"""
FedDSA-Gated: Distance-gated style dispatch.

Design insight (from GPT-5.4 research review):
    Current FedDSA does "unconditional raw style sharing" — dispatches random styles
    regardless of how similar they are. This works for large-style-gap datasets
    (PACS: sketch vs photo), but fails for small-style-gap datasets (Office: all
    real photos), where AdaIN mixing becomes off-manifold noise injection.

Fix:
    Compute inter-client style distance on server. Only dispatch styles whose
    distance to the client's own style exceeds a threshold. If all styles are
    too similar, skip style augmentation entirely.

    distance(i, j) = ||mu_i - mu_j||_2 + ||log(sigma_i) - log(sigma_j)||_2

Behavior:
    - PACS: all pairs have large d → all styles pass threshold → normal dispatch
    - Office: all pairs have small d → no styles pass → no style aug (effectively)

This preserves FedDSA identity (decouple + share + align) while making style
sharing regime-aware.

Hyperparameters added:
    dist_threshold (default 1.0): minimum style distance to dispatch
    dispatch_top_k_by_dist (default True): prefer most-distant styles
"""
import copy
import numpy as np
import torch

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
            'dist_threshold': 1.0,  # minimum distance to dispatch
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

    def _style_distance(self, a, b):
        """Compute distance between two style (mu, sigma) tuples."""
        mu_a, sigma_a = a
        mu_b, sigma_b = b
        mu_diff = (mu_a - mu_b).norm(p=2).item()
        log_sigma_diff = (torch.log(sigma_a.clamp(min=1e-6)) -
                          torch.log(sigma_b.clamp(min=1e-6))).norm(p=2).item()
        return mu_diff + log_sigma_diff

    def pack(self, client_id, mtype=0):
        """Distance-gated dispatch: only send styles sufficiently distant from own style."""
        dispatched_styles = None

        if len(self.style_bank) > 0 and self.current_round >= self.warmup_rounds:
            available = {cid: s for cid, s in self.style_bank.items() if cid != client_id}

            if len(available) > 0 and client_id in self.style_bank:
                # Have own style → compute distances
                own_style = self.style_bank[client_id]
                distances = []
                for cid, style in available.items():
                    d = self._style_distance(own_style, style)
                    distances.append((d, cid, style))

                # Sort by distance descending (most distant first)
                distances.sort(key=lambda x: -x[0])

                # Log once per round (only for first client) for diagnostics
                if not hasattr(self, '_logged_dist_round') or self._logged_dist_round != self.current_round:
                    self._logged_dist_round = self.current_round
                    d_max = distances[0][0] if distances else 0
                    d_min = distances[-1][0] if distances else 0
                    d_mean = sum(d for d, _, _ in distances) / max(len(distances), 1)
                    if hasattr(self, 'gv') and hasattr(self.gv, 'logger'):
                        self.gv.logger.info(
                            f"[R{self.current_round}] style_dist: max={d_max:.3f} mean={d_mean:.3f} "
                            f"min={d_min:.3f} threshold={self.dist_threshold}"
                        )

                # Gate: skip dispatch if most distant is still below threshold
                if distances[0][0] < self.dist_threshold:
                    dispatched_styles = None  # no style aug this round
                else:
                    # Take top-K most distant among those above threshold
                    n = min(self.style_dispatch_num, len(distances))
                    chosen = [(d, cid, style) for d, cid, style in distances[:n]
                              if d >= self.dist_threshold]
                    dispatched_styles = [style for _, _, style in chosen] if chosen else None
            elif len(available) > 0:
                # Fallback to random (shouldn't normally happen)
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


class Client(BaseClient):
    """Same as base."""
    pass


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    from algorithm.feddsa import init_global_module as base_init
    base_init(object)
