"""
FedDSA-RegimeGated: Regime-adaptive aggregation + style graph dispatch using
the decoupled style bank.

Motivation (from EXP-064 + EXP-066 findings):
    - Consensus QP aggregation wins on PACS (high style-gap regime) but only
      breaks even on Office (low style-gap regime, all real photos).
    - FDSE's KL BN regularization is too rigid and seed-sensitive.
    - Our decoupled style bank provides TWO free signals:
        1. regime score r = mean pairwise style distance across clients
        2. style graph G with edges d_ij = pairwise style distance

Method (single clean policy, two touch-points on the server):

    [1] Style graph neighbor dispatch
        Replace random K-style dispatch with K-nearest-neighbor dispatch in
        style space. Ensures on-manifold augmentation:
            - PACS: nearest neighbors are still other domains (still diverse)
            - Office: nearest neighbors are very close → gentle augmentation,
              avoids off-manifold noise from outlier styles

    [2] Regime-gated aggregation
        a. Always perform Consensus QP aggregation first (inherited from
           feddsa_consensus).
        b. Compute regime score r from the current style bank.
        c. If r < regime_threshold AND we have a previous aggregated
           pseudo-gradient, apply a server-side SAM look-ahead step:
              w ← w_consensus + sam_rho * prev_pseudo_grad / ||prev_pseudo_grad||
           This is FedGloSS-style server-side SAM with zero extra communication
           (reuses previous round's pseudo-gradient as the perturbation direction).
        d. Record current aggregated pseudo-gradient for the next round.

Key distinction from FedGloSS / FedSOL / FDSE / CCST:
    - FedGloSS: fixed-rho server-side SAM, no regime awareness, no style graph
    - FedSOL: client-side orthogonal proximal, parameter-space, no style
    - FDSE: consensus QP alone, no regime awareness
    - CCST: has style bank + random dispatch, no graph structure, no regime
    - Ours: style-graph dispatch + regime-gated server SAM, both driven by the
      decoupled style distance signal

Thesis:
    "The decoupled style bank gives us two free signals — a regime score and
     a style graph — that together let the server perform regime-adaptive
     aggregation AND on-manifold style dispatch. Low gap needs flatness +
     gentle augmentation; high gap needs consensus + diverse augmentation."

Keeps FedDSA identity:
    - Dual-head decouple + orth loss ✓
    - Global style bank + AdaIN augmentation ✓
    - InfoNCE prototype alignment ✓
    - style_head private ✓
"""
import os
import sys
import copy
import numpy as np
import torch

from algorithm.feddsa_consensus import Server as ConsensusServer, Client as ConsensusClient
from algorithm.feddsa import Server as FedDSAServer  # for pack() override reference


class Server(ConsensusServer):
    def initialize(self):
        # Call parent initialize which sets up init_algo_para
        super().initialize()

        # Extra hyper-parameters (not in algo_para tuple to avoid breaking parent).
        # They default to reasonable values and can be overridden via algo_para
        # if we extend the config schema in a future experiment.
        self.regime_threshold = getattr(self, 'regime_threshold', 0.0)
        self.sam_rho = getattr(self, 'sam_rho', 0.05)

        # Persistent state for SAM look-ahead
        self.prev_pseudo_grad = None
        self.regime_history = []  # list of (round, r, strategy)

    def _aggregate_shared_consensus(self, models):
        """Regime-gated aggregation wrapping the parent consensus QP.

        Flow:
            1. snapshot current global state (pre-aggregation)
            2. run parent consensus QP aggregation (mutates self.model)
            3. compute current aggregated pseudo-gradient d_t = w_new - w_old
            4. compute regime score r from style bank
            5. if r < regime_threshold AND prev_pseudo_grad is available:
                  apply SAM look-ahead: w ← w + sam_rho * prev / ||prev||
            6. store d_t as prev_pseudo_grad for next round
        """
        if len(models) == 0:
            return

        # ---- Step 1: snapshot pre-aggregation global state ----
        pre_state = {}
        global_dict = self.model.state_dict()
        for k in self.shared_keys:
            if 'num_batches_tracked' in k or 'running_' in k:
                continue
            pre_state[k] = global_dict[k].detach().clone()

        # ---- Step 2: run parent consensus QP (mutates self.model) ----
        super()._aggregate_shared_consensus(models)

        # ---- Step 3: compute current pseudo-gradient d_t = w_new - w_old ----
        post_state = self.model.state_dict()
        curr_pseudo_grad = {}
        for k, w_old in pre_state.items():
            curr_pseudo_grad[k] = (post_state[k] - w_old).detach()

        # ---- Step 4: compute regime score ----
        r = self._compute_regime_score()
        strategy = 'consensus'

        # ---- Step 5: regime-gated SAM look-ahead ----
        if (r is not None and r < self.regime_threshold
                and self.prev_pseudo_grad is not None):
            strategy = 'consensus+sam'
            self._apply_sam_lookahead()

        # ---- Step 6: log and save ----
        self.regime_history.append({
            'round': self.current_round,
            'r': r if r is not None else -1.0,
            'strategy': strategy,
            'num_bank_clients': len(self.style_bank),
        })
        if self.current_round % 10 == 0:
            print(f"[RegimeGated] round={self.current_round} "
                  f"r={r if r is not None else 'NA'} "
                  f"strategy={strategy} bank_size={len(self.style_bank)}",
                  flush=True)

        self.prev_pseudo_grad = curr_pseudo_grad

    def _compute_regime_score(self):
        """Mean pairwise style distance from the style bank.

        d_ij = ||mu_i - mu_j||_2 + ||log sigma_i - log sigma_j||_2

        Returns None if the bank has fewer than 2 clients (warmup).
        """
        if len(self.style_bank) < 2:
            return None

        cids = list(self.style_bank.keys())
        distances = []
        for i in range(len(cids)):
            mu_i, sigma_i = self.style_bank[cids[i]]
            for j in range(i + 1, len(cids)):
                mu_j, sigma_j = self.style_bank[cids[j]]
                mu_d = (mu_i - mu_j).norm().item()
                log_s_i = torch.log(sigma_i.clamp(min=1e-6))
                log_s_j = torch.log(sigma_j.clamp(min=1e-6))
                sig_d = (log_s_i - log_s_j).norm().item()
                distances.append(mu_d + sig_d)

        if len(distances) == 0:
            return None
        return float(np.mean(distances))

    def _apply_sam_lookahead(self):
        """Server-side SAM look-ahead using previous round's aggregated pseudo-gradient.

        w_new = w_consensus + sam_rho * prev_pseudo_grad / ||prev_pseudo_grad||
        """
        if self.prev_pseudo_grad is None:
            return

        # Compute global norm of prev_pseudo_grad over all tracked keys
        sq_sum = 0.0
        for g in self.prev_pseudo_grad.values():
            sq_sum += float((g ** 2).sum().item())
        total_norm = float(np.sqrt(sq_sum) + 1e-12)

        updated = self.model.state_dict()
        for k, g in self.prev_pseudo_grad.items():
            if k not in updated:
                continue
            # Same shape check - prev may have stale keys
            if updated[k].shape != g.shape:
                continue
            scale = self.sam_rho / total_norm
            updated[k] = updated[k] + g * scale

        self.model.load_state_dict(updated, strict=False)

    # ------------------------------------------------------------------
    # Style graph neighbor dispatch (replaces random dispatch in pack())
    # ------------------------------------------------------------------

    def pack(self, client_id, mtype=0):
        """Override the random style dispatch with K-nearest-neighbor dispatch
        based on the style graph.

        All other parts of pack() are identical to feddsa.Server.pack().
        """
        # K-nearest style dispatch (excluding self)
        dispatched_styles = None
        if len(self.style_bank) > 0 and self.current_round >= self.warmup_rounds:
            available = {cid: s for cid, s in self.style_bank.items() if cid != client_id}
            if len(available) == 0:
                available = self.style_bank
            n = min(self.style_dispatch_num, len(available))
            dispatched_styles = self._dispatch_knn_styles(client_id, available, n)

        return {
            'model': copy.deepcopy(self.model),
            'global_protos': copy.deepcopy(self.global_semantic_protos),
            'style_bank': dispatched_styles,
            'current_round': self.current_round,
        }

    def _dispatch_knn_styles(self, client_id, available, k):
        """Return the k styles in `available` whose (mu, sigma) are nearest
        in style distance to client_id's own style.

        Distance: d_ij = ||mu_i - mu_j||_2 + ||log sigma_i - log sigma_j||_2

        If client_id is not in the global bank yet (early rounds), fall back
        to random selection so we don't crash or favor an arbitrary direction.
        """
        my_style = self.style_bank.get(client_id)
        if my_style is None:
            # Fall back to random (first round after warmup for a fresh client)
            keys = list(available.keys())
            chosen = np.random.choice(keys, k, replace=False)
            return [available[c] for c in chosen]

        mu_i, sigma_i = my_style
        log_s_i = torch.log(sigma_i.clamp(min=1e-6))

        scored = []
        for cid, (mu_j, sigma_j) in available.items():
            mu_d = (mu_i - mu_j).norm().item()
            log_s_j = torch.log(sigma_j.clamp(min=1e-6))
            sig_d = (log_s_i - log_s_j).norm().item()
            scored.append((mu_d + sig_d, cid))

        # K nearest: smallest distance first
        scored.sort(key=lambda t: t[0])
        chosen_cids = [cid for _, cid in scored[:k]]
        return [available[c] for c in chosen_cids]

    def _style_graph_edges(self):
        """Return all pairwise edges (i, j, d_ij) of the current style graph.
        Useful for logging and debugging. i < j only (undirected)."""
        cids = list(self.style_bank.keys())
        edges = []
        for a in range(len(cids)):
            mu_a, sig_a = self.style_bank[cids[a]]
            log_a = torch.log(sig_a.clamp(min=1e-6))
            for b in range(a + 1, len(cids)):
                mu_b, sig_b = self.style_bank[cids[b]]
                log_b = torch.log(sig_b.clamp(min=1e-6))
                d = (mu_a - mu_b).norm().item() + (log_a - log_b).norm().item()
                edges.append((cids[a], cids[b], d))
        return edges


class Client(ConsensusClient):
    """Unchanged from consensus baseline — all regime logic is server-side."""
    pass


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    from algorithm.feddsa import init_global_module as base_init
    base_init(object)
