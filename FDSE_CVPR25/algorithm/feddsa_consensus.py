"""
FedDSA-Consensus: Replace FedAvg with consensus-aware aggregation for shared parameters.

Diagnosis (from EXP-060~063):
    All four style-side fixes (gated, noaug, softbeta, augschedule) HURT Office.
    This proves Office's failure is NOT about style augmentation being too aggressive.
    Following GPT-5.4 Q2 analysis, Office's bottleneck is aggregation conflict
    (subtle client drift causing destructive FedAvg averaging on shared params).

Fix:
    Keep everything about FedDSA (dual-head, style bank, AdaIN, orth, InfoNCE) intact.
    Only replace FedAvg aggregation of shared parameters with FDSE's consensus-
    maximization aggregation (minimize L2 of aggregated update direction).

    Adapted from fdse.py L124-144 + optim_lambda + quadprog.

Keeps FedDSA identity:
    - Dual-head decouple + orth loss ✓
    - Global style bank + AdaIN augmentation ✓
    - InfoNCE prototype alignment ✓
    - style_head private ✓
Only changes HOW shared parameters are aggregated on the server.
"""
import os
import sys
import copy
import numpy as np
import torch
import cvxopt

from algorithm.feddsa import Server as BaseServer, Client as BaseClient


class Server(BaseServer):
    def iterate(self):
        """Override to use consensus-maximization aggregation instead of weighted FedAvg."""
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)

        models = res['model']
        protos_list = res['protos']
        proto_counts_list = res['proto_counts']
        style_stats_list = res['style_stats']

        # 1. Aggregate shared parameters via CONSENSUS MAXIMIZATION
        self._aggregate_shared_consensus(models)

        # 2. Collect style bank (unchanged)
        for cid, style in zip(self.received_clients, style_stats_list):
            if style is not None:
                self.style_bank[cid] = style

        # 3. Aggregate prototypes (unchanged)
        self._aggregate_protos(protos_list, proto_counts_list)

    def _aggregate_shared_consensus(self, models):
        """
        Consensus-maximization aggregation (FDSE Eq. 8).
        For each shared layer, find aggregation weights u that minimize
        ||sum(u_k * d_k)||^2 where d_k is the normalized update direction of client k.
        """
        if len(models) == 0:
            return

        mdicts = [m.state_dict() for m in models]
        global_dict = self.model.state_dict()

        current_shared = {k: global_dict[k] for k in self.shared_keys}
        shared_dicts = [{k: md[k] for k in self.shared_keys} for md in mdicts]
        new_shared = {}

        for k in self.shared_keys:
            if 'num_batches_tracked' in k:
                continue

            # For BN running stats, just use mean (server doesn't train so these stay at init anyway)
            if 'running_' in k:
                new_shared[k] = torch.stack([md[k] for md in shared_dicts]).mean(dim=0)
                continue

            shape = shared_dicts[0][k].shape
            crt_vec = current_shared[k].reshape(-1).to(self.device)

            # Compute update directions: u_k = (theta_k - theta_global)
            k_vecs = [md[k].reshape(-1).to(self.device) - crt_vec for md in shared_dicts]
            k_norms = [t.norm() for t in k_vecs]

            # Handle degenerate case: all updates zero
            if all(n.item() < 1e-12 for n in k_norms):
                new_shared[k] = current_shared[k]
                continue

            # Normalize directions
            k_vecs_normed = [t / (tn + 1e-8) for t, tn in zip(k_vecs, k_norms)]

            # Optimize u to minimize ||sum(u_k * d_k)||^2 s.t. u_k >= 0, sum(u_k) = 1
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                op_lambda = self._optim_lambda(k_vecs_normed)
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout

            op_lambda = torch.tensor([ele[0] for ele in op_lambda]).float().to(self.device)

            # Reconstruct: sum(u_k * d_k) * mean(||u_k||) + theta_global
            new_dir = (op_lambda.unsqueeze(0) @ torch.stack(k_vecs_normed))[0]
            new_shared[k] = (torch.stack(k_norms).mean() * new_dir + crt_vec).reshape(shape)

        # Update the server model
        updated_dict = global_dict.copy()
        for k, v in new_shared.items():
            updated_dict[k] = v
        self.model.load_state_dict(updated_dict, strict=False)

    def _optim_lambda(self, grads):
        """Solve quadratic program to minimize ||sum(lambda_k * grad_k)||^2
        s.t. lambda_k >= 0, sum(lambda_k) = 1, lambda_k <= 1."""
        n = len(grads)
        Jt = np.array([g.detach().cpu().numpy() for g in grads])
        P = 2 * np.dot(Jt, Jt.T)  # [n, n]
        q = np.array([[0] for _ in range(n)])
        A = np.ones(n).T
        b = np.array([1])
        lb = np.zeros(n)
        ub = np.ones(n)
        G = np.zeros((2 * n, n))
        for i in range(n):
            G[i][i] = -1
            G[n + i][i] = 1
        h = np.zeros((2 * n, 1))
        for i in range(n):
            h[i] = -lb[i]
            h[n + i] = ub[i]
        return self._quadprog(P, q, G, h, A, b)

    def _quadprog(self, P, q, G, h, A, b):
        P = cvxopt.matrix(P.tolist())
        q = cvxopt.matrix(q.tolist(), tc='d')
        G = cvxopt.matrix(G.tolist())
        h = cvxopt.matrix(h.tolist())
        A = cvxopt.matrix(A.tolist())
        b = cvxopt.matrix(b.tolist(), tc='d')
        sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
        return np.array(sol['x'])


class Client(BaseClient):
    """Same as base — no client-side changes."""
    pass


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    from algorithm.feddsa import init_global_module as base_init
    base_init(object)
