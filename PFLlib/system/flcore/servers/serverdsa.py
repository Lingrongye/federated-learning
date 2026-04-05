"""
Server for DSA (Decouple-Share-Align) — Decoupled Prototype Learning with Style Asset Sharing

Differential aggregation:
- Backbone conv layers + Semantic head + Semantic classifier → FedAvg
- Style head → Private (not aggregated)
- BN layers → Private (FedBN principle)
"""
import copy
import time
import torch
import torch.nn.functional as F
import numpy as np
from flcore.clients.clientdsa import clientDSA
from flcore.servers.serverbase import Server


class FedDSA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientDSA)

        # Global semantic prototypes (per-class)
        self.global_semantic_protos = {}

        # Global style bank: list of (mu, sigma, client_id) tuples
        self.style_bank = []
        self.style_dedup_threshold = args.style_dedup if hasattr(args, 'style_dedup') else 0.95
        self.style_bank_max_size = args.style_bank_max if hasattr(args, 'style_bank_max') else 50
        self.style_dispatch_num = args.style_dispatch_num if hasattr(args, 'style_dispatch_num') else 5
        self.warmup_rounds = args.warmup_rounds if hasattr(args, 'warmup_rounds') else 10

        print(f"\n{'='*50}")
        print(f"FedDSA: Decouple-Share-Align")
        print(f"Join ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Style bank max size: {self.style_bank_max_size}")
        print(f"Style dispatch per client: {self.style_dispatch_num}")
        print(f"Warmup rounds: {self.warmup_rounds}")
        print(f"{'='*50}\n")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()

            self.selected_clients = self.select_clients()

            # Send global model + prototypes + style bank
            self.send_models()
            self.send_protos_and_styles(i)

            if i % self.eval_gap == 0:
                print(f"\n--- Round {i} ---")
                print(f"Style bank size: {len(self.style_bank)}")
                self.evaluate()

            # Client local training
            for client in self.selected_clients:
                client.set_round(i)
                client.train()

            # Receive and aggregate
            self.receive_models()

            # FIX: Build active client list from uploaded_ids to handle dropout
            active_clients = [self.clients[cid] for cid in self.uploaded_ids]

            self.aggregate_parameters_no_bn()
            self.aggregate_semantic_heads(active_clients)
            self.collect_styles(active_clients)
            self.aggregate_semantic_protos(active_clients)

            self.Budget.append(time.time() - s_t)
            print(f'  Round {i} time: {self.Budget[-1]:.1f}s')

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print(f"\nBest accuracy: {max(self.rs_test_acc):.4f}")
        print(f"Avg time/round: {sum(self.Budget[1:])/max(len(self.Budget)-1, 1):.1f}s")

        self.save_results()
        self.save_global_model()

    def send_protos_and_styles(self, round_num):
        """Send global semantic prototypes and style bank subset to each client."""
        for client in self.clients:
            client.set_global_semantic_protos(
                copy.deepcopy(self.global_semantic_protos)
            )

            if len(self.style_bank) > 0 and round_num >= self.warmup_rounds:
                n_dispatch = min(self.style_dispatch_num, len(self.style_bank))
                # Exclude client's own style if possible
                available_styles = [
                    s for s in self.style_bank if s[2] != client.id
                ]
                if len(available_styles) == 0:
                    available_styles = self.style_bank
                n_dispatch = min(n_dispatch, len(available_styles))
                indices = np.random.choice(
                    len(available_styles), n_dispatch, replace=False
                )
                dispatched = [(available_styles[j][0], available_styles[j][1])
                              for j in indices]
                client.set_style_bank(dispatched)
            else:
                client.set_style_bank(None)

    def aggregate_parameters_no_bn(self):
        """FedAvg aggregation excluding BN layers."""
        assert len(self.uploaded_models) > 0

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        global_dict = self.global_model.state_dict()

        for key in global_dict.keys():
            if 'bn' in key.lower() or 'running_' in key or 'num_batches_tracked' in key:
                continue

            weighted_sum = torch.zeros_like(global_dict[key])
            for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
                weighted_sum += w * client_model.state_dict()[key]
            global_dict[key] = weighted_sum

        self.global_model.load_state_dict(global_dict)

    def aggregate_semantic_heads(self, active_clients):
        """Aggregate semantic head + classifier parameters across active clients."""
        if not active_clients:
            return

        head_params_list = []
        weights = []
        for client in active_clients:
            head_params_list.append(client.get_semantic_head_params())
            weights.append(client.train_samples)

        total = sum(weights)
        weights = [w / total for w in weights]

        avg_params = {}
        for key in head_params_list[0].keys():
            avg_params[key] = torch.zeros_like(head_params_list[0][key])
            for w, params in zip(weights, head_params_list):
                avg_params[key] += w * params[key]

        # Send to ALL clients (not just active)
        for client in self.clients:
            client.set_semantic_head_params(copy.deepcopy(avg_params))

    def collect_styles(self, active_clients):
        """Collect style prototypes from active clients and manage the bank."""
        for client in active_clients:
            if client.style_proto_mu is not None:
                mu = client.style_proto_mu.detach().cpu()
                sigma = client.style_proto_sigma.detach().cpu()

                # Check for duplicates via cosine similarity
                is_dup = False
                for existing_mu, _, _ in self.style_bank:
                    cos_sim = F.cosine_similarity(
                        mu.unsqueeze(0), existing_mu.unsqueeze(0)
                    ).item()
                    if cos_sim > self.style_dedup_threshold:
                        # Update existing entry instead of skipping
                        is_dup = True
                        break

                if not is_dup:
                    self.style_bank.append((mu, sigma, client.id))

        # Trim bank if too large (keep most recent)
        if len(self.style_bank) > self.style_bank_max_size:
            self.style_bank = self.style_bank[-self.style_bank_max_size:]

    def aggregate_semantic_protos(self, active_clients):
        """Aggregate per-class semantic prototypes weighted by sample count."""
        proto_weighted = {}  # class -> (weighted_sum, total_count)

        for client in active_clients:
            if hasattr(client, 'semantic_protos') and client.semantic_protos:
                counts = getattr(client, 'semantic_proto_counts', {})
                for c, proto in client.semantic_protos.items():
                    count = counts.get(c, 1)
                    proto_cpu = proto.detach().cpu()
                    if c not in proto_weighted:
                        proto_weighted[c] = (proto_cpu * count, count)
                    else:
                        prev_sum, prev_count = proto_weighted[c]
                        proto_weighted[c] = (prev_sum + proto_cpu * count, prev_count + count)

        self.global_semantic_protos = {}
        for c, (weighted_sum, total_count) in proto_weighted.items():
            self.global_semantic_protos[c] = weighted_sum / total_count
