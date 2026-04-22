"""EXP-119 Sanity A — Centralized (non-federated) baseline
=============================================================

Merge all clients' training data into a single dataset and train one AlexNet
on a single GPU (no FedAvg, no BN locality). This establishes the capacity
ceiling of the backbone+data combination — FL accuracy <= centralized accuracy.

Design choice: we reuse flgo's Server loop for evaluation infrastructure (so
PerRunLogger still works and per-client test metrics are reported), but during
training we bypass federated rounds entirely:
    server.run() -> merge train data -> train one model for R*E epochs -> eval -> save

**Usage**:
    python run_single.py --task office_caltech10_c4 --algorithm centralized \\
        --config ./config/office/centralized_office.yml --seed 2

This is logically equivalent to treating the union of all clients as a single
centralised learner, while keeping evaluation per-client (so Best/Last metrics
are still comparable to FL runs).
"""
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import trange

import flgo.algorithm.fedbase as fab


class Server(fab.BasicServer):
    def run(self):
        """Override the standard federated loop — do centralized training instead."""
        # Ensure every client has a model stub (PerRunLogger expects c.model)
        for c in self.clients:
            if getattr(c, 'model', None) is None:
                c.model = copy.deepcopy(self.model)

        self.gv.logger.time_start('Total Time Cost')
        self.gv.logger.info('--------------Initial Evaluation--------------')
        self.gv.logger.time_start('Eval Time Cost')
        self.gv.logger.log_once()
        self.gv.logger.time_end('Eval Time Cost')

        # ---- Merge all clients' train data ----
        merged = ConcatDataset([c.train_data for c in self.clients])
        loader = DataLoader(
            merged,
            batch_size=int(self.option.get('batch_size', 50)),
            shuffle=True,
            num_workers=0,
        )

        device = self.device
        model = self.model.to(device)
        model.train()
        lr = float(self.option.get('learning_rate', 0.05))
        wd = float(self.option.get('weight_decay', 1e-3))
        momentum = float(getattr(self, 'momentum', 0.9))
        clip = float(self.option.get('clip_grad', 0))
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=wd
        )
        lr_decay = float(self.option.get('learning_rate_decay', 1.0))
        loss_fn = nn.CrossEntropyLoss()
        num_rounds = int(self.option.get('num_rounds', 200))
        num_epochs = int(self.option.get('num_epochs', 1))
        eval_interval = max(1, num_rounds // 100)  # evaluate ~100 times across training

        total_epochs = num_rounds * num_epochs
        self.gv.logger.info(
            f'[centralized] training for {total_epochs} epochs on {len(merged)} samples'
        )

        global_step = 0
        for round_idx in trange(num_rounds, desc='centralized'):
            for _ in range(num_epochs):
                for batch in loader:
                    # flgo uses calculator.to_device for the batch tuple
                    batch = self.clients[0].calculator.to_device(batch)
                    x, y = batch[0], batch[-1]
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    loss.backward()
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                    global_step += 1
            # LR decay per round (align with federated runs)
            for g in optimizer.param_groups:
                g['lr'] = lr * (lr_decay ** (round_idx + 1))

            # Broadcast the centralized model to every client's view (for eval)
            for c in self.clients:
                c.model = model

            # Evaluate at interval
            if (round_idx + 1) % eval_interval == 0 or (round_idx + 1) == num_rounds:
                self.gv.logger.time_start('Eval Time Cost')
                self.gv.logger.log_once()
                self.gv.logger.time_end('Eval Time Cost')

        self.gv.logger.info('=================End==================')
        self.gv.logger.time_end('Total Time Cost')
        self.gv.logger.save_output_as_json()
        return


class Client(fab.BasicClient):
    """Stub client — only provides test_data for per-client evaluation.

    Training is entirely handled by Server; this class is a passive participant.
    """
    def initialize(self, *args, **kwargs):
        # placeholder; Server.run will overwrite self.model after each round
        pass
