"""
FedDSA-StyleHeadBank: Connect style_head output to style bank & augmentation.

Design problem found in audit:
    - style_head maps h [1024] → z_sty [128], only used in orth loss
    - style bank stores (mu, sigma) of h [1024], used for AdaIN
    - The two "style" concepts are disconnected

Fix:
    - Style bank stores (mu, sigma) of z_sty [128] instead of h [1024]
    - AdaIN augmentation operates on z_sem [128] using z_sty statistics
    - This makes style_head the actual style extractor
    - loss_aug now classifies AdaIN(z_sem, z_sty_stats) directly

Combined with detach fix (EXP-058):
    z_sty = model.get_style(h.detach())
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.feddsa import Server as BaseServer, Client as BaseClient, FedDSAModel
import flgo.utils.fmodule as fuf


class Server(BaseServer):
    """Same as base, but style bank now stores z_sty stats [proj_dim] not h stats [1024]."""
    pass


class Client(BaseClient):
    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum
        )

        aux_w = min(1.0, self.current_round / max(self.warmup_rounds, 1))

        proto_sum = {}
        proto_count = {}
        style_sum = None
        style_sq_sum = None
        style_n = 0

        num_steps = self.num_steps
        for step in range(num_steps):
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            x, y = batch_data[0], batch_data[-1]

            model.zero_grad()

            h = model.encode(x)
            z_sem = model.get_semantic(h)
            z_sty = model.get_style(h.detach())  # detach fix from EXP-058

            # Loss 1: Task CE
            output = model.head(z_sem)
            loss_task = self.loss_fn(output, y)

            # Loss 2: Augmented CE — now in z_sem space using z_sty stats
            loss_aug = torch.tensor(0.0, device=x.device)
            if self.local_style_bank is not None and self.current_round >= self.warmup_rounds:
                z_sem_aug = self._style_augment_semantic(z_sem)
                output_aug = model.head(z_sem_aug)
                loss_aug = self.loss_fn(output_aug, y)

            # Loss 3: Decoupling
            loss_orth, loss_hsic = self._decouple_loss(z_sem, z_sty)

            # Loss 4: Semantic contrastive alignment
            loss_sem = torch.tensor(0.0, device=x.device)
            if self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_loss(z_sem, y)

            loss = loss_task + loss_aug + \
                   aux_w * self.lambda_orth * loss_orth + \
                   aux_w * self.lambda_hsic * loss_hsic + \
                   aux_w * self.lambda_sem * loss_sem

            loss.backward()
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
            optimizer.step()

            # Accumulate protos + style stats (last epoch)
            if step >= num_steps - len(self.train_data) // self.batch_size - 1:
                with torch.no_grad():
                    z_det = z_sem.detach().cpu()
                    z_sty_det = z_sty.detach()  # now collect z_sty stats, not h stats
                    for i, label in enumerate(y):
                        c = label.item()
                        if c not in proto_sum:
                            proto_sum[c] = z_det[i].clone()
                            proto_count[c] = 1
                        else:
                            proto_sum[c] += z_det[i]
                            proto_count[c] += 1

                    # Style stats from z_sty [proj_dim] instead of h [1024]
                    b = z_sty_det.size(0)
                    batch_mu = z_sty_det.mean(dim=0).cpu()
                    batch_sq = (z_sty_det ** 2).mean(dim=0).cpu()
                    if style_sum is None:
                        style_sum = batch_mu * b
                        style_sq_sum = batch_sq * b
                        style_n = b
                    else:
                        style_sum += batch_mu * b
                        style_sq_sum += batch_sq * b
                        style_n += b

        self._local_protos = {c: proto_sum[c] / proto_count[c] for c in proto_sum}
        self._local_proto_counts = proto_count

        if style_n > 1:
            mu = style_sum / style_n
            var = style_sq_sum / style_n - mu ** 2
            self._local_style_stats = (mu, var.clamp(min=1e-6).sqrt())
        else:
            self._local_style_stats = None

    def _style_augment_semantic(self, z_sem):
        """AdaIN on z_sem [proj_dim] using z_sty style bank stats [proj_dim]."""
        idx = np.random.randint(0, len(self.local_style_bank))
        mu_ext, sigma_ext = self.local_style_bank[idx]
        mu_ext = mu_ext.to(z_sem.device)
        sigma_ext = sigma_ext.to(z_sem.device)

        mu_local = z_sem.mean(dim=0)
        sigma_local = z_sem.std(dim=0, unbiased=False).clamp(min=1e-6)

        alpha = np.random.beta(0.1, 0.1)
        mu_mix = alpha * mu_local + (1 - alpha) * mu_ext
        sigma_mix = alpha * sigma_local + (1 - alpha) * sigma_ext

        z_norm = (z_sem - mu_local) / sigma_local
        return z_norm * sigma_mix + mu_mix


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    from algorithm.feddsa import init_global_module as base_init
    base_init(object)
