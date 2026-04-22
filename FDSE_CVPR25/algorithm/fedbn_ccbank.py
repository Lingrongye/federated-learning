"""EXP-119 Sanity B — FedBN + Class-Conditional BN Stats Bank (CC-Bank)
================================================================

最小实现, 用于验证 class-conditional feature-level AdaIN 是否 work.

**核心机制**:
  Server 维护 style_bank[class_id][client_id] = (mu, sigma) ∈ R^{1024} × R^{1024}
  (penultimate feature after encoder+fc2+bn7+relu, 不是 BN 层内部的)

  Client training (with p=0.5 probability, per-sample):
      从 style_bank[y][other_client] 随机采样 (mu', sigma')
      h_aug = sigma' * (h - mu_self_batch) / sigma_self_batch + mu'
      h_final = alpha * h_aug + (1 - alpha) * h   (alpha 固定, 来自 config)
      L = CE(classifier(h_final), y)

  Client upload: per-class batch-level (mu, sigma) if n_c >= min_samples (default 8)

**算法级差异 vs FedFA/FedCA**:
  FedFA: Gaussian reparameterization on domain-level BN stats (single pool per client)
  FedCA: domain-level style bank (shallow conv stats)
  我们: **class-conditional** bank, hard-sampled from other clients, fixed alpha

**跟 FedBN 的关系**:
  继承 FedBN 的 BN 本地化策略 (所有 bn.* key 本地保留, 不参与 FedAvg)
  只在 penultimate feature 上做 AdaIN, 不碰 BN 层

**Config 参数** (algo_para):
  - alpha: float in [0, 1] — AdaIN mix 强度 (固定, 不 learnable)
  - aug_prob: float in [0, 1] — per-sample 做 AdaIN 的概率, default 0.5
  - min_samples: int — 收 bank 的每类最小样本数, default 8
  - bank_smoothing: float in [0, 1] — EMA smoothing, 0 = 直接覆盖, 1 = 完全不更新, default 0 (先跑无 smoothing)

作者: Claude Code, 2026-04-22
"""
import copy
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import flgo.algorithm.fedbase as fab


# ------------------------------------------------------------------
# Default algo_para: [alpha, aug_prob, min_samples, bank_smoothing]
# ------------------------------------------------------------------
DEFAULT_ALGO_PARA = [0.5, 0.5, 8, 0.0]


# ------------------------------------------------------------------
# Feature extraction helper (penultimate 1024d)
# ------------------------------------------------------------------
def forward_with_feature(model, x):
    """Forward AlexNet and return (logits, penultimate 1024d feature).

    Penultimate = output of bn7 + relu (before fc3 classifier).
    Works with the AlexNet defined in benchmark/office_caltech10_classification/config.py.
    """
    h = model.features(x)
    h = model.avgpool(h)
    h = torch.flatten(h, 1)
    h = model.bn6(model.fc1(h))
    h = model.relu(h)
    h = model.bn7(model.fc2(h))
    h = model.relu(h)           # 1024d penultimate, 'h' in our math
    logits = model.fc3(h)
    return logits, h


# ------------------------------------------------------------------
# CC-Bank: class-conditional style bank (per-class, per-client)
# ------------------------------------------------------------------
class CCBank:
    """Dict-of-dict bank: bank[class_id][client_id] = (mu, sigma).

    mu, sigma are cpu tensors of shape [feat_dim] (e.g. 1024).
    Caller must ensure contribution comes from batches with >= min_samples of the given class.
    """
    def __init__(self, smoothing: float = 0.0):
        """smoothing in [0, 1): exponential moving avg, 0 = overwrite (no smoothing)."""
        self.smoothing = float(smoothing)
        # bank[c][cid] = (mu, sigma)
        self.bank = defaultdict(dict)

    def update(self, class_id: int, client_id: int, mu: torch.Tensor, sigma: torch.Tensor):
        """Update a single (class, client) entry."""
        mu_c = mu.detach().cpu().float()
        sig_c = sigma.detach().cpu().float()
        key = int(class_id)
        cid = int(client_id)
        if cid in self.bank[key] and self.smoothing > 0:
            prev_mu, prev_sig = self.bank[key][cid]
            new_mu = self.smoothing * prev_mu + (1 - self.smoothing) * mu_c
            new_sig = self.smoothing * prev_sig + (1 - self.smoothing) * sig_c
            self.bank[key][cid] = (new_mu, new_sig)
        else:
            self.bank[key][cid] = (mu_c, sig_c)

    def has(self, class_id: int, exclude_client: int = None) -> bool:
        """Check if bank has any (class_id, client!=exclude_client) entry."""
        key = int(class_id)
        if key not in self.bank:
            return False
        others = [cid for cid in self.bank[key] if cid != exclude_client]
        return len(others) > 0

    def sample(self, class_id: int, exclude_client: int, device=None):
        """Sample one (mu, sigma) from bank[class_id][k] where k != exclude_client.

        Returns (mu, sigma) on device, or None if no valid entry.
        """
        key = int(class_id)
        if key not in self.bank:
            return None
        candidates = [cid for cid in self.bank[key] if cid != exclude_client]
        if not candidates:
            return None
        pick = random.choice(candidates)
        mu, sig = self.bank[key][pick]
        if device is not None:
            mu = mu.to(device)
            sig = sig.to(device)
        return mu, sig

    def size(self) -> int:
        """Total (class, client) entries."""
        return sum(len(v) for v in self.bank.values())


# ------------------------------------------------------------------
# Server
# ------------------------------------------------------------------
class Server(fab.BasicServer):
    def initialize(self, *args, **kwargs):
        # Read algo_para, use defaults if missing
        raw = getattr(self, 'algo_para', None) or []
        pad = raw + DEFAULT_ALGO_PARA[len(raw):]
        self.cc_alpha = float(pad[0])
        self.cc_aug_prob = float(pad[1])
        self.cc_min_samples = int(pad[2])
        self.cc_smoothing = float(pad[3])

        # Bank lives on server (CPU memory, broadcast to clients each round)
        self.style_bank = CCBank(smoothing=self.cc_smoothing)

        # Push hyperparams to every client
        for c in self.clients:
            c.cc_alpha = self.cc_alpha
            c.cc_aug_prob = self.cc_aug_prob
            c.cc_min_samples = self.cc_min_samples

    # --------------------------------------------------------------
    def pack(self, client_id, mtype=0):
        """Send model + bank snapshot to the selected client."""
        return {
            'model': copy.deepcopy(self.model),
            'style_bank': self.style_bank,        # pass by reference (read-only on client side)
            'current_round': getattr(self, 'current_round', 0),
        }

    # --------------------------------------------------------------
    def iterate(self):
        """One federated round: sample -> broadcast -> train -> aggregate + bank update."""
        self.selected_clients = self.sample()
        received = self.communicate(self.selected_clients)
        if received is None or 'model' not in received:
            return False

        # 1. Aggregate models (FedBN semantic: skip BN running keys)
        new_model = self._aggregate_fedbn(received['model'])
        self.model = new_model

        # 2. Update bank with class-conditional stats uploaded by clients
        class_stats_list = received.get('class_stats', [])
        for cid, stats in zip(self.selected_clients, class_stats_list):
            if not stats:
                continue
            for class_id, (mu, sig) in stats.items():
                self.style_bank.update(class_id, cid, mu, sig)
        return True

    # --------------------------------------------------------------
    def _aggregate_fedbn(self, client_models):
        """FedAvg on everything except BN running stats (follow fedbn.py conventions)."""
        if not client_models:
            return self.model
        # equal-weight aggregation over selected clients
        n = len(client_models)
        template = copy.deepcopy(client_models[0])
        tmpl_sd = template.state_dict()
        # Figure out which keys to skip (BN running/num_batches_tracked)
        bn_keys = set()
        for k in tmpl_sd:
            if 'bn' in k.lower() or 'batch_norm' in k.lower() or 'batchnorm' in k.lower():
                bn_keys.add(k)
        avg_sd = copy.deepcopy(self.model.state_dict())
        for k in tmpl_sd:
            if k in bn_keys:
                # keep the global model's own BN (unchanged, since BN is local anyway)
                continue
            stacked = torch.stack([m.state_dict()[k].float() for m in client_models], dim=0)
            avg_sd[k] = stacked.mean(dim=0).to(tmpl_sd[k].dtype)
        template.load_state_dict(avg_sd)
        return template


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------
class Client(fab.BasicClient):
    def initialize(self, *args, **kwargs):
        # Placeholders — server.initialize() will overwrite after construction
        self.cc_alpha = getattr(self, 'cc_alpha', DEFAULT_ALGO_PARA[0])
        self.cc_aug_prob = getattr(self, 'cc_aug_prob', DEFAULT_ALGO_PARA[1])
        self.cc_min_samples = getattr(self, 'cc_min_samples', int(DEFAULT_ALGO_PARA[2]))
        self.style_bank = None
        self.model = None

    # --------------------------------------------------------------
    def unpack(self, received_pkg):
        """Preserve BN layers when loading server model (FedBN style)."""
        global_model = received_pkg['model']
        if self.model is None:
            self.model = copy.deepcopy(global_model)
        else:
            new_dict = self.model.state_dict()
            global_dict = global_model.state_dict()
            for key in new_dict.keys():
                if 'bn' in key.lower() or 'batch_norm' in key.lower() or 'batchnorm' in key.lower():
                    continue
                new_dict[key] = global_dict[key]
            self.model.load_state_dict(new_dict)
        # Store bank for use in train()
        self.style_bank = received_pkg.get('style_bank', None)
        return self.model

    # --------------------------------------------------------------
    def pack(self, model):
        """Compute per-class stats on local training data, upload with model."""
        class_stats = self._compute_class_stats(model)
        return {'model': model, 'class_stats': class_stats}

    # --------------------------------------------------------------
    def train(self, model):
        """One client round: E epochs with CC-Bank augmentation mixed into CE loss."""
        model.train()
        device = self.device
        model.to(device)

        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum,
        )
        loss_fn = nn.CrossEntropyLoss()

        loader = self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size)

        for epoch in range(self.num_epochs):
            for batch in loader:
                batch = self.calculator.to_device(batch)
                x, y = batch[0], batch[-1]
                optimizer.zero_grad()
                logits, h = forward_with_feature(model, x)  # h: [B, 1024]

                # ---- CC-Bank augmentation ----
                loss_ce = loss_fn(logits, y)
                loss_aug = self._compute_aug_loss(model, h, y, loss_fn)

                # If aug is valid, form h_final and use augmented CE; else fallback.
                # Important: loss_aug is either a scalar tensor (valid) or None (no valid bank sample).
                if loss_aug is not None:
                    loss = loss_ce + loss_aug
                else:
                    loss = loss_ce

                loss.backward()
                if getattr(self, 'clip_grad', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                optimizer.step()
        return model

    # --------------------------------------------------------------
    def _compute_aug_loss(self, model, h, y, loss_fn):
        """For each sample in batch with p=aug_prob, draw a cross-client same-class (mu, sigma) and compute CE on h_final.

        Returns scalar tensor OR None (when no valid aug happened for the whole batch).
        """
        if self.style_bank is None or self.style_bank.size() == 0:
            return None
        if self.cc_alpha <= 1e-6:  # alpha 0 means no augmentation
            return None

        B, D = h.shape
        device = h.device

        # Compute per-sample self (mu, sigma) over feature dim — for AdaIN normalization
        # Note: for pooled 1024d feature, "per-sample (mu, sigma)" over feature-axis
        # follows the MixStyle convention (channel-wise for conv, feature-wise for pooled).
        mu_self = h.mean(dim=1, keepdim=True)           # [B, 1]
        sigma_self = h.std(dim=1, keepdim=True) + 1e-5  # [B, 1]

        # For each sample, decide if augment and which bank entry to pick
        mask = torch.rand(B, device=device) < self.cc_aug_prob  # [B] bool

        # Build h_aug (default equals h; replace for masked samples with valid bank entry)
        h_aug = h.clone()
        valid_mask = torch.zeros(B, dtype=torch.bool, device=device)
        for i in range(B):
            if not mask[i]:
                continue
            yi = int(y[i].item())
            # Sample from bank[yi][k != self.id]
            sampled = self.style_bank.sample(yi, exclude_client=self.id, device=device)
            if sampled is None:
                continue
            mu_other, sig_other = sampled             # each [D]
            # AdaIN on feature axis: (h - μ_self)/σ_self × σ_other + μ_other
            # μ_other / σ_other here are per-feature-channel (not scalar), so broadcasting is across channels.
            # Reuse per-sample scalar (mu_self, sigma_self) for self-normalization.
            h_aug[i] = ((h[i] - mu_self[i]) / sigma_self[i]) * sig_other + mu_other
            valid_mask[i] = True

        if not valid_mask.any():
            return None

        # Mix: h_final = α · h_aug + (1-α) · h
        h_final = self.cc_alpha * h_aug + (1 - self.cc_alpha) * h
        logits_aug = model.fc3(h_final)
        # Only compute loss on the samples that actually got augmented
        loss_aug = loss_fn(logits_aug[valid_mask], y[valid_mask])
        return loss_aug

    # --------------------------------------------------------------
    def _compute_class_stats(self, model):
        """One extra forward pass over training data to compute per-class (mu, sigma) of penultimate feature.

        Only classes with >= min_samples are uploaded.
        Uses model.eval() + no_grad for stable estimates (no BN stat update here).
        """
        device = self.device
        model.eval()
        loader = self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size)

        feat_sum = defaultdict(lambda: None)    # class_id -> running sum vector
        feat_sq_sum = defaultdict(lambda: None) # for variance = E[x^2] - E[x]^2
        counts = defaultdict(int)

        with torch.no_grad():
            for batch in loader:
                batch = self.calculator.to_device(batch)
                x, y = batch[0], batch[-1]
                _, h = forward_with_feature(model, x)  # [B, D]
                for c in torch.unique(y):
                    ci = int(c.item())
                    mask = (y == c)
                    n_c = int(mask.sum().item())
                    if n_c == 0:
                        continue
                    h_c = h[mask]
                    s = h_c.sum(dim=0)
                    sq = (h_c ** 2).sum(dim=0)
                    if feat_sum[ci] is None:
                        feat_sum[ci] = s.clone()
                        feat_sq_sum[ci] = sq.clone()
                    else:
                        feat_sum[ci] += s
                        feat_sq_sum[ci] += sq
                    counts[ci] += n_c

        stats = {}
        for ci, n in counts.items():
            if n < self.cc_min_samples:
                continue
            mu = feat_sum[ci] / n
            var = feat_sq_sum[ci] / n - mu ** 2
            var = torch.clamp(var, min=1e-8)
            sigma = torch.sqrt(var)
            stats[ci] = (mu.detach().cpu(), sigma.detach().cpu())
        model.train()
        return stats


# ------------------------------------------------------------------
# Init hook (flgo plumbing) — use default benchmark model
# ------------------------------------------------------------------
def init_local_module(object):
    pass


def init_dataset(object):
    pass


def init_global_module(object):
    # Use the benchmark's default AlexNet (with full BN); no custom model.
    pass
