"""EXP-119 Sanity C — FedBN + Class-level Prototype Trajectory Prediction
==========================================================================

最小实现, 用于验证一阶 Taylor 预测 `p̂ = p + η·v` 是否比直接用 `p` 对齐显著更好.

**核心机制**:
  Server 维护 per-class 全局 prototype 的当前位置和上轮位置:
      p_c^t   : current round aggregated class-c prototype
      p_c^{t-1}: previous round
      v_c^t = p_c^t - p_c^{t-1}
      p̂_c^{t+1} = p_c^t + η · v_c^t     (η 固定来自 config)

  Client alignment loss:
      L_align = mean_{x, y=c} (1 - cos(h(x), stop_grad(p̂_c^{t+1})))

  Config η=0 退化为直接对齐当前 prototype (naive baseline)

**算法级差异 vs FedProto**:
  FedProto 用 MSE(h, p_c) 对齐**当前**全局原型, 每 round 独立;
  我们用 cos + **预测下一轮** 位置做对齐 (时间动力学).

**跟 FedBN 的关系**:
  继承 FedBN 的 BN 本地化, 只在 penultimate 1024d feature 上加 alignment loss.

**Config 参数** (algo_para):
  - eta: float in [0, 1] — 预测步长 (0 = 无预测 baseline)
  - lambda_align: float — alignment loss 权重, default 0.5
  - proto_ema: float in [0, 1) — EMA on prototype for smoothness, 0 = 不 smooth, 0.9 = 强平滑
  - warmup_rounds: int — 前 N round 不做 alignment (等模型有点形再对齐), default 5

作者: Claude Code, 2026-04-22
"""
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import flgo.algorithm.fedbase as fab


DEFAULT_ALGO_PARA = [0.5, 0.5, 0.0, 5]  # eta, lambda_align, proto_ema, warmup_rounds


def _unwrap_alexnet(model):
    """Unwrap FModule wrappers (see fedbn_ccbank._unwrap_alexnet for rationale)."""
    inner = model
    for _ in range(3):
        if hasattr(inner, 'features'):
            return inner
        if hasattr(inner, 'model'):
            inner = inner.model
            continue
        if hasattr(inner, 'net'):
            inner = inner.net
            continue
        break
    return inner


def forward_with_feature(model, x):
    """Same helper as fedbn_ccbank, return (logits, penultimate 1024d feature)."""
    net = _unwrap_alexnet(model)
    h = net.features(x)
    h = net.avgpool(h)
    h = torch.flatten(h, 1)
    h = net.bn6(net.fc1(h))
    h = net.relu(h)
    h = net.bn7(net.fc2(h))
    h = net.relu(h)
    logits = net.fc3(h)
    return logits, h


# ------------------------------------------------------------------
# Server
# ------------------------------------------------------------------
class Server(fab.BasicServer):
    def initialize(self, *args, **kwargs):
        raw = getattr(self, 'algo_para', None) or []
        pad = raw + DEFAULT_ALGO_PARA[len(raw):]
        self.traj_eta = float(pad[0])
        self.traj_lambda = float(pad[1])
        self.traj_proto_ema = float(pad[2])
        self.traj_warmup = int(pad[3])

        # Prototype state
        # p_current[c] = current aggregated prototype (tensor, cpu, 1024d)
        # p_prev[c]    = previous round prototype
        # None = never seen yet
        self.p_current = {}
        self.p_prev = {}

        # broadcast to clients
        for c in self.clients:
            c.traj_lambda = self.traj_lambda
            c.traj_warmup = self.traj_warmup

    # --------------------------------------------------------------
    def pack(self, client_id, mtype=0):
        """Broadcast model + predicted prototypes (p_hat_next) to client."""
        current_round = int(getattr(self, 'current_round', 0))
        predicted = self._predict_next()
        return {
            'model': copy.deepcopy(self.model),
            'pred_protos': predicted,       # {class_id: tensor[1024]} or {} for warmup rounds
            'current_round': current_round,
        }

    # --------------------------------------------------------------
    def _predict_next(self):
        """Compute p_hat_{t+1} = p_t + eta * (p_t - p_{t-1}) per class.

        If p_prev not available for a class, fallback to p_current (no prediction).
        Returns dict[class_id] -> tensor[1024] (cpu). Empty during warmup.
        """
        current_round = int(getattr(self, 'current_round', 0))
        if current_round < self.traj_warmup:
            return {}
        preds = {}
        for c, p_c in self.p_current.items():
            if c in self.p_prev:
                v_c = p_c - self.p_prev[c]
                pred = p_c + self.traj_eta * v_c
            else:
                pred = p_c.clone()   # no previous round, direct use
            preds[c] = pred.detach().cpu()
        return preds

    # --------------------------------------------------------------
    def iterate(self):
        self.selected_clients = self.sample()
        received = self.communicate(self.selected_clients)
        if received is None or 'model' not in received:
            return False

        # 1. Aggregate models with FedBN semantic
        new_model = self._aggregate_fedbn(received['model'])
        self.model = new_model

        # 2. Aggregate per-class prototypes (weighted by class sample count)
        client_protos = received.get('class_protos', [])
        client_counts = received.get('class_counts', [])
        self._update_prototypes(client_protos, client_counts)
        return True

    # --------------------------------------------------------------
    def _update_prototypes(self, client_protos, client_counts):
        """Weighted average of per-class prototypes from clients; EMA with previous value if configured.

        client_protos: list of dict[class_id -> tensor]
        client_counts: list of dict[class_id -> int]
        """
        # Shift current -> prev BEFORE updating (so p_prev stays one round behind)
        self.p_prev = {c: p.clone() for c, p in self.p_current.items()}

        # Aggregate fresh this round
        agg_sum = defaultdict(lambda: None)
        agg_count = defaultdict(int)
        for protos, counts in zip(client_protos, client_counts):
            if protos is None:
                continue
            for c, p in protos.items():
                n = int(counts.get(c, 0))
                if n <= 0:
                    continue
                p_w = p.detach().cpu().float() * n
                if agg_sum[c] is None:
                    agg_sum[c] = p_w.clone()
                else:
                    agg_sum[c] += p_w
                agg_count[c] += n

        # Write to p_current, optionally EMA-blend with previous
        for c, s in agg_sum.items():
            if agg_count[c] <= 0:
                continue
            p_new = s / agg_count[c]
            if self.traj_proto_ema > 0 and c in self.p_current:
                self.p_current[c] = (
                    self.traj_proto_ema * self.p_current[c]
                    + (1 - self.traj_proto_ema) * p_new
                )
            else:
                self.p_current[c] = p_new

    # --------------------------------------------------------------
    def _aggregate_fedbn(self, client_models):
        """Same FedBN aggregation as fedbn_ccbank."""
        if not client_models:
            return self.model
        template = copy.deepcopy(client_models[0])
        tmpl_sd = template.state_dict()
        bn_keys = {k for k in tmpl_sd
                   if 'bn' in k.lower() or 'batch_norm' in k.lower() or 'batchnorm' in k.lower()}
        avg_sd = copy.deepcopy(self.model.state_dict())
        for k in tmpl_sd:
            if k in bn_keys:
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
        self.traj_lambda = getattr(self, 'traj_lambda', DEFAULT_ALGO_PARA[1])
        self.traj_warmup = getattr(self, 'traj_warmup', int(DEFAULT_ALGO_PARA[3]))
        self.model = None
        self.pred_protos = {}

    def unpack(self, received_pkg):
        """FedBN-style: preserve local BN layers."""
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
        self.pred_protos = received_pkg.get('pred_protos', {}) or {}
        self.current_round = int(received_pkg.get('current_round', 0))
        return self.model

    def pack(self, model):
        class_protos, class_counts = self._compute_class_protos(model)
        return {
            'model': model,
            'class_protos': class_protos,
            'class_counts': class_counts,
        }

    # --------------------------------------------------------------
    def train(self, model):
        model.train()
        device = self.device
        model.to(device)

        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum,
        )
        loss_fn = nn.CrossEntropyLoss()
        loader = self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size)

        # Move predicted prototypes to device once per round (read-only)
        pred_protos_gpu = {c: v.to(device) for c, v in self.pred_protos.items()} if self.pred_protos else {}
        use_alignment = (self.current_round >= self.traj_warmup) and (len(pred_protos_gpu) > 0)

        for epoch in range(self.num_epochs):
            for batch in loader:
                batch = self.calculator.to_device(batch)
                x, y = batch[0], batch[-1]
                optimizer.zero_grad()
                logits, h = forward_with_feature(model, x)
                loss_ce = loss_fn(logits, y)

                if use_alignment:
                    loss_align = self._compute_alignment_loss(h, y, pred_protos_gpu)
                    loss = loss_ce + self.traj_lambda * loss_align
                else:
                    loss = loss_ce

                loss.backward()
                if getattr(self, 'clip_grad', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                optimizer.step()
        return model

    # --------------------------------------------------------------
    def _compute_alignment_loss(self, h, y, pred_protos_gpu):
        """L_align = mean_i (1 - cos(h_i, p_hat[y_i])).

        Only counts samples whose class is present in pred_protos_gpu (warmup-aware).
        """
        B = h.shape[0]
        # Build target tensor: [B, D], fill pred_protos[y_i] or fallback h_i (no gradient contribution)
        targets = torch.zeros_like(h)
        mask = torch.zeros(B, dtype=torch.bool, device=h.device)
        for i in range(B):
            yi = int(y[i].item())
            if yi in pred_protos_gpu:
                targets[i] = pred_protos_gpu[yi]
                mask[i] = True
        if not mask.any():
            return torch.tensor(0.0, device=h.device)
        h_m = h[mask]
        t_m = targets[mask].detach()    # stop grad on target
        cos = F.cosine_similarity(h_m, t_m, dim=-1)
        loss = (1.0 - cos).mean()
        return loss

    # --------------------------------------------------------------
    def _compute_class_protos(self, model):
        """Compute per-class mean feature (p_{c,k}) on local training data."""
        device = self.device
        model.eval()
        loader = self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size)

        feat_sum = defaultdict(lambda: None)
        counts = defaultdict(int)
        with torch.no_grad():
            for batch in loader:
                batch = self.calculator.to_device(batch)
                x, y = batch[0], batch[-1]
                _, h = forward_with_feature(model, x)
                for c in torch.unique(y):
                    ci = int(c.item())
                    mask = (y == c)
                    n_c = int(mask.sum().item())
                    if n_c == 0:
                        continue
                    s = h[mask].sum(dim=0)
                    if feat_sum[ci] is None:
                        feat_sum[ci] = s.clone()
                    else:
                        feat_sum[ci] += s
                    counts[ci] += n_c

        class_protos = {}
        class_counts = {}
        for ci, n in counts.items():
            if n <= 0:
                continue
            class_protos[ci] = (feat_sum[ci] / n).detach().cpu()
            class_counts[ci] = n
        model.train()
        return class_protos, class_counts


# NOTE: intentionally NOT defining init_global_module / init_local_module /
# init_dataset. flgo falls back to the benchmark's default model hook
# (benchmark/{office_caltech10,pacs,domainnet}_classification/model/default_model.py)
# Defining an empty pass hook would silently prevent model construction.
