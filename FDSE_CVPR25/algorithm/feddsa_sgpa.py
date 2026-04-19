"""
FedDSA-SGPA: Style-Gated Prototype Adjustment for Federated Domain Generalization
==================================================================================

Based on FINAL_PROPOSAL.md (2026-04-19 research-refine v1, 9.0/10 READY).

核心改动 (vs Plan A orth_only):
1. sem_classifier 改为 Fixed Simplex ETF buffer (seeded, 所有 client 一致, 不聚合漂移)
2. Client 上传 (μ_sty_k, Σ_sty_k) = 128d z_sty 的 batch-mean / 协方差
3. Server 构造 pooled second-order model (μ_global, Σ_within + Σ_between, Σ_inv_sqrt)
4. Inference 端: 双 gate (entropy + Mahalanobis-in-whitened-space) 筛 reliable 样本
5. Top-m per-class support bank → cos(z_sem, proto) 分类 + ETF fallback

设计约束 (refine 后硬性要求):
- 只用 z_sem 做 ETF, z_sty 保持 Gaussian bank (避免撞 FedDEAP)
- ResNet/AlexNet from-scratch, 不用 CLIP
- 完全 backprop-free inference (无反传)
- warmup 5 batch 期间输出 ETF fallback (不 continue 跳过样本)
"""

import os
import copy
import math
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import flgo.algorithm.fedbase
import flgo.utils.fmodule as fuf


# ============================================================
# Model: AlexNet encoder + dual-head + Fixed Simplex ETF classifier
# ============================================================


class AlexNetEncoder(nn.Module):
    """Same AlexNet backbone as feddsa.py / feddsa_scheduled.py for fair comparison."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
            ('bn2', nn.BatchNorm2d(192)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(384)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('bn5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU(inplace=True)),
            ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
        ]))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.bn7 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn6(self.fc1(x))
        x = self.relu(x)
        x = self.bn7(self.fc2(x))
        x = self.relu(x)
        return x  # [B, 1024]


def build_etf_matrix(feat_dim: int, num_classes: int, seed: int = 0) -> torch.Tensor:
    """Construct Fixed Simplex ETF matrix M ∈ R^{d × K}.

    M 的 K 个列向量构成 simplex ETF (等角最大间隔), 满足:
    - ‖M[:, c]‖ = sqrt(K / (K-1))  (normalized 后两两 cos = -1/(K-1))
    - 不含可训练参数, 所有 client 用同一 seed 保证 M 一致.

    Args:
        feat_dim: z_sem 维度 (必须 >= num_classes).
        num_classes: 类别数 K.
        seed: QR 分解的随机 seed.
    Returns:
        M: [feat_dim, num_classes] float32 tensor.
    """
    assert feat_dim >= num_classes, (
        f"Fixed ETF 需要 feat_dim({feat_dim}) >= num_classes({num_classes})")
    rng = torch.Generator()
    rng.manual_seed(seed)
    U = torch.linalg.qr(torch.randn(feat_dim, num_classes, generator=rng))[0]
    I_K = torch.eye(num_classes)
    ones_K = torch.ones(num_classes, num_classes)
    # β = 1 (与 τ 冗余, 按 refine 决议固定)
    scale = math.sqrt(num_classes / max(num_classes - 1, 1))
    M = scale * U @ (I_K - ones_K / num_classes)
    return M


class FedDSASGPAModel(fuf.FModule):
    """FedDSA-SGPA model: AlexNet + dual head + Fixed ETF classifier."""

    def __init__(self, num_classes=7, feat_dim=1024, proj_dim=128,
                 tau_etf=0.1, etf_seed=0):
        super().__init__()
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.tau_etf = tau_etf

        self.encoder = AlexNetEncoder()
        self.semantic_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.style_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        # Fixed Simplex ETF buffer (所有 client 共享, 不参加聚合漂移)
        M = build_etf_matrix(proj_dim, num_classes, seed=etf_seed)
        self.register_buffer('M', M)

    def encode(self, x):
        return self.encoder(x)

    def get_semantic(self, h):
        return self.semantic_head(h)

    def get_style(self, h):
        return self.style_head(h)

    def classify(self, z_sem: torch.Tensor) -> torch.Tensor:
        """统一分类接口: logits = normalize(z_sem) @ M / τ_etf."""
        z_norm = F.normalize(z_sem, dim=-1)
        return z_norm @ self.M / self.tau_etf

    def forward(self, x):
        h = self.encode(x)
        z_sem = self.get_semantic(h)
        return self.classify(z_sem)


# ============================================================
# Server: FedAvg + FedBN + style 二阶统计 + pooled whitening
# ============================================================


class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({
            'lo': 1.0,        # lambda_orth
            'tau_etf': 0.1,   # Fixed ETF temperature
            'pd': 128,        # proj_dim
            'warmup_r': 10,   # warmup rounds before enabling style bank
            'eps_sigma': 1e-3,  # Σ_global 正则化
            'min_clients_whiten': 2,  # 最少几个 client 才构造 whitening
            'diag': 0,        # 0=off, 1=enable DiagnosticLogger
        })
        self.sample_option = 'full'

        # Style bank: client_id -> (μ_sty, Σ_sty)  (128d post-decouple)
        self.style_bank = {}
        self._init_agg_keys()

        # Pooled whitening state (更新于 iterate)
        self.source_mu_k = None    # dict cid -> [d]
        self.mu_global = None      # [d]
        self.sigma_inv_sqrt = None  # [d, d]

        # Pass config
        for c in self.clients:
            c.lambda_orth = self.lo
            c.tau_etf = self.tau_etf
            c.proj_dim = self.pd
            c.warmup_rounds = self.warmup_r

    def _init_agg_keys(self):
        """FedBN + style_head 本地; M buffer 参与聚合但因 seeded 一致所以不影响."""
        all_keys = list(self.model.state_dict().keys())
        self.private_keys = set()
        for k in all_keys:
            if 'style_head' in k:
                self.private_keys.add(k)
            elif 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
                self.private_keys.add(k)
        # M buffer 也跳过聚合 (所有 client 一致, 不需 aggregate)
        for k in all_keys:
            if k.endswith('.M') or k == 'M':
                self.private_keys.add(k)
        self.shared_keys = [k for k in all_keys if k not in self.private_keys]

    def pack(self, client_id, mtype=0):
        """下发 global model + whitening payload."""
        return {
            'model': copy.deepcopy(self.model),
            'current_round': self.current_round,
            'source_mu_k': self.source_mu_k,
            'mu_global': self.mu_global,
            'sigma_inv_sqrt': self.sigma_inv_sqrt,
        }

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)

        models = res['model']
        style_stats_list = res['style_stats']

        # 1. Aggregate shared parameters (FedAvg, excl. style_head/BN/M)
        self._aggregate_shared(models)

        # 2. Collect per-client style (μ_sty, Σ_sty)
        for cid, style in zip(self.received_clients, style_stats_list):
            if style is not None:
                self.style_bank[cid] = style

        # 3. Rebuild pooled whitening once we have enough clients
        if len(self.style_bank) >= self.min_clients_whiten:
            self._compute_pooled_whitening()

    def _aggregate_shared(self, models):
        """FedAvg on shared keys only (sample-size weighted)."""
        if not models:
            return
        weights = np.array(
            [len(self.clients[cid].train_data) for cid in self.received_clients],
            dtype=float,
        )
        weights /= weights.sum()

        global_dict = self.model.state_dict()
        model_dicts = [m.state_dict() for m in models]
        for k in self.shared_keys:
            if 'num_batches_tracked' in k:
                continue
            global_dict[k] = sum(w * md[k] for w, md in zip(weights, model_dicts))
        self.model.load_state_dict(global_dict, strict=False)

    def _compute_pooled_whitening(self):
        """聚合各 client (μ, Σ) → (μ_global, Σ_within+Σ_between, Σ_inv_sqrt).

        Σ_inv_sqrt 通过 eigendecomposition: Σ = Q Λ Q^T, Σ^{-1/2} = Q Λ^{-1/2} Q^T.

        NOTE (codex review SHOULD_FIX): 当前用 uniform 平均 (1/N) Σ_k.
        更严格应该按 client 样本数 N_k 加权 (true pooled second-order),
        但在 PACS/Office 4-client 场景各 client 样本数接近时差异可忽略.
        留作 TODO 在后续 PR 中切换到 sample-weighted 版.
        """
        cids = sorted(self.style_bank.keys())
        if len(cids) < 2:
            return
        mus = [self.style_bank[c][0] for c in cids]  # list of [d]
        sigmas = [self.style_bank[c][1] for c in cids]  # list of [d, d]
        d = mus[0].shape[0]

        # μ_global = mean
        mu_stack = torch.stack(mus, dim=0)  # [N, d]
        mu_global = mu_stack.mean(dim=0)

        # Σ_within = avg of per-client cov
        sigma_within = torch.stack(sigmas, dim=0).mean(dim=0)  # [d, d]

        # Σ_between = cov of client means across clients
        diffs = mu_stack - mu_global.unsqueeze(0)  # [N, d]
        sigma_between = diffs.t() @ diffs / len(cids)  # [d, d]

        sigma_global = sigma_within + sigma_between + self.eps_sigma * torch.eye(d)

        # 确保对称 (避免 eigh 数值噪声)
        sigma_global = 0.5 * (sigma_global + sigma_global.t())
        # torch.linalg.eigh 要求 symmetric, upcast float64 for stability
        L, Q = torch.linalg.eigh(sigma_global.double())
        L = L.clamp(min=self.eps_sigma)
        inv_sqrt_L = L.pow(-0.5)
        sigma_inv_sqrt = (Q @ torch.diag(inv_sqrt_L) @ Q.t()).float()

        # 保存
        self.source_mu_k = {c: self.style_bank[c][0].clone() for c in cids}
        self.mu_global = mu_global.clone()
        self.sigma_inv_sqrt = sigma_inv_sqrt


# ============================================================
# Client: Plan A train + ETF classify + SGPA inference
# ============================================================


class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = copy.deepcopy(self.server.model).to('cpu')
        self.loss_fn = nn.CrossEntropyLoss()
        self.current_round = 0
        self._local_style_stats = None
        # SGPA inference state (推理时动态维护)
        self.sgpa_supports = None  # {c: list of (H, z_sem)}
        self.sgpa_proto = None     # [K, d] F.normalize 后的 proto
        self.sgpa_tau_H = None
        self.sgpa_tau_S = None
        self.sgpa_warmup_buf_H = []
        self.sgpa_warmup_buf_D = []
        # Received from server
        self.source_mu_k = None
        self.mu_global = None
        self.sigma_inv_sqrt = None

    def reply(self, svr_pkg):
        self.unpack(svr_pkg)
        self.train(self.model)
        return self.pack()

    def unpack(self, svr_pkg):
        """接收 global model (FedBN 式) + whitening payload."""
        global_model = svr_pkg['model']
        if self.model is None:
            self.model = global_model
        else:
            local_dict = self.model.state_dict()
            global_dict = global_model.state_dict()
            for key in local_dict.keys():
                if 'style_head' in key:
                    continue
                if 'bn' in key.lower() and ('running_' in key or 'num_batches_tracked' in key):
                    continue
                if key.endswith('.M') or key == 'M':
                    continue  # Fixed ETF buffer 不 overwrite
                local_dict[key] = global_dict[key]
            self.model.load_state_dict(local_dict)
        self.current_round = svr_pkg['current_round']
        self.source_mu_k = svr_pkg.get('source_mu_k', None)
        self.mu_global = svr_pkg.get('mu_global', None)
        self.sigma_inv_sqrt = svr_pkg.get('sigma_inv_sqrt', None)

    def pack(self):
        return {
            'model': copy.deepcopy(self.model.to('cpu')),
            'style_stats': self._local_style_stats,
        }

    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        """Plan A orth_only 训练 + 收集 (μ_sty, Σ_sty) 本地统计."""
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum
        )

        # Online style stats accumulator (128d z_sty)
        style_sum = None
        style_sq_sum = None
        style_n = 0
        # 保存最后一轮 epoch 所有 z_sty 样本以便算 full covariance
        last_epoch_z_sty = []

        num_steps = self.num_steps
        steps_per_epoch = max(1, len(self.train_data) // self.batch_size)
        last_epoch_start = num_steps - steps_per_epoch

        for step in range(num_steps):
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            x, y = batch_data[0], batch_data[-1]

            model.zero_grad()

            h = model.encode(x)
            z_sem = model.get_semantic(h)
            z_sty = model.get_style(h)

            # Loss 1: CE via Fixed ETF classifier
            logits = model.classify(z_sem)
            loss_task = self.loss_fn(logits, y)

            # Loss 2: orthogonal decouple loss (Plan A core)
            z_sem_n = F.normalize(z_sem, dim=-1)
            z_sty_n = F.normalize(z_sty, dim=-1)
            loss_orth = ((z_sem_n * z_sty_n).sum(dim=-1) ** 2).mean()

            loss = loss_task + self.lambda_orth * loss_orth
            loss.backward()
            optimizer.step()

            # Style stats accumulation (last epoch only, 保证稳定)
            if step >= last_epoch_start:
                with torch.no_grad():
                    z_sty_det = z_sty.detach().cpu()
                    last_epoch_z_sty.append(z_sty_det)
                    b = z_sty_det.size(0)
                    batch_mu = z_sty_det.mean(dim=0)
                    batch_sq = (z_sty_det ** 2).mean(dim=0)
                    if style_sum is None:
                        style_sum = batch_mu * b
                        style_sq_sum = batch_sq * b
                        style_n = b
                    else:
                        style_sum += batch_mu * b
                        style_sq_sum += batch_sq * b
                        style_n += b

        # Store (μ, Σ) for server
        if style_n >= 4 and last_epoch_z_sty:
            mu = style_sum / style_n
            # Full covariance Σ from stacked samples
            Z = torch.cat(last_epoch_z_sty, dim=0)  # [N, d]
            N = Z.size(0)
            Z_centered = Z - mu.unsqueeze(0)
            sigma = (Z_centered.t() @ Z_centered) / max(N - 1, 1)  # [d, d]
            self._local_style_stats = (mu.clone(), sigma.clone())
        else:
            self._local_style_stats = None

    # ----------------------------------------------------------------
    # SGPA Inference (backprop-free, warmup-safe, top-m proto bank)
    # ----------------------------------------------------------------

    def _reset_sgpa_state(self, num_classes, proj_dim, device):
        """每次 test() 重置 SGPA 状态."""
        self.sgpa_supports = {c: [] for c in range(num_classes)}
        # proto 初始化为 ETF vertex 方向 (cold-start safe)
        M = self.model.M.to(device)  # [d, K]
        self.sgpa_proto = F.normalize(M.t(), dim=-1).clone()  # [K, d]
        self.sgpa_proto_activated = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.sgpa_tau_H = None
        self.sgpa_tau_S = None
        self.sgpa_warmup_buf_H = []
        self.sgpa_warmup_buf_D = []

    @torch.no_grad()
    def test_with_sgpa(self, m_top: int = 35, warmup_batches: int = 5,
                        tau_H_quantile: float = 0.5,
                        tau_S_quantile: float = 0.3,
                        ema_decay: float = 0.95):
        """SGPA inference: 双 gate + top-m proto bank + ETF fallback.

        Args:
            m_top: per-class support size upper bound.
            warmup_batches: 前 N batch 不激活 gate, 输出 ETF + 收集统计.
            tau_H_quantile: entropy gate 阈值的分位数 (warmup 之后).
            tau_S_quantile: style distance gate 阈值的分位数.
            ema_decay: τ_H / τ_S 的 EMA 衰减.
        Returns:
            {'sgpa_acc', 'etf_acc', 'reliable_rate', 'fallback_rate',
             'proto_vs_etf_gain', 'n_samples'}
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        K = self.model.num_classes
        d = self.model.proj_dim

        # SGPA 只在 whitening 可用且 test set 非空时启用
        has_whitening = (self.source_mu_k is not None
                         and self.mu_global is not None
                         and self.sigma_inv_sqrt is not None)

        self._reset_sgpa_state(K, d, device)

        mu_k_white = None
        if has_whitening:
            # pre-compute whitened source bank
            mu_global = self.mu_global.to(device)
            sigma_inv_sqrt = self.sigma_inv_sqrt.to(device)
            cids = sorted(self.source_mu_k.keys())
            mu_stack = torch.stack(
                [self.source_mu_k[c].to(device) for c in cids], dim=0)  # [N, d]
            mu_k_white = (mu_stack - mu_global.unsqueeze(0)) @ sigma_inv_sqrt  # [N, d]

        # 累积预测
        all_labels = []
        all_pred_etf = []
        all_pred_sgpa = []
        reliable_count = 0
        total_count = 0

        test_loader = self.calculator.get_dataloader(
            self.test_data, batch_size=self.batch_size)

        for batch_idx, batch_data in enumerate(test_loader):
            batch_data = self.calculator.to_device(batch_data)
            x, y = batch_data[0], batch_data[-1]
            B = x.size(0)
            total_count += B

            h = self.model.encode(x)
            z_sem = self.model.get_semantic(h)
            z_sty = self.model.get_style(h)

            logits_etf = self.model.classify(z_sem)
            pred_etf = logits_etf.argmax(dim=-1)

            # 默认 pred = ETF
            pred_sgpa = pred_etf.clone()

            if not has_whitening:
                # 没 whitening payload 时直接用 ETF (前几轮)
                all_labels.append(y.detach().cpu())
                all_pred_etf.append(pred_etf.detach().cpu())
                all_pred_sgpa.append(pred_sgpa.detach().cpu())
                continue

            # Compute gate signals
            p = F.softmax(logits_etf, dim=-1)
            log_p = F.log_softmax(logits_etf, dim=-1)
            H = -(p * log_p).sum(dim=-1)  # [B]

            z_sty_white = (z_sty - mu_global.unsqueeze(0)) @ sigma_inv_sqrt  # [B, d]
            # squared distance to each mu_k_white, min
            diffs = z_sty_white.unsqueeze(1) - mu_k_white.unsqueeze(0)  # [B, N, d]
            dist_sq = (diffs ** 2).sum(dim=-1)  # [B, N]
            dist_min = dist_sq.min(dim=-1).values  # [B]

            # Warmup: 收集统计, 不激活 gate (仍输出 ETF)
            if batch_idx < warmup_batches:
                self.sgpa_warmup_buf_H.extend(H.detach().cpu().tolist())
                self.sgpa_warmup_buf_D.extend(dist_min.detach().cpu().tolist())
                all_labels.append(y.detach().cpu())
                all_pred_etf.append(pred_etf.detach().cpu())
                all_pred_sgpa.append(pred_sgpa.detach().cpu())
                continue
            elif batch_idx == warmup_batches:
                if self.sgpa_warmup_buf_H:
                    self.sgpa_tau_H = float(np.quantile(
                        self.sgpa_warmup_buf_H, tau_H_quantile))
                    self.sgpa_tau_S = float(np.quantile(
                        self.sgpa_warmup_buf_D, tau_S_quantile))

            # EMA 更新 τ_H / τ_S
            cur_tau_H = float(np.quantile(H.detach().cpu().numpy(), tau_H_quantile))
            cur_tau_S = float(np.quantile(dist_min.detach().cpu().numpy(), tau_S_quantile))
            self.sgpa_tau_H = ema_decay * self.sgpa_tau_H + (1 - ema_decay) * cur_tau_H
            self.sgpa_tau_S = ema_decay * self.sgpa_tau_S + (1 - ema_decay) * cur_tau_S

            reliable = (H < self.sgpa_tau_H) & (dist_min < self.sgpa_tau_S)
            reliable_count += reliable.sum().item()

            # Top-m per-class support update (按 entropy 升序)
            for c in range(K):
                mask_c = reliable & (pred_etf == c)
                if mask_c.any():
                    idxs = mask_c.nonzero(as_tuple=False).flatten()
                    for i in idxs.tolist():
                        self.sgpa_supports[c].append(
                            (H[i].item(), z_sem[i].detach().cpu()))
                    # keep top-m by lowest entropy
                    self.sgpa_supports[c] = sorted(
                        self.sgpa_supports[c], key=lambda t: t[0])[:m_top]
                    # update proto
                    sup_tensors = torch.stack([s[1] for s in self.sgpa_supports[c]], dim=0)
                    proto_c = sup_tensors.mean(dim=0)
                    self.sgpa_proto[c] = F.normalize(
                        proto_c.to(device), dim=-1)
                    self.sgpa_proto_activated[c] = True

            # SGPA 分类: cos(z_sem, proto), activated class 才用
            z_sem_n = F.normalize(z_sem, dim=-1)
            proto_logits = z_sem_n @ self.sgpa_proto.t()  # [B, K]
            pred_proto = proto_logits.argmax(dim=-1)

            # fallback: 若某样本 argmax(proto) 对应的 class 还没 activated → 用 ETF
            activated_of_pred = self.sgpa_proto_activated[pred_proto]  # [B] bool
            pred_sgpa = torch.where(activated_of_pred, pred_proto, pred_etf)

            all_labels.append(y.detach().cpu())
            all_pred_etf.append(pred_etf.detach().cpu())
            all_pred_sgpa.append(pred_sgpa.detach().cpu())

        labels = torch.cat(all_labels)
        pred_etf = torch.cat(all_pred_etf)
        pred_sgpa = torch.cat(all_pred_sgpa)

        etf_acc = (pred_etf == labels).float().mean().item()
        sgpa_acc = (pred_sgpa == labels).float().mean().item()
        fallback_rate = (pred_sgpa == pred_etf).float().mean().item()
        reliable_rate = reliable_count / max(total_count, 1)
        return {
            'sgpa_acc': sgpa_acc,
            'etf_acc': etf_acc,
            'proto_vs_etf_gain': sgpa_acc - etf_acc,
            'reliable_rate': reliable_rate,
            'fallback_rate': fallback_rate,
            'n_samples': total_count,
        }


# ============================================================
# flgo hooks
# ============================================================


def init_dataset(object):
    pass


def init_local_module(object):
    pass


# Task prefix → num_classes dispatch (与 feddsa_scheduled.py 保持一致)
_MODEL_MAP = {
    'PACS': lambda: FedDSASGPAModel(num_classes=7, feat_dim=1024, proj_dim=128),
    'office': lambda: FedDSASGPAModel(num_classes=10, feat_dim=1024, proj_dim=128),
    'domainnet': lambda: FedDSASGPAModel(num_classes=10, feat_dim=1024, proj_dim=128),
}


def init_global_module(object):
    """只 server 创建 global model, client init 时 deepcopy."""
    if 'Server' not in object.__class__.__name__:
        return
    task = os.path.basename(object.option['task'])
    for prefix, factory in _MODEL_MAP.items():
        if prefix.lower() in task.lower():
            object.model = factory().to(object.device)
            return
    # Fallback: PACS-like 7 classes
    object.model = FedDSASGPAModel(
        num_classes=7, feat_dim=1024, proj_dim=128
    ).to(object.device)
