"""
FedDSA-DualEnc: Dual-Encoder Decouple-Share-Align with AdaIN Decoder + Cycle Anatomy Consistency
Ported to flgo/FDSE framework.

Architecture (delta vs feddsa.py orth_only):
    1. E_sem (1024 -> 512)        # 语义码维度提升, 不再瓶颈
    2. E_sty (1024 -> 16d VAE)    # 风格码改 VAE, 16d (z_sty_dim)
    3. Decoder (z_sem, z_sty)     # AdaIN-style modulation, 重建图 (256x256)
    4. Cycle anatomy consistency  # swap z_sty -> decode -> re-encode -> match z_sem (detached)

Loss (4 项 + KL warmup):
    L = L_CE + lambda_rec * L_rec + lambda_saac * L_saac + lambda_dsct * L_dsct + kl_w * L_kl

Aggregation:
    encoder + E_sem + decoder + classifier  -> FedAvg
    E_sty (mu_head + logvar_head) + BN running stats -> private (local)
    Style bank ((mu, sigma) of z_sty)       -> server-managed, not parameters

Reference:
    EXP-128 实验计划: refine-logs/2026-04-25_FedDSA-DualEnc/EXPERIMENT_PLAN.md
    Lineage: BiProto (failed, mode collapse) -> redesign with cycle replacing EMA Pd self-loop
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flgo.algorithm.fedbase
import flgo.utils.fmodule as fuf
from collections import OrderedDict


# ============================================================
# Backbone (sync with feddsa.py for fair comparison)
# ============================================================

class AlexNetEncoder(nn.Module):
    """Same AlexNet as feddsa.py but pooled to 1024-d."""
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


# ============================================================
# Style Reconstruction Module + AdaIN (CDDSA / MUNIT style)
# ============================================================

class SRM(nn.Module):
    """16d z_sty -> per-channel (gamma, beta) for AdaIN modulation."""
    def __init__(self, style_dim, hidden, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(style_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2 * out_channels)
        self.out_channels = out_channels

    def forward(self, z_sty):
        h = F.relu(self.fc1(z_sty))
        gb = self.fc2(h)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta


def adain(F_map, gamma, beta, eps=1e-5):
    """Instance norm + per-sample affine.
    F_map: (B, C, H, W),  gamma/beta: (B, C)
    """
    mean = F_map.mean(dim=(2, 3), keepdim=True)
    std = F_map.std(dim=(2, 3), keepdim=True, unbiased=False).clamp(min=eps)
    F_norm = (F_map - mean) / std
    g = gamma.view(*gamma.shape, 1, 1)
    b = beta.view(*beta.shape, 1, 1)
    return g * F_norm + b


class AdaINBlock(nn.Module):
    """ConvTranspose -> AdaIN -> ReLU."""
    def __init__(self, in_c, out_c, style_dim, srm_hidden, kernel=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding)
        self.srm = SRM(style_dim, srm_hidden, out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, F_map, z_sty):
        F_map = self.conv(F_map)
        gamma, beta = self.srm(z_sty)
        F_map = adain(F_map, gamma, beta)
        return self.relu(F_map)


# ============================================================
# Decoder: z_sem (512) + z_sty (16) -> image (256x256)
# ============================================================

class Decoder(nn.Module):
    """3 AdaIN blocks (8->16->32->64) + 2 plain upsample (64->128->256)."""
    def __init__(self, sem_dim=512, sty_dim=16, srm_hidden=256, base_size=8, out_channels=3):
        super().__init__()
        self.base_size = base_size
        self.fc = nn.Linear(sem_dim, 256 * base_size * base_size)
        self.block1 = AdaINBlock(256, 128, sty_dim, srm_hidden)  # 8 -> 16
        self.block2 = AdaINBlock(128, 64, sty_dim, srm_hidden)   # 16 -> 32
        self.block3 = AdaINBlock(64, 32, sty_dim, srm_hidden)    # 32 -> 64
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 64 -> 128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, out_channels, 4, 2, 1),  # 128 -> 256
            nn.Tanh(),
        )

    def forward(self, z_sem, z_sty):
        x = self.fc(z_sem)
        x = x.view(-1, 256, self.base_size, self.base_size)
        x = self.block1(x, z_sty)
        x = self.block2(x, z_sty)
        x = self.block3(x, z_sty)
        x = self.upsample(x)
        return x  # [B, 3, 256, 256], range [-1, 1]


# ============================================================
# DualEnc Model: backbone + E_sem + E_sty (VAE) + decoder + classifier
# ============================================================

class FedDSADualEncModel(fuf.FModule):
    """Dual-encoder + AdaIN decoder + classifier.

    Note: 'classifier' field naming: framework expects `head` for default forward.
    We use `head` (single Linear) for compat, and route training through z_sem.
    """
    def __init__(self, num_classes=7, feat_dim=1024, sem_dim=512, sty_dim=16, srm_hidden=256):
        super().__init__()
        self.encoder = AlexNetEncoder()
        # E_sem: backbone pooled (1024) -> z_sem (512)
        self.semantic_head = nn.Linear(feat_dim, sem_dim)
        # E_sty: VAE encoder -> mu, logvar (each 16-d)
        self.style_mu_head = nn.Linear(feat_dim, sty_dim)
        self.style_logvar_head = nn.Linear(feat_dim, sty_dim)
        # Decoder
        self.decoder = Decoder(sem_dim=sem_dim, sty_dim=sty_dim, srm_hidden=srm_hidden)
        # Classifier (named `head` for flgo framework forward path compat)
        self.head = nn.Linear(sem_dim, num_classes)

        self.sem_dim = sem_dim
        self.sty_dim = sty_dim

    def forward(self, x):
        h = self.encoder(x)
        z_sem = self.semantic_head(h)
        return self.head(z_sem)

    def encode(self, x):
        return self.encoder(x)

    def get_semantic(self, h):
        return self.semantic_head(h)

    def get_style(self, h):
        """Return (mu, logvar). Sampling done in training step."""
        return self.style_mu_head(h), self.style_logvar_head(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_sem, z_sty):
        return self.decoder(z_sem, z_sty)


# ============================================================
# Server
# ============================================================

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        # algo_para 顺序 (与 yml 对齐):
        # 0 lambda_rec
        # 1 lambda_saac
        # 2 lambda_dsct
        # 3 lambda_kl
        # 4 kl_warmup_rounds
        # 5 saac_warmup_rounds
        # 6 sem_dim
        # 7 sty_dim
        # 8 srm_hidden
        # 9 bank_K
        # 10 tau (InfoNCE temperature for L_dsct)
        self.init_algo_para({
            'lambda_rec': 0.001,
            'lambda_saac': 1.0,
            'lambda_dsct': 0.01,
            'lambda_kl': 0.01,
            'kl_warmup_rounds': 10,
            'saac_warmup_rounds': 10,
            'sem_dim': 512,
            'sty_dim': 16,
            'srm_hidden': 256,
            'bank_K': 4,
            'tau': 0.1,
        })
        self.sample_option = 'full'

        # Style bank: client_id -> tensor (N, sty_dim) of z_sty samples (mu only)
        # 不存 (μ, σ) per channel of pooled feature, 改存 z_sty 实采样 — CDDSA 风格
        self.style_bank = {}

        self._init_agg_keys()

        # Pass config to clients
        for c in self.clients:
            c.lambda_rec = self.lambda_rec
            c.lambda_saac = self.lambda_saac
            c.lambda_dsct = self.lambda_dsct
            c.lambda_kl = self.lambda_kl
            c.kl_warmup_rounds = self.kl_warmup_rounds
            c.saac_warmup_rounds = self.saac_warmup_rounds
            c.bank_K = self.bank_K
            c.tau = self.tau
            # 模型超参 (multi_gpus wrap 后无法靠 self.model.* 读)
            c.sty_dim = self.sty_dim
            c.sem_dim = self.sem_dim

    def _init_agg_keys(self):
        """Classify keys into shared (FedAvg) vs private (local).
        Private: style_mu_head + style_logvar_head + BN running stats.
        Shared: encoder, semantic_head, decoder, head, BN gamma/beta (affine params).
        """
        all_keys = list(self.model.state_dict().keys())
        self.private_keys = set()
        for k in all_keys:
            if 'style_mu_head' in k or 'style_logvar_head' in k:
                self.private_keys.add(k)
            elif 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
                self.private_keys.add(k)
        self.shared_keys = [k for k in all_keys if k not in self.private_keys]

    def pack(self, client_id, mtype=0):
        """Send global model + style bank dispatch (exclude client's own).

        Codex IMPORTANT 修正: 旧版 fallback 到 self bank, 单 client / sparse-bank 时
        cross-client SAAC/DSCT 退化成 self-source, 语义错误. 修正: 没有别 client 风格
        时返回 None (client 端见 None 直接 saac_active=False, 不进 cycle).
        """
        dispatched = None
        active_round = max(0, self.current_round - 1) >= max(self.saac_warmup_rounds, 1)
        if active_round and len(self.style_bank) > 0:
            available = {
                cid: s for cid, s in self.style_bank.items()
                if cid != client_id and s is not None and s.numel() > 0
            }
            if len(available) > 0:
                dispatched = available
            # else: dispatched 保持 None, client 见 None 关闭 SAAC/DSCT

        return {
            'model': copy.deepcopy(self.model),
            'style_bank': dispatched,
            'current_round': self.current_round,
        }

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)

        models = res['model']
        style_samples_list = res['style_samples']

        # 1. Aggregate shared parameters (FedAvg, exclude E_sty + BN running stats)
        self._aggregate_shared(models)

        # 2. Update style bank: each client overwrites its own slot with fresh z_sty samples
        for cid, samples in zip(self.received_clients, style_samples_list):
            if samples is not None and samples.numel() > 0:
                self.style_bank[cid] = samples

    def _aggregate_shared(self, models):
        """Sample-count-weighted FedAvg on shared keys."""
        if len(models) == 0:
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


# ============================================================
# Client
# ============================================================

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = copy.deepcopy(self.server.model).to('cpu')
        self.loss_fn = nn.CrossEntropyLoss()
        self.current_round = 0
        self.received_bank = None
        # 上传给 server 的 z_sty 样本
        self._uploaded_style_samples = None
        # 本 client 上限 z_sty 采样数 (每轮收集),控通信代价
        self._max_style_samples = 200

    def reply(self, svr_pkg):
        model, style_bank, current_round = self.unpack(svr_pkg)
        self.current_round = current_round
        self.received_bank = style_bank
        self.train(model)
        return self.pack()

    def unpack(self, svr_pkg):
        """Apply only shared keys; keep style_*_head and BN running stats local."""
        global_model = svr_pkg['model']
        if self.model is None:
            self.model = global_model
        else:
            new_dict = self.model.state_dict()
            global_dict = global_model.state_dict()
            for key in new_dict.keys():
                if 'style_mu_head' in key or 'style_logvar_head' in key:
                    continue
                if 'bn' in key.lower() and ('running_' in key or 'num_batches_tracked' in key):
                    continue
                new_dict[key] = global_dict[key]
            self.model.load_state_dict(new_dict)
        return self.model, svr_pkg['style_bank'], svr_pkg['current_round']

    def pack(self):
        return {
            'model': copy.deepcopy(self.model.to('cpu')),
            'style_samples': self._uploaded_style_samples,
        }

    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum,
        )

        # KL warmup: round 0..kl_warmup_rounds-1 disabled, then linear ramp to lambda_kl
        kl_w = self._kl_weight()
        # SAAC active flag (cycle 需要至少 1 个别 client 风格)
        # Codex IMPORTANT 修正: warmup 用 1-based 跟 _kl_weight 对齐.
        # round=1 时 progress=0, 当 progress >= saac_warmup_rounds 才激活.
        saac_active = (
            self.received_bank is not None
            and len(self.received_bank) > 0
            and max(0, self.current_round - 1) >= max(self.saac_warmup_rounds, 1)
        )

        # 收集本轮 z_sty 样本用于上传 (last epoch 内)
        collected_z_sty = []
        # 收集 z_sty 用于 L_dsct (域间散开) 跨 client 负例: 直接复用 received_bank
        prebuilt_negatives = self._build_negative_pool() if saac_active else None

        num_steps = self.num_steps
        last_epoch_start = max(0, num_steps - max(1, len(self.train_data) // self.batch_size))

        for step in range(num_steps):
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            x, y = batch_data[0], batch_data[-1]
            B = x.size(0)

            model.zero_grad()

            # === Forward (原图) ===
            h = model.encode(x)                            # [B, 1024]
            z_sem = model.get_semantic(h)                   # [B, sem_dim]
            mu, logvar = model.get_style(h)                 # [B, sty_dim] each
            z_sty = model.reparameterize(mu, logvar)        # [B, sty_dim]

            # === L_CE (主任务) ===
            logits = model.head(z_sem)
            loss_ce = self.loss_fn(logits, y)

            # === L_rec (自重建) ===
            x_hat = model.decode(z_sem, z_sty)               # [B, 3, 256, 256]
            x_target = self._normalize_target(x)             # match Tanh range
            loss_rec = F.l1_loss(x_hat, x_target)

            # === L_saac (cycle: 双向一致性 — anatomy + style) ===
            # 关键修复 (codex CRITICAL 1): 必须让 cycle 同时回流梯度到 style encoder, 否则
            # style_*_head 几乎只受 0.001*L_rec + KL 约束, 容易学成 per-client centroid.
            # 双向 cycle (MUNIT/CDDSA 标准): 注入 swap style 后 anatomy 不变 + 注入 style
            # 通过 image-encoder cycle 后能被 style_head 识别回来.
            loss_saac = torch.tensor(0.0, device=x.device)
            if saac_active:
                z_sty_swap = self._sample_swap(B, x.device, prebuilt_negatives)
                x_swap = model.decode(z_sem, z_sty_swap)
                # re-encode (走完整 pipeline backbone -> E_sem + E_sty)
                h_swap = model.encode(x_swap)
                z_sem_swap = model.get_semantic(h_swap)
                mu_swap_after, _ = model.get_style(h_swap)
                # (a) anatomy 一致: GT detach 防 BiProto-style 自循环坍缩
                loss_anat = F.l1_loss(z_sem_swap, z_sem.detach())
                # (b) style cycle 一致: 注入的 z_sty_swap 应能被 style_head 识别 (z_sty_swap 来自
                #     bank 已经 detach, 这里反向梯度通过 decoder + encoder + E_sty 回流, 训练
                #     style encoder 能从图像里 "看出" 风格 — 防 codex CRITICAL 1)
                loss_style_cyc = F.l1_loss(mu_swap_after, z_sty_swap.detach())
                loss_saac = loss_anat + loss_style_cyc

            # === L_dsct (push current z_sty AWAY from other-client styles) ===
            # 关键修复 (codex CRITICAL 2): 不再把 batch 内同 client 样本互相拉近 (会塌成
            # per-client 单团), 改为 instance-vs-bank InfoNCE — 自己跟自己 = positive,
            # bank 中所有别 client 风格 = negatives.
            loss_dsct = torch.tensor(0.0, device=x.device)
            if saac_active and prebuilt_negatives is not None and prebuilt_negatives.numel() > 0:
                loss_dsct = self._dsct_loss(z_sty, prebuilt_negatives)

            # === L_kl (VAE KL to N(0, I)) ===
            loss_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()

            # === Total ===
            loss = (
                loss_ce
                + self.lambda_rec * loss_rec
                + self.lambda_saac * loss_saac
                + self.lambda_dsct * loss_dsct
                + kl_w * loss_kl
            )

            loss.backward()
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
            optimizer.step()

            # === 本 epoch 收集 z_sty 用于上传 ===
            # Codex IMPORTANT 修正: 旧版上传 mu (deterministic), 但 mix 时 sum|alpha|=1
            # 让 var 缩到 Σα²<1, 跟训练分布错位. 改上传 sampled z (含重参噪声), 真实
            # 反映 posterior, 跨 client mix 后方差量级合理.
            if step >= last_epoch_start:
                with torch.no_grad():
                    collected_z_sty.append(z_sty.detach().cpu())

        # 限制上传样本数, sub-sample 至 _max_style_samples
        self._uploaded_style_samples = self._compose_upload(collected_z_sty)

    # ---- Helper functions ----

    def _kl_weight(self):
        """Linear KL warmup from 0 to lambda_kl over kl_warmup_rounds.

        Codex IMPORTANT 修正: flgo current_round 从 1 开始计数, 0-based 公式会让
        round=1 就有 0.1*lambda_kl, warm 实际只走 (warm-1) 轮. 改 1-based:
        round=1 -> 0/warm, round=warm+1 -> 1.0.
        """
        warm = max(self.kl_warmup_rounds, 1)
        # 用 (current_round - 1), 在 round=1 时 ratio=0, 在 round=warm+1 时 ratio=1
        progress = max(0, self.current_round - 1)
        if progress < warm:
            ratio = progress / float(warm)
        else:
            ratio = 1.0
        return ratio * self.lambda_kl

    @staticmethod
    def _normalize_target(x):
        """If x is in [0, 1] (PFLlib style) -> map to [-1, 1] for Tanh.
        If x already in [-1, 1] (e.g. ImageNet-norm), pass through.
        Heuristic: if max <= 1.5 and min >= -0.5 -> [0,1] domain.
        """
        with torch.no_grad():
            x_min, x_max = x.min(), x.max()
        if x_min.item() >= -0.5 and x_max.item() <= 1.5:
            return x * 2.0 - 1.0
        return x

    def _build_negative_pool(self):
        """Concat all other-client z_sty samples into a pool tensor [N, sty_dim]."""
        if self.received_bank is None or len(self.received_bank) == 0:
            return None
        tensors = [s for s in self.received_bank.values() if s is not None and s.numel() > 0]
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)

    def _sample_swap(self, batch_size, device, pool):
        """For each sample, draw K z_sty from pool, mix via U(-1, 1) linear combo (CDDSA)."""
        if pool is None or pool.size(0) == 0:
            # fallback: 随机噪声 (理论上 saac_active=False 已挡住, 这里防御)
            return torch.randn(batch_size, self.sty_dim, device=device)
        N = pool.size(0)
        K = max(1, min(self.bank_K, N))
        # 对每个 batch 样本独立采 K 个并组合
        idx = torch.randint(0, N, (batch_size, K), device='cpu')
        chosen = pool[idx]  # [B, K, sty_dim]
        alpha = (torch.rand(batch_size, K) * 2.0 - 1.0)  # U(-1, 1)
        denom = alpha.abs().sum(dim=1, keepdim=True).clamp(min=1e-6)
        alpha = alpha / denom
        z_sty_swap = (alpha.unsqueeze(-1) * chosen).sum(dim=1)
        return z_sty_swap.to(device)

    def _dsct_loss(self, z_sty, negatives):
        """Instance discrimination: current z_sty (self) vs other-client styles (bank).

        Codex CRITICAL 2 修正: 旧版本把 batch 内除自己外都当正例, 强拉同 client 内
        z_sty 紧 → optimal solution = per-client single cluster, 灾难性破坏 bank 多样性.
        新版本: 单一 positive (自身, cosine=1/tau 固定) + bank 全部负例, 等价于 instance
        discrimination, 只 push z_sty 远离别 client 风格, 不强制同 client 内紧.
        """
        B = z_sty.size(0)
        if B < 1 or negatives.size(0) == 0:
            return torch.tensor(0.0, device=z_sty.device)
        z_n = F.normalize(z_sty, dim=1)
        neg_n = F.normalize(negatives.to(z_sty.device), dim=1)
        # 自相似 (B, 1) — z_n 已归一, z·z=1, 除以 tau
        self_sim = (z_n * z_n).sum(dim=1, keepdim=True) / self.tau
        # 跟 bank 中每个负例的相似度 (B, N)
        neg_sim = z_n @ neg_n.T / self.tau
        # CE: target = 0 (positive 永远在第 0 列)
        logits = torch.cat([self_sim, neg_sim], dim=1)
        targets = torch.zeros(B, dtype=torch.long, device=z_sty.device)
        return F.cross_entropy(logits, targets)

    def _compose_upload(self, collected):
        """Sub-sample to <= self._max_style_samples for communication budget."""
        if not collected:
            return None
        all_z = torch.cat(collected, dim=0)  # [N_total, sty_dim]
        N = all_z.size(0)
        if N == 0:
            return None
        if N > self._max_style_samples:
            idx = torch.randperm(N)[: self._max_style_samples]
            all_z = all_z[idx]
        return all_z


# ============================================================
# Model registration (called by flgo framework)
# ============================================================

# 注: flgo 分发 model_map 在 init 时一次性 instantiate, 所以 sem_dim/sty_dim 必须 server.init_algo_para 之前固定
# 实际项目中 algo_para 顺序约定后, 模型超参直接用 yml 默认 (sem_dim=512, sty_dim=16, srm_hidden=256)
def _make_model(num_classes, sem_dim=512, sty_dim=16, srm_hidden=256):
    return FedDSADualEncModel(
        num_classes=num_classes, feat_dim=1024,
        sem_dim=sem_dim, sty_dim=sty_dim, srm_hidden=srm_hidden,
    )


model_map = {
    'PACS': lambda: _make_model(num_classes=7),
    'office': lambda: _make_model(num_classes=10),
    'domainnet': lambda: _make_model(num_classes=10),
}


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    dataset = os.path.split(object.option['task'])[-1].split('_')[0]
    if 'Server' in object.__class__.__name__:
        object.model = model_map.get(dataset, lambda: _make_model(num_classes=7))().to(object.device)
