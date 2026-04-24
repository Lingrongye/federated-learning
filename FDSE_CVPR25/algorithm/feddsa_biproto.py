"""
FedDSA-BiProto: Federated Domain Prototype as a First-Class Shared Object via
                Straight-Through Hybrid Exclusion.

研究方案文档: obsidian_exprtiment_results/refine_logs/2026-04-24_feddsa-biproto-asym_v1/FINAL_PROPOSAL.md
大白话版本: obsidian_exprtiment_results/知识笔记/大白话_FedDSA-BiProto方案.md

继承 feddsa_scheduled.py (orth_only base) 并加:
  1. BiProtoAlexNetEncoder: AlexNet + 中间层 (μ,σ) tap 抽取 (state_dict 兼容 orth_only ckpt)
  2. StatisticEncoder: 2-layer MLP on conv1-3 (μ,σ) → z_sty (~1M params)
  3. Pc / Pd EMA buffers (server-aggregated, broadcast to clients)
  4. L_sty_proto = InfoNCE(z_sty, Pd) + 0.5 * MSE(z_sty, stopgrad(Pd[d]))
  5. L_proto_excl = present-classes-only cos² with hybrid ST domain axis
  6. freeze_encoder_sem flag for C0 matched-intervention gate
  7. save_best ckpt 继承自 feddsa_scheduled (se=1)

algo_para order (12 items):
  lo, lh, ls, tau, sdn, pd, sm, bp, bw, cr, gli, lm  ← inherit from feddsa_scheduled
  + lp = lambda_sty_proto (peak weight)
  + le = lambda_proto_excl (peak weight)
  + tp = tau_proto (InfoNCE temperature for L_sty_proto)
  + mc = mse_coef (MSE anchor weight inside L_sty_proto)
  + ws = warmup_start (first round of ramp-up)
  + we = warmup_end (peak phase begins)
  + rd = ramp_down (start of L_sty_proto ramp-down)
  + emd = ema_decay (Pc/Pd EMA decay)
  + se = save_endpoint (1 = save best ckpt)
  + fz = freeze_encoder_sem (1 = C0 gate mode)
"""
import os
import copy
import math
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flgo.algorithm.fedbase
import flgo.utils.fmodule as fuf
from collections import OrderedDict

from algorithm.feddsa_scheduled import (
    AlexNetEncoder as _BaseAlexNetEncoder,
    Server as _BaseServer,
    Client as _BaseClient,
)


# ============================================================
# Constants for AlexNet conv1-3 channel counts (used by StatisticEncoder)
# ============================================================
_TAP_CHANNELS = [64, 192, 384]  # conv1=64, conv2=192, conv3=384 (after relu+pool)
_TAP_NAMES = ('maxpool1', 'maxpool2', 'relu3')  # tap points in features Sequential


# ============================================================
# Model
# ============================================================

class BiProtoAlexNetEncoder(_BaseAlexNetEncoder):
    """AlexNet encoder + intermediate (μ,σ) tap exposure.

    state_dict 完全兼容 _BaseAlexNetEncoder, 可从 orth_only checkpoint 加载.
    """

    def forward_with_taps(self, x):
        """Forward + return list of intermediate activations at conv1-3 outputs.

        Returns:
            pooled: [B, 1024] — same as forward(x)
            taps: list of 3 tensors with shapes:
                [B, 64, h1, w1], [B, 192, h2, w2], [B, 384, h3, w3]
        """
        taps = []
        out = x
        for name, layer in self.features.named_children():
            out = layer(out)
            if name in _TAP_NAMES:
                taps.append(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.relu(self.bn6(self.fc1(out)))
        out = self.relu(self.bn7(self.fc2(out)))
        return out, taps


class StatisticEncoder(nn.Module):
    """风格 encoder: 从 conv1-3 (μ,σ) 统计量 → z_sty [128].

    参数量约 1M:
      Linear(2*640=1280, 512) + LayerNorm(512) + Linear(512, 128) ≈ 657K + 65K ≈ 722K
    """

    def __init__(self, tap_channels=_TAP_CHANNELS, hidden=512, out_dim=128):
        super().__init__()
        in_dim = 2 * sum(tap_channels)  # 2 (μ,σ) × 640 = 1280
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, taps):
        """taps: list of [B, C_l, H_l, W_l] (already detached by caller).

        Returns z_sty: [B, out_dim], NOT L2-normalized (caller normalizes).
        """
        feats = []
        for t in taps:
            mu = t.mean(dim=(2, 3))                        # [B, C]
            sigma = t.std(dim=(2, 3), unbiased=False) + 1e-5  # [B, C]
            feats.append(mu)
            feats.append(sigma)
        s = torch.cat(feats, dim=-1)  # [B, 2*sum(C)]
        return self.net(s)


class FedDSABiProtoModel(fuf.FModule):
    """Inherits orth_only model architecture but replaces style_head with
    StatisticEncoder fed by encoder's intermediate (μ,σ) taps.

    Pc/Pd are no-grad EMA buffers, server-aggregated and broadcast.

    state_dict layout (主要 keys):
      encoder.*           — same as feddsa_scheduled FedDSAModel.encoder (兼容 orth_only ckpt)
      semantic_head.*     — same
      head.*              — same (sem_classifier in proposal)
      encoder_sty.net.*   — NEW: statistic encoder MLP
      Pc                  — NEW buffer [num_classes, proj_dim]
      Pd                  — NEW buffer [num_clients, proj_dim]
    """

    def __init__(self, num_classes=7, num_clients=4, feat_dim=1024, proj_dim=128):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_clients = int(num_clients)
        self.proj_dim = int(proj_dim)

        self.encoder = BiProtoAlexNetEncoder()
        # NOTE: 命名 semantic_head/head 与 feddsa_scheduled 完全一致, 保证 ckpt 加载
        self.semantic_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.head = nn.Linear(proj_dim, num_classes)

        # 风格分支 (替代原 style_head): 仅在 model 命名空间, FedAvg 聚合
        self.encoder_sty = StatisticEncoder(out_dim=proj_dim)

        # Pc/Pd: no-grad EMA buffers (server 维护, broadcast 时同步)
        # 初始化为单位球面上的随机向量 (避免 cosine NaN)
        Pc_init = F.normalize(torch.randn(num_classes, proj_dim), dim=-1)
        Pd_init = F.normalize(torch.randn(num_clients, proj_dim), dim=-1)
        self.register_buffer('Pc', Pc_init)
        self.register_buffer('Pd', Pd_init)

    # ============== 兼容现有诊断脚本的 3 个接口 ==============

    def encode(self, x):
        """Alias for orth_only-compatible encode interface (returns pooled 1024d).

        诊断脚本 (visualize_tsne / run_capacity_probes) 用这个.
        """
        pooled, _ = self.encoder.forward_with_taps(x)
        return pooled

    def encode_with_taps(self, x):
        """Returns (pooled, taps) — used internally by training to feed encoder_sty."""
        return self.encoder.forward_with_taps(x)

    def get_semantic(self, h):
        return self.semantic_head(h)

    def get_style(self, h_or_taps):
        """Compute z_sty.

        Note: 训练时 caller 应该先 detach taps. 这里**不主动 detach**, 留给 caller 控制.
        诊断脚本 (无梯度场景) 输入 pooled (1024d) — 但 BiProto 不能直接从 pooled 得 z_sty,
        所以诊断脚本要先调 encode_with_taps(x). 因此本方法只接受 taps (list of tensors).
        """
        if isinstance(h_or_taps, list):
            return self.encoder_sty(h_or_taps)
        # 兼容: 如果传进来是 1024d pooled (诊断脚本), 报错引导
        raise ValueError(
            "BiProto.get_style requires taps (list of intermediate activations); "
            "use model.encode_with_taps(x) to obtain (pooled, taps) and pass taps."
        )

    def forward(self, x):
        """Standard forward — only used for inference/eval. Returns logits."""
        h, _ = self.encoder.forward_with_taps(x)
        z_sem = self.semantic_head(h)
        return self.head(z_sem)


# ============================================================
# Server: 继承 feddsa_scheduled.Server, override Pc/Pd 管理
# ============================================================

class Server(_BaseServer):
    def initialize(self):
        """**独立 init_algo_para** (不调 super) 避免索引冲突 (父类有 18 项, 我们 22 项).

        algo_para 顺序 (22 items, BiProto 自有定义):
          [0]  lo  - lambda_orth (always on, 1.0)
          [1]  lh  - lambda_hsic (legacy, 0=off, 推荐设 0)
          [2]  ls  - lambda_sem (legacy InfoNCE 不用, 设 1.0 占位)
          [3]  tau - InfoNCE temperature legacy (0.2)
          [4]  sdn - style_dispatch_num (5)
          [5]  pd  - proj_dim (128)
          [6]  sm  - schedule_mode (0=orth_only base, BiProto 自己有 schedule)
          [7]  bp  - bell_peak legacy
          [8]  bw  - bell_width legacy
          [9]  cr  - cutoff_round legacy
          [10] gli - grad_log_interval (0=off)
          [11] lm  - lambda_mse legacy
          --- BiProto 新增 (10 items) ---
          [12] lp  - lambda_sty_proto peak (0.5)
          [13] le  - lambda_proto_excl peak (0.3)
          [14] tp  - tau_proto (InfoNCE temperature for L_sty_proto, 0.1)
          [15] mc  - mse_coef (MSE anchor weight, 0.5)
          [16] ws  - warmup_start (50)
          [17] we  - warmup_end (80)
          [18] rd  - ramp_down (150)
          [19] emd - ema_decay (0.9)
          [20] se  - save_endpoint (1=save best ckpt)
          [21] fz  - freeze_encoder_sem (1=C0 gate mode)
        """
        # 直接 init_algo_para 22 项, 不 super (避免父类 18 项与我们冲突)
        self.init_algo_para({
            'lo': 1.0, 'lh': 0.0, 'ls': 1.0, 'tau': 0.2, 'sdn': 5, 'pd': 128,
            'sm': 0, 'bp': 60, 'bw': 30, 'cr': 80, 'gli': 0, 'lm': 1.0,
            'lp': 0.5, 'le': 0.3, 'tp': 0.1, 'mc': 0.5,
            'ws': 50, 'we': 80, 'rd': 150, 'emd': 0.9,
            'se': 0, 'fz': 0,
        })

        # base aliases (与父类同名, 给 super.train 等使用)
        self.lambda_orth = float(self.lo)
        self.lambda_hsic = float(self.lh)
        self.lambda_sem = float(self.ls)
        self.tau = float(self.tau)
        self.style_dispatch_num = int(self.sdn)
        self.proj_dim = int(self.pd)
        self.schedule_mode = int(self.sm)
        self.bell_peak = float(self.bp)
        self.bell_width = float(self.bw)
        self.cutoff_round = int(self.cr)
        self.grad_log_interval = int(self.gli)
        self.lambda_mse = float(self.lm)
        # 父类 train 函数读这些, 给默认值占位
        self.alpha_sparsity = 0.25
        self.style_aware_sem = 0
        self.style_aware_tau = 0.3
        self.scpr_mode = 0
        self.scpr_tau_val = 0.3
        self.sample_option = 'full'

        # BiProto-specific aliases
        self.lambda_sty_proto = float(self.lp)
        self.lambda_proto_excl = float(self.le)
        self.tau_proto = float(self.tp)
        self.mse_coef = float(self.mc)
        self.warmup_start = int(self.ws)
        self.warmup_end = int(self.we)
        self.ramp_down = int(self.rd)
        self.ema_decay = float(self.emd)
        self.save_errors = int(self.se)  # 复用父类 save_best 机制
        self.freeze_encoder_sem = int(self.fz)

        # 父类 Server.initialize 还做的事 (server-side state)
        self.client_sem_states = {}
        self.client_head_states = {}
        self.style_bank = {}
        self.scpr_style_bank = {}
        self.global_semantic_protos = {}
        self.client_class_protos = {}
        self.grad_conflict_log = {}
        self._best_avg_acc = -1.0
        self._best_round = 0
        self._best_global_state = None
        self._best_client_states = None
        self._best_fedavg_head_state = None

        self._init_agg_keys()

        # ---- Optional: load orth_only checkpoint (for S0 matched-intervention gate) ----
        ckpt_path = self.option.get('init_ckpt', None)
        if ckpt_path:
            self._load_orth_only_ckpt(ckpt_path)

        # Pass 给 client
        for c in self.clients:
            # base
            c.lambda_orth = self.lambda_orth
            c.lambda_hsic = self.lambda_hsic
            c.lambda_sem = self.lambda_sem
            c.tau = self.tau
            c.proj_dim = self.proj_dim
            c.schedule_mode = self.schedule_mode
            c.bell_peak = self.bell_peak
            c.bell_width = self.bell_width
            c.cutoff_round = self.cutoff_round
            c.grad_log_interval = self.grad_log_interval
            c.lambda_mse = self.lambda_mse
            # parent extras (provide neutral defaults so super().train doesn't break)
            c.alpha_sparsity = self.alpha_sparsity
            c.scpr_mode = self.scpr_mode
            c.scpr_tau_val = self.scpr_tau_val
            # BiProto
            c.lambda_sty_proto = self.lambda_sty_proto
            c.lambda_proto_excl = self.lambda_proto_excl
            c.tau_proto = self.tau_proto
            c.mse_coef = self.mse_coef
            c.warmup_start = self.warmup_start
            c.warmup_end = self.warmup_end
            c.ramp_down = self.ramp_down
            c.ema_decay = self.ema_decay
            c.freeze_encoder_sem = self.freeze_encoder_sem
            c.num_clients_total = len(self.clients)
            c.num_rounds = int(self.num_rounds)  # ★ Client schedule 用

    def _load_orth_only_ckpt(self, ckpt_path):
        """从 orth_only ckpt 加载 encoder/semantic_head/head 权重 (strict=False 跳过 BiProto-only keys).

        Used for S0 matched-intervention gate: 加载 EXP-105 orth_only Office R200 ckpt 后冻结 encoder.
        ckpt_path: 路径到 ~/fl_checkpoints/feddsa_*/global_model.pt 或绝对路径
        """
        import os
        import torch as _t
        path = os.path.expanduser(ckpt_path)
        if os.path.isdir(path):
            path = os.path.join(path, 'global_model.pt')
        if not os.path.isfile(path):
            try:
                self.gv.logger.warning(f"[BiProto] init_ckpt not found: {path}, skip load")
            except Exception:
                pass
            return
        try:
            sd = _t.load(path, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            res = self.model.load_state_dict(sd, strict=False)
            try:
                self.gv.logger.info(
                    f"[BiProto] loaded init_ckpt={path}, "
                    f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
                )
            except Exception:
                pass
        except Exception as e:
            try:
                self.gv.logger.warning(f"[BiProto] failed to load init_ckpt {path}: {e}")
            except Exception:
                pass

    def _init_agg_keys(self):
        """Override: encoder_sty 全部聚合 (FedAvg); BN running stats 本地; Pc/Pd 不参与 FedAvg
        (用 server-side EMA 聚合)."""
        all_keys = list(self.model.state_dict().keys())
        self.private_keys = set()
        for k in all_keys:
            # FedBN 原则: BN running stats 本地
            if 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
                self.private_keys.add(k)
            # Pc/Pd 不参与 FedAvg, server EMA
            if k in ('Pc', 'Pd'):
                self.private_keys.add(k)
        self.shared_keys = [k for k in all_keys if k not in self.private_keys]

    def iterate(self):
        """Override: 在父类 iterate (FedAvg shared keys) 后, 增加 Pc/Pd EMA 更新."""
        # 父类完成 shared_keys 的 FedAvg + sas 个性化 + style_bank 更新
        super().iterate()

        # ---- BiProto: server EMA aggregate Pc / Pd ----
        # 收集 client 上传的 (class_means, domain_mean, domain_id, client_n_classes)
        Pc_aggregates = collections.defaultdict(list)  # class_id -> [(mean, count), ...]
        Pd_aggregates = collections.defaultdict(list)  # domain_id -> [(mean, count), ...]
        for c in self.clients:
            payload = getattr(c, '_biproto_payload', None)
            if payload is None:
                continue
            for cls_id, (mean, count) in payload.get('class_means', {}).items():
                Pc_aggregates[int(cls_id)].append((mean, int(count)))
            d_id = payload.get('domain_id', None)
            d_mean = payload.get('domain_mean', None)
            d_count = payload.get('domain_count', 0)
            if d_id is not None and d_mean is not None and d_count > 0:
                Pd_aggregates[int(d_id)].append((d_mean, int(d_count)))

        m = float(self.ema_decay)
        with torch.no_grad():
            # Update Pc
            for c_id, items in Pc_aggregates.items():
                if not items:
                    continue
                total = sum(cnt for _, cnt in items)
                if total <= 0:
                    continue
                agg = sum(mean * (cnt / total) for mean, cnt in items)
                agg = F.normalize(agg.float().to(self.model.Pc.device), dim=-1)
                cur = self.model.Pc[c_id]
                self.model.Pc[c_id] = F.normalize(m * cur + (1 - m) * agg, dim=-1)
            # Update Pd
            for d_id, items in Pd_aggregates.items():
                if not items:
                    continue
                total = sum(cnt for _, cnt in items)
                if total <= 0:
                    continue
                agg = sum(mean * (cnt / total) for mean, cnt in items)
                agg = F.normalize(agg.float().to(self.model.Pd.device), dim=-1)
                cur = self.model.Pd[d_id]
                self.model.Pd[d_id] = F.normalize(m * cur + (1 - m) * agg, dim=-1)


# ============================================================
# Client: 继承 feddsa_scheduled.Client, override train (替换 5 loss)
# ============================================================

class Client(_BaseClient):
    """BiProto Client: orth_only-style training + L_sty_proto + L_proto_excl."""

    def _get_biproto_weights(self):
        """Bell schedule for L_sty_proto + L_proto_excl + MSE coef.

        Returns dict with: w_sty_proto, w_proto_excl, w_mse.
        """
        t = self.current_round
        ws = self.warmup_start
        we = self.warmup_end
        rd = self.ramp_down
        R = self.num_rounds

        if t < ws:
            ramp = 0.0
        elif t < we:
            ramp = (t - ws) / max(1.0, (we - ws))
        elif t < rd:
            ramp = 1.0
        else:
            # ramp-down for L_sty_proto only
            ramp = max(0.0, 1.0 - (t - rd) / max(1.0, (R - rd)))

        # L_proto_excl 在 ramp-up 后保持恒定 (低维 proto 空间梯度稀疏, 不需要 ramp-down)
        if t < ws:
            ramp_excl = 0.0
        elif t < we:
            ramp_excl = (t - ws) / max(1.0, (we - ws))
        else:
            ramp_excl = 1.0

        return {
            'w_sty_proto': self.lambda_sty_proto * ramp,
            'w_proto_excl': self.lambda_proto_excl * ramp_excl,
            'w_mse': self.mse_coef if t >= ws else 0.0,
        }

    def _maybe_freeze_encoder_sem(self, model):
        """C0 matched-intervention gate: freeze encoder.* params (但 semantic_head/head 仍可训)."""
        if int(getattr(self, 'freeze_encoder_sem', 0)) == 1:
            for name, p in model.named_parameters():
                if name.startswith('encoder.'):
                    p.requires_grad_(False)

    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        # Apply C0 freeze if configured (BEFORE building optimizer)
        self._maybe_freeze_encoder_sem(model)

        # Build optimizer with only trainable params (works for both freeze/no-freeze modes)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )

        weights = self._get_biproto_weights()

        # Online accumulators for Pc/Pd EMA targets
        class_sum = collections.defaultdict(lambda: torch.zeros(model.proj_dim))
        class_count = collections.defaultdict(int)
        domain_sum = torch.zeros(model.proj_dim)
        domain_count = 0
        domain_id = int(self.id) if hasattr(self, 'id') else 0

        num_steps = self.num_steps
        for step in range(num_steps):
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            x, y = batch_data[0], batch_data[-1]

            model.zero_grad()

            # ---- Forward (with taps) ----
            pooled, taps = model.encode_with_taps(x)
            taps_detached = [t.detach() for t in taps]  # 切断 sty 侧梯度回流到 encoder

            z_sem = model.semantic_head(pooled)            # [B, 128] — has grad to encoder
            z_sem_n = F.normalize(z_sem, dim=-1)

            z_sty = model.encoder_sty(taps_detached)        # [B, 128]
            z_sty_n = F.normalize(z_sty, dim=-1)

            logits = model.head(z_sem)
            L_CE = F.cross_entropy(logits, y)

            # ---- L_orth (recall: 在 normalized 向量上) ----
            cos_se_st = (z_sem_n * z_sty_n).sum(dim=-1)
            L_orth = (cos_se_st ** 2).mean()

            # ---- L_sty_proto split into InfoNCE + MSE anchor (decoupled weights) ----
            # CRITICAL fix (R3 reviewer): InfoNCE 和 MSE 用独立权重相加, 不要 InfoNCE * w_sty_proto
            # 后再 + MSE * w_mse (因为 ramp_down 时 w_sty_proto=0 会让 MSE 失效)
            L_info = torch.tensor(0.0, device=z_sty.device)
            L_mse = torch.tensor(0.0, device=z_sty.device)
            if weights['w_sty_proto'] > 0 or weights['w_mse'] > 0:
                Pd = model.Pd.to(z_sty.device)  # [D, 128]
                if Pd.shape[0] > 1 and weights['w_sty_proto'] > 0:
                    # InfoNCE: each sample's domain = client's domain (single-client=single-domain)
                    sim = torch.matmul(z_sty_n, Pd.t()) / max(self.tau_proto, 1e-3)
                    target = torch.full((z_sty_n.size(0),), domain_id,
                                         device=z_sty.device, dtype=torch.long)
                    L_info = F.cross_entropy(sim, target)
                if weights['w_mse'] > 0:
                    # MSE anchor (FPL-style, stopgrad on Pd)
                    Pd_target = Pd[domain_id].detach().expand_as(z_sty_n)  # [B, 128]
                    L_mse = F.mse_loss(z_sty_n, Pd_target)

            # ---- L_proto_excl: present-classes-only, hybrid ST domain axis ----
            L_proto_excl = torch.tensor(0.0, device=z_sty.device)
            if weights['w_proto_excl'] > 0:
                # Present classes (count >= 2 in batch)
                unique_y, counts = torch.unique(y, return_counts=True)
                present_classes = unique_y[counts >= 2].tolist()

                if len(present_classes) > 0:
                    # class_axis[c] = mean(z_sem) over y==c, normalized (has grad)
                    class_axes = []
                    for c in present_classes:
                        mask = (y == c)
                        ca = z_sem[mask].mean(dim=0)  # [128]
                        class_axes.append(F.normalize(ca, dim=-1))
                    class_axes = torch.stack(class_axes, dim=0)  # [num_present, 128]

                    # Domain axis (single domain per client batch): hybrid ST
                    # 单 client = 单 domain → 整 batch 都属同一 domain → 直接 mean over batch
                    bc_d = z_sty.mean(dim=0)  # [128], has grad
                    bc_d = F.normalize(bc_d, dim=-1)
                    Pd_anchor = model.Pd[domain_id].detach().to(z_sty.device)
                    raw = Pd_anchor + bc_d - bc_d.detach()
                    domain_axis = F.normalize(raw, dim=-1).unsqueeze(0)  # [1, 128]

                    # cos² over (present_classes, [single domain])
                    cos_cd = (class_axes * domain_axis).sum(dim=-1)  # [num_present]
                    L_proto_excl = (cos_cd ** 2).mean()

            # ---- Total loss (InfoNCE + MSE 解耦权重, 修复 R3 reviewer C3 bug) ----
            loss = (
                L_CE
                + self.lambda_orth * L_orth
                + weights['w_sty_proto'] * L_info       # InfoNCE
                + weights['w_mse'] * L_mse              # MSE anchor (独立权重)
                + weights['w_proto_excl'] * L_proto_excl
            )

            loss.backward()
            optimizer.step()

            # ---- Online accumulators (no_grad, for Pc/Pd EMA upload) ----
            with torch.no_grad():
                z_sem_cpu = z_sem.detach().cpu()
                z_sty_cpu = z_sty.detach().cpu()
                for i, yi in enumerate(y.cpu().tolist()):
                    class_sum[int(yi)] = class_sum[int(yi)] + F.normalize(z_sem_cpu[i], dim=-1)
                    class_count[int(yi)] += 1
                domain_sum = domain_sum + F.normalize(z_sty_cpu, dim=-1).sum(dim=0)
                domain_count += z_sty_cpu.size(0)

        # ---- Build payload for server EMA aggregation ----
        class_means = {}
        for c_id, cnt in class_count.items():
            if cnt > 0:
                class_means[c_id] = (class_sum[c_id] / cnt, cnt)
        domain_mean = (domain_sum / max(domain_count, 1)) if domain_count > 0 else None

        self._biproto_payload = {
            'class_means': class_means,
            'domain_id': domain_id,
            'domain_mean': domain_mean,
            'domain_count': domain_count,
        }


# ============================================================
# init_global_module: 提供给 main.py / flgo 创建 model 实例
# ============================================================

_TASK_NUM_CLASSES = {
    'pacs': 7,
    'office_caltech10': 10,
    'domainnet': 10,
    'digit5': 10,
}
_TASK_NUM_CLIENTS = {
    'pacs': 4,
    'office_caltech10': 4,
    'domainnet': 6,
    'digit5': 5,
}


def _resolve_task_meta(task_name: str):
    name = task_name.lower()
    nc, nk = 7, 4  # defaults (PACS-style)
    for key, val in _TASK_NUM_CLASSES.items():
        if key in name:
            nc = val
            break
    for key, val in _TASK_NUM_CLIENTS.items():
        if key in name:
            nk = val
            break
    return nc, nk


def init_global_module(object):
    """Standard flgo entry point. Receives a Server (or Client) instance.

    Only initializes model on Server; Client deepcopies via the framework.
    """
    if 'Server' not in object.__class__.__name__:
        return
    task = os.path.basename(object.option['task'])
    num_classes, num_clients = _resolve_task_meta(task)
    object.model = FedDSABiProtoModel(
        num_classes=num_classes,
        num_clients=num_clients,
    ).to(object.device)
