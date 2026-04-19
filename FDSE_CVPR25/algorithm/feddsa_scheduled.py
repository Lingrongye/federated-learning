"""
FedDSA-Scheduled: Controlled auxiliary loss scheduling for FedDSA.

Based on feddsa.py with key changes:
  1. L_orth always at full weight from R0 (no ramp) — properly tests decoupling
  2. Configurable schedule for InfoNCE + style augmentation:
     - mode 0: orth_only (no aug, no InfoNCE — pure decouple test)
     - mode 1: bell-curve (InfoNCE peaks at bell_peak then decays)
     - mode 2: cutoff (InfoNCE active until cutoff_round, then off)
     - mode 3: always-on (original behavior but with orth from R0)
  3. Gradient conflict diagnostic logging (every grad_log_interval rounds)

algo_para order:
  lo=lambda_orth, lh=lambda_hsic, ls=lambda_sem, tau,
  sdn=style_dispatch_num, pd=proj_dim,
  sm=schedule_mode, bp=bell_peak, bw=bell_width, cr=cutoff_round,
  gli=grad_log_interval
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


# ============================================================
# Model: AlexNet backbone + dual-head (same as feddsa.py)
# ============================================================

class AlexNetEncoder(nn.Module):
    """Same AlexNet as config.py but without the final classifier fc3."""
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


class FedDSAModel(fuf.FModule):
    def __init__(self, num_classes=7, feat_dim=1024, proj_dim=128):
        super().__init__()
        self.encoder = AlexNetEncoder()
        self.semantic_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.style_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.head = nn.Linear(proj_dim, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        z_sem = self.semantic_head(h)
        return self.head(z_sem)

    def encode(self, x):
        return self.encoder(x)

    def get_semantic(self, h):
        return self.semantic_head(h)

    def get_style(self, h):
        return self.style_head(h)


# ============================================================
# Server
# ============================================================

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({
            'lo': 1.0,    # lambda_orth
            'lh': 0.0,    # lambda_hsic
            'ls': 1.0,    # lambda_sem (base weight, multiplied by schedule)
            'tau': 0.2,   # InfoNCE temperature
            'sdn': 5,     # style_dispatch_num
            'pd': 128,    # proj_dim
            'sm': 0,      # schedule_mode: 0=orth_only, 1=bell, 2=cutoff, 3=always_on,
                          #   4=mse_anchor, 5=alpha_sparsity, 6=mse+alpha, 7=detach_aug
            'bp': 60,     # bell_peak (round where bell peaks)
            'bw': 30,     # bell_width (gaussian sigma)
            'cr': 80,     # cutoff_round (for mode 2)
            'gli': 10,    # grad_log_interval (0=off)
            'lm': 1.0,    # lambda_mse (MSE anchor weight, modes 4/6)
            'al': 0.25,   # alpha_sparsity (power for cos_sim, modes 5/6)
            'se': 0,      # save_errors: 0=off, 1=save client checkpoints on last round
            'sas': 0,     # style-aware aggregation scope:
                          #   0 = off (baseline B0)
                          #   1 = sem_head only, classifier FedAvg (Plan A / B1 / EXP-084)
                          #   2 = sem_head + classifier, both style-conditioned (A2 / sas-FH, OURS)
                          #   3 = sem_head sas + classifier uniform-avg (C2 counterfactual)
                          #   4 = sem_head sas + classifier fully local (C1 counterfactual)
            'sas_tau': 0.3,  # softmax temperature for style similarity
            # -------- SCPR (Self-Masked Style-Weighted Multi-Positive InfoNCE) --------
            # NEW (2026-04-19): refined via GPT-5.4 x5-round refine session
            # (obsidian_exprtiment_results/refine_logs/2026-04-18_SCPR_v1/)
            'scpr': 0,    # SCPR mode:
                          #   0 = off (fall back to sm schedule / Plan A, no SCPR loss)
                          #   1 = uniform multi-positive (= M3 / SCPR tau->inf lower bound)
                          #   2 = style-weighted multi-positive (SCPR, OURS)
                          # Always self-masked (j != k) when mode > 0.
                          # Uses domain-indexed class prototypes; NO new trainable params.
            'scpr_tau': 0.3,  # SCPR style attention temperature (inherited from SAS optimal)
        })
        # Readable aliases
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
        self.alpha_sparsity = float(self.al)
        self.save_errors = int(self.se)
        self.style_aware_sem = int(self.sas)
        self.style_aware_tau = float(self.sas_tau)
        # SCPR aliases
        self.scpr_mode = int(self.scpr)
        self.scpr_tau_val = float(self.scpr_tau)
        self.sample_option = 'full'

        # 方案 A：per-client semantic_head states (最近一轮上传的)
        self.client_sem_states = {}
        # sas-FH (A2/C1/C2)：per-client classifier head states
        self.client_head_states = {}

        # Style bank (1024d pool5 μ/σ — used by SAS for parameter-space routing)
        self.style_bank = {}
        # SCPR-dedicated style bank (128d z_sty μ/σ — post-decouple, per FINAL_PROPOSAL)
        # Introduced 2026-04-19 after EXP-095 v1 was found to wrongly share SAS's 1024d bank
        self.scpr_style_bank = {}
        # Global semantic prototypes
        self.global_semantic_protos = {}
        # SCPR: per-(client, class) prototypes bank (domain-indexed)
        #   key: client_id -> value: dict {class_idx: proto_tensor (cpu)}
        self.client_class_protos = {}
        # Gradient conflict log
        self.grad_conflict_log = {}
        # Best checkpoint tracking (for save_errors)
        self._best_avg_acc = -1.0
        self._best_round = 0
        self._best_global_state = None
        self._best_client_states = None
        # Same-round FedAvg head snapshot at best round (for Claim 2 swap diagnostic)
        self._best_fedavg_head_state = None

        self._init_agg_keys()

        # Pass config to clients
        for c in self.clients:
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
            c.alpha_sparsity = self.alpha_sparsity
            # SCPR params to client
            c.scpr_mode = self.scpr_mode
            c.scpr_tau_val = self.scpr_tau_val

    def _init_agg_keys(self):
        """Classify model keys into: shared (FedAvg), private (style_head+BN)."""
        all_keys = list(self.model.state_dict().keys())
        self.private_keys = set()
        for k in all_keys:
            if 'style_head' in k:
                self.private_keys.add(k)
            elif 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
                self.private_keys.add(k)
        self.shared_keys = [k for k in all_keys if k not in self.private_keys]

    def pack(self, client_id, mtype=0):
        """Send global model + protos + styles.

        sas 聚合策略（分两个部分分别处理）：
          sem_head:
            - sas in {1,2,3,4}: style-conditioned 个性化 (同 Plan A)
            - sas == 0: FedAvg (不做个性化)
          classifier head:
            - sas == 2 (A2): style-conditioned 个性化（OURS, sas-FH）
            - sas == 3 (C2): uniform-avg 聚合（均匀加权，不用 style）
            - sas == 4 (C1): local-only（用 client 自己最近一轮上传的 head）
            - sas in {0,1}: FedAvg (Plan A 行为)
        """
        dispatched_styles = None
        if self.schedule_mode != 0 and len(self.style_bank) > 0:
            available = {cid: s for cid, s in self.style_bank.items() if cid != client_id}
            if len(available) == 0:
                available = self.style_bank
            n = min(self.style_dispatch_num, len(available))
            if n > 0:
                keys = list(available.keys())
                chosen = np.random.choice(keys, n, replace=False)
                dispatched_styles = [available[k] for k in chosen]

        model_to_send = copy.deepcopy(self.model)
        model_state = model_to_send.state_dict()
        overridden = False

        # --- 1) sem_head 部分 ---
        if self.style_aware_sem in (1, 2, 3, 4) \
                and len(self.client_sem_states) >= 2 \
                and client_id in self.style_bank:
            personalized_sem = self._compute_style_weighted(
                self.client_sem_states, client_id
            )
            if personalized_sem is not None:
                for k, v in personalized_sem.items():
                    if k in model_state:
                        model_state[k] = v.to(model_state[k].device)
                overridden = True

        # --- 2) classifier head 部分 ---
        if self.style_aware_sem == 2 \
                and len(self.client_head_states) >= 2 \
                and client_id in self.style_bank:
            # A2 (sas-FH): style-conditioned personalized head
            personalized_head = self._compute_style_weighted(
                self.client_head_states, client_id
            )
            if personalized_head is not None:
                for k, v in personalized_head.items():
                    if k in model_state:
                        model_state[k] = v.to(model_state[k].device)
                overridden = True
        elif self.style_aware_sem == 3 and len(self.client_head_states) >= 2:
            # C2 counterfactual: uniform-avg head (每个 client 收到同一份均值 head)
            uniform_head = self._compute_uniform_avg(self.client_head_states)
            if uniform_head is not None:
                for k, v in uniform_head.items():
                    if k in model_state:
                        model_state[k] = v.to(model_state[k].device)
                overridden = True
        elif self.style_aware_sem == 4 and client_id in self.client_head_states:
            # C1 counterfactual: local-only head (client 拿自己上轮上传的 head)
            local_head = self.client_head_states[client_id]
            for k, v in local_head.items():
                if k in model_state:
                    model_state[k] = v.to(model_state[k].device)
            overridden = True

        if overridden:
            model_to_send.load_state_dict(model_state)

        # SCPR: compute self-masked style-weighted (or uniform) prototype payload
        # Returns None when scpr=0 OR bank insufficient; client falls back to non-SCPR path.
        scpr_payload = None
        if self.scpr_mode > 0 and len(self.client_class_protos) >= 2:
            scpr_payload = self._compute_scpr_payload(client_id)

        return {
            'model': model_to_send,
            'global_protos': copy.deepcopy(self.global_semantic_protos),
            'style_bank': dispatched_styles,
            'current_round': self.current_round,
            'scpr_payload': scpr_payload,
        }

    @staticmethod
    def _extract_style_vec(style):
        """从 style_bank 的 entry (tuple/dict/tensor) 提取展平的 style 向量。"""
        if isinstance(style, (tuple, list)):
            return style[0].flatten().cpu()
        elif isinstance(style, dict):
            v = style.get('mu', next(iter(style.values())))
            return v.flatten().cpu()
        else:
            return style.flatten().cpu()

    def _compute_style_weighted(self, client_states, target_cid):
        """通用版本：对 target client 计算其他 client 的 style-similarity 加权 state。

        Args:
            client_states: dict[cid -> state_dict]，比如 self.client_sem_states 或
                           self.client_head_states。所有 value 必须有相同的 key 集合。
            target_cid: 目标 client id。
        Returns:
            aggregated state dict (同 client_states[cid] 的结构)，或 None（无法聚合时）。
        """
        import torch as _t
        target_style = self.style_bank.get(target_cid)
        if target_style is None:
            return None
        target_vec = self._extract_style_vec(target_style)

        sims = []
        cids = []
        for cid in client_states:
            src_style = self.style_bank.get(cid)
            if src_style is None:
                continue
            src_vec = self._extract_style_vec(src_style)
            if src_vec.shape != target_vec.shape:
                continue
            dot = (src_vec * target_vec).sum()
            norm = src_vec.norm() * target_vec.norm() + 1e-8
            sims.append((dot / norm).item())
            cids.append(cid)

        if len(sims) == 0:
            return None

        sims_t = _t.tensor(sims, dtype=_t.float32) / max(self.style_aware_tau, 1e-3)
        weights = _t.softmax(sims_t, dim=0)

        accumulated = None
        for w, cid in zip(weights.tolist(), cids):
            st = client_states[cid]
            if accumulated is None:
                accumulated = {k: v.clone() * w for k, v in st.items()}
            else:
                for k in accumulated:
                    accumulated[k] += st[k] * w
        return accumulated

    def _compute_uniform_avg(self, client_states):
        """C2 counterfactual: 对所有 client state 做均匀平均（uniform mean），
        不使用 style 信息。每个 target client 都会收到同一份结果。

        Args:
            client_states: dict[cid -> state_dict]
        Returns:
            averaged state dict，或 None（空输入时）。
        """
        if len(client_states) == 0:
            return None
        cids = list(client_states.keys())
        n = float(len(cids))
        accumulated = None
        for cid in cids:
            st = client_states[cid]
            if accumulated is None:
                accumulated = {k: v.clone() / n for k, v in st.items()}
            else:
                for k in accumulated:
                    accumulated[k] += st[k] / n
        return accumulated

    def _compute_scpr_payload(self, target_cid):
        """Compute SCPR payload for target client: self-masked style-weighted
        (or uniform) prototype dispatch.

        scpr_mode=1 (uniform multi-positive, = M3 lower bound) — STYLE-FREE:
            w_{k->j}^{uniform} = 1 / (K-1) for all j != k that have proto(s)
            Does NOT require style bank (this is the whole point of M3 lower bound).
        scpr_mode=2 (style-weighted, OURS) — requires style bank:
            w_{k->j} = softmax_j( cos(s_k, s_j) / tau_SCPR ) * 1[j != k]

        Returns:
            None if insufficient bank. Otherwise dict:
              'weights': {source_cid: float}  (self-masked, sums to 1 over source_cids)
              'protos':  {source_cid: {class_idx: proto_tensor_cpu}}
        """
        import torch as _t

        # --- Stage 1: self-masked prototype source collection (style-independent) ---
        source_cids_raw = []
        source_protos_raw = {}
        for cid, protos in self.client_class_protos.items():
            if cid == target_cid:  # SELF-MASK (R1 reviewer fix)
                continue
            if protos is None or len(protos) == 0:
                continue
            source_cids_raw.append(cid)
            source_protos_raw[cid] = protos

        if len(source_cids_raw) == 0:
            return None

        # --- Stage 2: weight assignment ---
        if self.scpr_mode == 1:
            # Uniform (M3 lower bound; SCPR tau -> infinity). STYLE-FREE by design.
            # codex-fix #1: do NOT depend on style_bank here.
            uniform_w = 1.0 / len(source_cids_raw)
            weights = {cid: uniform_w for cid in source_cids_raw}
            return {'weights': weights, 'protos': source_protos_raw}

        if self.scpr_mode == 2:
            # Style-weighted softmax (SCPR main method).
            # Uses the SCPR-dedicated 128d z_sty bank (post-decouple), NOT the SAS 1024d h bank.
            # This aligns with FINAL_PROPOSAL: s_k := normalize(E_{x ∈ D_k}[z_sty(x)]).
            bank = self.scpr_style_bank if self.scpr_style_bank else self.style_bank
            target_style = bank.get(target_cid)
            if target_style is None:
                return None
            target_vec = self._extract_style_vec(target_style)

            source_cids = []
            source_vecs = []
            source_protos = {}
            for cid in source_cids_raw:
                src_style = bank.get(cid)
                if src_style is None:
                    continue
                src_vec = self._extract_style_vec(src_style)
                if src_vec.shape != target_vec.shape:
                    continue
                source_cids.append(cid)
                source_vecs.append(src_vec)
                source_protos[cid] = source_protos_raw[cid]

            if len(source_cids) == 0:
                return None

            sims = []
            for src_vec in source_vecs:
                dot = (src_vec * target_vec).sum()
                norm = src_vec.norm() * target_vec.norm() + 1e-8
                sims.append((dot / norm).item())
            sims_t = _t.tensor(sims, dtype=_t.float32) / max(self.scpr_tau_val, 1e-3)
            w_vec = _t.softmax(sims_t, dim=0).tolist()
            weights = {cid: float(w) for cid, w in zip(source_cids, w_vec)}
            return {'weights': weights, 'protos': source_protos}

        return None

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)

        models = res['model']
        protos_list = res['protos']
        proto_counts_list = res['proto_counts']
        style_stats_list = res['style_stats']
        grad_conflict_list = res['grad_conflict']

        # Aggregate shared parameters (FedAvg)
        self._aggregate_shared(models)

        # sas：记录每个 client 上传的 sem_head / head state（供 pack 时做聚合）
        if self.style_aware_sem in (1, 2, 3, 4):
            for cid, m in zip(self.received_clients, models):
                m_state = m.state_dict()
                sem_state = {k: v.clone().cpu() for k, v in m_state.items()
                             if 'semantic_head' in k}
                self.client_sem_states[cid] = sem_state
                # sas-FH (A2/C1/C2) 需要额外记录 classifier head
                if self.style_aware_sem in (2, 3, 4):
                    head_state = {k: v.clone().cpu() for k, v in m_state.items()
                                  if k.startswith('head.')}
                    if head_state:
                        self.client_head_states[cid] = head_state

        # Collect style bank (1024d pool5 — for SAS)
        for cid, style in zip(self.received_clients, style_stats_list):
            if style is not None:
                self.style_bank[cid] = style

        # SCPR: update per-(client, class) prototype bank (domain-indexed)
        # and per-client z_sty-based style bank (128d post-decouple, for SCPR attention)
        if self.scpr_mode > 0:
            style_stats_scpr_list = res.get('style_stats_scpr', [None] * len(self.received_clients))
            for cid, protos, sty_scpr in zip(self.received_clients, protos_list, style_stats_scpr_list):
                if protos is not None and len(protos) > 0:
                    self.client_class_protos[cid] = {
                        c: p.detach().clone().cpu() for c, p in protos.items()
                    }
                if sty_scpr is not None:
                    self.scpr_style_bank[cid] = sty_scpr

        # Aggregate global protos (weighted mean)
        self._aggregate_protos(protos_list, proto_counts_list)

        # Log gradient conflict
        for cid, gc in zip(self.received_clients, grad_conflict_list):
            if gc is not None:
                if self.current_round not in self.grad_conflict_log:
                    self.grad_conflict_log[self.current_round] = {}
                self.grad_conflict_log[self.current_round][cid] = gc
        if self.current_round in self.grad_conflict_log:
            entries = self.grad_conflict_log[self.current_round]
            vals = list(entries.values())
            mean_cos = sum(vals) / len(vals)
            self.gv.logger.info(
                f"[GradConflict] round={self.current_round} "
                f"mean_cos={mean_cos:.4f} per_client={entries}"
            )

        # 追踪 best（每轮）+ 最后一轮保存 best checkpoint
        if int(getattr(self, 'save_errors', 0)) == 1:
            self._track_best()
            if self.current_round >= self.num_rounds:
                self._save_best_checkpoints()

    def _track_best(self):
        """每轮聚合后缓存 best AVG acc 对应的模型快照。
        读取 flgo logger.output 拿最新 test metrics（上一轮的结果）。"""
        try:
            output = getattr(self.gv.logger, 'output', None)
            if not output or 'mean_local_test_accuracy' not in output:
                return
            history = output['mean_local_test_accuracy']
            if not history:
                return
            current = float(history[-1]) * 100
            if current > self._best_avg_acc:
                import copy
                self._best_avg_acc = current
                self._best_round = len(history)  # round index (1-based)
                self._best_global_state = copy.deepcopy(self.model.state_dict())
                self._best_client_states = [
                    copy.deepcopy(c.model.state_dict()) if getattr(c, 'model', None) is not None else None
                    for c in self.clients
                ]
                # sas-FH Claim 2：保存同轮 FedAvg head 快照（供 swap diagnostic 用）
                # 仅 A2 模式下有意义：此时 server.model 里 head 已是 FedAvg 版本，
                # 但下发给 client 的 head 是 style-conditioned personalized，
                # 两者有差异 → 需要保留这份 "same-round global_head"。
                self._best_fedavg_head_state = {
                    k: v.clone().cpu() for k, v in self.model.state_dict().items()
                    if k.startswith('head.')
                }
        except Exception as e:
            try:
                self.gv.logger.info(f"[TrackBest] skip: {e}")
            except Exception:
                pass

    def _save_best_checkpoints(self):
        """训练结束时把 best round 的模型快照写到 ~/fl_checkpoints/"""
        import os, torch, time, json
        if self._best_global_state is None:
            # 兜底：best 未追踪到（比如 test 没跑过），就存最后一轮
            self._best_global_state = self.model.state_dict()
            self._best_client_states = [
                c.model.state_dict() if getattr(c, 'model', None) is not None else None
                for c in self.clients
            ]
            self._best_round = self.current_round
            # 同时保一份最后一轮的 FedAvg head 作为 fallback snapshot
            if self._best_fedavg_head_state is None:
                self._best_fedavg_head_state = {
                    k: v.clone().cpu() for k, v in self.model.state_dict().items()
                    if k.startswith('head.')
                }
            fallback = True
        else:
            fallback = False
        seed = getattr(self, 'option', {}).get('seed', 'unknown')
        tag = f"feddsa_s{seed}_R{self.num_rounds}_best{self._best_round}_{int(time.time())}"
        save_dir = os.path.expanduser(f'~/fl_checkpoints/{tag}')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self._best_global_state, os.path.join(save_dir, 'global_model.pt'))
        saved = 0
        for cid, state in enumerate(self._best_client_states):
            if state is not None:
                torch.save(state, os.path.join(save_dir, f'client_{cid}.pt'))
                saved += 1
        # 同轮 FedAvg head 快照（Claim 2 swap diagnostic）
        if self._best_fedavg_head_state is not None:
            torch.save(self._best_fedavg_head_state,
                       os.path.join(save_dir, 'fedavg_head.pt'))
        meta = {
            'best_round': self._best_round,
            'best_avg_acc': self._best_avg_acc,
            'num_rounds': self.num_rounds,
            'num_clients': len(self.clients),
            'schedule_mode': int(self.schedule_mode),
            'lambda_orth': float(self.lambda_orth),
            'tau': float(self.tau),
            'sas': int(self.style_aware_sem),
            'sas_tau': float(self.style_aware_tau),
            'seed': seed,
            'has_fedavg_head': self._best_fedavg_head_state is not None,
            'is_last_round_fallback': fallback,
        }
        with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        self.gv.logger.info(
            f"[SaveBest] best@R{self._best_round} avg={self._best_avg_acc:.4f} "
            f"global+{saved} clients -> {save_dir} (fallback={fallback})"
        )

    def _aggregate_shared(self, models):
        if len(models) == 0:
            return
        weights = np.array(
            [len(self.clients[cid].train_data) for cid in self.received_clients],
            dtype=float,
        )
        weights /= weights.sum()

        global_dict = self.model.state_dict()
        model_dicts = [m.state_dict() for m in models]

        for key in self.shared_keys:
            global_dict[key] = sum(w * md[key] for w, md in zip(weights, model_dicts))
        self.model.load_state_dict(global_dict)

    def _aggregate_protos(self, protos_list, proto_counts_list):
        agg_sum = {}
        agg_cnt = {}
        for protos, counts in zip(protos_list, proto_counts_list):
            if protos is None:
                continue
            for c, p in protos.items():
                n = counts.get(c, 1)
                if c not in agg_sum:
                    agg_sum[c] = p * n
                    agg_cnt[c] = n
                else:
                    agg_sum[c] += p * n
                    agg_cnt[c] += n
        self.global_semantic_protos = {
            c: agg_sum[c] / agg_cnt[c] for c in agg_sum
        }


# ============================================================
# Client
# ============================================================

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.global_protos = None
        self.local_style_bank = None
        self.current_round = 0
        self._grad_conflict_log = None
        # SCPR state (populated by Server.initialize via `c.scpr_mode=...`)
        self.scpr_mode = 0
        self.scpr_tau_val = 0.3
        self.scpr_payload = None
        self._local_style_stats_scpr = None  # 128d z_sty μ/σ, populated in train()

    def reply(self, svr_pkg):
        model, global_protos, style_bank, current_round = self.unpack(svr_pkg)
        self.current_round = current_round
        self.global_protos = global_protos
        self.local_style_bank = style_bank
        # SCPR: extract per-round payload (self-masked weights + domain protos)
        self.scpr_payload = svr_pkg.get('scpr_payload', None)
        self.train(model)
        return self.pack()

    def unpack(self, svr_pkg):
        global_model = svr_pkg['model']
        if self.model is None:
            self.model = global_model
        else:
            local_dict = self.model.state_dict()
            global_dict = global_model.state_dict()
            for key in global_dict:
                if 'style_head' not in key and not (
                    'bn' in key.lower() and ('running_' in key or 'num_batches_tracked' in key)
                ):
                    local_dict[key] = global_dict[key]
            self.model.load_state_dict(local_dict)
        return (
            self.model,
            svr_pkg['global_protos'],
            svr_pkg['style_bank'],
            svr_pkg['current_round'],
        )

    def pack(self):
        return {
            'model': copy.deepcopy(self.model.to('cpu')),
            'protos': self._local_protos,
            'proto_counts': self._local_proto_counts,
            'style_stats': self._local_style_stats,            # 1024d pool5 (SAS interface)
            'style_stats_scpr': getattr(self, '_local_style_stats_scpr', None),  # 128d z_sty (SCPR)
            'grad_conflict': self._grad_conflict_log,
        }

    # ----------------------------------------------------------------
    # Schedule functions
    # ----------------------------------------------------------------

    @staticmethod
    def _bell_weight(t, t_peak, width):
        """Gaussian bell: peaks at t_peak, decays both sides."""
        if width <= 0:
            return 1.0 if t == t_peak else 0.0
        return math.exp(-0.5 * ((t - t_peak) / width) ** 2)

    def _get_aux_weight(self):
        """Compute the auxiliary loss weight based on schedule_mode.

        Returns:
            w_aux: weight for InfoNCE and style augmentation (0 to 1).
                   L_orth is ALWAYS full weight (not affected by this).
        """
        t = self.current_round
        mode = self.schedule_mode

        if mode == 0:
            # orth_only: no InfoNCE, no augmentation
            return 0.0
        elif mode == 1:
            # bell-curve: peaks at bell_peak, decays with bell_width
            return self._bell_weight(t, self.bell_peak, self.bell_width)
        elif mode == 2:
            # cutoff: full weight until cutoff_round, then off
            return 1.0 if t <= self.cutoff_round else 0.0
        elif mode in (3, 4, 5, 6, 7):
            # always-on variants: full aux weight, loss type varies in train()
            return 1.0
        else:
            return 0.0

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------

    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum
        )

        # Compute schedule weight for this round
        w_aux = self._get_aux_weight()

        # Gradient conflict logging
        should_log_grad = (
            self.grad_log_interval > 0
            and self.current_round % self.grad_log_interval == 0
        )
        self._grad_conflict_log = None

        # Online accumulators
        proto_sum = {}
        proto_count = {}
        style_sum = None          # 1024d pool5 mean (SAS)
        style_sq_sum = None
        style_n = 0
        # SCPR dedicated: 128d z_sty mean (post-decouple, per FINAL_PROPOSAL)
        style_sum_scpr = None
        style_sq_sum_scpr = None
        style_n_scpr = 0

        num_steps = self.num_steps
        steps_per_epoch = max(1, len(self.train_data) // self.batch_size)

        for step in range(num_steps):
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            x, y = batch_data[0], batch_data[-1]

            model.zero_grad()

            # Forward
            h = model.encode(x)
            z_sem = model.get_semantic(h)
            z_sty = model.get_style(h)

            # Loss 1: Task CE (always active)
            output = model.head(z_sem)
            loss_task = self.loss_fn(output, y)

            # Loss 2: Style-augmented CE (controlled by w_aux)
            # Mode 7: augmented features go to contrastive ONLY, not CE (PARDON insight)
            loss_aug = torch.tensor(0.0, device=x.device)
            z_sem_aug = None
            has_aug = (w_aux > 1e-4
                       and self.local_style_bank is not None
                       and len(self.local_style_bank) > 0)
            if has_aug:
                h_aug = self._style_augment(h)
                z_sem_aug = model.get_semantic(h_aug)
                if self.schedule_mode != 7:
                    # Normal modes: augmented features go through CE
                    output_aug = model.head(z_sem_aug)
                    loss_aug = self.loss_fn(output_aug, y)
                # Mode 7: z_sem_aug computed but NOT sent through CE

            # Loss 3: Decoupling — ALWAYS full weight from R0
            loss_orth, loss_hsic = self._decouple_loss(z_sem, z_sty)

            # Loss 4: Alignment (controlled by w_aux, type depends on mode)
            loss_sem = torch.tensor(0.0, device=x.device)
            loss_mse_anchor = torch.tensor(0.0, device=x.device)
            use_alpha = self.schedule_mode in (5, 6, 7)
            use_mse = self.schedule_mode in (4, 6, 7)
            # codex-fix #2: when SCPR is active, it is the sole alignment loss.
            # Disable the legacy global-mean InfoNCE/alpha/MSE block to preserve
            # the "scpr=1 strictly reduces to M3" mathematical contract.
            legacy_align_active = (
                self.scpr_mode == 0
                and w_aux > 1e-4
                and self.global_protos
                and len(self.global_protos) >= 2
            )
            if legacy_align_active:
                # Choose InfoNCE variant
                # For mode 7: use z_sem + z_sem_aug concatenated for richer contrastive
                z_for_contrast = z_sem
                y_for_contrast = y
                if self.schedule_mode == 7 and z_sem_aug is not None:
                    z_for_contrast = torch.cat([z_sem, z_sem_aug], dim=0)
                    y_for_contrast = torch.cat([y, y], dim=0)

                if use_alpha:
                    loss_sem = self._infonce_alpha_loss(z_for_contrast, y_for_contrast)
                else:
                    loss_sem = self._infonce_loss(z_for_contrast, y_for_contrast)

                if use_mse:
                    loss_mse_anchor = self._mse_anchor_loss(z_sem, y)

            # Total loss:
            # L_orth: ALWAYS full weight (key change from original)
            # L_aug + L_InfoNCE + L_MSE: controlled by w_aux schedule
            loss = (
                loss_task
                + w_aux * loss_aug
                + self.lambda_orth * loss_orth
                + self.lambda_hsic * loss_hsic
                + w_aux * self.lambda_sem * loss_sem
                + w_aux * self.lambda_mse * loss_mse_anchor
            )

            # SCPR loss: self-masked (style-)weighted multi-positive InfoNCE.
            # Independent from sm/w_aux schedule (works even when Plan A sm=0).
            # scpr_mode=0 → skip entirely; =1 → uniform multi-pos (M3 bound); =2 → style-weighted.
            if self.scpr_mode > 0 and self.scpr_payload is not None:
                loss_scpr = self._scpr_loss(z_sem, y)
                loss = loss + self.lambda_sem * loss_scpr

            # Gradient conflict logging (last batch of last epoch)
            is_last_batch = (step == num_steps - 1)
            if should_log_grad and is_last_batch and loss_sem.item() > 0:
                self._log_grad_conflict(
                    model, loss_task, w_aux * self.lambda_sem * loss_sem
                )

            loss.backward()
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
            optimizer.step()

            # Accumulate prototypes + style stats (last epoch only)
            if step >= num_steps - steps_per_epoch - 1:
                with torch.no_grad():
                    z_det = z_sem.detach().cpu()
                    h_det = h.detach()
                    z_sty_det = z_sty.detach()      # 128d post-decouple (for SCPR)

                    for i, label in enumerate(y):
                        c = label.item()
                        if c not in proto_sum:
                            proto_sum[c] = z_det[i].clone()
                            proto_count[c] = 1
                        else:
                            proto_sum[c] += z_det[i]
                            proto_count[c] += 1

                    b = h_det.size(0)
                    # SAS style stats: 1024d pool5 (pre-decouple)
                    batch_mu = h_det.mean(dim=0).cpu()
                    batch_sq = (h_det ** 2).mean(dim=0).cpu()
                    if style_sum is None:
                        style_sum = batch_mu * b
                        style_sq_sum = batch_sq * b
                        style_n = b
                    else:
                        style_sum += batch_mu * b
                        style_sq_sum += batch_sq * b
                        style_n += b
                    # SCPR style stats: 128d z_sty (post-decouple per FINAL_PROPOSAL)
                    batch_mu_scpr = z_sty_det.mean(dim=0).cpu()
                    batch_sq_scpr = (z_sty_det ** 2).mean(dim=0).cpu()
                    if style_sum_scpr is None:
                        style_sum_scpr = batch_mu_scpr * b
                        style_sq_sum_scpr = batch_sq_scpr * b
                        style_n_scpr = b
                    else:
                        style_sum_scpr += batch_mu_scpr * b
                        style_sq_sum_scpr += batch_sq_scpr * b
                        style_n_scpr += b

        # Store for pack()
        self._local_protos = {c: proto_sum[c] / proto_count[c] for c in proto_sum}
        self._local_proto_counts = proto_count

        if style_n > 1:
            mu = style_sum / style_n
            var = style_sq_sum / style_n - mu ** 2
            self._local_style_stats = (mu, var.clamp(min=1e-6).sqrt())
        else:
            self._local_style_stats = None

        # SCPR style stats (128d z_sty post-decouple)
        if style_n_scpr > 1:
            mu_s = style_sum_scpr / style_n_scpr
            var_s = style_sq_sum_scpr / style_n_scpr - mu_s ** 2
            self._local_style_stats_scpr = (mu_s, var_s.clamp(min=1e-6).sqrt())
        else:
            self._local_style_stats_scpr = None

    # ----------------------------------------------------------------
    # Gradient Conflict Logger
    # ----------------------------------------------------------------

    def _log_grad_conflict(self, model, loss_task, loss_align_weighted):
        """Compute cosine similarity between CE and InfoNCE gradients on encoder."""
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        if not encoder_params:
            return
        try:
            grad_ce = torch.autograd.grad(
                loss_task, encoder_params,
                retain_graph=True, allow_unused=True, create_graph=False
            )
            grad_align = torch.autograd.grad(
                loss_align_weighted, encoder_params,
                retain_graph=True, allow_unused=True, create_graph=False
            )
        except RuntimeError:
            return

        flat_ce = []
        flat_align = []
        for g_ce, g_al in zip(grad_ce, grad_align):
            if g_ce is not None and g_al is not None:
                flat_ce.append(g_ce.flatten())
                flat_align.append(g_al.flatten())
        if not flat_ce:
            return

        flat_ce = torch.cat(flat_ce)
        flat_align = torch.cat(flat_align)
        cos_sim = F.cosine_similarity(flat_ce.unsqueeze(0), flat_align.unsqueeze(0)).item()
        self._grad_conflict_log = cos_sim

    # ----------------------------------------------------------------
    # Helper functions (same as feddsa.py)
    # ----------------------------------------------------------------

    def _style_augment(self, h):
        """AdaIN-style augmentation using external style."""
        idx = np.random.randint(0, len(self.local_style_bank))
        mu_ext, sigma_ext = self.local_style_bank[idx]
        mu_ext = mu_ext.to(h.device)
        sigma_ext = sigma_ext.to(h.device)

        mu_local = h.mean(dim=0)
        sigma_local = h.std(dim=0, unbiased=False).clamp(min=1e-6)

        alpha = np.random.beta(0.1, 0.1)
        mu_mix = alpha * mu_local + (1 - alpha) * mu_ext
        sigma_mix = alpha * sigma_local + (1 - alpha) * sigma_ext

        h_norm = (h - mu_local) / sigma_local
        return h_norm * sigma_mix + mu_mix

    def _decouple_loss(self, z_sem, z_sty):
        """Orthogonal + HSIC dual constraint."""
        z_sem_n = F.normalize(z_sem, dim=1)
        z_sty_n = F.normalize(z_sty, dim=1)
        cos = (z_sem_n * z_sty_n).sum(dim=1)
        loss_orth = (cos ** 2).mean()

        loss_hsic = self._hsic(z_sem, z_sty)
        return loss_orth, loss_hsic

    def _hsic(self, x, y):
        n = x.size(0)
        if n < 4:
            return torch.tensor(0.0, device=x.device)
        Kx = self._gaussian_kernel(x)
        Ky = self._gaussian_kernel(y)
        H = torch.eye(n, device=x.device) - torch.ones(n, n, device=x.device) / n
        return torch.trace(Kx @ H @ Ky @ H) / (n * n)

    def _gaussian_kernel(self, x):
        n = x.size(0)
        dist = torch.cdist(x, x, p=2) ** 2
        nonzero = dist[dist > 0]
        if nonzero.numel() == 0:
            return torch.ones(n, n, device=x.device)
        median = torch.median(nonzero)
        bw = median / (2.0 * np.log(n + 1) + 1e-6)
        return torch.exp(-dist / (2.0 * bw.clamp(min=1e-6)))

    def _infonce_loss(self, z_sem, y):
        """InfoNCE: pull toward same-class global proto, push away others."""
        available = sorted([c for c, p in self.global_protos.items() if p is not None])
        if len(available) < 2:
            return torch.tensor(0.0, device=z_sem.device)

        proto_matrix = torch.stack([self.global_protos[c].to(z_sem.device) for c in available])
        class_to_idx = {c: i for i, c in enumerate(available)}

        z_n = F.normalize(z_sem, dim=1)
        p_n = F.normalize(proto_matrix, dim=1)
        logits = (z_n @ p_n.T) / self.tau

        targets = []
        valid = []
        for i in range(y.size(0)):
            label = y[i].item()
            if label in class_to_idx:
                targets.append(class_to_idx[label])
                valid.append(i)

        if len(valid) == 0:
            return torch.tensor(0.0, device=z_sem.device)

        valid_t = torch.tensor(valid, device=z_sem.device)
        targets_t = torch.tensor(targets, device=z_sem.device, dtype=torch.long)
        return F.cross_entropy(logits[valid_t], targets_t)

    def _infonce_alpha_loss(self, z_sem, y):
        """Alpha-sparsity InfoNCE (FedPLVM-inspired).
        cos_sim^alpha weakens positive-pair gradients, focuses on hard negatives.
        Requires cos_sim >= 0, so we clamp after normalization."""
        available = sorted([c for c, p in self.global_protos.items() if p is not None])
        if len(available) < 2:
            return torch.tensor(0.0, device=z_sem.device)

        proto_matrix = torch.stack([self.global_protos[c].to(z_sem.device) for c in available])
        class_to_idx = {c: i for i, c in enumerate(available)}

        z_n = F.normalize(z_sem, dim=1)
        p_n = F.normalize(proto_matrix, dim=1)
        cos_sim = (z_n @ p_n.T).clamp(min=0)  # clamp negatives to 0 for pow safety
        logits = cos_sim.pow(self.alpha_sparsity) / self.tau

        targets = []
        valid = []
        for i in range(y.size(0)):
            label = y[i].item()
            if label in class_to_idx:
                targets.append(class_to_idx[label])
                valid.append(i)

        if len(valid) == 0:
            return torch.tensor(0.0, device=z_sem.device)

        valid_t = torch.tensor(valid, device=z_sem.device)
        targets_t = torch.tensor(targets, device=z_sem.device, dtype=torch.long)
        return F.cross_entropy(logits[valid_t], targets_t)

    def _mse_anchor_loss(self, z_sem, y):
        """MSE anchor loss (FPL-inspired): pull z_sem toward own-class global proto.
        Acts as a stable 'gravitational center' preventing feature drift."""
        targets = []
        valid = []
        for i in range(y.size(0)):
            label = y[i].item()
            if label in self.global_protos and self.global_protos[label] is not None:
                targets.append(self.global_protos[label])
                valid.append(i)
        if not valid:
            return torch.tensor(0.0, device=z_sem.device)
        target_matrix = torch.stack(targets).to(z_sem.device).detach()
        valid_t = torch.tensor(valid, device=z_sem.device)
        return F.mse_loss(z_sem[valid_t], target_matrix)

    # ----------------------------------------------------------------
    # SCPR: Self-Masked Style-Weighted Multi-Positive InfoNCE
    # ----------------------------------------------------------------

    def _scpr_loss(self, z_sem, y):
        """Self-masked (style-)weighted multi-positive SupCon-style InfoNCE.

        For client k (self-masked by Server: j != k), given the domain-indexed
        prototype bank {p_c^j} (all j != k, all classes c) and client-pair
        weights w_{k->j}, the loss per sample (z_i, y_i=c) is the weighted
        SupCon multi-positive formulation WITH POSITIVES IN THE DENOMINATOR
        (standard SupCon, Khosla et al. 2020):

            L_i = - sum_{j in A_i^c} (w_{k->j} / Z_i^c) * log_prob(j; i)
            log_prob(j; i) = logits[i, j] - logsumexp_{a in ALL_entries} logits[i, a]
            logits[i, a] = sim(z_i, p_{c_a}^{j_a}) / tau_nce

        where:
          A_i^c = {j != k : p_c^j exists}    (per-class available clients)
          Z_i^c = sum_{j in A_i^c} w_{k->j}  (per-sample renormalization)
          ALL_entries = all (class, client) pairs in the bank passed in

        Under scpr_mode=1 (uniform weights) this is exactly domain-aware
        multi-positive SupCon (= M3 lower bound). Under scpr_mode=2
        (style-weighted softmax) it is the full SCPR method.

        All prototypes are detached (soft anchors, no grad back to bank).
        """
        if self.scpr_payload is None:
            return torch.tensor(0.0, device=z_sem.device)
        weights = self.scpr_payload.get('weights', {})
        domain_protos = self.scpr_payload.get('protos', {})
        if not weights or not domain_protos:
            return torch.tensor(0.0, device=z_sem.device)

        device = z_sem.device

        # Flatten into per-entry arrays: each entry = (class, source_cid, proto, weight)
        entry_protos = []
        entry_classes = []
        entry_weights = []
        for cid, protos in domain_protos.items():
            w = float(weights.get(cid, 0.0))
            for c, proto in protos.items():
                entry_protos.append(proto)
                entry_classes.append(int(c))
                entry_weights.append(w)

        if len(entry_protos) < 2:
            return torch.tensor(0.0, device=device)

        proto_matrix = torch.stack([p.to(device) for p in entry_protos]).detach()  # [N, D]
        entry_classes_t = torch.tensor(entry_classes, device=device, dtype=torch.long)  # [N]
        entry_weights_t = torch.tensor(entry_weights, device=device, dtype=torch.float32)  # [N]

        # Cosine-similarity logits with InfoNCE temperature
        z_n = F.normalize(z_sem, dim=1)
        p_n = F.normalize(proto_matrix, dim=1)
        logits = (z_n @ p_n.T) / self.tau  # [B, N]
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # [B, N]

        # Positive mask: entry.class == y[i]  -> [B, N]
        y_col = y.unsqueeze(1)  # [B, 1]
        pos_mask = (entry_classes_t.unsqueeze(0) == y_col).float()  # [B, N]

        # Per-sample renormalize positive weights over A_i^c (clients that have class c)
        pos_w = pos_mask * entry_weights_t.unsqueeze(0)  # [B, N]
        pos_w_sum = pos_w.sum(dim=1, keepdim=True)  # [B, 1]
        pos_w_norm = pos_w / pos_w_sum.clamp(min=1e-8)  # [B, N]

        # Weighted NLL; skip samples with no positive (A_i^c empty)
        loss_per_sample = -(pos_w_norm * log_prob).sum(dim=1)  # [B]
        has_pos = (pos_w_sum.squeeze(1) > 1e-8).float()  # [B]
        denom = has_pos.sum().clamp(min=1.0)
        return (loss_per_sample * has_pos).sum() / denom


# ============================================================
# Model initialization
# ============================================================

model_map = {
    'PACS': lambda: FedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128),
    'office': lambda: FedDSAModel(num_classes=10, feat_dim=1024, proj_dim=128),
    'domainnet': lambda: FedDSAModel(num_classes=10, feat_dim=1024, proj_dim=128),
}


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    if 'Server' not in object.__class__.__name__:
        return
    task = os.path.basename(object.option['task'])
    for prefix, factory in model_map.items():
        if prefix.lower() in task.lower():
            object.model = factory().to(object.device)
            return
    object.model = FedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128).to(object.device)
