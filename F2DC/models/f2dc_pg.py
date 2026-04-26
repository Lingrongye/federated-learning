"""
F2DC-PG (Prototype-Guided F2DC) v3.2 trainer
============================================
基于 F2DC._train_net 改造, 加 prototype 维护 + cross-client 聚合.

v3.2 (4 轮 review 整合):
  M1: client 端 sample 累加 (round 末算 mean), 不用 batch-mean EMA
  M2: server 端 L2-norm + 等权聚合 (raw 空间 EMA, 最后 normalize - RV2 fix)
  NV3: client 端 α_round=0 (无跨 round 平滑)
  NV4: server 端跨 round EMA β=0.8 (在 raw 空间, 不在 unit 球面)
  RV2: server EMA 改 raw 空间避免 unit 球面凸组合数学不一致

向后兼容: proto_weight=0 时所有 prototype 路径关闭, 等价 F2DC vanilla.
"""
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import numpy as np

from utils.args import *
from models.utils.federated_model import FederatedModel
from models.f2dc import F2DC, get_pred


class F2DCPG(F2DC):
    """PG-DFC v3.2 — F2DC + prototype-guided DFC."""
    NAME = 'f2dc_pg'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super().__init__(nets_list, args, transform)
        self.args = args

        # PG-DFC v3.2 超参 (从 args 读, 默认值见 main_run.py)
        self.proto_weight_target = getattr(args, 'pg_proto_weight', 0.3)
        self.warmup_rounds = getattr(args, 'pg_warmup_rounds', 30)
        self.ramp_rounds = getattr(args, 'pg_ramp_rounds', 20)
        self.server_ema_beta = getattr(args, 'pg_server_ema_beta', 0.8)
        # attn_temperature 在模型里, 通过 args 传给 backbone

        # Client 本地 prototype state (per-client per-class)
        # 结构: list of (num_classes, C) tensors, 每个 client 一个
        self.local_protos = [None] * args.parti_num
        self.local_class_counts = [None] * args.parti_num
        self.feat_dim = None  # backbone output channel, 第一次 forward 后填

        # Server raw prototype (用于跨 round EMA, RV2 fix - 在 raw 空间不在 unit 球面)
        self.global_proto_raw = None         # (num_classes, C), 未 normalize 的 EMA
        self.global_proto_unit = None        # (num_classes, C), normalize 后下发的版本

        # 诊断 logs
        self.proto_logs = []                 # 每 round 一条 diagnostic dict

    def _get_proto_weight(self):
        """warmup + ramp schedule"""
        ep = self.epoch_index
        if ep < self.warmup_rounds:
            return 0.0
        elif ep < self.warmup_rounds + self.ramp_rounds:
            ramp = (ep - self.warmup_rounds) / max(self.ramp_rounds, 1)
            return self.proto_weight_target * ramp
        else:
            return self.proto_weight_target

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients
        all_clients_loss = 0.0
        self.num_samples = []

        # 当前 round 的 proto_weight
        current_pw = self._get_proto_weight()

        # 把 proto_weight 同步到所有 client model 的 dfc_module + 把 global proto 下发到 buffer
        for i in online_clients:
            net = self.nets_list[i]
            if hasattr(net, 'dfc_module') and hasattr(net.dfc_module, 'set_proto_weight'):
                net.dfc_module.set_proto_weight(current_pw)
                # 下发 global prototype (m2 fix: persistent=False, 不被 FedAvg 误聚合)
                if self.global_proto_unit is not None:
                    net.dfc_module.class_proto.copy_(self.global_proto_unit.to(self.device))
                # 重置 client 端诊断
                if hasattr(net.dfc_module, 'reset_diag'):
                    net.dfc_module.reset_diag()

        # 训练所有 online clients
        round_diag_collect = []
        for i in online_clients:
            c_loss, c_samples = self._train_net_pg(i, self.nets_list[i], priloader_list[i])
            all_clients_loss += c_loss
            self.num_samples.append(c_samples)

            # 收集 client 端诊断
            net = self.nets_list[i]
            if hasattr(net, 'dfc_module') and hasattr(net.dfc_module, 'get_diag_summary'):
                d = net.dfc_module.get_diag_summary()
                if d is not None:
                    d['client_id'] = i
                    round_diag_collect.append(d)

        # F2DC 原 backbone 聚合 (FedAvg)
        self.aggregate_nets(None)

        # ★ Prototype 聚合 (v3.2 — L2-norm 等权 + server EMA β raw 空间)
        if self.epoch_index >= self.warmup_rounds - 1:
            self.aggregate_protos_v3()

        # 记录 round diagnostic
        round_summary = self._summarize_round_diag(round_diag_collect)
        round_summary['round'] = self.epoch_index
        round_summary['proto_weight_active'] = current_pw
        if self.global_proto_unit is not None:
            round_summary['global_proto_norm_mean'] = float(
                self.global_proto_unit.norm(dim=-1).mean().item()
            )
        self.proto_logs.append(round_summary)

        all_c_avg_loss = all_clients_loss / len(online_clients)
        return round(all_c_avg_loss, 3)

    def _train_net_pg(self, index, net, train_loader):
        """F2DC._train_net 加 sample 累加 prototype 更新"""
        try:
            n_avail = len(train_loader.sampler.indices) if hasattr(train_loader.sampler, "indices") else len(train_loader.dataset)
        except Exception:
            n_avail = 1
        if n_avail == 0:
            print(f"[skip] client {index} has empty dataloader")
            return 0.0, 0

        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)

        # 自动推断 feat_dim (第一次)
        if self.feat_dim is None:
            self.feat_dim = net.dfc_module.C if hasattr(net, 'dfc_module') else 512

        N_CLASSES = self.args.num_classes if hasattr(self.args, 'num_classes') else 7  # default PACS

        # ★ M1 fix: round 内 sample 累加 (不用 batch-mean EMA)
        sum_feat_round = torch.zeros(N_CLASSES, self.feat_dim, device=self.device)
        count_round = torch.zeros(N_CLASSES, device=self.device)

        num_c_samples = 0
        iterator = tqdm(range(self.local_epoch), disable=True)
        global_loss = 0.0
        global_samples = 0

        for iter in iterator:
            epoch_loss = 0.0
            epoch_samples = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)

                out, feat, ro_outputs, re_outputs, rec_outputs, ro_flatten, re_flatten = net(images)
                outputs = out
                wrong_high_labels = get_pred(out, labels)

                # F2DC 原 loss (完全不变)
                DFD_dis1_loss = torch.tensor(0.).to(self.device)
                if len(ro_outputs):
                    for ro_out in ro_outputs:
                        DFD_dis1_loss += criterion(ro_out, labels)
                    DFD_dis1_loss /= len(ro_outputs)
                DFD_dis2_loss = torch.tensor(0.).to(self.device)
                if len(re_outputs):
                    for re_out in re_outputs:
                        DFD_dis2_loss += criterion(re_out, wrong_high_labels)
                    DFD_dis2_loss /= len(re_outputs)
                l_cos = torch.cosine_similarity(ro_flatten, re_flatten, dim=1)
                DFD_sep_loss = (l_cos / self.tem).mean()
                DFD_loss = DFD_dis1_loss + DFD_dis2_loss + DFD_sep_loss

                DFC_loss = torch.tensor(0.).to(self.device)
                if len(rec_outputs):
                    for rec_out in rec_outputs:
                        DFC_loss += criterion(rec_out, labels)
                    DFC_loss /= len(rec_outputs)

                loss_DC = self.args.lambda1 * DFD_loss + self.args.lambda2 * DFC_loss
                loss_CE = criterion(outputs, labels)
                loss = loss_CE + loss_DC

                loss.backward()
                optimizer.step()

                # ★ Sample 累加 (M1 fix) — 用 r_feat (ro_flatten 已经是 pooled)
                # ro_flatten shape: (B, C)
                with torch.no_grad():
                    for c in range(N_CLASSES):
                        mask_c = (labels == c)
                        n_c = mask_c.sum().item()
                        if n_c > 0:
                            sum_feat_round[c] += ro_flatten[mask_c].sum(0)
                            count_round[c] += n_c

                batch_size = labels.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size

            global_loss += epoch_loss
            global_samples += epoch_samples
            num_c_samples = epoch_samples

        # ★ Round 末算 prototype (M1: sum/count, NV3: 不跨 round 平滑)
        local_proto = torch.zeros(N_CLASSES, self.feat_dim, device=self.device)
        for c in range(N_CLASSES):
            if count_round[c] > 0:
                local_proto[c] = sum_feat_round[c] / count_round[c]
            # n_c=0 时保持 0, 聚合时会被 mask 掉
        self.local_protos[index] = local_proto.cpu()
        self.local_class_counts[index] = count_round.cpu()

        global_avg_loss = global_loss / max(global_samples, 1)
        return round(global_avg_loss, 3), num_c_samples

    def aggregate_protos_v3(self):
        """
        v3.2 prototype 聚合:
        - L2-norm + 等权 (M2 fix)
        - RV2: server EMA 在 raw 空间 (不在 unit 球面) 避免凸组合数学不一致
        - NV4: 跨 round EMA β=0.8 平滑投票成员变化
        """
        N_CLASSES = self.local_protos[0].shape[0] if self.local_protos[0] is not None else 7
        C = self.feat_dim

        if self.global_proto_raw is None:
            self.global_proto_raw = torch.zeros(N_CLASSES, C)

        new_aggregated = torch.zeros(N_CLASSES, C)

        for c in range(N_CLASSES):
            valid_clients = [
                k for k in self.online_clients
                if self.local_protos[k] is not None
                   and self.local_class_counts[k] is not None
                   and self.local_class_counts[k][c].item() > 0
            ]
            if not valid_clients:
                # 无 client 有此类, 保持上 round 值
                new_aggregated[c] = self.global_proto_raw[c]
                continue

            # M2 fix: L2-normalize 每个 client 的 prototype, 再等权平均
            normed_list = []
            for k in valid_clients:
                p = self.local_protos[k][c]
                norm = p.norm()
                if norm > 1e-6:
                    normed_list.append(p / norm)
                # else: 跳过 0 prototype (NV5 safety)

            if not normed_list:
                new_aggregated[c] = self.global_proto_raw[c]
                continue

            new_aggregated[c] = torch.stack(normed_list).mean(0)
            # 注: 这里不再 normalize, 留到最后 unit 化时做

        # ★ RV2 fix: server EMA 在 raw 空间
        beta = self.server_ema_beta
        if self.global_proto_raw.abs().sum() < 1e-6:
            # 第一次, 直接用
            self.global_proto_raw = new_aggregated
        else:
            # raw 空间凸组合 (不 normalize 中间值)
            self.global_proto_raw = beta * self.global_proto_raw + (1 - beta) * new_aggregated

        # 只在最终下发时 normalize 到 unit (避免 NV5 NaN: 加 norm 阈值检查)
        unit_protos = torch.zeros_like(self.global_proto_raw)
        for c in range(N_CLASSES):
            norm = self.global_proto_raw[c].norm()
            if norm > 1e-3:                                  # NV5 safety
                unit_protos[c] = self.global_proto_raw[c] / norm
            else:
                # norm 太小 (反向时合成 ≈ 0), 用 new_aggregated 作 fallback
                fb_norm = new_aggregated[c].norm()
                if fb_norm > 1e-6:
                    unit_protos[c] = new_aggregated[c] / fb_norm
                # else: 保持 0
        self.global_proto_unit = unit_protos

    def _summarize_round_diag(self, client_diags):
        """聚合 client 端 diagnostic 成 round summary"""
        if not client_diags:
            return {}
        keys = ['mask_sparsity_mean', 'mask_sparsity_std', 'attn_entropy_mean', 'proto_signal_ratio_mean']
        summary = {}
        for k in keys:
            vals = [d[k] for d in client_diags if d.get(k) is not None]
            if vals:
                summary[k] = float(np.mean(vals))
        return summary
