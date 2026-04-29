"""
PG-DFC v3.3 — 加方案 A (outlier-aware aggregation) + 方案 B (client adaptive proto_weight)
基于 v3.2 (f2dc_pg.py) 的 aggregate_protos_v3 + loc_update 修改

v3.3 新增:
- A: server 聚合时检测 outlier client (跟 median 方向 cosine sim < threshold), mask 不参与
- B: client 端基于自己跟 global proto 的 sim 自适应 proto_weight (低 sim → 降 proto_weight)
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
from models.f2dc_pg import F2DCPG


class F2DCPGv33(F2DCPG):
    """
    PG-DFC v3.3 — outlier-aware + adaptive proto_weight
    继承 v3.2,只覆盖 aggregate_protos_v3 (加 outlier mask) 和 loc_update (加 adaptive pw)
    """
    NAME = 'f2dc_pgv33'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super().__init__(nets_list, args, transform)
        # v3.3 超参 (从 args 读, 默认值见 main_run.py)
        self.outlier_cosine_thresh = getattr(args, 'pg_outlier_cosine_thresh', 0.3)
        self.adaptive_pw_enable = getattr(args, 'pg_adaptive_pw', True)
        self.adaptive_pw_floor = 0.05  # 自适应 proto_weight 最小值

        # client 端 sim 跟 global proto, 用于 adaptive pw
        self.client_proto_global_sim = [0.5] * args.parti_num  # default 中等

    def _get_proto_weight_for_client(self, client_idx):
        """
        方案 B: client adaptive proto_weight
        基于 client 上 round 的 sim(local proto vs global proto), 自适应调整 proto_weight
        sim 高 (跟 global 一致) → 用 default proto_weight
        sim 低 (outlier) → 降到 floor (近 0 等价于 vanilla)
        """
        base_w = self._get_proto_weight()  # warmup + ramp 后的 target
        if not self.adaptive_pw_enable or base_w == 0:
            return base_w

        sim = self.client_proto_global_sim[client_idx]  # [-1, 1]
        # smooth sigmoid mapping: sim=0.7 → factor 1.0 / sim=0.3 → factor 0.5 / sim=0.0 → factor 0.1
        factor = float(torch.sigmoid(torch.tensor(8.0 * (sim - 0.3))).item())
        adaptive_w = max(self.adaptive_pw_floor, base_w * factor)
        return adaptive_w

    def loc_update(self, priloader_list):
        """覆盖父类 loc_update,加 client adaptive proto_weight"""
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients
        all_clients_loss = 0.0
        self.num_samples = []

        # 当前 round 的 base proto_weight
        base_pw = self._get_proto_weight()

        # 同步到所有 client model 的 dfc_module + 下发 global prototype
        for i in online_clients:
            net = self.nets_list[i]
            if hasattr(net, 'dfc_module') and hasattr(net.dfc_module, 'set_proto_weight'):
                # ★ v3.3 方案 B: 用 client 自适应 proto_weight
                client_pw = self._get_proto_weight_for_client(i)
                net.dfc_module.set_proto_weight(client_pw)
                if self.global_proto_unit is not None:
                    net.dfc_module.class_proto.copy_(self.global_proto_unit.to(self.device))
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

        # backbone 聚合
        self.aggregate_nets(None)

        # ★ v3.3 prototype 聚合 (outlier-aware)
        if self.epoch_index >= self.warmup_rounds - 1:
            self.aggregate_protos_v33()

        # class_proto is a non-persistent buffer and is skipped by state_dict.
        # Sync it explicitly so global evaluation uses the same PG-DFC path.
        self._sync_global_proto_to_global_net(base_pw)

        # 记录 round diagnostic
        round_summary = self._summarize_round_diag(round_diag_collect)
        round_summary['round'] = self.epoch_index
        round_summary['proto_weight_active'] = base_pw
        round_summary['client_pw_used'] = [self._get_proto_weight_for_client(i) for i in online_clients]
        round_summary['client_proto_sim'] = [self.client_proto_global_sim[i] for i in online_clients]
        if self.global_proto_unit is not None:
            round_summary['global_proto_norm_mean'] = float(
                self.global_proto_unit.norm(dim=-1).mean().item()
            )
        self.proto_logs.append(round_summary)

        all_c_avg_loss = all_clients_loss / len(online_clients)
        return round(all_c_avg_loss, 3)

    def aggregate_protos_v33(self):
        """
        v3.3 prototype 聚合 — outlier-aware:
        - 先算 valid client prototype 的 median 方向
        - 跟 median 方向 cosine sim < threshold 的 client 标记为 outlier, mask 不参与
        - 用 inlier 重新聚合 (L2-norm + 等权 + raw EMA β)
        - 同时记录每个 client 的 sim, 给 client 自适应 proto_weight 用 (方案 B 的输入)
        """
        # patch (2026-04-29): 用 args.num_classes 而不是 fallback 7
        N_CLASSES = (
            self.local_protos[0].shape[0]
            if (self.local_protos[0] is not None)
            else int(getattr(self.args, 'num_classes', 7))
        )
        C = self.feat_dim

        if self.global_proto_raw is None:
            self.global_proto_raw = torch.zeros(N_CLASSES, C)

        new_aggregated = torch.zeros(N_CLASSES, C)

        # 临时记录每个 client 的"跟 global 的 sim 之和"(用于 adaptive pw)
        client_sim_sum = {k: 0.0 for k in self.online_clients}
        client_sim_count = {k: 0 for k in self.online_clients}

        for c in range(N_CLASSES):
            valid_clients = [
                k for k in self.online_clients
                if self.local_protos[k] is not None
                   and self.local_class_counts[k] is not None
                   and self.local_class_counts[k][c].item() > 0
            ]
            if not valid_clients:
                new_aggregated[c] = self.global_proto_raw[c]
                continue

            # 1. L2-normalize 每个 valid client 的 prototype
            normed = {}
            for k in valid_clients:
                p = self.local_protos[k][c]
                norm = p.norm()
                if norm > 1e-6:
                    normed[k] = p / norm

            if not normed:
                new_aggregated[c] = self.global_proto_raw[c]
                continue

            # 2. 算 median 方向 (临时 mean 后 normalize, 类似一阶 Karcher mean)
            stacked = torch.stack(list(normed.values()))
            median_dir = F.normalize(stacked.mean(0), dim=0)

            # 3. ★ 方案 A: 算每个 client 跟 median 的 cosine sim, 标记 outlier
            inliers = []
            for k, n in normed.items():
                sim = float(F.cosine_similarity(n, median_dir, dim=0).item())
                client_sim_sum[k] += sim
                client_sim_count[k] += 1
                if sim >= self.outlier_cosine_thresh:
                    inliers.append(n)
                # else: 视为 outlier, 不参与本类聚合

            # 4. 用 inlier 重新 mean (如果 inlier 太少 fallback 到全部)
            if len(inliers) >= max(2, len(normed) // 2):
                new_aggregated[c] = torch.stack(inliers).mean(0)
            else:
                # inlier 太少, fallback 到全部 (避免 noise)
                new_aggregated[c] = stacked.mean(0)

        # 5. server EMA β=0.8 平滑 (raw 空间)
        beta = self.server_ema_beta
        if self.global_proto_raw.abs().sum() < 1e-6:
            self.global_proto_raw = new_aggregated
        else:
            self.global_proto_raw = beta * self.global_proto_raw + (1 - beta) * new_aggregated

        # 6. 输出 unit (NV5 safety)
        unit_protos = torch.zeros_like(self.global_proto_raw)
        for c in range(N_CLASSES):
            norm = self.global_proto_raw[c].norm()
            if norm > 1e-3:
                unit_protos[c] = self.global_proto_raw[c] / norm
            else:
                fb_norm = new_aggregated[c].norm()
                if fb_norm > 1e-6:
                    unit_protos[c] = new_aggregated[c] / fb_norm
        self.global_proto_unit = unit_protos

        # 7. ★ 更新 client_proto_global_sim (给方案 B 用,下 round 生效)
        for k in self.online_clients:
            if client_sim_count[k] > 0:
                self.client_proto_global_sim[k] = client_sim_sum[k] / client_sim_count[k]
