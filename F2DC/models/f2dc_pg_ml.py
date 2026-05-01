"""
PG-DFC v3.3 + Multi-Layer (f2dc_pg_ml)
======================================
基于 F2DCPGv33 (outlier-aware + adaptive proto_weight) 加 layer3 deep supervision.

设计:
  - 继承 F2DCPGv33, 重写 _train_net_pg, 加 aux3 deep supervision loss
  - L_total = L_PGv33_existing + ml_aux_alpha · CE(aux3_logits, labels)
  - ml_aux_alpha=0 时 → aux3_loss 不纳入计算 → lite 模块 grad=None (PyTorch SGD
    在 grad=None 时不更新参数), 整个 lite 分支静止 → 等价 PG-DFC v3.3.
    注意: backbone forward 仍会算 lite 分支 (浪费 ~10% compute), 但不影响 acc.
  - 不改 loc_update / aggregate_protos / class_proto 同步逻辑 (全继承)

新超参 (从 args 读取, 默认值见 main_run.py):
  - ml_aux_alpha:    deep sup loss 权重, 默认 0.1 (R10 smoke 起始值, 设 0 退化)
  - ml_lite_channel: DFD/DFC lite 内部 channel, 默认 32 (backbone 用)
  - ml_lite_tau:     lite gumbel tau, 默认 0.5 (backbone 用, 比 layer4 的 0.1 大避免坍塌)

诊断字段 (round summary 里):
  - mask3_sparsity_mean / mask3_sparsity_std: layer3 mask 平均稠密度, 健康 0.3-0.7
  - aux3_loss_mean / main_loss_mean: 两个 loss 量级
  - aux3_over_main_ratio: aux3 / main 比值, 大 = aux 主导, 小 = aux 辅助
  - ml_aux_alpha: 当前 round 用的 alpha 值 (后续可加 schedule 时反映动态值)
"""
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.args import *
from models.f2dc import get_pred
from models.f2dc_pgv33 import F2DCPGv33


class F2DCPGML(F2DCPGv33):
    """PG-DFC v3.3 + multi-layer deep supervision."""
    NAME = 'f2dc_pg_ml'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super().__init__(nets_list, args, transform)
        # ml deep sup 权重 (loss-side, backbone 已经构建好 module)
        self.ml_aux_alpha = float(getattr(args, 'ml_aux_alpha', 0.1))
        # Round 内 deep sup loss 累计 (用于 round summary)
        self._round_aux3_loss_sum = 0.0
        self._round_aux3_samples = 0
        self._round_main_loss_sum = 0.0  # 主 loss 累积, 用于诊断比值

    def _train_net_pg(self, index, net, train_loader):
        """覆盖父类: 在原 PG loss 基础上加 aux3 deep supervision loss."""
        try:
            n_avail = len(train_loader.sampler.indices) if hasattr(train_loader.sampler, "indices") \
                else len(train_loader.dataset)
        except Exception:
            n_avail = 1
        if n_avail == 0:
            print(f"[skip] client {index} has empty dataloader")
            return 0.0, 0

        net = net.to(self.device)
        net.train()

        # 重置 lite 分支的 round 内诊断
        if hasattr(net, 'dfd_lite') and hasattr(net.dfd_lite, 'reset_diag'):
            net.dfd_lite.reset_diag()

        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)

        if self.feat_dim is None:
            self.feat_dim = net.dfc_module.C if hasattr(net, 'dfc_module') else 512

        N_CLASSES = self.args.num_classes if hasattr(self.args, 'num_classes') else 7

        sum_feat_round = torch.zeros(N_CLASSES, self.feat_dim, device=self.device)
        count_round = torch.zeros(N_CLASSES, device=self.device)

        num_c_samples = 0
        iterator = tqdm(range(self.local_epoch), disable=True)
        global_loss = 0.0
        global_samples = 0
        # client local 累计
        c_aux3_loss_sum = 0.0
        c_main_loss_sum = 0.0
        c_aux3_samples = 0

        alpha = self.ml_aux_alpha

        for it in iterator:
            epoch_loss = 0.0
            epoch_samples = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)

                out, feat, ro_outputs, re_outputs, rec_outputs, ro_flatten, re_flatten = net(images)
                outputs = out
                wrong_high_labels = get_pred(out, labels)

                # F2DC 原 loss (不变)
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
                main_loss = loss_CE + loss_DC

                # ★ ML deep sup loss — 仅在 alpha > 0 时纳入计算
                if alpha > 0 and getattr(net, '_last_aux3_logits', None) is not None:
                    aux3_loss = criterion(net._last_aux3_logits, labels)
                    total_loss = main_loss + alpha * aux3_loss
                    # 诊断累积
                    bs = labels.size(0)
                    c_aux3_loss_sum += aux3_loss.item() * bs
                    c_main_loss_sum += main_loss.item() * bs
                    c_aux3_samples += bs
                else:
                    total_loss = main_loss

                total_loss.backward()
                optimizer.step()

                # M1: sample 累加 prototype state (用 r_feat pooled)
                with torch.no_grad():
                    sum_feat_round.index_add_(0, labels, ro_flatten.detach())
                    count_round += torch.bincount(labels, minlength=N_CLASSES).float()

                batch_size = labels.size(0)
                epoch_loss += total_loss.item() * batch_size
                epoch_samples += batch_size

            global_loss += epoch_loss
            global_samples += epoch_samples
            num_c_samples = epoch_samples

        # Round 末算 prototype
        local_proto = torch.zeros(N_CLASSES, self.feat_dim, device=self.device)
        for c in range(N_CLASSES):
            if count_round[c] > 0:
                local_proto[c] = sum_feat_round[c] / count_round[c]
        self.local_protos[index] = local_proto.cpu()
        self.local_class_counts[index] = count_round.cpu()

        # 累计到 model 级 round 诊断
        self._round_aux3_loss_sum += c_aux3_loss_sum
        self._round_aux3_samples += c_aux3_samples
        self._round_main_loss_sum += c_main_loss_sum

        global_avg_loss = global_loss / max(global_samples, 1)
        return round(global_avg_loss, 3), num_c_samples

    def _summarize_round_diag(self, client_diags):
        """覆盖父类 summary: 加上 mask3 7-stat / aux3-vs-main loss 比值.

        新增 (2026-05-01): mask3 7 个 healthcheck 指标 (unit_std / hard_ratio /
        mid_ratio / sample_std / channel_std / spatial_std), 用 compute_mask_stats
        在 backbone 收集. backward compat 保留 mask3_sparsity_mean / std.
        """
        import numpy as np
        summary = super()._summarize_round_diag(client_diags)

        # 收集 lite mask3 全部 7-stat (从所有 online client 的 dfd_lite.get_diag_summary)
        # mask3_*_mean keys: mean / unit_std / hard_ratio / mid_ratio /
        #                    sample_std / channel_std / spatial_std
        # backward compat keys: mask3_sparsity_mean / mask3_sparsity_std
        m3_collect = {}  # key -> list of values
        for k in self.online_clients:
            net = self.nets_list[k]
            if hasattr(net, 'dfd_lite') and hasattr(net.dfd_lite, 'get_diag_summary'):
                d = net.dfd_lite.get_diag_summary()
                if d is None:
                    continue
                for kk, vv in d.items():
                    if isinstance(vv, (int, float)) and vv is not None:
                        m3_collect.setdefault(kk, []).append(vv)
        for kk, vals in m3_collect.items():
            if vals:
                summary[kk] = float(np.mean(vals))

        # aux3 / main loss 比值
        if self._round_aux3_samples > 0:
            aux3_mean = self._round_aux3_loss_sum / self._round_aux3_samples
            main_mean = self._round_main_loss_sum / self._round_aux3_samples
            summary['aux3_loss_mean'] = float(aux3_mean)
            summary['main_loss_mean'] = float(main_mean)
            summary['aux3_over_main_ratio'] = float(aux3_mean / max(main_mean, 1e-8))
        summary['ml_aux_alpha'] = self.ml_aux_alpha

        # 重置 round 累积 (下 round 重新计)
        self._round_aux3_loss_sum = 0.0
        self._round_aux3_samples = 0
        self._round_main_loss_sum = 0.0

        # round summary 直接 print 出来, smoke test 能直接 grep
        # 7-stat 完整 print + aux3 + 旧字段 backward compat
        keys_show = [
            'round',
            # mask3 7-stat (layer3 lite, ML 新增)
            'mask3_mean_mean', 'mask3_unit_std_mean',
            'mask3_hard_ratio_mean', 'mask3_mid_ratio_mean',
            'mask3_sample_std_mean', 'mask3_channel_std_mean', 'mask3_spatial_std_mean',
            # aux3 deep sup
            'aux3_loss_mean', 'main_loss_mean', 'aux3_over_main_ratio',
            # mask4 7-stat (layer4, F2DC 原版 DFD via DFC_PG 收集)
            'mask_mean_mean', 'mask_unit_std_mean',
            'mask_hard_ratio_mean', 'mask_mid_ratio_mean',
            'mask_sample_std_mean', 'mask_channel_std_mean', 'mask_spatial_std_mean',
            # PG-DFC attention
            'attn_entropy_mean', 'proto_signal_ratio_mean',
            # backward compat
            'mask3_sparsity_mean', 'mask_sparsity_mean',
        ]
        printable = {k: summary.get(k) for k in keys_show if summary.get(k) is not None}
        if printable:
            print('[ML diag]', printable)

        return summary
