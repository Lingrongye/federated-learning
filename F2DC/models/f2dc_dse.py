"""
F2DC + DSE_Rescue3 + CCC + Magnitude (Progressive Shift Rescue)
==================================================================
基于纯 F2DC base, 加 layer3 DSE_Rescue3 修正 + CCC class-conditional consistency +
Magnitude safety guard.

设计 (基于 EXP-141 v3+v4 失败教训, 跟 codex 多轮 review 收敛):
  - 不依赖 PG-DFC / prototype attention / aux3 deep sup (那些在 mask3 ablation 已验证无效)
  - layer3 不切 hard mask, 改 soft correction adapter (DSE_Rescue3)
  - CCC 给 DSE 明确"去域"监督方向 (cosine similarity to layer3 raw class proto)
  - Magnitude guard 防止 delta 爆炸破坏 layer4 input
  - Warmup + Ramp 防止训练初期破坏 backbone 收敛

Loss:
  L_total = L_F2DC_main          # 标准 F2DC 损失 (DFD + DFC + CE)
          + lambda_cc_t * L_cc   # cosine: GAP(feat3_rescued) 接近 global_proto3_unit[label]
          + lambda_mag * L_mag   # max(0, ||rho*delta||/||feat|| - r_max)^2

★ 关键设计选择 (codex review 校正):
  - global_proto3 用 raw feat3 (NOT rescued!) 算
    -> 提供 "把修正后的 feat 推向 raw 跨域 class 共识中心" 的明确方向
    -> 如果用 rescued, target 跟 input 同源, 退化为 weak self-loop
  - cosine consistency (而非 KL on BN stats)
    -> 类似 FedProto / FPL, 只对齐方向不限 magnitude
  - Warmup + Ramp 同步 (rho 跟 lambda_cc):
    -> R0 ~ Rwarmup-1: rho=0, lambda_cc=0 (DSE 不影响主路, server EMA proto3)
    -> Rwarmup ~ Rwarmup+ramp-1: 同步线性 0 → max
    -> Rwarmup+ramp+: full

新超参 (CLI args, default 见 main_run.py):
  --dse_reduction               default 8    bottleneck reduction
  --dse_rho_max                 default 0.1  修正强度上限
  --dse_lambda_cc               default 0.1  CCC 权重
  --dse_lambda_mag              default 0.01 magnitude safety guard 权重
  --dse_r_max                   default 0.15 magnitude 触发阈值
  --dse_cc_warmup_rounds        default 5    CCC warmup
  --dse_cc_ramp_rounds          default 10   CCC ramp
  --dse_rho_warmup_rounds       default 5    rho warmup (跟 cc_warmup 同步)
  --dse_rho_ramp_rounds         default 10   rho ramp (跟 cc_ramp 同步)
  --dse_proto3_ema_beta         default 0.85 server proto3 EMA momentum

Round summary print (新增 dse_* / cc_* / proto3_* 全部诊断, [DSE diag] 前缀):
  rho_t, lambda_cc_t                                          - 当前 ramp 系数
  dse_delta_raw_ratio_mean, dse_delta_scaled_ratio_mean       - 修正幅度
  dse_delta_cos_feat_mean                                     - delta 跟 feat 方向
  cc_loss_mean, mag_loss_mean, mag_exceed_rate                - loss 数值 + 触发率
  proto3_valid_classes, proto3_update_norm                    - server proto3 状态
  raw_to_target_cos_mean, rescued_to_target_cos_mean          - CCC 改善证据
  ccc_improvement = rescued - raw                             - DSE 是否真"擦域"

训练流程 (loc_update):
  1. 算 rho_t / lambda_cc_t (warmup + ramp 公式)
  2. 给所有 client model set_rho_t + set_global_proto3_unit
  3. reset client dse diag
  4. 重置 round 级 server proto3 累积
  5. for each online client: _train_net_dse (call F2DC main loss + CCC + Mag, 顺便累 raw proto3)
  6. backbone FedAvg 聚合 (走 standard aggregate_nets)
  7. server 端聚合 raw_proto3 → EMA → L2-norm → 同步给 client
  8. print round diag

Test-time:
  - eval 不用 CCC / Mag, DSE_Rescue3 forward 仍跑 (rho_t 是当前 ramp 值)
  - 干预实验: rho_t = 0 时等价 F2DC vanilla, 可在 eval 时强制 set_rho_t(0) 看 acc_drop
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from utils.args import *
from models.f2dc import F2DC, get_pred


class F2DCDSE(F2DC):
    """F2DC + DSE_Rescue3 + CCC + Magnitude.

    继承 F2DC base, 重写 _train_net 加 CCC + Mag, 重写 loc_update 加 server proto3 同步.
    backbone factory 走 resnet10_f2dc_dse (具 dse_rescue3 module + global_proto3_unit_buf).
    """
    NAME = 'f2dc_dse'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super().__init__(nets_list, args, transform)
        self.args = args
        # DSE 超参 (CLI default 见 main_run.py)
        # dse修正强度上限
        self.dse_rho_max = float(getattr(args, 'dse_rho_max', 0.1))
        # ccc loss 最大权重
        self.dse_lambda_cc = float(getattr(args, 'dse_lambda_cc', 0.1))
        # magnitude loss权重
        self.dse_lambda_mag = float(getattr(args, 'dse_lambda_mag', 0.01))
        # 允许的最大修正比例
        self.dse_r_max = float(getattr(args, 'dse_r_max', 0.15))
        # ccc loss什么时候启动 怎么增大
        self.dse_cc_warmup = int(getattr(args, 'dse_cc_warmup_rounds', 5))
        self.dse_cc_ramp = int(getattr(args, 'dse_cc_ramp_rounds', 10))
        # dse主的路注入强度什么时候启动 怎么增大
        self.dse_rho_warmup = int(getattr(args, 'dse_rho_warmup_rounds', 5))
        self.dse_rho_ramp = int(getattr(args, 'dse_rho_ramp_rounds', 10))
        # 每个server的prototype 的ema平滑系数
        self.dse_proto3_ema_beta = float(getattr(args, 'dse_proto3_ema_beta', 0.85))
        # ★ Digits 专用 (default 0 = off, backward compat 跟 PACS/Office 不传该参数完全一致):
        # L_orth = cos(delta3, feat3)^2 mean, 强制 adapter delta 朝 feat 正交方向 (横向校正)
        # 解决 Digits svhn 灾难根因: random init 让 R7 adapter delta 锁定同向 → svhn corrupt
        # 仅 Digits 实验传非零值 (e.g., --dse_lambda_orth 0.05), PACS/Office 不传 = 0 = 关闭
        self.dse_lambda_orth = float(getattr(args, 'dse_lambda_orth', 0.0))
        # ★ Fix (codex 复审): CCC 诊断改成"每个 (client, epoch) 固定前 N batch", 不再 10% 随机
        # 保证 R10 smoke / 小 batch / proto 还没 ready 时也总有诊断数据，每个client 每个epoch固定前几个batch 记录ccc诊断
        self._ccc_fixed_batches = int(getattr(args, 'dse_ccc_fixed_batches', 2))

        # global proto3 (server 跨 round EMA buffer)
        # raw: 跨 client 聚合的 raw feat3 GAP class mean (CPU tensor)
        # unit: L2-norm 版给 client 当 CCC target
        self.global_proto3_raw = None
        self.global_proto3_unit = None
        self.epoch_index = 0

        # ★ Fix (codex 复审): proto_logs 跟 PG-DFC 现有机制对齐, 让 utils/diagnostic.py
        # dump_heavy_snapshot 自动捡 latest dict 写到 proto_logs.jsonl + npz proto_diag_*
        # (之前 _print_round_diag 只 print, 训练日志一丢就拿不到 cc_loss/mag_ratio/proto3_ema_delta)
        self.proto_logs = []

        # round 级累积 (诊断 + server aggregate)
        self._round_local_proto3_sum = None       # (C, dim) 跨 client 跨 batch 累 raw feat3 GAP
        self._round_local_proto3_count = None     # (C,) class count
        self._round_cc_loss_sum = 0.0
        self._round_mag_loss_sum = 0.0
        # ★ Fix (codex 复审): per-sample exceed count + 真正参与 mag 计算的 sample 总数
        # (之前 batch scalar > r_max 单一计数, 单 sample 爆但 batch mean 不爆会漏报)
        self._round_mag_exceed_samples = 0      # Σ (ratio_per > r_max).sum()
        self._round_mag_eval_samples = 0        # Σ batch_size  (rho>0 时才累)
        self._round_total_batches = 0
        self._round_raw_to_target_cos = []        # 10% sample, raw to target cos
        self._round_rescued_to_target_cos = []    # 10% sample, rescued to target cos
        # ★ Fix #5 (codex): mag per-sample p95/max diag (batch scalar 掩盖单 sample 异常)
        self._round_mag_ratio_p95 = []
        self._round_mag_ratio_max = []
        # ★ Fix #7 (codex): proto3 EMA delta 真实更新幅度 (vs proto3_update_norm 只是 mean norm)
        self._round_proto3_ema_delta_norm = 0.0
        # ★ L_orth 累计 (Digits 专用, lambda_orth=0 时不累, 也不影响 PACS/Office)
        self._round_orth_loss_sum = 0.0
        self._round_orth_cos_abs_sum = 0.0
        self._round_orth_eval_samples = 0

        # 当前 round 的 ramp 值 (loc_update 算好, _train_net_dse 用)
        self._cur_rho_t = 0.0
        self._cur_lambda_cc_t = 0.0

    def _compute_ramp_value(self, value_max, warmup, ramp, t):
        """Warmup + linear ramp:
            t < warmup:                  0
            warmup <= t < warmup+ramp:   value_max * (t - warmup) / ramp
            t >= warmup+ramp:            value_max
        """
        if t < warmup:
            return 0.0
        if ramp <= 0:
            return value_max
        if t < warmup + ramp:
            return value_max * (t - warmup) / ramp
        return value_max

    def loc_update(self, priloader_list):
        # === Step 1: 算当前 round 的 rho_t / lambda_cc_t ===
        t = self.epoch_index
        rho_t = self._compute_ramp_value(self.dse_rho_max,
                                          self.dse_rho_warmup, self.dse_rho_ramp, t)
        lambda_cc_t = self._compute_ramp_value(self.dse_lambda_cc,
                                                self.dse_cc_warmup, self.dse_cc_ramp, t)
        self._cur_rho_t = rho_t
        self._cur_lambda_cc_t = lambda_cc_t

        # === Step 2: 给所有 client model + global_net 同步 rho_t + global proto3 unit + reset diag ===
        # ★ Fix #2 (codex): rho_t 跟 global_proto3_unit 是 persistent=False buffer, 不走 FedAvg
        # state_dict 同步, 必须显式设给 global_net (eval 时用 global_net, 否则 DSE 等价 off)
        all_nets = list(self.nets_list)
        if hasattr(self, 'global_net') and self.global_net is not None:
            all_nets.append(self.global_net)
        for net in all_nets:
            if hasattr(net, 'set_rho_t'):
                net.set_rho_t(rho_t)
            if hasattr(net, 'set_global_proto3_unit') and self.global_proto3_unit is not None:
                net.set_global_proto3_unit(self.global_proto3_unit)
            if hasattr(net, 'reset_dse_diag'):
                net.reset_dse_diag()

        # === Step 3: 重置 round 级累积 ===
        # ★ Fix #3 (codex): N_CLASSES 从 backbone 推 (而非 args.num_classes default 7),
        # 防止 Office/Digits (10 类) 没传 --num_classes 时越界
        N_CLASSES = self.nets_list[0].linear.out_features
        feat3_dim = self.nets_list[0].feat3_dim if hasattr(self.nets_list[0], 'feat3_dim') else 256
        self._round_local_proto3_sum = torch.zeros(N_CLASSES, feat3_dim, device=self.device)
        self._round_local_proto3_count = torch.zeros(N_CLASSES, device=self.device)
        self._round_cc_loss_sum = 0.0
        self._round_mag_loss_sum = 0.0
        self._round_mag_exceed_samples = 0
        self._round_mag_eval_samples = 0
        self._round_total_batches = 0
        self._round_raw_to_target_cos = []
        self._round_rescued_to_target_cos = []
        # ★ Fix #5/#7 reset
        self._round_mag_ratio_p95 = []
        self._round_mag_ratio_max = []
        self._round_proto3_ema_delta_norm = 0.0
        # ★ L_orth reset (Digits 专用)
        self._round_orth_loss_sum = 0.0
        self._round_orth_cos_abs_sum = 0.0
        self._round_orth_eval_samples = 0

        # === Step 4: 标准 F2DC loc_update (call _train_net_dse) ===
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num,
                                                    replace=False).tolist()
        self.online_clients = online_clients
        all_clients_loss = 0.0
        self.num_samples = []
        for i in online_clients:
            c_loss, c_samples = self._train_net_dse(i, self.nets_list[i], priloader_list[i])
            all_clients_loss += c_loss
            self.num_samples.append(c_samples)

        # === Step 5: backbone FedAvg 聚合 (走 standard) ===
        # 注意: dse_rescue3 weight 走 FedAvg (默认行为, 暂不 personalized)
        self.aggregate_nets(None)

        # === Step 6: server 端聚合 raw_proto3 + EMA + L2-norm + 同步给 client ===
        self._aggregate_proto3()

        # === Step 7: round summary print ===
        self._print_round_diag()

        self.epoch_index += 1
        all_c_avg_loss = all_clients_loss / len(online_clients)
        return round(all_c_avg_loss, 3)

    def _aggregate_proto3(self):
        """server: 跨 client 累的 local_proto3_raw → global_proto3_raw (EMA + L2-norm).

        注意:
          - skip class with count=0 (没 client 这 round 见到的 class)
          - EMA 只更新 valid class (有 count>0 的)
          - L2-norm → global_proto3_unit (传给 client 当 CCC target)
        """
        if self._round_local_proto3_sum is None:
            return
        valid_mask = self._round_local_proto3_count > 0
        if not valid_mask.any():
            return
        new_proto = torch.zeros_like(self._round_local_proto3_sum)
        new_proto[valid_mask] = (
            self._round_local_proto3_sum[valid_mask]
            / self._round_local_proto3_count[valid_mask].unsqueeze(-1)
        )
        # EMA on CPU (避免 device migration)
        new_proto_cpu = new_proto.cpu()
        valid_cpu = valid_mask.cpu()
        if self.global_proto3_raw is None:
            # 第一轮: 直接用 (新 valid class 部分; 没 valid 的 class 留 0)
            self.global_proto3_raw = torch.zeros_like(new_proto_cpu)
            self.global_proto3_raw[valid_cpu] = new_proto_cpu[valid_cpu]
            self._round_proto3_ema_delta_norm = 0.0  # 第一轮没 delta
        else:
            beta = self.dse_proto3_ema_beta
            # ★ Fix #7 (codex): 算 EMA 前后 delta norm (真正"更新幅度", 之前 proto3_update_norm
            # 只是 mean norm 不能反映 EMA 收敛速度)
            old_proto = self.global_proto3_raw[valid_cpu].clone()
            self.global_proto3_raw[valid_cpu] = (
                beta * old_proto
                + (1 - beta) * new_proto_cpu[valid_cpu]
            )
            delta = self.global_proto3_raw[valid_cpu] - old_proto
            self._round_proto3_ema_delta_norm = delta.norm(dim=-1).mean().item()
        # L2-norm (skip 0 vector)
        norms = self.global_proto3_raw.norm(dim=-1, keepdim=True)
        norms = torch.where(norms > 1e-8, norms, torch.ones_like(norms))
        self.global_proto3_unit = self.global_proto3_raw / norms

        # ★ Fix #2 (codex): 立刻 sync proto3 给 global_net + nets_list
        # Why: rho_t / global_proto3_unit_buf 是 persistent=False buffer, FedAvg 不传,
        #      上面 Step 2 sync 已经过了, 不在这里 sync 的话 global_net 在本 round eval
        #      用的是上一轮的 proto3 (差一轮), 影响 CCC target 评估的及时性
        all_nets = list(self.nets_list)
        if hasattr(self, 'global_net') and self.global_net is not None:
            all_nets.append(self.global_net)
        for net in all_nets:
            if hasattr(net, 'set_global_proto3_unit'):
                net.set_global_proto3_unit(self.global_proto3_unit)

    def _train_net_dse(self, index, net, train_loader):
        """覆盖 F2DC _train_net: 加 CCC loss + Mag loss + 收 local_proto3_raw."""
        try:
            n_avail = (
                len(train_loader.sampler.indices)
                if hasattr(train_loader.sampler, "indices")
                else len(train_loader.dataset)
            )
        except Exception:
            n_avail = 1
        if n_avail == 0:
            print(f"[skip] client {index} has empty dataloader")
            return 0.0, 0

        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr,
                              momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)
        # ★ Fix #3 (codex 复审): N_CLASSES 必须从 backbone 推, 不能用 args.num_classes
        # Why: args.num_classes 默认 7, Office/Digits 真值 10, 没传 --num_classes 时
        #      bincount(labels, minlength=N) 不会自动扩到 10, += 累加 10 类 buffer 时
        #      shape (10) vs (7) 直接 RuntimeError. T9b 没抓到是因为强制塞 label 9 触发自动扩
        N_CLASSES = net.linear.out_features

        # client local 本 round 累积
        c_cc_loss_sum = 0.0
        c_mag_loss_sum = 0.0
        # ★ Fix (codex 复审): per-sample exceed (Σ ratio_per > r_max) + 真正参与计算的 sample 数
        c_mag_exceed_samples = 0
        c_mag_eval_samples = 0
        c_batches = 0
        global_loss = 0.0
        global_samples = 0
        num_c_samples = 0

        for it in range(self.local_epoch):
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)

                out, feat, ro_outputs, re_outputs, rec_outputs, ro_flat, re_flat = net(images)
                outputs = out
                wrong_high_labels = get_pred(out, labels)

                # === F2DC main loss (原版, 不变) ===
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
                l_cos = torch.cosine_similarity(ro_flat, re_flat, dim=1)
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

                # === CCC loss (cosine, only when lambda_cc_t > 0) ===
                cc_loss = torch.tensor(0.).to(self.device)
                if (self._cur_lambda_cc_t > 0
                        and self.global_proto3_unit is not None
                        and net._last_feat3_rescued is not None):
                    feat3_rescued = net._last_feat3_rescued  # (B, dim, H, W)
                    h3_rescued = F.adaptive_avg_pool2d(feat3_rescued, 1).flatten(1)  # (B, dim)
                    h3_rescued_unit = F.normalize(h3_rescued, dim=-1, eps=1e-8)
                    target_unit = net.global_proto3_unit_buf[labels].detach()  # (B, dim)
                    # 防止 target 是 0 (cold start, 某 class 还没收到 proto)
                    target_norm = target_unit.norm(dim=-1)
                    valid = target_norm > 1e-8
                    if valid.any():
                        # 把修正后的feat 拉向全局的同类prototype
                        cos_sim = (h3_rescued_unit[valid] * target_unit[valid]).sum(-1)
                        cc_loss = (1.0 - cos_sim).mean()

                # === Magnitude loss (safety guard) ===
                # ★ Fix #5 (codex): per-sample ratio + p95/max diag (batch scalar 掩盖单 sample 异常)
                mag_loss = torch.tensor(0.).to(self.device)
                ratio_scalar = 0.0
                ratio_p95 = 0.0
                ratio_max = 0.0
                # ★ Fix (codex 复审): per-sample exceed 计数 (放外层供后面累加)
                exceed_count_this_batch = 0
                if self._cur_rho_t > 0 and net._last_delta3 is not None:
                    delta3 = net._last_delta3
                    feat3_raw = net._last_feat3_raw
                    # per-sample norm (B,) — flatten 后 norm
                    delta_norm_per = delta3.flatten(1).norm(dim=-1)
                    feat_norm_per = feat3_raw.flatten(1).norm(dim=-1) + 1e-8
                    # 先计算每个样本的修正比例
                    ratio_per_t = self._cur_rho_t * delta_norm_per / feat_norm_per  # (B,)
                    # batch-level scalar (保留, 用于 ratio_scalar 兼容旧 diag)
                    feat_norm = feat3_raw.norm() + 1e-8
                    delta_norm = delta3.norm()
                    ratio_t = self._cur_rho_t * delta_norm / feat_norm
                    ratio_scalar = ratio_t.item()
                    # mag_loss: 用 per-sample ratio max(0, ratio_per - r_max)^2 mean over batch
                    # 让 outlier sample 真正受惩罚 (而非整 batch 平均拉低)，如果修正比例没有超过r_max
                    mag_loss = F.relu(ratio_per_t - self.dse_r_max).pow(2).mean()
                    # diag (no_grad 取值)
                    with torch.no_grad():
                        ratio_p95 = torch.quantile(ratio_per_t, 0.95).item()
                        ratio_max = ratio_per_t.max().item()
                        exceed_count_this_batch = int(
                            (ratio_per_t > self.dse_r_max).sum().item()
                        )

                # === Orthogonality loss (Digits 专用, default lambda=0 时完全不影响) ===
                # 强制 adapter delta 朝 feat3 正交方向 (横向校正), 防止 random init 锁同向
                # cos(delta3, feat3)^2 mean → 0 表示横向, → 1 表示同/反向
                # PACS/Office 默认 lambda_orth=0, 完全 backward compat
                orth_loss = torch.tensor(0.).to(self.device)
                orth_cos_abs_batch = 0.0
                if (getattr(self, 'dse_lambda_orth', 0.0) > 0
                        and self._cur_rho_t > 0
                        and net._last_delta3 is not None):
                    delta3_for_orth = net._last_delta3
                    feat3_for_orth = net._last_feat3_raw
                    delta_flat = delta3_for_orth.flatten(1)  # (B, C*H*W)
                    feat_flat = feat3_for_orth.flatten(1)
                    cos_df = F.cosine_similarity(delta_flat, feat_flat, dim=1)  # (B,)
                    orth_loss = cos_df.pow(2).mean()
                    with torch.no_grad():
                        orth_cos_abs_batch = cos_df.abs().mean().item()

                # === total loss + backward ===
                total_loss = (
                    main_loss
                    + self._cur_lambda_cc_t * cc_loss
                    + self.dse_lambda_mag * mag_loss
                    + getattr(self, 'dse_lambda_orth', 0.0) * orth_loss  # Digits 专用 (default 0 时无影响)
                )
                total_loss.backward()
                optimizer.step()

                # === 收集 local_proto3_raw (★ 从 raw feat3, 不是 rescued) ===
                with torch.no_grad():
                    if net._last_feat3_raw is not None:
                        feat3_raw = net._last_feat3_raw  # (B, dim, H, W)
                        h3_raw = F.adaptive_avg_pool2d(feat3_raw, 1).flatten(1)  # (B, dim)
                        # accumulator class-wise
                        self._round_local_proto3_sum.index_add_(0, labels, h3_raw)
                        self._round_local_proto3_count += torch.bincount(
                            labels, minlength=N_CLASSES).float()

                        # raw_to_target / rescued_to_target cos (★ Fix codex 复审: 改成
                        # 固定前 N batch 记录, 不再随机采样, 保证 R10 smoke / 前几轮 proto
                        # 没 ready 时也总有诊断数据)

                        if (batch_idx < getattr(self, '_ccc_fixed_batches', 2)
                                and self.global_proto3_unit is not None):
                            target_unit = net.global_proto3_unit_buf[labels]
                            target_norm = target_unit.norm(dim=-1)
                            valid = target_norm > 1e-8
                            if valid.any():
                                raw_unit = F.normalize(h3_raw, dim=-1, eps=1e-8)
                                rescued_unit = F.normalize(
                                    F.adaptive_avg_pool2d(net._last_feat3_rescued, 1).flatten(1),
                                    dim=-1, eps=1e-8
                                )
                                # raw的feat跟prototype的相似度
                                self._round_raw_to_target_cos.append(
                                    (raw_unit[valid] * target_unit[valid]).sum(-1).mean().item()
                                )
                                # 修正后的
                                self._round_rescued_to_target_cos.append(
                                    (rescued_unit[valid] * target_unit[valid]).sum(-1).mean().item()
                                )

                bs = labels.size(0)
                c_cc_loss_sum += cc_loss.item() * bs
                c_mag_loss_sum += mag_loss.item() * bs
                # ★ Fix (codex 复审): per-sample exceed 累 (而非 batch scalar > r_max)
                if self._cur_rho_t > 0:
                    c_mag_exceed_samples += exceed_count_this_batch
                    c_mag_eval_samples += bs
                    self._round_mag_ratio_p95.append(ratio_p95)
                    self._round_mag_ratio_max.append(ratio_max)
                # ★ L_orth 累计 (Digits 专用, lambda_orth=0 时不累)
                if getattr(self, 'dse_lambda_orth', 0.0) > 0 and self._cur_rho_t > 0:
                    self._round_orth_loss_sum += orth_loss.item() * bs
                    self._round_orth_cos_abs_sum += orth_cos_abs_batch * bs
                    self._round_orth_eval_samples += bs
                c_batches += 1
                global_loss += total_loss.item() * bs
                global_samples += bs
                num_c_samples = global_samples  # 保留最新值

        # 回写到 round 级累积
        self._round_cc_loss_sum += c_cc_loss_sum
        self._round_mag_loss_sum += c_mag_loss_sum
        self._round_mag_exceed_samples += c_mag_exceed_samples
        self._round_mag_eval_samples += c_mag_eval_samples
        self._round_total_batches += c_batches

        global_avg_loss = global_loss / max(global_samples, 1)
        return round(global_avg_loss, 3), num_c_samples

    def _print_round_diag(self):
        """收集 per-client dse diag + 加 round 级 metric, [DSE diag] 前缀 print."""
        # 收 per-client dse diag (delta_raw_ratio / delta_scaled_ratio / delta_cos_feat / rho_t)
        dse_diag_collect = []
        for k in self.online_clients:
            net = self.nets_list[k]
            if hasattr(net, 'get_dse_diag_summary'):
                d = net.get_dse_diag_summary()
                if d is not None:
                    dse_diag_collect.append(d)
        merged = {}
        if dse_diag_collect:
            keys = list(dse_diag_collect[0].keys())
            for k in keys:
                vals = [d[k] for d in dse_diag_collect if d.get(k) is not None]
                if vals:
                    merged[k] = float(np.mean(vals))

        # round 级补充
        merged['round'] = self.epoch_index
        merged['rho_t'] = self._cur_rho_t
        merged['lambda_cc_t'] = self._cur_lambda_cc_t
        if self._round_total_batches > 0:
            merged['cc_loss_mean'] = self._round_cc_loss_sum / max(
                sum(self.num_samples), 1)
            merged['mag_loss_mean'] = self._round_mag_loss_sum / max(
                sum(self.num_samples), 1)
            # ★ Fix (codex 复审): per-sample exceed rate (单 sample 爆但 batch 不爆也算)
            if self._round_mag_eval_samples > 0:
                merged['mag_exceed_rate'] = (
                    self._round_mag_exceed_samples / self._round_mag_eval_samples
                )
            else:
                merged['mag_exceed_rate'] = 0.0
        # ★ Fix #5: mag per-sample diag
        if self._round_mag_ratio_p95:
            merged['mag_ratio_p95_mean'] = float(np.mean(self._round_mag_ratio_p95))
        if self._round_mag_ratio_max:
            merged['mag_ratio_max_mean'] = float(np.mean(self._round_mag_ratio_max))
        # proto3 状态
        if self.global_proto3_raw is not None:
            valid_mask_cpu = (self._round_local_proto3_count > 0).cpu()
            valid = int(valid_mask_cpu.sum().item())
            total_classes = int(self.global_proto3_raw.size(0))
            merged['proto3_valid_classes'] = valid
            # ★ Fix #7: 改名 proto3_update_norm → proto3_mean_norm (诚实命名),
            # 加 proto3_ema_delta_norm (真"更新幅度", EMA 前后差)
            all_norms = self.global_proto3_raw.norm(dim=-1)
            merged['proto3_mean_norm'] = all_norms.mean().item()
            merged['proto3_ema_delta_norm'] = float(self._round_proto3_ema_delta_norm)
            # ★ Fix (codex 复审): proto3_mean_norm 包含本 round 没出现 class (norm=0 或 stale),
            # 解释起来不准. 加 valid_mean_norm (只算 count>0 class) + valid_ratio
            if valid > 0:
                merged['proto3_valid_mean_norm'] = all_norms[valid_mask_cpu].mean().item()
            else:
                merged['proto3_valid_mean_norm'] = 0.0
            merged['proto3_valid_ratio'] = valid / max(total_classes, 1)
        # raw vs rescued cos (CCC 改善证据)
        if self._round_raw_to_target_cos:
            merged['raw_to_target_cos_mean'] = float(np.mean(self._round_raw_to_target_cos))
        if self._round_rescued_to_target_cos:
            merged['rescued_to_target_cos_mean'] = float(np.mean(self._round_rescued_to_target_cos))
        if ('raw_to_target_cos_mean' in merged
                and 'rescued_to_target_cos_mean' in merged):
            merged['ccc_improvement'] = (
                merged['rescued_to_target_cos_mean']
                - merged['raw_to_target_cos_mean']
            )

        # ★ L_orth 诊断 (Digits 专用, lambda_orth=0 时不输出)
        if getattr(self, 'dse_lambda_orth', 0.0) > 0 and getattr(self, '_round_orth_eval_samples', 0) > 0:
            merged['orth_loss_mean'] = (
                getattr(self, '_round_orth_loss_sum', 0.0) / self._round_orth_eval_samples
            )
            merged['orth_cos_abs_mean'] = (
                getattr(self, '_round_orth_cos_abs_sum', 0.0) / self._round_orth_eval_samples
            )
            merged['lambda_orth'] = getattr(self, 'dse_lambda_orth', 0.0)

        # 排序 print (key 顺序)
        order = [
            'round', 'rho_t', 'lambda_cc_t', 'lambda_orth',
            'dse_delta_raw_ratio_mean', 'dse_delta_scaled_ratio_mean',
            'dse_delta_cos_feat_mean',
            'cc_loss_mean',
            'mag_loss_mean', 'mag_exceed_rate',
            'mag_ratio_p95_mean', 'mag_ratio_max_mean',  # ★ Fix #5
            'orth_loss_mean', 'orth_cos_abs_mean',  # ★ L_orth (Digits 专用)
            'proto3_valid_classes', 'proto3_valid_ratio',
            'proto3_mean_norm', 'proto3_valid_mean_norm', 'proto3_ema_delta_norm',  # ★ Fix #7 + 复审
            'raw_to_target_cos_mean', 'rescued_to_target_cos_mean',
            'ccc_improvement',
        ]
        printable = {k: merged.get(k) for k in order if merged.get(k) is not None}
        if printable:
            print('[DSE diag]', printable)

        # ★ Fix (codex 复审): merged 写到 self.proto_logs[], diagnostic.py heavy dump
        # 自动会拿 proto_logs[-1] 写 npz proto_diag_* + proto_logs.jsonl
        if merged:
            self.proto_logs.append(dict(merged))
