"""
PG-DFC (Prototype-Guided DFC) v3.2 backbone
============================================
基于 F2DC ResNet_DC.py 改造, 加 prototype-guided DFC.

v3.2 (基于 4 轮 review) 核心 fix:
  - NV1: cosine attention (q/k 都 L2-normalize)
  - NV2: proto 路径 mask 预先 detach
  - RV1: query 走 nr_feat.detach() 阻断梯度反传到 mask
  - 第四轮专家 review: query 用 r_pooled (非 nr_pooled) — robust feature 当 query 才有信号
  - m2: register_buffer(persistent=False)
  - 安全: F.normalize 加 norm 阈值检查, 防 NaN

向后兼容: proto_weight=0 时退化等价 F2DC 原版.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# 复用 F2DC 原 DFD 和 BasicBlock(避免重复定义)
from backbone.gumbel_sigmoid import GumbelSigmoid
from backbone.ResNet_DC import DFD, BasicBlock, Bottleneck


class DFC_PG(nn.Module):
    """
    Prototype-Guided DFC v3.2.
    proto_weight=0 时完全等价 F2DC 原版 DFC.
    """
    def __init__(self, size, num_classes, num_channel=64,
                 proto_weight=0.3, attn_temperature=0.3):
        super().__init__()
        C, H, W = size
        self.C, self.H, self.W = C, H, W
        self.num_classes = num_classes
        self.proto_weight = proto_weight
        self.attn_temperature = attn_temperature

        # 路径 1: 原 F2DC conv 残差 (保留, 100% 等价)
        self.net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

        # 路径 2: prototype attention (新增)
        self.q_proj = nn.Linear(C, C)
        self.k_proj = nn.Linear(C, C)
        self.v_proj = nn.Linear(C, C)

        # class_proto buffer: 由 server 下发, persistent=False 防 FedAvg 误聚合 (m2 fix)
        self.register_buffer('class_proto', torch.zeros(num_classes, C),
                             persistent=False)

        # 诊断 buffers (训练时记录, 不参与 backprop)
        self._diag_mask_sparsity = []
        self._diag_attn_entropy = []
        self._diag_proto_signal_ratio = []

    def set_proto_weight(self, w):
        """warmup ramp 调用"""
        self.proto_weight = w

    def reset_diag(self):
        """每 round 末重置诊断 buffer"""
        self._diag_mask_sparsity = []
        self._diag_attn_entropy = []
        self._diag_proto_signal_ratio = []

    def get_diag_summary(self):
        """返回 round 内诊断指标的 mean (供 NOTE.md / log)"""
        if not self._diag_mask_sparsity:
            return None
        import numpy as np
        return dict(
            mask_sparsity_mean=float(np.mean(self._diag_mask_sparsity)),
            mask_sparsity_std=float(np.std(self._diag_mask_sparsity)),
            attn_entropy_mean=float(np.mean(self._diag_attn_entropy)) if self._diag_attn_entropy else None,
            proto_signal_ratio_mean=float(np.mean(self._diag_proto_signal_ratio)) if self._diag_proto_signal_ratio else None,
        )

    def forward(self, nr_feat, mask, r_feat=None):
        """
        nr_feat: (B, C, H, W) non-robust feature
        mask:    (B, C, H, W) DFD mask
        r_feat:  (B, C, H, W) robust feature (NEW v3.2 — query 用 r_pooled)
        """
        B, C, H, W = nr_feat.shape

        # 诊断: mask sparsity (1% 采样, 不影响速度)
        if self.training and torch.rand(1).item() < 0.01:
            with torch.no_grad():
                self._diag_mask_sparsity.append(mask.mean().item())

        # ============================================================
        # 路径 1: 原 F2DC conv 残差 (保留)
        # ============================================================
        rec_units = self.net(nr_feat)

        # ============================================================
        # 路径 2: prototype-guided attention (v3.2 核心)
        # ============================================================
        if self.proto_weight > 0 and self.class_proto.abs().sum() > 1e-6:
            # 第四轮 review 关键 fix: query 来自 r_feat (不是 nr_feat)
            # 语义: 我用 robust 部分去字典确认/强化该类的标准方向
            #       不是用噪声向量去匹配 (那样 attention 学不到东西)
            # 同时加 .detach() 阻断梯度反传到 mask (RV1 fix)
            query_source = r_feat if r_feat is not None else nr_feat
            q_pooled = F.adaptive_avg_pool2d(query_source.detach(), 1).reshape(B, C)
            q = self.q_proj(q_pooled)                              # (B, C)

            # k/v 来自 class_proto (buffer, 不需 detach)
            k = self.k_proj(self.class_proto)                       # (K, C)
            v = self.v_proj(self.class_proto)                       # (K, C)

            # NV1 fix: cosine attention (消除 magnitude 错配, 同时解决 M4)
            q_norm = F.normalize(q, dim=-1, eps=1e-8)               # unit
            k_norm = F.normalize(k, dim=-1, eps=1e-8)               # unit
            attn_logits = (q_norm @ k_norm.T) / self.attn_temperature  # (B, K)
            attn = F.softmax(attn_logits, dim=-1)                   # (B, K)

            # 诊断: attention entropy
            if self.training and torch.rand(1).item() < 0.01:
                with torch.no_grad():
                    entropy = -(attn * (attn + 1e-9).log()).sum(-1).mean().item()
                    max_ent = float(torch.log(torch.tensor(attn.size(-1), dtype=torch.float)).item())
                    self._diag_attn_entropy.append(entropy / max_ent)  # 归一化 [0, 1]

            # 加权 select v
            proto_clue = attn @ v                                   # (B, C)
            proto_clue = proto_clue.reshape(B, C, 1, 1).expand(-1, -1, H, W)

            # NV2 fix: proto 路径上 mask 预先 detach (阻断 proto → mask 反传)
            # rec_units 路径保留 mask 梯度 (F2DC 原版梯度流不变)
            mask_for_proto = mask.detach()

            # 诊断: proto_signal / rec_units 量级比 (M3 验证)
            if self.training and torch.rand(1).item() < 0.01:
                with torch.no_grad():
                    proto_mag = (self.proto_weight * proto_clue.abs().mean()).item()
                    rec_mag = rec_units.abs().mean().item()
                    self._diag_proto_signal_ratio.append(proto_mag / max(rec_mag, 1e-8))

            rec_feat = nr_feat \
                + (1 - mask) * rec_units \
                + (1 - mask_for_proto) * self.proto_weight * proto_clue
        else:
            # warmup 期 / class_proto 还没初始化 → 完全等价 F2DC 原版
            rec_feat = nr_feat + (1 - mask) * rec_units

        return rec_feat


class ResNet_PG(nn.Module):
    """ResNet + DFD + DFC_PG (PG-DFC v3.2)"""
    def __init__(self, block, num_blocks, num_classes=10, tau=0.1, image_size=(32, 32),
                 name='f2dc_pg', proto_weight=0.3, attn_temperature=0.3):
        super().__init__()
        self.name = name
        self.in_planes = 64
        self.tau = tau
        self.image_size = image_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        H_out, W_out = int(image_size[0] / 8), int(image_size[1] / 8)
        self.dfd_module = DFD(size=(512, H_out, W_out), tau=self.tau)
        self.dfc_module = DFC_PG(size=(512, H_out, W_out),
                                  num_classes=num_classes,
                                  proto_weight=proto_weight,
                                  attn_temperature=attn_temperature)
        self.aux = nn.Sequential(nn.Linear(512, num_classes))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, is_eval=False):
        r_outputs = []
        nr_outputs = []
        rec_outputs = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        r_feat, nr_feat, mask = self.dfd_module(out, is_eval=is_eval)
        ro_flat = torch.nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1)
        re_flat = torch.nn.AdaptiveAvgPool2d(1)(nr_feat).reshape(nr_feat.shape[0], -1)
        r_outputs.append(self.aux(ro_flat))
        nr_outputs.append(self.aux(re_flat))

        # ★ v3.2 关键改动: 把 r_feat 也传给 DFC, 用 r_pooled 当 query
        rec_feat = self.dfc_module(nr_feat, mask, r_feat=r_feat)
        rec_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(rec_feat).reshape(rec_feat.shape[0], -1))
        rec_outputs.append(rec_out)

        out = r_feat + rec_feat
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        feat = out
        out = self.linear(out)
        return out, feat, r_outputs, nr_outputs, rec_outputs, ro_flat, re_flat


def resnet10_dc_pg(num_classes=7, gum_tau=0.1,
                   proto_weight=0.3, attn_temperature=0.3):
    return ResNet_PG(BasicBlock, [1, 1, 1, 1], num_classes=num_classes,
                     tau=gum_tau, image_size=(128, 128),
                     proto_weight=proto_weight, attn_temperature=attn_temperature)


def resnet10_dc_pg_office(num_classes=10, gum_tau=0.1,
                          proto_weight=0.3, attn_temperature=0.3):
    return ResNet_PG(BasicBlock, [1, 1, 1, 1], num_classes=num_classes,
                     tau=gum_tau, image_size=(32, 32),
                     proto_weight=proto_weight, attn_temperature=attn_temperature)
