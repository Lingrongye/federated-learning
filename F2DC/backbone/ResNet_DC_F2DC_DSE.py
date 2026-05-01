"""
Pure F2DC + DSE_Rescue3 layer3 修正
========================================
基于纯 F2DC ResNet (DFD + DFC at layer4), 在 layer3 后加 1 个轻量 rescue adapter
做 progressive domain shift correction. 不依赖 PG-DFC / prototype attention / aux3.

设计原则 (重要):
  1. layer4 主路完全等价 F2DC vanilla (DFD + DFC, 不动)
  2. layer3 后加 DSE_Rescue3: 1×1 reduce → GroupNorm/ReLU → depthwise 3×3 → 1×1 expand
     - GroupNorm 而不是 BN: 不维护 running stats, train/eval 一致, 不需 server 聚合白名单
     - 1×1 expand zero-init: 训练初期 delta3=0, 不破坏 layer4 input
  3. feat3_rescued = feat3_raw + rho_t * delta3
     - rho_t 由 trainer 注入 (warmup + ramp), 默认 0
  4. ★ proto3 用 raw feat3 算 (不是 rescued!), 提供 explicit "去域" 监督方向

Forward 接口跟 F2DC 一致: 返回 7-tuple (out, feat, ro_outputs, re_outputs,
rec_outputs, ro_flat, re_flat). 不变, 兼容 utils/training.global_evaluate.

Transient attributes (训练时被 trainer 读取, eval 时不需):
  - self._last_feat3_raw     : layer3 输出 (B, 256, H, W) — proto3 update + raw_cos diag
  - self._last_feat3_rescued : feat3 + rho_t * delta3       — CCC input + rescued_cos diag
  - self._last_delta3        : DSE_Rescue3 输出 (B, 256, H, W) — magnitude diag

Buffer:
  - self.rho_t                    : 当前 ramp 系数 (trainer set_rho_t 注入)
  - self.global_proto3_unit_buf   : server 下发的 layer3 class proto (L2-norm), 给 CCC 算 target

Diagnostic (round 内 10% sample 累积, get_dse_diag_summary 取):
  - delta_raw_ratio    = ||delta3|| / ||feat3_raw||
  - delta_scaled_ratio = ||rho_t * delta3|| / ||feat3_raw||
  - delta_cos_feat     = cos(delta3 flatten, feat3_raw flatten)
  - rho_t              = 当前 round rho_t

Shape 速查 (PACS image_size=128):
  - layer1 → 64×128×128
  - layer2 → 128×64×64
  - layer3 → 256×32×32  ← DSE_Rescue3 接这里
  - layer4 → 512×16×16  ← 收 feat3_rescued

Office/Digits image_size=32:
  - layer3 → 256×8×8
  - layer4 → 512×4×4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.ResNet_DC import (
    BasicBlock, ResNet, DFD, DFC, compute_mask_stats
)


class DSE_Rescue3(nn.Module):
    """layer3 后的 domain shift rescue adapter (FDSE-inspired bottleneck).

    架构: 1×1 reduce → GroupNorm/ReLU → depthwise 3×3 → 1×1 expand (zero-init)
    输出: delta3 (B, C, H, W), 跟 feat3_raw 同 shape

    Bottleneck: 256 → 32 (reduction=8) → 32 → 256, 强制 low-rank domain shift residual,
    防止 delta 重写整个 feature 主语义.

    GroupNorm 而非 BatchNorm:
      - federated 场景下 BN running stats 跨 round 漂移, train/eval 不一致
      - GN 用单 batch instance-level normalization, 不依赖 batch stats, 完全一致
      - 不需要在 server 聚合白名单加 dse_rescue3.bn.running_*

    Zero-init expand:
      - training 初始 delta3 = 0, layer4 收到的就是 feat3_raw
      - 等价 F2DC vanilla, backbone 先正常学 5 round
      - rho_t ramp 起来后 DSE 才开始作用
    """
    def __init__(self, channels=256, reduction=8, dw_size=3):
        super().__init__()
        # 至少 16 channel 给 GN 留点 capacity
        mid = max(channels // reduction, 16)
        self.channels = channels
        self.mid = mid
        # 1×1 reduce (无 bias, GN 自带 affine)
        self.reduce = nn.Conv2d(channels, mid, 1, bias=False)
        # GroupNorm: num_groups 取 min(8, mid) 保证 mid % num_groups == 0
        num_groups = min(8, mid)
        while mid % num_groups != 0:
            num_groups -= 1
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=mid)
        # depthwise 3×3 (groups=mid, per-channel spatial 修正)
        self.dw = nn.Conv2d(mid, mid, dw_size, padding=dw_size // 2,
                             groups=mid, bias=False)
        # 1×1 expand (zero-init weight, 训练初始 delta=0)
        self.expand = nn.Conv2d(mid, channels, 1, bias=False)
        nn.init.zeros_(self.expand.weight)

    def forward(self, feat3):
        x = self.reduce(feat3)
        x = F.relu(self.gn(x), inplace=True)
        x = self.dw(x)
        delta3 = self.expand(x)
        return delta3


class ResNet_F2DC_DSE(ResNet):
    """ResNet (F2DC base) + DSE_Rescue3 layer3 修正.

    继承 F2DC ResNet (有 DFD + DFC at layer4), 重写 forward 在 layer3 后插入
    DSE_Rescue3 修正. layer4 主路 + DFD + DFC + linear 完全不变.

    rho_t 由 trainer 通过 set_rho_t(rho) 注入 (默认 0 = DSE 不影响主路).
    global_proto3_unit_buf 由 trainer 通过 set_global_proto3_unit() 注入 (CCC target).
    """
    def __init__(self, block, num_blocks, num_classes=10, tau=0.1,
                 image_size=(32, 32), name='f2dc_dse',
                 dse_reduction=8, dse_dw_size=3):
        super().__init__(block, num_blocks, num_classes=num_classes,
                         tau=tau, image_size=image_size, name=name)
        # layer3 输出 channel = 256 * block.expansion (BasicBlock=1)
        C3 = 256 * block.expansion
        self.feat3_dim = C3
        self.dse_rescue3 = DSE_Rescue3(channels=C3, reduction=dse_reduction,
                                        dw_size=dse_dw_size)
        # rho_t buffer (trainer 注入)
        self.register_buffer('rho_t', torch.tensor(0.0), persistent=False)
        # global_proto3_unit (trainer 注入, CCC target)
        # 注意: 第一轮 server 还没聚合, 是 None. trainer 会在 _aggregate_proto3 后 set_global_proto3_unit
        self.register_buffer('global_proto3_unit_buf',
                             torch.zeros(num_classes, C3),
                             persistent=False)
        # transient (forward 时写, trainer 抓走)
        self._last_feat3_raw = None
        self._last_feat3_rescued = None
        self._last_delta3 = None
        # 诊断 buffer (round 内 10% sample 累积)
        self._diag_dse_stats = []

    def set_rho_t(self, rho):
        """trainer 每 round 开始注入当前 ramp 系数."""
        self.rho_t.fill_(float(rho))

    def set_global_proto3_unit(self, proto_unit_tensor):
        """trainer 每 round 末聚合 proto3 后, 同步 L2-norm proto 给 client."""
        self.global_proto3_unit_buf.data.copy_(proto_unit_tensor.to(self.global_proto3_unit_buf.device))

    def reset_dse_diag(self):
        self._diag_dse_stats = []

    def get_dse_diag_summary(self):
        if not self._diag_dse_stats:
            return None
        import numpy as np
        keys = list(self._diag_dse_stats[0].keys())
        return {f'dse_{k}_mean': float(np.mean([s[k] for s in self._diag_dse_stats]))
                for k in keys}

    def forward(self, x, is_eval=False):
        r_outputs = []
        nr_outputs = []
        rec_outputs = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        feat3_raw = self.layer3(out)

        # ★ DSE_Rescue3: layer3 后的轻量修正
        delta3 = self.dse_rescue3(feat3_raw)
        feat3_rescued = feat3_raw + self.rho_t * delta3

        # transient: trainer 抓 feat3_raw (proto3 update + raw_cos diag) +
        # rescued (CCC input + rescued_cos diag) + delta3 (mag diag)
        self._last_feat3_raw = feat3_raw
        self._last_feat3_rescued = feat3_rescued
        self._last_delta3 = delta3

        # 诊断 (10% sample, 训练时, 不影响速度)
        if self.training and torch.rand(1).item() < 0.1:
            with torch.no_grad():
                feat_norm = feat3_raw.norm()
                if feat_norm > 1e-8:
                    delta_norm = delta3.norm()
                    scaled_norm = (self.rho_t * delta3).norm()
                    raw_ratio = (delta_norm / feat_norm).item()
                    scaled_ratio = (scaled_norm / feat_norm).item()
                    # cosine of delta3 跟 feat3_raw (per-sample, 然后 mean)
                    cos_feat = F.cosine_similarity(
                        delta3.flatten(1), feat3_raw.flatten(1), dim=-1
                    ).mean().item()
                    self._diag_dse_stats.append({
                        'delta_raw_ratio': raw_ratio,
                        'delta_scaled_ratio': scaled_ratio,
                        'delta_cos_feat': cos_feat,
                        'rho_t': self.rho_t.item(),
                    })

        # 主路 layer4 用 rescued feat
        out = self.layer4(feat3_rescued)
        # 标准 F2DC: DFD → r_feat / nr_feat / mask
        r_feat, nr_feat, mask = self.dfd_module(out, is_eval=is_eval)
        ro_flat = nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1)
        re_flat = nn.AdaptiveAvgPool2d(1)(nr_feat).reshape(nr_feat.shape[0], -1)
        r_outputs.append(self.aux(ro_flat))
        nr_outputs.append(self.aux(re_flat))
        # 标准 F2DC: DFC 重建
        rec_feat = self.dfc_module(nr_feat, mask)
        rec_out = self.aux(nn.AdaptiveAvgPool2d(1)(rec_feat).reshape(rec_feat.shape[0], -1))
        rec_outputs.append(rec_out)
        out = r_feat + rec_feat
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        feat = out
        out = self.linear(out)
        return out, feat, r_outputs, nr_outputs, rec_outputs, ro_flat, re_flat


def resnet10_f2dc_dse(num_classes=7, gum_tau=0.1, dse_reduction=8):
    return ResNet_F2DC_DSE(BasicBlock, [1, 1, 1, 1], num_classes=num_classes,
                           tau=gum_tau, image_size=(128, 128),
                           dse_reduction=dse_reduction)


def resnet10_f2dc_dse_office(num_classes=10, gum_tau=0.1, dse_reduction=8):
    return ResNet_F2DC_DSE(BasicBlock, [1, 1, 1, 1], num_classes=num_classes,
                           tau=gum_tau, image_size=(32, 32),
                           dse_reduction=dse_reduction)


def resnet10_f2dc_dse_digits(num_classes=10, gum_tau=0.1, dse_reduction=8):
    return ResNet_F2DC_DSE(BasicBlock, [1, 1, 1, 1], num_classes=num_classes,
                           tau=gum_tau, image_size=(32, 32),
                           dse_reduction=dse_reduction)
