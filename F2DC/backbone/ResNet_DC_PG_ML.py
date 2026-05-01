"""
Multi-Layer PG-DFC backbone (ResNet_DC_PG_ML)
=============================================
基于 ResNet_DC_PG.py (PG-DFC v3.2/v3.3) 加 layer3 deep supervision lite 分支.

设计原则 (重要):
  1. layer4 主路完全等价 PG-DFC v3.3 — 用原 feat3 喂 layer4, 不用 cleaned 版本
  2. layer3 lite 分支 (DFD_lite + DFC_lite + aux3) 只产生 aux3 logits 给 trainer
     做 deep supervision loss (α·CE), 不影响 layer4 / linear / 主预测
  3. ml_aux_alpha=0 时整个 lite 分支没梯度 → 退化成 PG-DFC v3.3 (acc 不会更差)
  4. dfd_lite/dfc_lite/aux3 的参数走 F2DC 默认全 state_dict FedAvg 聚合
     (跟 dfc_module/dfd_module/linear/aux 一致, 不引入差异化聚合)

Forward 接口:
  - 仍返回原 7-tuple (out, feat, ro_outputs, re_outputs, rec_outputs, ro_flatten, re_flatten)
    保持跟 utils/training.py global_evaluate / utils/diagnostic.py 兼容
  - aux3 logits 通过 module attribute self._last_aux3_logits 暴露给 trainer
  - is_eval=True 时 lite 分支也走 deterministic gumbel (mask 用 0.5 noise)

Shape 速查 (PACS image_size=128):
  - layer1 → 64×128×128       (stride 1)
  - layer2 → 128×64×64        (stride 2)
  - layer3 → 256×32×32  ←★ lite 分支接这里
  - layer4 → 512×16×16        (stride 2, 喂主 PG-DFC)

Office/Digits image_size=32:
  - layer3 → 256×8×8
  - layer4 → 512×4×4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.gumbel_sigmoid import GumbelSigmoid
from backbone.ResNet_DC import BasicBlock, compute_mask_stats
from backbone.ResNet_DC_PG import ResNet_PG, DFC_PG


class DFD_lite(nn.Module):
    """layer3 mask 切分 lite 版.

    跟 ResNet_DC.DFD 同结构, 但 num_channel 默认 32 (vs DFD 的 64).
    PACS 上 layer3 的 spatial 是 32×32, 比 layer4 的 16×16 大 4 倍.

    tau 历史:
      - 旧版 tau=0.5 (软 GumbelSigmoid, 怕 collapse) → 实测 pre_mask3 hard% < 1%,
        mid% > 80%, 完全卡 0.5, mask3 没真学 (EXP-141 v2 R0/R1 验证)
      - 新版 tau=0.1 (跟 layer4 DFD tau 一致, 强 push 0/1) → 预期 pre_mask3 真二值化
        (EXP-141 v3 验证)
    """
    def __init__(self, size, num_channel=32, tau=0.1, diag_sample_rate=0.1):
        super().__init__()
        C, H, W = size
        self.C, self.H, self.W = C, H, W
        self.tau = tau
        # 诊断采样率: I2 fix — 1% 在 R10 短跑可能采不到样本, 改 10% 确保有数据
        self.diag_sample_rate = diag_sample_rate
        self.net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False),
        )
        # 诊断 buffers: pre-Gumbel + post-Gumbel mask3 7-stat
        self._diag_pre_mask_stats = []
        self._diag_mask_stats = []

    def reset_diag(self):
        self._diag_pre_mask_stats = []
        self._diag_mask_stats = []

    def get_diag_summary(self):
        """返回 round 内 mask3 7-stat mean (跨 batch).

        新增 pre-Gumbel mask3 概率诊断 (★ 真正反映模型学到):
          pre_mask3_*_mean : sigmoid(rob_map) 的概率分布 7-stat
        post-Gumbel mask3 (训练时实际用):
          mask3_*_mean : GumbelSigmoid 后 7-stat (受 tau 噪声影响, hard_ratio 天然 50%)
        backward compat: mask3_sparsity_mean / mask3_sparsity_std.
        """
        if not self._diag_pre_mask_stats:
            return None
        import numpy as np
        keys = list(self._diag_pre_mask_stats[0].keys())
        agg = {}
        # pre-Gumbel
        for k in keys:
            agg[f'pre_mask3_{k}_mean'] = float(np.mean([s[k] for s in self._diag_pre_mask_stats]))
        # post-Gumbel (跟之前的 mask3_*_mean 保持兼容)
        if self._diag_mask_stats:
            for k in keys:
                agg[f'mask3_{k}_mean'] = float(np.mean([s[k] for s in self._diag_mask_stats]))
            agg['mask3_sparsity_mean'] = agg.get('mask3_mean_mean')
            agg['mask3_sparsity_std'] = float(np.std([s['mean'] for s in self._diag_mask_stats]))
        return agg

    def forward(self, feat, is_eval=False):
        rob_map = self.net(feat)
        mask = rob_map.reshape(rob_map.shape[0], 1, -1)
        mask = torch.sigmoid(mask)
        # ★ 同一次采样 pre + post mask 形态 (10% 采样, R10 smoke 能采到)
        do_diag = self.training and torch.rand(1).item() < self.diag_sample_rate
        if do_diag:
            pre_4d = mask[:, 0].reshape(mask.shape[0], self.C, self.H, self.W)
            self._diag_pre_mask_stats.append(compute_mask_stats(pre_4d))
        mask = GumbelSigmoid(tau=self.tau)(mask, is_eval=is_eval)
        mask = mask[:, 0].reshape(mask.shape[0], self.C, self.H, self.W)
        if do_diag:
            self._diag_mask_stats.append(compute_mask_stats(mask))
        r_feat = feat * mask
        nr_feat = feat * (1 - mask)
        return r_feat, nr_feat, mask


class DFC_lite(nn.Module):
    """layer3 重建 lite 版 — bottleneck 结构 (1×1 → 3×3 → 1×1) 不带 prototype.

    浅一点的 layer3 还没成熟语义, 拿 class prototype 查 attention 没参考意义,
    所以这里只走 conv 残差 (类似 FDSE 的"擦"思路).
    """
    def __init__(self, size, num_channel=32):
        super().__init__()
        C, H, W = size
        self.C, self.H, self.W = C, H, W
        self.net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=1, bias=False),
        )

    def forward(self, nr_feat, mask):
        rec_units = self.net(nr_feat)
        rec_feat = nr_feat + (1 - mask) * rec_units
        return rec_feat


class ResNet_PG_ML(ResNet_PG):
    """ResNet + DFD + PG-DFC + layer3 lite 分支.

    继承 ResNet_PG, 新增:
      - dfd_lite, dfc_lite, aux3 (额外 module)
      - 重写 forward: layer3 出来后岔出去算 lite 分支, 主路用原 feat3 喂 layer4
      - module attribute self._last_aux3_logits 给 trainer 做 deep sup loss
    """
    def __init__(self, block, num_blocks, num_classes=10, tau=0.1, image_size=(32, 32),
                 name='f2dc_pg_ml', proto_weight=0.3, attn_temperature=0.3,
                 ml_lite_channel=32, ml_lite_tau=0.1, ml_main_rho=0.0):
        super().__init__(block, num_blocks, num_classes=num_classes, tau=tau,
                         image_size=image_size, name=name,
                         proto_weight=proto_weight, attn_temperature=attn_temperature)
        # layer3 输出 shape: 256ch, image_size/4 spatial
        H3 = max(1, int(image_size[0] / 4))
        W3 = max(1, int(image_size[1] / 4))
        C3 = 256 * block.expansion
        self.dfd_lite = DFD_lite(size=(C3, H3, W3), num_channel=ml_lite_channel, tau=ml_lite_tau)
        self.dfc_lite = DFC_lite(size=(C3, H3, W3), num_channel=ml_lite_channel)
        self.aux3 = nn.Linear(C3, num_classes)
        # ml_main_rho: feat3_clean 注入主路比例 (0=不注入 等价旧 design, 0.1-0.2 弱注入,
        # 让 mask3 接通 main loss 梯度. EXP-141 v3 验证 tau ablation 失败后启用)
        self.ml_main_rho = ml_main_rho
        # transient attributes — 不存 state_dict, 训练时被 trainer 读取
        self._last_aux3_logits = None
        self._last_mask3 = None

    def forward(self, x, is_eval=False):
        r_outputs = []
        nr_outputs = []
        rec_outputs = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        feat3 = self.layer3(out)

        # ★ layer3 lite 分支 — deep supervision only, 不污染主路
        r3, nr3, mask3 = self.dfd_lite(feat3, is_eval=is_eval)
        rec3 = self.dfc_lite(nr3, mask3)
        feat3_clean = r3 + rec3
        feat3_pooled = nn.AdaptiveAvgPool2d(1)(feat3_clean).reshape(feat3_clean.shape[0], -1)
        # transient — 只在当前 forward 有效, 下次 forward 会覆盖
        self._last_aux3_logits = self.aux3(feat3_pooled)
        self._last_mask3 = mask3

        # ★ layer4 主路 — rho=0 等价旧 design, rho>0 弱注入 cleaned 让 mask3 接通 main loss 梯度
        if self.ml_main_rho > 0:
            feat3_main = feat3 + self.ml_main_rho * (feat3_clean - feat3)
        else:
            feat3_main = feat3
        out = self.layer4(feat3_main)

        r_feat, nr_feat, mask = self.dfd_module(out, is_eval=is_eval)
        ro_flat = nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1)
        re_flat = nn.AdaptiveAvgPool2d(1)(nr_feat).reshape(nr_feat.shape[0], -1)
        r_outputs.append(self.aux(ro_flat))
        nr_outputs.append(self.aux(re_flat))

        # PG-DFC 走 r_feat 当 query
        rec_feat = self.dfc_module(nr_feat, mask, r_feat=r_feat)
        rec_out = self.aux(nn.AdaptiveAvgPool2d(1)(rec_feat).reshape(rec_feat.shape[0], -1))
        rec_outputs.append(rec_out)

        out = r_feat + rec_feat
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        feat = out
        out = self.linear(out)
        return out, feat, r_outputs, nr_outputs, rec_outputs, ro_flat, re_flat


def resnet10_dc_pg_ml(num_classes=7, gum_tau=0.1,
                      proto_weight=0.3, attn_temperature=0.3,
                      ml_lite_channel=32, ml_lite_tau=0.1, ml_main_rho=0.0):
    return ResNet_PG_ML(BasicBlock, [1, 1, 1, 1], num_classes=num_classes,
                        tau=gum_tau, image_size=(128, 128),
                        proto_weight=proto_weight, attn_temperature=attn_temperature,
                        ml_lite_channel=ml_lite_channel, ml_lite_tau=ml_lite_tau,
                        ml_main_rho=ml_main_rho)


def resnet10_dc_pg_ml_office(num_classes=10, gum_tau=0.1,
                             proto_weight=0.3, attn_temperature=0.3,
                             ml_lite_channel=32, ml_lite_tau=0.1, ml_main_rho=0.0):
    return ResNet_PG_ML(BasicBlock, [1, 1, 1, 1], num_classes=num_classes,
                        tau=gum_tau, image_size=(32, 32),
                        proto_weight=proto_weight, attn_temperature=attn_temperature,
                        ml_lite_channel=ml_lite_channel, ml_lite_tau=ml_lite_tau,
                        ml_main_rho=ml_main_rho)


def resnet10_dc_pg_ml_digits(num_classes=10, gum_tau=0.1,
                             proto_weight=0.3, attn_temperature=0.3,
                             ml_lite_channel=32, ml_lite_tau=0.1, ml_main_rho=0.0):
    return ResNet_PG_ML(BasicBlock, [1, 1, 1, 1], num_classes=num_classes,
                        tau=gum_tau, image_size=(32, 32),
                        proto_weight=proto_weight, attn_temperature=attn_temperature,
                        ml_lite_channel=ml_lite_channel, ml_lite_tau=ml_lite_tau,
                        ml_main_rho=ml_main_rho)
