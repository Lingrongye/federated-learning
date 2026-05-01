import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.gumbel_sigmoid import GumbelSigmoid


@torch.no_grad()
def compute_mask_stats(mask: torch.Tensor) -> dict:
    """7-stat mask diagnostic over a single batch tensor (B, C, H, W) ∈ [0, 1].

    Returns: dict with keys
      - mean        : 全 element 平均 (旧 sparsity)
      - unit_std    : 单 batch 内全 element std (>0.2 真 selective, <0.05 接近常数)
      - hard_ratio  : (mask<0.1 OR >0.9) 比例 (>0.7 真二值化)
      - mid_ratio   : (0.4<mask<0.6) 比例 (>0.7 卡 0.5)
      - sample_std  : per-sample mean 的 std (>0.05 样本间真区分)
      - channel_std : per-channel mean 的 std (>0.1 通道真区分)
      - spatial_std : per-spatial-position mean 的 std (>0.05 位置真区分)
    """
    m = mask.detach().float()
    return dict(
        mean=m.mean().item(),
        unit_std=m.std().item(),
        hard_ratio=((m < 0.1) | (m > 0.9)).float().mean().item(),
        mid_ratio=((m > 0.4) & (m < 0.6)).float().mean().item(),
        sample_std=m.mean(dim=[1, 2, 3]).std().item() if m.size(0) > 1 else 0.0,
        channel_std=m.mean(dim=[0, 2, 3]).std().item() if m.size(1) > 1 else 0.0,
        spatial_std=m.mean(dim=[0, 1]).std().item() if (m.size(2) > 1 and m.size(3) > 1) else 0.0,
    )


# feature decoupling
class DFD(torch.nn.Module):
    def __init__(self, size, num_channel=64, tau=0.1, diag_sample_rate=0.1):
        super(DFD, self).__init__()
        C, H, W = size
        self.C, self.H, self.W = C, H, W
        self.tau = tau
        self.diag_sample_rate = diag_sample_rate
        # 一个小的cnn标准的，用来预测每个feature unit是否robust
        self.net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )
        # 诊断 buffers (pre-Gumbel + post-Gumbel mask 的 7-stat)
        self._diag_pre_mask_stats = []   # sigmoid(rob_map) 概率 (Gumbel 噪声前)
        self._diag_post_mask_stats = []  # GumbelSigmoid(...) 后的 mask (实际用)

    def reset_diag(self):
        self._diag_pre_mask_stats = []
        self._diag_post_mask_stats = []

    def get_diag_summary(self):
        """返回 pre-Gumbel 跟 post-Gumbel mask 7-stat (各自跨 batch mean).
        pre_mask_*_mean : sigmoid(rob_map) 的概率分布 7-stat (★ 真正反映模型学到的)
        mask_*_mean     : post-Gumbel mask (训练时实际用的, 受 tau noise 影响)
        """
        if not self._diag_pre_mask_stats:
            return None
        import numpy as np
        keys = list(self._diag_pre_mask_stats[0].keys())
        agg = {}
        # pre-Gumbel
        for k in keys:
            agg[f'pre_mask_{k}_mean'] = float(np.mean([s[k] for s in self._diag_pre_mask_stats]))
        # post-Gumbel
        if self._diag_post_mask_stats:
            for k in keys:
                agg[f'mask_{k}_mean'] = float(np.mean([s[k] for s in self._diag_post_mask_stats]))
            agg['mask_sparsity_mean'] = agg.get('mask_mean_mean')
            agg['mask_sparsity_std'] = float(np.std([s['mean'] for s in self._diag_post_mask_stats]))
        return agg

    def forward(self, feat, is_eval=False):
        rob_map = self.net(feat)
        # 先生成一个robust map,还要去sigmoid到【0，1】
        mask = rob_map.reshape(rob_map.shape[0], 1, -1)
        mask = torch.nn.Sigmoid()(mask)

        # ★ 同一次采样 pre + post mask 形态 (训练时 sample, eval 不采)
        do_diag = self.training and torch.rand(1).item() < self.diag_sample_rate
        if do_diag:
            pre_4d = mask[:, 0].reshape(mask.shape[0], self.C, self.H, self.W)
            self._diag_pre_mask_stats.append(compute_mask_stats(pre_4d))

        # 把连续的概率通过一个gumbel层
        mask = GumbelSigmoid(tau=self.tau)(mask, is_eval=is_eval)
        # 取第0个就是属于robust的概率
        mask = mask[:, 0].reshape(mask.shape[0], self.C, self.H, self.W)
        if do_diag:
            self._diag_post_mask_stats.append(compute_mask_stats(mask))
        # mask接近于1的位置，就是r_feat
        r_feat = feat * mask
        nr_feat = feat * (1 - mask)

        return r_feat, nr_feat, mask

# 不要直接去丢掉no_rb的信息，还可以进行恢复跟校准
class DFC(nn.Module):
    def __init__(self, size, num_channel=64):
        super(DFC, self).__init__()
        C, H, W = size
        # 仍然是一个小的cnn
        self.net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, nr_feat, mask):
        # 通过这个小的cnn去生成一个校正后的feature
        rec_units = self.net(nr_feat)
        # 确保只拿到r_feature
        rec_units = rec_units * (1 - mask)
        # 相加得到重建特征
        rec_feat = nr_feat + rec_units
        return rec_feat


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, tau=0.1, image_size=(32, 32), name='f2dc'):
        super(ResNet, self).__init__()
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

        self.dfd_module = DFD(size=(512, int(self.image_size[0] / 8), 
                                           int(self.image_size[1] / 8)), tau=self.tau)
        self.dfc_module = DFC(size=(512, int(self.image_size[0] / 8), 
                                                 int(self.image_size[1] / 8))) 
        # 这是一个辅助的分类器
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
        # print(out.shape)
        # 只在第四层
        r_feat, nr_feat, mask = self.dfd_module(out, is_eval=is_eval)
        # 池化后
        ro_flat = torch.nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1)
        re_flat = torch.nn.AdaptiveAvgPool2d(1)(nr_feat).reshape(nr_feat.shape[0], -1)
        r_outputs.append(self.aux(ro_flat))
        nr_outputs.append(self.aux(re_flat))

        rec_feat = self.dfc_module(nr_feat, mask)
        rec_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(rec_feat).reshape(rec_feat.shape[0], -1))
        rec_outputs.append(rec_out)
        # out = r_feat + rec_feat = mask * feat + (1-mask) * feat + rec_units = feat + rec_units 
        out = r_feat + rec_feat
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        feat = out
        out = self.linear(out)
        return out, feat, r_outputs, nr_outputs, rec_outputs, ro_flat, re_flat


def ResNet18_FSR(num_classes=10, tau=0.1, image_size=(32, 32)):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, tau=tau, image_size=image_size)

def resnet10_dc(num_classes=7, gum_tau=0.1):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, tau=gum_tau, image_size=(128, 128))

def resnet34_dc(num_classes=7, gum_tau=0.1):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, tau=gum_tau, image_size=(128, 128))

def resnet10_dc_office(num_classes=10, gum_tau=0.1):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, tau=gum_tau, image_size=(32, 32))

def resnet10_dc_digits(num_classes=10, gum_tau=0.1):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, tau=gum_tau, image_size=(32, 32))