"""ResNet-10 backbone for FDSE (CVPR 2025) ported into F2DC framework.

完整保留 FDSE 论文的 layer 分解思想 (DFE = domain-agnostic feature extractor,
DSE = domain-specific shift eraser): 每个 Conv2d 替换成 DSEConv, 这样后续
Server.iterate 能按 dfe_/dse_ 关键字差异化聚合.

适配:
- 输入大小: PACS 128x128 (跟 F2DC ResNet_DC.py 一致, parti_num=10 协议),
            Office/Digits 32x32 (跟 F2DC ResNet_DC.py 一致)
- backbone: ResNet-10 (跟 F2DC paper Sec 5.1 'all compared methods use the same model' 一致)
- 完全照抄 FDSE 原仓库的 DSEConv / DSELinear (FDSE_CVPR25/algorithm/fdse.py)

参数命名约定 (Server.iterate 依赖):
- 名字含 'dfe_' = domain-agnostic, 走 QP 优化的差异化加权聚合
- 名字含 'head' = 分类头, 也走 dfe 通道 (跟 FDSE 原 shared_names = ['dfe','head'] 一致)
- 名字含 'dse_bn.running_' = 完全本地 (BN running stats 不聚合)
- 其他 (含 dse_conv 等) = personalized, 走 cosine similarity softmax 加权聚合
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSEConv(nn.Module):
    """完全照抄 FDSE_CVPR25/algorithm/fdse.py 的 DSEConv (line 14-50)."""
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, padding=0,
                 use_relu=True, bias=True, use_dse_activate=False, use_dse_bn=True, shortcut=True):
        super().__init__()
        self.oup = oup
        self.use_relu = use_relu
        self.use_dse_activate = use_dse_activate
        self.use_dse_bn = use_dse_bn
        self.ratio = ratio
        self.shortcut = shortcut
        if shortcut:
            init_channels = math.ceil(oup / ratio) if ratio > 1 else oup
            new_channels = init_channels * (ratio - 1) if ratio > 1 else oup
        else:
            init_channels = math.ceil(oup / ratio) if ratio > 1 else oup
            new_channels = oup
        self.dfe_conv = nn.Conv2d(inp, init_channels, kernel_size, stride, padding, bias=True)
        self.dse_bn = nn.BatchNorm2d(init_channels)
        self.dfe_bias = nn.Parameter(torch.zeros(oup)) if bias else None
        self.dse_conv = nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2,
                                   groups=init_channels, bias=False)
        self.dfe_bn = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x1 = self.dfe_conv(x)
        if self.use_dse_bn: x1 = self.dse_bn(x1)
        if self.use_dse_activate: x1 = self.leakyrelu(x1)
        if self.shortcut:
            x2 = self.dse_conv(x1)
            x = torch.cat([x1, x2], dim=1) if self.ratio > 1 else x2 + x1
        else:
            x = self.dse_conv(x1)
        if self.dfe_bias is not None:
            x = x[:, :self.oup, :, :] + self.dfe_bias.expand(x.shape[0], self.dfe_bias.shape[0])\
                                              .reshape(x.shape[0], self.dfe_bias.shape[0], 1, 1)
        x = self.dfe_bn(x)
        if self.use_relu: x = self.relu(x)
        return x


class FDSEBasicBlock(nn.Module):
    """ResNet BasicBlock with FDSE DSEConv 替换两个 3x3 Conv."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = DSEConv(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                              bias=False, use_relu=True)
        # DSEConv 内部已带 dfe_bn + relu, 所以这里不需要外层 BN/ReLU
        self.conv2 = DSEConv(planes, planes, kernel_size=3, stride=1, padding=1,
                              bias=False, use_relu=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet_FDSE(nn.Module):
    """ResNet-10 with FDSE DSEConv blocks, 适配 F2DC 协议 (PACS 128x128 / Office,Digits 32x32)."""
    def __init__(self, num_classes=7, image_size=(128, 128)):
        super().__init__()
        self.in_planes = 64
        self.image_size = image_size
        # 第一层 conv 用普通 Conv2d (跟 F2DC ResNet.py 一致, 不做 DSE 分解)
        # 因为第一层是 stem 抽 low-level edge/color, 不存在 domain bias
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 4 个 stage (跟 F2DC ResNet.py [1,1,1,1] 一致)
        self.layer1 = self._make_layer(FDSEBasicBlock, 64, 1, stride=1)
        self.layer2 = self._make_layer(FDSEBasicBlock, 128, 1, stride=2)
        self.layer3 = self._make_layer(FDSEBasicBlock, 256, 1, stride=2)
        self.layer4 = self._make_layer(FDSEBasicBlock, 512, 1, stride=2)
        # 分类 head (用 'head' 命名, FDSE shared_names 包含 'head')
        self.head = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.head(out)
        return out

    def features(self, x):
        """For MOON-like methods that need feature representation."""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out); out = self.layer4(out)
        out = nn.AdaptiveAvgPool2d(1)(out)
        return out.view(out.size(0), -1)


def resnet10_fdse_pacs(num_classes=7):
    return ResNet_FDSE(num_classes=num_classes, image_size=(128, 128))

def resnet10_fdse_office(num_classes=10):
    return ResNet_FDSE(num_classes=num_classes, image_size=(32, 32))

def resnet10_fdse_digits(num_classes=10):
    return ResNet_FDSE(num_classes=num_classes, image_size=(32, 32))
