"""
Digits-5 联邦学习任务配置 (digit5_c5 = 5 client, 每 client 1 个 domain).

=============================================================
数据集来源: FedBN (ICLR 2021) 官方预处理数据, HuggingFace 托管
   https://huggingface.co/datasets/Jemary/FedBN_Dataset/resolve/main/digit_dataset.zip
   解压后路径: $FLGO_DATA_ROOT/digit5/
   目录结构:
     digit5/
     ├── MNIST/        partitions/ (10 份) + test.pkl
     ├── MNIST_M/      partitions/ (10 份) + test.pkl
     ├── SVHN/         partitions/ (10 份) + test.pkl
     ├── SynthDigits/  partitions/ (10 份) + test.pkl
     └── USPS/         partitions/ (10 份) + test.pkl

每个 partitions/train_part{0..9}.pkl 是 pickle 打包的 (X, y) 元组:
   X shape 因 domain 而异, 见下方 _raw_shape
   y 是 int 列表, 10 类 (数字 0-9)

FedBN paper Table 2 标准划分: **5 client = 5 domain**, 每个 domain 作为一个 client
的本地数据. 每 client 的 train 集是该 domain 10 份 partition 合并 (7430 样本),
test 集是该 domain 的 test.pkl (1860~97791 样本, 取决于 domain).

=============================================================
关键决策 (已经用户确认):

1. **图像预处理统一到 32×32×3** — FedBN 原 repo 的 transform 做法. 这样 CNN 只需一套
   结构, 不用为每个 domain 分别适配. 具体:
   - MNIST / USPS: 灰度 28×28 (或 16×16) → Resize 32 → Grayscale→RGB (复制 3 通道)
   - MNIST_M: 28×28×3 → Resize 32
   - SVHN / SynthDigits: 32×32×3 → 保持

2. **严格 FedBN 评估协议**: train / test 分开, 不 concat (跟 PACS/Office/DomainNet 的
   concat+holdout 不同). 每 client:
     - train = 该 domain 的 10 partitions 合并, 共 7430 样本
     - test  = 该 domain 的 test.pkl, subsample 到 TEST_SIZE_PER_DOMAIN (固定 1860)
   这样 reviewer 完全认可, paper 主表 Digits-5 可以直接和 FedBN / FedPLVM / MP-FedCL
   paper 的数字对齐比较 (他们都用 per-domain 固定 test set).

3. **test 均衡到 1860 per domain** (用户选项 #4): 5 个 domain 原 test.pkl 大小差异
   极大 (USPS 1860, MNIST/MNIST_M 14000, SVHN 19858, SynthDigits 97791). 如果不做
   subsample, AVG Best (per-client 简单平均) 会被 USPS/MNIST 的小 test set 主导,
   和 ALL Best (加权) 偏差极大. 统一到 min = 1860 消除这个偏差.
   - 用固定 seed 42 的 numpy.random.default_rng 采样, 每次实验一致
   - 1860 × 5 = 9300 total test, 相对 PACS 1500 / Office 750 适中

4. **Model 用 FedBN 标准 DigitModel** (5-layer CNN: 3 Conv + 2 FC), 约 20M 参数,
   32×32 输入. 经过 2 次 MaxPool 到 8×8, flatten 后 128*8*8=8192 → 2048 FC → 512 FC
   → 10 logits. 这是 FedBN / FedPLVM / FPL / MP-FedCL 等所有 Digits-5 benchmark 的
   标配, reviewer 最认可.

=============================================================
"""

import os
import pickle

import flgo.benchmark
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
from PIL import Image
from torchvision import transforms

# ----------------------------------------------------------
# 基础配置
# ----------------------------------------------------------

# 5 个 domain 名字 (注意 FedBN 用的顺序是 MNIST -> SVHN -> USPS -> SYN -> MNIST_M,
# 但我们按字母序排列更清晰, paper 主表里 per-domain 列顺序可以自由调整).
domain_list = ['MNIST', 'MNIST_M', 'SVHN', 'SynthDigits', 'USPS']

# 数据根目录: flgo 默认数据目录下的 digit5/
# 部署在服务器上时, 需要把下载解压后的 digit_dataset 目录 symlink 或 mv 到这里:
#   ln -s /root/autodl-tmp/digit_dataset $(flgo.benchmark.data_root)/digit5
path = os.path.join(flgo.benchmark.data_root, 'digit5')

# 10 个类别 (数字 0-9, 所有 domain 共享同一标签空间)
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 每个域 raw shape 不一致, 记录在这里便于 __getitem__ 分支处理.
# (H, W, C) 形式, C 省略表示单通道灰度.
_raw_shape = {
    'MNIST': (28, 28),          # 灰度 28x28
    'MNIST_M': (28, 28, 3),     # RGB 28x28
    'SVHN': (32, 32, 3),        # RGB 32x32
    'SynthDigits': (32, 32, 3), # RGB 32x32 (合成数字)
    'USPS': (16, 16),           # 灰度 16x16 (邮政手写)
}

# test set 均衡到 per-domain = 1860 (= USPS 全量, 其他 domain 的物理最小下限).
# 避免 SynthDigits 97791 / USPS 1860 这种 50× 量级差异让 AVG 指标失去意义.
# 采样用固定 seed 42, 每次实验一致, 保证可复现.
TEST_SIZE_PER_DOMAIN = 1860
TEST_SUBSAMPLE_SEED = 42


# ----------------------------------------------------------
# 单域 Dataset 类
# ----------------------------------------------------------

class Digit5DomainDataset(torch.utils.data.Dataset):
    """
    单个 Digits-5 域的 PyTorch Dataset, 从 pkl 文件加载.

    构造参数:
      root:   数据根目录 (path), 下含 MNIST/MNIST_M/... 子目录
      domain: 域名字 (必须在 domain_list 中)
      split:  'train' 或 'test'
                - train: 合并该 domain 的 10 个 partitions/train_part*.pkl
                - test:  该 domain 根目录下的 test.pkl

    transform 策略 (统一所有 domain 到 (3, 32, 32) 浮点张量, 归一化到 ±1):
      1. numpy array -> PIL Image
      2. Resize((32, 32))  — MNIST/USPS 原 28×28 或 16×16 会被放大
      3. 若单通道 (MNIST/USPS), Grayscale→RGB 三通道复制
      4. ToTensor (0-255 uint8 → 0-1 float)
      5. Normalize mean=0.5 std=0.5 (FedBN 原 code 做法), 输出 range [-1, 1]
    """

    def __init__(self, root, domain, split='train'):
        super().__init__()
        assert domain in domain_list, f'未知 domain: {domain}'
        assert split in ('train', 'test'), f'split 必须 train 或 test'
        self.root = root
        self.domain = domain
        self.split = split

        domain_dir = os.path.join(root, domain)
        if split == 'train':
            # --- train: 合并 10 个 partition, 拼成单一大数组 (7430 样本) ---
            X_list, y_list = [], []
            for i in range(10):
                pkl_path = os.path.join(domain_dir, 'partitions', f'train_part{i}.pkl')
                with open(pkl_path, 'rb') as f:
                    Xi, yi = pickle.load(f)
                X_list.append(Xi)
                y_list.append(np.asarray(yi).reshape(-1))
            self.X = np.concatenate(X_list, axis=0)
            self.y = np.concatenate(y_list, axis=0).astype(np.int64)
        else:
            # --- test: 加载 test.pkl, 然后 subsample 到 TEST_SIZE_PER_DOMAIN 均衡大小 ---
            with open(os.path.join(domain_dir, 'test.pkl'), 'rb') as f:
                X, y = pickle.load(f)
            X = X
            y = np.asarray(y).reshape(-1).astype(np.int64)

            n_orig = len(X)
            target_n = min(TEST_SIZE_PER_DOMAIN, n_orig)
            if n_orig > target_n:
                # 固定 seed 随机 subsample, 保证每次跑同一子集
                rng = np.random.default_rng(TEST_SUBSAMPLE_SEED)
                idx = rng.choice(n_orig, size=target_n, replace=False)
                idx.sort()   # 排序便于 debug, 不影响随机性
                X = X[idx]
                y = y[idx]
            self.X = X
            self.y = y

        # 预先构造 transform (只构造一次, 避免每 __getitem__ 重建)
        is_gray = (len(_raw_shape[domain]) == 2)  # MNIST / USPS 灰度
        tfs = [
            transforms.ToPILImage(),                 # numpy -> PIL, 自动识别通道
            transforms.Resize((32, 32)),             # 统一到 32x32
        ]
        if is_gray:
            # 灰度图扩展到 3 通道 (简单复制), FedBN 原 code 的做法
            tfs.append(transforms.Grayscale(num_output_channels=3))
        tfs += [
            transforms.ToTensor(),                   # (H,W,C) uint8 -> (C,H,W) float [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
        ]
        self.transform = transforms.Compose(tfs)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]

        # dtype 规范化 — SynthDigits 可能是 float, 需要转 uint8 给 PIL
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                # 浮点 [0, 1] 范围 (罕见)
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                # 浮点 [0, 255] 范围 (常见, SynthDigits 实际如此)
                img = img.clip(0, 255).astype(np.uint8)

        # 形状规范化 — 单通道 (H, W, 1) 的情况下 ToPILImage 要求 2D array
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(-1)

        img_t = self.transform(img)            # (3, 32, 32) float
        label = int(self.y[idx])
        return img_t, label


# ----------------------------------------------------------
# 构造 5 client 的 train + test 数据
# ----------------------------------------------------------

# 为每个 domain 构造 train/test Dataset
# 注意: 这是 **module-level 的数据加载**, import config.py 就会触发磁盘读取 +
# pickle 解码 + numpy concat. Digits-5 原始 pkl 全部加载后约 500MB 内存, OK.
domain_data_train = [Digit5DomainDataset(path, domain=d, split='train') for d in domain_list]
domain_data_test = [Digit5DomainDataset(path, domain=d, split='test') for d in domain_list]

# 严格 FedBN 评估协议 (用户已确认):
#   train_data = list of 5 个 Dataset, 每个对应一个 client 的 train 数据 (7430 样本)
#   test_data  = list of 5 个 Dataset, 每个对应一个 client 的 test 数据 (1860 样本,
#                  已经 subsample 过的)
# flgo 的 from_dataset_pipe 支持 train/test 都是 list 形式, each client 独立地用 train/test.
# 这样评估用固定 test.pkl (而不是 train 的 holdout), 和 FedBN/FedPLVM/MP-FedCL paper 一致.
train_data = list(domain_data_train)
val_data = None           # 不单独留 val, flgo 可选用 train_holdout 做 val (config.yml 控制)
test_data = list(domain_data_test)


# ----------------------------------------------------------
# 训练相关函数 (loss / eval / data_to_device, 与 domainnet_c6 接口一致)
# ----------------------------------------------------------

loss_fn = nn.CrossEntropyLoss()


def data_to_device(batch_data, device):
    """把一个 batch (image, label) 移到 device."""
    return batch_data[0].to(device), batch_data[1].to(device)


def eval(model, data_loader, device) -> dict:
    """在给定 DataLoader 上跑完整 evaluation, 返回 {accuracy, loss}."""
    model.eval()
    model.to(device)
    total_loss = 0.0
    num_correct = 0
    num_samples = 0
    # 不开 autograd, 节省显存加速
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = data_to_device(batch_data, device)
            outputs = model(batch_data[0])
            batch_mean_loss = loss_fn(outputs, batch_data[-1]).item()
            y_pred = outputs.data.max(1, keepdim=True)[1]
            correct = y_pred.eq(batch_data[-1].data.view_as(y_pred)).long().cpu().sum()
            num_correct += correct.item()
            num_samples += len(batch_data[-1])
            total_loss += batch_mean_loss * len(batch_data[-1])
    return {
        'accuracy': 1.0 * num_correct / num_samples if num_samples > 0 else 0.0,
        'loss': total_loss / num_samples if num_samples > 0 else 0.0,
    }


def compute_loss(model, batch_data, device) -> dict:
    """单 batch loss, 用于训练循环. 返回 {loss: Tensor}."""
    tdata = data_to_device(batch_data, device)
    outputs = model(tdata[0])
    loss = loss_fn(outputs, tdata[-1])
    return {'loss': loss}


# ----------------------------------------------------------
# DigitModel — FedBN (ICLR 2021) paper 标准 5-layer CNN
# ----------------------------------------------------------

class DigitModel(nn.Module):
    """
    Digits-5 标准 backbone (FedBN paper appendix + FedPLVM/FPL/MP-FedCL 沿用).

    结构 (输入 3×32×32):
        conv1  3 -> 64   5x5 pad=2   bn1   ReLU   MaxPool 2x2  -> 64×16×16
        conv2 64 -> 64   5x5 pad=2   bn2   ReLU   MaxPool 2x2  -> 64×8×8
        conv3 64 -> 128  5x5 pad=2   bn3   ReLU                -> 128×8×8
        flatten -> 128*8*8 = 8192
        fc1   8192 -> 2048  bn4 (1D)  ReLU
        fc2   2048 ->  512  bn5 (1D)  ReLU
        fc3    512 ->   10  (logits)

    参数量: ~20M (主要在 fc1 8192->2048). 比 AlexNet (60M) 小, 符合 digits 任务规模.

    为什么 FedBN 选这个架构:
      1. BN 密集 (每个 conv/fc 后都有 BN): FedBN 的核心是把 BN 层本地化,
         BN 越密集本地化效果越明显.
      2. 没有 Dropout: 便于 BN 本地化消融 (Dropout 会引入额外不确定性).
      3. 参数量适中: 10 FL round × 100 local step × 5 client 能在 1h 内跑完
         标准设置.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # --- 卷积块 ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)

        # --- 全连接 ---
        self.fc1 = nn.Linear(128 * 8 * 8, 2048)
        self.bn4 = nn.BatchNorm1d(2048)

        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1: conv + bn + relu + pool
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)                 # 32x32 -> 16x16

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)                 # 16x16 -> 8x8

        # Block 3 (无 pool)
        x = F.relu(self.bn3(self.conv3(x)))    # 保持 8x8

        # Flatten + FC
        x = x.view(x.size(0), -1)              # -> (B, 8192)
        x = F.relu(self.bn4(self.fc1(x)))      # -> (B, 2048)
        x = F.relu(self.bn5(self.fc2(x)))      # -> (B, 512)
        x = self.fc3(x)                        # -> (B, 10)
        return x


def get_model():
    """flgo 加载 Model 时调用. 返回未初始化的 DigitModel 实例."""
    return DigitModel(num_classes=len(classes))


# ----------------------------------------------------------
# 调试入口: python config.py 会打印每域几张示例图
# ----------------------------------------------------------
if __name__ == '__main__':
    print(f'Digits-5 task config 加载成功')
    print(f'  path = {path}')
    print(f'  domains = {domain_list}')
    print(f'  TEST_SIZE_PER_DOMAIN = {TEST_SIZE_PER_DOMAIN} (subsample seed={TEST_SUBSAMPLE_SEED})')
    for i, d in enumerate(domain_list):
        tr, te = domain_data_train[i], domain_data_test[i]
        sample_img, sample_label = tr[0]
        print(f'  {d}: train={len(tr)} test={len(te)} '
              f'raw_shape={_raw_shape[d]} sample_tensor={tuple(sample_img.shape)} '
              f'sample_label={sample_label}')
    print(f'  共 {len(train_data)} 个 client')
    print(f'  train per client: {[len(d) for d in train_data]}')
    print(f'  test  per client: {[len(d) for d in test_data]}')
    print(f'  Model get_model(): type={type(get_model()).__name__}')
