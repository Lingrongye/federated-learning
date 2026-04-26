# EXP-130 | F2DC 主表 baseline 复现 (3 algo × 3 dataset × 3 seed)

## 基本信息
- **日期**: 2026-04-26 启动
- **服务器**: sc3 (新 AutoDL 实例, port 19385, GPU 0 RTX 4090 24GB, 完全空闲)
- **目的**: 在 F2DC release 框架下复现 F2DC paper 主表数字 (PACS/Office/Digits)
- **状态**: 🟡 smoke 测试中

## 这个实验做什么 (大白话)

把 F2DC 论文 (CVPR 2026) 的整套实验在我们环境下完整复现一遍, 包括:
1. **F2DC 自身** — 看我们能不能复现 PACS 76.47% / Office 66.82% / Digits 87.23% 这三个论文报数
2. **FedAvg** — F2DC 的 FedAvg 极限 (论文报: PACS 66.39 / Office 55.86 / Digits 81.24)
3. **MOON** — F2DC 的 MOON 极限 (论文报: PACS 62.64 / Office 51.41 / Digits 76.60)

跑完后这 9 个数字直接进 paper 主表 baseline 对照, 跟我们 FedDSA 数字横向对比.

## 关键背景: F2DC release 代码不能直接跑

F2DC release (https://github.com/mala-lab/F2DC) 在 PyTorch 2.6 上**直接踩 6 个 bug**, 必须先 patch 才能跑:

| # | 文件 | Bug | 修复 |
|---|------|-----|------|
| 1 | backbone/VGGNet_DC.py | release 完全缺失 | 补 stub |
| 2 | backbone/WRN.py | release 完全缺失 | 补 stub |
| 3 | datasets/pacs.py:80 | `array == []` shape mismatch | `len() == 0` |
| 4 | datasets/utils/federated_dataset.py:218,353 | `np.random.randint(0)` when prob_del 空 | `if len()==0: break` |
| 5 | backbone/ResNet_DC.py:162 | forward 5-tuple ≠ f2dc.py 期望 7-tuple | 补返回 ro_flat/re_flat |
| 6 | models/f2dc.py:94 | `tensor(0.) += [B]` PyTorch 2.x 不兼容 | `(l_cos/τ).mean()` |
| 7 | models/f2dc.py:_train_net | rand_dataset 同域多 client → 空 dataloader 除零 | skip empty client |
| 8 | utils/training.py:29 | eval 也按 5-tuple 解包 | 改 7-tuple |

详见 git commit `25ab52a fix(F2DC): 6 个 patch 让 release 代码在 PyTorch 2.6 上跑通`.

## 实验配置

### 数据集 (对齐 F2DC paper Sec 5.1)

| 数据集 | parti_num | percent | 域 | N classes | backbone |
|--------|:---------:|:-------:|----|:---------:|----------|
| PACS | 10 | 30% | photo / art / cartoon / sketch | 7 | resnet10_dc (128×128) |
| Office-Caltech10 | 10 | 20% | caltech / amazon / webcam / dslr | 10 | resnet10_dc (32×32) |
| Digits | 20 | 1% | mnist / usps / svhn / syn | 10 | resnet10_dc (32×32) |

### 算法

| 算法 | 来源 | 关键超参 |
|------|------|----------|
| F2DC | 论文 method | gum_tau=0.1, tem=0.06, λ1=0.8, λ2=1.0 (论文默认) |
| FedAvg | 论文 baseline | 无额外 |
| MOON | 论文 baseline | temperature=0.5, mu=5 (best_args 默认) |

### 训练 hyperparameters (所有算法 + 数据集统一)
```
communication_epoch = 100      # F2DC 论文标准
local_epoch        = 10
local_lr           = 0.01
batch_size         = 64
optimizer          = SGD(momentum=0.9, weight_decay=1e-5)
online_ratio       = 1.0       # 全部 client 参与
averaging          = 'weight'  # by sample count
```

### Seeds
- **{2, 15, 333}** — 跟我们 FedDSA 主表对齐, 3 seed mean ± std

### 总实验数
**3 algo × 3 dataset × 3 seed = 27 runs**

## Smoke test (启动前必须 0 error)

### Smoke 配置
- R=3, E=3 (轻量), seed=55, parti_num 按数据集
- 串行跑, 第 1 个失败立即 abort
- 9 个 smoke (3 algo × 3 dataset)
- 路径: `sc3:/root/autodl-tmp/federated-learning/F2DC/_smoke_test/`

### Smoke 验证项
- [ ] F2DC × PACS: ✅ 之前 R=5 烟雾测试已通过 (R0→R4: 10.5→30.9 acc 单调上涨)
- [ ] F2DC × Office: 🟡 测试中
- [ ] F2DC × Digits: 🟡 测试中 (需要 SynthDigits .pkl 解包成 ImageFolder)
- [ ] FedAvg × {PACS, Office, Digits}: 🟡 测试中
- [ ] MOON × {PACS, Office, Digits}: 🟡 测试中

### 数据准备状态
| 数据集 | 路径 | 状态 |
|--------|------|------|
| PACS | `rundata/dataset/PACS_7/{4 域}` | ✅ symlink (9991 张) |
| Office | `rundata/dataset/Office_Caltech_10/{4 域}` | ✅ symlink (2533 张) |
| MNIST | `rundata/dataset/MNIST/raw/*` | ✅ torchvision 自动下载 |
| USPS | `rundata/dataset/usps{,.t}.bz2` | ✅ |
| SVHN | `rundata/dataset/{train,test}_32x32.mat` | ✅ |
| SynthDigits | `rundata/dataset/syn/{train(7438),val(97791)}/<class>/` | ✅ 从 .pkl 解包成功 |

## Phase C 启动计划

烟雾全过后启动 27 runs greedy launcher:
- **MIN_FREE_MB**: 5500 (单 F2DC ~5GB, 留 500MB 余量)
- **GPU**: sc3 GPU 0 (24 GB free)
- **理论并行**: 24 / 5.5 = 4 并行
- **总 wall 估算**: 27 / 4 ≈ 7 wave × 单 wave (PACS R=100 ~12.5h / Office ~6h / Digits ~3h) ≈ 实际取决于数据集分布
- **保守预估**: 24-36h wall

## 胜负判决 (paper 角度)

| 指标 | F2DC 论文报 | 我们复现目标 | 含义 |
|------|:----------:|:----------:|------|
| F2DC PACS AVG | 76.47 | 73-78 | release 跑出与论文 ±3pp 即认为复现 OK |
| F2DC Office AVG | 66.82 | 64-69 | 同上 |
| F2DC Digits AVG | 87.23 | 84-89 | 同上 |
| FedAvg PACS AVG | 66.39 | 63-69 | 同上 |
| FedAvg Office AVG | 55.86 | 53-59 | 同上 |
| FedAvg Digits AVG | 81.24 | 78-84 | 同上 |
| MOON PACS AVG | 62.64 | 60-66 | 同上 |
| MOON Office AVG | 51.41 | 48-55 | 同上 |
| MOON Digits AVG | 76.60 | 73-80 | 同上 |

**如果跑出来差距 > 3pp**: 说明 release 代码确实是早期版本, 论文用的是更完整的内部代码, 我们 paper 对比时**只用我们环境复现的数字**而不是论文报数 (公平比较).

## 📎 相关文件
- F2DC 仓库: `F2DC/` (vendored, commit bb355ba clean baseline + 25ab52a 6 patches)
- Smoke runner: `sc3:/root/autodl-tmp/federated-learning/F2DC/_smoke_test/run_smokes.sh`
- 服务器: sc3 (`~/.ssh/config` 别名, port 19385)
- 数据 symlink: `sc3:/root/autodl-tmp/federated-learning/F2DC/rundata/dataset/`

## 结果回填

(等 27 runs 跑完后回填这里)

### 主表数字

| Algo | PACS AVG (mean ± std) | Office AVG | Digits AVG |
|------|:--:|:--:|:--:|
| FedAvg | TBD | TBD | TBD |
| MOON | TBD | TBD | TBD |
| F2DC | TBD | TBD | TBD |

### 与论文对比

| Algo × Dataset | 论文 | 我们复现 | Δ |
|---|:--:|:--:|:--:|
| F2DC PACS | 76.47 | TBD | TBD |
| F2DC Office | 66.82 | TBD | TBD |
| F2DC Digits | 87.23 | TBD | TBD |
| ... | | | |

### 结论 (跑完后写)

TBD
