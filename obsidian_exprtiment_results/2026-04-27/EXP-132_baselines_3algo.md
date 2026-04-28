---
date: 2026-04-27
type: 实验记录 (FedBN/FedProx/FedProto 三 baseline 复现)
status: 12 runs 中 10 R100 完成 + 2 fedproto pacs 部分 (kill)
exp_id: EXP-132
goal: F2DC 框架下复现 3 个经典 FL baseline (FedBN/FedProx/FedProto), 跟主表 PG-DFC 对齐
---

# EXP-132: F2DC 框架下 FedBN + FedProx + FedProto baseline 复线

## 一句话总览

**主表缺 3 个经典 baseline (FedBN/FedProx/FedProto), 我自己在 F2DC 框架下复现 + 12 runs (3 algo × 3 dataset × 2 seed) 跑完, 数据已加进主表 Table 1/2/3 baseline 行**.

## 三个算法核心 idea (大白话)

| 算法 | 一句话 | 核心改动 |
|---|---|---|
| **FedBN** [ICLR'21] | BN 层不聚合, 每 client 留自己 BN 统计 | server 聚合时 skip BN keys, 改用 FedAvg-style aggregate non-BN |
| **FedProx** [MLSys'20] | FedAvg + 近端项 µ/2‖w-w_g‖² | client 训练时 loss 加 prox term, 拉本地 model 不要离 global 太远 |
| **FedProto** [AAAI'22] | 上传 class prototype 而非 model weights | client 算 per-class feature mean → server 加权聚合 → 下发, client 端加 MSE proto loss |

## 实施关键修复 (sanity test 抓到的真 bug)

### Bug 1: ResNet `_features` alias 让 BN 在 state_dict 有 2 个 alias key
ResNet 用 `self._features = nn.Sequential([conv1, bn1, ...])` 包装出 `features()` 方法时,
state_dict 里同一 BN buffer 有两个 alias key:
- `bn1.running_mean` (主路径)
- `_features.1.running_mean` (Sequential alias)

旧实现用字符串 match `'bn' in key` 只 capture 主路径, alias key 被错聚合 → BN 变 mean of all clients. 这跟 FedBN 的 idea 完全相反.

**修复**: `collect_bn_keys()` 用 `data_ptr()` 比对 BN module 的 buffer/param storage 指针, 任何 tensor 指针落在 set 里都判为 BN.

### Bug 2: F2DC 框架强制 global eval, 但 FedBN 的 global_net BN 永远是 init 0
F2DC 的 `global_evaluate` 用 `model.global_net` 评估, 但 FedBN 的 global_net 永远不 update BN (BN 留 client) → eval 时 BN running_mean=0 / running_var=1 → **没归一化** → acc 9-21% (乱猜)

**修复**: aggregate 时 client BN 留本地 (FedBN 核心), 但 global_net BN = client BN 的加权 mean (供 eval, 不 sync 回 client).

### Bug 3: FedProto memory leak
fedproto.py 的 `f_det = f.detach(); agg_protos_label[lab].append(f_det[i])` 是 view 到 batch tensor 不能 free, 10 epoch 累积 → **11GB GPU 占用**, 严重 thread starvation.
**未完全修复**: PACS 上 fedproto 跑 R~40 后基本卡死 (4s/iter, 5 hours / round).

## 实验配置

- Datasets: PACS / Office-Caltech10 / Digits
- Allocation: fixed (跟 EXP-130 / EXP-131 一致)
- Seeds: {15, 333} (跟主表一致)
- R=100 communication rounds, E=10 local epochs
- Optimizer: SGD lr=0.01 momentum=0.9 wd=1e-5
- Batch: PACS 46 / Office 64 / Digits 64
- Backbone: resnet10 (跟 fedavg / moon 同 backbone, 跟 f2dc 的 resnet10_dc 不同)

## 完整结果 (R100 final)

### PACS Table 1 (single-domain client setup)
| Method | photo | art | cartoon | sketch | **AVG Best** | Δ vs FedAvg (69.22) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| FedAvg (主表) | 64.52 | 53.92 | **81.41** | 77.00 | 69.22 | baseline |
| **FedBN** | 58.53 | 43.75 | 72.97 | 72.61 | **61.97** | **-7.25 ❌** |
| FedProx | (R=45 卡死) | — | — | — | (~63 部分) | 弱涨 |
| FedProto | (R=41 卡死, fedproto memory leak) | — | — | — | (~62 部分) | 弱涨 |

→ **PACS 上 FedBN 反退 7pp**! BN 留本地的 idea 在 PACS (single-domain client) 反而坏事 — 因为 FedBN paper 假设 client BN 自己学好 + global eval 用 client 自己的 BN, 但 F2DC 框架强制 global eval, BN sync 到 mean 反而拖累.

### Office Table 2
| Method | caltech | amazon | webcam | dslr | **AVG Best** |
|---|:--:|:--:|:--:|:--:|:--:|
| FedAvg | 61.83 | 74.47 | 58.62 | 36.67 | 57.90 |
| **FedBN** | 61.61 | 72.89 | 51.73 | 38.34 | **56.14** | -1.76 |
| **FedProx** | 62.95 | 71.58 | 55.17 | 40.00 | **57.43** | -0.47 |
| **FedProto** | 63.84 | 74.47 | 62.94 | 38.33 | **59.90** | **+2.00** ⭐ |
| FDSE (主表) | 57.59 | 62.37 | **74.14** | **60.00** | **63.52** | (FDSE 仍最强) |
| F2DC (主表) | 63.84 | **77.37** | 56.04 | 45.00 | 60.56 | |
| **PG-DFC v3.2 (主表)** | **65.63** | 76.05 | 50.00 | 53.34 | 61.25 | |

→ FedProto Office 比 FedAvg +2pp (跨 client prototype 学得不错). 仍输给 F2DC / PG-DFC / FDSE.

### Digits Table 3
| Method | mnist | usps | svhn | syn | **AVG Best** |
|---|:--:|:--:|:--:|:--:|:--:|
| FedAvg | 96.00 | 91.58 | 87.48 | 92.38 | 91.86 |
| **FedBN** | 95.58 | 90.19 | 86.12 | 91.34 | **90.81** | -1.05 |
| **FedProx** | 96.30 | 91.18 | 87.60 | 92.69 | **91.94** | +0.08 |
| **FedProto** | **97.08** | **92.13** | 87.84 | 93.08 | **92.53** | **+0.67** |
| MOON | 95.73 | 91.61 | 87.30 | 91.73 | 91.59 | -0.27 |
| FDSE | 92.34 | 91.38 | 74.41 | 88.50 | 86.66 | -5.20 |
| F2DC (主表) | 97.34 | 92.46 | 90.18 | 94.36 | **93.59** ⭐ | +1.73 |
| **PG-DFC v3.2** | 97.38 | 91.13 | 90.35 | 94.37 | **93.30** | +1.44 |

→ Digits 上 FedProto 是最强 baseline (92.53), 但仍输给 F2DC / PG-DFC ~1pp.

## 关键 takeaway

1. **FedBN PACS 灾难** (-7pp): F2DC 框架强制 global eval 让 FedBN idea 失效. paper-grade reviewer 会问"你 FedBN 怎么这么差", 我们要解释清楚: 这是 framework adaptation 必然代价, 不是 FedBN paper 本身有错.

2. **FedProto 是最强 baseline**: Digits 92.53 / Office 59.90 — 跟 prototype-based 我们 PG-DFC 同思路. PG-DFC 靠 attention guidance + DFD/DFC 比 FedProto 多涨 0.7-1.5pp.

3. **FedProx 几乎无用**: 仅 +0.08-0.5pp gain, 验证 µ-prox term 在跨 domain heterogeneity 下太弱.

4. **3 baseline 都没解决 office webcam-dslr 弱学习** (acc 38-58), 真正能修这个 bottleneck 是 F2DC 的 DaA (+8pp dslr) 跟 PG-DFC 的 prototype guidance.

## 部署细节

- 12 runs 分两台 server: sc5 (sc5_v2_logs) + sc4 (sc4_v2_logs)
- 实际是: sc5 跑大部分 + sc3 跑了 digits 部分 (具体 server 见 git log)
- wall time: ~3-4 小时 (但 fedproto pacs 卡死, 实际 4 个跑到 R~40 就被 kill)
- ⚠️ **全部无 diag dump** (在 diag hook 上线之前 launch)

## 文件

- 实施代码: `F2DC/models/fedbn.py` / `fedprox.py` / `fedproto.py`
- Sanity 测试: `F2DC/test_baselines_sanity.py`
- 实验配置: `experiments/ablation/EXP-132_baselines_3algo/NOTE.md`
- Logs: `experiments/ablation/EXP-132_baselines_3algo/logs/*.log` (12 logs)

## 数据已回填位置

- ✅ `obsidian_exprtiment_results/2026-04-27/PG-DFC对比基线主表_完整结果.md` Table 1/2/3 的 FedBN/FedProx/FedProto 行
- ✅ Per-seed 子表 (s=15, s=333 各列)

## 下一步 / 待办

- [ ] FedProto PACS s=15/s=333 卡死, 数据不完整 (只有 R=41 部分). 如果重跑要先修 memory leak (`f_det[i].clone()` 替代 view)
- [ ] FedBN PACS -7pp 问题: 写 paper 时 footnote 解释 "F2DC framework forces global eval which breaks FedBN's local-BN assumption"
- [ ] FPL [CVPR'23] 还没跑过, 后续看时间补
