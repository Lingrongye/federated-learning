# FedDSA Paper 主表 — 2026-04-10

> 所有实验: R200, AlexNet backbone, 3-seed mean (seed=2,15,333)
> PACS: E=5, B=50; Office-Caltech10: E=1, B=50
> ALL = 按样本加权准确率; AVG = client 简单平均准确率
> 数据来源: SC1 + SC2 服务器 json records 合并

---

## Table 1: PACS R200 (AVG Best, 3-seed mean ± std)

| Method | s2 | s15 | s333 | **Mean ± Std** |
|--------|------|------|------|----------------|
| FedAvg | 71.46 | 71.50 | 70.40 | 71.12 ± 0.62 |
| FedProx | 72.03 | 72.15 | 69.83 | 71.34 ± 1.31 |
| MOON | 71.26 | 72.46 | — | ~71.86 (2-seed) |
| Ditto | 79.51 | 74.77 | 77.64 | 77.31 ± 2.38 |
| FedBN | 79.19 | 78.70 | 77.74 | 78.54 ± 0.73 |
| FDSE | 82.16 | 79.00 | 79.93 | 80.36 ± 1.67 |
| **FedDSA (Ours)** | **81.15** | **80.59** | **81.05** | **80.93 ± 0.30** |

### PACS 分析
- FedDSA vs FDSE: **+0.57** (微赢, 但 FedDSA std=0.30 远小于 FDSE std=1.67)
- FedDSA vs FedBN: **+2.39** (显著)
- FedDSA vs Ditto: **+3.62**
- FedDSA **方差最小** (0.30), 最稳定

---

## Table 2: Office-Caltech10 R200 (AVG Best, 3-seed mean ± std) ✅ COMPLETE

| Method | s2 | s15 | s333 | **Mean ± Std** |
|--------|------|------|------|----------------|
| FedAvg | 85.92 | 83.31 | 87.78 | 85.67 ± 2.24 |
| MOON | 85.13 | 87.10 | 86.74 | 86.33 ± 1.05 |
| Ditto | 87.87 | 90.26 | 86.47 | 88.20 ± 1.92 |
| FedProx | 87.78 | 88.20 | 88.76 | 88.25 ± 0.49 |
| FedBN | 88.99 | 88.27 | 88.68 | 88.65 ± 0.36 |
| **FedDSA (Ours)** | **89.95** | **91.08** | **86.35** | **89.13 ± 2.42** |
| FDSE | 92.39 | 91.24 | 88.11 | **90.58 ± 2.22** |

### Office 分析
- FedDSA vs FDSE: **-1.45** (FDSE 赢)
- FedDSA vs FedBN: **+0.48**
- FedDSA vs FedProx: **+0.88**
- FedDSA vs FedAvg: **+3.46**
- FedDSA 排名**第二**,仅次于 FDSE

---

## 对比 FDSE 论文 (R500, 5-seed)

### PACS
| Method | 论文 R500 AVG | 我们 R200 AVG | 差距 |
|--------|--------------|--------------|------|
| FedAvg | 72.10 | 71.12 | -0.98 |
| FedBN | 79.47 | 78.54 | -0.93 |
| Ditto | 80.03 | 77.31 | -2.72 |
| FDSE | **82.17** | 80.36 | -1.81 |
| **FedDSA** | — | **80.93** | — |

### Office
| Method | 论文 R500 AVG | 我们 R200 AVG | 差距 |
|--------|--------------|--------------|------|
| FedAvg | 86.26 | 85.67 | -0.59 |
| FedBN | 87.01 | 88.65 | +1.64 |
| Ditto | 88.72 | ~89.07 | +0.35 |
| FDSE | **91.58** | 90.58 | -1.00 |
| **FedDSA** | — | **89.13** | — |

---

## 待补数据
| 实验 | 缺失 | 状态 |
|------|------|------|
| MOON PACS s333 | 1 run | ⏳ SC2 在跑 |
| MOON Office s15/s333 | 2 runs | ⏳ SC2 在跑 |
| Ditto Office s333 | 1 run | ⏳ SC2 在跑 |
| FedDSA Office LR=0.05 s15/s333 | 2 runs | ⏳ SC2 在跑 |

---

## Ablation Summary

### LR Grid Search (PACS, seed=2)
| LR | decay | AVG Best | 判定 |
|----|-------|----------|------|
| **0.1** | 0.9998 | **82.24** | **最优** |
| 0.05 | 0.9998 | 79.31 | ❌ -2.93 |
| 0.2 | 0.9998 | 79.05 | ❌ -3.19, 崩 |
| 0.1 | 0.998 | 80.31 | ❌ -1.93 |

### LR Grid Search (Office, seed=2)
| LR | AVG Best | 判定 |
|----|----------|------|
| 0.1 | 89.95 | 基线 |
| **0.05** | **90.82** | **+0.87, 最优** |

### 架构/Loss Ablation (PACS, seed=2)
| Variant | AVG Best | Gap | 判定 |
|---------|----------|-----|------|
| FedDSA 原版 | 82.24 | 6.78 | 基线 |
| EXP-058 Detach | 79.05 | — | ❌ -3.19 |
| EXP-059 StyleHeadBank | 80.02 | — | ❌ -2.22 |
| EXP-048 FixBN | 80.73 | 4.46 | ❌ trade-off |
| EXP-047A AugDown | 81.33 | 7.50 | ❌ 无效 |
| EXP-047D NoAugLate | 82.32 | 8.51 | ❌ 无效 |
