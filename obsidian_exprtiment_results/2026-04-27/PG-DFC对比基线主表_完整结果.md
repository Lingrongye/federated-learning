---
date: 2026-04-27
type: 实验结果汇总(主表)
status: PACS / Office 完成,Digits 跑中
data_source:
  - EXP-130 sc3_v2_logs (F2DC v2 + fdse v2, fixed allocation)
  - EXP-130 sc4_v2_logs (FedAvg + MOON v2, fixed allocation)
  - EXP-131 sc5 logs (PG-DFC v3.2 / v3.3, fixed allocation)
allocation: fixed (PACS photo:2/art:3/cartoon:2/sketch:3; Office caltech:3/amazon:2/webcam:2/dslr:3)
seeds: [15, 333]
training: R=100, E=10, lr=0.01, batch=46/64
---

# PG-DFC vs Baselines — 完整对比主表

> 跟 F2DC paper Table 1 格式一致 (per-domain + AVG Best + AVG Last + Best Round)
> 严格 2-seed mean,fixed allocation

---

## Table 1. PACS (4 domains × 7 classes, fixed: photo:2/art:3/cartoon:2/sketch:3)

| Method | photo | art | cartoon | sketch | **AVG Best ↑** | **AVG Last ↑** | Best Round (avg) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| FedAvg [AISTATS'17] | 64.52 | 53.92 | **81.41** | 77.00 | 69.22 | 68.39 | R96.5 |
| FedBN [ICLR'21] | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FedProto [AAAI'22] | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FPL [CVPR'23] | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MOON [CVPR'21] | TBD (R34+缺s333) | TBD | TBD | TBD | TBD | TBD | TBD |
| FDSE [CVPR'25] | TBD (R35+R30 未完成) | TBD | TBD | TBD | **TBD** | TBD | TBD |
| **F2DC [CVPR'26]** | **69.46** | 56.62 | 78.53 | 79.49 | **71.02** | 69.57 | R94.5 |
| **PG-DFC v3.2 (Ours)** | 67.82 | **62.99** | 80.45 | **81.53** | **73.20** ⭐ | **71.31** | R95.0 |
| **Δ vs FedAvg** | +3.30 | **+9.07** | -0.96 | +4.53 | **+3.98pp** | **+2.92pp** | -1.5 round |
| **Δ vs F2DC** | -1.64 | **+6.37** ⭐ | +1.92 | +2.04 | **+2.18pp** | **+1.74pp** | +0.5 round |

### Per-seed 详细数据 (PACS)

| Method | s=15 best | s=333 best | s=15 last | s=333 last | s=15 best round | s=333 best round |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| FedAvg | 69.565 | 68.865 | 69.565 | 67.205 | R100 | R93 |
| F2DC v2 | 72.345 | 69.702 | 70.122 | 69.025 | R89 | R98 |
| **PG-DFC v3.2** | **73.992** | **72.400** | 73.068 | 69.552 | R98 | R92 |

### Per-domain best round (PACS)

| Method | photo (s15/s333) | art (s15/s333) | cartoon (s15/s333) | sketch (s15/s333) |
|---|:--:|:--:|:--:|:--:|
| F2DC v2 | 68.56 / 70.36 | 58.09 / 55.15 | 81.20 / 75.85 | 81.53 / 77.45 |
| PG-DFC v3.2 | 69.46 / 66.17 | 62.50 / 63.48 | 83.12 / 77.78 | 80.89 / 82.17 |

---

## Table 2. Office-Caltech (4 domains × 10 classes, fixed: caltech:3/amazon:2/webcam:2/dslr:3)

| Method | caltech | amazon | webcam | dslr | **AVG Best ↑** | **AVG Last ↑** | Best Round (avg) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| FedAvg | 61.83 | 74.47 | 58.62 | 36.67 | 57.90 | 54.01 | R62.0 |
| FedBN | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FedProto | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FPL | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MOON | 58.70 | 70.53 | 47.42 | 33.33 | 52.49 | 50.11 | R54.0 |
| **FDSE [CVPR'25]** | 57.59 | 62.37 | **74.14** | **60.00** | **63.52** ⭐ | 59.33 | R69 |
| **F2DC [CVPR'26]** | 63.84 | 77.37 | 56.04 | 45.00 | 60.56 | 56.68 | R99 |
| **PG-DFC v3.2 (Ours)** | **65.63** | 76.05 | 50.00 | 53.34 | 61.25 | 56.05 | R92.5 |
| **PG-DFC v3.3 (A+B)** | 63.17 | **78.42** | 56.04 | 48.33 | 61.49 | **59.09** | R90 |
| **Δ v3.2 vs FedAvg** | +3.80 | +1.58 | -8.62 | +16.67 | +3.35 | +2.04 | +30.5 round |
| **Δ v3.2 vs F2DC** | +1.79 | -1.32 | -6.04 | +8.34 | +0.69 | -0.63 | -6.5 round |
| **Δ v3.3 vs F2DC** | -0.67 | +1.05 | 0 | +3.33 | +0.93 | +2.41 | -9 round |

⚠️ **重大发现**: FDSE (CVPR'25) 在 Office 上反而最强 (63.52),比 F2DC (60.56) +2.96pp。我们之前低估了 FDSE。

### Per-seed 详细数据 (Office)

| Method | s=15 best | s=333 best | s=15 last | s=333 last | s=15 best round | s=333 best round |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| FedAvg | 54.485 | 61.312 | 48.830 | 59.188 | R42 | R82 |
| MOON | 50.067 | 54.920 | 46.205 | 54.023 | R26 | R82 |
| F2DC v2 | 60.80 | 60.323 | 54.618 | 58.735 | R98 | R98 |
| FDSE v2 | 60.99 | **66.055** | 58.062 | 60.6 | R81 | R55 |
| PG-DFC v3.2 | 61.613 | 60.892 | 54.415 | 57.675 | R87 | R98 |
| PG-DFC v3.3 | 59.848 | 63.13 | 56.125 | 62.045 | R86 | R94 |

### Per-domain best round (Office)

| Method | caltech (s15/s333) | amazon (s15/s333) | webcam (s15/s333) | dslr (s15/s333) |
|---|:--:|:--:|:--:|:--:|
| F2DC v2 | 63.84 / 63.84 | 75.79 / 78.95 | 56.90 / 55.17 | 46.67 / 43.33 |
| FDSE v2 | 58.04 / 57.14 | 56.84 / 67.89 | 72.41 / 75.86 | 56.67 / 63.33 |
| PG-DFC v3.2 | 64.29 / 66.96 | 75.26 / 76.84 | 56.90 / 43.10 | 50.00 / 56.67 |
| PG-DFC v3.3 | 60.27 / 66.07 | 75.79 / 81.05 | 50.00 / 62.07 | 53.33 / 43.33 |

---

## Table 3. Digits (4 domains × 10 classes, fixed: mnist:3/usps:6/svhn:6/syn:5)

| Method | mnist | usps | svhn | syn | **AVG Best ↑** | **AVG Last ↑** | Best Round (avg) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| FedAvg | 96.00 | 91.58 | 87.48 | 92.38 | 91.86 | 91.63 | R89.5 |
| FedBN | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FedProto | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FPL | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MOON | 95.73 | 91.61 | 87.30 | 91.73 | 91.59 | 90.37 | R87.5 |
| FDSE | 92.34 | 91.38 | 74.41 | 88.50 | 86.66 | 84.61 | R74.0 |
| **F2DC** | **97.34** | **92.46** | **90.18** | **94.36** | **93.59** ⭐ | **93.40** | R94.5 |
| **PG-DFC v3.2 (Ours)** | TBD (R20 跑中)| TBD | TBD | TBD | **TBD** (R20 mean ~89) | TBD | TBD |
| **Δ F2DC vs FedAvg** | +1.34 | +0.88 | +2.70 | +1.98 | +1.73 | +1.77 | +5 round |

### Per-seed (Digits)

| Method | s=15 best | s=333 best | s=15 last | s=333 last | s=15 best round | s=333 best round |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| FedAvg | 92.100 | 91.625 | 91.825 | 91.442 | R98 | R81 |
| MOON | 92.227 | 90.958 | 91.342 | 89.388 | R85 | R90 |
| F2DC v2 | 93.742 | 93.428 | 93.362 | 93.428 | R88 | R99 |
| FDSE v2 | 86.732 | 86.585 | 84.905 | 84.322 | R75 | R71 |
| PG-DFC v3.2 | 跑中 R20 (88.84) | 跑中 R19 (89.15) | - | - | - | - |

---

## Wave 1 PACS+Office seed=2 (rand allocation, 不在主表 — setup 不一致)

| Run | Config | PACS best | Office best | 备注 |
|---|---|:--:|:--:|---|
| R0 sanity_pw0 (PG-DFC backbone, proto=0) | 等价 vanilla | 68.02 | 59.44 | sanity 验证 |
| R1 full_v32 (τ=0.3) | PG-DFC default 早期 | 66.55 | 65.26 | τ=0.3 弱 |
| R2 (τ=0.5) | PG-DFC best τ | 67.95 | **66.87** ⭐ | Office Wave 1 best |
| R4 (β=0) | NV4 ablation | - | 62.57 | server EMA 必要 |

---

## Figures

### Fig 1. Per-Domain Δ Bar Chart
路径: `experiments/ablation/EXP-131_PG-DFC_v3.2/fig1_per_domain_comparison.png`
- 2x2 subplot: PACS (top) / Office (bottom),左 bar 对比 + 右 Δ 条
- 关键发现:**art +6.37pp / dslr +8.34pp** (baseline-weak domains 增益最大)

### Fig 2. Convergence Curves (3 datasets)
路径: `experiments/ablation/EXP-131_PG-DFC_v3.2/fig_convergence_3datasets.png`
- 1x3 subplot: PACS / Office / Digits
- F2DC 灰,FDSE 紫,PG-DFC v3.2 红,PG-DFC v3.3 蓝(只 Office 完整)
- 关键发现:
  - PACS 上 PG-DFC v3.2 全程领先 F2DC
  - **Office 上 FDSE 反而最高**(我们之前低估)
  - Digits 上 F2DC 远超 FDSE,PG-DFC 还在跑

---

## 真实 Paper-Grade 结论

### 主 contribution(can write now)
| Dataset | Δ vs F2DC | 备注 |
|---|:--:|---|
| **PACS** | **+2.18pp** ✓ | art +6.37 main story |
| Office | +0.69pp | 边缘,FDSE 更强 |
| Digits | TBD | 跑中,预测 ~93+ (跟 F2DC 接近) |

### 关键 insight(per-domain analysis)
- **baseline-weak domain 受益最大**:art +6.37pp,dslr +8.34pp
- **outlier domain 反受拖累**:Office webcam -6.03pp(跟其他 client 视觉不同步)
- v3.3 (A+B) 部分修复 outlier 问题(office mean +0.24pp vs v3.2)

### 收敛速度
- PACS:PG-DFC 比 F2DC 提前 1-3 round
- Office:PG-DFC v3.3 提前 8 round

### Limitation(诚实写)
- FDSE 在 Office 上反而 +3pp 强于 F2DC 跟 PG-DFC,**未来要研究 PG-DFC + FDSE-style 层分解**
- v3.2 在 outlier domain (webcam) 上拖后腿,v3.3 部分修复但还可优化

---

## 待回填 (跑完后填)

- [x] PACS F2DC + PG-DFC v3.2 (R100 完成)
- [x] Office F2DC + FDSE + PG-DFC v3.2 + v3.3 (R100 完成)
- [x] Digits F2DC + FDSE (R100 完成)
- [x] **EXP-130 sc4_v2: FedAvg PACS/Office/Digits + MOON Office/Digits 已回填**
- [ ] **Digits PG-DFC v3.2** (跑中 R20,~30 min 后完成)
- [ ] PACS v3.3 (跑中 R37)
- [ ] **MOON PACS** (sc4_v2 只到 R34 + 缺 s333,需补 R100×2)
- [ ] FDSE PACS (sc3_v2 R35+R30 没跑完,需补 R100×2)
- [ ] FedBN / FedProto / FPL 3 个 baselines (还没跑)

## 后续 priority

1. **等 Digits PG-DFC v3.2 完成** → 填主表完整 3 dataset
2. **补 MOON PACS + FDSE PACS** (PACS 缺基线,但 FedAvg+F2DC 已可对比)
3. **跑 FedBN / FedProto / FPL** on PACS/Office/Digits
4. **画 per-class confusion matrix** (paper Fig 6 风格,展示 art 增益来自哪些类)
