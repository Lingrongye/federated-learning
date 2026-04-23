# EXP-123 Stage A — PACS Art Diagnostic Report
生成时间: 2026-04-23 06:08 (v1)
更新时间: 2026-04-23 (v2 paper-standard metric)

## ⚠️ Metric 修正说明

- **v1 (wrong)**: per-(seed, client) 各自在自己的 best round 取峰值 → 每 client 不同 round, 高估
- **v2 (correct, paper standard)**: 每 seed 找 mean_local_test_accuracy 的 best global round, 所有 client 都在**同一 round** 取值 → 真实反映该 run 在 best 时刻的域表现

**v2 是 FDSE paper 及联邦学习通用规范的做法, 后续以 v2 为准。**

## Data sources
- **orth_only**: 3 seeds [2, 15, 333], config `feddsa_scheduled_lo1.0_lh0.0_ls1.0_tau0.2_*_Mfeddsa_scheduled`
- **fdse**: 3 seeds [2, 15, 333], config `fdse_lmbd0.5_tau0.5_beta0.001_Mfdse`
- LR = 5.00e-02 (both), R = 200

## Per-seed 详细 (paper-standard, R_best 见 best_round 列)

| Method | Seed | R_best | AVG Best | AVG Last | Art | Cartoon | Photo | Sketch |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| orth_only | 2 | 150 | 81.87 | 80.37 | 70.10 | 86.32 | 82.04 | 89.03 |
| orth_only | 15 | 98 | 80.08 | 79.65 | 60.78 | 88.03 | 81.44 | 90.05 |
| orth_only | 333 | 135 | 79.28 | 78.23 | 59.31 | 86.32 | 81.44 | 90.05 |
| fdse | 2 | 185 | 80.81 | 78.09 | 73.53 | 85.04 | 78.44 | 86.22 |
| fdse | 15 | 120 | 79.00 | 76.64 | 63.73 | 84.62 | 79.64 | 88.01 |
| fdse | 333 | 181 | 79.93 | 77.92 | 62.25 | 83.76 | 84.43 | 89.29 |

## 3-seed mean ± std (paper-standard)

| Method | AVG Best | AVG Last | Art | Cartoon | Photo | Sketch |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| orth_only | **80.41 ± 1.08** | **79.42 ± 0.89** | 63.40 ± 4.78 | **86.89 ± 0.81** | **81.64 ± 0.28** | **89.71 ± 0.48** |
| fdse | 79.91 ± 0.74 | 77.55 ± 0.65 | **66.50 ± 5.00** | 84.47 ± 0.53 | 80.84 ± 2.59 | 87.84 ± 1.26 |

## orth_only vs FDSE per-domain Δ (paper-standard)

| Domain | orth_only | FDSE | Δ (orth − FDSE) |
|---|:-:|:-:|:-:|
| **Art** | 63.40 ± 4.78 | **66.50 ± 5.00** | **-3.10 ❌** |
| **Cartoon** | **86.89** | 84.47 | **+2.42 ✅** |
| **Photo** | **81.64** | 80.84 | **+0.80 ✅** |
| **Sketch** | **89.71** | 87.84 | **+1.87 ✅** |
| **AVG Best** | **80.41** | 79.91 | **+0.50 ✅** |
| **AVG Last** | **79.42** | 77.55 | **+1.87 ✅** |

## 核心结论

1. **整体 orth_only 赢 FDSE** (+0.50 Best / +1.87 Last) — 满足 CLAUDE.md 硬指标要求 (>79.91)
2. **Art 域 FDSE 赢 -3.10** — 且 3/3 seeds 一致 (非波动), 是系统性劣势
3. **Art 方差极大** (≈5.0 for both) — seed 间差 10+pp, 说明 Art 本身难度高 (光影/抽象风格)
4. **Last 指标 orth_only 领先更多** (+1.87 vs +0.50 for Best) — orth_only 训练更稳定, 不过拟合
5. **orth_only 劣势集中在 Art 域** — 其他 3 域全赢, 只要攻下 Art 就能全面领先

## v1 vs v2 对比

| 指标 | v1 (wrong) | v2 (correct) | 差异 |
|---|:-:|:-:|:-:|
| Art orth_only | 65.20 | 63.40 | -1.80 |
| Art FDSE | 69.61 | 66.50 | -3.11 |
| **Art Δ** | **-4.41** | **-3.10** | +1.31 (v1 高估劣势) |

v2 的 Art 差距比 v1 小 ~1.3pp, 但**方向不变** (FDSE 仍赢)。v1 把 FDSE 的 Art 高估了 3pp。

## Figures
- `fig1_per_domain_curves.png` — per-domain curve 3-seed mean ± std (v1 metric, 仅参考曲线形状)
- `fig2_domain_gap.png` — domain gap (max - min) over rounds

## 下一步决策 (基于 v2 数据)

### 选项 A: 直接开 Stage B (9 runs × diagnostic hooks)

跑 FedBN + orth_only + FDSE × 3 seeds 并 hook 进 diagnostic 信号 (per-round per-domain confusion matrix, ECE, t-SNE features)。
- 目标: 理解 FDSE 为什么在 Art 上赢 3pp
- 成本: seetacloud2 ~5h wall
- 价值: 方法设计有数据支撑, 不拍脑袋

### 选项 B: 先基于 v2 数据做 paper-review, 不跑 Stage B

v2 已经有 per-domain 3-seed mean + peak round + std, 其实诊断信号够用:
- FDSE R_best 普遍靠后 (185/120/181) → FDSE 是"慢热"型
- orth_only R_best (150/98/135) → 中早期达峰
- Art 在 FDSE 的 R_best 通常也是 Art 的峰 → FDSE 给 Art 更多训练时间
- 方差 5pp → Art 需要 seed-robust 的方法

→ 可以直接设计 targeted intervention for Art

### 选项 C: 轻量化 Stage B

只跑 3 seeds × orth_only 加 hook, 不跑 FDSE (FDSE 记录已经有了, 只缺我们自己的细节)
- 成本: ~2h
- 价值: 对准 orth_only 的 Art 失败模式
