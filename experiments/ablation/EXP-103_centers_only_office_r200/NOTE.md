# EXP-103: Plan A + Class-Centers-only Office R200 3-seed

**日期**: 2026-04-20 启动 / 待完成
**状态**: 🟡 部署中

## 大白话

> 去掉 whitening, 只保留 class_centers 收集. 看 class_centers 单独有多少贡献.

## Claim

| Claim | 判定 |
|-------|------|
| class_centers 单独贡献 ≥ 1% | AVG Best ≥ 83.55 |
| class_centers << whitening | Δ centers < Δ whitening (EXP-102 结果) |

## 配置

| use_etf | use_whitening | use_centers | diag |
|---------|---------------|-------------|------|
| 0 | **0** | **1** | 0 |

Seeds {2, 15, 333}, R200, LR=0.05, E=1

## 🏆 结果 (待回填)

| 配置 | seed | AVG Best | Δ vs Plan A |
|------|------|---------|-------------|
| Plan A | mean | 82.55 | baseline |
| EXP-102 whitening only | mean | **89.26 ± 0.83** | **+6.71** 🔥 |
| **本实验 (centers only)** | mean | **88.41 ± 0.53** | **+5.86** |
| EXP-100 full (whitening+centers+diag) | mean | 88.75 | +6.20 |

**Per-seed (本实验, centers only)**:
| seed | ALL Best | AVG Best | AVG Last | Caltech | Amazon | DSLR | Webcam |
|------|---------|---------|---------|---------|--------|------|--------|
| 2 | 80.57 | 87.90 | 86.93 | 68.8/67.0 | 86.3/84.2 | 100.0/100.0 | 96.6/96.6 |
| 15 | 82.54 | 88.20 | 85.53 | 72.3/74.1 | 87.4/85.3 | 100.0/100.0 | 93.1/82.8 |
| 333 | 84.53 | 89.14 | 87.51 | 73.2/72.3 | 93.7/94.7 | 100.0/93.3 | 89.7/89.7 |
| **mean** | 82.55 ± 1.62 | **88.41 ± 0.53** | **86.66 ± 0.83** | 71.4/71.1 | 89.1/88.1 | 100.0/97.8 | 93.1/89.7 |

**消融结论**:
- whitening 单独贡献: **+6.71pp** (Plan A 82.55 → whitening only 89.26)
- centers 单独贡献: **+5.86pp** (Plan A → centers only 88.41)
- whitening+centers+diag 组合 (EXP-100): 88.75pp (比单 whitening 略低, 说明 diag 本身或 centers 可能轻微干扰)
- **whitening only (EXP-102) 反而是 Office 最强 config** (89.26pp)

## 📎 相关

- Config: `FDSE_CVPR25/config/office/feddsa_centers_only_office_r200.yml`
