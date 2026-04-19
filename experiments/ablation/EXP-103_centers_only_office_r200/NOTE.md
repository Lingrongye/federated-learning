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
| EXP-102 whitening only | mean | 待填 | 待填 |
| **本实验 (centers only)** | mean | 待填 | 待填 |
| EXP-100 full (whitening+centers+diag) | mean | 88.75 | +6.20 |

## 📎 相关

- Config: `FDSE_CVPR25/config/office/feddsa_centers_only_office_r200.yml`
