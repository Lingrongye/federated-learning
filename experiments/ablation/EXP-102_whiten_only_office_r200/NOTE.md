# EXP-102: Plan A + Whitening-only Office R200 3-seed

**日期**: 2026-04-20 启动 / 待完成
**算法**: `feddsa_sgpa` (use_etf=0, use_whitening=1, use_centers=0, diag=0)
**服务器**: seetacloud2 (与 EXP-098 PACS 并行)
**状态**: 🟡 部署中

## 这个实验做什么 (大白话)

> EXP-100 Linear+whitening 在 Office 达 88.75% (+6.20 vs Plan A), 但**不知道**是 pooled whitening 还是 class_centers 收集带来的 gain. 这次去掉 class_centers 和 diag, **只保留 pooled whitening**, 看 Plan A + 纯 whitening 能到多少.

## Claim

| Claim | 判定 | 失败含义 |
|-------|------|---------|
| **whitening 单独贡献 ≥ 3%** | AVG Best ≥ 85.55 (Plan A 82.55 + 3) | whitening 贡献有限, gain 来自别处 |
| **whitening ≈ 完整基础设施** | AVG Best ≈ EXP-100 88.75 | whitening 是唯一真正有效机制 |

## 配置

| use_etf | use_whitening | use_centers | diag |
|---------|---------------|-------------|------|
| 0 | **1** | **0** | 0 |

Seeds {2, 15, 333}, R200, LR=0.05, E=1 (同 EXP-097/100)

## 🏆 结果 (待回填)

| 配置 | seed | ALL Best | AVG Best | AVG Last | Caltech | Amazon | DSLR | Webcam |
|------|------|---------|---------|---------|---------|--------|------|--------|
| Plan A orth_only | mean | 88.61 | 82.55 | 81.35 | 72.6/70.2 | 90.9/88.1 | 100.0/97.8 | 94.3/93.1 |
| Linear+whitening (EXP-100) | mean | 82.81 | **88.75** | 86.91 | 72.3/70.5 | 88.4/87.4 | 100.0/97.8 | 94.3/92.0 |
| **本实验 (whitening only)** | mean | 待填 | **待填** | 待填 | — | — | — | — |
| Δ 本 − Plan A | — | — | 待填 | — | — | — | — | — |
| Δ 本 − EXP-100 (centers+diag 副作用) | — | — | 待填 | — | — | — | — | — |

## 📎 相关

- EXP-100 对照: `experiments/ablation/EXP-100_linear_office_r200/NOTE.md`
- Config: `FDSE_CVPR25/config/office/feddsa_whiten_only_office_r200.yml`
