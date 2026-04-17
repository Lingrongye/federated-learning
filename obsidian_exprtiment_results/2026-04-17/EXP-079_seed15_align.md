# EXP-079 | Seed 对齐补跑 — s=15 for orth_only / mse_alpha

## 基本信息
- **日期**: 2026-04-16 启动 / 2026-04-17 回填
- **算法**: feddsa_scheduled (mode 0 / mode 6)
- **服务器**: SC2 GPU 0
- **状态**: ✅ R200 全部完成

## 动机

EXP-076/078 的 orth_only / mse_alpha 用 seeds {2, 333, 42}，与 FDSE 官方默认 {2, 15, 333} 不对齐。补 s=15 让公共 seed 集变成 {2, 15, 333}，与基线严格对齐。

## 变体通俗解释

- **orth_only (mode=0)**: 纯 CE + L_orth（正交约束从 R0 全开）
- **mse_alpha (mode=6)**: CE + L_orth + alpha-sparsity InfoNCE + MSE 锚点

## 结果（R200 完整，ALL Best/Last \| AVG Best/Last）

### PACS s=15

| Config | R | ALL Best | ALL Last | AVG Best | AVG Last | drop (AVG) |
|--------|---|---------|---------|---------|---------|-----------|
| orth_only LR=0.1 | 201 | 81.94 | 76.82 | 80.29 | 73.77 | **6.52** ❌ |
| mse_alpha LR=0.1 | 201 | 80.94 | 74.92 | 79.15 | 72.16 | **6.99** ❌ |

### Office s=15

| Config | R | ALL Best | ALL Last | AVG Best | AVG Last | drop (AVG) |
|--------|---|---------|---------|---------|---------|-----------|
| orth_only LR=0.1 | 201 | 81.37 | 80.56 | 88.43 | 86.07 | 2.36 |
| **mse_alpha LR=0.1** | 201 | **83.35** | **82.95** | **89.55** | **89.28** | **0.27** ★ |

## 3-seed {2, 15, 333} 合并（含 EXP-076 s=2/333 + 本次 s=15）

### PACS R200 3-seed mean

| 方法 | ALL Best | ALL Last | AVG Best | AVG Last |
|------|---------|---------|---------|---------|
| **orth_only LR=0.1** | 82.98 | 75.08 | 81.35 | **72.21** ❌ |
| mse_alpha | 82.38 | 76.82 | 80.53 | 74.23 |
| FDSE (EXP-049) | **81.57** | **79.60** | **79.91** | **77.55** |

→ **LR=0.1 下 orth_only 不稳**；LR=0.05 版本（EXP-080）才能稳超 FDSE

### Office R200 3-seed mean

| 方法 | ALL Best | ALL Last | AVG Best | AVG Last |
|------|---------|---------|---------|---------|
| orth_only | 82.42 | 81.62 | 88.64 | 87.34 |
| mse_alpha | 82.68 | 81.48 | 87.93 | 87.25 |
| **FDSE** | **86.38** | **85.05** | **90.58** | **89.22** |

## 关键发现

1. **s=15 没救回 PACS orth_only LR=0.1** — 3-seed {2,15,333} AVG Last 72.21 仍远低于 FDSE 77.55
2. **Office mse_alpha s=15 drop 仅 0.27** — Office E=1 场景下 InfoNCE 变体可用，与 PACS E=5 的全面失败形成鲜明对比
3. **3-seed 严格同 seed 对比已可做** — 后续所有 PACS/Office 实验可直接对比 FDSE EXP-049/051
4. **LR=0.1 本身是问题的一部分** — 见 EXP-080，LR=0.05 才是真正突破

## 下一步

- LR=0.05 × PACS s=15/333 已在 EXP-080 完成
- Office LR=0.05 × s=2/15/333 也在 EXP-080 完成
- 本 EXP 结果用于 baseline 参考
