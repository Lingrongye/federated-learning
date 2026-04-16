# EXP-079 | Seed 对齐补跑 — s=15 for orth_only / mse_alpha

## 基本信息
- **日期**: 2026-04-16 启动 / 2026-04-17 结果
- **算法**: feddsa_scheduled (mode 0 / mode 6)
- **服务器**: SC2 GPU 0
- **状态**: 🔄 运行中

## 动机

今日 EXP-076/078 的 orth_only / mse_alpha 用了 seeds {2, 333, 42}，而所有基线（FDSE 论文、EXP-051 Office FedDSA/FDSE 复现、EXP-049 FDSE 5-seed）使用的是 **FDSE 官方源码 `run.py` 默认 seeds {2, 15, 333}**。seed 不对齐导致对比不公平。

本实验补跑 s=15，让我们今日实验的 seed 集扩展为 {2, 15, 333, 42} 或提取公共 {2, 15, 333} 3-seed 直接对齐基线。

## 变体通俗解释

- **orth_only (mode=0)**: 纯 CE + 正交约束（L_orth=1.0 从 R0），**无增强无 InfoNCE**。这是今日最稳定方案。
- **mse_alpha (mode=6)**: CE + L_orth + alpha-sparsity InfoNCE (tau=0.07, alpha=0.25) + MSE 锚点。EXP-077 R50 最佳但 R200 长期失效。

## 实验矩阵 (4 runs)

| # | Config | 数据集 | Seed | 算法文件 |
|---|--------|--------|------|---------|
| 1 | feddsa_orth_only.yml | PACS_c4 | 15 | feddsa_scheduled (mode=0) |
| 2 | feddsa_mse_alpha_r200.yml | PACS_c4 | 15 | feddsa_scheduled (mode=6) |
| 3 | feddsa_orth_only.yml | office_caltech10_c4 | 15 | feddsa_scheduled (mode=0) |
| 4 | feddsa_mse_alpha.yml | office_caltech10_c4 | 15 | feddsa_scheduled (mode=6) |

## 预期结果

基于今日 s=2/333/42 的 3-seed mean 推测 s=15 应当落在类似区间：

| Config | 数据集 | 预期 AVG Best | 预期 AVG Last |
|--------|--------|--------------|--------------|
| orth_only | PACS | 80-82 | 79-81 |
| mse_alpha | PACS | 80-82 | 76-78 (会下降) |
| orth_only | Office | 88-90 | 88-89 |
| mse_alpha | Office | 86-88 | 86-87 |

## 成功标准

- 补 s=15 后，**公共 seed 集 {2, 333}** → **3-seed {2, 15, 333}** 对比基线 EXP-049/051
- orth_only 3-seed mean (2,15,333) ≥ FDSE 3-seed mean (EXP-049 PACS 80.24, EXP-051 Office 90.58 AVG Best)

## 结果（待填）

### PACS
| Config | s | R | ALL Best | ALL Last | AVG Best | AVG Last | drop |
|--------|---|---|----------|---------|----------|---------|------|
| orth_only | 15 | - | - | - | - | - | - |
| mse_alpha | 15 | - | - | - | - | - | - |

### Office-Caltech10
| Config | s | R | ALL Best | ALL Last | AVG Best | AVG Last | drop |
|--------|---|---|----------|---------|----------|---------|------|
| orth_only | 15 | - | - | - | - | - | - |
| mse_alpha | 15 | - | - | - | - | - | - |

## 合并后的 3-seed 对比 {2, 15, 333}（待填）

| 方法 | 数据集 | ALL Best 3-seed | AVG Best 3-seed | vs FDSE |
|------|--------|----------------|----------------|---------|
| orth_only | PACS | - | - | - |
| orth_only | Office | - | - | - |
| FDSE (基线) | PACS | - | 80.24 (EXP-049) | baseline |
| FDSE (基线) | Office | 86.38 | 90.58 (EXP-051) | baseline |
