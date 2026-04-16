# 2026-04-17 | 每日实验概览

## 今日部署（基于 2026-04-16 分析的后续工作）

截至 04-17 凌晨（04-16 晚 23:00+ 启动），SC2 GPU 0 上同时运行 **10 个并发实验**，分 3 组：

### EXP-079 | Seed 对齐补跑（s=15）

**目的**：今日主实验用 seeds {2, 333, 42}，与 FDSE 官方默认 seeds {2, 15, 333} 不对齐。补 s=15 让我们的 3-seed 可以提取 {2, 15, 333} 与基线严格同 seed 对比。

| # | 数据集 | Config | Seed |
|---|--------|--------|------|
| 1 | PACS | feddsa_orth_only | 15 |
| 2 | PACS | feddsa_mse_alpha_r200 | 15 |
| 3 | Office | feddsa_orth_only | 15 |
| 4 | Office | feddsa_mse_alpha | 15 |

### EXP-080 | orth_only 超参扫

**目的**：今日验证 orth_only (L_orth=1.0, no HSIC, LR=0.1) 是最稳定方案。但超参未扫过，可能有优化空间。

| # | Config | 变化点 | 预期 |
|---|--------|--------|------|
| 5 | feddsa_orth_lo2 | L_orth=2.0 | 强解耦是否更好 |
| 6 | feddsa_orth_lo0p5 | L_orth=0.5 | 弱解耦是否够用 |
| 7 | feddsa_orth_hsic | +HSIC (lh=0.1) | 非线性独立是否助提升 |
| 8 | feddsa_orth_lr05 | LR=0.05 | 低 LR 是否更稳 |

全部 × PACS × s=2（对照 EXP-076 baseline AVG Best 80.1）

### EXP-081 | FDSE 基线 s=42 补齐

**目的**：让 FDSE 也有 s=42 数据，可构成 FDSE {2, 333, 42} 3-seed 与我们 orth_only {2, 333, 42} 3-seed 的严格同 seed 对比。

| # | 数据集 | 算法 | Seed |
|---|--------|------|------|
| 9 | PACS | fdse | 42 |
| 10 | Office | fdse | 42 |

## GPU 资源

- **SC2 GPU 0**: 10 runs 并发，显存 15GB/24GB（61%），GPU utilization 63%
- **SC4**: 实例关机状态，今日不可用
- **Lab-lry**: 本次跳过

## ETA

- Office runs (E=1): ~2h 完成
- PACS runs (E=5): ~6h 完成
- **所有 10 runs 全部完成预计 04-17 早上 6-8 点**

## 关键对比目标（待填）

### PACS R200 AVG Best — 严格 {2, 15, 333} 3-seed 对比

| 方法 | s=2 | s=15 | s=333 | 3-seed mean | vs FDSE |
|------|-----|------|-------|-------------|---------|
| FDSE (EXP-049/046) | 80.81 | ? | ? | 80.24 ± 0.75 | baseline |
| orth_only | 80.1 | **EXP-079 in progress** | 83.1 | - | - |

### Office R200 — 同 {2, 15, 333} 3-seed 对比

| 方法 | ALL Best 3s | AVG Best 3s | vs FDSE |
|------|------------|------------|---------|
| FDSE (EXP-051) | 86.38 ± 2.01 | 90.58 ± 2.22 | baseline |
| orth_only | 待填 | 待填 | - |

## 今日已完成实验的 NOTE.md 引用

- [EXP-079 seed 对齐补跑](EXP-079_seed15_align.md)
- [EXP-080 orth 超参扫](EXP-080_orth_hparam_sweep.md)
- [EXP-081 FDSE s=42 基线对齐](EXP-081_fdse_s42_align.md)

## 下一步（等实验跑完后）

1. 提取全部 10 runs 的 ALL + AVG 完整轨迹
2. 更新 EXP-079/080/081 NOTE.md 回填结果
3. 合并到 04-17 daily_summary，给出最终 3-seed 同 seed 对比结论
4. 若 EXP-080 某变体 > baseline → 扩展到 3-seed + Office 验证
5. 若 FDSE s=42 结果拉高其 3-seed mean → 重估 orth_only 相对优势
