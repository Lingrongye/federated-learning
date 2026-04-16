# EXP-081 | FDSE 基线补 s=42 — 同 seed 严格对比

## 基本信息
- **日期**: 2026-04-16 启动 / 2026-04-17 结果
- **算法**: fdse (原版)
- **服务器**: SC2 GPU 0
- **状态**: 🔄 运行中

## 动机

今日 EXP-076/078 的 orth_only / mse_alpha 用了 seed {2, 333, 42}，但 FDSE 基线（EXP-049 PACS 5-seed / EXP-051 Office 3-seed）没有 s=42。

虽然 EXP-079 正在补 s=15 对齐基线的方向，但双向对齐更可靠：也让 FDSE 补 s=42，这样两边可以同时用 3-seed {2, 333, 42} 做同 seed 均值对比（最严格）。

## 实验矩阵 (2 runs)

| # | 数据集 | Seed | Config | 对比对象 |
|---|--------|------|--------|---------|
| 1 | PACS_c4 | 42 | fdse_r200.yml | EXP-076 orth_only s=42 |
| 2 | office_caltech10_c4 | 42 | fdse_office_r200.yml | EXP-076 orth_only s=42 |

## 成功标准

本实验不是为了得到改进，而是**补齐基线数据**，让 summary 可以报告三种对齐方式：

1. **原始 3-seed 对比**（seed 不对齐）
2. **公共 seed {2, 333}** 2-seed 子集
3. **补齐后 3-seed {2, 333, 42}**（EXP-081 结果 + 今日 EXP-076）

以及 EXP-079 提供的 {2, 15, 333} 另一个方向的 3-seed。

## 基线参考（现有 seed）

### PACS R200 FDSE (EXP-049 5-seed)
| seed | AVG Best | AVG Last |
|------|----------|---------|
| 2 | 80.81 | 78.09 |
| 15 | ? | ? |
| 333 | ? | ? |
| 4388 | ~80 | ~75 |
| 967 | ? | ? |
| **mean** | **80.24 ± 0.75** | 75.57 |

### Office R200 FDSE (EXP-051 3-seed)
| seed | ALL Best | AVG Best |
|------|----------|----------|
| 2 | 88.10 | 92.39 |
| 15 | 86.92 | 91.24 |
| 333 | 84.12 | 88.11 |
| **mean** | **86.38 ± 2.01** | **90.58 ± 2.22** |

## 结果（待填）

### PACS FDSE s=42
| R | ALL Best | ALL Last | AVG Best | AVG Last | drop |
|---|----------|---------|----------|---------|------|
| - | - | - | - | - | - |

### Office FDSE s=42
| R | ALL Best | ALL Last | AVG Best | AVG Last | drop |
|---|----------|---------|----------|---------|------|
| - | - | - | - | - | - |

## 合并后的严格同 seed 对比（待填）

### PACS {2, 333, 42} 3-seed mean
| 方法 | AVG Best | AVG Last | vs orth_only |
|------|----------|---------|--------------|
| FDSE | ? | ? | ? |
| orth_only (EXP-076) | 81.4 | 80.7 | baseline |

### Office {2, 333, 42} 3-seed mean
| 方法 | ALL Best | AVG Best | vs orth_only |
|------|----------|----------|--------------|
| FDSE | ? | ? | ? |
| orth_only (EXP-076) | ? | 89.4 | baseline |
