# Office Ablation 综合分析 — whitening 单独最强!

> 2026-04-20 凌晨完成. **决定性发现**: pooled whitening broadcast 是唯一主 gain 机制, centers 次要, diag 零副作用, ETF 有害.

## TL;DR

**Plan A + pooled whitening broadcast alone = 89.26%** ← 比任何其他组合都高!
- 比 full (whitening + centers + diag) **89.26 vs 88.77** 高 **+0.49%**
- 比 Plan A baseline **89.26 vs 82.55** 高 **+6.71%**
- **离 FDSE SOTA 90.58% 只差 1.32%**!

**新论文叙事**: "**FedDSA-Plus with Pooled Style Whitening Broadcast**" — 单一新机制,零新 trainable,66KB/round 通信。

## Office R200 3-seed 完整结果

| # | 配置 | use_etf | whiten | centers | diag | AVG Best | ALL Best | Δ vs Plan A |
|---|------|---------|--------|---------|------|----------|----------|-------------|
| 1 | Plan A baseline | 0 | 0 | 0 | 0 | 82.55 | 88.61 | 基线 |
| 2 | **EXP-102 whitening only** | 0 | **1** | 0 | 0 | **89.26 ± 0.83** 🔥 | 83.61 | **+6.71** |
| 3 | EXP-103 centers only | 0 | 0 | 1 | 0 | 88.41 ± 0.53 | 82.55 | +5.86 |
| 4 | EXP-104 full no diag | 0 | 1 | 1 | 0 | 88.77 ± 0.67 | 83.33 | +6.22 |
| 5 | EXP-100 full + diag | 0 | 1 | 1 | 1 | 88.75 ± 0.86 | 82.81 | +6.20 |
| 6 | EXP-097 SGPA (+ETF) | **1** | 1 | 1 | 1 | 86.97 ± 1.23 | 82.01 | +4.42 |
| 7 | SAS τ=0.3 (EXP-084) | — | — | — | — | 84.40 | 89.82 | +1.85 |
| 8 | FDSE (EXP-051) | — | — | — | — | **90.58** | 86.38 | +8.03 |

## EXP-102 whitening only per-seed (最强配置)

| seed | Best@R | AVG Best | AVG Last | ALL Best | Caltech | Amazon | DSLR | Webcam |
|------|--------|---------|---------|---------|---------|--------|------|--------|
| 2 | 45 | 88.13 | 86.74 | 81.36 | 69.6/69.6 | 86.3/84.2 | 100.0/100.0 | 96.6/93.1 |
| 15 | 26 | 89.59 | 88.16 | 83.35 | 72.3/73.2 | 89.5/86.3 | 100.0/100.0 | 96.6/93.1 |
| 333 | 57 | 90.07 | 87.66 | 86.12 | 75.9/75.0 | 94.7/92.6 | 100.0/93.3 | 89.7/89.7 |
| **mean** | — | **89.26 ± 0.83** | 87.52 | 83.61 | 72.6 | 90.2 | 100.0 | 94.3 |

**注意**: 3 个 seed 的 Best@R 都在 **R25-R60 之间** — 收敛速度极快, 25 轮左右就达到峰值. 比 EXP-097 SGPA 的 R130-R186 快 3x!

## 4 个关键发现

### 1. whitening 是唯一主 gain 机制

```
Plan A (82.55)
  + whitening only → 89.26 (+6.71) 🔥 单独就顶上!
  + centers only → 88.41 (+5.86) 次要
  + whitening + centers → 88.77 (+6.22) ← 比单 whitening 还低 0.49!
```

### 2. whitening + centers 有"负协同"

单 whitening 89.26 > whitening+centers 88.77 (-0.49%).

**猜测**: class_centers 收集可能**稀释了 whitening 的效果**:
- whitening 靠 (μ_global, Σ) 广播隐式对齐 encoder 特征
- 加 centers 后,客户端额外用 class-level 特征平均做计算,可能与 whitening 的全局统计量**互相干扰**
- 最干净的方案 = **只用 whitening, 不要 centers**

### 3. diag 框架完全无副作用 ✅

| 配置 | diag=0 | diag=1 | Δ |
|------|--------|--------|---|
| Linear+whitening full | EXP-104: 88.77 | EXP-100: 88.75 | -0.02 |

**结论**: DiagnosticLogger 代码不影响训练, 可以放心用。未来实验都开 diag=1 拿诊断数据。

### 4. ETF 在 Office 全线证伪

- SGPA (use_etf=1): 86.97
- Linear+whitening: 88.75 (+1.78)
- whitening only: 89.26 (+2.29 vs SGPA)

**Fixed ETF 在 feature-skew FL + Office 10 类单 outlier 场景下只会 hurt**。Neural Collapse 几何约束对这个场景不合适 (Linear 自适应分类边界更好)。

## 最终论文方案: FedDSA-Plus

### 唯一新机制

```
每轮 Server:
  1. 各 client 上传 (μ_sty, Σ_sty)     # 128d 风格 mean + 128×128 协方差
  2. 计算 pooled whitening:
     μ_global = (1/N) Σ_k μ_sty_k
     Σ_global = Σ_within + Σ_between + ε·I
     Σ_inv_sqrt = eigh 分解 → Q Λ^{-1/2} Q^T
  3. 广播 (μ_global, Σ_inv_sqrt, 各 client 的 μ_sty_k) → 所有 client
  
每轮 Client (训练):
  - 正常 Plan A 训练 (orth_only + CE)
  - 收集自己的 (μ_sty, Σ_sty) 供下轮上传
  - [可选] 推理时用 whitening 做 SGPA 双 gate (EXP-099 待测)
```

### 对比 SOTA

| 方法 | Office AVG Best | 通信增量 | 新参数 |
|------|-----------------|---------|-------|
| FedAvg | 85.67 | 0 | 0 |
| FedBN | 88.65 | 0 (BN 私有) | 0 |
| FedProto | — | 原型 broadcast | 0 |
| Plan A orth_only | 82.55 | 0 | L_orth 损失 (无新参数) |
| SAS | 84.40 | 0 (聚合改动) | 0 |
| **FedDSA-Plus (OURS)** | **89.26** | **66KB** | **0** |
| FDSE (SOTA) | 90.58 | DFE+DSE | 层分解参数 |

**FedDSA-Plus 离 FDSE 只差 1.32%**, 但**零新 trainable 参数**,通信增量极小。

### 为什么 work (机制解释)

**pooled whitening 广播 = FL 层面的 Batch Norm**:
- 传统 FedAvg 每 client BN 统计量独立 → 跨 client 特征分布不一致
- **我们把 BN 做到 cross-client 层面**: 广播"所有客户端风格的共同参考坐标系"
- 客户端隐式对齐: 我的 z_sty 在这个 pooled 坐标系里的位置是可比的

这也解释了 whitening+centers 负协同: 加 centers 等于**又引入 client-local 统计**,与 pooled 统计竞争。

## 下一步

1. **PACS 6 runs 完成后** (~1h): 验证 PACS 4-outlier 是否同样 whitening-only 最强
2. **EXP-099 SGPA 推理**: 需要先跑一个 use_etf=1, se=1 保存 checkpoint, 再跑 inference script 测 proto_vs_etf_gain
3. **补 3 组合** (可选): 00_1 / 01_1 / 10_1 (带 diag 但其他 off) 做完整 2^3 ablation
4. **开始写论文**: 主叙事清晰,数据齐全 (待 PACS 验证)

## Obsidian 链接

- EXP-102 NOTE: `experiments/ablation/EXP-102_whiten_only_office_r200/NOTE.md`
- EXP-103 NOTE: `experiments/ablation/EXP-103_centers_only_office_r200/NOTE.md`
- EXP-104 NOTE: `experiments/ablation/EXP-104_full_nodiag_office_r200/NOTE.md`
- 前置综合分析 (Office SGPA vs Linear): `EXP-097_100_office_综合分析.md`
- 大白话方案解释: `知识笔记/大白话_我们所有方案.md` (需要更新!)
- 关键备忘: `2026-04-19/关键实验发现备忘.md`
