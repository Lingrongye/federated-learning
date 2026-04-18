# EXP-091 | LR=0.025 + 方案 A (sas=1) 组合

## 基本信息
- **日期**: 2026-04-18
- **服务器**: Lab-lry GPU 1
- **算法**: feddsa_scheduled (sm=0, sas=1, sas_tau=0.3, LR=**0.025**)
- **状态**: ❌ 全部 SIGKILL 放弃（Lab-lry 被 wjc 抢 GPU，SC2 未重跑）

## 动机分析

基于现有结果反思：
1. **方案 A Office 3-seed 已验证有效**（AVG Best +1.21, Caltech +2.4/+3.6）
2. **LR=0.05 已经是 orth_only 最稳配置**（drop ~1%）
3. **但 Office 仍差 FDSE -0.94% AVG Last**
4. **PACS 方案 A 还在跑**（EXP-086）

**假设**：LR=0.05 → 0.025 可以进一步稳定后期收敛，**结合 sas=1 的个性化聚合**，看能否把 AVG Last 再拉近 FDSE。

之前 LR=0.05 在 PACS 上 drop 从 8.19% 降到 0.99% — **还能更低吗？LR=0.025 + decay 0.9998 在 R200 时 LR ≈ 0.017**，更稳定。

风险：LR 太低可能让 Best 也降（收敛不足）。要看 R200 下 LR=0.025 的 Best vs Last。

## 实验矩阵 (6 runs)

| #   | Task   | LR    | sas | Seed       |
| --- | ------ | ----- | --- | ---------- |
| 1-3 | PACS   | 0.025 | 1   | 2, 15, 333 |
| 4-6 | Office | 0.025 | 1   | 2, 15, 333 |

## 对照基线

**PACS**：
| 方法 | AVG Best | AVG Last | drop |
|------|---------|---------|------|
| orth_only LR=0.05 (baseline) | 80.41 | 79.42 | 0.99 |
| **LR=0.025 + sas** | 待 | 待 | 待 |

**Office**：
| 方法 | AVG Best | AVG Last |
|------|---------|---------|
| baseline sas=0 LR=0.05 | 88.61 | 87.30 |
| **sas=1 LR=0.05 (EXP-084)** | **89.82** | **88.28** |
| FDSE | 90.58 | 89.22 |
| **LR=0.025 + sas (本实验)** | 待 | 待 |

## 成功标准

1. **AVG Last 继续超 baseline**（PACS > 80; Office > 88）
2. **drop < 1%**（保持方案 A 的稳定性）
3. **如果 Office 能到 89+** → 逼近 FDSE → 新 SOTA 候选

## 结果（待填）

| Task | seed | ALL Best/Last | AVG Best/Last | drop |
|------|------|---------------|---------------|------|
| PACS | 2 | ❌ SIGKILL | ❌ SIGKILL | — |
| PACS | 15 | ❌ SIGKILL | ❌ SIGKILL | — |
| PACS | 333 | ❌ SIGKILL | ❌ SIGKILL | — |
| Office | 2 | ❌ SIGKILL | ❌ SIGKILL | — |
| Office | 15 | ❌ SIGKILL | ❌ SIGKILL | — |
| Office | 333 | ❌ SIGKILL | ❌ SIGKILL | — |

**放弃原因**：Lab-lry 6 runs 全被 wjc SIGKILL（log rounds=0）。LR=0.05 + sas τ=0.3 已在 Office 验证有效（EXP-084 AVG 89.82），LR=0.025 的额外收益不确定且投入过高（需重占 SC2 6h），放弃此方向。
