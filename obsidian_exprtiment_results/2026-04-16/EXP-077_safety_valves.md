# EXP-077 | Safety Valves — R50 快速验证 MSE 锚点 / Alpha-Sparsity / 组合

## 基本信息
- **日期**: 2026-04-16
- **算法**: feddsa_scheduled.py (modes 4-7)
- **服务器**: SC2
- **状态**: ✅ 全部完成 (R51)

## 动机

5-agent 并行审查 FPL/FedPLVM/PARDON 论文发现：所有成功的 FL 原型对齐方法都有"安全阀"（MSE 锚点 / alpha-sparsity / triplet margin），我们的 InfoNCE 一个都没有。
R50 快速验证哪个修复最有效——核心指标是 cos_sim(grad_CE, grad_InfoNCE) 是否仍为正。

## 变体解释

| Config | Mode | 一句话 | 灵感来源 |
|--------|------|--------|---------|
| **mse_anchor** | 4 | 标准 InfoNCE + MSE(z_sem, 同类原型)，tau=0.05 | FPL (CVPR'23) 的 L_UPCR |
| **alpha_sparse** | 5 | cos_sim^0.25 弱化正例梯度，tau=0.07 | FedPLVM (NeurIPS'24) 的核心创新 |
| **mse_alpha** | 6 | MSE 锚点 + alpha-sparsity 双重安全阀 | 组合两家之长 |
| **detach_aug** | 7 | 增强特征只走对比不走 CE + MSE + alpha | PARDON/DFA 思路 |

所有变体: L_orth 从 R0 全开，梯度冲突日志每 5 round 记录。

## 结果 ✅ COMPLETE

| 变体 | max | last | drop | **cos_sim@R50** | 评价 |
|------|-----|------|------|----------------|------|
| **077c mse+alpha (mode=6)** | **82.2** | **82.2** | **0.0** | **+0.365** | **🏆 最佳! 追平原始 FedDSA peak** |
| 077d detach_aug (mode=7) | 80.4 | 80.1 | -0.3 | +0.362 | 很好 |
| 077a mse_anchor (mode=4) | 80.1 | 79.8 | -0.3 | **+0.678** | cos 最高，MSE 锚点极稳 |
| 077b alpha_sparse (mode=5) | 78.6 | 78.6 | -0.1 | +0.021 | cos 边缘，alpha 单独不够 |

### 对比历史实验在 R50 的 cos_sim

| 方法 | R50 cos_sim | R50 Acc | 后续崩？ |
|------|------------|---------|---------|
| **077c mse+alpha** | **+0.365** | **82.2** | 极可能不崩 |
| **077a mse_anchor** | **+0.678** | 79.8 | 极可能不崩 |
| 原始 gradual_shallow | -0.010 | 79.2 | 已确认崩了 |
| 原始 gradual_noaug | +0.028 | 81.2 | 崩到 51% |

## 核心结论

1. **MSE 锚点 (FPL 思路) 是最有效的梯度冲突防护** — cos_sim 从接近零提升到 +0.678
2. **Alpha-sparsity 单独不够** — cos 仅 +0.021，需要配合 MSE
3. **MSE + alpha 组合最强** — 82.2% 追平 peak 且零下降
4. **cos_sim > 0.3 是安全线** — 077c 和 077d 都稳定在 +0.36

## 后续: → EXP-078 (R200 × 3 seeds 完整验证)
