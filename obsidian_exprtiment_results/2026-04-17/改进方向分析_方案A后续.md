# 方案 A 后续改进方向分析

*日期: 2026-04-18 02:10 CST*

## 📊 现有结果盘点

### Office R200 3-seed {2,15,333} 已完成
| 方法 | ALL Best | AVG Best | AVG Last | Caltech |
|------|---------|---------|---------|---------|
| baseline (sas=0) | 82.55 | 88.61 | 87.30 | 72.6/70.2 |
| **方案 A (sas=1 τ=0.3)** | 84.40 | 89.82 | 88.28 | **75.0/73.8** |
| **Δ vs baseline** | +1.85 | +1.21 | **+0.98** | **+2.4/+3.6** ★ |
| FDSE | 86.38 | 90.58 | 89.22 | 78.9/77.7 |
| **gap 方案A-FDSE** | -1.98 | -0.76 | **-0.94** | -3.9 |

**Office 方案 A 已比 baseline 好，但距 FDSE 仍差 0.94-3.9%**。

### PACS R200 3-seed 还在跑（EXP-086 R76-77）

预期 3-5h 后出结果。已知 orth_only LR=0.05 baseline AVG Best 80.41 / Last 79.42。

## 🎯 改进方向候选（按可行性排序）

### A. 进一步降 LR（EXP-091 正在验证）
- **假设**：LR=0.025 + decay → R200 LR ≈ 0.017，后期更稳定收敛
- **风险**：LR 太低 → Best 也降（收敛不足）
- **已部署**：PACS + Office × 3-seed × LR=0.025 + sas=1

### B. sas_tau 最优搜索（EXP-088 正在跑 Office 5 档）
- 当前 τ=0.3，5 档扫 {0.05, 0.1, 0.5, 1.0, 3.0}
- 若 τ=0.1 或 0.5 明显好 → 扩 3-seed + 更新方案 A 主实验

### C. **sas 扩展到 classifier 层** (未做)
- 当前 sas 只作用在 `semantic_head`，`head (final FC classifier)` 还是 FedAvg
- 改：对 `head` 也做 style-aware 聚合
- 预期：Caltech 进一步 +1-2%（分类器也个性化）
- 风险：过度个性化 → 跨 client 知识共享减少

### D. **方案 A + Prototype 推理混合** (未做, 但 EXP-083 checkpoint 已存)
- 方案 B eval 发现 FC vs Proto 推理几乎一致（+0.13%）
- 但**融合两者**可能有 +0.3-0.5%：`logit = 0.7·FC + 0.3·ProtoDist`
- 实现简单，零成本（不改训练，只改推理）

### E. **FDSE 风格的 BN 聚合** (未做)
- FDSE 用 "similarity-aware DSE 聚合" — 跟我们方案 A 类似但针对 BN 模块
- 对齐 FDSE 的 per-layer personalization 强度
- 预期：+0.3-1%

### F. **FDSE 的 L_Con 一致性正则** (未做)
- FDSE 有一个 `L_Con` 损失项（拉近 DSE 输出的 BN 统计量与全局）
- 我们没有这个
- 风险：可能引入梯度冲突（之前 MSE anchor 已失败）

## 🔍 不值得做的方向

1. **PACS Local 复现** — flgo bug，引用论文值即可
2. **扩 5-seed** — FDG 领域 3-seed 已是惯例，变 paired t-test 附录即可
3. **MSE/InfoNCE 类损失项** — 已多次失败（078a/c），理论上不稳

## 📋 下一步部署（已完成）

| EXP         | 内容                                           | 服务器     | ETA  |
| ----------- | -------------------------------------------- | ------- | ---- |
| EXP-086     | PACS 方案 A × 3-seed (主)                       | SC2     | 3-4h |
| EXP-088     | Office sas_tau 5 档扫                          | SC2     | ~1h  |
| **EXP-091** | **LR=0.025 + sas 组合 × PACS/Office × 3-seed** | Lab-lry | ~4h  |

## 🔧 后续若验证成功可做

- **方案 A + 最优 sas_tau + 最优 LR** 组合 × 5-seed main table
- **方案 C (classifier 也 sas)** 做消融
- **方案 D (proto inference 融合)** 做 test-time ablation
- **E (BN 也 sas)** 做 per-module ablation

## 监控机制

每 30 分钟自动检查 SC2 + Lab-lry 进度，发现新完成 run → 回填 NOTE + 同步 Obsidian。
