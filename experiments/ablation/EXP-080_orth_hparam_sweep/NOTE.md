# EXP-080 | orth_only 超参扫 — L_orth 权重 + HSIC 独立性

## 基本信息
- **日期**: 2026-04-16 启动 / 2026-04-17 结果
- **算法**: feddsa_scheduled (mode=0, orth_only base)
- **服务器**: SC2 GPU 0
- **状态**: 🔄 运行中

## 动机

今日 EXP-076 验证了 orth_only (L_orth=1.0, 无 HSIC) 是最稳定方案，PACS 3-seed AVG Best mean 81.4%。但：

1. **L_orth=1.0 是否最优**？没扫过 L_orth 权重。可能 0.5（弱正交）或 2.0（强正交）更优。
2. **HSIC 独立性是否有帮助**？当前 lambda_hsic=0.0，只用了线性余弦正交。HSIC (Hilbert-Schmidt Independence Criterion) 可以捕获非线性依赖。理论上 HSIC=0 比余弦=0 更强的独立性。

如果 L_orth=2.0 或 +HSIC 能进一步提升，说明解耦强度还不够；如果不变或下降，说明 L_orth=1.0 已接近最优。

## 变体通俗解释

- **lo2**: `L_orth = 2.0 × cos²(z_sem, z_sty)`，正交约束加倍
- **lo0p5**: `L_orth = 0.5 × cos²(...)`，正交约束减半（测试 Pareto 下界）
- **hsic**: `L_orth = 1.0 × cos² + 0.1 × HSIC(z_sem, z_sty)`，线性正交 + 非线性独立

## 实验矩阵 (3 runs × s=2)

| # | Config | L_orth | HSIC | 预期 |
|---|--------|--------|------|------|
| 1 | feddsa_orth_lo2.yml | 2.0 | 0.0 | 更强解耦可能 +0.5% 或下降（过度约束） |
| 2 | feddsa_orth_lo0p5.yml | 0.5 | 0.0 | 弱解耦应略低（-0.5-1%） |
| 3 | feddsa_orth_hsic.yml | 1.0 | 0.1 | HSIC 可能捕获非线性依赖 +0.3-0.8% |

baseline: orth_only (L_orth=1.0, 无 HSIC) PACS s=2 R201 AVG Best 80.1% (EXP-076)

## 成功标准

- 任一变体 PACS AVG Best ≥ 81% 且 drop < 1% → 后续扩至 3-seed + Office
- HSIC 版本如果 > cos²-only 基线 → 支持我们原 FedDSA 方案的"HSIC 双层解耦"设计

## 结果（待填）

| Config | s | R | ALL Best | AVG Best | AVG Last | drop | vs baseline |
|--------|---|---|----------|---------|---------|------|-------------|
| lo0p5 | 2 | - | - | - | - | - | - |
| **baseline lo1.0** | 2 | 201 | ? | **80.1** | 79.2 | -0.9 | ref |
| lo2 | 2 | - | - | - | - | - | - |
| hsic | 2 | - | - | - | - | - | - |

## 下一步

- 如果 hparam 有明显改进 → 扩至 3-seed {2, 15, 333} PACS + Office 验证
- 如果 lo2 过度（特征空间过度约束导致 CE 退化）→ 搜 lo ∈ {1.2, 1.5}
- 如果 HSIC 有效 → 深究 HSIC kernel 带宽的影响
