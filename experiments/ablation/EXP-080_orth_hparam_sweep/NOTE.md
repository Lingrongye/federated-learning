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

## 实验矩阵 (4 runs × s=2)

| # | Config | L_orth | HSIC | LR | 预期 |
|---|--------|--------|------|------|------|
| 1 | feddsa_orth_lo2.yml | 2.0 | 0.0 | 0.1 | 更强解耦可能 +0.5% 或下降（过度约束） |
| 2 | feddsa_orth_lo0p5.yml | 0.5 | 0.0 | 0.1 | 弱解耦应略低（-0.5-1%） |
| 3 | feddsa_orth_hsic.yml | 1.0 | 0.1 | 0.1 | HSIC 可能捕获非线性依赖 +0.3-0.8% |
| 4 | feddsa_orth_lr05.yml | 1.0 | 0.0 | 0.05 | 低 LR 平滑收敛（看 drop 是否 < baseline） |

baseline: orth_only (L_orth=1.0, 无 HSIC) PACS s=2 R201 AVG Best 80.1% (EXP-076)

## 成功标准

- 任一变体 PACS AVG Best ≥ 81% 且 drop < 1% → 后续扩至 3-seed + Office
- HSIC 版本如果 > cos²-only 基线 → 支持我们原 FedDSA 方案的"HSIC 双层解耦"设计

## 结果 (R146-148 中期)

| Config | s | R | ALL Best | AVG Best | AVG Last | drop | vs baseline |
|--------|---|---|----------|---------|---------|------|-------------|
| **baseline lo=1.0** (EXP-076 R201) | 2 | 201 | ~90 | **80.1** | 79.2 | **0.9** | ref |
| lo0p5 (lo=0.5) | 2 | 148 | 91.07 | 80.82 | 77.48 | 3.34 | +0.7 peak, -1.7 last ❌ |
| **lo2 (lo=2.0)** | 2 | 148 | 90.05 | **82.02** ★ | 78.72 | 3.30 | **+1.9 peak**，drop 大 |
| hsic (+lh=0.1) | 2 | 147 | ? | 81.48 | 77.42 | 4.06 | +1.4 peak，drop 最差 |
| **lr05 (LR=0.05)** | 2 | 147 | ? | **81.68** ★ | **81.49** ★★★ | **0.19** ★★★ | **+1.6 peak + drop 最小！** |

## 🏆 关键发现 (中期 R147-148)

1. **LR=0.05 是当前最大改进** — drop 仅 0.19%，AVG Last 81.49 比 baseline last 79.2 **+2.3%**！
2. **lo=2.0 peak 82.02** — 最高 peak，但 drop 3.3 → overshoot，强正交导致后期波动
3. **HSIC 无明显收益** — peak +1.4，但 drop 4.1（反而更不稳）
4. **lo=0.5 全面输** — 弱正交不如标准，证明 orth 需要足够强度

## 下一步（立即）

扩展 **LR=0.05** 到多 seed 和 Office：

| Follow-up | 部署 | 目的 |
|-----------|------|------|
| PACS orth_lr05 × s={15, 333} | SC2 | 验证 LR=0.05 是否 3-seed 都稳 |
| PACS orth_lo2 × s={15, 333} | SC2 | 验证 lo=2.0 peak 高是否可重现 |
| Office orth_lr05 × s={2, 15, 333} | SC2 | Office 上是否同样受益 |
