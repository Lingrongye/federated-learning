# EXP-086 | PACS 方案 A (sas=1) ❌ 失败实验

## 基本信息
- **日期**: 2026-04-17 启动 / 2026-04-18 完成
- **算法**: feddsa_scheduled (sm=0, sas=1, sas_tau=0.3, LR=0.05)
- **服务器**: SC2 GPU 0
- **状态**: ✅ R200 全部完成，**结果为负**

## 动机

EXP-084 方案 A 在 Office 上 Caltech +3.6% 显著提升，AVG +1.21/+0.98。假设：**PACS 的 Art 也是 style outlier，方案 A 应该也能帮 Art**。

## 🚨 核心结果：方案 A 在 PACS 无效，甚至变差

### PACS 3-seed {2, 15, 333}

| seed | ALL Best | ALL Last | AVG Best | AVG Last | Art(c0) B/L |
|------|---------|---------|---------|---------|------------|
| 2 | 81.84 | 81.24 | 80.11 | 79.37 | 65.7/63.2 |
| 15 | 81.94 | 81.64 | 80.02 | 79.63 | 62.7/60.8 |
| 333 | 81.04 | 78.63 | 79.16 | 76.17 | 64.2/56.9 |
| **mean** | **81.61** | **80.50** | **79.76** | **78.39** | **64.2/60.3** |

### 对比 orth_only LR=0.05 (no sas) baseline

| 指标 | baseline (MASTER) | 方案 A (sas=1) | Δ |
|------|-------------------|---------------|---|
| ALL Best | 82.31 | 81.61 | -0.70 |
| ALL Last | 81.17 | 80.50 | -0.67 |
| AVG Best | 80.41 | 79.76 | **-0.65** ❌ |
| AVG Last | 79.42 | 78.39 | **-1.03** ❌ |

Art per-seed Δ（sas - baseline）：
- s=2: **-4.9/-2.0** ❌
- s=15: +0.4/-0.5 ≈ 0
- s=333: +1.5/-2.9 (Last drop 2.9%)

## 🔍 根因分析

**Office 上方案 A 有效**：
- Caltech 是唯一 style outlier（自然图）
- Amazon/DSLR/Webcam 都是白底/棚拍，相互相似度高
- sas 让 Caltech 不被 3 个相似域稀释 → 受益

**PACS 上方案 A 失效**：
- **Art / Cartoon / Photo / Sketch 4 域风格互相都差异大**
- 每个 client 的 style_proto 跟其他 3 个都远 → sas 个性化后每个 client 都"退化到 FedBN-like 本地化"
- 失去跨域知识共享 → 性能降

**机制假设**：
> 方案 A 仅对"多数相似 client + 少数 outlier"的分布有效。
> 对"全部都 outlier"的分布反而有害。

## 📋 论文决策

1. **PACS 主方法用 orth_only LR=0.05**（MASTER AVG 80.41/79.42, 超 FDSE +0.50/+1.87）
2. **Office 主方法用 orth_only + sas τ=?** (τ 最优值待 EXP-092 3-seed 确认，可能 0.3/0.5/3.0)
3. **论文叙事**：方案 A 有适用条件 — Office Caltech 式 outlier 场景。PACS 的 4-domain 均匀 outlier 是 counter-example，正好验证机制假设。

## 下一步

- ✅ 不再对 PACS 部署 sas 变体
- 🎯 Office 继续探索最优 sas_tau（EXP-092 τ=0.5/3.0 × 3-seed, LR=0.025 × 3-seed）
- 📝 论文 abstract 修改："We propose sas aggregation for **outlier domain enhancement** (Office), while orth_only already excels on PACS"
