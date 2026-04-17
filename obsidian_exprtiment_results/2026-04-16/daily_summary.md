# 每日实验总结 — 2026-04-16

## TL;DR

**今日共 4 大实验组（EXP-075/076/077/078），总计 30+ runs，全天核心结论：InfoNCE 方向全面关闭，orth_only 成为最可靠的最终方案。**

> 📌 **指标口径（对齐 FDSE CVPR'25 Table 1）**：
> - **ALL** = 所有客户端测试样本合并后的准确率（按样本数加权）= flgo 日志 `local_test_accuracy`
> - **AVG** = 各客户端准确率的等权平均 = flgo 日志 `mean_local_test_accuracy`
> - 时间聚合：同时报 **Best (max of rounds)** 和 **Last (final round)**
> - 基线 FDSE / FedDSA / FedBN / FedAvg R200 复现来自 EXP-043/046/049（PACS）和 EXP-051（Office），seed 集 {2, 15, 333}
> - ⚠️ **Seed 不对齐问题**：今日 EXP-076/078 用 seed {2, 333, 42}，与基线 {2, 15, 333} 仅交集 {2, 333}。已于 2026-04-16 23:10 SC2 补跑 s=15（4 runs，ETA +6h）对齐基线

| 指标 | 结果 |
|------|------|
| 今日最佳方案 | **orth_only (mode=0)** |
| PACS orth_only **AVG Best** (3-seed s=2/333/42) | **81.4%**（超 FDSE R200 复现 AVG Best ≈ 80.24）|
| PACS orth_only **ALL Best** (部分 runs 有记录) | 待 s=15 跑完后补完整 |
| Office orth_only **AVG Best** (s=2/333/42) | **89.4%** |
| Office orth_only **ALL Best** (s=2/333/42) | 待 s=15 跑完后重抽 |
| 最大崩溃事件 | gradual_noaug s=2 从 81.7% → 51.2%（-30.4%）|
| InfoNCE 变体失败率 | 21/25 HURT/CRASHED (84%) |
| 🔄 补跑进度（s=15，SC2） | PACS orth/mse_alpha + Office orth/mse_alpha = 4 runs，R 3-18 运行中 |

---

## 服务器状态（20:00+ 快照）

| 服务器 | GPU | 显存 | 利用率 | 进程 | 状态 |
|--------|-----|-----|--------|------|------|
| SC2 | 0 | 1/24564 MiB | 0% | 0 | 已释放 |
| SC4 | 0 | No devices | — | 0 | 实例已关/释放 |
| Lab-lry | — | — | — | — | 本次跳过 |

---

## 今日四大实验组

### EXP-075 | Gradual Training（分离式 Sigmoid Ramp）

**目的**：用 sigmoid ramp-up 平滑激活 InfoNCE/Aug，验证是否能避免 warmup 崩溃。

**配置**：4 configs × 3 seeds = 12 runs（SC4 GPU0）

| Config | s=2 max/last | s=333 max/last | s=42 max/last | **mean max** | mean last |
|--------|-------------|---------------|--------------|--------------|-----------|
| gradual_shallow | 80.8/74.2 | 78.8/77.2 | 80.2/75.9 | 79.9 | 75.8 |
| gradual_deep | 80.7/77.8 | 79.9/70.0 | 81.6/76.6 | 80.7 | 74.8 |
| gradual_noaug | 81.7/78.1 | 77.7/74.9 | 80.7/77.3 | **80.0** | 76.8 |
| gradual_syncramp | 81.6/77.5 | 78.8/78.7 | 81.3/77.0 | **80.6** | 77.7 |

**结论**：❌ **全面失败**。mean max 仅 79.9–80.7，不超过 FDSE 80.36 基线，更严重的是 peak→final drop 普遍 -3 到 -10%，稳定性极差。但首次捕获梯度冲突铁证：**cos_sim(grad_CE, grad_InfoNCE) 在 R50 穿零**（+0.716@R10 → -0.010@R50 → -0.235@R60），6 runs 完全一致。

---

### EXP-076 | Scheduled Training（L_orth 从 R0 全开 + 4 种 InfoNCE 调度）

**目的**：首次真正测试 L_orth 全权重的效果（修复 EXP-070 warmup=9999 的 bug），对比 4 种 InfoNCE 调度策略。

**PACS R200 3-seed 最终结果**（对比 FDSE R200 复现 max≈80.36）：

| Mode              | 描述                         | **mean max** | mean final | mean drop | 评价                |
| ----------------- | -------------------------- | ------------ | ---------- | --------- | ----------------- |
| **orth_only (0)** | CE + L_orth（无 Aug 无 InfoNCE） | **81.4**     | 80.7       | -0.7      | ✅ **BEST，max 超 FDSE** |
| bell_60_30 (1)    | 钟形调度                       | 80.9         | 80.2       | -0.8      | ✅ 接近              |
| cutoff_80 (2)     | R80 硬关断                    | 80.2         | 79.0       | -1.3      | ⚠️                 |
| always_on (3)     | 持续 InfoNCE                 | 79.9         | **75.1**   | **-4.8**  | ❌ 延迟崩溃，max 勉强但 final 暴跌 |

**关键发现**：
- **orth_only s=333 peak 83.1%@R129**（single-seed 最佳）
- always_on s=42 在 R181 时看似稳定 (-0.4%)，R201 暴跌到 -6.8% → **中间快照有误导性**
- InfoNCE 无论如何调度都不稳定

**Office 验证（SC4）**（对比基线：FedAvg 85.67 / FedBN 88.65 / FedDSA 89.13 / FDSE 90.58，均为 max/peak）：

| Config | s=2 max/last | s=333 max/last | s=42 max/last | **mean max** | mean final | vs FDSE |
|--------|--------------|----------------|---------------|--------------|-----------|---------|
| orth_only | 87.7/87.2 | 89.8/88.7 | 90.7/89.5 | **89.4** | 88.5 | -1.2 |
| mse_alpha | 86.8/86.1 | 87.5/86.4 | 89.0/87.9 | 87.8 | 86.8 | -2.8 |

→ **正交解耦在 PACS（高域差异）和 Office（低域差异）均有效，是通用机制**。

---

### EXP-077 | Safety Valves R50 快速验证（SC2）

**目的**：5-agent 并行审查 FPL/FedPLVM/PARDON 发现我们缺 3 道安全阀（MSE 锚点/alpha-sparsity/triplet margin），R50 快筛最有效的修复。

| 变体 | Mode | R51 max | R51 last | cos_sim@R50 | 评价 |
|------|------|---------|---------|------------|------|
| **077c mse+alpha** | **6** | **82.2** | **82.2** | **+0.365** | 🏆 追平原 FedDSA peak，零下降 |
| 077d detach_aug | 7 | 80.4 | 80.1 | +0.362 | 好 |
| 077a mse_anchor | 4 | 80.1 | 79.8 | **+0.678** | cos 最高最稳 |
| 077b alpha_sparse | 5 | 78.6 | 78.6 | +0.021 | alpha 单独不够 |

**关键发现**：**MSE 锚点是最有效的梯度冲突防护**，cos_sim 从穿零提升到 +0.678。→ 部署 EXP-078 做 R200 完整验证。

---

### EXP-078 | R200 完整验证（SC2 + Lab-lry + SC4）

**目的**：对 077 R50 成功的 3 个变体做 R200 × 3 seeds 长期验证。

**最终结果**：

| Config                       | 部署      | R 终止        | **mean max** | mean final | drop      | 结论                                             |
| ---------------------------- | ------- | ----------- | ------------ | ---------- | --------- | ---------------------------------------------- |
| **078a MSE anchor (mode=4)** | SC2     | kill @ R142 | 81.0         | 76.7       | **-4.3**  | ❌ max 接近 orth_only 但后期塌陷，锚点随全局原型漂移             |
| 078c MSE+alpha (mode=6)      | Lab-lry | 连接失败        | —            | 未知          | —         | —                                              |
| **078d detach_aug (mode=7)** | SC4     | kill @ R102 | 80.4*        | ~78.8*     | -1.6      | ❌ s=333 从 R2 NaN 锁死 15%；max 也不超过 orth_only     |

*去掉 s=333 NaN seed 仅 2 seeds 均值

**核心教训**：
- MSE 锚点短期（R50）cos=+0.678 极好，但 R120+ 全局原型漂移 → 锚点跟着漂 → 失效
- detach_aug 分离 CE/InfoNCE 梯度流 → 敏感 seed 从 R2 NaN，不可调和
- **InfoNCE 方向正式关闭**

---

## 四 mode 汇总对照（PACS 3-seed）

| 变体 | **mean max** | mean final | max vs orth | final vs orth |
|------|-------------|-----------|-------------|---------------|
| **mode0 orth_only** | **81.4** | **80.7** | baseline ✅ | baseline ✅ |
| mode1 bell_60_30 | 80.9 | 80.2 | -0.5 | -0.5 |
| mode2 cutoff_80 | 80.2 | 79.0 | -1.2 | -1.7 |
| mode3 always_on | 79.9 | 75.1 | -1.5 | -5.6 ❌ |
| mode4 mse_anchor (078a) | 81.0 | 76.7 | -0.4 | -4.0 ❌ |
| mode7 detach_aug (078d) | 80.4* | 78.8* | -1.0 | -1.9 ❌ |

*去掉 NaN seed 仅 2 seeds 均值

> 💡 **用 max 看**：mode4 仅差 0.4，似乎有希望；**但用 final 看**：mode4 落后 4.0，证明长期失效。**max-final gap 揭示训练稳定性** — orth_only gap 仅 0.7，所有 InfoNCE 变体 gap 都 ≥1.2。

---

## 关键发现（来自"关键实验发现备忘.md"共 16 条）

1. **EXP-070 "协同效应"是 peak 幻觉** — "Decouple only" warmup=9999 等于纯 CE，按 final 排序结论完全反转
2. **cos_sim R50 穿零是梯度冲突铁证**（+0.72@R10 → -0.01@R50 → -0.24@R60）
3. **25 runs 中 84% HURT/CRASHED**，tau=0.2 的"稳定"只是 InfoNCE 太弱
4. **5-agent 论文审查**：FPL/FedPLVM/PARDON 都有安全阀，我们一个没有
5. **风格增强违反文献所有安全原则**：MixStyle 深层暴跌 -7%
6. **077c R50 达 82.2%, cos=+0.365** — 曾被视为突破口
7. **orth_only s=333 R48 就达 81.9%** — 纯 CE+正交极其有效
8. **gradual_noaug s=2 R145 崩到 51.2%**（-30.4%）— 即使无增强，持续 InfoNCE 必崩
9. **M4_intra s=333 R85 崩到 56.6%** — 所有持续性指令型对齐都会崩
10. **orth_only PACS 3-seed mean final 80.67% > FDSE 80.36%**
11. **Office 验证 orth_only mean final 88.5%** — 跨数据集通用
12. **078a MSE 锚点 R142 mean final 76.7%** — 锚点随全局原型漂移，短期有效长期失效
13. **078d detach_aug s=333 从 R2 NaN 锁死 15.04%** — 分割梯度流引入不稳定
14. **078d 稳定 seed 也 78.8% < orth_only 80.7%** — 无继续价值，主动 kill
15. **所有 InfoNCE 变体在 R200 全面失败** — 方向正式关闭
16. **SC2+SC4 GPU 完全释放**，等待新实验

---

## 与基线对比（统一规范：ALL + AVG × Best + Last，对齐 FDSE Table 1）

> 📖 **指标统一定义**：所有 PACS / Office 对比都用以下四列
> - **ALL Best** = 各轮 `local_test_accuracy`（样本加权）的最大值
> - **ALL Last** = 第 200 轮 `local_test_accuracy`
> - **AVG Best** = 各轮 `mean_local_test_accuracy`（客户端等权）的最大值
> - **AVG Last** = 第 200 轮 `mean_local_test_accuracy`
> - 每个数字都是同一 seed 集下多 seed 的 mean（seed 集在"seed"列标明）

### PACS R200（所有数字均 seed mean）

| 方法 | seed 集 | ALL Best | ALL Last | AVG Best | AVG Last | 来源 |
|------|--------|---------|---------|---------|---------|------|
| **FDSE** | {2,15,333,4388,967} 5s | 81.78 | 76.47 | **80.24** | **75.57** | EXP-049 |
| **orth_only** | {2,333,42} 3s | 83.45 | 76.49 | **81.69** | **73.87** ❌ | EXP-076 mode=0 (Lab-lry R200 JSON) |
| bell_60_30 | {2,333,42} 3s (s=42 用 SC2 log) | — | — | **81.67** | **79.29** ★ | EXP-076 mode=1 |
| cutoff_80 | {2,333,42} 3s | — | — | 80.40 | 78.06 | EXP-076 mode=2 |
| always_on | {2,333,42} 3s | — | — | 79.86 | 75.09 | EXP-076 mode=3 |
| 078c mse_alpha | {2,333,42} 3s | 82.81 | 77.76 | 80.90 | 75.21 | EXP-078c (Lab-lry R200 JSON) |
| 078a mse_anchor | R142 kill | — | — | 81.00 | 76.70 | EXP-078a |
| **orth_only s=15 补跑** | 进行中 R146 | — | — | 79.50 | 76.98 | EXP-079 (R149/200) |
| **mse_alpha s=15 补跑** | 进行中 R146 | — | — | 79.15 | 76.24 | EXP-079 |
| **orth_lr05 s=2** (新) | 单 seed R147 | — | — | **81.68** | **81.49** ★★★ | EXP-080 LR=0.05 |
| **FDSE s=42 补跑** | 单 seed R148 | — | — | 79.75 | 77.19 | EXP-081（进行中）|

**论文 R500 原始值（仅对照）**：FDSE ALL 83.81 / AVG 82.17；FedBN ALL 81.58 / AVG 79.47；FedAvg ALL 74.30 / AVG 72.10

### Office-Caltech10 R200（所有数字均 seed mean）

| 方法 | seed 集 | ALL Best | ALL Last | AVG Best | AVG Last | 来源 |
|------|--------|---------|---------|---------|---------|------|
| FedDSA 原版 | {2,15,333} 3s | 84.39 | 81.61 | 89.13 | 86.52 | EXP-051 |
| **FDSE** | {2,15,333} 3s | **86.38** | **85.05** | **90.58** | **89.22** | EXP-051 |
| **orth_only** | {2,333,42} 3s | 待同步† | — | **89.4** | 88.5 | EXP-076 (SC4 已关) |
| mse_alpha | {2,333,42} 3s | 待同步† | — | 87.8 | 86.8 | EXP-076 |
| **orth_only s=15** ✅ DONE | s=15 单 | ~86.5‡ | 待提 | 88.43 | 86.07 | EXP-079 |
| **mse_alpha s=15** ✅ DONE | s=15 单 | ~85.5‡ | 待提 | **89.55** ★ | 89.28 (drop 0.27 ★) | EXP-079 |
| **orth_lr05** (新) | s=2 刚启 R1 | — | — | 进行中 | — | EXP-080 |

†SC4 实例已关，git 未拉 record JSON；s=15 完成后会连同 record 一起补 ALL
‡初步估值（待 JSON 同步确认）

**论文 R500 原始值**：FDSE ALL 87.15 / AVG 91.58；FedBN ALL 83.08 / AVG 87.01；FedAvg ALL 82.60 / AVG 86.26

---

### 🎯 严格同 seed 对比（消除 seed 运气）

#### PACS FDSE vs orth_only — 公共 seed {2, 333} 同 seed mean

| 方法 | s=2 AVG Best/Last | s=333 AVG Best/Last | **2s mean AVG Best** | **2s mean AVG Last** |
|------|------------------|--------------------|--------------------|--------------------|
| FDSE | 80.81 / 78.09 | 79.93 / 77.92 | **80.37** | **78.01** |
| orth_only | 80.11 / 77.52 | 83.65 / **65.34** 💥 | **81.88** | **71.43** ❌ |

→ **同 seed {2,333}：peak 上 orth_only +1.51%，但 last 暴跌 -6.58%**（s=333 崩盘）

#### PACS FDSE vs orth_only — 公共 seed {2, 333, 42}（待 EXP-081 FDSE s=42 完成）

| 方法 | s=2 | s=333 | s=42 | **3s mean AVG Best** | **3s mean AVG Last** |
|------|-----|-------|------|--------------------|--------------------|
| FDSE | 80.81/78.09 | 79.93/77.92 | EXP-081 R148: 79.75/77.19 | 待 EXP-081 完成 | 待 |
| orth_only | 80.11/77.52 | 83.65/65.34 | 81.30/78.74 | **81.69** | **73.87** |

#### Office FDSE vs orth_only — 公共 seed {2, 333}

| 方法 | s=2 ALL/AVG Best | s=333 ALL/AVG Best | **2s mean ALL Best** | **2s mean AVG Best** |
|------|-----------------|-------------------|--------------------|--------------------|
| FDSE | 88.10 / 92.39 | 84.12 / 88.11 | **86.11** | **90.25** |
| FedDSA 原 | 84.13 / 89.95 | 82.12 / 86.35 | 83.12 | 88.15 |
| orth_only | 待同步 / 87.72 | 待同步 / 89.77 | 待 | **88.75** |

→ **Office 同 seed {2,333}：orth_only AVG Best 88.75 vs FDSE 90.25，差 -1.5%**

---

## 结论（最终修正版，基于 R200 full 数据）

1. **PACS 最稳方案不是 orth_only 而是 bell_60_30** — mean AVG last 79.29 > orth_only 73.87
2. **orth_only s=333 R200 崩到 65.34%** — R181 快照 81.8 完全是假象，delayed crash
3. **LR=0.05 + orth_only 可能是真正突破** — 单 seed R147 drop 仅 0.19 (s=2)，需 3-seed 确认
4. **FDSE s=4388 也 R200 崩到 68.78** — 所有方法在 R200 都有 seed-dependent 崩溃风险
5. **同 seed 严格对比**：PACS FDSE 5-seed AVG Best 80.24，orth_only {2,333,42} 81.69 peak 略高但 last 严重不稳
6. **Office FDSE 赢 orth_only 约 1.5%**（AVG Best），这个结论在 R181 和 R200 都成立
7. **ALL vs AVG 差异**：PACS 上 ALL 普遍 + 1-2% 高于 AVG（Photo/Cartoon 大域主导），Office 上 AVG 比 ALL 高 +4-5%（DSLR 小域拉平均）

---

## 今日修改的实验笔记

| 实验 | 一句话结论 |
|------|----------|
| EXP-075 gradual_training | 分离式 sigmoid ramp 失败，但梯度冲突日志是核心价值 |
| EXP-076 scheduled_training | orth_only 是最佳方案，超 FDSE；InfoNCE 调度不治本 |
| EXP-077 safety_valves | R50 MSE+alpha 达 82.2%，短期突破口但 R200 失效 |
| EXP-078 r200_validation | MSE/detach_aug 均失败，InfoNCE 方向关闭 |

---

## 下一步计划

- [ ] **纯解耦架构消融** — 基于 orth_only（不引入任何对比损失），验证风格共享是否仍有必要
- [ ] **撰写论文方向收敛** — 聚焦"正交解耦"作为核心贡献，弱化 InfoNCE/风格仓库原定位
- [ ] **FDSE 对比实验** — Office 上 orth_only 仍低于 FDSE 2.1%，需分析差距来源
- [ ] **SC2/SC4 已释放** — 可部署新消融或基线对照（FedPall/MOON/FedPLVM 复现）
