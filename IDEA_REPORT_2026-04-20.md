# Research Idea Report — FedDSA-SGPA 方案升级 (04-20)

**Direction**: 解决 "FedDSA-SGPA Linear+whitening 在 PACS -1.49pp / z_sty 被磨 95%" 的数据集边界问题, 且正面回应 "统计约束 L_orth+HSIC 无法告诉模型什么是风格" 的质疑.
**Generated**: 2026-04-20
**Ideas evaluated**: 12 生成 → 6 存活 → 3 重点推荐
**文献输入**: `LITERATURE_SURVEY_DANN_AND_DECOUPLING.md` (30 篇 2024-2026)

---

## Landscape Summary

文献综述揭示 3 个核心发现:

1. **Style Blind DG Segmentation (arXiv 2403.06122)** 明确: "feature normalization confuses semantic features because content and style are entangled" → **印证我们 PACS 失败的诊断**.
2. **RobustNet / ISW (CVPR 2021)** 证明选择性白化 > 全局白化, 但 **FL 场景无人做**, 这是空白.
3. **Deep Feature Disentanglement SCL (Cog. Comp. 2025)** 用 class label 监督 z_sem vs z_sty, 比 cos² 强很多, 但**没搬到 FL**.

**我们 2×2 象限 (解耦+风格资产化+FL) 仍无竞争者**, 可保留 novelty 主 claim.

---

## 12 个 Idea (首次 brainstorm)

### I1. Style-Aware Selective Whitening (SASW)

**Hypothesis**: 不是全擦风格, 按通道选择性擦除. Style-sensitive 通道强 whitening, content-sensitive 通道弱或不 whitening, PACS z_sty 得以保留.

**Mechanism**: 基于 RobustNet ISW 思路:
- 用 photometric transform 成对样本 → 特征 covariance 差异定义哪些维度是 style
- 服务器广播 per-channel whitening 权重 λ (style 通道 λ→1, content 通道 λ→0)
- 最终 whitening: `Σ_inv_sqrt_weighted = Σ_inv_sqrt * diag(λ)`

**Minimum experiment**: PACS R200 3 seeds, 对比 Linear+whitening(全擦) vs Linear+SASW.

**Risk**: **MEDIUM** - photometric transform 在 sketch 上不适用, 需退化到 class-level covariance.

**Effort**: 1-2 周. **Novelty**: 7.5/10.

---

### I2. Constrained DANN with Dual-Head (CDANN-Dual) ⭐主推

**Hypothesis**: 给 z_sem 反向梯度 (DANN 式), 给 z_sty 正向梯度到 domain classifier, "显式告诉模型"什么是风格.

**Mechanism**:
```
                 ┌→ sem_classifier (正向 CE on y) → z_sem 学类别
z → decouple ────┤
                 ├→ dom_head_sem (反向 GRL) → z_sem 不能识别 domain
                 └→ dom_head_sty (正向) → z_sty 必须能识别 domain
```
每 client 知道自己 domain id (= client id, 天然标签). 所有头本地算, 不泄露.

**Loss**:
```
L = L_CE(y, sem_classifier(z_sem))
  + λ_orth · cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)
  + λ_adv · CE(d, dom_head_sty(z_sty))
  − λ_adv · CE(d, dom_head_sem(GRL(z_sem)))   # GRL 内置负号
```

**Minimum experiment**: Office R100 + PACS R100, 2 seeds. 对比 baseline vs CDANN-Dual.

**Risk**: **LOW-MEDIUM** - DANN 在 FL 已有 ADCOL/FedPall 验证可行, 我们 "constrained + dual" 理论稳定.

**Effort**: 1 周 (加 dom_head + GRL 大概 50 行代码 + 单测).

**Novelty**: 9/10 - Deep Feature Disentanglement SCL 的 FL 扩展 + constrained, 零直接 prior.

---

### I3. Style Strength Detector (SSD) + 自适应 Whitening

**Hypothesis**: 每个数据集"风格强度"不同 (Office 弱, PACS 强), whitening 力度应自适应.

**Mechanism**:
- 每 client 算 `s = ||μ_sty_client - μ_sty_global|| / ||z_sem_mean||`
- 服务器 `s̄ = mean(s_k)`, 若 `s̄ > threshold` → 弱 whitening (λ=0.3), 否则 → 强 whitening (λ=1.0)

**Risk**: **MEDIUM-HIGH** - "一套方法应两数据集"是强 claim, reviewer 质疑 overfit to 2 datasets.

**Effort**: 1 周. **Novelty**: 6/10.

---

### I4. SCFlow-Lite: 可逆重构 Loss 替代 cos²

**Hypothesis**: 用 "(z_sem, z_sty) → z 重构" 可逆性定义解耦.

**Mechanism**: 加一个轻量 merge_head (MLP), loss `||merge_head(z_sem, z_sty) - z_original||²`.

**Risk**: **HIGH** - merge_head 可能 bypass z_sty. EXP-041 VAE 风格头已失败过.

**Novelty**: 5/10 (简化版已被做过).

---

### I5. Class-Conditional z_sty (CC-Style)

**Hypothesis**: z_sty 应该和 class label 独立 (同类跨域 z_sty 不同, 跨类同域 z_sty 相似).

**Mechanism**: loss 加 `-HSIC(z_sty, domain_id) + HSIC(z_sty, class_id)`.

**Risk**: **LOW**. **Effort**: 3-5 天. **Novelty**: 6/10 (增量).

---

### I6. Correlated Style Uncertainty Whitening (CSUW)

**Hypothesis**: PACS 风格有 correlation, pooled whitening 假设 independent 是错的.

**Mechanism**: 广播 `Σ_style_correlation` 代替 diagonal Σ_inv_sqrt.

**Risk**: **MEDIUM** - PACS 4 client 可能算不准 correlation.

**Novelty**: 5/10.

---

### I7. Frequency-Domain Style Separation — ❌ **Eliminate**

高频=风格, 低频=内容的假设对 PACS 错误 (sketch 风格在低频).

---

### I8. FedPall Amplifier 反转 — 合并到 I2 (本质等价)

---

### I9. Photometric SSL 定义风格

Self-supervised 对 sketch 不适用, 且 Risk MEDIUM, 新颖度中.

---

### I10. Prototype-Conditional Whitening — ❌ PACS 每类每 client 60 样本不够算 Σ

---

### I11. VGG 教师风格 — ❌ 违背 "AlexNet from scratch" 原则, 且同质 FISC/PARDON

---

### I12. MI Adversarial — HIGH risk, MI estimator 在小 batch FL 不稳

---

## First-Pass Filtering 结果

| # | Idea | Pass? | 备注 |
|---|------|-------|------|
| I1 | SASW | ✅ | FL selective whitening 空白 |
| **I2** | **CDANN-Dual** | ✅ **TOP** | 最高 novelty + 正面回答质疑 |
| I3 | SSD | ⚠️ | 可行但 "适配 2 数据集" claim 风险 |
| I4 | SCFlow-Lite | ⚠️ | 风险大 |
| I5 | CC-Style | ✅ | 轻量增强 |
| I6 | CSUW | ⚠️ | 数据不够 |
| I7 | FDSS | ❌ 假设错 |
| I8 | Amplifier | merge to I2 |
| I9 | Photometric | ⚠️ |
| I10 | Per-class wh | ❌ 数据 |
| I11 | VGG 教师 | ❌ 原则 |
| I12 | MI adv | ⚠️ 风险 |

**Top 3 推荐**: **I2 CDANN-Dual** (主推) + I1 SASW + I5 CC-Style.

---

## Top 3 Novelty Check

### I2 CDANN-Dual — Novelty 9/10

| Prior | 重合 | 差异 |
|-------|------|------|
| Federated Adversarial DA (2019) | ❌ 擦除所有 domain | 我们**保留** z_sty |
| ADCOL (ICML 2023) | ❌ 擦除 | 我们 constrained + 双头 |
| FedPall (ICCV 2025) | ⚠️ 最近, 混合空间对抗 | 我们解耦后 + 正反双向 |
| Deep Feature Disentanglement SCL (2025) | ⚠️ 思路类似 | FL 扩展 + constrained |
| FediOS / FedSeProto / FedDP | ❌ 擦除/正交 | 监督信号类型不同 |

**无直接 prior**. Constrained DANN 正反双头 + FL + 保留风格 = 首次.

### I1 SASW — Novelty 7.5/10

FL + selective whitening 是空白. RobustNet 2021 单机; Switchable Whitening 2019 无 FL.

### I5 CC-Style — Novelty 6/10

增量 novelty, 配合 I2 做 stack.

---

## Devil's Advocate Review

### I2 最强 objection

> "你的 double-directional DANN 本质是 FedPall Amplifier 的对称版, Deep Feature SCL 已证明有效, 你的贡献仅是 FL 扩展"

**反驳**:
1. 我们的 z_sem (反向) 和 z_sty (正向) 有**强耦合约束** (cos⊥ + HSIC), FedPall 没有解耦结构
2. FL 场景下每 client 只知自己 domain id (非 global), 需要特殊实现 (本地 dom_head 训 + 全局 GRL)
3. **量化**: 本地 vs 全局 domain classifier 的精度差距是新诊断维度 (无人报告过)

### I1 最强 objection

> "RobustNet 2021 已做, FL 实现是工程"

**反驳**:
1. FL 场景需服务器聚合 per-channel weights, 涉及通信开销和聚合策略
2. RobustNet 用 photometric pair 识别 style 通道, PACS 不适用 sketch; 我们用 class-level covariance
3. RobustNet 在 Cityscapes/GTA5 (分割), 没验证过 PACS 风格

---

## Final Ranking + Execution Order

| 排名 | Idea | Novelty | Feasibility | Risk | 策略 |
|------|------|---------|-------------|------|------|
| **1st** | **I2 CDANN-Dual** | 9/10 | 1 周 | LOW-MED | **主推** |
| 2nd | I1 SASW | 7.5/10 | 1-2 周 | MED | 备选 / 混合 |
| 3rd | I5 CC-Style | 6/10 | 3-5 天 | LOW | I2 增强 (stack) |

### Pilot 顺序

**Week 1**: I2 CDANN-Dual pilot
- Office R100 1 seed + PACS R100 1 seed (~4h 单卡)
- 判定:
  - ✅ 若 PACS 从 -1.49pp 回到 ≥ 0 → I2 成功, 扩 3 seeds
  - ❌ 若 Office 从 88.75 掉到 < 87 → 过度监督 fail, pivot

**Week 2 (如 I2 成功)**: I2 + I5 stack 扩 3 seeds full
**Week 2 (如 I2 失败)**: Pivot I1 SASW

---

## 最终建议

进入 **research-refine 流程**, 主 idea = **I2 CDANN-Dual**. 方案初稿:

> **FedDSA-CDANN** = FedDSA-SGPA + 双向 domain head:
> - 保留: 正交解耦 + Linear+whitening + FedBN
> - 新增:
>   - `dom_head_sem`: GRL 反向 → z_sem 不能识别 domain (标准 DANN)
>   - `dom_head_sty`: 正向 → z_sty 必须能识别 domain (新)
> - Domain label = client id (天然)
> - 两 heads 本地训, 不聚合 (隐私 + 防止过早收敛)

**正面回应**:
1. **"统计约束不够"**: 加了 class/domain label 监督信号
2. **"PACS 失败"**: dom_head_sty 强制 z_sty 保留 domain-discriminative 信息 (= PACS 风格类别判别信号)
3. **"novelty 在哪"**: Constrained DANN + 双头解耦 + FL 三位一体

**下一步**: 3 轮 reviewer 式精炼 → 写 NOTE + 知识笔记 (双版本) → 决定实现.
