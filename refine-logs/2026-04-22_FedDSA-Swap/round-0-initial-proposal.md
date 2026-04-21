# FedDSA-Swap: Feature-Level Swap Consistency for Federated Domain Generalization

**Round**: 0 (初始提案)
**时间**: 2026-04-22
**作者**: Claude + 用户
**背景**: 原 FedDSA (Decouple-Share-Align) 经过 EXP-108/109/110/111 反事实 probe 验证,发现 L_orth + L_aug 组合在 linear probe 下解耦成功 (probe_sty_class=0.24) 但 MLP-256 probe 下仍 0.71 (非线性泄漏)。Phase 0-3 调研 28 篇论文 (5 份深度精读)后,推出 Swap 方案作为方法 novelty。

---

## Problem

FL 双头解耦 (z_sem + z_sty) 在 shared-trunk 架构下面临**非线性 class 信号从 z_sem 主干溢出到 z_sty 分支**的根本问题:

- EXP-111 最优配置 (lo=3): probe_sty_class linear=0.34, MLP-64=0.20, **MLP-256=0.71**
- Moyer NeurIPS 2018 早已证明: 损失级 adversary / 正交约束无法消除非线性泄漏
- FL 领域 8 篇同类方法 (FedPall, FedSTAR, FedSeProto, FedDP, FISC, FedPLVM, FediOS, D²IFLN) 无一做严格 probe 验证 — 全靠下游 accuracy 间接代理

## One-sentence Novelty

**我们提出 FedDSA-Swap: 首个在 feature 空间通过 style swap consistency 实现的 FL 双头解耦方法, 无需 decoder, 基于 MUNIT 2018 swap idea 但改造为 MLP-learned style modulation, 配合 Moyer 2018 0/1/2/3 层递增 adversary sweep 作为 evaluation, 实证暴露 FL 双头方法共享弱点的同时提供首个 probe-validated 解耦**。

## Method (架构 + Loss)

### 架构 (在现有 FedDSA 基础上 incremental)
```
AlexNet encoder → h ∈ R^{1024}
                 ├── semantic_head → z_sem ∈ R^{128}
                 └── style_head   → z_sty ∈ R^{128}  (保留私有,不参与全局聚合)

新增: style_modulator = MLP(z_sty → R^{256}) → split into (γ, β) ∈ R^{128}, R^{128}
                         (用 style vector 通过小 MLP 生成 channel-wise 调制)
```

### Loss 组合
```
L_total = L_CE(sem_classifier(z_sem_A), y_A)                # 1. 原主任务
        + L_CE(sem_classifier(z_sem_A_styled), y_A)         # 2. 原 aug 分类 (保留)
        + 1.0 · L_swap                                       # 3. ★ 新 MUNIT-style swap
        + 0.3 · L_cycle_sem                                  # 4. ★ 新 cycle consistency
        + 0.1 · L_orth(z_sem, z_sty)                         # 5. 软正交 (原有,权重降低)
        + 0.1 · L_HSIC(z_sem, z_sty)                         # 6. 原有
```

其中关键新 loss:

**L_swap** (feature swap consistency):
```python
perm = torch.randperm(B)                           # batch 内随机配对
z_sty_other = z_sty[perm]                          # 借用另一样本的 z_sty
(γ, β) = style_modulator(z_sty_other)              # MLP 生成调制参数
z_sem_swap = γ · z_sem + β                         # channel-wise FiLM 调制
ŷ_swap = sem_classifier(z_sem_swap)
L_swap = CE(ŷ_swap, y)                             # 必须仍预测自己的 y (非 perm 后的)
```

**L_cycle_sem** (MUNIT cycle consistency):
```python
(γ, β) = style_modulator(z_sty_self)               # 用自己的 z_sty 调制
z_sem_cycle = γ · z_sem_swap + β                   # 二次调制回
L_cycle_sem = ||z_sem_cycle - z_sem||_2
```

### Style 来源 (风格仓库保留)
保留原 FedDSA 的跨 client 风格仓库 (FedDSA 原叙事):
- 每 client 维护本地 z_sty 统计量 (μ_k, σ_k)
- Server 合成全局 style pool
- L_aug 使用 bank style (原逻辑)
- **L_swap 使用 batch 内 z_sty (新)**
- 两者互补: bank 给宏观 domain-level style,batch 给 instance-level style

## Evaluation (双 contribution)

### 方法指标 (主结果表)
- **PACS AVG Best** 3-seed (对照 orth_only 80.64, CDANN 80.08, FediOS 推算 ~80)
- **Office AVG Best** 3-seed

### ★ Moyer probe sweep (FL 领域首次)
- **0 层 (logistic)** / **1 层 MLP hidden=64** / **2 层 MLP hidden=64,64** / **3 层 hidden=64,64,64**
- 全部用 absolute error loss + BN + Adam lr=0.001 (Moyer 2018 原参数)
- 每层 probe 在 z_sty → class, z_sem → domain 两个方向测
- 目标: **3 层 MLP probe(z_sty → class) ≤ random + 10%**

### 辅助诊断 (Phase 0 推荐的 Tier 2)
- Hewitt-Liang selectivity (control task 防 probe 过拟合)
- DCI-RF disentanglement score
- MIG 用 (domain, class) 作双 GT factor

### Baseline 对照
- orth_only (EXP-109): linear=0.24, MLP-256=0.81
- CDANN (EXP-108): linear=0.96, MLP-256=0.96
- FediOS (复现): linear=0.96, MLP-256=0.88 (预测)

## 预期结果

| Metric | orth_only | CDANN | FediOS (pred) | **FedDSA-Swap (pred)** |
|--------|:---------:|:-----:|:-------------:|:----------------------:|
| Linear probe (sty→class) | 0.24 | 0.96 | 0.80 | **0.10-0.15** |
| MLP-64 probe | 0.69 | 0.96 | 0.82 | **0.25-0.35** |
| MLP-256 probe | **0.71** | 0.96 | 0.88 | **0.40-0.50** |
| 3-layer MLP probe | ~0.80 (est) | ~0.90 (est) | ~0.90 (est) | **0.45-0.55** |
| PACS AVG Best 3-seed | 80.64 | 80.08 | ~80 | **81.0-82.0** |

## Contributions

1. **方法**: 首个 FL + feature-level swap consistency + MLP-learned style modulation 组合
2. **理论**: 基于 MUNIT 2018 content-style cycle + von Kügelgen NeurIPS 2021 contrastive identifiability
3. **评估**: FL 领域首个 Moyer 0/1/2/3 层 adversary sweep 评估协议
4. **诊断贡献**: 用严格评估暴露 FediOS / FedPall / CDANN 等 FL 双头方法的共享弱点

## Difference from 原 FedDSA (重要区分)

| 维度 | 原 FedDSA | FedDSA-Swap |
|------|----------|-------------|
| L_aug (style injection) | bank 静态 μ/σ AdaIN | 保留 + 新 MLP-learned γ/β |
| Swap 机制 | ❌ 无 | ★ L_swap + L_cycle |
| Evaluation | linear probe only | ★ Moyer 0/1/2/3 层 sweep |
| 理论背书 | 无 | MUNIT + von Kügelgen |

**本质: 原 FedDSA 的 "L_aug 保留" + "新加 L_swap/L_cycle" + "全新 Moyer evaluation"**

## Implementation Plan

| Week | 内容 |
|------|------|
| W1 | 实现 style_modulator MLP + L_swap + L_cycle, smoke test PACS s=2 |
| W2 | 3-seed PACS 完整跑 + Moyer 0/1/2/3 层 probe sweep |
| W3 | Office 3-seed + ablation (去 L_swap / 去 L_cycle / 各一个 run) |
| W4 | 论文 draft (method + evaluation 章节) |

## Risk

1. **L_swap 与 L_orth 可能冲突**: swap 让 z_sem 学到独立于 z_sty 的表示,orth 让几何正交,可能梯度打架 → 应对: orth 权重降到 0.1
2. **batch=50 swap 噪声大**: 4 client 每个 12-13 样本,swap 空间小 → memory bank 技巧
3. **Style modulator MLP 过拟合 class shortcut**: MLP 可能学到"从 z_sty 读 class"的捷径 → L_swap label 必须是原 y_A (不是 perm 后 y_B)
4. **FL 通信成本**: style_modulator 参数参加 FedAvg 聚合 (~小 MLP 10k 参数)

## 7-dim Self-Assessment

| 维度 | 自评 | 理由 |
|------|:---:|-----|
| Problem Fidelity | 8.5/10 | PACS 非线性泄漏有实证,FL disentanglement 验证空白 |
| Method Specificity | 7.5/10 | Loss 组合明确, MLP modulator 结构待实验确定 |
| Contribution Quality | 6.5/10 | 方法是 MUNIT 在 FL 的迁移, novelty 中等偏上 |
| Frontier Leverage | 7.5/10 | 用了 MUNIT'18 + von Kügelgen'21 + Moyer'18 三条 |
| Feasibility | 7.5/10 | 工程增量小 (1 MLP + 2 loss),数据已就绪 |
| Validation Focus | 8.5/10 | Moyer sweep 是真正的严格评估 |
| Venue Readiness | 6.5/10 | 论文 novelty 中等, 需要 Codex 评审判断是否够 |

**总评** (weighted): ~7.3/10 — 可进 refine 打磨到 8+ 或 pivot

---

## 待 Codex Round 1 回答的 4 个核心问题

1. **L_swap 是否真的与原 L_aug 本质不同?** 或者只是换 style 来源的 incremental 改动?
2. **MLP-learned γ/β (FiLM) vs 静态 AdaIN 的优势真的存在吗?** 有直接证据吗?
3. **MUNIT cycle consistency 在 discriminative FL 场景迁移合理性**? 无 decoder 情况下 cycle 真的是 cycle 吗?
4. **Moyer sweep 作为 evaluation contribution 是否够 paper 主卖点?** 还是只能作辅助?
