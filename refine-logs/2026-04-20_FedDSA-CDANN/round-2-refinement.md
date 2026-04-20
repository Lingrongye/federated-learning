# Round 2 Refinement — FedDSA-CDANN v2

## Problem Anchor (verbatim from round 0)

- **Bottom-line problem**: 联邦学习中跨域客户端(Feature-skew FL)的语义-风格解耦对不同数据集性质的自适应处理. FedDSA-SGPA 的 Linear+whitening 在 Office +6.20pp, 在 PACS -1.49pp.
- **Must-solve bottleneck**: 现有统计解耦 (cos²=0 + HSIC) **没告诉模型什么是风格**, 导致类别判别信号被错归到 z_sty 并被 whitening 磨掉 (PACS z_sty_norm 塌 95%).
- **Non-goals**: 不追求万能机制; 不用大模型; 不做 label noise / 少样本.
- **Constraints**: seetacloud2 4090; Office/PACS client=domain bijection; AlexNet/DINOv2 portability.
- **Success condition**: PACS 3-seed AVG Best ≥ 82.2, Office ≥ 88.0, z_sty_norm R200 ≥ 1.5, 有 frozen probe 证据.

## Anchor Check

- **原 bottleneck**: 统计解耦缺方向性, whitening 误擦含类信号的风格.
- **V2 方法是否仍解决**: **仍解决**. 机制保持 (domain supervision + GRL on z_sem encoder path + shared dom_head). 本轮只做 precise framing 调整 + 诊断工具升级.
- **Reviewer suggestions rejected as drift**: 无. 本轮 3 个反馈都是 clarity/diagnostic 强化, 没动机制.

## Simplicity Check

- **V2 Dominant contribution**: `asymmetric encoder-gradient domain supervision (GRL on z_sem only, identity on z_sty) with a shared dom_head prevents whitening-induced style collapse in client=domain FedDG`. (比 V1 更精确: asymmetry 明确在 encoder 梯度而非 head objective.)
- **Components removed/merged in V2**:
  - **Clarification**: dom_head 本身 **non-adversarial**, 训练目标是 standard CE from both branches. 对抗发生在 encoder 梯度通过 GRL 反向传 z_sem 路径上. (Change #1)
  - **替换诊断**: "training dom_head accuracy" → "**frozen post-hoc linear probe** on final z_sem / z_sty" (Change #2)
  - **明确 C-port 辅助性**: 把 portability check 从 "optional supporting contribution" 降级为 "auxiliary sanity check", 明确不 claim 第二 contribution (Change #3)
- **Reviewer suggestions rejected as unnecessary complexity**: 无. 所有建议都是减法 / 澄清, 无加法.
- **为什么 V2 仍是最小充分**:
  - 新 trainable 仍是 1 个 MLP (dom_head)
  - frozen probe 只在训练完后跑, 不是新 trainable, 是 eval tool
  - C-port 只换 encoder, 不加新模块

## Changes Made

### 1. Precise mechanism framing: asymmetry 在 encoder gradient, 不在 dom_head objective (R2 CRITICAL)

- **Reviewer said**: "Right now the prose slightly overstates 'shared head with opposing gradients.' The head sees standard CE on both branches; only the `z_sem` encoder path is adversarial."
- **Action**: 重写 method 描述关键段落:

**V1 (imprecise)**:
> `dom_head` 被正反两路梯度对抗 → 最纯粹的"非对称监督"机制

**V2 (precise)**:
> `dom_head` 的训练目标是**普通 Cross-Entropy** 在两路 branch 上 (both `dom_head(GRL(z_sem))` 和 `dom_head(z_sty)` 都 minimize CE to domain labels). **Asymmetry 来自 encoder 端**: GRL 把 z_sem → dom_head 的梯度反向传到 encoder (encoder 被训练使 z_sem domain-indistinguishable); z_sty → dom_head 的梯度正向 (encoder 被训练使 z_sty domain-discriminative). 因此 dom_head 自身不做对抗游戏, 而是同时"两路都正向"作为 discriminator, 由 encoder 的两条路径**被动**接收非对称更新.

- **Reasoning**: 审稿人会误解为 "dom_head 被对抗", 实际 dom_head 永远 minimize CE. 精确表述避免 pseudo-novelty 指控.
- **Impact**: Method Specificity 进一步 +1, Contribution Quality 的 novelty description 更 defensible.

### 2. 加 frozen post-hoc linear probe 替代 training head accuracy (R2 IMPORTANT)

- **Reviewer said**: "A frozen post-hoc linear domain probe on final `z_sem` and `z_sty` would be a cleaner mechanism readout than reusing the same head."
- **Action**: 把 C-main 的 "domain confusion readout (from training dom_head)" **替换**为:

```python
# At the end of R200 training, freeze all params. For each client:
for cid in range(N_clients):
    z_sem_all, z_sty_all, d_all = collect_features_from_testset()
    probe_sem = LinearProbe(128, N_clients).fit(z_sem_all, d_all)  # 新训练 linear
    probe_sty = LinearProbe(128, N_clients).fit(z_sty_all, d_all)  # 同上
    report(probe_sem.test_acc, probe_sty.test_acc)
```

- **Expected evidence**:
  - `probe_sem accuracy ≈ 1/N_clients` (random-level, GRL 有效磨掉 z_sem 里的 domain)
  - `probe_sty accuracy ≈ 1.0` (positive supervision 成功)
  - Baseline (Linear+whitening, no CDANN): `probe_sem ≈ probe_sty` 且远高于 random (统计解耦没分开 domain)
- **Reasoning**: Training head 是 jointly optimized, 不是 clean representation diagnostic. Frozen probe 直接测 representation 的 domain-disentangle 程度.
- **Impact**: Validation Focus +1, Contribution Quality 的 evidence 强度 +1.

### 3. C-port 明确定位 auxiliary sanity check (R2 IMPORTANT)

- **Reviewer said**: "Make the portability check clearly auxiliary, not a second contribution. Keep it as one small sanity check that the mechanism is not AlexNet-specific."
- **Action**:
  - 从 "Optional supporting contribution" 删除 (V1 定义为此), 放进 "Appendix / Sanity Checks" 位置
  - Claim section 只留 **1 个 dominant contribution**, 无 supporting contribution
  - 实验 section 仍跑 C-port, 但在 paper 结构中只占 1 段 (supplementary appendix)
  - 表述 "mechanism not AlexNet-specific" 而非 "portable to modern backbones"
- **Reasoning**: 防止 reviewer 把 DINOv2 当作 "dual contribution" 要求更多实验. 只是 sanity check.
- **Impact**: Contribution Quality 更收敛 +0.5, Venue Readiness +0.5 (focus 更清).

### 4. Simplification: compute tight 时 drop AVG Last, 保留 AVG Best + mechanism metrics

- **Reviewer said**: "If compute gets tight, drop `AVG Last` from the headline table and keep `AVG Best` plus mechanism metrics."
- **Action**: paper headline table 主 metric 改为 AVG Best + domain probe acc (z_sem/z_sty) + z_sty_norm R200. AVG Last 进 appendix.
- **Reasoning**: Mechanism evidence > raw trajectory detail.
- **Impact**: Paper 表格更清.

---

## Revised Proposal (V2, full anchored)

# Research Proposal: FedDSA-CDANN v2 — Asymmetric Encoder-Gradient Domain Supervision

## Problem Anchor (same as R0/R1)

## Technical Gap

**当前方法失败点**:
- `L_orth = cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)` 只要求统计独立, 无方向
- Whitening `Σ_inv_sqrt` 广播无差别归一化, 跨 client z_sty 差异被磨平
- 诊断: PACS whitening 后 z_sty_norm R10→R200 塌 95% → 当风格携带类信号时信息丢失

**Smallest adequate intervention**: 给解耦加**方向性监督**. Domain label (= client id) 零成本可用. z_sem 应 domain-indistinguishable, z_sty 应 domain-discriminative. 一个共享 `dom_head` (non-adversarial, standard CE discriminator) + GRL **只作用在 z_sem encoder path** 实现非对称监督.

**Core technical claim**: **Asymmetric encoder-gradient domain supervision with a shared `dom_head`** prevents whitening-induced style collapse in `client=domain` FedDG where style carries class signal.

## Method Thesis

- **One-sentence**: GRL 只反转 z_sem → encoder 的梯度路径, 让同一个非对抗 dom_head 给 encoder 一条 domain-invariant (z_sem) 一条 domain-discriminative (z_sty) 的监督, 配合原 pooled whitening 广播, 既保留 Office 的 +6.20pp 收益又恢复 PACS 被磨掉的风格类信号.
- **Why smallest adequate**: 1 MLP (9K params) + 1 GRL (无参) + 3 行 loss; encoder/聚合/whitening 不动.
- **Why timely**: FL + asymmetric encoder-gradient domain supervision + 保留风格的三重交集在 2024-2026 30 篇综述无直接 prior.

## Contribution Focus

- **Dominant contribution (only)**: **Asymmetric encoder-gradient supervision via shared non-adversarial dom_head** 防止 whitening 擦掉含类信号的风格, 在 `client=domain where style carries class signal` 的 FedDG 场景有效. 由 frozen post-hoc linear probe 证实 representation-level disentanglement.
- **Appendix sanity check (不是 contribution)**: C-port 用 frozen DINOv2 替换 AlexNet encoder, mechanism 依然有效, 证明**不 AlexNet-specific**.
- **Explicit non-contributions**:
  - 不 claim 新聚合 / 新 whitening / 新分类器
  - 不 claim "dataset boundary diagnosis"
  - 不 claim "portable to all modern backbones" (只是 AlexNet-non-specific)
  - 不做 theoretical convergence

## Proposed Method

### Complexity Budget

- **Frozen / reused**: AlexNet encoder, 双头 sem_head/sty_head, sem_classifier, pooled whitening, L_orth + HSIC, FedBN
- **New trainable**: **1 个** `dom_head`: MLP 128→64→N_clients (~9K params)
- **GRL**: 无参
- **Post-hoc probes**: 训练完冻结模型后**新**训练 `LinearProbe(128, N_clients)` on z_sem 和 z_sty (但这是**evaluation-only**, 不算 new trainable)
- **Excluded tempting additions**: MI estimator, CC-Style HSIC, selective whitening, VGG/CLIP teacher

### System Overview

```
                                                     ┌──── sem_classifier(z_sem) ──→ L_CE(y)          [task, encoder 正向]
  x ─ encoder ─ feature ─┬─ sem_head → z_sem ─ (wh) ─┤
                         │                            └──── dom_head(GRL(z_sem, λ)) ──→ L_CE(d)        [dom_head 正向 CE, encoder 被反向 via GRL]
                         └─ sty_head → z_sty ─ (wh) ──────── dom_head(z_sty) ──→ L_CE(d)               [dom_head 正向 CE, encoder 正向 (no GRL)]

  z_sem ⊥ z_sty:  λ_orth · cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)

  dom_head 自己非对抗 (两路都 minimize CE). Asymmetry 只在 encoder 的梯度方向.
  Inference: 只走 z_sem → sem_classifier 预测 y. z_sty 用于下游 pooled style bank 广播.
```

### Core Mechanism (精确表述, R2 修正)

- **dom_head 的目标**: standard Cross-Entropy discriminator, 两路 branch 都最小化
  ```
  L_dom_head = CE(d, dom_head(GRL(z_sem, λ))) + CE(d, dom_head(z_sty))
  ```
  dom_head 被这个 L 训练, 对 z_sem 和 z_sty 都 "想分对 domain". 但因为 GRL 层 forward 恒等, dom_head 看到的 feature 和 z_sem / z_sty 相同.
- **Encoder 端接收的梯度**:
  - z_sem → dom_head 这条路径: 反向梯度 (GRL × -λ), encoder 被推向产 domain-indistinguishable z_sem
  - z_sty → dom_head 这条路径: 正向梯度 (无 GRL), encoder 被推向产 domain-discriminative z_sty
  - z_sem → sem_classifier: 正向, encoder 产 class-discriminative z_sem (原 L_task)
- **关键**: 同一个 dom_head 参数同时服务两路 discriminate 任务. Asymmetry **不在 head objective**, 在 **encoder 的两条 upstream path 梯度方向**.

### Architecture & Loss

```python
dom_head = Linear(128, 64) → ReLU → Dropout(0.1) → Linear(64, N_clients)
GRL(x, λ): forward(x) = x, backward(grad) = -λ · grad

L_task    = CE(y, sem_classifier(z_sem))
L_dec     = λ_orth · cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)
L_dom_sem = CE(d, dom_head(GRL(z_sem, λ_adv)))
L_dom_sty = CE(d, dom_head(z_sty))
L_total   = L_task + L_dec + L_dom_sem + L_dom_sty
```

- **λ_adv schedule**: `λ_adv(r) = min(1.0, max(0, (r - 20) / 20))` (R=0..20 off, R=20..40 linear, R≥40 full)
- **Features entering dom_head**: post-whitening z_sem / z_sty (同 sem_classifier 的 feature space)

### Aggregation Strategy

- `dom_head` 通过 **FedAvg 聚合**. 本地 data 单域会让本地 head degenerate, 聚合后的 head 能区分 N_clients 个 domain, GRL 才有信号.
- 隐私: head 参数聚合, 不是 data; client id 本就公开.

### Integration

- 代码位置: `FDSE_CVPR25/algorithm/feddsa_sgpa.py`
- 新增: `GradientReverseLayer` (~15 行) + `dom_head` MLP (~10 行) + L_dom_sem/sty 计算 (~20 行)
- 聚合: `dom_head` 参与 FedAvg
- 开关: algo_para 加 `ca` (cdann active, 0/1 with default 0)

### Training Plan

- **Joint training**, λ_adv 三段 schedule
- 同 EXP-102 config: LR=0.05, E=1 (Office) / E=5 (PACS), R=200
- Loss weights: λ_orth=1.0, λ_hsic=0.1

### Failure Modes

- **FM1: 对抗训练发散** — 检测 L_task 不降; 缓解 λ_adv schedule + grad clip=10
- **FM2: dom_head 欠拟合** — 检测 head acc 过低; 缓解 Dropout=0.1 已加
- **FM3: z_sty 过度携带信息** (λ_adv 太大) — 检测 frozen probe_sty > 80% 但 sem_classifier 掉 → 降 λ_adv 到 0.5 (**不加 HSIC(z_sty,y) 约束**, 冲突 anchor)
- **FM4: Office 不需 CDANN** — scope 已明确限定 PACS-like settings, Office parity 即可

### Novelty and Elegance Argument

- **Closest**: Deep Feature Disentanglement SCL (Cog. Comp. 2025), **non-FL + symmetric both-positive supervision + separate heads**. 我们: **FL + asymmetric encoder-gradient + shared non-adversarial head**.
- **Vs FedPall**: 混合空间擦除 vs 我们解耦后保留
- **Vs ADCOL / Federated Adversarial DA**: 擦除派 vs 我们保留
- **Pseudo-novelty defense**: "共享 head + encoder 梯度非对称" 的 precise framing 避免被读成 "两个独立 DANN head". 关键 delta 是 **head 本身 non-adversarial**, 对抗游戏只发生在 encoder 通过 GRL 被反向推.

---

## Claim-Driven Validation Sketch

### Claim C-main (Primary): CDANN 修 PACS 不伤 Office

- **Minimal experiment**: PACS R200 3 seeds + Office R200 3 seeds, vs Linear+whitening baseline
- **Headline metrics**:
  - AVG Best
  - **frozen post-hoc probe accuracy** on z_sem (应 ≈ 1/N_clients, random) 和 z_sty (应 ≈ 1.0)
  - z_sty_norm R200
- **Expected evidence**:
  - PACS AVG Best 82-84 (from -1.49pp to ≥0 vs Plan A 81.69)
  - Office AVG Best ≥ 88.0 (maintain baseline)
  - probe_sem ≈ 0.25 (random, N=4); probe_sty ≈ 0.95
  - z_sty_norm R200 ≥ 1.5 (from baseline 0.15)

### Claim C-ablate (必做): z_sty 正向监督的必要性

- **Minimal experiment**: PACS R200 2 seeds × 3 variants:
  - (V1) baseline: Linear+whitening (no CDANN)
  - (V2) z_sem-only: 只 L_dom_sem (标准 DANN)
  - (V3) full CDANN (ours)
  - [V3 - V2] vs [V2 - V1] 对比 z_sty 正向监督的增量
- **Expected**: V3 > V2 ≥ V1, 关键证据是 "只反向不够, 正向 z_sty 才锁住风格"

### Appendix sanity check C-port (auxiliary, **not a contribution**)

- **Minimal experiment**: PACS R100 1 seed, encoder 换 frozen DINOv2-S/14 (其他不变)
- **Goal**: 证明 mechanism not AlexNet-specific
- **Expected**: AVG Best ≥ AlexNet CDANN (希望略好), 主要是验证 mechanism 通用

## Experiment Handoff Inputs

- **Must-prove**: C-main (12h), C-ablate (24h)
- **Appendix sanity**: C-port (2h)
- **Critical metrics**: AVG Best, frozen probe acc (sem/sty), z_sty_norm R200
- **Highest-risk assumptions**:
  - A1: client=domain bijection (成立)
  - A2: 同一个 dom_head 同时服务两路不互相抵消 (pilot 验证)
  - A3: λ_adv schedule 稳 (standard DANN, 已有 mitigation)

## Compute & Timeline

- **Pilot**: Office + PACS R100 × 1 seed = 2.5h
- **Full C-main**: 12h
- **C-ablate**: 24h
- **C-port**: 2h
- **Probes**: 30min
- **Total**: ~41 GPU·h
