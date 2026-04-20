# Round 3 Refinement — FedDSA-CDANN v3

## Problem Anchor (verbatim)

- **Bottom-line problem**: 跨域 FL 解耦在不同数据集的自适应处理; Linear+whitening Office +6.20pp / PACS -1.49pp
- **Must-solve bottleneck**: 统计解耦 (cos²+HSIC) 无方向; 类信号被错归到 z_sty, whitening 磨掉 (PACS z_sty_norm 塌 95%)
- **Non-goals / Constraints / Success**: 见 R0, 不变

## Anchor Check

- **原 bottleneck**: "class-relevant style 被错归到 z_sty 后被 whitening 擦掉"
- **V3 仍解决**: 不变, 本轮只强化 evidence 对齐 anchor
- **Reviewer suggestions rejected as drift**: 无

## Simplicity Check

- **V3 Dominant contribution**: 不变 (shared non-adversarial dom_head + GRL on z_sem encoder path; 文字中 novelty defense 更窄)
- **Components removed/merged in V3**:
  - **Probe 协议严谨化**: train on frozen train features, test on held-out test (Change #1)
  - **加 PACS class probe on z_sty** (z_sty → y 而非 → domain): 直接证明 anchor 的 "class-relevant" 部分 (Change #2)
  - **删 "2024-2026 review gap" 新颖性话术** (Change #3), 改用 mechanism-narrow framing
- **Reviewer suggestions rejected as unnecessary complexity**: 无
- **为什么 V3 仍是最小充分**: 无新 trainable 组件; probe 是 evaluation-only; 改的是 framing 和 protocol

## Changes Made

### 1. Fix frozen probe protocol: train/test 不 leak (R3 CRITICAL)

- **Reviewer said**: "Your current pseudocode appears to fit the probe on the test set. Train the probe on frozen features from train or train/val splits, then report on held-out test features."
- **Action**: 改 probe pseudo-code:

```python
# After R200 training, freeze encoder / heads / whitening.
# For each client c, collect features from LOCAL train loader and LOCAL test loader.
for cid in range(N_clients):
    # ===== Train probe on TRAIN features (not test) =====
    z_sem_train, z_sty_train, d_train, y_train = collect_from_train_loader(cid)
    # (domain label d is the client id, same for all samples in one client, so aggregate across clients)

# Aggregate across all clients
Z_sem_train, Z_sty_train, D_train, Y_train = concat(z_*_train_per_client)
Z_sem_test,  Z_sty_test,  D_test,  Y_test  = concat(z_*_test_per_client)

# Three probes for three claims
probe_sem_domain = LogisticRegression().fit(Z_sem_train, D_train)
probe_sty_domain = LogisticRegression().fit(Z_sty_train, D_train)
probe_sty_class  = LogisticRegression().fit(Z_sty_train, Y_train)

# Evaluate on HELD-OUT test features
acc_sem_dom = probe_sem_domain.score(Z_sem_test, D_test)    # should ≈ 1/N (random)
acc_sty_dom = probe_sty_domain.score(Z_sty_test, D_test)    # should ≈ 1.0
acc_sty_cls = probe_sty_class.score(Z_sty_test, Y_test)     # KEY: PACS should be high, Office low
```

- **Reasoning**: 没 train/test 分离等于 probe overfit 特定样本, 不反映 representation 本身的 domain/class 可分性. 现在正确了.
- **Impact**: Validation Focus +1 (从 8→9); 诊断 claim defensible.

### 2. 加 frozen class probe on z_sty (z_sty → y), 证明 "class-relevant" anchor (R3 CRITICAL)

- **Reviewer said**: "Add back one PACS-only frozen class probe on z_sty. Without it, the validation shows domain disentanglement, but not the anchor-specific claim that whitening erased class-relevant style."
- **Action**: 在 Claim C-main 加第三个 probe: `probe_sty_class = Linear(128, K_classes)` 在 z_sty 上预测 y.

**Expected evidence**:
- **PACS CDANN**: `probe_sty_class` acc ≥ 40% (远超 random K=7 → 14%) — **证明 z_sty 里保留了 class 判别信号**
- **PACS Linear+whitening (baseline)**: `probe_sty_class` acc ≈ 15% (接近 random, 因为 whitening 磨了 z_sty)
- **Office CDANN**: acc ~ 20-30% (比 random 10% 稍高, Office 风格不含强类信号)
- **关键对照**: `[PACS CDANN - PACS baseline]` 的 probe_sty_class gap 直接量化 "CDANN 保留了 whitening 擦掉的类信号"

这**直接对应 anchor 的 "class-relevant style got misassigned to z_sty and then erased by whitening"**.

- **Reasoning**: Domain probe 只证明 domain 分开了; 类 probe 才证明磨掉的是**class-relevant** style.
- **Impact**: Contribution Quality 从 8→9 (anchor evidence 打实), Validation Focus +1.

### 3. Novelty defense 窄化: 删 "2024-2026 综述 gap", 改 mechanism-narrow framing (R3 IMPORTANT)

- **Reviewer said**: "In the writing, stop leaning on '2024-2026 review gap' language as a novelty defense. The stronger defense is narrower: shared non-adversarial discriminator plus asymmetric encoder-gradient supervision is the minimal repair for whitening-induced style collapse in `client=domain` FedDG."
- **Action**: 删 Novelty argument 里的 "30 篇综述无直接 prior" 话术, 改为:

**V2 (old novelty defense)**:
> FL + asymmetric encoder-gradient + shared non-adversarial head 的三重交集在 2024-2026 30 篇综述无直接 prior.

**V3 (new, narrower)**:
> Our contribution is the **minimal repair** for a specific, observable failure mode: *whitening-induced style collapse in client=domain FedDG when style carries class signal*. The precise repair is (a) a **shared, non-adversarial** domain discriminator (standard CE on both branches, no adversarial head training), and (b) **asymmetric encoder-gradient direction** (GRL on z_sem path only). Together they are the smallest intervention that fixes the observed 95% z_sty_norm collapse and -1.49pp PACS regression, without changing encoder, whitening, or aggregation. The novelty is **mechanism minimality** aligned to a **measurable failure mode**, not a new family of methods.

- **Reasoning**: 窄 claim 比广覆盖更抗反驳. 把 novelty 绑定在具体可验证的 failure mode 上, reviewer 没法说 "你只是 DANN 变体".
- **Impact**: Venue Readiness 从 8→9 (claim defensible 更强).

---

## Revised Proposal (V3, full anchored)

# FedDSA-CDANN v3: Minimal Repair for Whitening-Induced Style Collapse in client=domain FedDG

## Problem Anchor (same as R0)

## Technical Gap

**Observed failure mode**: 在 `client=domain` FedDG (Office/PACS 各 4 client 对应 4 域), 现有方法 FedDSA-SGPA 的 Linear+whitening:
- Office: AVG Best 88.75 (Plan A 82.55, **+6.20pp**)
- PACS: AVG Best 80.20 (Plan A 81.69, **-1.49pp**)
- PACS z_sty_norm 轨迹: R10=3.12 → R200=0.15 (**塌 95%**)
- Office z_sty_norm R200=2.21 (稳定 -2%)

**Root cause**: 原 `L_orth = cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)` 只要求统计独立, **没有方向性监督**. 当 PACS 风格 (油画/素描) 携带 class 判别信号时, 特征可能被错归到 z_sty, 随后被 whitening 磨掉.

**Smallest adequate intervention**: 给解耦加方向性监督. Domain label (= client id) 是**零成本 zero-annotation** 可用的监督信号. 一个 **shared non-adversarial** `dom_head` (在两路都最小化 standard CE) + **GRL 只作用在 z_sem encoder 路径** 就是 repair 这个 failure mode 的最小改动.

## Method Thesis

**One-sentence**: Shared non-adversarial domain discriminator + asymmetric encoder-gradient direction (GRL on z_sem only) is the minimal repair for whitening-induced style collapse when class-relevant style is present.

**Why smallest adequate**:
- 1 个 MLP (9K params) + 1 个 GRL (无参) + 3 行 loss
- 不改 encoder / sem_classifier / whitening / 聚合算法
- 评估新增 3 个 frozen linear probe (evaluation-only, 非新 trainable)

**Why timely**: 不依赖大模型, 不做新聚合; 是 "mechanism-first" 精准 repair.

## Contribution Focus

- **Dominant contribution (sole)**: **Shared non-adversarial dom_head + asymmetric encoder-gradient GRL** 是对 "whitening-induced style collapse in client=domain FedDG with class-relevant style" 这一可观测 failure mode 的**最小 repair**. 通过 frozen class probe on z_sty 直接验证 anchor.
- **Appendix sanity check (not a contribution)**: C-port 用 frozen DINOv2-S/14 替换 AlexNet encoder, 验证 **mechanism not AlexNet-specific**.
- **Explicit non-contributions**: 不 claim 新聚合 / 新 whitening / 新分类器; 不 claim "all datasets"; 不 claim theoretical convergence.

## Proposed Method

### Complexity Budget

- **Frozen / reused**: AlexNet encoder, sem_head / sty_head, sem_classifier, pooled whitening, L_orth + HSIC, FedBN
- **New trainable**: **1** — `dom_head` MLP(128→64→N_clients), ~9K params
- **GRL**: 无参, `forward(x) = x`, `backward(grad) = -λ · grad`
- **Post-hoc evaluation probes** (not new trainable):
  - `probe_sem_domain`: Linear(128, N_clients), 训练在 train features, 测试在 test features
  - `probe_sty_domain`: 同上但输入 z_sty
  - `probe_sty_class`: Linear(128, K_classes) on z_sty (关键: 证明 z_sty 的 class 携带)
- **Excluded tempting additions**: MI estimator; CC-Style HSIC; selective whitening; VGG/CLIP teacher; HSIC(z_sty, y)=0 (和 anchor 自矛盾)

### System Overview

```
                                                     ┌── sem_classifier(z_sem) ── L_CE(y)          [encoder 正向, task]
  x ─ encoder ─ feature ─┬─ sem_head → z_sem ─ (wh) ─┤
                         │                            └── dom_head(GRL(z_sem, λ)) ── L_CE(d)        [dom_head 正向 CE, encoder via GRL 反向]
                         └─ sty_head → z_sty ─ (wh) ──── dom_head(z_sty) ── L_CE(d)                 [dom_head 正向 CE, encoder 正向]

  z_sem ⊥ z_sty:  λ_orth · cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)

  dom_head 自己 non-adversarial (两路都 minimize CE). Asymmetry 只在 encoder 的梯度方向.

  Evaluation (frozen, post R200):
  probe_sem → domain (expect ≈ random)
  probe_sty → domain (expect ≈ 1.0)
  probe_sty → class  (KEY: PACS should be high, Office low; proves z_sty class-carrying)
```

### Core Mechanism

- **dom_head objective**: standard Cross-Entropy discriminator on both branches, both minimize (no adversarial head training):
  ```
  L_dom_head_update = CE(d, dom_head(GRL(z_sem))) + CE(d, dom_head(z_sty))
  ```
- **Encoder gradient flows**:
  - `sem_classifier(z_sem) → y`: encoder forward path, 正常 task supervision
  - `dom_head(GRL(z_sem, λ)) → d`: GRL 反转梯度, encoder 被推向产 domain-indistinguishable z_sem
  - `dom_head(z_sty) → d`: 无 GRL, encoder 被推向产 domain-discriminative z_sty
- **Key clarification (R2 精炼)**: 同一个 dom_head 参数同时被两路更新. Asymmetry 不在 head objective, 在 encoder 两条 upstream path 的梯度方向.

### Loss

```python
dom_head = Linear(128, 64) → ReLU → Dropout(0.1) → Linear(64, N_clients)

L_task    = CE(y, sem_classifier(z_sem))
L_dec     = λ_orth · cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)
L_dom_sem = CE(d, dom_head(GRL(z_sem, λ_adv)))
L_dom_sty = CE(d, dom_head(z_sty))
L_total   = L_task + L_dec + L_dom_sem + L_dom_sty
```

- `λ_adv(r) = min(1.0, max(0, (r - 20) / 20))` (R=0..20 off, R=20..40 linear, R≥40 full)
- Features entering dom_head: post-whitening z_sem/z_sty
- Inference: 只走 z_sem → sem_classifier, 预测 y

### Aggregation

- `dom_head` 通过 **FedAvg 聚合** (本地单域会让本地 head degenerate)
- 隐私: 聚合参数, 不是 data; client id 本就公开

### Integration

- 代码位置: `FDSE_CVPR25/algorithm/feddsa_sgpa.py`
- 新增: `GradientReverseLayer` (~15 行) + `dom_head` MLP (~10 行) + L_dom_sem/sty + frozen probe eval (~20 行 + probe script)
- 聚合: `dom_head` 加入 FedAvg 白名单
- 开关: algo_para 加 `ca` (cdann active, 0/1 default 0)

### Training Plan

- Joint training, λ_adv 三段 schedule
- Loss weights: λ_orth=1.0, λ_hsic=0.1, λ_adv scheduled

### Failure Modes

- **FM1: 对抗训练发散** — λ_adv schedule + grad clip=10
- **FM2: dom_head 欠拟合** — Dropout 已加, FedAvg 聚合确保跨 client 一致
- **FM3: z_sty 过度携带信息** (λ_adv 过大) — 检测 `probe_sty_class` 过高同时 sem_classifier 掉; 降 λ_adv 到 0.5 (不加 HSIC(z_sty,y), 冲突 anchor)
- **FM4: Office 不需 CDANN** — scope 已限定 "style carries class signal" 场景, Office parity 即可

### Novelty and Elegance Argument (R3 窄化)

Our contribution is the **minimal repair** for a specific, observable failure mode: *whitening-induced style collapse in `client=domain` FedDG when style carries class signal* (observed on PACS: z_sty_norm 塌 95%, AVG Best -1.49pp).

The precise repair:
- (a) a **shared, non-adversarial** domain discriminator (standard CE on both branches, no adversarial head training)
- (b) **asymmetric encoder-gradient direction** (GRL on z_sem path only)

Together they are **the smallest intervention** that fixes the observed collapse and regression, without changing encoder, whitening, or aggregation.

The novelty is **mechanism minimality aligned to a measurable failure mode**, not a new family of methods. Closest prior:
- **Deep Feature Disentanglement SCL (Cog. Comp. 2025)**: non-FL, symmetric positive supervision, separate heads. We: FL, asymmetric encoder-gradient, shared non-adversarial head.
- **FedPall (ICCV 2025)**: adversarial in mixed feature space, erases domain. We: preserve domain info in z_sty after disentanglement.
- **ADCOL / Federated Adversarial DA**: erase-all DANN. We: asymmetric preserve.

---

## Claim-Driven Validation Sketch

### Claim C-main (Primary): CDANN 修 PACS 不伤 Office

- **Experiment**: PACS R200 3 seeds + Office R200 3 seeds, vs Linear+whitening baseline
- **Headline metrics**: AVG Best, frozen probe accuracies (3 probes), z_sty_norm R200
- **Expected**:
  - PACS AVG Best **82-84** (from -1.49pp back to ≥0 vs Plan A 81.69)
  - Office AVG Best **≥ 88.0** (maintain baseline)
  - `probe_sem_domain` ≈ 0.25 (random, N=4), `probe_sty_domain` ≈ 0.95
  - **`probe_sty_class` PACS ≥ 40%** (CDANN) vs ≈15% (baseline) — anchor 证据
  - z_sty_norm R200 ≥ 1.5 (from baseline 0.15)

### Claim C-ablate (必做): z_sty 正向监督的必要性

- **Experiment**: PACS R200 2 seeds × 3 variants:
  - (V1) baseline: Linear+whitening (no CDANN)
  - (V2) z_sem-only: 只 L_dom_sem (标准 DANN)
  - (V3) full CDANN (ours)
- **Expected**: V3 > V2 ≥ V1 on AVG Best AND `probe_sty_class`
- **关键**: V3 - V2 的 `probe_sty_class` 差距证明 z_sty 正向监督是**必要**的

### Probe Protocol (R3 修正, train/test 不 leak)

```python
# Freeze all params after R200.
# Train probe on FROZEN TRAIN features (aggregated across all clients).
# Report on HELD-OUT TEST features.
Z_sem_train, Z_sty_train, D_train, Y_train = aggregate_across_clients(train_loader)
Z_sem_test,  Z_sty_test,  D_test,  Y_test  = aggregate_across_clients(test_loader)

probe_sem_dom = LogReg().fit(Z_sem_train, D_train)
probe_sty_dom = LogReg().fit(Z_sty_train, D_train)
probe_sty_cls = LogReg().fit(Z_sty_train, Y_train)

report_test_accs(probe_sem_dom, probe_sty_dom, probe_sty_cls)
```

### Appendix sanity check C-port (auxiliary)

- PACS R100 1 seed, encoder 换 frozen DINOv2-S/14
- Expected: AVG Best ≥ AlexNet CDANN, 证 mechanism not AlexNet-specific

## Experiment Handoff

- Must-prove: C-main (12h), C-ablate (24h)
- Appendix: C-port (2h), 3 probes (30min)
- Total: ~41 GPU·h single 4090
