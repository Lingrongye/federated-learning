# Round 2 Refinement — FedDSA-BiProto v3

**Date**: 2026-04-24
**Based on**: Round 2 review (7.8/10 REVISE)

## Problem Anchor (逐字复刻自 round-0, 不变)

- Bottom-line: 在 cross-domain 联邦学习 (PACS / Office-Caltech10) 下, 用 AlexNet from scratch, 3-seed {2,15,333} × R200 mean AVG Best 必须同时严格超过 FDSE 本地复现 baseline (PACS > 79.91 / Office > 90.58). 当前 orth_only: PACS 80.64 ✅ (+0.73), Office 89.09 ❌ (−1.49).
- Must-solve bottleneck: Office 补回 −1.49pp 且至少 +0.5pp, PACS ≥ 80.91.
- Non-goals: 不换数据集; 不做诊断论文; 不堆模块; 不预训练; 不换骨干.
- Constraints: 4090/3090; Pilot 预算 ≤ 50 GPU-h hard; AlexNet + FedBN; PACS E=5 / Office E=1; 1 周.
- Success: 3-seed mean AVG Best PACS ≥ 80.91 且 Office ≥ 91.08; AVG Last 不退; 3 套可视化 evidence.

## Anchor Check
- Bottleneck 不变, 修订后方法依然直接指向 Office 恢复 + PACS 不退
- Reviewer 没要求 drift, 全部建议都是紧 mechanism

## Simplicity Check

- **修订后 dominant contribution (单一 headline)**:  
  **"Federated Domain Prototype Pd as a first-class shared object, excluded against online class centroids in low-dimensional prototype space."**
- **被删/降级的**:
  - **Pc 从 "class prototype dual bank" 完全降级**: 仅作为 monitoring/diagnostic EMA buffer, 不参与 loss, 不出现在 novelty table 作为 contribution
  - L_proto_excl 的"online class centroid" 描述替代"class prototype"叙事
- **修订后为何仍 smallest adequate**:
  - 核心新增: 1 个 encoder_sty (1M) + 1 个 federated Pd + 2 条 loss (L_sty_proto, L_proto_excl)
  - Pc 是 no-loss EMA, 仅做诊断. 相比 v2 又减 1 个"看起来像 contribution 的组件"
  - L_proto_excl 现在通过 **hybrid axis** 让 headline (Pd) 和 implementation (gradient path) 对齐 — 消除 reviewer 指出的"headline/impl 错配"

## Changes Made

### 1. Fix C0 Gate (CRITICAL)
- **Reviewer said**: "Freeze encoder_sem + semantic_head + sem_classifier → inference path 冻结, L_proto_excl 不能 move predictions"
- **Action**: C0 改为**只冻 encoder_sem**, semantic_head 保持 trainable, sem_classifier 保持 trainable. 这样:
  - L_proto_excl 通过 batch class centroid 把 gradient 推到 semantic_head (因为 centroid computed from z_sem = semantic_head(pooled_sem))
  - Prediction logits 通过 sem_classifier, 能实际变化
  - baseline 是同样条件下 head-only fine-tune (没有 encoder_sty / Pd / L_proto_excl)
- **Reasoning**: reviewer 完全对 — 原 C0 冻死 prediction path 导致测不到任何 accuracy 动
- **Impact**: Validation Focus 从 6 → 预期 8

### 2. L_proto_excl 改 Hybrid Axis (CRITICAL)
- **Reviewer said**: "headline 说 key 是 federated Pd, 但 L_proto_excl 用 batch-local centroid, 不是 Pd → weakens central claim"
- **Action**: L_proto_excl 采用 **straight-through / detach-reparametrization trick**:
  ```python
  # On batch of size B, with samples from multiple classes c ∈ {c present in batch}
  #                                       and domain d (单 client 通常一个 domain)
  batch_class_centroid[c] = mean_{i: y_i==c} z_sem_i               # has grad
  batch_domain_centroid[d] = mean_{i: domain(i)==d} z_sty_i         # has grad
  
  # Hybrid axis: forward value = federated Pd, gradient goes to batch centroid
  Pd_hybrid[d] = Pd[d].detach() + batch_domain_centroid[d] - batch_domain_centroid[d].detach()
  # ↑ Forward value equals Pd[d] (shared federated object)
  # ↑ Gradient flows through batch_domain_centroid[d] (reaches encoder_sty)
  
  # Pc_hybrid defined similarly if we want symmetric treatment, OR just use
  # on-the-fly batch_class_centroid directly (more honest: class side is not federated)
  
  # Exclusion over all (c, d) pairs:
  L_proto_excl = mean_{c, d} cos²(F.normalize(batch_class_centroid[c]), F.normalize(Pd_hybrid[d]))
  ```
- **Reasoning**:
  - **Forward semantics**: exclusion 是相对 federated Pd (真正 cross-client shared prototype), 不是 client 本地的 batch centroid — **headline claim 对齐**
  - **Gradient path**: gradient 通过 batch_domain_centroid → z_sty → encoder_sty, 通过 batch_class_centroid → z_sem → semantic_head → encoder_sem. 真正有训练信号
  - **Stability**: forward 用 Pd (cross-round EMA smoothed), 比 batch-local 更稳. Gradient 用 batch 级让梯度多样
- **Impact**: Contribution Quality + Method Specificity 同时上升

### 3. Pc 完全降级为 Monitor (IMPORTANT)
- **Reviewer said**: "Pd is real federated training object; Pc is mostly EMA monitor. Calling them 'dual' overstates symmetry"
- **Action**:
  - **删除 Pc 作为 contribution** — novelty table 里只出现 Pd, 不再写 "class + domain dual bank"
  - Pc 保留为 server-side EMA buffer 用于 Vis-C 监控 (Pc-Pd cosine trajectory), 不参与任何 loss
  - L_proto_excl 用 **batch_class_centroid** (on-the-fly, 有 gradient), 不依赖 Pc
- **New headline sentence**: *"Domain should be modeled as a first-class federated prototype object Pd, excluded in prototype space against online class centroids derived from z_sem."*
- **Impact**: Contribution Quality 从 7 → 预期 9 (真正单一 crisp headline), Venue Readiness 上升

### 4. Sparse-batch Rules 明确 (IMPORTANT)
- **Reviewer said**: "Batch-local class centroids can be noisy under sparse class support. Without explicit singleton/low-count rules, L_proto_excl may inject variance not geometry."
- **Action**: 明确 exclusion loss 的 sparse-batch handling:
  ```python
  # Pseudocode
  present_classes = {c : count_in_batch(c) >= 2}  # 硬要求 ≥ 2 样本
  present_domains = {d : count_in_batch(d) >= 2}  # 通常 = 1 (单 client, 整 batch 同 domain)
  
  for c in present_classes:
      batch_class_centroid[c] = F.normalize(mean of z_sem where y==c, dim=-1)
  for d in present_domains:
      batch_domain_centroid[d] = F.normalize(mean of z_sty where domain==d, dim=-1)
  
  # 若某 class 在 batch 内缺失: fallback to stopgrad(Pc[c])
  for c in all_classes:
      if c not in present_classes:
          batch_class_centroid[c] = Pc[c].detach()  # EMA fallback, no grad
  
  # L_proto_excl computed only over present (or EMA-fallback) pairs
  L_proto_excl = mean_{c in all_classes, d in present_domains} cos²(batch_class_centroid[c], Pd_hybrid[d])
  ```
- **z_sem source 明确**: semantic_head(pooled) 的输出, L2-normalized 到 128d. 不取中间层
- **Impact**: Validation Focus + Feasibility 提升

### 5. Claim Scope Tightening (IMPORTANT)
- **Reviewer said**: "PACS/Office D=K=4, careful not to oversell empirical separation from client-bank"
- **Action**:
  - **明确说明 setup**: "In this paper's setup (PACS & Office-Caltech10, 4 clients each with distinct domain), D=K=4 and the domain-indexed Pd degenerates to a per-client bank. The domain-indexed formulation generalizes to multi-client-per-domain or client-with-multi-domain settings, but such generalization is left to future work."
  - **Mechanism honesty**: 从 "trunk decontamination" → "low-dimensional geometric exclusion against a shared federated domain axis". 不 claim 修 trunk
- **Impact**: Venue Readiness 上升

## Revised Proposal (v3)

### Title
**FedDSA-BiProto: Federated Domain Prototype as a First-Class Shared Object for Cross-Domain Federated Learning**

### One-sentence thesis (v3)
**Domain should be modeled as a first-class federated prototype object Pd, excluded in low-dimensional prototype space against online class centroids derived from the semantic encoder — yielding cross-domain semantic consistency without feature-level adversarial training.**

### Technical Gap (unchanged from v2)

现有 FL cross-domain 解耦方法对 domain 信息只有 3 种处置: 擦除 / 私有 / 本地对抗. 没有工作把 domain 建模为**与 class 方向几何互斥的联邦共享原型对象**. BiProto 填这个 gap.

### Contribution Focus (v3)

- **Dominant (C1, ONLY headline)**:  
  **Federated Domain Prototype Pd ∈ ℝ^{D×d_z}, excluded in prototype space against online class centroids computed from z_sem**.
  Mechanism:
  - Pd via per-round EMA aggregation across participating clients, domain-indexed
  - Exclusion loss via hybrid axis: forward value = federated Pd, gradient = batch-local centroid (straight-through)
  - L2-normalized cosine² over (present class, present domain) pairs, sparse-batch fallback to EMA Pc
- **Enabling infra**: Asymmetric statistic encoder_sty (~1M MLP) for proper Pd input representation — **not a separate contribution**
- **Monitor only**: Pc (EMA class centroid) for visualization and sparse-batch fallback, not a contribution
- **Explicit non-contributions**: not MI-optimal, not comm-efficient, no convergence, no cross-backbone, no DP, **not trunk decontamination**, **not dual prototype bank**

### Method

#### Complexity Budget (v3)

| Component | Role | Training |
|---|---|---|
| encoder_sem (60M, inherit) | class branch | L_CE + L_CE_aug + L_orth + L_proto_excl (via batch class centroid) |
| semantic_head, sem_classifier (inherit) | class branch | 同上 |
| BN local (FedBN) | — | not aggregated |
| style bank (μ,σ) AdaIN (inherit) | augmentation | not new |
| **encoder_sty (new, ~1M)** | domain branch (statistic MLP) | L_orth + L_sty_proto + L_proto_excl (via batch domain centroid) |
| **Pd (new, core federated object)** | domain prototype bank | server EMA, no-backward buffer |
| Pc (monitor only, derived) | class EMA centroid | server EMA, no-backward buffer, **not in any loss** |
| **L_sty_proto (new)** | InfoNCE(z_sty, Pd, domain_label) + MSE anchor | trains encoder_sty |
| **L_proto_excl (new, hybrid axis)** | cos²(batch_class_centroid, Pd_hybrid) over (c, d) pairs | trains encoder_sem + encoder_sty via ST |
| ~~L_sem_proto~~ | deleted | — |

#### Interface Specification (v3, 精化版)

**Statistic encoder_sty** (不变):
- Input: (μ, σ) from conv1-3 **BN-post pre-ReLU** taps, detached
- Architecture: Linear(in, 512) → LayerNorm → ReLU → Linear(512, 128)
- Output: z_sty ∈ ℝ^128, L2-normalized

**Pd update** (server, after each round, domain-indexed):
```
m = 0.9  # EMA decay
For each domain d ∈ {0, ..., D-1}:
    participating = {client k : participated this round AND domain_id_k == d}
    if len(participating) == 0:
        Pd[d] <- Pd[d]  # no update
    else:
        count_total = sum_{k in participating} n_k
        agg_d = sum_{k in participating} (n_k / count_total) · client_mean_{k,d}
        Pd[d] <- F.normalize(m · Pd[d] + (1-m) · F.normalize(agg_d), dim=-1)
```

**Pc update** (monitor only, server EMA, same structure but indexed by class):
- 不参与 loss, 仅用于 Vis-C 监控 Pd ⊥ Pc cosine 轨迹
- Sparse-batch fallback 时作为 class axis 替代 (detached, 仅 forward 不 backward)

**L_proto_excl computation** (client-side, per batch):
```
present_classes = {c : count_in_batch(c) >= 2}
present_domains = {d : count_in_batch(d) >= 2}  # 通常单 client = 1 个 domain

# Build class axis for all C classes
class_axis = {}
for c in all_classes:
    if c in present_classes:
        class_axis[c] = F.normalize(mean_{i: y_i==c} z_sem_i, dim=-1)  # on-the-fly, has grad
    else:
        class_axis[c] = Pc[c].detach()  # EMA fallback, no grad

# Build domain axis (hybrid: forward=Pd, grad=batch centroid via ST)
domain_axis = {}
for d in present_domains:
    bc_d = F.normalize(mean_{i: domain_i==d} z_sty_i, dim=-1)  # has grad
    domain_axis[d] = Pd[d].detach() + bc_d - bc_d.detach()  # straight-through
    # Forward == Pd[d], gradient flows through bc_d

# Exclusion: mean squared cosine over all (c, d) pairs with at least one in each
L_proto_excl = 0
for c in all_classes:
    for d in present_domains:
        L_proto_excl += torch.cos(class_axis[c], domain_axis[d]).pow(2)
L_proto_excl /= (len(all_classes) * len(present_domains))
```

**z_sem source**: semantic_head output (after L2-norm), shape [B, 128]. 这是 encoder_sem pooled 后经 semantic_head 的最终 128d 表征, 和 sem_classifier 输入一致.

**z_sty source**: encoder_sty (statistic MLP) 输出, L2-normalized 128d.

**Gradient-flow table (v3 精化)**:

| Module | L_CE | L_CE_aug | L_orth | L_sty_proto | L_proto_excl (hybrid) |
|---|:-:|:-:|:-:|:-:|:-:|
| encoder_sem | ✅ | ✅ | ✅ (via z_sem) | ❌ (taps detach) | ✅ (via batch class centroid in class_axis) |
| semantic_head | ✅ | ✅ | ✅ | ❌ | ✅ |
| sem_classifier | ✅ | ✅ | ❌ | ❌ | ❌ |
| encoder_sty | ❌ | ❌ | ✅ (via z_sty) | ✅ | ✅ (via batch domain centroid through ST) |
| Pc, Pd | ❌ | ❌ | ❌ | ❌ (MSE uses stopgrad) | ❌ (EMA only, forward anchor via ST) |

**关键点 (响应 R2 reviewer)**:
- Hybrid axis 让 forward value = federated Pd (headline 对齐), gradient 仍通过 batch centroid 流到 encoder_sty
- Sparse-batch rules: class 缺席 fallback to Pc detach, domain 缺席 (不太会发生) skip
- Pc 只做 fallback 和 monitoring, 不单独训练

#### Loss and Schedule (v3, 和 v2 相同)

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{CE,aug}} + \lambda_1 \mathcal{L}_{\text{orth}} + \lambda_2 \mathcal{L}_{\text{sty\_proto}} + \lambda_3 \mathcal{L}_{\text{proto\_excl}}
$$

Bell schedule + MSE anchor (α-sparsity 默认关闭). Schedule table 同 v2.

#### FL Aggregation (v3, 和 v2 相同)

- FedAvg: encoder_sem, semantic_head, sem_classifier, encoder_sty, LayerNorm
- Local: BN running stats (FedBN)
- Server EMA: Pd (core), Pc (monitor)
- Total comm overhead ≤ +2%

#### Failure Modes (v3)

| Failure | Detection | Fallback |
|---|---|---|
| C0 gate 不过 | S0 ablation (freeze encoder_sem only) Office R20-30 提升 < +0.3 vs head-only baseline | Kill BiProto, 改投聚合侧 |
| z_sty 坍缩 | norm 轨迹 monitor | 增 L_sty_proto or L_sty_norm_reg |
| InfoNCE 崩 | R100+ AVG 掉 > 2pp | Bell ramp-down 已内置; 补 α-sparsity |
| Batch sparsity (class 大多单样本) | present_classes ratio < 0.3 on Office | 降 batch_size 或 switch L_proto_excl 到 EMA-only mode (降级到 batch_class_centroid 全部 fallback Pc) |
| Pd 低区分度 | pairwise cos(Pd[i], Pd[j]) > 0.9 | 增 λ_sty_proto |
| L_proto_excl vs L_CE 冲突 | L_CE 异常涨 | λ_proto_excl 0.3 → 0.1 |

### Novelty Differentiation (v3, Pc 从表中移除)

| 方法 | Disent? | Federated Domain Prototype Pd? | Excl. Level | Params |
|---|:-:|:-:|:-:|:-:|
| FedProto / FPL / FedPLVM / MP-FedCL / I2PFL | — | ❌ (class only, no domain proto) | — | 1× |
| FedSTAR | FiLM | ❌ (style **local**, not federated) | feature | 1.1× |
| FDSE / FedDP / FedSeProto | erasure | ❌ (erase) | feature | ~1× |
| FedFSL-CFRD | recon | ❌ | feature | 1.2× |
| FedPall | ❌ | ❌ (class only) | feature (adv) | 1.1× |
| FISC/PARDON | — | ❌ (style stats shared, no proto) | — | 1× |
| CDANN | GRL | ❌ | feature (adv) | ~1× |
| **FedDSA-BiProto (v3)** | stat-encoder | ✅ **FIRST** | **proto (low-dim)** | **~1.02×** |

**Canonical sentence (v3)**: *"We propose Federated Domain Prototype Pd, the first federated prototype object that represents domain as a first-class shared entity dual to class at the prototype level. Pd is constructed via server-side EMA aggregation of per-client statistic encoder outputs, and excluded against online class centroids in low-dimensional prototype space via straight-through gradient hybrid axis. Under PACS and Office-Caltech10 setups where each client holds one domain (D=K=4), Pd degenerates to a per-client bank; the domain-indexed formulation generalizes to multi-client-per-domain settings, which is left to future work."*

### Claim-Driven Validation Sketch (v3)

#### Claim 0 (PRE-REQUISITE): Fixed C0 Matched Intervention Gate
- **Setup**: 加载 EXP-105 orth_only Office R200 seed=2 checkpoint
  - **Freeze encoder_sem only** (关键修复)
  - **semantic_head + sem_classifier 保持 trainable** (关键修复)
  - 加 encoder_sty + Pd + L_proto_excl + L_sty_proto (完整 domain 支路)
  - Office R20-R30, seed=2
- **Baseline**: 同 checkpoint, **freeze encoder_sem only**, head-only fine-tune (without encoder_sty / Pd / L_proto_excl), Office R20-R30
- **Decision**:
  - Δ = BiProto-lite - head-only ≥ +1.0pp → strong → go S1
  - +0.3 ~ +1.0pp → weak → proceed but temper
  - < +0.3pp → kill BiProto
- **Cost**: 2 GPU-h
- **这个 C0 真正测量 L_proto_excl 对 Office 的实际增量** (inference path 可动)

#### Claim 1 (DOMINANT): Accuracy Win — Stage-gated
- S1: Office seed=2 R200 smoke (4 GPU-h) → AVG Best ≥ 90.0 才 promote
- S2: Office 3-seed R200 (20 GPU-h) → 3-seed mean ≥ 91.08 才 promote
- S3: PACS 3-seed R200 (30 GPU-h) → 3-seed mean ≥ 80.91
- S4: Ablations (40 GPU-h)
  - BiProto − Pd (用 batch-only centroid 作 domain axis, 看 federated EMA 贡献)
  - BiProto − encoder_sty (换回 orth_only 的 style_head, 看统计 encoder 贡献)
  - BiProto − L_proto_excl (只有 L_sty_proto, 看 exclusion 必要)
- **τ sweep 移出 core ablation**, 仅在 S4 pass 后 fallback
- Kill: R50 崩 or R150 drop > 3pp

#### Claim 2 (SUPPORTING): 3-Suite Visual Evidence (v3, 和 v2 相同)
- Vis-A: t-SNE dual panel (z_sem class / z_sty domain) × 4 methods × silhouette
- Vis-B: Probe ladder (linear/MLP-64/MLP-256 × 4 directions)
- Vis-C: Prototype + feature health matrix (Pd separation + Pd⊥Pc cosine + norm/rank + orth trajectory)

### Compute Plan (v3, stage-gated 和 v2 相同)

| Stage | GPU-h |
|---|:-:|
| S0 (fixed C0) | 2 |
| Impl | 0 |
| S1 smoke | 4 |
| S2 Office 3-seed | 20 |
| S3 PACS 3-seed | 30 |
| S4 ablations | 40 |
| Vis | 2 |
| **Total** | ≤ 98 |
| **Pilot (S0+S1+S2)** | **≤ 26** ✅ fits 50 GPU-h anchor |

## Intentionally Excluded (v3)

DomainNet, ResNet-18, DP, learnable τ/α, FINCH multi-cluster, GRL, HSIC, Kendall, L_sem_proto, **Pc as contribution / dual bank claim / trunk decontamination claim** (全部因 reviewer 反对而删).

## v2 → v3 Deltas

| 项 | v2 | v3 | 理由 |
|---|---|---|---|
| C0 gate | freeze encoder_sem + head + classifier | **freeze encoder_sem only, head trainable** | R2 CRITICAL: inference path 不能冻死 |
| L_proto_excl forward axis | batch-local centroid | **hybrid: Pd (forward) + batch (grad) via ST** | R2 CRITICAL: headline-impl 对齐 |
| Pc 角色 | derived EMA + 在 novelty table 作为 "dual bank" | **monitor only**, novelty table 移除 | R2 IMPORTANT: pseudo-duality 风险 |
| Sparse-batch | unspecified | **明确 ≥2 count rule + Pc fallback** | R2 IMPORTANT: 避免 singleton variance |
| Scope | claim "domain-indexed" 全场景通用 | **明确 PACS/Office D=K=4 下等价 per-client bank, generalization 是 future work** | R2 IMPORTANT: oversell 风险 |
| τ sweep | core ablation | **moved to S4 fallback** | R2 simplification: not first-order |
