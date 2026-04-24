# FedDSA-BiProto: Federated Domain Prototype via Straight-Through Hybrid Exclusion

**Final version**: v4 (Round 4, refined from 4 rounds of GPT-5.4 xhigh review)
**Date**: 2026-04-24
**Status**: Near-READY (8.75/10, REVISE). Remaining gap is empirical (D=K=4 ablation evidence), not design.

---

## Problem Anchor (immutable)

- **Bottom-line**: 在 FL cross-domain 下用 AlexNet from scratch, 3-seed {2,15,333} × R200 mean AVG Best **必须同时严格超过 FDSE 本地复现 baseline**: PACS > 79.91 (当前 orth_only 80.64 ✅) 且 Office-Caltech10 > 90.58 (当前 orth_only 89.09 ❌ −1.49).
- **Must-solve bottleneck**: Office 补回 −1.49 pp 且至少 +0.5 pp, PACS 不退 (≥ 80.91).
- **Non-goals**: 不换数据集 / 不做诊断论文 / 不堆模块凑 novelty / 不预训练 / 不换骨干.
- **Constraints**: AutoDL 4090 + lab 3090; Pilot 预算 ≤ 50 GPU-h hard; ≤ 120 GPU-h 总 incl. ablations; AlexNet + FedBN; PACS E=5 / Office E=1; 1 周.
- **Success**: 3-seed mean AVG Best PACS ≥ 80.91 且 Office ≥ 91.08; AVG Last 不退; t-SNE + probe ladder + prototype/feature metrics 三套可视化 evidence 齐备.

---

## Thesis (benchmark-scoped)

**To our knowledge, we are the first to maintain a federated domain prototype object Pd — a server-side EMA centroid across participating clients — and use it as the forward anchor of low-dimensional prototype-space class-domain exclusion via a straight-through hybrid gradient axis. Empirical validation is scoped to D=K=4 benchmarks (PACS, Office-Caltech10), with full-scope generalization left as future work.**

---

## Contribution Focus

- **Dominant (C1, only headline)**:  
  Federated Domain Prototype **Pd ∈ ℝ^{D×d_z}** with prototype-space class-domain exclusion via **straight-through hybrid axis**. Pd is server-side EMA across participating clients, used as forward anchor of exclusion; gradient flows through batch-local centroid to encoder_sty. This replaces feature-level adversarial/MI-minimization routes (CDANN / FedDP / FedSeProto) that were falsified in our setup (EXP-108: probe 0.96, zero accuracy gain).
- **Enabling infrastructure (NOT a contribution)**:  
  Asymmetric statistic encoder_sty (~1M MLP) for proper Pd input representation — structurally cannot encode class discriminative signal because it only processes (μ, σ) statistics, not spatial features.
- **Monitor only (NOT a contribution)**:  
  Pc = EMA class centroid, used only in Vis-C for visualization; never participates in any loss.
- **Explicit non-contributions**: not MI-optimal, not communication efficient, no convergence proof, no cross-backbone, no DP, **not trunk decontamination**, **not dual prototype bank**.

---

## Technical Gap

Current FL cross-domain 解耦 methods 对 domain 信息只有 3 种处置, 均在我们 setup 下被 evidence 约束:

| 派 | 代表 | 对 domain 的处置 | 我们 setup 下的证据 |
|---|---|---|---|
| 擦除 | FDSE, FedDP, FedSeProto | noise, feature-level MI min | FDSE 本地复现 PACS 79.91 被 orth_only 80.64 超 (+0.73); FDSE 是主要 Office 对手 (90.58) |
| 私有 | FedSTAR, FedBN, FedSDAF | 本地保留, 不跨 client | FedBN 是 orth_only 的前身组件, 已是基线 |
| 共享不解耦 | FISC, StyleDDG, FedCCRL, MixStyle | 混合空间 AdaIN | EXP-060/061 实验: 在 Office 反而 −1.97 pp |
| Feature-level 对抗 | CDANN 变体 | GRL 推 z_sem 去 domain | EXP-108 3-seed 证伪: probe 0.96 但 accuracy 0 增益 |

**Missing**: 没有工作把 domain 建模为**与 class 方向几何互斥的联邦共享原型对象**. 本方案填这个 gap, 并在 low-dim proto 空间操作, 避开 feature-level MI/对抗的历史坑 (EXP-017 HSIC / EXP-108 CDANN).

---

## Proposed Method

### Architecture Overview

```
         ┌─────────── x (input image) ───────────┐
         │                                        │
         ▼                                        │
  AlexNet encoder_sem (60M, inherit orth_only)   │
  ├── conv1 → BN₁ → pool ──┐                      │
  ├── conv2 → BN₂ → pool ──┼── stat taps          │
  ├── conv3 → BN₃ ─────────┘  (μ,σ per-channel,   │
  │                            detached)          │
  └── ... deeper + pool → 1024d                   │
                  │                               │
                  ├──► semantic_head → z_sem [128] (L2-norm)
                  │                    │
                  │                    ├──► sem_classifier → L_CE
                  │                    │
                  │                    └──► AdaIN(μ,σ from style bank)
                  │                         → z_sem_aug → L_CE_aug (inherit FedDSA)
                  │
                  ↓
    statistic_encoder_sty [MLP ~1M]
              ↓
          z_sty [128] (L2-norm)
              │
              ├──► cos²(z_sem, z_sty) ───────► L_orth (inherit, λ=1)
              └──► InfoNCE + MSE anchor vs Pd[domain] ──► L_sty_proto
              
    On-the-fly batch centroids:
      class_axis[c]  = F.normalize(mean z_sem[y==c])
      domain_axis[d] = F.normalize(Pd[d].detach() + bc_d − bc_d.detach())  # ST hybrid
              │
              └──► cos²(class_axis[c], domain_axis[d]) over present pairs ──► L_proto_excl
```

### Interface Specification

**Statistic encoder_sty** (~1M, 新增):
- Input: (μ, σ) per-channel from conv1-3 **BN-post pre-ReLU** activations, detached to prevent gradient flow to encoder_sem
- Architecture: `Linear(2·Σ_l C_l, 512) → LayerNorm → ReLU → Linear(512, 128)`
- Output: z_sty ∈ ℝ^{B×128}, L2-normalized

**Pc / Pd EMA update** (server-side, after each round):
- Both L2-normalized and stored as [C, 128] / [D, 128] buffers (no gradient)
- Decay m = 0.9
```python
# Pd update (Pc identical, just class-indexed)
for d in range(D):
    participating = {k : client_k participated AND domain(k) == d}
    if participating:
        agg = weighted_mean_{k ∈ participating} (client_mean_{k,d})
        Pd[d] ← F.normalize(0.9 · Pd[d] + 0.1 · F.normalize(agg), dim=-1)
    # else: no-op (preserve previous value)
```

**Initialization** (R4 AI-1): Pd[d] 初始化为 F.normalize(torch.randn(d_z)) with seed-fixed RNG, 在 R0-R50 warmup 期间通过 L_sty_proto 和 AdaIN 辅助训练累积 "first-available client centroid" style 信息. Pc 同理. 不从零初始化 (会导致 cosine 计算中 NaN); 不从第一 client batch 计算 (partial participation 时可能 missing).

**L_proto_excl computation** (client-side per batch, present-classes-only):
```python
present_classes = {c : count_in_batch(c) ≥ 2}
present_domains = {d : count_in_batch(d) ≥ 2}  # typically = 1 (single client = single domain)

if not present_classes or not present_domains:
    L_proto_excl = 0.0  # no-op, rare on Office/PACS batch=50 (<0.5% probability)
else:
    class_axis = {c: F.normalize(mean z_sem[y==c], dim=-1) for c in present_classes}

    domain_axis = {}
    for d in present_domains:
        bc_d = F.normalize(mean z_sty[domain==d], dim=-1)       # has grad
        raw = Pd[d].detach() + bc_d - bc_d.detach()              # forward = Pd[d]
        domain_axis[d] = F.normalize(raw, dim=-1)                # final renormalize

    total, count = 0.0, 0
    for c in present_classes:
        for d in present_domains:
            total += F.cosine_similarity(class_axis[c].unsqueeze(0),
                                          domain_axis[d].unsqueeze(0)).pow(2).squeeze()
            count += 1
    L_proto_excl = total / count
```

**关于 Straight-Through Hybrid Axis (R4 AI-4, 明确透明)**:  
> *The straight-through composition `Pd[d].detach() + bc_d - bc_d.detach()` is a **biased but intentional training surrogate**: it fixes the forward cosine value to the federated Pd[d] (matching the headline claim), while routing the gradient through the minibatch domain centroid bc_d to reach encoder_sty. The bias vanishes as Pd and bc_d converge (EMA stabilizes post-warmup), empirically monitored via Vis-C Pd⊥bc_d cosine trajectory.*

### Gradient-flow table

| Module | L_CE | L_CE_aug | L_orth | L_sty_proto | L_proto_excl |
|---|:-:|:-:|:-:|:-:|:-:|
| encoder_sem | ✅ | ✅ | ✅ (via z_sem) | ❌ (taps detach) | ✅ (via batch class centroid) |
| semantic_head | ✅ | ✅ | ✅ | ❌ | ✅ |
| sem_classifier | ✅ | ✅ | ❌ | ❌ | ❌ |
| encoder_sty | ❌ | ❌ | ✅ (via z_sty) | ✅ | ✅ (via ST through batch domain centroid) |
| Pc, Pd | ❌ | ❌ | ❌ | ❌ (MSE uses stopgrad) | ❌ (EMA only, forward anchor via ST) |

### Loss Composition & Bell Schedule

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{CE,aug}} + \lambda_1 \mathcal{L}_{\text{orth}} + \lambda_2 \mathcal{L}_{\text{sty\_proto}} + \lambda_3 \mathcal{L}_{\text{proto\_excl}}
$$

- **L_CE, L_CE_aug**: 继承 FedDSA 原方案 (CE on logits, CE on AdaIN-augmented z_sem)
- **L_orth** = cos²(z_sem, z_sty), λ₁=1.0 全程
- **L_sty_proto** = InfoNCE(z_sty, Pd, domain_label; τ=0.1) + 0.5·MSE(z_sty, stopgrad(Pd[d]))
- **L_proto_excl** = present-classes-only (见上公式)

| Round | λ_orth | λ_sty_proto | λ_proto_excl | MSE coef | Phase |
|:-:|:-:|:-:|:-:|:-:|---|
| 0-49 | 1.0 | 0 | 0 | 0 | Warmup: orth_only + AdaIN, Pc/Pd EMA 预热 |
| 50-80 | 1.0 | 0→0.5 | 0→0.3 | 0.5 | Ramp-up |
| 80-150 | 1.0 | 0.5 | 0.3 | 0.5 | Peak |
| 150-200 | 1.0 | 0.5→0 | 0.3 | 0.5 | L_sty_proto ramp-down |

**Safety valves** (EXP-076/077 validated): Bell + MSE anchor. α-sparsity **disabled by default**, added only if pilot collapse appears.

### FL Aggregation

| Parameter group | Strategy | Comm cost |
|---|---|---|
| encoder_sem (60M) | FedAvg | ~240 MB/client/round |
| semantic_head, sem_classifier | FedAvg | <1 MB |
| encoder_sty (~1M) | FedAvg | ~4 MB/client/round (+1.7%) |
| BN running stats | Local (FedBN) | 0 |
| LayerNorm (sty side) | FedAvg | <1 KB |
| Pc [C × 128] | Server EMA | <5 KB broadcast |
| Pd [D × 128] | Server EMA | <2 KB broadcast |
| style bank (μ, σ) | 继承 FedDSA 原方案 | 已有 |

**Total comm overhead ≤ +2%** vs baseline.

### Failure Modes & Fallbacks

| Failure | Detection | Fallback |
|---|---|---|
| C0 gate fails | Δ(BiProto-lite vs head-only) < +0.3pp on Office R20-30 | **Kill BiProto**, 改投聚合/outlier 侧 |
| z_sty 坍缩 | z_sty_norm < 0.3 over R ≥ 10 round | 增 λ_sty_proto 或加 L_sty_norm_reg |
| InfoNCE late collapse | AVG Last 掉 > 2pp post-R100 | Bell ramp-down built-in; 补 α-sparsity 作第三层安全阀 |
| Sparse batches | present_classes ratio < 0.3 on Office | 降 batch_size 或启 Pc fallback mode (appendix variant) |
| Pd 低区分度 | pairwise cos(Pd[i], Pd[j]) > 0.9 | 增 λ_sty_proto |
| L_proto_excl vs L_CE 冲突 | L_CE 异常涨 > 20% | λ_proto_excl 0.3 → 0.1 |

---

## Claim-Driven Validation

### Claim 0 (PRE-REQUISITE): C0 Matched-Intervention Pruning Test

**Role disclaimer**: *C0 is a fast intervention pruning test, not a causal proof. A positive C0 is **necessary but not sufficient** for S1-S3 R200 success; a negative C0 is **sufficient** to kill BiProto.*

**Setup**:
- Load EXP-105 orth_only Office R200 seed=2 checkpoint
- **Freeze encoder_sem only** (semantic_head + sem_classifier remain trainable)
- Add encoder_sty + Pd + L_sty_proto + L_proto_excl
- Office R20-R30, seed=2

**Baseline**: 同 checkpoint, 同 freeze scope, head-only fine-tune without encoder_sty / Pd / L_proto_excl

**Decision rule**:
- Δ ≥ +1.0pp → strong pruning signal → S1
- Δ +0.3 ~ +1.0pp → weak pruning signal → proceed but temper expectations
- Δ < +0.3pp → **kill BiProto**, 改投聚合侧

**Cost**: 2 GPU-h

### Claim 1 (DOMINANT): Stage-Gated Accuracy Win

| Stage | Content | GPU-h | Promote Gate |
|---|---|:-:|---|
| S1 | Office seed=2 R200 smoke | 4 | AVG Best ≥ 90.0 |
| S2 | Office 3-seed {2,15,333} R200 | 20 | 3-seed mean ≥ 91.08 |
| S3 | PACS 3-seed R200 | 30 | 3-seed mean ≥ 80.91 |
| S4 | Ablations (见下表) | up to 40 | — |

**Kill criteria**: R50 崩 (PACS < 80, Office < 86) or R150 drop > 3pp

### Ablation Schedule (R4 AI-3, 精确定义)

| Ablation | Priority | 变更 (只改一个 factor) | 测的是什么 |
|---|:-:|---|---|
| **−Pd** | **MANDATORY** | 改 `domain_axis[d] = F.normalize(bc_d)` (删 Pd.detach() anchor + ST). 其他一切保持. | Federated Pd 的真实增量 (vs batch-local + 服务端 buffer) |
| −L_proto_excl | RECOMMENDED | 删 L_proto_excl 项. 保留 Pd, L_sty_proto, encoder_sty. | Exclusion 机制必要性 |
| −encoder_sty | RECOMMENDED | 换回 orth_only 的 symmetric `style_head = semantic_head` 架构. z_sty 来自 pooled_sem 经 128d head. 其他一切保持. **只改 encoder 结构, 不删 Pd 也不删 L_proto_excl.** | 非对称统计 encoder 的 inductive bias 必要性 |
| τ sweep | OPTIONAL | τ ∈ {0.05, 0.1, 0.2} on L_sty_proto. 其他一切保持. | Sensitivity, 仅 S4 pass 后跑 |

### Pre-registered Claim Downgrade (R4 AI-2)

**If the −Pd ablation (MANDATORY) falls within 0.5 pp of full BiProto on Office 3-seed mean AVG Best**, the central claim **downgrades pre-emptively** to:

> *"Our method demonstrates that a low-dimensional prototype-space class-domain geometric exclusion mechanism — implemented with a local domain centroid (plus optional server-side EMA for monitoring) — substantially improves cross-domain FL on Office-Caltech10 and maintains PACS. Whether the server-side EMA Pd contributes beyond the local domain centroid is an open empirical question that requires D≠K benchmarks to resolve."*

该 wording 写入 paper method section 作为 conditional contingency, 避免 −Pd 出结果后的 post-hoc 叙事漂移.

### Claim 2 (SUPPORTING): 3-Suite Visual Evidence

**Vis-A: t-SNE Dual Panel** (paper Fig 2)
- 4 methods (orth_only / FDSE / CDANN / BiProto) × 2 datasets × 2 features (z_sem, z_sty) = 16 subplots
- z_sem colored by class (expect class cluster across domains), z_sty colored by domain (expect domain cluster)
- Silhouette score quantification per panel

**Vis-B: Probe Ladder** (paper Fig 3 / Table 2)
- Linear / MLP-64 / MLP-256 × 4 directions (z_sem→class, z_sem→domain, z_sty→class, z_sty→domain) × 4 methods
- Target matrix:

| | Linear | MLP-64 | MLP-256 |
|---|:-:|:-:|:-:|
| z_sem→class | >0.85 | >0.90 | >0.92 |
| z_sem→domain | <0.35 | <0.45 | <0.55 |
| z_sty→class | <0.30 | <0.40 | <0.50 ← 关键进步 (vs orth_only 0.81) |
| z_sty→domain | >0.90 | >0.95 | >0.97 |

**Vis-C: Prototype + Feature Health Matrix** (paper Fig 4, single combined figure)
- Pc / Pd pairwise cosine off-diagonal mean trajectory (every 10 round)
- Pd ⊥ Pc cosine matrix heatmap (final value)
- z_sem / z_sty norm + effective rank (SVD) trajectory
- cos(z_sem, z_sty) + HSIC (computed, not loss) trajectory
- **Pd ⊥ bc_d cosine trajectory** (verifies ST estimator bias vanishing claim from R4 AI-4)

---

## Limitations (R4 AI-3 派生, paper 小节)

**(a) Benchmark-scoped empirical validation.** 本文 empirical validation 限定于 D=K=4 benchmarks (PACS, Office-Caltech10). 该 configuration 下 domain-indexed Pd 在实现上退化为 per-client prototype bank, "federated domain object" 和 "per-client centroid with server aggregation" 的概念区别无法 empirically 完全 disambiguate. −Pd ablation (§C.3, MANDATORY) 通过对比 Pd-anchored hybrid ST axis vs batch-local-only centroid 部分回应, 但完整 validation 需要 D ≠ K benchmarks (如 DomainNet multi-client-per-domain), 留作 future work.

**(b) Straight-through estimator bias.** Hybrid ST axis 引入 mild estimator bias: forward cosine 相对 federated Pd[d], gradient 来自 batch-local domain centroid. 该 bias 在 Pd 和 bc_d 收敛后消失 (EMA 稳定), 通过 Vis-C Pd ⊥ bc_d cosine trajectory 监测 (预期 post-R100 接近 1.0).

**(c) No cross-backbone generalization.** 实验限定于 AlexNet from scratch. ResNet-18 / ViT 等 backbone 的 transferability 留作 future work.

---

## Compute & Timeline

| Stage | GPU-h | Wall | Promote Gate |
|---|:-:|:-:|---|
| S0 C0 gate | 2 | 2h | Δ ≥ +0.3 pp 才进 S1 |
| Implementation | 0 | 0.5d | ast.parse + unit test + codex review |
| S1 smoke | 4 | 4h | Office seed=2 AVG Best ≥ 90.0 |
| S2 Office 3-seed | 20 | 12h | 3-seed mean ≥ 91.08 |
| S3 PACS 3-seed | 30 | 20h | 3-seed mean ≥ 80.91 |
| S4 ablations (−Pd mandatory + recommendeds) | 40 | 24h | — |
| Vis | 2 | 0.5d | — |
| **Total** | **≤ 98** | ~4-5d | — |
| **Pilot (S0+S1+S2)** | **≤ 26** | ~18h | **Fits ≤50 GPU-h anchor** ✅ |
| **To-first-kill** | **≤ 6** (S0 + Impl + S1) | ~8h | — |

---

## Novelty Differentiation

| 方法 | Disentanglement | **Federated Domain Prototype?** | Exclusion level | Params overhead |
|---|:-:|:-:|:-:|:-:|
| FedProto / FPL / FedPLVM / MP-FedCL / I2PFL | — | ❌ (class only) | — | 1× |
| FedSTAR (2024) | FiLM | ❌ (style **local**, not federated) | feature | 1.1× |
| FDSE (CVPR'25) | layer decomp | ❌ (erase) | feature | 0.65× |
| FedDP / FedSeProto (2024-25) | MI min | ❌ (erase) | feature | ~1× |
| FedFSL-CFRD (AAAI'25) | reconstruction | ❌ (common/personal, 非 class/domain) | feature | 1.2× |
| FedPall (ICCV'25) | — | ❌ (class only) | feature (adv) | 1.1× |
| FISC / PARDON (ICDCS'25) | — | ❌ (style stats shared, no proto) | — | 1× |
| CDANN variants | GRL | ❌ | feature (adv) | ~1× |
| **FedDSA-BiProto (ours)** | stat-encoder | ✅ **FIRST** | **proto (low-dim)** | **~1.02×** |

---

## Intentionally Excluded

- **DomainNet** (scope, Limitations (a))
- **ResNet-18 / ViT backbones** (scope, Limitations (c))
- **DP / privacy analysis** (not method venue)
- **Learnable τ / α** (EXP-028 Kendall weighting 已证伪)
- **FINCH multi-cluster Pd** (EXP-095 SCPR 已证伪)
- **GRL / adversarial dom_head** (EXP-108 CDANN 已证伪)
- **HSIC / feature-level MI min** (EXP-017 已证伪)
- **L_sem_proto** (R1 reviewer: contribution sprawl)
- **Pc as contribution / dual bank claim** (R2 reviewer: pseudo-duality)
- **Trunk decontamination claim** (R2 reviewer: 不诚实, mechanism ≠ story)
- **Absent-class Pc fallback** (R3 reviewer: pilot 简化, 移到附录)
- **α-sparsity by default** (R1 reviewer: too many safety valves)

---

## Next Steps

1. **/experiment-plan**: 把本 FINAL_PROPOSAL 转为详细 experiment roadmap (每 stage task 分解, S0/S1/S2/S3/S4 具体命令 + kill criteria)
2. **/run-experiment S0**: 2 GPU-h 跑 C0 matched-gate, 判决 BiProto 是否启动
3. **如果 S0 通过** (Δ ≥ +0.3pp): 实现 feddsa_biproto.py (~250 行) + unit tests + codex review → S1 smoke
4. **如果 S0 失败** (Δ < +0.3pp): **Kill BiProto**, 回 Calibrator 兜底或聚合侧优化 (SAS τ tune / Caltech 权重)

**Reviewer 剩余 empirical 疑虑** (R4 confirmed): Pd 是否真有 "federated object" 增量. 该问题由 −Pd MANDATORY ablation 的数据直接回答, 无法通过进一步 method refinement 解决. 这正是 Phase 5 stop 的合理理由.
