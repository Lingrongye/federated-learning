# FedDSA-CDANN: Constrained Dual-Directional DANN for Federated Domain Generalization — 技术方案

**日期**: 2026-04-20
**对应大白话版**: [大白话_FedDSA-CDANN.md](大白话_FedDSA-CDANN.md)
**Final refined proposal**: [refine-logs/2026-04-20_FedDSA-CDANN/FINAL_PROPOSAL.md](../../refine-logs/2026-04-20_FedDSA-CDANN/FINAL_PROPOSAL.md)
**Score**: 8.75/10 REVISE (5 rounds Codex gpt-5.4 xhigh review, proposal-complete at review-time ceiling)

---

## Problem Statement

### Observed failure mode

在 `client=domain` Federated Domain Generalization (FedDG) 场景下, FedDSA-SGPA (Linear+whitening) 表现出**数据集依赖**的不一致行为:

| Dataset | AVG Best | vs Plan A (orth_only baseline) | z_sty_norm R10→R200 |
|---------|----------|-------------------------------|---------------------|
| Office-Caltech10 | 88.75 | **+6.20pp** 🔥 | 3.12 → 2.21 (-2%) |
| PACS_c4 | 80.20 | **-1.49pp** ⚠️ | 3.12 → 0.15 (**-95%**) |

**Root cause analysis (from diagnostic jsonl 数据)**:
- 原 decoupling loss `L_orth = cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)` 要求统计独立但**无方向性**
- 当数据集的 domain-shift 表现为 style 信号携带 class-discriminative 信息时 (e.g., PACS sketch 的线条结构是 class 识别的核心), 模型可能把 class-relevant 特征 route 到 z_sty
- Subsequent pooled whitening broadcast `Σ_inv_sqrt` 广播归一化无差别 magnifies z_sty attenuation across clients, erasing class signal

### Scope disclaimer

本方案仅适用:
1. `client=domain` bijection (Office/PACS 4 clients = 4 domains)
2. Style carries class signal (measured empirically via `probe_sty_class` ≫ random)

多域/client 扩展 (DomainNet) 留作 future work, 需要 domain cluster pseudo-labels.

---

## Method Thesis

**Novelty (locked one sentence, verbatim in Abstract / Method / Discussion)**:

> A shared non-adversarial domain discriminator plus asymmetric encoder-gradient supervision (GRL on z_sem path only) is the minimal repair for whitening-induced style collapse in client=domain FedDG when style carries class signal.

---

## System Architecture

```
                                                     ┌── sem_classifier(z_sem) ── L_task = CE(y)    [task, encoder forward]
  x ─ encoder ─ feat ──┬─ sem_head → z_sem ─ (WH) ──┤
                       │                             └── dom_head(GRL(z_sem, λ)) ── L_dom_sem       [dom_head forward CE; encoder via GRL reversed]
                       └─ sty_head → z_sty ─ (WH) ──── dom_head(z_sty)            ── L_dom_sty       [dom_head forward CE; encoder forward]

                       Decoupling:  L_dec = λ_orth · cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)

  dom_head: non-adversarial discriminator, both branches minimize standard CE.
  Asymmetry resides in encoder gradient direction (GRL on z_sem path only).

  Inference: argmax sem_classifier(z_sem). z_sty only used for downstream pooled style bank broadcast (preserve Plan A).
```

---

## Core Mechanism (Precise Formulation)

### Components

1. **Encoder** (frozen from FedDSA-SGPA): AlexNet → feature $\mathbf{f} \in \mathbb{R}^{B \times 1024}$
2. **Dual heads** (frozen): `sem_head`, `sty_head`: $\mathbb{R}^{1024} \to \mathbb{R}^{128}$
3. **Pooled whitening** (frozen from FedDSA-SGPA): broadcast $(\mu_{global}, \Sigma_{inv\_sqrt})$, produce post-whitening $\tilde{\mathbf{z}}_{sem}, \tilde{\mathbf{z}}_{sty} \in \mathbb{R}^{128}$
4. **New: shared domain head**:
   $$\text{dom\_head}: \mathbb{R}^{128} \to \mathbb{R}^{N_{clients}}, \quad \text{MLP}(128 \to 64 \to N_{clients}) \text{ with Dropout}(0.1), \sim 9K \text{ params}$$
5. **New: Gradient Reversal Layer (GRL)**:
   $$\text{GRL}_\lambda: \text{forward}(x) = x, \quad \text{backward}(\nabla) = -\lambda \cdot \nabla$$

### Loss functions

$$
\mathcal{L}_{task} = \mathrm{CE}(y, \text{sem\_classifier}(\tilde{\mathbf{z}}_{sem}))
$$

$$
\mathcal{L}_{dec} = \lambda_{orth} \cdot \cos^2(\tilde{\mathbf{z}}_{sem}, \tilde{\mathbf{z}}_{sty}) + \lambda_{hsic} \cdot \text{HSIC}(\tilde{\mathbf{z}}_{sem}, \tilde{\mathbf{z}}_{sty})
$$

$$
\mathcal{L}_{dom\_sem} = \mathrm{CE}(d, \text{dom\_head}(\text{GRL}_{\lambda_{adv}}(\tilde{\mathbf{z}}_{sem})))
$$

$$
\mathcal{L}_{dom\_sty} = \mathrm{CE}(d, \text{dom\_head}(\tilde{\mathbf{z}}_{sty}))
$$

$$
\boxed{\mathcal{L}_{total} = \mathcal{L}_{task} + \mathcal{L}_{dec} + \mathcal{L}_{dom\_sem} + \mathcal{L}_{dom\_sty}}
$$

### Gradient flow analysis (key clarification)

**dom\_head 参数更新** (通过 $\mathcal{L}_{dom\_sem} + \mathcal{L}_{dom\_sty}$):
- Both branches provide standard CE gradient
- dom\_head 是 **non-adversarial** discriminator (两路都 minimize CE, 不对抗)
- Same head parameters are updated by both branches

**Encoder upstream gradients** (asymmetry):
- From $\mathcal{L}_{task}$: positive gradient, forces $\tilde{\mathbf{z}}_{sem}$ to be class-discriminative
- From $\mathcal{L}_{dom\_sem}$ through GRL: **reversed gradient** with factor $-\lambda_{adv}$, forces $\tilde{\mathbf{z}}_{sem}$ to be **domain-indistinguishable**
- From $\mathcal{L}_{dom\_sty}$: positive gradient (no GRL), forces $\tilde{\mathbf{z}}_{sty}$ to be **domain-discriminative**

**Critical insight**: Asymmetry resides **not in dom_head objective** (which is symmetric standard CE on both branches) but in **encoder's two upstream path gradient directions** (reversed on z_sem, forward on z_sty).

### Scheduling

$$
\lambda_{adv}(r) = \min(1.0, \max(0, (r - 20) / 20))
$$

- $r \in [0, 20]$: warmup baseline (λ_adv = 0, equivalent to FedDSA-SGPA Plan A + whitening)
- $r \in [20, 40]$: linear ramp-up
- $r \geq 40$: full strength

Inherited from Plan A: $\lambda_{orth} = 1.0$, $\lambda_{hsic} = 0.1$.

### Aggregation strategy

- `dom_head` parameters participate in **FedAvg** aggregation (same as encoder / sem_classifier)
- **Rationale**: local client sees only one domain → local dom_head degenerates (constant classifier). Only post-aggregation dom_head discriminates across $N_{clients}$ domains, which is essential for meaningful GRL signal.
- **Privacy**: aggregating head parameters (not data); domain label $= \text{client id}$ is not sensitive in FL header.

---

## Evaluation Protocol

### Frozen post-hoc probes (post-whitening feature space)

**Critical detail**: All three probes operate on **post-whitening features** (same space as sem_classifier training). This is essential because the claimed failure mode is whitening-induced collapse.

```python
# After R=200 training, freeze encoder / heads / whitening parameters.
# Aggregate features across all clients.
Z_sem_train, Z_sty_train, D_train, Y_train = aggregate_across_clients(train_loader)
Z_sem_test,  Z_sty_test,  D_test,  Y_test  = aggregate_across_clients(test_loader)

# Train probes ON TRAIN features
probe_sem_dom = LogisticRegression().fit(Z_sem_train, D_train)   # z_sem → domain
probe_sty_dom = LogisticRegression().fit(Z_sty_train, D_train)   # z_sty → domain
probe_sty_cls = LogisticRegression().fit(Z_sty_train, Y_train)   # z_sty → class (anchor-aligned)

# Report HELD-OUT TEST accuracies
print(probe_sem_dom.score(Z_sem_test, D_test))   # expect ≈ 1/N (random)
print(probe_sty_dom.score(Z_sty_test, D_test))   # expect ≈ 1.0
print(probe_sty_cls.score(Z_sty_test, Y_test))   # KEY: PACS ≥ 40% (CDANN) vs ≈ 15% (baseline)
```

### Claim-driven experiments

**Claim C-main (Primary)**: CDANN fixes PACS without hurting Office
- Experiment: PACS R200 3 seeds + Office R200 3 seeds vs Linear+whitening baseline
- Metrics: AVG Best, probe_sem_domain, probe_sty_domain, **PACS probe_sty_class**, z_sty_norm R200
- Expected:
  - PACS AVG Best ∈ [82, 84] (recover from -1.49pp to ≥0 vs Plan A 81.69)
  - Office AVG Best ≥ 88.0
  - PACS probe_sty_class ≥ 40% (vs baseline ≈ 15%)
  - probe_sem_domain ≈ 0.25 (random), probe_sty_domain ≈ 0.95
  - z_sty_norm R200 ≥ 1.5 (vs baseline 0.15)

**Claim C-ablate (Required)**: Necessity of z_sty positive supervision
- Variants on PACS R200 × 2 seeds:
  - V1 baseline: Linear+whitening (no CDANN)
  - V2 z_sem-only: only $\mathcal{L}_{dom\_sem}$ (standard DANN)
  - V3 full CDANN (ours)
- Expected: V3 > V2 ≥ V1 on both AVG Best AND probe_sty_class
- Key evidence: `probe_sty_class` gap between V3 and V2 is **consistent with** anchor claim "z_sty positive supervision preserves class-relevant style" (not formal causal proof)

**Appendix sanity check C-port (not a contribution)**:
- PACS R100 1 seed with frozen DINOv2-S/14 encoder (everything else unchanged)
- Goal: mechanism not AlexNet-specific
- Expected: AVG Best ≥ AlexNet CDANN

---

## Novelty Analysis

### vs. closest prior work

| Prior | Relation | Our delta |
|-------|----------|-----------|
| Deep Feature Disentanglement SCL (Cog. Comp. 2025) | Non-FL, symmetric positive supervision, **separate heads** | FL + **asymmetric** encoder-gradient + **shared non-adversarial** head |
| FedPall (ICCV 2025) | Adversarial Amplifier in **mixed feature** space, **erase** domain | Decoupled space, **preserve** domain in z_sty |
| ADCOL (ICML 2023) | DANN in FL, **erase** all client differences | Asymmetric preserve (selectively retain on z_sty) |
| Federated Adversarial DA (arXiv 1911.02054) | Vanilla DANN in FL, feature-level disentanglement | Encoder-gradient asymmetry + shared non-adversarial discriminator |
| FediOS (ML 2025) | Orthogonal generic/personalized subspaces, no domain supervision | Domain-supervised asymmetric gradient |
| FDSE (CVPR 2025) | Layer decomposition, erase domain offsets | Feature decoupling, preserve style |
| FedSeProto (ECAI 2024) / FedDP (TMC 2025) | MI minimization, **discard** domain features | Preserve via supervision |

**Defense against pseudo-novelty accusation**: Precise framing emphasizes:
1. **dom_head is non-adversarial** (both branches minimize standard CE, not a GAN-style game)
2. **Asymmetry is encoder-side** (GRL only on z_sem → encoder path, identity on z_sty → encoder path)
3. **Mechanism minimality**: 1 MLP (9K params), no aggregation / whitening changes, no additional trainables

This combination is not explicitly formulated in 2024-2026 literature (confirmed via 30-paper Landscape Survey).

---

## Complexity Budget

| Component | Status | Params | Communication/round |
|-----------|--------|--------|---------------------|
| Encoder (AlexNet) | Frozen reused | ~60M | FedAvg (same as before) |
| sem_head / sty_head | Frozen reused | ~256K each | FedAvg |
| sem_classifier | Frozen reused | ~1K (K classes × 128) | FedAvg |
| Pooled whitening broadcast | Frozen reused | N/A (buffer) | ~66KB/round |
| L_orth + HSIC | Frozen reused | - | - |
| FedBN local BN | Frozen reused | ~1K | Local (not aggregated) |
| **NEW: dom_head** | **New trainable** | **~9K** | **~36KB/round** |
| **NEW: GRL** | **New module** | **0** | - |
| **NEW: Frozen probes** | **Eval-only** | ~0.5K each | Local post-hoc |

**Total new trainable**: 1 component (~9K params, 36KB/round comm), meets `MAX_NEW_TRAINABLE_COMPONENTS ≤ 2` budget.

---

## Failure Modes & Mitigation

| Mode | Detection | Mitigation |
|------|-----------|------------|
| FM1: Adversarial training divergence (classical DANN issue) | L_task not decreasing after R20; z_sem_norm collapsing to < 1 | λ_adv schedule (R0-20 off); grad clip at 10 |
| FM2: dom_head underfitting | dom_head training acc < 0.3 | Dropout(0.1) + FedAvg aggregation (cross-client signal) |
| FM3: z_sty over-absorbs class info (λ_adv too large) | probe_sty_class > 0.8 AND sem_classifier drops | Reduce λ_adv to 0.5 (do NOT add HSIC(z_sty, y)=0, conflicts anchor) |
| FM4: Office doesn't benefit from CDANN | Office CDANN < Linear+whitening baseline | Expected: scope discipline limits to PACS-like settings; Office parity (not strict gain) is acceptable |

---

## Compute Budget

| Experiment | Config | GPU·h (single 4090) |
|-----------|--------|---------------------|
| Pilot (Office + PACS R100 × 1 seed) | Quick signal | 2.5 |
| C-main (Office + PACS R200 × 3 seeds) | Primary claim | 12 |
| C-ablate (PACS R200 × 3 variants × 2 seeds) | Mechanism decomposition | 24 |
| C-port (PACS R100 × 1 seed with DINOv2) | Appendix sanity | 2 |
| Probes (4 configs × 30 min) | Frozen eval | 2 |
| **Total** | | **~42 GPU·h** |

---

## Implementation Plan

### Code changes (`FDSE_CVPR25/algorithm/feddsa_sgpa.py`)

1. Add `GradientReverseLayer(torch.autograd.Function)` (~15 lines)
2. Add `dom_head` MLP in `FedDSASGPAModel.__init__` (~10 lines)
3. Add `L_dom_sem` + `L_dom_sty` computation in `Client.train` (~20 lines)
4. Update `aggregate_keys` to include `dom_head.*` (~3 lines)
5. Add algo_para flag `ca` (cdann active, 0/1, default 0) for backward compat (~5 lines)
6. Update FedAvg server-side aggregation to route `dom_head` params (~5 lines)

### Evaluation script (new `scripts/run_frozen_probes.py`)

1. Load checkpoint (reuse EXP-105 `se=1` mechanism)
2. Aggregate features from train/test loaders per client
3. Fit 3 probes (sklearn LogisticRegression)
4. Report held-out test accuracies

### Unit tests (`tests/test_feddsa_sgpa.py` addition)

1. TestGRL: forward identity, backward sign-reversed
2. TestDomHead: shape, parameter count
3. TestLAdvPaths: z_sem gradient reversed, z_sty gradient not reversed (via hooks)
4. TestLambdaSchedule: ramp-up curve
5. TestShouldAggregate: dom_head in FedAvg white list
6. TestCaFlag: ca=0 equivalent to baseline (regression test)

### Codex code review (before deployment)

- Full codex review on ~50-line changes
- Fix all CRITICAL/IMPORTANT issues before commit

---

## Known Limitations (Honest Disclosure)

1. **Novelty ceiling is intrinsic** (5 rounds Codex reviewer verdict): strong reviewer will read as "clean asymmetric DANN repair", not new method family. Paper upside requires empirical overperformance.
2. **Scope is narrow**: applicable only to `client=domain` FedDG with style-carries-class-signal property. Extending to DomainNet (multi-domain-per-client) requires cluster pseudo-labels (future work).
3. **No theoretical convergence proof**: empirical-only validation. Formal causal proof of "whitening erased → CDANN preserved" requires counterfactual analysis, not included.
4. **DANN training instability risk**: depends on λ_adv schedule tuning; mitigated but not eliminated.

---

## References

- **Upstream**: [FINAL_PROPOSAL.md](../../refine-logs/2026-04-20_FedDSA-CDANN/FINAL_PROPOSAL.md) + [REFINEMENT_REPORT.md](../../refine-logs/2026-04-20_FedDSA-CDANN/REFINEMENT_REPORT.md)
- **Literature grounding**: [LITERATURE_SURVEY_DANN_AND_DECOUPLING.md](../../LITERATURE_SURVEY_DANN_AND_DECOUPLING.md)
- **Diagnostic evidence**: [诊断指标分析_数据集边界证据.md](../2026-04-20/诊断指标分析_数据集边界证据.md)
- **Codebase**: `FDSE_CVPR25/algorithm/feddsa_sgpa.py`
