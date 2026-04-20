# EXP-108 FedDSA-CDANN 完整技术流程 (学术版)

**对应大白话版**: [EXP-108_CDANN流程_大白话版](EXP-108_CDANN流程_大白话版.md)
**方案**: Constrained Dual-Directional DANN for Federated Domain Generalization
**状态**: Phase 0-6 完成, Phase 7 部署待启动
**Codex proposal score**: 8.75/10 (5 rounds, proposal-complete, near review-time ceiling)
**Codex code review**: REVISE → all fixed → APPROVED (via 55/55 tests)

---

## 1. Problem Formulation

### 1.1 Observed Failure Mode

在 `client=domain` Federated Domain Generalization (FedDG) 场景下, FedDSA-SGPA (Linear classifier + pooled whitening) 表现出**数据集依赖**的不一致行为:

| Dataset | AVG Best | vs Plan A baseline | z_sty_norm R10→R200 | 解读 |
|---------|----------|-------------------|---------------------|------|
| Office-Caltech10 | 88.75 | **+6.20pp** 🔥 | 3.12 → 2.21 (-2%) | 风格信号弱, whitening 有益 |
| PACS_c4 | 80.20 | **-1.49pp** ⚠️ | 3.12 → 0.15 (**-95%**) | 风格信号强, whitening 误擦 |

### 1.2 Root Cause Hypothesis

FedDSA-SGPA 的解耦损失 `L_dec = λ_orth · cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)` 只要求**统计独立**, 无方向性监督. 当数据集中的 domain shift 以 "style carries class signal" 形式出现时:

1. Class-discriminative 特征可能被 route 到 `z_sty` 分支 (因为 cos²=0 只要求正交, 不指定谁包含类信息)
2. Server-side `Σ_inv_sqrt` broadcast 无差别归一化, 将跨 client z_sty 的差异磨平
3. 最终 `z_sty_norm` 塌缩 → class signal 同时被擦除

### 1.3 Scope Disclaimer

本方案仅适用于:
1. **client=domain bijection** (Office-Caltech10 / PACS_c4, 每 client 一个域)
2. **style carries class signal** (可由 `probe_sty_class` ≫ random 验证)

多域/client (DomainNet) 需要 domain cluster pseudo-labels, 留作 future work.

---

## 2. Method Thesis (Locked One-Sentence Novelty)

> **"A shared non-adversarial domain discriminator plus asymmetric encoder-gradient supervision (GRL on z_sem path only) is the minimal repair for whitening-induced style collapse in client=domain FedDG when style carries class signal."**

---

## 3. System Architecture

```
                                                     ┌── sem_classifier(z_sem) ─── L_task = CE(y)       [encoder forward pass]
                                                     │
  x ─ encoder ─ feat ──┬── sem_head → z_sem ─ (WH) ──┤
                       │                              └── dom_head(GRL(z_sem, λ)) ─── L_dom_sem          [dom_head forward CE; encoder via GRL: reversed grad]
                       │
                       └── sty_head → z_sty ─ (WH) ──── dom_head(z_sty) ─── L_dom_sty                    [dom_head forward CE; encoder: forward grad]

  Decoupling: L_dec = λ_orth · cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)

  Inference: argmax sem_classifier(z_sem). z_sty 仅用于 downstream pooled style bank broadcast (保持 Plan A).
```

## 4. Core Mechanism (Precise Formulation)

### 4.1 Components

| Component | Status | Params | Notes |
|-----------|--------|--------|-------|
| Encoder (AlexNet) | Frozen reused | ~60M | FedAvg aggregated |
| sem_head / sty_head | Frozen reused | ~256K each | FedAvg |
| sem_classifier | Frozen reused | ~1K (128×K) | FedAvg |
| Pooled whitening | Frozen reused | - (buffer) | 66KB/round broadcast |
| L_orth + HSIC | Frozen reused | - | λ_orth=1.0, λ_hsic=0.1 |
| FedBN (local BN) | Frozen reused | ~1K | Not aggregated |
| **[NEW] dom_head** | **New trainable** | **~9K** | **FedAvg aggregated** |
| **[NEW] GRL** | **New layer** | **0** | No params |
| **[EVAL] 3 probes** | Post-hoc | ~0.5K each | Train on train, test on held-out |

### 4.2 Gradient Reversal Layer

```python
class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = float(lam)
        return x.view_as(x)  # identity in forward

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None  # reverse gradient by -lam
```

### 4.3 Shared Non-Adversarial dom_head

```python
self.dom_head = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(inplace=True),
    nn.Dropout(0.1),
    nn.Linear(64, num_clients),  # N_clients = 4 for Office/PACS
)
```

**Key property**: dom_head 自身目标是 **standard CE on both branches** (non-adversarial). Asymmetry 来自 encoder 端的 upstream path gradient direction.

### 4.4 Loss Functions

$$
\mathcal{L}_{task} = \mathrm{CE}(y, \text{sem\_classifier}(\tilde{\mathbf{z}}_{sem}))
$$

$$
\mathcal{L}_{dec} = \lambda_{orth} \cdot \cos^2(\tilde{\mathbf{z}}_{sem}, \tilde{\mathbf{z}}_{sty}) + \lambda_{hsic} \cdot \mathrm{HSIC}(\tilde{\mathbf{z}}_{sem}, \tilde{\mathbf{z}}_{sty})
$$

$$
\mathcal{L}_{dom\_sem} = \mathrm{CE}(d, \text{dom\_head}(\mathrm{GRL}_{\lambda_{adv}}(\tilde{\mathbf{z}}_{sem})))
$$

$$
\mathcal{L}_{dom\_sty} = \mathrm{CE}(d, \text{dom\_head}(\tilde{\mathbf{z}}_{sty}))
$$

$$
\boxed{\mathcal{L}_{total} = \mathcal{L}_{task} + \mathcal{L}_{dec} + \mathcal{L}_{dom\_sem} + \mathcal{L}_{dom\_sty}}
$$

### 4.5 λ_adv 3-Stage Schedule (Codex R4 locked)

```
CDANN_WARMUP_START = 20
CDANN_WARMUP_END = 40

λ_adv(r) = 0.0                         if r < 20  (CDANN completely disabled, baseline-equivalent)
         = (r - 20) / 20               if 20 ≤ r < 40  (linear ramp 0→1)
         = 1.0                         if r ≥ 40  (full strength)
```

**Codex CRITICAL fix**: warmup 阶段**完全跳过** L_dom_sem + L_dom_sty 计算 (不只是 λ_adv=0), 否则 dom_head + style_head 仍被 L_dom_sty 更新, 与 baseline 不等价.

```python
cdann_active = (ca_flag == 1 
                and model.dom_head is not None 
                and self.current_round >= CDANN_WARMUP_START)
if cdann_active:
    # compute L_dom_sem + L_dom_sty
    ...
    loss = loss + L_dom_sem + L_dom_sty
else:
    # CDANN 完全禁用, 严格等价 baseline
    pass
```

### 4.6 Aggregation Strategy

- `dom_head` 参与 **FedAvg** (和 encoder / sem_classifier 一致)
- **Rationale**: local client 只见一个 domain, 本地 dom_head 退化 (constant classifier). 必须 aggregate 才能判别 N domains, GRL 才有信号
- **Privacy**: 聚合参数, 不是 data; client id 本就是 FL header 公开信息

---

## 5. Experimental Setup

### 5.1 Configuration Alignment (严格对齐 baseline, 只改 ca+se+dg)

#### Office CDANN (vs EXP-102)

| Parameter | EXP-102 baseline | EXP-108 CDANN | 说明 |
|-----------|------------------|---------------|------|
| algo_para (13 位) | 10 位 + default | `[1.0, 0.1, 128, 10, 1e-3, 2, 1, 0, 1, 0, 1, 0, 1]` | 改 3 位 |
| `dg` (diag) | 0 | **1** | 用于诊断 jsonl |
| `ue` (use_etf) | 0 (Linear) | 0 (Linear) | 同 |
| `uw` (use_whitening) | 1 | 1 | 保留 +6.20pp 基础 |
| `uc` (use_centers) | 0 | 0 | 同 |
| `se` (save_endpoint) | 0 | **1** | 保存 checkpoint 供 probe |
| `ca` (CDANN) | - | **1** | CDANN ON |
| `lp` (lambda_etf_pull) | 0 | 0 | 同 |
| R / E / LR | 200 / 1 / 0.05 | 同 | |
| λ_orth / λ_hsic | 1.0 / 0.1 | 同 | |
| Seeds | {2, 15, 333} | 同 | 严格对齐 |

Config: `FDSE_CVPR25/config/office/feddsa_cdann_office_r200.yml`

#### PACS CDANN (vs EXP-098 Linear)

| Parameter | EXP-098 Linear | EXP-108 CDANN |
|-----------|----------------|---------------|
| R / E / LR | 200 / **5** / 0.05 | 同 (PACS 惯例 E=5) |
| `uc` | 1 (implicit) | 1 (严格对齐) |
| 其他 | 同 Office CDANN | 同 |

Config: `FDSE_CVPR25/config/pacs/feddsa_cdann_pacs_r200.yml`

### 5.2 Implementation Plan

| File | Change | Lines | Purpose |
|------|--------|-------|---------|
| `feddsa_sgpa.py` | **+120 lines** | Core | GRL + dom_head + L_dom + warmup gate + dynamic rebuild |
| `test_feddsa_sgpa.py` | **+14 tests** | Test | 55/55 green (GRL × 4, λ_adv × 4, CDANNModel × 5, backward compat × 1) |
| `run_frozen_probes.py` | New | Script | Post-hoc probe on frozen model |
| `feddsa_cdann_office_r200.yml` | New | Config | Office CDANN run |
| `feddsa_cdann_pacs_r200.yml` | New | Config | PACS CDANN run |

---

## 6. Diagnostic Framework (Two Layers)

### 6.1 Training-Time Diagnostics (26 metrics)

**Switch**: `algo_para[6] = dg = 1` → activates `SGPADiagnosticLogger`
**Output**: `task/<TASK>/diag_logs/R200_S<seed>_cdann/{diag_train_client{0,1,2,3}.jsonl, diag_aggregate_client-1.jsonl}`
**Variant label**: 自动区分 `_etf` / `_linear` / `_cdann` 避免 jsonl 污染

#### Layer 1 — Feature Geometry (Client-side, 每 5 round)

| Metric | Definition | Expected Pattern |
|--------|-----------|------------------|
| `orth` | `mean(cos²(z_sem, z_sty))` | → 0 (independence) |
| `etf_align_mean` | mean class-wise `z_sem · M_y / \|z_sem\|` | ≈ 0 (Linear mode) |
| `intra_cls_sim` | intra-class cosine mean | → 1.0 (tight within-class) |
| `inter_cls_sim` | inter-class cosine mean | → -1/(K-1) (max separation) |
| `loss_task` | CE(y, sem_classifier) | → 0 |
| `loss_orth` | L_orth value | monitor |
| **`z_sem_norm_{mean, std, min, max}`** | ‖z_sem‖ statistics | CDANN 不应 collapse z_sem |
| **`z_sty_norm_{mean, std, min, max}`** | ‖z_sty‖ statistics | **PACS CDANN R200 ≥ 1.5** (baseline 0.15) |

#### Layer 2 — Server-Side Aggregation (每 round)

| Metric | Definition | Notes |
|--------|-----------|-------|
| `client_center_var` | variance of class centers across clients | cross-client consistency |
| `param_drift` | ‖θ_t - θ_{t-1}‖² / ‖θ_{t-1}‖² | training stability |
| `n_valid_classes` | active classes in aggregation | data balance |

#### 🆕 CDANN-Specific Metrics (5, ca=1 时写入)

| Metric | Definition | Expected Trajectory | Diagnostic Claim |
|--------|-----------|---------------------|------------------|
| `lambda_adv` | GRL coefficient | R<20=0, 20-40 ramp, ≥40=1.0 | Schedule executed correctly |
| `loss_dom_sem` | CE(d, dom_head(GRL(z_sem))) | R<20=0 (gated), R40+ ↑ ≈ log(N_clients) | GRL adversarial pressure effective |
| `loss_dom_sty` | CE(d, dom_head(z_sty)) | R<20=0, R40+ ↓ ≈ 0 | Positive supervision succeeds |
| `dom_sem_acc_train` | `argmax(logits_sem) == d` rate | R40+ → 1/N (random) | z_sem becomes domain-blind |
| `dom_sty_acc_train` | `argmax(logits_sty) == d` rate | R40+ → 100% | z_sty becomes domain-expert |

### 6.2 Frozen Post-Hoc Probes (3 probes, 训练后单独跑)

**Script**: `FDSE_CVPR25/scripts/run_frozen_probes.py`
**Dependency**: Config `se=1` 保存 checkpoint (已改)

#### Protocol (Codex R3 修正, no leakage)

```python
# 1. Load checkpoint: encoder + heads + whitening + client BN states (frozen)
# 2. Collect post-whitening z_sem, z_sty from ALL clients' train/test loaders
#    (whitened features, same space as sem_classifier)
# 3. Train probe ON FROZEN TRAIN FEATURES (sklearn LogisticRegression)
# 4. Report accuracy ON HELD-OUT TEST FEATURES
```

#### Three Probes (all post-whitening features)

| Probe | Input | Target | Expected | Claim Alignment |
|-------|-------|--------|----------|-----------------|
| `probe_sem_domain` | z_sem | domain (client id) | ≈ 1/N_clients = 0.25 | **C-domain**: GRL erases domain from z_sem |
| `probe_sty_domain` | z_sty | domain | ≥ 0.95 | **C-domain**: positive supervision makes z_sty domain-aware |
| **`probe_sty_class`** | **z_sty** | **class (y)** | **PACS ≥ 0.40, Office ~ 0.20-0.30** | **C-probe (anchor evidence)**: z_sty retains class-relevant style |

#### Why 2 Layers

Training-time `dom_sem_acc_train` is the accuracy of the **jointly trained** dom_head, which could overfit. Frozen post-hoc probes with independent LogisticRegression trained on held-out features provide **clean representation-level evidence** independent of the training objective.

---

## 7. Claims and Validation

### 7.1 Primary Claim (C-main)

**Statement**: CDANN fixes PACS regression without hurting Office parity.

**Experiment**: PACS R200 × 3 seeds + Office R200 × 3 seeds vs Linear+whitening baseline

**Metrics**: AVG Best, AVG Last, per-domain Best/Last

**Success Criterion**:
- PACS 3-seed mean AVG Best ≥ 82.2 (recover from -1.49pp to ≥0 vs Plan A 81.69)
- Office 3-seed mean AVG Best ≥ 88.0 (maintain baseline 88.75 ± 0.75)

### 7.2 Supporting Claim (C-probe)

**Statement**: z_sty retains class-relevant signal on PACS; magnitude consistent with anchor claim.

**Experiment**: `run_frozen_probes.py` on trained checkpoints

**Success Criterion**:
- PACS `probe_sty_class` test accuracy ≥ 0.40 (vs baseline ≈ 0.15)
- Gap ≥ 25pp directly supports "CDANN preserved class-relevant style that whitening otherwise erased"

### 7.3 Mechanism Claim (C-domain)

**Statement**: GRL effectively removes domain from z_sem; positive supervision makes z_sty domain-discriminative.

**Success Criterion**:
- `probe_sem_domain` test accuracy ≈ 1/N = 0.25 (random level)
- `probe_sty_domain` test accuracy ≥ 0.95

### 7.4 Ablation Claim (C-ablate, GPU 余量做)

**Statement**: Positive supervision on z_sty (L_dom_sty) is necessary; single-direction DANN insufficient.

**Experiment**: PACS R200 × 2 seeds × 3 variants
- V1 baseline: Linear+whitening (EXP-098 已有)
- V2 z_sem-only: L_dom_sem only (standard DANN)
- V3 full CDANN: both L_dom_sem + L_dom_sty (ours)

**Success Criterion**: V3 > V2 ≥ V1 on both AVG Best AND `probe_sty_class`

---

## 8. Failure Modes and Diagnostics

| Failure Mode | Detection (metrics) | Mitigation |
|--------------|---------------------|------------|
| FM1: Adversarial divergence (classical DANN) | `loss_task` not decreasing after R20; `z_sem_norm` → 0 | λ_adv schedule already has R<20 warmup; add grad clip = 10 (config) |
| FM2: dom_head underfitting | `dom_head acc < 30%` at R50 | Already: Dropout(0.1) + FedAvg cross-client aggregation |
| FM3: z_sty over-absorbs class signal | `probe_sty_class` > 0.8 AND sem_classifier accuracy drops | Reduce λ_adv to 0.5 (NOT add HSIC(z_sty, y)=0, conflicts anchor) |
| FM4: Office without CDANN benefit | Office CDANN < Linear+whitening | Expected within scope; parity (not strict gain) is acceptable per R4 positioning |

---

## 9. Compute and Timeline

### 9.1 Compute Budget

| Experiment | Config | GPU·h (single 4090) |
|-----------|--------|---------------------|
| C-main Full | Office + PACS × 3 seeds × R200 | 12 |
| C-ablate V2 (z_sem-only) | PACS × 2 seeds × R200 (GPU 余量) | 6 |
| Frozen probes | 6 ckpt × ~5 min each | 0.5 |
| **Total planned** | | **~18.5** |

### 9.2 Timeline

- Day 0 (2026-04-20): Phase 0-6 完成 (proposal + 代码 + 测试 + codex review)
- Day 0-1 evening/overnight: Phase 7 部署 C-main 6 runs (~12h)
- Day 1 (2026-04-21) AM: 检查结果, 跑 probes (30 min)
- Day 1 PM: 回填 NOTE + Obsidian, 如 C-main 成功则部署 C-ablate (6h)
- Day 2: C-ablate 完成, 综合 Obsidian summary + 已做实验总览 update

---

## 10. Codex Review History

### 10.1 Proposal Refine (5 rounds, research-refine skill)

| Round | Overall | Verdict | Key Issue → Fix |
|-------|---------|---------|-----------------|
| R1 | 7.1 | REVISE | Contribution sprawl → merged 2 heads to 1 shared; narrowed scope |
| R2 | 8.35 | REVISE | Mechanism overstated → precise reframing (non-adversarial head + encoder gradient asymmetry) |
| R3 | 8.4 | REVISE | Probe leak + missing class probe → train/test split + probe_sty_class added |
| R4 | 8.75 | REVISE | Evidence framing → "consistent with" not "formal proof"; post-whitening explicit; one-sentence locked |
| R5 | 8.75 | REVISE (near ceiling) | Intrinsic novelty ceiling, not fixable framing — accept, move to execution |

Full history: `refine-logs/2026-04-20_FedDSA-CDANN/REVIEW_SUMMARY.md`

### 10.2 Code Review (single round)

| Level | Issue | Fix |
|-------|-------|-----|
| CRITICAL | warmup 非 baseline-equivalent (R<20 仍跑 L_dom_sty) | Added explicit `cdann_active` gate; R<20 skip both L_dom_* entirely |
| IMPORTANT | `_TASK_NUM_CLIENTS` heuristic vs runtime mismatch | Server.initialize dynamically rebuilds dom_head to `len(clients)` |
| IMPORTANT | `ca=1 + ue=1` conflict unchecked | raise ValueError with clear message |
| IMPORTANT | `client.id` bound not validated | assert `cid < out_dim` in Client.train |
| MINOR | `c.num_clients_total` dead state | Removed |

All fixed, 55/55 tests still green.

---

## 11. Novelty Analysis

### 11.1 Delta vs Closest Prior Work

| Prior | Relation to ours | Delta |
|-------|------------------|-------|
| Deep Feature Disentanglement SCL (Cog. Comp. 2025) | Non-FL, symmetric dual-head positive supervision | **FL + asymmetric encoder-gradient + shared non-adversarial head** |
| FedPall (ICCV 2025) | Adversarial Amplifier in mixed feature space, erase domain | **Decoupled space, preserve z_sty domain info** |
| ADCOL (ICML 2023) | DANN in FL, erase all client differences | **Asymmetric preserve (selective retention on z_sty)** |
| Federated Adversarial DA (2019) | Vanilla DANN in FL | **Encoder-gradient asymmetry + shared non-adversarial discriminator** |
| FediOS (ML 2025) | Orthogonal generic/personalized subspaces | **Domain-supervised asymmetric gradient (not mere orthogonality)** |
| FDSE (CVPR 2025) | Layer-decomposition domain erasure | **Feature decoupling with style preservation** |
| FedSeProto (ECAI 2024) / FedDP (TMC 2025) | MI minimization, discard domain features | **Preserve via positive supervision on z_sty** |

### 11.2 Anti-Pseudo-Novelty Defense

Precise framing emphasizes three inseparable points:
1. **dom_head is non-adversarial** (both branches minimize standard CE, not GAN-style game)
2. **Asymmetry is encoder-side** (GRL only on z_sem → encoder; z_sty → encoder is identity)
3. **Mechanism minimality** (1 MLP, 9K params, no aggregation / whitening / classifier changes)

This combination is not explicitly formulated in 30-paper 2024-2026 Landscape Survey.

---

## 12. Known Limitations (Honest Disclosure)

1. **Novelty ceiling intrinsic** (R5 reviewer verdict): Strong reviewer may read as "clean asymmetric DANN repair" rather than new method family. Empirical overperformance is the upgrade path.
2. **Narrow scope**: Applicable only to `client=domain` FedDG with style-carries-class-signal property. DomainNet extension requires future work on cluster pseudo-labels.
3. **No theoretical convergence proof**: Empirical-only. Formal causal proof of "whitening erased → CDANN preserved" requires counterfactual analysis.
4. **DANN instability risk**: λ_adv schedule tuning dependent; mitigated via 3-stage warmup but not eliminated.

---

## 13. Related Documents

### Proposal Stage
- Final proposal: `refine-logs/2026-04-20_FedDSA-CDANN/FINAL_PROPOSAL.md`
- Review summary (5 rounds): `refine-logs/2026-04-20_FedDSA-CDANN/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/2026-04-20_FedDSA-CDANN/REFINEMENT_REPORT.md`
- Score evolution: `refine-logs/2026-04-20_FedDSA-CDANN/score-history.md`

### Literature Stage
- Landscape survey (30 papers): `LITERATURE_SURVEY_DANN_AND_DECOUPLING.md`
- Idea report (12 candidates): `IDEA_REPORT_2026-04-20.md`

### Implementation Stage
- Algorithm: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (+120 lines)
- Tests: `FDSE_CVPR25/tests/test_feddsa_sgpa.py` (55/55 green)
- Probe script: `FDSE_CVPR25/scripts/run_frozen_probes.py`
- Configs: `FDSE_CVPR25/config/{office,pacs}/feddsa_cdann_*_r200.yml`
- Codex code review: `refine-logs/2026-04-20_FedDSA-CDANN/codex-code-review-result.md`

### Experiment Stage
- Main NOTE: `experiments/ablation/EXP-108_cdann_office_pacs_r200/NOTE.md`
- Experiment flow: `experiments/ablation/EXP-108_cdann_office_pacs_r200/EXPERIMENT_FLOW.md`
- Deploy script: `rr/deploy_exp108_cdann.sh`

### Knowledge Notes
- 大白话版: [EXP-108_CDANN流程_大白话版](EXP-108_CDANN流程_大白话版.md)
- 学术方案版: `obsidian_exprtiment_results/知识笔记/FedDSA-CDANN_技术方案.md`
- 大白话方案版: `obsidian_exprtiment_results/知识笔记/大白话_FedDSA-CDANN.md`

### Baseline Comparisons
- Office baseline (EXP-102 Linear+whitening): AVG 88.75, z_sty_norm R200 = 2.21
- PACS baseline (EXP-098 Linear+whitening): AVG 80.20, z_sty_norm R200 = 0.15
- Office Plan A (EXP-083 orth_only): AVG 82.55
- PACS Plan A (EXP-080 orth_only): AVG 81.69

---

## 14. Current Status (2026-04-20)

**Phase Progress**:
- ✅ Phase 0: Problem identification via diagnostic analysis
- ✅ Phase 1: Literature landscape (30 papers)
- ✅ Phase 2: Idea brainstorm (12 candidates, top-3 novelty-checked)
- ✅ Phase 3: Novelty verification (zero direct prior)
- ✅ Phase 4: 5-round Codex research-refine (7.1 → 8.75, proposal-complete)
- ✅ Phase 5: Code implementation (+120 lines, 14 new tests)
- ✅ Phase 6: Single-round Codex code review (REVISE → all fixed)
- ⏳ **Phase 7: Deploy 6 runs (PENDING, ~12h on seetacloud2 4090)**
- ⏳ Phase 8: Results backfill + frozen probes + Obsidian sync

**Next action**: Launch deploy via `wsl bash -lc 'ssh seetacloud2 bash < rr/deploy_exp108_cdann.sh'`
