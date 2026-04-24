# Round 3 Refinement — FedDSA-BiProto v4

**Date**: 2026-04-24
**Based on**: Round 3 review (8.25/10 REVISE, 5 细节级 AI)

## Problem Anchor (不变)

Bottom-line / bottleneck / non-goals / constraints / success — 全部逐字继承自 round-0.

## Anchor Check
- Anchor 不变, 修订全是 estimator 语义和 claim 措辞
- Reviewer 没要求任何 drift

## Simplicity Check
- Dominant 单一 headline (federated Pd + hybrid ST exclusion) 已稳
- Pc 已降级为 monitor, 无需再降
- v4 修订只做 "collapse Pilot loss to present_classes only" 进一步简化

## Changes Made (v3 → v4)

### 1. ST axis normalization variant 选定 (AI-1)
- **Reviewer**: 需要明确 normalization point 影响 gradient scale
- **Decision**: 采用 **final-renormalize** 方案:
  ```python
  bc_d = F.normalize(mean_{i: domain_i==d} z_sty_i, dim=-1)  # batch centroid, has grad, already L2-norm
  # 组合 (Pd already L2-norm, bc also L2-norm)
  raw_axis = Pd[d].detach() + bc_d - bc_d.detach()
  domain_axis[d] = F.normalize(raw_axis, dim=-1)  # final renormalize
  ```
- **Reasoning**:
  - Pd 和 bc_d 都已 L2-norm (见 Pd EMA update 和 batch centroid 定义)
  - Forward value = Pd[d] (因为 bc_d - bc_d.detach() = 0 forward)
  - 浮点误差 + gradient 注入后 `raw_axis` 的 L2 norm 可能 drift, final F.normalize 确保 cosine 计算一致
  - Gradient 通过 F.normalize 的 Jacobian 流回 bc_d → encoder_sty, 在 unit sphere 切线空间上 well-defined
- **Alternative rejected**: 分别 normalize Pd 和 bc 再加会让 forward semantics 不等于 Pd (破坏 headline)
- **Impact**: Method Specificity 上升

### 2. L_proto_excl averaging set = present_classes only (AI-2)
- **Reviewer**: absent class 用 Pc fallback 会让这些 term 只推 sty 不推 sem, 要么明确 intentional, 要么去掉
- **Decision**: **Pilot 阶段只用 present_classes**, 完全删除 absent-class fallback. Pc 回归 pure monitoring:
  ```python
  present_classes = {c : count_in_batch(c) >= 2}
  present_domains = {d : count_in_batch(d) >= 2}
  
  class_axis = {c: F.normalize(mean z_sem_i where y_i==c) for c in present_classes}
  domain_axis = {d: hybrid_st_axis(Pd[d], mean z_sty_i where domain_i==d) for d in present_domains}
  
  # 若 present_classes 为空 (极罕见, batch=50 + 7/10 class 不太可能): skip L_proto_excl this step
  if not present_classes or not present_domains:
      L_proto_excl = 0.0  # no-op
  else:
      L_proto_excl = mean_{c in present_classes, d in present_domains} cos²(class_axis[c], domain_axis[d])
  ```
- **Reasoning**: 
  - Absent-class fallback 引入 second loss behavior (push sty only) + 需要额外解释, 违反 simplicity
  - Office batch=50, K=10 class 下 `P(class c absent) ≈ (9/10)^50 ≈ 0.5%`, 几乎不会发生 fallback
  - PACS batch=50, K=7 class 下 `P(class c absent) ≈ (6/7)^50 ≈ 0.04%` 更不会发生
  - Fallback 可作为附录 "robustness" 变体 (multi-site setups 数据稀疏时可能需要), pilot 不用
- **Impact**: Feasibility + Validation Focus 上升

### 3. Claim language: "to our knowledge" + benchmark-scoped limitations (AI-3)
- **Reviewer**: D=K=4 下 federated object claim 实证弱, 要紧
- **Decision**: 新增 **Limitations** 小节, thesis 和 abstract 加 "to our knowledge"
- **Revised canonical sentence**:
  > *"To our knowledge, this is the first work in federated cross-domain learning to explicitly maintain a **federated domain prototype object** Pd, defined as a server-side EMA across participating clients and used as the forward anchor of prototype-space class-domain exclusion. We empirically validate this on PACS and Office-Caltech10 under a one-client-one-domain setup (D=K=4), where Pd degenerates to a per-client bank. Generalization to D<K (multi-client-per-domain) and D>K (single-client-per-multi-domain) requires additional benchmarks and is left as future work (see §Limitations)."*
- **New Limitations section content**:
  - "Our empirical validation is scoped to D=K=4 benchmarks (PACS, Office-Caltech10). Under this configuration, the domain-indexed formulation of Pd is implementation-equivalent to a per-client prototype bank; the conceptual distinction between 'federated domain object' and 'per-client centroid with server aggregation' cannot be empirically disambiguated in this setup. The −Pd ablation (§C.3) partially addresses this by comparing Pd-anchored hybrid ST axis against batch-local-only domain centroid, but full empirical validation requires D ≠ K benchmarks such as DomainNet with multi-client-per-domain partitioning, left as future work."
  - "The hybrid straight-through axis introduces a mild estimator bias: the forward cosine value is computed against federated Pd[d], but the gradient signal is from the current minibatch's domain centroid. This bias vanishes as Pd and bc_d converge (EMA stabilizes), which we empirically observe post-R100 via Pd ⊥ bc_d cosine monitoring in Vis-C."
- **Impact**: Venue Readiness + Contribution Quality 上升

### 4. −Pd ablation 升为 MANDATORY (AI-4)
- **Reviewer**: 因为 D=K=4, −Pd (batch-local-only domain axis) 是唯一能 empirically 证"真 federated vs local with server buffer"的实验
- **Decision**: 消融表新排序 (MANDATORY 在前):
  ```
  MANDATORY (必须跑, 无论预算):
    - BiProto − Pd    (domain_axis = F.normalize(batch_centroid_d), no ST, no federated EMA)
                       → 测 federated Pd 的实际增量. 若 -Pd 和 full BiProto 差 <0.5pp, 则 claim 弱化为 "local domain axis works, federated aggregation marginal"
  
  RECOMMENDED (budget 充裕跑):
    - BiProto − L_proto_excl  (只 L_sty_proto, 测 exclusion 必要)
    - BiProto − encoder_sty   (换 orth_only 的 style_head, 测 asymmetric 必要)
  
  OPTIONAL (S4 pass 之后):
    - τ sweep {0.05, 0.1, 0.2}
    - MSE coef sweep {0.25, 0.5, 1.0}
  ```
- **Impact**: Validation Focus 上升

### 5. C0 role 精确措辞 (AI-5)
- **Reviewer**: C0 是 pruning test, 不是 causal proof
- **Decision**: C0 描述加精确限定词:
  > "**Claim 0 (C0) is a fast intervention pruning test to determine whether the BiProto add-on (encoder_sty + Pd + L_proto_excl) has any measurable effect on Office accuracy when trained on top of a frozen orth_only encoder_sem with semantic_head/classifier trainable. A positive C0 is a necessary but not sufficient condition for S1-S3 R200 full training to succeed: it shows the add-on can move Office at all, but does not prove that the mechanism continues to dominate under full end-to-end training or at R200 convergence. A negative C0 is sufficient to kill BiProto (no add-on value = no amount of full training can rescue the headline)."**
- **Impact**: Validation Focus 上升

## Revised key sections (v4 精化, only changed parts)

### One-sentence thesis (v4, benchmark-scoped)

**To our knowledge, we are the first to maintain a federated domain prototype object Pd — a server-side EMA centroid across participating clients — and use it as the forward anchor of low-dimensional prototype-space class-domain exclusion via a straight-through hybrid gradient axis, empirically validated on D=K=4 benchmarks (PACS, Office-Caltech10) with full-scope generalization left as future work.**

### Claim 0 (FIXED + role-tightened, v4)

- **Setup**: Load EXP-105 orth_only Office R200 seed=2 checkpoint. Freeze encoder_sem only. semantic_head + sem_classifier trainable. Add encoder_sty + Pd + L_sty_proto + L_proto_excl, train Office R20-R30, seed=2.
- **Baseline**: Same checkpoint, same freeze scope, head-only fine-tune without encoder_sty/Pd/L_proto_excl.
- **Role disclaimer**: C0 is a pruning test, not a causal proof. Positive C0 necessary but not sufficient for full R200 success. Negative C0 sufficient to kill.
- **Decision**:
  - Δ ≥ +1.0pp (BiProto-lite vs head-only) → strong pruning signal → S1
  - Δ +0.3~+1.0pp → weak pruning signal → proceed with tempered expectation
  - Δ < +0.3pp → **kill BiProto**
- **Cost**: 2 GPU-h

### L_proto_excl computation (v4, final form)

```python
# Settings: m=0.9 EMA decay, τ=0.1 InfoNCE
# Per batch:

present_classes = {c : count_in_batch(c) >= 2}
present_domains = {d : count_in_batch(d) >= 2}  # usually 1 single-client = single-domain

if not present_classes or not present_domains:
    L_proto_excl = 0.0  # no-op (extremely rare on Office/PACS batch=50)
else:
    # Class axis: batch-local centroid, on-the-fly, has grad
    class_axis = {c: F.normalize(mean(z_sem[y==c]), dim=-1) for c in present_classes}
    
    # Domain axis: hybrid straight-through with final renormalize
    domain_axis = {}
    for d in present_domains:
        bc_d = F.normalize(mean(z_sty[domain==d]), dim=-1)     # batch centroid, has grad
        raw = Pd[d].detach() + bc_d - bc_d.detach()             # forward = Pd[d], grad via bc_d
        domain_axis[d] = F.normalize(raw, dim=-1)               # final renormalize for cosine
    
    # Exclusion over present pairs
    total = 0.0
    count = 0
    for c in present_classes:
        for d in present_domains:
            total += F.cosine_similarity(class_axis[c].unsqueeze(0), domain_axis[d].unsqueeze(0)).pow(2).squeeze()
            count += 1
    L_proto_excl = total / count
```

### Ablation table (v4, MANDATORY 重排)

| Ablation | Priority | Setup | 测的是什么 |
|---|:-:|---|---|
| **BiProto − Pd** | **MANDATORY** | 改 `domain_axis[d] = F.normalize(bc_d)`, 无 Pd forward anchor, 无 ST | Federated Pd 是否真有增量 (vs local + server buffer) |
| BiProto − L_proto_excl | RECOMMENDED | 删 L_proto_excl, 保留其他全部 | Exclusion 机制必要性 |
| BiProto − encoder_sty | RECOMMENDED | 换 orth_only 的 symmetric style_head | Asymmetric 统计 encoder 必要性 |
| τ sweep | OPTIONAL | τ ∈ {0.05, 0.1, 0.2} | Sensitivity, S4 pass 后跑 |
| MSE coef sweep | OPTIONAL | coef ∈ {0.25, 0.5, 1.0} | Sensitivity |

### Limitations (v4, 新增小节)

> 本文实证 validation 限定于 D=K=4 的 PACS 和 Office-Caltech10 benchmarks. 该 configuration 下 domain-indexed Pd 在实现上退化为 per-client prototype bank, "federated domain object" 和 "per-client centroid with server aggregation" 的概念区别无法 empirically disambiguate. −Pd ablation 通过对比 Pd-anchored hybrid ST axis vs batch-local-only domain centroid 部分回应该问题, 但完整 empirical validation 需要 D ≠ K 的 benchmarks (如 DomainNet multi-client-per-domain partitioning), 留作 future work.
> 
> Hybrid straight-through axis 引入 mild estimator bias: forward cosine 相对 federated Pd[d], gradient 来自 batch-local domain centroid. 该 bias 在 Pd 和 bc_d 收敛后消失 (EMA 稳定), 通过 Vis-C 的 Pd ⊥ bc_d cosine 轨迹监测.

## v3 → v4 Deltas Summary

| 项 | v3 | v4 | 理由 |
|---|---|---|---|
| ST axis normalization | unspecified | **F.normalize(Pd.detach()+bc-bc.detach())** final renormalize | R3 AI-1: stability clarity |
| L_proto_excl absent class | Pc[c] fallback | **pilot: present-classes only, no fallback** | R3 AI-2: simplification |
| Claim language | "first federated..." | **"to our knowledge" + D=K=4 scope + Limitations §** | R3 AI-3: narrow defensible |
| −Pd ablation | optional | **MANDATORY** | R3 AI-4: key test of federated value |
| C0 role wording | gate | **"pruning test, positive necessary but not sufficient; negative sufficient to kill"** | R3 AI-5: avoid oversell |
| Thesis sentence | (prior form) | **benchmark-scoped 版本** | R3 AI-3 派生 |

Method graph, gradient-flow table, loss schedule, FL aggregation, compute plan, novelty table — 全部维持 v3 不变.
