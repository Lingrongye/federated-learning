# Round 2 Review (GPT-5.4 xhigh, resume thread)

**Date**: 2026-04-24
**Session**: 019dc056-e5fa-7fd2-8721-994c42c8b3ca (resume)

## Score Table

| Dimension | R1 | R2 | Δ | Reason |
|---|:-:|:-:|:-:|---|
| Problem Fidelity | 8 | **9** | +1 | no drift, 9 是 reviewer 给的 ceiling |
| Method Specificity | 6 | **8** | +2 | gradient-flow table + EMA equations + shapes 全到位 |
| Contribution Quality | 6 | **7** | +1 | 单 headline 但 "dual" 叙事被反对 |
| Frontier Leverage | 8 | **9** | +1 | 明确接受 "appropriate conservatism" |
| Feasibility | 5 | **7** | +2 | stage-gate 预算通过, 但 exclusion signal noise 风险 |
| Validation Focus | 6 | **6** | 0 | C0 gate 被判定 invalid, 分数没涨 |
| Venue Readiness | 6 | **7** | +1 | sharper 但未到 top-venue-ready bar |

## OVERALL: 7.8 / 10 — **REVISE** (距离 READY 9.0 还差 1.2)

## 剩余关键问题 (3 个)

### [CRITICAL 1] C0 Gate 目前无效
- **Reviewer 原话**: "If you freeze encoder_sem + semantic_head + sem_classifier, then training encoder_sty + Pd + L_proto_excl cannot change predictions, because the inference path is frozen. This gate cannot measure the claimed effect."
- **Fix**: **只冻 encoder_sem**, 保持 semantic_head 可训 — 这样 L_proto_excl 能通过 on-the-fly batch class centroid 推到 semantic_head, prediction path 可以动. Baseline 是 matched head-only fine-tune

### [CRITICAL 2] Headline / 实现错配
- **Reviewer 原话**: "The proposal says the key mechanism is federated Pd with proto-level exclusion, but L_proto_excl currently uses batch-local domain centroids, not Pd. That weakens the central claim."
- **Fix**: L_proto_excl 采用 **hybrid axis**: `stopgrad(Pd[d]) + (batch_centroid_d - stopgrad(batch_centroid_d))` — 前向 anchor 是 federated Pd, gradient 仍能流到 encoder_sty via detach-reparametrization trick. 这样 headline 和 implementation 对齐

### [IMPORTANT 3] Pseudo-duality 风险
- **Reviewer 原话**: "Pd is a real federated training object; Pc is mostly an EMA monitor/reference. Calling them fully 'dual' overstates symmetry."
- **Fix**: 收紧 claim language — "federated domain prototypes excluded against **online class centroids** (+ Pc as monitor)" 而非 "dual prototype bank". Novelty table 里 Pc 从 "class+domain dual bank" 降为 "domain bank with class reference"

## IMPORTANT Fixes

### [IMPORTANT 4] Sparse-batch rules
- 单 sample class centroid 在 Office (E=1, batch=50, 10 classes) 可能频繁出现 singleton, 注入 variance 不是 geometry
- **Fix**: min class count per batch ≥ 2 skip rule; z_sem 明确取 semantic_head 输出 L2-normalized 后的 128d; class centroid on-the-fly 但若该 class 无 ≥2 sample, fallback 到 EMA Pc[c] 那个 class 的前值 (stopgrad)

### [IMPORTANT 5] Claim language tightening
- PACS/Office D=K=4 下 Pd concat 和 client bank 不可区分, 不能在 paper 里 oversell 实现上的 "domain-indexed separation"
- **Fix**: 明确 scope — "for setups where client-domain mapping is 1-1 or known, Pd degenerates to a per-client bank; the domain-indexed formulation generalizes to multi-client-per-domain settings"

## Simplification Opportunities (reviewer 全接受)
1. 若 Pc 只做 monitoring, 从 novelty table 和 headline 中移除, claim 变为 "federated domain prototypes excluded against online class centroids"
2. 合并 "global Pd" 和 "local domain centroid" 为一个 hybrid axis, 不维护两套语义
3. τ sweep 降级 — 从 core ablation 移到 fallback, 仅在 full method 通过 Office 之后跑

## Modernization Opportunities
NONE (确认)

## Drift Warning
NONE on research problem

## Verdict
**REVISE** — 方案相比 v1 已显著改进 (6.5 → 7.8), 有 real center, 但 headline 和 training path 必须对齐, C0 gate 必须可测. Round 3 这三条 CRITICAL + 2 IMPORTANT 修完有机会上 9.0 READY bar.

<details>
<summary>Raw reviewer output (link)</summary>

See `round-2-review-raw.txt`.

</details>
