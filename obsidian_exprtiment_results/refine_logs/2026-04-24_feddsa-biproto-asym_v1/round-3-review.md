# Round 3 Review (GPT-5.4 xhigh, resume thread)

**Date**: 2026-04-24
**Session**: 019dc056-e5fa-7fd2-8721-994c42c8b3ca

## Quick Calls (reviewer 总评)

| Item | Status |
|---|:-:|
| Problem Anchor | ✅ Preserved |
| Dominant Contribution | ✅ Sharp enough |
| Headline/Impl Alignment | ✅ Largely resolved (hybrid ST) |
| C0 Gate | ✅ Valid as pruning gate |
| Method Size | ✅ No longer overbuilt |
| Frontier Leverage | ✅ Appropriate |

## Score Table

| Dimension | R2 | R3 | Δ | Reason |
|---|:-:|:-:|:-:|---|
| Problem Fidelity | 9 | **9** | 0 | 维持 |
| Method Specificity | 8 | **8** | 0 | 剩 estimator/normalization 细节 |
| Contribution Quality | 7 | **8** | +1 | Pc 降级 + Pd dominant 后 sharper |
| Frontier Leverage | 9 | **9** | 0 | 维持 |
| Feasibility | 7 | **8** | +1 | stage-gate + 简化后 realistic |
| Validation Focus | 6 | **8** | +2 | C0 fix + 3-suite visual 合理 |
| Venue Readiness | 7 | **7** | 0 | D=K=4 约束 + ST 需要更 principled 解释, 硬天花板 |

## OVERALL: 8.25 / 10 — **REVISE** (距 READY 9.0 还差 0.75)

## Drift Warning: NONE

## 剩余 5 个 Action Items (全是细节级)

### [AI-1] ST axis normalization point 精化
- **Reviewer**: `domain_axis = normalize(Pd.detach() + bc - bc.detach())` 还是 normalize Pd/bc separately 再 ST 组合? 影响 gradient scale 和 stability
- **Fix**: 选定具体 variant + 理由

### [AI-2] L_proto_excl averaging set 明确
- **Reviewer**: 若 absent class 用 Pc[c].detach(), 这些 term 只推 encoder_sty 不推 encoder_sem. 这是 intentional 还是 bug?
- **Fix**: Pilot 只用 present_classes (不做 fallback); 附录留 full-class 变体比较

### [AI-3] Federated claim narrow/defensible
- **Reviewer**: D=K=4 下 "federated domain prototype" 概念 valid 但实证弱. "to our knowledge" 语气 + benchmark scope
- **Fix**: Paper 中明确 limitation 小节 "empirical validation scoped to D=K=4, generalization to D<K / D>K as future work"

### [AI-4] -Pd 升为 MANDATORY ablation
- **Reviewer**: 因为 benchmark 下 domain=client, -Pd (换 batch-local-only 作 domain axis, 无 federated EMA) 是真正测"paper 是 federated object 还是 local domain axis + server buffer"的唯一方式
- **Fix**: 消融表 -Pd 从 "optional" 升为 "MANDATORY"

### [AI-5] C0 role 精确措辞
- **Reviewer**: "C0 是 fast intervention pruning test, 不是 end-to-end causal proof. 要把 role 写精确避免 oversell"
- **Fix**: C0 描述明确 "快速排除 add-on 是否有机会 move Office, 不保证 R200 full run 同原因有效"

## Simplification Opportunities (reviewer 接受)
1. Pilot 只用 present_classes for L_proto_excl, 不做 fallback (简化 loss)
2. -encoder_sty 从 core ablation 移至 optional (核心是 -Pd 和 -L_proto_excl)
3. L_sty_norm_reg 作为 contingency, 不上 paper

## Verdict
REVISE — 方向已充分锁定, 剩下的都是 estimator 语义和 claim 措辞的细修. R4 修完这 5 条 AI 有机会突破 9.0 READY bar.

<details>
<summary>Raw</summary>
See round-3-review-raw.txt.
</details>
