# Round 3 Review — sas-FH Proposal (R3)

**Reviewer**: GPT-5.4 via codex (reasoning=high, same thread 019d9ebd-…)
**Date**: 2026-04-18
**Verdict**: **REVISE** (9.1/10, 距 READY 仅差一档)

## Scores

| Dim | Score | Rationale |
|-----|-------|-----------|
| PF  | 10/10 | 完全保留，框架更清晰。 |
| MS  | 10/10 | Configs/更新规则/checkpoint/整合方式 — 全部无歧义。 |
| CQ  | 9/10  | 单一主导机制，counterfactual 干净。novelty 仍偏窄但可辩护。 |
| FL  | 9/10  | 克制恰当。 |
| Feas| 9/10  | 可信。 |
| VF  | 8/10  | 大幅改善。遗留项：最终论文对 C2 的处理仍需更强。 |
| VR  | 8/10  | 接近 READY。未完全 READY 的原因：在 strong-margin 分支下，关键 novelty baseline 可能验证不足。 |

**Overall: 9.1/10**

## Remaining Blocking Issue (single)

- 当前 decision rule: "if `A2 − max(B1, C2) ≥ 2.0`, keep C2 single-seed" → 对 triage 可以，**但对 top-venue final paper 不够**
- 理由：如果 thesis 声称 style-conditioned routing 是 causal，reviewer 会要求 matched baseline C2 与 A2 同等 seeds
- Fix：**C2 必须 3-seed 出 final table，不能因 strong-margin 而跳过**

## Other Action Items (MINOR)

1. 3-seed 的 paired t-test 统计力弱 → 重点放 effect size + per-seed consistency
2. 澄清 C2 的 `S` 集合（all clients vs participating clients），要跟 A2 identical
3. Claim 3 的 heatmap 量化预测（"Caltech self > 0.7"）改为**定性**：Caltech should be more self-concentrated than other rows

## Simplification
- C1 明确为 secondary（预算紧 → 保 A2+C2）
- PACS 仅放 appendix
- Claim 3 可 fold 进 Claim 1/2 mechanism evidence section 而非独立 claim

## Drift
NONE.

## Raw Response

见 `round-3-review-raw.txt` 第 210-271 行。

Key quote:
> The only blocking issue is evidentiary: the key baseline `C2` must be promoted to full final-table status whenever the paper makes the causal routing claim. Once that is fixed, this is very close to `READY`.
