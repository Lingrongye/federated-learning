# Refinement Report — FedDSA-CDANN

**Problem**: 联邦学习跨域客户端的风格-语义解耦在 Office vs PACS 的自适应边界; 统计解耦 (cos²+HSIC) 没有方向性导致 whitening 磨掉 class-relevant style
**Initial Approach**: 双向 DANN 头 (z_sem 反向 + z_sty 正向), 保留风格的 class 信号
**Date**: 2026-04-20
**Rounds**: 5 / 5 (MAX_ROUNDS reached)
**Final Score**: 8.75 / 10
**Final Verdict**: **REVISE (proposal-complete, near review-time ceiling)**

## Problem Anchor (verbatim)

- **Bottom-line problem**: 跨域 FL 解耦在 Office (风格弱, whitening +6.20pp) vs PACS (风格强, -1.49pp) 表现相反
- **Must-solve bottleneck**: 原 L_orth+HSIC 无方向监督; PACS z_sty_norm 被 whitening 磨掉 95%
- **Non-goals / Constraints / Success**: 见 FINAL_PROPOSAL.md

## Output Files

- Review summary: `refine-logs/2026-04-20_FedDSA-CDANN/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/2026-04-20_FedDSA-CDANN/FINAL_PROPOSAL.md`
- Score evolution: `refine-logs/2026-04-20_FedDSA-CDANN/score-history.md`
- Raw rounds: `round-{0..4}-refinement.md`, `round-{1..5}-review.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1     | 8                | 8                  | 6                    | 6                 | 8           | 8                | 6               | 7.1     | REVISE |
| 2     | 9                | 9                  | 8                    | 8                 | 8           | 8                | 7               | 8.35    | REVISE |
| 3     | 9                | 9                  | 8                    | 8                 | 8           | 8                | 8               | 8.4     | REVISE |
| 4     | 9                | 9                  | 9                    | 8                 | 8           | 9                | 8               | 8.75    | REVISE |
| 5     | 9                | 9                  | 9                    | 8                 | 8           | 9                | 8               | 8.75    | REVISE (near ceiling) |

**Delta summary**: Overall 7.1 → 8.75 (+1.65), 稳定上升, 无回退. Problem Fidelity 8→9, Contribution Quality 6→9 (最大涨幅, 来自 R1 合并 dom_heads + 收敛 single contribution), Venue Readiness 6→8 (scope 窄化 + novelty one-sentence).

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|-------|-----------------------|------------------|--------|
| 1 | 2 heads 像 "加模块"; dataset diagnosis 发散; AlexNet-only dated; Venue Readiness 6 | 合并 shared head; 降级 diagnosis; 窄化 scope; 加 DINOv2 | Resolved (6 → 8 on Contrib) |
| 2 | "shared head with opposing gradients" 误导 (head non-adversarial); training head acc 不 clean; C-port 可能变 second | 精确重写 asymmetry 在 encoder gradient 不在 head objective; frozen post-hoc probe; C-port 降 appendix | Resolved (8.35 → 8.4) |
| 3 | Probe fit on test leaked; 缺 class probe on z_sty; "30-paper gap" novelty 话术弱 | Fix train/test split; 加 `probe_sty_class`; 删综述 gap, 改窄化 novelty 句 | Resolved (8.4 → 8.75) |
| 4 | Probe 过 formal proof; 未明确 post-whitening; novelty 句不统一 | "evidence consistent with anchor"; 明确 post-whitening; 锁 one-sentence novelty verbatim | Partial (8.75, framing ceiling) |
| 5 | **Intrinsic novelty ceiling, not fixable** | 无 proposal 改动, 只是 framing discipline | **Ceiling acknowledged** (8.75) |

## Final Proposal Snapshot

- Canonical clean version: `refine-logs/2026-04-20_FedDSA-CDANN/FINAL_PROPOSAL.md`

**Final thesis (3 bullets)**:
1. **Dominant contribution**: Shared non-adversarial `dom_head` (两路 standard CE) + **asymmetric encoder-gradient direction** (GRL on z_sem path only, identity on z_sty path) = minimal repair for whitening-induced style collapse in `client=domain` FedDG when style carries class signal.
2. **Key evidence**: Frozen post-hoc `probe_sty_class` on z_sty (post-whitening features), PACS expected ≥40% (CDANN) vs ≈15% (Linear+whitening baseline), 差距量化 "CDANN 保留了被 whitening 擦掉的 class-relevant style".
3. **Scope discipline**: 明确限定 `client=domain FedDG where style carries class signal`; 多域每 client (DomainNet) 留作 future work.

## Method Evolution Highlights

1. **最重要的 simplification**: R1 合并 2 dom_heads 为 1 shared head, 把 asymmetry 从 "head 结构" 移到 "encoder gradient direction". 减半 trainable params, 消除 "加模块" 印象.
2. **最重要的 mechanism clarification**: R2 精确表述 dom_head non-adversarial (两路都 minimize standard CE), 避免被读成 "adversarially optimized head".
3. **最重要的 evidence alignment**: R3 加 `probe_sty_class` (z_sty → y) 直接对应 anchor 的 "class-relevant style" 部分, 不只是 domain disentanglement.
4. **最重要的 modernization balance**: R1 加 DINOv2 appendix sanity check, R2/R3/R4 保持 appendix 位置不变, 避免变成 second contribution.

## Pushback / Drift Log

| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 1 | "Dataset boundary diagnosis" 可作 second contribution | Author 主动降级为 supporting analysis (非 contribution) | Accepted (R1 CRITICAL fix, 主动收敛) |
| 1 | 加 gradient projection (PCGrad) 处理 λ 冲突? | Author 拒绝: 增加复杂度, λ_adv schedule 已够 | Rejected (保持最小机制) |
| 1 | 加跨 domain cluster pseudo-label 扩展 DomainNet | Author 拒绝: 超出当前 scope, 留作 future work | Rejected (scope discipline) |
| 2 | "Training dom_head accuracy" 作 diagnostic | Author 替换为 frozen post-hoc probe | Accepted (R2 CRITICAL) |
| 2 | C-port 可能扩成 second contribution | Author 明确降 appendix sanity | Accepted (R2 IMPORTANT) |
| 3 | "Need class probe on z_sty" | Author 加 `probe_sty_class` | Accepted (R3 CRITICAL) |
| 3 | Probe fit on test 是 leak | Author 改 train on train / test on held-out | Accepted (R3 CRITICAL) |
| 3 | 停用 "30-paper review gap" 作 novelty | Author 改窄化 one-sentence novelty | Accepted (R3 IMPORTANT) |
| 4 | Probe claims "formal proof" | Author 改 "evidence consistent with anchor" | Accepted (R4 IMPORTANT) |
| 5 | "Intrinsic novelty ceiling" | Author 接受 reviewer 判断, 不再做 proposal 改动, 转向 empirical 执行 | Accepted (MAX_ROUNDS reached) |

**无 drift warning 全程 5 轮** (reviewer 明确 "Preserved" 5/5 次), 诚实保留窄 scope, 未被推广成 "universal method".

## Remaining Weaknesses (Honest Disclosure)

1. **Novelty ceiling 是内在的**: R5 reviewer: "proposal is near its review-time ceiling. Further proposal-side refinement is unlikely to materially change the score."
2. **Venue readiness 停在 8/10**: 即使完美执行, 仍被强 reviewer 读成 "very clean asymmetric DANN-style repair inside an existing FedDG pipeline", 非 new method family.
3. **唯一上升路径**: 实验结果 overperform:
   - PACS AVG Best 大幅超越 Plan A (> +2pp) 且 `probe_sty_class` 差距 > 30pp → empirical 支撑 top-venue pitch
   - 若结果只是 "回到 Plan A" 没 +2pp 额外 gain, 论文只能 mid-tier (BMVC/WACV)

## Raw Reviewer Responses

<details>
<summary>Round 1 Review (Overall 7.1, REVISE)</summary>

见 `refine-logs/2026-04-20_FedDSA-CDANN/round-1-review.md` (verbatim Codex output)

Key objections:
- Contribution Quality 6/10: 过包装, "asymmetric dual-direction DANN" 是真 contribution, dataset diagnosis 和 style asset 稀释 focus
- Frontier Leverage 6/10: AlexNet-from-scratch 显 dated; 建议加 frozen DINOv2 portability check
- Venue Readiness 6/10: 过宽 claim, 应窄化到 `client=domain` 场景

</details>

<details>
<summary>Round 2 Review (Overall 8.35, REVISE)</summary>

见 `refine-logs/2026-04-20_FedDSA-CDANN/round-2-review.md`

Key remaining objections:
- Mechanism 描述 "shared head with opposing gradients" 略 overstated, 实际 head non-adversarial
- Training dom_head accuracy 不是 clean representation diagnostic
- C-port 需明确 auxiliary 定位

</details>

<details>
<summary>Round 3 Review (Overall 8.4, REVISE)</summary>

见 `refine-logs/2026-04-20_FedDSA-CDANN/round-3-review.md`

Key remaining objections:
- Probe protocol fit on test 泄漏
- 缺 anchor-aligned class probe on z_sty (只有 domain probe 不够)
- "2024-2026 review gap" novelty 话术弱, 应改用 mechanism-minimal framing

</details>

<details>
<summary>Round 4 Review (Overall 8.75, REVISE)</summary>

见 `refine-logs/2026-04-20_FedDSA-CDANN/round-4-review.md`

Key remaining objections (均 IMPORTANT, 非 CRITICAL):
- Class probe 描述应 "consistent with anchor" 非 formal proof
- 明确三 probe 都在 post-whitening feature space
- One-sentence novelty discipline

"Mainly venue bar, not method sloppiness"

</details>

<details>
<summary>Round 5 Review (Overall 8.75, REVISE — proposal-complete)</summary>

见 `refine-logs/2026-04-20_FedDSA-CDANN/round-5-review.md`

**Reviewer final judgment**:
> "Yes. This proposal is near its review-time ceiling. Further proposal-side refinement is unlikely to materially change the score. The only path upward now is execution quality: if results are unusually strong, consistent, and clearly beat the strongest ablations, the paper could overperform the current proposal score. But that is an empirical upside, not a proposal-design fix."

> "This is proposal-complete. The remaining gap is not a fixable weakness in framing or mechanism specification; it is the method's inherent novelty ceiling relative to the READY bar."

</details>

## Next Steps

- ❌ **Do not continue proposal refinement** (ceiling reached, diminishing return)
- ✅ **Proceed to `/experiment-plan`** for detailed execution-ready experiment roadmap (根据 FINAL_PROPOSAL.md 的 C-main / C-ablate / C-port + probes 展开具体 config / commands / decision gates)
- ✅ **Implement code**: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` 加 GRL + dom_head + L_dom + probe 脚本 (~50 行 + 单测 + Codex code review)
- ✅ **Run pilot**: PACS + Office R100 × 1 seed (seetacloud2 2.5h), 决定 Full 或 pivot
- ✅ **If pilot signal positive**: Full C-main (12h) + C-ablate (24h) + C-port (2h) + Probes (30min) = 41 GPU·h
- ✅ **Write knowledge notes**: 大白话版 + 学术版 (obsidian_exprtiment_results/知识笔记/)
