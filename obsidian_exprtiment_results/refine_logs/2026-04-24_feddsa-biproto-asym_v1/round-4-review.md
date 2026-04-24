# Round 4 Review (GPT-5.4 xhigh, resume thread)

**Date**: 2026-04-24

## Quick Calls (reviewer 总评)

| Item | Status |
|---|:-:|
| Problem Anchor | ✅ Preserved |
| ST normalization spec | ✅ Adequate now |
| C0 role wording | ✅ Accurate now |
| −Pd MANDATORY | ✅ Right call |
| "to our knowledge" + Limitations | 🟡 Sufficient for honest positioning, NOT sufficient to erase empirical scope risk (D=K=4 constrained) |

## Score Table

| Dimension | R3 | R4 | Δ | Reason |
|---|:-:|:-:|:-:|---|
| Problem Fidelity | 9 | **9.0** | 0 | 维持 |
| Method Specificity | 8 | **9.0** | +1 | ST form + 接口 + 梯度路径全部具体 |
| Contribution Quality | 8 | **8.5** | +0.5 | 单一 dominant, "to our knowledge" narrowing 生效 |
| Frontier Leverage | 9 | **9.0** | 0 | 维持 |
| Feasibility | 8 | **8.5** | +0.5 | present-classes-only 简化 |
| Validation Focus | 8 | **9.0** | +1 | −Pd MANDATORY + C0 role 清晰 |
| Venue Readiness | 7 | **8.0** | +1 | sharper 但 empirical ceiling 由 D=K=4 决定 |

## OVERALL: 8.75 / 10 — **REVISE** (距 9.0 READY bar 0.25)

## Reviewer 关键判断

> "v4 resolves the main mechanism and interface objections. The proposal is now coherent, implementable, and appropriately scoped. The remaining reason it is not READY is not a design flaw; it is that the venue case still depends on a narrow but crucial empirical question: whether Pd shows real value beyond a local domain-axis surrogate on D=K=4 benchmarks."

**关键**: 剩下的 gap 不是 design flaw, 是 empirical question — 只能通过**实际跑 −Pd ablation** 来回答. 这意味着 further method refinement **边际收益为零**, 继续 refine 无法把分推过 9.0 bar.

## 4 个 Remaining AI (非 blocking, 可在 implementation 阶段并行处理)

| # | Item | 优先级 |
|:-:|---|:-:|
| 1 | Pd 初始化和 warmup behavior 明确 (from first client centroid vs zeros vs random normalized) | Implementation detail |
| 2 | **Pre-register −Pd ablation 的 claim downgrade wording** 避免 post hoc drift | Paper integrity |
| 3 | −encoder_sty ablation 精确定义 (只改 input bias, 不改多 factor) | Ablation rigor |
| 4 | Method section 加 1 句明确 "ST estimator 是 biased but intentional training surrogate" | Transparency |

## Drift Warning: NONE

## Simplification / Modernization
- Simplification: −encoder_sty 降为 secondary; present_domains 在 local batch = 1 写明
- Modernization: NONE

## Stop Decision

**用户于 Round 4 后选择 stop**. 当前状态:
- Overall 8.75 (near-READY, 差 0.25)
- 所有 design-level critique 已解决
- 剩余 0.25 gap 由 empirical 验证 (−Pd ablation) 决定, 非 refine 可推进
- 4 个 remaining AI 留给 implementation / experiment-plan 阶段处理
