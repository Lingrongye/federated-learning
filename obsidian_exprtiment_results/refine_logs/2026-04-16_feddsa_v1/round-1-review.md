# Round 1 Review — GPT-5.4 Crisis Diagnosis Assessment

## Scores

| Dimension | Score | Weight |
|-----------|-------|--------|
| Problem Fidelity | **9/10** | 15% |
| Method Specificity | 7/10 | 25% |
| Contribution Quality | **8/10** | 25% |
| Frontier Leverage | 6/10 | 15% |
| Feasibility | 8/10 | 10% |
| Validation Focus | 7/10 | 5% |
| Venue Readiness | 6/10 | 5% |
| **Overall** | **7.4/10** | |

## Verdict: REVISE

Not READY (method story too broken), not RETHINK (diagnosis is valuable, credible paper exists if narrowed).

## Key Assessments

1. The diagnosis is accurate — the failure pattern is systematic, not noise
2. The gradient cosine trajectory is the most important evidence in the package
3. The "stable tau=0.2" result is not success — it is just underpowered InfoNCE
4. Still missing: cleaner attribution between "bad target" vs "bad schedule," and one representational diagnostic (prototype collapse vs domain overmixing vs class discrimination degradation)

## Recommended Route: C + A, with D as fallback

Priority order:
1. Reframe around Decouple + diagnosis (Route C)
2. Test bell-curve InfoNCE as the single repair (Route A)
3. If that fails, replace InfoNCE with one constraint-type consistency objective (Route D)
4. Deprioritize style-sharing rescue (Route B is weakest)

## Best Paper Thesis (from reviewer)

> "Cross-domain federated learning benefits from feature decoupling, but persistent directive alignment objectives induce late-stage gradient conflict with supervised learning. We show that alignment must be transient or constraint-based rather than continuously optimized."

## Weaknesses & Fixes

### Frontier Leverage (6/10) — IMPORTANT
- Weakness: Search space too centered on "how to rescue InfoNCE" rather than "how to respect objective interference in FL"
- Fix: Add one modern conflict-aware alternative — gradient surgery/PCGrad, stop-gradient teacher consistency (BYOL/SimSiam-style), or prototype EMA alignment without negatives

### Venue Readiness (6/10) — CRITICAL
- Weakness: The original three-module mechanism appears overcomplete and partly invalidated
- Fix: Rewrite around a narrower thesis: decoupling is robust, directive alignment causes late-stage conflict, and alignment must be transient or constraint-based

## Simplification Opportunities

- Drop the three-module story
- Treat Decouple as core, Share/Align as conditional
- Remove dual alignment (intra+cross)
- Stop exploring style-bank fixes
- Reduce alignment design to two options: early-only InfoNCE or non-contrastive consistency

## Modernization Opportunities

- PCGrad/gradient projection
- BYOL-style stop-gradient consistency
- BN-statistics consistency
- CKA/prototype drift diagnostics
