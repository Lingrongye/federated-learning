# Review Summary — FedDSA-SGPA

**Problem**: Cross-domain FL feature skew — improve over Plan A (orth_only + LR=0.05 + opt SAS) by ≥1% AVG without any persistent-alignment mechanism.

**Initial Approach**: FedDSA-ETA — Fixed ETF classifier + SATA style-exchange gate + T3A prototype TTA (3 parallel contributions).

**Date**: 2026-04-19
**Rounds**: 3 / 5
**Final Score**: 9.0 / 10
**Final Verdict**: READY

## Problem Anchor (verbatim, preserved all 3 rounds)
- Bottom-line: PACS/Office feature-skew FL,在 Plan A 之上 +≥1% AVG
- Must-solve: (a) classifier drift under FedAvg; (b) z_sty/style_bank 推理时未用
- Non-goals (Round 2 narrowed): no persistent alignment loss, no raw sample/feature sharing但允许 lightweight summary stats, no CLIP, no "FL+ETF+decouple 第一"
- Success: PACS AVG ≥81.5%, Office AVG ≥84% (w/o SAS) or ≥90.5% (w/ SAS), std ≤1.5%

## Round-by-Round Resolution Log

| Round | Main Concerns | What Changed | Resolved? | Remaining |
|-------|--------------|-------------|-----------|-----------|
| 1 | (1) Contribution sprawl 2.5 ways CRITICAL; (2) AdaIN(z_sem) math bug CRITICAL; (3) Venue story not singular | Renamed FedDSA-ETA→FedDSA-SGPA (SGPA as singular headline, ETF demoted); AdaIN deleted, replaced with z_sty distance gate; one-line thesis + "orthogonal to FedDEAP" rephrase | (1) ✅ RESOLVED; (2) 🟡 PARTIAL (new FedBN comparability issue); (3) ✅ RESOLVED | FedBN private BN → cross-client z_sty comparability |
| 2 | FedBN z_sty comparability CRITICAL; `bn.track_running_stats=False` bug; order-sensitive calibration; σ unused; minor drift on source_style_bank broadcast | Mahalanobis whitening via pooled Σ (within + between); removed track_running_stats flag; 5-batch warm-up + running EMA; σ now used; non-goal narrowed | ✅ all addressed, but "provably invariant" wording overreach | wording + warm-up code bug |
| 3 | "provably invariant" overreach; warm-up `continue` bug silently drops predictions; need diagnostic ablations | Reworded to "pooled second-order approximation"; warm-up now emits ETF fallback predictions; added whitening on/off + Σ decomposition ablations; symeig → torch.linalg.eigh | ✅ all 4 wording/pseudocode items applied | NONE |

## Overall Evolution
- **R0 → R1**: 3-way contribution sprawl → 1 dominant SGPA + 1 supporting ETF
- **R1 → R2**: AdaIN math bug → Mahalanobis whitening (principled, uses σ)
- **R2 → R3**: Statistical claim overreach → honest "pooled second-order approximation" framing
- **Method stayed FedDSA-compatible throughout** — Plan A backbone + orth decouple never touched

## Final Status
- **Anchor**: preserved 3/3 rounds
- **Focus**: tight, singular dominant contribution (SGPA)
- **Modernity**: 2025-era (NC + TTA + disentanglement), deliberate CLIP exclusion defensible
- **Strongest parts**:
  - SGPA dual gate in whitened Mahalanobis space is principled yet computationally minimal
  - ETF classifier genuinely eliminates FedAvg drift on classifier
  - backprop-free inference = zero training risk, ideal for reusing Plan A checkpoints
  - FedDEAP differentiation framed as "orthogonal: training-time prompt vs inference-time prototype correction" (reviewer-approved)
- **Remaining weaknesses**:
  - Formal proof of whitening invariance is partial (pooled, not per-client affine)
  - PACS N=4 clients → Σ_between rank ≤ 3 → may need fallback to Σ_within only
  - First 5 warm-up batches don't benefit from SGPA (ETF-only predictions)
