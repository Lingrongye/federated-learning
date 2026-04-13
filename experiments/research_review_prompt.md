# Research Review Request — FedDSA Current State (2026-04-12)

You are a senior ML reviewer at NeurIPS/ICML level. Please critically review the following research project and tell me: (1) is the story publishable, (2) what's missing, (3) how to best frame the contribution.

## Method: FedDSA (Decouple-Share-Align)

A cross-domain federated learning method that:
1. **Decouple**: Dual-head (semantic + style) with orthogonal + HSIC constraints on backbone features
2. **Share**: Global style bank (per-client per-class μ/σ stats) + AdaIN cross-client style augmentation
3. **Align**: InfoNCE contrastive alignment of semantic features to global prototypes

Backbone: AlexNet (same as FDSE baseline for fair comparison)
Aggregation: FedAvg on encoder+semantic_head+classifier; style_head and BN layers kept private (FedBN principle)

## Main Competitor: FDSE (CVPR 2025)

FDSE decomposes each layer into DFE (domain-free) + DSE (domain-shift eraser), uses consensus-max aggregation (QP), and KL BN consistency regularization. It treats domain/style information as "noise to erase."

Our key philosophical difference: we treat style as an "asset to share," not noise to erase.

## Complete Experimental Results (3-seed, R200, same AlexNet backbone)

### PACS (4 domains: Photo, Art, Cartoon, Sketch — HIGH style gap)

| Method | s=2 | s=15 | s=333 | Mean ± Std |
|---|---|---|---|---|
| FedAvg | - | - | - | ~72.10 (FDSE paper) |
| FedBN | - | - | - | ~79.47 (FDSE paper) |
| Ditto | - | - | - | ~80.03 (FDSE paper) |
| FDSE (paper AVG) | - | - | - | ~82.17 (paper R500) |
| **FDSE (our R200)** | **80.81** | ~79.93 | — | **~80.36** |
| **FedDSA baseline** | **82.24** | **80.59** | **81.05** | **81.29 ± 0.86** |
| FedDSA + Consensus QP | 83.04 | 79.39 | 79.80 | 80.74 ± 1.63 |
| FedDSA + Consensus + KNN nearest | 81.11 | 78.70 | 78.57 | 79.46 ± 1.17 |
| FedDSA + Consensus + Farthest + ProjBank | 80.64 | 78.89 | 78.19 | 79.24 ± 1.01 |

### Office-Caltech10 (4 domains: Amazon, DSLR, Webcam, Caltech — LOW style gap)

| Method | s=2 | s=15 | s=333 | Mean ± Std |
|---|---|---|---|---|
| FedAvg | - | - | - | ~86.26 (FDSE paper) |
| FedBN | - | - | - | ~87.01 (FDSE paper) |
| Ditto | - | - | - | ~88.72 (FDSE paper) |
| FDSE (paper AVG) | - | - | - | ~91.58 (paper R500) |
| **FDSE (our R200)** | **92.39** | — | — | **~90.58** (est) |
| **FedDSA baseline** | 89.95 | 91.08 | 86.35 | **89.13 ± 2.42** |
| **FedDSA + Consensus QP** | 89.40 | 90.11 | 89.99 | **89.83 ± 0.40** |
| FedDSA + Consensus + KNN nearest | 88.91 | 90.03 | 90.22 | 89.72 ± 0.58 |
| **FedDSA + Consensus + Farthest + ProjBank** | 89.48 | 90.22 | 89.77 | **89.82 ± 0.30** |

### Key observations:
- FedDSA baseline already beats FDSE on PACS (81.29 > 80.36)
- FedDSA loses to FDSE on Office (89.82 vs 90.58, gap = 0.76)
- Adding Consensus QP hurts PACS but helps Office stability (std 2.42 → 0.30)
- Style dispatch direction (nearest/farthest) has < 1% effect
- NOTE: FDSE paper uses R500, we use R200. Our FDSE R200 reproduction is ~80.36 (PACS) and ~90.58 (Office)

## Additional Validated Findings

1. **Style_head projection space is discriminative as regime signal**: PACS r ≈ 12 vs Office r ≈ 3 (3.6x ratio) when using style_head output for pairwise distance
2. **H1 (style aug is harmful on Office) falsified**: 4 style-side ablations (Gated, NoAug, SoftBeta, AugSchedule) all hurt Office
3. **H2 (aggregation conflict is the bottleneck) partially validated**: Consensus helps Office stability but hurts PACS

## What We Have NOT Tried Yet
- Hyperparameter tuning of FedDSA (lambda_orth, lambda_sem, tau, warmup — all at default)
- Server-side SAM with calibrated regime threshold
- DomainNet (3rd dataset, 6 domains)
- Early stopping / best checkpoint selection

## Questions for Reviewer

1. **Is this publishable?** Given FedDSA beats FDSE on PACS but loses on Office, is the story strong enough?
2. **What's the minimum additional experiment** to make this a solid paper?
3. **How should we frame the contribution?** Options:
   a) "Style as asset, not noise" (philosophical, but Office gap weakens it)
   b) "Regime-dependent behavior in FedDG" (novel finding, but we don't fully exploit it)
   c) "Decoupled style diagnostics for federated aggregation" (validated signal, but no performance gain from exploiting it)
   d) Something else?
4. **Is the "Consensus hurts PACS" finding novel and interesting?** This seems to be a genuine contribution — showing that single-mechanism aggregation is regime-dependent
5. **What would a skeptical reviewer's main attack be?** And how do we preempt it?
6. **Mock NeurIPS review**: Please write a realistic review with Strengths, Weaknesses, Questions, Score (1-10), and what would move toward accept.

Be brutally honest. False optimism wastes months.
