OpenAI Codex v0.117.0 (research preview)
--------
workdir: D:\桌面文件\联邦学习
model: gpt-5.4
provider: fox
approval: never
sandbox: read-only
reasoning effort: high
reasoning summaries: none
session id: 019d81fd-426e-7031-bc86-cbeb56dcf07b
--------
user
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

codex
I’m checking the cited FDSE paper first so the comparison is grounded in the actual baseline design, then I’ll give you a hard-nosed publishability assessment against your reported results.
web search: 
web search: FDSE CVPR 2025 federated domain generalization style eraser openaccess
web search: 
web search: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Federated_Learning_with_Domain_Shift_Eraser_CVPR_2025_paper.pdf
web search: 
web search: 'PACS' in https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Federated_Learning_with_Domain_Shift_Eraser_CVPR_2025_paper.pdf
codex
I’ve verified the FDSE paper setup: AlexNet backbone, three datasets including DomainNet, 500 communication rounds, and tuned hyperparameters in the official comparison. That makes the current weak point pretty clear: your story is plausible, but your present evidence is not yet at the standard needed to claim superiority over FDSE.
codex
**Verdict**

In its current state: not a main-track NeurIPS/ICML paper. It is potentially a good workshop paper or a main-track submission after a focused repositioning plus a few mandatory experiments.

The reason is simple: your current evidence does **not** support the claim that FedDSA is a better method than FDSE. It supports a weaker but more interesting claim: **style sharing is regime-dependent in cross-domain FL**. PACS helps you; Office hurts you. That is not fatal if you frame it correctly. It is fatal if you frame it as universal algorithmic superiority.

**1. Is this publishable?**

Yes, but not with the current story.

If the story is:
- “FedDSA beats FDSE” -> no, not credible enough yet.
- “Style is an asset, not noise” -> too strong; Office directly weakens it.
- “The utility of style sharing and consensus in federated cross-domain learning is regime-dependent, and we can diagnose that regime from decoupled style representations” -> this is the publishable story.

Right now the paper reads like a method paper with mixed results. It should become either:
- a **regime-aware method paper**, or
- a **diagnostic/analysis paper with a method as the vehicle**.

Without that pivot, reviewers will say: one win, one loss, missing third dataset, unfair protocol mismatch, unclear takeaway.

**2. What is missing?**

These are the real gaps, in order of importance:

1. **Matched-protocol evidence against FDSE**
- FDSE’s official paper uses AlexNet, three datasets, **500 rounds**, and tuned hyperparameters over a grid; results are averaged over **5 trials**.
- Your current comparison is **R200**, **3 seeds**, and largely default FedDSA hyperparameters.
- A skeptical reviewer will say your PACS gain may be a training-horizon artifact and your Office loss may be a tuning artifact.

2. **Third dataset**
- You need **DomainNet**. With only PACS and Office, the “regime” thesis is underdetermined.
- DomainNet is especially important because FDSE reports it, so omitting it looks selective.

3. **True DG-style evaluation if you want to say FedDG**
- Your table is standard cross-domain FL on participating clients.
- If you want to claim “FedDG,” you need unseen-domain evidence: leave-one-domain-out, unseen-client adaptation, or the FDSE-style held-out-domain adaptation protocol.
- Otherwise reviewers will call this cross-domain FL, not federated domain generalization.

4. **A clean component ablation**
- You need: `Decouple only`, `+Share`, `+Align`, and full model.
- Right now the story mixes many ingredients: orthogonal, HSIC, private BN, style bank, AdaIN, prototypes, InfoNCE. A reviewer can fairly say this is a bundle, not an insight.

5. **Exploit the diagnostic, not just report it**
- Your style-head distance signal is interesting.
- But if it only diagnoses and never changes the algorithm, many reviewers will view it as an observation, not a contribution.

6. **Privacy/communication discussion**
- A per-client per-class `mu/sigma` style bank is not “free” in FL.
- Reviewers may ask whether it leaks label/domain information and what the communication/storage overhead is relative to FDSE.

**3. Minimum additional experiment**

If you force me to pick the single highest-value next step:

**Run DomainNet under a fully matched FDSE protocol**:
- AlexNet
- 500 rounds
- same evaluation protocol
- tuned key hyperparameters for FedDSA
- at least 5 seeds if feasible

Then report one extra figure:
- x-axis: your style-dispersion score
- y-axis: FedDSA minus FDSE gain

If PACS = high dispersion / positive gain, Office = low dispersion / negative or neutral gain, DomainNet = intermediate and lands accordingly, then you have a real paper.

If you can afford one more addition, make it this:
- a **simple regime-adaptive switch** using your diagnostic
- e.g. attenuate style sharing / consensus when style dispersion is low
- it does not need to be fancy; a thresholded gate is enough

That would convert the paper from “interesting observation” to “actionable method.”

**4. How to frame the contribution**

Best framing is not your options a/b/c exactly. It should be:

**“Style is conditionally useful shared structure in federated cross-domain learning.”**

More concretely:

- Existing methods like FDSE assume domain/style factors should be erased to maximize consensus.
- Your results suggest that assumption is **regime-dependent**.
- In high style-divergence regimes, explicit style decoupling and sharing can help.
- In low style-divergence regimes, aggressive consensus/personalization mechanisms can over-constrain training and hurt.
- A decoupled style branch gives a measurable regime signal that can guide algorithm choice.

That framing uses Office as evidence, not embarrassment.

I would avoid:
- “style as asset, not noise” as the main claim
because the honest version is:
- “style is **not always** nuisance, and **not always** an asset; its role depends on inter-client style dispersion.”

That is stronger intellectually and harder for reviewers to dismiss.

**5. Is “Consensus hurts PACS” novel and interesting?**

Yes, but only if you make it systematic.

As a one-off result, reviewers may dismiss it as:
- optimization instability,
- a bad QP implementation,
- horizon mismatch,
- or hyperparameter sensitivity.

It becomes interesting if you show:
- the effect repeats across PACS / Office / DomainNet,
- it correlates with your style-dispersion metric,
- and it varies predictably with consensus strength.

Then the finding becomes:

**Consensus is not uniformly beneficial in cross-domain FL; its effect depends on regime.**

That is novel enough to matter.

**6. Main skeptical attacks, and how to preempt them**

The top attacks will be:

1. **“This is not a fair comparison to FDSE.”**
- Preempt with matched rounds, matched seeds, matched backbone, matched evaluation, tuned hypers for both.

2. **“The method is a bag of tricks.”**
- Preempt with a strict incremental ablation and a compact conceptual decomposition:
  `decouple -> expose style factor`,
  `share -> leverage cross-client style diversity`,
  `align -> preserve semantic consistency`.

3. **“You win only on one dataset and lose on another.”**
- Preempt by explicitly framing the paper as regime-dependent, not universally superior.
- Add DomainNet and show the trend follows style dispersion.

4. **“Your diagnostic is post hoc and not useful.”**
- Preempt with a simple gating policy driven by the diagnostic.

5. **“This is not FedDG.”**
- Preempt with unseen-domain / held-out-client evaluation.

6. **“The style bank may leak information.”**
- Preempt with a privacy and overhead discussion, ideally a short analysis table.

**My honest recommendation**

Do not submit this as “FedDSA beats FDSE.”

Submit it, if at all, as:

**Regime-Aware Federated Cross-Domain Learning: When Style Sharing Helps and When Consensus Hurts**

FedDSA is then the experimental instrument and baseline method, not the sole claim.

**Mock NeurIPS Review**

**Summary**
The paper proposes FedDSA, a federated cross-domain learning method that decouples semantic and style representations, shares class-conditional style statistics across clients for augmentation, and aligns semantic features via global prototypes. The core thesis is that style should not always be erased as nuisance; instead, style can be a useful cross-client resource. Experiments show gains over an FDSE reproduction on PACS but weaker performance on Office-Caltech10, alongside an interesting observation that consensus-based aggregation improves stability in low-style-gap settings but hurts in high-style-gap settings.

**Strengths**
- Tackles a real and underexplored question in federated cross-domain learning: whether domain/style factors should be erased or exploited.
- The decouple-share-align design is conceptually coherent.
- The PACS result and the “consensus hurts PACS” observation are interesting.
- The style-space diagnostic appears promising as a regime indicator.
- The paper has a plausible broader contribution beyond a single algorithmic tweak: understanding when cross-client style sharing is beneficial.

**Weaknesses**
- The empirical case against the main baseline is not yet convincing. The comparison appears not fully matched in training budget and tuning, while the baseline paper reports stronger settings.
- Results are mixed: gains on PACS but losses on Office-Caltech10. As written, this weakens the central claim of method superiority.
- The work currently looks like a bundle of components rather than a sharply isolated contribution; stronger ablations are needed.
- The paper does not yet convincingly exploit the proposed regime signal for decision-making.
- If the paper claims federated domain generalization, unseen-domain evaluation is missing or insufficiently emphasized.
- Privacy and communication implications of sharing per-client per-class style statistics are under-discussed.

**Questions**
- How do results change under the exact FDSE training protocol, including 500 rounds and tuned hyperparameters?
- What happens on DomainNet, which is used in the FDSE paper and would provide a crucial third regime?
- Does the proposed style-dispersion metric predict performance gains across datasets/splits/seeds?
- Can the method use this signal to switch between stronger style sharing and more conservative training?
- What is the privacy leakage surface of the style bank compared with prototype-sharing baselines?

**Score**
4/10

**Confidence**
High

**What would move this toward accept**
- A fully matched comparison to FDSE under the same protocol.
- DomainNet results.
- Unseen-domain / held-out-client evaluation if claiming FedDG.
- A clean incremental ablation.
- A simple regime-adaptive policy using the diagnostic, showing reduced harm on Office while preserving PACS gains.

**Bottom line**

Current state: **interesting, but reject for main track**.

Best salvage path:
- stop claiming universal superiority,
- pivot to **regime-dependent utility of style sharing**,
- run **matched 500-round tuned comparisons plus DomainNet**,
- and, ideally, add a **simple diagnostic-driven gate**.

That gives you a paper with a real idea instead of a method with mixed tables.

Source check: I verified from the official FDSE CVPR 2025 paper that it uses AlexNet, evaluates on Office-Caltech10, PACS, and DomainNet, and reports 500-round tuned experiments averaged over 5 trials: https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Federated_Learning_with_Domain_Shift_Eraser_CVPR_2025_paper.html and PDF: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Federated_Learning_with_Domain_Shift_Eraser_CVPR_2025_paper.pdf

If you want, I can next turn this into a concrete rebuttal-style paper plan: title, abstract angle, experiment priority list, and exact figure/table structure.
tokens used
28,452
**Verdict**

In its current state: not a main-track NeurIPS/ICML paper. It is potentially a good workshop paper or a main-track submission after a focused repositioning plus a few mandatory experiments.

The reason is simple: your current evidence does **not** support the claim that FedDSA is a better method than FDSE. It supports a weaker but more interesting claim: **style sharing is regime-dependent in cross-domain FL**. PACS helps you; Office hurts you. That is not fatal if you frame it correctly. It is fatal if you frame it as universal algorithmic superiority.

**1. Is this publishable?**

Yes, but not with the current story.

If the story is:
- “FedDSA beats FDSE” -> no, not credible enough yet.
- “Style is an asset, not noise” -> too strong; Office directly weakens it.
- “The utility of style sharing and consensus in federated cross-domain learning is regime-dependent, and we can diagnose that regime from decoupled style representations” -> this is the publishable story.

Right now the paper reads like a method paper with mixed results. It should become either:
- a **regime-aware method paper**, or
- a **diagnostic/analysis paper with a method as the vehicle**.

Without that pivot, reviewers will say: one win, one loss, missing third dataset, unfair protocol mismatch, unclear takeaway.

**2. What is missing?**

These are the real gaps, in order of importance:

1. **Matched-protocol evidence against FDSE**
- FDSE’s official paper uses AlexNet, three datasets, **500 rounds**, and tuned hyperparameters over a grid; results are averaged over **5 trials**.
- Your current comparison is **R200**, **3 seeds**, and largely default FedDSA hyperparameters.
- A skeptical reviewer will say your PACS gain may be a training-horizon artifact and your Office loss may be a tuning artifact.

2. **Third dataset**
- You need **DomainNet**. With only PACS and Office, the “regime” thesis is underdetermined.
- DomainNet is especially important because FDSE reports it, so omitting it looks selective.

3. **True DG-style evaluation if you want to say FedDG**
- Your table is standard cross-domain FL on participating clients.
- If you want to claim “FedDG,” you need unseen-domain evidence: leave-one-domain-out, unseen-client adaptation, or the FDSE-style held-out-domain adaptation protocol.
- Otherwise reviewers will call this cross-domain FL, not federated domain generalization.

4. **A clean component ablation**
- You need: `Decouple only`, `+Share`, `+Align`, and full model.
- Right now the story mixes many ingredients: orthogonal, HSIC, private BN, style bank, AdaIN, prototypes, InfoNCE. A reviewer can fairly say this is a bundle, not an insight.

5. **Exploit the diagnostic, not just report it**
- Your style-head distance signal is interesting.
- But if it only diagnoses and never changes the algorithm, many reviewers will view it as an observation, not a contribution.

6. **Privacy/communication discussion**
- A per-client per-class `mu/sigma` style bank is not “free” in FL.
- Reviewers may ask whether it leaks label/domain information and what the communication/storage overhead is relative to FDSE.

**3. Minimum additional experiment**

If you force me to pick the single highest-value next step:

**Run DomainNet under a fully matched FDSE protocol**:
- AlexNet
- 500 rounds
- same evaluation protocol
- tuned key hyperparameters for FedDSA
- at least 5 seeds if feasible

Then report one extra figure:
- x-axis: your style-dispersion score
- y-axis: FedDSA minus FDSE gain

If PACS = high dispersion / positive gain, Office = low dispersion / negative or neutral gain, DomainNet = intermediate and lands accordingly, then you have a real paper.

If you can afford one more addition, make it this:
- a **simple regime-adaptive switch** using your diagnostic
- e.g. attenuate style sharing / consensus when style dispersion is low
- it does not need to be fancy; a thresholded gate is enough

That would convert the paper from “interesting observation” to “actionable method.”

**4. How to frame the contribution**

Best framing is not your options a/b/c exactly. It should be:

**“Style is conditionally useful shared structure in federated cross-domain learning.”**

More concretely:

- Existing methods like FDSE assume domain/style factors should be erased to maximize consensus.
- Your results suggest that assumption is **regime-dependent**.
- In high style-divergence regimes, explicit style decoupling and sharing can help.
- In low style-divergence regimes, aggressive consensus/personalization mechanisms can over-constrain training and hurt.
- A decoupled style branch gives a measurable regime signal that can guide algorithm choice.

That framing uses Office as evidence, not embarrassment.

I would avoid:
- “style as asset, not noise” as the main claim
because the honest version is:
- “style is **not always** nuisance, and **not always** an asset; its role depends on inter-client style dispersion.”

That is stronger intellectually and harder for reviewers to dismiss.

**5. Is “Consensus hurts PACS” novel and interesting?**

Yes, but only if you make it systematic.

As a one-off result, reviewers may dismiss it as:
- optimization instability,
- a bad QP implementation,
- horizon mismatch,
- or hyperparameter sensitivity.

It becomes interesting if you show:
- the effect repeats across PACS / Office / DomainNet,
- it correlates with your style-dispersion metric,
- and it varies predictably with consensus strength.

Then the finding becomes:

**Consensus is not uniformly beneficial in cross-domain FL; its effect depends on regime.**

That is novel enough to matter.

**6. Main skeptical attacks, and how to preempt them**

The top attacks will be:

1. **“This is not a fair comparison to FDSE.”**
- Preempt with matched rounds, matched seeds, matched backbone, matched evaluation, tuned hypers for both.

2. **“The method is a bag of tricks.”**
- Preempt with a strict incremental ablation and a compact conceptual decomposition:
  `decouple -> expose style factor`,
  `share -> leverage cross-client style diversity`,
  `align -> preserve semantic consistency`.

3. **“You win only on one dataset and lose on another.”**
- Preempt by explicitly framing the paper as regime-dependent, not universally superior.
- Add DomainNet and show the trend follows style dispersion.

4. **“Your diagnostic is post hoc and not useful.”**
- Preempt with a simple gating policy driven by the diagnostic.

5. **“This is not FedDG.”**
- Preempt with unseen-domain / held-out-client evaluation.

6. **“The style bank may leak information.”**
- Preempt with a privacy and overhead discussion, ideally a short analysis table.

**My honest recommendation**

Do not submit this as “FedDSA beats FDSE.”

Submit it, if at all, as:

**Regime-Aware Federated Cross-Domain Learning: When Style Sharing Helps and When Consensus Hurts**

FedDSA is then the experimental instrument and baseline method, not the sole claim.

**Mock NeurIPS Review**

**Summary**
The paper proposes FedDSA, a federated cross-domain learning method that decouples semantic and style representations, shares class-conditional style statistics across clients for augmentation, and aligns semantic features via global prototypes. The core thesis is that style should not always be erased as nuisance; instead, style can be a useful cross-client resource. Experiments show gains over an FDSE reproduction on PACS but weaker performance on Office-Caltech10, alongside an interesting observation that consensus-based aggregation improves stability in low-style-gap settings but hurts in high-style-gap settings.

**Strengths**
- Tackles a real and underexplored question in federated cross-domain learning: whether domain/style factors should be erased or exploited.
- The decouple-share-align design is conceptually coherent.
- The PACS result and the “consensus hurts PACS” observation are interesting.
- The style-space diagnostic appears promising as a regime indicator.
- The paper has a plausible broader contribution beyond a single algorithmic tweak: understanding when cross-client style sharing is beneficial.

**Weaknesses**
- The empirical case against the main baseline is not yet convincing. The comparison appears not fully matched in training budget and tuning, while the baseline paper reports stronger settings.
- Results are mixed: gains on PACS but losses on Office-Caltech10. As written, this weakens the central claim of method superiority.
- The work currently looks like a bundle of components rather than a sharply isolated contribution; stronger ablations are needed.
- The paper does not yet convincingly exploit the proposed regime signal for decision-making.
- If the paper claims federated domain generalization, unseen-domain evaluation is missing or insufficiently emphasized.
- Privacy and communication implications of sharing per-client per-class style statistics are under-discussed.

**Questions**
- How do results change under the exact FDSE training protocol, including 500 rounds and tuned hyperparameters?
- What happens on DomainNet, which is used in the FDSE paper and would provide a crucial third regime?
- Does the proposed style-dispersion metric predict performance gains across datasets/splits/seeds?
- Can the method use this signal to switch between stronger style sharing and more conservative training?
- What is the privacy leakage surface of the style bank compared with prototype-sharing baselines?

**Score**
4/10

**Confidence**
High

**What would move this toward accept**
- A fully matched comparison to FDSE under the same protocol.
- DomainNet results.
- Unseen-domain / held-out-client evaluation if claiming FedDG.
- A clean incremental ablation.
- A simple regime-adaptive policy using the diagnostic, showing reduced harm on Office while preserving PACS gains.

**Bottom line**

Current state: **interesting, but reject for main track**.

Best salvage path:
- stop claiming universal superiority,
- pivot to **regime-dependent utility of style sharing**,
- run **matched 500-round tuned comparisons plus DomainNet**,
- and, ideally, add a **simple diagnostic-driven gate**.

That gives you a paper with a real idea instead of a method with mixed tables.

Source check: I verified from the official FDSE CVPR 2025 paper that it uses AlexNet, evaluates on Office-Caltech10, PACS, and DomainNet, and reports 500-round tuned experiments averaged over 5 trials: https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Federated_Learning_with_Domain_Shift_Eraser_CVPR_2025_paper.html and PDF: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Federated_Learning_with_Domain_Shift_Eraser_CVPR_2025_paper.pdf

If you want, I can next turn this into a concrete rebuttal-style paper plan: title, abstract angle, experiment priority list, and exact figure/table structure.
