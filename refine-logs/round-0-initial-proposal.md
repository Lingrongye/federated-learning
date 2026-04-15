# Research Proposal: Decoupled Domain-Aware Dual Alignment for Federated Prototype Learning (M4)

## Problem Anchor
- **Bottom-line problem**: M3 domain-aware prototype alignment (+5.1%) retains all (class, client) prototypes with equal weight and flat SupCon InfoNCE, lacking quality distinction and intra/cross-domain differentiation
- **Must-solve bottleneck**: Equal-weight prototypes + single SupCon conflates intra-domain stability and cross-domain generalization into one loss. FedDAP proved dual alignment + cosine weighting is superior (PACS 84.63%) but operates in entangled space
- **Non-goals**: No changes to Decouple/Share, no new trainable modules, no backbone change
- **Constraints**: feddsa_adaptive.py, AlexNet, z_sem 128d, PACS/Office/DomainNet, tau=0.2, lab-lry GPU1
- **Success condition**: PACS 3-seed mean > 81.29%, stable training (peak-final < 2%), no communication increase

## Technical Gap

Current M3 treats all domain prototypes as equal positive samples in a flat SupCon InfoNCE. This conflates two distinct alignment objectives:

1. **Intra-domain stability**: Features should be close to same-domain prototype of same class (domain-specific quality)
2. **Cross-domain generalization**: Features should also approach other-domain prototypes of same class (domain-invariant semantics)

Flat SupCon cannot weight these differently. FedDAP (arXiv 2025) solves this with cosine-weighted fusion + dual alignment (L_DPA + L_CPCL), achieving PACS 84.63%. However, FedDAP operates in entangled feature space where prototypes contain style noise.

Our opportunity: Apply dual alignment in FedDSA's orthogonally-decoupled z_sem space, where prototypes are pure semantic signals. Style noise removal should make both weighted fusion and dual alignment more effective.

## Method Thesis
- **One-sentence thesis**: Cosine-weighted domain prototype fusion + dual alignment (intra-domain stability + cross-domain contrastive) in decoupled z_sem space achieves more effective prototype alignment than equal-weight flat SupCon or entangled-space dual alignment
- **Why smallest adequate**: Only changes server-side aggregation weighting + client-side loss decomposition. Zero new trainable parameters
- **Why timely**: Directly improves upon FedDAP (2025) by leveraging orthogonal decoupling

## Contribution Focus
- **Dominant contribution**: Dual alignment in decoupled z_sem space with cosine-weighted fusion
- **Supporting contribution**: Empirical evidence that decoupled-space alignment > entangled-space alignment
- **Non-contributions**: No new clustering, no new architecture, no new trainable components

## Proposed Method

### Complexity Budget
- **Frozen/reused**: AlexNet, semantic_head, style_head, classifier, AdaIN augmentation, orthogonal constraints
- **New trainable components**: None (zero additional parameters)
- **Intentionally excluded**: FINCH clustering (unstable with 4 protos), trainable prototypes (adds complexity), alpha-sparsity (optional ablation only)

### System Overview

Server (each round after warmup):
  1. Collect z_sem class prototypes: {(class, client_id) -> proto}
  2. Cosine-weighted fusion per class:
     - S_j = sum_{k!=j} cos(p_j, p_k)  (consistency score)
     - w_j = softmax(S_j / tau_agg)     (attention weight)
  3. Dispatch to client m in domain d_m:
     - Intra-domain protos: {P^(c, d_m)} for all classes c
     - Cross-domain protos: {P^(c, d')} for all c, d' != d_m

Client m (training):
  1. Forward: h -> z_sem, z_sty -> CE + orthogonal
  2. Style augment: AdaIN -> z_sem_aug -> CE_aug
  3. Dual alignment:
     a) L_intra = mean(1 - cos(z_sem_i, P^(y_i, d_m)))  (pull toward same-domain proto)
     b) L_cross = InfoNCE with cross-domain protos (same-class=positive, diff-class=negative)
     c) L_align = lambda_intra * L_intra + lambda_cross * L_cross

### Core Mechanism

**Server: Cosine-Weighted Fusion**
- For each class, compute pairwise cosine similarity among domain prototypes
- Weight each prototype by its consistency with others (semantically coherent protos get higher weight)
- In decoupled z_sem space, this consistency truly reflects semantic agreement (no style confusion)

**Client: Dual Alignment Loss**
- L_intra (cosine similarity): Ensures features are stable within own domain
- L_cross (InfoNCE contrastive): Encourages features to generalize across domains
- The two losses serve complementary objectives that flat SupCon cannot separate

### Novelty Argument

| Method | Space | Fusion | Alignment |
|--------|-------|--------|-----------|
| FedProto | entangled | mean | MSE |
| FPL | entangled | FINCH | hierarchical InfoNCE + MSE |
| FedPLVM | entangled | double FINCH | alpha-sparsity InfoNCE |
| FedDAP | entangled | cosine-weighted | dual (intra + cross) |
| M3 (current) | decoupled z_sem | none (keep all) | flat SupCon |
| **M4 (proposed)** | **decoupled z_sem** | **cosine-weighted** | **dual (intra + cross)** |

Key difference from FedDAP: In entangled space, cosine similarity between prototypes is corrupted by style differences. In decoupled z_sem space, cosine similarity reflects pure SEMANTIC consistency.

### Failure Modes
1. L_intra dominates -> overfit to own domain -> ablate lambda_intra {0.1, 0.5, 1.0}
2. Cosine weighting degenerates (all equal) -> gracefully degrades to equal-weight M3
3. Training instability -> mitigated by tau=0.2 (validated)

## Claim-Driven Validation Sketch

### Claim 1: Dual alignment > flat SupCon in decoupled space
- Experiment: M4 vs M3, PACS 3-seed
- Expected: M4 > M3 by 1-2%

### Claim 2: Decoupled-space dual alignment > entangled-space dual alignment
- Experiment: M4 (z_sem) vs same dual alignment in h-space
- Expected: z_sem version higher (purer prototypes)

### Claim 3: Cosine-weighted > equal-weight
- Experiment: M4 weighted vs M4 equal-weight
- Expected: weighted +0.5-1%

## Compute and Timeline
- Code changes: ~80 lines
- GPU-hours: ~48h (12 configs x 4h)
- Timeline: 1 day implement + 2 days experiments
