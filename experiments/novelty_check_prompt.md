# Novelty Check — Phase C (Cross-Model Verification)

You are a senior ML reviewer at NeurIPS/ICML level. Your job: verify the novelty of 3 proposed innovation directions for a cross-domain federated learning method, given the prior art I will list below. Be **BRUTALLY HONEST**. False novelty claims waste months of research time.

## Our Setup

We are developing **FedDSA (Decouple-Share-Align)**: a federated prototypical learning method with:
- Dual-head decoupling (semantic head + style head) with orthogonal + HSIC constraints
- Global style bank storing per-client per-class (μ, σ) statistics, enabling cross-client style sharing via AdaIN augmentation
- InfoNCE prototype alignment
- ResNet-18 backbone, FedBN-style local BN

Goal: beat FDSE (CVPR 2025) on PACS and Office-Caltech10.

**Unique asset**: Because we maintain a global style bank, we can FREELY compute pairwise style distance `d_ij = ||μ_i - μ_j||_2 + ||log σ_i - log σ_j||_2`. This gives us a free "regime signal" that most FL methods don't have. On PACS (sketch, photo, art, cartoon) d is large ("high style-gap regime"). On Office-Caltech10 (all real photos) d is small ("low style-gap regime").

**Empirical trigger for this novelty check**: We already tried porting two FDSE modules verbatim into our FedDSA framework:
1. FDSE's Consensus-max QP aggregation (Eq.8): works on PACS (83.04 > baseline 82.24) but only matches baseline on Office (89.40 vs 89.95).
2. FDSE's KL BN-consistency regularization (module C): extremely seed-sensitive on Office — seed=333 jumps to 91.52 (+5.17) but seed=2/15 drops by 1 to 3.7.

Diagnosis: FDSE's single-mechanism aggregation is insufficient for low style-gap domains (where all clients have similar features). We need a different route that is NOT just porting more FDSE components.

## Proposed Innovation Directions

### Direction A — SGA-SAM (Style-Gap-Adaptive Server-Side SAM)
**Core Claims**:
- A1: Replace FDSE KL-BN regularization with server-side SAM (FedGloSS-style) — push the aggregated global model toward flat minima
- A2: The SAM perturbation radius ρ is **adaptive based on the mean pairwise style distance measured from the style bank**. Low-gap regime → larger ρ (aggressively seek flat minima). High-gap regime → smaller ρ
- A3: **Head-selective SAM**: apply SAM only to the semantic head parameters. Backbone uses vanilla FedAvg. Rationale: semantic head carries cross-domain invariance, needs flat minima; backbone can be sharper

### Direction B — PGO (Prototype-Gradient Orthogonality)
**Core Claims**:
- B1: Extend FedSOL's local-proximal orthogonality idea (CVPR 2024)
- B2: The proximal target is changed from "parameter distance gradient" (FedSOL) to **"semantic prototype drift gradient"** `∇||φ(x) - global_prototype_y||²`. I.e., local task gradient should be orthogonal to the direction that pulls features toward the global class prototype — preserves prototype alignment without over-constraining parameters
- B3: **Three-way orthogonality**: `∇L_task ⊥ ∇L_proto ⊥ ∇L_style`. Reuses our existing orthogonal decoupling loss to tie into the FedSOL family

### Direction C — RGHA (Regime-Gated Hybrid Aggregation)
**Core Claims**:
- C1: Compute a regime score `r ∈ [0,1]` = normalized mean pairwise style distance from the style bank
- C2: Hybrid aggregation weighted by r:
  `aggregation = r · Consensus_QP + (1−r) · FedGloSS_SAM + per-layer · ALA_gate`
- C3: A single method with fixed hyperparameters beats FDSE on BOTH PACS and Office-Caltech10 — which FDSE itself cannot do due to its single-mechanism design
- C4: Main paper claim: "The decoupled style bank is a free diagnostic signal for federated aggregation regime selection"

## Phase B Prior Art Findings

### Directly relevant for Direction A

| Paper | Year | Venue | Mechanism | Overlap with A |
|---|---|---|---|---|
| **FedGloSS** | 2025 | CVPR | Server-side SAM with previous-round pseudo-gradient as ε̂, fixed ρ, whole-model perturbation. Experiments: CIFAR-10/100 + Landmarks-User-160k under Dirichlet label skew. **NO PACS/Office/DG benchmarks.** | A1 (base method) ≈ 70%. A2 (style-adaptive ρ) and A3 (head-selective) NOT in FedGloSS. |
| **FedLESAM** | 2024 | ICML Spotlight | Client-side, uses consecutive global model difference to estimate global perturbation, saves 1 backprop | Still local-side, non style-adaptive |
| **FedSCAM** | 2025/2026 | arxiv:2601.00853 | **Early-batch gradient-norm client heterogeneity score inversely modulates SAM radius + heterogeneity-aware aggregation**. Label-skew experiments only (CIFAR-10/F-MNIST) | ⚠️ **CLOSEST COLLISION FOR A2**: "per-client adaptive ρ from heterogeneity score" already exists. But score is grad-norm, NOT style statistics. No DG experiments. |
| **ASAM** | 2021 | ICML | Input-sensitivity-based adaptive ρ (scale-invariant). Non-FL | Philosophy overlap, not FL |
| **DGSAM** | 2025 | arxiv:2503.23430 | Domain-level individual sharpness for DG (centralized). Avoids fake flat minima | Centralized DG, non FL |
| **DP²-FedSAM** | 2024 | arxiv:2409.13645 | DP-preserving + separate shared extractor / personal head, applies SAM to personal head | ⚠️ A3 partial overlap: "head-differentiated SAM" exists but for DP, not style-gap |

### Directly relevant for Direction B

| Paper | Year | Venue | Mechanism | Overlap with B |
|---|---|---|---|---|
| **FedSOL** | 2024 | CVPR | `g_local ⊥ g_proximal` where `L_p = ||w-w_g||²`. Parameter-space only. No prototype involvement | B1 baseline. B2 (prototype gradient target) not in FedSOL. |
| **FedORGP / FedOC** | 2025 | arxiv:2502.16119 | **Inter-class orthogonality regularization on server-side global prototypes** — maximizes class-angular separation | ⚠️ **CLOSEST COLLISION FOR B2**: "prototype + orthogonality" combo already taken. But it's class-vs-class orthogonal (static server reg), NOT gradient-level, NOT semantic-vs-style decomposition |
| **FedPGO** | 2026 | ScienceDirect IoT | Decomposes client gradient into global-aligned + orthogonal components, per-layer re-weight | ⚠️ B1/B3 near-neighbor: "gradient orthogonal decomposition" in FL exists, but decomposition basis is global direction, not prototype shift / style direction |
| **FedProTIP** | 2025 | arxiv:2509.21606 | Project client update onto orthogonal complement of learned representation subspace (FL continual learning) | Non prototype-gradient |
| **GDOD** | 2023 | arxiv:2301.13465 | Multi-task orthogonal gradient decomposition for conflicting gradients | Non-FL, non-prototype |

### Directly relevant for Direction C

| Paper | Year | Venue | Mechanism | Overlap with C |
|---|---|---|---|---|
| **CCST** | 2023 | WACV | **Explicit server-side style bank storing per-client (μ,σ)**, image-space AdaIN cross-client style transfer | ⚠️⚠️ **TERM COLLISION**: "style bank" term already used since 2023! But CCST uses bank for augmentation, NOT as regime diagnostic, NOT for gating aggregation, does not compute pairwise style distance |
| **FedAWA** | 2025 | CVPR | Client-vector (update direction) adaptive aggregation weights, plug-and-play | Adaptive aggregation exists, but signal is update direction not style |
| **FedDisco** | 2023 | ICML | Per-client aggregation weight = data_size × local-vs-global category-distribution discrepancy | ⚠️ C1 philosophy overlap: "discrepancy-signal-based aggregation weight" exists, but signal is LABEL distribution, not style |
| **FedCA (medical)** | 2026 | ESWA | Cross-client feature style transfer + adaptive style alignment at test-time aggregation | ⚠️ C1/C2 partial: "style-based adaptive aggregation" exists but test-time/per-image, not regime-level gate, and not QP+SAM hybrid |
| **StyleDDG** | 2025 | arxiv:2504.06235 | Decentralized P2P style sharing, convergence analysis | Share style, not regime-gated, not hybrid |
| **FISC/PARDON** | 2025 | ICDCS | FINCH aggregate style → median interpolation → AdaIN, no decoupling, no gate | Non-hybrid |
| **pFedGraph** | 2023 | ICML | Pairwise model similarity → collaboration graph | Signal is model sim, not style |
| **FedSDAF** | 2025 | arxiv:2505.02515 | Dual adapter: domain-aware (local) + domain-invariant (shared) | Decoupled at adapter level, not aggregation level |

### Red Flags from Phase B
- NO direct "style-adaptive SAM" paper found (FedSCAM is closest but uses grad-norm)
- NO "prototype-space gradient orthogonal proximal" paper found (FedSOL is parameter-space)
- NO "regime-gated hybrid aggregation with style distance" paper found
- PARTIAL: "style bank" is a taken term (CCST 2023) — but semantic usage differs. C4 needs rephrasing

## Your Task — Be BRUTALLY HONEST

For each of the 3 directions, answer:

1. **Novelty assessment per claim**: Is each sub-claim (A1/A2/A3, B1/B2/B3, C1/C2/C3/C4) novel? Rate HIGH / MEDIUM / LOW. Identify the **closest prior work** for each.

2. **Reviewer ammunition**: What would a skeptical NeurIPS reviewer cite as "this is incremental"? Give me the concrete argument they would use.

3. **Overall direction score**: 0–10 scale where:
   - 0–3: ABANDON (already done or trivial extension)
   - 4–6: PROCEED WITH CAUTION (needs strong reframing)
   - 7–8: PROCEED (novel contribution, needs solid experiments)
   - 9–10: PROCEED AGGRESSIVELY (genuinely new insight)

4. **Positioning advice**: Given the prior art, give me a 2–3 sentence statement of contribution that maximizes novelty perception while being honest.

5. **Missing search**: What prior art do you suspect we missed that I should search for?

6. **Critical question**: Is FedSCAM a real threat to direction A? Given it uses grad-norm (a gradient magnitude signal) and we use style distance (a feature statistics signal) — is the conceptual delta enough, or is this too close?

7. **Final recommendation**: Among the 3 directions, which ONE should we go first and why? Rank them.

8. **Alternative angle**: If all 3 directions have >medium novelty risk, suggest a different angle we should consider based on our unique assets (decoupled style bank + dual-head + regime-dependence finding).

Please output in structured Markdown. No softballs. If something is too close to prior art, say so directly with specific citation.
