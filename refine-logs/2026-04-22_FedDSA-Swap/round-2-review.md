# Round 2 Review — FedDSA-VIB (A) / FedDSA-VSC (B)

Reviewer: Senior NeurIPS/ICML (external).
Prior rounds: R1 RETHINK 6.2 (L_swap repackaging, MUNIT cycle broken, von Kügelgen misused). R2 is attempt #4.

---

## Q1. New blockers the author missed in self-review

Yes — four the self-review does not acknowledge:

1. **Prior is non-stationary and chicken-and-egg.** `semantic_prototype[y]` is itself updated from q(z_sem|x,y). If prior tracks its own posterior, KL(q‖p) collapses to ~0 and VIB provides no compression pressure. The R0–20 warmup "freeze prototype" is not described; without an EMA-lagged / stop-grad prior, L_VIB degenerates to an identity regularizer. This is the single biggest risk and kills the headline result.
2. **FedAvg on stochastic encoders mixes σ heads across domains.** Sigma represents **domain-conditional aleatoric uncertainty**; averaging σ across PACS-Sketch and PACS-Photo shrinks to the majority domain and makes R (rate) artificially uniform. The "fallback to FedBN-style σ localization" in item 5 is listed as optional — it should be the **default**, not a fallback.
3. **VIB rate/distortion is not identifiable from class labels alone.** With a Gaussian prior centered on y's prototype, the minimum-rate solution is `z_sem = prototype[y]` deterministically — i.e. the encoder collapses to a lookup table indexed by label, passes CE trivially, and destroys all within-class structure (including domain-invariant nuisances). Probe numbers will look great but the representation is degenerate. Need either a lower-bounded σ_prior *learned* or a reconstruction / feature-matching term — neither is in the proposal.
4. **L_HSIC + L_VIB are partially redundant.** Both compress z_sem's domain information (HSIC penalizes z_sem ⊥ z_sty; VIB penalizes I(z_sem; x) which includes domain). Running both without ablating which one does the work will produce an uninterpretable paper. Reviewers will force an ablation anyway.

Minor: SupCon in FL has a known issue (positives from other clients are unreachable without feature exchange) — does B use only local positives? If yes, it's a weaker contrastive than stated; if it uses global prototypes as positives, it's closer to ProtoNCE and novelty claim shrinks.

---

## Q2. Score — Variant A (FedDSA-VIB)

| Dimension | Score | Weight | Notes |
|---|---|---|---|
| Problem Fidelity | 7 | 15% | Leak metric well-defined; domain=client assumption still implicit |
| Method Specificity | 6 | 25% | VIB is specified; prior-update schedule and σ locality are underspecified (see Q1.1, Q1.2) |
| Contribution Quality | 5 | 25% | "FL-first VIB with prototype prior" is a combination, not a mechanism. Author admits this |
| Frontier Leverage | 6 | 15% | Moyer 2018 is 7yr old; no engagement with IB-IRM (Ahuja 2021), IP-IRM (Wang 2022), or VIB-DG variants |
| Feasibility | 8 | 10% | Closed-form KL, reparam trick, ~50 LOC delta. Compute OK |
| Validation Focus | 7 | 5% | Strong diagnostic suite; but predicted Δ (0.5–1.0 pp) is within orth_only noise |
| Venue Readiness | 5 | 5% | Would be desk-reject risk at ICML if presented as novelty; workshop OK |

**Weighted total A = 6.15**

Calc: 0.15·7 + 0.25·6 + 0.25·5 + 0.15·6 + 0.10·8 + 0.05·7 + 0.05·5 = 1.05 + 1.50 + 1.25 + 0.90 + 0.80 + 0.35 + 0.25 = **6.10**.

---

## Q3. Score — Variant B (FedDSA-VSC)

| Dimension | Score | Weight | Notes |
|---|---|---|---|
| Problem Fidelity | 7 | 15% | Same as A |
| Method Specificity | 6 | 25% | SupCon positives across clients are undefined (see Q1 note) |
| Contribution Quality | 5 | 25% | VIB + SupCon is still a combination; SupCon+IB appears in Tian 2021, Tsai 2021 |
| Frontier Leverage | 6 | 15% | Slight uptick: SupCon is more current than InfoNCE-prototype |
| Feasibility | 7 | 10% | SupCon memory bank in FL is nontrivial; may need local-only positives → weaker |
| Validation Focus | 7 | 5% | Same diagnostic suite |
| Venue Readiness | 5 | 5% | Same concern as A |

**Weighted total B = 6.15**

Calc: 0.15·7 + 0.25·6 + 0.25·5 + 0.15·6 + 0.10·7 + 0.05·7 + 0.05·5 = 1.05 + 1.50 + 1.25 + 0.90 + 0.70 + 0.35 + 0.25 = **6.00**.

Essentially tied with A. The author's self-score of B > A by 0.2 is not justified by the proposal — SupCon is not clearly a net win in FL with ≤10 clients and no memory bank.

---

## Q4. Is A+B parallel useful or wasteful?

**Mostly wasteful as currently framed.** Reason: if A fails because of the prior-collapse issue (Q1.1/Q1.3), B will fail identically — SupCon cannot rescue a VIB whose KL term is vacuous. The two variants share the same critical failure mode.

Running both only adds value **if** you also run a third: **A without VIB but with SupCon** (i.e., orth+HSIC+SupCon, no stochastic encoder). This isolates whether the gain comes from VIB or from the contrastive upgrade. Without it, A vs B cannot tell VIB-contribution from SupCon-contribution.

Recommendation: run A, drop B, add "orth+SupCon (no VIB)" as a third cell. This is a 2-factor design (VIB yes/no × contrastive type) instead of a 1-factor.

---

## Q5. Is the 50+ metric suite sufficient?

Strong on breadth, **missing three keys**:

1. **KL collapse detector** — log `mean_y ‖μ(x,y) − prototype[y]‖₂ / σ_prior` per round. If this drops below 0.3 by R80, the posterior is collapsed onto the prior and VIB is doing nothing (or everything — either way, report is degenerate). This directly tests Q1.3.
2. **Domain-conditional rate R_d = E_{x∈d}[KL(q‖p)]** per domain. If R_Sketch ≫ R_Photo, the compression is unfair across domains (the FedPLVM α-sparsity problem re-emerging). 50+ metrics but none stratified by domain × VIB term.
3. **IRM / environment penalty** as a sanity probe. If VIB really removes spurious domain info, V(∇_w L | domain=d) across d should shrink. Current suite has grad_cos(CE, VIB) but not cross-domain gradient variance. This is cheap and directly relevant to the leak claim.

Moyer 0/1/2/3 layer sweep is good and underused by most FL-DG papers — keep it.

---

## Q6. Are 5 seeds enough to beat orth_only std=1.46?

**No, not for a 1 pp effect.** Standard power analysis: to detect Δ=1.0 pp at σ=1.46 with α=0.05, power=0.8, two-sample t-test → n ≈ 2·(1.46)²·(1.96+0.84)² / 1² ≈ **33 seeds per arm**.

For Δ=1.5 pp: n ≈ 15 per arm. For Δ=2.0 pp: n ≈ 8 per arm.

5 seeds is only enough if the true effect is ≥2.5 pp, which contradicts the author's own predicted range (0.5–1.5 pp for A, similar for B). The experiment as designed is **underpowered by ~3–6×** relative to the claim it intends to support.

Options: (a) run 10 seeds on PACS (the noisy one) and 3 on Office (the stable one); (b) use paired design — same 5 seeds across all arms and run **paired t-test** on seed-matched deltas, which exploits seed correlation and gives ~2× effective power; (c) accept that you can only claim "within noise" for A/B vs orth_only and pivot the headline to **the probe metrics** (where the predicted effect is 5–25 pp, easily detectable with 5 seeds).

Option (c) is honest and is what I'd recommend.

---

## Q7. Do I endorse "decoder-free VIB with prototype prior" as genuine improvement?

**Conditionally, leaning incremental.**

Positive: moving from cos² orthogonality (a weak geometric constraint) to KL-divergence-based compression (an information-theoretic constraint) is a genuine step up and addresses R1's legitimate complaint that L_orth + L_HSIC are both geometric and leak nonlinearly. VIB with a class-conditional prior is a reasonable and well-motivated choice.

Negative: (i) the mechanism is 7 years old (Moyer 2018) and has been tried in centralized DG with mixed results; (ii) prototype-as-prior is 3 years old (FedProto 2022) in FL; (iii) the combination is the contribution, and combinations of known mechanisms are incremental by ICML/NeurIPS standards unless they produce a **qualitatively new** outcome (e.g. a property no component alone produces). The proposal does not identify such a property.

**Verdict: genuine improvement over orth_only, incremental as a standalone paper contribution.** Viable for a workshop or a strong tier-2 venue (TMLR, UAI). For ICML/NeurIPS main track, needs an additional ingredient — theory tying FL aggregation to VIB rate distortion, or a novel prior design, or a non-obvious empirical finding.

---

## Q8. Attempt #4 — continue or abandon dual-head decoupling?

**Continue, but narrow the claim.**

Evidence for continuing: the counterfactual probe in EXP-109 showed orth_only legitimately reduces class leakage in z_sty vs CDANN (linear 0.24 vs 0.96). That is a real, measurable effect, and it's not explained by any competitor paper you've surveyed. The decoupling is not fake — what's fake is the *accuracy* claim. The ~1 pp accuracy gap between orth_only and CDANN/FedAvg is noise.

Evidence for abandoning: four rounds in, you still cannot show a clean accuracy win. FL-DG leaderboards are unforgiving; reviewers will focus on accuracy unless you reframe.

**Recommendation: pivot the paper from "a better FL-DG method" to "a diagnostic / representation-quality study of FL-DG decoupling methods."** The current work actually *is* stronger as a diagnostic paper — 50+ metrics, 2×2 landscape, novel probe comparisons — than as a new-SOTA paper. Treat VIB/SupCon as a representation-quality improvement (validated on probes) that maintains accuracy parity, and frame it as the first rigorous decoupling audit for FL. This is publishable, honest, and leverages what you've actually found.

Do **not** continue adding mechanisms (VIB, SupCon, next round VAE, next round diffusion prior…) hoping accuracy will crack. It won't, because accuracy is bottlenecked by the dataset, not the method.

---

## Q9. Final verdicts

- **Variant A (FedDSA-VIB): RETHINK (6.1)** — prior-collapse and σ-aggregation blockers are unaddressed; 5 seeds underpowered; contribution is a 2-component combination.
- **Variant B (FedDSA-VSC): RETHINK (6.0)** — same blockers as A; SupCon-in-FL positives undefined; no independent justification for preferring B over A.

Neither is RETHINK-at-low-end (5.5) because the diagnostic suite, probe framing, and counterfactual baselines are genuinely strong Round-2 work. Neither is REVISE (7+) because the headline mechanism has not survived self-critique.

---

## Q10. Three concrete changes to reach READY (REVISE 7+)

1. **Fix the prior** — make `semantic_prototype[y]` an **EMA-lagged, stop-grad** target with lag ≥20 rounds, and learn `log σ_prior[y]` as a separate server-side parameter (not a hand-picked scalar in {0.3, 1.0, 3.0}). This closes the collapse loophole (Q1.1, Q1.3). Add a diagnostic: `KL_true = KL(q‖p_ema_lagged) − KL(q‖p_current)` to prove the lag is doing work.

2. **Localize σ-head (FedBN-style) as default**, not fallback. Only `μ`-head and backbone aggregate. Add an ablation `σ-global vs σ-local` on PACS at 3 seeds — should be done first (before main 5-seed runs) because if σ-global breaks the method, the whole variant changes. Also drop B (SupCon variant) and replace with **"orth+HSIC+SupCon (no VIB)"** so the 2×2 VIB × contrastive design is identifiable (Q4).

3. **Reframe claim and powering.** Abandon "PACS +1pp over orth_only" as the headline. Run 3 seeds × 4 cells (orth, A, orth+SupCon, A+SupCon) on PACS and Office, but headline on **probe leakage reduction** (where predicted effect is 20–50 pp, trivially significant at n=3) and accuracy **parity** (one-sided non-inferiority test, margin 1 pp, which 5 seeds can actually support). Add the KL-collapse detector, domain-conditional rate R_d, and cross-domain gradient variance (Q5) to the suite. Update the paper title/abstract to "representation-quality" framing (Q8).

With those three, A becomes a defensible REVISE-7.5 paper for UAI / TMLR / a strong workshop, and a plausible main-track submission if one genuinely novel ingredient (learned σ_prior, theory on FL-VIB rate under heterogeneity, or a new leak metric) is added.

---

*Total: ~1450 words.*
